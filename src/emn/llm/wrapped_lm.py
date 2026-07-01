"""
EMNWrappedCausalLM
==================
Wraps a HuggingFace CausalLM (or the Anthropic API) with an EMN memory
system. Retrieval-augmented generation using uncertainty-gated memories.

Usage:
------
    model = EMNWrappedCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        memory_capacity=1000,
    )
    model.add_memory("The Eiffel Tower is in Paris.")
    response = model.generate("Where is the Eiffel Tower?")

Or with the Anthropic API:
    model = EMNWrappedCausalLM(
        backend="anthropic",
        anthropic_model="claude-sonnet-4-6",
        d_model=384,
        memory_capacity=1000,
    )

Memory insertion:
    Text → sentence encoder → (d_model,) → EvidentialWriteGate → EpistemicMemoryStore

Retrieval:
    Query → encoder → top-k memories → prepend as context → generation
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from emn.memory.store import EpistemicMemoryStore
from emn.memory.entry import MemoryEntry
from emn.gates.write_gate import EvidentialWriteGate
from emn.retrieval.retriever import UncertaintyWeightedRetriever
from emn.types import MemoryStats


class EMNWrappedCausalLM:
    """
    LLM with EMN memory augmentation.

    Supports two generation backends:
    - "hf"        : local HuggingFace transformers model
    - "anthropic" : Anthropic API (claude-*)

    The memory system (store, write gate, retriever) is always local
    regardless of the generation backend.

    Parameters
    ----------
    backend          : "hf" | "anthropic"
    hf_model_name    : str — HuggingFace model ID (for backend="hf")
    anthropic_model  : str — Anthropic model string (for backend="anthropic")
    d_model          : int — sentence encoder output dim (for memory embeddings)
    memory_capacity  : int — max memories
    retrieval_k      : int — memories to prepend per generation
    retrieval_backend: "brute" | "faiss"
    vacuity_weight   : float — retrieval vacuity downweighting strength
    device           : str
    """

    def __init__(
        self,
        backend: str = "hf",
        hf_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        anthropic_model: str = "claude-sonnet-4-6",
        d_model: int = 384,
        memory_capacity: int = 1000,
        retrieval_k: int = 5,
        retrieval_backend: str = "brute",
        vacuity_weight: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.backend = backend
        self.hf_model_name = hf_model_name
        self.anthropic_model = anthropic_model
        self.d_model = d_model
        self.retrieval_k = retrieval_k
        self.device = device

        # Memory system (always local)
        self.write_gate = EvidentialWriteGate(d_model=d_model).to(device)
        self.store = EpistemicMemoryStore(
            capacity=memory_capacity,
            d_model=d_model,
            device=device,
        )
        self.retriever = UncertaintyWeightedRetriever(
            store=self.store,
            backend=retrieval_backend,
            vacuity_weight=vacuity_weight,
            device=device,
        )

        # Sentence encoder for converting text → embeddings
        self._encoder = None  # lazy-loaded

        # Generation backends (lazy-loaded)
        self._hf_model = None
        self._hf_tokenizer = None
        self._anthropic_client = None

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        memory_capacity: int = 1000,
        retrieval_k: int = 5,
        device: str = "cpu",
        **kwargs,
    ) -> "EMNWrappedCausalLM":
        """Convenience constructor for HF models."""
        return cls(
            backend="hf",
            hf_model_name=model_name,
            memory_capacity=memory_capacity,
            retrieval_k=retrieval_k,
            device=device,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_memory(
        self,
        text: str,
        task_id: str = "",
        metadata: Optional[dict] = None,
    ) -> MemoryEntry:
        """
        Encode text and store as a new memory.

        Parameters
        ----------
        text     : str — the memory content
        task_id  : str — optional task label
        metadata : dict | None — optional metadata (stored as-is)

        Returns
        -------
        MemoryEntry — the stored entry with its assigned vacuity
        """
        embedding = self._encode_text(text)
        # Compute vacuity via write gate
        t = torch.from_numpy(embedding).unsqueeze(0).to(self.device)
        with torch.no_grad():
            gate_output = self.write_gate(t)
        vacuity = float(gate_output.vacuity.squeeze(0).cpu())

        meta = metadata or {}
        meta["source_text"] = text

        return self.store.write(
            vector=embedding,
            task_id=task_id,
            metadata=meta,
            vacuity=vacuity,
        )

    def retrieve_memory(
        self,
        query: str,
        k: Optional[int] = None,
        return_scores: bool = False,
    ) -> List[MemoryEntry] | tuple:
        """
        Retrieve top-k memories relevant to query.

        Parameters
        ----------
        query        : str — query text
        k            : int | None (use constructor default)
        return_scores: bool

        Returns
        -------
        List[MemoryEntry] or (entries, scores) if return_scores=True
        """
        k = k or self.retrieval_k
        q_emb = self._encode_text(query)
        return self.retriever.retrieve(q_emb, k=k, return_scores=return_scores)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        task_id: str = "",
        store_response: bool = False,
        **kwargs,
    ) -> str:
        """
        Generate a response, augmented with retrieved memories.

        1. Retrieve top-k relevant memories
        2. Prepend as context to the prompt
        3. Generate via configured backend

        Parameters
        ----------
        prompt          : str
        max_new_tokens  : int
        temperature     : float
        task_id         : str — for memory storage of response
        store_response  : bool — if True, also store response in memory
        **kwargs        : passed to generation backend

        Returns
        -------
        str — generated response text
        """
        # Retrieve relevant memories
        memories = self.retrieve_memory(prompt)
        augmented_prompt = self._build_augmented_prompt(prompt, memories)

        # Generate
        if self.backend == "hf":
            response = self._generate_hf(augmented_prompt, max_new_tokens, temperature, **kwargs)
        elif self.backend == "anthropic":
            response = self._generate_anthropic(augmented_prompt, max_new_tokens, temperature, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

        # Optionally store response
        if store_response and response.strip():
            self.add_memory(response, task_id=task_id, metadata={"type": "response"})

        return response

    def memory_stats(self) -> MemoryStats:
        """Return statistics about the current memory store."""
        return self.store.stats()

    # ------------------------------------------------------------------
    # Generation backends
    # ------------------------------------------------------------------

    def _generate_hf(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        **kwargs,
    ) -> str:
        """Generate using a local HuggingFace model."""
        model, tokenizer = self._load_hf_model()

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                **kwargs,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _generate_anthropic(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        system: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate using the Anthropic API."""
        client = self._load_anthropic_client()

        messages = [{"role": "user", "content": prompt}]
        create_kwargs = {
            "model": self.anthropic_model,
            "max_tokens": max_new_tokens,
            "messages": messages,
        }
        if system:
            create_kwargs["system"] = system
        # Note: Anthropic API temperature range is 0-1
        if temperature != 1.0:
            create_kwargs["temperature"] = min(max(temperature, 0.0), 1.0)

        response = client.messages.create(**create_kwargs)
        return response.content[0].text

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_augmented_prompt(
        self, prompt: str, memories: List[MemoryEntry]
    ) -> str:
        """Prepend retrieved memories as context."""
        if not memories:
            return prompt

        context_parts = []
        for i, entry in enumerate(memories):
            src = entry.metadata.get("source_text", "")
            if src:
                conf = f"[confidence: {entry.confidence():.2f}]"
                context_parts.append(f"[Memory {i+1} {conf}]: {src}")

        if not context_parts:
            return prompt

        context = "\n".join(context_parts)
        return (
            f"Relevant context from memory:\n{context}\n\n"
            f"Question: {prompt}"
        )

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def _encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to a (d_model,) float32 numpy vector.
        Uses sentence-transformers/all-MiniLM-L6-v2 (384-dim).
        If d_model != 384, a learned projection is applied.
        """
        encoder = self._load_encoder()
        with torch.no_grad():
            embedding = encoder.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]  # (384,)

        if embedding.shape[0] != self.d_model:
            # Pad or truncate to match d_model
            if embedding.shape[0] > self.d_model:
                embedding = embedding[: self.d_model]
            else:
                embedding = np.pad(embedding, (0, self.d_model - embedding.shape[0]))

        return embedding.astype(np.float32)

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _load_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        return self._encoder

    def _load_hf_model(self):
        if self._hf_model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_name,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            ).to(self.device)
            self._hf_model.eval()
        return self._hf_model, self._hf_tokenizer

    def _load_anthropic_client(self):
        if self._anthropic_client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required for Anthropic backend. "
                    "Install with: pip install anthropic"
                )
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "ANTHROPIC_API_KEY environment variable not set."
                )
            self._anthropic_client = anthropic.Anthropic(api_key=api_key)
        return self._anthropic_client

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save_memory(self, path: str) -> None:
        """Save the memory store to disk."""
        self.store.save(path)

    def load_memory(self, path: str) -> None:
        """Load memory store from disk."""
        self.store = EpistemicMemoryStore.load(path, device=self.device)
        self.retriever = UncertaintyWeightedRetriever(
            store=self.store,
            backend=self.retriever.backend,
            vacuity_weight=self.retriever.vacuity_weight,
            device=self.device,
        )

    def __repr__(self) -> str:
        return (
            f"EMNWrappedCausalLM("
            f"backend={self.backend!r}, "
            f"model={self.hf_model_name if self.backend == 'hf' else self.anthropic_model!r}, "
            f"memory={len(self.store)}/{self.store.capacity})"
        )
