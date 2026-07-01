"""
UncertaintyWeightedRetriever
============================
Component 3 of EMN: Uncertainty-Downweighted Retrieval.

Retrieval score formula:
    score_i = cosine(query, memory_i) * (1 - vacuity_i)

High-vacuity (uncertain) memories score lower even if they are geometrically
close to the query — the system prefers confident memories.

Two backends:
  "brute"  — pure PyTorch/NumPy cosine similarity, O(N * d_model)
  "faiss"  — FAISS IndexFlatIP on L2-normalised vectors for ANN search,
             then post-reranks top-2k candidates by vacuity-adjusted score

Backend is selectable at construction time and via retrieve(backend=...).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from emn.memory.entry import MemoryEntry
from emn.memory.store import EpistemicMemoryStore


class UncertaintyWeightedRetriever:
    """
    Retriever that downweights uncertain memories during scoring.

    Parameters
    ----------
    store                 : EpistemicMemoryStore
    backend               : "brute" | "faiss"
    vacuity_weight        : float — weight of vacuity penalty in [0, 1]
                            0.0 = pure cosine similarity (ablation)
                            1.0 = full EMN downweighting (default)
    faiss_probe_factor    : int — FAISS retrieves k * factor candidates
                            before reranking; higher = more accurate
    device                : "cpu" | "cuda:N" — for torch ops
    """

    def __init__(
        self,
        store: EpistemicMemoryStore,
        backend: str = "brute",
        vacuity_weight: float = 1.0,
        faiss_probe_factor: int = 2,
        device: str = "cpu",
    ) -> None:
        self.store = store
        self.backend = backend
        self.vacuity_weight = vacuity_weight
        self.faiss_probe_factor = faiss_probe_factor
        self.device = device
        self._faiss_index = None
        self._faiss_dirty = True  # needs rebuild when store changes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: np.ndarray,
        k: int = 5,
        backend: Optional[str] = None,
        return_scores: bool = False,
    ) -> List[MemoryEntry] | Tuple[List[MemoryEntry], np.ndarray]:
        """
        Retrieve top-k memories by uncertainty-weighted cosine similarity.

        Parameters
        ----------
        query        : (d_model,) float32
        k            : int — number of results
        backend      : "brute" | "faiss" | None (use constructor default)
        return_scores: bool — if True return (entries, scores) tuple

        Returns
        -------
        List[MemoryEntry] — sorted best-first
        (optionally tuple with score array)
        """
        if len(self.store) == 0:
            return ([], np.array([])) if return_scores else []

        backend = backend or self.backend
        k = min(k, len(self.store))

        if backend == "brute":
            entries, scores = self._retrieve_brute(query, k)
        elif backend == "faiss":
            entries, scores = self._retrieve_faiss(query, k)
        else:
            raise ValueError(f"Unknown backend: {backend!r}. Use 'brute' or 'faiss'.")

        if return_scores:
            return entries, scores
        return entries

    def retrieve_batch(
        self,
        queries: np.ndarray,
        k: int = 5,
        backend: Optional[str] = None,
    ) -> List[List[MemoryEntry]]:
        """
        Retrieve for a batch of queries.

        Parameters
        ----------
        queries : (batch, d_model)
        k       : int

        Returns
        -------
        List of lists, one per query
        """
        return [self.retrieve(q, k=k, backend=backend) for q in queries]

    # ------------------------------------------------------------------
    # Brute-force backend (NumPy)
    # ------------------------------------------------------------------

    def _retrieve_brute(
        self, query: np.ndarray, k: int
    ) -> Tuple[List[MemoryEntry], np.ndarray]:
        vectors = self.store.all_vectors()   # (size, d_model)
        vacuities = self.store.all_vacuities()  # (size,)
        entries = self.store.all_entries()

        # L2-normalise
        q_norm = query / (np.linalg.norm(query) + 1e-9)
        v_norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
        v_normalised = vectors / v_norms

        cosine = v_normalised @ q_norm  # (size,)
        score = cosine * (1.0 - self.vacuity_weight * vacuities)

        top_idx = np.argpartition(score, -k)[-k:]
        top_idx = top_idx[np.argsort(score[top_idx])[::-1]]

        return [entries[i] for i in top_idx], score[top_idx]

    # ------------------------------------------------------------------
    # FAISS backend
    # ------------------------------------------------------------------

    def _retrieve_faiss(
        self, query: np.ndarray, k: int
    ) -> Tuple[List[MemoryEntry], np.ndarray]:
        """
        FAISS IndexFlatIP (inner product on normalised vectors = cosine sim).
        Retrieves k * faiss_probe_factor candidates, then reranks by vacuity score.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for the FAISS backend. "
                "Install with: pip install faiss-cpu"
            )

        self._maybe_rebuild_faiss_index(faiss)

        n_probe = min(k * self.faiss_probe_factor, len(self.store))

        q_norm = query / (np.linalg.norm(query) + 1e-9)
        q_f32 = q_norm.astype(np.float32).reshape(1, -1)

        distances, indices = self._faiss_index.search(q_f32, n_probe)
        # distances are inner products (= cosine on normalised vectors)
        distances = distances[0]  # (n_probe,)
        indices = indices[0]      # (n_probe,)

        # Filter invalid indices (-1 means fewer results than requested)
        valid = indices >= 0
        distances = distances[valid]
        indices = indices[valid]

        if len(indices) == 0:
            return [], np.array([])

        entries = self.store.all_entries()
        vacuities = self.store.all_vacuities()

        # Rerank by vacuity-adjusted score
        scores = distances * (1.0 - self.vacuity_weight * vacuities[indices])
        order = np.argsort(scores)[::-1][:k]

        top_entries = [entries[indices[i]] for i in order]
        top_scores = scores[order]

        return top_entries, top_scores

    def _maybe_rebuild_faiss_index(self, faiss) -> None:
        """Rebuild FAISS index if store has changed since last build."""
        # We rebuild whenever called — for production, could track a dirty flag
        # by hooking into store.write(); for benchmark scale this is fine
        vectors = self.store.all_vectors()  # (size, d_model)
        if len(vectors) == 0:
            return

        d = vectors.shape[1]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
        v_norm = (vectors / norms).astype(np.float32)

        index = faiss.IndexFlatIP(d)
        index.add(v_norm)
        self._faiss_index = index

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def score_all(self, query: np.ndarray) -> np.ndarray:
        """
        Return the uncertainty-weighted score for every entry in the store.
        Useful for visualising the memory landscape.

        Returns
        -------
        scores : (size,) — sorted same order as store.all_entries()
        """
        if len(self.store) == 0:
            return np.array([])
        vectors = self.store.all_vectors()
        vacuities = self.store.all_vacuities()
        q_norm = query / (np.linalg.norm(query) + 1e-9)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
        cosine = (vectors / norms) @ q_norm
        return cosine * (1.0 - self.vacuity_weight * vacuities)

    def __repr__(self) -> str:
        return (
            f"UncertaintyWeightedRetriever("
            f"backend={self.backend!r}, "
            f"vacuity_weight={self.vacuity_weight}, "
            f"store_size={len(self.store)})"
        )
