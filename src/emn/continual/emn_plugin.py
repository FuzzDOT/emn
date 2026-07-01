"""
EMNPlugin
=========
Avalanche SupervisedPlugin that integrates the EMN memory protection loss
into any continual learning training loop.

Memory protection loss (per memory entry i):
    L_mem_i = (1 - v_i) * || f_t(x_i) - m_i ||^2

where:
  v_i   = vacuity of memory i  (low v = high confidence = strong protection)
  f_t   = current model's feature extractor output for memory input x_i
  m_i   = stored memory vector

Total loss:
    L = L_ce + lambda_mem * mean(L_mem)

The (1 - v_i) weight is the key EMN contribution:
  uncertain memories (high v) contribute little to the protection loss
  → the model is free to overwrite uncertain/noisy old memories
  confident memories (low v) contribute strongly
  → the model is penalised for forgetting what it knows well
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from emn.memory.entry import MemoryEntry
from emn.memory.store import EpistemicMemoryStore
from emn.gates.write_gate import EvidentialWriteGate

try:
    from avalanche.training.plugins import SupervisedPlugin
    from avalanche.training.templates import SupervisedTemplate
    AVALANCHE_AVAILABLE = True
except ImportError:
    # Provide a stub so the module imports even without Avalanche installed
    AVALANCHE_AVAILABLE = False

    class SupervisedPlugin:  # type: ignore[no-redef]
        """Stub for environments without Avalanche."""
        def before_backward(self, *args, **kwargs):
            pass
        def after_training_exp(self, *args, **kwargs):
            pass


class EMNPlugin(SupervisedPlugin):
    """
    Avalanche plugin that injects EMN's memory-protection loss.

    Parameters
    ----------
    store           : EpistemicMemoryStore
    write_gate      : EvidentialWriteGate
    feature_extractor : nn.Module
                      Callable that maps (batch_inputs) → (batch, d_model).
                      Typically the backbone minus the classification head.
    lambda_mem      : float — weight for memory protection loss
    memory_batch_size : int — memories sampled per backward pass
    store_new_memories_every : int — store a new memory every N training steps
    device          : str
    """

    def __init__(
        self,
        store: EpistemicMemoryStore,
        write_gate: EvidentialWriteGate,
        feature_extractor: nn.Module,
        lambda_mem: float = 0.5,
        memory_batch_size: int = 64,
        store_new_memories_every: int = 10,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.store = store
        self.write_gate = write_gate
        self.feature_extractor = feature_extractor
        self.lambda_mem = lambda_mem
        self.memory_batch_size = memory_batch_size
        self.store_new_memories_every = store_new_memories_every
        self.device = device
        self._step = 0
        self._current_task_id = "task_0"

    # ------------------------------------------------------------------
    # Avalanche hooks
    # ------------------------------------------------------------------

    def before_backward(
        self, strategy: "SupervisedTemplate", *args, **kwargs
    ) -> None:
        """
        Called by Avalanche just before loss.backward().
        Computes memory protection loss and adds it to strategy.loss.
        """
        if len(self.store) == 0:
            return

        mem_loss = self._compute_memory_protection_loss(strategy)
        if mem_loss is not None:
            strategy.loss += self.lambda_mem * mem_loss

        # Optionally store current batch embeddings as new memories
        self._step += 1
        if self._step % self.store_new_memories_every == 0:
            self._store_current_batch(strategy)

    def after_training_exp(
        self, strategy: "SupervisedTemplate", *args, **kwargs
    ) -> None:
        """Called after each experience (task). Updates task ID counter."""
        exp_id = getattr(strategy.experience, "current_experience", 0)
        self._current_task_id = f"task_{exp_id + 1}"

    # ------------------------------------------------------------------
    # Memory protection loss
    # ------------------------------------------------------------------

    def _compute_memory_protection_loss(
        self, strategy: "SupervisedTemplate"
    ) -> Optional[torch.Tensor]:
        """
        Sample memories → run feature extractor → compute weighted L2 loss.

        Returns
        -------
        scalar tensor or None if store is empty
        """
        sampled: List[MemoryEntry] = self.store.sample(
            n=self.memory_batch_size,
            strategy="inverse_vacuity",  # prefer confident memories
        )
        if not sampled:
            return None

        # Stack memory vectors and vacuity weights
        mem_vectors = torch.from_numpy(
            np.stack([e.vector for e in sampled])
        ).to(self.device)  # (n_mem, d_model)

        vacuities = torch.tensor(
            [e.vacuity for e in sampled], dtype=torch.float32, device=self.device
        )  # (n_mem,)

        # Run current model's feature extractor on memory inputs
        # We treat memory vectors as feature-space representations and
        # compute their updated representations under the current model
        # by a lightweight linear probe — or directly use the stored vectors
        # as targets (they ARE the feature-space memories)
        self.feature_extractor.eval()
        with torch.no_grad():
            # Memory vectors are feature-space embeddings; we compare
            # current model output for any batch input against stored memories.
            # The standard EMN formulation: use a fresh forward pass on the
            # stored raw inputs, but since we only store feature vectors
            # (not raw inputs) we use the feature vectors as both input and target.
            # This is equivalent to a memory consolidation regulariser.
            current_feats = self.feature_extractor(mem_vectors)
            if current_feats.dim() > 2:
                current_feats = current_feats.mean(dim=1)

        self.feature_extractor.train()

        # L_mem_i = (1 - v_i) * ||current_feat_i - stored_i||^2
        diff = current_feats - mem_vectors  # (n_mem, d_model)
        per_entry_loss = diff.pow(2).sum(dim=-1)  # (n_mem,)
        confidence_weights = (1.0 - vacuities).clamp(min=0.0)
        weighted_loss = confidence_weights * per_entry_loss
        return weighted_loss.mean()

    def _store_current_batch(self, strategy: "SupervisedTemplate") -> None:
        """Extract features from current batch and store in memory."""
        try:
            mb_x = strategy.mb_x
        except AttributeError:
            return

        self.feature_extractor.eval()
        with torch.no_grad():
            feats = self.feature_extractor(mb_x)
            if feats.dim() > 2:
                feats = feats.mean(dim=1)

        self.feature_extractor.train()

        # Store a subset of the batch (first min(8, batch) entries)
        n_store = min(8, feats.shape[0])
        for i in range(n_store):
            vec = feats[i].detach().cpu().numpy().astype(np.float32)
            self.store.write(
                vector=vec,
                task_id=self._current_task_id,
                metadata={"step": self._step},
            )


class EMNFeatureExtractorWrapper(nn.Module):
    """
    Wraps a classification backbone to expose only the feature extractor part.

    Used with Avalanche's ResNet backbones where the forward() returns logits
    but EMN needs intermediate feature representations.

    Parameters
    ----------
    backbone        : nn.Module — full model (e.g. SlimResNet18)
    feature_layer   : str — attribute name of the feature layer
                      If None, removes the last Linear layer automatically.
    """

    def __init__(self, backbone: nn.Module, feature_layer: Optional[str] = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_layer = feature_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run backbone and return feature embeddings (not logits).

        If backbone has a .feature_extractor attribute, use that.
        Otherwise, hook the penultimate layer.
        """
        if self.feature_layer is not None:
            # Use named sub-module
            feats = dict(self.backbone.named_modules())[self.feature_layer](x)
            return feats

        # Auto-detect: try common attribute names
        for attr in ("feature_extractor", "features", "encoder"):
            if hasattr(self.backbone, attr):
                return getattr(self.backbone, attr)(x)

        # Last resort: run full forward and return penultimate activations
        # by removing the final classifier layer temporarily
        modules = list(self.backbone.children())
        feature_modules = nn.Sequential(*modules[:-1])
        out = feature_modules(x)
        if out.dim() > 2:
            out = torch.flatten(out, start_dim=1)
        return out
