"""
EvidentialWriteGate
===================
Component 1 of EMN: Evidential Write Gate.

Every incoming memory passes through this gate before storage.
The gate uses NOVA's EvidentialHead (via VacuityExtractor) to assign
a Dirichlet-based vacuity score that travels with the memory forever.

Architecture (inherited from NOVA):
    LayerNorm → Linear(d_model, d_model//2) → SiLU → Linear(d_model//2, n_classes)
    → Softplus + 1  →  alpha
    →  belief = (alpha - 1) / S
    →  vacuity = n_classes / S   (clamped to [0, 1])

This is a proper nn.Module so gradients flow through it during training.
The store calls it at write time; the returned vacuity is persisted with the
memory and used for eviction and retrieval scoring.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from emn.evidential.nova_bridge import VacuityExtractor
from emn.types import WriteGateOutput


class EvidentialWriteGate(nn.Module):
    """
    Evidential write gate — assigns vacuity to incoming memory embeddings.

    Parameters
    ----------
    d_model   : int  — embedding dimensionality
    n_classes : int  — Dirichlet classes (256 matches NOVA default)

    Usage
    -----
    gate = EvidentialWriteGate(d_model=512)
    output = gate(x)          # x: (batch, d_model)
    vacuity = output.vacuity  # (batch,) in [0, 1]
    """

    def __init__(self, d_model: int, n_classes: int = 256) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.vacuity_extractor = VacuityExtractor(d_model=d_model, n_classes=n_classes)

    def forward(self, x: torch.Tensor) -> WriteGateOutput:
        """
        Parameters
        ----------
        x : (batch, d_model) — incoming memory embedding

        Returns
        -------
        WriteGateOutput
          .alpha   : (batch, n_classes) — Dirichlet concentration params
          .belief  : (batch, n_classes) — expected belief masses, sums to 1
          .vacuity : (batch,)           — uncertainty scalar in [0, 1]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        alpha, epistemic_unc, _aleatoric_unc = self.vacuity_extractor.full_forward(x)

        # belief_k = (alpha_k - 1) / S — normalised evidence masses
        S = alpha.sum(dim=-1, keepdim=True)
        belief = (alpha - 1.0) / S  # (batch, n_classes)

        return WriteGateOutput(
            alpha=alpha,
            belief=belief,
            vacuity=epistemic_unc,  # = n_classes / S clamped to [0,1]
        )

    def compute_vacuity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience: return only the vacuity tensor.

        Parameters
        ----------
        x : (batch, d_model) or (d_model,)

        Returns
        -------
        vacuity : (batch,) in [0, 1]
        """
        return self.forward(x).vacuity

    def evidential_loss(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        kl_weight: float = 0.01,
    ) -> torch.Tensor:
        """
        Compute NOVA evidential loss for training the write gate.

        Parameters
        ----------
        x       : (batch, d_model)
        targets : (batch,) integer class labels in [0, n_classes)
        kl_weight : float — KL regularisation weight

        Returns
        -------
        loss : scalar tensor
        """
        output = self.forward(x)
        return self.vacuity_extractor.evidential_head.evidential_loss(
            output.alpha, targets, kl_weight=kl_weight
        )

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n_classes={self.n_classes}"
