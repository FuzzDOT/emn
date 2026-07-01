"""
NOVA Bridge
===========
Thin wrapper around NOVA's EvidentialHead that exposes a single clean
interface for EMN components that only need the scalar vacuity score.

Design principle: EMN never reimplements evidential learning — it only
calls into the canonical NOVA implementation via this bridge.

Vacuity = K / S  where K = n_classes, S = sum(alpha)
High vacuity  → low total evidence → uncertain memory → candidate for eviction
Low vacuity   → high total evidence → confident memory → protected from eviction
"""

from __future__ import annotations

import torch
import torch.nn as nn
from emn.evidential.nova_uncertainty import EvidentialHead


class VacuityExtractor(nn.Module):
    """
    Wraps NOVA's EvidentialHead to extract only the vacuity scalar.

    Used by:
    - EvidentialWriteGate   (assigns vacuity at memory write time)
    - EpistemicMemoryStore  (stores vacuity per entry)
    - UncertaintyWeightedRetriever (downweights high-vacuity memories)

    Parameters
    ----------
    d_model   : int   — dimensionality of input embeddings
    n_classes : int   — number of Dirichlet classes (default 256, matching NOVA)
    """

    def __init__(self, d_model: int, n_classes: int = 256) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.evidential_head = EvidentialHead(d_model=d_model, n_classes=n_classes)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden : (batch, d_model) — pooled memory embedding

        Returns
        -------
        vacuity : (batch,) — scalar uncertainty score in [0, 1]
                  0 = maximally confident
                  1 = maximally uncertain
        """
        _alpha, epistemic_unc, _aleatoric_unc = self.evidential_head(hidden)
        return epistemic_unc  # vacuity = n_classes / S clamped to [0,1]

    def full_forward(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full NOVA evidential output for callers that need alpha + both
        uncertainty components (e.g. EvidentialWriteGate).

        Returns
        -------
        alpha         : (batch, n_classes)
        epistemic_unc : (batch,)   — vacuity score
        aleatoric_unc : (batch,)   — normalised entropy
        """
        return self.evidential_head(hidden)

    def vacuity_from_numpy(self, vector: "np.ndarray") -> float:  # type: ignore[name-defined]
        """
        Convenience: compute scalar vacuity from a single numpy vector.
        Used by EpistemicMemoryStore.write() when the caller passes a raw array.

        Parameters
        ----------
        vector : (d_model,) float32 numpy array

        Returns
        -------
        float — vacuity in [0, 1]
        """
        import numpy as np
        t = torch.from_numpy(vector.astype(np.float32)).unsqueeze(0)
        device = next(self.parameters()).device
        t = t.to(device)
        with torch.no_grad():
            vac = self.forward(t)
        return float(vac.squeeze(0).cpu().item())
