"""
EMN Types
=========
Lightweight dataclasses that mirror NOVA's research_core.types.
These are derived by reading nova_uncertainty.py and implementing
only what EMN needs — no dependency on the full NOVA research_core package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List
try:
    import torch
except ImportError:
    pass

# Constants matching NOVA defaults
ABSTENTION_THRESHOLD: float = 0.7
MIN_CONFIDENCE_TO_ANSWER: float = 0.05


@dataclass
class CognitiveState:
    """
    Mirrors NOVA's CognitiveState.
    Carries hidden representations and uncertainty signals through the pipeline.
    """
    hidden: torch.Tensor                          # (batch, seq, d_model)
    uncertainty: torch.Tensor                     # (batch, seq) — blended total uncertainty
    epistemic_uncertainty: torch.Tensor           # (batch, seq)
    aleatoric_uncertainty: torch.Tensor           # (batch, seq)
    attention_mask: torch.Tensor                  # (batch, seq) — 1 = real token
    memory_keys: Optional[torch.Tensor] = None    # (batch, n_mem, d_model)
    memory_values: Optional[torch.Tensor] = None  # (batch, n_mem, d_model)
    routing_weights: Optional[torch.Tensor] = None
    computation_depth: int = 0
    should_abstain: Optional[torch.Tensor] = None # (batch,) bool
    trace: List[str] = field(default_factory=list)

    def record(self, msg: str) -> None:
        """Append a trace message — mirrors NOVA's CognitiveState.record()."""
        self.trace.append(msg)

    @classmethod
    def from_hidden(
        cls,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> "CognitiveState":
        """Construct a minimal CognitiveState from a raw hidden tensor."""
        B, T, D = hidden.shape
        device = hidden.device
        if attention_mask is None:
            attention_mask = torch.ones(B, T, device=device)
        zeros = torch.zeros(B, T, device=device)
        return cls(
            hidden=hidden,
            uncertainty=zeros.clone(),
            epistemic_uncertainty=zeros.clone(),
            aleatoric_uncertainty=zeros.clone(),
            attention_mask=attention_mask,
        )


@dataclass
class UncertaintyBundle:
    """
    Mirrors NOVA's UncertaintyBundle.
    Scalar summary of uncertainty signals for logging and downstream use.
    """
    total: float
    epistemic: float
    aleatoric: float
    conformal_score: float
    entropy: float
    mutual_information: float
    should_abstain: bool
    abstention_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "epistemic": self.epistemic,
            "aleatoric": self.aleatoric,
            "conformal_score": self.conformal_score,
            "entropy": self.entropy,
            "mutual_information": self.mutual_information,
            "should_abstain": self.should_abstain,
            "abstention_reason": self.abstention_reason,
        }


@dataclass
class WriteGateOutput:
    """Output of EvidentialWriteGate.forward()."""
    alpha: torch.Tensor    # (batch, n_classes) Dirichlet concentration params
    belief: torch.Tensor   # (batch, n_classes) expected belief masses
    vacuity: torch.Tensor  # (batch,) scalar uncertainty score in [0, 1]

    def mean_vacuity(self) -> float:
        return self.vacuity.mean().item()


@dataclass
class MemoryStats:
    """Returned by EpistemicMemoryStore.stats()."""
    n_entries: int
    capacity: int
    capacity_used_pct: float
    mean_vacuity: float
    max_vacuity: float
    min_vacuity: float
    task_id_distribution: dict  # {task_id: count}

    def to_dict(self) -> dict:
        return {
            "n_entries": self.n_entries,
            "capacity": self.capacity,
            "capacity_used_pct": self.capacity_used_pct,
            "mean_vacuity": self.mean_vacuity,
            "max_vacuity": self.max_vacuity,
            "min_vacuity": self.min_vacuity,
            "task_id_distribution": self.task_id_distribution,
        }
