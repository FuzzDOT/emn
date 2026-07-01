"""
MemoryEntry
===========
The atomic unit stored in EpistemicMemoryStore.
Every memory carries its vacuity score through its full lifecycle:
write → retrieval → eviction.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class MemoryEntry:
    """
    A single memory slot in the Epistemic Memory Store.

    Fields
    ------
    vector    : float32 numpy array, shape (d_model,)
                The actual memory embedding.
    vacuity   : float in [0, 1]
                Vacuity score at write time (from NOVA's EvidentialHead).
                Low  = high confidence  = protected from eviction.
                High = low confidence   = first candidate for eviction.
    timestamp : float
                Unix timestamp at write time (time.time()).
    task_id   : str
                Task label for continual learning bookkeeping.
                "" for non-CL usage.
    metadata  : dict
                Arbitrary JSON-serialisable payload (source text, labels, etc.)
    entry_id  : str
                UUID4 string — globally unique identifier for this entry.
    """

    vector: np.ndarray
    vacuity: float
    timestamp: float = field(default_factory=time.time)
    task_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if not isinstance(self.vector, np.ndarray):
            raise TypeError(f"vector must be np.ndarray, got {type(self.vector)}")
        if self.vector.dtype != np.float32:
            self.vector = self.vector.astype(np.float32)
        if self.vector.ndim != 1:
            raise ValueError(f"vector must be 1-D, got shape {self.vector.shape}")
        if not (0.0 <= self.vacuity <= 1.0):
            raise ValueError(f"vacuity must be in [0,1], got {self.vacuity}")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict (vector stored as list)."""
        return {
            "entry_id": self.entry_id,
            "vector": self.vector.tolist(),
            "vacuity": float(self.vacuity),
            "timestamp": float(self.timestamp),
            "task_id": self.task_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        """Reconstruct from a dict produced by to_dict()."""
        return cls(
            vector=np.array(d["vector"], dtype=np.float32),
            vacuity=float(d["vacuity"]),
            timestamp=float(d["timestamp"]),
            task_id=d.get("task_id", ""),
            metadata=d.get("metadata", {}),
            entry_id=d.get("entry_id", str(uuid.uuid4())),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "MemoryEntry":
        return cls.from_dict(json.loads(s))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def d_model(self) -> int:
        return self.vector.shape[0]

    def confidence(self) -> float:
        """1 - vacuity — higher = more confident."""
        return 1.0 - self.vacuity

    def __repr__(self) -> str:
        return (
            f"MemoryEntry(id={self.entry_id[:8]}, "
            f"d={self.d_model}, "
            f"vacuity={self.vacuity:.4f}, "
            f"task={self.task_id!r})"
        )
