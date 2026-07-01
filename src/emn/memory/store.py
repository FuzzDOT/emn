"""
EpistemicMemoryStore
====================
Component 2 of EMN: Confidence-Weighted Eviction.

Key invariant: when the store is at capacity and a new memory arrives,
the entry with the HIGHEST vacuity (= most uncertain = least trustworthy)
is evicted.  This is NOT FIFO, NOT LRU, NOT random.

The store maintains a parallel numpy array of vacuity floats so that
argmax-eviction is O(N) without deserialising full entries.

For N <= 100_000 entries O(N) argmax on a float32 array takes ~0.1ms
on a modern CPU — fast enough for real-time use.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from emn.memory.entry import MemoryEntry
from emn.types import MemoryStats


class EpistemicMemoryStore:
    """
    Fixed-capacity memory store with evidential confidence-weighted eviction.

    Parameters
    ----------
    capacity    : int   — maximum number of memories
    d_model     : int   — embedding dimensionality
    vacuity_extractor : VacuityExtractor | None
                  If provided, vacuity is computed automatically on write.
                  If None, caller must supply vacuity explicitly.
    device      : str   — "cpu" or "cuda:N" for torch ops during retrieval

    Storage layout
    --------------
    _vectors  : np.ndarray  shape (capacity, d_model) float32  — memory matrix
    _vacuity  : np.ndarray  shape (capacity,)         float32  — parallel vacuity
    _entries  : List[Optional[MemoryEntry]]           — full entry objects
    _size     : int                                   — current occupancy
    """

    def __init__(
        self,
        capacity: int,
        d_model: int,
        vacuity_extractor=None,  # VacuityExtractor | None
        device: str = "cpu",
        eviction_strategy: str = "vacuity",   # "vacuity" | "random" | "lru" — ablation hook
        retrieval_vacuity_weight: float = 1.0, # ablation: 0.0 disables vacuity weighting
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")

        self.capacity = capacity
        self.d_model = d_model
        self.vacuity_extractor = vacuity_extractor
        self.device = device
        self.eviction_strategy = eviction_strategy
        self.retrieval_vacuity_weight = retrieval_vacuity_weight

        # Pre-allocate arrays for O(1) index operations
        self._vectors = np.zeros((capacity, d_model), dtype=np.float32)
        self._vacuity = np.ones(capacity, dtype=np.float32)   # init high = evictable
        self._timestamps = np.zeros(capacity, dtype=np.float64)
        self._entries: List[Optional[MemoryEntry]] = [None] * capacity
        self._size: int = 0

        # LRU support (for ablation)
        self._access_times = np.zeros(capacity, dtype=np.float64)

    # ------------------------------------------------------------------
    # Core write / evict
    # ------------------------------------------------------------------

    def write(
        self,
        vector: np.ndarray,
        task_id: str = "",
        metadata: Optional[dict] = None,
        vacuity: Optional[float] = None,
    ) -> MemoryEntry:
        """
        Write a new memory to the store.

        Parameters
        ----------
        vector   : (d_model,) float32
        task_id  : str
        metadata : dict | None
        vacuity  : float | None — if None, computed via vacuity_extractor

        Returns
        -------
        entry : MemoryEntry — the stored entry (with assigned vacuity)

        Raises
        ------
        RuntimeError if vacuity is None and no extractor is configured.
        """
        if vector.shape != (self.d_model,):
            raise ValueError(
                f"vector shape mismatch: expected ({self.d_model},), got {vector.shape}"
            )

        # Compute vacuity if not provided
        if vacuity is None:
            if self.vacuity_extractor is None:
                raise RuntimeError(
                    "vacuity not provided and no vacuity_extractor configured"
                )
            vacuity = self.vacuity_extractor.vacuity_from_numpy(vector)

        entry = MemoryEntry(
            vector=vector.astype(np.float32),
            vacuity=float(vacuity),
            timestamp=time.time(),
            task_id=task_id,
            metadata=metadata or {},
        )

        if self._size < self.capacity:
            idx = self._size
            self._size += 1
        else:
            idx = self._evict()

        self._vectors[idx] = entry.vector
        self._vacuity[idx] = entry.vacuity
        self._timestamps[idx] = entry.timestamp
        self._access_times[idx] = entry.timestamp
        self._entries[idx] = entry

        return entry

    def _evict(self) -> int:
        """
        Choose an index to evict and return it.

        Strategy dispatch:
        - "vacuity"  : evict argmax(vacuity)   — EMN default
        - "random"   : evict uniformly at random  — ablation baseline
        - "lru"      : evict least-recently-accessed — ablation baseline
        """
        if self.eviction_strategy == "vacuity":
            idx = int(np.argmax(self._vacuity[: self._size]))
        elif self.eviction_strategy == "random":
            idx = int(np.random.randint(0, self._size))
        elif self.eviction_strategy == "lru":
            idx = int(np.argmin(self._access_times[: self._size]))
        else:
            raise ValueError(f"Unknown eviction_strategy: {self.eviction_strategy!r}")
        return idx

    def evict_entry(self, entry_id: str) -> bool:
        """Evict a specific entry by ID. Returns True if found and removed."""
        for i in range(self._size):
            e = self._entries[i]
            if e is not None and e.entry_id == entry_id:
                self._remove_at(i)
                return True
        return False

    def _remove_at(self, idx: int) -> None:
        """Remove entry at idx by swapping with last slot."""
        last = self._size - 1
        if idx != last:
            self._vectors[idx] = self._vectors[last]
            self._vacuity[idx] = self._vacuity[last]
            self._timestamps[idx] = self._timestamps[last]
            self._access_times[idx] = self._access_times[last]
            self._entries[idx] = self._entries[last]
        self._entries[last] = None
        self._size -= 1

    # ------------------------------------------------------------------
    # Retrieval (delegates to UncertaintyWeightedRetriever logic inline
    # for simple brute-force; full retriever is in retrieval/retriever.py)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: np.ndarray,
        k: int = 5,
        return_scores: bool = False,
    ) -> List[MemoryEntry] | Tuple[List[MemoryEntry], np.ndarray]:
        """
        Retrieve top-k memories by uncertainty-weighted cosine similarity.

        score_i = cosine(query, memory_i) * (1 - vacuity_i * weight)

        Parameters
        ----------
        query         : (d_model,) float32
        k             : int
        return_scores : bool — if True return (entries, scores) tuple

        Returns
        -------
        List[MemoryEntry] — top-k entries, ranked by score descending
        (optionally with their scores)
        """
        if self._size == 0:
            return ([], np.array([])) if return_scores else []

        active = self._vectors[: self._size]
        active_vac = self._vacuity[: self._size]

        # Cosine similarity
        q_norm = query / (np.linalg.norm(query) + 1e-9)
        norms = np.linalg.norm(active, axis=1, keepdims=True) + 1e-9
        v_norm = active / norms
        cosine_sim = v_norm @ q_norm  # (size,)

        # Uncertainty-downweighted score
        score = cosine_sim * (1.0 - self.retrieval_vacuity_weight * active_vac)

        k_actual = min(k, self._size)
        top_idx = np.argpartition(score, -k_actual)[-k_actual:]
        top_idx = top_idx[np.argsort(score[top_idx])[::-1]]

        # Update access times for LRU
        now = time.time()
        for i in top_idx:
            self._access_times[i] = now

        entries = [self._entries[i] for i in top_idx]

        if return_scores:
            return entries, score[top_idx]
        return entries

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> MemoryStats:
        """Return summary statistics of the current store contents."""
        if self._size == 0:
            return MemoryStats(
                n_entries=0,
                capacity=self.capacity,
                capacity_used_pct=0.0,
                mean_vacuity=0.0,
                max_vacuity=0.0,
                min_vacuity=0.0,
                task_id_distribution={},
            )
        active_vac = self._vacuity[: self._size]
        task_ids = [
            e.task_id for e in self._entries[: self._size] if e is not None
        ]
        return MemoryStats(
            n_entries=self._size,
            capacity=self.capacity,
            capacity_used_pct=100.0 * self._size / self.capacity,
            mean_vacuity=float(active_vac.mean()),
            max_vacuity=float(active_vac.max()),
            min_vacuity=float(active_vac.min()),
            task_id_distribution=dict(Counter(task_ids)),
        )

    def all_entries(self) -> List[MemoryEntry]:
        """Return all live entries."""
        return [e for e in self._entries[: self._size] if e is not None]

    def all_vectors(self) -> np.ndarray:
        """Return (size, d_model) array of all live vectors."""
        return self._vectors[: self._size].copy()

    def all_vacuities(self) -> np.ndarray:
        """Return (size,) array of all live vacuity scores."""
        return self._vacuity[: self._size].copy()

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return (
            f"EpistemicMemoryStore("
            f"size={self._size}/{self.capacity}, "
            f"d_model={self.d_model}, "
            f"eviction={self.eviction_strategy})"
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the store to disk.

        Saves two files:
          <path>.npy   — vector + vacuity arrays (fast numpy binary)
          <path>.json  — entry metadata list
        """
        path = str(path)
        arrays = {
            "vectors": self._vectors[: self._size],
            "vacuity": self._vacuity[: self._size],
            "timestamps": self._timestamps[: self._size],
            "access_times": self._access_times[: self._size],
            "size": np.array([self._size]),
            "capacity": np.array([self.capacity]),
            "d_model": np.array([self.d_model]),
        }
        np.save(path + ".npy", arrays, allow_pickle=True)

        meta = {
            "entries": [
                e.to_dict() for e in self._entries[: self._size] if e is not None
            ],
            "eviction_strategy": self.eviction_strategy,
            "retrieval_vacuity_weight": self.retrieval_vacuity_weight,
        }
        with open(path + ".json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str, vacuity_extractor=None, device: str = "cpu") -> "EpistemicMemoryStore":
        """
        Load a previously saved store.

        Parameters
        ----------
        path              : str  — same path prefix used in save()
        vacuity_extractor : optional VacuityExtractor for new writes
        device            : str
        """
        path = str(path)
        arrays = np.load(path + ".npy", allow_pickle=True).item()

        size = int(arrays["size"][0])
        capacity = int(arrays["capacity"][0])
        d_model = int(arrays["d_model"][0])

        with open(path + ".json") as f:
            meta = json.load(f)

        store = cls(
            capacity=capacity,
            d_model=d_model,
            vacuity_extractor=vacuity_extractor,
            device=device,
            eviction_strategy=meta.get("eviction_strategy", "vacuity"),
            retrieval_vacuity_weight=meta.get("retrieval_vacuity_weight", 1.0),
        )
        store._size = size
        store._vectors[:size] = arrays["vectors"]
        store._vacuity[:size] = arrays["vacuity"]
        store._timestamps[:size] = arrays["timestamps"]
        store._access_times[:size] = arrays["access_times"]

        for i, ed in enumerate(meta["entries"]):
            store._entries[i] = MemoryEntry.from_dict(ed)

        return store

    # ------------------------------------------------------------------
    # Sampling (used by EMNPlugin for memory-protection loss)
    # ------------------------------------------------------------------

    def sample(
        self,
        n: int,
        strategy: str = "inverse_vacuity",
        seed: Optional[int] = None,
    ) -> List[MemoryEntry]:
        """
        Sample n entries from the store.

        Strategies
        ----------
        "inverse_vacuity"  : sample proportional to (1 - vacuity)
                             → prefer confident memories (default for loss)
        "uniform"          : uniform random

        Parameters
        ----------
        n        : int — number of entries to sample (clamped to _size)
        strategy : str
        seed     : int | None — for reproducibility

        Returns
        -------
        List[MemoryEntry]
        """
        if self._size == 0:
            return []
        rng = np.random.default_rng(seed)
        n = min(n, self._size)
        active_vac = self._vacuity[: self._size]

        if strategy == "inverse_vacuity":
            weights = 1.0 - active_vac
            weights = weights / (weights.sum() + 1e-9)
            indices = rng.choice(self._size, size=n, replace=False, p=weights)
        elif strategy == "uniform":
            indices = rng.choice(self._size, size=n, replace=False)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy!r}")

        return [self._entries[i] for i in indices if self._entries[i] is not None]
