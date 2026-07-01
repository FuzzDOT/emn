"""
Unit tests for EpistemicMemoryStore.
Tests: write/evict/retrieve correctness, eviction targets max-vacuity NOT FIFO,
serialisation round-trip.
"""

import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pytest
import numpy as np

from emn.memory.store import EpistemicMemoryStore
from emn.memory.entry import MemoryEntry


D = 64  # test d_model


@pytest.fixture
def store():
    return EpistemicMemoryStore(capacity=10, d_model=D)


def rand_vec(seed=None):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(D).astype(np.float32)


# ── Write ─────────────────────────────────────────────────────────────────────

def test_write_increases_size(store):
    assert len(store) == 0
    store.write(rand_vec(), vacuity=0.3)
    assert len(store) == 1
    store.write(rand_vec(), vacuity=0.5)
    assert len(store) == 2


def test_write_returns_entry(store):
    vec = rand_vec()
    entry = store.write(vec, vacuity=0.4, task_id="task_0", metadata={"x": 1})
    assert isinstance(entry, MemoryEntry)
    assert entry.vacuity == pytest.approx(0.4)
    assert entry.task_id == "task_0"
    assert entry.metadata["x"] == 1
    assert entry.d_model == D


def test_write_stores_correct_vector(store):
    vec = rand_vec(42)
    store.write(vec, vacuity=0.3)
    stored = store.all_vectors()
    np.testing.assert_allclose(stored[0], vec, rtol=1e-6)


def test_write_fills_to_capacity(store):
    for i in range(10):
        store.write(rand_vec(i), vacuity=float(i) / 10)
    assert len(store) == 10


# ── Eviction: argmax(vacuity), NOT FIFO ───────────────────────────────────────

def test_eviction_removes_highest_vacuity():
    """
    Critical invariant: when at capacity, the entry with the HIGHEST vacuity
    is evicted, regardless of insertion order.
    """
    store = EpistemicMemoryStore(capacity=5, d_model=D)

    # Insert 5 entries with known vacuities
    vacuities = [0.1, 0.9, 0.3, 0.5, 0.2]
    for v in vacuities:
        store.write(rand_vec(), vacuity=v)

    assert len(store) == 5

    # Insert a 6th — should evict the entry with vacuity=0.9
    store.write(rand_vec(), vacuity=0.05)

    assert len(store) == 5
    remaining_vacuities = sorted(store.all_vacuities().tolist())
    assert 0.9 not in remaining_vacuities, \
        "Entry with vacuity=0.9 should have been evicted"
    assert max(remaining_vacuities) <= 0.5, \
        f"Max remaining vacuity should be <= 0.5, got {max(remaining_vacuities)}"


def test_eviction_is_not_fifo():
    """First-inserted entry should survive if it has low vacuity."""
    store = EpistemicMemoryStore(capacity=3, d_model=D)
    e1 = store.write(rand_vec(0), vacuity=0.05)  # inserted first, very confident
    e2 = store.write(rand_vec(1), vacuity=0.95)  # high vacuity → eviction candidate
    e3 = store.write(rand_vec(2), vacuity=0.90)  # also high vacuity

    assert len(store) == 3

    # Insert 4th → should evict e2 (vacuity=0.95), NOT e1
    store.write(rand_vec(3), vacuity=0.2)

    entry_ids = {e.entry_id for e in store.all_entries()}
    assert e1.entry_id in entry_ids, "First entry (low vacuity) should not be evicted"
    assert e2.entry_id not in entry_ids, "Highest-vacuity entry should be evicted"


def test_eviction_is_not_lru():
    """Most recently accessed entry with high vacuity should still be evicted."""
    store = EpistemicMemoryStore(capacity=3, d_model=D)
    e_confident = store.write(rand_vec(0), vacuity=0.05)
    e_uncertain1 = store.write(rand_vec(1), vacuity=0.95)
    e_uncertain2 = store.write(rand_vec(2), vacuity=0.80)

    # Access e_uncertain1 (most recently accessed = would survive LRU)
    store.retrieve(rand_vec(99), k=1)

    # Insert new entry → should evict e_uncertain1 (highest vacuity), not oldest
    store.write(rand_vec(3), vacuity=0.2)

    entry_ids = {e.entry_id for e in store.all_entries()}
    assert e_confident.entry_id in entry_ids
    assert e_uncertain1.entry_id not in entry_ids, \
        "Highest-vacuity entry should be evicted regardless of access recency"


def test_eviction_strategy_random():
    """With random eviction strategy, store should not exceed capacity."""
    store = EpistemicMemoryStore(capacity=5, d_model=D, eviction_strategy="random")
    for i in range(10):
        store.write(rand_vec(i), vacuity=0.5)
    assert len(store) == 5


def test_eviction_strategy_lru():
    """LRU eviction: least-recently-accessed entry is evicted."""
    store = EpistemicMemoryStore(capacity=3, d_model=D, eviction_strategy="lru")
    for i in range(3):
        store.write(rand_vec(i), vacuity=0.5)

    # Access entry 0 (making it most recently used)
    entries_before = store.all_entries()
    store.retrieve(entries_before[0].vector, k=1)

    # Insert 4th — LRU should evict the least recently accessed
    store.write(rand_vec(99), vacuity=0.5)
    assert len(store) == 3


# ── Retrieval ─────────────────────────────────────────────────────────────────

def test_retrieve_returns_list(store):
    for i in range(5):
        store.write(rand_vec(i), vacuity=0.3)
    results = store.retrieve(rand_vec(99), k=3)
    assert isinstance(results, list)
    assert len(results) == 3


def test_retrieve_k_larger_than_size():
    store = EpistemicMemoryStore(capacity=10, d_model=D)
    for i in range(3):
        store.write(rand_vec(i), vacuity=0.3)
    results = store.retrieve(rand_vec(99), k=10)
    assert len(results) == 3  # clamps to available


def test_retrieve_empty_store():
    store = EpistemicMemoryStore(capacity=10, d_model=D)
    results = store.retrieve(rand_vec(), k=5)
    assert results == []


def test_retrieve_prefers_low_vacuity():
    """
    Uncertainty-weighted score: confident memories should score higher
    than uncertain ones even when they have similar cosine similarity.
    """
    store = EpistemicMemoryStore(capacity=5, d_model=D, retrieval_vacuity_weight=1.0)
    query = np.ones(D, dtype=np.float32) / np.sqrt(D)

    # Two memories equally close to query, but different vacuities
    confident_vec = query.copy()  # cosine = 1.0
    uncertain_vec = query.copy()  # cosine = 1.0

    e_confident = store.write(confident_vec, vacuity=0.05)
    e_uncertain  = store.write(uncertain_vec, vacuity=0.95)

    results, scores = store.retrieve(query, k=2, return_scores=True)
    assert results[0].entry_id == e_confident.entry_id, \
        "Confident memory should rank first in retrieval"
    assert scores[0] > scores[1], "Confident memory should have higher retrieval score"


def test_retrieve_with_scores():
    store = EpistemicMemoryStore(capacity=5, d_model=D)
    for i in range(5):
        store.write(rand_vec(i), vacuity=0.3)
    results, scores = store.retrieve(rand_vec(99), k=3, return_scores=True)
    assert len(results) == 3
    assert len(scores) == 3
    assert isinstance(scores, np.ndarray)
    # Scores should be in descending order
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), \
        "Scores should be sorted descending"


# ── Statistics ────────────────────────────────────────────────────────────────

def test_stats_empty():
    store = EpistemicMemoryStore(capacity=10, d_model=D)
    stats = store.stats()
    assert stats.n_entries == 0
    assert stats.capacity == 10


def test_stats_correct(store):
    for i in range(5):
        store.write(rand_vec(i), vacuity=float(i) / 10, task_id=f"task_{i % 2}")
    stats = store.stats()
    assert stats.n_entries == 5
    assert stats.capacity == 10
    assert stats.capacity_used_pct == pytest.approx(50.0)
    assert stats.mean_vacuity == pytest.approx(np.mean([0.0, 0.1, 0.2, 0.3, 0.4]), abs=1e-4)
    assert stats.task_id_distribution["task_0"] == 3
    assert stats.task_id_distribution["task_1"] == 2


# ── Sampling ──────────────────────────────────────────────────────────────────

def test_sample_inverse_vacuity(store):
    for i in range(10):
        store.write(rand_vec(i), vacuity=float(i) / 10)
    sampled = store.sample(n=5, strategy="inverse_vacuity", seed=42)
    assert len(sampled) == 5
    # All sampled should be valid entries
    for e in sampled:
        assert isinstance(e, MemoryEntry)


def test_sample_uniform(store):
    for i in range(10):
        store.write(rand_vec(i), vacuity=0.5)
    sampled = store.sample(n=5, strategy="uniform", seed=42)
    assert len(sampled) == 5


def test_sample_n_larger_than_size():
    store = EpistemicMemoryStore(capacity=10, d_model=D)
    for i in range(3):
        store.write(rand_vec(i), vacuity=0.3)
    sampled = store.sample(n=10)
    assert len(sampled) == 3


def test_sample_empty_store():
    store = EpistemicMemoryStore(capacity=10, d_model=D)
    sampled = store.sample(n=5)
    assert sampled == []


# ── Serialisation round-trip ──────────────────────────────────────────────────

def test_save_load_roundtrip():
    store = EpistemicMemoryStore(capacity=10, d_model=D)
    vecs = [rand_vec(i) for i in range(5)]
    vacuities = [0.1, 0.3, 0.5, 0.7, 0.9]
    entries = []
    for v, vac in zip(vecs, vacuities):
        e = store.write(v, vacuity=vac, task_id="task_0", metadata={"idx": len(entries)})
        entries.append(e)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "store")
        store.save(path)

        # Load it back
        loaded = EpistemicMemoryStore.load(path)

        assert len(loaded) == 5
        np.testing.assert_allclose(
            sorted(loaded.all_vacuities().tolist()),
            sorted(vacuities),
            rtol=1e-5,
        )
        loaded_vecs = loaded.all_vectors()
        # All original vectors should be recoverable
        for orig in vecs:
            assert any(
                np.allclose(orig, loaded_vecs[i], rtol=1e-5)
                for i in range(len(loaded))
            ), "Vector not found after round-trip"


def test_save_creates_two_files():
    store = EpistemicMemoryStore(capacity=5, d_model=D)
    store.write(rand_vec(), vacuity=0.3)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "store")
        store.save(path)
        assert Path(path + ".npy").exists()
        assert Path(path + ".json").exists()


def test_eviction_strategy_preserved_across_load():
    store = EpistemicMemoryStore(capacity=5, d_model=D, eviction_strategy="lru")
    store.write(rand_vec(), vacuity=0.3)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "store")
        store.save(path)
        loaded = EpistemicMemoryStore.load(path)
        assert loaded.eviction_strategy == "lru"


# ── Error handling ────────────────────────────────────────────────────────────

def test_write_wrong_shape_raises():
    store = EpistemicMemoryStore(capacity=5, d_model=D)
    with pytest.raises(ValueError):
        store.write(np.zeros(D + 1, dtype=np.float32), vacuity=0.3)


def test_invalid_eviction_strategy_raises():
    store = EpistemicMemoryStore(capacity=5, d_model=D, eviction_strategy="invalid")
    for i in range(5):
        store.write(rand_vec(i), vacuity=0.5)
    with pytest.raises(ValueError):
        store.write(rand_vec(99), vacuity=0.5)  # triggers eviction


def test_vacuity_out_of_range_raises():
    store = EpistemicMemoryStore(capacity=5, d_model=D)
    with pytest.raises(ValueError):
        store.write(rand_vec(), vacuity=1.5)
    with pytest.raises(ValueError):
        store.write(rand_vec(), vacuity=-0.1)
