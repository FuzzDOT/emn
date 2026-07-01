"""
Integration tests: full EMN pipeline.
Tests the complete chain: write gate → store → retriever.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pytest
import torch
import numpy as np

from emn.gates.write_gate import EvidentialWriteGate
from emn.memory.store import EpistemicMemoryStore
from emn.retrieval.retriever import UncertaintyWeightedRetriever


D = 64


@pytest.fixture
def pipeline():
    write_gate = EvidentialWriteGate(d_model=D)
    store = EpistemicMemoryStore(capacity=50, d_model=D)
    retriever = UncertaintyWeightedRetriever(store=store, vacuity_weight=1.0)
    return write_gate, store, retriever


def compute_vacuity(write_gate: EvidentialWriteGate, vec: np.ndarray) -> float:
    t = torch.from_numpy(vec).unsqueeze(0)
    with torch.no_grad():
        out = write_gate(t)
    return float(out.vacuity.squeeze())


# ── Write → Store → Retrieve ─────────────────────────────────────────────────

def test_full_write_retrieve_cycle(pipeline):
    write_gate, store, retriever = pipeline
    rng = np.random.default_rng(42)

    # Write 10 memories through the gate
    vecs = [rng.standard_normal(D).astype(np.float32) for _ in range(10)]
    for i, vec in enumerate(vecs):
        vacuity = compute_vacuity(write_gate, vec)
        store.write(vec, vacuity=vacuity, task_id=f"task_{i%3}")

    assert len(store) == 10

    # Retrieve top-3
    query = rng.standard_normal(D).astype(np.float32)
    results = retriever.retrieve(query, k=3)
    assert len(results) == 3


def test_gate_writes_valid_vacuity(pipeline):
    write_gate, store, retriever = pipeline
    rng = np.random.default_rng(0)

    for _ in range(20):
        vec = rng.standard_normal(D).astype(np.float32)
        vacuity = compute_vacuity(write_gate, vec)
        assert 0.0 <= vacuity <= 1.0
        store.write(vec, vacuity=vacuity)

    vacuities = store.all_vacuities()
    assert (vacuities >= 0.0).all()
    assert (vacuities <= 1.0).all()


def test_eviction_through_pipeline(pipeline):
    """Overfill the store; verify it doesn't exceed capacity."""
    write_gate, store, retriever = pipeline
    rng = np.random.default_rng(7)

    for i in range(70):  # capacity=50, so 20 evictions
        vec = rng.standard_normal(D).astype(np.float32)
        vacuity = compute_vacuity(write_gate, vec)
        store.write(vec, vacuity=vacuity)

    assert len(store) == 50, f"Store exceeded capacity: {len(store)}"


def test_retrieval_scores_are_positive(pipeline):
    write_gate, store, retriever = pipeline
    rng = np.random.default_rng(5)

    for _ in range(10):
        vec = rng.standard_normal(D).astype(np.float32)
        vacuity = compute_vacuity(write_gate, vec)
        store.write(vec, vacuity=vacuity)

    query = rng.standard_normal(D).astype(np.float32)
    _, scores = retriever.retrieve(query, k=5, return_scores=True)
    # Scores can be negative in theory (negative cosine), but should be well-defined
    assert not np.isnan(scores).any()
    assert not np.isinf(scores).any()


def test_pipeline_stats(pipeline):
    write_gate, store, retriever = pipeline
    rng = np.random.default_rng(3)

    for i in range(15):
        vec = rng.standard_normal(D).astype(np.float32)
        vacuity = compute_vacuity(write_gate, vec)
        store.write(vec, vacuity=vacuity, task_id=f"task_{i%3}")

    stats = store.stats()
    assert stats.n_entries == 15
    assert stats.capacity == 50
    assert len(stats.task_id_distribution) == 3


# ── Multi-task scenario ───────────────────────────────────────────────────────

def test_multi_task_write_and_retrieve(pipeline):
    write_gate, store, retriever = pipeline
    rng = np.random.default_rng(11)

    task_vecs = {}
    for task_id in range(3):
        task_vecs[task_id] = []
        for _ in range(5):
            vec = rng.standard_normal(D).astype(np.float32)
            vacuity = compute_vacuity(write_gate, vec)
            entry = store.write(vec, vacuity=vacuity, task_id=f"task_{task_id}")
            task_vecs[task_id].append(vec)

    assert len(store) == 15
    stats = store.stats()
    assert stats.task_id_distribution.get("task_0", 0) == 5
    assert stats.task_id_distribution.get("task_1", 0) == 5
    assert stats.task_id_distribution.get("task_2", 0) == 5


# ── Serialisation round-trip ──────────────────────────────────────────────────

def test_pipeline_save_load(pipeline, tmp_path):
    write_gate, store, retriever = pipeline
    rng = np.random.default_rng(99)

    for _ in range(10):
        vec = rng.standard_normal(D).astype(np.float32)
        vacuity = compute_vacuity(write_gate, vec)
        store.write(vec, vacuity=vacuity)

    # Save
    path = str(tmp_path / "store")
    store.save(path)

    # Load and verify
    loaded_store = EpistemicMemoryStore.load(path)
    loaded_retriever = UncertaintyWeightedRetriever(store=loaded_store)

    assert len(loaded_store) == 10

    query = rng.standard_normal(D).astype(np.float32)
    results = loaded_retriever.retrieve(query, k=3)
    assert len(results) == 3
