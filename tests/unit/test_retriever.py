"""
Unit tests for UncertaintyWeightedRetriever.
Tests: score formula, top-k count, both backends agree.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pytest
import numpy as np

from emn.memory.store import EpistemicMemoryStore
from emn.retrieval.retriever import UncertaintyWeightedRetriever


D = 64


def make_store(n: int = 10, seed: int = 0) -> EpistemicMemoryStore:
    rng = np.random.default_rng(seed)
    store = EpistemicMemoryStore(capacity=100, d_model=D)
    for i in range(n):
        vec = rng.standard_normal(D).astype(np.float32)
        store.write(vec, vacuity=rng.uniform(0.0, 1.0))
    return store


def make_retriever(n: int = 10, vacuity_weight: float = 1.0, backend: str = "brute") -> UncertaintyWeightedRetriever:
    return UncertaintyWeightedRetriever(
        store=make_store(n),
        backend=backend,
        vacuity_weight=vacuity_weight,
    )


# ── Score formula ─────────────────────────────────────────────────────────────

def test_score_formula_correctness():
    """Verify: score = cosine(q, m) * (1 - vacuity)."""
    store = EpistemicMemoryStore(capacity=5, d_model=D)
    rng = np.random.default_rng(42)

    # Add a known memory
    mem_vec = rng.standard_normal(D).astype(np.float32)
    vacuity = 0.4
    store.write(mem_vec, vacuity=vacuity)

    query = rng.standard_normal(D).astype(np.float32)

    retriever = UncertaintyWeightedRetriever(store=store, vacuity_weight=1.0)
    _, scores = retriever.retrieve(query, k=1, return_scores=True)

    # Manually compute expected score
    q_norm = query / (np.linalg.norm(query) + 1e-9)
    m_norm = mem_vec / (np.linalg.norm(mem_vec) + 1e-9)
    cosine = float(q_norm @ m_norm)
    expected_score = cosine * (1.0 - vacuity)

    assert scores[0] == pytest.approx(expected_score, abs=1e-4), \
        f"Score formula wrong: got {scores[0]:.4f}, expected {expected_score:.4f}"


def test_zero_vacuity_weight_equals_pure_cosine():
    """vacuity_weight=0.0 should give pure cosine retrieval."""
    store = EpistemicMemoryStore(capacity=5, d_model=D)
    rng = np.random.default_rng(0)

    vecs = [rng.standard_normal(D).astype(np.float32) for _ in range(5)]
    for v in vecs:
        store.write(v, vacuity=rng.uniform(0.0, 1.0))

    query = rng.standard_normal(D).astype(np.float32)
    retriever = UncertaintyWeightedRetriever(store=store, vacuity_weight=0.0)
    _, scores_no_vac = retriever.retrieve(query, k=5, return_scores=True)

    # Manually compute pure cosine scores
    vectors = store.all_vectors()
    q_norm = query / (np.linalg.norm(query) + 1e-9)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    cosines = (vectors / norms) @ q_norm
    top5 = np.sort(cosines)[::-1][:5]

    np.testing.assert_allclose(scores_no_vac, top5, rtol=1e-4, atol=1e-4)


def test_full_vacuity_weight_downweights_uncertain():
    """High vacuity should lower retrieval score vs low vacuity with same similarity."""
    store = EpistemicMemoryStore(capacity=5, d_model=D)
    query = np.ones(D, dtype=np.float32) / np.sqrt(D)  # unit vector

    e_conf = store.write(query.copy(), vacuity=0.05)   # same direction, confident
    e_unc  = store.write(query.copy(), vacuity=0.95)   # same direction, uncertain

    retriever = UncertaintyWeightedRetriever(store=store, vacuity_weight=1.0)
    results, scores = retriever.retrieve(query, k=2, return_scores=True)

    assert results[0].entry_id == e_conf.entry_id, \
        "Confident entry should rank first"
    assert scores[0] > scores[1], \
        "Confident entry should have higher score"


# ── Top-k count ───────────────────────────────────────────────────────────────

def test_top_k_exact_count():
    retriever = make_retriever(n=20)
    query = np.random.randn(D).astype(np.float32)
    results = retriever.retrieve(query, k=5)
    assert len(results) == 5


def test_top_k_clamped_to_store_size():
    retriever = make_retriever(n=3)
    query = np.random.randn(D).astype(np.float32)
    results = retriever.retrieve(query, k=100)
    assert len(results) == 3


def test_top_k_one():
    retriever = make_retriever(n=10)
    query = np.random.randn(D).astype(np.float32)
    results = retriever.retrieve(query, k=1)
    assert len(results) == 1


def test_empty_store():
    store = EpistemicMemoryStore(capacity=10, d_model=D)
    retriever = UncertaintyWeightedRetriever(store=store)
    results = retriever.retrieve(np.random.randn(D).astype(np.float32), k=5)
    assert results == []


# ── Scores are sorted ─────────────────────────────────────────────────────────

def test_scores_are_descending():
    retriever = make_retriever(n=20)
    query = np.random.randn(D).astype(np.float32)
    _, scores = retriever.retrieve(query, k=10, return_scores=True)
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], \
            f"Scores not sorted at index {i}: {scores[i]:.4f} < {scores[i+1]:.4f}"


# ── Brute vs FAISS agreement ──────────────────────────────────────────────────

def test_brute_and_faiss_top1_agree():
    """Both backends should return the same top-1 entry."""
    pytest.importorskip("faiss")

    store = make_store(n=50, seed=7)
    rng = np.random.default_rng(7)
    query = rng.standard_normal(D).astype(np.float32)

    retriever_brute = UncertaintyWeightedRetriever(store=store, backend="brute")
    retriever_faiss = UncertaintyWeightedRetriever(store=store, backend="faiss")

    results_brute = retriever_brute.retrieve(query, k=1)
    results_faiss = retriever_faiss.retrieve(query, k=1, backend="faiss")

    assert results_brute[0].entry_id == results_faiss[0].entry_id, \
        "Brute and FAISS backends disagree on top-1 result"


def test_brute_and_faiss_top5_overlap():
    """Top-5 from both backends should have >= 4 entries in common."""
    pytest.importorskip("faiss")

    store = make_store(n=100, seed=13)
    rng = np.random.default_rng(13)
    query = rng.standard_normal(D).astype(np.float32)

    r_brute = UncertaintyWeightedRetriever(store=store, backend="brute")
    r_faiss = UncertaintyWeightedRetriever(store=store, backend="faiss")

    ids_brute = {e.entry_id for e in r_brute.retrieve(query, k=5)}
    ids_faiss = {e.entry_id for e in r_faiss.retrieve(query, k=5)}

    overlap = len(ids_brute & ids_faiss)
    assert overlap >= 4, \
        f"Brute and FAISS top-5 overlap only {overlap}/5 entries"


# ── Batch retrieval ───────────────────────────────────────────────────────────

def test_batch_retrieve_shape():
    retriever = make_retriever(n=20)
    rng = np.random.default_rng(0)
    queries = rng.standard_normal((5, D)).astype(np.float32)
    all_results = retriever.retrieve_batch(queries, k=3)
    assert len(all_results) == 5
    for results in all_results:
        assert len(results) == 3


# ── score_all ─────────────────────────────────────────────────────────────────

def test_score_all_length():
    retriever = make_retriever(n=10)
    query = np.random.randn(D).astype(np.float32)
    scores = retriever.score_all(query)
    assert len(scores) == 10


def test_score_all_empty():
    store = EpistemicMemoryStore(capacity=10, d_model=D)
    retriever = UncertaintyWeightedRetriever(store=store)
    scores = retriever.score_all(np.random.randn(D).astype(np.float32))
    assert len(scores) == 0
