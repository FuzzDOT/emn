"""
Integration tests for EMNWrappedCausalLM.
Uses a tiny mock model to avoid requiring TinyLlama for CI.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pytest
import torch
import torch.nn as nn
import numpy as np

from emn.llm.wrapped_lm import EMNWrappedCausalLM
from emn.memory.store import EpistemicMemoryStore
from emn.retrieval.retriever import UncertaintyWeightedRetriever


D = 64


# ── Mock text encoder for tests ───────────────────────────────────────────────

class MockEncoder:
    """Deterministic mock sentence encoder — accepts all sentence-transformers kwargs."""
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False,
               convert_to_numpy=True, convert_to_tensor=False, batch_size=32, **kwargs):
        rng = np.random.default_rng(hash(texts[0]) % (2**31))
        emb = rng.standard_normal(384).astype(np.float32)
        if normalize_embeddings:
            emb = emb / (np.linalg.norm(emb) + 1e-9)
        return np.array([emb])


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def wrapped_lm():
    """EMNWrappedCausalLM with a mock encoder (no actual LLM loaded)."""
    lm = EMNWrappedCausalLM(backend="hf", d_model=384, memory_capacity=50)
    lm._encoder = MockEncoder()
    return lm


# ── add_memory tests ──────────────────────────────────────────────────────────

def test_add_memory_stores_entry(wrapped_lm):
    entry = wrapped_lm.add_memory("The Eiffel Tower is in Paris.")
    assert len(wrapped_lm.store) == 1
    assert entry.metadata.get("source_text") == "The Eiffel Tower is in Paris."


def test_add_memory_multiple(wrapped_lm):
    for i in range(10):
        wrapped_lm.add_memory(f"Fact number {i}.")
    assert len(wrapped_lm.store) == 10


def test_add_memory_returns_valid_vacuity(wrapped_lm):
    entry = wrapped_lm.add_memory("Water freezes at 0°C.")
    assert 0.0 <= entry.vacuity <= 1.0


def test_add_memory_with_task_id(wrapped_lm):
    entry = wrapped_lm.add_memory("Test fact.", task_id="test_task")
    assert entry.task_id == "test_task"


def test_add_memory_with_metadata(wrapped_lm):
    entry = wrapped_lm.add_memory("Test.", metadata={"source": "Wikipedia"})
    assert entry.metadata.get("source") == "Wikipedia"
    assert entry.metadata.get("source_text") == "Test."  # also stored


# ── retrieve_memory tests ─────────────────────────────────────────────────────

def test_retrieve_memory_returns_entries(wrapped_lm):
    for i in range(10):
        wrapped_lm.add_memory(f"Fact {i}.")
    results = wrapped_lm.retrieve_memory("What is fact 5?", k=3)
    assert len(results) == 3


def test_retrieve_memory_empty_store(wrapped_lm):
    results = wrapped_lm.retrieve_memory("Any question.", k=5)
    assert results == []


def test_retrieve_with_scores(wrapped_lm):
    for i in range(5):
        wrapped_lm.add_memory(f"Memory {i}.")
    entries, scores = wrapped_lm.retrieve_memory("query", k=3, return_scores=True)
    assert len(entries) == 3
    assert len(scores) == 3


# ── memory_stats tests ────────────────────────────────────────────────────────

def test_memory_stats(wrapped_lm):
    for i in range(5):
        wrapped_lm.add_memory(f"Fact {i}.")
    stats = wrapped_lm.memory_stats()
    assert stats.n_entries == 5
    assert stats.capacity == 50


# ── save/load memory tests ────────────────────────────────────────────────────

def test_save_load_memory(wrapped_lm, tmp_path):
    for i in range(5):
        wrapped_lm.add_memory(f"Memory {i}.")
    path = str(tmp_path / "lm_memory")
    wrapped_lm.save_memory(path)
    wrapped_lm.load_memory(path)
    assert len(wrapped_lm.store) == 5


# ── _build_augmented_prompt tests ────────────────────────────────────────────

def test_augmented_prompt_contains_memories(wrapped_lm):
    wrapped_lm.add_memory("Paris is in France.")
    entries = wrapped_lm.store.all_entries()
    prompt = wrapped_lm._build_augmented_prompt("Where is Paris?", entries)
    assert "Paris is in France." in prompt
    assert "memory" in prompt.lower() or "context" in prompt.lower()


def test_augmented_prompt_no_memories(wrapped_lm):
    result = wrapped_lm._build_augmented_prompt("Where is Paris?", [])
    assert result == "Where is Paris?"


# ── repr ──────────────────────────────────────────────────────────────────────

def test_repr(wrapped_lm):
    r = repr(wrapped_lm)
    assert "EMNWrappedCausalLM" in r
    assert "backend='hf'" in r