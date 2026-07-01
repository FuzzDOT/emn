"""
End-to-end tests: full EMN pipeline from raw data to metrics.
Exercises the complete chain without real models.
"""

import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pytest
import numpy as np
import torch

from emn.gates.write_gate import EvidentialWriteGate
from emn.memory.store import EpistemicMemoryStore
from emn.retrieval.retriever import UncertaintyWeightedRetriever
from emn.benchmarks.confabulation.dataset import generate_dataset, save_dataset, load_dataset
from emn.utils.reproducibility import set_seed, get_git_hash, build_run_metadata
from emn.utils.metrics import (
    classification_metrics,
    continual_learning_metrics,
    expected_calibration_error,
    auroc,
    auprc,
)
from emn.utils.tables import make_table1, make_table2, make_table3


D = 64


# ── Reproducibility ───────────────────────────────────────────────────────────

def test_set_seed_is_deterministic():
    set_seed(42)
    a = np.random.randn(10)
    set_seed(42)
    b = np.random.randn(10)
    np.testing.assert_allclose(a, b)


def test_get_git_hash_returns_string():
    h = get_git_hash()
    assert isinstance(h, str)
    assert len(h) > 0


def test_build_run_metadata(tmp_path):
    meta = build_run_metadata(
        experiment_name="test",
        seed=42,
        config={"lr": 0.01, "epochs": 5},
        output_dir=str(tmp_path),
    )
    assert meta["seed"] == 42
    assert meta["experiment"] == "test"
    assert "hardware" in meta
    assert (tmp_path / "run_metadata.json").exists()


# ── Dataset generation ────────────────────────────────────────────────────────

def test_generate_confabulation_dataset():
    items = generate_dataset(seed=42)
    assert len(items) == 1000
    assert all(1 <= item.severity <= 5 for item in items)
    assert len({item.fact_id for item in items}) == 1000  # all unique IDs


def test_dataset_determinism():
    items_a = generate_dataset(seed=42)
    items_b = generate_dataset(seed=42)
    assert all(a.fact_id == b.fact_id for a, b in zip(items_a, items_b))
    assert all(a.correct_answer == b.correct_answer for a, b in zip(items_a, items_b))


def test_dataset_save_load_roundtrip(tmp_path):
    items = generate_dataset(seed=42)
    path = str(tmp_path / "dataset.jsonl")
    save_dataset(items, path)
    loaded = load_dataset(path)
    assert len(loaded) == len(items)
    for orig, load in zip(items, loaded):
        assert orig.fact_id == load.fact_id
        assert orig.severity == load.severity
        assert orig.correct_answer == load.correct_answer


def test_dataset_all_severities_present():
    items = generate_dataset(seed=42)
    severities = {item.severity for item in items}
    assert severities == {1, 2, 3, 4, 5}


# ── Metrics ───────────────────────────────────────────────────────────────────

def test_classification_metrics_perfect():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    m = classification_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)


def test_classification_metrics_all_wrong():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 1])
    m = classification_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(0.0)


def test_continual_learning_metrics():
    """
    Simple 3-task accuracy matrix.
    Diagonal: per-task accuracy after training on that task.
    Last row: final accuracy on all tasks.
    """
    # Simulated: model learns each task but forgets slightly
    acc = np.array([
        [0.9, 0.0, 0.0],
        [0.7, 0.8, 0.0],
        [0.5, 0.6, 0.9],
    ])
    m = continual_learning_metrics(acc)
    assert m["average_accuracy"] == pytest.approx((0.5 + 0.6 + 0.9) / 3, abs=1e-3)
    # BWT: forgetting on tasks 0 and 1
    # task 0: acc[2,0] - acc[0,0] = 0.5 - 0.9 = -0.4
    # task 1: acc[2,1] - acc[1,1] = 0.6 - 0.8 = -0.2
    # BWT = mean = -0.3
    assert m["backward_transfer"] == pytest.approx(-0.3, abs=1e-3)


def test_ece_perfectly_calibrated():
    """A perfectly calibrated model should have ECE near 0."""
    rng = np.random.default_rng(0)
    n = 1000
    confidences = rng.uniform(0.0, 1.0, n)
    # Match accuracy to confidence exactly (perfectly calibrated by construction)
    is_correct = rng.uniform(0, 1, n) < confidences
    ece = expected_calibration_error(confidences, is_correct.astype(float))
    assert ece < 0.05, f"ECE for calibrated model should be < 0.05, got {ece:.4f}"


def test_auroc_and_auprc():
    rng = np.random.default_rng(0)
    n = 200
    scores = rng.uniform(0, 1, n)
    labels = (scores > 0.5).astype(int) + rng.integers(0, 2, n) * 0
    labels = np.clip(labels, 0, 1)
    a = auroc(scores, labels)
    ap = auprc(scores, labels)
    assert 0.0 <= a <= 1.0
    assert 0.0 <= ap <= 1.0


# ── Table generation ──────────────────────────────────────────────────────────

def test_make_table1(tmp_path):
    results = {
        "standard_transformer": {"accuracy": 0.55, "accuracy_std": 0.02, "precision": 0.55,
                                   "precision_std": 0.02, "recall": 0.53, "recall_std": 0.02,
                                   "f1": 0.54, "f1_std": 0.02},
        "emn": {"accuracy": 0.72, "accuracy_std": 0.01, "precision": 0.71,
                 "precision_std": 0.01, "recall": 0.73, "recall_std": 0.01,
                 "f1": 0.72, "f1_std": 0.01},
    }
    make_table1(results, output_dir=str(tmp_path))
    assert (tmp_path / "table1_selective_forgetting.csv").exists()
    assert (tmp_path / "table1_selective_forgetting.tex").exists()


def test_make_table2(tmp_path):
    results = {
        "sequential_ft": {"average_accuracy": 0.32, "average_accuracy_std": 0.02,
                           "backward_transfer": -0.28, "backward_transfer_std": 0.01,
                           "forward_transfer": 0.05, "forward_transfer_std": 0.01,
                           "forgetting": 0.35, "forgetting_std": 0.01},
        "emn": {"average_accuracy": 0.55, "average_accuracy_std": 0.01,
                 "backward_transfer": -0.08, "backward_transfer_std": 0.01,
                 "forward_transfer": 0.10, "forward_transfer_std": 0.01,
                 "forgetting": 0.14, "forgetting_std": 0.01},
    }
    make_table2(results, output_dir=str(tmp_path))
    assert (tmp_path / "table2_continual_learning.csv").exists()
    assert (tmp_path / "table2_continual_learning.tex").exists()


def test_make_table3(tmp_path):
    results = {
        "softmax": {"confabulation_rate": 0.38, "hedging_rate": 0.15, "update_rate": 0.62,
                    "auroc": 0.72, "auprc": 0.68, "ece": 0.14},
        "emn_vacuity": {"confabulation_rate": 0.21, "hedging_rate": 0.28, "update_rate": 0.79,
                         "auroc": 0.85, "auprc": 0.82, "ece": 0.06},
    }
    make_table3(results, output_dir=str(tmp_path))
    assert (tmp_path / "table3_confabulation.csv").exists()
    assert (tmp_path / "table3_confabulation.tex").exists()


# ── Full mini pipeline ────────────────────────────────────────────────────────

def test_write_gate_to_store_to_retriever_pipeline():
    """Complete mini pipeline: gate → store → retriever → verify."""
    set_seed(42)
    gate = EvidentialWriteGate(d_model=D)
    store = EpistemicMemoryStore(capacity=20, d_model=D)
    retriever = UncertaintyWeightedRetriever(store=store)

    rng = np.random.default_rng(42)
    n = 15

    for i in range(n):
        vec = rng.standard_normal(D).astype(np.float32)
        t = torch.from_numpy(vec).unsqueeze(0)
        with torch.no_grad():
            gate_out = gate(t)
        vacuity = float(gate_out.vacuity.squeeze())
        store.write(vec, vacuity=vacuity, task_id=f"task_{i % 3}")

    assert len(store) == n

    query = rng.standard_normal(D).astype(np.float32)
    results, scores = retriever.retrieve(query, k=5, return_scores=True)

    assert len(results) == 5
    assert all(0.0 <= e.vacuity <= 1.0 for e in results)
    # Scores descending
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

    stats = store.stats()
    assert stats.n_entries == n
    assert len(stats.task_id_distribution) == 3
