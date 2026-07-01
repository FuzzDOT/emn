"""
Shared metric helpers for EMN experiments.

Covers:
- Classification metrics (accuracy, precision, recall, F1)
- Continual learning metrics (BWT, FWT, Average Accuracy, Forgetting)
- Calibration metrics (ECE)
- Ranking metrics (AUROC, AUPRC)
- Statistical tests (t-test, bootstrap CI)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.stats


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Accuracy, precision, recall, F1 (binary or macro-averaged)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float((y_true == y_pred).mean())
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Continual learning metrics
# ---------------------------------------------------------------------------

def backward_transfer(
    accuracy_matrix: np.ndarray,
) -> float:
    """
    Backward Transfer (BWT).
    BWT = (1 / (T-1)) * sum_{i=1}^{T-1} (R_{T,i} - R_{i,i})

    Parameters
    ----------
    accuracy_matrix : (T, T) float
        accuracy_matrix[i, j] = accuracy on task j after training on task i

    Returns
    -------
    float — negative BWT = forgetting, positive = positive transfer
    """
    T = accuracy_matrix.shape[0]
    if T < 2:
        return 0.0
    bwt = 0.0
    for i in range(T - 1):
        bwt += accuracy_matrix[T - 1, i] - accuracy_matrix[i, i]
    return float(bwt / (T - 1))


def forward_transfer(
    accuracy_matrix: np.ndarray,
    random_init_acc: Optional[np.ndarray] = None,
) -> float:
    """
    Forward Transfer (FWT).
    FWT = (1 / (T-1)) * sum_{i=2}^{T} (R_{i-1,i} - b_i)

    where b_i = random init accuracy on task i (default 0.0)
    """
    T = accuracy_matrix.shape[0]
    if T < 2:
        return 0.0
    if random_init_acc is None:
        random_init_acc = np.zeros(T)
    fwt = 0.0
    for i in range(1, T):
        fwt += accuracy_matrix[i - 1, i] - random_init_acc[i]
    return float(fwt / (T - 1))


def average_accuracy(accuracy_matrix: np.ndarray) -> float:
    """Average accuracy over all tasks after training on all tasks."""
    T = accuracy_matrix.shape[0]
    return float(accuracy_matrix[T - 1, :].mean())


def forgetting(accuracy_matrix: np.ndarray) -> float:
    """
    Average forgetting.
    F = (1 / (T-1)) * sum_{i=1}^{T-1} (max_{t<=T} R_{t,i} - R_{T,i})
    """
    T = accuracy_matrix.shape[0]
    if T < 2:
        return 0.0
    f = 0.0
    for i in range(T - 1):
        peak = accuracy_matrix[:, i].max()
        final = accuracy_matrix[T - 1, i]
        f += peak - final
    return float(f / (T - 1))


def continual_learning_metrics(
    accuracy_matrix: np.ndarray,
) -> Dict[str, float]:
    """All four CL metrics from a single accuracy matrix."""
    return {
        "average_accuracy": average_accuracy(accuracy_matrix),
        "backward_transfer": backward_transfer(accuracy_matrix),
        "forward_transfer": forward_transfer(accuracy_matrix),
        "forgetting": forgetting(accuracy_matrix),
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def expected_calibration_error(
    confidences: np.ndarray,
    is_correct: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE — see evaluator.py for docs."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(confidences)
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = is_correct[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / N) * abs(float(acc) - float(conf))
    return float(ece)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC."""
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5
    return float(roc_auc_score(labels, scores))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUPRC."""
    from sklearn.metrics import average_precision_score
    labels = np.asarray(labels)
    if labels.sum() == 0:
        return 0.0
    return float(average_precision_score(labels, scores))


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def paired_ttest(
    a: np.ndarray, b: np.ndarray, alternative: str = "two-sided"
) -> Tuple[float, float]:
    """
    Paired t-test.

    Returns
    -------
    (t_statistic, p_value)
    """
    result = scipy.stats.ttest_rel(a, b, alternative=alternative)
    return float(result.statistic), float(result.pvalue)


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for the mean.

    Returns
    -------
    (mean, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    means = [
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_bootstrap)
    ]
    alpha = (1 - ci) / 2
    lower = float(np.quantile(means, alpha))
    upper = float(np.quantile(means, 1 - alpha))
    return float(values.mean()), lower, upper


def summary_stats(values: List[float]) -> Dict[str, float]:
    """Mean, std, min, max, median of a list."""
    arr = np.array(values)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
    }
