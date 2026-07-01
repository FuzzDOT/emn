"""
Publication-quality figure generation for EMN.
All figures: 300 DPI, saved as both PDF and PNG.

Figures:
  1. EMN architecture diagram
  2. Selective forgetting results (bar chart)
  3. Continual learning comparison (line plot)
  4. Confabulation AUROC (ROC curves)
  5. Memory lifecycle visualization
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyArrow
import numpy as np
import seaborn as sns

# Publication-quality settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

DPI = 300
COLORS = {
    "standard_transformer": "#7f7f7f",
    "random_eviction": "#d62728",
    "lru_eviction": "#ff7f0e",
    "titans_surprise": "#2ca02c",
    "emn": "#1f77b4",
    "sequential_ft": "#7f7f7f",
    "ewc": "#e377c2",
    "si": "#bcbd22",
    "gem": "#17becf",
    "softmax": "#7f7f7f",
    "temperature": "#ff7f0e",
    "mc_dropout": "#2ca02c",
    "emn_vacuity": "#1f77b4",
}

DISPLAY_NAMES = {
    "standard_transformer": "Standard Transformer",
    "random_eviction": "Random Eviction",
    "lru_eviction": "LRU",
    "titans_surprise": "Titans (Surprise)",
    "emn": "EMN (Ours)",
    "sequential_ft": "Sequential FT",
    "ewc": "EWC",
    "si": "SI",
    "gem": "GEM",
    "softmax": "Softmax",
    "temperature": "Temp. Scaling",
    "mc_dropout": "MC Dropout",
    "emn_vacuity": "EMN Vacuity (Ours)",
}


def _save_figure(fig: plt.Figure, path_stem: str) -> None:
    """Save figure as both PDF and PNG at 300 DPI."""
    for ext in ("pdf", "png"):
        p = f"{path_stem}.{ext}"
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: EMN Architecture Diagram
# ---------------------------------------------------------------------------

def figure1_architecture(output_dir: str = "figures") -> None:
    """Draw EMN architecture as a flow diagram."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Epistemic Memory Networks (EMN) Architecture", fontsize=14, fontweight="bold")

    # Helper: draw box
    def box(x, y, w, h, label, color="#d0e8ff", fontsize=9):
        rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                              facecolor=color, edgecolor="black", linewidth=1.2, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", zorder=3, wrap=True)

    # Helper: draw arrow
    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # Components
    box(1.0, 2.5, 1.6, 0.7, "Input\nEmbedding", "#ffe0c0")
    box(3.2, 2.5, 1.8, 0.7, "Evidential\nWrite Gate\n(NOVA EDL)", "#d0e8ff")
    box(5.5, 3.5, 1.8, 0.7, "Vacuity\nScore υ", "#e0ffe0")
    box(5.5, 1.5, 1.8, 0.7, "Memory\nVector m", "#ffe0e0")
    box(8.0, 2.5, 1.8, 0.7, "Epistemic\nMemory Store", "#f0e0ff")

    # Arrows
    arrow(1.8, 2.5, 2.3, 2.5)
    arrow(4.1, 2.8, 4.8, 3.5)
    arrow(4.1, 2.2, 4.8, 1.5)
    arrow(6.4, 3.5, 7.2, 2.8)
    arrow(6.4, 1.5, 7.2, 2.2)

    # Labels on arrows
    ax.text(4.55, 3.2, "vacuity", fontsize=8, color="#006600", ha="center")
    ax.text(4.55, 1.85, "vector", fontsize=8, color="#cc0000", ha="center")

    # Eviction note
    ax.text(8.0, 1.0, "Evict: argmax(υ)\n(NOT FIFO / LRU / Random)",
            ha="center", va="center", fontsize=9, color="#333333",
            style="italic",
            bbox=dict(facecolor="#fffbe0", edgecolor="#aaa", boxstyle="round,pad=0.3"))

    # Retrieval note
    ax.text(5.5, 0.3, "Retrieval: score = cos(q, m) × (1 - υ)",
            ha="center", va="center", fontsize=9, color="#333333",
            style="italic",
            bbox=dict(facecolor="#e8f0ff", edgecolor="#aaa", boxstyle="round,pad=0.3"))

    _save_figure(fig, str(Path(output_dir) / "figure1_architecture"))
    print(f"Figure 1 saved: {output_dir}/figure1_architecture.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Figure 2: Selective Forgetting Results
# ---------------------------------------------------------------------------

def figure2_selective_forgetting(
    results: Dict[str, Dict[str, float]],
    output_dir: str = "figures",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Selective Forgetting — MemoryAgentBench FactConsolidation", fontsize=13)

    names = list(results.keys())
    colors = [COLORS.get(n, "#888888") for n in names]
    display = [DISPLAY_NAMES.get(n, n) for n in names]

    for ax, metric, label in [
        (axes[0], "accuracy", "Accuracy"),
        (axes[1], "f1", "F1 Score"),
    ]:
        means = [results[n].get(metric, 0.0) for n in names]
        stds = [results[n].get(f"{metric}_std", 0.0) for n in names]
        bars = ax.bar(range(len(names)), means, color=colors,
                      alpha=0.85, edgecolor="black", linewidth=0.8)
        ax.errorbar(range(len(names)), means, yerr=stds,
                    fmt="none", color="black", capsize=4, linewidth=1.2)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(display, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.05)
        ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    _save_figure(fig, str(Path(output_dir) / "figure2_selective_forgetting"))
    print(f"Figure 2 saved: {output_dir}/figure2_selective_forgetting.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Figure 3: Continual Learning Comparison
# ---------------------------------------------------------------------------

def figure3_continual_learning(
    results: Dict[str, Dict[str, float]],
    output_dir: str = "figures",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Continual Learning — Split-CIFAR100 (10 Tasks, SlimResNet18)", fontsize=13)

    names = list(results.keys())
    colors = [COLORS.get(n, "#888888") for n in names]
    display = [DISPLAY_NAMES.get(n, n) for n in names]

    for ax, metric, label, higher in [
        (axes[0], "average_accuracy", "Average Accuracy ↑", True),
        (axes[1], "forgetting", "Forgetting ↓", False),
    ]:
        means = [results[n].get(metric, 0.0) for n in names]
        stds = [results[n].get(f"{metric}_std", 0.0) for n in names]
        ax.bar(range(len(names)), means, color=colors,
               alpha=0.85, edgecolor="black", linewidth=0.8)
        ax.errorbar(range(len(names)), means, yerr=stds,
                    fmt="none", color="black", capsize=4, linewidth=1.2)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(display, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(label)

    plt.tight_layout()
    _save_figure(fig, str(Path(output_dir) / "figure3_continual_learning"))
    print(f"Figure 3 saved: {output_dir}/figure3_continual_learning.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Figure 4: Confabulation ROC Curves
# ---------------------------------------------------------------------------

def figure4_confabulation_roc(
    per_method_scores: Dict[str, Dict[str, np.ndarray]],
    output_dir: str = "figures",
) -> None:
    """
    Parameters
    ----------
    per_method_scores : {method_name: {"scores": np.ndarray, "labels": np.ndarray}}
    """
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title("Confabulation Benchmark — ROC Curves", fontsize=13)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")

    for method, data in per_method_scores.items():
        scores = data["scores"]
        labels = data["labels"]
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        color = COLORS.get(method, "#888888")
        disp = DISPLAY_NAMES.get(method, method)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{disp} (AUC={roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    plt.tight_layout()
    _save_figure(fig, str(Path(output_dir) / "figure4_confabulation_roc"))
    print(f"Figure 4 saved: {output_dir}/figure4_confabulation_roc.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Figure 5: Memory Lifecycle Visualization
# ---------------------------------------------------------------------------

def figure5_memory_lifecycle(
    vacuity_trajectories: Optional[Dict] = None,
    output_dir: str = "figures",
) -> None:
    """
    Visualise how vacuity scores evolve across the memory lifecycle.
    If no real data provided, generates synthetic illustration.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("EMN Memory Lifecycle: Vacuity Score Dynamics", fontsize=13)

    rng = np.random.default_rng(42)
    n_memories = 50
    steps = 100

    # Panel 1: Vacuity at write time (distribution)
    ax = axes[0]
    vacuities = rng.beta(1.5, 5.0, n_memories)  # Most memories are confident
    ax.hist(vacuities, bins=15, color="#1f77b4", alpha=0.8, edgecolor="black", lw=0.5)
    ax.axvline(vacuities.mean(), color="red", lw=2, linestyle="--", label=f"Mean={vacuities.mean():.2f}")
    ax.set_xlabel("Vacuity at Write Time")
    ax.set_ylabel("Count")
    ax.set_title("(a) Vacuity Distribution at Write", fontsize=11)
    ax.legend()

    # Panel 2: Eviction pattern — high vacuity evicted first
    ax = axes[1]
    memory_ages = np.arange(n_memories)
    eviction_order = np.argsort(vacuities)[::-1]  # high vacuity evicted first
    scatter_colors = ["#d62728" if i in eviction_order[:10] else "#1f77b4"
                      for i in range(n_memories)]
    ax.scatter(memory_ages, vacuities, c=scatter_colors, alpha=0.8, s=40, edgecolors="black", lw=0.3)
    evicted_patch = mpatches.Patch(color="#d62728", label="Evicted (top-10 vacuity)")
    kept_patch = mpatches.Patch(color="#1f77b4", label="Retained")
    ax.set_xlabel("Memory Index (age)")
    ax.set_ylabel("Vacuity Score")
    ax.set_title("(b) Eviction Pattern (Red = Evicted)", fontsize=11)
    ax.legend(handles=[evicted_patch, kept_patch], fontsize=8)
    ax.axhline(vacuities[eviction_order[9]], color="orange", lw=1.5, linestyle=":",
               label="Eviction threshold")

    # Panel 3: Retrieval score vs cosine similarity
    ax = axes[2]
    cos_sims = rng.uniform(0.3, 1.0, n_memories)
    vacuity_scatter = rng.beta(1.5, 5.0, n_memories)
    retrieval_scores = cos_sims * (1.0 - vacuity_scatter)
    sc = ax.scatter(cos_sims, retrieval_scores, c=vacuity_scatter,
                    cmap="RdYlGn_r", alpha=0.8, s=40, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="Vacuity")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="score = sim (no penalty)")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Retrieval Score = sim × (1 − υ)")
    ax.set_title("(c) Retrieval Score vs Similarity", fontsize=11)
    ax.legend(fontsize=8)

    plt.tight_layout()
    _save_figure(fig, str(Path(output_dir) / "figure5_memory_lifecycle"))
    print(f"Figure 5 saved: {output_dir}/figure5_memory_lifecycle.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Master figure generation
# ---------------------------------------------------------------------------

def generate_all_figures(
    results_dir: str = "results",
    output_dir: str = "figures",
    also_copy_to_paper: bool = True,
) -> None:
    """
    Generate all 5 figures from result files.
    Loads CSVs from results_dir, outputs to figures/ and paper/figures/.
    """
    results_path = Path(results_dir)
    fig_path = Path(output_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    paper_fig_path = Path("paper/figures")
    paper_fig_path.mkdir(parents=True, exist_ok=True)

    # Figure 1: architecture (no data needed)
    figure1_architecture(output_dir=str(fig_path))

    # Load experiment results if available
    exp1_path = results_path / "exp1_results.json"
    exp2_path = results_path / "exp2_results.json"
    exp3_path = results_path / "exp3_results.json"

    if exp1_path.exists():
        with open(exp1_path) as f:
            exp1_results = json.load(f)
        figure2_selective_forgetting(exp1_results, output_dir=str(fig_path))
    else:
        print(f"Warning: {exp1_path} not found, skipping Figure 2")

    if exp2_path.exists():
        with open(exp2_path) as f:
            exp2_results = json.load(f)
        figure3_continual_learning(exp2_results, output_dir=str(fig_path))
    else:
        print(f"Warning: {exp2_path} not found, skipping Figure 3")

    if exp3_path.exists():
        with open(exp3_path) as f:
            exp3_data = json.load(f)
        # Build per-method score arrays from saved results
        per_method_scores = {}
        for method, data in exp3_data.items():
            if "scores" in data and "labels" in data:
                per_method_scores[method] = {
                    "scores": np.array(data["scores"]),
                    "labels": np.array(data["labels"]),
                }
        if per_method_scores:
            figure4_confabulation_roc(per_method_scores, output_dir=str(fig_path))
    else:
        print(f"Warning: {exp3_path} not found, skipping Figure 4")

    # Figure 5: lifecycle (synthetic illustration)
    figure5_memory_lifecycle(output_dir=str(fig_path))

    # Copy to paper/figures
    if also_copy_to_paper:
        import shutil
        for f in fig_path.glob("figure*.*"):
            shutil.copy2(f, paper_fig_path / f.name)
        print(f"Figures copied to {paper_fig_path}")
