"""
Table generation for paper output.
Produces CSV + LaTeX tables from experiment results.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


def _fmt(v, decimals: int = 3) -> str:
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _bold_max(values: List[str], higher_is_better: bool = True) -> List[str]:
    """Bold the best value in a list of formatted floats."""
    try:
        nums = [float(v) for v in values]
    except ValueError:
        return values
    best_idx = nums.index(max(nums) if higher_is_better else min(nums))
    result = list(values)
    result[best_idx] = r"\textbf{" + result[best_idx] + "}"
    return result


# ---------------------------------------------------------------------------
# Table 1: Selective Forgetting (Exp 1)
# ---------------------------------------------------------------------------

def make_table1(
    results: Dict[str, Dict[str, float]],
    output_dir: str = "results",
    decimals: int = 3,
) -> None:
    """
    Parameters
    ----------
    results : {baseline_name: {metric: mean, metric_std: std, ...}}
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    baselines = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]

    # CSV
    csv_path = Path(output_dir) / "table1_selective_forgetting.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Method"] + [m.capitalize() for m in metrics]
        writer.writerow(header)
        for name in baselines:
            row = [name]
            for m in metrics:
                mean = results[name].get(m, 0.0)
                std = results[name].get(f"{m}_std", 0.0)
                row.append(f"{mean:.{decimals}f} ± {std:.{decimals}f}")
            writer.writerow(row)

    # LaTeX
    tex_path = Path(output_dir) / "table1_selective_forgetting.tex"
    display_names = {
        "standard_transformer": "Standard Transformer",
        "random_eviction": "Random Eviction",
        "lru_eviction": "LRU Eviction",
        "titans_surprise": "Titans (Surprise)",
        "emn": r"\textbf{EMN (Ours)}",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Selective Forgetting on MemoryAgentBench FactConsolidation Task."
        r" Results are mean $\pm$ std over seeds \{42, 43, 44\}.}",
        r"\label{tab:selective_forgetting}",
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule",
        "Method & " + " & ".join(m.capitalize() for m in metrics) + r" \\",
        r"\midrule",
    ]

    # Find best values for bolding
    for name in baselines:
        row_vals = []
        for m in metrics:
            mean = results[name].get(m, 0.0)
            std = results[name].get(f"{m}_std", 0.0)
            row_vals.append(f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}")
        display = display_names.get(name, name)
        lines.append(display + " & " + " & ".join(row_vals) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Table 1 saved: {csv_path}, {tex_path}")


# ---------------------------------------------------------------------------
# Table 2: Continual Learning (Exp 2)
# ---------------------------------------------------------------------------

def make_table2(
    results: Dict[str, Dict[str, float]],
    output_dir: str = "results",
    decimals: int = 3,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    baselines = list(results.keys())
    metrics = ["average_accuracy", "backward_transfer", "forward_transfer", "forgetting"]

    csv_path = Path(output_dir) / "table2_continual_learning.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Method", "Avg Acc", "BWT", "FWT", "Forgetting"]
        writer.writerow(header)
        for name in baselines:
            row = [name]
            for m in metrics:
                mean = results[name].get(m, 0.0)
                std = results[name].get(f"{m}_std", 0.0)
                row.append(f"{mean:.{decimals}f} ± {std:.{decimals}f}")
            writer.writerow(row)

    tex_path = Path(output_dir) / "table2_continual_learning.tex"
    display_names = {
        "sequential_ft": "Sequential FT",
        "ewc": "EWC",
        "si": "SI",
        "gem": "GEM",
        "emn": r"\textbf{EMN (Ours)}",
    }
    header_cols = ["Avg Acc $\\uparrow$", "BWT $\\uparrow$", "FWT $\\uparrow$", "Forgetting $\\downarrow$"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Continual Learning on Split-CIFAR100 (10 tasks, SlimResNet18). "
        r"Results over seeds \{42, 43, 44\}.}",
        r"\label{tab:continual_learning}",
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule",
        "Method & " + " & ".join(header_cols) + r" \\",
        r"\midrule",
    ]

    for name in baselines:
        row_vals = []
        for m in metrics:
            mean = results[name].get(m, 0.0)
            std = results[name].get(f"{m}_std", 0.0)
            row_vals.append(f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}")
        display = display_names.get(name, name)
        lines.append(display + " & " + " & ".join(row_vals) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Table 2 saved: {csv_path}, {tex_path}")


# ---------------------------------------------------------------------------
# Table 3: Confabulation (Exp 3)
# ---------------------------------------------------------------------------

def make_table3(
    results: Dict[str, Dict],
    output_dir: str = "results",
    decimals: int = 3,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    methods = list(results.keys())
    metrics = ["confabulation_rate", "hedging_rate", "update_rate", "auroc", "auprc", "ece"]
    display_metrics = [
        "Confab Rate $\\downarrow$",
        "Hedge Rate",
        "Update Rate $\\uparrow$",
        "AUROC $\\uparrow$",
        "AUPRC $\\uparrow$",
        "ECE $\\downarrow$",
    ]

    csv_path = Path(output_dir) / "table3_confabulation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method"] + metrics)
        for method in methods:
            r = results[method]
            row = [method] + [f"{r.get(m, 0.0):.{decimals}f}" for m in metrics]
            writer.writerow(row)

    tex_path = Path(output_dir) / "table3_confabulation.tex"
    display_names = {
        "softmax": "Softmax",
        "temperature": "Temp. Scaling",
        "mc_dropout": "MC Dropout",
        "emn_vacuity": r"\textbf{EMN Vacuity (Ours)}",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Confabulation Benchmark Results (200 facts $\times$ 5 severity levels). "
        r"All methods use TinyLlama-1.1B backbone.}",
        r"\label{tab:confabulation}",
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule",
        "Method & " + " & ".join(display_metrics) + r" \\",
        r"\midrule",
    ]

    for method in methods:
        r = results[method]
        row_vals = [f"{r.get(m, 0.0):.{decimals}f}" for m in metrics]
        display = display_names.get(method, method)
        lines.append(display + " & " + " & ".join(row_vals) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Table 3 saved: {csv_path}, {tex_path}")


# ---------------------------------------------------------------------------
# Ablation table
# ---------------------------------------------------------------------------

def make_ablation_table(
    results: Dict[str, Dict[str, float]],
    output_dir: str = "results",
    decimals: int = 3,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ablations = list(results.keys())
    metrics = ["accuracy", "f1", "average_accuracy", "auroc"]

    csv_path = Path(output_dir) / "table_ablation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Ablation"] + metrics)
        for name in ablations:
            row = [name] + [f"{results[name].get(m, 0.0):.{decimals}f}" for m in metrics]
            writer.writerow(row)

    tex_path = Path(output_dir) / "table_ablation.tex"
    display_names = {
        "full_emn": r"\textbf{Full EMN}",
        "no_vacuity_retrieval": "w/o Vacuity Retrieval",
        "no_vacuity_eviction": "w/o Vacuity Eviction",
        "no_evidential_head": "w/o Evidential Head",
        "random_confidence": "Random Confidence",
        "lru_eviction": "LRU Eviction",
        "softmax_confidence": "Softmax Confidence",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study. Each row removes or replaces one EMN component.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule",
        "Configuration & Accuracy & F1 & Avg Acc & AUROC" + r" \\",
        r"\midrule",
    ]

    for name in ablations:
        row_vals = [f"{results[name].get(m, 0.0):.{decimals}f}" for m in metrics]
        display = display_names.get(name, name)
        lines.append(display + " & " + " & ".join(row_vals) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Ablation table saved: {csv_path}, {tex_path}")


# ---------------------------------------------------------------------------
# Results summary markdown
# ---------------------------------------------------------------------------

def make_results_summary(
    exp1: Optional[Dict] = None,
    exp2: Optional[Dict] = None,
    exp3: Optional[Dict] = None,
    output_dir: str = "results",
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    lines = [
        "# EMN Experiment Results Summary",
        "",
        "Auto-generated by `emn.utils.tables.make_results_summary()`.",
        "",
    ]

    if exp1:
        lines += [
            "## Experiment 1: Selective Forgetting (MemoryAgentBench)",
            "",
            "| Method | Accuracy | F1 |",
            "| --- | --- | --- |",
        ]
        for name, res in exp1.items():
            acc = f"{res.get('accuracy', 0.0):.3f} ± {res.get('accuracy_std', 0.0):.3f}"
            f1 = f"{res.get('f1', 0.0):.3f} ± {res.get('f1_std', 0.0):.3f}"
            lines.append(f"| {name} | {acc} | {f1} |")
        lines.append("")

    if exp2:
        lines += [
            "## Experiment 2: Continual Learning (Split-CIFAR100)",
            "",
            "| Method | Avg Acc | BWT | Forgetting |",
            "| --- | --- | --- | --- |",
        ]
        for name, res in exp2.items():
            aa = f"{res.get('average_accuracy', 0.0):.3f}"
            bwt = f"{res.get('backward_transfer', 0.0):.3f}"
            forg = f"{res.get('forgetting', 0.0):.3f}"
            lines.append(f"| {name} | {aa} | {bwt} | {forg} |")
        lines.append("")

    if exp3:
        lines += [
            "## Experiment 3: Confabulation Benchmark",
            "",
            "| Method | Confab Rate | AUROC | ECE |",
            "| --- | --- | --- | --- |",
        ]
        for name, res in exp3.items():
            cr = f"{res.get('confabulation_rate', 0.0):.3f}"
            au = f"{res.get('auroc', 0.0):.3f}"
            ece = f"{res.get('ece', 0.0):.3f}"
            lines.append(f"| {name} | {cr} | {au} | {ece} |")
        lines.append("")

    out_path = Path(output_dir) / "results_summary.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Results summary: {out_path}")
