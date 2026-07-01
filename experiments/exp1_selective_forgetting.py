"""
Experiment 1: Selective Forgetting — MemoryAgentBench FactConsolidation
=========================================================================
Requires MemoryAgentBench cloned at external/MemoryAgentBench.
The run_all_experiments.sh script handles cloning.

Usage:
    python experiments/exp1_selective_forgetting.py
    python experiments/exp1_selective_forgetting.py --seeds 42 43 44 --capacity 50
    python experiments/exp1_selective_forgetting.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from emn.benchmarks.memory_agent.bench_runner import (
    get_bench_path,
    load_fact_consolidation_task,
    run_all_baselines,
)
from emn.utils.reproducibility import set_seed, build_run_metadata
from emn.utils.tables import make_table1, make_results_summary
from emn.utils.plotting import figure2_selective_forgetting


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EMN Experiment 1: Selective Forgetting")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44],
                   help="Random seeds for repeated evaluation")
    p.add_argument("--capacity", type=int, default=50,
                   help="Memory store capacity for all baselines")
    p.add_argument("--d-model", type=int, default=384,
                   help="Sentence embedding dimensionality")
    p.add_argument("--device", default="cpu", help="cpu or cuda:N")
    p.add_argument("--results-dir", default="results",
                   help="Directory to write CSV/tex/json results")
    p.add_argument("--figures-dir", default="figures",
                   help="Directory to write figures")
    p.add_argument("--max-items", type=int, default=None,
                   help="Limit benchmark items (for fast testing)")
    p.add_argument("--no-figures", action="store_true",
                   help="Skip figure generation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seeds[0])

    print("=" * 60)
    print("EMN Experiment 1: Selective Forgetting")
    print("=" * 60)

    # Locate and load the benchmark
    bench_path = get_bench_path()
    print(f"MemoryAgentBench found at: {bench_path}")
    task_data = load_fact_consolidation_task(bench_path)

    if args.max_items:
        task_data = task_data[: args.max_items]
        print(f"  (limited to {args.max_items} items)")

    print(f"Loaded {len(task_data)} FactConsolidation items")

    # Build run metadata
    config = vars(args)
    build_run_metadata(
        experiment_name="exp1_selective_forgetting",
        seed=args.seeds[0],
        config=config,
        output_dir=args.results_dir,
    )

    # Run all baselines × all seeds
    print(f"\nRunning 5 baselines × {len(args.seeds)} seeds...")
    results = run_all_baselines(
        task_data=task_data,
        seeds=args.seeds,
        capacity=args.capacity,
        d_model=args.d_model,
        device=args.device,
        verbose=True,
    )

    # Save raw results
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    results_path = Path(args.results_dir) / "exp1_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved: {results_path}")

    # Print summary table
    print("\n" + "=" * 50)
    print(f"{'Method':<30} {'Accuracy':>10} {'F1':>10}")
    print("-" * 50)
    for name, res in results.items():
        acc = f"{res['accuracy']:.3f}±{res['accuracy_std']:.3f}"
        f1 = f"{res['f1']:.3f}±{res['f1_std']:.3f}"
        marker = " ◀" if name == "emn" else ""
        print(f"{name:<30} {acc:>10} {f1:>10}{marker}")

    # Generate tables
    make_table1(results, output_dir=args.results_dir)

    # Copy tables to paper/
    import shutil
    paper_tables = Path("paper/tables")
    paper_tables.mkdir(parents=True, exist_ok=True)
    for f in Path(args.results_dir).glob("table1_*"):
        shutil.copy2(f, paper_tables / f.name)

    # Generate figures
    if not args.no_figures:
        figure2_selective_forgetting(results, output_dir=args.figures_dir)

    print("\nExperiment 1 complete.")


if __name__ == "__main__":
    main()
