# Epistemic Memory Networks (EMN)

> *Memory architecture treating confidence as a first-class variable.*

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Standard memory systems treat all memories equally. EMN breaks this symmetry: every memory carries a **vacuity score** — a Dirichlet-based measure of evidential uncertainty from NOVA's architecture — that governs the memory's entire lifecycle from write to eviction to retrieval.

**Core insight**: uncertain memories (high vacuity) should be evicted first and down-ranked in retrieval, while confident memories (low vacuity) should be protected.

**Relationship to NOVA**: EMN does not reimplement evidential learning. NOVA's `EvidentialHead` is used verbatim (patched import only). EMN is three thin architectural wrappers that add memory semantics on top.

---

## Three Components

| Component | Location | Role |
| --- | --- | --- |
| EvidentialWriteGate | `src/emn/gates/write_gate.py` | Assigns vacuity at write time |
| EpistemicMemoryStore | `src/emn/memory/store.py` | Evicts `argmax(vacuity)` — not FIFO/LRU/random |
| UncertaintyWeightedRetriever | `src/emn/retrieval/retriever.py` | `score = cos(q,m) × (1−v)` |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/FuzzDOT/emn.git
cd emn
conda env create -f environment.yml
conda activate emn

# 2. Run all experiments
bash scripts/run_all_experiments.sh

# 3. Fast smoke test (5 minutes)
bash scripts/run_all_experiments.sh --fast-test
```

---

## Experiments

### Experiment 1: Selective Forgetting (MemoryAgentBench)

```bash
python experiments/exp1_selective_forgetting.py --seeds 42 43 44
```

5 baselines: Standard Transformer, Random Eviction, LRU, Titans (Surprise), EMN.
Metrics: Accuracy, Precision, Recall, F1.

### Experiment 2: Continual Learning (Split-CIFAR100)

```bash
python experiments/exp2_continual_learning.py --seeds 42 43 44
```

10 tasks × 10 classes, SlimResNet18. Baselines: SeqFT, EWC, SI, GEM, EMN.
Metrics: Average Accuracy, Backward Transfer, Forward Transfer, Forgetting.

### Experiment 3: Confabulation Benchmark

```bash
python experiments/exp3_confabulation.py --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

200 facts × 5 severity levels = 1000 items. All 4 methods on the same TinyLlama backbone.
Methods: Softmax, Temperature Scaling, MC Dropout, EMN Vacuity.
Metrics: ConfabulationRate, HedgingRate, UpdateRate, AUROC, AUPRC, ECE.

---

## Repository Structure

```
emn/
├── src/emn/
│   ├── evidential/        NOVA EvidentialHead (verbatim) + VacuityExtractor
│   ├── memory/            MemoryEntry + EpistemicMemoryStore
│   ├── gates/             EvidentialWriteGate (Component 1)
│   ├── retrieval/         UncertaintyWeightedRetriever (Component 3)
│   ├── continual/         EMNPlugin for Avalanche
│   ├── llm/               EMNWrappedCausalLM
│   ├── benchmarks/        Confabulation dataset + MemoryAgentBench runner
│   └── utils/             Metrics, tables, plots, reproducibility
├── experiments/           exp1_*.py, exp2_*.py, exp3_*.py
├── scripts/               run_all_experiments.sh + SLURM scripts
├── tests/                 unit, integration, e2e
├── configs/               base.yaml, experiment1/2/3, 6 ablations
├── docs/                  architecture.md, experiments.md, api.md, hpc.md
└── results/               (generated) JSON, CSV, LaTeX
```

---

## Tests

```bash
# Run all tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# With coverage
pytest tests/ --cov=emn --cov-report=html
```

---

## HPC (NCSU Henry2)

```bash
sbatch scripts/run_exp1.slurm   # 1× A100, 4h
sbatch scripts/run_exp2.slurm   # 4× A100, 12h, DDP
sbatch scripts/run_exp3.slurm   # 8× A100, 8h
```

See `docs/hpc.md` for full setup guide.

---

## Citation

```bibtex
@article{mohamed2025emn,
  title   = {Epistemic Memory Networks: Treating Confidence as a First-Class Architectural Variable},
  author  = {Mohamed, Faaz},
  year    = {2025},
  note    = {Manuscript in preparation, targeting ACL Rolling Review}
}
```

NOVA (foundational architecture):
```bibtex
@misc{nova2025,
  title  = {NOVA: Cognitive Architecture with Evidential Uncertainty Decomposition},
  author = {Mohamed, Faaz},
  year   = {2025},
  doi    = {10.5281/zenodo.20562861},
  url    = {https://zenodo.org/doi/10.5281/zenodo.20562861}
}
```

---

## License

MIT. See [LICENSE](LICENSE).
