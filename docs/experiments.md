# EMN Experiments

## Experiment 1: Selective Forgetting (MemoryAgentBench)

**Script**: `experiments/exp1_selective_forgetting.py`

**Benchmark**: HUST-AI-HYZ/MemoryAgentBench, FactConsolidation task.

**Baselines** (5):
1. Standard Transformer — fixed context window, FIFO drop
2. Random Eviction — EMN store, random eviction strategy
3. LRU Eviction — EMN store, least-recently-used eviction
4. Titans (Surprise) — evict by gradient-magnitude surprise score
5. EMN — vacuity eviction + vacuity-weighted retrieval (full system)

**Metrics**: Accuracy, Precision, Recall, F1. Mean ± std over seeds {42, 43, 44}.

**Output**: `results/exp1_results.json`, `results/table1_selective_forgetting.{csv,tex}`

## Experiment 2: Continual Learning (Split-CIFAR100)

**Script**: `experiments/exp2_continual_learning.py`

**Benchmark**: SplitCIFAR100, 10 tasks × 10 classes, SlimResNet18 backbone.

**Baselines** (5):
1. Sequential FT — naive fine-tuning, no memory
2. EWC — Elastic Weight Consolidation (λ=0.4)
3. SI — Synaptic Intelligence (λ=0.1)
4. GEM — Gradient Episodic Memory
5. EMN — evidential memory protection loss

**EMN memory protection loss**:
```
L_mem_i = (1 - v_i) × ||f_t(x_i) - m_i||²
L_total = L_ce + 0.5 × mean(L_mem)
```
Memory sampled by inverse vacuity (64 entries per backward pass).

**Metrics**: Average Accuracy (AA), Backward Transfer (BWT), Forward Transfer (FWT), Forgetting (F).

**Output**: `results/exp2_results.json`, `results/table2_continual_learning.{csv,tex}`

## Experiment 3: Confabulation Benchmark

**Script**: `experiments/exp3_confabulation.py`

**Benchmark**: 200 facts × 5 contradiction severity levels = 1000 items.

All four methods use **TinyLlama-1.1B-Chat-v1.0** as the shared backbone. Apples-to-apples comparison: same model, same prompts, different confidence estimation method.

**Methods** (4):
1. Softmax — max softmax probability at last token
2. Temperature Scaling — softmax with T=1.5
3. MC Dropout — 10 stochastic passes, mutual information
4. EMN Vacuity — NOVA EvidentialHead on TinyLlama hidden states

**Severity levels**:
- Level 1: Subtle synonym swap
- Level 2: Attribute contradiction (plausible wrong value)
- Level 3: Authority contradiction (expert source cited)
- Level 4: Repetition contradiction (repeated 4×)
- Level 5: Unanimous contradiction (multiple authoritative sources)

**Metrics**: ConfabulationRate, HedgingRate, UpdateRate, AUROC, AUPRC, ECE.

**Output**: `results/exp3_results.json` (per-item scores+labels), `results/table3_confabulation.{csv,tex}`

## Ablations (6)

All ablations run via config override — the only changes are YAML flags, no code modification.

| Ablation | Change |
| --- | --- |
| no_vacuity_retrieval | `retrieval_vacuity_weight: 0.0` |
| no_vacuity_eviction | `eviction_strategy: random` |
| no_evidential_head | `confidence_source: none` |
| random_confidence | `confidence_source: random` |
| lru_eviction | `eviction_strategy: lru` |
| softmax_confidence | `confidence_source: softmax` |
