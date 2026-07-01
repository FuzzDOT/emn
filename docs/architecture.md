# EMN Architecture

## Overview

Epistemic Memory Networks (EMN) treats memory confidence as a first-class architectural variable. Built on NOVA's Dirichlet evidential uncertainty framework, EMN adds three components to any memory-augmented neural system.

## Core Principle

Standard memory systems treat all memories equally — a confident fact and a confabulated guess occupy the same slot and compete equally for retrieval. EMN breaks this symmetry. Every memory carries a **vacuity score** derived from NOVA's EvidentialHead, and this score governs the memory's entire lifecycle.

**Vacuity** = K / S where K = n\_classes (256), S = sum of Dirichlet alpha parameters. High vacuity = low total evidence = uncertain memory.

## The Three Components

### Component 1: EvidentialWriteGate

`src/emn/gates/write_gate.py`

Every incoming memory embedding passes through this gate before storage. The gate is a thin wrapper around NOVA's EvidentialHead that assigns a Dirichlet-based vacuity score.

```
Input x (batch, d_model)
  → LayerNorm → Linear(d_model, d_model//2) → SiLU
  → Linear(d_model//2, n_classes) → Softplus+1
  → alpha (Dirichlet params)
  → vacuity = n_classes / sum(alpha)
```

Output: `WriteGateOutput(alpha, belief, vacuity)`.

The vacuity score is stored alongside the memory vector and never recomputed — it travels with the memory for its entire lifecycle.

### Component 2: EpistemicMemoryStore

`src/emn/memory/store.py`

Fixed-capacity memory store. When at capacity and a new memory arrives, the entry with the **highest vacuity** (most uncertain) is evicted.

This is the central claim: **evict the least trustworthy memory, not the oldest, least-recently-used, or a random one.**

Implementation: a parallel `np.float32` array of vacuity values enables O(N) `argmax` eviction without deserialising full entries.

```python
# Eviction in 1 line:
idx = np.argmax(self._vacuity[:self._size])
```

Ablation hook: `eviction_strategy` can be set to `"random"` or `"lru"` to run ablation experiments without changing any other code.

### Component 3: UncertaintyWeightedRetriever

`src/emn/retrieval/retriever.py`

Retrieval score: `score_i = cosine(query, memory_i) × (1 - vacuity_i)`

High-vacuity memories score lower even if they are geometrically close to the query. The retriever prefers memories with both high similarity and high confidence.

Two backends:
- **brute**: pure NumPy cosine similarity, O(N×d)
- **faiss**: FAISS IndexFlatIP on L2-normalised vectors, O(log N) search, then post-reranks top-2k candidates

## NOVA Dependency

EMN does **not** reimplement evidential learning. The canonical implementation is NOVA's `evidential_uncertainty.py`, copied verbatim with one import patch (`research_core.types` → `emn.types`).

Source: NOVA Project Coffeemaker, Stage 0. DOI: 10.5281/zenodo.20562861.

## Memory Lifecycle

```
Text/embedding
     ↓
EvidentialWriteGate
     ↓
WriteGateOutput { alpha, belief, vacuity }
     ↓
EpistemicMemoryStore.write(vector, vacuity)
     ├── If at capacity: evict argmax(vacuity)
     └── Store entry with persistent vacuity
     ↓
Retrieval query
     ↓
UncertaintyWeightedRetriever
     └── score = cosine × (1 - vacuity)
     └── Return top-k by score
```

## Continual Learning Integration

`src/emn/continual/emn_plugin.py`

`EMNPlugin` is an Avalanche `SupervisedPlugin` that injects a memory protection loss:

```
L_total = L_ce + λ × mean_i[(1 - v_i) × ||f_t(x_i) - m_i||²]
```

The `(1 - v_i)` weight means:
- **Uncertain memories (high v)**: contribute little to the protection loss — the model is free to overwrite them
- **Confident memories (low v)**: contribute strongly — the model is penalised for forgetting what it knows well
