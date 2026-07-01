# EMN API Reference

## Core Classes

### EvidentialWriteGate

```python
from emn.gates.write_gate import EvidentialWriteGate

gate = EvidentialWriteGate(d_model=384, n_classes=256)
output = gate(x)          # x: (batch, d_model) tensor
# output.alpha    : (batch, 256) Dirichlet params
# output.belief   : (batch, 256) normalised belief masses
# output.vacuity  : (batch,) in [0,1] — 0=confident, 1=uncertain
```

### EpistemicMemoryStore

```python
from emn.memory.store import EpistemicMemoryStore

store = EpistemicMemoryStore(
    capacity=1000,
    d_model=384,
    eviction_strategy="vacuity",   # "vacuity" | "random" | "lru"
    retrieval_vacuity_weight=1.0,
)

entry = store.write(vector, vacuity=0.3, task_id="task_0")
entries = store.retrieve(query, k=5)
entries, scores = store.retrieve(query, k=5, return_scores=True)
stats = store.stats()
store.save("path/to/store")
store = EpistemicMemoryStore.load("path/to/store")
```

### UncertaintyWeightedRetriever

```python
from emn.retrieval.retriever import UncertaintyWeightedRetriever

retriever = UncertaintyWeightedRetriever(
    store=store,
    backend="brute",     # "brute" | "faiss"
    vacuity_weight=1.0,
)

entries = retriever.retrieve(query, k=5)
entries, scores = retriever.retrieve(query, k=5, return_scores=True)
all_scores = retriever.score_all(query)
```

### EMNWrappedCausalLM

```python
from emn.llm.wrapped_lm import EMNWrappedCausalLM

model = EMNWrappedCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    memory_capacity=1000,
)
entry = model.add_memory("Water freezes at 0°C.")
response = model.generate("At what temperature does water freeze?")
stats = model.memory_stats()
```

### EMNPlugin (Avalanche)

```python
from emn.continual.emn_plugin import EMNPlugin

plugin = EMNPlugin(
    store=store,
    write_gate=write_gate,
    feature_extractor=feature_extractor,
    lambda_mem=0.5,
    memory_batch_size=64,
)
# Plug into any Avalanche strategy:
strategy = Naive(model, optimizer, criterion, plugins=[plugin])
```
