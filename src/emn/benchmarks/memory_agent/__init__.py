"""EMN MemoryAgentBench integration."""

from emn.benchmarks.memory_agent.bench_runner import (
    run_all_baselines,
    evaluate_agent,
    load_fact_consolidation_task,
    get_bench_path,
    EMNAgent,
    StandardTransformerAgent,
    RandomEvictionAgent,
    LRUEvictionAgent,
    TitansSurpriseAgent,
)

__all__ = [
    "run_all_baselines",
    "evaluate_agent",
    "load_fact_consolidation_task",
    "get_bench_path",
    "EMNAgent",
    "StandardTransformerAgent",
    "RandomEvictionAgent",
    "LRUEvictionAgent",
    "TitansSurpriseAgent",
]
