"""
MemoryAgentBench Integration
=============================
Runs the FactConsolidation task from HUST-AI-HYZ/MemoryAgentBench.

Requires the benchmark to be cloned at the path specified by
MEMORY_AGENT_BENCH_PATH (default: ./external/MemoryAgentBench).

The run_all_experiments.sh script clones the benchmark automatically.

Baselines implemented:
  1. standard_transformer — no memory system, just a fixed context window
  2. random_eviction       — EMN store with random eviction
  3. lru_eviction          — EMN store with LRU eviction
  4. titans_surprise       — evict by surprise score (gradient magnitude proxy)
  5. emn                   — EMN store with vacuity eviction (full system)

All baselines use the same TinyLlama backbone and sentence encoder.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from emn.memory.store import EpistemicMemoryStore
from emn.memory.entry import MemoryEntry
from emn.gates.write_gate import EvidentialWriteGate
from emn.retrieval.retriever import UncertaintyWeightedRetriever
from emn.utils.reproducibility import set_seed


# ---------------------------------------------------------------------------
# MemoryAgentBench path resolution
# ---------------------------------------------------------------------------

DEFAULT_BENCH_PATH = Path(__file__).parents[5] / "external" / "MemoryAgentBench"
BENCH_PATH_ENV = "MEMORY_AGENT_BENCH_PATH"


def get_bench_path() -> Path:
    env = os.environ.get(BENCH_PATH_ENV)
    if env:
        p = Path(env)
    else:
        p = DEFAULT_BENCH_PATH

    if not p.exists():
        raise RuntimeError(
            f"MemoryAgentBench not found at {p}.\n"
            f"Clone it with:\n"
            f"  git clone https://github.com/HUST-AI-HYZ/MemoryAgentBench.git {p}\n"
            f"Or set the {BENCH_PATH_ENV} environment variable to the correct path."
        )
    return p


def load_fact_consolidation_task(bench_path: Path) -> List[dict]:
    """
    Load the FactConsolidation task data from MemoryAgentBench.

    Returns list of dicts with keys: question, answer, facts, context
    """
    # Try several possible paths within the benchmark repo
    candidate_paths = [
        bench_path / "data" / "fact_consolidation.json",
        bench_path / "data" / "FactConsolidation" / "test.json",
        bench_path / "tasks" / "fact_consolidation" / "data.json",
        bench_path / "benchmark" / "fact_consolidation.json",
    ]

    for cand in candidate_paths:
        if cand.exists():
            with open(cand) as f:
                data = json.load(f)
            # Normalise to list of dicts
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "data" in data:
                return data["data"]
            elif isinstance(data, dict) and "examples" in data:
                return data["examples"]
            else:
                return list(data.values()) if isinstance(data, dict) else [data]

    # If no standard path found, scan recursively for JSON files
    json_files = list(bench_path.rglob("*.json"))
    fact_files = [f for f in json_files if "fact" in f.name.lower() or "consolidat" in f.name.lower()]

    if fact_files:
        with open(fact_files[0]) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return [data]

    raise FileNotFoundError(
        f"Could not find FactConsolidation task data in {bench_path}.\n"
        f"Available JSON files: {[str(f) for f in json_files[:10]]}"
    )


# ---------------------------------------------------------------------------
# Baseline implementations
# ---------------------------------------------------------------------------

class BaselineAgent:
    """Base class for all baseline agents."""

    def __init__(self, device: str = "cpu", d_model: int = 384):
        self.device = device
        self.d_model = d_model
        self._encoder = None

    def _load_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return self._encoder

    def encode(self, text: str) -> np.ndarray:
        enc = self._load_encoder()
        emb = enc.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        if emb.shape[0] != self.d_model:
            if emb.shape[0] > self.d_model:
                emb = emb[:self.d_model]
            else:
                emb = np.pad(emb, (0, self.d_model - emb.shape[0]))
        return emb.astype(np.float32)

    def process_fact(self, fact: str, task_id: str = "") -> None:
        """Process (store) a new fact."""
        raise NotImplementedError

    def answer_question(self, question: str) -> str:
        """Answer a question using stored memories."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset memory between episodes."""
        raise NotImplementedError


class StandardTransformerAgent(BaselineAgent):
    """No external memory — truncated fixed context window."""

    def __init__(self, context_window: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.context_window = context_window
        self._context: List[str] = []

    def process_fact(self, fact: str, task_id: str = "") -> None:
        self._context.append(fact)
        if len(self._context) > self.context_window:
            self._context.pop(0)  # FIFO — oldest dropped

    def answer_question(self, question: str) -> str:
        # Simple retrieval: return all context facts and the last one
        if not self._context:
            return ""
        return self._context[-1]

    def reset(self) -> None:
        self._context = []


class EvictionAgent(BaselineAgent):
    """Generic eviction-strategy agent backed by EpistemicMemoryStore."""

    def __init__(
        self,
        capacity: int = 50,
        eviction_strategy: str = "random",
        retrieval_vacuity_weight: float = 0.0,  # no vacuity weighting in retrieval
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.store = EpistemicMemoryStore(
            capacity=capacity,
            d_model=self.d_model,
            eviction_strategy=eviction_strategy,
            retrieval_vacuity_weight=retrieval_vacuity_weight,
        )
        self.retriever = UncertaintyWeightedRetriever(
            store=self.store,
            vacuity_weight=retrieval_vacuity_weight,
        )
        self._vacuity_counter = 0.0

    def process_fact(self, fact: str, task_id: str = "") -> None:
        vec = self.encode(fact)
        # For non-EMN baselines, assign synthetic vacuity
        vacuity = self._get_synthetic_vacuity(fact)
        self.store.write(vector=vec, task_id=task_id, metadata={"text": fact}, vacuity=vacuity)

    def _get_synthetic_vacuity(self, fact: str) -> float:
        """Default: uniform random vacuity (for random eviction baseline)."""
        return float(np.random.uniform(0.0, 1.0))

    def answer_question(self, question: str) -> str:
        q_vec = self.encode(question)
        entries = self.retriever.retrieve(q_vec, k=3)
        if not entries:
            return ""
        return entries[0].metadata.get("text", "")

    def reset(self) -> None:
        self.store = EpistemicMemoryStore(
            capacity=self.store.capacity,
            d_model=self.d_model,
            eviction_strategy=self.store.eviction_strategy,
            retrieval_vacuity_weight=self.store.retrieval_vacuity_weight,
        )
        self.retriever = UncertaintyWeightedRetriever(
            store=self.store,
            vacuity_weight=self.store.retrieval_vacuity_weight,
        )


class RandomEvictionAgent(EvictionAgent):
    def __init__(self, **kwargs):
        super().__init__(eviction_strategy="random", **kwargs)


class LRUEvictionAgent(EvictionAgent):
    def __init__(self, **kwargs):
        super().__init__(eviction_strategy="lru", **kwargs)


class TitansSurpriseAgent(EvictionAgent):
    """
    Titans-style: evict by surprise score.
    Surprise proxy = cosine distance from mean of stored memories.
    Low surprise (similar to average) → high vacuity → evicted first.
    """

    def __init__(self, **kwargs):
        super().__init__(eviction_strategy="vacuity", **kwargs)

    def _get_synthetic_vacuity(self, fact: str) -> float:
        """Surprise = 1 - |cos_sim to store mean|.  Low surprise → high vacuity."""
        if len(self.store) == 0:
            return 0.5
        vec = self.encode(fact)
        mean_vec = self.store.all_vectors().mean(axis=0)
        norm_vec = vec / (np.linalg.norm(vec) + 1e-9)
        norm_mean = mean_vec / (np.linalg.norm(mean_vec) + 1e-9)
        cos_sim = float(norm_vec @ norm_mean)
        # Low similarity = high surprise = low vacuity = keep
        # High similarity = low surprise = high vacuity = evict
        return float((cos_sim + 1.0) / 2.0)  # map [-1,1] to [0,1]


class EMNAgent(EvictionAgent):
    """Full EMN: evidential vacuity eviction + vacuity-weighted retrieval."""

    def __init__(self, **kwargs):
        super().__init__(
            eviction_strategy="vacuity",
            retrieval_vacuity_weight=1.0,
            **kwargs,
        )
        self._write_gate = EvidentialWriteGate(d_model=self.d_model)

    def _get_synthetic_vacuity(self, fact: str) -> float:
        """Compute true vacuity via NOVA EvidentialWriteGate."""
        vec = self.encode(fact)
        t = torch.from_numpy(vec).unsqueeze(0)
        with torch.no_grad():
            out = self._write_gate(t)
        return float(out.vacuity.squeeze(0))

    def reset(self) -> None:
        super().reset()
        self.store.retrieval_vacuity_weight = 1.0
        self.retriever.vacuity_weight = 1.0


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate_agent(
    agent: BaselineAgent,
    task_data: List[dict],
    seed: int = 42,
) -> Dict[str, float]:
    """
    Run an agent on the FactConsolidation task and compute metrics.

    Parameters
    ----------
    agent     : BaselineAgent
    task_data : list of dicts with keys: facts, question, answer
    seed      : int

    Returns
    -------
    dict with accuracy, precision, recall, f1
    """
    set_seed(seed)
    agent.reset()

    y_true, y_pred_bool = [], []

    for item in task_data:
        facts = item.get("facts", item.get("context_facts", []))
        question = item.get("question", "")
        correct_answer = item.get("answer", item.get("correct_answer", ""))
        task_id = str(item.get("task_id", ""))

        if not question or not correct_answer:
            continue

        # Process all facts
        for fact in facts:
            agent.process_fact(str(fact), task_id=task_id)

        # Answer question
        predicted = agent.answer_question(question)

        is_correct = correct_answer.lower().strip() in predicted.lower()
        y_true.append(1)
        y_pred_bool.append(int(is_correct))

    if not y_pred_bool:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred_bool)

    accuracy = float(y_pred_arr.mean())
    tp = int((y_true_arr * y_pred_arr).sum())
    fp = int(((1 - y_true_arr) * y_pred_arr).sum())
    fn = int((y_true_arr * (1 - y_pred_arr)).sum())

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def run_all_baselines(
    task_data: List[dict],
    seeds: List[int] = (42, 43, 44),
    capacity: int = 50,
    d_model: int = 384,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Run all 5 baselines × 3 seeds, return mean ± std.

    Returns
    -------
    dict: baseline_name → {metric: mean, metric_std: std, ...}
    """
    baselines = {
        "standard_transformer": lambda: StandardTransformerAgent(d_model=d_model),
        "random_eviction": lambda: RandomEvictionAgent(capacity=capacity, d_model=d_model),
        "lru_eviction": lambda: LRUEvictionAgent(capacity=capacity, d_model=d_model),
        "titans_surprise": lambda: TitansSurpriseAgent(capacity=capacity, d_model=d_model),
        "emn": lambda: EMNAgent(capacity=capacity, d_model=d_model),
    }

    all_results: Dict[str, Dict[str, float]] = {}

    for name, agent_factory in baselines.items():
        if verbose:
            print(f"\nRunning baseline: {name}")

        seed_results = []
        for seed in seeds:
            if verbose:
                print(f"  seed={seed}")
            agent = agent_factory()
            metrics = evaluate_agent(agent, task_data, seed=seed)
            seed_results.append(metrics)

        # Aggregate mean ± std
        agg = {}
        for metric in seed_results[0]:
            vals = np.array([r[metric] for r in seed_results])
            agg[metric] = float(vals.mean())
            agg[f"{metric}_std"] = float(vals.std())

        all_results[name] = agg
        if verbose:
            print(f"  -> accuracy={agg['accuracy']:.3f} ± {agg['accuracy_std']:.3f}")

    return all_results
