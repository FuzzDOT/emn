"""
Confabulation Benchmark Evaluator
===================================
Evaluates four confidence estimation methods on the confabulation benchmark:
  1. Softmax confidence
  2. Temperature scaling
  3. MC Dropout
  4. EMN Vacuity

Metrics:
  - ConfabulationRate  : fraction of items where model asserts wrong answer with high confidence
  - HedgingRate        : fraction of items where model gives hedged/uncertain response
  - UpdateRate         : fraction of items where model updates toward correct answer after context
  - AUROC              : treating confidence as score for distinguishing correct vs wrong predictions
  - AUPRC              : area under precision-recall curve
  - ECE                : Expected Calibration Error of the confidence signal
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None
from emn.benchmarks.confabulation.dataset import ConfabulationItem


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ItemResult:
    fact_id: str
    severity: int
    correct_answer: str
    wrong_answer: str
    model_answer: str
    is_correct: bool
    is_hedged: bool
    confidence: float          # confidence score in [0, 1]
    vacuity: Optional[float]   # EMN vacuity score (None for non-EMN methods)
    method: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalResults:
    method: str
    confabulation_rate: float
    hedging_rate: float
    update_rate: float
    auroc: float
    auprc: float
    ece: float
    n_items: int
    per_severity: Dict[int, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Hedging detection
# ---------------------------------------------------------------------------

HEDGE_PATTERNS = [
    r"\b(i'm not sure|i am not sure)\b",
    r"\b(i don't know|i do not know)\b",
    r"\b(i'm uncertain|i am uncertain)\b",
    r"\buncertain\b",
    r"\bpossibly\b",
    r"\bperhaps\b",
    r"\bmight be\b",
    r"\bcould be\b",
    r"\bit's unclear\b",
    r"\bit is unclear\b",
    r"\bcannot confirm\b",
    r"\bcan't confirm\b",
    r"\bI cannot be sure\b",
    r"\bnot confident\b",
]
_HEDGE_RE = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)


def is_hedged_response(text: str) -> bool:
    """Return True if the response contains hedging language."""
    return bool(_HEDGE_RE.search(text))


def answer_matches(model_answer: str, reference: str) -> bool:
    """
    Fuzzy match: True if reference is a substring of model_answer (case-insensitive).
    """
    return reference.lower() in model_answer.lower()


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def compute_ece(
    confidences: np.ndarray,
    is_correct: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error.

    Parameters
    ----------
    confidences : (N,) float in [0, 1]
    is_correct  : (N,) bool
    n_bins      : int

    Returns
    -------
    float — ECE in [0, 1]
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(confidences)

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = is_correct[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / N) * abs(acc - conf)

    return float(ece)


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    AUROC treating `scores` as classifier scores for `labels` (1=positive=correct).

    Parameters
    ----------
    scores : (N,) — confidence or (1-vacuity)
    labels : (N,) — 1 if correct, 0 if wrong

    Returns
    -------
    float — AUROC in [0, 1]
    """
    from sklearn.metrics import roc_auc_score
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5  # degenerate case
    return float(roc_auc_score(labels, scores))


def compute_auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUPRC."""
    from sklearn.metrics import average_precision_score
    if labels.sum() == 0:
        return 0.0
    return float(average_precision_score(labels, scores))


# ---------------------------------------------------------------------------
# Confidence methods
# ---------------------------------------------------------------------------

class SoftmaxConfidence:
    """Standard max softmax probability as confidence."""

    def __init__(self, model: nn.Module, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def score(self, prompt: str, candidate: str) -> float:
        """Return max softmax prob on the candidate tokens."""
        full = prompt + " " + candidate
        tokens = self.tokenizer(
            full, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**tokens)
            logits = out.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
        return float(probs.max().cpu())

    def generate(self, prompt: str, max_new_tokens: int = 64) -> Tuple[str, float]:
        tokens = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        gen_ids = out.sequences[0][tokens["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Confidence = geometric mean of max-softmax per token
        if out.scores:
            per_token_conf = [
                float(F.softmax(s, dim=-1).max().cpu()) for s in out.scores
            ]
            confidence = float(np.exp(np.mean(np.log(np.clip(per_token_conf, 1e-9, 1.0)))))
        else:
            confidence = 0.5
        return text, confidence


class TemperatureScaledConfidence(SoftmaxConfidence):
    """Softmax with temperature scaling calibration."""

    def __init__(self, model: nn.Module, tokenizer, temperature: float = 1.5, device: str = "cpu"):
        super().__init__(model, tokenizer, device)
        self.temperature = temperature

    def generate(self, prompt: str, max_new_tokens: int = 64) -> Tuple[str, float]:
        tokens = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        gen_ids = out.sequences[0][tokens["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        if out.scores:
            per_token_conf = [
                float(F.softmax(s / self.temperature, dim=-1).max().cpu())
                for s in out.scores
            ]
            confidence = float(np.exp(np.mean(np.log(np.clip(per_token_conf, 1e-9, 1.0)))))
        else:
            confidence = 0.5
        return text, confidence


class MCDropoutConfidence:
    """MC Dropout uncertainty estimation — multiple stochastic forward passes."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        n_passes: int = 10,
        dropout_p: float = 0.1,
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.n_passes = n_passes
        self.dropout_p = dropout_p
        self.device = device

    def _enable_dropout(self):
        """Enable dropout at inference time."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def generate(self, prompt: str, max_new_tokens: int = 64) -> Tuple[str, float]:
        tokens = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        # First pass: deterministic for the answer
        self.model.eval()
        with torch.no_grad():
            det_out = self.model.generate(
                **tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        gen_ids = det_out[0][tokens["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Multiple stochastic passes for uncertainty estimation
        self._enable_dropout()
        logit_samples = []
        with torch.no_grad():
            for _ in range(self.n_passes):
                out = self.model(**tokens)
                logit_samples.append(F.softmax(out.logits[:, -1, :], dim=-1).cpu())

        self.model.eval()

        # Confidence = 1 - variance in predictions (mutual information proxy)
        stacked = torch.stack(logit_samples, dim=0)  # (passes, batch, vocab)
        mean_pred = stacked.mean(dim=0)
        predictive_entropy = -(mean_pred * (mean_pred + 1e-9).log()).sum(dim=-1)
        aleatoric = (stacked * (stacked + 1e-9).log()).sum(dim=-1).mean(dim=0).neg()
        mutual_info = (predictive_entropy - aleatoric).clamp(min=0.0)

        # Confidence = 1 - normalised mutual information
        vocab_size = stacked.shape[-1]
        max_entropy = float(torch.log(torch.tensor(float(vocab_size))))
        confidence = float(1.0 - (mutual_info / (max_entropy + 1e-9)).clamp(0.0, 1.0).mean())

        return text, confidence


class EMNVacuityConfidence:
    """
    EMN vacuity-based confidence via EvidentialWriteGate applied to hidden states.
    Confidence = 1 - vacuity.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        write_gate,    # EvidentialWriteGate
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.write_gate = write_gate
        self.device = device

    def generate(self, prompt: str, max_new_tokens: int = 64) -> Tuple[str, float]:
        tokens = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(
                **tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        gen_ids = out.sequences[0][tokens["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Extract last hidden state from the last generated position
        # out.hidden_states is a tuple of (n_layers+1, batch, seq, d_model)
        try:
            if out.hidden_states and len(out.hidden_states) > 0:
                # Last generation step, last layer, pool over sequence
                last_step_hidden = out.hidden_states[-1][-1]  # (batch, seq, d_model)
                pooled = last_step_hidden.mean(dim=1)  # (batch, d_model)

                # Ensure d_model matches write_gate
                gate_d = self.write_gate.d_model
                if pooled.shape[-1] != gate_d:
                    # Truncate or pad
                    if pooled.shape[-1] > gate_d:
                        pooled = pooled[:, :gate_d]
                    else:
                        pad = torch.zeros(pooled.shape[0], gate_d - pooled.shape[-1], device=self.device)
                        pooled = torch.cat([pooled, pad], dim=-1)

                gate_out = self.write_gate(pooled)
                vacuity = float(gate_out.vacuity.mean().cpu())
                confidence = 1.0 - vacuity
            else:
                confidence = 0.5
        except Exception:
            confidence = 0.5

        return text, confidence


# ---------------------------------------------------------------------------
# Full evaluator
# ---------------------------------------------------------------------------

class ConfabulationEvaluator:
    """
    Runs all four confidence methods over the benchmark dataset.

    Parameters
    ----------
    items         : List[ConfabulationItem]
    model         : HF CausalLM (TinyLlama or similar)
    tokenizer     : HF tokenizer
    write_gate    : EvidentialWriteGate (for EMN method)
    device        : str
    max_new_tokens: int
    """

    def __init__(
        self,
        items: List[ConfabulationItem],
        model: nn.Module,
        tokenizer,
        write_gate,
        device: str = "cpu",
        max_new_tokens: int = 64,
        temperature_scale: float = 1.5,
        mc_passes: int = 10,
    ):
        self.items = items
        self.model = model
        self.tokenizer = tokenizer
        self.write_gate = write_gate
        self.device = device
        self.max_new_tokens = max_new_tokens

        self.methods = {
            "softmax": SoftmaxConfidence(model, tokenizer, device),
            "temperature": TemperatureScaledConfidence(model, tokenizer, temperature_scale, device),
            "mc_dropout": MCDropoutConfidence(model, tokenizer, mc_passes, device=device),
            "emn_vacuity": EMNVacuityConfidence(model, tokenizer, write_gate, device),
        }

    def _build_prompt(self, item: ConfabulationItem) -> str:
        return (
            f"Context: {item.contradiction}\n\n"
            f"Question: {item.question}\n"
            f"Please answer concisely."
        )

    def evaluate_method(
        self,
        method_name: str,
        max_items: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[List[ItemResult], EvalResults]:
        """
        Run a single confidence method over all benchmark items.

        Parameters
        ----------
        method_name : str — one of "softmax", "temperature", "mc_dropout", "emn_vacuity"
        max_items   : int | None — limit for fast testing
        verbose     : bool

        Returns
        -------
        (item_results, aggregate_metrics)
        """
        scorer = self.methods[method_name]
        items = self.items[:max_items] if max_items else self.items
        results: List[ItemResult] = []

        for i, item in enumerate(items):
            if verbose and i % 50 == 0:
                print(f"  [{method_name}] {i}/{len(items)}")

            prompt = self._build_prompt(item)
            try:
                model_answer, confidence = scorer.generate(
                    prompt, max_new_tokens=self.max_new_tokens
                )
            except Exception as e:
                model_answer = ""
                confidence = 0.5
                if verbose:
                    print(f"  Warning: generation failed for {item.fact_id}: {e}")

            is_correct = answer_matches(model_answer, item.correct_answer)
            is_hedged = is_hedged_response(model_answer)

            # Extract vacuity for EMN method
            vacuity = None
            if method_name == "emn_vacuity":
                vacuity = 1.0 - confidence

            results.append(ItemResult(
                fact_id=item.fact_id,
                severity=item.severity,
                correct_answer=item.correct_answer,
                wrong_answer=item.wrong_answer,
                model_answer=model_answer,
                is_correct=is_correct,
                is_hedged=is_hedged,
                confidence=confidence,
                vacuity=vacuity,
                method=method_name,
            ))

        aggregate = self._compute_aggregate(results, method_name)
        return results, aggregate

    def evaluate_all(
        self,
        max_items: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, EvalResults]:
        """Run all four methods and return aggregated results."""
        all_results = {}
        for method_name in self.methods:
            if verbose:
                print(f"\nEvaluating method: {method_name}")
            _, agg = self.evaluate_method(method_name, max_items=max_items, verbose=verbose)
            all_results[method_name] = agg
        return all_results

    def _compute_aggregate(
        self, results: List[ItemResult], method_name: str
    ) -> EvalResults:
        is_correct_arr = np.array([r.is_correct for r in results], dtype=float)
        is_hedged_arr = np.array([r.is_hedged for r in results], dtype=float)
        confidence_arr = np.array([r.confidence for r in results], dtype=float)

        # ConfabulationRate: wrong answer + high confidence (>= 0.5) + not hedged
        confab_mask = (
            (~is_correct_arr.astype(bool)) &
            (confidence_arr >= 0.5) &
            (~is_hedged_arr.astype(bool))
        )
        confabulation_rate = float(confab_mask.mean())
        hedging_rate = float(is_hedged_arr.mean())

        # UpdateRate: we don't have a pre/post comparison here, so
        # use proxy: correct answer despite contradiction context
        update_rate = float(is_correct_arr.mean())

        # AUROC/AUPRC: confidence as score, correctness as label
        auroc = compute_auroc(confidence_arr, is_correct_arr)
        auprc = compute_auprc(confidence_arr, is_correct_arr)
        ece = compute_ece(confidence_arr, is_correct_arr.astype(bool))

        # Per-severity breakdown
        per_severity: Dict[int, Dict[str, float]] = {}
        for sev in range(1, 6):
            sev_mask = np.array([r.severity == sev for r in results])
            if sev_mask.sum() == 0:
                continue
            sev_correct = is_correct_arr[sev_mask]
            sev_conf = confidence_arr[sev_mask]
            sev_hedge = is_hedged_arr[sev_mask]
            sev_confab = (
                (~sev_correct.astype(bool)) &
                (sev_conf >= 0.5) &
                (~sev_hedge.astype(bool))
            )
            per_severity[sev] = {
                "confabulation_rate": float(sev_confab.mean()),
                "hedging_rate": float(sev_hedge.mean()),
                "accuracy": float(sev_correct.mean()),
                "mean_confidence": float(sev_conf.mean()),
            }

        return EvalResults(
            method=method_name,
            confabulation_rate=confabulation_rate,
            hedging_rate=hedging_rate,
            update_rate=update_rate,
            auroc=auroc,
            auprc=auprc,
            ece=ece,
            n_items=len(results),
            per_severity=per_severity,
        )
