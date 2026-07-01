"""
NOVA Uncertainty — patched for EMN self-contained use.

This is NOVA's evidential_uncertainty.py verbatim, with one change:
the import of research_core.types is replaced with emn.types so the
repository has no external dependency on the NOVA research_core package.

Original source: NOVA Project Coffeemaker, Stage 0
DOI: 10.5281/zenodo.20562861
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from emn.types import CognitiveState, UncertaintyBundle, ABSTENTION_THRESHOLD, MIN_CONFIDENCE_TO_ANSWER


class EvidentialHead(nn.Module):
    """
    Evidential uncertainty head.

    Maps hidden states to Dirichlet concentration parameters (alpha).
    Evidence e_k = alpha_k - 1 >= 0 for all classes k.

    Loss: Negative log marginal likelihood of Dirichlet-Categorical.
    Epistemic uncertainty: 1 / (sum of alpha) — inverse total evidence.
    Aleatoric uncertainty: spread of Dirichlet distribution.
    """

    def __init__(self, d_model: int, n_classes: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, n_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        nn.init.constant_(self.head[-1].bias, 0.0)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        hidden : (batch, d_model)

        Returns
        -------
        alpha         : (batch, n_classes) Dirichlet concentration params >= 1
        epistemic_unc : (batch,)  vacuity = n_classes / S
        aleatoric_unc : (batch,)  normalised entropy of expected distribution
        """
        logits = self.head(hidden)
        alpha = F.softplus(logits) + 1.0

        S = alpha.sum(dim=-1)
        p = alpha / S.unsqueeze(-1)

        epistemic_unc = (self.n_classes / S).clamp(0.0, 1.0)

        eps = 1e-9
        aleatoric_unc = -(p * (p + eps).log()).sum(dim=-1) / np.log(self.n_classes)
        aleatoric_unc = aleatoric_unc.clamp(0.0, 1.0)

        return alpha, epistemic_unc, aleatoric_unc

    def evidential_loss(
        self,
        alpha: torch.Tensor,
        targets: torch.Tensor,
        kl_weight: float = 0.01,
    ) -> torch.Tensor:
        n_classes = alpha.shape[-1]
        S = alpha.sum(dim=-1)

        one_hot = F.one_hot(targets, n_classes).float()
        nll = -((one_hot * (alpha / S.unsqueeze(-1) + 1e-9).log()).sum(dim=-1))

        alpha_tilde = one_hot + (1 - one_hot) * alpha
        ones = torch.ones_like(alpha_tilde)
        S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)
        S_ones = ones.sum(dim=-1, keepdim=True)

        kl = (
            torch.lgamma(S_tilde) - torch.lgamma(S_ones)
            - (torch.lgamma(alpha_tilde) - torch.lgamma(ones)).sum(dim=-1, keepdim=True)
            + ((alpha_tilde - ones) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(
                dim=-1, keepdim=True
            )
        ).squeeze(-1)

        loss = (nll + kl_weight * kl).mean()
        return loss


class ConformalCalibrator(nn.Module):
    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()
        self.alpha = alpha
        self._calibration_scores: list[float] = []
        self._quantile: float = 0.9

    def update_calibration(self, scores: list[float]) -> None:
        self._calibration_scores.extend(scores)
        if len(self._calibration_scores) >= 10:
            n = len(self._calibration_scores)
            level = np.ceil((n + 1) * (1 - self.alpha)) / n
            level = float(np.clip(level, 0.0, 1.0))
            self._quantile = float(np.quantile(self._calibration_scores, level))

    def nonconformity_score(self, probs: torch.Tensor) -> torch.Tensor:
        return 1.0 - probs.max(dim=-1).values

    def is_anomalous(self, scores: torch.Tensor) -> torch.Tensor:
        return scores > self._quantile


class UncertaintyEstimationModule(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        abstention_threshold: float = ABSTENTION_THRESHOLD,
        conformal_alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.abstention_threshold = abstention_threshold
        self.n_uncertainty_classes = 256
        self.evidential = EvidentialHead(d_model, self.n_uncertainty_classes)
        self.conformal = ConformalCalibrator(alpha=conformal_alpha)

        self.abstention_gate = nn.Sequential(
            nn.Linear(3, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        for m in self.abstention_gate:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        state: CognitiveState,
        logits: torch.Tensor,
    ) -> tuple[CognitiveState, UncertaintyBundle]:
        B, T, V = logits.shape

        mask = state.attention_mask.float().unsqueeze(-1)
        pooled = (state.hidden * mask).sum(1) / mask.sum(1).clamp(min=1)

        alpha, epistemic, aleatoric = self.evidential(pooled)

        last_logits = logits[:, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        eps = 1e-9
        entropy = -(probs * (probs + eps).log()).sum(dim=-1) / np.log(V)

        conf_score = self.conformal.nonconformity_score(probs)
        mutual_info = (epistemic * (1.0 - aleatoric)).clamp(0.0, 1.0)

        gate_input = torch.stack([epistemic, aleatoric, conf_score], dim=-1)
        abstain_prob = self.abstention_gate(gate_input).squeeze(-1)
        should_abstain = abstain_prob > self.abstention_threshold

        max_prob = probs.max(dim=-1).values
        should_abstain = should_abstain | (max_prob < MIN_CONFIDENCE_TO_ANSWER)

        total_unc = (0.4 * epistemic + 0.4 * aleatoric + 0.2 * entropy).clamp(0.0, 1.0)

        abstention_reason = ""
        if should_abstain.any():
            reasons = []
            if epistemic.mean() > 0.7:
                reasons.append("high_epistemic")
            if conf_score.mean() > self.conformal._quantile:
                reasons.append("ood_conformal")
            if max_prob.mean() < MIN_CONFIDENCE_TO_ANSWER:
                reasons.append("low_confidence")
            abstention_reason = "|".join(reasons) if reasons else "threshold"

        bundle = UncertaintyBundle(
            total=total_unc.mean().item(),
            epistemic=epistemic.mean().item(),
            aleatoric=aleatoric.mean().item(),
            conformal_score=conf_score.mean().item(),
            entropy=entropy.mean().item(),
            mutual_information=mutual_info.mean().item(),
            should_abstain=should_abstain.any().item(),
            abstention_reason=abstention_reason,
        )

        per_token_probs = torch.softmax(logits, dim=-1)
        per_token_entropy = -(per_token_probs * (per_token_probs + 1e-9).log()).sum(dim=-1)
        per_token_entropy = (per_token_entropy / np.log(V)).clamp(0.0, 1.0)
        updated_per_token_unc = (0.6 * state.uncertainty + 0.4 * per_token_entropy).clamp(0.0, 1.0)

        new_state = CognitiveState(
            hidden=state.hidden,
            uncertainty=updated_per_token_unc,
            epistemic_uncertainty=epistemic.unsqueeze(-1).expand_as(state.epistemic_uncertainty),
            aleatoric_uncertainty=aleatoric.unsqueeze(-1).expand_as(state.aleatoric_uncertainty),
            attention_mask=state.attention_mask,
            memory_keys=state.memory_keys,
            memory_values=state.memory_values,
            routing_weights=state.routing_weights,
            computation_depth=state.computation_depth,
            should_abstain=should_abstain,
            trace=list(state.trace),
        )
        new_state.record(
            f"uncertainty:epistemic={bundle.epistemic:.3f},"
            f"aleatoric={bundle.aleatoric:.3f},"
            f"abstain={bundle.should_abstain}"
        )
        return new_state, bundle

    def evidential_loss(self, state: CognitiveState, targets: torch.Tensor) -> torch.Tensor:
        mask = state.attention_mask.float().unsqueeze(-1)
        pooled = (state.hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        alpha, _, _ = self.evidential(pooled)
        return self.evidential.evidential_loss(alpha, targets)

    @property
    def name(self) -> str:
        return "uncertainty_estimation"

    def get_interpretability_hooks(self) -> dict[str, torch.Tensor]:
        return {}
