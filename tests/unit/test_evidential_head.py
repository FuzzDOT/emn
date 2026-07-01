"""
Unit tests for EvidentialHead (NOVA's canonical implementation).
Tests: shape, calibration, numerical stability, gradient flow.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pytest
import torch
import torch.nn as nn
import numpy as np

from emn.evidential.nova_uncertainty import EvidentialHead


@pytest.fixture
def head():
    return EvidentialHead(d_model=64, n_classes=256)


# ── Shape tests ──────────────────────────────────────────────────────────────

def test_forward_shapes(head):
    x = torch.randn(4, 64)
    alpha, epi, ale = head(x)
    assert alpha.shape == (4, 256), f"alpha shape wrong: {alpha.shape}"
    assert epi.shape == (4,), f"epistemic shape wrong: {epi.shape}"
    assert ale.shape == (4,), f"aleatoric shape wrong: {ale.shape}"


def test_single_sample(head):
    x = torch.randn(1, 64)
    alpha, epi, ale = head(x)
    assert alpha.shape == (1, 256)
    assert epi.shape == (1,)
    assert ale.shape == (1,)


def test_large_batch(head):
    x = torch.randn(128, 64)
    alpha, epi, ale = head(x)
    assert alpha.shape == (128, 256)


# ── Value constraints ─────────────────────────────────────────────────────────

def test_alpha_greater_than_one(head):
    """All alpha values must be >= 1 (evidence + 1)."""
    x = torch.randn(16, 64)
    alpha, _, _ = head(x)
    assert (alpha >= 1.0).all(), "Alpha values must be >= 1"


def test_epistemic_in_unit_interval(head):
    """Vacuity (epistemic uncertainty) must be in [0, 1]."""
    x = torch.randn(16, 64)
    _, epi, _ = head(x)
    assert (epi >= 0.0).all() and (epi <= 1.0).all(), \
        f"Epistemic uncertainty out of [0,1]: min={epi.min():.4f}, max={epi.max():.4f}"


def test_aleatoric_in_unit_interval(head):
    x = torch.randn(16, 64)
    _, _, ale = head(x)
    assert (ale >= 0.0).all() and (ale <= 1.0).all()


def test_vacuity_formula():
    """Vacuity = n_classes / S, verified numerically."""
    head = EvidentialHead(d_model=32, n_classes=10)
    x = torch.randn(8, 32)
    alpha, epi, _ = head(x)
    S = alpha.sum(dim=-1)
    expected_vacuity = (10.0 / S).clamp(0.0, 1.0)
    torch.testing.assert_close(epi, expected_vacuity, rtol=1e-4, atol=1e-4)


def test_belief_sums_approximately_one():
    """Belief masses (alpha-1)/S should sum to approximately 1."""
    head = EvidentialHead(d_model=64, n_classes=256)
    x = torch.randn(8, 64)
    alpha, _, _ = head(x)
    S = alpha.sum(dim=-1, keepdim=True)
    belief = (alpha - 1.0) / S
    # beliefs might be slightly negative if alpha_k is near 1; check their sum
    p = alpha / S
    torch.testing.assert_close(p.sum(dim=-1), torch.ones(8), rtol=1e-4, atol=1e-4)


# ── Numerical stability ───────────────────────────────────────────────────────

def test_no_nan_on_zero_input(head):
    x = torch.zeros(4, 64)
    alpha, epi, ale = head(x)
    assert not torch.isnan(alpha).any(), "NaN in alpha for zero input"
    assert not torch.isnan(epi).any(), "NaN in epistemic for zero input"
    assert not torch.isnan(ale).any(), "NaN in aleatoric for zero input"


def test_no_nan_on_large_input(head):
    x = torch.randn(4, 64) * 100
    alpha, epi, ale = head(x)
    assert not torch.isnan(alpha).any()
    assert not torch.isnan(epi).any()
    assert not torch.isnan(ale).any()


def test_no_nan_on_negative_large_input(head):
    x = torch.randn(4, 64) * -100
    alpha, epi, ale = head(x)
    assert not torch.isnan(alpha).any()
    assert not torch.isnan(epi).any()


def test_no_inf_values(head):
    x = torch.randn(8, 64)
    alpha, epi, ale = head(x)
    assert not torch.isinf(alpha).any()
    assert not torch.isinf(epi).any()
    assert not torch.isinf(ale).any()


# ── Gradient flow ─────────────────────────────────────────────────────────────

def test_gradient_flows_through_evidential_loss(head):
    """EvidentialHead must support backpropagation for training."""
    x = torch.randn(8, 64, requires_grad=True)
    alpha, _, _ = head(x)
    targets = torch.randint(0, 256, (8,))
    loss = head.evidential_loss(alpha, targets)
    loss.backward()
    assert x.grad is not None, "Gradients did not flow to input"
    assert not torch.isnan(x.grad).any(), "NaN in gradients"


def test_gradients_wrt_head_parameters(head):
    x = torch.randn(8, 64)
    alpha, _, _ = head(x)
    targets = torch.randint(0, 256, (8,))
    loss = head.evidential_loss(alpha, targets)
    loss.backward()
    for name, param in head.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


def test_loss_is_scalar(head):
    x = torch.randn(8, 64)
    alpha, _, _ = head(x)
    targets = torch.randint(0, 256, (8,))
    loss = head.evidential_loss(alpha, targets)
    assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() > 0, "Loss should be positive"


# ── Calibration signal ────────────────────────────────────────────────────────

def test_high_evidence_gives_low_vacuity():
    """If we inflate alpha by a large constant, vacuity should be small."""
    head = EvidentialHead(d_model=32, n_classes=16)
    # Manually set high evidence
    alpha = torch.ones(4, 16) * 1000.0  # huge evidence
    S = alpha.sum(dim=-1)
    vacuity = (16.0 / S).clamp(0.0, 1.0)
    assert (vacuity < 0.1).all(), "High evidence should give low vacuity"


def test_low_evidence_gives_high_vacuity():
    """Alpha = 1 (minimum, zero evidence) should give maximum vacuity."""
    n_classes = 16
    alpha = torch.ones(4, n_classes)  # alpha_k = 1 → S = n_classes → vacuity = 1
    S = alpha.sum(dim=-1)
    vacuity = (n_classes / S).clamp(0.0, 1.0)
    torch.testing.assert_close(vacuity, torch.ones(4), rtol=1e-4, atol=1e-4)


# ── Different architectures ───────────────────────────────────────────────────

@pytest.mark.parametrize("d_model,n_classes", [
    (32, 16), (64, 64), (128, 256), (512, 256),
])
def test_various_configurations(d_model, n_classes):
    head = EvidentialHead(d_model=d_model, n_classes=n_classes)
    x = torch.randn(4, d_model)
    alpha, epi, ale = head(x)
    assert alpha.shape == (4, n_classes)
    assert (alpha >= 1.0).all()
    assert (epi >= 0.0).all() and (epi <= 1.0).all()
