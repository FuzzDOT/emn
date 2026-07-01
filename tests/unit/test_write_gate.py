"""
Unit tests for EvidentialWriteGate.
Tests: shapes, belief sums to 1, vacuity in [0,1], gradient flow.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pytest
import torch
import numpy as np

from emn.gates.write_gate import EvidentialWriteGate
from emn.types import WriteGateOutput


D = 64


@pytest.fixture
def gate():
    return EvidentialWriteGate(d_model=D)


# ── Shape tests ───────────────────────────────────────────────────────────────

def test_output_is_write_gate_output(gate):
    x = torch.randn(4, D)
    out = gate(x)
    assert isinstance(out, WriteGateOutput)


def test_alpha_shape(gate):
    x = torch.randn(4, D)
    out = gate(x)
    assert out.alpha.shape == (4, 256)


def test_belief_shape(gate):
    x = torch.randn(4, D)
    out = gate(x)
    assert out.belief.shape == (4, 256)


def test_vacuity_shape(gate):
    x = torch.randn(4, D)
    out = gate(x)
    assert out.vacuity.shape == (4,)


def test_single_sample(gate):
    x = torch.randn(1, D)
    out = gate(x)
    assert out.vacuity.shape == (1,)
    assert out.alpha.shape == (1, 256)


def test_1d_input_auto_unsqueezed(gate):
    """1D input (d_model,) should be auto-unsqueezed to (1, d_model)."""
    x = torch.randn(D)
    out = gate(x)
    assert out.vacuity.shape == (1,)


# ── Value constraints ─────────────────────────────────────────────────────────

def test_vacuity_in_unit_interval(gate):
    x = torch.randn(32, D)
    out = gate(x)
    assert (out.vacuity >= 0.0).all()
    assert (out.vacuity <= 1.0).all()


def test_alpha_greater_than_one(gate):
    x = torch.randn(16, D)
    out = gate(x)
    assert (out.alpha >= 1.0).all(), "Alpha values must be >= 1 (evidence + 1)"


def test_expected_probabilities_sum_to_one(gate):
    """p = alpha / sum(alpha) should sum to 1."""
    x = torch.randn(8, D)
    out = gate(x)
    S = out.alpha.sum(dim=-1, keepdim=True)
    p = out.alpha / S
    sums = p.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones(8), rtol=1e-4, atol=1e-4)


def test_vacuity_matches_nova_formula(gate):
    """vacuity = n_classes / sum(alpha), clamped to [0,1]."""
    x = torch.randn(8, D)
    out = gate(x)
    n_classes = gate.n_classes
    S = out.alpha.sum(dim=-1)
    expected = (n_classes / S).clamp(0.0, 1.0)
    torch.testing.assert_close(out.vacuity, expected, rtol=1e-4, atol=1e-4)


def test_no_nan_values(gate):
    x = torch.randn(8, D)
    out = gate(x)
    assert not torch.isnan(out.alpha).any()
    assert not torch.isnan(out.belief).any()
    assert not torch.isnan(out.vacuity).any()


def test_no_inf_values(gate):
    x = torch.randn(8, D)
    out = gate(x)
    assert not torch.isinf(out.alpha).any()
    assert not torch.isinf(out.vacuity).any()


# ── Gradient flow ─────────────────────────────────────────────────────────────

def test_gradient_flows_through_gate(gate):
    x = torch.randn(4, D, requires_grad=True)
    out = gate(x)
    out.vacuity.sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_gate_parameters_receive_gradients(gate):
    x = torch.randn(4, D)
    targets = torch.randint(0, gate.n_classes, (4,))
    loss = gate.evidential_loss(x, targets)
    loss.backward()
    for name, param in gate.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_evidential_loss_is_positive(gate):
    x = torch.randn(8, D)
    targets = torch.randint(0, gate.n_classes, (8,))
    loss = gate.evidential_loss(x, targets)
    assert loss.item() > 0


# ── compute_vacuity convenience ───────────────────────────────────────────────

def test_compute_vacuity_convenience(gate):
    x = torch.randn(4, D)
    v = gate.compute_vacuity(x)
    assert v.shape == (4,)
    assert (v >= 0.0).all() and (v <= 1.0).all()


def test_mean_vacuity(gate):
    x = torch.randn(4, D)
    out = gate(x)
    mv = out.mean_vacuity()
    assert isinstance(mv, float)
    assert 0.0 <= mv <= 1.0


# ── Different configurations ──────────────────────────────────────────────────

@pytest.mark.parametrize("d_model,n_classes", [
    (32, 16), (64, 256), (128, 512),
])
def test_various_configurations(d_model, n_classes):
    gate = EvidentialWriteGate(d_model=d_model, n_classes=n_classes)
    x = torch.randn(4, d_model)
    out = gate(x)
    assert out.alpha.shape == (4, n_classes)
    assert (out.vacuity >= 0.0).all() and (out.vacuity <= 1.0).all()


# ── extra_repr ────────────────────────────────────────────────────────────────

def test_extra_repr(gate):
    r = gate.extra_repr()
    assert "d_model=64" in r
    assert "n_classes=256" in r
