# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Tests for MultiIntegralOperator -- closed-form integration over several axes.

Validated against the existing single-axis path (which it must reproduce
exactly), against tensor-product quadrature, and against hand-computed integrals
for DC features.
"""

import numpy as np
import pytest
import torch

from fastlsq.basis import (
    IntegralOperator,
    MultiIntegralOperator,
    Op,
    SinusoidalBasis,
)

torch.set_default_dtype(torch.float64)


def test_single_axis_reduces_to_definite_integral():
    """With one axis it must be bit-identical to IntegralOperator.definite."""
    torch.manual_seed(0)
    basis = SinusoidalBasis.random(2, 24, sigma=2.0)
    x = torch.rand(6, 2)
    got = MultiIntegralOperator.definite([0], [0.0], [1.0], d=2).apply(basis, x)
    ref = IntegralOperator.definite(dim=0, lower=0.0, upper=1.0, d=2).apply(basis, x)
    assert torch.allclose(got, ref, atol=1e-14)


def test_single_axis_volterra_reduces_too():
    torch.manual_seed(1)
    basis = SinusoidalBasis.random(2, 24, sigma=2.0)
    x = torch.rand(6, 2)
    got = MultiIntegralOperator.volterra([1], [0.0], d=2).apply(basis, x)
    ref = IntegralOperator.volterra(dim=1, lower=0.0, d=2).apply(basis, x)
    assert torch.allclose(got, ref, atol=1e-14)


def test_double_definite_matches_tensor_quadrature():
    """∫₀¹∫₀¹ u dx dy against a 2-D trapezoid reference."""
    torch.manual_seed(2)
    basis = SinusoidalBasis.random(2, 20, sigma=2.0, normalize=False)
    beta = torch.randn(20, 1)

    n = 1200
    g = torch.linspace(0, 1, n)
    XX, YY = torch.meshgrid(g, g, indexing="ij")
    pts = torch.stack([XX.reshape(-1), YY.reshape(-1)], dim=1)
    vals = (basis.evaluate(pts) @ beta).reshape(n, n)
    ref = torch.trapz(torch.trapz(vals, g, dim=1), g)

    got = MultiIntegralOperator.definite([0, 1], [0.0, 0.0], [1.0, 1.0], d=2)
    out = got.apply(basis, torch.rand(4, 2)) @ beta
    assert (out - ref).abs().max() < 1e-5
    # A fully definite integral is a functional: identical for every row.
    assert (out - out[0]).abs().max() < 1e-14


def test_mixed_definite_and_volterra():
    """Definite in axis 0, running in axis 1 -- the space-time memory shape."""
    torch.manual_seed(3)
    basis = SinusoidalBasis.random(2, 16, sigma=2.0, normalize=False)
    beta = torch.randn(16, 1)
    x = torch.tensor([[0.3, 0.7], [0.5, 0.25]])

    got = (
        MultiIntegralOperator([0, 1], [0.0, 0.0], d=2, uppers=[1.0, None])
        .apply(basis, x)
        @ beta
    ).squeeze()

    g = torch.linspace(0, 1, 800)
    for i, (_, t_row) in enumerate(x.tolist()):
        tg = torch.linspace(0, t_row, 800)
        inner = []
        for tv in tg:
            pg = torch.stack([g, torch.full_like(g, tv)], dim=1)
            inner.append(torch.trapz((basis.evaluate(pg) @ beta).squeeze(), g))
        ref = torch.trapz(torch.stack(inner), tg)
        assert abs(got[i].item() - ref.item()) < 1e-5


def test_dc_features_are_exact_and_finite():
    """A feature with zero frequency on an axis integrates to Delta, not 0/0."""
    W = torch.tensor([[0.0, 1.5], [2.0, 0.0]])
    b = torch.tensor([[0.4, 0.9]])
    basis = SinusoidalBasis(W, b, normalize=False)
    got = MultiIntegralOperator.definite([0, 1], [0.0, 0.0], [1.0, 1.0], d=2).apply(
        basis, torch.zeros(1, 2)
    )
    assert torch.isfinite(got).all()
    # feature 0 = sin(2y+0.4): ∫₀¹∫₀¹ = (cos 0.4 − cos 2.4)/2
    assert abs(got[0, 0].item() - (np.cos(0.4) - np.cos(2.4)) / 2) < 1e-13
    # feature 1 = sin(1.5x+0.9): ∫₀¹∫₀¹ = (cos 0.9 − cos 2.4)/1.5
    assert abs(got[0, 1].item() - (np.cos(0.9) - np.cos(2.4)) / 1.5) < 1e-13


def test_three_axes():
    """Extends past two axes -- ∫∫∫ over the unit cube."""
    torch.manual_seed(4)
    basis = SinusoidalBasis.random(3, 12, sigma=1.5, normalize=False)
    beta = torch.randn(12, 1)
    got = (
        MultiIntegralOperator.definite([0, 1, 2], [0.0] * 3, [1.0] * 3, d=3)
        .apply(basis, torch.rand(2, 3))
        @ beta
    )
    n = 60
    g = torch.linspace(0, 1, n)
    G = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), dim=-1).reshape(-1, 3)
    vals = (basis.evaluate(G) @ beta).reshape(n, n, n)
    ref = torch.trapz(torch.trapz(torch.trapz(vals, g, dim=2), g, dim=1), g)
    assert abs(got[0].item() - ref.item()) < 1e-3


def test_composes_with_differential_operator():
    torch.manual_seed(5)
    basis = SinusoidalBasis.random(2, 16, sigma=2.0)
    x = torch.rand(5, 2)
    L = Op.partial(0, 1, d=2) + 2.0 * MultiIntegralOperator.volterra([0, 1], [0.0, 0.0], d=2)
    got = L.apply(basis, x)
    ref = basis.derivative(x, (1, 0)) + 2.0 * MultiIntegralOperator.volterra(
        [0, 1], [0.0, 0.0], d=2
    ).apply(basis, x)
    assert torch.allclose(got, ref, atol=1e-14)


def test_repeated_axis_raises():
    torch.manual_seed(6)
    basis = SinusoidalBasis.random(2, 8, sigma=1.0)
    with pytest.raises(ValueError, match="distinct"):
        MultiIntegralOperator.definite([0, 0], [0.0, 0.0], [1.0, 1.0], d=2).apply(
            basis, torch.rand(3, 2)
        )


def test_mismatched_limit_lengths_raise():
    torch.manual_seed(7)
    basis = SinusoidalBasis.random(2, 8, sigma=1.0)
    with pytest.raises(ValueError, match="one entry per dim"):
        MultiIntegralOperator([0, 1], [0.0], d=2).apply(basis, torch.rand(3, 2))
