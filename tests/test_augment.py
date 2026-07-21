# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Tests for polynomial / DC augmentation columns.

The columns must carry *exact* operator images -- not a zero-derivative stub --
so these check monomial derivatives, antiderivatives and iterated integrals
against closed forms and quadrature, then check that an AugmentedBasis is
transparent to every existing operator.
"""

import math

import numpy as np
import pytest
import torch

from fastlsq.augment import AugmentedBasis, PolynomialColumns
from fastlsq.basis import IntegralOperator, Op, SinusoidalBasis, SymbolOperator

torch.set_default_dtype(torch.float64)


# ======================================================================
# PolynomialColumns: exact operator images
# ======================================================================

def test_column_layout_is_graded():
    p = PolynomialColumns(degree=2, dim=2)
    assert p.n_columns == 6
    assert p.exponents.tolist() == [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]
    # The first column is always the constant (the DC column).
    assert p.exponents[0].tolist() == [0, 0]


def test_derivative_matches_closed_form():
    """d/dx x^p = p x^(p-1), including the column it annihilates."""
    p = PolynomialColumns(degree=3, dim=1)
    x = torch.linspace(0.3, 2.0, 11).reshape(-1, 1)
    got = p.derivative(x, (1,))
    ref = torch.cat([torch.zeros_like(x)] + [k * x ** (k - 1) for k in (1, 2, 3)], dim=1)
    assert torch.allclose(got, ref, atol=1e-13)


def test_antiderivative_matches_closed_form():
    """A negative multi-index integrates: x^p -> x^(p+1)/(p+1)."""
    p = PolynomialColumns(degree=3, dim=1)
    x = torch.linspace(0.3, 2.0, 11).reshape(-1, 1)
    got = p.derivative(x, (-1,))
    ref = torch.cat([x ** (k + 1) / (k + 1) for k in (0, 1, 2, 3)], dim=1)
    assert torch.allclose(got, ref, atol=1e-13)


def test_derivative_annihilates_low_order_columns():
    """d^2/dx^2 kills the constant and the ramp, exactly."""
    p = PolynomialColumns(degree=3, dim=1)
    x = torch.linspace(0.3, 2.0, 7).reshape(-1, 1)
    got = p.derivative(x, (2,))
    assert torch.allclose(got[:, :2], torch.zeros(7, 2), atol=1e-15)
    assert torch.allclose(got[:, 2], 2 * torch.ones(7), atol=1e-13)


def test_mixed_partial_in_2d():
    p = PolynomialColumns(degree=2, dim=2)
    x = torch.rand(6, 2) + 0.5
    got = p.derivative(x, (1, 1))
    idx = p.exponents.tolist().index([1, 1])
    assert torch.allclose(got[:, idx], torch.ones(6), atol=1e-13)
    # d^2/dxdy annihilates every other column of degree <= 2.
    others = [j for j in range(p.n_columns) if j != idx]
    assert torch.allclose(got[:, others], torch.zeros(6, len(others)), atol=1e-14)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_iterated_integral_matches_quadrature(order):
    """The n-fold iterated integral of the columns matches n-fold cumulative trapezoid."""
    p = PolynomialColumns(degree=3, dim=1)
    lo = 0.25
    xg = torch.linspace(lo, 2.0, 40001).reshape(-1, 1)

    cur = p.evaluate(xg)
    for _ in range(order):
        nxt = torch.zeros_like(cur)
        nxt[1:] = torch.cumulative_trapezoid(cur, xg.squeeze(), dim=0)
        cur = nxt

    got = p.iterated_integral(xg, 0, lo, order=order)
    rel = (torch.norm(got - cur) / torch.norm(cur)).item()
    assert rel < 1e-7, f"order {order}: rel-L2 = {rel:.2e}"


def test_monomials_at_negative_coordinates():
    """Negative coordinates must not produce NaN.

    The columns are built as `x ** e` with a float-typed exponent, and a negative
    base raised to a non-integral float power is NaN.  torch special-cases
    integral-valued exponents so this is well defined -- pinned here because a
    domain centred away from the origin (or any lower integration limit < 0)
    would otherwise silently poison the whole design matrix.
    """
    p = PolynomialColumns(degree=3, dim=1)
    x = torch.tensor([[-2.0], [-0.5], [0.0], [1.5]])
    v = p.evaluate(x)
    assert torch.isfinite(v).all()
    assert torch.allclose(v, torch.cat([x ** k for k in range(4)], dim=1), atol=1e-14)

    # ...and with a negative lower limit in the iterated integral.
    got = p.iterated_integral(torch.tensor([[0.85]]), 0, -0.4, order=2)
    assert torch.isfinite(got).all()

    p2 = PolynomialColumns(degree=2, dim=2)
    assert torch.isfinite(p2.evaluate(torch.tensor([[-1.0, -2.0], [0.5, -0.5]]))).all()


def test_definite_integral_of_constant_column():
    """int_a^b 1 dt = b - a, independent of the evaluation point."""
    p = PolynomialColumns(degree=0, dim=1)
    x = torch.rand(5, 1)
    got = p.definite_integral(x, 0, 0.5, upper=2.0)
    assert torch.allclose(got, 1.5 * torch.ones(5, 1), atol=1e-13)


def test_iterated_integral_is_cauchy_not_a_difference():
    """order>=2 is the Cauchy repeated integral, matching the (x-t)^(n-1)/(n-1)! kernel.

    Same distinction the sinusoidal path draws: it is NOT F_n(hi) - F_n(lo),
    which would drop the polynomial terms.
    """
    p = PolynomialColumns(degree=2, dim=1)
    lo, x_eval, n = -0.4, 0.85, 3
    tg = torch.linspace(lo, x_eval, 60001)
    vals = p.evaluate(tg.reshape(-1, 1))
    kernel = (x_eval - tg) ** (n - 1) / math.factorial(n - 1)
    ref = torch.trapz(kernel.unsqueeze(1) * vals, tg, dim=0)

    got = p.iterated_integral(torch.tensor([[x_eval]]), 0, lo, order=n).squeeze(0)
    assert torch.allclose(got, ref, rtol=1e-7)


# ======================================================================
# AugmentedBasis: transparent to every operator
# ======================================================================

def _aug(n_feat=32, degree=1, dim=1, sigma=2.0, seed=0):
    torch.manual_seed(seed)
    basis = SinusoidalBasis.random(dim, n_feat, sigma=sigma)
    return basis, AugmentedBasis(basis, PolynomialColumns(degree=degree, dim=dim))


def test_shapes_and_block_structure():
    basis, aug = _aug()
    x = torch.rand(9, 1)
    A = aug.evaluate(x)
    assert A.shape == (9, 32 + 2)
    assert torch.allclose(A[:, :32], basis.evaluate(x), atol=1e-14)


def test_diff_operator_is_transparent():
    """DiffOperator sees the augmented basis as just a wider basis."""
    basis, aug = _aug()
    x = torch.rand(9, 1)
    got = Op.partial(0, 2, d=1).apply(aug, x)
    assert torch.allclose(got[:, :32], basis.derivative(x, (2,)), atol=1e-14)
    # d^2/dx^2 annihilates a degree-1 augmentation entirely.
    assert torch.allclose(got[:, 32:], torch.zeros(9, 2), atol=1e-15)


def test_integral_operator_is_transparent():
    basis, aug = _aug()
    x = torch.rand(9, 1) + 0.1
    op = IntegralOperator.volterra(dim=0, lower=0.0, d=1, order=2)
    got = op.apply(aug, x)
    assert got.shape == (9, 34)
    assert torch.allclose(
        got[:, :32],
        basis.iterated_integral(x, 0, 0.0, order=2),
        atol=1e-12,
    )


def test_cache_is_shared_and_consistent():
    """Passing an explicit cache gives the same answer as recomputing."""
    _, aug = _aug()
    x = torch.rand(9, 1)
    cache = aug.cache(x)
    assert torch.allclose(aug.evaluate(x, cache=cache), aug.evaluate(x), atol=1e-15)
    assert torch.allclose(
        aug.derivative(x, (1,), cache=cache), aug.derivative(x, (1,)), atol=1e-15
    )


def test_gradient_and_laplacian_blocks():
    basis, aug = _aug(dim=2, degree=2, seed=3)
    x = torch.rand(7, 2)
    g = aug.gradient(x)
    assert g.shape == (7, 2, 32 + 6)
    assert torch.allclose(g[:, :, :32], basis.gradient(x), atol=1e-14)
    lap = aug.laplacian(x)
    # Laplacian of [1, x, y, x^2, xy, y^2] = [0, 0, 0, 2, 0, 2].
    assert torch.allclose(
        lap[:, 32:],
        torch.tensor([0.0, 0, 0, 2, 0, 2]).expand(7, 6),
        atol=1e-13,
    )


def test_operator_mixed_terms():
    """A multi-term operator applies term-by-term to both blocks."""
    basis, aug = _aug(degree=2)
    x = torch.rand(9, 1) + 0.5
    terms = [(1.0, (2,)), (3.0, (0,))]
    got = aug.operator(x, terms)
    ref_feat = basis.operator(x, terms)
    assert torch.allclose(got[:, :32], ref_feat, atol=1e-14)
    # (d^2/dx^2 + 3)[1, x, x^2] = [3, 3x, 2 + 3x^2]
    poly_ref = torch.cat([3 * torch.ones_like(x), 3 * x, 2 + 3 * x ** 2], dim=1)
    assert torch.allclose(got[:, 32:], poly_ref, atol=1e-13)


def test_split_recovers_blocks():
    _, aug = _aug(degree=1)
    beta = torch.randn(34, 1)
    bf, bp = aug.split(beta)
    assert bf.shape == (32, 1) and bp.shape == (2, 1)


def test_dimension_mismatch_raises():
    torch.manual_seed(0)
    basis = SinusoidalBasis.random(2, 16, sigma=1.0)
    with pytest.raises(ValueError, match="dimension mismatch"):
        AugmentedBasis(basis, PolynomialColumns(degree=1, dim=1))


# ======================================================================
# Symbol operators on augmentation columns
# ======================================================================

def test_symbol_on_dc_column_uses_m_at_zero():
    """L applied to the constant column is m(0) -- the zero-frequency plane wave."""
    _, aug = _aug(degree=0)
    x = torch.rand(6, 1)
    # A symbol that is identically 7 gives 7 on the constant column.
    got = SymbolOperator(lambda W: 7.0 * torch.ones(1, W.shape[1])).apply(aug, x)
    assert torch.allclose(got[:, 32:], 7.0 * torch.ones(6, 1), atol=1e-14)


def test_fractional_laplacian_annihilates_the_constant():
    """(-Delta)^s of a constant is 0, since the symbol ||xi||^(2s) vanishes at xi=0."""
    _, aug = _aug(degree=0)
    x = torch.rand(6, 1)
    got = SymbolOperator.fractional_laplacian(0.6).apply(aug, x)
    assert got[:, 32:].abs().max() < 1e-12


def test_symbol_on_higher_degree_columns_raises():
    """A non-constant monomial has no function-valued multiplier image -- say so."""
    _, aug = _aug(degree=1)
    x = torch.rand(6, 1)
    with pytest.raises(NotImplementedError, match="degree > 0"):
        SymbolOperator.fractional_laplacian(0.5).apply(aug, x)


# ======================================================================
# End to end
# ======================================================================

def test_dc_column_improves_a_large_offset_solution():
    """u'' = f with a large DC offset: the explicit column beats synthesising it.

    The constant is in ker(d^2/dx^2), so the PDE rows say nothing about it and it
    has to come from the basis plus the boundary rows.  A sinusoidal bank can
    approximate a constant from near-DC features, so the plain basis is not
    hopeless -- but an exact column is measurably better as the offset grows.
    """
    from fastlsq.linalg import solve_lstsq

    x = torch.linspace(0, 1, 400).reshape(-1, 1)
    xb = torch.tensor([[0.0], [1.0]])

    def run(C, augmented):
        torch.manual_seed(0)
        basis = SinusoidalBasis.random(1, 120, sigma=4.0)
        B = AugmentedBasis(basis, PolynomialColumns(0, 1)) if augmented else basis
        A_pde = Op.partial(0, 2, d=1).apply(B, x)
        f = -((2 * np.pi) ** 2) * torch.sin(2 * np.pi * x)
        w = 100.0
        A = torch.cat([A_pde, w * B.evaluate(xb)])
        rhs = torch.cat([f, w * (torch.sin(2 * np.pi * xb) + C)])
        beta = solve_lstsq(A, rhs, mu=1e-12)
        u = B.evaluate(x) @ beta
        exact = torch.sin(2 * np.pi * x) + C
        return (torch.norm(u - exact) / torch.norm(exact)).item()

    # Both are accurate; the column is the better-conditioned way to carry a
    # large offset, and never worse.
    for C in (0.0, 10.0, 100.0, 1000.0):
        plain, augmented = run(C, False), run(C, True)
        assert augmented <= plain * 1.5, f"C={C}: augmented {augmented:.2e} vs plain {plain:.2e}"
    assert run(1000.0, True) < run(1000.0, False) / 5.0
