# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Tests for separable (degenerate) kernels and Fredholm equations.

The references are analytic, from degenerate-kernel theory, not from a reference
quadrature: for K = Σ g_m h_m the equation u − λKu = f reduces to an R × R
linear system whose solution is exact.
"""

import math

import numpy as np
import pytest
import torch

from fastlsq.basis import Op, SinusoidalBasis
from fastlsq.kernels import (
    SeparableKernelOperator,
    degenerate_eigenvalues,
    fredholm_second_kind,
)
from fastlsq.linalg import solve_lstsq

torch.set_default_dtype(torch.float64)


def _product_kernel():
    return SeparableKernelOperator(
        [lambda x: x[:, 0]], [lambda y: y[:, 0]], 0.0, 1.0, d=1
    )


# ======================================================================
# Inner products and rank structure
# ======================================================================

def test_inner_products_match_closed_form():
    """C[0, j] = ∫₀¹ y sin(w_j y + b_j) dy, which is elementary."""
    W = torch.tensor([[2.0, 5.0, 0.5]])
    b = torch.tensor([[0.3, 1.1, 2.0]])
    basis = SinusoidalBasis(W, b, normalize=False)
    C = _product_kernel().inner_products(basis)

    w, bb = W[0], b[0]
    # ∫₀¹ y sin(wy+b) dy = [sin(w+b) - sin(b)]/w² - cos(w+b)/w
    ref = (torch.sin(w + bb) - torch.sin(bb)) / w ** 2 - torch.cos(w + bb) / w
    assert torch.allclose(C[0], ref, atol=1e-12)


def test_quadrature_is_converged_for_a_typical_basis():
    torch.manual_seed(0)
    basis = SinusoidalBasis.random(1, 200, sigma=6.0)
    assert _product_kernel().check_quadrature(basis) < 1e-10


def test_check_quadrature_flags_an_underresolved_basis():
    """A very high-bandwidth basis with few nodes must be reported, not hidden."""
    torch.manual_seed(1)
    basis = SinusoidalBasis.random(1, 60, sigma=200.0)
    coarse = SeparableKernelOperator(
        [lambda x: x[:, 0]], [lambda y: y[:, 0]], 0.0, 1.0, d=1, n_quad=8
    )
    assert coarse.check_quadrature(basis) > 1e-3


def test_block_rank_equals_number_of_terms():
    """A degenerate kernel of rank R assembles to a block of rank exactly R."""
    torch.manual_seed(2)
    basis = SinusoidalBasis.random(1, 120, sigma=5.0)
    x = torch.linspace(0, 1, 400).reshape(-1, 1)

    assert int(torch.linalg.matrix_rank(_product_kernel().apply(basis, x), tol=1e-10)) == 1

    K2 = SeparableKernelOperator(
        [lambda x: x[:, 0], lambda x: torch.sin(x[:, 0])],
        [lambda y: y[:, 0], lambda y: torch.cos(y[:, 0])],
        0.0, 1.0, d=1,
    )
    assert int(torch.linalg.matrix_rank(K2.apply(basis, x), tol=1e-10)) == 2
    assert K2.rank == 2


def test_from_inner_products_skips_quadrature():
    """Analytic inner products may be supplied directly."""
    torch.manual_seed(3)
    basis = SinusoidalBasis.random(1, 40, sigma=3.0)
    x = torch.rand(10, 1)
    C = _product_kernel().inner_products(basis)
    K = SeparableKernelOperator.from_inner_products([lambda x: x[:, 0]], C)
    assert torch.allclose(K.apply(basis, x), _product_kernel().apply(basis, x), atol=1e-14)


def test_mismatched_term_counts_raise():
    with pytest.raises(ValueError, match="equal length"):
        SeparableKernelOperator([lambda x: x[:, 0]], [], 0.0, 1.0, d=1)


# ======================================================================
# Characteristic values
# ======================================================================

def test_product_kernel_characteristic_value_is_three():
    """K = xy on [0,1] has S = ∫y·y = 1/3, so the single singular λ is 3."""
    torch.manual_seed(4)
    basis = SinusoidalBasis.random(1, 80, sigma=4.0)
    lams = degenerate_eigenvalues(_product_kernel(), basis)
    assert lams.numel() == 1
    assert abs(lams.real.item() - 3.0) < 1e-9
    assert abs(lams.imag.item()) < 1e-9


# ======================================================================
# Fredholm second kind, against the exact degenerate-kernel solution
# ======================================================================

@pytest.mark.parametrize("lam", [0.25, 0.5, 1.0, 2.0, 2.9])
def test_fredholm_matches_analytic_solution(lam):
    """u − λ∫₀¹ xy u(y)dy = f  has  u = f + λcx,  c = ∫yf / (1 − λ/3).

    Swept up to λ = 2.9, close to the singular λ = 3, so the test also shows the
    solution degrading gracefully rather than silently rather than blowing up.
    """
    torch.manual_seed(5)
    basis = SinusoidalBasis.random(1, 220, sigma=6.0)
    x = torch.linspace(0, 1, 500).reshape(-1, 1)
    w = 2.0
    f = lambda t: torch.sin(w * t)

    # ∫₀¹ y sin(wy) dy = (sin w − w cos w)/w²
    int_yf = (math.sin(w) - w * math.cos(w)) / w ** 2
    c = int_yf / (1.0 - lam / 3.0)
    exact = f(x) + lam * c * x

    L = fredholm_second_kind(_product_kernel(), lam, d=1)
    beta = solve_lstsq(L.apply(basis, x), f(x), mu=1e-12)
    u = basis.evaluate(x) @ beta
    rel = (torch.norm(u - exact) / torch.norm(exact)).item()
    assert rel < 1e-6, f"lam={lam}: rel-L2 = {rel:.2e}"


def test_fredholm_rank2_matches_analytic_solution():
    """Rank-2 kernel: (I − λS)c = b reduces the problem to a 2×2 solve."""
    torch.manual_seed(6)
    basis = SinusoidalBasis.random(1, 220, sigma=6.0)
    x = torch.linspace(0, 1, 500).reshape(-1, 1)
    lam, w = 0.4, 2.0
    g = [lambda t: t[:, 0], lambda t: torch.sin(t[:, 0])]
    h = [lambda t: t[:, 0], lambda t: torch.cos(t[:, 0])]
    K = SeparableKernelOperator(g, h, 0.0, 1.0, d=1)

    S = np.array([
        [1.0 / 3.0, math.sin(1.0) - math.cos(1.0)],
        [math.cos(1.0) + math.sin(1.0) - 1.0, 0.5 * math.sin(1.0) ** 2],
    ])
    b0 = (math.sin(w) - w * math.cos(w)) / w ** 2
    b1 = 0.5 * ((1 - math.cos(w + 1)) / (w + 1) + (1 - math.cos(w - 1)) / (w - 1))
    c = np.linalg.solve(np.eye(2) - lam * S, np.array([b0, b1]))

    f = torch.sin(w * x)
    exact = f + lam * (float(c[0]) * x + float(c[1]) * torch.sin(x))

    L = fredholm_second_kind(K, lam, d=1)
    beta = solve_lstsq(L.apply(basis, x), f, mu=1e-12)
    u = basis.evaluate(x) @ beta
    rel = (torch.norm(u - exact) / torch.norm(exact)).item()
    assert rel < 1e-6, f"rank-2 Fredholm rel-L2 = {rel:.2e}"


def test_fredholm_needs_no_boundary_rows():
    """A second-kind equation is well posed on its own -- the identity pins it.

    Solved with ``mu=0``: the point is that the *formulation* needs no boundary
    rows, and a Tikhonov ridge would confound that.  (With ``mu=1e-12`` the
    residual is ~1e-8 rather than ~1e-13, entirely from the ridge -- cond(A) is
    ~2e17 here, so even a tiny mu perturbs the solution measurably.)
    """
    torch.manual_seed(7)
    basis = SinusoidalBasis.random(1, 200, sigma=5.0)
    x = torch.linspace(0, 1, 400).reshape(-1, 1)
    rhs = torch.sin(2 * x)
    A = fredholm_second_kind(_product_kernel(), 0.5, d=1).apply(basis, x)
    beta = solve_lstsq(A, rhs, mu=0.0)
    resid = (torch.norm(A @ beta - rhs) / torch.norm(rhs)).item()
    assert resid < 1e-10, f"residual {resid:.2e}"


def test_kernel_composes_with_differential_terms():
    """Separable kernels compose with Op through the duck-typed operator protocol."""
    torch.manual_seed(8)
    basis = SinusoidalBasis.random(1, 40, sigma=3.0)
    x = torch.rand(12, 1)
    K = _product_kernel()
    L = Op.partial(0, 1, d=1) - 0.7 * K
    got = L.apply(basis, x)
    ref = basis.derivative(x, (1,)) - 0.7 * K.apply(basis, x)
    assert torch.allclose(got, ref, atol=1e-13)


def test_lambda_may_be_a_parameter():
    """λ can be an nn.Parameter, so the coupling is differentiable."""
    torch.manual_seed(9)
    basis = SinusoidalBasis.random(1, 30, sigma=3.0)
    x = torch.rand(8, 1)
    lam = torch.nn.Parameter(torch.tensor(0.5))
    A = fredholm_second_kind(_product_kernel(), lam, d=1).apply(basis, x)
    A.sum().backward()
    assert lam.grad is not None and torch.isfinite(lam.grad).all()
