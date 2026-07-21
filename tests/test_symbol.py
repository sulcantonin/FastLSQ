# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Tests for SymbolOperator -- Fourier-multiplier (nonlocal) operators.

The claim under test is that a multiplier acts *exactly* and diagonally on a
sinusoidal feature bank, so these check against independent references rather
than against the implementation:

  * integer orders must reproduce the existing exact differential operators;
  * the fractional order must agree with the singular-integral (textbook)
    definition of (-Delta)^s, including its normalising constant;
  * convolution must agree with direct numerical quadrature of the convolution.
"""

import math

import numpy as np
import pytest
import torch

from fastlsq.basis import SinusoidalBasis, SymbolOperator, Op

torch.set_default_dtype(torch.float64)


# ======================================================================
# Consistency with the exact differential operators already in the package
# ======================================================================

def test_s_equals_one_reproduces_negative_laplacian():
    """(-Delta)^1 must equal -Delta to machine precision (it is the same operator)."""
    torch.manual_seed(0)
    basis = SinusoidalBasis.random(2, 64, sigma=2.0)
    x = torch.rand(50, 2)
    got = SymbolOperator.fractional_laplacian(s=1.0).apply(basis, x)
    assert torch.allclose(got, -basis.laplacian(x), atol=1e-14)


def test_s_equals_two_reproduces_biharmonic():
    """(-Delta)^2 == Delta^2, cross-checking the symbol against the 4th-order operator."""
    torch.manual_seed(1)
    basis = SinusoidalBasis.random(2, 48, sigma=1.5)
    x = torch.rand(40, 2)
    got = SymbolOperator.fractional_laplacian(s=2.0).apply(basis, x)
    ref = Op.biharmonic(2).apply(basis, x)
    assert torch.allclose(got, ref, atol=1e-12)


def test_first_derivative_symbol():
    """Symbol i*xi_k must reproduce d/dx_k (the imaginary part drives the cos phase)."""
    torch.manual_seed(2)
    basis = SinusoidalBasis.random(2, 32, sigma=2.0)
    x = torch.rand(30, 2)
    op = SymbolOperator(
        lambda W: torch.complex(torch.zeros_like(W[0:1, :]), W[0:1, :])
    )
    assert torch.allclose(op.apply(basis, x), basis.derivative(x, (1, 0)), atol=1e-14)


def test_semigroup_property():
    """Symbols multiply: applying s=0.3 then s=0.7 to the coefficients equals s=1.0."""
    torch.manual_seed(3)
    basis = SinusoidalBasis.random(2, 40, sigma=2.0)
    x = torch.rand(25, 2)
    W = basis.W
    m_a = SymbolOperator.fractional_laplacian(0.3).symbol(W)
    m_b = SymbolOperator.fractional_laplacian(0.7).symbol(W)
    m_1 = SymbolOperator.fractional_laplacian(1.0).symbol(W)
    assert torch.allclose(m_a * m_b, m_1, rtol=1e-12)


# ======================================================================
# The fractional Laplacian against its singular-integral definition
# ======================================================================

def _cnst_1d(s):
    """Normalising constant C(1, s) of the singular-integral definition in 1-D."""
    return 4.0 ** s * math.gamma(0.5 + s) / (math.sqrt(math.pi) * abs(math.gamma(-s)))


def _int_one_minus_cos(s, n=2_000_000, t_split=1.0, t_max=4.0e4):
    """Numerically evaluate I(s) = int_0^inf (1 - cos t) / t^(1+2s) dt.

    Near 0 the integrand behaves like t^(1-2s), which is integrable but has
    unbounded derivatives, so uniform-grid trapezoid converges badly for larger
    s.  Grade the near field with t = v^4 (dt = 4 v^3 dv), which turns the
    integrand into 4 (1 - cos v^4) v^(-1-4a) ~ 2 v^(7-4a) -- bounded with
    bounded derivatives for every s < 1.
    """
    a = 2.0 * s
    v = np.linspace(0.0, t_split ** 0.25, n // 2)[1:]
    t1 = v ** 4
    # 1 - cos t = 2 sin^2(t/2): avoids cancellation for small t.
    f1 = 4.0 * v ** 3 * 2.0 * np.sin(t1 / 2.0) ** 2 / t1 ** (1.0 + a)
    near = np.trapz(f1, v)

    t2 = np.logspace(np.log10(t_split), np.log10(t_max), n // 2)
    f2 = (1.0 - np.cos(t2)) / t2 ** (1.0 + a)
    far = np.trapz(f2, t2)

    # Tail int_{t_max}^inf: the cos part averages out, leaving 1/(a t_max^a).
    tail = 1.0 / (a * t_max ** a)
    return near + far + tail


@pytest.mark.parametrize("s", [0.25, 0.5, 0.75])
def test_symbol_matches_singular_integral_definition(s):
    """The symbol ||xi||^(2s) IS the textbook singular-integral (-Delta)^s.

    For u = sin(w x + b),
        C(1,s) PV int (u(x) - u(y)) / |x-y|^(1+2s) dy
            = 2 C(1,s) sin(w x + b) |w|^(2s) int_0^inf (1 - cos t)/t^(1+2s) dt,
    so the operator is diagonal with eigenvalue |w|^(2s) exactly when
    2 C(1,s) I(s) = 1.  Verifying that identity numerically pins the
    normalisation, i.e. that this is the standard operator and not a rescaling.
    """
    got = 2.0 * _cnst_1d(s) * _int_one_minus_cos(s)
    assert abs(got - 1.0) < 2e-3, f"s={s}: 2*C(1,s)*I(s) = {got}, want 1"


@pytest.mark.parametrize("s", [0.25, 0.5, 0.75])
def test_fractional_eigenvalue_on_a_single_feature(s):
    """Applied to one feature, (-Delta)^s must scale it by exactly |w|^(2s)."""
    w, b = 2.3, 0.7
    W = torch.tensor([[w]])
    bb = torch.tensor([[b]])
    basis = SinusoidalBasis(W, bb, normalize=False)
    x = torch.linspace(-1.0, 1.0, 17).reshape(-1, 1)

    got = SymbolOperator.fractional_laplacian(s).apply(basis, x)
    ref = abs(w) ** (2 * s) * torch.sin(w * x + b)
    assert torch.allclose(got, ref, atol=1e-13)


def test_fractional_order_is_learnable():
    """s may be an nn.Parameter: gradient descent recovers a planted order."""
    torch.manual_seed(7)
    basis = SinusoidalBasis.random(1, 24, sigma=2.0)
    x = torch.linspace(-1, 1, 200).reshape(-1, 1)
    beta = torch.randn(24, 1)

    s_true = 0.65
    target = SymbolOperator.fractional_laplacian(s_true).apply(basis, x) @ beta

    s = torch.nn.Parameter(torch.tensor(0.2))
    opt = torch.optim.Adam([s], lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        pred = SymbolOperator.fractional_laplacian(s).apply(basis, x) @ beta
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        opt.step()
    assert abs(s.item() - s_true) < 1e-3, f"recovered s={s.item():.4f}, want {s_true}"


# ======================================================================
# Convolution
# ======================================================================

def test_gaussian_convolution_matches_quadrature():
    """k * u via the symbol must equal direct numerical convolution.

    Gaussian kernel k(y) = exp(-a y^2), khat(xi) = sqrt(pi/a) exp(-xi^2/4a).
    """
    a = 3.0
    torch.manual_seed(11)
    basis = SinusoidalBasis.random(1, 12, sigma=1.5, normalize=False)
    beta = torch.randn(12, 1)
    x = torch.linspace(-0.5, 0.5, 9).reshape(-1, 1)

    K = SymbolOperator.convolution(
        lambda W: math.sqrt(math.pi / a) * torch.exp(-(W ** 2).sum(0, keepdim=True) / (4 * a))
    )
    got = (K.apply(basis, x) @ beta).squeeze()

    # Direct quadrature: (k*u)(x) = int k(y) u(x-y) dy, Gaussian decays fast.
    y = torch.linspace(-12.0, 12.0, 400001)
    kern = torch.exp(-a * y ** 2)
    ref = []
    for xi in x.squeeze():
        u = (basis.evaluate((xi - y).reshape(-1, 1)) @ beta).squeeze()
        ref.append(torch.trapz(kern * u, y))
    ref = torch.stack(ref)

    rel = (torch.norm(got - ref) / torch.norm(ref)).item()
    assert rel < 1e-8, f"symbol vs quadrature rel-L2 = {rel:.2e}"


def test_convolution_of_even_kernel_is_real():
    """A real even kernel has a real symbol, so no cos component appears."""
    a = 2.0
    torch.manual_seed(13)
    basis = SinusoidalBasis.random(2, 20, sigma=1.0)
    x = torch.rand(15, 2)
    K = SymbolOperator.convolution(
        lambda W: torch.exp(-(W ** 2).sum(0, keepdim=True) / (4 * a))
    )
    A = K.apply(basis, x)
    scale = torch.exp(-(basis.W ** 2).sum(0, keepdim=True) / (4 * a))
    assert torch.allclose(A, scale * basis.evaluate(x), atol=1e-14)


# ======================================================================
# Riesz operators and composition
# ======================================================================

def test_riesz_potential_inverts_fractional_laplacian():
    """(-Delta)^-s composed with (-Delta)^s is the identity away from DC."""
    torch.manual_seed(17)
    basis = SinusoidalBasis.random(2, 32, sigma=2.0)
    x = torch.rand(20, 2)
    s = 0.4
    m_fwd = SymbolOperator.fractional_laplacian(s).symbol(basis.W)
    m_inv = SymbolOperator.riesz_potential(s).symbol(basis.W)
    assert torch.allclose(m_fwd * m_inv, torch.ones_like(m_fwd), rtol=1e-12)


def test_riesz_transform_is_complex_and_bounded():
    """R_k has symbol -i xi_k/||xi||, of unit modulus away from DC."""
    torch.manual_seed(19)
    basis = SinusoidalBasis.random(2, 32, sigma=2.0)
    m = SymbolOperator.riesz_transform(0).symbol(basis.W)
    assert m.is_complex()
    assert (m.abs() <= 1.0 + 1e-12).all()


def test_composition_with_differential_operator():
    """SymbolOperator + DiffOperator assembles into one design matrix."""
    torch.manual_seed(23)
    basis = SinusoidalBasis.random(2, 32, sigma=2.0)
    x = torch.rand(20, 2)
    L = SymbolOperator.fractional_laplacian(0.5) + 3.0 * Op.identity(2)
    got = L.apply(basis, x)
    ref = (
        SymbolOperator.fractional_laplacian(0.5).apply(basis, x)
        + 3.0 * basis.evaluate(x)
    )
    assert torch.allclose(got, ref, atol=1e-14)


def test_rejects_wrongly_shaped_symbol():
    torch.manual_seed(29)
    basis = SinusoidalBasis.random(2, 16, sigma=1.0)
    x = torch.rand(5, 2)
    with pytest.raises(ValueError, match="broadcast"):
        SymbolOperator(lambda W: torch.ones(3, 7)).apply(basis, x)


def test_fractional_system_solves_to_machine_precision():
    """End-to-end: a manufactured (-Delta)^s u = f is solved to ~1e-13 residual.

    Note this asserts the *residual*, not coefficient recovery.  A random-feature
    design matrix is severely rank-deficient (cond ~1e18-1e21 here), so many
    coefficient vectors reproduce the same data and least squares returns the
    minimum-norm one, which is not the planted vector.  That is a property of
    random features, not of the operator: the assembled system is satisfied to
    machine precision, which is what the symbol calculus is responsible for.
    """
    from fastlsq.linalg import solve_lstsq

    torch.manual_seed(31)
    s = 0.6
    n_feat = 300
    basis = SinusoidalBasis.random(1, n_feat, sigma=4.0)
    x = torch.linspace(-1, 1, 600).reshape(-1, 1)

    A = SymbolOperator.fractional_laplacian(s).apply(basis, x)
    f = A @ (torch.randn(n_feat, 1) / n_feat)

    beta = solve_lstsq(A, f, mu=0.0)
    rel = (torch.norm(A @ beta - f) / torch.norm(f)).item()
    assert rel < 1e-10, f"fractional system residual rel-L2 = {rel:.2e}"
