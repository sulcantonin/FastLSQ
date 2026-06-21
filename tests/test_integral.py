# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Verification that the closed-form *integral* calculus of the sinusoidal basis is
exact -- the mirror image of ``test_derivatives.py``.

Because the derivative of sin(W.x+b) is a phase-shifted sinusoid, so is its
antiderivative: integration is differentiation of *negative* order, with a
reciprocal prefactor.  These tests make that a *tested* property:

* indefinite antiderivative inverts the derivative (d/dx ∫ = id) to machine precision;
* mixed integro-differential multi-indices match the analytic closed form and autodiff;
* the DC guard keeps near-zero-frequency antiderivatives finite;
* definite / running (Volterra) integrals agree with high-resolution quadrature;
* differential and integral terms compose into one (M, N) design matrix, through which
  gradients flow to learnable coefficients;
* a Volterra integro-differential boundary-value problem is solved in one LSQ to <1e-5.

Run with ``pytest`` or directly as a script.
"""

import numpy as np
import torch

from fastlsq import (
    SinusoidalBasis, Op, IntegralOperator, IntegroDifferentialOperator,
    solve_lstsq, sample_box,
)

torch.set_default_dtype(torch.float64)
PI = np.pi


def _nested_autodiff(fn, x, alpha):
    """D^alpha fn via repeated autograd (fn: (M,d)->(M,1)); alpha entries >= 0."""
    x = x.clone().requires_grad_(True)
    val = fn(x)
    for dim, order in enumerate(alpha):
        for _ in range(order):
            val = torch.autograd.grad(val.sum(), x, create_graph=True)[0][:, dim:dim + 1]
    return val


def test_antiderivative_inverts_derivative():
    """d^n/dx^n of the n-fold antiderivative recovers the basis value to machine precision,
    and the closed form matches the hand-derived primitive of sin."""
    torch.manual_seed(0)
    basis = SinusoidalBasis.random(1, 64, sigma=3.0, normalize=False)
    beta = torch.randn(64, 1)
    x = sample_box(200, 1)

    # closed-form single antiderivative == -cos(Z)/W * inv_norm
    Z = x @ basis.W + basis.b
    hand = (-torch.cos(Z) / basis.W) * basis._inv_norm
    assert torch.allclose(basis.derivative(x, (-1,)), hand, atol=1e-12, rtol=1e-9)

    # D^n applied to the (-n)-order antiderivative == value, checked by autodiff
    for n in (1, 2, 3):
        anti = lambda z, n=n: basis.derivative(z, (-n,)) @ beta
        rt = _nested_autodiff(anti, x, (n,))
        val = basis.evaluate(x) @ beta
        assert torch.allclose(rt, val, atol=1e-8, rtol=1e-6), \
            f"order {n}: max|diff|={(rt - val).abs().max():.2e}"


def test_mixed_integro_differential_index():
    """A mixed sign multi-index (integrate x0, differentiate x1) matches the analytic
    closed form (W1/W0) sin(Z) and autodiff (∂/∂x1 of the x0-antiderivative)."""
    torch.manual_seed(2)
    basis = SinusoidalBasis.random(2, 48, sigma=3.0, normalize=False)
    beta = torch.randn(48, 1)
    x = sample_box(120, 2)

    Z = x @ basis.W + basis.b
    hand = ((basis.W[1:2, :] / basis.W[0:1, :]) * torch.sin(Z)) * basis._inv_norm
    assert torch.allclose(basis.derivative(x, (-1, 1)), hand, atol=1e-11, rtol=1e-8)

    got = basis.derivative(x, (-1, 1)) @ beta
    ad = _nested_autodiff(lambda z: basis.derivative(z, (-1, 0)) @ beta, x, (0, 1))
    assert torch.allclose(got, ad, atol=1e-9, rtol=1e-6)


def test_dc_guard_no_blowup():
    """A near-DC feature has no sinusoidal antiderivative; its column is zeroed, not inf."""
    torch.manual_seed(1)
    basis = SinusoidalBasis.random(1, 16, sigma=2.0, normalize=False)
    basis.W[0, 3] = 1e-14  # near-zero frequency
    anti = basis.derivative(sample_box(20, 1), (-1,))
    assert torch.isfinite(anti).all()
    assert torch.all(anti[:, 3] == 0)


def test_volterra_matches_quadrature():
    """The running integral ∫_0^x phi matches a fine-grid cumulative trapezoid, including
    a near-DC feature (exercises the numerically stable sinc path, which must NOT zero it)."""
    torch.manual_seed(3)
    basis = SinusoidalBasis.random(1, 32, sigma=3.0, normalize=False)
    basis.W[0, 5] = 1e-13  # near-DC: its running integral ~ const*(x-0), genuinely finite
    beta = torch.randn(32, 1)

    xg = torch.linspace(0.0, 1.0, 4001).reshape(-1, 1)
    fg = (basis.evaluate(xg) @ beta).squeeze()
    cum = torch.zeros_like(fg)
    cum[1:] = torch.cumulative_trapezoid(fg, xg.squeeze())

    V = IntegralOperator.volterra(dim=0, lower=0.0, d=1)
    v_cf = (V.apply(basis, xg) @ beta).squeeze()
    assert torch.isfinite(v_cf).all()
    rel = (torch.norm(v_cf - cum) / torch.norm(cum)).item()
    assert rel < 1e-3, f"volterra vs trapz rel-L2 = {rel:.2e}"


def test_definite_matches_quadrature():
    """A definite integral over [a, b] matches torch.trapz, and is the same for every row."""
    torch.manual_seed(4)
    basis = SinusoidalBasis.random(1, 48, sigma=4.0, normalize=False)
    beta = torch.randn(48, 1)

    a, b = 0.2, 0.9
    xg = torch.linspace(a, b, 6001).reshape(-1, 1)
    exact = torch.trapz((basis.evaluate(xg) @ beta).squeeze(), xg.squeeze())

    I = IntegralOperator.definite(dim=0, lower=a, upper=b, d=1)
    rows = I.apply(basis, torch.rand(5, 1)) @ beta  # arbitrary x: result independent of it in 1D
    assert torch.allclose(rows, rows[:1].expand_as(rows), atol=1e-10)
    assert (rows[0].squeeze() - exact).abs() < 1e-6


def test_integro_differential_compose():
    """Differential + integral terms compose into one (M, N) matrix equal to the sum of parts."""
    torch.manual_seed(5)
    basis = SinusoidalBasis.random(1, 40, sigma=3.0, normalize=False)
    x = sample_box(64, 1)

    D = Op.partial(0, 1, d=1)
    V = IntegralOperator.volterra(dim=0, lower=0.0, d=1)
    L = D + V
    assert isinstance(L, IntegroDifferentialOperator)

    A = L.apply(basis, x)
    A_parts = D.apply(basis, x) + V.apply(basis, x)
    assert A.shape == (64, basis.n_features)
    assert torch.allclose(A, A_parts, atol=1e-12)

    # scaling and subtraction also stay consistent
    L2 = D - 2.0 * V
    assert torch.allclose(L2.apply(basis, x), D.apply(basis, x) - 2.0 * V.apply(basis, x),
                          atol=1e-12)


def test_learnable_coeff_grad_flows():
    """A learnable nn.Parameter coefficient on the integral term receives gradients
    through the composed operator (mirrors test_op_learnable_parameter)."""
    torch.manual_seed(6)
    lam = torch.nn.Parameter(torch.tensor(0.7))
    L = Op.partial(0, 1, d=1) + lam * IntegralOperator.volterra(dim=0, lower=0.0, d=1)
    basis = SinusoidalBasis.random(1, 50, sigma=3.0, normalize=False)
    x = torch.rand(20, 1)
    L.apply(basis, x).sum().backward()
    assert lam.grad is not None
    assert torch.isfinite(lam.grad).all()


def test_volterra_ivp_forward_solve():
    """One-shot solve of  u'(x) + ∫_0^x u(s)ds = f(x),  u(0)=0  recovers u*(x)=sin(w x)."""
    torch.manual_seed(0)
    w, L = 3.0, 1.0
    u_star = lambda x: torch.sin(w * x)
    f_rhs = lambda x: w * torch.cos(w * x) + (1.0 - torch.cos(w * x)) / w

    basis = SinusoidalBasis.random(1, 800, sigma=5.0, normalize=False)
    x_col = sample_box(4000, 1) * L
    x_ic = torch.zeros(1, 1)

    op = Op.partial(0, 1, d=1) + IntegralOperator.volterra(dim=0, lower=0.0, d=1)
    W_IC = 100.0
    A = torch.cat([op.apply(basis, x_col), W_IC * basis.evaluate(x_ic)])
    b = torch.cat([f_rhs(x_col), W_IC * u_star(x_ic)])
    beta = solve_lstsq(A, b, mu=1e-10)

    xt = sample_box(3000, 1) * L
    err = (torch.norm(basis.evaluate(xt) @ beta - u_star(xt)) / torch.norm(u_star(xt))).item()
    assert err < 1e-5, f"forward integro-diff rel-L2 = {err:.2e}"


if __name__ == "__main__":
    test_antiderivative_inverts_derivative()
    test_mixed_integro_differential_index()
    test_dc_guard_no_blowup()
    test_volterra_matches_quadrature()
    test_definite_matches_quadrature()
    test_integro_differential_compose()
    test_learnable_coeff_grad_flows()
    test_volterra_ivp_forward_solve()
    print("ALL INTEGRAL TESTS PASSED")
