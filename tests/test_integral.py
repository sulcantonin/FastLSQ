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
* higher-order (n-fold ITERATED) integrals agree with n-fold cumulative trapezoid and
  with the Cauchy repeated-integration kernel -- and are *not* the naive difference of
  n-th antiderivatives, which drops the polynomial terms;
* near-DC features integrate to the exact ramp phi*Delta^n/n! at every order, rather
  than being zeroed as the standalone (genuinely divergent) antiderivative is;
* differential and integral terms compose into one (M, N) design matrix, through which
  gradients flow to learnable coefficients;
* a Volterra integro-differential boundary-value problem is solved in one LSQ to <1e-5.

Run with ``pytest`` or directly as a script.
"""

import math

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


def _iterated_trapz(fg, xs, n):
    """n-fold cumulative trapezoid of ``fg`` on grid ``xs`` (all lower limits at xs[0])."""
    cur = fg
    for _ in range(n):
        nxt = torch.zeros_like(cur)
        nxt[1:] = torch.cumulative_trapezoid(cur, xs)
        cur = nxt
    return cur


def test_iterated_volterra_matches_quadrature():
    """order>=2 is the n-fold ITERATED (Volterra) integral, matching an n-fold cumulative
    trapezoid.  Regression guard: the old code returned F_n(hi) - F_n(lo), which drops the
    polynomial terms sum_j F_j(lo)*(x-lo)^(n-j)/(n-j)! and was ~380% off for n=2."""
    torch.manual_seed(3)
    basis = SinusoidalBasis.random(1, 32, sigma=3.0, normalize=False)
    beta = torch.randn(32, 1)

    xg = torch.linspace(0.0, 1.0, 20001).reshape(-1, 1)
    xs = xg.squeeze()
    fg = (basis.evaluate(xg) @ beta).squeeze()

    for n in (1, 2, 3, 4):
        ref = _iterated_trapz(fg, xs, n)
        got = (IntegralOperator.volterra(dim=0, lower=0.0, d=1, order=n).apply(basis, xg)
               @ beta).squeeze()
        assert torch.isfinite(got).all()
        rel = (torch.norm(got - ref) / torch.norm(ref)).item()
        assert rel < 1e-6, f"order {n}: volterra vs {n}x-trapz rel-L2 = {rel:.2e}"


def test_iterated_volterra_matches_cauchy_kernel():
    """Independent reference: the n-fold iterated integral equals the SINGLE integral
    int_lo^x (x-t)^(n-1)/(n-1)! phi(t) dt (Cauchy formula for repeated integration)."""
    torch.manual_seed(11)
    basis = SinusoidalBasis.random(1, 24, sigma=2.5, normalize=False)
    beta = torch.randn(24, 1)

    lo, x_eval = -0.4, 0.85
    tg = torch.linspace(lo, x_eval, 40001)
    fg = (basis.evaluate(tg.reshape(-1, 1)) @ beta).squeeze()

    for n in (2, 3, 5):
        kernel = (x_eval - tg) ** (n - 1) / float(math.factorial(n - 1))
        ref = torch.trapz(kernel * fg, tg)
        got = (IntegralOperator.volterra(dim=0, lower=lo, d=1, order=n)
               .apply(basis, torch.tensor([[x_eval]])) @ beta).squeeze()
        rel = ((got - ref).abs() / ref.abs()).item()
        assert rel < 1e-6, f"order {n}: vs Cauchy kernel rel = {rel:.2e}"


def test_iterated_definite_matches_quadrature():
    """The order-n definite integral over [a, b] equals the n-fold running integral
    evaluated at b, and is identical for every row in 1D."""
    torch.manual_seed(4)
    basis = SinusoidalBasis.random(1, 48, sigma=4.0, normalize=False)
    beta = torch.randn(48, 1)

    a, b = 0.2, 0.9
    xg = torch.linspace(a, b, 20001).reshape(-1, 1)
    fg = (basis.evaluate(xg) @ beta).squeeze()

    for n in (2, 3):
        ref = _iterated_trapz(fg, xg.squeeze(), n)[-1]
        I = IntegralOperator.definite(dim=0, lower=a, upper=b, d=1, order=n)
        rows = I.apply(basis, torch.rand(5, 1)) @ beta  # result independent of x in 1D
        assert torch.allclose(rows, rows[:1].expand_as(rows), atol=1e-10)
        rel = ((rows[0].squeeze() - ref).abs() / ref.abs()).item()
        assert rel < 1e-6, f"order {n}: definite vs {n}x-trapz rel = {rel:.2e}"


def test_iterated_order_one_matches_definite_integral():
    """order=1 of the general iterated form reproduces the sinc closed form exactly, so the
    two code paths in IntegralOperator.apply cannot drift apart."""
    torch.manual_seed(12)
    basis = SinusoidalBasis.random(2, 64, sigma=3.0, normalize=False)
    x = sample_box(200, 2)
    for dim, lo, hi in ((0, -0.4, None), (1, 0.1, 0.75)):
        ref = basis.definite_integral(x, dim, lo, upper=hi)
        got = basis.iterated_integral(x, dim, lo, upper=hi, order=1)
        assert torch.allclose(got, ref, atol=1e-14, rtol=1e-11)


def test_iterated_near_dc_is_exact_not_zeroed():
    """Documented semantics: unlike the DC-guarded standalone antiderivative (whose 1/W
    primitive genuinely diverges), the *limit-bearing* iterated integral of a near-DC
    feature is finite and exact -- phi * Delta^n / n! -- and must NOT be zeroed.  This keeps
    order>=2 consistent with the order=1 sinc path (cf. test_dc_guard_no_blowup)."""
    W = torch.tensor([[0.0, 1e-13, 2.0]])
    b = torch.tensor([[0.7, 0.7, 0.7]])
    basis = SinusoidalBasis(W, b, normalize=False)

    delta = 1.3
    x = torch.tensor([[delta]])
    for n in (1, 2, 3, 4):
        got = basis.iterated_integral(x, 0, 0.0, order=n)[0]
        assert torch.isfinite(got).all()
        exact_dc = np.sin(0.7) * delta ** n / float(math.factorial(n))
        # both the exactly-DC and the near-DC column hit the ramp value, not zero
        assert abs(got[0].item() - exact_dc) < 1e-13, f"order {n}: DC col = {got[0].item()}"
        assert abs(got[1].item() - exact_dc) < 1e-13
        assert got[0].abs().item() > 1e-3  # explicitly: not zeroed


def test_iterated_differs_from_antiderivative_difference():
    """Pins the actual bug: for n=2 the correct value is F2(x)-F2(lo)-F1(lo)*(x-lo), so it
    must differ from the naive difference F2(x)-F2(lo) by exactly that ramp term."""
    torch.manual_seed(13)
    basis = SinusoidalBasis.random(1, 16, sigma=2.0, normalize=False)
    x = torch.linspace(0.1, 1.0, 9).reshape(-1, 1)
    lo = 0.0

    naive = basis.derivative(x, (-2,)) - basis.derivative(torch.full_like(x, lo), (-2,))
    got = basis.iterated_integral(x, 0, lo, order=2)
    ramp = basis.derivative(torch.full_like(x, lo), (-1,)) * (x - lo)

    assert torch.allclose(got, naive - ramp, atol=1e-10, rtol=1e-8)
    assert not torch.allclose(got, naive, atol=1e-6)  # the dropped term is not negligible


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
    test_iterated_volterra_matches_quadrature()
    test_iterated_volterra_matches_cauchy_kernel()
    test_iterated_definite_matches_quadrature()
    test_iterated_order_one_matches_definite_integral()
    test_iterated_near_dc_is_exact_not_zeroed()
    test_iterated_differs_from_antiderivative_difference()
    test_integro_differential_compose()
    test_learnable_coeff_grad_flows()
    test_volterra_ivp_forward_solve()
    print("ALL INTEGRAL TESTS PASSED")
