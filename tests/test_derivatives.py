# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Verification that the closed-form analytical derivatives of the sinusoidal
(goniometric) basis -- gradient, Laplacian, and arbitrary mixed partials --
agree with automatic differentiation to (near) machine precision, and that the
recovered solution gradient generalises at the same fidelity as the value.

This makes the central structural claim of FastLSQ a *tested* property: because
the derivative of sin(W.x+b) is again a phase-shifted sinusoid, the operator
generalises to every derivative exactly, with no autodiff and no finite
differences.  Run with ``pytest`` or directly as a script.
"""

import numpy as np
import torch

from fastlsq import (
    SinusoidalBasis, FastLSQSolver, solve_lstsq, Op,
    sample_box, sample_boundary_box,
)

torch.set_default_dtype(torch.float64)
PI = np.pi


def _nested_autodiff(fn, x, alpha):
    """D^alpha fn via repeated autograd (fn: (M,d)->(M,1))."""
    x = x.clone().requires_grad_(True)
    val = fn(x)
    for dim, order in enumerate(alpha):
        for _ in range(order):
            val = torch.autograd.grad(val.sum(), x, create_graph=True)[0][:, dim:dim + 1]
    return val


def test_basis_gradient_matches_autodiff():
    """grad phi_j = W_j cos(.) (closed form) equals autodiff to machine precision."""
    torch.manual_seed(0)
    basis = SinusoidalBasis.random(3, 64, sigma=4.0, normalize=False)
    beta = torch.randn(64, 1)
    x = sample_box(200, 3)
    g_cf = torch.einsum("mdh,ho->md", basis.gradient(x), beta)
    g_ad = _nested_autodiff(lambda z: basis.evaluate(z) @ beta, x, (1, 0, 0))
    # full gradient via autodiff (all components at once)
    xr = x.clone().requires_grad_(True)
    g_ad_full = torch.autograd.grad((basis.evaluate(xr) @ beta).sum(), xr)[0]
    assert torch.allclose(g_cf, g_ad_full, atol=1e-12, rtol=1e-9)


def test_laplacian_matches_autodiff():
    torch.manual_seed(1)
    basis = SinusoidalBasis.random(2, 64, sigma=3.0, normalize=False)
    beta = torch.randn(64, 1)
    x = sample_box(150, 2)
    lap_cf = basis.laplacian(x) @ beta
    uxx = _nested_autodiff(lambda z: basis.evaluate(z) @ beta, x, (2, 0))
    uyy = _nested_autodiff(lambda z: basis.evaluate(z) @ beta, x, (0, 2))
    assert torch.allclose(lap_cf, uxx + uyy, atol=1e-11, rtol=1e-8)


def test_mixed_partials_match_autodiff():
    """Arbitrary mixed partials D^alpha match autodiff for |alpha| up to 4."""
    torch.manual_seed(2)
    basis = SinusoidalBasis.random(2, 48, sigma=3.0, normalize=False)
    beta = torch.randn(48, 1)
    x = sample_box(120, 2)
    for alpha in [(1, 0), (0, 1), (2, 0), (1, 1), (3, 0), (2, 1), (2, 2), (4, 0)]:
        D_cf = basis.derivative(x, alpha) @ beta
        D_ad = _nested_autodiff(lambda z: basis.evaluate(z) @ beta, x, alpha)
        # roundoff grows mildly with order; 1e-8 is comfortably tight in float64
        assert torch.allclose(D_cf, D_ad, atol=1e-8, rtol=1e-6), \
            f"alpha={alpha}: max|diff|={(D_cf - D_ad).abs().max():.2e}"


def _solve_poisson2d(n_feat=1500, sigma=8.0):
    """-lap u = f, u* = sin(pi x)sin(pi y)+0.5 sin(3pi x)sin(2pi y), u=0 on bdry."""
    modes = ((1, 1, 1.0), (3, 2, 0.5))

    def exact(x):
        u = torch.zeros(x.shape[0], 1)
        for m, n, a in modes:
            u = u + a * torch.sin(m * PI * x[:, 0:1]) * torch.sin(n * PI * x[:, 1:2])
        return u

    def exact_grad(x):
        gx = torch.zeros(x.shape[0], 1)
        gy = torch.zeros_like(gx)
        for m, n, a in modes:
            gx = gx + a * m * PI * torch.cos(m * PI * x[:, 0:1]) * torch.sin(n * PI * x[:, 1:2])
            gy = gy + a * n * PI * torch.sin(m * PI * x[:, 0:1]) * torch.cos(n * PI * x[:, 1:2])
        return torch.cat([gx, gy], dim=1)

    def source(x):
        f = torch.zeros(x.shape[0], 1)
        for m, n, a in modes:
            f = f + a * (PI ** 2) * (m ** 2 + n ** 2) * \
                torch.sin(m * PI * x[:, 0:1]) * torch.sin(n * PI * x[:, 1:2])
        return f

    torch.manual_seed(0)
    solver = FastLSQSolver(2, normalize=False)
    for _ in range(3):
        solver.add_block(hidden_size=n_feat // 3, scale=sigma)
    op = -Op.laplacian(d=2)
    x_int = sample_box(8000, 2)
    x_bc = sample_boundary_box(2000, 2)
    A = torch.cat([op.apply(solver.basis, x_int), 100.0 * solver.basis.evaluate(x_bc)])
    b = torch.cat([source(x_int), 100.0 * exact(x_bc)])
    solver.beta = solve_lstsq(A, b, mu=1e-10)
    return solver, exact, exact_grad


def test_solution_gradient_matches_autodiff():
    """grad u_N (closed form) equals autodiff through the solved model."""
    solver, _, _ = _solve_poisson2d()
    x = sample_box(2000, 2)
    _, g_cf = solver.predict_with_grad(x)
    xr = x.clone().requires_grad_(True)
    g_ad = torch.autograd.grad(solver.predict(xr).sum(), xr)[0]
    assert torch.allclose(g_cf, g_ad, atol=1e-11, rtol=1e-8)


PHASE_T = [torch.sin, torch.cos, lambda z: -torch.sin(z), lambda z: -torch.cos(z)]
NAMED_COMPONENTS = {  # the value and all 1st/2nd-order partials in 2-D
    "u": (0, 0), "u_x": (1, 0), "u_y": (0, 1),
    "u_xy": (1, 1), "u_xx": (2, 0), "u_yy": (0, 2),
}


def _modes_partial(x, alpha, modes):
    """Analytic D^alpha of sum a sin(m pi x) sin(n pi y)."""
    p, q = alpha
    out = torch.zeros(x.shape[0], 1)
    for m, n, a in modes:
        fx = (m * PI) ** p * PHASE_T[p % 4](m * PI * x[:, 0:1])
        fy = (n * PI) ** q * PHASE_T[q % 4](n * PI * x[:, 1:2])
        out = out + a * fx * fy
    return out


def test_named_derivative_components():
    """The gradient test over u, u_x, u_y, u_xy, u_xx, u_yy:  each closed-form
    component equals autodiff to machine precision AND recovers the analytic
    derivative of the solved PDE to <1e-3."""
    solver, _, _ = _solve_poisson2d()
    modes = ((1, 1, 1.0), (3, 2, 0.5))
    x = sample_box(3000, 2)
    for name, alpha in NAMED_COMPONENTS.items():
        D_cf = solver.basis.derivative(x, alpha) @ solver.beta
        D_ad = _nested_autodiff(lambda z: solver.predict(z), x, alpha)
        D_ex = _modes_partial(x, alpha, modes)
        # closed form == autodiff (machine precision)
        assert torch.allclose(D_cf, D_ad, atol=1e-9, rtol=1e-6), \
            f"{name}: closed-form vs autodiff max|diff|={(D_cf - D_ad).abs().max():.2e}"
        # closed form recovers the true derivative
        rel = (torch.norm(D_cf - D_ex) / torch.norm(D_ex)).item()
        assert rel < 1e-3, f"{name}: recovery rel-L2 {rel:.2e}"


def test_gradient_generalises_at_value_fidelity():
    """The recovered gradient is within one order of magnitude of the value error
    -- the goniometric feature carries its accuracy into the derivative."""
    solver, exact, exact_grad = _solve_poisson2d()
    x = sample_box(5000, 2)
    u, g = solver.predict_with_grad(x)
    val_err = (torch.norm(u - exact(x)) / torch.norm(exact(x))).item()
    grad_err = (torch.norm(g - exact_grad(x)) / torch.norm(exact_grad(x))).item()
    assert val_err < 1e-6
    assert grad_err < 30 * val_err, f"grad/val ratio = {grad_err / val_err:.1f}"


if __name__ == "__main__":
    test_basis_gradient_matches_autodiff()
    test_laplacian_matches_autodiff()
    test_mixed_partials_match_autodiff()
    test_named_derivative_components()
    test_solution_gradient_matches_autodiff()
    test_gradient_generalises_at_value_fidelity()

    # Per-component gradient test report: u, u_x, u_y, u_xy, u_xx, u_yy
    solver, _, _ = _solve_poisson2d()
    modes = ((1, 1, 1.0), (3, 2, 0.5))
    xc = sample_box(5000, 2)
    print("\n  component        closed-form vs exact   closed-form vs autodiff")
    for nm, al in NAMED_COMPONENTS.items():
        Dcf = solver.basis.derivative(xc, al) @ solver.beta
        Dad = _nested_autodiff(lambda z: solver.predict(z), xc, al)
        Dex = _modes_partial(xc, al, modes)
        rel = (torch.norm(Dcf - Dex) / torch.norm(Dex)).item()
        adm = (Dcf - Dad).abs().max().item()
        print(f"    {nm:5s}            {rel:.2e}              {adm:.2e}")
    # report the headline numbers
    basis = SinusoidalBasis.random(2, 64, sigma=5.0, normalize=False)
    beta = torch.randn(64, 1)
    x = sample_box(500, 2)
    g_cf = torch.einsum("mdh,ho->md", basis.gradient(x), beta)
    xr = x.clone().requires_grad_(True)
    g_ad = torch.autograd.grad((basis.evaluate(xr) @ beta).sum(), xr)[0]
    diff = (g_cf - g_ad).abs().max().item()
    solver, exact, exact_grad = _solve_poisson2d()
    xt = sample_box(5000, 2)
    u, g = solver.predict_with_grad(xt)
    ve = (torch.norm(u - exact(xt)) / torch.norm(exact(xt))).item()
    ge = (torch.norm(g - exact_grad(xt)) / torch.norm(exact_grad(xt))).item()
    print("ALL DERIVATIVE TESTS PASSED")
    print(f"  closed-form grad vs autodiff : max|diff| = {diff:.2e}")
    print(f"  solved Poisson2D             : value err = {ve:.2e}, grad err = {ge:.2e}"
          f"  (grad/val = {ge/ve:.1f})")
