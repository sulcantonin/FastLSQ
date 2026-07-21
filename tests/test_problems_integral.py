# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Accuracy regressions for the integral-equation Problem classes.

These run through `solve_linear` -- the same harness as the PDE problems -- so a
regression in the operators, the solver or the problem definitions shows up as a
number, not as a silently worse fit.  Thresholds are set roughly an order of
magnitude looser than the observed error, so they catch real regressions without
being brittle to reseeding.

Every reference is analytic (degenerate-kernel theory for the Fredholm cases,
the equivalent ODE for the Volterra ones), so these validate the whole stack
against mathematics rather than against a stored golden value.
"""

import numpy as np
import pytest
import torch

from fastlsq.api import solve_linear
from fastlsq.problems import (
    FredholmProductKernel,
    FredholmRank2Kernel,
    IntegroDifferentialODE,
    VolterraSecondKind,
)

torch.set_default_dtype(torch.float64)

# Modest, fixed configuration: these are regressions, not benchmarks.
CFG = dict(
    scale=5.0, auto_scale=False, n_blocks=1, hidden_size=200,
    n_pde=1200, n_test=1200, verbose=False,
)


def _run(problem, **over):
    cfg = {**CFG, **over}
    torch.manual_seed(0)
    return solve_linear(problem, **cfg)["metrics"]["val_err"]


# ======================================================================
# The problems solve to the accuracy their closed forms allow
# ======================================================================

@pytest.mark.parametrize("lam", [0.5, 2.0])
def test_fredholm_product_kernel_accuracy(lam):
    err = _run(FredholmProductKernel(lam=lam))
    assert err < 1e-8, f"lam={lam}: val_err = {err:.2e}"


def test_fredholm_rank2_accuracy():
    err = _run(FredholmRank2Kernel(lam=0.4))
    assert err < 1e-8, f"val_err = {err:.2e}"


def test_volterra_second_kind_accuracy():
    err = _run(VolterraSecondKind(lam=1.5, omega=3.0))
    assert err < 1e-6, f"val_err = {err:.2e}"


def test_integro_differential_ode_accuracy():
    err = _run(IntegroDifferentialODE(lam=4.0, u0=0.3))
    assert err < 1e-6, f"val_err = {err:.2e}"


# ======================================================================
# The problem definitions are self-consistent
# ======================================================================

def test_fredholm_exact_solution_satisfies_its_own_equation():
    """Verify `exact` really solves u − λ∫Ku = f, by independent quadrature.

    Guards against the closed form and the assembled operator being wrong in the
    same direction, which an end-to-end solve alone would not catch.
    """
    P = FredholmProductKernel(lam=0.5, freq=2.0)
    x = torch.linspace(0, 1, 41).reshape(-1, 1)
    y = torch.linspace(0, 1, 200001).reshape(-1, 1)
    uy = P.exact(y).squeeze()
    # ∫₀¹ x y u(y) dy = x · ∫₀¹ y u(y) dy
    inner = torch.trapz(y.squeeze() * uy, y.squeeze())
    lhs = P.exact(x) - P.lam * x * inner
    assert torch.allclose(lhs, P.source(x), atol=1e-9)


def test_fredholm_rank2_exact_solution_satisfies_its_own_equation():
    """Same independent check at rank 2, where the reduced system is 2x2.

    The rank-2 closed form needs six hand-derived integrals (S is 2x2, b is 2x1);
    checking the resulting u against the original equation by quadrature is what
    catches an algebra slip in any of them, which comparing against the same
    formulas cannot.
    """
    P = FredholmRank2Kernel(lam=0.4, freq=2.0)
    x = torch.linspace(0, 1, 41).reshape(-1, 1)
    y = torch.linspace(0, 1, 200001).reshape(-1, 1)
    uy = P.exact(y).squeeze()
    ys = y.squeeze()

    # ∫₀¹ K(x,y) u(y) dy with K = x·y + sin x·cos y
    i0 = torch.trapz(ys * uy, ys)                    # ∫ y u
    i1 = torch.trapz(torch.cos(ys) * uy, ys)         # ∫ cos y · u
    Ku = x * i0 + torch.sin(x) * i1
    lhs = P.exact(x) - P.lam * Ku
    assert torch.allclose(lhs, P.source(x), atol=1e-9)


def test_exact_grad_matches_finite_differences():
    """`exact_grad` is required by the harness -- verify it against `exact`."""
    for P in (FredholmProductKernel(0.5), FredholmRank2Kernel(0.4),
              VolterraSecondKind(1.5), IntegroDifferentialODE(4.0)):
        x = torch.linspace(0.1, 0.9, 25).reshape(-1, 1)
        h = 1e-6
        fd = (P.exact(x + h) - P.exact(x - h)) / (2 * h)
        rel = (torch.norm(P.exact_grad(x) - fd) / torch.norm(fd)).item()
        assert rel < 1e-7, f"{P.name}: exact_grad vs FD rel = {rel:.2e}"


def test_volterra_exact_solution_satisfies_its_own_equation():
    """u(x) − λ∫₀ˣ u = f(x), checked pointwise against cumulative quadrature."""
    P = VolterraSecondKind(lam=1.5, omega=3.0)
    t = torch.linspace(0, 1, 200001)
    u = P.exact(t.reshape(-1, 1)).squeeze()
    run = torch.zeros_like(u)
    run[1:] = torch.cumulative_trapezoid(u, t)
    lhs = u - P.lam * run
    rhs = P.source(t.reshape(-1, 1)).squeeze()
    rel = (torch.norm(lhs - rhs) / torch.norm(rhs)).item()
    assert rel < 1e-8, f"rel = {rel:.2e}"


def test_integro_differential_exact_satisfies_its_own_equation():
    """u' + λ∫₀ˣ u = f, with u' by finite difference and the integral by quadrature."""
    P = IntegroDifferentialODE(lam=4.0, u0=0.3)
    t = torch.linspace(0, 1, 200001)
    u = P.exact(t.reshape(-1, 1)).squeeze()
    run = torch.zeros_like(u)
    run[1:] = torch.cumulative_trapezoid(u, t)
    du = torch.gradient(u, spacing=(t,))[0]
    lhs = du + P.lam * run
    rhs = P.source(t.reshape(-1, 1)).squeeze()
    # Interior only: one-sided differences at the ends are first-order.
    s = slice(10, -10)
    rel = (torch.norm(lhs[s] - rhs[s]) / torch.norm(rhs[s])).item()
    assert rel < 1e-7, f"rel = {rel:.2e}"


def test_characteristic_value_is_reported():
    """The problem exposes the λ at which it becomes singular (analytically 3)."""
    from fastlsq.basis import SinusoidalBasis

    torch.manual_seed(0)
    P = FredholmProductKernel(lam=0.5)
    basis = SinusoidalBasis.random(1, 80, sigma=4.0)
    lams = P.characteristic_values(basis)
    assert abs(lams.real.item() - 3.0) < 1e-9


# ======================================================================
# Harness integration
# ======================================================================

def test_problems_expose_the_solve_linear_contract():
    """Each problem carries the members `solve_linear` consults."""
    for P in (FredholmProductKernel(), FredholmRank2Kernel(),
              VolterraSecondKind(), IntegroDifferentialODE()):
        assert isinstance(P.name, str) and P.dim == 1
        x_pde, bcs, f_pde = P.get_train_data(n_pde=32, n_bc=1)
        assert x_pde.shape == (32, 1) and f_pde.shape == (32, 1)
        assert isinstance(bcs, list)
        assert P.exact(P.get_test_points(16)).shape == (16, 1)


def test_second_kind_problems_need_no_boundary_rows():
    """Fredholm/Volterra second-kind problems return an empty bcs list."""
    for P in (FredholmProductKernel(), FredholmRank2Kernel(), VolterraSecondKind()):
        _, bcs, _ = P.get_train_data(n_pde=16, n_bc=4)
        assert bcs == [], f"{P.name} should need no boundary rows"
    # ...but the integro-differential ODE genuinely does need one.
    _, bcs, _ = IntegroDifferentialODE().get_train_data(n_pde=16, n_bc=1)
    assert len(bcs) == 1
