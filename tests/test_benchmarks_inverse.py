# Copyright (c) 2026 Antonin Sulc -- MIT.
"""Smoke tests for the benchmark PDE equations and the inverse-problem workflows.

Exercises the forward benchmark problems through the public ``solve_linear`` /
``solve_nonlinear`` API and two inverse pipelines -- parameter recovery via an
outer optimiser, and SINDy-style PDE discovery via analytical derivatives -- so
the v0.2.3 QR / N-scaled-collocation solver path is covered end-to-end, not just
on the single Poisson problem in ``test_basic``.

Scales are fixed (not auto-selected) and the RNG is seeded so the smoke test is
fast and deterministic; tolerances carry ~10x headroom over measured errors.

``ElasticWave2D`` -- a coupled 2-output vector problem -- exercises the
block-stacked vector path (``n_outputs = 2``, ``block_concat`` assembly,
``unpack_beta`` -> ``(N, 2)`` beta); it carries a per-case ``n_blocks`` bump
since the coupled solve needs more features than the scalar benchmarks.

``Wave2D_MS`` -- a long-time anisotropic wave -- likewise bumps ``n_blocks``;
its ``t_max`` was reduced from 100 to 4 so the normalised-time solution spans
~3.5 temporal cycles rather than ~87.  The PDE's second time-derivative
amplifies the random-feature representation error by ``Omega**2``, so the
one-shot collocation only resolves a few cycles (see the class docstring) --
the old t_max=100 gave rel-err 1.0 in every configuration.
"""
import numpy as np
import pytest
import torch

from fastlsq import (
    solve_linear, solve_nonlinear, solve_lstsq, Op, SinusoidalBasis,
    sample_box, sample_boundary_box,
)
from fastlsq.problems import linear as L
from fastlsq.problems import nonlinear as NL


# (class, fixed scale, val_err tolerance, solver-config overrides)
LINEAR_CASES = [
    (L.PoissonND,    0.5, 5e-3, {}),
    (L.HeatND,       0.5, 1e-1, {}),
    (L.Wave1D,      15.0, 5e-3, {}),
    (L.Helmholtz2D, 10.0, 1e-5, {}),
    (L.Maxwell2D_TM, 2.0, 5e-3, {}),
    # Long-time anisotropic wave: temporal-matched bandwidth + more features
    # (t_max reduced 100 -> 4 so the collocation can resolve the ~3.5 cycles).
    (L.Wave2D_MS,    3.0, 1e-2, {"n_blocks": 3}),
    # Coupled 2-output vector problem: needs more features than the scalars.
    (L.ElasticWave2D, 6.0, 1e-1, {"n_blocks": 3}),
]

NONLINEAR_CASES = [
    (NL.NLPoisson2D,     8.0, 1e-4),
    (NL.Bratu2D,        15.0, 1e-4),
    (NL.SteadyBurgers1D,10.0, 1e-4),
    (NL.NLHelmholtz2D,   5.0, 1e-4),
    (NL.AllenCahn1D,    15.0, 2e-1),
]


@pytest.mark.parametrize(
    "cls,scale,tol,solver_kw", LINEAR_CASES, ids=[c[0].__name__ for c in LINEAR_CASES]
)
def test_linear_benchmark_solves(cls, scale, tol, solver_kw):
    """Each linear benchmark equation solves end-to-end via the public API."""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    cfg = dict(n_blocks=2, hidden_size=300, n_test=1500,
               auto_scale=False, verbose=False)
    cfg.update(solver_kw)
    r = solve_linear(cls(), scale=scale, **cfg)
    ve = r["metrics"]["val_err"]
    assert np.isfinite(ve), f"{cls.__name__}: non-finite val_err"
    assert ve < tol, f"{cls.__name__}: val_err={ve:.2e} exceeds tol {tol:.0e}"


@pytest.mark.parametrize(
    "cls,scale,tol", NONLINEAR_CASES, ids=[c[0].__name__ for c in NONLINEAR_CASES]
)
def test_nonlinear_benchmark_solves(cls, scale, tol):
    """Each nonlinear benchmark equation converges via Newton + the public API."""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    r = solve_nonlinear(cls(), scale=scale, n_blocks=2, hidden_size=300,
                        n_test=1500, max_iter=15, auto_scale=False, verbose=False)
    ve = r["metrics"]["val_err"]
    assert r["n_iters"] > 0, f"{cls.__name__}: no Newton iterations ran"
    assert np.isfinite(ve), f"{cls.__name__}: non-finite val_err"
    assert ve < tol, f"{cls.__name__}: val_err={ve:.2e} exceeds tol {tol:.0e}"


def test_inverse_source_position():
    """Recover a Gaussian source position from sensor data (forward solve + L-BFGS)."""
    opt = pytest.importorskip("scipy.optimize")
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    pde_op = -Op.laplacian(d=2)
    basis = SinusoidalBasis.random(input_dim=2, n_features=700, sigma=5.0,
                                   normalize=True)
    x_pde = sample_box(3000, 2)
    x_bc = sample_boundary_box(400, 2)
    n_bc = x_bc.shape[0]
    cache = basis.cache(x_pde)
    A = torch.cat([pde_op.apply(basis, x_pde, cache=cache),
                   100.0 * basis.evaluate(x_bc)])
    x_sens = torch.tensor([[0.3, 0.3], [0.7, 0.7], [0.3, 0.7], [0.7, 0.3]])

    def forward(xs, ys):
        b = torch.exp(-((x_pde[:, 0] - xs) ** 2
                        + (x_pde[:, 1] - ys) ** 2) / 0.1).unsqueeze(1)
        b = torch.cat([b, torch.zeros(n_bc, 1, dtype=b.dtype)])
        beta = solve_lstsq(A, b)
        return (basis.evaluate(x_sens) @ beta).detach().cpu().numpy().ravel()

    true = np.array([0.4, 0.6])
    rng = np.random.default_rng(0)
    u_obs = forward(*true) + 0.005 * rng.standard_normal(4)

    res = opt.minimize(
        lambda p: float(np.sum((forward(float(p[0]), float(p[1])) - u_obs) ** 2)),
        x0=[0.5, 0.5], method="L-BFGS-B", bounds=[(0.1, 0.9)] * 2,
    )
    assert np.linalg.norm(res.x - true) < 0.06, f"recovered {res.x} vs {true}"


def test_pde_discovery_recovers_governing_equation():
    """SINDy-style discovery via analytical derivatives recovers u_xx = a*u + b*u_x.

    Synthetic damped oscillator u = exp(-x/2) sin(2x) -> u_xx = -4.25 u - 1.0 u_x.
    The dominant restoring term is recovered tightly; the damping term is harder
    from 2% noise, so it is only bounded in sign/magnitude.
    """
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)

    M = 500
    x = torch.linspace(0, 2 * np.pi, M).reshape(-1, 1)
    u_true = torch.exp(-0.5 * x) * torch.sin(2 * x)
    u_noisy = u_true + 0.02 * torch.randn_like(u_true)

    basis = SinusoidalBasis.random(input_dim=1, n_features=400, sigma=4.0,
                                   normalize=True)
    beta = solve_lstsq(basis.evaluate(x), u_noisy, mu=1e-3)
    cache = basis.cache(x)
    u = basis.evaluate(x, cache=cache) @ beta
    u_x = basis.derivative(x, alpha=(1,), cache=cache) @ beta
    u_xx = basis.derivative(x, alpha=(2,), cache=cache) @ beta

    coef = solve_lstsq(torch.cat([u, u_x], dim=1), u_xx)  # u_xx = a*u + b*u_x
    a, b = float(coef[0]), float(coef[1])
    assert abs(a - (-4.25)) < 0.3, f"restoring coeff a={a:.3f} (want -4.25)"
    assert b < 0 and abs(b - (-1.0)) < 0.5, f"damping coeff b={b:.3f} (want -1.0)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
