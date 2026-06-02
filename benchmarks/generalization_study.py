#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License.

"""Generalization study for FastLSQ.

Three distinct generalization regimes that the fixed-N, fixed-RHS,
in-distribution comparison in benchmark_comparison.py does NOT cover:

  (A) Cross-RHS reuse.
      The operator matrix A and its factorisation depend only on the basis
      and the PDE operator. Re-solving for a new forcing term f reduces to a
      single back-substitution. We solve a 2D Poisson against 50 different
      RHS functions and measure amortised time + error.

  (B) Out-of-distribution (OOD) extrapolation.
      We collocate the solver on a SUB-domain [0.2, 0.8]² and evaluate the
      learned solution on the FULL [0, 1]² grid (so ~64% of the test grid
      is geometrically outside the collocation hull). Measures the
      basis-induced inductive bias.

  (C) Boundary-condition perturbation.
      Train with homogeneous Dirichlet BC, evaluate against a perturbed
      problem whose true solution has a non-trivial trace. Measures how
      gracefully the soft-penalty formulation degrades under model
      mismatch.
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastlsq.basis import SinusoidalBasis  # noqa: E402

torch.set_default_dtype(torch.float64)

N_FEAT = 1000
SIGMA = 4.0
LAM_BC = 100.0
MU_REG = 1e-8


# ---------------------------------------------------------------------------
def make_basis(seed: int = 0):
    torch.manual_seed(seed)
    return SinusoidalBasis.random(2, N_FEAT, sigma=SIGMA)


def grid(n_int: int, n_bc: int, box=(0.0, 1.0), seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    lo, hi = box
    pts_int = lo + (hi - lo) * torch.rand(n_int, 2, generator=g)
    side = n_bc // 4
    parts = []
    for s in range(4):
        t = lo + (hi - lo) * torch.rand(side, generator=g)
        if s == 0: parts.append(torch.stack([torch.full_like(t, lo), t], dim=1))
        elif s == 1: parts.append(torch.stack([torch.full_like(t, hi), t], dim=1))
        elif s == 2: parts.append(torch.stack([t, torch.full_like(t, lo)], dim=1))
        else: parts.append(torch.stack([t, torch.full_like(t, hi)], dim=1))
    return pts_int, torch.cat(parts, dim=0)


def test_pts(box=(0.0, 1.0), n: int = 60):
    xs = torch.linspace(box[0], box[1], n)
    X, Y = torch.meshgrid(xs, xs, indexing="ij")
    return torch.stack([X.ravel(), Y.ravel()], dim=1)


# ---------------------------------------------------------------------------
# (A) Cross-RHS reuse
# ---------------------------------------------------------------------------

def study_A_cross_rhs(n_rhs: int = 50):
    basis = make_basis(0)
    pts_int, pts_bc = grid(4000, 1200)
    A_pde = -basis.laplacian(pts_int)
    A_bc = LAM_BC * basis.evaluate(pts_bc)
    A = torch.cat([A_pde, A_bc], dim=0)

    # Cholesky factorise once
    t0 = time.perf_counter()
    AtA = A.T @ A + MU_REG * torch.eye(N_FEAT)
    L = torch.linalg.cholesky(AtA)
    t_setup = time.perf_counter() - t0

    rng = np.random.default_rng(0)
    pts_test = test_pts()
    errors = []
    t0 = time.perf_counter()
    for _ in range(n_rhs):
        # Random superposition of low-frequency sine modes as exact solution
        n_modes = 4
        ks = rng.integers(1, 4, (n_modes, 2))
        cs = rng.normal(0, 1, n_modes)

        def u_ex_np(x, y):
            out = np.zeros_like(x)
            for c, (kx, ky) in zip(cs, ks):
                out += c * np.sin(kx * np.pi * x) * np.sin(ky * np.pi * y)
            return out

        def rhs_np(x, y):
            out = np.zeros_like(x)
            for c, (kx, ky) in zip(cs, ks):
                out += c * ((kx ** 2 + ky ** 2) * np.pi ** 2) * \
                       np.sin(kx * np.pi * x) * np.sin(ky * np.pi * y)
            return out

        b_pde = torch.tensor(rhs_np(pts_int[:, 0].numpy(), pts_int[:, 1].numpy()))
        b_bc = LAM_BC * torch.tensor(u_ex_np(pts_bc[:, 0].numpy(), pts_bc[:, 1].numpy()))
        b = torch.cat([b_pde, b_bc])
        Atb = A.T @ b
        beta = torch.cholesky_solve(Atb.unsqueeze(1), L).squeeze(1)
        pred = basis.evaluate(pts_test) @ beta
        ref = torch.tensor(u_ex_np(pts_test[:, 0].numpy(), pts_test[:, 1].numpy()))
        errors.append(float((pred - ref).norm() / ref.norm()))
    t_solve_all = time.perf_counter() - t0

    print(f"[A] Cross-RHS reuse (one factorisation, {n_rhs} different f)")
    print(f"     setup (assemble + Cholesky): {t_setup*1000:>7.1f} ms")
    print(f"     amortised per RHS:           {1000*t_solve_all/n_rhs:>7.1f} ms")
    print(f"     median rel L2:               {np.median(errors):>10.2e}")
    print(f"     max    rel L2:               {np.max(errors):>10.2e}")
    return dict(setup_ms=t_setup * 1000, per_rhs_ms=1000 * t_solve_all / n_rhs,
                median_err=float(np.median(errors)), max_err=float(np.max(errors)))


# ---------------------------------------------------------------------------
# (B) OOD extrapolation
# ---------------------------------------------------------------------------

def study_B_ood():
    basis = make_basis(1)
    # Collocate on inner box [0.2, 0.8]²; evaluate on full [0, 1]²
    pts_int, pts_bc = grid(4000, 1200, box=(0.2, 0.8))
    A_pde = -basis.laplacian(pts_int)
    A_bc = LAM_BC * basis.evaluate(pts_bc)
    A = torch.cat([A_pde, A_bc], dim=0)

    def u_ex(x): return torch.sin(np.pi * x[:, 0]) * torch.sin(np.pi * x[:, 1])
    def rhs(x): return 2.0 * np.pi**2 * torch.sin(np.pi*x[:,0]) * torch.sin(np.pi*x[:,1])

    b = torch.cat([rhs(pts_int), LAM_BC * u_ex(pts_bc)])
    AtA = A.T @ A + MU_REG * torch.eye(N_FEAT)
    beta = torch.linalg.solve(AtA, A.T @ b)

    # In-distribution test
    pts_id = test_pts(box=(0.2, 0.8))
    err_id = float(((basis.evaluate(pts_id) @ beta) - u_ex(pts_id)).norm() / u_ex(pts_id).norm())

    # OOD test: full box minus inner ring
    pts_full = test_pts(box=(0.0, 1.0))
    mask = (pts_full[:, 0] < 0.2) | (pts_full[:, 0] > 0.8) | \
           (pts_full[:, 1] < 0.2) | (pts_full[:, 1] > 0.8)
    pts_ood = pts_full[mask]
    err_ood = float(((basis.evaluate(pts_ood) @ beta) - u_ex(pts_ood)).norm() / u_ex(pts_ood).norm())

    print(f"[B] OOD extrapolation (collocate on [0.2,0.8]², test on full [0,1]²)")
    print(f"     in-distribution rel L2: {err_id:>10.2e}")
    print(f"     extrapolation rel L2:   {err_ood:>10.2e}  "
          f"(degradation × {err_ood/err_id:.1f})")
    return dict(err_id=err_id, err_ood=err_ood)


# ---------------------------------------------------------------------------
# (C) Boundary perturbation
# ---------------------------------------------------------------------------

def study_C_bc_perturbation():
    basis = make_basis(2)
    pts_int, pts_bc = grid(4000, 1200)
    A_pde = -basis.laplacian(pts_int)
    A_bc = LAM_BC * basis.evaluate(pts_bc)
    A = torch.cat([A_pde, A_bc], dim=0)

    def u_ex(x):  # u = sin(πx) sin(πy) + 0.3 * x * (1 − y)   (non-zero trace)
        return torch.sin(np.pi * x[:, 0]) * torch.sin(np.pi * x[:, 1]) \
               + 0.3 * x[:, 0] * (1.0 - x[:, 1])

    def rhs(x):  # -Δu = 2π² sin sin   (linear term is harmonic)
        return 2 * np.pi ** 2 * torch.sin(np.pi * x[:, 0]) * torch.sin(np.pi * x[:, 1])

    pts_test = test_pts()
    errs = {}
    for label, g_fn in [("homogeneous", lambda x: torch.zeros(x.shape[0])),
                        ("true (non-zero)", lambda x: u_ex(x))]:
        b = torch.cat([rhs(pts_int), LAM_BC * g_fn(pts_bc)])
        AtA = A.T @ A + MU_REG * torch.eye(N_FEAT)
        beta = torch.linalg.solve(AtA, A.T @ b)
        pred = basis.evaluate(pts_test) @ beta
        ref = u_ex(pts_test)
        errs[label] = float((pred - ref).norm() / ref.norm())

    print(f"[C] BC perturbation (true solution has non-zero trace)")
    print(f"     soft penalty with homogeneous BC: rel L2 = {errs['homogeneous']:.2e}")
    print(f"     soft penalty with true BC values: rel L2 = {errs['true (non-zero)']:.2e}")
    return errs


if __name__ == "__main__":
    study_A_cross_rhs()
    print()
    study_B_ood()
    print()
    study_C_bc_perturbation()
