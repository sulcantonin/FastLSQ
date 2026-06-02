#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License.

"""Iso-time Pareto comparison: FastLSQ vs PIELM (tanh) on 2D Poisson.

Sweeps the feature count N for both methods and reports (time, error) pairs
so that, at any fixed wall-clock budget, one can read off the best achievable
relative L2 error per method. This is the "fair-time" baseline that
complements the fixed-N comparison in benchmark_comparison.py.

Both solvers consume identical collocation grids, identical regularisation,
and identical RHS, so the only thing that differs is the feature family
(sinusoid vs tanh) and how the operator matrix A is assembled (analytical
closed form vs problem-specific symbolic calculus, which here is
auto-handled by the FeatureBasis adapter for tanh).
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastlsq.basis import SinusoidalBasis  # noqa: E402
from fastlsq.solvers import PIELMSolver  # noqa: E402

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# Problem: -Δu = f on [0,1]², u_exact = sin(πx) sin(πy)
# ---------------------------------------------------------------------------

def u_exact(x):
    return torch.sin(np.pi * x[:, 0]) * torch.sin(np.pi * x[:, 1])


def rhs(x):
    return 2.0 * (np.pi ** 2) * torch.sin(np.pi * x[:, 0]) * torch.sin(np.pi * x[:, 1])


def make_grid(m_int: int, m_bc: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    pts_int = torch.rand(m_int, 2, generator=g)
    # 4-side boundary
    per_side = m_bc // 4
    bc = torch.empty(0, 2)
    for side in range(4):
        t = torch.rand(per_side, generator=g)
        if side == 0:   xs, ys = torch.zeros_like(t), t
        elif side == 1: xs, ys = torch.ones_like(t), t
        elif side == 2: xs, ys = t, torch.zeros_like(t)
        else:           xs, ys = t, torch.ones_like(t)
        bc = torch.cat([bc, torch.stack([xs, ys], dim=1)], dim=0)
    return pts_int, bc


def test_grid(n: int = 60):
    xs = torch.linspace(0, 1, n)
    X, Y = torch.meshgrid(xs, xs, indexing="ij")
    return torch.stack([X.ravel(), Y.ravel()], dim=1)


# ---------------------------------------------------------------------------
# Solvers (one-shot least-squares with identical pipeline)
# ---------------------------------------------------------------------------

LAM_BC = 100.0
MU_REG = 1e-10


def _assemble_and_solve(A_pde, A_bc, b_pde, b_bc):
    A = torch.cat([A_pde, LAM_BC * A_bc], dim=0)
    b = torch.cat([b_pde, LAM_BC * b_bc], dim=0)
    AtA = A.T @ A + MU_REG * torch.eye(A.shape[1], dtype=A.dtype)
    Atb = A.T @ b
    return torch.linalg.solve(AtA, Atb)


def run_fastlsq(n_feat: int, m_int: int, m_bc: int, sigma: float = 3.0):
    pts_int, pts_bc = make_grid(m_int, m_bc)
    basis = SinusoidalBasis.random(2, n_feat, sigma=sigma)
    t0 = time.perf_counter()
    A_pde = -basis.laplacian(pts_int)                    # −Δ
    A_bc = basis.evaluate(pts_bc)
    b_pde = rhs(pts_int)
    b_bc = u_exact(pts_bc)
    beta = _assemble_and_solve(A_pde, A_bc, b_pde, b_bc)
    dt = time.perf_counter() - t0

    pts_test = test_grid()
    pred = basis.evaluate(pts_test) @ beta
    ref = u_exact(pts_test)
    rel = (pred - ref).norm() / ref.norm()
    return dt, float(rel)


def run_pielm(n_feat: int, m_int: int, m_bc: int, scale: float = 5.0):
    pts_int, pts_bc = make_grid(m_int, m_bc)
    solver = PIELMSolver(input_dim=2)
    solver.add_block(hidden_size=n_feat, scale=scale)
    basis = solver.basis
    t0 = time.perf_counter()
    A_pde = -basis.laplacian(pts_int)
    A_bc = basis.evaluate(pts_bc)
    b_pde = rhs(pts_int)
    b_bc = u_exact(pts_bc)
    beta = _assemble_and_solve(A_pde, A_bc, b_pde, b_bc)
    solver.beta = beta
    dt = time.perf_counter() - t0

    pts_test = test_grid()
    pred = solver.predict(pts_test)
    ref = u_exact(pts_test)
    rel = (pred - ref).norm() / ref.norm()
    return dt, float(rel)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep():
    Ns = [200, 400, 800, 1500, 3000]
    M_INT, M_BC = 4000, 1200
    print(f"{'method':<10} {'N':>6} {'time [s]':>10} {'rel L2':>12}")
    print("-" * 44)
    results = []
    for N in Ns:
        # warm-up to avoid first-call torch overhead inflating timings
        run_fastlsq(N, M_INT, M_BC); run_pielm(N, M_INT, M_BC)
        t_f, e_f = run_fastlsq(N, M_INT, M_BC)
        t_p, e_p = run_pielm(N, M_INT, M_BC)
        print(f"{'FastLSQ':<10} {N:>6d} {t_f:>10.4f} {e_f:>12.3e}")
        print(f"{'PIELM':<10} {N:>6d} {t_p:>10.4f} {e_p:>12.3e}")
        results.append(dict(method="FastLSQ", N=N, time=t_f, rel_l2=e_f))
        results.append(dict(method="PIELM",   N=N, time=t_p, rel_l2=e_p))
    return results


if __name__ == "__main__":
    sweep()
