#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License.

"""Sparse-BPM orbit reconstruction on a FODO storage ring.

Recover x(s) along an N_CELLS-cell FODO ring from 12 noisy BPM readings,
using the Hill's-equation PDE residual as a soft physical prior. Setup
mirrors the EFIT-style sparse-probe Grad-Shafranov example in gs_inverse.py:
data fidelity term + PDE residual term + periodic BC penalty, combined into
one least-squares problem.

In a real storage ring the corrector kicks would be unknown; here we
synthesise observations from the manufactured x_exact of orbit_hill.py so the
script runs without external data.
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastlsq.basis import SinusoidalBasis  # noqa: E402
from orbit_hill import (  # noqa: E402
    L_RING, K_of_s, x_exact, theta_required, build_basis,
    sample_points, MU_REG, LAM_BC, N_FEAT,
)

N_BPM = 12
LAM_DATA = 50.0
LAM_PDE = 1.0
NOISE_LEVEL = 0.01


def main():
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    basis = build_basis()
    pts_int = sample_points()

    # --- PDE block ---
    s_np = pts_int[:, 0].cpu().numpy()
    K_vals = K_of_s(s_np)
    ddphi = basis.derivative(pts_int, alpha=(2,)).cpu().numpy().astype(np.float64)
    phi_i = basis.evaluate(pts_int).cpu().numpy().astype(np.float64)
    A_pde = LAM_PDE * (ddphi + K_vals[:, None] * phi_i)
    b_pde = LAM_PDE * theta_required(s_np)

    # --- Periodic BC block ---
    M_BC = 200
    s_a = np.linspace(0.0, L_RING - 1e-8, M_BC).astype(np.float32)[:, None]
    s_b = (s_a + L_RING).astype(np.float32)
    pts_a = torch.tensor(s_a); pts_b = torch.tensor(s_b)
    phi_a = basis.evaluate(pts_a).cpu().numpy().astype(np.float64)
    phi_b = basis.evaluate(pts_b).cpu().numpy().astype(np.float64)
    dphi_a = basis.derivative(pts_a, alpha=(1,)).cpu().numpy().astype(np.float64)
    dphi_b = basis.derivative(pts_b, alpha=(1,)).cpu().numpy().astype(np.float64)
    A_per = LAM_BC * np.vstack([phi_a - phi_b, dphi_a - dphi_b])
    b_per = np.zeros(2 * M_BC)

    # --- Sparse BPM observations ---
    s_bpm = np.linspace(0.0, L_RING, N_BPM, endpoint=False).astype(np.float32)
    pts_bpm = torch.tensor(s_bpm[:, None])
    phi_bpm = basis.evaluate(pts_bpm).cpu().numpy().astype(np.float64)
    x_bpm_true = x_exact(s_bpm)
    noise = rng.normal(0.0, NOISE_LEVEL * np.std(x_bpm_true), N_BPM)
    x_bpm_obs = x_bpm_true + noise
    A_data = LAM_DATA * phi_bpm
    b_data = LAM_DATA * x_bpm_obs

    A_full = np.vstack([A_pde, A_per, A_data])
    b_full = np.concatenate([b_pde, b_per, b_data])

    t0 = time.perf_counter()
    AtA = A_full.T @ A_full + MU_REG * np.eye(N_FEAT)
    beta = np.linalg.solve(AtA, A_full.T @ b_full)
    t_solve = time.perf_counter() - t0

    s_eval = np.linspace(0.0, L_RING, 2000)
    pts_eval = torch.tensor(s_eval[:, None], dtype=torch.float32)
    Phi = basis.evaluate(pts_eval).cpu().numpy().astype(np.float64)
    x_hat = Phi @ beta
    x_ref = x_exact(s_eval)
    rel_l2 = float(np.linalg.norm(x_hat - x_ref) / np.linalg.norm(x_ref))

    print(f"[Orbit-Inverse] N_BPM={N_BPM} noise={NOISE_LEVEL:.0%}")
    print(f"  solve : {t_solve:.3f} s")
    print(f"  rel L2: {rel_l2:.2e}")

    return dict(rel_l2=rel_l2, t_solve=t_solve)


if __name__ == "__main__":
    main()
