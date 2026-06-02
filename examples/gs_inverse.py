#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License.

"""
Sparse-probe Grad-Shafranov reconstruction.

Recover ψ(R, Z) across a tokamak poloidal cross-section from a small set of
magnetic probes (~30) that measure the poloidal magnetic field components

    B_R = (1/R) ∂ψ/∂Z,    B_Z = −(1/R) ∂ψ/∂R

along with the Grad-Shafranov PDE constraint as a soft regulariser. Setup
mirrors EFIT-style equilibrium reconstruction from MAST-U public diagnostics,
but here we synthesize probe readings from the manufactured ground truth in
``grad_shafranov.py`` so the script runs without external data.

The reconstruction is a single least-squares problem combining:
  * sparse probe data (B_R, B_Z) at N_PROBES locations,
  * PDE residual at M_INT interior collocation points,
  * Dirichlet boundary at M_BC boundary points.

All gradients of φ_j are obtained in closed form from ``SinusoidalBasis``.
"""

import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastlsq.basis import SinusoidalBasis  # noqa: E402
from grad_shafranov import (  # noqa: E402
    R0, R1, Z0, Z1, psi_exact, gs_rhs,
    build_basis, sample_points, MU_REG, LAM_BC,
)

# ---------------------------------------------------------------------------
N_PROBES = 30
LAM_PDE = 1.0
LAM_DATA = 50.0
NOISE_LEVEL = 0.01


def synth_probes(rng: np.random.Generator):
    """Place probes around the boundary; values are (B_R, B_Z) from ψ_exact."""
    angles = rng.uniform(0.0, 2 * np.pi, N_PROBES)
    R_mid = 0.5 * (R0 + R1); Z_mid = 0.5 * (Z0 + Z1)
    r_radius = 0.45 * min(R1 - R0, Z1 - Z0)
    R_p = R_mid + r_radius * np.cos(angles)
    Z_p = Z_mid + r_radius * np.sin(angles)

    # Closed-form derivatives of psi_exact = R²(R²−R_a²)Z²:
    #   ∂ψ/∂R = (4 R³ − 2 R_a² R) Z²
    #   ∂ψ/∂Z = 2 R² (R² − R_a²) Z
    R_a = 1.0
    dpsi_dR = (4 * R_p ** 3 - 2 * R_a ** 2 * R_p) * Z_p ** 2
    dpsi_dZ = 2 * R_p ** 2 * (R_p ** 2 - R_a ** 2) * Z_p
    B_R = dpsi_dZ / R_p
    B_Z = -dpsi_dR / R_p

    noise_R = rng.normal(0.0, NOISE_LEVEL * (np.std(B_R) + 1e-12), N_PROBES)
    noise_Z = rng.normal(0.0, NOISE_LEVEL * (np.std(B_Z) + 1e-12), N_PROBES)
    return (np.stack([R_p, Z_p], axis=1).astype(np.float32),
            B_R + noise_R, B_Z + noise_Z)


def main():
    torch.manual_seed(1)
    rng = np.random.default_rng(1)

    basis = build_basis()
    pts_int, pts_bc = sample_points()

    # --- PDE block (same as forward solver) ---
    lap = basis.laplacian(pts_int).cpu().numpy().astype(np.float64)
    grad_int = basis.gradient(pts_int).cpu().numpy().astype(np.float64)
    R_int = pts_int[:, 0].cpu().numpy().astype(np.float64)
    A_pde = LAM_PDE * (lap - (1.0 / R_int)[:, None] * grad_int[:, 0, :])
    b_pde = LAM_PDE * gs_rhs(R_int, pts_int[:, 1].cpu().numpy().astype(np.float64))

    # --- Boundary block ---
    A_bc = LAM_BC * basis.evaluate(pts_bc).cpu().numpy().astype(np.float64)
    R_b = pts_bc[:, 0].cpu().numpy().astype(np.float64)
    Z_b = pts_bc[:, 1].cpu().numpy().astype(np.float64)
    b_bc = LAM_BC * psi_exact(R_b, Z_b)

    # --- Sparse probes (B_R, B_Z) ---
    probes_np, B_R_obs, B_Z_obs = synth_probes(rng)
    pts_probe = torch.tensor(probes_np)
    grad_probe = basis.gradient(pts_probe).cpu().numpy().astype(np.float64)
    R_p = probes_np[:, 0].astype(np.float64)
    A_BR = LAM_DATA * (1.0 / R_p)[:, None] * grad_probe[:, 1, :]
    A_BZ = LAM_DATA * (-1.0 / R_p)[:, None] * grad_probe[:, 0, :]
    b_BR = LAM_DATA * B_R_obs
    b_BZ = LAM_DATA * B_Z_obs

    A_full = np.vstack([A_pde, A_bc, A_BR, A_BZ])
    b_full = np.concatenate([b_pde, b_bc, b_BR, b_BZ])

    t0 = time.perf_counter()
    AtA = A_full.T @ A_full + MU_REG * np.eye(A_full.shape[1])
    Atb = A_full.T @ b_full
    beta = np.linalg.solve(AtA, Atb)
    t_solve = time.perf_counter() - t0

    # Evaluate
    n_eval = 80
    Rg, Zg = np.meshgrid(np.linspace(R0, R1, n_eval),
                         np.linspace(Z0, Z1, n_eval), indexing="ij")
    pts_eval = torch.tensor(np.stack([Rg.ravel(), Zg.ravel()], axis=1),
                            dtype=torch.float32)
    Phi = basis.evaluate(pts_eval).cpu().numpy().astype(np.float64)
    psi_hat = (Phi @ beta).reshape(Rg.shape)
    psi_ref = psi_exact(Rg, Zg)
    rel_l2 = np.linalg.norm(psi_hat - psi_ref) / np.linalg.norm(psi_ref)

    print(f"[GS-Inverse] N_probes={N_PROBES} noise={NOISE_LEVEL:.2%}")
    print(f"  solve : {t_solve:.3f} s")
    print(f"  rel L2: {rel_l2:.2e}")

    return dict(rel_l2=rel_l2, t_solve=t_solve)


if __name__ == "__main__":
    main()
