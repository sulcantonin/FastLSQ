#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License.

"""
Grad-Shafranov plasma equilibrium via FastLSQ (forward solve).

The axisymmetric MHD equilibrium ψ(R, Z) of an ideal magnetised plasma obeys

    Δ* ψ ≡ ∂²ψ/∂R² − (1/R) ∂ψ/∂R + ∂²ψ/∂Z² = −μ₀ R J_φ(R, ψ).

For prescribed profiles p'(ψ) and FF'(ψ), the toroidal current density is

    J_φ = R p'(ψ) + (1/μ₀ R) F F'(ψ),

so when p'(ψ) and FF'(ψ) are *linearised about a reference profile* the
right-hand side becomes a known function of (R, Z) and the operator is linear
in ψ. This makes it a clean fit for FastLSQ's one-shot least-squares solver:

    A_ij = Δ*[φ_j](R_i, Z_i),  φ_j(R, Z) = sin(W_j · (R, Z) + b_j)

and Δ* itself decomposes analytically into `basis.laplacian` minus
`(1/R) · ∂/∂R` -- both available in closed form from `SinusoidalBasis`.

This script:
  1. Builds a manufactured Solov'ev-style exact solution on a rectangular
     poloidal cross-section.
  2. Assembles A analytically (no autodiff).
  3. Solves a single least-squares system.
  4. Reports L_2 error and wall-clock time.
"""

import os
import sys
import time
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastlsq.basis import SinusoidalBasis  # noqa: E402

# ---------------------------------------------------------------------------
# Domain & manufactured solution
# ---------------------------------------------------------------------------

R0, R1 = 0.6, 1.4       # major-radius range [m] (MAST-U scale)
Z0, Z1 = -0.8, 0.8      # vertical range [m]

# Solov'ev-style closed-form ψ_exact:  ψ(R, Z) = R² (R² − R_a²) Z²  (up to a const)
R_AXIS = 1.0


def psi_exact(R, Z):
    return (R ** 2) * (R ** 2 - R_AXIS ** 2) * (Z ** 2)


def gs_rhs(R, Z):
    """Apply Δ* analytically to psi_exact to obtain the source term.

    ψ = R²(R² − R_a²) Z² = (R⁴ − R_a² R²) Z²
    ∂ψ/∂R   = (4 R³ − 2 R_a² R) Z²
    ∂²ψ/∂R² = (12 R² − 2 R_a²) Z²
    ∂²ψ/∂Z² = 2 R² (R² − R_a²)
    Δ*ψ = ∂²ψ/∂R² − (1/R) ∂ψ/∂R + ∂²ψ/∂Z²
        = (12 R² − 2 R_a²) Z² − (4 R² − 2 R_a²) Z² + 2 R² (R² − R_a²)
        = (8 R²) Z² + 2 R² (R² − R_a²)
    """
    return 8.0 * R ** 2 * Z ** 2 + 2.0 * R ** 2 * (R ** 2 - R_AXIS ** 2)


# ---------------------------------------------------------------------------
# Sampling & basis
# ---------------------------------------------------------------------------

N_FEAT_PER_BLOCK = 400
SIGMAS = [3.0, 7.0]
N_FEAT = N_FEAT_PER_BLOCK * len(SIGMAS)

M_INT = 4000        # interior collocation
M_BC = 1200         # boundary collocation
LAM_BC = 100.0
MU_REG = 1e-9


def build_basis() -> SinusoidalBasis:
    Ws, bs = [], []
    for sigma in SIGMAS:
        blk = SinusoidalBasis.random(2, N_FEAT_PER_BLOCK, sigma=sigma)
        Ws.append(blk.W)
        bs.append(blk.b)
    return SinusoidalBasis(torch.cat(Ws, dim=1), torch.cat(bs, dim=1), normalize=True)


def sample_points():
    rng = np.random.default_rng(0)
    R_int = rng.uniform(R0, R1, M_INT)
    Z_int = rng.uniform(Z0, Z1, M_INT)
    pts_int = torch.tensor(np.stack([R_int, Z_int], axis=1), dtype=torch.float32)

    # Boundary: equal weight on all four edges
    m_side = M_BC // 4
    R_b1 = np.full(m_side, R0); Z_b1 = rng.uniform(Z0, Z1, m_side)
    R_b2 = np.full(m_side, R1); Z_b2 = rng.uniform(Z0, Z1, m_side)
    Z_b3 = np.full(m_side, Z0); R_b3 = rng.uniform(R0, R1, m_side)
    Z_b4 = np.full(m_side, Z1); R_b4 = rng.uniform(R0, R1, m_side)
    R_b = np.concatenate([R_b1, R_b2, R_b3, R_b4])
    Z_b = np.concatenate([Z_b1, Z_b2, Z_b3, Z_b4])
    pts_bc = torch.tensor(np.stack([R_b, Z_b], axis=1), dtype=torch.float32)
    return pts_int, pts_bc


# ---------------------------------------------------------------------------
# Operator: Δ* = laplacian − (1/R) ∂/∂R
# ---------------------------------------------------------------------------

def assemble(basis: SinusoidalBasis, pts_int: torch.Tensor, pts_bc: torch.Tensor):
    lap = basis.laplacian(pts_int).cpu().numpy()                  # (M_int, N)
    grad = basis.gradient(pts_int).cpu().numpy()                  # (M_int, 2, N)
    dphi_dR = grad[:, 0, :]
    R_int = pts_int[:, 0].cpu().numpy()

    A_pde = lap - (1.0 / R_int)[:, None] * dphi_dR                # closed-form Δ*

    A_bc = LAM_BC * basis.evaluate(pts_bc).cpu().numpy()
    A_full = np.vstack([A_pde, A_bc])

    R_b = pts_bc[:, 0].cpu().numpy(); Z_b = pts_bc[:, 1].cpu().numpy()
    b_pde = gs_rhs(R_int, pts_int[:, 1].cpu().numpy())
    b_bc = LAM_BC * psi_exact(R_b, Z_b)
    b_full = np.concatenate([b_pde, b_bc])

    return A_full, b_full


def solve(A, b):
    # Promote to float64 so normal-equation Cholesky stays positive definite
    # even at small regularisation (basis is generated in float32).
    A64 = A.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)
    AtA = A64.T @ A64 + MU_REG * np.eye(A64.shape[1])
    Atb = A64.T @ b64
    cho = cho_factor(AtA)
    return cho_solve(cho, Atb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(0)
    basis = build_basis()
    pts_int, pts_bc = sample_points()

    t0 = time.perf_counter()
    A, b = assemble(basis, pts_int, pts_bc)
    t_assemble = time.perf_counter() - t0

    t0 = time.perf_counter()
    beta = solve(A, b)
    t_solve = time.perf_counter() - t0

    # Evaluate on a grid for error reporting
    n_eval = 80
    R_eval = np.linspace(R0, R1, n_eval)
    Z_eval = np.linspace(Z0, Z1, n_eval)
    Rg, Zg = np.meshgrid(R_eval, Z_eval, indexing="ij")
    pts_eval = torch.tensor(np.stack([Rg.ravel(), Zg.ravel()], axis=1),
                            dtype=torch.float32)
    Phi = basis.evaluate(pts_eval).cpu().numpy()
    psi_hat = (Phi @ beta).reshape(Rg.shape)
    psi_ref = psi_exact(Rg, Zg)
    rel_l2 = np.linalg.norm(psi_hat - psi_ref) / np.linalg.norm(psi_ref)

    print(f"[Grad-Shafranov] N_feat={N_FEAT}  M_int={M_INT}  M_bc={M_BC}")
    print(f"  assemble: {t_assemble:.3f} s")
    print(f"  solve   : {t_solve:.3f} s")
    print(f"  rel L2  : {rel_l2:.2e}")

    return dict(rel_l2=rel_l2, t_assemble=t_assemble, t_solve=t_solve,
                psi_hat=psi_hat, psi_ref=psi_ref, R_grid=Rg, Z_grid=Zg)


if __name__ == "__main__":
    main()
