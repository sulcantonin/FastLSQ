#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License.

"""
Hill's equation for transverse beam orbit on a periodic FODO storage ring.

Equation:
    d^2 x/ds^2 + K(s) x(s) = theta(s),
    x(s + L) = x(s),  x'(s + L) = x'(s).

K(s) is a 24-cell FODO lattice (focusing-defocusing alternating quadrupoles
with drift sections in between); theta(s) is a corrector-kick distribution.

FastLSQ assembles A_{ij} = phi_j''(s_i) + K(s_i) phi_j(s_i) analytically -
the same variable-coefficient pattern used in inverse_magnetostatics.py.
Periodic boundary conditions are enforced as paired soft penalties on the
zeroth and first derivatives at s = 0 and s = L.

This script:
  1. Builds a 24-cell FODO lattice.
  2. Picks a manufactured periodic exact orbit and computes the corresponding
     corrector distribution theta_required = x_exact'' + K x_exact.
  3. Runs FastLSQ and reports rel L2 error + wall-clock time.
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastlsq.basis import SinusoidalBasis  # noqa: E402

# ---------------------------------------------------------------------------
# Lattice
# ---------------------------------------------------------------------------

N_CELLS = 24
CELL_LEN = 1.0                  # [m] per FODO cell
L_RING = N_CELLS * CELL_LEN     # ring circumference

K0 = 1.2                        # [1/m^2] quadrupole strength
QUAD_FRACTION = 0.20            # fraction of each cell occupied by a quad


def K_of_s(s: np.ndarray) -> np.ndarray:
    """Piecewise-constant FODO focusing function.

    Each cell is [F-quad | drift | D-quad | drift], where F and D quads each
    span QUAD_FRACTION/2 of the cell, alternating sign.
    """
    s = np.asarray(s, dtype=np.float64) % L_RING
    phase = (s / CELL_LEN) % 1.0
    q = QUAD_FRACTION
    f_quad = phase < (q / 2)
    d_quad = (phase >= 0.5) & (phase < 0.5 + q / 2)
    out = np.zeros_like(s)
    out[f_quad] = +K0
    out[d_quad] = -K0
    return out


# ---------------------------------------------------------------------------
# Manufactured exact orbit
# ---------------------------------------------------------------------------

def x_exact(s):
    """Smooth periodic exact orbit, period L_RING."""
    s = np.asarray(s)
    return 0.30 * np.sin(2 * np.pi * s / L_RING) \
         + 0.10 * np.cos(4 * np.pi * s / L_RING)


def x_exact_dd(s):
    s = np.asarray(s)
    k1 = 2 * np.pi / L_RING
    k2 = 4 * np.pi / L_RING
    return -0.30 * (k1 ** 2) * np.sin(k1 * s) - 0.10 * (k2 ** 2) * np.cos(k2 * s)


def theta_required(s):
    """RHS for which x_exact is the unique periodic solution of Hill's equation."""
    return x_exact_dd(s) + K_of_s(s) * x_exact(s)


# ---------------------------------------------------------------------------
# Basis & sampling
# ---------------------------------------------------------------------------

N_FEAT_PER_BLOCK = 500
SIGMAS = [3.0, 7.0]
N_FEAT = N_FEAT_PER_BLOCK * len(SIGMAS)

M_INT = 4000
M_BC = 200          # paired periodic-BC pseudo-points (s=0 vs s=L)
LAM_BC = 200.0
MU_REG = 1e-9


def build_basis() -> SinusoidalBasis:
    """Multi-block 1-D sinusoidal basis."""
    Ws, bs = [], []
    for sigma in SIGMAS:
        blk = SinusoidalBasis.random(1, N_FEAT_PER_BLOCK, sigma=sigma)
        Ws.append(blk.W)
        bs.append(blk.b)
    return SinusoidalBasis(torch.cat(Ws, dim=1), torch.cat(bs, dim=1), normalize=True)


def sample_points(seed: int = 0):
    rng = np.random.default_rng(seed)
    s_int = rng.uniform(0.0, L_RING, M_INT)
    pts_int = torch.tensor(s_int[:, None], dtype=torch.float32)
    return pts_int


# ---------------------------------------------------------------------------
# Operator: phi''(s) + K(s) phi(s)
# ---------------------------------------------------------------------------

def assemble(basis: SinusoidalBasis, pts_int: torch.Tensor):
    """Variable-coefficient operator A_ij = phi_j''(s_i) + K(s_i) phi_j(s_i)."""
    s_np = pts_int[:, 0].cpu().numpy()
    K_vals = K_of_s(s_np)

    # second derivative along the only spatial axis
    ddphi = basis.derivative(pts_int, alpha=(2,)).cpu().numpy().astype(np.float64)
    phi   = basis.evaluate(pts_int).cpu().numpy().astype(np.float64)
    A_pde = ddphi + K_vals[:, None] * phi

    # Periodic BC: phi_j(0) = phi_j(L) and phi_j'(0) = phi_j'(L), paired rows.
    s_pair = np.linspace(0.0, 1e-6, 1)  # single representative pair is enough; reused below
    # Build many paired anchor points across the period? A single pair is sufficient for
    # 1-D periodicity since the basis is global. We use M_BC paired evaluations spread
    # around the period to spread the constraint mass across features.
    s_a = np.linspace(0.0, L_RING - 1e-8, M_BC).astype(np.float32)[:, None]
    s_b = (s_a + L_RING).astype(np.float32)         # mod-period equivalent via shift
    # Since the basis is sin(W s + b), evaluating at s + L gives a different value
    # unless W (2pi/L) is an integer; the soft penalty pulls the recovered solution
    # toward periodicity but does NOT bind features to be periodic (acceptable
    # because x_exact itself is periodic).
    pts_a = torch.tensor(s_a)
    pts_b = torch.tensor(s_b)
    phi_a = basis.evaluate(pts_a).cpu().numpy().astype(np.float64)
    phi_b = basis.evaluate(pts_b).cpu().numpy().astype(np.float64)
    dphi_a = basis.derivative(pts_a, alpha=(1,)).cpu().numpy().astype(np.float64)
    dphi_b = basis.derivative(pts_b, alpha=(1,)).cpu().numpy().astype(np.float64)
    A_per_val = LAM_BC * (phi_a - phi_b)
    A_per_drv = LAM_BC * (dphi_a - dphi_b)

    A_full = np.vstack([A_pde, A_per_val, A_per_drv])

    b_pde = theta_required(s_np)
    b_per_val = np.zeros(M_BC)
    b_per_drv = np.zeros(M_BC)
    b_full = np.concatenate([b_pde, b_per_val, b_per_drv])

    return A_full, b_full


def solve(A, b):
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
    pts_int = sample_points()

    t0 = time.perf_counter()
    A, b = assemble(basis, pts_int)
    t_assemble = time.perf_counter() - t0

    t0 = time.perf_counter()
    beta = solve(A, b)
    t_solve = time.perf_counter() - t0

    # Evaluate on a dense grid
    s_eval = np.linspace(0.0, L_RING, 2000)
    pts_eval = torch.tensor(s_eval[:, None], dtype=torch.float32)
    Phi = basis.evaluate(pts_eval).cpu().numpy().astype(np.float64)
    x_hat = Phi @ beta
    x_ref = x_exact(s_eval)
    rel_l2 = float(np.linalg.norm(x_hat - x_ref) / np.linalg.norm(x_ref))

    print(f"[Orbit-Hill] N_feat={N_FEAT} M_int={M_INT} M_bc={M_BC}")
    print(f"  assemble: {t_assemble:.3f} s")
    print(f"  solve   : {t_solve:.3f} s")
    print(f"  rel L2  : {rel_l2:.2e}")

    return dict(rel_l2=rel_l2, t_assemble=t_assemble, t_solve=t_solve,
                s_eval=s_eval, x_hat=x_hat, x_ref=x_ref, basis=basis)


if __name__ == "__main__":
    main()
