#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License.

"""
Linearised power-grid swing equation on the IEEE 14-bus network.

Equation (quasi-steady DC power-flow / linearised swing):

    L @ delta = p

where L is the network graph Laplacian (entries B_ij between connected
buses), delta in R^14 is the rotor-angle vector and p in R^14 is the net
power injection vector. This is a *discrete* analogue of the continuous
Laplacian solves elsewhere in the paper - FastLSQ here is a regularised
graph-Laplacian least-squares solver, with the operator matrix supplied
in closed form (a single matrix multiplication, no autodiff).

The IEEE 14-bus topology is hard-coded so the example has zero external
dependencies. Susceptances B_ij are taken from the standard benchmark
(uniformised at 1.0 here for simplicity - the discrete-graph framing
abstracts away electrical detail).

This script:
  1. Builds the 14x14 graph Laplacian.
  2. Constructs an exact angle vector and the matching injection p = L @ delta.
  3. Runs the FastLSQ (LSQ over delta with Tikhonov reg) and compares to
     scipy.sparse.linalg.spsolve as a ground-truth baseline.
"""

from __future__ import annotations

import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# IEEE 14-bus topology (undirected edges, 1-based bus indexing)
EDGES = [
    (1, 2), (1, 5), (2, 3), (2, 4), (2, 5),
    (3, 4), (4, 5), (4, 7), (4, 9), (5, 6),
    (6, 11), (6, 12), (6, 13), (7, 8), (7, 9),
    (9, 10), (9, 14), (10, 11), (12, 13), (13, 14),
]
N_BUS = 14

# Slack bus (reference) - pinned at zero angle to remove the Laplacian's null
# space (the all-ones vector). Standard practice in DC power flow.
SLACK_BUS = 1   # 1-based; converted to 0-based internally


def build_laplacian(b_uniform: float = 1.0) -> np.ndarray:
    """Graph Laplacian L with susceptance b on every edge."""
    L = np.zeros((N_BUS, N_BUS), dtype=np.float64)
    for i, j in EDGES:
        i0, j0 = i - 1, j - 1
        L[i0, j0] -= b_uniform
        L[j0, i0] -= b_uniform
        L[i0, i0] += b_uniform
        L[j0, j0] += b_uniform
    return L


def pin_slack(L: np.ndarray, slack: int = SLACK_BUS) -> tuple[np.ndarray, list[int]]:
    """Remove the slack bus from L to get a non-singular reduced system.

    Returns (L_red, keep_idx) where keep_idx are the remaining bus indices
    (0-based) and L_red is the corresponding (N-1) x (N-1) submatrix.
    """
    keep = [i for i in range(N_BUS) if i != (slack - 1)]
    return L[np.ix_(keep, keep)], keep


# ---------------------------------------------------------------------------
# FastLSQ-style solve (with Tikhonov regularisation - the relevant knob
# from the continuous setting carries over verbatim).
# ---------------------------------------------------------------------------

MU_REG = 1e-12


def fastlsq_solve(L: np.ndarray, p: np.ndarray, mu: float = MU_REG) -> np.ndarray:
    """Tikhonov-regularised least-squares solve of L @ x = p.

    Mirrors the continuous-domain FastLSQ pipeline:
        A = L,    b = p
        x = (A^T A + mu I)^{-1} A^T b.
    The 'operator matrix A' is the graph Laplacian itself, supplied in
    closed form via build_laplacian().
    """
    A = L
    AtA = A.T @ A + mu * np.eye(A.shape[0])
    return np.linalg.solve(AtA, A.T @ p)


# ---------------------------------------------------------------------------
def main():
    L_full = build_laplacian(b_uniform=1.0)
    L_red, keep = pin_slack(L_full)
    n_red = L_red.shape[0]

    rng = np.random.default_rng(0)
    delta_exact = rng.normal(0.0, 1.0, n_red)
    p_red = L_red @ delta_exact

    # Sparse-LU baseline (the gold standard for this DC-power-flow problem)
    t0 = time.perf_counter()
    delta_lu = spsolve(csr_matrix(L_red), p_red)
    t_lu = time.perf_counter() - t0

    # FastLSQ
    t0 = time.perf_counter()
    delta_fl = fastlsq_solve(L_red, p_red)
    t_fl = time.perf_counter() - t0

    rel_err_fl = float(np.linalg.norm(delta_fl - delta_exact) / np.linalg.norm(delta_exact))
    rel_err_vs_lu = float(np.linalg.norm(delta_fl - delta_lu) / np.linalg.norm(delta_lu))

    print(f"[Grid-Swing] IEEE 14-bus  (reduced dim {n_red}, edges {len(EDGES)})")
    print(f"  sparse-LU baseline : {1000*t_lu:.3f} ms")
    print(f"  FastLSQ            : {1000*t_fl:.3f} ms")
    print(f"  rel err vs exact   : {rel_err_fl:.2e}")
    print(f"  rel err vs LU      : {rel_err_vs_lu:.2e}")

    return dict(rel_err_exact=rel_err_fl, rel_err_lu=rel_err_vs_lu,
                t_lu=t_lu, t_fl=t_fl, n_red=n_red)


if __name__ == "__main__":
    main()
