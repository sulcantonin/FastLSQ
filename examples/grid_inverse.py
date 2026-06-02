#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License.

"""Sparse-PMU rotor-angle reconstruction on the IEEE 14-bus grid.

Recover the full angle vector delta in R^13 from k <= 13 noisy PMU readings
plus the graph-Laplacian prior:

    minimize lambda_pde * || L delta - p ||^2 + lambda_data * || S delta - y ||^2

where S in R^{k x 13} is the selection matrix picking the observed buses
and y are the noisy PMU measurements. The solver is a single least-squares
call combining both blocks - identical pipeline to gs_inverse.py.

Real-world analogue: ENTSO-E publishes minute-scale frequency / angle data
from a handful of PMUs across each control area; recovering the full bus
state requires combining sparse measurements with a power-flow physics
prior.
"""

from __future__ import annotations

import time
import numpy as np

from grid_swing import (
    EDGES, N_BUS, SLACK_BUS, build_laplacian, pin_slack,
)

N_PMU = 5
LAM_PDE = 1.0
LAM_DATA = 50.0
NOISE_LEVEL = 0.01
MU_REG = 1e-10


def main():
    rng = np.random.default_rng(1)
    L_full = build_laplacian(b_uniform=1.0)
    L_red, keep = pin_slack(L_full)
    n_red = L_red.shape[0]

    # Ground-truth angles + matching injections
    delta_exact = rng.normal(0.0, 1.0, n_red)
    p_red = L_red @ delta_exact

    # Sparse PMUs: pick N_PMU random buses to observe
    pmu_idx = rng.choice(n_red, N_PMU, replace=False)
    S = np.zeros((N_PMU, n_red))
    S[np.arange(N_PMU), pmu_idx] = 1.0
    y = delta_exact[pmu_idx] + rng.normal(0.0, NOISE_LEVEL * np.std(delta_exact), N_PMU)

    # Stacked LSQ:  [ lambda_pde * L ; lambda_data * S ] @ delta = [ lambda_pde * p ; lambda_data * y ]
    A_pde = LAM_PDE * L_red
    A_data = LAM_DATA * S
    A = np.vstack([A_pde, A_data])
    b = np.concatenate([LAM_PDE * p_red, LAM_DATA * y])

    t0 = time.perf_counter()
    AtA = A.T @ A + MU_REG * np.eye(n_red)
    delta_hat = np.linalg.solve(AtA, A.T @ b)
    t_solve = time.perf_counter() - t0

    rel_l2 = float(np.linalg.norm(delta_hat - delta_exact) / np.linalg.norm(delta_exact))
    print(f"[Grid-Inverse] {N_PMU}/{n_red} PMUs at {NOISE_LEVEL:.0%} noise")
    print(f"  solve : {1000*t_solve:.3f} ms")
    print(f"  rel L2: {rel_l2:.2e}")

    return dict(rel_l2=rel_l2, t_solve=t_solve, n_pmu=N_PMU, n_red=n_red)


if __name__ == "__main__":
    main()
