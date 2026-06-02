# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License.

"""Smoke tests for the IEEE 14-bus grid swing-equation example."""

import os
import sys

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")
sys.path.insert(0, EXAMPLES_DIR)


def test_grid_swing_forward_accuracy():
    import grid_swing as gs

    result = gs.main()
    # Discrete-graph Laplacian LSQ should match sparse-LU to machine precision.
    assert result["rel_err_exact"] < 1e-8, f"rel err too high: {result['rel_err_exact']:.2e}"
    assert result["rel_err_lu"] < 1e-8


def test_grid_inverse_sparse_pmu():
    import grid_inverse as inv

    result = inv.main()
    # 5/13 PMUs at 1% noise: 10^-2 ballpark is acceptable.
    assert result["rel_l2"] < 5e-2, f"inverse rel L2 too high: {result['rel_l2']:.2e}"


def test_grid_rl_smoke():
    import grid_rl_control as rl
    import numpy as np

    env = rl.GridEnv(seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (rl.N_PMU,)
    obs2, reward, term, trunc, _ = env.step(np.full(rl.N_GEN, 0.05))
    assert obs2.shape == (rl.N_PMU,)
    assert not term and not trunc
    assert isinstance(reward, float)
