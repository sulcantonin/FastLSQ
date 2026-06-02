# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License.

"""Smoke tests for the Grad-Shafranov example."""

import os
import sys

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")
sys.path.insert(0, EXAMPLES_DIR)


def test_grad_shafranov_forward_accuracy():
    import grad_shafranov as gs

    result = gs.main()
    # Closed-form Δ* assembly should reach the paper's 10^-6 ballpark.
    assert result["rel_l2"] < 5e-6, f"rel L2 too high: {result['rel_l2']:.2e}"
    # Assembly is the headline claim -- must stay well under a second.
    assert result["t_assemble"] < 0.5


def test_grad_shafranov_sparse_inverse():
    import gs_inverse as inv

    result = inv.main()
    # 30 noisy probes + PDE prior should still resolve ψ to ~10^-3.
    assert result["rel_l2"] < 5e-3, f"inverse rel L2 too high: {result['rel_l2']:.2e}"


def test_grad_shafranov_rl_smoke():
    import gs_rl_control as rl

    env = rl.GradShafranovEnv(seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (rl.N_PROBES,)
    import numpy as np
    obs2, reward, term, trunc, info = env.step(np.full(rl.N_COILS, 0.05))
    assert obs2.shape == (rl.N_PROBES,)
    assert not term and not trunc
    assert isinstance(reward, float)
