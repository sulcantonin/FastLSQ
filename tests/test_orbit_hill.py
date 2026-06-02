# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License.

"""Smoke tests for the orbit / Hill's-equation example."""

import os
import sys

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")
sys.path.insert(0, EXAMPLES_DIR)


def test_orbit_hill_forward_accuracy():
    import orbit_hill as oh

    result = oh.main()
    # Manufactured periodic orbit with closed-form Hill's-equation assembly
    # should resolve to the paper's <= 10^-6 ballpark.
    assert result["rel_l2"] < 5e-6, f"rel L2 too high: {result['rel_l2']:.2e}"
    # Analytical assembly is the headline claim - must stay well under a second.
    assert result["t_assemble"] < 0.5


def test_orbit_inverse_sparse_bpm():
    import orbit_inverse as inv

    result = inv.main()
    # 12 BPMs across a 24-cell ring at 1% noise: 10^-2 ballpark is acceptable.
    assert result["rel_l2"] < 5e-2, f"inverse rel L2 too high: {result['rel_l2']:.2e}"


def test_orbit_rl_smoke():
    import orbit_rl as rl
    import numpy as np

    env = rl.OrbitEnv(seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (rl.N_BPM,)
    obs2, reward, term, trunc, _ = env.step(np.full(rl.N_CORR, 0.05))
    assert obs2.shape == (rl.N_BPM,)
    assert not term and not trunc
    assert isinstance(reward, float)
