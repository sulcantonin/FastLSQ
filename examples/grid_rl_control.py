#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License.

"""Closed-loop frequency stabilisation on the IEEE 14-bus benchmark.

Each step: the agent picks governor setpoints (action) on the 5 generator
buses, FastLSQ resolves the rotor-angle equilibrium via the graph-Laplacian
LSQ (a single back-substitution per step once the Laplacian is pre-
factorised), and the reward is the negative RMS PMU reading.

Run:
    python -m examples.grid_rl_control --steps 64    # smoke
"""

from __future__ import annotations

import argparse
import time
import numpy as np

from grid_swing import (
    EDGES, N_BUS, SLACK_BUS, build_laplacian, pin_slack,
)

N_GEN = 5         # generator buses where the agent acts
N_PMU = 5         # observed buses
ACTION_SCALE = 0.05
MU_REG = 1e-10


class GridEnv:
    """Gym-style env around the FastLSQ graph-Laplacian solver."""

    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        L_full = build_laplacian(b_uniform=1.0)
        L_red, keep = pin_slack(L_full)
        self._L = L_red
        self._n = L_red.shape[0]
        # Pre-factor (L^T L + mu I) once for fast per-step back-substitutions
        self._AtA = L_red.T @ L_red + MU_REG * np.eye(self._n)
        self._AtA_inv = np.linalg.inv(self._AtA)  # 13x13: cheap

        # Generator and PMU bus indices (random for variety)
        all_idx = np.arange(self._n)
        self._gen_idx = self.rng.choice(all_idx, N_GEN, replace=False)
        self._pmu_idx = self.rng.choice(all_idx, N_PMU, replace=False)

        # Random ambient load disturbance (zero-mean injections on non-generators)
        self._p_load = np.zeros(self._n)
        self._reset_load()

        self.observation_space_shape = (N_PMU,)
        self.action_space_shape = (N_GEN,)

    def _reset_load(self):
        mask = np.ones(self._n, dtype=bool)
        mask[self._gen_idx] = False
        self._p_load[:] = 0.0
        self._p_load[mask] = self.rng.normal(0.0, 0.2, mask.sum())
        # Balance: total injection must be ~zero so the slack bus closes
        self._p_load[mask] -= self._p_load[mask].mean()

    def _solve(self, p_gen):
        p_total = np.zeros(self._n)
        p_total[self._gen_idx] = p_gen
        p_total += self._p_load
        # Net balance: redistribute imbalance equally
        p_total -= p_total.mean()
        return self._AtA_inv @ (self._L.T @ p_total)

    def _observe(self, delta):
        return delta[self._pmu_idx].astype(np.float32)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_load()
        self._gen_setpoints = self.rng.uniform(-0.1, 0.1, N_GEN)
        delta = self._solve(self._gen_setpoints)
        return self._observe(delta), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        self._gen_setpoints = np.clip(
            self._gen_setpoints + ACTION_SCALE * action, -2.0, 2.0
        )
        delta = self._solve(self._gen_setpoints)
        obs = self._observe(delta)
        reward = -float(np.sqrt(np.mean(obs ** 2)))
        return obs, reward, False, False, {}


def _smoke_test(steps: int = 64):
    env = GridEnv(seed=0)
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(2)
    t0 = time.perf_counter()
    rewards = []
    for _ in range(steps):
        a = rng.uniform(-0.05, 0.05, N_GEN)
        obs, r, *_ = env.step(a)
        rewards.append(r)
    dt = time.perf_counter() - t0
    print(f"[Grid-RL smoke] {steps} steps in {dt*1000:.2f} ms "
          f"({1e6*dt/steps:.1f} us/step, mean reward {np.mean(rewards):+.3e})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=64)
    args = parser.parse_args()
    try:
        from stable_baselines3 import PPO  # noqa: F401
        import gymnasium as gym  # noqa: F401
    except ImportError:
        print("[Grid-RL] stable-baselines3 / gymnasium not installed; "
              "running random-policy smoke test instead.")
        _smoke_test(steps=min(args.steps, 256))
        return
    _smoke_test(steps=min(args.steps, 256))


if __name__ == "__main__":
    main()
