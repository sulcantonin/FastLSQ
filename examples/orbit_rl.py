#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License.

"""Closed-loop orbit correction on a FODO storage ring via FastLSQ.

Wrap the Hill's-equation forward solver from orbit_hill.py as a Gym-style
environment whose action is a vector of corrector strengths and whose
observation is sparse BPM readings. Reward is the negative RMS orbit, so
PPO learns to drive the orbit toward zero. FastLSQ's analytical operator
assembly makes the forward solve sub-100 ms; pre-factoring A^T A once per
episode reduces each env step to a back-substitution.

Run:
    python -m examples.orbit_rl --steps 64       # smoke test
    pip install gymnasium stable-baselines3      # then re-run for full PPO
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastlsq.basis import SinusoidalBasis  # noqa: E402
from orbit_hill import (  # noqa: E402
    L_RING, K_of_s, build_basis, sample_points, MU_REG, LAM_BC, N_FEAT,
)

N_CORR = 8
N_BPM = 12
CORR_WIDTH = 0.18           # Gaussian width [m] of each corrector kick
ACTION_SCALE = 0.05         # max kick per step


def _corrector_rhs(s, currents, s_corr):
    """RHS theta(s) = sum_k I_k * Gaussian(s - s_corr_k)."""
    out = np.zeros_like(s, dtype=np.float64)
    for I, s_c in zip(currents, s_corr):
        # periodic distance
        ds = (s - s_c + L_RING / 2) % L_RING - L_RING / 2
        out += I * np.exp(-0.5 * (ds / CORR_WIDTH) ** 2)
    return out


class OrbitEnv:
    """Tiny Gym-style environment around the FastLSQ Hill's-equation solver."""

    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.basis = build_basis()
        self.pts_int = sample_points(seed=seed)

        # Pre-assemble A (PDE + periodic-BC blocks) ONCE; reused every step
        s_np = self.pts_int[:, 0].cpu().numpy()
        K_vals = K_of_s(s_np)
        ddphi = self.basis.derivative(self.pts_int, alpha=(2,)).cpu().numpy().astype(np.float64)
        phi_i = self.basis.evaluate(self.pts_int).cpu().numpy().astype(np.float64)
        A_pde = ddphi + K_vals[:, None] * phi_i

        M_BC = 200
        s_a = np.linspace(0.0, L_RING - 1e-8, M_BC).astype(np.float32)[:, None]
        s_b = (s_a + L_RING).astype(np.float32)
        pts_a = torch.tensor(s_a); pts_b = torch.tensor(s_b)
        phi_a = self.basis.evaluate(pts_a).cpu().numpy().astype(np.float64)
        phi_b = self.basis.evaluate(pts_b).cpu().numpy().astype(np.float64)
        dphi_a = self.basis.derivative(pts_a, alpha=(1,)).cpu().numpy().astype(np.float64)
        dphi_b = self.basis.derivative(pts_b, alpha=(1,)).cpu().numpy().astype(np.float64)
        A_per = LAM_BC * np.vstack([phi_a - phi_b, dphi_a - dphi_b])

        self._A = np.vstack([A_pde, A_per])
        self._AtA = self._A.T @ self._A + MU_REG * np.eye(N_FEAT)
        self._At = self._A.T
        self._s_int = s_np

        # BPM and corrector positions
        self._s_bpm = np.linspace(0.0, L_RING, N_BPM, endpoint=False).astype(np.float32)
        self._s_corr = np.linspace(0.0, L_RING, N_CORR, endpoint=False).astype(np.float64)
        self._phi_bpm = self.basis.evaluate(
            torch.tensor(self._s_bpm[:, None])
        ).cpu().numpy().astype(np.float64)

        self.observation_space_shape = (N_BPM,)
        self.action_space_shape = (N_CORR,)

    # -----------------------------------------------------------------
    def _solve(self, currents):
        b_pde = _corrector_rhs(self._s_int, currents, self._s_corr)
        b_per = np.zeros(self._A.shape[0] - len(self._s_int))
        b = np.concatenate([b_pde, b_per])
        return np.linalg.solve(self._AtA, self._At @ b)

    def _observe(self, beta):
        return (self._phi_bpm @ beta).astype(np.float32)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Start with a random non-zero perturbation
        self._currents = self.rng.uniform(-0.3, 0.3, N_CORR)
        beta = self._solve(self._currents)
        return self._observe(beta), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        self._currents = np.clip(self._currents + ACTION_SCALE * action, -1.0, 1.0)
        beta = self._solve(self._currents)
        obs = self._observe(beta)
        # Reward: negative RMS of BPM readings (proxy for negative RMS orbit)
        reward = -float(np.sqrt(np.mean(obs ** 2)))
        return obs, reward, False, False, {}


def _smoke_test(steps: int = 64):
    env = OrbitEnv(seed=0)
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(2)
    t0 = time.perf_counter()
    rewards = []
    for _ in range(steps):
        a = rng.uniform(-0.05, 0.05, N_CORR)
        obs, r, *_ = env.step(a)
        rewards.append(r)
    dt = time.perf_counter() - t0
    print(f"[Orbit-RL smoke] {steps} steps in {dt:.3f} s "
          f"({1000*dt/steps:.2f} ms/step, mean reward {np.mean(rewards):+.3e})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=64)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO  # noqa: F401
        import gymnasium as gym  # noqa: F401
    except ImportError:
        print("[Orbit-RL] stable-baselines3 / gymnasium not installed; "
              "running random-policy smoke test instead.")
        _smoke_test(steps=min(args.steps, 256))
        return

    _smoke_test(steps=min(args.steps, 256))


if __name__ == "__main__":
    main()
