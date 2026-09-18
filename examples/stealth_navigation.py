#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Stealth navigation: a FastLSQ world model refit inside a control loop.

A drone crosses a radar interference field corner to corner.  It must reach the
goal without the intensity it senses ever tripping a detector, and it has no map
-- only a small cross of field samples around its current position, streaming in
as it moves.

    radar field      u(x, y) = sum_k sin(kappa r_k) / (r_k + eps)
    detector         trips when  u^2 > TAU
    sensing          5-point cross at the drone's position, every step
    world model      beta = (H^T H + mu I)^-1 H^T u   refit every REFIT_EVERY steps
    control          descend  w_goal * ||x - goal||^2 + w_stealth * u_hat(x)^2
                     using the surrogate's ANALYTIC gradient

The point of the demo is the third and fourth lines.  The surrogate is a
`SinusoidalBasis`, so once beta is fit, grad u_hat is a closed-form expression --
no finite differences, no autodiff graph, no re-solve.  Refitting is a single
Tikhonov least-squares solve over the samples gathered so far, cheap enough to
run hundreds of times per episode.

The world model is trustworthy only where the drone has actually sensed, which is
why the controller weights the stealth term by a confidence that decays with
distance to the nearest sample.

Run:
    python examples/stealth_navigation.py
    python examples/stealth_navigation.py --plot      # writes a PNG
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastlsq.basis import SinusoidalBasis  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 0

# Radar field
N_EMITTERS = 5
KAPPA = 34.0               # wavenumber -- sets the fringe spacing
EPS = 0.06                 # softening, keeps the field finite at an emitter
# Detector threshold on u^2.  Calibrated against the field: u^2 reaches 236 over
# the domain and the straight start->goal line peaks at 79, so a threshold of 60
# means the direct route WOULD be detected and the drone has to find a corridor.
TAU = 60.0

# Surrogate
N_FEAT = 1000
SIGMAS = (6.0, 14.0)       # two bandwidth blocks, N_FEAT split evenly
MU_REG = 1e-6              # Tikhonov ridge
REFIT_EVERY = 3

# Sensing
CROSS_H = 0.018            # arm length of the 5-point sensing cross

# Control
START = np.array([0.06, 0.06])
GOAL = np.array([0.94, 0.94])
STEP = 0.0045              # metres per step (domain is the unit square)
MAX_STEPS = 1200
GOAL_TOL = 0.03
W_GOAL = 1.0
W_STEALTH = 0.85
CONF_LEN = 0.10            # confidence decay length for the surrogate


# ---------------------------------------------------------------------------
# Ground-truth radar field (the drone never sees this, only point samples)
# ---------------------------------------------------------------------------
def make_emitters(rng):
    """Emitters placed away from the start/goal diagonal, so the corridor is
    genuinely non-trivial but a stealthy path exists."""
    return np.array([
        [0.50, 0.18],
        [0.22, 0.58],
        [0.78, 0.44],
        [0.55, 0.82],
        [0.88, 0.14],
    ])[:N_EMITTERS]


def radar_field(xy, emitters):
    """u(x, y) = sum_k sin(kappa r_k) / (r_k + eps).  xy is (M, 2)."""
    xy = np.atleast_2d(xy)
    d = xy[:, None, :] - emitters[None, :, :]        # (M, K, 2)
    r = np.linalg.norm(d, axis=2)                    # (M, K)
    return np.sum(np.sin(KAPPA * r) / (r + EPS), axis=1)


# ---------------------------------------------------------------------------
# FastLSQ surrogate
# ---------------------------------------------------------------------------
def build_basis(seed=SEED):
    torch.manual_seed(seed)
    per = N_FEAT // len(SIGMAS)
    Ws, bs = [], []
    for sigma in SIGMAS:
        blk = SinusoidalBasis.random(2, per, sigma=sigma)
        Ws.append(blk.W)
        bs.append(blk.b)
    return SinusoidalBasis(torch.cat(Ws, dim=1), torch.cat(bs, dim=1), normalize=True)


class WorldModel:
    """Tikhonov least-squares surrogate of the radar field, with analytic grad."""

    def __init__(self, basis):
        self.basis = basis
        self.beta = None
        self.n_refits = 0
        self.refit_times = []

    def refit(self, X, y):
        """One regularised least-squares solve over every sample seen so far."""
        t0 = time.perf_counter()
        Xt = torch.tensor(X, dtype=torch.float64)
        H = self.basis.evaluate(Xt).cpu().numpy().astype(np.float64)   # (M, N)
        n = H.shape[1]
        # normal equations with a ridge: the system is small in N and this is the
        # cheapest stable route when M can be < N early in the episode
        A = H.T @ H + MU_REG * np.eye(n)
        self.beta = np.linalg.solve(A, H.T @ y)
        self.refit_times.append((time.perf_counter() - t0) * 1e3)
        self.n_refits += 1

    def predict(self, xy):
        Xt = torch.tensor(np.atleast_2d(xy), dtype=torch.float64)
        H = self.basis.evaluate(Xt).cpu().numpy().astype(np.float64)
        return H @ self.beta

    def gradient(self, xy):
        """Closed-form gradient of the surrogate -- the whole point of the demo."""
        Xt = torch.tensor(np.atleast_2d(xy), dtype=torch.float64)
        G = self.basis.gradient(Xt).cpu().numpy().astype(np.float64)   # (M, 2, N)
        return G @ self.beta                                           # (M, 2)


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------
def sense_cross(pos, emitters):
    """The drone's only observation: a 5-point cross around its position."""
    offs = np.array([[0.0, 0.0], [CROSS_H, 0.0], [-CROSS_H, 0.0],
                     [0.0, CROSS_H], [0.0, -CROSS_H]])
    pts = np.clip(pos[None, :] + offs, 0.0, 1.0)
    return pts, radar_field(pts, emitters)


def run_episode(verbose=True, w_stealth=W_STEALTH):
    rng = np.random.default_rng(SEED)
    emitters = make_emitters(rng)
    basis = build_basis()
    wm = WorldModel(basis)

    pos = START.copy()
    X_buf, y_buf = [], []
    path = [pos.copy()]
    sensed_u2 = []
    detected = False

    for step in range(MAX_STEPS):
        pts, vals = sense_cross(pos, emitters)
        X_buf.append(pts)
        y_buf.append(vals)

        u_here = float(vals[0])
        sensed_u2.append(u_here ** 2)
        if u_here ** 2 > TAU:
            detected = True
            if verbose:
                print(f"  DETECTED at step {step}: u^2 = {u_here**2:.2f} > {TAU}")
            break

        if step % REFIT_EVERY == 0:
            wm.refit(np.vstack(X_buf), np.concatenate(y_buf))

        to_goal = GOAL - pos
        dist = np.linalg.norm(to_goal)
        if dist < GOAL_TOL:
            break

        dir_goal = to_goal / (dist + 1e-12)

        # Stealth term from the surrogate's analytic gradient: descend u_hat^2.
        # Both terms are unit vectors so neither can swamp the other -- an
        # unnormalised grad(u^2) reaches |2 u grad u| ~ 1e3 here and would pin the
        # drone in the first null it finds.  The stealth direction is weighted by
        # how close the *predicted* intensity is to the detector threshold, and by
        # how much we trust the model at this point.
        if wm.beta is not None and w_stealth > 0.0:
            u_hat = float(wm.predict(pos)[0])
            g_hat = wm.gradient(pos)[0]
            grad_u2 = 2.0 * u_hat * g_hat
            gn = np.linalg.norm(grad_u2)
            if gn > 1e-9:
                dir_stealth = -grad_u2 / gn
                # trust the model only near where we have actually sensed
                Xa = np.vstack(X_buf)
                near = np.min(np.linalg.norm(Xa - pos[None, :], axis=1))
                conf = float(np.exp(-(near / CONF_LEN) ** 2))
                danger = min(1.0, (u_hat ** 2) / TAU)
                stealth = w_stealth * conf * danger * dir_stealth
            else:
                stealth = np.zeros(2)
        else:
            stealth = np.zeros(2)

        drive = W_GOAL * dir_goal + stealth
        nrm = np.linalg.norm(drive)
        drive = dir_goal if nrm < 1e-9 else drive / nrm

        pos = np.clip(pos + STEP * drive, 0.0, 1.0)
        path.append(pos.copy())

    path = np.array(path)
    reached = np.linalg.norm(path[-1] - GOAL) < GOAL_TOL
    return dict(
        path=path, emitters=emitters, world_model=wm, detected=detected,
        reached=reached, steps=len(path) - 1, peak_u2=max(sensed_u2),
        n_samples=len(X_buf) * 5,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot", action="store_true", help="write a PNG of the episode")
    args = ap.parse_args()

    print("Stealth navigation -- FastLSQ world model refit in the loop")
    print(f"  radar: {N_EMITTERS} emitters, kappa={KAPPA}, detector trips at u^2 > {TAU}")
    print(f"  surrogate: N={N_FEAT} features, sigma={SIGMAS}, refit every {REFIT_EVERY} steps\n")

    t0 = time.perf_counter()
    r = run_episode()
    wall = time.perf_counter() - t0

    # Ablation: identical episode with the stealth term switched off.  This is
    # what the drone does if it ignores the surrogate's gradient and just drives
    # at the goal -- the control that shows the world model is doing the work.
    base = run_episode(verbose=False, w_stealth=0.0)

    wm = r["world_model"]
    rt = np.array(wm.refit_times)
    outcome = ("goal - undetected" if r["reached"] and not r["detected"]
               else "DETECTED" if r["detected"] else "did not reach goal")

    print("  Episode summary")
    print(f"    outcome           {outcome}")
    print(f"    steps             {r['steps']}")
    print(f"    refits            {wm.n_refits}")
    print(f"    refit cost        {np.median(rt):.1f} ms median  "
          f"({rt.min():.1f}-{rt.max():.1f} ms)")
    print(f"    samples gathered  {r['n_samples']}")
    print(f"    peak sensed u^2   {r['peak_u2']:.2f}  (detector at {TAU})")
    print(f"    wall clock        {wall:.1f} s")

    b_out = ("goal - undetected" if base["reached"] and not base["detected"]
             else "DETECTED" if base["detected"] else "did not reach goal")
    print("\n  Ablation -- same episode, stealth term off (drive straight at goal)")
    print(f"    outcome           {b_out}")
    print(f"    steps             {base['steps']}")
    print(f"    peak sensed u^2   {base['peak_u2']:.2f}  (detector at {TAU})")
    if base["detected"]:
        # The baseline stopped because it was detected, not because it arrived,
        # so its step count is where it died -- not a path length to compare.
        print(f"\n  Driving straight at the goal trips the detector at step "
              f"{base['steps']} of a {r['steps']}-step crossing.")
        print(f"  Steering on the surrogate's analytic gradient completes the "
              f"crossing with peak exposure {r['peak_u2']:.1f}, "
              f"{100.0 * (1 - r['peak_u2'] / TAU):.0f}% under the threshold.")
    else:
        print(f"\n  Gradient steering cut peak exposure "
              f"{base['peak_u2'] / r['peak_u2']:.2f}x for "
              f"{100.0 * (r['steps'] / base['steps'] - 1.0):+.0f}% path length.")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gx, gy = np.meshgrid(np.linspace(0, 1, 300), np.linspace(0, 1, 300))
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        U = radar_field(grid, r["emitters"]).reshape(gx.shape)

        # The surrogate is only meaningful where the drone actually sensed, so
        # fade it out with distance to the path rather than showing extrapolation
        # as if it were prediction.
        from scipy.spatial import cKDTree
        tree = cKDTree(r["path"])
        dist_to_path, _ = tree.query(grid)
        conf = np.exp(-(dist_to_path.reshape(gx.shape) / CONF_LEN) ** 2)

        fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))
        panels = [(U, "true radar field", None),
                  (wm.predict(grid).reshape(gx.shape),
                   "FastLSQ world model (faded where unsensed)", conf)]
        for a, (field, title, alpha) in zip(ax, panels):
            im = a.imshow(field, extent=(0, 1, 0, 1), origin="lower",
                          cmap="magma", vmin=U.min(), vmax=U.max(),
                          alpha=None if alpha is None else np.clip(alpha, 0.06, 1.0))
            a.plot(r["path"][:, 0], r["path"][:, 1], "-", color="#8ff", lw=1.8)
            a.plot(*START, "o", color="w", ms=6)
            a.plot(*GOAL, "*", color="#8f8", ms=14)
            a.plot(r["emitters"][:, 0], r["emitters"][:, 1], "x", color="w", ms=7)
            a.set_title(title, fontsize=10)
            a.set_xticks([]); a.set_yticks([])
            fig.colorbar(im, ax=a, fraction=0.046)
        fig.suptitle(f"{r['steps']} steps - {outcome} - "
                     f"{wm.n_refits} refits at {np.median(rt):.1f} ms", fontsize=11)
        fig.tight_layout()
        out = os.path.join(os.path.dirname(__file__), "..", "misc",
                           "stealth_navigation.png")
        fig.savefig(out, dpi=140)
        print(f"\n  Saved figure -> {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
