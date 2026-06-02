#!/usr/bin/env python
"""Closed-loop Observe-Fit-Act on a simulated ALS-U ring.

This script answers the user's question:
   ``s01_sofb_observe_fit_act has a fixed set of magnets; how do we do
   exploration?  This sounds more like we retrieve data from archiver
   and as more data come, we improve the simulation.''

That earlier scenario was indeed PASSIVE streaming inference: an
archive replay never lets the agent's choices change what data arrives
next.  Here we close the loop properly:

  World          : a pyAT-grounded linearised orbit-response model of
                   the ALS-U storage ring,
                       x(t)  =  R * theta(t)  +  d(t),
                   where R is the design Hill Green's function evaluated
                   from the parsed lattice (the same R used in
                   s01_orbit_inverse.py) and d(t) is a slowly varying
                   external orbit drift (Earth-tide-like, M2 + 8 min comb)
                   that the agent must compensate for.

  Sensor         : 12 BPM channels (one per sector), 12-vector at each
                   discrete time step.

  Actuator       : 12 correctors (one per sector); each step the agent
                   chooses a theta in R^12.

  Surrogate (FF) : a streaming FFF model of the residual orbit field
                   r_hat(s, t)  =  x_hat(s, t)  -  R @ theta_now,
                   refit each step over a rolling FIFO of past
                   (theta, x) interactions.

  Policy         : greedy step that picks the next theta to minimise
                   predicted ||x|| using the surrogate's prediction;
                   that is,
                       theta_next  =  - R^+ ( x_meas  +  r_hat )
                   --- response-matrix inversion *combined with* the
                   FFF-recovered drift forecast.

  Baselines      : (a) RANDOM agent (no surrogate, random thetas),
                   (b) RESPONSE-only agent (knows R exactly but has no
                       model of d(t)), and
                   (c) ORACLE agent (knows R and the true d(t)).

The exploration is real here: every theta the agent issues changes the
next BPM observation, which changes the buffer the surrogate refits on,
which changes the next theta.

Outputs /tmp/alsu_observe_fit_act_sim.pdf and .png.
"""
from __future__ import annotations

import os, sys, time, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fastlsq
from fastlsq import SinusoidalBasis, Op, solve_lstsq

from s01_orbit_inverse import alsu_optics, response_matrix

torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

N_BPM      = 12              # 1 per sector
N_CORR     = 12              # 1 per sector, mid-arc
N_STEPS    = 400             # episode length
DT         = 1.0             # seconds between Observe-Fit-Act calls
N_FEAT     = 600             # random Fourier features in the surrogate
BUFFER_LEN = 80              # rolling FIFO length (Observe-Fit-Act paper
                              #  uses ~M=200-2500 here)
MU_REG     = 1e-4
SEED       = 0
SENSOR_SIGMA = 0.01          # BPM read noise (mm-equivalent units)


# ----------------------------------------------------------------------
# World: linearised orbit-response, anchored on the real ALS-U lattice
# ----------------------------------------------------------------------

def build_world():
    """Returns response matrix R (N_BPM, N_CORR) plus an Earth-tide-like
    drift d(t) function from the actual ALS-U lattice optics."""
    opt = alsu_optics()
    n_per_sec = opt["n_bpm_sp"]
    # one BPM per sector (mid-of-sector)
    bpm_idx = np.array([(k * n_per_sec) + n_per_sec // 2
                        for k in range(N_BPM)])
    # one corrector per sector, slightly offset (where you'd put a VCM)
    corr_idx = np.array([(k * n_per_sec) + n_per_sec // 2 + 1
                         for k in range(N_CORR)])
    R = response_matrix(opt["beta_y"][bpm_idx], opt["mu_y"][bpm_idx],
                        opt["beta_y"][corr_idx], opt["mu_y"][corr_idx],
                        opt["Q_y"])
    # Drift d(t):  M2 (12.42 h) + 8 min + diurnal random walk
    def d_of_t(t_seconds):
        omega_M2 = 2 * np.pi / (12.42 * 3600.0)
        omega_8m = 2 * np.pi / (8 * 60.0)
        # 12 BPM channels with phase-shifted amplitudes:
        sectors = np.arange(N_BPM, dtype=float)
        phi_M2 = sectors * 2 * np.pi / 12   # azimuthal phase
        phi_8m = sectors * 2 * np.pi / 12 + 0.7
        amp_M2 = 0.20 + 0.05 * np.cos(sectors * 2 * np.pi / 12)
        amp_8m = 0.10 + 0.04 * np.sin(sectors * 2 * np.pi / 12 + 1.1)
        d = (amp_M2 * np.sin(omega_M2 * t_seconds + phi_M2)
             + amp_8m * np.sin(omega_8m * t_seconds + phi_8m))
        return d
    return R, d_of_t, opt


# ----------------------------------------------------------------------
# Agents
# ----------------------------------------------------------------------

class RandomAgent:
    """No model.  Issues random small corrections."""
    name = "random"
    def __init__(self, R, scale=0.5):
        self.scale = scale
        self.n = R.shape[1]
        self.rng = np.random.default_rng(SEED + 1)
    def act(self, step, t, x_meas, history):
        return self.rng.normal(scale=self.scale, size=self.n)
    def fit(self, *a, **k):
        return 0.0


class ResponseAgent:
    """Knows R perfectly, has no model of d(t).  Best linear controller
    that ignores temporal structure: theta = - R^+ x_meas."""
    name = "response-only"
    def __init__(self, R):
        self.R_pinv = np.linalg.pinv(R)
    def act(self, step, t, x_meas, history):
        return - self.R_pinv @ x_meas
    def fit(self, *a, **k):
        return 0.0


class OracleAgent:
    """Knows R AND the true drift d(t).  The unreachable upper bound."""
    name = "oracle"
    def __init__(self, R, d_fn):
        self.R_pinv = np.linalg.pinv(R)
        self.d_fn   = d_fn
    def act(self, step, t, x_meas, history):
        # Cancel the drift one step ahead so x(t+1) ~ 0.
        d_next = self.d_fn(t + DT)
        return - self.R_pinv @ d_next
    def fit(self, *a, **k):
        return 0.0


class FFFAgent:
    """Observe-Fit-Act agent: FFF surrogate of the residual orbit field
    r_hat(s_bpm, t) refit on a rolling (theta, x) buffer.  The action is
    a forward-aware correction:  predict r(t+1), then theta = -R^+
    (r_hat(t+1)).  Knows R exactly (it's the design response matrix), but
    not d(t) -- that's what FFF learns from the rolling buffer.
    """
    name = "FFF (Observe-Fit-Act)"

    def __init__(self, R, buffer_len=BUFFER_LEN, n_features=N_FEAT,
                 mu=MU_REG):
        self.R = R
        self.R_pinv = np.linalg.pinv(R)
        self.buffer_len = buffer_len
        self.mu = mu
        # 2-D basis: (sector_index, t).
        torch.manual_seed(SEED)
        sigma_s = 2 * np.pi / N_BPM * 2.0
        sigma_t = 2 * np.pi / 60.0          # ~ 1-minute scale
        self.basis = SinusoidalBasis.random_anisotropic(
            input_dim=2, n_features=n_features,
            sigma=[sigma_s, sigma_t])
        self.beta = None
        self.last_refit_ms = 0.0

    def fit(self, history):
        if len(history) < 8:
            self.beta = None
            return 0.0
        recent = history[-self.buffer_len:]
        N = self.R.shape[0]
        rows = []; vals = []
        for (step, t, theta, x_meas) in recent:
            # Strip the *known* response component R*theta off x_meas to
            # isolate the drift field d that FFF must learn.
            r = x_meas - self.R @ theta
            sectors = np.arange(N, dtype=float)
            for k in range(N):
                rows.append([float(k), float(t)])
                vals.append(r[k])
        x_t = torch.tensor(rows, dtype=torch.float64)
        y_t = torch.tensor(vals, dtype=torch.float64).reshape(-1, 1)
        phi = self.basis.evaluate(x_t)
        t0 = time.perf_counter()
        self.beta = solve_lstsq(phi, y_t, mu=self.mu)
        t1 = time.perf_counter()
        self.last_refit_ms = (t1 - t0) * 1000.0
        return self.last_refit_ms

    def predict_r(self, t_query):
        sectors = np.arange(self.R.shape[0], dtype=float)
        rows = np.stack([sectors,
                         np.full(self.R.shape[0], t_query)], axis=1)
        x_t = torch.tensor(rows, dtype=torch.float64)
        phi = self.basis.evaluate(x_t)
        return (phi @ self.beta).reshape(-1).numpy()

    def act(self, step, t, x_meas, history):
        if self.beta is None:
            # No model yet: fall back to plain response inversion.
            return - self.R_pinv @ x_meas
        # Predict the drift at next step and pre-cancel it.
        r_next = self.predict_r(t + DT)
        return - self.R_pinv @ r_next


# ----------------------------------------------------------------------
# Closed-loop episode runner
# ----------------------------------------------------------------------

def run_episode(agent, R, d_fn, n_steps=N_STEPS):
    history = []   # list of (step, t, theta, x_meas)
    x_norms = np.zeros(n_steps)
    fit_ms  = np.zeros(n_steps)
    rng = np.random.default_rng(SEED + 2)
    # Initial state: no kick, only drift
    t = 0.0
    theta = np.zeros(R.shape[1])
    for step in range(n_steps):
        # World: observe x = R theta + d(t) + noise
        x_clean = R @ theta + d_fn(t)
        x_meas = x_clean + rng.normal(scale=SENSOR_SIGMA, size=x_clean.shape)
        x_norms[step] = float(np.linalg.norm(x_meas))
        # Append the EFFECT of the most recent theta to history first
        history.append((step, t, theta.copy(), x_meas.copy()))
        # Fit (cheap, one Tikhonov solve)
        fit_ms[step] = agent.fit(history)
        # Act: choose the next theta
        theta = agent.act(step, t, x_meas, history)
        t += DT
    return {"x_norms": x_norms, "fit_ms": fit_ms, "history": history}


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def run():
    print(">> Closed-loop Observe-Fit-Act on simulated ALS-U", flush=True)
    R, d_fn, opt = build_world()
    print(f"   world:  N_BPM = {N_BPM},  N_CORR = {N_CORR},  "
          f"|R| = {np.linalg.norm(R):.2f},  Q_y = {opt['Q_y']:.4f}")
    agents = [
        RandomAgent(R),
        ResponseAgent(R),
        FFFAgent(R),
        OracleAgent(R, d_fn),
    ]
    results = {}
    for ag in agents:
        print(f"\n   === running agent: {ag.name} ===")
        out = run_episode(ag, R, d_fn)
        # Summary
        x_final_rms = float(np.sqrt(np.mean(out["x_norms"][-50:] ** 2)))
        x_first_rms = float(np.sqrt(np.mean(out["x_norms"][:50] ** 2)))
        median_ms   = float(np.median(out["fit_ms"][out["fit_ms"] > 0])
                            if (out["fit_ms"] > 0).any() else 0.0)
        print(f"   ||x|| RMS first 50 steps = {x_first_rms:.4f}")
        print(f"   ||x|| RMS last 50 steps  = {x_final_rms:.4f}")
        print(f"   ratio early/late          = "
              f"{x_first_rms / max(x_final_rms, 1e-30):.2f}x")
        print(f"   refit median ms           = {median_ms:.2f}")
        results[ag.name] = {**out, "x_first": x_first_rms,
                            "x_final": x_final_rms,
                            "median_ms": median_ms}
    # Final comparison
    print()
    print("   ===  final-stage  ||x||  RMS (last 50 steps)  ===")
    for n, r in results.items():
        print(f"     {n:25s}  =  {r['x_final']:.4f}")
    # Ratios
    base = results["response-only"]["x_final"]
    oracle = results["oracle"]["x_final"]
    fff  = results["FFF (Observe-Fit-Act)"]["x_final"]
    print()
    print(f"   FastLSQ vs response-only :  "
          f"{fff / max(base, 1e-30):.3f}x   "
          f"(lower is better;  FFF wins if < 1)")
    print(f"   FastLSQ vs oracle gap    :  "
          f"({fff - oracle:.4f}  remaining RMS,  "
          f"{(fff-oracle)/max(base-oracle, 1e-30)*100:.1f}% of "
          f"response-only-to-oracle gap)")
    return results


def make_figure(res, out_pdf="/tmp/alsu_observe_fit_act_sim.pdf"):
    if res is None:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                             gridspec_kw=dict(height_ratios=[1, 0.55]))
    ax = axes[0]
    colours = {
        "random":                "#bc4444",
        "response-only":         "#888888",
        "FFF (Observe-Fit-Act)": "#3666b4",
        "oracle":                "#54a04e",
    }
    for name, r in res.items():
        x = np.arange(len(r["x_norms"]))
        ax.semilogy(x, r["x_norms"], lw=1.0, color=colours.get(name, "k"),
                    label=f"{name}   (final RMS {r['x_final']:.3f})")
    ax.set_xlabel("Observe-Fit-Act step")
    ax.set_ylabel("$||x_{meas}||$  (12-BPM L2 norm)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Closed-loop control on simulated ALS-U ring with "
                 "tidal+8-min drift\n"
                 "FastLSQ surrogate learns the drift field "
                 "online from the (theta, x) interaction buffer",
                 fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax = axes[1]
    r_fff = res["FFF (Observe-Fit-Act)"]
    mask = r_fff["fit_ms"] > 0
    ax.plot(np.arange(len(r_fff["fit_ms"]))[mask], r_fff["fit_ms"][mask],
            "o-", color="#3666b4", lw=0.6, markersize=2,
            label=f"FFF refit median {r_fff['median_ms']:.1f} ms")
    ax.set_xlabel("step")
    ax.set_ylabel("refit ms")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight", dpi=140)
    plt.savefig(out_pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=130)
    print(f"   wrote {out_pdf}")


if __name__ == "__main__":
    res = run()
    make_figure(res)
