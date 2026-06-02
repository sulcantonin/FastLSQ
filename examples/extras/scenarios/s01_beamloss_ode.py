#!/usr/bin/env python
"""FastLSQ-load-bearing recovery of the time-varying beam-loss rate from
the ALS-U DCCT trace.

Between top-off injection pulses the stored beam current obeys the
first-order linear decay ODE

    dI/dt(t)  =  - gamma(t) * I(t)  +  s(t) ,                       (*)

where
  * gamma(t) is the *time-varying* total loss rate
    (Touschek + residual gas + RF-bucket spillover),
  * s(t) is the impulsive drive from the top-off injection pulses,
    detected directly from the DCCT trace as sharp positive jumps.

Knowing the impulses, gamma(t) is the only remaining unknown, and we
identify it as a SMOOTH Fast-Fourier-Features field
``gamma(t) ~ phi_g(t) beta_g`` so that the closed-form constant-gamma
fit becomes the N_g = 1 special case.  The full LSQ system is *linear*
once I_hat(t) and dI_hat/dt(t) have been pre-fit through the FFF basis.

Pipeline:
  1. fetch SR:DCCT (~10 Hz native, 24 h ~ 800k samples) and regrid to a
     dense regular grid;
  2. fit I(t) on an FFF basis with an explicit bias feature so the 500 mA
     DC level is absorbed; compute dI/dt analytically through the FFF
     cyclic identity (Op.partial), not by finite differences;
  3. detect top-off events as positive outliers of dI/dt;
  4. build the known impulse train s_hat(t);
  5. solve the linear LSQ  (I_hat * phi_g) beta_g = - dI_hat + s_hat
     for gamma(t) on the FFF gamma-basis (very smooth);
  6. report constant-gamma residual, FFF-gamma residual, and the
     resulting <tau> = 1/<gamma>.

Outputs /tmp/alsu_beamloss_ode.pdf and .png.
"""
from __future__ import annotations

import os, sys, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

import fastlsq
from fastlsq import SinusoidalBasis, Op, solve_lstsq

from s01_betatron_tune import _ensure_daq, _series_to_arrays


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

PV_DCCT = "SR:DCCT"
HOURS = 24.0
N_GRID = 8000           # regular grid for the FFF fit (~3 s spacing)
N_FEAT_I = 800          # FFF features for I(t) (saw-tooth resolved)
N_FEAT_G = 24           # FFF features for gamma(t)  (smooth)


# ----------------------------------------------------------------------
# Fetch DCCT
# ----------------------------------------------------------------------

def fetch_dcct(hours=HOURS):
    get_pv = _ensure_daq()
    end = datetime.utcnow()
    start = end - timedelta(hours=hours)
    resp, status = get_pv(PV_DCCT, start=start, end=end)
    if status != 200:
        return None, None, f"HTTP {status}"
    t, v = _series_to_arrays(resp)
    return t, v, None


def regrid(t_raw, v_raw, t_target):
    return np.interp(t_target, t_raw, v_raw)


# ----------------------------------------------------------------------
# Build FFF basis with an explicit bias feature, plus its derivative basis
# ----------------------------------------------------------------------

def fff_with_bias(tt, n_features, sigma, seed=0):
    """Build a sinusoidal FFF basis and add an explicit bias column,
    then return:
       phi   (M, N+1)   bias-augmented basis matrix
       dphi  (M, N+1)   d/dt of the same (bias derivative = 0)
    """
    torch.manual_seed(seed)
    basis = SinusoidalBasis.random(input_dim=1, n_features=n_features,
                                   sigma=sigma)
    phi = basis.evaluate(tt)
    dphi = Op.partial(0, 1, 1).apply(basis, tt)
    M = tt.shape[0]
    ones_col = torch.ones(M, 1, dtype=torch.float64)
    phi_aug = torch.cat([phi, ones_col], dim=1)
    dphi_aug = torch.cat([dphi, torch.zeros(M, 1, dtype=torch.float64)],
                         dim=1)
    return basis, phi_aug, dphi_aug


# ----------------------------------------------------------------------
# Stage A: fit I(t) on FFF basis with bias, get dI/dt analytically
# ----------------------------------------------------------------------

def fit_current(t, I):
    tt = torch.tensor(t, dtype=torch.float64).reshape(-1, 1)
    I_t = torch.tensor(I, dtype=torch.float64).reshape(-1, 1)
    # Cover frequencies up to ~0.5 Hz (Nyquist for the regrid)
    sigma_I = 2 * np.pi * 0.3        # rad/s
    basis, phi, dphi = fff_with_bias(tt, N_FEAT_I, sigma_I, seed=0)
    beta = solve_lstsq(phi, I_t, mu=1e-6)
    I_hat = (phi @ beta).reshape(-1).numpy()
    dI_hat = (dphi @ beta).reshape(-1).numpy()
    rel = float(np.linalg.norm(I_hat - I) / max(np.linalg.norm(I), 1e-30))
    return I_hat, dI_hat, rel


# ----------------------------------------------------------------------
# Stage B: detect injection events in the smoothed dI/dt
# ----------------------------------------------------------------------

def detect_events(t, I, min_jump_mA=0.3, min_gap_s=20.0):
    """Top-off events are sharp positive jumps in raw I(t).
    Use a median-baseline robust jump test."""
    di = np.diff(I, prepend=I[0])
    # robust threshold: median + k * MAD, k chosen so injection jumps stick out
    med = np.median(di)
    mad = np.median(np.abs(di - med)) + 1e-30
    thr = max(min_jump_mA, med + 6.0 * 1.4826 * mad)
    above = di > thr
    events = []
    in_evt = False
    start = 0
    for i, a in enumerate(above):
        if a and not in_evt:
            in_evt = True; start = i
        elif (not a) and in_evt:
            in_evt = False
            k = start + int(np.argmax(di[start:i+1]))
            if not events or (t[k] - events[-1][0] > min_gap_s):
                events.append((t[k], di[k]))
    return events


def impulse_train(t, events, width_s=5.0):
    """Sum of Gaussians at each detected event time, area-normalised so
    that integral of s(t) around event = the dI jump (mA)."""
    s = np.zeros_like(t)
    norm = 1.0 / (width_s * np.sqrt(2.0 * np.pi))
    for tc, dI_jump in events:
        s += dI_jump * norm * np.exp(-0.5 * ((t - tc) / width_s) ** 2)
    return s


# ----------------------------------------------------------------------
# Stage C: solve linearly for gamma(t) on FFF basis
# ----------------------------------------------------------------------

def fit_gamma(t, I_hat, dI_hat, s_hat):
    """Linear LSQ:  (I_hat * phi_g) @ beta_g = - dI_hat + s_hat,
       evaluated only on the *decay* portion (event mask off).
    """
    T = float(t[-1] - t[0])
    tt = torch.tensor(t, dtype=torch.float64).reshape(-1, 1)
    sigma_g = 2 * np.pi / max(T * 0.4, 1.0)
    basis_g, phi_g_aug, _ = fff_with_bias(tt, N_FEAT_G, sigma_g, seed=1)
    rhs = -dI_hat + s_hat
    # Build (I_hat .* phi_g) row-wise:
    I_col = torch.tensor(I_hat, dtype=torch.float64).reshape(-1, 1)
    A = I_col * phi_g_aug         # (M, N_g+1)
    b = torch.tensor(rhs, dtype=torch.float64).reshape(-1, 1)
    # Tikhonov on the FFF features (NOT on the bias column), pushing
    # gamma(t) toward a constant in the absence of evidence.
    n_g = phi_g_aug.shape[1]
    # Stack a weighted-identity regulariser for the FFF features only
    reg_diag = torch.cat([torch.ones(n_g - 1, dtype=torch.float64),
                          torch.tensor([0.0], dtype=torch.float64)])
    R = torch.diag(reg_diag) * 1e4
    AA = torch.cat([A, R], dim=0)
    bb = torch.cat([b, torch.zeros(n_g, 1, dtype=torch.float64)], dim=0)
    beta_g = solve_lstsq(AA, bb, mu=1e-8)
    gamma_t = (phi_g_aug @ beta_g).reshape(-1).numpy()
    # Also a CONSTANT-gamma baseline obtained the same way (drop FFF features)
    A_const = (I_col)                     # only the bias path
    AA_const = A_const
    bb_const = b
    g_const = float((solve_lstsq(AA_const, bb_const, mu=1e-8)).item())
    return gamma_t, g_const, beta_g.reshape(-1).numpy()


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def run():
    print(">> Beam-loss-rate ODE inverse on real ALS-U DCCT", flush=True)
    t_raw, I_raw, err = fetch_dcct(hours=HOURS)
    if err or t_raw is None:
        print(f"   could not fetch DCCT: {err}")
        return None
    print(f"   {len(t_raw)} raw samples over "
          f"{(t_raw[-1] - t_raw[0])/3600:.2f} h")
    T = float(t_raw[-1] - t_raw[0])
    # Clip to the longest "clean" sub-window where I stays > 400 mA.
    keep = I_raw > 400.0
    if keep.any():
        # find longest contiguous run
        idx = np.where(keep)[0]
        breaks = np.where(np.diff(idx) > 1)[0]
        if len(breaks) == 0:
            s_idx, e_idx = idx[0], idx[-1]
        else:
            segs = np.split(idx, breaks + 1)
            longest = max(segs, key=len)
            s_idx, e_idx = longest[0], longest[-1]
        t_raw = t_raw[s_idx:e_idx + 1] - t_raw[s_idx]
        I_raw = I_raw[s_idx:e_idx + 1]
        print(f"   clipped to clean window: "
              f"{(t_raw[-1])/3600:.2f} h, "
              f"<I> = {I_raw.mean():.3f} mA, "
              f"range [{I_raw.min():.3f}, {I_raw.max():.3f}]")
    T = float(t_raw[-1] - t_raw[0])
    t = np.linspace(0.0, T, N_GRID)
    I = regrid(t_raw, I_raw, t)
    print(f"   regridded onto {N_GRID} samples, "
          f"<I> = {I.mean():.3f} mA, range [{I.min():.3f}, {I.max():.3f}]")
    # Stage A
    I_hat, dI_hat, rel_fit = fit_current(t, I)
    print(f"   FFF fit of I(t):  rel error = {rel_fit:.3e}")
    # Stage B
    events = detect_events(t, I)
    print(f"   detected {len(events)} top-off injection events "
          f"(<dt_event> = {np.mean(np.diff([e[0] for e in events])):.1f} s)")
    s_hat = impulse_train(t, events)
    # Stage C
    gamma_t, g_const, beta_g = fit_gamma(t, I_hat, dI_hat, s_hat)
    # Residuals
    ode_const = dI_hat + g_const * I_hat - s_hat
    ode_var   = dI_hat + gamma_t * I_hat - s_hat
    # Mask out the event windows so the residual reflects decay only
    s_max = max(s_hat.max(), 1e-30)
    decay = s_hat < 0.05 * s_max
    rel_const = float(np.std(ode_const[decay]) /
                      max(np.std(dI_hat[decay]), 1e-30))
    rel_var   = float(np.std(ode_var[decay]) /
                      max(np.std(dI_hat[decay]), 1e-30))
    tau_const_h = 1.0 / max(g_const, 1e-30) / 3600.0
    tau_var_h   = 1.0 / max(gamma_t.mean(), 1e-30) / 3600.0
    print()
    print(f"   const-gamma residual (decay-only) = {rel_const:.3e}")
    print(f"   FFF  -gamma residual (decay-only) = {rel_var:.3e}")
    print(f"   improvement                       = "
          f"{rel_const / max(rel_var, 1e-30):.2f}x")
    print(f"   const gamma = {g_const:.3e} /s    tau = {tau_const_h:.2f} h")
    print(f"   FFF <gamma> = {gamma_t.mean():.3e} /s   "
          f"<tau> = {tau_var_h:.2f} h")
    print(f"   FFF gamma spread = "
          f"+/- {gamma_t.std()/abs(gamma_t.mean())*100:.1f}% RMS over 24 h")
    return {"t": t, "I": I, "I_hat": I_hat, "dI_hat": dI_hat,
            "events": events, "s_hat": s_hat,
            "gamma_t": gamma_t, "g_const": g_const,
            "rel_const": rel_const, "rel_var": rel_var,
            "tau_const_h": tau_const_h, "tau_var_h": tau_var_h,
            "rel_fit": rel_fit}


def make_figure(d, out_pdf="/tmp/alsu_beamloss_ode.pdf"):
    if d is None:
        return
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    t_h = d["t"] / 3600.0
    ax = axes[0]
    ax.plot(t_h, d["I"], color="#666666", lw=0.4, label="DCCT $I(t)$")
    ax.plot(t_h, d["I_hat"], color="#c4424b", lw=0.8,
            label="FFF reconstruction $\\hat I(t)$  "
                  f"(rel err {d['rel_fit']:.1e})")
    for tc, _ in d["events"]:
        ax.axvline(tc / 3600.0, color="#54a04e", lw=0.2, alpha=0.4)
    ax.set_ylabel("beam current (mA)")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("FFF inversion of the beam-loss ODE "
                 "$dI/dt = -\\gamma(t)\\,I(t) + s(t)$ on ALS-U DCCT\n"
                 f"24 h trace,  {len(d['events'])} injection events,  "
                 f"$\\langle\\tau\\rangle_{{FFF}}$ = {d['tau_var_h']:.1f} h   "
                 f"vs   $\\tau_{{const}}$ = {d['tau_const_h']:.1f} h",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(t_h, d["dI_hat"], color="#3b6db4", lw=0.5,
            label="$d\\hat I/dt$  (analytical via FFF cyclic identity)")
    ax.plot(t_h, d["s_hat"], color="#54a04e", lw=0.6,
            label="$s(t)$ = impulse train from detected events")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_ylabel("mA / s")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    ax = axes[2]
    ax.plot(t_h, d["gamma_t"], color="#3b6db4", lw=1.0,
            label="$\\gamma(t)$  (FFF field)")
    ax.axhline(d["g_const"], color="#c4424b", lw=1.0, ls="--",
               label=f"constant $\\gamma$ = {d['g_const']:.2e} /s "
                     f"($\\tau$ = {d['tau_const_h']:.1f} h)")
    ax.set_ylabel("loss rate $\\gamma$ (1/s)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    ax = axes[3]
    res_const = (d["dI_hat"] + d["g_const"] * d["I_hat"] - d["s_hat"])
    res_var   = (d["dI_hat"] + d["gamma_t"] * d["I_hat"] - d["s_hat"])
    s_max = max(d["s_hat"].max(), 1e-30)
    mask = d["s_hat"] < 0.05 * s_max
    res_const_m = np.where(mask, res_const, np.nan)
    res_var_m   = np.where(mask, res_var,   np.nan)
    ax.plot(t_h, res_const_m, color="#c4424b", lw=0.5,
            label=f"const-$\\gamma$ residual ({d['rel_const']:.2e})")
    ax.plot(t_h, res_var_m,   color="#3b6db4", lw=0.5,
            label=f"FFF-$\\gamma$  residual ({d['rel_var']:.2e})")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_xlabel("time (h since start of window)")
    ax.set_ylabel("ODE residual (mA/s)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight", dpi=140)
    plt.savefig(out_pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=130)
    print(f"\n   wrote {out_pdf}")


if __name__ == "__main__":
    d = run()
    make_figure(d)
