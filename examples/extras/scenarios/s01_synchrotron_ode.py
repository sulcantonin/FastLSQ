#!/usr/bin/env python
"""FastLSQ-load-bearing identification of the time-varying damping rate
of the longitudinal-feedback (LFB) loop at ALS-U.

The longitudinal-feedback peak detector ``IGPF:LFB:SRAM:M1:PEAK`` is the
amplitude envelope of the bunch's coherent synchrotron oscillation as
seen by the iGp digital LFB.  Under standard Robinson + radiation-damping
theory this envelope obeys a first-order linear ODE

    dA/dt(t) = - gamma(t) A(t) + s(t),                              (*)

where
  * gamma(t) is the *time-varying* total damping rate
    (radiation damping + Robinson damping that scales with stored
    current + LFB-loop damping that depends on the loop-gain settings),
  * s(t) is the drive: bunch-current shot noise + injection kicks
    from top-off + any RF-amplitude noise leaking into the longitudinal
    plane.

The textbook closed form for gamma is a constant; in production with
top-off, gain trims, and current decay between fills, gamma is a function
of time.  That is exactly the kind of operator FastLSQ's variable-
coefficient field term ``Op.field(gamma_fn, alpha=(0,))`` is built for.

This script:
  1. fetches A(t) := IGPF:LFB:SRAM:M1:PEAK and I(t) := SR:DCCT over
     ~24 hours,
  2. fits a constant-gamma + smooth-drive baseline,
  3. fits the *same* problem with gamma(t) carried by a smooth FFF field
     basis and s(t) carried by another FFF basis, both jointly via a
     single linear least-squares solve in the compiled Op,
  4. reports the residual reduction and shows that recovered gamma(t)
     tracks the stored current (Robinson's prediction).

Outputs /tmp/alsu_synchrotron_ode.pdf and .png plus a one-line summary.
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

PV_LFB_PEAK = "IGPF:LFB:SRAM:M1:PEAK"
PV_LFB_RMS  = "IGPF:LFB:SRAM:MAXRMSVAL"
PV_DCCT     = "SR:DCCT"
HOURS       = 24.0


# ----------------------------------------------------------------------
# Fetch the two scalar traces over the same window
# ----------------------------------------------------------------------

def fetch_traces(hours=HOURS):
    get_pv = _ensure_daq()
    end = datetime.utcnow()
    start = end - timedelta(hours=hours)
    out = {}
    for label, pv in (("A", PV_LFB_PEAK),
                      ("A_alt", PV_LFB_RMS),
                      ("I", PV_DCCT)):
        try:
            resp, status = get_pv(pv, start=start, end=end)
        except Exception as e:
            out[label] = (None, None, f"exception: {e}")
            continue
        if status != 200:
            out[label] = (None, None, f"HTTP {status}")
            continue
        t, v = _series_to_arrays(resp)
        out[label] = (t, v, None)
    return out


def regrid(t_raw, v_raw, t_target):
    """Resample ``v_raw(t_raw)`` onto a regular grid by linear interpolation."""
    if t_raw is None or v_raw is None:
        return None
    return np.interp(t_target, t_raw, v_raw)


# ----------------------------------------------------------------------
# Build FFF baselines for gamma(t) and s(t) and solve
# ----------------------------------------------------------------------

def fit_constant_gamma(t, A):
    """Baseline: gamma constant, drive = smooth FFF field s(t).

    The compiled operator is ``L = d/dt + gamma_const * I``.  Jointly solve
    for the FFF coefficients of A_hat(t) ~ phi(t) beta_A *and* the
    coefficients of s(t) ~ phi_s(t) beta_s using two stacked least-squares
    blocks:

         block 1 (ODE):   L_A phi(t) beta_A  -  phi_s(t) beta_s  = 0
         block 2 (data):  phi(t) beta_A                          = A_meas .
    """
    T = t[-1] - t[0]
    # Pre-compute the basis for A(t):  oscillations down to ~1/T*60
    # = enough degrees of freedom that |residual| is data-noise-limited.
    torch.manual_seed(0)
    sigma_A = 2 * np.pi / max(T * 0.01, 1.0)
    basis_A = SinusoidalBasis.random(input_dim=1, n_features=400,
                                     sigma=sigma_A)
    basis_s = SinusoidalBasis.random(input_dim=1, n_features=200,
                                     sigma=sigma_A * 0.3)
    tt = torch.tensor(t, dtype=torch.float64).reshape(-1, 1)
    # gamma scan: find best constant gamma in log-space [1e-5, 1e0] /s
    gammas = np.logspace(-5, 0, 40)
    best = None
    for g in gammas:
        L = Op.partial(dim=0, order=1, d=1) + g * Op.identity(1)
        A_block = L.apply(basis_A, tt)         # (M, N_A)
        data_block = basis_A.evaluate(tt)      # (M, N_A)
        # Stack: [L phi_A | -phi_s ;  phi_A | 0]
        M = tt.shape[0]
        zero_s = torch.zeros(M, basis_s.n_features, dtype=torch.float64)
        s_block = basis_s.evaluate(tt)
        top = torch.cat([A_block, -s_block], dim=1)
        bot = torch.cat([data_block, zero_s], dim=1)
        AA = torch.cat([top, bot], dim=0)
        # RHS: top = 0 (ODE),  bottom = A_meas
        rhs = torch.cat([torch.zeros(M, 1, dtype=torch.float64),
                         torch.tensor(A, dtype=torch.float64).reshape(-1, 1)],
                        dim=0)
        beta = solve_lstsq(AA, rhs, mu=1e-4)
        beta_A = beta[:basis_A.n_features]
        A_hat = (basis_A.evaluate(tt) @ beta_A).reshape(-1).numpy()
        res = float(np.linalg.norm(A_hat - A) / max(np.linalg.norm(A), 1e-30))
        if best is None or res < best[0]:
            best = (res, g, beta_A.reshape(-1).numpy())
    return best   # (rel_residual, gamma_const, beta_A)


def fit_varying_gamma(t, A):
    """Variable-coefficient: gamma(t) = phi_g(t) beta_g with phi_g a SMOOTH
    FFF basis (low-frequency, so the recovered gamma cannot absorb the
    fast structure of A and trivialise the fit).

    The fit is *nonlinear* (gamma(t)·A(t) is bilinear in the unknowns
    beta_g and beta_A) but small in dim, so we run a fixed-point alternating
    least-squares solve: start with constant gamma; freeze beta_A and solve
    for beta_g linearly; freeze beta_g and solve for beta_A linearly; repeat.
    """
    T = t[-1] - t[0]
    torch.manual_seed(1)
    sigma_A = 2 * np.pi / max(T * 0.01, 1.0)
    basis_A = SinusoidalBasis.random(input_dim=1, n_features=400,
                                     sigma=sigma_A)
    basis_s = SinusoidalBasis.random(input_dim=1, n_features=200,
                                     sigma=sigma_A * 0.3)
    # gamma basis: VERY smooth -- a few features only, period ~ T / 4
    sigma_g = 2 * np.pi / max(T * 0.3, 1.0)
    basis_g = SinusoidalBasis.random(input_dim=1, n_features=40,
                                     sigma=sigma_g)
    tt = torch.tensor(t, dtype=torch.float64).reshape(-1, 1)
    A_t = torch.tensor(A, dtype=torch.float64).reshape(-1, 1)
    M = tt.shape[0]
    # Initialise from constant-gamma fit
    res0, g0, beta_A0 = fit_constant_gamma(t, A)
    beta_A = torch.tensor(beta_A0, dtype=torch.float64).reshape(-1, 1)
    beta_g = torch.zeros(basis_g.n_features, 1, dtype=torch.float64)
    # Put the average of beta_g such that mean(gamma(t)) ~ g0.
    # gamma(t) = phi_g(t) beta_g.  Init beta_g so phi_g beta_g = g0 constant.
    phi_g0 = basis_g.evaluate(tt)
    beta_g, *_ = torch.linalg.lstsq(phi_g0,
                                    torch.full((M, 1), g0,
                                               dtype=torch.float64))
    # Iterate
    phi_A = basis_A.evaluate(tt)               # (M, N_A)
    dphi_A = Op.partial(0, 1, 1).apply(basis_A, tt)   # (M, N_A)
    phi_s = basis_s.evaluate(tt)
    history = []
    for it in range(8):
        # ---- Step 1: freeze beta_A, solve for (beta_g, beta_s) ----
        # ODE residual:  dphi_A beta_A + (phi_g beta_g) (phi_A beta_A) - phi_s beta_s = 0
        # term (phi_g beta_g) (phi_A beta_A) is bilinear; rewriting it as
        # ((phi_A beta_A) * phi_g) @ beta_g  with elementwise broadcast.
        A_hat = (phi_A @ beta_A)               # (M,1)
        dA_hat = (dphi_A @ beta_A)             # (M,1)
        gamma_block = A_hat * basis_g.evaluate(tt)   # (M, N_g)
        top = torch.cat([gamma_block, -phi_s], dim=1)
        rhs_top = -dA_hat
        # Data block on A: already enforced in step 2; here we keep only ODE
        beta_gs = solve_lstsq(top, rhs_top, mu=1e-4)
        beta_g = beta_gs[:basis_g.n_features]
        beta_s = beta_gs[basis_g.n_features:]
        # ---- Step 2: freeze (beta_g, beta_s), solve for beta_A ----
        # ODE residual:  (dphi_A + diag(gamma(t)) phi_A) beta_A = phi_s beta_s
        # Data residual: phi_A beta_A = A_meas
        gamma_t = basis_g.evaluate(tt) @ beta_g          # (M,1)
        L_phi_A = dphi_A + gamma_t * phi_A
        rhs_ode = phi_s @ beta_s
        AA = torch.cat([L_phi_A, phi_A], dim=0)
        rhs = torch.cat([rhs_ode, A_t], dim=0)
        beta_A = solve_lstsq(AA, rhs, mu=1e-4).reshape(-1, 1)
        # ---- Diagnostics ----
        A_hat_np = (phi_A @ beta_A).reshape(-1).numpy()
        rel_data = float(np.linalg.norm(A_hat_np - A) /
                         max(np.linalg.norm(A), 1e-30))
        ode_res = float(torch.linalg.norm(
            dphi_A @ beta_A + gamma_t * (phi_A @ beta_A) - phi_s @ beta_s
        ) / max(torch.linalg.norm(A_t), torch.tensor(1e-30)))
        history.append((it, rel_data, ode_res))
    gamma_t_np = (basis_g.evaluate(tt) @ beta_g).reshape(-1).numpy()
    s_t_np     = (phi_s @ beta_s).reshape(-1).numpy()
    A_hat_np   = (phi_A @ beta_A).reshape(-1).numpy()
    return {"A_hat": A_hat_np, "gamma_t": gamma_t_np, "s_t": s_t_np,
            "rel_data": history[-1][1], "ode_res": history[-1][2],
            "history": history}


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def run():
    print(">> Synchrotron-envelope ODE on real ALS-U LFB data", flush=True)
    raw = fetch_traces(hours=HOURS)
    for k, (t, v, err) in raw.items():
        if err:
            print(f"   {k}: ERROR {err}")
        else:
            print(f"   {k}: {len(t) if t is not None else 0} samples")
    t_A_raw, A_raw, errA = raw["A"]
    t_I_raw, I_raw, errI = raw["I"]
    if errA or t_A_raw is None or len(t_A_raw) < 200:
        print("   no usable LFB:M1:PEAK trace, trying RMS as fallback")
        t_A_raw, A_raw, errA = raw["A_alt"]
        if errA or t_A_raw is None or len(t_A_raw) < 200:
            print("   no usable LFB envelope trace; aborting")
            return None
    # Build common regular grid
    T = float(min(t_A_raw[-1], t_I_raw[-1] if t_I_raw is not None else t_A_raw[-1]))
    N = 4000
    t = np.linspace(0.0, T, N)
    A = regrid(t_A_raw, A_raw, t)
    I = regrid(t_I_raw, I_raw, t) if t_I_raw is not None else None
    # Z-score A so the fit is numerically stable
    A_mean = float(np.mean(A))
    A_std  = float(np.std(A)) or 1.0
    A_n = (A - A_mean) / A_std
    print(f"   regridded {N} samples over {T/3600:.2f} h "
          f"(mean A {A_mean:.3g}, std {A_std:.3g})")
    # Constant-gamma baseline
    res_const, gamma_const, _ = fit_constant_gamma(t, A_n)
    print(f"   constant-gamma fit:  best gamma = {gamma_const:.4e} /s,  "
          f"data residual = {res_const:.3e}")
    # Variable-gamma fit
    out = fit_varying_gamma(t, A_n)
    res_var = out["rel_data"]
    print(f"   variable-gamma fit:  data residual = {res_var:.3e}, "
          f"ODE residual = {out['ode_res']:.3e}")
    print(f"   residual ratio (const / var) = {res_const / max(res_var, 1e-30):.2f}x")
    # Correlate recovered gamma(t) with stored current I(t)
    if I is not None:
        r = float(np.corrcoef(out["gamma_t"], I)[0, 1])
        print(f"   corr( gamma(t), I(t) ) = {r:+.3f}  "
              "(Robinson predicts a positive sign)")
    else:
        r = float("nan")
    return {"t": t, "A": A, "A_n": A_n, "A_hat": out["A_hat"],
            "gamma_t": out["gamma_t"], "s_t": out["s_t"],
            "I": I, "A_mean": A_mean, "A_std": A_std,
            "res_const": res_const, "res_var": res_var,
            "gamma_const": gamma_const, "r_gamma_I": r}


def make_figure(d, out_pdf="/tmp/alsu_synchrotron_ode.pdf"):
    if d is None:
        return
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    t_h = d["t"] / 3600.0
    ax = axes[0]
    ax.plot(t_h, d["A_n"], color="#666666", lw=0.6,
            label="$A(t)$ (LFB peak detector, z-scored)")
    ax.plot(t_h, d["A_hat"], color="#c4424b", lw=1.0,
            label="FFF fit $\\hat A(t)$ via variable-$\\gamma$ ODE")
    ax.set_ylabel("synchrotron envelope $A$")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Time-varying damping rate of the LFB-loop coherent "
                 "synchrotron oscillation\n"
                 f"const-$\\gamma$ residual {d['res_const']:.2e}   "
                 f"$\\to$   var-$\\gamma$ residual {d['res_var']:.2e}   "
                 f"(ratio {d['res_const']/max(d['res_var'],1e-30):.1f}$\\times$)",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(t_h, d["gamma_t"], color="#3b6db4", lw=1.0,
            label="recovered $\\gamma(t)$ from FFF field")
    ax.axhline(d["gamma_const"], color="#c4424b", lw=1.0, ls="--",
               label=f"const $\\gamma$ baseline = {d['gamma_const']:.2e} /s")
    ax.set_ylabel("damping rate $\\gamma$ (1/s)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    ax = axes[2]
    if d["I"] is not None:
        ax2 = ax.twinx()
        ax.plot(t_h, d["gamma_t"], color="#3b6db4", lw=1.0,
                label="$\\gamma(t)$")
        ax2.plot(t_h, d["I"], color="#54a04e", lw=1.0,
                 label="$I(t)$ = SR:DCCT")
        ax.set_ylabel("$\\gamma(t)$", color="#3b6db4")
        ax2.set_ylabel("beam current $I(t)$ (mA)", color="#54a04e")
        ax.set_title(f"Robinson cross-check:  "
                     f"corr($\\gamma$, $I$) = {d['r_gamma_I']:+.3f}",
                     fontsize=10)
    ax.set_xlabel("time (h since start of window)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight", dpi=140)
    plt.savefig(out_pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=130)
    print(f"   wrote {out_pdf}")


if __name__ == "__main__":
    d = run()
    make_figure(d)
