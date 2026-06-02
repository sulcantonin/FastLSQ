#!/usr/bin/env python
"""Scenario 05 -- solar p-mode helioseismology.

Real GONG / SDO/HMI / SoHO data requires FITS parsing of large multi-
gigabyte time series.  We ship a synthetic 1-year disk-integrated
velocity trace built from a published solar p-mode frequency comb
(Lazrek et al. 1997 stable line list, n=20-24 at low l) plus
turbulent broadening noise.

If a real `/tmp/gong_velocity.npz` with keys (t_sec, v_m_s) is
present, it overrides the synthetic.
"""
from __future__ import annotations

import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import fit_1d_fff, ls_periodogram_peaks, spectral_summary

# Published p-mode frequencies (uHz) near the 5-min band, low-l (Lazrek 1997).
# Each entry is (n, l, nu_uHz).  We pick a few well-known modes.
MODES = [
    (20, 0, 2899.13), (21, 0, 3033.93), (22, 0, 3168.80),
    (20, 1, 2963.42), (21, 1, 3098.39),
    (19, 2, 2898.05),
]
LARGE_SEP_UHZ = 134.9                  # canonical solar Delta_nu

T_END_S   = 30 * 86400.0               # 30 days
DT_S      = 60.0                       # 1 minute sampling


def synth_pmodes(seed=0):
    rng = np.random.default_rng(seed)
    n = int(T_END_S / DT_S)
    t = np.arange(n) * DT_S
    v = np.zeros_like(t)
    for (nn, ll, nu_uHz) in MODES:
        omega = 2 * np.pi * nu_uHz * 1e-6
        amp = 0.2 + 0.1 * rng.random()      # m/s, near solar 5-min RMS
        phi = rng.uniform(0, 2 * np.pi)
        v += amp * np.sin(omega * t + phi)
    v += 0.1 * v.std() * rng.standard_normal(n)
    return t, v


def load():
    cache = "/tmp/gong_velocity.npz"
    if os.path.exists(cache):
        d = np.load(cache); return d["t_sec"], d["v_m_s"], "real GONG"
    t, v = synth_pmodes()
    return t, v, "synthetic (Lazrek 1997 frequency comb)"


def main():
    print(">> Scenario 05: solar p-mode helioseismology\n", flush=True)
    t, v, src = load()
    print(f"   source: {src}")
    print(f"   {len(t)} samples, span {t[-1]/86400:.1f} days, "
          f"dt = {(t[1]-t[0]):.1f} s\n")
    v_c = v - v.mean()

    # FFF fit at the 5-min band
    sigma_W = 2 * np.pi * 5e-3                 # 5 mHz scale
    basis, beta, rmse, _ = fit_1d_fff(t, v_c, sigma_W,
                                      n_features=2000, mu_reg=1e-8)
    print(f"   FFF fit RMSE = {rmse:.3e} m/s\n")

    # LS periodogram, 1-10 mHz band
    W_min = 2 * np.pi * 1e-3
    W_max = 2 * np.pi * 6e-3
    peaks = ls_periodogram_peaks(t, v_c, W_min, W_max,
                                 n_grid=8000, n_peaks=10,
                                 suppress_log_frac=0.005)
    print("   Top peaks:")
    print(f"     {'nu (uHz)':14s} {'match':30s} {'rel err':10s}")
    found = {}
    for W, _ in peaks:
        nu_uHz = W / (2 * np.pi) * 1e6
        match = ""; rel = ""
        for (nn, ll, nu_known) in MODES:
            key = f"(n={nn}, l={ll})"
            if key not in found and abs(nu_uHz - nu_known) < 2.0:
                match = f"n={nn} l={ll} (known {nu_known:.2f})"
                rel = f"{(nu_uHz - nu_known) / nu_known:+.2e}"
                found[key] = True
                break
        print(f"     {nu_uHz:<14.3f} {match:30s} {rel}")
    # Large-frequency separation Delta nu = nu_{n+1,l} - nu_{n,l}
    rec_nu_l0 = sorted([2 * np.pi / W * 0 + W / (2 * np.pi) * 1e6
                        for W, _ in peaks if 2800 < W/(2*np.pi)*1e6 < 3300])
    if len(rec_nu_l0) >= 2:
        delta_nu = np.median(np.diff(rec_nu_l0))
        print(f"\n   Large frequency separation Delta_nu = {delta_nu:.1f} uHz "
              f"(canonical {LARGE_SEP_UHZ} uHz)")
    print(f"\n   Modes recovered: {len(found)} / {len(MODES)}")
    s = spectral_summary(basis.W.numpy(), beta.numpy(), k_top=6)
    print(f"   Expansion: top-6 energy = {s['energy_K_top']:.3f}, "
          f"K_99 = {s['K_target']}/{s['N']}")


if __name__ == "__main__":
    main()
