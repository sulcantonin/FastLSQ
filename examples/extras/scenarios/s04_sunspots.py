#!/usr/bin/env python
"""Scenario 04 -- recover Schwabe / Hale / Gleissberg cycles from the
SILSO daily total sunspot record (1818 -- today).

Auto-downloads the daily file from the Royal Observatory of Belgium
(no API key), fits a Fast Fourier Features basis, and runs a Lomb-
Scargle periodogram on the fit to recover the named solar cycles.

Expected dominant peaks (in years):
    Schwabe     ~11
    Hale        ~22  (weak in scalar count data)
    Gleissberg  ~80-90
    Suess       ~200 (at the edge of detectability in a 300-yr record)

Usage:  python s04_sunspots.py
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import (download_url, fit_1d_fff,
                     ls_periodogram_peaks, spectral_summary)

SILSO_URL = "https://www.sidc.be/SILSO/INFO/sndtotcsv.php"


def load_silso(cache="/tmp/silso_dtot.csv"):
    """Parse SILSO daily-total CSV.  Columns: year, month, day,
    fractional year, daily total SN (-1 missing), sigma, N_obs,
    flag."""
    download_url(SILSO_URL, cache, timeout=30)
    arr = np.genfromtxt(cache, delimiter=";")
    yr_frac = arr[:, 3]
    sn = arr[:, 4]
    mask = sn >= 0
    return yr_frac[mask], sn[mask]


def main():
    print(">> Scenario 04: SILSO sunspot cycles (1818-today)\n", flush=True)
    t_yr, sn = load_silso()
    print(f"   loaded {len(sn)} daily observations, "
          f"{t_yr[0]:.2f} -> {t_yr[-1]:.2f}")
    # 27-day rolling mean to suppress solar-rotation modulation, then
    # variance-stabilising sqrt transform.
    win = 27
    sn_smooth = np.convolve(sn, np.ones(win) / win, mode="same")
    y = np.sqrt(np.maximum(sn_smooth, 0.0))
    # Coarsen to monthly to keep the linear system manageable.
    keep = np.arange(0, len(y), 30)
    t = t_yr[keep]; y = y[keep]
    print(f"   coarsened to {len(y)} monthly samples\n")

    # Work in raw years.  Detrend to remove the long-term mean + any
    # multi-century secular drift -- the LS power otherwise collapses
    # onto the lowest frequency in the search window.
    t_yr_centred = t - t.mean()
    A = np.stack([np.ones_like(t_yr_centred), t_yr_centred,
                  t_yr_centred ** 2], axis=1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_detr = y - A @ c
    print(f"   detrended (degree-2 polynomial removed), "
          f"std before/after: {y.std():.3f} -> {y_detr.std():.3f}\n")

    # FFF fit in years.  sigma_W = 2 pi / 11 covers Schwabe + harmonics.
    sigma_W = 2 * np.pi / 11.0
    basis, beta, rmse, _ = fit_1d_fff(t_yr_centred, y_detr, sigma_W,
                                      n_features=1500, mu_reg=1e-6)
    print(f"   FFF fit  RMSE = {rmse:.3e}  "
          f"(signal std = {y_detr.std():.3e})\n")

    # LS periodogram for periods 5--500 years (W in rad/yr).
    W_min = 2 * np.pi / 500.0
    W_max = 2 * np.pi / 5.0
    peaks = ls_periodogram_peaks(t_yr_centred, y_detr, W_min, W_max,
                                 n_grid=4000, n_peaks=8,
                                 suppress_log_frac=0.08)
    print("   Top spectral peaks (sorted by LS power):")
    print(f"     {'W (rad/yr)':12s} {'period (yr)':14s} {'tag':14s}")
    NAMED = [("Schwabe", 11.0, 1.5),
             ("Hale",    22.0, 3.0),
             ("Gleissberg", 85.0, 25.0),
             ("Suess",   200.0, 50.0)]
    found = {n: False for n, _, _ in NAMED}
    for W, _ in peaks:
        period = 2 * np.pi / W
        tag = ""
        for name, T0, tol in NAMED:
            if abs(period - T0) < tol and not found[name]:
                tag = name; found[name] = True; break
        print(f"     {W:<12.4f} {period:<14.2f} {tag}")
    # Symbolic head report
    print("\n   Recovered solar cycles:")
    for name, T0, tol in NAMED:
        status = "yes" if found[name] else "no"
        print(f"     {name:12s}  expected ~ {T0:>5.1f} yr     "
              f"recovered: {status}")
    # Spectral summary of the expansion
    s = spectral_summary(basis.W.numpy(), beta.numpy(), k_top=5)
    bw_yr = 2 * np.pi / s["bandwidth"]
    print(f"\n   Expansion: top-5 energy fraction = {s['energy_K_top']:.3f},  "
          f"K_99 = {s['K_target']} / {s['N']},  "
          f"dominant period 1/bar W -> {bw_yr:.1f} yr")
    return {
        "n_obs": len(y), "rmse": rmse, "found": found,
        "bandwidth_yr": bw_yr, "K_99": s["K_target"], "N": s["N"]
    }


if __name__ == "__main__":
    main()
