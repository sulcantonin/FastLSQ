#!/usr/bin/env python
"""Scenario 07 -- IERS Earth-rotation parameters.

Auto-downloads the standard IERS finals2000A.all file (no API key)
and recovers the Chandler wobble (~432 d), annual wobble (365.25 d),
and 18.6-yr lunar nutation from the polar-motion x_p time series.

Usage:  python s07_iers_earth_rotation.py
"""
from __future__ import annotations

import os, sys, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import download_url, fit_1d_fff, \
                    ls_periodogram_peaks, spectral_summary

URL = "https://datacenter.iers.org/data/9/finals2000A.all"


def load_iers(cache="/tmp/iers_finals.txt"):
    """Parse IERS finals2000A.  Returns (MJD, x_p_arcsec, y_p_arcsec)
    for rows with observed (I) polar motion only."""
    download_url(URL, cache, timeout=60)
    mjd_list, xp_list, yp_list = [], [], []
    float_pat = re.compile(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?")
    with open(cache) as f:
        for line in f:
            if len(line) < 60: continue
            flag_pos = line[16:17].strip()
            if flag_pos != "I":          # require observed polar motion
                continue
            nums = float_pat.findall(line[:80])
            if len(nums) < 5: continue
            try:
                mjd = float(nums[0])
                xp = float(nums[1])
                yp = float(nums[3])
            except (ValueError, IndexError):
                continue
            mjd_list.append(mjd); xp_list.append(xp); yp_list.append(yp)
    return (np.array(mjd_list), np.array(xp_list), np.array(yp_list))


def main():
    print(">> Scenario 07: IERS Earth rotation\n", flush=True)
    mjd, xp, yp = load_iers()
    print(f"   loaded {len(mjd)} daily observed records")
    print(f"   MJD range {mjd[0]:.1f} -> {mjd[-1]:.1f}  "
          f"(~{(mjd[-1] - mjd[0]) / 365.25:.1f} years)")
    # Use polar-motion x in arcseconds; convert MJD to days from start.
    t = mjd - mjd[0]
    y = xp - xp.mean()
    print(f"   x_p std = {y.std() * 1000:.2f} mas\n")

    # FFF basis: cover periods 100 d to 10000 d.  sigma_W matched
    # to the Chandler band.
    sigma_W = 2 * np.pi / 432.0           # centred at Chandler
    basis, beta, rmse, _ = fit_1d_fff(t, y, sigma_W,
                                      n_features=1500, mu_reg=1e-8)
    print(f"   FFF fit  RMSE = {rmse * 1000:.2f} mas\n")

    # LS periodogram, periods 100 -- 10000 days
    W_min = 2 * np.pi / 10000.0
    W_max = 2 * np.pi / 100.0
    peaks = ls_periodogram_peaks(t, y, W_min, W_max,
                                 n_grid=4000, n_peaks=8,
                                 suppress_log_frac=0.04)
    print("   Top peaks (sorted by LS power):")
    print(f"     {'W (rad/d)':12s} {'period (d)':14s} {'period (yr)':14s} "
          f"{'tag':14s}")
    NAMED = [("annual",   365.25,   15),
             ("Chandler", 432.0,    15),
             ("18.6yr",   6798.0,  500)]
    found = {n: False for n, _, _ in NAMED}
    for W, _ in peaks:
        period_d = 2 * np.pi / W
        period_y = period_d / 365.25
        tag = ""
        for name, T0, tol in NAMED:
            if not found[name] and abs(period_d - T0) < tol:
                tag = name; found[name] = True; break
        print(f"     {W:<12.6f} {period_d:<14.2f} {period_y:<14.2f} {tag}")
    print("\n   Recovered modes:")
    for name, T0, _ in NAMED:
        st = "yes" if found[name] else "no"
        print(f"     {name:10s}  expected period {T0:>8.1f} d   {st}")
    s = spectral_summary(basis.W.numpy(), beta.numpy(), k_top=5)
    print(f"\n   Expansion: top-5 energy = {s['energy_K_top']:.3f},  "
          f"K_99 = {s['K_target']}/{s['N']}")
    return {"found": found, "rmse_mas": rmse * 1000}


if __name__ == "__main__":
    main()
