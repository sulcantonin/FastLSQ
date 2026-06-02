#!/usr/bin/env python
"""Scenario 09 -- ENSO (Nino 3.4 SST) and QBO (Singapore 30 hPa zonal
wind).  Two parallel real-data fits showing two different "characters":
ENSO is broad-band, QBO is a relatively sharp ~28-month peak.

Sources:
   ENSO Nino 3.4:  NOAA PSL,
       https://psl.noaa.gov/data/correlation/nina34.data
   QBO:            Free University Berlin,
       https://www.geo.fu-berlin.de/met/ag/strat/produkte/qbo/singapore.dat

Both are plain text, no API key.
"""
from __future__ import annotations

import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import download_url, fit_1d_fff, \
                    ls_periodogram_peaks, spectral_summary

NINO_URL = "https://psl.noaa.gov/data/correlation/nina34.data"
QBO_URL  = "https://www.geo.fu-berlin.de/met/ag/strat/produkte/qbo/singapore.dat"


def load_nino(cache="/tmp/nina34.dat"):
    download_url(NINO_URL, cache, timeout=30)
    with open(cache) as f:
        lines = f.readlines()
    header = lines[0].split()
    y0 = int(header[0])
    t_yr, vals = [], []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) != 13: continue
        try:
            yr = int(parts[0])
            for m in range(12):
                v = float(parts[1 + m])
                if v <= -90: continue
                t_yr.append(yr + (m + 0.5) / 12.0)
                vals.append(v)
        except ValueError:
            continue
    return np.array(t_yr), np.array(vals)


def load_qbo(cache="/tmp/qbo_singapore.dat", level=30):
    """Parse Singapore monthly mean zonal wind at the chosen pressure
    level (hPa).  Returns (decimal year, wind in 0.1 m/s)."""
    download_url(QBO_URL, cache, timeout=30)
    t_yr, vals = [], []
    cur_year = None
    with open(cache) as f:
        for ln in f:
            s = ln.strip()
            if not s: continue
            # Year header line: a 4-digit number alone
            if len(s) == 4 and s.isdigit():
                cur_year = int(s); continue
            parts = s.split()
            if len(parts) < 13 or not parts[0].isdigit() or cur_year is None:
                continue
            try:
                plev = int(parts[0])
            except ValueError:
                continue
            if plev != level: continue
            for m in range(12):
                try:
                    v = float(parts[1 + m])
                except (ValueError, IndexError):
                    continue
                if abs(v) > 9000: continue
                t_yr.append(cur_year + (m + 0.5) / 12.0)
                vals.append(v)
    return np.array(t_yr), np.array(vals)


def analyse(name, t, y, T_target_yr, label_tag, sigma_yr,
            search_min_yr, search_max_yr, NAMED):
    print(f"\n--- {name}")
    print(f"   {len(y)} monthly observations, "
          f"{t[0]:.2f} -> {t[-1]:.2f}")
    y_c = y - y.mean()
    sigma_W = 2 * np.pi / sigma_yr
    basis, beta, rmse, _ = fit_1d_fff(t - t.mean(), y_c, sigma_W,
                                      n_features=800, mu_reg=1e-6)
    print(f"   FFF fit RMSE = {rmse:.3f}  (std = {y_c.std():.3f})")
    W_min = 2 * np.pi / search_max_yr
    W_max = 2 * np.pi / search_min_yr
    peaks = ls_periodogram_peaks(t - t.mean(), y_c, W_min, W_max,
                                 n_grid=4000, n_peaks=5,
                                 suppress_log_frac=0.05)
    print(f"   Top peaks:")
    print(f"     {'W (rad/yr)':12s} {'period (yr)':14s} {'tag':10s}")
    found = {n: False for n, _, _ in NAMED}
    for W, _ in peaks:
        period = 2 * np.pi / W
        tag = ""
        for nm, T0, tol in NAMED:
            if not found[nm] and abs(period - T0) < tol:
                tag = nm; found[nm] = True; break
        print(f"     {W:<12.4f} {period:<14.4f} {tag}")
    s = spectral_summary(basis.W.numpy(), beta.numpy(), k_top=3)
    print(f"   Top-3 energy = {s['energy_K_top']:.3f}, "
          f"K_99 = {s['K_target']}/{s['N']}  "
          f"(low compressibility = broad-band signal)")
    return found


def main():
    print(">> Scenario 09: ENSO and QBO real-data spectra\n", flush=True)
    # ENSO Nino 3.4: ~3-7 yr broad band
    t_e, y_e = load_nino()
    NAMED_ENSO = [("ENSO band", 4.0, 2.0)]
    analyse("Nino 3.4 SST", t_e, y_e, 4.0, "ENSO",
            sigma_yr=4.0, search_min_yr=1.0, search_max_yr=15.0,
            NAMED=NAMED_ENSO)
    # QBO: ~28-month sharp peak
    t_q, y_q = load_qbo(level=30)
    NAMED_QBO = [("QBO ~28 mo", 28.0/12.0, 6.0/12.0)]
    analyse("QBO at 30 hPa (Singapore)", t_q, y_q, 28.0/12.0, "QBO",
            sigma_yr=2.0, search_min_yr=0.5, search_max_yr=10.0,
            NAMED=NAMED_QBO)


if __name__ == "__main__":
    main()
