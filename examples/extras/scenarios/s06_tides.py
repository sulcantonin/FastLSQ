#!/usr/bin/env python
"""Scenario 06 -- recover the named astronomical tide constituents
(M2, S2, N2, K1, O1, ...) from a real NOAA tide-gauge record.

Auto-downloads hourly water level for a chosen NOAA station via the
NOAA Tides & Currents REST API (no key).  Fits a Fast Fourier
Features basis to the deseasoned water level, runs a Lomb-Scargle
periodogram, and matches the recovered peaks to the published
astronomical periods.

The astronomical periods are *exact* (derived from celestial
mechanics), so this is the cleanest possible recovery test.

Usage:  python s06_tides.py
"""
from __future__ import annotations

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import download_url, fit_1d_fff, \
                    ls_periodogram_peaks, spectral_summary

STATION = "9414290"          # San Francisco Presidio (long record)
YEARS   = ("2021", "2022")   # NOAA hourly_height caps at 1 yr per request

def url_for_year(yr):
    return ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
            "?product=hourly_height&application=fastlsq"
            f"&begin_date={yr}0101&end_date={yr}1231"
            f"&datum=MLLW&station={STATION}"
            "&time_zone=GMT&units=metric&format=csv")

# Named astronomical tide constituents.  Periods in hours.
CONSTITUENTS = [
    ("M2", 12.4206012),       # principal lunar semidiurnal
    ("S2", 12.0000000),       # principal solar semidiurnal
    ("N2", 12.6583480),       # larger elliptic lunar
    ("K2", 11.9672348),       # luni-solar semidiurnal
    ("K1", 23.9344696),       # luni-solar diurnal
    ("O1", 25.8193387),       # principal lunar diurnal
    ("P1", 24.0658876),       # principal solar diurnal
    ("Q1", 26.8683567),       # larger lunar elliptic diurnal
    ("Mf", 327.85906),        # lunar fortnightly
    ("Mm", 661.31111),        # lunar monthly
]


def load_tides():
    """Concatenate per-year requests because NOAA's hourly_height
    endpoint caps each call at one calendar year."""
    chunks = []
    for yr in YEARS:
        cache = f"/tmp/noaa_tides_{STATION}_{yr}.csv"
        download_url(url_for_year(yr), cache, timeout=60)
        raw = np.genfromtxt(cache, delimiter=",", skip_header=1,
                            usecols=(1,), invalid_raise=False)
        raw = raw[~np.isnan(raw)]
        if len(raw) < 100:
            raise RuntimeError(f"NOAA returned {len(raw)} rows for {yr}; "
                               "likely API change or station outage")
        chunks.append(raw)
    arr = np.concatenate(chunks)
    t = np.arange(len(arr), dtype=float)
    return t, arr


def main():
    print(f">> Scenario 06: NOAA tide-gauge {STATION}, "
          f"{YEARS[0]}-{YEARS[-1]}\n", flush=True)
    t, h = load_tides()
    print(f"   loaded {len(h)} hourly readings  "
          f"(span = {len(h) / 24:.1f} days)")
    # Detrend (mean offset is huge relative to tidal amplitude)
    h_centred = h - h.mean()
    print(f"   mean water level = {h.mean():.3f} m, "
          f"residual std = {h_centred.std():.3f} m\n")

    # FFF basis: t is in hours, periods of interest 11.9 -- 700 hours
    # so W ranges from 2 pi / 700 ~ 0.009 to 2 pi / 11.9 ~ 0.53 rad/hr.
    sigma_W = 0.6                 # covers the semidiurnal band with margin
    basis, beta, rmse, _ = fit_1d_fff(t, h_centred, sigma_W,
                                      n_features=2000, mu_reg=1e-10)
    print(f"   FFF fit  RMSE = {rmse:.3e} m\n")

    # LS periodogram from period 8 h (faster than M2's harmonics) to
    # period 700 h (~ 29 days, captures Mm).
    W_max = 2 * np.pi / 8.0
    W_min = 2 * np.pi / 700.0
    peaks = ls_periodogram_peaks(t, h_centred, W_min, W_max,
                                 n_grid=8000, n_peaks=12,
                                 suppress_log_frac=0.01)
    print("   Top peaks (sorted by LS power):")
    print(f"     {'W (rad/h)':12s} {'period (h)':14s} {'tag':10s} "
          f"{'rel err':10s}")
    matched = {n: False for n, _ in CONSTITUENTS}
    for W, _ in peaks:
        period = 2 * np.pi / W
        tag = ""; rel = ""
        for name, T0 in CONSTITUENTS:
            if not matched[name] and abs(period - T0) / T0 < 0.005:
                tag = name; matched[name] = True
                rel = f"{(period - T0) / T0:+.2e}"
                break
        print(f"     {W:<12.5f} {period:<14.5f} {tag:10s} {rel}")
    n_match = sum(matched.values())
    print(f"\n   Constituents recovered: {n_match} / {len(CONSTITUENTS)}")
    for name, T0 in CONSTITUENTS:
        status = "yes" if matched[name] else "no"
        print(f"     {name:5s} T_known = {T0:9.4f} h    {status}")
    # Spectral expansion summary
    s = spectral_summary(basis.W.numpy(), beta.numpy(), k_top=7)
    print(f"\n   Expansion: top-7 energy = {s['energy_K_top']:.3f},  "
          f"K_99 = {s['K_target']}/{s['N']}")
    return {"n_matched": n_match, "rmse": rmse}


if __name__ == "__main__":
    main()
