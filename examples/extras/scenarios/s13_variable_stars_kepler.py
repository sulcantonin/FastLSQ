#!/usr/bin/env python
"""Scenario 13 -- Cepheid pulsation period from real Kepler photometry.

Target: V1154 Cyg (KIC 7548061), a classical Cepheid in the Kepler
field with a textbook ~4.93-day pulsation period (Derekas et al.
2012).  We auto-download the available long-cadence FITS files from
the public STScI Kepler archive, concatenate them, and recover the
pulsation period via the same Fourier-feature + Lomb-Scargle pipeline
as the rest of the cross-domain demos.

Requires: astropy (in code/requirements.txt).
"""
from __future__ import annotations

import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import download_url, fit_1d_fff, ls_periodogram_peaks, \
                    ls_power, spectral_summary

# V1154 Cyg = KIC 7548061; published Cepheid period 4.9255 d
KIC = "007548061"
P_KNOWN_D = 4.9255

# A few quarter-long FITS segments (each ~30 days).  STScI archive lists
# these explicitly; we just take the first 4 to keep the demo light.
FITS_FILES = [
    "kplr007548061-2009166043257_llc.fits",
    "kplr007548061-2009259160929_llc.fits",
    "kplr007548061-2009350155506_llc.fits",
    "kplr007548061-2010078095331_llc.fits",
]
BASE_URL = (f"https://archive.stsci.edu/missions/kepler/lightcurves/"
            f"{KIC[:4]}/{KIC}/")


def load_kepler():
    """Concatenate the chosen FITS files into one (BJD, PDC flux) array."""
    from astropy.io import fits
    bjd_all, flux_all = [], []
    for fn in FITS_FILES:
        cache = f"/tmp/kepler_{fn}"
        download_url(BASE_URL + fn, cache, timeout=60)
        with fits.open(cache) as hdul:
            tab = hdul[1].data
            t = np.asarray(tab["TIME"], dtype=float) + 2454833.0
            f = np.asarray(tab["PDCSAP_FLUX"], dtype=float)
            ok = np.isfinite(t) & np.isfinite(f)
            bjd_all.append(t[ok]); flux_all.append(f[ok])
    bjd = np.concatenate(bjd_all); flux = np.concatenate(flux_all)
    return bjd, flux


def main():
    print(f">> Scenario 13: Kepler Cepheid V1154 Cyg (KIC {KIC})\n",
          flush=True)
    bjd, flux = load_kepler()
    print(f"   loaded {len(flux)} long-cadence points, "
          f"span {bjd[-1] - bjd[0]:.1f} d")
    # Normalise flux and centre time.
    flux_n = flux / np.median(flux) - 1.0
    t = bjd - bjd[0]
    print(f"   modulation depth = {flux_n.std():.4f}\n")

    # FFF fit.  P_known ~ 5 d -> W = 2 pi / 5 = 1.26 rad/d.  sigma_W
    # spans up to 4*fundamental for harmonic content.
    sigma_W = 4 * (2 * np.pi / P_KNOWN_D)
    basis, beta, rmse, _ = fit_1d_fff(t, flux_n, sigma_W,
                                      n_features=1500, mu_reg=1e-10)
    print(f"   FFF fit RMSE = {rmse:.3e}\n")

    # LS periodogram, periods 1 -- 30 d
    W_min = 2 * np.pi / 30.0
    W_max = 2 * np.pi / 1.0
    peaks = ls_periodogram_peaks(t, flux_n, W_min, W_max,
                                 n_grid=4000, n_peaks=5,
                                 suppress_log_frac=0.02)
    print(f"   Top peaks:")
    print(f"     {'W (rad/d)':14s} {'period (d)':14s} {'tag':14s}")
    fund = None
    for W, _ in peaks:
        P = 2 * np.pi / W
        tag = ""
        if fund is None:
            fund = P; tag = "fundamental"
        elif abs(P - fund / 2) / (fund / 2) < 0.02:
            tag = "2nd harmonic"
        elif abs(P - fund / 3) / (fund / 3) < 0.02:
            tag = "3rd harmonic"
        print(f"     {W:<14.5f} {P:<14.6f} {tag}")
    rel = abs(fund - P_KNOWN_D) / P_KNOWN_D
    print(f"\n   Recovered fundamental: {fund:.6f} d  "
          f"(known {P_KNOWN_D} d, rel err {rel:.2e})")
    # Fourier amplitude ratios (Simon-Lee Cepheid diagnostic)
    W0 = 2 * np.pi / fund
    P1 = ls_power(t, flux_n, W0)
    P2 = ls_power(t, flux_n, 2 * W0)
    P3 = ls_power(t, flux_n, 3 * W0)
    R21 = float(np.sqrt(max(P2 / max(P1, 1e-30), 0.0)))
    R31 = float(np.sqrt(max(P3 / max(P1, 1e-30), 0.0)))
    print(f"   Amplitude ratios:  R21 = {R21:.3f},  R31 = {R31:.3f}  "
          f"(Simon-Lee Cepheid Fourier diagnostic)")
    s = spectral_summary(basis.W.numpy(), beta.numpy(), k_top=5)
    print(f"\n   Expansion: top-5 energy = {s['energy_K_top']:.3f},  "
          f"K_99 = {s['K_target']}/{s['N']}")


if __name__ == "__main__":
    main()
