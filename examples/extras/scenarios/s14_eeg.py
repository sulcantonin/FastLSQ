#!/usr/bin/env python
"""Scenario 14 -- EEG band decomposition on real PhysioNet data.

Auto-downloads one resting-state EDF recording from the PhysioNet
EEG Motor Movement/Imagery Database (S001R02, eyes-open baseline; no
API key needed), parses it with mne, and recovers the canonical alpha
rhythm (8-12 Hz) from an occipital channel via the FFF + LS
periodogram pipeline.

Requires: mne (in requirements.txt).
"""
from __future__ import annotations

import os, sys, warnings
import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import download_url, fit_1d_fff, ls_periodogram_peaks, \
                    ls_power, spectral_summary

URL = "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R02.edf"


def load_eeg(cache="/tmp/s001r02.edf"):
    download_url(URL, cache, timeout=60)
    import mne
    raw = mne.io.read_raw_edf(cache, preload=True, verbose="ERROR")
    # Pick an occipital channel (alpha rhythm is strongest there).
    # The EDF uses labels like "O1..", "Oz..", "O2.."
    ch_names = raw.ch_names
    occip = [c for c in ch_names if c.strip(". ").lower() in ("o1", "o2", "oz")]
    if not occip:
        occip = ch_names[:1]
    ch = occip[0]
    sfreq = raw.info["sfreq"]
    data = raw.get_data(picks=[ch])[0]
    return sfreq, data, ch


def main():
    print(">> Scenario 14: EEG alpha rhythm from PhysioNet\n", flush=True)
    sfreq, x, ch = load_eeg()
    t = np.arange(len(x)) / sfreq
    print(f"   channel {ch},  fs = {sfreq:.0f} Hz,  "
          f"duration = {t[-1]:.1f} s")
    print(f"   signal std = {x.std() * 1e6:.2f} uV\n")
    # Take a 10-second clip well into the recording
    i0 = int(10 * sfreq); i1 = int(20 * sfreq)
    t_clip = t[i0:i1] - t[i0]; x_clip = x[i0:i1]
    x_clip = x_clip - x_clip.mean()

    # FFF fit covering 1-50 Hz
    sigma_W = 2 * np.pi * 30.0
    basis, beta, rmse, _ = fit_1d_fff(t_clip, x_clip, sigma_W,
                                      n_features=1500, mu_reg=1e-12)
    print(f"   FFF fit RMSE = {rmse:.3e}\n")

    # LS periodogram, 1-50 Hz
    W_min = 2 * np.pi * 1.0
    W_max = 2 * np.pi * 50.0
    peaks = ls_periodogram_peaks(t_clip, x_clip, W_min, W_max,
                                 n_grid=6000, n_peaks=5,
                                 suppress_log_frac=0.1)
    print("   Top peaks:")
    print(f"     {'f (Hz)':12s} {'band':10s}")
    BANDS = [("delta", 0.5, 4), ("theta", 4, 8), ("alpha", 8, 12),
             ("beta", 12, 30), ("gamma", 30, 100)]
    for W, _ in peaks:
        f = W / (2 * np.pi)
        band = ""
        for name, lo, hi in BANDS:
            if lo <= f < hi:
                band = name; break
        print(f"     {f:<12.4f} {band}")
    # Band-power readout: aggregate LS power across each band
    print("\n   Band-power decomposition (sum of LS power inside each band):")
    grid = np.linspace(W_min, W_max, 2000)
    powers = np.array([ls_power(t_clip, x_clip, W) for W in grid])
    total = float(powers.sum())
    print(f"     {'band':10s} {'frac of total':14s}")
    for name, lo, hi in BANDS:
        mask = (grid >= 2 * np.pi * lo) & (grid < 2 * np.pi * hi)
        frac = float(powers[mask].sum() / max(total, 1e-30))
        print(f"     {name:10s} {frac:<14.3f}")
    s = spectral_summary(basis.W.numpy(), beta.numpy(), k_top=5)
    print(f"\n   Expansion: top-5 energy = {s['energy_K_top']:.3f}, "
          f"K_99 = {s['K_target']}/{s['N']}")


if __name__ == "__main__":
    main()
