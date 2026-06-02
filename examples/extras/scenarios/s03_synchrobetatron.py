#!/usr/bin/env python
"""Scenario 03 -- synchro-betatron sidebands: a modulated betatron
signal produces a triplet (carrier + two sidebands at Q_x +/- Q_s).
The top-3 peaks of the LS periodogram on the FFF fit *are* the
physical triplet.

Synthetic data (same justification as s01).
"""
from __future__ import annotations

import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import fit_1d_fff, ls_periodogram_peaks, spectral_summary

Q_X = 0.27
Q_S = 0.01
ALPHA = 0.15
N_TURNS = 8192


def synth(Q_x=Q_X, Q_s=Q_S, alpha=ALPHA, n_turns=N_TURNS, noise=0.005,
          seed=11):
    rng = np.random.default_rng(seed)
    n = np.arange(n_turns)
    x = (np.cos(2 * np.pi * Q_x * n + 0.5)
         * (1.0 + alpha * np.cos(2 * np.pi * Q_s * n + 0.1))
         + noise * rng.standard_normal(n_turns))
    return n.astype(float), x


def main():
    print(">> Scenario 03: synchro-betatron sidebands\n", flush=True)
    n, x = synth()
    print(f"   {len(x)} turns;  Q_x = {Q_X}, Q_s = {Q_S}, alpha = {ALPHA}\n")

    sigma_W = 2 * np.pi * 0.5
    basis, beta, rmse, _ = fit_1d_fff(n, x, sigma_W,
                                      n_features=2500, mu_reg=1e-12)
    print(f"   FFF fit RMSE = {rmse:.3e}\n")

    W_min = 2 * np.pi * 0.01
    W_max = 2 * np.pi * 0.49
    peaks = ls_periodogram_peaks(n, x, W_min, W_max,
                                 n_grid=10000, n_peaks=4,
                                 suppress_log_frac=0.003)
    print(f"   Top 4 spectral peaks (expect carrier + 2 sidebands):")
    print(f"     {'Q_frac':14s} {'tag':18s}")
    Qs = [W / (2 * np.pi) for W, _ in peaks]
    Qs.sort()
    for q in Qs:
        if abs(q - Q_X) < Q_S / 2:
            tag = "carrier Q_x"
        elif abs(q - (Q_X - Q_S)) < Q_S / 2:
            tag = "lower sideband Q_x - Q_s"
        elif abs(q - (Q_X + Q_S)) < Q_S / 2:
            tag = "upper sideband Q_x + Q_s"
        else:
            tag = ""
        print(f"     {q:<14.6f} {tag}")
    # Estimate Q_s from the triplet symmetry
    triplet = sorted([q for q in Qs if abs(q - Q_X) < 2 * Q_S])
    if len(triplet) >= 3:
        Q_s_est = (triplet[-1] - triplet[0]) / 2
        print(f"\n   Estimated Q_s from sideband spacing = {Q_s_est:.6f}  "
              f"(truth {Q_S})")
    s = spectral_summary(basis.W.numpy(), beta.numpy(), k_top=3)
    print(f"\n   Expansion: top-3 energy = {s['energy_K_top']:.3f},  "
          f"K_99 = {s['K_target']}/{s['N']}")


if __name__ == "__main__":
    main()
