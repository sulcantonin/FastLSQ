#!/usr/bin/env python
"""Scenario 02 -- plasma wakefield: recover the plasma wavelength
lambda_p from a single longitudinal Ez(xi) trace.

Synthetic data (no public PIC archive needed): we generate a wake of
the form
    Ez(xi) = E0 * exp(-(xi-xi_drv)^2/sigma^2) * sin(kp xi + phi)
          + nonlinear correction at 2 kp for "blow-out" demonstration.

The FFF fit recovers kp from the dominant spectral peak, and the
2nd-harmonic amplitude diagnoses the regime (linear vs nonlinear).

If a WarpX-generated dataset is provided as
`/tmp/warpx_lwfa_Ez.npz` with keys (xi, Ez), it overrides the
synthetic source.
"""
from __future__ import annotations

import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import fit_1d_fff, ls_periodogram_peaks, \
                    ls_power, spectral_summary


def synth_wake(lambda_p_um=20.0, length_um=200.0, n_pts=1000,
               a0=0.5, seed=0):
    """Synthetic LWFA-like wake in vacuum-scaled units."""
    rng = np.random.default_rng(seed)
    xi = np.linspace(-length_um, 0.0, n_pts)        # behind the driver
    kp = 2 * np.pi / lambda_p_um
    # Envelope: Gaussian drive at xi=0, decaying behind.
    env = np.exp(-(xi + 30.0) ** 2 / (40.0 ** 2))
    # Fundamental + 2nd-harmonic with amplitude scaling like a0^2
    Ez = env * np.sin(kp * xi) + 0.4 * a0 ** 2 * env * np.sin(2 * kp * xi)
    Ez += rng.standard_normal(n_pts) * 0.01
    return xi, Ez


def load_data():
    cache = "/tmp/warpx_lwfa_Ez.npz"
    if os.path.exists(cache):
        d = np.load(cache)
        return d["xi"], d["Ez"], "WarpX"
    xi, Ez = synth_wake()
    return xi, Ez, "synthetic"


def main():
    print(">> Scenario 02: plasma wakefield Ez(xi) -> lambda_p\n",
          flush=True)
    xi, Ez, src = load_data()
    print(f"   source: {src}, {len(xi)} points, "
          f"xi range = ({xi[0]:.1f}, {xi[-1]:.1f}) um\n")

    # FFF fit.  lambda_p ~ 20 um expected, so kp ~ 0.31 rad/um.  sigma_W
    # spans up to 4*kp to capture harmonics.
    sigma_W = 4 * (2 * np.pi / 20.0)
    basis, beta, rmse, _ = fit_1d_fff(xi, Ez, sigma_W,
                                      n_features=1000, mu_reg=1e-10)
    print(f"   FFF fit  RMSE = {rmse:.3e}  (|Ez|_max = {np.max(np.abs(Ez)):.3f})")

    # LS periodogram: search for lambda_p in [5, 60] um
    k_min = 2 * np.pi / 60.0
    k_max = 2 * np.pi / 5.0
    peaks = ls_periodogram_peaks(xi, Ez, k_min, k_max,
                                 n_grid=4000, n_peaks=5,
                                 suppress_log_frac=0.05)
    print(f"\n   Top spectral peaks:")
    print(f"     {'kp (rad/um)':14s} {'lambda_p (um)':15s} {'tag':10s}")
    lambda_p_recovered = None
    for k, _ in peaks:
        lp = 2 * np.pi / k
        tag = ""
        if lambda_p_recovered is None:
            lambda_p_recovered = lp
            tag = "fundamental"
        elif abs(lp - lambda_p_recovered / 2) / (lambda_p_recovered / 2) < 0.05:
            tag = "2nd harmonic"
        print(f"     {k:<14.5f} {lp:<15.4f} {tag}")
    # 2nd-harmonic amplitude ratio
    kp = 2 * np.pi / lambda_p_recovered
    P1 = ls_power(xi, Ez, kp)
    P2 = ls_power(xi, Ez, 2 * kp)
    amp_ratio = float(np.sqrt(max(P2 / max(P1, 1e-30), 0.0)))
    print(f"\n   Recovered lambda_p = {lambda_p_recovered:.4f} um")
    print(f"   2nd-harmonic amplitude ratio A2/A1 = {amp_ratio:.4f}")
    print(f"     (linear regime: < 0.05;  weak nonlinear: 0.05-0.2;  "
          f"strong: > 0.2)")
    s = spectral_summary(basis.W.numpy(), beta.numpy(), k_top=3)
    print(f"\n   Expansion: top-3 energy = {s['energy_K_top']:.3f},  "
          f"K_99 = {s['K_target']}/{s['N']}")
    return {"lambda_p": lambda_p_recovered, "A2/A1": amp_ratio}


if __name__ == "__main__":
    main()
