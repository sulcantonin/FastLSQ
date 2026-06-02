#!/usr/bin/env python
"""Scenario 11 -- structural modal analysis on a synthetic cantilever
under ambient excitation.

The Euler-Bernoulli cantilever beam has eigenfrequencies given by
    omega_n = (beta_n L)^2 sqrt(E I / (rho A L^4))
with (beta_n L) the roots of cos(b)cosh(b) + 1 = 0:
    beta_1 L = 1.8751,   beta_2 L = 4.6941,
    beta_3 L = 7.8548,   beta_4 L = 10.9955.

For unit parameters, the first four eigenfrequencies (in rad/s) are
exactly (beta_n L)^2.  We simulate ambient white-noise excitation, then
recover the eigenfrequencies from the resulting acceleration trace.

If a real OMA dataset has been provided at /tmp/oma_accel.npz with
keys (t, acc), it overrides the synthetic source.  (Z24 Bridge requires
manual download; see code/plans/APIS_NEEDED.md.)
"""
from __future__ import annotations

import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import fit_1d_fff, ls_periodogram_peaks, spectral_summary


# Euler-Bernoulli cantilever clamped-free: roots of cos*cosh + 1 = 0
BETA_L = np.array([1.8751, 4.6941, 7.8548, 10.9955])
ZETA   = 0.02                                       # 2% modal damping
T_END  = 60.0                                       # 60 s of "ambient"
DT     = 1.0 / 1000.0                                # 1 kHz sampling
SEED   = 7


def synth_cantilever():
    """Ringing-down ambient response: sum of damped sinusoids at the
    eigenfrequencies plus measurement noise.  This is the canonical
    "operational modal analysis" signal shape and lets the LS
    periodogram pick out each mode."""
    rng = np.random.default_rng(SEED)
    omega_n = BETA_L ** 2                            # in rad/s
    t = np.arange(0.0, T_END, DT)
    acc = np.zeros_like(t)
    # Random impulse-train forcing: each impulse triggers a damped
    # ring-down of every mode.
    n_impulses = 80
    impulse_times = rng.uniform(0, T_END, size=n_impulses)
    for t_imp in impulse_times:
        mask = t >= t_imp
        dt_loc = t[mask] - t_imp
        for omega in omega_n:
            phase = rng.uniform(0, 2 * np.pi)
            acc[mask] += np.exp(-ZETA * omega * dt_loc) \
                         * np.cos(omega * dt_loc + phase)
    acc += 0.02 * acc.std() * rng.standard_normal(len(t))
    return t, acc


def load():
    cache = "/tmp/oma_accel.npz"
    if os.path.exists(cache):
        d = np.load(cache); return d["t"], d["acc"], "external"
    t, acc = synth_cantilever()
    return t, acc, "synthetic Euler-Bernoulli cantilever"


def main():
    print(">> Scenario 11: structural modal analysis\n", flush=True)
    t, acc, src = load()
    print(f"   source: {src}")
    print(f"   {len(t)} samples, span {t[-1]:.2f} s,  dt = {t[1]-t[0]:.4f} s\n")
    acc_c = acc - acc.mean()
    sigma_W = 2 * (BETA_L[-1] ** 2)                 # cover up to 4th mode
    basis, beta, rmse, _ = fit_1d_fff(t, acc_c, sigma_W,
                                      n_features=2500, mu_reg=1e-10)
    print(f"   FFF fit RMSE = {rmse:.3e}  (std = {acc_c.std():.3e})\n")

    # LS periodogram over the modal range (0.1 Hz to 50 Hz say)
    W_min = 2 * np.pi * 0.1
    W_max = 2 * np.pi * 40.0
    peaks = ls_periodogram_peaks(t, acc_c, W_min, W_max,
                                 n_grid=8000, n_peaks=6,
                                 suppress_log_frac=0.15)
    print("   Top peaks (sorted by LS power):")
    print(f"     {'W (rad/s)':12s} {'freq (Hz)':12s} "
          f"{'tag':12s} {'rel err':10s}")
    expected = BETA_L ** 2
    found = [False] * len(expected)
    for W, _ in peaks:
        f = W / (2 * np.pi)
        tag = ""; rel = ""
        for k, om in enumerate(expected):
            if not found[k] and abs(W - om) / om < 0.05:
                tag = f"mode {k+1}"; found[k] = True
                rel = f"{(W - om) / om:+.2e}"; break
        print(f"     {W:<12.4f} {f:<12.4f} {tag:12s} {rel}")
    n_match = sum(found)
    print(f"\n   Recovered modes: {n_match} of {len(expected)}  "
          f"(omega_n = (beta_n L)^2 for unit-parameter Euler-Bernoulli)")
    print(f"     truth: {expected}")
    s = spectral_summary(basis.W.numpy(), beta.numpy(), k_top=4)
    print(f"\n   Expansion: top-4 energy = {s['energy_K_top']:.3f},  "
          f"K_99 = {s['K_target']}/{s['N']}")


if __name__ == "__main__":
    main()
