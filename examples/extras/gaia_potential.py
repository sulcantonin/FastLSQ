#!/usr/bin/env python
"""Discover a galactic potential from stellar accelerations (Gaia-style).

The setup mirrors Loop C of the NeSy paper but on a different forward map.
A galactic potential Phi(r) generates accelerations  a = -grad Phi(r).
Given a sample of stars at positions r_i with measured accelerations a_i
(plus noise), we ask: what symbolic functional form has Phi?

Grammar of potential primitives:
    Plummer(M, a)       :  Phi = -M / sqrt(r^2 + a^2)
    Hernquist(M, a)     :  Phi = -M / (r + a)
    NFW(M, a)           :  Phi = -M log(1 + r/a) / r
    Logarithmic(v, a)   :  Phi = 0.5 v^2 log(r^2 + a^2)
    Sum(P1, P2)         :  composition

We generate synthetic Gaia-like data from a known potential, then enumerate
the grammar, fit each primitive's parameters by Adam, and select by BIC.

Usage:  python gaia_potential.py
"""
from __future__ import annotations

import time
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
DEV = "cpu"


# ----------------------------------------------------------------------
# Spherical potentials + their gradients (returns radial acceleration)
# ----------------------------------------------------------------------

def plummer_acc(r, params):
    """params = [log_M, log_a]. Returns -grad Phi = -M r / (r^2 + a^2)^{3/2}."""
    M = torch.exp(params[0]); a = torch.exp(params[1])
    return -M * r / (r ** 2 + a ** 2) ** 1.5


def hernquist_acc(r, params):
    """params = [log_M, log_a]. -grad Phi = -M / (r + a)^2."""
    M = torch.exp(params[0]); a = torch.exp(params[1])
    return -M / (r + a) ** 2


def nfw_acc(r, params):
    """params = [log_M, log_a]. -grad Phi for an NFW-like profile
    Phi = -M ln(1 + r/a) / r, hence
    -grad Phi = -M [ 1/(r(r+a)) - ln(1+r/a)/r^2 ].
    """
    M = torch.exp(params[0]); a = torch.exp(params[1])
    return -M * (1.0 / (r * (r + a)) - torch.log(1 + r / a) / r ** 2)


def logarithmic_acc(r, params):
    """params = [log_v, log_a]. Phi = 0.5 v^2 ln(r^2 + a^2);
    -grad Phi = -v^2 r / (r^2 + a^2)."""
    v = torch.exp(params[0]); a = torch.exp(params[1])
    return -(v ** 2) * r / (r ** 2 + a ** 2)


PRIMS = [
    ("Plummer",  plummer_acc,  2, lambda rng: torch.tensor(
        [np.log(rng.uniform(0.3, 3.0)), np.log(rng.uniform(2.0, 10.0))], requires_grad=True)),
    ("Hernquist", hernquist_acc, 2, lambda rng: torch.tensor(
        [np.log(rng.uniform(0.3, 3.0)), np.log(rng.uniform(2.0, 15.0))], requires_grad=True)),
    ("NFW",       nfw_acc,       2, lambda rng: torch.tensor(
        [np.log(rng.uniform(1.0, 20.0)), np.log(rng.uniform(8.0, 40.0))], requires_grad=True)),
    ("LogPot",    logarithmic_acc, 2, lambda rng: torch.tensor(
        [np.log(rng.uniform(0.5, 2.0)), np.log(rng.uniform(2.0, 12.0))], requires_grad=True)),
]


# ----------------------------------------------------------------------
# Synthetic Gaia-like dataset
# ----------------------------------------------------------------------

def make_gaia_like_data(true_potential, true_params, n_stars=2000, noise_rel=0.05, seed=0):
    """Generate stars at random radii in [0.5, 30] kpc; compute a_true(r).
    Add noise_rel relative Gaussian noise to each measured acceleration.
    Returns (r, a_noisy, a_clean).
    """
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.5, 30.0, size=n_stars)
    r_t = torch.tensor(r)
    p_t = torch.tensor(np.asarray(true_params))
    with torch.no_grad():
        a_clean = true_potential(r_t, p_t).numpy()
    sig = noise_rel * np.std(a_clean)
    a_noisy = a_clean + rng.normal(0, sig, size=a_clean.shape)
    return r, a_noisy, a_clean


# ----------------------------------------------------------------------
# Fit a single primitive (or a sum of two) by Adam
# ----------------------------------------------------------------------

def fit_primitive(prim_idx, r, a_obs, n_epochs=600, lr=0.03, seed=0):
    rng = np.random.default_rng(seed * 9 + prim_idx)
    name, fn, n_par, init_fn = PRIMS[prim_idx]
    params = init_fn(rng)
    opt = torch.optim.Adam([params], lr=lr)
    r_t = torch.tensor(r); a_t = torch.tensor(a_obs)
    for ep in range(n_epochs):
        opt.zero_grad()
        loss = ((fn(r_t, params) - a_t) ** 2).mean()
        loss.backward(); opt.step()
    return params.detach(), float(loss.detach()), n_par


def fit_sum_of_two(idx_a, idx_b, r, a_obs, n_epochs=800, lr=0.03, seed=0):
    rng_a = np.random.default_rng(seed * 7 + idx_a)
    rng_b = np.random.default_rng(seed * 11 + idx_b + 1)
    pa = PRIMS[idx_a][3](rng_a)
    pb = PRIMS[idx_b][3](rng_b)
    fn_a, fn_b = PRIMS[idx_a][1], PRIMS[idx_b][1]
    n_par = PRIMS[idx_a][2] + PRIMS[idx_b][2]
    opt = torch.optim.Adam([pa, pb], lr=lr)
    r_t = torch.tensor(r); a_t = torch.tensor(a_obs)
    for ep in range(n_epochs):
        opt.zero_grad()
        pred = fn_a(r_t, pa) + fn_b(r_t, pb)
        loss = ((pred - a_t) ** 2).mean()
        loss.backward(); opt.step()
    return (pa.detach(), pb.detach()), float(loss.detach()), n_par


def bic(loss, n_obs, n_par):
    return n_obs * np.log(max(loss, 1e-30)) + n_par * np.log(max(n_obs, 1))


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main():
    print(">> Galactic-potential discovery from stellar accelerations")
    print(f"   K = {len(PRIMS)} primitive potentials\n", flush=True)

    SCENARIOS = [
        # (name, true_function, true_params_tensor)
        ("Plummer (M=1, a=5)",         plummer_acc,     [np.log(1.0), np.log(5.0)]),
        ("Hernquist (M=2, a=8)",       hernquist_acc,   [np.log(2.0), np.log(8.0)]),
        ("NFW (M=8, a=20)",            nfw_acc,         [np.log(8.0), np.log(20.0)]),
        ("LogPot (v=1, a=5)",          logarithmic_acc, [np.log(1.0), np.log(5.0)]),
    ]

    rows = []
    for sc_name, true_fn, true_p in SCENARIOS:
        r, a_obs, a_clean = make_gaia_like_data(true_fn, true_p, n_stars=2000,
                                                noise_rel=0.05, seed=0)
        t0 = time.perf_counter()
        # Singletons
        scored = []
        for p_idx, (p_name, _, _, _) in enumerate(PRIMS):
            params, loss, n_par = fit_primitive(p_idx, r, a_obs)
            scored.append((p_name, loss, n_par, bic(loss, len(r), n_par), params))
        # Sums (small set of canonical bulge+halo combos)
        canonical_sums = [(0, 2), (1, 2), (0, 3), (1, 3)]
        for ia, ib in canonical_sums:
            n_ab = f"{PRIMS[ia][0]}+{PRIMS[ib][0]}"
            (pa, pb), loss, n_par = fit_sum_of_two(ia, ib, r, a_obs)
            scored.append((n_ab, loss, n_par, bic(loss, len(r), n_par), (pa, pb)))
        scored.sort(key=lambda x: x[3])  # by BIC
        elapsed = time.perf_counter() - t0

        winner = scored[0][0]
        winner_loss = scored[0][1]
        print(f"--- {sc_name}")
        for s in scored[:5]:
            print(f"     {s[0]:25s} loss={s[1]:.3e} BIC={s[3]:.1f}", flush=True)
        print(f"   winner: {winner}   ({elapsed:.1f}s)\n", flush=True)
        rows.append({"truth": sc_name, "winner": winner,
                     "winner_loss": winner_loss, "time_s": elapsed})

    print("Summary:")
    for r in rows:
        print(f"  truth={r['truth']:30s} winner={r['winner']}")


if __name__ == "__main__":
    main()
