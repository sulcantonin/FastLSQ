#!/usr/bin/env python
"""Discover the drift and diffusion of an interest-rate SDE from data.

A short-rate model has the generic form
    dr = mu(r,t) dt + sigma(r,t) dW.
The four classical templates:
    Vasicek:        mu = kappa(theta - r),       sigma = sigma_0
    CIR:            mu = kappa(theta - r),       sigma = sigma_0 sqrt(r)
    CKLS:           mu = kappa(theta - r),       sigma = sigma_0 r^gamma
    Hull-White:     mu = (theta(t) - a r),       sigma = sigma_0
For a fixed time-step approximation we have, increment by increment,
    dr_i = mu(r_i) dt + sigma(r_i) sqrt(dt) z_i,   z_i ~ N(0,1).
So given a series r_1,...,r_N we can estimate the conditional mean and
conditional variance of  dr_i  given  r_i  by binning, and fit a symbolic
mu(r) and sigma^2(r) to the binned estimates.

We try to download the FRED DGS10 series (10-year Treasury constant
maturity); if not available offline we fall back to synthetic Vasicek
or CIR data.

Usage:  python fred_sde.py
"""
from __future__ import annotations

import time
import io
import urllib.request
import numpy as np

np.set_printoptions(precision=4, suppress=True)


# ----------------------------------------------------------------------
# Data fetching (or synthetic fallback).
# ----------------------------------------------------------------------

def fetch_fred_dgs10(timeout=10):
    """Try to download FRED DGS10 (10-year CMT) daily yields.
    Returns (dates, values_in_decimal) or None if no internet."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as e:
        print(f"   FRED fetch failed ({type(e).__name__}: {e})", flush=True)
        return None
    # parse: DATE,DGS10 columns; skip header
    dates, vals = [], []
    for line in raw.splitlines()[1:]:
        try:
            d, v = line.split(",")
            f = float(v)
            dates.append(d); vals.append(f / 100.0)   # percent → decimal
        except (ValueError, IndexError):
            continue
    if len(vals) < 500:
        return None
    return np.array(dates), np.array(vals)


def synth_vasicek(n=3000, kappa=0.10, theta=0.045, sigma0=0.015, r0=0.04,
                  dt=1/252, seed=0):
    rng = np.random.default_rng(seed)
    r = np.empty(n); r[0] = r0
    for i in range(1, n):
        eps = rng.standard_normal()
        r[i] = r[i-1] + kappa*(theta - r[i-1])*dt + sigma0*np.sqrt(dt)*eps
    return r


def synth_cir(n=3000, kappa=0.08, theta=0.05, sigma0=0.07, r0=0.04,
              dt=1/252, seed=0):
    rng = np.random.default_rng(seed)
    r = np.empty(n); r[0] = r0
    for i in range(1, n):
        eps = rng.standard_normal()
        diff = sigma0 * np.sqrt(max(r[i-1], 1e-8)) * np.sqrt(dt) * eps
        r[i] = max(r[i-1] + kappa*(theta - r[i-1])*dt + diff, 1e-8)
    return r


# ----------------------------------------------------------------------
# Conditional moment estimation: bin r and compute E[dr|r], Var[dr|r].
# ----------------------------------------------------------------------

def conditional_moments(r, n_bins=20, dt=1/252):
    """Return (r_bin_centres, mu_hat(r_bin), sigma2_hat(r_bin)) per bin."""
    dr = np.diff(r)
    r_at_step = r[:-1]
    bins = np.quantile(r_at_step, np.linspace(0, 1, n_bins + 1))
    bins[0] -= 1e-9; bins[-1] += 1e-9
    idx = np.digitize(r_at_step, bins) - 1
    r_c, m_c, v_c, n_c = [], [], [], []
    for k in range(n_bins):
        mask = (idx == k)
        if mask.sum() < 5:
            continue
        r_c.append(r_at_step[mask].mean())
        m_c.append(dr[mask].mean() / dt)            # ≈ mu(r)
        v_c.append(dr[mask].var()  / dt)            # ≈ sigma^2(r)
        n_c.append(mask.sum())
    return (np.asarray(r_c), np.asarray(m_c),
            np.asarray(v_c), np.asarray(n_c))


# ----------------------------------------------------------------------
# Symbolic candidates for mu(r) and sigma^2(r).
# ----------------------------------------------------------------------

def fit_drift(r_c, mu_hat, n_c):
    """Fit candidate drift forms; report best by AIC.
    Candidates:
       (Const)         mu = a
       (Linear-MR)     mu = a + b*r          (mean-reverting: b<0 means kappa>0)
       (Quadratic)     mu = a + b*r + c*r^2
    """
    w = np.sqrt(n_c)
    cands = []
    for name, basis_fn in [
        ("Const",     lambda r: np.stack([np.ones_like(r)], axis=1)),
        ("Linear-MR", lambda r: np.stack([np.ones_like(r), r], axis=1)),
        ("Quadratic", lambda r: np.stack([np.ones_like(r), r, r**2], axis=1)),
    ]:
        X = basis_fn(r_c)
        Xw = X * w[:, None]; yw = mu_hat * w
        coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        resid = (X @ coef - mu_hat) * w
        rss = float(resid @ resid)
        k = X.shape[1]
        aic = len(r_c) * np.log(max(rss / max(len(r_c), 1), 1e-30)) + 2 * k
        cands.append((name, coef, rss, aic))
    cands.sort(key=lambda x: x[3])
    return cands


def fit_diffusion(r_c, sigma2_hat, n_c):
    """Fit candidate sigma^2(r) forms; report best by AIC.
       (Const)         sigma^2 = a
       (Linear-r)      sigma^2 = a * r           ← CIR (after rescaling)
       (Linear-r^2)    sigma^2 = a * r^2         ← geometric BM
       (Power)         sigma^2 = a * r^(2*gamma) — fit gamma in log space
    """
    w = np.sqrt(n_c)
    cands = []
    # Const, linear-r, linear-r^2 by weighted LSQ
    for name, basis_fn in [
        ("Const",      lambda r: np.stack([np.ones_like(r)], axis=1)),
        ("Linear-r",   lambda r: np.stack([r],                axis=1)),
        ("Linear-r^2", lambda r: np.stack([r**2],             axis=1)),
    ]:
        X = basis_fn(r_c)
        Xw = X * w[:, None]; yw = sigma2_hat * w
        coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        resid = (X @ coef - sigma2_hat) * w
        rss = float(resid @ resid)
        k = X.shape[1]
        aic = len(r_c) * np.log(max(rss / max(len(r_c), 1), 1e-30)) + 2 * k
        cands.append((name, coef, rss, aic))
    # Power: log(sigma^2) = log(a) + 2*gamma * log(r), needs r > 0
    pos = r_c > 0
    if pos.sum() > 3:
        lx = np.log(r_c[pos]); ly = np.log(np.maximum(sigma2_hat[pos], 1e-20))
        ww = w[pos]
        X = np.stack([np.ones_like(lx), lx], axis=1)
        Xw = X * ww[:, None]; yw = ly * ww
        coef_p, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        log_a, two_gamma = coef_p
        # AIC in original space
        pred = np.exp(log_a) * (r_c[pos] ** two_gamma)
        resid = (pred - sigma2_hat[pos]) * w[pos]
        rss = float(resid @ resid)
        k = 2
        aic = pos.sum() * np.log(max(rss / max(pos.sum(), 1), 1e-30)) + 2 * k
        cands.append((f"Power(gamma={two_gamma/2:.2f})", coef_p, rss, aic))
    cands.sort(key=lambda x: x[3])
    return cands


# ----------------------------------------------------------------------
# Pretty-print one analysis
# ----------------------------------------------------------------------

def pretty_drift(name, coef):
    if name == "Const":
        return f"mu(r) = {coef[0]:+.4f}"
    if name == "Linear-MR":
        a, b = coef
        # mean reversion: mu = a + b*r = b*(a/b + r), kappa = -b, theta = -a/b
        kappa = -b
        theta = -a / b if abs(b) > 1e-12 else float("nan")
        return (f"mu(r) = {a:+.4f} + {b:+.4f} r   "
                f"(kappa={kappa:+.3f}, theta={theta:+.4f})")
    if name == "Quadratic":
        return (f"mu(r) = {coef[0]:+.4f} + {coef[1]:+.4f} r + {coef[2]:+.4f} r^2")
    return f"{name}: coefs={coef}"


def pretty_diffusion(name, coef):
    if name == "Const":
        return f"sigma^2(r) = {coef[0]:.4g}        => sigma = {np.sqrt(max(coef[0],0)):.4f}"
    if name == "Linear-r":
        return f"sigma^2(r) = {coef[0]:.4g} * r    (CIR-like)"
    if name == "Linear-r^2":
        return f"sigma^2(r) = {coef[0]:.4g} * r^2 (geom-BM-like)"
    if name.startswith("Power"):
        log_a, two_gamma = coef
        return (f"sigma^2(r) = {np.exp(log_a):.4g} * r^{two_gamma:.3f}    "
                f"({name})")
    return f"{name}: coefs={coef}"


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def analyse(label, r, dt=1/252):
    print(f"\n=== {label}   (N={len(r)}  observations)")
    r_c, mu_hat, sig2_hat, n_c = conditional_moments(r, n_bins=20, dt=dt)
    print(f"   r range:        {r_c.min():.4f} .. {r_c.max():.4f}")
    print(f"   binned drifts:  range {mu_hat.min():+.4f} .. {mu_hat.max():+.4f}")
    print(f"   binned vars:    range {sig2_hat.min():.4g} .. {sig2_hat.max():.4g}")
    drift_cands  = fit_drift(r_c, mu_hat, n_c)
    diff_cands   = fit_diffusion(r_c, sig2_hat, n_c)
    print("   Drift candidates (sorted by AIC):")
    for nm, coef, rss, aic in drift_cands:
        marker = "←" if nm == drift_cands[0][0] else " "
        print(f"     {marker} {nm:12s}  AIC={aic:9.1f}  {pretty_drift(nm, coef)}")
    print("   Diffusion candidates (sorted by AIC):")
    for nm, coef, rss, aic in diff_cands:
        marker = "←" if nm == diff_cands[0][0] else " "
        print(f"     {marker} {nm:20s}  AIC={aic:9.1f}  {pretty_diffusion(nm, coef)}")
    return drift_cands[0], diff_cands[0]


def main():
    print(">> Short-rate SDE discovery (FRED + synthetic checks)\n", flush=True)

    # --- 1. Synthetic Vasicek
    r_vas = synth_vasicek(n=3000, kappa=0.10, theta=0.045, sigma0=0.015, seed=0)
    analyse("Synthetic Vasicek  truth: drift Linear-MR (kappa=0.10, theta=0.045); "
            "sigma^2 = const = 0.000225", r_vas)

    # --- 2. Synthetic CIR
    r_cir = synth_cir(n=3000, kappa=0.08, theta=0.05, sigma0=0.07, seed=0)
    analyse("Synthetic CIR      truth: drift Linear-MR (kappa=0.08, theta=0.050); "
            "sigma^2 = 0.0049 * r", r_cir)

    # --- 3. Real FRED data, if reachable
    res = fetch_fred_dgs10()
    if res is None:
        print("\n=== FRED DGS10 unavailable (offline) --- skipping real-data run.")
    else:
        dates, r_real = res
        # drop NaN-ish entries (FRED uses '.' for missing)
        r_real = r_real[~np.isnan(r_real)]
        analyse(f"FRED DGS10 (10-yr Treasury daily yields, {len(r_real)} obs)",
                r_real)


if __name__ == "__main__":
    main()
