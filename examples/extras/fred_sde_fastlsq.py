#!/usr/bin/env python
"""Short-rate SDE drift/diffusion discovery using fastlsq.

Given a time series r_t of interest-rate-like data, fit
        dr = mu(r) dt + sigma(r) sqrt(dt) z,    z ~ N(0,1)
by:

 1. Bin r and estimate conditional moments  E[dr|r],  Var[dr|r]  per bin.
 2. For each parametric candidate for mu(r) and sigma^2(r), build a
    weighted design matrix and solve it with fastlsq.solve_lstsq.  Score
    candidates by AIC.

Candidates for mu(r):
        Const             mu = a
        Linear-MR         mu = a + b r       (mean reverting if b < 0)
        Quadratic         mu = a + b r + c r^2

Candidates for sigma^2(r):
        Const             sigma^2 = a
        Linear-r          sigma^2 = a r              (CIR family)
        Linear-r^2        sigma^2 = a r^2            (geometric BM family)

Synthetic test cases (Vasicek, CIR) and an optional FRED DGS10 pull are
provided.  The same fastlsq.solve_lstsq call powers every parametric fit.

Usage:  python fred_sde_fastlsq.py
"""
from __future__ import annotations

import io
import os
import subprocess
import time
import urllib.request
import numpy as np
import torch

from fastlsq import SinusoidalBasis, Op, solve_lstsq
from spectral_expansion import spectral_expansion_report

torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=4, suppress=True)


# ----------------------------------------------------------------------
# Data: synthetic + (optional) FRED
# ----------------------------------------------------------------------

def synth_vasicek(n=3000, kappa=0.10, theta=0.045, sigma0=0.015, r0=0.04,
                  dt=1/252, seed=0):
    rng = np.random.default_rng(seed)
    r = np.empty(n); r[0] = r0
    for i in range(1, n):
        eps = rng.standard_normal()
        r[i] = r[i-1] + kappa * (theta - r[i-1]) * dt + sigma0 * np.sqrt(dt) * eps
    return r


def synth_cir(n=3000, kappa=0.08, theta=0.05, sigma0=0.07, r0=0.04,
              dt=1/252, seed=0):
    rng = np.random.default_rng(seed)
    r = np.empty(n); r[0] = r0
    for i in range(1, n):
        eps = rng.standard_normal()
        diff = sigma0 * np.sqrt(max(r[i-1], 1e-8)) * np.sqrt(dt) * eps
        r[i] = max(r[i-1] + kappa * (theta - r[i-1]) * dt + diff, 1e-8)
    return r


def fetch_fred_dgs10(timeout=30, cache="/tmp/fred_dgs10.csv"):
    """Fetch the FRED DGS10 series (10-year Treasury constant maturity
    daily yield) via curl; fall back to local cache if already on disk."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
    if not os.path.exists(cache):
        try:
            subprocess.run(["curl", "-s", "--max-time", str(timeout),
                            url, "-o", cache], check=True)
        except Exception as e:
            print(f"   FRED fetch failed ({type(e).__name__}: {e})", flush=True)
            return None
    if not os.path.exists(cache) or os.path.getsize(cache) < 1000:
        return None
    with open(cache) as f:
        raw = f.read()
    dates, vals = [], []
    for line in raw.splitlines()[1:]:
        try:
            d, v = line.split(",")
            vals.append(float(v) / 100.0)
            dates.append(d)
        except (ValueError, IndexError):
            continue
    if len(vals) < 500:
        return None
    return np.asarray(dates), np.asarray(vals)


# ----------------------------------------------------------------------
# Conditional moments by binning
# ----------------------------------------------------------------------

def conditional_moments(r, n_bins=20, dt=1/252):
    dr = np.diff(r); r_at = r[:-1]
    bins = np.quantile(r_at, np.linspace(0, 1, n_bins + 1))
    bins[0] -= 1e-9; bins[-1] += 1e-9
    idx = np.digitize(r_at, bins) - 1
    rc, mc, vc, nc = [], [], [], []
    for k in range(n_bins):
        m = (idx == k)
        if m.sum() < 5: continue
        rc.append(r_at[m].mean())
        mc.append(dr[m].mean() / dt)
        vc.append(dr[m].var()  / dt)
        nc.append(m.sum())
    return (np.asarray(rc), np.asarray(mc),
            np.asarray(vc), np.asarray(nc))


# ----------------------------------------------------------------------
# Parametric fits via fastlsq.solve_lstsq (weighted)
# ----------------------------------------------------------------------

def fit_candidate(r_c, target, n_c, basis_fn, mu_reg=1e-12):
    """Weighted-LSQ solve via fastlsq.solve_lstsq, return (coef, rss, k)."""
    w = np.sqrt(n_c)
    X = basis_fn(r_c)                       # (M, k)
    Xw = X * w[:, None]
    yw = target * w
    A = torch.as_tensor(Xw, dtype=torch.float64)
    b = torch.as_tensor(yw, dtype=torch.float64).reshape(-1, 1)
    coef = solve_lstsq(A, b, mu=mu_reg).reshape(-1).numpy()
    resid = (X @ coef - target) * w
    rss = float(resid @ resid)
    return coef, rss, X.shape[1]


def aic(rss, n_pts, k):
    return n_pts * np.log(max(rss / max(n_pts, 1), 1e-30)) + 2 * k


DRIFT_CANDIDATES = [
    ("Const",     lambda r: np.stack([np.ones_like(r)], axis=1)),
    ("Linear-MR", lambda r: np.stack([np.ones_like(r), r], axis=1)),
    ("Quadratic", lambda r: np.stack([np.ones_like(r), r, r**2], axis=1)),
]
DIFF_CANDIDATES = [
    ("Const",      lambda r: np.stack([np.ones_like(r)], axis=1)),
    ("Linear-r",   lambda r: np.stack([r],                axis=1)),
    ("Linear-r^2", lambda r: np.stack([r**2],             axis=1)),
]


def fit_ckls_gamma(r_c, sigma2_hat, n_c, mu_reg=1e-12,
                   gamma_grid=None):
    """Fit sigma^2(r) = a * r^(2 gamma) with continuous gamma, with
    original-space weighted LSQ (the same loss as the discrete
    candidates so the AIC values are directly comparable).

    For each gamma on a grid, the column r^(2 gamma) is a single
    feature; fastlsq.solve_lstsq gives the best 'a' in closed form,
    and we pick the gamma minimising the original-space weighted RSS.
    A few coordinate-descent polish steps refine the winner.

    Returns (gamma_hat, a_hat, rss_in_original_space, n_used).
    """
    pos = (r_c > 0) & (sigma2_hat > 0)
    if pos.sum() < 3:
        return None
    r_p   = r_c[pos]
    s2_p  = sigma2_hat[pos]
    w     = np.sqrt(n_c[pos])
    yw_t  = torch.as_tensor(s2_p * w,
                            dtype=torch.float64).reshape(-1, 1)

    def fit_at_gamma(g):
        x = r_p ** (2 * g)
        Xw = (x * w).reshape(-1, 1)
        A = torch.as_tensor(Xw, dtype=torch.float64)
        a_hat = float(solve_lstsq(A, yw_t, mu=mu_reg).item())
        pred = a_hat * x
        resid = (pred - s2_p) * w
        rss = float(resid @ resid)
        return a_hat, rss

    if gamma_grid is None:
        gamma_grid = np.linspace(-0.25, 2.0, 31)
    best = (np.inf, 0.0, 0.0)
    for g in gamma_grid:
        a_hat, rss = fit_at_gamma(g)
        if rss < best[0]:
            best = (rss, g, a_hat)
    # Polish gamma by coordinate descent
    g_cur = best[1]; rss_cur = best[0]; a_cur = best[2]
    step = (gamma_grid[1] - gamma_grid[0]) * 0.5
    for _ in range(8):
        improved = False
        for s in (-1, +1):
            g_try = g_cur + s * step
            a_try, rss_try = fit_at_gamma(g_try)
            if rss_try < rss_cur:
                rss_cur, g_cur, a_cur, improved = rss_try, g_try, a_try, True
        if not improved:
            step *= 0.5
    return g_cur, a_cur, rss_cur, int(pos.sum())


# ----------------------------------------------------------------------
# Interpretability: fit mu_hat(r) on a sinusoidal basis and read off the
# local mean-reversion speed kappa(r) = - d mu / d r in closed form from
# the cyclic identity.  For Vasicek/CIR the truth is kappa(r) = const;
# the data-driven kappa(r) curve says whether the linear-MR fit is
# defensible across the full rate range.
# ----------------------------------------------------------------------

def fit_drift_basis_and_kappa(r_c, mu_hat, n_c, n_features=80, sigma=8.0,
                              mu_reg=1e-2, seed=0):
    """Fit mu(r) on a 1D sinusoidal basis under the same weights as the
    parametric drifts; return the analytical d mu / d r at the bin
    centres, plus the basis fit for spectral interrogation."""
    torch.manual_seed(seed)
    basis = SinusoidalBasis.random(input_dim=1, n_features=n_features,
                                   sigma=sigma)
    r_t  = torch.tensor(r_c,    dtype=torch.float64).reshape(-1, 1)
    y_t  = torch.tensor(mu_hat, dtype=torch.float64).reshape(-1, 1)
    w_t  = torch.tensor(np.sqrt(n_c), dtype=torch.float64).reshape(-1, 1)
    phi  = basis.evaluate(r_t)
    A = phi * w_t
    b = y_t * w_t
    c = solve_lstsq(A, b, mu=mu_reg).reshape(-1)
    dphi = Op.partial(dim=0, order=1, d=1).apply(basis, r_t)
    mu_at_r   = (phi  @ c.reshape(-1, 1)).reshape(-1).numpy()
    dmu_at_r  = (dphi @ c.reshape(-1, 1)).reshape(-1).numpy()
    kappa_at_r = -dmu_at_r
    return mu_at_r, kappa_at_r, basis, c


def analyse(label, r, dt=1/252):
    print(f"\n=== {label}  (N={len(r)})")
    rc, mc, vc, nc = conditional_moments(r, n_bins=20, dt=dt)
    t0 = time.perf_counter()
    drift = []
    for name, basis_fn in DRIFT_CANDIDATES:
        coef, rss, k = fit_candidate(rc, mc, nc, basis_fn)
        drift.append((name, coef, rss, aic(rss, len(rc), k)))
    diff = []
    for name, basis_fn in DIFF_CANDIDATES:
        coef, rss, k = fit_candidate(rc, vc, nc, basis_fn)
        diff.append((name, coef, rss, aic(rss, len(rc), k)))
    # Continuous-gamma CKLS family
    ckls = fit_ckls_gamma(rc, vc, nc)
    gamma_hat = None
    if ckls is not None:
        gamma_hat, a_hat, rss_c, n_used = ckls
        # Match the AIC scale used for the discrete candidates: n_pts is
        # len(rc) for those; here we used only positive bins, so report
        # AIC over the same len(rc) count to keep them comparable, with
        # k = 2 parameters (log_a, gamma).
        aic_c = aic(rss_c, len(rc), 2)
        diff.append((f"CKLS(gamma={gamma_hat:+.3f})",
                     np.array([a_hat, gamma_hat]), rss_c, aic_c))
    # Interpretability: kappa(r) from analytical d mu / d r on a basis
    mu_at_r, kappa_at_r, basis_mu, c_mu = fit_drift_basis_and_kappa(rc, mc, nc)
    elapsed = time.perf_counter() - t0
    drift.sort(key=lambda x: x[3]); diff.sort(key=lambda x: x[3])
    print("   Drift candidates (sorted by AIC):")
    for name, coef, rss, a in drift:
        print(f"     {name:24s}  AIC={a:8.2f}  coef={coef}")
    print("   Diffusion candidates (sorted by AIC):")
    for name, coef, rss, a in diff:
        print(f"     {name:24s}  AIC={a:8.2f}  coef={coef}")
    # Local mean-reversion at three quantiles of r.
    q = [0.1, 0.5, 0.9]
    idxs = [int(qi * (len(rc) - 1)) for qi in q]
    print("   kappa(r) = - d mu / d r at quantiles 10 / 50 / 90 %:")
    for qi, i in zip(q, idxs):
        print(f"     q={qi:.1f}  r={rc[i]:.4f}  kappa(r) = {kappa_at_r[i]:+.4f}")
    # Spectral expansion of mu_hat(r) on the basis
    spec = spectral_expansion_report(basis_mu.W.numpy(),
                                     basis_mu.b.numpy(),
                                     c_mu.numpy(),
                                     label="mu", k_top=5)
    print(f"   t_fastlsq = {elapsed:.3f}s")
    return drift[0][0], diff[0][0], elapsed, gamma_hat, spec


def main():
    print(">> SDE drift/diffusion discovery (fastlsq backend)\n", flush=True)
    rows = []

    # 1. Vasicek (truth: Linear-MR drift, Const diffusion; CKLS gamma -> 0)
    r = synth_vasicek(n=10000, kappa=0.30, theta=0.045, sigma0=0.012, seed=0)
    d_w, s_w, t, g, _spec = analyse("Synthetic Vasicek (truth: Linear-MR + Const, gamma=0)", r)
    rows.append(("Vasicek",    "Linear-MR", "Const",    d_w, s_w, t, g, 0.0))

    # 2. CIR (truth: Linear-MR drift, Linear-r diffusion; CKLS gamma -> 0.5)
    r = synth_cir(n=10000, kappa=0.30, theta=0.05, sigma0=0.07, seed=0)
    d_w, s_w, t, g, _spec = analyse("Synthetic CIR (truth: Linear-MR + Linear-r, gamma=0.5)", r)
    rows.append(("CIR",        "Linear-MR", "Linear-r", d_w, s_w, t, g, 0.5))

    # 3. FRED DGS10 if reachable
    res = fetch_fred_dgs10()
    if res is not None:
        dates, real = res
        mask = ~np.isnan(real)
        real = real[mask]; dates = dates[mask]
        print(f"\n   FRED DGS10 covers {dates[0]} ... {dates[-1]}")
        d_w, s_w, t, g, _spec = analyse(f"FRED DGS10 (real Treasury yield, {len(real)} obs)", real)
        rows.append(("FRED-DGS10", "?", "?", d_w, s_w, t, g, None))

    print("\nSummary:")
    print(f"  {'series':14s} {'drift fit':12s} {'diff fit':12s} "
          f"{'gamma_hat':10s} {'gamma_truth':12s} {'t (s)':8s}")
    n_correct = 0
    for series, dt_truth, st_truth, dt_fit, st_fit, t, g, g_true in rows:
        d_ok = (dt_fit == dt_truth) or (dt_truth == "?")
        s_ok = (st_fit == st_truth) or (st_truth == "?")
        if dt_truth != "?" and st_truth != "?":
            n_correct += int(d_ok and s_ok)
        g_str = f"{g:+.3f}" if g is not None else "  n/a"
        gt_str = f"{g_true:+.2f}" if g_true is not None else "  n/a"
        print(f"  {series:14s} {dt_fit:12s} {st_fit:12s} "
              f"{g_str:10s} {gt_str:12s} {t:<8.3f}")
    synth_n = sum(1 for r in rows if r[1] != "?")
    print(f"  >>> {n_correct}/{synth_n} synthetic cases recovered")
    return rows, n_correct


if __name__ == "__main__":
    main()
