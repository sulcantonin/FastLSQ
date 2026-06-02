#!/usr/bin/env python
"""Symbolic factor-loading discovery on **real** Fama--French data.

We pull two public datasets from Ken French's data library
(https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html,
public, no API key):

    F-F_Research_Data_Factors_daily.csv      -- Mkt-RF, SMB, HML, RF
    10_Industry_Portfolios_Daily.csv         -- 10 value-weighted industry
                                                portfolio daily returns

For each industry portfolio we form excess returns r_i = R_i - RF and
ask: what is the symbolic expression for r_i in terms of (Mkt-RF, SMB,
HML)?  The Fama--French 3-factor model says r_i is *linear* in these
three factors plus a constant alpha, with industry-specific betas.
Decades of empirical asset pricing back this up.

We assemble a richer dictionary that includes the three linear factors
*plus* squares, products, signed powers, and the lagged factor values,
and run sequentially thresholded least squares (STLSQ) with fastlsq's
solve_lstsq as the inner solver.  Two things are interesting:

    (a) STLSQ should keep the three linear factors and reject the
        nonlinear primitives -- a positive recovery test, because we
        already know what the right symbolic answer is.
    (b) The recovered linear coefficients should match the known
        industry tilts: HiTec is growth (negative HML), Energy is value
        (positive HML), Utilities is low-beta (Mkt-RF < 1), and so on.

This is a real-data test on which the canonical symbolic answer exists
and has been independently verified in the asset-pricing literature.

Usage:  python numerai_alpha_fastlsq.py
"""
from __future__ import annotations

import os
import subprocess
import time
import zipfile
import numpy as np
import torch

from fastlsq import SinusoidalBasis, Op, solve_lstsq
from spectral_expansion import spectral_expansion_report

torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------
# Fama--French daily data via Ken French's data library
# ----------------------------------------------------------------------

FF_URL_3FAC = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
               "ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip")
FF_URL_IND  = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
               "ken.french/ftp/10_Industry_Portfolios_daily_CSV.zip")


def fetch_zip(url, cache_zip, member_csv, cache_csv, timeout=30):
    if not os.path.exists(cache_csv):
        subprocess.run(["curl", "-sL", "--max-time", str(timeout),
                        url, "-o", cache_zip], check=True)
        with zipfile.ZipFile(cache_zip) as zf:
            zf.extract(member_csv, path=os.path.dirname(cache_zip))
        os.rename(os.path.join(os.path.dirname(cache_zip), member_csv),
                  cache_csv)
    return cache_csv


def load_ff_factors():
    """Return (date_int, factors[Mkt-RF,SMB,HML,RF])."""
    csv = fetch_zip(FF_URL_3FAC, "/tmp/ff3.zip",
                    "F-F_Research_Data_Factors_daily.csv",
                    "/tmp/ff3.csv")
    dates, rows = [], []
    seen_header = False
    with open(csv) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith(",Mkt-RF"):
                seen_header = True
                continue
            if not seen_header:
                continue
            if ln.startswith("Copyright"):
                break
            parts = ln.split(",")
            if len(parts) != 5:
                continue
            try:
                d  = int(parts[0])
                vs = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            dates.append(d); rows.append(vs)
    return np.asarray(dates), np.asarray(rows)


def load_ff_industries():
    """Return (date_int, returns[10], industry_names) for the *value-
    weighted* daily industry portfolios."""
    csv = fetch_zip(FF_URL_IND, "/tmp/ff10ind.zip",
                    "10_Industry_Portfolios_Daily.csv",
                    "/tmp/ff10ind.csv")
    section = None
    dates, rows, names = [], [], None
    with open(csv) as f:
        for ln in f:
            stripped = ln.strip()
            if "Average Value Weighted Returns" in stripped:
                section = "vw"
                continue
            if "Average Equal Weighted Returns" in stripped:
                break
            if section != "vw":
                continue
            if stripped.startswith(","):
                names = [s.strip() for s in stripped.split(",")[1:]]
                continue
            parts = stripped.split(",")
            if len(parts) < 11 or not parts[0].isdigit():
                continue
            try:
                d  = int(parts[0])
                vs = [float(x) for x in parts[1:11]]
            except ValueError:
                continue
            if any(v <= -90 for v in vs):                # missing-data sentinel
                continue
            dates.append(d); rows.append(vs)
    return np.asarray(dates), np.asarray(rows), names


# ----------------------------------------------------------------------
# Dictionary of candidate primitives over the 3 factors
# ----------------------------------------------------------------------

def build_dictionary(F):
    """F has columns [Mkt-RF, SMB, HML]; return Theta and primitive names."""
    n, d = F.shape
    cols, names = [], []
    # 1. constant
    cols.append(np.ones(n));               names.append("1")
    # 2. linear
    for i, nm in enumerate(["Mkt-RF", "SMB", "HML"]):
        cols.append(F[:, i]);              names.append(nm)
    # 3. squares
    for i, nm in enumerate(["Mkt-RF", "SMB", "HML"]):
        cols.append(F[:, i] ** 2);         names.append(f"{nm}^2")
    # 4. cross products
    pairs = [(0, 1, "Mkt-RF*SMB"),
             (0, 2, "Mkt-RF*HML"),
             (1, 2, "SMB*HML")]
    for i, j, nm in pairs:
        cols.append(F[:, i] * F[:, j]);    names.append(nm)
    # 5. signed factor (truncates direction information from magnitude)
    for i, nm in enumerate(["Mkt-RF", "SMB", "HML"]):
        cols.append(np.sign(F[:, i]));     names.append(f"sign({nm})")
    return np.stack(cols, axis=1), names


# ----------------------------------------------------------------------
# STLSQ with fastlsq.solve_lstsq inner solver
# ----------------------------------------------------------------------

def stlsq_fastlsq(Theta, y, threshold=0.05, mu=1e-8, max_iter=25):
    Theta_t = torch.as_tensor(Theta, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.float64).reshape(-1, 1)
    coeffs = solve_lstsq(Theta_t, y_t, mu=mu).reshape(-1)
    for _ in range(max_iter):
        small = torch.abs(coeffs) < threshold
        if not torch.any(~small):
            return coeffs.numpy()
        keep = ~small
        if not torch.any(keep):
            return coeffs.numpy() * 0.0
        coeffs_new = torch.zeros_like(coeffs)
        sub = solve_lstsq(Theta_t[:, keep], y_t, mu=mu).reshape(-1)
        coeffs_new[keep] = sub
        if torch.equal(small, torch.abs(coeffs_new) < threshold):
            return coeffs_new.numpy()
        coeffs = coeffs_new
    return coeffs.numpy()


def discover(Theta, y, threshold=0.05):
    """Standardise, STLSQ, then de-standardise the surviving support."""
    col_norms = np.linalg.norm(Theta, axis=0) + 1e-30
    y_norm = np.linalg.norm(y) + 1e-30
    Theta_s = Theta / col_norms
    y_s = y / y_norm
    c_s = stlsq_fastlsq(Theta_s, y_s, threshold=threshold)
    support = np.abs(c_s) > 0
    coeffs = np.zeros(Theta.shape[1])
    if np.any(support):
        Theta_sub = torch.as_tensor(Theta[:, support], dtype=torch.float64)
        y_t = torch.as_tensor(y, dtype=torch.float64).reshape(-1, 1)
        sub = solve_lstsq(Theta_sub, y_t, mu=1e-10).reshape(-1).numpy()
        coeffs[support] = sub
    return coeffs


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

LINEAR_TRUE = {"1", "Mkt-RF", "SMB", "HML"}      # constant + 3 factors


# ----------------------------------------------------------------------
# Interpretability: time-varying market beta with an analytical
# time-derivative.  We model y_t = beta(t) * Mkt_t with beta(t) on a 1D
# sinusoidal basis.  The design row is A_{t,j} = Mkt_t * phi_j(t).
# After the solve we have both beta(t) and beta'(t) in closed form via
# the cyclic identity -- the derivative is *free*.
# ----------------------------------------------------------------------

def fit_time_varying_beta(t_norm, mkt, y, n_features=80, sigma=4.0,
                          mu_reg=1e-3, seed=0):
    """t_norm: time in [0,1].  mkt, y: same length.  Returns (basis, c)
    where beta(t) = sum_j c_j phi_j(t) and beta'(t) is the same sum with
    the derivative basis."""
    torch.manual_seed(seed)
    basis = SinusoidalBasis.random(input_dim=1, n_features=n_features,
                                   sigma=sigma)
    t_t = torch.tensor(t_norm, dtype=torch.float64).reshape(-1, 1)
    phi   = basis.evaluate(t_t)                                  # (T, N)
    A = phi * torch.tensor(mkt, dtype=torch.float64).reshape(-1, 1)
    b = torch.tensor(y,   dtype=torch.float64).reshape(-1, 1)
    c = solve_lstsq(A, b, mu=mu_reg).reshape(-1)
    return basis, c


def evaluate_beta_and_dbeta(basis, c, t_grid):
    t_t = torch.tensor(t_grid, dtype=torch.float64).reshape(-1, 1)
    phi   = basis.evaluate(t_t)                                  # (M, N)
    dphi  = Op.partial(dim=0, order=1, d=1).apply(basis, t_t)    # (M, N)
    beta_t   = (phi  @ c.reshape(-1, 1)).reshape(-1).numpy()
    dbeta_t  = (dphi @ c.reshape(-1, 1)).reshape(-1).numpy()
    return beta_t, dbeta_t


def main():
    print(">> Fama--French factor-loading discovery (REAL data)\n",
          flush=True)
    d_fac, F = load_ff_factors()
    d_ind, R, ind_names = load_ff_industries()
    # Align on common dates
    common, i_fac, i_ind = np.intersect1d(d_fac, d_ind, return_indices=True)
    F = F[i_fac];   R = R[i_ind]
    Mkt, SMB, HML, RF = F[:, 0], F[:, 1], F[:, 2], F[:, 3]
    factors = np.stack([Mkt, SMB, HML], axis=1)
    print(f"   loaded {len(common)} common trading days "
          f"({common[0]} -> {common[-1]})")
    print(f"   industries: {ind_names}")
    print(f"   factor mean (Mkt-RF, SMB, HML): "
          f"({Mkt.mean():+.3f}, {SMB.mean():+.3f}, {HML.mean():+.3f}) bp/day\n",
          flush=True)

    Theta, prim_names = build_dictionary(factors)
    # Chronological 80/20 split for honest OOS evaluation
    split = int(0.8 * len(common))
    Theta_tr, Theta_te = Theta[:split], Theta[split:]

    rows = []
    for j, ind in enumerate(ind_names):
        y = R[:, j] - RF                          # excess return
        y_tr, y_te = y[:split], y[split:]
        t0 = time.perf_counter()
        coeffs = discover(Theta_tr, y_tr, threshold=0.05)
        elapsed = time.perf_counter() - t0
        active = [(prim_names[i], coeffs[i])
                  for i in range(len(coeffs)) if abs(coeffs[i]) > 1e-12]
        # Out-of-sample fit
        y_hat = Theta_te @ coeffs
        ss_res = float(((y_te - y_hat) ** 2).sum())
        ss_tot = float(((y_te - y_te.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot
        # Symbolic recovery criterion: did STLSQ keep ONLY linear factors
        # (any subset of LINEAR_TRUE is fine -- STLSQ is supposed to drop
        # the factors that don't matter for this industry; the failure
        # mode would be keeping a NONLINEAR primitive).
        names_active = set(n for n, _ in active)
        found_extra  = names_active - LINEAR_TRUE
        clean_linear = (len(found_extra) == 0)
        # Pretty print
        print(f"--- {ind:6s}  ({len(active)} active, t={elapsed:.2f}s, "
              f"OOS R^2 = {r2:+.3f})")
        for n, c in sorted(active, key=lambda nc: -abs(nc[1])):
            tag = "  " if n in LINEAR_TRUE else "**"
            print(f"      {tag} {n:14s}  {c:+.4f}")
        print(f"      {'CLEAN: only linear factors kept' if clean_linear else 'extras leaked in'}; "
              f"extras = {sorted(found_extra) or '[]'}\n",
              flush=True)
        # Pull beta on Mkt-RF and HML for narrative
        coef_dict = {n: c for n, c in active}
        rows.append({"industry": ind, "n_active": len(active),
                     "clean_linear": clean_linear,
                     "n_extras": len(found_extra),
                     "beta_mkt": coef_dict.get("Mkt-RF", 0.0),
                     "beta_smb": coef_dict.get("SMB",    0.0),
                     "beta_hml": coef_dict.get("HML",    0.0),
                     "oos_r2":   r2,
                     "time":     elapsed})

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("Summary:  (LINEAR_TRUE = {1, Mkt-RF, SMB, HML})")
    print(f"  {'industry':10s} {'beta_Mkt':9s} {'beta_SMB':9s} {'beta_HML':9s} "
          f"{'clean':7s} {'extras':6s} {'OOS R^2':9s}")
    for r in rows:
        ok = "yes" if r["clean_linear"] else "no"
        print(f"  {r['industry']:10s} {r['beta_mkt']:+9.3f} {r['beta_smb']:+9.3f} "
              f"{r['beta_hml']:+9.3f} {ok:7s} {r['n_extras']:<6d} "
              f"{r['oos_r2']:+9.3f}")
    n_clean = sum(int(r["clean_linear"]) for r in rows)
    print(f"\n  {n_clean}/{len(rows)} industries: STLSQ kept only linear factors "
          f"(no nonlinear primitive leaked in)")

    # ------------------------------------------------------------------
    # Interpretability: time-varying beta(t) on a sinusoidal basis,
    # with the analytical d beta / d t.  We focus on HiTec, where the
    # asset-pricing literature documents a market-beta regime shift
    # around the dot-com bust (2000-2002) and another around the 2008
    # crisis.  fastlsq gives beta(t) and beta'(t) in closed form.
    # ------------------------------------------------------------------
    print("\n  Interpretability: time-varying market beta for HiTec")
    j = ind_names.index("HiTec")
    y = R[:, j] - RF
    # Use the post-1970 window where HiTec has comparable composition;
    # earlier data has a different sector definition.
    start_yyyymmdd = 19700101
    mask = common >= start_yyyymmdd
    t_raw = common[mask].astype(np.float64)
    t_norm = (t_raw - t_raw.min()) / (t_raw.max() - t_raw.min())
    basis_t, c_t = fit_time_varying_beta(t_norm, Mkt[mask], y[mask])
    # Spectral expansion of beta(t): top-K time-frequency components.
    spec_beta = spectral_expansion_report(basis_t.W.numpy(),
                                          basis_t.b.numpy(),
                                          c_t.numpy(),
                                          label="beta_HiTec(t_norm)",
                                          k_top=5)
    # Probe beta(t) and dbeta/dt at year-end dates of interest.
    probe_dates = [19800101, 19900101, 19951231, 20000101, 20021231,
                   20081231, 20151231, 20200101, 20211231, 20251231]
    probe_norm = [(d - t_raw.min()) / (t_raw.max() - t_raw.min())
                  for d in probe_dates if d >= t_raw.min() and d <= t_raw.max()]
    probe_dates = [d for d in probe_dates if d >= t_raw.min() and d <= t_raw.max()]
    b_t, db_t = evaluate_beta_and_dbeta(basis_t, c_t, np.asarray(probe_norm))
    print(f"  {'date':12s} {'beta_Mkt(t)':12s} {'d beta / d t':14s}")
    for d, b, db in zip(probe_dates, b_t, db_t):
        print(f"  {str(d):12s} {b:+12.3f} {db:+14.3e}")
    # Detect the largest |dbeta/dt| in the interior (skip 5% boundaries
    # where the basis is under-constrained).  Map normalised t back to
    # a calendar year by linear interpolation against the integer dates
    # (those YYYYMMDD codes are not linear in time, so we have to map
    # via index).
    grid = np.linspace(0.05, 0.95, 400)
    _, dbg = evaluate_beta_and_dbeta(basis_t, c_t, grid)
    k = int(np.argmax(np.abs(dbg)))
    idx_real = int(round(grid[k] * (len(t_raw) - 1)))
    yyyymmdd = int(t_raw[idx_real])
    print(f"  peak |d beta / d t| in interior near {yyyymmdd} "
          f"(d beta / d t = {dbg[k]:+.3e})")

    return rows, n_clean


if __name__ == "__main__":
    main()
