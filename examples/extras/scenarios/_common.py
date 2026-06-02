"""Shared utilities for FFF cross-domain scenario scripts.

* download_url(url, cache, timeout=...)
        Auto-fetch via curl, cached at the given path; safe to re-run.

* fit_1d_fff(t, y, sigma_W, n_features, mu_reg)
        Fit y(t) on a 1D sinusoidal random Fourier-feature basis.

* ls_power(t, y, W) / ls_periodogram_peaks(t, y, W_min, W_max, ...)
        Lomb-Scargle-style spectral readout for univariate or
        multivariate y, returns top peaks for the symbolic head.

* spectral_summary(W, beta)
        Top-K, K_99, energy-weighted bandwidth -- the
        "expansion-as-symbolic-answer" readouts used throughout the
        paper.

* report(rows, columns, title)
        Plain-text tabular printer.
"""
from __future__ import annotations

import os
import subprocess
import numpy as np
import torch

torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------
# Auto-fetch
# ----------------------------------------------------------------------

def download_url(url, cache, timeout=60, headers=None, verbose=True):
    """curl-based download with a sticky cache.  Returns the cache path."""
    if os.path.exists(cache) and os.path.getsize(cache) > 0:
        if verbose:
            print(f"   cached  {cache}")
        return cache
    if verbose:
        print(f"   fetching {url}  ->  {cache}")
    cmd = ["curl", "-sL", "--max-time", str(timeout)]
    if headers:
        for k, v in headers.items():
            cmd += ["-H", f"{k}: {v}"]
    cmd += [url, "-o", cache]
    subprocess.run(cmd, check=True)
    if not os.path.exists(cache) or os.path.getsize(cache) < 8:
        raise RuntimeError(f"download failed: {url}")
    return cache


# ----------------------------------------------------------------------
# 1D Fast Fourier Features fit
# ----------------------------------------------------------------------

def fit_1d_fff(t, y, sigma_W, n_features=1000, mu_reg=1e-10, seed=0,
               weights=None):
    """Fit y(t) = sum_j beta_j sin(W_j t + b_j) by fastlsq.solve_lstsq.

    Returns (basis, beta, rmse, hat).

    weights: optional 1D array of per-sample weights for weighted LSQ.
    """
    from fastlsq import SinusoidalBasis, solve_lstsq
    torch.manual_seed(seed)
    basis = SinusoidalBasis.random(input_dim=1, n_features=n_features,
                                   sigma=sigma_W)
    t_t = torch.tensor(t, dtype=torch.float64).reshape(-1, 1)
    y_t = torch.tensor(y, dtype=torch.float64).reshape(-1, 1)
    phi = basis.evaluate(t_t)
    if weights is not None:
        w = torch.tensor(np.sqrt(weights),
                         dtype=torch.float64).reshape(-1, 1)
        beta = solve_lstsq(phi * w, y_t * w, mu=mu_reg).reshape(-1)
    else:
        beta = solve_lstsq(phi, y_t, mu=mu_reg).reshape(-1)
    y_hat = (phi @ beta.reshape(-1, 1)).reshape(-1).numpy()
    rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))
    return basis, beta, rmse, y_hat


# ----------------------------------------------------------------------
# Lomb-Scargle periodogram on data
# ----------------------------------------------------------------------

def ls_power(t, y, W):
    """LS power at a single angular frequency W for a 1D signal y."""
    if y.ndim == 1:
        y_arr = y[:, None]
    else:
        y_arr = y
    sw = np.sin(W * t); cw = np.cos(W * t)
    denom_s = float(sw @ sw); denom_c = float(cw @ cw); cross = float(sw @ cw)
    det = denom_s * denom_c - cross * cross
    if det <= 0: return 0.0
    power = 0.0
    for k in range(y_arr.shape[1]):
        col = y_arr[:, k]
        ys = float(sw @ col); yc = float(cw @ col)
        a = (denom_c * ys - cross * yc) / det
        b = (denom_s * yc - cross * ys) / det
        power += a * ys + b * yc
    return float(power)


def ls_periodogram_peaks(t, y, W_min, W_max, n_grid=4000, n_peaks=5,
                         suppress_log_frac=0.05):
    """Sweep log-spaced W, find top non-overlapping LS peaks.
    y can be 1D or 2D (each column is a component, e.g. x,y,z)."""
    grid = np.exp(np.linspace(np.log(W_min), np.log(W_max), n_grid))
    powers = np.array([ls_power(t, y, W) for W in grid])
    peaks = []
    used_logs = []
    for k in np.argsort(-powers):
        if powers[k] <= 0: continue
        lW = np.log(grid[k])
        if any(abs(lW - u) < suppress_log_frac for u in used_logs):
            continue
        # micro-refine ±3 grid steps
        lo = max(k - 3, 0); hi = min(k + 3, len(grid) - 1)
        W_local = grid[lo:hi+1]
        p_local = powers[lo:hi+1]
        j = int(np.argmax(p_local))
        peaks.append((float(W_local[j]), float(p_local[j])))
        used_logs.append(lW)
        if len(peaks) >= n_peaks: break
    return peaks


# ----------------------------------------------------------------------
# Spectral expansion summary (top-K, K_99, bandwidth)
# ----------------------------------------------------------------------

def spectral_summary(W, beta, k_top=5, energy_target=0.99):
    W = np.asarray(W).reshape(-1)
    beta = np.asarray(beta).reshape(-1)
    e_j = beta ** 2
    e_tot = float(e_j.sum())
    if e_tot <= 0:
        return None
    order = np.argsort(-np.abs(beta))
    cum = np.cumsum(e_j[order]) / e_tot
    K_target = int(np.searchsorted(cum, energy_target) + 1)
    energy_K_top = float(cum[min(k_top, len(beta)) - 1])
    bandwidth = float(np.sum(np.abs(W) * e_j) / e_tot)
    return {
        "energy_K_top": energy_K_top,
        "K_target":     K_target,
        "bandwidth":    bandwidth,
        "N":            len(beta),
        "top_idx":      order[:k_top].tolist(),
    }


# ----------------------------------------------------------------------
# Pretty-printer
# ----------------------------------------------------------------------

def print_row(d, fmt):
    """Print one dict d with format spec dict fmt (key -> format string)."""
    s = "  ".join(format(d[k], fmt[k]) for k in fmt)
    print("  " + s, flush=True)
