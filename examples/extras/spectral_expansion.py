"""Spectral-expansion interpretability for fastlsq sinusoidal fits.

A fastlsq fit produces an explicit closed-form symbolic expression
            f_hat(x) = sum_j beta_j sin(W_j^T x + b_j),
so the beta vector together with (W, b) IS the symbolic answer.  Three
robust quantitative readouts of that expansion are useful in any
scenario:

  1. Sparse symbolic compression.  Sort features by |beta_j| descending
     and report the top K terms.  These give a K-term trigonometric
     polynomial that captures the bulk of the energy in the function;
     for moderate K (5-20) this is a readable, finite, named expression
     for the recovered function.

  2. Compressibility K_99.  The smallest K such that the top K terms
     hold at least 99% of the total l2 energy of beta.  A function that
     can be re-expressed as a short trigonometric sum has small K_99;
     a function spread across many basis directions has large K_99.

  3. Effective bandwidth.  The energy-weighted mean of |W_j| under the
     distribution beta_j^2 / sum beta_j^2.  This is the dominant
     frequency scale of the function in the units of the input domain
     and gives a physically interpretable "characteristic scale" --- a
     length scale for spatial fits, a timescale for temporal ones.

All three readouts come for free from the same beta vector that fastlsq
already solves for.
"""
from __future__ import annotations
import numpy as np


def spectral_expansion_report(W, b, beta, label="f", k_top=8,
                              energy_target=0.99):
    """Print and return a spectral readout of a fastlsq fit.

    Parameters
    ----------
    W : (d, N) array      -- feature frequencies
    b : (1, N) array      -- feature phases
    beta : (N,) array     -- recovered coefficients
    label : str           -- function name used in printout
    k_top : int           -- how many top-amplitude terms to report
    energy_target : float -- threshold for K_target (e.g. 0.99)

    Returns
    -------
    dict with keys:
        top_terms       list of {beta, W, b} dicts
        energy_K_top    fraction of energy in the top k_top features
        K_target        smallest K reaching energy_target fraction
        bandwidth       energy-weighted mean |W|
    """
    W = np.asarray(W); b = np.asarray(b); beta = np.asarray(beta).reshape(-1)
    N = beta.size
    Wnorm = np.linalg.norm(W, axis=0) if W.ndim == 2 else np.abs(W)
    e_j = beta ** 2
    energy_total = float(e_j.sum())
    if energy_total <= 0:
        return None
    order = np.argsort(-np.abs(beta))
    cum = np.cumsum(e_j[order]) / energy_total
    # K such that cum[K-1] >= energy_target
    K_target = int(np.searchsorted(cum, energy_target) + 1)
    energy_K_top = float(cum[min(k_top, N) - 1])
    bandwidth = float(np.sum(Wnorm * e_j) / energy_total)
    top = order[:k_top]
    print(f"   Spectral expansion of {label}(x) "
          f"= sum_j beta_j sin(W_j^T x + b_j):")
    print(f"     top {k_top} terms: E(top {k_top})/E(total) = {energy_K_top:.3f};"
          f"  K_{int(energy_target*100):02d} = {K_target}/{N} features for "
          f"{int(energy_target*100)}% energy")
    print(f"     effective bandwidth (energy-weighted mean |W|) = "
          f"{bandwidth:.3f}")
    top_terms = []
    for rank, j in enumerate(top):
        if W.ndim == 2:
            Wj = W[:, j].ravel()
        else:
            Wj = np.array([W[j]])
        bj = float(b.ravel()[j])
        bej = float(beta[j])
        if Wj.size == 1:
            print(f"       {rank+1:2d}. beta = {bej:+.4f}, "
                  f"W = {Wj[0]:+.4f}, b = {bj:+.4f}")
        else:
            wstr = "[" + ", ".join(f"{w:+.3f}" for w in Wj) + "]"
            print(f"       {rank+1:2d}. beta = {bej:+.4f}, "
                  f"W = {wstr}, b = {bj:+.4f}")
        top_terms.append({"beta": bej, "W": Wj.tolist(), "b": bj})
    print()
    return {"top_terms":     top_terms,
            "energy_K_top":  energy_K_top,
            "K_target":      K_target,
            "bandwidth":     bandwidth,
            "N":             N}
