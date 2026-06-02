#!/usr/bin/env python
"""Galactic-potential discovery on **real** rotation curves from SPARC.

The Spitzer Photometry and Accurate Rotation Curves (SPARC) database
[Lelli, McGaugh, Schombert 2016, AJ 152, 157] contains observed rotation
curves V_obs(R) for 175 disk galaxies, measured from HI 21-cm and
H-alpha emission lines.  We pull the curve table via VizieR (no API key,
public).

For a star on a circular orbit at radius R, centripetal balance gives
        a(R) = - V_c(R)^2 / R   (radial, inward),
where a(R) = -d Phi / dR.  Given (R_i, V_obs_i) for one galaxy we
reconstruct -d Phi / dR on the data, then identify which classical
spherical-potential family best fits:

    Plummer       Phi = -M / sqrt(r^2 + a^2)
    Hernquist     Phi = -M / (r + a)
    NFW           Phi = -M ln(1 + r/a) / r
    Logarithmic   Phi = 0.5 v^2 ln(r^2 + a^2)

The first step uses fastlsq's analytical-derivative basis to fit Phi(r)
as a smooth function (the linear system has design rows
A_ij = -basis.derivative(r_i, alpha=(1,))_j).  The second step searches
the (log M, log a) plane for each candidate family and ranks by BIC.

Usage:  python gaia_potential_fastlsq.py
"""
from __future__ import annotations

import os
import subprocess
import time
import numpy as np
import torch

from fastlsq import SinusoidalBasis, Op, solve_lstsq
from spectral_expansion import spectral_expansion_report

torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------
# SPARC fetcher (VizieR)
# ----------------------------------------------------------------------

SPARC_URL = (
    "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
    "?-source=J/AJ/152/157/table2"
    "&-out=Name,Rad,Vobs,e_Vobs&-out.max=unlimited"
)


def fetch_sparc(cache="/tmp/sparc.tsv", timeout=30):
    if not os.path.exists(cache) or os.path.getsize(cache) < 10_000:
        subprocess.run(["curl", "-sL", "--max-time", str(timeout),
                        SPARC_URL, "-o", cache], check=True)
    rows = []
    with open(cache) as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#"):
                continue
            if ln.startswith("Name") or ln.startswith("-") or "kpc" in ln:
                continue
            parts = ln.split("\t")
            if len(parts) < 4:
                continue
            try:
                name = parts[0].strip()
                r    = float(parts[1])
                v    = float(parts[2])
                ev   = float(parts[3])
            except ValueError:
                continue
            if r <= 0 or v <= 0:
                continue
            rows.append((name, r, v, ev))
    by_galaxy = {}
    for name, r, v, ev in rows:
        by_galaxy.setdefault(name, []).append((r, v, ev))
    return {n: np.asarray(rs) for n, rs in by_galaxy.items()}


# ----------------------------------------------------------------------
# Candidate spherical potentials: a(r) = -d Phi / dr
# ----------------------------------------------------------------------

def acc_plummer(r, logM, loga):
    M = np.exp(logM); a = np.exp(loga)
    return -M * r / (r ** 2 + a ** 2) ** 1.5


def acc_hernquist(r, logM, loga):
    M = np.exp(logM); a = np.exp(loga)
    return -M / (r + a) ** 2


def acc_nfw(r, logM, loga):
    M = np.exp(logM); a = np.exp(loga)
    return -M * (1.0 / (r * (r + a)) - np.log(1 + r / a) / r ** 2)


def acc_log(r, logv, loga):
    v = np.exp(logv); a = np.exp(loga)
    return -(v ** 2) * r / (r ** 2 + a ** 2)


# Parameter search ranges in log-units (broad enough to cover SPARC scales).
# M and a are in units consistent with v^2 in (km/s)^2 and r in kpc; the
# scoring is invariant under rescaling, we just need the search box to
# bracket each family's optimum.
PRIMS = [
    ("Plummer",   acc_plummer,   ( 6.0, 13.0), (-1.0, 3.0)),
    ("Hernquist", acc_hernquist, ( 6.0, 13.0), (-1.0, 3.0)),
    ("NFW",       acc_nfw,       ( 6.0, 14.0), ( 0.0, 4.0)),
    ("LogPot",    acc_log,       ( 4.0, 6.5),  (-1.0, 3.0)),
]


# ----------------------------------------------------------------------
# Nonparametric Phi_hat via fastlsq
# ----------------------------------------------------------------------

def fit_potential_fastlsq(r, a_obs, n_features=200, sigma=0.4, mu_reg=1e-4,
                          seed=0):
    torch.manual_seed(seed)
    basis = SinusoidalBasis.random(input_dim=1, n_features=n_features,
                                   sigma=sigma)
    r_t = torch.tensor(r, dtype=torch.float64).reshape(-1, 1)
    dPhi = Op.partial(dim=0, order=1, d=1)
    A = -dPhi.apply(basis, r_t)
    y = torch.tensor(a_obs, dtype=torch.float64).reshape(-1, 1)
    beta = solve_lstsq(A, y, mu=mu_reg).reshape(-1)
    a_hat = (A @ beta.reshape(-1, 1)).reshape(-1).numpy()
    rmse = float(np.sqrt(np.mean((a_hat - a_obs) ** 2)))
    return basis, beta, rmse


# ----------------------------------------------------------------------
# Interpretability: inferred density profile from the *second* derivative.
# ----------------------------------------------------------------------
#
# In spherical symmetry the Poisson equation reads
#     4 pi G rho(r) = nabla^2 Phi = Phi''(r) + (2/r) Phi'(r).
# Both Phi'(r) and Phi''(r) come from the cyclic identity at zero cost on
# the sinusoidal basis -- the order-2 derivative is just the order-1
# derivative with a different phase tag.  After computing rho_hat(r) we
# symbolic-regress it against named density profiles.

def infer_density(basis, beta, r):
    """Return rho_hat(r) up to a positive normalisation (the 4 pi G
    factor cancels in family identification)."""
    r_t = torch.tensor(r, dtype=torch.float64).reshape(-1, 1)
    dPhi  = Op.partial(dim=0, order=1, d=1).apply(basis, r_t)
    d2Phi = Op.partial(dim=0, order=2, d=1).apply(basis, r_t)
    grad = (dPhi  @ beta.reshape(-1, 1)).reshape(-1).numpy()
    hess = (d2Phi @ beta.reshape(-1, 1)).reshape(-1).numpy()
    return hess + (2.0 / r) * grad


# Named density profiles (up to overall normalisation; one scale parameter)
def rho_plummer  (r, log_a): a = np.exp(log_a); return (r ** 2 + a ** 2) ** -2.5
def rho_hernquist(r, log_a): a = np.exp(log_a); return 1.0 / (r * (r + a) ** 3)
def rho_nfw      (r, log_a): a = np.exp(log_a); return 1.0 / (r * (r + a) ** 2)
def rho_burkert  (r, log_a): a = np.exp(log_a); return 1.0 / ((r + a) * (r ** 2 + a ** 2))
def rho_iso      (r, log_a): a = np.exp(log_a); return 1.0 / (r ** 2 + a ** 2)


DENSITIES = [
    ("Plummer-rho",   rho_plummer,   (-1.0, 3.5)),
    ("Hernquist-rho", rho_hernquist, (-1.0, 3.5)),
    ("NFW-rho",       rho_nfw,       (-1.0, 4.5)),
    ("Burkert-rho",   rho_burkert,   (-1.0, 3.5)),
    ("Isothermal-rho",rho_iso,       (-1.0, 3.5)),
]


def fit_density_family(rho_hat, r, n_grid=20, polish=6):
    """For each candidate rho_family(r; a) fit a positive amplitude C >= 0
    by closed-form least squares (one column), search a on a log-grid,
    polish.  Score by log-residual sum of squares because rho spans many
    decades.  Returns (name, log_a, C, log_rss)."""
    pos = rho_hat > 0
    if pos.sum() < 4:
        return []
    r_p   = r[pos]
    y     = np.log(rho_hat[pos])
    scored = []
    for name, fn, ar in DENSITIES:
        grid = np.linspace(*ar, n_grid)
        def loss_at(la):
            x = fn(r_p, la)
            if np.any(x <= 0):
                return np.inf, 0.0
            log_x = np.log(x)
            c_log = float(np.mean(y - log_x))               # best log-amplitude
            resid = y - log_x - c_log
            return float(resid @ resid), c_log
        best = (np.inf, grid[0], 0.0)
        for la in grid:
            l, c_log = loss_at(la)
            if l < best[0]:
                best = (l, la, c_log)
        l_cur, la_cur, c_cur = best
        step = (grid[1] - grid[0]) * 0.5
        for _ in range(polish):
            improved = False
            for s in (-1, +1):
                la_try = la_cur + s * step
                l_try, c_try = loss_at(la_try)
                if l_try < l_cur:
                    l_cur, la_cur, c_cur, improved = l_try, la_try, c_try, True
            if not improved:
                step *= 0.5
        scored.append((name, la_cur, c_cur, l_cur))
    scored.sort(key=lambda s: s[3])
    return scored


# ----------------------------------------------------------------------
# Symbolic identification by grid + polish
# ----------------------------------------------------------------------

def fit_one_primitive(fn, p1r, p2r, r, a_obs, n_grid=14, polish_steps=4):
    p1 = np.linspace(*p1r, n_grid)
    p2 = np.linspace(*p2r, n_grid)
    best = (np.inf, p1[0], p2[0])
    for x in p1:
        for y in p2:
            l = float(np.mean((fn(r, x, y) - a_obs) ** 2))
            if l < best[0]:
                best = (l, x, y)
    cur_l, cur_x, cur_y = best
    dx = (p1[1] - p1[0]) * 0.5
    dy = (p2[1] - p2[0]) * 0.5
    for _ in range(polish_steps):
        for cand in [(cur_x - dx, cur_y), (cur_x + dx, cur_y),
                     (cur_x, cur_y - dy), (cur_x, cur_y + dy)]:
            l = float(np.mean((fn(r, *cand) - a_obs) ** 2))
            if l < cur_l:
                cur_l, cur_x, cur_y = l, *cand
        dx *= 0.5; dy *= 0.5
    return cur_l, cur_x, cur_y


def fit_pair(fn1, p1r_a, p2r_a, fn2, p1r_b, p2r_b, r, a_obs,
             polish_steps=6):
    """Greedy + joint polish for a(r) = fn1(r;.) + fn2(r;.).

    Mirrors the Op DSL: the symbolic answer is a *sum* of two named
    primitives, with their (M, a) pairs jointly polished.  This is the
    multi-component analogue of fit_one_primitive and is what galactic
    dynamicists actually do when they decompose a rotation curve.
    """
    # Step 1: warm-start with best fn1 alone.
    _, x1, y1 = fit_one_primitive(fn1, p1r_a, p2r_a, r, a_obs)
    resid = a_obs - fn1(r, x1, y1)
    # Step 2: best fn2 against the residual.
    _, x2, y2 = fit_one_primitive(fn2, p1r_b, p2r_b, r, resid)
    # Step 3: jointly polish all four parameters by coordinate descent.
    p = [x1, y1, x2, y2]
    def loss_at(q):
        return float(np.mean((fn1(r, q[0], q[1]) + fn2(r, q[2], q[3])
                              - a_obs) ** 2))
    cur_l = loss_at(p)
    step = np.array([(p1r_a[1]-p1r_a[0])/12, (p2r_a[1]-p2r_a[0])/12,
                     (p1r_b[1]-p1r_b[0])/12, (p2r_b[1]-p2r_b[0])/12])
    for _ in range(polish_steps):
        improved = False
        for k in range(4):
            for s in (-1, +1):
                q = p.copy(); q[k] = q[k] + s * step[k]
                l = loss_at(q)
                if l < cur_l:
                    cur_l, p, improved = l, q, True
        if not improved:
            step *= 0.5
    return cur_l, tuple(p)


def bic(loss, n_obs, n_par):
    return n_obs * np.log(max(loss, 1e-30)) + n_par * np.log(max(n_obs, 1))


# ----------------------------------------------------------------------
# Driver: analyse one galaxy
# ----------------------------------------------------------------------

def analyse_galaxy(name, rvc):
    r   = rvc[:, 0]                       # kpc
    v   = rvc[:, 1]                       # km/s
    a_obs = -v ** 2 / r                   # (km/s)^2 / kpc, inward
    t0 = time.perf_counter()
    basis, beta, rmse_np = fit_potential_fastlsq(r, a_obs)
    t_np = time.perf_counter() - t0
    # ---- Interpretability: density profile from Phi'' (free 2nd deriv) ----
    rho_hat = infer_density(basis, beta, r)
    rho_fits = fit_density_family(rho_hat, r)
    # ---- Interpretability: sparse trigonometric expansion of Phi ----
    spec = spectral_expansion_report(basis.W.numpy(), basis.b.numpy(),
                                     beta.numpy(),
                                     label=f"Phi[{name}]",
                                     k_top=5)
    t0 = time.perf_counter()
    # Single-primitive candidates
    scored = []
    for prim_name, fn, pr1, pr2 in PRIMS:
        loss, x, y = fit_one_primitive(fn, pr1, pr2, r, a_obs)
        scored.append((prim_name, loss, bic(loss, len(r), 2), "single"))
    # Sum-of-two candidates -- bulge+halo decompositions a galactic
    # dynamicist would actually consider.  We enumerate ordered pairs
    # (fn_inner, fn_outer) where the inner is a centrally-concentrated
    # primitive (Plummer or Hernquist) and the outer is a flat-curve
    # primitive (LogPot or NFW).  This is the symbolic side of the Op
    # DSL: a sum of two named primitives, jointly fitted.
    inner_idx = [0, 1]                                 # Plummer, Hernquist
    outer_idx = [3, 2]                                 # LogPot, NFW
    for ii in inner_idx:
        for io in outer_idx:
            n_in, fn_in, pr1_in, pr2_in = PRIMS[ii]
            n_ou, fn_ou, pr1_ou, pr2_ou = PRIMS[io]
            loss, _ = fit_pair(fn_in, pr1_in, pr2_in,
                               fn_ou, pr1_ou, pr2_ou, r, a_obs)
            scored.append((f"{n_in}+{n_ou}", loss,
                           bic(loss, len(r), 4), "sum"))
    scored.sort(key=lambda s: s[2])
    t_sym = time.perf_counter() - t0
    return scored, rho_fits, spec, t_np, t_sym, rmse_np, len(r)


def main():
    print(">> Galactic-potential discovery on REAL SPARC rotation curves",
          flush=True)
    data = fetch_sparc()
    print(f"   loaded {len(data)} galaxies from SPARC (Lelli+2016)\n",
          flush=True)

    # Pick a representative mix: bright spirals, classic flat-curve targets,
    # a dwarf, a low-surface-brightness galaxy.  All are real SPARC entries.
    targets = ["NGC3198", "NGC2403", "NGC6503", "NGC3521", "NGC7814",
               "DDO154",  "UGC02953"]
    targets = [t for t in targets if t in data]
    if len(targets) < 5:
        # fall back to any 6 galaxies with >= 12 points
        targets = sorted([n for n, rs in data.items() if len(rs) >= 12])[:6]
    print(f"   running on {len(targets)} galaxies: {targets}\n", flush=True)

    rows = []
    win_kind = {"single": 0, "sum": 0}
    family_counts = {}
    density_counts = {}
    for name in targets:
        scored, rho_fits, spec, t_np, t_sym, rmse_np, npts = analyse_galaxy(
            name, data[name])
        winner = scored[0]
        win_kind[winner[3]] += 1
        family_counts[winner[0]] = family_counts.get(winner[0], 0) + 1
        best_single = next(s for s in scored if s[3] == "single")
        best_sum    = next(s for s in scored if s[3] == "sum")
        delta = best_single[2] - best_sum[2]
        rho_winner = rho_fits[0][0] if rho_fits else "n/a"
        density_counts[rho_winner] = density_counts.get(rho_winner, 0) + 1
        print(f"--- {name}   ({npts} points)")
        print(f"   Potential family (a = -d Phi / d r):")
        for s in scored[:4]:
            print(f"     {s[0]:22s}  BIC={s[2]:7.1f}   [{s[3]}]")
        print(f"   Density profile (rho = nabla^2 Phi, via d^2 Phi / d r^2):")
        for n, la, c_log, l in rho_fits[:3]:
            print(f"     {n:18s}  a~{np.exp(la):.2f} kpc   "
                  f"log10 amp ~ {c_log/np.log(10):+.2f}  "
                  f"log-RSS={l:.2f}")
        print(f"   winner: Phi -> {winner[0]} ({winner[3]});  "
              f"rho -> {rho_winner}    rmse_np={rmse_np:.2e}\n",
              flush=True)
        rows.append({"galaxy": name, "n_pts": npts,
                     "winner": winner[0], "winner_kind": winner[3],
                     "delta_bic": delta,
                     "density": rho_winner,
                     "energy_K5": spec["energy_K_top"],
                     "K_99":      spec["K_target"],
                     "bandwidth": spec["bandwidth"],
                     "rmse_np": rmse_np, "t_np": t_np, "t_sym": t_sym})

    print("Summary:")
    print(f"  {'galaxy':12s} {'n':5s} {'Phi family':24s} {'rho family':16s} "
          f"{'E(top5)':9s} {'K99':5s} {'BW(1/kpc)':10s}")
    for r in rows:
        print(f"  {r['galaxy']:12s} {r['n_pts']:<5d} {r['winner']:24s} "
              f"{r['density']:16s} {r['energy_K5']:<9.3f} "
              f"{r['K_99']:<5d} {r['bandwidth']:<10.3f}")
    print(f"\n  Phi family wins:    {family_counts}")
    print(f"  rho family wins:    {density_counts}")
    print(f"  winner kinds (Phi): {win_kind}")
    return rows, family_counts


if __name__ == "__main__":
    main()
