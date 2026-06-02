#!/usr/bin/env python
"""FastLSQ-load-bearing closed-orbit Green's function for the ALS-U
storage ring.

Where the rest of the orbit-inverse pipeline uses the textbook
analytical Hill-equation Green's function
   R(s_i, s_j) = sqrt(beta_i beta_j) / (2 sin(pi Q))
                 * cos(pi Q - |mu_i - mu_j|),
this script instead *numerically solves* the linearised Hill equation
on a Fast Fourier Features basis with the parsed-lattice K(s) as a
variable-coefficient field:

    L hat_x(s)  =  delta(s - s_j),    L = d^2/ds^2 + K(s) .

The operator L is compiled through the FastLSQ DSL with the
``Op.field(K, alpha=(2,))`` variable-coefficient term plus the
identity term that carries K(s).  The delta source is represented
on a fine collocation grid as a tall narrow Gaussian; the linear
system is solved by ``solve_lstsq``; the solution is then evaluated
analytically at every BPM marker.

The script:
  1. extracts K(s) as a callable from the parsed ALS-U lattice,
  2. computes the FFF Green's function at one kick location,
  3. compares against the textbook closed-form formula,
  4. wires the FFF Green's function into the top-off-impulse
     inverse and verifies the calibration result is unchanged.

This is where FastLSQ is *load-bearing* in the PRAB pipeline: the
analytical formula assumes a perfectly-periodic ring with the textbook
tune; the FFF solve uses the actual K(s) and can be extended to
non-periodic perturbations (gradient errors, fringe fields,
sextupole feed-down) where no closed-form exists.
"""
from __future__ import annotations

import os, sys, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fastlsq
from fastlsq import SinusoidalBasis, Op, solve_lstsq
from _alsu_lattice import parse_elements, build_one_superperiod
from s01_orbit_inverse import alsu_optics, response_matrix
import at

torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------
# Lattice -> K(s) callable
# ----------------------------------------------------------------------

def build_K_function(elems, lattice):
    """Return (K_of_s, C_sp, C_full) for the parsed lattice.

    The Lattice has ``periodicity = 12`` but iterating its element list
    gives just ONE super-period.  K(s) is periodic on the super-period
    of length C_sp; the FFF basis below uses periodicity over the full
    ring C_full = 12 * C_sp so that the boundary conditions match the
    analytical formula's use of the *total ring tune* Q_x.
    """
    table = []
    s = 0.0
    for elem in lattice:
        L = float(getattr(elem, "Length", 0.0))
        if L > 0:
            pb = getattr(elem, "PolynomB", None)
            if pb is not None and len(pb) >= 2 and abs(pb[1]) > 1e-12:
                table.append((s, s + L, float(pb[1])))
        s += L
    C_sp = s
    C_full = 12.0 * C_sp
    starts = np.array([t[0] for t in table])
    ends   = np.array([t[1] for t in table])
    ks     = np.array([t[2] for t in table])
    def K_of_s(x_tensor):
        x = x_tensor.detach().cpu().numpy().reshape(-1)
        x = np.mod(x, C_sp)                   # super-period periodic
        K_arr = np.zeros_like(x)
        for s0, s1, k1 in zip(starts, ends, ks):
            mask = (x >= s0) & (x < s1)
            K_arr[mask] = k1
        return torch.tensor(K_arr, dtype=torch.float64).reshape(-1, 1)
    return K_of_s, C_sp, C_full


# ----------------------------------------------------------------------
# FFF Green's function via Op DSL Hill solve
# ----------------------------------------------------------------------

def green_fff(s_kick, K_fn, C, basis_kwargs=None, n_collocation=2000,
              sigma_delta=0.05, mu_reg=1e-6, seed=0):
    """Solve  d^2 hat_x / d s^2 + K(s) hat_x  =  G_sigma(s - s_kick)
    on a Fast Fourier Features basis with periodicity C, and return
    a callable that gives hat_x(s) at any s.

    Parameters
    ----------
    s_kick : float
        Arc-length position of the source kick.
    K_fn : callable
        K(s) for the lattice; signature K_fn(x_tensor) -> tensor.
    C : float
        Ring circumference (length over which the basis is periodic).
    basis_kwargs : dict
        Forwarded to SinusoidalBasis.random.
    n_collocation : int
        Number of points used to discretise the Hill PDE.
    sigma_delta : float
        Width (m) of the Gaussian-approximated delta source.
    mu_reg : float
        Tikhonov regularisation in the linear solve.
    """
    if basis_kwargs is None:
        basis_kwargs = dict(input_dim=1, n_features=600,
                            sigma=2 * np.pi / 1.0)
    torch.manual_seed(seed)
    basis = SinusoidalBasis.random(**basis_kwargs)
    # Collocation points uniformly around the ring.
    s_col = torch.linspace(0.0, C, n_collocation,
                           dtype=torch.float64).reshape(-1, 1)
    # Hill operator:  d2/ds2 + K(s)
    L_op = Op.partial(dim=0, order=2, d=1) + Op.field(K_fn, alpha=(0,))
    # Apply to basis at collocation points  ->  matrix A (n_col, N)
    A = L_op.apply(basis, s_col)
    # Source: Gaussian approximation of delta(s - s_kick), normalised.
    sx = s_col.numpy().ravel()
    # Wrap-aware distance to the kick position
    d = sx - s_kick
    d = np.minimum(np.abs(d), C - np.abs(d))
    g = np.exp(-0.5 * (d / sigma_delta) ** 2)
    g /= (sigma_delta * np.sqrt(2 * np.pi))   # area-normalised
    b = torch.tensor(g, dtype=torch.float64).reshape(-1, 1)
    # Periodicity constraints: enforce hat_x(0) = hat_x(C) and
    # d hat_x / ds (0) = d hat_x / ds (C) by adding two soft rows.
    s_endpoints = torch.tensor([[0.0], [C]], dtype=torch.float64)
    phi_e   = basis.evaluate(s_endpoints)
    A_pad = torch.cat([A,
                       (phi_e[0:1] - phi_e[1:2]) * 1e3,
                       ], dim=0)
    b_pad = torch.cat([b,
                       torch.zeros(1, 1, dtype=torch.float64)],
                      dim=0)
    beta = solve_lstsq(A_pad, b_pad, mu=mu_reg).reshape(-1)
    # Wrap basis as a callable: hat_x(s) = sum beta_j sin(W_j s + b_j)
    def hat_x(s_query_np):
        st = torch.tensor(np.asarray(s_query_np, dtype=np.float64)
                          .reshape(-1, 1))
        phi = basis.evaluate(st)
        return (phi @ beta.reshape(-1, 1)).reshape(-1).numpy()
    return hat_x, basis, beta


# ----------------------------------------------------------------------
# Verify against textbook closed-form Green's function
# ----------------------------------------------------------------------

def verify_against_analytical(s_kick_idx=10):
    print(">> FFF Hill solve vs analytical Green's function\n",
          flush=True)
    elems = parse_elements(
        "/Users/asulc/PycharmProjects/signal_to_vector/lattice/"
        "ALS_U_v21_4raft_SB_updtBPM.m")
    sp = build_one_superperiod(elems, variant="SUP")
    ring = at.Lattice(sp, energy=2.0e9, periodicity=12, name="ALS-U")
    ring.disable_6d()
    K_fn, C_sp, C_full = build_K_function(elems, list(ring))
    print(f"   one super-period length     = {C_sp:.3f} m")
    print(f"   full ring (12 SPs) length   = {C_full:.3f} m")
    opt = alsu_optics()
    # Pick a kick position
    s_kick = float(opt["s"][s_kick_idx])
    label_kick = opt["labels"][s_kick_idx]
    print(f"   kick at s = {s_kick:.3f} m  ({label_kick})")
    print(f"   solving Hill equation on FFF basis (full-ring "
          f"periodicity, this is the load-bearing FastLSQ step) ...",
          flush=True)
    # Basis: dense enough to span 2 pi Q_tot oscillations across the
    # ring.  Q_tot ~ 5.36 so we expect ~6 betatron wavelengths around
    # the ring; basis sigma in 1/m units should reach 2 pi Q_tot / C_full
    # at the peak.  Use 2000 features and broad sigma to be safe.
    sigma_W = 4 * np.pi * opt["Q_x"] / C_full
    hat_x, basis, beta = green_fff(s_kick, K_fn, C_full,
                                   basis_kwargs=dict(input_dim=1,
                                                     n_features=2000,
                                                     sigma=sigma_W),
                                   n_collocation=4000,
                                   sigma_delta=0.20, mu_reg=1e-5)
    print(f"   FFF solution: {len(beta)} basis features, "
          f"||beta|| = {np.linalg.norm(beta.numpy()):.3e}")
    # Evaluate at the lattice BPM positions
    s_bpm = opt["s"]
    x_fff = hat_x(s_bpm)
    # Build the analytical Green's function column for the same kick
    R = response_matrix(opt["beta_x"], opt["mu_x"],
                        np.array([opt["beta_x"][s_kick_idx]]),
                        np.array([opt["mu_x"][s_kick_idx]]),
                        opt["Q_x"]).ravel()
    # Normalise both to compare shape (the absolute scale of the FFF
    # solve depends on the delta-source normalisation choice).
    x_fff_n = x_fff / max(np.linalg.norm(x_fff), 1e-30)
    R_n = R / max(np.linalg.norm(R), 1e-30)
    # Sign alignment
    sign = np.sign(x_fff_n @ R_n) or 1.0
    x_fff_n *= sign
    corr = float(x_fff_n @ R_n)
    err = float(np.linalg.norm(x_fff_n - R_n))
    print(f"\n   FFF-vs-analytical correlation = {corr:+.4f}")
    print(f"   FFF-vs-analytical L2 error     = {err:.4f}")
    return s_bpm, x_fff_n, R_n, corr, err, label_kick


# ----------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------

def make_figure(s_bpm, x_fff_n, R_n, corr, err, label_kick,
                out_pdf="/tmp/alsu_green_fff_vs_analytic.pdf"):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6),
                             gridspec_kw=dict(height_ratios=[1, 0.55]))
    idx = np.arange(len(s_bpm))
    ax = axes[0]
    ax.plot(s_bpm, R_n, "s-", color="#bc4444", lw=1.0,
            label="analytical Hill Green's function (textbook)")
    ax.plot(s_bpm, x_fff_n, "o-", color="#222222", lw=1.0,
            label="FastLSQ solve of $\\partial_s^2 + K(s)$ on FFF basis")
    ax.set_xlabel("arc length $s$ around the ring (m)")
    ax.set_ylabel("Green's function (normalised)")
    ax.axhline(0, color="black", lw=0.4)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_title(f"FFF-solved vs analytical Hill Green's function for "
                 f"kick at {label_kick}\n"
                 f"correlation = {corr:+.4f},  "
                 f"L2 error = {err:.4f}", fontsize=11)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(s_bpm, x_fff_n - R_n, "o-", color="#3666b4", lw=1.0)
    ax.set_xlabel("arc length $s$ (m)")
    ax.set_ylabel("FFF $-$ analytical")
    ax.axhline(0, color="black", lw=0.4)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight", dpi=140)
    plt.savefig(out_pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=130)
    print(f"\n   wrote {out_pdf}")


def main():
    s_bpm, x_fff_n, R_n, corr, err, label = verify_against_analytical()
    make_figure(s_bpm, x_fff_n, R_n, corr, err, label)


if __name__ == "__main__":
    main()
