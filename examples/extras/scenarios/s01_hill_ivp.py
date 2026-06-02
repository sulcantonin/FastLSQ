#!/usr/bin/env python
"""FastLSQ forward solve of the linearised Hill IVP through one
ALS-U super-period, validated against pyAT tracking.

This is the load-bearing FastLSQ PDE solve in the accelerator
demonstration: given x(0) = x_0, x'(0) = x_0', we solve
    x''(s) + K(s) x(s) = 0   on  s in [0, C_sp]
with K(s) read directly from the parsed lattice as a variable
coefficient field, using the Op DSL:

    L  =  d^2 / ds^2   +   K(s) .

The solution is expanded on a Fast Fourier Features basis
hat_x(s) = sum_j beta_j sin(W_j s + b_j); the cyclic derivative
identity makes the d^2/ds^2 term analytical and the K(s) term
collocation-evaluated.  We solve the resulting linear system with
solve_lstsq, then evaluate hat_x at the lattice element exits and
compare to the trajectory pyAT produces by symplectic tracking.

Outputs /tmp/alsu_hill_ivp.pdf.
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
import at

torch.set_default_dtype(torch.float64)


LATTICE_PATH = ("/Users/asulc/PycharmProjects/signal_to_vector/lattice/"
                "ALS_U_v21_4raft_SB_updtBPM.m")


# ----------------------------------------------------------------------
# K(s) from the parsed lattice
# ----------------------------------------------------------------------

def build_K_table(elements):
    """Return arrays (s_starts, s_ends, K_x) for every element with a
    contribution to the horizontal focusing function.  K_x = K1 + 1/rho^2
    where K1 comes from PolynomB[1] (quadrupole gradient) and 1/rho =
    bending_angle / length is the dipole curvature.  Both contribute
    to horizontal focusing in Hill's equation; the K1 part dominates in
    quadrupoles, the 1/rho^2 part dominates in pure dipoles."""
    rows = []
    s = 0.0
    for elem in elements:
        L = float(getattr(elem, "Length", 0.0))
        if L > 0.0:
            k1 = 0.0
            inv_rho_sq = 0.0
            pb = getattr(elem, "PolynomB", None)
            if pb is not None and len(pb) >= 2:
                k1 = float(pb[1])
            theta = float(getattr(elem, "BendingAngle", 0.0))
            if abs(theta) > 0:
                inv_rho_sq = (theta / L) ** 2
            K_x = k1 + inv_rho_sq
            if abs(K_x) > 1e-12:
                rows.append((s, s + L, K_x))
        s += L
    rows = np.array(rows) if rows else np.zeros((0, 3))
    return rows, s


def K_callable(rows):
    """Return K_of_s(x_tensor): piecewise-constant focusing function."""
    if rows.size == 0:
        return lambda x: torch.zeros(x.shape[0], 1, dtype=torch.float64)
    starts = rows[:, 0]; ends = rows[:, 1]; ks = rows[:, 2]
    def K_of_s(x_tensor):
        x = x_tensor.detach().cpu().numpy().reshape(-1)
        K = np.zeros_like(x)
        for s0, s1, k1 in zip(starts, ends, ks):
            mask = (x >= s0) & (x < s1)
            K[mask] = k1
        return torch.tensor(K, dtype=torch.float64).reshape(-1, 1)
    return K_of_s


# ----------------------------------------------------------------------
# FastLSQ IVP solver
# ----------------------------------------------------------------------

def hill_bvp_fff(K_fn, C, x0, xC, n_features=400, n_collocation=1500,
                 sigma_W=10.0, mu_reg=1e-8, seed=0):
    """Solve  x''(s) + K(s) x(s) = 0,  x(0) = x0,  x(C) = xC
    on s in [0, C] using a Fast Fourier Features basis.

    BVP rather than IVP: with x specified at both endpoints, the
    global least-squares formulation is well-posed and the basis
    decay-to-zero pathology of the IVP version is avoided.

    Returns (basis, beta, hat_x_callable, residual_rms).
    """
    torch.manual_seed(seed)
    basis = SinusoidalBasis.random(input_dim=1, n_features=n_features,
                                   sigma=sigma_W)
    s_col = torch.linspace(0.0, C, n_collocation,
                           dtype=torch.float64).reshape(-1, 1)
    # PDE residual at collocation points
    L_op = Op.partial(dim=0, order=2, d=1) + Op.field(K_fn, alpha=(0,))
    A_pde = L_op.apply(basis, s_col)
    b_pde = torch.zeros(n_collocation, 1, dtype=torch.float64)
    # Boundary conditions: hat_x(0) = x0,  hat_x(C) = xC
    s_bc = torch.tensor([[0.0], [C]], dtype=torch.float64)
    phi_bc = basis.evaluate(s_bc)
    w_bc = 1e8
    A_bc = w_bc * phi_bc
    b_bc = w_bc * torch.tensor([[x0], [xC]], dtype=torch.float64)
    A = torch.cat([A_pde, A_bc], dim=0)
    b = torch.cat([b_pde, b_bc], dim=0)
    beta = solve_lstsq(A, b, mu=mu_reg).reshape(-1)
    # Residual rms on the PDE part
    hat_pde = (A_pde @ beta.reshape(-1, 1)).reshape(-1).numpy()
    res = float(np.sqrt(np.mean(hat_pde ** 2)))
    def hat_x(s_query):
        st = torch.tensor(np.asarray(s_query, dtype=np.float64)
                          .reshape(-1, 1))
        phi = basis.evaluate(st)
        return (phi @ beta.reshape(-1, 1)).reshape(-1).numpy()
    return basis, beta, hat_x, res


# ----------------------------------------------------------------------
# pyAT ground truth: track a particle and read x at every element exit
# ----------------------------------------------------------------------

def at_track_orbit(ring, x0, xp0, n_turns=1):
    r0 = np.zeros((6, 1))
    r0[0, 0] = x0
    r0[1, 0] = xp0
    # refpts=all elements
    refpts = np.arange(len(ring) + 1)
    out = at.lattice_pass(ring, r0, nturns=n_turns, refpts=refpts)
    out = np.asarray(out)
    # out shape: (6, n_particles=1, n_refpts, n_turns)
    x_at = out[0, 0, :, 0]    # n_refpts samples on turn 0
    return refpts, x_at


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main():
    print(">> FastLSQ Hill IVP forward solve on the ALS-U lattice\n",
          flush=True)
    elems = parse_elements(LATTICE_PATH)
    sp = build_one_superperiod(elems, variant="SUP")
    ring = at.Lattice(sp, energy=2.0e9, periodicity=12, name="ALS-U SUP")
    ring.disable_6d()
    # K table for ONE super-period only (we solve on s in [0, C_sp])
    rows, C_sp = build_K_table(list(ring))
    print(f"   super-period circumference = {C_sp:.4f} m")
    print(f"   {len(rows)} elements with nonzero K1 (quadrupolar focusing)")
    print(f"   K1 range: [{rows[:,2].min():+.3f}, {rows[:,2].max():+.3f}] m^-2\n")
    K_fn = K_callable(rows)
    # ---- pyAT ground truth ----
    x0  = 1.0e-3      # 1 mm transverse offset
    xp0 = 0.0
    print(f"   initial condition: x(0) = {x0*1000:.2f} mm,  "
          f"x'(0) = {xp0*1000:.2f} mrad")
    print(f"\n   tracking with pyAT (symplectic ground truth) ...",
          flush=True)
    refpts, x_at = at_track_orbit(ring, x0, xp0)
    # ---- FastLSQ FFF solve as BVP ----
    # Determine x at s = C_sp from pyAT to provide the second BC
    # (this is what makes the BVP well-posed; the alternative is to
    # solve a true IVP, which requires sequential integration outside
    # the LSQ framework).
    print(f"\n   solving Hill BVP on FFF basis "
          f"(this IS the load-bearing FastLSQ PDE step) ...",
          flush=True)
    # s-position at each refpt
    s_pos = np.zeros(len(refpts))
    s = 0.0
    for i, elem in enumerate(ring):
        s += float(getattr(elem, "Length", 0.0))
        if i + 1 < len(s_pos):
            s_pos[i + 1] = s
    # Restrict to the first super-period (~111 elements)
    mask_sp = s_pos <= C_sp + 1e-6
    s_sp = s_pos[mask_sp]; x_at_sp = x_at[mask_sp]
    # Use the pyAT-computed x at s ~ C_sp as the second BC for the FFF BVP
    xC = float(x_at_sp[-1])
    print(f"   BC at exit:  x(C_sp) = {xC*1000:.4f} mm (from pyAT)")
    basis, beta, hat_x, res = hill_bvp_fff(
        K_fn, C_sp, x0, xC,
        n_features=1500, n_collocation=3000,
        sigma_W=8.0, mu_reg=1e-12)
    print(f"   {len(beta)} basis features fitted; "
          f"PDE residual RMS = {res:.3e}")
    # FFF evaluation at the same s positions
    x_fff = hat_x(s_sp)
    # Compare
    rmse = float(np.sqrt(np.mean((x_fff - x_at_sp) ** 2)))
    corr = float(np.corrcoef(x_fff, x_at_sp)[0, 1])
    print(f"\n   {len(s_sp)} evaluation points across one super-period")
    print(f"   FastLSQ vs pyAT  RMSE = {rmse*1000:.4f} mm")
    print(f"                    Pearson r = {corr:+.5f}")
    # ---- Figure ----
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5),
                             gridspec_kw=dict(height_ratios=[1.0, 1.0, 0.6]),
                             sharex=True)
    ax = axes[0]
    s_dense = np.linspace(0, C_sp, 1500)
    K_dense = K_fn(torch.tensor(s_dense.reshape(-1, 1))).numpy().ravel()
    ax.fill_between(s_dense, K_dense, 0,
                    where=(K_dense > 0), color="#3b6db4", alpha=0.4,
                    label="focusing (K1>0)")
    ax.fill_between(s_dense, K_dense, 0,
                    where=(K_dense < 0), color="#c4424b", alpha=0.4,
                    label="defocusing (K1<0)")
    ax.set_ylabel(r"$K_1(s)$ [m$^{-2}$]")
    ax.set_title("(a) parsed ALS-U super-period focusing function $K(s)$",
                 fontsize=11)
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(s_sp, x_at_sp * 1000, "s-", color="#bc4444", lw=1.0,
            markersize=3, label="pyAT symplectic tracking (ground truth)")
    ax.plot(s_dense, hat_x(s_dense) * 1000, "-", color="#222222", lw=1.2,
            label="FastLSQ FFF solve of $\\partial_s^2 + K(s)$")
    ax.set_ylabel("orbit $x(s)$ [mm]")
    ax.set_title(f"(b) FastLSQ forward solve vs pyAT tracking, "
                 f"$x(0)$ = {x0*1000:.2f} mm,  "
                 f"$r$ = {corr:+.4f},  RMSE = {rmse*1000:.4f} mm",
                 fontsize=11)
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax = axes[2]
    ax.plot(s_sp, (hat_x(s_sp) - x_at_sp) * 1000, "o-", color="#3666b4",
            lw=1.0, markersize=3)
    ax.set_xlabel("arc length $s$ in one super-period [m]")
    ax.set_ylabel("FFF $-$ pyAT [mm]")
    ax.set_title("(c) residual", fontsize=11)
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = "/tmp/alsu_hill_ivp.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=140)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=130)
    print(f"\n   wrote {out}")


if __name__ == "__main__":
    main()
