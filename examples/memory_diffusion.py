#!/usr/bin/env python3
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Integro-differential PDE for free: memory diffusion via the eigenfunction trick.

A material with *perfect memory* of its past fluxes obeys the Gurtin-Pipkin equation -- a
heat equation whose flux integrates the entire spatial-Laplacian history:

        ∂_t u(x, t) = D ∫_0^t Δ_x u(x, s) ds,   u(x, 0) = u_0(x),   u(0,t)=u(1,t)=0.

Differentiating once in t collapses it to  ∂_tt u = D Δu -- the WAVE equation.  So a constant
memory kernel turns diffusion into *propagation*: the solution is a standing wave, not a
decaying bump.  (The "second initial condition" u_t(x,0)=0 is not imposed -- it falls out of
the integral equation at t=0, where ∫_0^0 = 0.)

The point of the demo is HOW the integral term is assembled.  Sinusoidal features are
eigenfunctions of the Laplacian, Δ_x φ_j = -‖W_x,j‖² φ_j, so the spatial Laplacian *inside*
the time-integral is just a per-column rescaling of the closed-form Volterra matrix:

        ∫_0^t Δ_x φ_j ds = -‖W_x,j‖² · ∫_0^t φ_j ds = -‖W_x,j‖² · (Volterra_t φ_j).

We pass that eigenvalue as the (vector) coefficient of one ``IntegralOperator.volterra`` term,
so the whole integro-differential PDE is a single linear-least-squares block -- differential
and integral operators commute trivially because both are diagonal in this basis.

Exact solution (two standing-wave modes):
        u*(x,t) = sin(πx) cos(√D·π t) + ½ sin(3πx) cos(√D·3π t).

Usage:  python memory_diffusion.py
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastlsq import SinusoidalBasis, Op, IntegralOperator, solve_lstsq
from fastlsq.geometry import sample_box

PI = np.pi


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    # ------------------------------------------------------------------
    # Problem (dim 0 = space x, dim 1 = time t); analytic ground truth
    # ------------------------------------------------------------------
    D, T = 1.0, 2.0
    cD = np.sqrt(D)
    u_star = lambda x, t: (torch.sin(PI * x) * torch.cos(cD * PI * t)
                           + 0.5 * torch.sin(3 * PI * x) * torch.cos(cD * 3 * PI * t))
    u0_fn = lambda x: torch.sin(PI * x) + 0.5 * torch.sin(3 * PI * x)

    # ------------------------------------------------------------------
    # Basis + collocation on [0,1] x [0,T]
    # ------------------------------------------------------------------
    basis = SinusoidalBasis.random(input_dim=2, n_features=2000, sigma=5.0, normalize=False)
    XT = sample_box(8000, 2)            # interior, t in [0,1]
    XT[:, 1] *= T                       # rescale time axis to [0, T]

    n_b = 80
    tb = torch.linspace(0, T, n_b).reshape(-1, 1)
    bc0 = torch.cat([torch.zeros(n_b, 1), tb], dim=1)   # x = 0
    bc1 = torch.cat([torch.ones(n_b, 1), tb], dim=1)    # x = 1
    n_ic = 120
    xi = torch.linspace(0, 1, n_ic).reshape(-1, 1)
    ic = torch.cat([xi, torch.zeros(n_ic, 1)], dim=1)   # t = 0

    # ------------------------------------------------------------------
    # Operator:  ∂_t u  -  D ∫_0^t Δ_x u ds   (one integro-differential block)
    #
    # ∫_0^t Δ_x u has per-column coefficient -‖W_x‖² (the Laplacian eigenvalue);
    # multiplying by -D gives the volterra coefficient +D·‖W_x‖² below.
    # ------------------------------------------------------------------
    w_x_sq = basis.W[0:1, :] ** 2                       # (1, N) spatial |W|^2
    op = (Op.partial(1, 1, d=2)                          # ∂_t u
          + (D * w_x_sq) * IntegralOperator.volterra(dim=1, lower=0.0, d=2))
    print("Operator: ∂_t u - D ∫_0^t Δ_x u ds   (D =", D, ")")

    # Assemble [PDE ; weighted BC ; weighted IC] = [0 ; 0 ; weighted u0] and solve once.
    W_BC, W_IC = 50.0, 50.0
    A = torch.cat([
        op.apply(basis, XT),
        W_BC * basis.evaluate(bc0),
        W_BC * basis.evaluate(bc1),
        W_IC * basis.evaluate(ic),
    ])
    b = torch.cat([
        torch.zeros(XT.shape[0], 1),
        torch.zeros(n_b, 1),
        torch.zeros(n_b, 1),
        W_IC * u0_fn(xi),
    ])
    beta = solve_lstsq(A, b, mu=1e-8)

    # ------------------------------------------------------------------
    # Accuracy on a dense space-time grid
    # ------------------------------------------------------------------
    nx, nt = 160, 160
    xs = torch.linspace(0, 1, nx)
    ts = torch.linspace(0, T, nt)
    Xg, Tg = torch.meshgrid(xs, ts, indexing="ij")
    pts = torch.stack([Xg.reshape(-1), Tg.reshape(-1)], dim=1)
    u_pred = (basis.evaluate(pts) @ beta).reshape(nx, nt)
    u_true = u_star(Xg, Tg)
    val_err = (torch.norm(u_pred - u_true) / torch.norm(u_true)).item()
    print(f"  solution rel-L2 error (space-time): {val_err:.2e}")

    # ------------------------------------------------------------------
    # Plot:  exact / FastLSQ / error  as space-time heatmaps
    # ------------------------------------------------------------------
    UP, UT = u_pred.numpy(), u_true.numpy()
    extent = [0, T, 0, 1]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    vlim = np.max(np.abs(UT))
    for ax, dat, title in (
        (axs[0], UT, "exact u*(x,t)"),
        (axs[1], UP, "FastLSQ (one shot)"),
    ):
        im = ax.imshow(dat, origin="lower", aspect="auto", extent=extent,
                       cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        ax.set_title(title); ax.set_xlabel("t"); ax.set_ylabel("x")
        fig.colorbar(im, ax=ax, fraction=0.046)
    im = axs[2].imshow(np.abs(UP - UT), origin="lower", aspect="auto", extent=extent,
                       cmap="magma")
    axs[2].set_title(f"|error|   (rel-L2 = {val_err:.1e})")
    axs[2].set_xlabel("t"); axs[2].set_ylabel("x")
    fig.colorbar(im, ax=axs[2], fraction=0.046)

    fig.suptitle("Memory diffusion  ∂_t u = D ∫₀ᵗ Δu ds  →  standing wave", y=1.03)
    plt.tight_layout()
    out = "memory_diffusion.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")


if __name__ == "__main__":
    main()
