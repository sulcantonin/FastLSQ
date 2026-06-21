#!/usr/bin/env python3
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Integro-differential equation in one shot: exact closed-form *calculus* -> linear LSQ.

FastLSQ solves PDEs by exploiting that every *derivative* of a sinusoidal feature is
closed-form.  The same identity runs backwards -- integration is differentiation of
negative order -- so *integral* terms are closed-form too.  Differential and integral
terms then compose into a single linear-least-squares design matrix.

Here we solve the Volterra integro-differential boundary-value problem

        u'(x) + ∫_0^x u(s) ds = f(x),   u(0) = 0,   x in [0, L],

whose exact solution is u*(x) = sin(w x), giving the (known) forcing

        f(x) = w cos(w x) + (1 - cos(w x)) / w.

The operator is assembled as one object,

        L = Op.partial(0, 1, d=1) + IntegralOperator.volterra(dim=0, lower=0.0, d=1),

stacked with the initial condition, and solved in a single ``solve_lstsq`` call -- no
time stepping, no quadrature, no autodiff.  We then verify a global conservation
identity ∫_0^L u with ``IntegralOperator.definite``.

Usage:  python integro_differential_demo.py
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastlsq import SinusoidalBasis, Op, IntegralOperator, solve_lstsq
from fastlsq.geometry import sample_box


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    # ------------------------------------------------------------------
    # Problem definition (analytic ground truth -> exact forcing)
    # ------------------------------------------------------------------
    w, L = 3.0, 1.0
    u_star = lambda x: torch.sin(w * x)                                  # exact solution
    f_rhs = lambda x: w * torch.cos(w * x) + (1.0 - torch.cos(w * x)) / w  # known forcing
    u0 = 0.0                                                             # initial condition

    # ------------------------------------------------------------------
    # Basis + collocation
    # ------------------------------------------------------------------
    basis = SinusoidalBasis.random(input_dim=1, n_features=800, sigma=5.0, normalize=False)
    x_col = sample_box(4000, 1) * L          # interior collocation points in [0, L]
    x_ic = torch.zeros(1, 1)                  # x = 0 for the initial condition

    # ------------------------------------------------------------------
    # One unified integro-differential operator:  u'(x) + ∫_0^x u(s) ds
    # ------------------------------------------------------------------
    op = Op.partial(0, 1, d=1) + IntegralOperator.volterra(dim=0, lower=0.0, d=1)
    print("Operator:", op)

    # Assemble [PDE rows ; weighted IC row]  =  [f ; weighted u0]  and solve once.
    W_IC = 100.0
    A = torch.cat([op.apply(basis, x_col), W_IC * basis.evaluate(x_ic)])
    b = torch.cat([f_rhs(x_col),           W_IC * torch.full((1, 1), u0)])
    beta = solve_lstsq(A, b, mu=1e-10)

    # ------------------------------------------------------------------
    # Report accuracy on held-out points
    # ------------------------------------------------------------------
    x_test = sample_box(3000, 1) * L
    u_pred = basis.evaluate(x_test) @ beta
    u_true = u_star(x_test)
    val_err = (torch.norm(u_pred - u_true) / (torch.norm(u_true) + 1e-15)).item()

    # Global conservation identity:  ∫_0^L u  (closed form, single functional row)
    int_num = float(IntegralOperator.definite(0, 0.0, L, d=1).apply(basis, x_test[:1]) @ beta)
    int_exact = float((1.0 - np.cos(w * L)) / w)

    print(f"  solution rel-L2 error      : {val_err:.2e}")
    print(f"  ∫_0^L u  (closed form)     : {int_num:.6f}   exact {int_exact:.6f}")

    # ------------------------------------------------------------------
    # Plot:  solution vs exact, and pointwise residual
    # ------------------------------------------------------------------
    xs = torch.linspace(0, L, 400).reshape(-1, 1)
    us = (basis.evaluate(xs) @ beta).squeeze().numpy()
    ue = u_star(xs).squeeze().numpy()
    xg = xs.squeeze().numpy()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
    ax0.plot(xg, ue, "k-", lw=2.5, label="exact  sin(w x)")
    ax0.plot(xg, us, "C1--", lw=1.8, label="FastLSQ")
    ax0.set_title("u'(x) + ∫₀ˣ u ds = f(x),  u(0)=0")
    ax0.set_xlabel("x"); ax0.set_ylabel("u(x)"); ax0.legend()
    ax1.semilogy(xg, np.abs(us - ue) + 1e-18, "C3-")
    ax1.set_title(f"pointwise |error|   (rel-L2 = {val_err:.1e})")
    ax1.set_xlabel("x"); ax1.set_ylabel("|u_pred - u*|")
    plt.tight_layout()
    out = "integro_differential_demo.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")


if __name__ == "__main__":
    main()
