#!/usr/bin/env python3
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
PDE Discovery via Fast-LSQ + Sparse Regression (SINDy-style).

Demonstrates how the analytical derivative structure of sinusoidal random
features can be used to discover the governing PDE of an unknown system
directly from data.  Because every differential operator evaluates to a
simple closed-form expression on sin(W^T x + b), building a dictionary
of candidate terms (u, u_x, u_xx, u^2, ...) costs O(1) per entry.

The workflow:
  1. Fit the data with a Fast-LSQ model (Tikhonov-regularised).
  2. Use ``SinusoidalBasis.derivative()`` to analytically evaluate a
     dictionary of differential operators -- no autodiff or finite diffs.
  3. Apply LASSO (L1-regularised regression) to discover which terms are
     active, recovering the governing equation in symbolic form.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

from fastlsq.basis import SinusoidalBasis
from fastlsq.linalg import solve_lstsq
from fastlsq.utils import device


def main():
    torch.set_default_dtype(torch.float64)

    # ---------------------------------------------------------
    # 1. Generate Synthetic Data: Damped Harmonic Oscillator
    #    True equation: u_xx = -4.25*u - 1.0*u_x
    # ---------------------------------------------------------
    M = 500
    x_np = np.linspace(0, 2 * np.pi, M)
    x = torch.tensor(x_np, device=device).reshape(-1, 1)

    u_true = torch.exp(-0.5 * x) * torch.sin(2 * x)

    torch.manual_seed(42)
    u_noisy = u_true + torch.randn_like(u_true) * 0.02

    # ---------------------------------------------------------
    # 2. Fit Data with SinusoidalBasis
    # ---------------------------------------------------------
    N = 400
    sigma = 4.0
    basis = SinusoidalBasis.random(
        input_dim=1, n_features=N, sigma=sigma, normalize=True,
    )

    H = basis.evaluate(x)
    mu = 1e-3
    beta = solve_lstsq(H, u_noisy, mu=mu)

    # ---------------------------------------------------------
    # 3. Build Analytical Derivative Dictionary
    #    Uses SinusoidalBasis.derivative() -- the core of Fast-LSQ.
    #    Each call is O(1): no autodiff, no finite differences.
    # ---------------------------------------------------------
    cache = basis.cache(x)

    u_hat = (basis.evaluate(x, cache=cache) @ beta).flatten()

    # First derivative via cyclic identity: d/dx sin(wx+b) = w*cos(wx+b)
    u_x_hat = (basis.derivative(x, alpha=(1,), cache=cache) @ beta).flatten()

    # Second derivative: d^2/dx^2 sin(wx+b) = -w^2*sin(wx+b)
    u_xx_hat = (basis.derivative(x, alpha=(2,), cache=cache) @ beta).flatten()

    # Nonlinear candidate term (u^2) to test sparsity
    u_sq_hat = u_hat ** 2

    # Move to numpy for sklearn
    Theta = torch.stack([u_hat, u_x_hat, u_sq_hat], dim=1).detach().cpu().numpy()
    Y_target = u_xx_hat.detach().cpu().numpy()
    feature_names = ["u", "u_x", "u^2"]

    # ---------------------------------------------------------
    # 4. Sweep LASSO Penalty & Track Discovered Equations
    # ---------------------------------------------------------
    alphas = np.logspace(-5, 1, 100)
    losses = []
    equations = []

    for alpha in alphas:
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=20000)
        model.fit(Theta, Y_target)

        preds = model.predict(Theta)
        mse = mean_squared_error(Y_target, preds)
        losses.append(mse)

        eq_str = "u_xx = "
        terms = []
        for coef, name in zip(model.coef_, feature_names):
            if abs(coef) > 1e-2:
                terms.append(f"{coef:.2f}*{name}")

        eq_str += " + ".join(terms) if terms else "0"
        equations.append(eq_str)

    # ---------------------------------------------------------
    # 5. Visualisation
    # ---------------------------------------------------------
    u_hat_np = u_hat.detach().cpu().numpy()
    u_true_np = u_true.flatten().detach().cpu().numpy()
    u_noisy_np = u_noisy.flatten().detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(x_np, u_true_np, "k-", lw=2, label="True solution")
    ax.scatter(x_np[::5], u_noisy_np[::5],
               s=8, alpha=0.4, color="gray", label="Noisy data")
    ax.plot(x_np, u_hat_np, "b--", lw=1.5, label="Fast-LSQ fit")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("u(x)", fontsize=12)
    ax.set_title("Step 1: Fit Data with SinusoidalBasis", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(alphas, losses, color="blue", linewidth=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("LASSO Penalty (Alpha)", fontsize=12)
    ax.set_ylabel("Mean Squared Error", fontsize=12)
    ax.set_title("Step 2: Discover PDE via Sparse Regression", fontsize=13)

    unique_eqs: set[str] = set()
    for a, l, eq in zip(alphas, losses, equations):
        eq_structure = "".join(
            c for c in eq if c.isalpha() or c == "_" or c == "^"
        )
        if eq_structure not in unique_eqs:
            unique_eqs.add(eq_structure)
            ax.scatter(a, l, color="red", zorder=5, s=60)
            ax.annotate(
                eq, (a, l),
                textcoords="offset points", xytext=(-10, 15),
                ha="right", fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="white", ec="gray", alpha=0.9,
                ),
            )

    ax.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("pde_discovery.png", dpi=150, bbox_inches="tight")
    print("Saved: pde_discovery.png")
    plt.show()

    best_idx = np.argmin(losses)
    print(f"\nBest equation (alpha={alphas[best_idx]:.2e}):")
    print(f"  {equations[best_idx]}")
    print(f"  True: u_xx = -4.25*u + -1.00*u_x")


if __name__ == "__main__":
    main()
