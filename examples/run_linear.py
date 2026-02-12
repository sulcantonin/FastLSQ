#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Linear PDE benchmark: Fast-LSQ (sin) vs PIELM (tanh).

For each problem the script performs a grid search over the bandwidth
parameter sigma, compares both solvers in terms of value and gradient
accuracy, and saves sensitivity plots to PDF.
"""

import torch
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastlsq.solvers import FastLSQSolver, PIELMSolver
from fastlsq.linalg import solve_lstsq
from fastlsq.utils import device, setup
from fastlsq.problems.linear import (
    PoissonND, HeatND, Wave1D, Wave2D_MS, Helmholtz2D, Maxwell2D_TM,
)
from fastlsq.problems.regression import (
    Burgers1D_Regression, KdV_Regression, ReactionDiffusion_Regression,
    SineGordon_Regression, KleinGordon_Regression, GrayScott_Pulse,
    NavierStokes2D_Kovasznay, Bratu2D_Regression, NLHelmholtz2D_Regression,
)


# ======================================================================
# Helpers
# ======================================================================

def run_solver_with_scale(solver_class, problem, scale, seed=42):
    """Run a single solver configuration and return (time, val_err, grad_err)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    solver = solver_class(problem.dim)

    # Unpack training data (2-tuple or 3-tuple)
    data = problem.get_train_data()
    if len(data) == 3:
        x_pde, bcs, f_pde = data
        build_args = (bcs, f_pde)
    else:
        x_pde, bcs = data
        build_args = (bcs,)

    # Anisotropic scaling when requested
    if hasattr(problem, "scale_multipliers"):
        effective_scale = [scale * m for m in problem.scale_multipliers]
    else:
        effective_scale = scale

    start = time.time()
    for _ in range(3):
        solver.add_block(hidden_size=500, scale=effective_scale)

    A, b = problem.build(solver, x_pde, *build_args)
    solver.beta = solve_lstsq(A, b)
    runtime = time.time() - start

    # Evaluate
    torch.manual_seed(999)
    xt = problem.get_test_points(2000)
    u_true = problem.exact(xt)
    grad_true = problem.exact_grad(xt)
    u_pred, grad_pred = solver.predict_with_grad(xt)

    val_l2 = (torch.norm(u_pred - u_true) / (torch.norm(u_true) + 1e-8)).item()
    norm_gt = torch.norm(grad_true)
    if norm_gt < 1e-6:
        norm_gt = torch.tensor(1.0)
    grad_l2 = (torch.norm(grad_pred - grad_true) / norm_gt).item()

    return runtime, val_l2, grad_l2


def grid_search_scale(solver_class, problem, scales, n_trials=3):
    best_scale = scales[0]
    best_val_error = float("inf")
    best_grad_error = float("inf")
    best_runtime = 0.0
    results = {}

    for scale in scales:
        val_errors, grad_errors, runtimes = [], [], []
        for seed in range(n_trials):
            try:
                rt, v_err, g_err = run_solver_with_scale(
                    solver_class, problem, scale, seed=seed
                )
                if np.isnan(v_err) or np.isinf(v_err):
                    v_err = 1e10
                if np.isnan(g_err) or np.isinf(g_err):
                    g_err = 1e10
                val_errors.append(v_err)
                grad_errors.append(g_err)
                runtimes.append(rt)
            except Exception:
                val_errors.append(1e10)
                grad_errors.append(1e10)
                runtimes.append(0.0)

        mean_val = np.mean(val_errors)
        mean_grad = np.mean(grad_errors)
        mean_runtime = np.mean(runtimes)
        results[scale] = (mean_val, mean_grad)

        if mean_val < best_val_error:
            best_val_error = mean_val
            best_grad_error = mean_grad
            best_scale = scale
            best_runtime = mean_runtime

    return best_scale, best_val_error, best_grad_error, best_runtime, results


# ======================================================================
# Plotting
# ======================================================================

def plot_and_save_sensitivity(problem_name, scales, results_rff, results_pielm):
    """Plot error-vs-scale sensitivity and save to PDF."""
    val_rff = [results_rff[s][0] for s in scales]
    grad_rff = [results_rff[s][1] for s in scales]
    val_pielm = [results_pielm[s][0] for s in scales]
    grad_pielm = [results_pielm[s][1] for s in scales]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(scales, val_rff, "b-o", label="Fast-LSQ (sin)", linewidth=2, markersize=6)
    ax1.plot(scales, val_pielm, "r--s", label="PIELM (tanh)", linewidth=2, markersize=6)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel(r"Scale (Frequency Bandwidth $\sigma$)")
    ax1.set_ylabel("Relative L2 Error (Value)")
    ax1.set_title(f"{problem_name}: Function Approx.")
    ax1.legend(); ax1.grid(True, which="both", ls="-", alpha=0.3)

    ax2.plot(scales, grad_rff, "b-o", label="Fast-LSQ (sin)", linewidth=2, markersize=6)
    ax2.plot(scales, grad_pielm, "r--s", label="PIELM (tanh)", linewidth=2, markersize=6)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel(r"Scale (Frequency Bandwidth $\sigma$)")
    ax2.set_ylabel("Relative L2 Error (Gradient)")
    ax2.set_title(f"{problem_name}: Derivative Accuracy")
    ax2.legend(); ax2.grid(True, which="both", ls="-", alpha=0.3)

    plt.tight_layout()
    clean_name = problem_name.replace(" ", "_").replace("/", "_")
    filename = f"Sensitivity_{clean_name}.pdf"
    plt.savefig(filename)
    plt.close()
    print(f"   -> Saved plot: {filename}")


# ======================================================================
# Main
# ======================================================================

def run_fair_comparison():
    problems = [
        PoissonND(), HeatND(), Wave1D(), Wave2D_MS(),
        Burgers1D_Regression(), KdV_Regression(),
        ReactionDiffusion_Regression(), Helmholtz2D(),
        SineGordon_Regression(), Maxwell2D_TM(),
        KleinGordon_Regression(), GrayScott_Pulse(),
        NavierStokes2D_Kovasznay(),
        Bratu2D_Regression(), NLHelmholtz2D_Regression(),
    ]

    scales = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
              8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]

    header = (f"{'PROBLEM':<15} | {'METHOD':<15} | {'BEST SCALE':<10} | "
              f"{'TIME (s)':<10} | {'VALUE ERROR':<12} | {'GRAD ERROR':<12}")
    print(f"\n{'=' * 115}")
    print(header)
    print(f"{'=' * 115}")

    for problem in problems:
        s_rff, v_rff, g_rff, t_rff, res_rff = grid_search_scale(
            FastLSQSolver, problem, scales
        )
        print(f"{problem.name:<15} | {'Fast-LSQ (sin)':<15} | "
              f"{s_rff:<10.1f} | {t_rff:<10.4f} | {v_rff:.2e}     | {g_rff:.2e}")

        s_pielm, v_pielm, g_pielm, t_pielm, res_pielm = grid_search_scale(
            PIELMSolver, problem, scales
        )
        print(f"{'':<15} | {'PIELM (tanh)':<15} | "
              f"{s_pielm:<10.1f} | {t_pielm:<10.4f} | {v_pielm:.2e}     | {g_pielm:.2e}")

        plot_and_save_sensitivity(problem.name, scales, res_rff, res_pielm)
        print("-" * 115)


if __name__ == "__main__":
    setup(dtype=torch.float32)
    run_fair_comparison()
