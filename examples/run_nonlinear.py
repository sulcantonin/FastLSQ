#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Nonlinear PDE benchmark: Newton-Raphson Fast-LSQ.

Runs Newton iteration (with Tikhonov regularisation, 1/sqrt(N) feature
normalisation, and continuation for Burgers) on five nonlinear test
problems.  Reports errors, iteration counts, and generates convergence
plots.
"""

import torch
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastlsq.solvers import FastLSQSolver
from fastlsq.linalg import solve_lstsq
from fastlsq.newton import (
    build_solver_with_scale, get_initial_guess,
    newton_solve, continuation_solve,
)
from fastlsq.utils import device, setup, evaluate_error
from fastlsq.problems.nonlinear import (
    NLPoisson2D, Bratu2D, SteadyBurgers1D, NLHelmholtz2D, AllenCahn1D,
)


# ======================================================================
# Newton runner
# ======================================================================

def run_newton_problem(problem, scale, n_blocks=3, hidden=500,
                       max_iter=30, damping=1.0, mu=1e-10, verbose=True):
    """Full pipeline: build solver, warm-start, Newton, measure error."""
    torch.manual_seed(42)
    np.random.seed(42)

    solver = build_solver_with_scale(problem.dim, scale, n_blocks, hidden)
    x_pde, bcs, f_pde = problem.get_train_data()

    t0 = time.time()

    if getattr(problem, "use_continuation", False):
        schedule = list(problem.continuation_schedule)
        if schedule[-1] != problem.nu_target:
            schedule.append(problem.nu_target)
        schedule = [v for v in schedule if v >= problem.nu_target]

        history = continuation_solve(
            solver, problem, x_pde, bcs, f_pde,
            param_name="nu", param_schedule=schedule,
            max_newton_per_step=max_iter // max(len(schedule), 1) + 5,
            mu=mu, verbose=verbose,
        )
        problem.nu = problem.nu_target
    else:
        get_initial_guess(solver, problem, x_pde, bcs, f_pde, mu=mu)
        history = newton_solve(
            solver, problem, x_pde, bcs, f_pde,
            max_iter=max_iter, mu=mu, damping=damping, verbose=verbose,
        )

    total_time = time.time() - t0
    val_err, grad_err = evaluate_error(solver, problem)
    return val_err, grad_err, total_time, len(history), history


def grid_search_newton(problem, scales, n_blocks=3, hidden=500,
                       max_iter=30, damping=1.0, mu=1e-10, verbose=False):
    """Grid search over scales for the Newton solver."""
    best = {"scale": scales[0], "val": float("inf"), "grad": float("inf"),
            "time": 0, "iters": 0}
    for scale in scales:
        try:
            ve, ge, t, ni, _ = run_newton_problem(
                problem, scale, n_blocks, hidden, max_iter, damping,
                mu=mu, verbose=verbose,
            )
            if np.isnan(ve) or np.isinf(ve):
                ve = 1e10
            if ve < best["val"]:
                best = {"scale": scale, "val": ve, "grad": ge,
                        "time": t, "iters": ni}
        except Exception as exc:
            if verbose:
                print(f"    Scale {scale:.1f} failed: {exc}")
    return best


# ======================================================================
# Plotting
# ======================================================================

def plot_convergence(histories, labels, problem_name, filename):
    """Plot Newton convergence: residual and relative solution change."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for hist, label in zip(histories, labels):
        iters = [h["iter"] for h in hist]
        residuals = [h["residual"] for h in hist]
        rel_dus = [h["rel_du"] for h in hist]
        ax1.semilogy(iters, residuals, "-o", label=label, markersize=4)
        ax2.semilogy(iters, rel_dus, "-o", label=label, markersize=4)

    ax1.set_xlabel("Newton Iteration")
    ax1.set_ylabel("Residual norm")
    ax1.set_title(f"{problem_name}: Residual")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Newton Iteration")
    ax2.set_ylabel("Relative solution change")
    ax2.set_title(f"{problem_name}: Convergence")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {filename}")


def run_spectral_sweep(problem, scales, mu=1e-10):
    """Sweep scales and return value/gradient errors."""
    val_errors, grad_errors, runtimes = [], [], []
    print(f"   -> Sweeping scales: {scales}")
    for scale in scales:
        try:
            ve, ge, t, _, _ = run_newton_problem(
                problem, scale, n_blocks=3, hidden=500,
                max_iter=30, damping=1.0, mu=mu, verbose=False,
            )
            if np.isnan(ve) or np.isinf(ve) or ve > 1e5:
                ve = 1.0
            if np.isnan(ge) or np.isinf(ge) or ge > 1e5:
                ge = 1.0
            val_errors.append(ve)
            grad_errors.append(ge)
            runtimes.append(t)
        except Exception:
            val_errors.append(1.0)
            grad_errors.append(1.0)
            runtimes.append(0.0)
    return val_errors, grad_errors, runtimes


def plot_spectral_sensitivity(problem_name, scales, val_errs, grad_errs):
    """Plot error vs scale on a log-log plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(scales, val_errs, "b-o", label=r"Value Error ($L_2$)", linewidth=2)
    ax.plot(scales, grad_errs, "r--s", label=r"Gradient Error ($L_2$)", linewidth=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Scale $\sigma$ (Bandwidth)", fontsize=12)
    ax.set_ylabel("Relative Error", fontsize=12)
    ax.set_title(f"Spectral Sensitivity: {problem_name}", fontsize=14)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend(fontsize=11)
    clean = (problem_name.replace(" ", "_").replace("(", "")
             .replace(")", "").replace("=", ""))
    filename = f"Sensitivity_{clean}.pdf"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"   -> Plot saved: {filename}")


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    setup(dtype=torch.float64)

    problems = [
        NLPoisson2D(),
        Bratu2D(lam=1.0),
        SteadyBurgers1D(nu=0.1),
        NLHelmholtz2D(k=3.0, alpha=0.5),
        AllenCahn1D(eps=0.1),
    ]

    scales = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0]
    MU = 1e-10

    print("\n" + "=" * 110)
    print("  NEWTON-FAST-LSQ: Tikhonov + 1/sqrt(N) normalisation + continuation")
    print("=" * 110)

    all_results = []

    for problem in problems:
        print(f"\n{'=' * 70}")
        print(f">>> {problem.name}")
        print(f"{'=' * 70}")

        best = grid_search_newton(problem, scales, mu=MU, verbose=False)
        print(f"\n  Best scale: sigma = {best['scale']}")
        print(f"  Re-running with verbose output ...\n")

        ve, ge, t, ni, hist = run_newton_problem(
            problem, best["scale"], mu=MU, verbose=True,
        )
        print()
        print(f"  RESULT: L2={ve:.2e}  |grad|={ge:.2e}  "
              f"iters={ni}  time={t:.3f}s")

        all_results.append({
            "name": problem.name, "scale": best["scale"],
            "iters": ni, "time": t, "val_err": ve, "grad_err": ge,
            "history": hist,
        })

        clean = (problem.name.replace(" ", "_").replace("(", "")
                 .replace(")", "").replace("=", "").replace("^", ""))
        plot_convergence(
            [hist], [f"sigma={best['scale']}"],
            problem.name, f"Newton_{clean}.pdf",
        )

    # Summary table
    print("\n\n" + "=" * 110)
    print("SUMMARY: Newton-Fast-LSQ on Nonlinear PDEs")
    print("=" * 110)
    print(f"{'PROBLEM':<28} | {'sigma':<6} | {'ITERS':<6} | {'TIME (s)':<10} | "
          f"{'VALUE L2':<12} | {'GRAD L2':<12}")
    print("-" * 110)
    for r in all_results:
        print(f"{r['name']:<28} | {r['scale']:<6.1f} | {r['iters']:<6d} | "
              f"{r['time']:<10.4f} | {r['val_err']:.2e}     | {r['grad_err']:.2e}")
    print("=" * 110)

    # Comparison: Newton vs Regression
    print("\n\n" + "=" * 110)
    print("COMPARISON: Newton Solver Mode vs Regression Mode")
    print("=" * 110)

    for problem in [NLPoisson2D(), AllenCahn1D(eps=0.1), SteadyBurgers1D(nu=0.1)]:
        print(f"\n>>> {problem.name}")

        best_n = grid_search_newton(problem, scales, mu=MU, verbose=False)
        print(f"  Newton (solver):     L2 = {best_n['val']:.2e}  "
              f"in {best_n['time']:.3f}s  ({best_n['iters']} Newton iters)")

        best_reg_val = float("inf")
        for scale in scales:
            torch.manual_seed(42)
            solver = build_solver_with_scale(problem.dim, scale)
            x_train = torch.rand(5000, problem.dim, device=device)
            u_train = problem.exact(x_train)
            H, _, _ = solver.get_features(x_train)
            solver.beta = solve_lstsq(H, u_train, mu=MU)
            ve, _ = evaluate_error(solver, problem)
            if ve < best_reg_val:
                best_reg_val = ve

        print(f"  Regression (data):   L2 = {best_reg_val:.2e}  "
              f"(uses exact solution as training data)")
