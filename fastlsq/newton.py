# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Newton-Raphson iteration for nonlinear PDEs solved with Fast-LSQ.

Includes:
  - Tikhonov-regularised Newton steps
  - Backtracking line search with Armijo condition
  - Continuation / homotopy for advection-dominated problems
"""

import torch
import numpy as np

from fastlsq.solvers import FastLSQSolver
from fastlsq.linalg import solve_lstsq
from fastlsq.utils import device


# ======================================================================
# Builder helper
# ======================================================================

def build_solver_with_scale(input_dim, scale, n_blocks=3, hidden=500):
    """Create a normalised FastLSQSolver with zero initial weights."""
    solver = FastLSQSolver(input_dim, normalize=True)
    for _ in range(n_blocks):
        solver.add_block(hidden_size=hidden, scale=scale)
    solver.beta = torch.zeros(solver.n_features, 1, device=device)
    return solver


def get_initial_guess(solver, problem, x_pde, bcs, f_pde, mu=1e-10):
    """Solve the linear part of the PDE as a warm start for Newton."""
    if hasattr(problem, "build_linear_init"):
        A, b = problem.build_linear_init(solver, x_pde, bcs, f_pde)
        solver.beta = solve_lstsq(A, b, mu=mu)


# ======================================================================
# Newton-Raphson driver
# ======================================================================

def newton_solve(solver, problem, x_pde, bcs, f_pde,
                 max_iter=30, tol_res=1e-12, tol_du=1e-13,
                 damping=1.0, mu=1e-10, verbose=True):
    """Newton-Raphson iteration with Tikhonov-regularised Fast-LSQ steps.

    Convergence is checked via two criteria (both must be small):

    1. Residual norm:             ||R|| < tol_res
    2. Relative solution change:  ||du|| / ||u|| < tol_du
       computed at collocation points (not via ||d_beta||).

    Parameters
    ----------
    solver : FastLSQSolver
    problem : object
        Must implement ``build_newton_step(solver, x_pde, bcs, f_pde)``.
    x_pde, bcs, f_pde :
        Training data returned by ``problem.get_train_data()``.
    max_iter : int
    tol_res, tol_du : float
    damping : float
        Initial step size for backtracking.
    mu : float
        Tikhonov regularisation parameter.
    verbose : bool

    Returns
    -------
    history : list[dict]
    """
    history = []

    for it in range(max_iter):
        J, neg_R = problem.build_newton_step(solver, x_pde, bcs, f_pde)
        res_norm = torch.norm(neg_R).item()

        delta_beta = solve_lstsq(J, neg_R, mu=mu)

        # Solution-level change (meaningful convergence metric)
        H_pde, _, _ = solver.get_features(x_pde)
        u_current = H_pde @ solver.beta
        du = H_pde @ delta_beta
        u_norm = torch.norm(u_current).item()
        du_norm = torch.norm(du).item()
        rel_du = du_norm / max(u_norm, 1e-15)

        # Backtracking line search (Armijo-like)
        alpha = damping
        beta_old = solver.beta.clone()

        for _ in range(10):
            solver.beta = beta_old + alpha * delta_beta
            _, new_neg_R = problem.build_newton_step(solver, x_pde, bcs, f_pde)
            new_res = torch.norm(new_neg_R).item()
            if new_res < res_norm * (1.0 - 1e-4 * alpha) + 1e-15:
                break
            alpha *= 0.5
        else:
            solver.beta = beta_old + alpha * delta_beta

        history.append({
            "iter": it, "residual": res_norm,
            "rel_du": rel_du, "du_norm": du_norm,
            "step_size": alpha,
        })

        if verbose:
            print(f"  Newton {it:2d}: |R|={res_norm:.2e}  "
                  f"|du|/|u|={rel_du:.2e}  alpha={alpha:.3f}")

        if res_norm < tol_res and rel_du < tol_du:
            if verbose:
                print(f"  Converged in {it + 1} iterations "
                      f"(|R|={res_norm:.1e}, |du|/|u|={rel_du:.1e})")
            break

        if res_norm < tol_res * 0.01:
            if verbose:
                print(f"  Residual converged in {it + 1} iterations "
                      f"(|R|={res_norm:.1e})")
            break

    return history


# ======================================================================
# Continuation / homotopy driver
# ======================================================================

def continuation_solve(solver, problem, x_pde, bcs, f_pde_final,
                       param_name, param_schedule,
                       max_newton_per_step=15, mu=1e-10, verbose=True):
    """Solve a sequence of problems with gradually increasing nonlinearity.

    At each stage, the previous solution is used as the initial guess
    for Newton on the next (harder) parameter value.

    The problem object must support:
      - ``setattr(problem, param_name, value)``
      - ``problem.source(x)``  (recomputes f for the new parameter)
      - ``problem.build_newton_step(...)``
    """
    all_history = []

    for step_idx, param_val in enumerate(param_schedule):
        setattr(problem, param_name, param_val)
        f_pde = problem.source(x_pde)

        if verbose:
            print(f"\n  --- Continuation step {step_idx + 1}/"
                  f"{len(param_schedule)}: {param_name}={param_val} ---")

        if step_idx == 0 and torch.norm(solver.beta).item() < 1e-10:
            get_initial_guess(solver, problem, x_pde, bcs, f_pde, mu=mu)

        history = newton_solve(
            solver, problem, x_pde, bcs, f_pde,
            max_iter=max_newton_per_step, mu=mu, verbose=verbose,
        )
        all_history.extend(history)

    return all_history
