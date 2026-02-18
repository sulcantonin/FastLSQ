# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
High-level API for solving PDEs with FastLSQ.

This module provides simple, one-line functions to solve linear and nonlinear
PDEs without needing to manually construct solvers, assemble systems, or
handle training data.
"""

import torch
import time
from typing import Optional, Dict, Any, Callable, Tuple

from fastlsq.solvers import FastLSQSolver
from fastlsq.linalg import solve_lstsq
from fastlsq.newton import (
    build_solver_with_scale, get_initial_guess,
    newton_solve, continuation_solve,
)
from fastlsq.utils import device, evaluate_error
from fastlsq.tuning import auto_select_scale


# ======================================================================
# High-level solve functions
# ======================================================================

def solve_linear(
    problem,
    *,
    scale: Optional[float] = None,
    n_blocks: int = 3,
    hidden_size: int = 500,
    n_pde: int = 10000,
    n_bc: int = 2000,
    n_test: int = 5000,
    mu: float = 0.0,
    auto_scale: bool = True,
    auto_scale_trials: int = 5,
    return_solver: bool = False,
    return_metrics: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Solve a linear PDE in one shot.

    Parameters
    ----------
    problem : object
        Problem instance with methods:
        - `get_train_data(n_pde, n_bc)` -> (x_pde, bcs, f_pde)
        - `build(solver, x_pde, bcs, f_pde)` -> (A, b)
        - `exact(x)`, `exact_grad(x)`, `get_test_points(n)`
    scale : float, optional
        Feature bandwidth parameter. If None and `auto_scale=True`,
        automatically selects via grid search.
    n_blocks : int
        Number of feature blocks.
    hidden_size : int
        Features per block.
    n_pde, n_bc : int
        Number of collocation and boundary points.
    n_test : int
        Number of test points for error evaluation.
    mu : float
        Tikhonov regularisation parameter (0 = no regularisation).
    auto_scale : bool
        If True and scale=None, automatically select scale via grid search.
    auto_scale_trials : int
        Number of trials per scale in auto-selection.
    return_solver : bool
        If True, include the solver object in the return dict.
    return_metrics : bool
        If True, compute and return error metrics.
    verbose : bool
        Print progress information.

    Returns
    -------
    result : dict
        Contains:
        - `solver` : FastLSQSolver (if return_solver=True)
        - `u_fn` : callable(x) -> u(x) prediction function
        - `metrics` : dict with 'val_err', 'grad_err', 'runtime' (if return_metrics=True)
        - `scale` : float, the scale used
    """
    t0 = time.time()

    # Auto-select scale if needed
    if scale is None and auto_scale:
        if verbose:
            print("Auto-selecting optimal scale...")
        scale = auto_select_scale(
            problem, solver_class=FastLSQSolver,
            n_blocks=n_blocks, hidden_size=hidden_size,
            n_pde=n_pde, n_bc=n_bc, n_trials=auto_scale_trials,
            verbose=verbose,
        )
        if verbose:
            print(f"Selected scale: {scale:.3f}")

    # Build solver
    solver = FastLSQSolver(problem.dim, normalize=False)
    if hasattr(problem, "scale_multipliers"):
        effective_scale = [scale * m for m in problem.scale_multipliers]
    else:
        effective_scale = scale

    for _ in range(n_blocks):
        solver.add_block(hidden_size=hidden_size, scale=effective_scale)

    # Get training data
    data = problem.get_train_data(n_pde=n_pde, n_bc=n_bc)
    if len(data) == 3:
        x_pde, bcs, f_pde = data
        build_args = (bcs, f_pde)
    else:
        x_pde, bcs = data
        build_args = (bcs,)

    # Assemble and solve
    A, b = problem.build(solver, x_pde, *build_args)
    solver.beta = solve_lstsq(A, b, mu=mu)

    runtime = time.time() - t0

    # Create prediction function
    def u_fn(x):
        """Evaluate solution at points x."""
        if isinstance(x, (list, tuple)):
            x = torch.tensor(x, device=device, dtype=solver.beta.dtype)
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=device, dtype=solver.beta.dtype)
        return solver.predict(x)

    result = {
        "u_fn": u_fn,
        "scale": scale,
        "runtime": runtime,
    }

    if return_solver:
        result["solver"] = solver

    if return_metrics:
        val_err, grad_err = evaluate_error(solver, problem, n_test=n_test)
        result["metrics"] = {
            "val_err": val_err,
            "grad_err": grad_err,
            "runtime": runtime,
        }
        if verbose:
            print(f"Value error: {val_err:.2e}, Gradient error: {grad_err:.2e}")

    return result


def solve_nonlinear(
    problem,
    *,
    scale: Optional[float] = None,
    n_blocks: int = 3,
    hidden_size: int = 500,
    n_pde: int = 5000,
    n_bc: int = 1000,
    n_test: int = 5000,
    max_iter: int = 30,
    tol_res: float = 1e-12,
    tol_du: float = 1e-13,
    damping: float = 1.0,
    mu: float = 1e-10,
    auto_scale: bool = True,
    auto_scale_trials: int = 3,
    return_solver: bool = False,
    return_history: bool = True,
    return_metrics: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Solve a nonlinear PDE via Newton-Raphson iteration.

    Parameters
    ----------
    problem : object
        Problem instance with methods:
        - `get_train_data(n_pde, n_bc)` -> (x_pde, bcs, f_pde)
        - `build_newton_step(solver, x_pde, bcs, f_pde)` -> (J, -R)
        - `exact(x)`, `exact_grad(x)`, `get_test_points(n)`
        - Optional: `build_linear_init(...)`, `use_continuation`, `continuation_schedule`
    scale : float, optional
        Feature bandwidth parameter. If None and `auto_scale=True`,
        automatically selects via grid search.
    n_blocks : int
        Number of feature blocks.
    hidden_size : int
        Features per block.
    n_pde, n_bc : int
        Number of collocation and boundary points.
    n_test : int
        Number of test points for error evaluation.
    max_iter : int
        Maximum Newton iterations.
    tol_res, tol_du : float
        Convergence tolerances (residual norm, relative solution change).
    damping : float
        Initial step size for backtracking line search.
    mu : float
        Tikhonov regularisation parameter for Newton steps.
    auto_scale : bool
        If True and scale=None, automatically select scale via grid search.
    auto_scale_trials : int
        Number of trials per scale in auto-selection.
    return_solver : bool
        If True, include the solver object in the return dict.
    return_history : bool
        If True, include Newton iteration history.
    return_metrics : bool
        If True, compute and return error metrics.
    verbose : bool
        Print progress information.

    Returns
    -------
    result : dict
        Contains:
        - `solver` : FastLSQSolver (if return_solver=True)
        - `u_fn` : callable(x) -> u(x) prediction function
        - `history` : list of iteration dicts (if return_history=True)
        - `metrics` : dict with 'val_err', 'grad_err', 'runtime', 'n_iters' (if return_metrics=True)
        - `scale` : float, the scale used
    """
    t0 = time.time()

    # Auto-select scale if needed
    if scale is None and auto_scale:
        if verbose:
            print("Auto-selecting optimal scale...")
        scale = auto_select_scale(
            problem, solver_class=lambda dim: FastLSQSolver(dim, normalize=True),
            n_blocks=n_blocks, hidden_size=hidden_size,
            n_pde=n_pde, n_bc=n_bc, n_trials=auto_scale_trials,
            verbose=verbose, newton_mode=True,
        )
        if verbose:
            print(f"Selected scale: {scale:.3f}")

    # Build solver
    solver = build_solver_with_scale(problem.dim, scale, n_blocks, hidden_size)

    # Get training data
    x_pde, bcs, f_pde = problem.get_train_data(n_pde=n_pde, n_bc=n_bc)

    # Check for continuation
    if getattr(problem, "use_continuation", False):
        schedule = list(problem.continuation_schedule)
        if schedule[-1] != getattr(problem, "nu_target", None):
            schedule.append(getattr(problem, "nu_target", None))
        schedule = [v for v in schedule if v >= getattr(problem, "nu_target", 0.0)]

        history = continuation_solve(
            solver, problem, x_pde, bcs, f_pde,
            param_name="nu", param_schedule=schedule,
            max_newton_per_step=max_iter // max(len(schedule), 1) + 5,
            mu=mu, verbose=verbose,
        )
        # Restore target parameter
        if hasattr(problem, "nu_target"):
            problem.nu = problem.nu_target
    else:
        # Standard Newton
        get_initial_guess(solver, problem, x_pde, bcs, f_pde, mu=mu)
        history = newton_solve(
            solver, problem, x_pde, bcs, f_pde,
            max_iter=max_iter, tol_res=tol_res, tol_du=tol_du,
            damping=damping, mu=mu, verbose=verbose,
        )

    runtime = time.time() - t0

    # Create prediction function
    def u_fn(x):
        """Evaluate solution at points x."""
        if isinstance(x, (list, tuple)):
            x = torch.tensor(x, device=device, dtype=solver.beta.dtype)
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=device, dtype=solver.beta.dtype)
        return solver.predict(x)

    result = {
        "u_fn": u_fn,
        "scale": scale,
        "runtime": runtime,
        "n_iters": len(history),
    }

    if return_solver:
        result["solver"] = solver

    if return_history:
        result["history"] = history

    if return_metrics:
        val_err, grad_err = evaluate_error(solver, problem, n_test=n_test)
        result["metrics"] = {
            "val_err": val_err,
            "grad_err": grad_err,
            "runtime": runtime,
            "n_iters": len(history),
        }
        if verbose:
            print(f"Value error: {val_err:.2e}, Gradient error: {grad_err:.2e}, "
                  f"Iterations: {len(history)}")

    return result
