# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Auto-tuning utilities for FastLSQ (scale selection, defaults)."""

import torch
import numpy as np
from typing import Optional, Callable

from fastlsq.solvers import FastLSQSolver
from fastlsq.linalg import solve_lstsq
from fastlsq.newton import build_solver_with_scale, get_initial_guess, newton_solve
from fastlsq.utils import device, evaluate_error


def auto_select_scale(
    problem,
    solver_class: Callable = FastLSQSolver,
    *,
    n_blocks: int = 3,
    hidden_size: int = 500,
    n_pde: int = 5000,
    n_bc: int = 1000,
    scales: Optional[list] = None,
    n_trials: int = 3,
    newton_mode: bool = False,
    verbose: bool = True,
) -> float:
    """Automatically select optimal scale via grid search.

    Parameters
    ----------
    problem : object
        Problem instance.
    solver_class : callable
        Constructor for solver (e.g., FastLSQSolver or lambda dim: FastLSQSolver(dim, normalize=True)).
    n_blocks, hidden_size : int
        Solver architecture.
    n_pde, n_bc : int
        Training data sizes.
    scales : list[float], optional
        Scale candidates. Default: [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0]
    n_trials : int
        Number of random seeds per scale.
    newton_mode : bool
        If True, use Newton iteration; otherwise direct solve.
    verbose : bool
        Print progress.

    Returns
    -------
    best_scale : float
    """
    if scales is None:
        scales = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0]

    best_scale = scales[0]
    best_error = float("inf")

    for scale in scales:
        errors = []
        for seed in range(n_trials):
            try:
                torch.manual_seed(seed)
                np.random.seed(seed)

                if newton_mode:
                    solver = solver_class(problem.dim)
                    for _ in range(n_blocks):
                        solver.add_block(hidden_size=hidden_size, scale=scale)
                    solver.beta = torch.zeros(solver.n_features, 1, device=device)

                    x_pde, bcs, f_pde = problem.get_train_data(n_pde=n_pde, n_bc=n_bc)
                    get_initial_guess(solver, problem, x_pde, bcs, f_pde, mu=1e-10)
                    history = newton_solve(
                        solver, problem, x_pde, bcs, f_pde,
                        max_iter=15, mu=1e-10, verbose=False,
                    )
                    val_err, _ = evaluate_error(solver, problem, n_test=1000)
                else:
                    solver = solver_class(problem.dim)
                    if hasattr(problem, "scale_multipliers"):
                        effective_scale = [scale * m for m in problem.scale_multipliers]
                    else:
                        effective_scale = scale

                    for _ in range(n_blocks):
                        solver.add_block(hidden_size=hidden_size, scale=effective_scale)

                    data = problem.get_train_data(n_pde=n_pde, n_bc=n_bc)
                    if len(data) == 3:
                        x_pde, bcs, f_pde = data
                        build_args = (bcs, f_pde)
                    else:
                        x_pde, bcs = data
                        build_args = (bcs,)

                    A, b = problem.build(solver, x_pde, *build_args)
                    solver.beta = solve_lstsq(A, b)
                    val_err, _ = evaluate_error(solver, problem, n_test=1000)

                if np.isnan(val_err) or np.isinf(val_err):
                    val_err = 1e10
                errors.append(val_err)
            except Exception:
                errors.append(1e10)

        mean_error = np.mean(errors)
        if verbose:
            print(f"  Scale {scale:5.1f}: error = {mean_error:.2e}")

        if mean_error < best_error:
            best_error = mean_error
            best_scale = scale

    return best_scale
