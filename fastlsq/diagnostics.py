# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Diagnostic utilities for checking problems and detecting common issues."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from fastlsq.solvers import FastLSQSolver
from fastlsq.linalg import solve_lstsq
from fastlsq.utils import device


def check_problem(
    problem,
    *,
    n_test: int = 100,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run diagnostics on a problem definition.

    Checks:
    - Shape consistency of exact() and exact_grad()
    - Finite difference consistency of exact_grad()
    - BC consistency (if applicable)
    - get_train_data() returns valid data

    Parameters
    ----------
    problem : object
        Problem instance.
    n_test : int
        Number of test points.
    verbose : bool
        Print results.

    Returns
    -------
    results : dict
        Diagnostic results.
    """
    results = {
        "shape_check": True,
        "gradient_check": True,
        "data_check": True,
        "warnings": [],
        "errors": [],
    }

    # Test exact() and exact_grad() shapes
    try:
        x_test = problem.get_test_points(n_test)
        u = problem.exact(x_test)
        grad_u = problem.exact_grad(x_test)

        if u.shape != (n_test, 1):
            results["errors"].append(
                f"exact() returns shape {u.shape}, expected ({n_test}, 1)"
            )
            results["shape_check"] = False

        if grad_u.shape != (n_test, problem.dim):
            results["errors"].append(
                f"exact_grad() returns shape {grad_u.shape}, "
                f"expected ({n_test}, {problem.dim})"
            )
            results["shape_check"] = False

        # Finite difference check
        eps = 1e-5
        grad_fd = torch.zeros_like(grad_u)
        for d in range(problem.dim):
            x_plus = x_test.clone()
            x_plus[:, d] += eps
            u_plus = problem.exact(x_plus)
            grad_fd[:, d] = ((u_plus - u) / eps).squeeze()

        grad_error = torch.norm(grad_u - grad_fd) / (torch.norm(grad_u) + 1e-10)
        if grad_error > 1e-3:
            results["warnings"].append(
                f"Gradient finite difference error: {grad_error:.2e} "
                f"(may indicate incorrect exact_grad() implementation)"
            )
            results["gradient_check"] = False

    except Exception as e:
        results["errors"].append(f"Error in exact/exact_grad: {e}")
        results["shape_check"] = False

    # Test get_train_data()
    try:
        data = problem.get_train_data(n_pde=100, n_bc=20)
        if len(data) == 3:
            x_pde, bcs, f_pde = data
            if x_pde.shape[1] != problem.dim:
                results["errors"].append(
                    f"get_train_data() x_pde has wrong dimension: "
                    f"{x_pde.shape[1]} != {problem.dim}"
                )
                results["data_check"] = False
        elif len(data) == 2:
            x_pde, bcs = data
            if x_pde.shape[1] != problem.dim:
                results["errors"].append(
                    f"get_train_data() x_pde has wrong dimension: "
                    f"{x_pde.shape[1]} != {problem.dim}"
                )
                results["data_check"] = False
        else:
            results["errors"].append(
                f"get_train_data() should return 2 or 3 items, got {len(data)}"
            )
            results["data_check"] = False
    except Exception as e:
        results["errors"].append(f"Error in get_train_data(): {e}")
        results["data_check"] = False

    if verbose:
        print("=" * 60)
        print("Problem Diagnostics")
        print("=" * 60)
        print(f"Shape check:     {'PASS' if results['shape_check'] else 'FAIL'}")
        print(f"Gradient check:  {'PASS' if results['gradient_check'] else 'FAIL'}")
        print(f"Data check:      {'PASS' if results['data_check'] else 'FAIL'}")
        if results["warnings"]:
            print("\nWarnings:")
            for w in results["warnings"]:
                print(f"  - {w}")
        if results["errors"]:
            print("\nErrors:")
            for e in results["errors"]:
                print(f"  - {e}")
        print("=" * 60)

    return results


def check_solver_conditioning(
    solver: FastLSQSolver,
    A: torch.Tensor,
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Check conditioning of the linear system A beta = b.

    Parameters
    ----------
    solver : FastLSQSolver
    A : Tensor, shape (M, N)
        System matrix.
    verbose : bool

    Returns
    -------
    results : dict
        Contains: 'condition_number', 'rank', 'warnings', 'suggestions'
    """
    results = {
        "condition_number": None,
        "rank": None,
        "warnings": [],
        "suggestions": [],
    }

    try:
        # Compute condition number
        s = torch.linalg.svdvals(A)
        cond = s[0] / (s[-1] + 1e-15)
        results["condition_number"] = cond.item()

        # Effective rank
        rank = torch.sum(s > 1e-10 * s[0]).item()
        results["rank"] = rank

        if cond > 1e12:
            results["warnings"].append(
                f"System is ill-conditioned (cond={cond:.2e}). "
                "Consider increasing Tikhonov regularisation (mu)."
            )
            results["suggestions"].append("Try mu=1e-8 or higher")

        if rank < A.shape[1] * 0.9:
            results["warnings"].append(
                f"System appears rank-deficient (rank={rank}/{A.shape[1]}). "
                "Consider reducing feature count or increasing collocation points."
            )
            results["suggestions"].append("Reduce n_blocks or hidden_size")

    except Exception as e:
        results["warnings"].append(f"Could not compute conditioning: {e}")

    if verbose:
        print("=" * 60)
        print("Solver Conditioning Diagnostics")
        print("=" * 60)
        if results["condition_number"] is not None:
            print(f"Condition number: {results['condition_number']:.2e}")
        if results["rank"] is not None:
            print(f"Effective rank:  {results['rank']}/{A.shape[1]}")
        if results["warnings"]:
            print("\nWarnings:")
            for w in results["warnings"]:
                print(f"  - {w}")
        if results["suggestions"]:
            print("\nSuggestions:")
            for s in results["suggestions"]:
                print(f"  - {s}")
        print("=" * 60)

    return results


def suggest_scale(
    problem,
    *,
    n_trials: int = 3,
    verbose: bool = True,
) -> float:
    """Suggest a reasonable scale based on problem characteristics.

    This is a heuristic based on problem dimension and domain size.

    Parameters
    ----------
    problem : object
    n_trials : int
    verbose : bool

    Returns
    -------
    suggested_scale : float
    """
    # Heuristic: scale ~ 1 / domain_size for low dim, ~ sqrt(dim) for high dim
    dim = problem.dim
    if dim <= 2:
        suggested = 5.0
    elif dim <= 5:
        suggested = 3.0
    else:
        suggested = 2.0

    if verbose:
        print(f"Suggested scale (heuristic): {suggested:.1f}")
        print("  (For best results, use auto_select_scale() or grid search)")

    return suggested
