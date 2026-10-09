# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Diagnostic utilities for checking problems and detecting common issues."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from fastlsq.solvers import FastLSQSolver
from fastlsq.linalg import solve_lstsq


def _fd_settings(dtype: torch.dtype):
    """Central-difference step and acceptance threshold for ``dtype``.

    With step ``h`` the central difference has truncation error ``O(h^2)`` and
    round-off error ``O(eps_machine / h)``; ``h ~ eps_machine^(1/3)`` balances the
    two.  The thresholds leave an order of magnitude of slack over that optimum.
    """
    if dtype in (torch.float64, torch.double):
        return 1e-6, 1e-5
    return 1e-3, 1e-2          # float32 (and anything narrower)


def check_problem(
    problem,
    *,
    n_test: int = 100,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run diagnostics on a problem definition.

    Checks:

    - shape consistency of ``exact()`` and ``exact_grad()`` (scalar problems
      return ``(n, 1)`` / ``(n, dim)``; vector problems with ``n_outputs = k``
      return ``(n, k)`` / ``(n, dim, k)``),
    - ``exact_grad()`` against a **central** finite difference of ``exact()``,
      with a step and tolerance chosen for the active dtype (float32 problems
      are checked loosely rather than falsely failed),
    - Dirichlet boundary data against ``exact()`` at the boundary points, for
      every ``(x_bc, u_bc)`` (or ``(x_bc, u_bc, "dirichlet")``) entry whose
      shapes allow the comparison,
    - ``get_train_data()`` returns a valid ``(x_pde, bcs, f_pde)`` triple (or
      the legacy ``(x_pde, bcs)`` pair).

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
        ``shape_check``, ``gradient_check``, ``bc_check``, ``data_check`` (bools),
        plus ``warnings`` and ``errors`` (lists of str).
    """
    results = {
        "shape_check": True,
        "gradient_check": True,
        "bc_check": True,
        "data_check": True,
        "warnings": [],
        "errors": [],
    }
    k = int(getattr(problem, "n_outputs", 1))
    dim = int(problem.dim)

    # ---- exact() / exact_grad(): shapes and a central finite difference ----
    try:
        x_test = problem.get_test_points(n_test)
        u = problem.exact(x_test)
        grad_u = problem.exact_grad(x_test)

        n = x_test.shape[0]
        exp_u = (n, k)
        exp_g = (n, dim) if k == 1 else (n, dim, k)
        if tuple(u.shape) != exp_u:
            results["errors"].append(
                f"exact() returns shape {tuple(u.shape)}, expected {exp_u}"
            )
            results["shape_check"] = False
        if tuple(grad_u.shape) != exp_g:
            results["errors"].append(
                f"exact_grad() returns shape {tuple(grad_u.shape)}, expected {exp_g}"
            )
            results["shape_check"] = False

        if results["shape_check"]:
            h, thresh = _fd_settings(x_test.dtype)
            grad_fd = torch.zeros_like(grad_u)
            for d in range(dim):
                x_plus, x_minus = x_test.clone(), x_test.clone()
                x_plus[:, d] += h
                x_minus[:, d] -= h
                du = (problem.exact(x_plus) - problem.exact(x_minus)) / (2.0 * h)
                if k == 1:
                    grad_fd[:, d] = du.reshape(-1)
                else:
                    grad_fd[:, d, :] = du
            grad_error = (torch.norm(grad_u - grad_fd)
                          / (torch.norm(grad_u) + 1e-30)).item()
            if not np.isfinite(grad_error) or grad_error > thresh:
                results["warnings"].append(
                    f"exact_grad() disagrees with a central finite difference of "
                    f"exact(): relative error {grad_error:.2e} > {thresh:.0e} "
                    f"(step {h:.0e}, dtype {str(x_test.dtype).replace('torch.', '')})"
                )
                results["gradient_check"] = False
        else:
            results["gradient_check"] = False

    except Exception as e:
        results["errors"].append(f"Error in exact/exact_grad: {e}")
        results["shape_check"] = False
        results["gradient_check"] = False

    # ---- get_train_data(): structure, then Dirichlet data vs exact() ----
    bcs = None
    try:
        data = problem.get_train_data(n_pde=100, n_bc=20)
        if len(data) in (2, 3):
            x_pde, bcs = data[0], data[1]
            if x_pde.shape[1] != dim:
                results["errors"].append(
                    f"get_train_data() x_pde has wrong dimension: "
                    f"{x_pde.shape[1]} != {dim}"
                )
                results["data_check"] = False
            if len(data) == 2:
                results["warnings"].append(
                    "get_train_data() returns (x_pde, bcs) without f_pde; the "
                    "documented contract is (x_pde, bcs, f_pde) and some entry "
                    "points (train_bandwidth, the Newton-mode scale search) "
                    "unpack three values"
                )
        else:
            results["errors"].append(
                f"get_train_data() should return 2 or 3 items, got {len(data)}"
            )
            results["data_check"] = False
    except Exception as e:
        results["errors"].append(f"Error in get_train_data(): {e}")
        results["data_check"] = False

    if bcs is not None:
        try:
            for i, entry in enumerate(bcs):
                if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                    continue
                kind = entry[2] if len(entry) >= 3 else "dirichlet"
                if not (isinstance(kind, str) and kind.lower() == "dirichlet"):
                    continue
                x_bc, u_bc = entry[0], entry[1]
                if not (torch.is_tensor(x_bc) and torch.is_tensor(u_bc)):
                    continue
                u_ref = problem.exact(x_bc)
                if tuple(u_ref.shape) != tuple(u_bc.shape):
                    continue
                _, thresh = _fd_settings(u_bc.dtype)
                err = (torch.norm(u_bc - u_ref)
                       / (torch.norm(u_ref) + 1e-30)).item()
                if not np.isfinite(err) or err > thresh:
                    results["warnings"].append(
                        f"boundary entry {i}: Dirichlet data differs from exact() "
                        f"at the boundary points (relative error {err:.2e})"
                    )
                    results["bc_check"] = False
        except Exception as e:
            results["warnings"].append(f"BC consistency check skipped: {e}")

    if verbose:
        print("=" * 60)
        print("Problem Diagnostics")
        print("=" * 60)
        print(f"Shape check:     {'PASS' if results['shape_check'] else 'FAIL'}")
        print(f"Gradient check:  {'PASS' if results['gradient_check'] else 'FAIL'}")
        print(f"BC check:        {'PASS' if results['bc_check'] else 'FAIL'}")
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
    """Suggest a starting scale from the problem's dimension alone.

    A coarse heuristic (5 for d <= 2, 3 for d <= 5, 2 above); it does not look at
    the domain size, wavenumber or solution content.  Use
    :func:`~fastlsq.tuning.auto_select_scale` (or ``solve_linear(scale=None)``)
    for a data-driven choice.

    Parameters
    ----------
    problem : object
    n_trials : int
        Accepted for backward compatibility and ignored: no trials are run.
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
