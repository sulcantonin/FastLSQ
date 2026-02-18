# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
FastLSQ -- Solving PDEs in one shot via Fourier features
with exact analytical derivatives.

Reference:
    A. Sulc, "Solving PDEs in One Shot via Fourier Features with Exact
    Analytical Derivatives," arXiv:2602.10541, 2026.
"""

from fastlsq.solvers import FastLSQSolver, PIELMSolver
from fastlsq.linalg import solve_lstsq
from fastlsq.api import solve_linear, solve_nonlinear
from fastlsq.tuning import auto_select_scale
from fastlsq.plotting import (
    plot_solution_1d,
    plot_solution_2d_slice,
    plot_solution_2d_contour,
    plot_convergence,
    plot_spectral_sensitivity,
)
from fastlsq.geometry import (
    sample_box,
    sample_ball,
    sample_sphere,
    sample_interval,
    sample_boundary_box,
    get_sampler,
)
from fastlsq.diagnostics import check_problem, check_solver_conditioning, suggest_scale
from fastlsq.export import (
    to_numpy,
    to_dict,
    from_dict,
    save_checkpoint,
    load_checkpoint,
)

__version__ = "0.1.0"
__all__ = [
    # Core solvers
    "FastLSQSolver",
    "PIELMSolver",
    "solve_lstsq",
    # High-level API
    "solve_linear",
    "solve_nonlinear",
    # Auto-tuning
    "auto_select_scale",
    # Plotting
    "plot_solution_1d",
    "plot_solution_2d_slice",
    "plot_solution_2d_contour",
    "plot_convergence",
    "plot_spectral_sensitivity",
    # Geometry
    "sample_box",
    "sample_ball",
    "sample_sphere",
    "sample_interval",
    "sample_boundary_box",
    "get_sampler",
    # Diagnostics
    "check_problem",
    "check_solver_conditioning",
    "suggest_scale",
    # Export
    "to_numpy",
    "to_dict",
    "from_dict",
    "save_checkpoint",
    "load_checkpoint",
]
