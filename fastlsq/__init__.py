# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
FastLSQ -- Solving PDEs in one shot via Fourier features
with exact analytical derivatives.

Reference:
    A. Sulc, "Solving PDEs in One Shot via Fourier Features with Exact
    Analytical Derivatives," arXiv:2602.10541, 2026.
"""

from fastlsq.device import resolve_device, set_device, get_device, device_info
from fastlsq.basis import (
    SinusoidalBasis,
    BasisCache,
    DiffOperator,
    Op,
    IntegralOperator,
    IntegroDifferentialOperator,
    GaussianWindowedBasis,
    ProjectionOperator,
    FeatureBasis,
)
from fastlsq.solvers import FastLSQSolver, PIELMSolver
from fastlsq.vector  import VectorBasis, VectorFastLSQSolver
from fastlsq.linalg import solve_lstsq
from fastlsq.block import block_concat, pack_beta, unpack_beta
from fastlsq.api import solve_linear, solve_nonlinear
from fastlsq.tuning import auto_select_scale
from fastlsq.learnable import LearnableFastLSQ, train_bandwidth
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
from fastlsq import viz

__version__ = "0.4.0"
__all__ = [
    # Device selection (CPU / CUDA / Apple-MPS, dtype-aware)
    "resolve_device",
    "set_device",
    "get_device",
    "device_info",
    # Basis and operators (foundation)
    "SinusoidalBasis",
    "BasisCache",
    "DiffOperator",
    "Op",
    "IntegralOperator",
    "IntegroDifferentialOperator",
    "GaussianWindowedBasis",
    "ProjectionOperator",
    "FeatureBasis",
    # Vector-valued basis  (0.1.5)
    "VectorBasis",
    "VectorFastLSQSolver",
    # Core solvers
    "FastLSQSolver",
    "PIELMSolver",
    "solve_lstsq",
    # Block assembly for vector-valued u
    "block_concat",
    "pack_beta",
    "unpack_beta",
    # Learnable bandwidth
    "LearnableFastLSQ",
    "train_bandwidth",
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
