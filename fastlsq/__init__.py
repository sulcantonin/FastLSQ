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

__version__ = "0.1.0"
__all__ = [
    "FastLSQSolver",
    "PIELMSolver",
    "solve_lstsq",
]
