#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Nonlinear PDE tutorial: Solving with Newton-Raphson.

This demonstrates the high-level API for solving nonlinear PDEs.
"""

import torch
from fastlsq import solve_nonlinear
from fastlsq.problems.nonlinear import NLPoisson2D
from fastlsq.plotting import plot_convergence, plot_solution_2d_contour

# Setup
torch.set_default_dtype(torch.float64)

# Create problem
problem = NLPoisson2D()

# Solve in one line!
result = solve_nonlinear(
    problem,
    scale=5.0,  # Or leave None for auto-selection
    n_blocks=3,
    hidden_size=500,
    max_iter=30,
    verbose=True,
)

# Access results
u_fn = result["u_fn"]
history = result["history"]
metrics = result["metrics"]

print(f"\nSolution computed!")
print(f"Value error: {metrics['val_err']:.2e}")
print(f"Gradient error: {metrics['grad_err']:.2e}")
print(f"Iterations: {metrics['n_iters']}")

# Plot convergence
plot_convergence(
    history,
    problem_name=problem.name,
    save_path="tutorial_nlpoisson_convergence.png",
)

# Plot solution
plot_solution_2d_contour(
    result["solver"],
    problem,
    save_path="tutorial_nlpoisson_solution.png",
)
