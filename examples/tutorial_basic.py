#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Basic tutorial: Solving a linear PDE in one line.

This demonstrates the high-level API for solving linear PDEs.
"""

import torch
from fastlsq import solve_linear
from fastlsq.problems.linear import PoissonND
from fastlsq.plotting import plot_solution_1d

# Setup
torch.set_default_dtype(torch.float32)

# Create problem
problem = PoissonND()

# Solve in one line!
result = solve_linear(
    problem,
    scale=5.0,  # Or leave None for auto-selection
    n_blocks=3,
    hidden_size=500,
    verbose=True,
)

# Access results
u_fn = result["u_fn"]
metrics = result["metrics"]
print(f"\nSolution computed!")
print(f"Value error: {metrics['val_err']:.2e}")
print(f"Gradient error: {metrics['grad_err']:.2e}")

# Evaluate at custom points
x_test = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5]])
u_pred = u_fn(x_test)
print(f"\nPrediction at [0.5, 0.5, 0.5, 0.5, 0.5]: {u_pred.item():.6f}")

# Plot (for 1D problems)
# plot_solution_1d(result["solver"], problem, save_path="tutorial_poisson.png")
