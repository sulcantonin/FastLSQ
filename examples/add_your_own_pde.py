#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Tutorial: Adding your own PDE problem.

This example shows how to define a custom PDE problem and solve it with FastLSQ.
"""

import torch
import numpy as np
from fastlsq import solve_linear, check_problem
from fastlsq.geometry import sample_box, sample_boundary_box


class CustomPoisson2D:
    """Custom 2D Poisson problem: -Laplacian(u) = f on [0,1]^2.

    Exact solution: u(x,y) = sin(pi*x) * sin(pi*y)
    """

    def __init__(self):
        self.name = "Custom Poisson 2D"
        self.dim = 2

    def exact(self, x):
        """Analytical solution."""
        return torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2])

    def exact_grad(self, x):
        """Analytical gradient."""
        sx = torch.sin(np.pi * x[:, 0:1])
        cx = torch.cos(np.pi * x[:, 0:1])
        sy = torch.sin(np.pi * x[:, 1:2])
        cy = torch.cos(np.pi * x[:, 1:2])
        return torch.cat([
            np.pi * cx * sy,  # du/dx
            np.pi * sx * cy,  # du/dy
        ], dim=1)

    def source(self, x):
        """Right-hand side f(x)."""
        # f = -Laplacian(u) = 2*pi^2 * u
        return 2 * np.pi ** 2 * self.exact(x)

    def get_train_data(self, n_pde=5000, n_bc=1000):
        """Generate training data: collocation and boundary points."""
        # Collocation points (interior)
        x_pde = sample_box(n_pde, self.dim, bounds=(0.0, 1.0))
        f_pde = self.source(x_pde)

        # Boundary points
        x_bc = sample_boundary_box(n_bc, self.dim, bounds=(0.0, 1.0))
        u_bc = self.exact(x_bc)

        # Return format: (x_pde, bcs, f_pde)
        # bcs is a list of tuples: [(x_bc, u_bc)]
        return x_pde, [(x_bc, u_bc)], f_pde

    def build(self, solver, x_pde, bcs, f_pde):
        """Assemble the linear system A beta = b."""
        # Get features and derivatives
        _, _, ddH = solver.get_features(x_pde)

        # Laplacian operator: u_xx + u_yy = sum(ddH)
        lap = torch.sum(ddH, dim=1)
        A = -lap  # Negative Laplacian
        b = f_pde

        # Add boundary conditions (penalty method)
        As, bs = [A], [b]
        for (x_bc, u_bc) in bcs:
            H_bc, _, _ = solver.get_features(x_bc)
            weight = 100.0  # Penalty weight
            As.append(weight * H_bc)
            bs.append(weight * u_bc)

        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=5000):
        """Random test points for evaluation."""
        return sample_box(n, self.dim, bounds=(0.0, 1.0))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)

    # Create problem
    problem = CustomPoisson2D()

    # Check problem definition
    print("Checking problem definition...")
    check_problem(problem)

    # Solve
    print("\nSolving...")
    result = solve_linear(problem, scale=5.0, verbose=True)

    # Results
    print(f"\nSolution computed!")
    print(f"Value error: {result['metrics']['val_err']:.2e}")
    print(f"Gradient error: {result['metrics']['grad_err']:.2e}")

    # Evaluate at a point
    x_test = torch.tensor([[0.5, 0.5]])
    u_pred = result["u_fn"](x_test)
    u_exact = problem.exact(x_test)
    print(f"\nAt (0.5, 0.5):")
    print(f"  Predicted: {u_pred.item():.6f}")
    print(f"  Exact:     {u_exact.item():.6f}")
