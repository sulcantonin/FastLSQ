#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Tutorial: Adding your own PDE problem.

Shows how to define a custom PDE using the symbolic `Op` interface.
The operator is evaluated analytically via the `SinusoidalBasis` --
no manual tensor algebra required.
"""

import torch
import numpy as np
from fastlsq import solve_linear, check_problem, Op
from fastlsq.geometry import sample_box, sample_boundary_box


# ======================================================================
# Define the PDE using DiffOperator (Op)
# ======================================================================

class CustomPoisson2D:
    """Custom 2D Poisson problem: -Laplacian(u) = f on [0,1]^2.

    Exact solution: u(x,y) = sin(pi*x) * sin(pi*y)
    """

    def __init__(self):
        self.name = "Custom Poisson 2D"
        self.dim = 2
        self.pde_op = -Op.laplacian(d=2)

    def exact(self, x):
        return torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2])

    def exact_grad(self, x):
        sx = torch.sin(np.pi * x[:, 0:1])
        cx = torch.cos(np.pi * x[:, 0:1])
        sy = torch.sin(np.pi * x[:, 1:2])
        cy = torch.cos(np.pi * x[:, 1:2])
        return torch.cat([np.pi * cx * sy, np.pi * sx * cy], dim=1)

    def source(self, x):
        return 2 * np.pi ** 2 * self.exact(x)

    def get_train_data(self, n_pde=5000, n_bc=1000):
        x_pde = sample_box(n_pde, self.dim, bounds=(0.0, 1.0))
        f_pde = self.source(x_pde)
        x_bc = sample_boundary_box(n_bc, self.dim, bounds=(0.0, 1.0))
        u_bc = self.exact(x_bc)
        return x_pde, [(x_bc, u_bc)], f_pde

    def build(self, solver, x_pde, bcs, f_pde):
        """Assemble A beta = b using the symbolic operator."""
        basis = solver.basis
        cache = basis.cache(x_pde)

        A_pde = self.pde_op.apply(basis, x_pde, cache=cache)
        As, bs = [A_pde], [f_pde]

        for (x_bc, u_bc) in bcs:
            As.append(100.0 * basis.evaluate(x_bc))
            bs.append(100.0 * u_bc)

        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=5000):
        return sample_box(n, self.dim, bounds=(0.0, 1.0))


# ======================================================================
# Run
# ======================================================================

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)

    problem = CustomPoisson2D()
    print("Custom Poisson 2D via Op-based interface")
    check_problem(problem)
    result = solve_linear(problem, scale=5.0, verbose=True)
    print(f"Value error:    {result['metrics']['val_err']:.2e}")
    print(f"Gradient error: {result['metrics']['grad_err']:.2e}")

    # Composing operators symbolically
    print("\n\nBonus: Composing operators symbolically")
    print("-" * 50)

    d = 3
    dt2 = Op.partial(dim=2, order=2, d=d)
    lap_xy = Op.laplacian(d=d, dims=[0, 1])
    c = 2.0
    wave_op = dt2 - c**2 * lap_xy
    print(f"  Wave operator:     {wave_op}")

    k = 10.0
    helmholtz = Op.laplacian(d=2) + k**2 * Op.identity(d=2)
    print(f"  Helmholtz (k=10):  {helmholtz}")

    bih = Op.biharmonic(d=2)
    print(f"  Biharmonic:        {bih}")

    # Learnable coefficients: use nn.Parameter for AdamW optimisation
    # k = torch.nn.Parameter(torch.tensor(10.0))
    # helmholtz = Op.laplacian(d=2) + k**2 * Op.identity(d=2)
    # See examples/learnable_helmholtz.py for the full workflow.
