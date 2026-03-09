# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Basic tests for FastLSQ."""

import torch
import numpy as np
import pytest

from fastlsq import (
    FastLSQSolver, solve_linear, solve_nonlinear,
    check_problem, sample_box, sample_ball, sample_boundary_box, to_numpy, Op,
)
from fastlsq.problems.linear import PoissonND
from fastlsq.problems.nonlinear import NLPoisson2D


def test_trivial_pde():
    """Test a trivial 2D Poisson with known exact solution u = sin(pi*x)*sin(pi*y)."""
    torch.set_default_dtype(torch.float64)

    class TrivialPoisson2D:
        """-Laplacian(u) = 2*pi^2 * sin(pi*x)*sin(pi*y) on [0,1]^2, u=0 on boundary."""
        name = "Trivial Poisson 2D"
        dim = 2
        pde_op = -Op.laplacian(d=2)

        def exact(self, x):
            return torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2])

        def exact_grad(self, x):
            sx = torch.sin(np.pi * x[:, 0:1])
            cx = torch.cos(np.pi * x[:, 0:1])
            sy = torch.sin(np.pi * x[:, 1:2])
            cy = torch.cos(np.pi * x[:, 1:2])
            return torch.cat([np.pi * cx * sy, np.pi * sx * cy], dim=1)

        def source(self, x):
            return 2 * np.pi**2 * self.exact(x)

        def get_train_data(self, n_pde=2000, n_bc=400):
            x_pde = sample_box(n_pde, self.dim)
            f_pde = self.source(x_pde)
            x_bc = sample_boundary_box(n_bc, self.dim)
            u_bc = self.exact(x_bc)
            return x_pde, [(x_bc, u_bc)], f_pde

        def build(self, solver, x_pde, bcs, f_pde):
            basis = solver.basis
            cache = basis.cache(x_pde)
            A_pde = self.pde_op.apply(basis, x_pde, cache=cache)
            As, bs = [A_pde], [f_pde]
            for (x_bc, u_bc) in bcs:
                As.append(100.0 * basis.evaluate(x_bc))
                bs.append(100.0 * u_bc)
            return torch.cat(As), torch.cat(bs)

        def get_test_points(self, n=2000):
            return sample_box(n, self.dim)

    problem = TrivialPoisson2D()
    result = solve_linear(
        problem,
        scale=5.0,
        n_blocks=2,
        hidden_size=300,
        n_pde=2000,
        n_bc=400,
        n_test=2000,
        verbose=False,
    )
    assert result["metrics"]["val_err"] < 1e-3, (
        f"Trivial PDE value error {result['metrics']['val_err']:.2e} should be < 1e-3"
    )
    assert result["metrics"]["grad_err"] < 1e-2, (
        f"Trivial PDE gradient error {result['metrics']['grad_err']:.2e} should be < 1e-2"
    )


def test_solver_creation():
    """Test solver can be created and basis used."""
    solver = FastLSQSolver(input_dim=2)
    solver.add_block(hidden_size=10, scale=1.0)
    assert solver.n_features == 10

    x = torch.rand(5, 2)
    basis = solver.basis
    cache = basis.cache(x)
    H = basis.evaluate(x, cache=cache)
    dH = basis.gradient(x, cache=cache)
    ddH = basis.hessian_diag(x, cache=cache)
    assert H.shape == (5, 10)
    assert dH.shape == (5, 2, 10)
    assert ddH.shape == (5, 2, 10)


def test_solve_linear():
    """Test high-level linear solve API."""
    torch.set_default_dtype(torch.float32)
    problem = PoissonND()
    result = solve_linear(
        problem,
        scale=5.0,
        n_blocks=2,
        hidden_size=200,
        n_pde=1000,
        n_bc=200,
        verbose=False,
    )
    assert "u_fn" in result
    assert "metrics" in result
    assert result["metrics"]["val_err"] < 1.0  # Reasonable error


def test_solve_nonlinear():
    """Test high-level nonlinear solve API."""
    torch.set_default_dtype(torch.float64)
    problem = NLPoisson2D()
    result = solve_nonlinear(
        problem,
        scale=5.0,
        n_blocks=2,
        hidden_size=200,
        n_pde=500,
        n_bc=100,
        max_iter=10,
        verbose=False,
    )
    assert "u_fn" in result
    assert "history" in result
    assert result["n_iters"] > 0


def test_check_problem():
    """Test problem diagnostics."""
    problem = PoissonND()
    results = check_problem(problem, verbose=False)
    assert results["shape_check"]
    assert results["data_check"]


def test_geometry_samplers():
    """Test geometry samplers."""
    x_box = sample_box(100, dim=2)
    assert x_box.shape == (100, 2)
    assert torch.all(x_box >= 0) and torch.all(x_box <= 1)

    x_ball = sample_ball(100, dim=3, radius=1.0)
    assert x_ball.shape == (100, 3)
    norms = torch.norm(x_ball, dim=1)
    assert torch.all(norms <= 1.0)


def test_op_learnable_parameter():
    """Test Op accepts nn.Parameter and gradients flow through apply()."""
    from fastlsq.basis import SinusoidalBasis

    k = torch.nn.Parameter(torch.tensor(10.0))
    helmholtz = Op.laplacian(d=2) + k**2 * Op.identity(d=2)
    basis = SinusoidalBasis.random(input_dim=2, n_features=50, sigma=3.0)
    x = torch.rand(20, 2)
    A = helmholtz.apply(basis, x)
    assert A.shape == (20, 50)
    loss = A.sum()
    loss.backward()
    assert k.grad is not None
    solver = FastLSQSolver(input_dim=2)
    solver.add_block(hidden_size=10, scale=1.0)
    solver.beta = torch.randn(10, 1)

    x = torch.rand(5, 2)
    u_np = to_numpy(solver, x)
    assert u_np.shape == (5, 1)
    assert isinstance(u_np, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
