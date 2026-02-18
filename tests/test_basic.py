# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Basic tests for FastLSQ."""

import torch
import numpy as np
import pytest

from fastlsq import (
    FastLSQSolver, solve_linear, solve_nonlinear,
    check_problem, sample_box, to_numpy,
)
from fastlsq.problems.linear import PoissonND
from fastlsq.problems.nonlinear import NLPoisson2D


def test_solver_creation():
    """Test solver can be created and features computed."""
    solver = FastLSQSolver(input_dim=2)
    solver.add_block(hidden_size=10, scale=1.0)
    assert solver.n_features == 10

    x = torch.rand(5, 2)
    H, dH, ddH = solver.get_features(x)
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


def test_export():
    """Test export utilities."""
    solver = FastLSQSolver(input_dim=2)
    solver.add_block(hidden_size=10, scale=1.0)
    solver.beta = torch.randn(10, 1)

    x = torch.rand(5, 2)
    u_np = to_numpy(solver, x)
    assert u_np.shape == (5, 1)
    assert isinstance(u_np, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
