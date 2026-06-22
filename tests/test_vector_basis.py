# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License.

"""Tests for the 0.1.5 vector-valued basis."""

import numpy as np
import pytest
import torch

import fastlsq
from fastlsq import (
    VectorBasis, VectorFastLSQSolver,
    SinusoidalBasis, solve_lstsq,
)
from fastlsq.utils import device


# ----------------------------------------------------------------------
# Module-level sanity
# ----------------------------------------------------------------------

def test_version():
    assert fastlsq.__version__ == "0.4.0"


def test_imports():
    assert VectorBasis is not None
    assert VectorFastLSQSolver is not None


# ----------------------------------------------------------------------
# VectorBasis construction
# ----------------------------------------------------------------------

def test_random_constructor_shapes():
    K = 3; N = 200
    vb = VectorBasis.random(input_dim=2, n_features=N, sigma=2.0,
                            n_components=K)
    assert vb.n_components == K
    assert vb.input_dim == 2
    assert vb.n_features_per_component == N
    assert vb.n_features_total == K * N


def test_per_component_sigmas():
    sigmas = [1.0, 3.0]
    vb = VectorBasis.random(input_dim=2, n_features=100, n_components=2,
                            sigmas=sigmas)
    # the std of each component's W column should be roughly sigma
    for c, s in zip(vb.components, sigmas):
        empirical = float(c.W.std())
        assert abs(empirical - s) / s < 0.3, (empirical, s)


def test_construct_from_components_validates_input_dim():
    a = SinusoidalBasis.random(2, 50, sigma=1.0)
    b = SinusoidalBasis.random(3, 50, sigma=1.0)
    with pytest.raises(ValueError):
        VectorBasis([a, b])


def test_construct_from_solvers():
    from fastlsq.solvers import FastLSQSolver
    s1 = FastLSQSolver(2)
    s2 = FastLSQSolver(2)
    s1.add_block(50, scale=1.0)
    s2.add_block(50, scale=2.0)
    vb = VectorBasis.from_solvers([s1, s2])
    assert vb.n_components == 2


# ----------------------------------------------------------------------
# Stacked evaluators
# ----------------------------------------------------------------------

def test_stacked_evaluate_shape():
    K, M, N = 3, 16, 80
    vb = VectorBasis.random(input_dim=2, n_features=N, n_components=K)
    x = torch.rand(M, 2, device=device)
    H = vb.evaluate(x)
    assert H.shape == (M, K, N)


def test_stacked_gradient_and_laplacian_shapes():
    K, M, N, d = 2, 20, 60, 3
    vb = VectorBasis.random(input_dim=d, n_features=N, n_components=K,
                            sigma=2.0)
    x = torch.rand(M, d, device=device)
    G = vb.gradient(x)
    L = vb.laplacian(x)
    H = vb.hessian_diag(x)
    Dxy = vb.derivative(x, alpha=(1, 1, 0))
    assert G.shape == (M, K, d, N)
    assert L.shape == (M, K, N)
    assert H.shape == (M, K, d, N)
    assert Dxy.shape == (M, K, N)


def test_stacked_matches_per_component():
    """The stacked output must equal the per-component outputs."""
    K, M, N = 2, 8, 40
    vb = VectorBasis.random(input_dim=2, n_features=N, n_components=K,
                            sigma=1.5)
    x = torch.rand(M, 2, device=device)
    H_stacked = vb.evaluate(x)
    for k in range(K):
        assert torch.allclose(H_stacked[:, k, :],
                              vb.component(k).evaluate(x))


# ----------------------------------------------------------------------
# Block-diagonal helpers
# ----------------------------------------------------------------------

def test_block_diag_evaluate_shape():
    K, M, N = 2, 7, 30
    vb = VectorBasis.random(input_dim=2, n_features=N, n_components=K)
    x = torch.rand(M, 2, device=device)
    B = vb.block_diag_evaluate(x)
    assert B.shape == (K * M, K * N)
    # block (0,0) should match component 0 evaluate
    assert torch.allclose(B[:M, :N], vb.component(0).evaluate(x))
    # off-diagonal blocks must be zero
    assert torch.allclose(B[:M, N : 2 * N], torch.zeros_like(B[:M, N : 2 * N]))


def test_block_diag_laplacian_shape():
    K, M, N = 2, 5, 20
    vb = VectorBasis.random(input_dim=2, n_features=N, n_components=K)
    x = torch.rand(M, 2, device=device)
    BL = vb.block_diag_laplacian(x)
    assert BL.shape == (K * M, K * N)


# ----------------------------------------------------------------------
# Coefficient packing and prediction
# ----------------------------------------------------------------------

def test_stack_and_unstack_betas_roundtrip():
    K, N = 3, 50
    vb = VectorBasis.random(input_dim=2, n_features=N, n_components=K)
    betas = [torch.randn(N, 1, device=device) for _ in range(K)]
    stacked = vb.stack_betas(betas)
    assert stacked.shape == (K * N, 1)
    restored = vb.unstack_beta(stacked)
    for a, b in zip(betas, restored):
        assert torch.allclose(a, b)


def test_predict_accepts_list_stacked_and_matrix():
    K, M, N = 2, 10, 40
    vb = VectorBasis.random(input_dim=2, n_features=N, n_components=K)
    x = torch.rand(M, 2, device=device)
    betas = [torch.randn(N, 1, device=device) for _ in range(K)]
    y1 = vb.predict(x, betas)
    y2 = vb.predict(x, vb.stack_betas(betas))
    y3 = vb.predict(x, torch.cat(betas, dim=1))      # (N, K)
    assert y1.shape == (M, K)
    assert torch.allclose(y1, y2)
    assert torch.allclose(y1, y3)


# ----------------------------------------------------------------------
# VectorFastLSQSolver
# ----------------------------------------------------------------------

def test_vector_solver_basic():
    K = 2
    solver = VectorFastLSQSolver(input_dim=2, n_components=K)
    solver.add_block(hidden_size=80, scale=1.5)
    solver.add_block(hidden_size=80, scale=1.5)
    assert solver.n_components == K
    assert solver.n_features_per_component == 160
    assert solver.n_features_total == 320
    vb = solver.basis
    assert isinstance(vb, VectorBasis)
    assert vb.n_components == K


def test_vector_solver_per_component_scale():
    K = 2
    solver = VectorFastLSQSolver(input_dim=2, n_components=K)
    solver.add_block(hidden_size=100, scale=[1.0, 4.0])
    s0 = float(solver.component_solver(0).basis.W.std())
    s1 = float(solver.component_solver(1).basis.W.std())
    assert s1 > 2.5 * s0   # very different bandwidths


def test_vector_solver_predict_end_to_end():
    """Smoke test: fit two unrelated scalar fields jointly with one
    VectorFastLSQSolver.  Component 0 := sin(pi x) sin(pi y),
    component 1 := exp(-(x^2 + y^2))."""
    torch.manual_seed(0); np.random.seed(0)
    K = 2; N = 400; M = 1000
    solver = VectorFastLSQSolver(input_dim=2, n_components=K, normalize=True)
    solver.add_block(hidden_size=N, scale=4.0)

    x_pts = torch.rand(M, 2, device=device) * 2 - 1
    y0 = (torch.sin(np.pi * x_pts[:, 0:1]) *
          torch.sin(np.pi * x_pts[:, 1:2]))
    y1 = torch.exp(-(x_pts[:, 0:1] ** 2 + x_pts[:, 1:2] ** 2))

    # Solve each component independently via a tiny regression
    betas = []
    for k, y in enumerate([y0, y1]):
        H = solver.basis.component(k).evaluate(x_pts)
        beta = solve_lstsq(H, y, mu=1e-8)
        betas.append(beta)
    solver.beta = betas

    # Evaluate at new points
    x_test = torch.rand(50, 2, device=device) * 2 - 1
    y_pred = solver.predict(x_test)
    y0_ex = (torch.sin(np.pi * x_test[:, 0:1]) *
             torch.sin(np.pi * x_test[:, 1:2]))
    y1_ex = torch.exp(-(x_test[:, 0:1] ** 2 + x_test[:, 1:2] ** 2))
    err0 = (y_pred[:, 0:1] - y0_ex).abs().mean().item()
    err1 = (y_pred[:, 1:2] - y1_ex).abs().mean().item()
    assert err0 < 0.05, err0
    assert err1 < 0.05, err1
