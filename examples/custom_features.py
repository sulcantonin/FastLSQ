#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Example: Extending FastLSQ with custom feature maps.

Demonstrates how to create cosine features using the quarter-wave identity
cos(z) = sin(z + pi/2).  Since `SinusoidalBasis` handles arbitrary
biases, cosine features are just sine features with a bias shift -- no
subclassing needed.  All derivative machinery (gradient, Hessian, Laplacian)
works automatically.
"""

import torch
import numpy as np
from fastlsq.basis import SinusoidalBasis
from fastlsq.linalg import solve_lstsq
from fastlsq.utils import device


# ======================================================================
# Create cosine features via phase shift
# ======================================================================

def cosine_basis(input_dim, n_features, sigma=5.0):
    """Create cosine features: cos(z) = sin(z + pi/2)."""
    W = torch.randn(input_dim, n_features, device=device) * sigma
    b = torch.rand(1, n_features, device=device) * 2 * np.pi + (np.pi / 2)
    return SinusoidalBasis(W, b, normalize=False)


if __name__ == "__main__":
    from fastlsq.problems.linear import PoissonND

    torch.set_default_dtype(torch.float32)
    problem = PoissonND()

    print("Cosine features via SinusoidalBasis with cos(z) = sin(z + pi/2)")
    basis = cosine_basis(problem.dim, n_features=1500, sigma=5.0)
    x_pde, bcs, f_pde = problem.get_train_data(n_pde=5000, n_bc=1000)

    cache = basis.cache(x_pde)
    A_pde = -basis.laplacian(x_pde, cache=cache)
    H_bc = basis.evaluate(bcs[0][0])
    A = torch.cat([A_pde, 100.0 * H_bc])
    b = torch.cat([f_pde, 100.0 * bcs[0][1]])
    beta = solve_lstsq(A, b)

    x_test = problem.get_test_points(2000)
    u_pred = basis.evaluate(x_test) @ beta
    u_true = problem.exact(x_test)
    err = (torch.norm(u_pred - u_true) / (torch.norm(u_true) + 1e-15)).item()
    print(f"  Value error: {err:.2e}")
