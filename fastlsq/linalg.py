# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Linear algebra helpers for FastLSQ."""

import torch


def solve_lstsq(A, b, mu=0.0):
    """Solve  min ||Ax - b||^2 + mu ||x||^2  (Tikhonov-regularised least squares).

    Parameters
    ----------
    A : Tensor, shape (M, N)
    b : Tensor, shape (M, 1)
    mu : float
        Tikhonov regularisation parameter.  When mu = 0, falls back to
        ``torch.linalg.lstsq``.  When mu > 0, forms the normal equations
        (A^T A + mu I) x = A^T b  and solves via Cholesky.

    Returns
    -------
    x : Tensor, shape (N, 1)
    """
    if mu <= 0:
        return torch.linalg.lstsq(A, b).solution
    AtA = A.T @ A
    Atb = A.T @ b
    AtA.diagonal().add_(mu)
    try:
        L = torch.linalg.cholesky(AtA)
        return torch.cholesky_solve(Atb, L)
    except torch.linalg.LinAlgError:
        return torch.linalg.solve(AtA, Atb)
