#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Example: Extending FastLSQ with custom feature maps.

This shows how to create a solver with custom activation functions
(e.g., cosine instead of sine, or learned features).
"""

import torch
import numpy as np
from fastlsq.solvers import FastLSQSolver
from fastlsq.linalg import solve_lstsq
from fastlsq.utils import device


class CustomFeatureSolver(FastLSQSolver):
    """FastLSQ solver with cosine activation instead of sine."""

    def get_features(self, x):
        """Override to use cosine activation."""
        Hs, dHs, ddHs = [], [], []
        for W, b in zip(self.W_list, self.b_list):
            Z = x @ W + b
            cos_Z = torch.cos(Z)
            sin_Z = torch.sin(Z)
            Hs.append(cos_Z)
            dHs.append(-sin_Z.unsqueeze(1) * W.unsqueeze(0))
            ddHs.append(-cos_Z.unsqueeze(1) * (W ** 2).unsqueeze(0))

        H = torch.cat(Hs, -1)
        dH = torch.cat(dHs, -1)
        ddH = torch.cat(ddHs, -1)

        if self.normalize:
            norm = np.sqrt(self._n_features)
            H = H / norm
            dH = dH / norm
            ddH = ddH / norm

        return H, dH, ddH


# Example usage
if __name__ == "__main__":
    from fastlsq.problems.linear import PoissonND

    torch.set_default_dtype(torch.float32)
    problem = PoissonND()

    # Use custom solver
    solver = CustomFeatureSolver(problem.dim, normalize=False)
    for _ in range(3):
        solver.add_block(hidden_size=500, scale=5.0)

    x_pde, bcs, f_pde = problem.get_train_data(n_pde=5000, n_bc=1000)
    A, b = problem.build(solver, x_pde, bcs, f_pde)
    solver.beta = solve_lstsq(A, b)

    # Evaluate
    from fastlsq.utils import evaluate_error
    val_err, grad_err = evaluate_error(solver, problem)
    print(f"Custom cosine solver: val_err={val_err:.2e}, grad_err={grad_err:.2e}")
