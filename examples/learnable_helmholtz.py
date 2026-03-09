#!/usr/bin/env python3
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Learnable Helmholtz wavenumber: plug nn.Parameter into the Op + AdamW.

Demonstrates how to optimise PDE operator coefficients (e.g. k in Helmholtz)
by building the operator inside forward() so gradients flow through the
prebuilt linear solve.
"""

import torch
import torch.nn as nn
import numpy as np

from fastlsq.basis import SinusoidalBasis, Op
from fastlsq.geometry import sample_box, sample_boundary_box
from fastlsq.utils import device


class LearnableHelmholtz(nn.Module):
    """Helmholtz (Δ + k²)u = f with learnable wavenumber k.

    The operator is rebuilt each forward() so k is always current.
    Gradients flow: loss -> A @ beta - b -> lstsq -> A -> k.
    """

    def __init__(self, n_features: int = 1000, init_k: float = 8.0):
        super().__init__()
        self.basis = SinusoidalBasis.random(
            input_dim=2, n_features=n_features, sigma=5.0, normalize=True
        )
        self.k = nn.Parameter(torch.tensor(init_k, device=device))
        self.beta: torch.Tensor | None = None

    def build_system(self, x_pde, x_bc, f_pde, u_bc):
        """Assemble A beta = b. Operator uses current self.k."""
        helmholtz = Op.laplacian(d=2) + self.k**2 * Op.identity(d=2)
        cache = self.basis.cache(x_pde)
        A_pde = helmholtz.apply(self.basis, x_pde, cache=cache)
        A = torch.cat([A_pde, 100.0 * self.basis.evaluate(x_bc)])
        b = torch.cat([f_pde, 100.0 * u_bc])
        return A, b

    def forward(self, x_pde, x_bc, f_pde, u_bc):
        A, b = self.build_system(x_pde, x_bc, f_pde, u_bc)
        self.beta = torch.linalg.lstsq(A, b).solution
        return self.basis.evaluate(x_pde) @ self.beta

    def predict(self, x):
        return self.basis.evaluate(x) @ self.beta


def main():
    # Synthetic Helmholtz: u = sin(k*x)*sin(k*y), source f = -k² u
    k_true = 10.0

    def exact(x):
        return torch.sin(k_true * x[:, 0:1]) * torch.sin(k_true * x[:, 1:2])

    def source(x):
        return -(k_true**2) * exact(x)

    x_pde = sample_box(2000, 2)
    x_bc = sample_boundary_box(400, 2)
    f_pde = source(x_pde)
    u_bc = exact(x_bc)

    model = LearnableHelmholtz(n_features=500, init_k=8.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    print("Optimising Helmholtz wavenumber k (true k = 10.0)")
    print("-" * 50)
    for step in range(50):
        optimizer.zero_grad()
        u_pred = model(x_pde, x_bc, f_pde, u_bc)
        loss = torch.nn.functional.mse_loss(u_pred, exact(x_pde))
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            print(f"  Step {step:3d}: k = {model.k.item():.4f}, loss = {loss.item():.2e}")

    print("-" * 50)
    print(f"Final k = {model.k.item():.4f} (true = {k_true})")


if __name__ == "__main__":
    main()
