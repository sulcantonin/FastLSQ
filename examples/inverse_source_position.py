#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Minimal inverse problem: recover Gaussian source position (x_s, y_s) from sensor data.

PDE: -Delta u = f,  f = Gaussian at (x_s, y_s),  u=0 on boundary.
No exact solution needed -- synthetic observations from forward model.
"""

import torch
import numpy as np
from scipy.optimize import minimize

from fastlsq import Op, solve_lstsq
from fastlsq.basis import SinusoidalBasis
from fastlsq.geometry import sample_box, sample_boundary_box


def main():
    pde_op = -Op.laplacian(d=2)
    basis = SinusoidalBasis.random(input_dim=2, n_features=800, sigma=5.0, normalize=True)
    x_pde = sample_box(3000, 2)
    x_bc = sample_boundary_box(400, 2)
    cache = basis.cache(x_pde)
    A_pde = pde_op.apply(basis, x_pde, cache=cache)
    A = torch.cat([A_pde, 100 * basis.evaluate(x_bc)])
    x_sens = torch.tensor([[0.3, 0.3], [0.7, 0.7], [0.3, 0.7], [0.7, 0.3]])

    def fwd(xs, ys):
        b = torch.exp(-((x_pde[:, 0] - xs) ** 2 + (x_pde[:, 1] - ys) ** 2) / 0.1).unsqueeze(1)
        b = torch.cat([b, torch.zeros(400, 1, device=b.device, dtype=b.dtype)])
        beta = solve_lstsq(A, b)
        return (basis.evaluate(x_sens) @ beta).detach().cpu().numpy().ravel()

    # True source at (0.4, 0.6), add small noise
    u_obs = fwd(0.4, 0.6) + 0.01 * np.random.randn(4)

    def loss(p):
        return np.sum((fwd(float(p[0]), float(p[1])) - u_obs) ** 2)

    result = minimize(loss, [0.5, 0.5], method="L-BFGS-B", bounds=[(0.1, 0.9)] * 2)
    print(f"True:  (0.4, 0.6)")
    print(f"Found: ({result.x[0]:.4f}, {result.x[1]:.4f})")
    print(f"Loss:  {result.fun:.2e}")


if __name__ == "__main__":
    main()
