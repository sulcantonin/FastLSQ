# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Nonlinear PDE problems solved via Newton-Raphson iteration.

Each class provides:
    exact(x)                 -- analytical solution
    exact_grad(x)            -- analytical gradient
    source(x)                -- right-hand side
    get_train_data()         -- collocation and boundary points
    build_newton_step(…)     -- Jacobian J and negative residual -R
    build_linear_init(…)     -- linear system for warm start
    get_test_points()        -- random test points for evaluation
"""

import torch
import numpy as np

from fastlsq.utils import device


# ======================================================================
# Helper: boundary points on the unit square
# ======================================================================

def _unit_square_boundary(n_bc):
    """Generate n_bc random points on the boundary of [0,1]^2."""
    n_side = n_bc // 4
    r = lambda n: torch.rand(n, 1, device=device)
    z = lambda n: torch.zeros(n, 1, device=device)
    o = lambda n: torch.ones(n, 1, device=device)
    return torch.cat([
        torch.cat([z(n_side), r(n_side)], 1),
        torch.cat([o(n_side), r(n_side)], 1),
        torch.cat([r(n_side), z(n_side)], 1),
        torch.cat([r(n_side), o(n_side)], 1),
    ], 0)


# ======================================================================
# Nonlinear Poisson with cubic term (2-D)
# ======================================================================

class NLPoisson2D:
    """Nonlinear Poisson:  -Laplacian(u) + u^3 = f  on [0,1]^2.

    Exact solution: u = sin(pi x) sin(pi y).
    """

    def __init__(self):
        self.name = "NL-Poisson 2D (u^3)"
        self.dim = 2
        self.lam_bc = 100.0

    def exact(self, x):
        return torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2])

    def exact_grad(self, x):
        sx = torch.sin(np.pi * x[:, 0:1])
        cx = torch.cos(np.pi * x[:, 0:1])
        sy = torch.sin(np.pi * x[:, 1:2])
        cy = torch.cos(np.pi * x[:, 1:2])
        return torch.cat([np.pi * cx * sy, np.pi * sx * cy], dim=1)

    def source(self, x):
        u = self.exact(x)
        return 2 * np.pi ** 2 * u + u ** 3

    def get_train_data(self, n_pde=5000, n_bc=1000):
        x_pde = torch.rand(n_pde, 2, device=device)
        f_pde = self.source(x_pde)
        x_bc = _unit_square_boundary(n_bc)
        u_bc = self.exact(x_bc)
        return x_pde, [(x_bc, u_bc)], f_pde

    def build_newton_step(self, solver, x_pde, bcs, f_pde):
        H, dH, ddH = solver.get_features(x_pde)
        beta = solver.beta
        lap_feat = torch.sum(ddH, dim=1)
        u_k = H @ beta
        lap_uk = lap_feat @ beta
        R = -lap_uk + u_k ** 3 - f_pde
        J_pde = -lap_feat + 3 * (u_k ** 2) * H
        rows_J, rows_b = [J_pde], [-R]
        for (x_bc, u_bc) in bcs:
            H_bc, _, _ = solver.get_features(x_bc)
            rows_J.append(self.lam_bc * H_bc)
            rows_b.append(self.lam_bc * (u_bc - H_bc @ beta))
        return torch.cat(rows_J, 0), torch.cat(rows_b, 0)

    def build_linear_init(self, solver, x_pde, bcs, f_pde):
        H, dH, ddH = solver.get_features(x_pde)
        lap = torch.sum(ddH, dim=1)
        rows_A, rows_b = [-lap], [f_pde]
        for (x_bc, u_bc) in bcs:
            H_bc, _, _ = solver.get_features(x_bc)
            rows_A.append(self.lam_bc * H_bc)
            rows_b.append(self.lam_bc * u_bc)
        return torch.cat(rows_A, 0), torch.cat(rows_b, 0)

    def get_test_points(self, n=5000):
        return torch.rand(n, 2, device=device)


# ======================================================================
# Bratu equation (2-D)
# ======================================================================

class Bratu2D:
    """Bratu equation:  -Laplacian(u) - lambda exp(u) = f  on [0,1]^2.

    Exact solution: u = sin(pi x) sin(pi y).
    """

    def __init__(self, lam=1.0):
        self.name = f"Bratu 2D (lam={lam})"
        self.dim = 2
        self.lam = lam
        self.lam_bc = 100.0

    def exact(self, x):
        return torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2])

    def exact_grad(self, x):
        sx = torch.sin(np.pi * x[:, 0:1])
        cx = torch.cos(np.pi * x[:, 0:1])
        sy = torch.sin(np.pi * x[:, 1:2])
        cy = torch.cos(np.pi * x[:, 1:2])
        return torch.cat([np.pi * cx * sy, np.pi * sx * cy], dim=1)

    def source(self, x):
        u = self.exact(x)
        return 2 * np.pi ** 2 * u - self.lam * torch.exp(u)

    def get_train_data(self, n_pde=5000, n_bc=1000):
        x_pde = torch.rand(n_pde, 2, device=device)
        f_pde = self.source(x_pde)
        x_bc = _unit_square_boundary(n_bc)
        u_bc = self.exact(x_bc)
        return x_pde, [(x_bc, u_bc)], f_pde

    def build_newton_step(self, solver, x_pde, bcs, f_pde):
        H, dH, ddH = solver.get_features(x_pde)
        beta = solver.beta
        lap_feat = torch.sum(ddH, dim=1)
        u_k = H @ beta
        lap_uk = lap_feat @ beta
        exp_uk = torch.exp(torch.clamp(u_k, max=20.0))
        R = -lap_uk - self.lam * exp_uk - f_pde
        J_pde = -lap_feat - self.lam * exp_uk * H
        rows_J, rows_b = [J_pde], [-R]
        for (x_bc, u_bc) in bcs:
            H_bc, _, _ = solver.get_features(x_bc)
            rows_J.append(self.lam_bc * H_bc)
            rows_b.append(self.lam_bc * (u_bc - H_bc @ beta))
        return torch.cat(rows_J, 0), torch.cat(rows_b, 0)

    def build_linear_init(self, solver, x_pde, bcs, f_pde):
        H, dH, ddH = solver.get_features(x_pde)
        lap = torch.sum(ddH, dim=1)
        rows_A, rows_b = [-lap], [f_pde]
        for (x_bc, u_bc) in bcs:
            H_bc, _, _ = solver.get_features(x_bc)
            rows_A.append(self.lam_bc * H_bc)
            rows_b.append(self.lam_bc * u_bc)
        return torch.cat(rows_A, 0), torch.cat(rows_b, 0)

    def get_test_points(self, n=5000):
        return torch.rand(n, 2, device=device)


# ======================================================================
# Steady viscous Burgers equation (1-D)
# ======================================================================

class SteadyBurgers1D:
    """Steady Burgers:  u u_x - nu u_xx = f  on [0,1].

    Exact solution: u = sin(2 pi x).
    Supports continuation (viscosity marching).
    """

    def __init__(self, nu=0.1):
        self.name = f"Steady Burgers 1D (nu={nu})"
        self.dim = 1
        self.nu = nu
        self.nu_target = nu
        self.lam_bc = 200.0
        self.use_continuation = True
        self.continuation_schedule = [1.0, 0.5, 0.2, 0.1]

    def exact(self, x):
        return torch.sin(2 * np.pi * x[:, 0:1])

    def exact_grad(self, x):
        return 2 * np.pi * torch.cos(2 * np.pi * x[:, 0:1])

    def source(self, x):
        u = self.exact(x)
        ux = self.exact_grad(x)
        uxx = -(2 * np.pi) ** 2 * torch.sin(2 * np.pi * x[:, 0:1])
        return u * ux - self.nu * uxx

    def get_train_data(self, n_pde=3000, n_bc=200):
        x_pde = torch.rand(n_pde, 1, device=device)
        f_pde = self.source(x_pde)
        x_bc = torch.cat([
            torch.zeros(n_bc // 2, 1, device=device),
            torch.ones(n_bc // 2, 1, device=device),
        ], 0)
        u_bc = torch.zeros(n_bc, 1, device=device)
        return x_pde, [(x_bc, u_bc)], f_pde

    def build_newton_step(self, solver, x_pde, bcs, f_pde):
        H, dH, ddH = solver.get_features(x_pde)
        beta = solver.beta
        u_k = H @ beta
        ux_feat = dH[:, 0, :]
        ux_k = ux_feat @ beta
        uxx_feat = ddH[:, 0, :]
        uxx_k = uxx_feat @ beta
        R = u_k * ux_k - self.nu * uxx_k - f_pde
        J_pde = u_k * ux_feat + ux_k * H - self.nu * uxx_feat
        rows_J, rows_b = [J_pde], [-R]
        for (x_bc, u_bc) in bcs:
            H_bc, _, _ = solver.get_features(x_bc)
            rows_J.append(self.lam_bc * H_bc)
            rows_b.append(self.lam_bc * (u_bc - H_bc @ beta))
        return torch.cat(rows_J, 0), torch.cat(rows_b, 0)

    def build_linear_init(self, solver, x_pde, bcs, f_pde):
        _, _, ddH = solver.get_features(x_pde)
        uxx_feat = ddH[:, 0, :]
        rows_A, rows_b = [-self.nu * uxx_feat], [f_pde]
        for (x_bc, u_bc) in bcs:
            H_bc, _, _ = solver.get_features(x_bc)
            rows_A.append(self.lam_bc * H_bc)
            rows_b.append(self.lam_bc * u_bc)
        return torch.cat(rows_A, 0), torch.cat(rows_b, 0)

    def get_test_points(self, n=5000):
        return torch.rand(n, 1, device=device)


# ======================================================================
# Nonlinear Helmholtz with cubic term (2-D)
# ======================================================================

class NLHelmholtz2D:
    """Nonlinear Helmholtz:  Laplacian(u) + k^2 u + alpha u^3 = f  on [0,1]^2.

    Exact solution: u = sin(k x) sin(k y).
    """

    def __init__(self, k=3.0, alpha=0.5):
        self.name = f"NL-Helmholtz 2D (k={k})"
        self.dim = 2
        self.k = k
        self.alpha = alpha
        self.lam_bc = 100.0

    def exact(self, x):
        return torch.sin(self.k * x[:, 0:1]) * torch.sin(self.k * x[:, 1:2])

    def exact_grad(self, x):
        k = self.k
        sx = torch.sin(k * x[:, 0:1])
        cx = torch.cos(k * x[:, 0:1])
        sy = torch.sin(k * x[:, 1:2])
        cy = torch.cos(k * x[:, 1:2])
        return torch.cat([k * cx * sy, k * sx * cy], dim=1)

    def source(self, x):
        u = self.exact(x)
        return -self.k ** 2 * u + self.alpha * u ** 3

    def get_train_data(self, n_pde=5000, n_bc=1000):
        x_pde = torch.rand(n_pde, 2, device=device)
        f_pde = self.source(x_pde)
        x_bc = _unit_square_boundary(n_bc)
        u_bc = self.exact(x_bc)
        return x_pde, [(x_bc, u_bc)], f_pde

    def build_newton_step(self, solver, x_pde, bcs, f_pde):
        H, dH, ddH = solver.get_features(x_pde)
        beta = solver.beta
        lap_feat = torch.sum(ddH, dim=1)
        u_k = H @ beta
        lap_uk = lap_feat @ beta
        R = lap_uk + self.k ** 2 * u_k + self.alpha * u_k ** 3 - f_pde
        J_pde = lap_feat + self.k ** 2 * H + 3 * self.alpha * (u_k ** 2) * H
        rows_J, rows_b = [J_pde], [-R]
        for (x_bc, u_bc) in bcs:
            H_bc, _, _ = solver.get_features(x_bc)
            rows_J.append(self.lam_bc * H_bc)
            rows_b.append(self.lam_bc * (u_bc - H_bc @ beta))
        return torch.cat(rows_J, 0), torch.cat(rows_b, 0)

    def build_linear_init(self, solver, x_pde, bcs, f_pde):
        H, _, ddH = solver.get_features(x_pde)
        lap = torch.sum(ddH, dim=1)
        A_pde = lap + self.k ** 2 * H
        rows_A, rows_b = [A_pde], [f_pde]
        for (x_bc, u_bc) in bcs:
            H_bc, _, _ = solver.get_features(x_bc)
            rows_A.append(self.lam_bc * H_bc)
            rows_b.append(self.lam_bc * u_bc)
        return torch.cat(rows_A, 0), torch.cat(rows_b, 0)

    def get_test_points(self, n=5000):
        return torch.rand(n, 2, device=device)


# ======================================================================
# Steady Allen-Cahn equation (1-D)
# ======================================================================

class AllenCahn1D:
    """Allen-Cahn:  eps u_xx + u - u^3 = f  on [0,1].

    Exact solution: u = sin(pi x).
    """

    def __init__(self, eps=0.1):
        self.name = f"Allen-Cahn 1D (eps={eps})"
        self.dim = 1
        self.eps = eps
        self.lam_bc = 200.0

    def exact(self, x):
        return torch.sin(np.pi * x[:, 0:1])

    def exact_grad(self, x):
        return np.pi * torch.cos(np.pi * x[:, 0:1])

    def source(self, x):
        u = self.exact(x)
        uxx = -(np.pi ** 2) * u
        return self.eps * uxx + u - u ** 3

    def get_train_data(self, n_pde=3000, n_bc=200):
        x_pde = torch.rand(n_pde, 1, device=device)
        f_pde = self.source(x_pde)
        x_bc = torch.cat([
            torch.zeros(n_bc // 2, 1, device=device),
            torch.ones(n_bc // 2, 1, device=device),
        ], 0)
        u_bc = torch.zeros(n_bc, 1, device=device)
        return x_pde, [(x_bc, u_bc)], f_pde

    def build_newton_step(self, solver, x_pde, bcs, f_pde):
        H, dH, ddH = solver.get_features(x_pde)
        beta = solver.beta
        uxx_feat = ddH[:, 0, :]
        u_k = H @ beta
        uxx_k = uxx_feat @ beta
        R = self.eps * uxx_k + u_k - u_k ** 3 - f_pde
        J_pde = self.eps * uxx_feat + (1 - 3 * u_k ** 2) * H
        rows_J, rows_b = [J_pde], [-R]
        for (x_bc, u_bc) in bcs:
            H_bc, _, _ = solver.get_features(x_bc)
            rows_J.append(self.lam_bc * H_bc)
            rows_b.append(self.lam_bc * (u_bc - H_bc @ beta))
        return torch.cat(rows_J, 0), torch.cat(rows_b, 0)

    def build_linear_init(self, solver, x_pde, bcs, f_pde):
        H, _, ddH = solver.get_features(x_pde)
        uxx = ddH[:, 0, :]
        rows_A = [self.eps * uxx + H]
        rows_b = [f_pde]
        for (x_bc, u_bc) in bcs:
            H_bc, _, _ = solver.get_features(x_bc)
            rows_A.append(self.lam_bc * H_bc)
            rows_b.append(self.lam_bc * u_bc)
        return torch.cat(rows_A, 0), torch.cat(rows_b, 0)

    def get_test_points(self, n=5000):
        return torch.rand(n, 1, device=device)
