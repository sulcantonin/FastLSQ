# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Linear PDE problems solved in *solver mode* (single least-squares solve).

Each class provides:
    exact(x)         -- analytical solution
    exact_grad(x)    -- analytical gradient
    source(x)        -- right-hand side (if applicable)
    get_train_data() -- collocation and boundary points
    build(solver, …) -- assemble the linear system  A beta = b
    get_test_points() -- random test points for evaluation
"""

import torch
import numpy as np

from fastlsq.utils import device


# ======================================================================
# Poisson equation on the unit hypercube (5-D)
# ======================================================================

class PoissonND:
    """Poisson equation  -Laplacian(u) = f  on [0,1]^5."""

    def __init__(self):
        self.name = "Poisson 5D"
        self.dim = 5

    def exact(self, x):
        return torch.sum(torch.sin(np.pi / 2 * x), dim=1, keepdim=True)

    def exact_grad(self, x):
        return (np.pi / 2) * torch.cos(np.pi / 2 * x)

    def source(self, x):
        return (np.pi ** 2 / 4) * torch.sum(
            torch.sin(np.pi / 2 * x), dim=1, keepdim=True
        )

    def get_train_data(self, n_pde=10000, n_bc=2000):
        x_pde = torch.rand(n_pde, self.dim, device=device)
        f_pde = self.source(x_pde)
        x_bc = torch.rand(n_bc, self.dim, device=device)
        mask_dim = torch.randint(0, self.dim, (n_bc,), device=device)
        mask_val = torch.randint(0, 2, (n_bc,), device=device).float()
        for i in range(n_bc):
            x_bc[i, mask_dim[i]] = mask_val[i]
        u_bc = self.exact(x_bc)
        return x_pde, [(x_bc, u_bc)], f_pde

    def build(self, slv, x_pde, bcs, f_pde):
        basis = slv.basis
        cache = basis.cache(x_pde)
        A = -basis.laplacian(x_pde, cache=cache)
        b = f_pde
        As, bs = [A], [b]
        for (pts, vals) in bcs:
            h = basis.evaluate(pts)
            As.append(h * 100.0)
            bs.append(vals * 100.0)
        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=10000):
        return torch.rand(n, self.dim, device=device)


# ======================================================================
# Heat equation on the unit ball x time (5 spatial dims + time)
# ======================================================================

class HeatND:
    """Heat equation  u_t - k Laplacian(u) = f  on B^5 x [0,1]."""

    def __init__(self):
        self.name = "Heat 5D"
        self.dim = 6
        self.d = 5
        self.k = 1.0 / 5.0

    def exact(self, x):
        r2 = torch.sum(x[:, 0:5] ** 2, dim=1, keepdim=True)
        t = x[:, 5:6]
        return torch.exp(0.5 * r2 + t)

    def exact_grad(self, x):
        u = self.exact(x)
        grad_spatial = x[:, 0:5] * u
        grad_time = u
        return torch.cat([grad_spatial, grad_time], dim=1)

    def source(self, x):
        r2 = torch.sum(x[:, 0:5] ** 2, dim=1, keepdim=True)
        return -(1.0 / self.d) * r2 * self.exact(x)

    def sample_sphere_time(self, n):
        pts = torch.randn(n, 5, device=device)
        pts = pts / torch.norm(pts, dim=1, keepdim=True)
        r = torch.rand(n, 1, device=device) ** (1 / 5.0)
        spatial = pts * r
        time = torch.rand(n, 1, device=device)
        return torch.cat([spatial, time], dim=1)

    def get_train_data(self, n_pde=10000, n_bc=2000):
        x_pde = self.sample_sphere_time(n_pde)
        f_pde = self.source(x_pde)
        x_ic_space = self.sample_sphere_time(n_bc)[:, 0:5]
        x_ic = torch.cat([x_ic_space, torch.zeros(n_bc, 1, device=device)], 1)
        u_ic = self.exact(x_ic)
        x_bc_space = torch.randn(n_bc, 5, device=device)
        x_bc_space = x_bc_space / torch.norm(x_bc_space, dim=1, keepdim=True)
        t_bc = torch.rand(n_bc, 1, device=device)
        x_bc = torch.cat([x_bc_space, t_bc], 1)
        g_bc = self.exact(x_bc)
        return x_pde, [
            (x_ic, u_ic, "dirichlet"),
            (x_bc, g_bc, "neumann_n"),
        ], f_pde

    def build(self, slv, x_pde, bcs, f_pde):
        basis = slv.basis
        cache = basis.cache(x_pde)
        lap = basis.laplacian(x_pde, dims=range(5), cache=cache)
        u_t = basis.gradient(x_pde, cache=cache)[:, 5, :]
        A = u_t - self.k * lap
        b = f_pde
        As, bs = [A], [b]
        for (pts, vals, type_) in bcs:
            h = basis.evaluate(pts)
            dh = basis.gradient(pts)
            w = 100.0
            if type_ == "dirichlet":
                As.append(h * w)
            elif type_ == "neumann_n":
                neumann_term = torch.zeros_like(h)
                for i in range(5):
                    neumann_term += pts[:, i : i + 1] * dh[:, i, :]
                As.append(neumann_term * w)
            bs.append(vals * w)
        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=10000):
        return self.sample_sphere_time(n)


# ======================================================================
# Wave equation in 1-D
# ======================================================================

class Wave1D:
    """Wave equation  u_tt = c^2 u_xx  on [0,1] x [0,1]."""

    def __init__(self):
        self.name = "Wave 1D"
        self.dim = 2
        self.c2 = 4.0

    def exact(self, x):
        xv, tv = x[:, 0:1], x[:, 1:2]
        return (torch.sin(np.pi * xv) * torch.cos(2 * np.pi * tv)
                + 0.5 * torch.sin(4 * np.pi * xv) * torch.cos(8 * np.pi * tv))

    def exact_grad(self, x):
        xv, tv = x[:, 0:1], x[:, 1:2]
        ux = (np.pi * torch.cos(np.pi * xv) * torch.cos(2 * np.pi * tv)
              + 2.0 * np.pi * torch.cos(4 * np.pi * xv) * torch.cos(8 * np.pi * tv))
        ut = (-2 * np.pi * torch.sin(np.pi * xv) * torch.sin(2 * np.pi * tv)
              - 4.0 * np.pi * torch.sin(4 * np.pi * xv) * torch.sin(8 * np.pi * tv))
        return torch.cat([ux, ut], dim=1)

    def get_train_data(self, n_pde=5000, n_bc=1000):
        x_pde = torch.rand(n_pde, 2, device=device)
        x_ic = torch.cat([
            torch.rand(n_bc, 1, device=device),
            torch.zeros(n_bc, 1, device=device),
        ], 1)
        u_ic = (torch.sin(np.pi * x_ic[:, 0:1])
                + 0.5 * torch.sin(4 * np.pi * x_ic[:, 0:1]))
        ut_ic = torch.zeros_like(u_ic)
        t_bc = torch.rand(n_bc, 1, device=device)
        x_bc_l = torch.cat([torch.zeros_like(t_bc), t_bc], 1)
        x_bc_r = torch.cat([torch.ones_like(t_bc), t_bc], 1)
        u_bc = torch.zeros_like(t_bc)
        return x_pde, [
            (x_ic, u_ic, "dirichlet"),
            (x_ic, ut_ic, "neumann_t"),
            (x_bc_l, u_bc, "dirichlet"),
            (x_bc_r, u_bc, "dirichlet"),
        ]

    def build(self, slv, x_pde, bcs):
        basis = slv.basis
        cache = basis.cache(x_pde)
        hess_diag = basis.hessian_diag(x_pde, cache=cache)
        A = hess_diag[:, 1, :] - self.c2 * hess_diag[:, 0, :]
        b = torch.zeros(len(x_pde), 1, device=device)
        As, bs = [A], [b]
        for (pts, vals, type_) in bcs:
            h = basis.evaluate(pts)
            dh = basis.gradient(pts)
            w = 100.0
            if type_ == "dirichlet":
                As.append(h * w)
            elif type_ == "neumann_t":
                As.append(dh[:, 1, :] * w)
            bs.append(vals * w)
        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=10000):
        return torch.rand(n, self.dim, device=device)


# ======================================================================
# Wave equation in 2-D (multi-scale, long time)
# ======================================================================

class Wave2D_MS:
    """Wave 2-D multi-scale with time normalisation and frequency compensation.

    Domain: [0,1]^2 x [0, t_max]  (t normalised to [0,1]).
    """

    def __init__(self):
        self.name = "Wave 2D-MS"
        self.dim = 3
        self.a2 = 2.0
        self.t_max = 100.0
        self.scale_multipliers = [1.0, 1.0, 300.0]

    def exact(self, x_in):
        xv = x_in[:, 0:1]
        yv = x_in[:, 1:2]
        tv = x_in[:, 2:3] * self.t_max
        omega = np.pi * np.sqrt(1 + self.a2)
        return (torch.sin(np.pi * xv) * torch.sin(np.pi * yv)
                * torch.cos(omega * tv))

    def exact_grad(self, x_in):
        xv = x_in[:, 0:1]
        yv = x_in[:, 1:2]
        tv = x_in[:, 2:3] * self.t_max
        omega = np.pi * np.sqrt(1 + self.a2)
        u_x = (np.pi * torch.cos(np.pi * xv) * torch.sin(np.pi * yv)
               * torch.cos(omega * tv))
        u_y = (np.pi * torch.sin(np.pi * xv) * torch.cos(np.pi * yv)
               * torch.cos(omega * tv))
        u_t_phys = (-omega * torch.sin(np.pi * xv) * torch.sin(np.pi * yv)
                     * torch.sin(omega * tv))
        u_t_norm = u_t_phys * self.t_max
        return torch.cat([u_x, u_y, u_t_norm], dim=1)

    def get_train_data(self, n_pde=5000, n_bc=1000):
        x_pde = torch.rand(n_pde, 3, device=device)
        x_ic = torch.cat([
            torch.rand(n_bc, 2, device=device),
            torch.zeros(n_bc, 1, device=device),
        ], 1)
        u_ic = self.exact(x_ic)
        ut_ic = torch.zeros(n_bc, 1, device=device)
        x_bc = torch.rand(n_bc, 3, device=device)
        mask = torch.randint(0, 4, (n_bc,), device=device)
        x_bc[mask == 0, 0] = 0
        x_bc[mask == 1, 0] = 1
        x_bc[mask == 2, 1] = 0
        x_bc[mask == 3, 1] = 1
        u_bc = self.exact(x_bc)
        return x_pde, [
            (x_ic, u_ic, "dirichlet"),
            (x_ic, ut_ic, "neumann_t"),
            (x_bc, u_bc, "dirichlet"),
        ], None

    def build(self, slv, x_pde, bcs, f_pde_ignored):
        basis = slv.basis
        cache = basis.cache(x_pde)
        hess_diag = basis.hessian_diag(x_pde, cache=cache)
        u_xx = hess_diag[:, 0, :]
        u_yy = hess_diag[:, 1, :]
        u_tt_norm = hess_diag[:, 2, :]
        A = u_tt_norm - (self.t_max ** 2) * (u_xx + self.a2 * u_yy)
        b = torch.zeros(len(x_pde), 1, device=device)
        As, bs = [A], [b]
        w_bc = 1000.0
        for (pts, vals, type_) in bcs:
            h = basis.evaluate(pts)
            dh = basis.gradient(pts)
            if type_ == "dirichlet":
                As.append(h * w_bc)
            elif type_ == "neumann_t":
                As.append(dh[:, 2, :] * w_bc)
            bs.append(vals * w_bc)
        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=2000):
        return torch.rand(n, 3, device=device)


# ======================================================================
# 2D Elastic wave equation (displacement formulation)
# ======================================================================

class ElasticWave2D:
    """2D elastic wave equation: 2 equations, 2 unknowns (u_x, u_y), 2 wave speeds.

    Displacement formulation:
        u_x_tt = c_p² u_x_xx + c_s² u_x_yy + (c_p² - c_s²) u_y_xy
        u_y_tt = c_p² u_y_yy + c_s² u_y_xx + (c_p² - c_s²) u_x_xy

    Domain: (x, y, t) in [0,1]³ with t normalised. Uses a P-wave + S-wave
    superposition as exact solution.
    """

    def __init__(self, c_p: float = 2.0, c_s: float = 1.0, t_max: float = 2.0):
        self.name = "Elastic Wave 2D"
        self.dim = 3  # x, y, t
        self.c_p = c_p
        self.c_s = c_s
        self.c_p2 = c_p ** 2
        self.c_s2 = c_s ** 2
        self.c_cross = self.c_p2 - self.c_s2  # coupling coefficient
        self.t_max = t_max
        # Wave numbers for exact solution (P-wave mode)
        self.kx = np.pi
        self.ky = np.pi
        self.omega_p = self.c_p * np.sqrt(self.kx ** 2 + self.ky ** 2)

    def exact_ux(self, x_in):
        """u_x component of exact solution (P-wave: dilatational)."""
        xv, yv, tv = x_in[:, 0:1], x_in[:, 1:2], x_in[:, 2:3] * self.t_max
        return (self.kx * torch.cos(self.kx * xv) * torch.sin(self.ky * yv)
                * torch.cos(self.omega_p * tv))

    def exact_uy(self, x_in):
        """u_y component of exact solution."""
        xv, yv, tv = x_in[:, 0:1], x_in[:, 1:2], x_in[:, 2:3] * self.t_max
        return (self.ky * torch.sin(self.kx * xv) * torch.cos(self.ky * yv)
                * torch.cos(self.omega_p * tv))

    def exact(self, x_in):
        """Stack (u_x, u_y) for BCs. Returns (N, 2)."""
        return torch.cat([self.exact_ux(x_in), self.exact_uy(x_in)], dim=1)

    def exact_ut(self, x_in):
        """Time derivative (u_x_t, u_y_t) for IC."""
        xv, yv, tv = x_in[:, 0:1], x_in[:, 1:2], x_in[:, 2:3] * self.t_max
        fac = -self.omega_p * self.t_max * torch.sin(self.omega_p * tv)
        ux_t = (self.kx * torch.cos(self.kx * xv) * torch.sin(self.ky * yv) * fac)
        uy_t = (self.ky * torch.sin(self.kx * xv) * torch.cos(self.ky * yv) * fac)
        return torch.cat([ux_t, uy_t], dim=1)

    def get_train_data(self, n_pde=5000, n_bc=1000):
        x_pde = torch.rand(n_pde, 3, device=device)
        x_ic = torch.cat([
            torch.rand(n_bc, 2, device=device),
            torch.zeros(n_bc, 1, device=device),
        ], 1)
        u_ic = self.exact(x_ic)
        ut_ic = self.exact_ut(x_ic)
        n_wall = n_bc // 4
        r_t = torch.rand(n_wall, 1, device=device)
        r_s = torch.rand(n_wall, 1, device=device)
        zeros = torch.zeros(n_wall, 1, device=device)
        ones = torch.ones(n_wall, 1, device=device)
        x_bc = torch.cat([
            torch.cat([zeros, r_s, r_t], 1),
            torch.cat([ones, r_s, r_t], 1),
            torch.cat([r_s, zeros, r_t], 1),
            torch.cat([r_s, ones, r_t], 1),
        ], 0)
        u_bc = self.exact(x_bc)
        return x_pde, [
            (x_ic, u_ic, "dirichlet"),
            (x_ic, ut_ic, "neumann_t"),
            (x_bc, u_bc, "dirichlet"),
        ], None

    def build(self, slv, x_pde, bcs, f_pde_ignored):
        """Build block system for coupled (u_x, u_y). Returns A (M, 2N), b (M, 1)."""
        basis = slv.basis
        cache = basis.cache(x_pde)
        N = basis.n_features

        # Derivatives for (x, y, t) with t as dim 2
        u_xx = basis.derivative(x_pde, (2, 0, 0), cache=cache)
        u_yy = basis.derivative(x_pde, (0, 2, 0), cache=cache)
        u_tt = basis.derivative(x_pde, (0, 0, 2), cache=cache)
        u_xy = basis.derivative(x_pde, (1, 1, 0), cache=cache)

        # t is normalised to [0,1]; physical d²/dt² = (1/t_max)² d²/dτ²
        t_scale = self.t_max ** 2

        # PDE1: u_x_tt - c_p² u_x_xx - c_s² u_x_yy - (c_p² - c_s²) u_y_xy = 0
        A1_x = t_scale * u_tt - self.c_p2 * u_xx - self.c_s2 * u_yy
        A1_y = -self.c_cross * u_xy

        # PDE2: u_y_tt - c_p² u_y_yy - c_s² u_y_xx - (c_p² - c_s²) u_x_xy = 0
        A2_x = -self.c_cross * u_xy
        A2_y = t_scale * u_tt - self.c_p2 * u_yy - self.c_s2 * u_xx

        A_pde = torch.cat([
            torch.cat([A1_x, A1_y], dim=1),
            torch.cat([A2_x, A2_y], dim=1),
        ], dim=0)
        b_pde = torch.zeros(2 * len(x_pde), 1, device=device)

        As, bs = [A_pde], [b_pde]
        w_bc = 1000.0

        for (pts, vals, type_) in bcs:
            h = basis.evaluate(pts)
            dh = basis.gradient(pts)
            n_pts = len(pts)
            if type_ == "dirichlet":
                # vals: (N_pts, 2) for u_x, u_y
                H_block_x = torch.cat([h, torch.zeros_like(h)], dim=1)
                H_block_y = torch.cat([torch.zeros_like(h), h], dim=1)
                A_bc = torch.cat([H_block_x, H_block_y], dim=0) * w_bc
                b_bc = torch.cat([vals[:, 0:1], vals[:, 1:2]], dim=0) * w_bc
            elif type_ == "neumann_t":
                dh_t = dh[:, 2, :]
                D_block_x = torch.cat([dh_t, torch.zeros_like(dh_t)], dim=1)
                D_block_y = torch.cat([torch.zeros_like(dh_t), dh_t], dim=1)
                A_bc = torch.cat([D_block_x, D_block_y], dim=0) * w_bc
                b_bc = torch.cat([vals[:, 0:1], vals[:, 1:2]], dim=0) * w_bc
            else:
                continue
            As.append(A_bc)
            bs.append(b_bc)

        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=2000):
        return torch.rand(n, 3, device=device)


# ======================================================================
# Helmholtz equation in 2-D (linear)
# ======================================================================

class Helmholtz2D:
    """Helmholtz equation  u_xx + u_yy + k^2 u = f  on [0,1]^2.

    Exact solution: u = sin(k x) sin(k y).
    """

    def __init__(self):
        self.name = "Helmholtz 2D"
        self.dim = 2
        self.k = 10.0

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
        return -(self.k ** 2) * self.exact(x)

    def get_train_data(self, n_pde=10000, n_bc=2000):
        x_pde = torch.rand(n_pde, 2, device=device)
        f_pde = self.source(x_pde)
        n_side = n_bc // 4
        y_rand = torch.rand(n_side, 1, device=device)
        x_rand = torch.rand(n_side, 1, device=device)
        zeros = torch.zeros(n_side, 1, device=device)
        ones = torch.ones(n_side, 1, device=device)
        x_bc = torch.cat([
            torch.cat([zeros, y_rand], 1),
            torch.cat([ones, y_rand], 1),
            torch.cat([x_rand, zeros], 1),
            torch.cat([x_rand, ones], 1),
        ], 0)
        u_bc = self.exact(x_bc)
        return x_pde, [(x_bc, u_bc, "dirichlet")], f_pde

    def build(self, slv, x_pde, bcs, f_pde):
        basis = slv.basis
        cache = basis.cache(x_pde)
        A = basis.laplacian(x_pde, cache=cache) + (self.k ** 2) * basis.evaluate(x_pde, cache=cache)
        b = f_pde
        As, bs = [A], [b]
        for (pts, vals, _type) in bcs:
            h = basis.evaluate(pts)
            w = 100.0
            As.append(h * w)
            bs.append(vals * w)
        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=10000):
        return torch.rand(n, self.dim, device=device)


# ======================================================================
# Maxwell 2-D TM mode in a PEC cavity
# ======================================================================

class Maxwell2D_TM:
    """Maxwell equations, TM mode:  u_tt = c^2 (u_xx + u_yy).

    Solves for E_z(x, y, t) in a unit PEC cavity.
    """

    def __init__(self):
        self.name = "Maxwell 2D (TM)"
        self.dim = 3
        self.c = 1.0
        self.kx = np.pi
        self.ky = np.pi
        self.omega = self.c * np.sqrt(self.kx ** 2 + self.ky ** 2)

    def exact(self, x_in):
        xv, yv, tv = x_in[:, 0:1], x_in[:, 1:2], x_in[:, 2:3]
        return (torch.sin(self.kx * xv) * torch.sin(self.ky * yv)
                * torch.cos(self.omega * tv))

    def exact_grad(self, x_in):
        xv, yv, tv = x_in[:, 0:1], x_in[:, 1:2], x_in[:, 2:3]
        du_dx = (self.kx * torch.cos(self.kx * xv)
                 * torch.sin(self.ky * yv) * torch.cos(self.omega * tv))
        du_dy = (self.ky * torch.sin(self.kx * xv)
                 * torch.cos(self.ky * yv) * torch.cos(self.omega * tv))
        du_dt = (-self.omega * torch.sin(self.kx * xv)
                 * torch.sin(self.ky * yv) * torch.sin(self.omega * tv))
        return torch.cat([du_dx, du_dy, du_dt], dim=1)

    def get_train_data(self, n_pde=5000, n_bc=1000):
        x_pde = torch.rand(n_pde, 3, device=device)
        x_ic_space = torch.rand(n_bc, 2, device=device)
        x_ic = torch.cat([x_ic_space, torch.zeros(n_bc, 1, device=device)], 1)
        u_ic = self.exact(x_ic)
        ut_ic = torch.zeros_like(u_ic)
        n_wall = n_bc // 4
        r_t = torch.rand(n_wall, 1, device=device)
        r_s = torch.rand(n_wall, 1, device=device)
        zeros = torch.zeros(n_wall, 1, device=device)
        ones = torch.ones(n_wall, 1, device=device)
        x_bc = torch.cat([
            torch.cat([zeros, r_s, r_t], 1),
            torch.cat([ones, r_s, r_t], 1),
            torch.cat([r_s, zeros, r_t], 1),
            torch.cat([r_s, ones, r_t], 1),
        ], 0)
        u_bc = torch.zeros(len(x_bc), 1, device=device)
        return x_pde, [
            (x_ic, u_ic, "dirichlet"),
            (x_ic, ut_ic, "neumann_t"),
            (x_bc, u_bc, "dirichlet"),
        ], None

    def build(self, slv, x_pde, bcs, f_pde_ignored):
        basis = slv.basis
        cache = basis.cache(x_pde)
        hess_diag = basis.hessian_diag(x_pde, cache=cache)
        u_xx = hess_diag[:, 0, :]
        u_yy = hess_diag[:, 1, :]
        u_tt = hess_diag[:, 2, :]
        A = u_tt - (self.c ** 2) * (u_xx + u_yy)
        b = torch.zeros(len(x_pde), 1, device=device)
        As, bs = [A], [b]
        for (pts, vals, type_) in bcs:
            h = basis.evaluate(pts)
            dh = basis.gradient(pts)
            w = 100.0
            if type_ == "dirichlet":
                As.append(h * w)
            elif type_ == "neumann_t":
                As.append(dh[:, 2, :] * w)
            bs.append(vals * w)
        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=2000):
        return torch.rand(n, 3, device=device)
