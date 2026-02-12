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
        _, _, ddH = slv.get_features(x_pde)
        A = -torch.sum(ddH, dim=1)
        b = f_pde
        As, bs = [A], [b]
        for (pts, vals) in bcs:
            h, _, _ = slv.get_features(pts)
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
        _, dH, ddH = slv.get_features(x_pde)
        lap = torch.sum(ddH[:, 0:5, :], dim=1)
        u_t = dH[:, 5, :]
        A = u_t - self.k * lap
        b = f_pde
        As, bs = [A], [b]
        for (pts, vals, type_) in bcs:
            h, dh, _ = slv.get_features(pts)
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
        _, _, ddH = slv.get_features(x_pde)
        A = ddH[:, 1, :] - self.c2 * ddH[:, 0, :]
        b = torch.zeros(len(x_pde), 1, device=device)
        As, bs = [A], [b]
        for (pts, vals, type_) in bcs:
            h, dh, _ = slv.get_features(pts)
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
        _, dH, ddH = slv.get_features(x_pde)
        u_xx = ddH[:, 0, :]
        u_yy = ddH[:, 1, :]
        u_tt_norm = ddH[:, 2, :]
        A = u_tt_norm - (self.t_max ** 2) * (u_xx + self.a2 * u_yy)
        b = torch.zeros(len(x_pde), 1, device=device)
        As, bs = [A], [b]
        w_bc = 1000.0
        for (pts, vals, type_) in bcs:
            h, dh, _ = slv.get_features(pts)
            if type_ == "dirichlet":
                As.append(h * w_bc)
            elif type_ == "neumann_t":
                As.append(dh[:, 2, :] * w_bc)
            bs.append(vals * w_bc)
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
        _, _, ddH = slv.get_features(x_pde)
        lap = torch.sum(ddH, dim=1)
        H, _, _ = slv.get_features(x_pde)
        A = lap + (self.k ** 2) * H
        b = f_pde
        As, bs = [A], [b]
        for (pts, vals, _type) in bcs:
            h, _, _ = slv.get_features(pts)
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
        _, dH, ddH = slv.get_features(x_pde)
        u_xx = ddH[:, 0, :]
        u_yy = ddH[:, 1, :]
        u_tt = ddH[:, 2, :]
        A = u_tt - (self.c ** 2) * (u_xx + u_yy)
        b = torch.zeros(len(x_pde), 1, device=device)
        As, bs = [A], [b]
        for (pts, vals, type_) in bcs:
            h, dh, _ = slv.get_features(pts)
            w = 100.0
            if type_ == "dirichlet":
                As.append(h * w)
            elif type_ == "neumann_t":
                As.append(dh[:, 2, :] * w)
            bs.append(vals * w)
        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=2000):
        return torch.rand(n, 3, device=device)
