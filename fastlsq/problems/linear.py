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
from fastlsq.block import block_concat


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
    """Wave 2-D multi-scale (anisotropic, normalised time).

    Anisotropic wave  u_tt = u_xx + a2 u_yy  on [0,1]^2 x [0, t_max], with time
    normalised to tau = t / t_max in [0,1].  ``build`` therefore carries the
    spatial term's t_max^2 factor (d^2/dt^2 = t_max^-2 d^2/dtau^2), so the
    discretised operator  u_tautau - t_max^2 (u_xx + a2 u_yy)  is satisfied
    exactly by ``exact`` (the (1,1) standing mode, omega = pi sqrt(1+a2)).

    Resolvability constraint on ``t_max``.  In normalised time the solution
    oscillates at  Omega = omega * t_max, i.e. ~ sqrt(1+a2) * t_max / 2 temporal
    cycles over tau in [0,1].  The PDE's second time-derivative amplifies the
    random-feature *representation* error by Omega^2, so the one-shot
    least-squares collocation only resolves a handful of cycles before that
    amplified error swamps the solution -- the original ``t_max = 100`` (~87
    cycles) did not solve in *any* configuration (rel-err 1.0, the [0.2.4] known
    issue), even at 8000 features with near-hard boundary constraints, because
    the best representable solution itself carries a huge PDE residual.
    ``t_max = 4`` keeps it at ~3.5 cycles (solves to ~1e-3 at 900 features); the
    anisotropic ``scale_multipliers`` place the temporal feature bandwidth at
    ~Omega while the spatial bandwidth stays ~pi.
    """

    def __init__(self):
        self.name = "Wave 2D-MS"
        self.dim = 3
        self.a2 = 2.0
        self.t_max = 4.0          # ~3.5 temporal cycles -- see class docstring
        # Anisotropic feature bandwidth: temporal ~ Omega = pi*sqrt(1+a2)*t_max
        # ~= 21.8, matched at scale ~3 (multiplier 7); spatial bandwidth ~ pi.
        self.scale_multipliers = [1.0, 1.0, 7.0]

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
        self.n_outputs = 2  # (u_x, u_y) -- block-stacked vector solve
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

    def exact_grad(self, x_in):
        """Jacobian of (u_x, u_y). Returns (M, d, k) with J[:, j, c] = du_c/dx_j.

        Time is normalised (t_phys = t * t_max), so the t-derivatives pick up a
        t_max chain-rule factor -- matching ``exact_ut`` and ``Wave2D_MS`` and the
        normalised inputs ``predict_with_grad`` differentiates against.
        """
        xv, yv, tv = x_in[:, 0:1], x_in[:, 1:2], x_in[:, 2:3] * self.t_max
        kx, ky = self.kx, self.ky
        cx, sx = torch.cos(kx * xv), torch.sin(kx * xv)
        cy, sy = torch.cos(ky * yv), torch.sin(ky * yv)
        ct, st = torch.cos(self.omega_p * tv), torch.sin(self.omega_p * tv)
        dt = -self.omega_p * self.t_max * st  # d/dt_norm of cos(omega_p * t_phys)

        # u_x = kx cos(kx x) sin(ky y) cos(omega_p t)
        ux_x = kx * (-kx * sx) * sy * ct
        ux_y = kx * cx * (ky * cy) * ct
        ux_t = kx * cx * sy * dt
        # u_y = ky sin(kx x) cos(ky y) cos(omega_p t)
        uy_x = ky * (kx * cx) * cy * ct
        uy_y = ky * sx * (-ky * sy) * ct
        uy_t = ky * sx * cy * dt

        grad_ux = torch.cat([ux_x, ux_y, ux_t], dim=1)  # (M, 3)
        grad_uy = torch.cat([uy_x, uy_y, uy_t], dim=1)  # (M, 3)
        return torch.stack([grad_ux, grad_uy], dim=-1)  # (M, 3, 2)

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
        """Block-stacked system for the coupled (u_x, u_y) solve.

        Two column blocks (u_x, u_y coefficients); each equation / BC adds a
        block row. ``block_concat`` assembles A in R^{Mk x Nk}, b in R^{Mk x 1}
        (k = n_outputs = 2) so ``unpack_beta`` recovers a (N, 2) beta.
        """
        basis = slv.basis
        cache = basis.cache(x_pde)

        # Derivatives for (x, y, t) with t as dim 2
        u_xx = basis.derivative(x_pde, (2, 0, 0), cache=cache)
        u_yy = basis.derivative(x_pde, (0, 2, 0), cache=cache)
        u_tt = basis.derivative(x_pde, (0, 0, 2), cache=cache)
        u_xy = basis.derivative(x_pde, (1, 1, 0), cache=cache)

        # t is normalised to [0,1]; physical d²/dt² = (1/t_max)² d²/dτ², so the
        # spatial + cross terms carry a t_max² factor (consistent with Wave2D_MS).
        t_scale = self.t_max ** 2
        cross = t_scale * self.c_cross

        # PDE1: u_x_ττ = t_max²·(c_p² u_x_xx + c_s² u_x_yy + (c_p²-c_s²) u_y_xy)
        A1_x = u_tt - t_scale * (self.c_p2 * u_xx + self.c_s2 * u_yy)
        A1_y = -cross * u_xy
        # PDE2: u_y_ττ = t_max²·(c_p² u_y_yy + c_s² u_y_xx + (c_p²-c_s²) u_x_xy)
        A2_x = -cross * u_xy
        A2_y = u_tt - t_scale * (self.c_p2 * u_yy + self.c_s2 * u_xx)

        z_pde = torch.zeros(len(x_pde), 1, device=device)
        rows = [[A1_x, A1_y], [A2_x, A2_y]]   # block rows: [u_x col, u_y col]
        rhs = [[z_pde], [z_pde]]              # matching RHS column blocks

        w_bc = 1000.0
        for (pts, vals, type_) in bcs:
            if type_ == "dirichlet":
                op = basis.evaluate(pts) * w_bc
            elif type_ == "neumann_t":
                op = basis.gradient(pts)[:, 2, :] * w_bc
            else:
                continue
            # vals: (n_pts, 2). One block row per component:
            #   u_x -> [op, None],  u_y -> [None, op]
            rows += [[op, None], [None, op]]
            rhs += [[vals[:, 0:1] * w_bc], [vals[:, 1:2] * w_bc]]

        return block_concat(rows), block_concat(rhs)

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
