# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Regression-mode problems: fit a known analytical solution directly.

These problems test the *function approximation* quality of the solver
(A = H, b = u_exact) rather than the PDE-constrained solve.  Problems
that also have a nonlinear solver-mode counterpart (Bratu, NL-Helmholtz)
inherit the exact solution from :mod:`fastlsq.problems.nonlinear` to
avoid code duplication.
"""

import torch
import numpy as np

from fastlsq.utils import device
from fastlsq.problems.nonlinear import Bratu2D, NLHelmholtz2D


# ======================================================================
# Generic regression build (shared by all classes below)
# ======================================================================

def _regression_build(slv, x_pde, bcs):
    """Standard regression build: A = H, b = u."""
    As, bs = [], []
    for (pts, vals, _) in bcs:
        h, _, _ = slv.get_features(pts)
        As.append(h)
        bs.append(vals)
    return torch.cat(As), torch.cat(bs)


# ======================================================================
# Burgers 1-D (traveling shock)
# ======================================================================

class Burgers1D_Regression:
    """Viscous Burgers:  u_t + u u_x = nu u_xx.

    Exact solution: traveling shock  u = 0.5 (1 - tanh(z)).
    """

    def __init__(self):
        self.name = "Burgers (Shock)"
        self.dim = 2
        self.nu = 0.02

    def exact(self, x_in):
        xv, tv = x_in[:, 0:1], x_in[:, 1:2]
        z = (xv - 0.5 * tv) / (4 * self.nu)
        return 0.5 * (1 - torch.tanh(z))

    def exact_grad(self, x_in):
        xv, tv = x_in[:, 0:1], x_in[:, 1:2]
        z = (xv - 0.5 * tv) / (4 * self.nu)
        sech2_z = 1 - torch.tanh(z) ** 2
        du_dz = -0.5 * sech2_z
        dz_dx = 1.0 / (4 * self.nu)
        dz_dt = -0.5 / (4 * self.nu)
        return torch.cat([du_dz * dz_dx, du_dz * dz_dt], dim=1)

    def get_train_data(self, n_samples=5000):
        x_pde = torch.rand(n_samples, 2, device=device)
        u_true = self.exact(x_pde)
        return x_pde, [(x_pde, u_true, "data_fit")]

    def build(self, slv, x_pde, bcs):
        return _regression_build(slv, x_pde, bcs)

    def get_test_points(self, n=10000):
        return torch.rand(n, self.dim, device=device)


# ======================================================================
# KdV (single soliton)
# ======================================================================

class KdV_Regression:
    """Korteweg-de Vries:  u_t + u u_x + u_xxx = 0.

    Exact solution: single soliton  u = 3c sech^2(...).
    """

    def __init__(self):
        self.name = "KdV (Soliton)"
        self.dim = 2
        self.c = 30.0
        self.x0 = -1.0

    def exact(self, x_in):
        xv, tv = x_in[:, 0:1], x_in[:, 1:2]
        sqrt_c = np.sqrt(self.c)
        z = (sqrt_c / 2.0) * (xv - self.c * tv - self.x0)
        sech_z = 1.0 / torch.cosh(z)
        return 3 * self.c * (sech_z ** 2)

    def exact_grad(self, x_in):
        xv, tv = x_in[:, 0:1], x_in[:, 1:2]
        sqrt_c = np.sqrt(self.c)
        z = (sqrt_c / 2.0) * (xv - self.c * tv - self.x0)
        sech_z = 1.0 / torch.cosh(z)
        u = 3 * self.c * (sech_z ** 2)
        tanh_z = torch.tanh(z)
        du_dz = -2.0 * u * tanh_z
        k = sqrt_c / 2.0
        return torch.cat([du_dz * k, du_dz * (-k * self.c)], dim=1)

    def get_train_data(self, n_samples=5000):
        x_space = torch.rand(n_samples, 1, device=device) * 4 - 2
        t_time = torch.rand(n_samples, 1, device=device) * 0.1
        x_pde = torch.cat([x_space, t_time], dim=1)
        u_true = self.exact(x_pde)
        return x_pde, [(x_pde, u_true, "data_fit")]

    def build(self, slv, x_pde, bcs):
        return _regression_build(slv, x_pde, bcs)

    def get_test_points(self, n=10000):
        x_space = torch.rand(n, 1, device=device) * 4 - 2
        t_time = torch.rand(n, 1, device=device) * 0.1
        return torch.cat([x_space, t_time], dim=1)


# ======================================================================
# Reaction-Diffusion (Fisher's equation, traveling wavefront)
# ======================================================================

class ReactionDiffusion_Regression:
    """Fisher's equation:  u_t - u_xx - u(1-u) = 0.

    Exact solution: traveling wavefront  u = (1 + exp(z))^{-2}.
    """

    def __init__(self):
        self.name = "Reaction-Diffusion"
        self.dim = 2
        self.rho = 1.0

    def exact(self, x_in):
        xv, tv = x_in[:, 0:1], x_in[:, 1:2]
        alpha = np.sqrt(self.rho / 6.0)
        c = (5.0 / 6.0) * np.sqrt(self.rho * 6.0)
        z = alpha * (xv - c * tv)
        return (1.0 + torch.exp(z)).pow(-2)

    def exact_grad(self, x_in):
        xv, tv = x_in[:, 0:1], x_in[:, 1:2]
        alpha = np.sqrt(self.rho / 6.0)
        c = (5.0 / 6.0) * np.sqrt(self.rho * 6.0)
        z = alpha * (xv - c * tv)
        E = torch.exp(z)
        du_dz = -2.0 * ((1.0 + E).pow(-3)) * E
        return torch.cat([du_dz * alpha, du_dz * (-alpha * c)], dim=1)

    def get_train_data(self, n_samples=5000):
        x_space = torch.rand(n_samples, 1, device=device) * 20 - 10
        t_time = torch.rand(n_samples, 1, device=device)
        x_pde = torch.cat([x_space, t_time], dim=1)
        u_true = self.exact(x_pde)
        return x_pde, [(x_pde, u_true, "data_fit")]

    def build(self, slv, x_pde, bcs):
        return _regression_build(slv, x_pde, bcs)

    def get_test_points(self, n=10000):
        x_space = torch.rand(n, 1, device=device) * 20 - 10
        t_time = torch.rand(n, 1, device=device)
        return torch.cat([x_space, t_time], dim=1)


# ======================================================================
# Sine-Gordon (breather)
# ======================================================================

class SineGordon_Regression:
    """Sine-Gordon equation breather solution."""

    def __init__(self):
        self.name = "Sine-Gordon"
        self.dim = 2
        self.omega = 0.5

    def exact(self, x_in):
        xv, tv = x_in[:, 0:1], x_in[:, 1:2]
        w = self.omega
        k = np.sqrt(1 - w ** 2)
        num = k * torch.sin(w * tv)
        denom = w * torch.cosh(k * xv)
        return 4.0 * torch.atan(num / denom)

    def exact_grad(self, x_in):
        xv, tv = x_in[:, 0:1], x_in[:, 1:2]
        w = self.omega
        k = np.sqrt(1 - w ** 2)
        sin_wt = torch.sin(w * tv)
        cos_wt = torch.cos(w * tv)
        cosh_kx = torch.cosh(k * xv)
        sinh_kx = torch.sinh(k * xv)
        num = k * sin_wt
        denom = w * cosh_kx
        A = num / denom
        du_dA = 4.0 / (1.0 + A ** 2)
        dA_dx = num * (-1.0 / (denom ** 2)) * (w * k * sinh_kx)
        dA_dt = (1.0 / denom) * (k * w * cos_wt)
        return torch.cat([du_dA * dA_dx, du_dA * dA_dt], dim=1)

    def get_train_data(self, n_samples=5000):
        x_space = torch.rand(n_samples, 1, device=device) * 20 - 10
        t_time = torch.rand(n_samples, 1, device=device) * 20
        x_pde = torch.cat([x_space, t_time], dim=1)
        u_true = self.exact(x_pde)
        return x_pde, [(x_pde, u_true, "data_fit")]

    def build(self, slv, x_pde, bcs):
        return _regression_build(slv, x_pde, bcs)

    def get_test_points(self, n=2000):
        x_space = torch.rand(n, 1, device=device) * 20 - 10
        t_time = torch.rand(n, 1, device=device) * 20
        return torch.cat([x_space, t_time], dim=1)


# ======================================================================
# Klein-Gordon (nonlinear, smooth oscillatory solution)
# ======================================================================

class KleinGordon_Regression:
    """Nonlinear Klein-Gordon:  u_tt - u_xx + u^2 = f.

    Exact solution: u = sin(pi x) cos(2 pi t).
    """

    def __init__(self):
        self.name = "Klein-Gordon (NL)"
        self.dim = 2

    def exact(self, x_in):
        xv, tv = x_in[:, 0:1], x_in[:, 1:2]
        return torch.sin(np.pi * xv) * torch.cos(2 * np.pi * tv)

    def exact_grad(self, x_in):
        xv, tv = x_in[:, 0:1], x_in[:, 1:2]
        du_dx = np.pi * torch.cos(np.pi * xv) * torch.cos(2 * np.pi * tv)
        du_dt = -2 * np.pi * torch.sin(np.pi * xv) * torch.sin(2 * np.pi * tv)
        return torch.cat([du_dx, du_dt], dim=1)

    def get_train_data(self, n_samples=5000):
        x_space = torch.rand(n_samples, 1, device=device) * 2 - 1
        t_time = torch.rand(n_samples, 1, device=device)
        x_pde = torch.cat([x_space, t_time], dim=1)
        u_true = self.exact(x_pde)
        return x_pde, [(x_pde, u_true, "data_fit")]

    def build(self, slv, x_pde, bcs):
        return _regression_build(slv, x_pde, bcs)

    def get_test_points(self, n=2000):
        x_space = torch.rand(n, 1, device=device) * 2 - 1
        t_time = torch.rand(n, 1, device=device)
        return torch.cat([x_space, t_time], dim=1)


# ======================================================================
# Navier-Stokes 2-D (Kovasznay flow, steady state)
# ======================================================================

class NavierStokes2D_Kovasznay:
    """Kovasznay flow: analytic steady-state solution to Navier-Stokes.

    Domain: [-0.5, 1.0] x [-0.5, 1.5].
    """

    def __init__(self):
        self.name = "NS (Kovasznay)"
        self.dim = 2
        self.Re = 20.0
        self.lam = 0.5 * self.Re - np.sqrt(0.25 * self.Re ** 2 + 4 * np.pi ** 2)

    def exact(self, x_in):
        xv, yv = x_in[:, 0:1], x_in[:, 1:2]
        return 1.0 - torch.exp(self.lam * xv) * torch.cos(2 * np.pi * yv)

    def exact_grad(self, x_in):
        xv, yv = x_in[:, 0:1], x_in[:, 1:2]
        exp_term = torch.exp(self.lam * xv)
        cos_term = torch.cos(2 * np.pi * yv)
        sin_term = torch.sin(2 * np.pi * yv)
        du_dx = -self.lam * exp_term * cos_term
        du_dy = 2 * np.pi * exp_term * sin_term
        return torch.cat([du_dx, du_dy], dim=1)

    def get_train_data(self, n_samples=5000):
        x_space = torch.rand(n_samples, 1, device=device) * 1.5 - 0.5
        y_space = torch.rand(n_samples, 1, device=device) * 2.0 - 0.5
        x_pde = torch.cat([x_space, y_space], dim=1)
        u_true = self.exact(x_pde)
        return x_pde, [(x_pde, u_true, "data_fit")]

    def build(self, slv, x_pde, bcs):
        return _regression_build(slv, x_pde, bcs)

    def get_test_points(self, n=2000):
        x_space = torch.rand(n, 1, device=device) * 1.5 - 0.5
        y_space = torch.rand(n, 1, device=device) * 2.0 - 0.5
        return torch.cat([x_space, y_space], dim=1)


# ======================================================================
# Gray-Scott (synthetic traveling pulse)
# ======================================================================

class GrayScott_Pulse:
    """Gray-Scott-like synthetic pulse: u = exp(-((x - ct)^2) / sigma)."""

    def __init__(self):
        self.name = "Gray-Scott (Pulse)"
        self.dim = 2
        self.c = 0.5
        self.sigma = 0.02

    def exact(self, x_in):
        xv, tv = x_in[:, 0:1], x_in[:, 1:2]
        arg = -((xv - self.c * tv) ** 2) / self.sigma
        return torch.exp(arg)

    def exact_grad(self, x_in):
        xv, tv = x_in[:, 0:1], x_in[:, 1:2]
        arg = -((xv - self.c * tv) ** 2) / self.sigma
        u = torch.exp(arg)
        darg_dx = -2 * (xv - self.c * tv) / self.sigma
        darg_dt = 2 * self.c * (xv - self.c * tv) / self.sigma
        return torch.cat([u * darg_dx, u * darg_dt], dim=1)

    def get_train_data(self, n_samples=5000):
        x_pde = torch.rand(n_samples, 2, device=device)
        u_true = self.exact(x_pde)
        return x_pde, [(x_pde, u_true, "data_fit")]

    def build(self, slv, x_pde, bcs):
        return _regression_build(slv, x_pde, bcs)

    def get_test_points(self, n=2000):
        return torch.rand(n, 2, device=device)


# ======================================================================
# Bratu 2-D (regression mode) -- inherits exact solution from nonlinear
# ======================================================================

class Bratu2D_Regression(Bratu2D):
    """Bratu 2-D in regression mode (fits exact solution directly).

    Shares the exact solution with :class:`~fastlsq.problems.nonlinear.Bratu2D`.
    """

    def __init__(self):
        super().__init__(lam=1.0)
        self.name = "Bratu 2D (Reg)"

    def get_train_data(self, n_samples=5000):
        x_pde = torch.rand(n_samples, 2, device=device)
        u_true = self.exact(x_pde)
        return x_pde, [(x_pde, u_true, "data_fit")]

    def build(self, slv, x_pde, bcs):
        return _regression_build(slv, x_pde, bcs)

    def get_test_points(self, n=5000):
        return torch.rand(n, 2, device=device)


# ======================================================================
# Nonlinear Helmholtz 2-D (regression mode) -- inherits from nonlinear
# ======================================================================

class NLHelmholtz2D_Regression(NLHelmholtz2D):
    """NL-Helmholtz 2-D in regression mode (fits exact solution directly).

    Shares the exact solution with
    :class:`~fastlsq.problems.nonlinear.NLHelmholtz2D`.
    """

    def __init__(self):
        super().__init__(k=3.0, alpha=0.5)
        self.name = "NL-Helmholtz (Reg)"

    def get_train_data(self, n_samples=5000):
        x_pde = torch.rand(n_samples, 2, device=device)
        u_true = self.exact(x_pde)
        return x_pde, [(x_pde, u_true, "data_fit")]

    def build(self, slv, x_pde, bcs):
        return _regression_build(slv, x_pde, bcs)

    def get_test_points(self, n=5000):
        return torch.rand(n, 2, device=device)
