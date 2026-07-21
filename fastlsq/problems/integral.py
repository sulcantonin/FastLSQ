# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Integral and integro-differential equations, in *solver mode*.

These run through the same `solve_linear` harness as the PDE problems, so an
integral-equation results table is produced alongside the PDE one rather than
living only in example scripts.

Every problem here has a **closed-form** solution -- from degenerate-kernel
theory for the Fredholm cases, by differentiating the equivalent ODE for the
Volterra ones -- so the reported error is against analysis, not against a
reference quadrature.

The Problem contract (duck-typed; there is no base class)
--------------------------------------------------------
Required by `fastlsq.api.solve_linear`:

    self.name                       -- str, for reporting
    self.dim                        -- int, input dimension
    get_train_data(n_pde, n_bc)     -- (x_pde, bcs, f_pde), bcs a list of
                                       (points, values) pairs
    build(solver, x_pde, bcs, f_pde) -> (A, b)
    exact(x)                        -- analytical solution, (M, 1)
    exact_grad(x)                   -- analytical gradient, (M, dim); required,
                                       `utils.evaluate_error` calls it
                                       unconditionally to report `grad_err`
    get_test_points(n)              -- (n, dim)

Optional, consulted when present:

    n_outputs, scale_multipliers, lam_bc

Note that an integral equation of the second kind needs **no boundary rows at
all**: the identity term makes it well posed on its own. The `bcs` list is
returned empty, and `n_bc` is accepted only to satisfy the shared signature.
"""

import numpy as np
import torch

from fastlsq.basis import IntegralOperator, Op
from fastlsq.kernels import (
    SeparableKernelOperator,
    degenerate_eigenvalues,
    fredholm_second_kind,
)
from fastlsq.utils import device


# ======================================================================
# Fredholm equations of the second kind (degenerate kernels)
# ======================================================================

class FredholmProductKernel:
    """u(x) − λ ∫₀¹ x y u(y) dy = f(x),  the classic rank-1 degenerate kernel.

    Degenerate-kernel theory gives the solution in closed form: writing
    c = ∫₀¹ y u(y) dy, the equation is u = f + λ c x, and substituting back,

        c = (∫₀¹ y f(y) dy) / (1 − λ/3),     u(x) = f(x) + λ c x

    singular only at λ = 3, the single characteristic value of the kernel.
    """

    def __init__(self, lam=0.5, freq=2.0):
        self.name = f"Fredholm xy (lam={lam})"
        self.dim = 1
        self.lam = lam
        self.freq = freq
        self.kernel = SeparableKernelOperator(
            [lambda x: x[:, 0]], [lambda y: y[:, 0]], 0.0, 1.0, d=1
        )
        # c = int_0^1 y f(y) dy for f = sin(freq*y), in closed form
        w = freq
        int_yf = (np.sin(w) - w * np.cos(w)) / w ** 2
        self.c = int_yf / (1.0 - lam / 3.0)

    def source(self, x):
        return torch.sin(self.freq * x[:, 0:1])

    def exact(self, x):
        return self.source(x) + self.lam * self.c * x[:, 0:1]

    def exact_grad(self, x):
        return self.freq * torch.cos(self.freq * x[:, 0:1]) + self.lam * self.c

    def get_train_data(self, n_pde=4000, n_bc=0):
        x_pde = torch.rand(n_pde, 1, device=device)
        return x_pde, [], self.source(x_pde)

    def build(self, slv, x_pde, bcs, f_pde):
        L = fredholm_second_kind(self.kernel, self.lam, d=1)
        return L.apply(slv.basis, x_pde), f_pde

    def get_test_points(self, n=4000):
        return torch.rand(n, 1, device=device)

    def characteristic_values(self, basis):
        """The λ at which this equation is singular (analytically, 3.0)."""
        return degenerate_eigenvalues(self.kernel, basis)


class FredholmRank2Kernel:
    """u(x) − λ ∫₀¹ [x y + sin x cos y] u(y) dy = f(x),  a rank-2 degenerate kernel.

    Same theory at rank 2: with c_m = ∫₀¹ h_m u, the equation reduces to the
    2 × 2 linear system (I − λS) c = b, where S_{mn} = ∫ h_m g_n and
    b_m = ∫ h_m f, and then u = f + λ Σ_m c_m g_m.  Solved here exactly (a 2 × 2
    solve, no quadrature) to give the reference.
    """

    def __init__(self, lam=0.4, freq=2.0):
        self.name = f"Fredholm rank-2 (lam={lam})"
        self.dim = 1
        self.lam = lam
        self.freq = freq
        self.g = [lambda x: x[:, 0], lambda x: torch.sin(x[:, 0])]
        self.h = [lambda y: y[:, 0], lambda y: torch.cos(y[:, 0])]
        self.kernel = SeparableKernelOperator(self.g, self.h, 0.0, 1.0, d=1)
        self._solve_coefficients()

    def _solve_coefficients(self):
        """Exact c from the 2x2 reduced system, via analytic 1-D integrals."""
        w = self.freq
        # S[m,n] = int_0^1 h_m(y) g_n(y) dy
        S = np.array([
            [1.0 / 3.0, np.sin(1.0) - np.cos(1.0)],                    # ∫y·y, ∫y·sin y
            [np.cos(1.0) + np.sin(1.0) - 1.0, 0.5 * np.sin(1.0) ** 2],  # ∫cos y·y, ∫cos y·sin y
        ])
        # b[m] = int_0^1 h_m(y) f(y) dy  with f = sin(w y)
        b0 = (np.sin(w) - w * np.cos(w)) / w ** 2
        if abs(w - 1.0) < 1e-12:
            b1 = 0.5 * np.sin(1.0) ** 2
        else:
            b1 = 0.5 * (
                (1 - np.cos((w + 1))) / (w + 1) + (1 - np.cos((w - 1))) / (w - 1)
            )
        b = np.array([b0, b1])
        self.c = np.linalg.solve(np.eye(2) - self.lam * S, b)

    def source(self, x):
        return torch.sin(self.freq * x[:, 0:1])

    def exact(self, x):
        u = self.source(x)
        for cm, gm in zip(self.c, self.g):
            u = u + self.lam * float(cm) * gm(x).reshape(-1, 1)
        return u

    def exact_grad(self, x):
        """u' = w cos(wx) + λ(c₀ + c₁ cos x), differentiating g = [x, sin x]."""
        t = x[:, 0:1]
        return (
            self.freq * torch.cos(self.freq * t)
            + self.lam * (float(self.c[0]) + float(self.c[1]) * torch.cos(t))
        )

    def get_train_data(self, n_pde=4000, n_bc=0):
        x_pde = torch.rand(n_pde, 1, device=device)
        return x_pde, [], self.source(x_pde)

    def build(self, slv, x_pde, bcs, f_pde):
        L = fredholm_second_kind(self.kernel, self.lam, d=1)
        return L.apply(slv.basis, x_pde), f_pde

    def get_test_points(self, n=4000):
        return torch.rand(n, 1, device=device)


# ======================================================================
# Volterra equations of the second kind
# ======================================================================

class VolterraSecondKind:
    """u(x) − λ ∫₀ˣ u(t) dt = f(x),  with the running (Volterra) integral.

    Differentiating gives u' − λu = f', u(0) = f(0), so

        u(x) = f(x) + λ ∫₀ˣ e^{λ(x−t)} f(t) dt

    and for f(x) = cos(ω x) that integral is elementary, giving the closed form
    used below.  Unlike the Fredholm cases this operator carries limits, so it
    is assembled with :class:`~fastlsq.basis.IntegralOperator`, not a separable
    kernel -- the two integral classes meet the same solver.
    """

    def __init__(self, lam=1.5, omega=3.0):
        self.name = f"Volterra 2nd kind (lam={lam})"
        self.dim = 1
        self.lam = lam
        self.omega = omega

    def source(self, x):
        return torch.cos(self.omega * x[:, 0:1])

    def exact(self, x):
        # u = f + lam * int_0^x e^{lam (x-t)} cos(w t) dt, integrated exactly:
        #   = [lam^2 e^{lam x} + w^2 sin(w x) * lam / w ... ] -- assembled below
        lam, w = self.lam, self.omega
        t = x[:, 0:1]
        denom = lam ** 2 + w ** 2
        # int_0^x e^{lam(x-t)} cos(wt) dt = [lam(cos wx) + w sin(wx) - lam e^{lam x}] / -denom
        integral = (lam * torch.exp(lam * t) - lam * torch.cos(w * t) + w * torch.sin(w * t)) / denom
        return torch.cos(w * t) + lam * integral

    def exact_grad(self, x):
        """From the equation itself: differentiating u − λ∫₀ˣu = f gives
        u' − λu = f', i.e. u' = λu − ω sin(ωx) -- no need to differentiate the
        closed form by hand."""
        t = x[:, 0:1]
        return self.lam * self.exact(x) - self.omega * torch.sin(self.omega * t)

    def get_train_data(self, n_pde=4000, n_bc=0):
        x_pde = torch.rand(n_pde, 1, device=device)
        return x_pde, [], self.source(x_pde)

    def build(self, slv, x_pde, bcs, f_pde):
        L = Op.identity(d=1) - self.lam * IntegralOperator.volterra(
            dim=0, lower=0.0, d=1
        )
        return L.apply(slv.basis, x_pde), f_pde

    def get_test_points(self, n=4000):
        return torch.rand(n, 1, device=device)


class IntegroDifferentialODE:
    """u'(x) + λ ∫₀ˣ u(t) dt = f(x),  u(0) = u₀ -- a first-order memory ODE.

    Differentiating gives u'' + λu = f', a harmonic oscillator, so with
    f(x) = cos(ω x) and the manufactured choice below the solution is a plain
    combination of sin/cos.  This is the shape of the RLC / memory-diffusion
    examples, expressed as a Problem so it reports in the same table.

    Here the initial condition **is** needed -- unlike a second-kind equation,
    a differential term leaves a genuine constant of integration -- so this
    problem returns one boundary row.
    """

    def __init__(self, lam=4.0, u0=0.3):
        self.name = f"Integro-diff ODE (lam={lam})"
        self.dim = 1
        self.lam = lam
        self.u0 = u0
        self.w = np.sqrt(lam)          # u'' + lam u = f' ; homogeneous freq
        self.lam_bc = 100.0

    def exact(self, x):
        """Manufactured: u = u0 cos(w x) + sin(w x), so u(0) = u0."""
        t = x[:, 0:1]
        return self.u0 * torch.cos(self.w * t) + torch.sin(self.w * t)

    def exact_grad(self, x):
        t = x[:, 0:1]
        return -self.u0 * self.w * torch.sin(self.w * t) + self.w * torch.cos(self.w * t)

    def source(self, x):
        """f = u' + lam * int_0^x u, computed analytically from `exact`."""
        t = x[:, 0:1]
        w, lam, u0 = self.w, self.lam, self.u0
        up = -u0 * w * torch.sin(w * t) + w * torch.cos(w * t)
        integral = (u0 / w) * torch.sin(w * t) + (1.0 - torch.cos(w * t)) / w
        return up + lam * integral

    def get_train_data(self, n_pde=4000, n_bc=1):
        x_pde = torch.rand(n_pde, 1, device=device)
        x_ic = torch.zeros(1, 1, device=device)
        return x_pde, [(x_ic, self.exact(x_ic))], self.source(x_pde)

    def build(self, slv, x_pde, bcs, f_pde):
        basis = slv.basis
        L = Op.partial(0, 1, d=1) + self.lam * IntegralOperator.volterra(
            dim=0, lower=0.0, d=1
        )
        As, bs = [L.apply(basis, x_pde)], [f_pde]
        for (pts, vals) in bcs:
            As.append(basis.evaluate(pts) * self.lam_bc)
            bs.append(vals * self.lam_bc)
        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=4000):
        return torch.rand(n, 1, device=device)
