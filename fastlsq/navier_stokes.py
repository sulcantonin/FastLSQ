# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Steady incompressible Navier-Stokes in closed form.

    (u . grad) u + grad p - nu lap u = 0,        div u = 0

The velocity is written as a free stream plus a disturbance,

    u = u_inf + u',     u' = sum_j c_j t_j sin(W_j . x + b_j)  [+ singular columns]

with the sinusoids in divergence-free form (:class:`~fastlsq.DivergenceFreeBasis`),
so continuity holds exactly and never enters the solve.  The pressure has its own
basis.  For flow past a body, :class:`~fastlsq.singular.StokesSingularities`
adds closed-form Stokeslets and doublets seated inside it, which carry the
near-wall structure a band-limited basis resolves slowly.

Every derivative is closed form, so a Newton step is one linear least-squares
solve.  Linearising about the previous iterate ``U``,

    (U . grad) u' + (u' . grad) U + grad p - nu lap u' = ((U - u_inf) . grad) U,

and the first step, with no iterate yet, is Stokes flow.  :func:`solve_navier_stokes`
runs that loop with the rows accumulated chunk by chunk into normal equations, so
memory grows with the number of unknowns squared, not with the number of rows.

>>> vel = DivergenceFreeBasis.random(2, 800, sigma=[2.0, 6.0])
>>> pre = SinusoidalBasis.random(2, 400, sigma=4.0)
>>> flow = IncompressibleFlow(vel, pre, nu=1 / 40)
>>> res = solve_navier_stokes(flow, x_int, [Dirichlet(x_b, u_b), PressurePoint(x0, 0.0)])
>>> u, grad_u, p = flow.evaluate(x, res.theta)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import torch

from fastlsq.linalg import NormalEquations

TensorLike = Union[torch.Tensor, Sequence[float], float]


class IncompressibleFlow:
    """Velocity, pressure and optional singular columns as one linear model.

    Parameters
    ----------
    velocity : DivergenceFreeBasis
        The disturbance velocity ``u'``.
    pressure : basis
        Anything with ``evaluate`` (M, N) and ``gradient`` (M, d, N):
        :class:`~fastlsq.SinusoidalBasis`, or :class:`~fastlsq.MirrorBasis` with
        ``parity=+1`` to match a mirror-symmetric velocity.
    nu : float
        Kinematic viscosity (``1/Re`` in units of the reference length and speed).
    u_inf : sequence of float, optional
        Free stream, lifted out of the unknown (a constant is not in the span of a
        sinusoidal basis).  Default zero.
    singular : StokesSingularities, optional
        Closed-form columns for flow past a body (3-D only).

    The unknowns are ``theta = [velocity | pressure | singular]``.
    """

    def __init__(self, velocity, pressure, nu: float, u_inf: Optional[TensorLike] = None, singular=None):
        self.velocity = velocity
        self.pressure = pressure
        self.singular = singular
        self.nu = float(nu)
        d = velocity.input_dim
        if singular is not None and d != 3:
            raise ValueError("singular columns are 3-D")
        W = velocity.basis.W
        self.u_inf = (torch.zeros(d, device=W.device, dtype=W.dtype) if u_inf is None
                      else torch.as_tensor(u_inf, device=W.device, dtype=W.dtype).reshape(d))
        self.n_vel = velocity.n_unknowns
        self.n_pre = pressure.n_features
        self.n_sing = 0 if singular is None else singular.n_unknowns

    @property
    def dim(self) -> int:
        return self.velocity.input_dim

    @property
    def n_unknowns(self) -> int:
        return self.n_vel + self.n_pre + self.n_sing

    def split(self, theta: torch.Tensor):
        """``theta`` -> (velocity, pressure, singular) coefficient views."""
        a, b = self.n_vel, self.n_vel + self.n_pre
        return theta[:a], theta[a:b], theta[b:]

    # ------------------------------------------------------------------
    # Design blocks
    # ------------------------------------------------------------------

    def blocks(self, x: torch.Tensor, laplacian: bool = True) -> dict:
        """Design blocks over all unknowns at ``x`` (M, d).

        ``val`` (M, d, n) and ``grad`` (M, d, d, n) of the disturbance velocity,
        ``p`` (M, n), ``pgrad`` (M, d, n) and, with ``laplacian``, ``stokes``
        (M, d, n) = ``-nu lap u' + grad p``: the linear part of the momentum
        operator.
        """
        M, d = x.shape
        kw = dict(device=x.device, dtype=x.dtype)
        val, grad, lap = self.velocity.blocks(x, laplacian=laplacian)
        c = self.pressure.cache(x) if hasattr(self.pressure, "cache") else None
        pv = self.pressure.evaluate(x, cache=c)
        pg = self.pressure.gradient(x, cache=c)
        Zp = torch.zeros(M, d, self.n_pre, **kw)
        Zv = torch.zeros(M, self.n_vel, **kw)
        out = dict(
            val=torch.cat([val, Zp], -1),
            grad=torch.cat([grad, Zp[:, :, None].expand(M, d, d, self.n_pre)], -1),
            p=torch.cat([Zv, pv], -1),
            pgrad=torch.cat([Zv[:, None].expand(M, d, self.n_vel), pg], -1),
            stokes=torch.cat([-self.nu * lap, pg], -1) if laplacian else None,
        )
        if self.singular is not None:
            s = self.singular.blocks(x)
            for key in ("val", "grad", "p", "pgrad"):
                out[key] = torch.cat([out[key], s[key]], -1)
            if laplacian:
                out["stokes"] = torch.cat([out["stokes"], s["stokes"]], -1)
        return out

    def evaluate(self, x: torch.Tensor, theta: torch.Tensor, chunk: int = 4000):
        """``u`` (M, d) with the free stream, ``grad u`` (M, d, d) and ``p`` (M,)."""
        us, gs, ps = [], [], []
        for i in range(0, x.shape[0], chunk):
            B = self.blocks(x[i:i + chunk], laplacian=False)
            us.append(B["val"] @ theta + self.u_inf)
            gs.append(B["grad"] @ theta)
            ps.append(B["p"] @ theta)
        return torch.cat(us), torch.cat(gs), torch.cat(ps)

    # ------------------------------------------------------------------
    # Rows
    # ------------------------------------------------------------------

    def momentum_rows(self, x: torch.Tensor, U: Optional[torch.Tensor] = None,
                      GU: Optional[torch.Tensor] = None):
        """Momentum rows, ``d * M`` of them, component-major.

        Without ``U`` these are Stokes rows.  With the previous iterate's velocity
        ``U`` (M, d, free stream included) and gradient ``GU`` (M, d, d) they are
        the Newton linearisation; at ``theta`` equal to that iterate,
        ``A @ theta - b`` is the full nonlinear residual.
        """
        B = self.blocks(x, laplacian=True)
        A = B["stokes"]
        b = torch.zeros(x.shape[0], self.dim, device=x.device, dtype=x.dtype)
        if U is not None:
            A = (A + torch.einsum("nm,nkmp->nkp", U, B["grad"])
                 + torch.einsum("nkm,nmp->nkp", GU, B["val"]))
            b = torch.einsum("nm,nkm->nk", U - self.u_inf, GU)
        return A.transpose(0, 1).reshape(-1, self.n_unknowns), b.T.reshape(-1)

    def dirichlet_rows(self, x: torch.Tensor, value: Optional[TensorLike] = None):
        """``u = value`` (M, d); default zero, i.e. no-slip."""
        B = self.blocks(x, laplacian=False)
        g = torch.zeros(x.shape[0], self.dim, device=x.device, dtype=x.dtype)
        if value is not None:
            g = g + torch.as_tensor(value, device=x.device, dtype=x.dtype)
        return B["val"].transpose(0, 1).reshape(-1, self.n_unknowns), (g - self.u_inf).T.reshape(-1)

    def slip_rows(self, x: torch.Tensor, normal: Union[int, TensorLike]):
        """Free slip: ``u . n = 0`` and no tangential shear, ``(I - n n^T) d_n u = 0``.

        ``normal`` is an axis index (a plane wall or symmetry plane) or unit normals
        (M, d) or (d,).  ``d + 1`` rows per point; one shear row is redundant.
        """
        n = self._normal(x, normal)
        B = self.blocks(x, laplacian=False)
        un = torch.einsum("nk,nkp->np", n, B["val"])
        dn = torch.einsum("nm,nkmp->nkp", n, B["grad"])
        shear = dn - n[:, :, None] * torch.einsum("nk,nkp->np", n, dn)[:, None]
        A = torch.cat([un, shear.transpose(0, 1).reshape(-1, self.n_unknowns)])
        b = torch.cat([-(n @ self.u_inf), torch.zeros(self.dim * x.shape[0], device=x.device, dtype=x.dtype)])
        return A, b

    def traction_rows(self, x: torch.Tensor, normal: Union[int, TensorLike], value: Optional[TensorLike] = None):
        """Prescribed traction ``sigma n = value`` (default zero: a free outlet),
        ``sigma = -p I + nu (grad u + grad u^T)``."""
        n = self._normal(x, normal)
        B = self.blocks(x, laplacian=False)
        A = self._traction(B, n)
        t = torch.zeros(x.shape[0], self.dim, device=x.device, dtype=x.dtype)
        if value is not None:
            t = t + torch.as_tensor(value, device=x.device, dtype=x.dtype)
        return A.transpose(0, 1).reshape(-1, self.n_unknowns), t.T.reshape(-1)

    def pressure_rows(self, x: torch.Tensor, value: TensorLike = 0.0):
        """``p = value``: fixes the pressure level when no boundary does."""
        B = self.blocks(x, laplacian=False)
        v = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype) + torch.as_tensor(value, device=x.device,
                                                                                      dtype=x.dtype)
        return B["p"], v

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def residual(self, x: torch.Tensor, theta: torch.Tensor, chunk: int = 2000) -> dict:
        """Pointwise momentum residual and the size of the terms it balances.

        Returns ``res`` (M, d) = ``(u . grad) u + grad p - nu lap u`` and
        ``convection``, ``viscous`` (= ``nu lap u``) and ``pressure`` (= ``grad p``),
        each (M, d), plus ``relative`` = ``|res| / (|conv| + |visc| + |grad p|)``,
        the scale-free measure (M,).
        """
        parts = {k: [] for k in ("res", "convection", "viscous", "pressure")}
        for i in range(0, x.shape[0], chunk):
            B = self.blocks(x[i:i + chunk], laplacian=True)
            u = B["val"] @ theta + self.u_inf
            gu = B["grad"] @ theta
            gp = B["pgrad"] @ theta
            st = B["stokes"] @ theta
            conv = torch.einsum("nm,nkm->nk", u, gu)
            parts["res"].append(conv + st)
            parts["convection"].append(conv)
            parts["viscous"].append(gp - st)
            parts["pressure"].append(gp)
        out = {k: torch.cat(v) for k, v in parts.items()}
        scale = out["convection"].norm(dim=1) + out["viscous"].norm(dim=1) + out["pressure"].norm(dim=1)
        out["relative"] = out["res"].norm(dim=1) / scale.clamp_min(1e-300)
        return out

    def traction(self, x: torch.Tensor, normal: Union[int, TensorLike], theta: torch.Tensor) -> torch.Tensor:
        """``sigma n`` (M, d) at ``x``.  With ``n`` the body's outward normal this is
        the force per unit area the fluid exerts on the body."""
        n = self._normal(x, normal)
        return self._traction(self.blocks(x, laplacian=False), n) @ theta

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def plain(self, theta: torch.Tensor) -> dict:
        """Everything a renderer needs, without fastlsq: velocity
        ``u_inf + sin(x @ W.T + b) @ C`` and pressure ``sin(x @ Wp.T + bp) @ Cp``,
        plus the singular strengths (see :meth:`StokesSingularities.plain`)."""
        cv, cp, cs = self.split(theta)
        W, b, C = self.velocity.plain(cv)
        pb = getattr(self.pressure, "basis", self.pressure)
        Wp, bp, sign = (self.pressure.plain() if hasattr(self.pressure, "plain")
                        else (pb.W, pb.b, torch.ones(pb.n_features, device=pb.W.device, dtype=pb.W.dtype)))
        reps = Wp.shape[1] // cp.shape[0]
        Cp = cp.repeat(reps) * sign * pb._inv_norm
        out = dict(W=W, b=b, C=C, Wp=Wp.T, bp=bp.reshape(-1), Cp=Cp, u_inf=self.u_inf, nu=self.nu)
        if self.singular is not None:
            out["singular"] = self.singular.plain(cs)
        return out

    # ------------------------------------------------------------------

    def _normal(self, x, normal):
        if isinstance(normal, int):
            n = torch.zeros(x.shape[0], self.dim, device=x.device, dtype=x.dtype)
            n[:, normal] = 1.0
            return n
        return torch.as_tensor(normal, device=x.device, dtype=x.dtype).expand(x.shape[0], self.dim)

    def _traction(self, B, n):
        sym = torch.einsum("nm,nkmp->nkp", n, B["grad"]) + torch.einsum("nk,nkmp->nmp", n, B["grad"])
        return -n[:, :, None] * B["p"][:, None] + self.nu * sym


# ----------------------------------------------------------------------
# Boundary conditions and the Newton driver
# ----------------------------------------------------------------------


def _per_point(v, sl, ndim):
    """Slice ``v`` if it holds one entry per point (``ndim`` dims), else pass it on."""
    return v[sl] if torch.is_tensor(v) and v.ndim == ndim else v


@dataclass
class Dirichlet:
    """``u = value`` at ``x``: (M, d) per point or (d,) constant; ``None`` is no-slip."""
    x: torch.Tensor
    value: Optional[TensorLike] = None
    weight: float = 1.0

    def rows(self, flow: IncompressibleFlow, sl: slice):
        return flow.dirichlet_rows(self.x[sl], _per_point(self.value, sl, 2))


@dataclass
class Slip:
    """Free slip (wall or symmetry plane): ``normal`` is an axis, (d,) or (M, d)."""
    x: torch.Tensor
    normal: Union[int, TensorLike]
    weight: float = 1.0

    def rows(self, flow: IncompressibleFlow, sl: slice):
        return flow.slip_rows(self.x[sl], _per_point(self.normal, sl, 2))


@dataclass
class Traction:
    """``sigma n = value``; the default zero is a traction-free outlet."""
    x: torch.Tensor
    normal: Union[int, TensorLike]
    value: Optional[TensorLike] = None
    weight: float = 1.0

    def rows(self, flow: IncompressibleFlow, sl: slice):
        return flow.traction_rows(self.x[sl], _per_point(self.normal, sl, 2), _per_point(self.value, sl, 2))


@dataclass
class PressurePoint:
    """``p = value`` at ``x`` ((M,) or scalar), to fix the pressure level."""
    x: torch.Tensor
    value: TensorLike = 0.0
    weight: float = 1.0

    def rows(self, flow: IncompressibleFlow, sl: slice):
        return flow.pressure_rows(self.x[sl], _per_point(self.value, sl, 1))


@dataclass
class NavierStokesResult:
    theta: torch.Tensor
    flow: IncompressibleFlow
    history: List[dict] = field(default_factory=list)
    converged: bool = False

    def velocity(self, x: torch.Tensor) -> torch.Tensor:
        return self.flow.evaluate(x, self.theta)[0]

    def pressure(self, x: torch.Tensor) -> torch.Tensor:
        return self.flow.evaluate(x, self.theta)[2]


def _boundary_system(flow, conditions, chunk) -> NormalEquations:
    ne = NormalEquations(flow.n_unknowns, device=flow.u_inf.device, dtype=flow.u_inf.dtype)
    for c in conditions:
        for i in range(0, c.x.shape[0], chunk):
            A, b = c.rows(flow, slice(i, i + chunk))
            ne.add(A, b, c.weight)
    return ne


def solve_navier_stokes(
    flow: IncompressibleFlow,
    x: torch.Tensor,
    conditions: Sequence,
    *,
    weights: Optional[torch.Tensor] = None,
    theta: Optional[torch.Tensor] = None,
    max_iter: int = 10,
    tol: float = 1e-4,
    mu: float = 1e-12,
    chunk: int = 2500,
    newton: bool = True,
    verbose: bool = False,
) -> NavierStokesResult:
    """Steady Navier-Stokes by Newton's method, one least-squares solve per step.

    Parameters
    ----------
    flow : IncompressibleFlow
    x : (M, d) tensor
        Interior collocation points.
    conditions : sequence of Dirichlet, Slip, Traction, PressurePoint
        Boundary rows.  They do not depend on the iterate, so their part of the
        normal equations is built once and reused by every step.
    weights : (M,) tensor, optional
        Per-point weights on the momentum rows.  Keep them fixed across steps (and
        across designs, when comparing): weights that depend on the iterate make
        the discrete problem depend on the path taken to it.
    theta : tensor, optional
        Starting iterate.  Without one the first step is Stokes flow.
    max_iter, tol :
        Stop when the relative velocity change on ``x`` falls below ``tol``.
        (The change in ``theta`` itself is a poor criterion: a rich basis has
        near-null directions that move without changing the field.)
    mu : float
        Relative ridge, raised automatically if the equations are not positive
        definite (see :class:`~fastlsq.NormalEquations`).
    newton : bool
        ``False`` stops after the first (Stokes or linearised) solve.
    """
    t_start = time.time()
    base = _boundary_system(flow, conditions, chunk)
    w = None if weights is None else weights.to(x)
    U = GU = None
    if theta is not None:
        U, GU, _ = flow.evaluate(x, theta)
    hist, converged = [], False
    for it in range(max_iter if newton else 1):
        t0 = time.time()
        ne = base.copy()
        for i in range(0, x.shape[0], chunk):
            sl = slice(i, i + chunk)
            A, b = flow.momentum_rows(x[sl], None if U is None else U[sl], None if GU is None else GU[sl])
            ne.add(A, b, 1.0 if w is None else w[sl].repeat(flow.dim))
        theta = ne.solve(mu=mu)
        Un, GUn, _ = flow.evaluate(x, theta)
        step = float((Un - U).norm() / Un.norm()) if U is not None else float("inf")
        U, GU = Un, GUn
        hist.append(dict(iteration=it, step=step, mu=ne.mu_used, seconds=time.time() - t0))
        if verbose:
            print(f"  newton {it}: du/u {step:.2e}  ({time.time() - t0:.1f}s)", flush=True)
        if step < tol:
            converged = True
            break
    if verbose:
        print(f"  {len(hist)} solves, {time.time() - t_start:.1f}s", flush=True)
    return NavierStokesResult(theta=theta, flow=flow, history=hist, converged=converged)
