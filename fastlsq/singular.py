# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Singular columns for flow past a body: the method of fundamental solutions.

A global sinusoid resolves the flow next to a body slowly.  The reason is
analytic: the flow's continuation *into* the body is singular a short distance
behind the wall, and a band-limited basis has to spend many features imitating
that singularity from outside.  The method of fundamental solutions puts the
singularities there explicitly -- closed-form point solutions seated just inside
the surface -- and lets the least-squares solve choose their strengths.

Per source point ``y`` and unit direction ``e_j`` (``r = x - y``, ``R = |r|``):

* **Stokeslet** (point force), velocity and pressure for viscosity ``nu``::

      G_ij = d_ij / R + r_i r_j / R^3,          p_j = 2 nu r_j / R^3

* **Source doublet** (potential dipole), harmonic and pressure-free::

      D_ij = d_ij / R^3 - 3 r_i r_j / R^5

* **Pressure poles**, with no velocity: a monopole ``1/R`` and dipoles
  ``r_j / R^3``.  They are harmonic, so they carry the part of the pressure that
  the nonlinear terms need and the Stokes columns cannot supply.

Each force or doublet column is an exact solution of the Stokes equations,
``-nu lap u + grad p = 0`` and ``div u = 0``, so in a collocation row its viscous
and pressure terms cancel analytically: only the convective terms of
Navier-Stokes see it.  Optionally the force columns are **Oseenlets**, exact
solutions of Navier-Stokes linearised about a free stream ``U e_x``, each with its
own wake.

Stokes flow past a sphere is exactly one Stokeslet plus one doublet at the
centre; :class:`StokesSingularities` with sources spread *inside* a general body
is the same idea, made to fit any shape.  All closed forms are 3-D.

>>> sing = StokesSingularities(sources, nu=0.02, mirror_axis=1)
>>> B = sing.blocks(x)              # val (M,3,P), grad (M,3,3,P), p, pgrad, stokes
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


# ----------------------------------------------------------------------
# Closed forms.  r: (..., 3), R = |r|: (...).  Gradients are [..., i, j, m]
# = d/dx_m of entry (i, j); pressure gradients [..., j, m].
# ----------------------------------------------------------------------


def _eye(r: torch.Tensor) -> torch.Tensor:
    return torch.eye(3, device=r.device, dtype=r.dtype)


def stokeslet(r: torch.Tensor, R: Optional[torch.Tensor] = None):
    """Stokeslet ``G``, its gradient, pressure ``P`` and pressure gradient.

    Returns ``G`` (..., 3, 3), ``dG`` (..., 3, 3, 3), ``P`` (..., 3) and
    ``dP`` (..., 3, 3).  Column ``j`` of ``G`` with pressure ``nu * P[..., j]``
    solves ``-nu lap u + grad p = 0``, ``div u = 0`` away from ``r = 0``.
    """
    R = r.norm(dim=-1) if R is None else R
    I = _eye(r)
    R3 = (R ** 3)[..., None, None]
    R5 = (R ** 5)[..., None, None]
    ri, rj = r[..., :, None], r[..., None, :]
    G = I / R[..., None, None] + ri * rj / R3
    rm = r[..., None, None, :]
    dG = (-I[..., None] * rm / R3[..., None]
          + (I[:, None, :] * r[..., None, :, None] + I[None, :, :] * r[..., :, None, None]) / R3[..., None]
          - 3 * ri[..., None] * rj[..., None] * rm / R5[..., None])
    P = 2 * r / R3[..., 0]
    dP = 2 * (I / R3 - 3 * ri * rj / R5)
    return G, dG, P, dP


def stokes_doublet(r: torch.Tensor, R: Optional[torch.Tensor] = None):
    """Potential source doublet ``D = I/R^3 - 3 r r^T / R^5`` and its gradient.

    Each column is harmonic and divergence-free with zero pressure, so it is a
    Stokes solution on its own.
    """
    R = r.norm(dim=-1) if R is None else R
    I = _eye(r)
    R3, R5, R7 = ((R ** k)[..., None, None] for k in (3, 5, 7))
    ri, rj = r[..., :, None], r[..., None, :]
    D = I / R3 - 3 * ri * rj / R5
    rm = r[..., None, None, :]
    dD = (-3 * I[..., None] * rm / R5[..., None]
          - 3 * (I[:, None, :] * r[..., None, :, None] + I[None, :, :] * r[..., :, None, None]) / R5[..., None]
          + 15 * ri[..., None] * rj[..., None] * rm / R7[..., None])
    return D, dD


def _oseen_f(s: torch.Tensor):
    """``f(s) = (1 - e^-s)/s`` and its first two derivatives; a series near 0."""
    small = s < 0.05
    ss = torch.where(small, torch.ones_like(s), s)
    e = torch.exp(-ss)
    g = e * (1 + ss) - 1
    f, f1, f2 = (1 - e) / ss, g / ss ** 2, -e / ss - 2 * g / ss ** 3
    s2, s3, s4 = s * s, s ** 3, s ** 4
    f_s = 1 - s / 2 + s2 / 6 - s3 / 24 + s4 / 120
    f1_s = -0.5 + s / 3 - s2 / 8 + s3 / 30 - s4 / 144
    f2_s = 1 / 3 - s / 4 + s2 / 10 - s3 / 36 + s4 / 168
    return torch.where(small, f_s, f), torch.where(small, f1_s, f1), torch.where(small, f2_s, f2)


def oseenlet(r: torch.Tensor, k: float, R: Optional[torch.Tensor] = None):
    """Oseen fundamental solution for a free stream along ``+x``, ``k = U / (2 nu)``.

    ``O_ij = 2 e^-s / R d_ij - k f'(s) a_i a_j - f(s) b_ij`` with ``s = k (R - r_x)``,
    ``a = r/R - e_x``, ``b = (I - r r^T / R^2) / R`` and ``f(s) = (1 - e^-s)/s``.
    Column ``j`` with pressure ``nu * P[..., j]`` (the Stokeslet's) solves
    ``U d_x u - nu lap u + grad p = 0``; near the source it tends to the Stokeslet.
    Same return layout as :func:`stokeslet`.
    """
    R = r.norm(dim=-1) if R is None else R
    I = _eye(r)
    Rx = R[..., None]
    a = r / Rx - I[0]
    s = k * (R - r[..., 0])
    f, f1, f2 = _oseen_f(s)
    es = torch.exp(-s)
    ri, rj = r[..., :, None], r[..., None, :]
    b = (I - ri * rj / Rx[..., None] ** 2) / Rx[..., None]
    ai, aj = a[..., :, None], a[..., None, :]
    O = (2 * es / R)[..., None, None] * I - (k * f1)[..., None, None] * ai * aj - f[..., None, None] * b
    am, rm = a[..., None, None, :], r[..., None, None, :]
    R3 = (R ** 3)[..., None, None, None]
    R5 = (R ** 5)[..., None, None, None]
    T1 = 2 * I[..., None] * (-(es * k / R)[..., None, None, None] * am - es[..., None, None, None] * rm / R3)
    bim, bjm = b[..., :, None, :], b[..., None, :, :]            # d_m a_i = b_im
    T2 = -k * ((k * f2)[..., None, None, None] * am * ai[..., None] * aj[..., None]
               + f1[..., None, None, None] * (bim * aj[..., None] + ai[..., None] * bjm))
    dbij = (-I[..., None] * rm / R3
            - (I[:, None, :] * r[..., None, :, None] + I[None, :, :] * r[..., :, None, None]) / R3
            + 3 * ri[..., None] * rj[..., None] * rm / R5)
    T3 = -((k * f1)[..., None, None, None] * am * b[..., None] + f[..., None, None, None] * dbij)
    P = 2 * r / (R ** 3)[..., None]
    dP = 2 * (I / (R ** 3)[..., None, None] - 3 * ri * rj / (R ** 5)[..., None, None])
    return O, T1 + T2 + T3, P, dP


# ----------------------------------------------------------------------
# The column bank
# ----------------------------------------------------------------------


class StokesSingularities:
    """Stokeslets, source doublets and pressure poles at fixed source points.

    Parameters
    ----------
    sources : (S, 3) tensor
        Source points, strictly inside the body (see :func:`seat_sources`).  With
        ``mirror_axis`` set, give only the half with ``x[mirror_axis] > 0``.
    nu : float
        Kinematic viscosity; scales the Stokeslet pressure.
    mirror_axis : int, optional
        Pair every column with its mirror image across ``x[mirror_axis] = 0``,
        matching :class:`~fastlsq.DivergenceFreeBasis` with the same axis.
    oseen : bool
        Use Oseenlets for the force columns (free stream ``u_inf`` along ``+x``).
        The doublets then carry the linearised Bernoulli pressure ``-u_inf u_x``.
    u_inf : float
        Free-stream speed for ``oseen=True``.

    Ten unknowns per source, ordered in groups, source-major within each::

        [ force 3S | doublet 3S | pressure monopole S | pressure dipole 3S ]

    :meth:`blocks` returns ``val`` (M, 3, P), ``grad`` (M, 3, 3, P) with ``[:, k, m]``
    the derivative of component ``k`` along ``x_m``, ``p`` (M, P), ``pgrad``
    (M, 3, P) and ``stokes`` (M, 3, P) = ``-nu lap u + grad p`` per column, which is
    known in closed form without a Laplacian: zero for the force and doublet
    columns (``-u_inf d_x u`` for Oseen ones) and ``grad p`` for the pressure poles.
    """

    PER_SOURCE = 10

    def __init__(self, sources: torch.Tensor, nu: float, mirror_axis: Optional[int] = None,
                 oseen: bool = False, u_inf: float = 1.0):
        if sources.ndim != 2 or sources.shape[1] != 3:
            raise ValueError(f"sources must be (S, 3), got {tuple(sources.shape)}")
        self.sources = sources
        self.nu = float(nu)
        self.mirror_axis = mirror_axis
        self.oseen = oseen
        self.u_inf = float(u_inf)
        self.k = self.u_inf / (2 * self.nu)

    @property
    def n_sources(self) -> int:
        return self.sources.shape[0]

    @property
    def n_unknowns(self) -> int:
        return self.PER_SOURCE * self.n_sources

    def _reflection(self, x):
        r = torch.ones(3, device=x.device, dtype=x.dtype)
        r[self.mirror_axis] = -1.0
        return r

    def blocks(self, x: torch.Tensor):
        """Design blocks at ``x`` (M, 3); see the class docstring for shapes."""
        M, S = x.shape[0], self.n_sources
        kw = dict(device=x.device, dtype=x.dtype)
        V = torch.zeros(M, 3, 6 * S, **kw)
        dV = torch.zeros(M, 3, 3, 6 * S, **kw)
        Pv = torch.zeros(M, 10 * S, **kw)
        dPv = torch.zeros(M, 3, 10 * S, **kw)
        y = self.sources.to(**kw)
        images = [(y, None)]
        if self.mirror_axis is not None:
            Rm = self._reflection(x)
            images.append((y * Rm, Rm))
        for src, Rm in images:
            r = x[:, None, :] - src[None, :, :]                       # (M, S, 3)
            Rn = r.norm(dim=-1)
            G, dG, P, dP = oseenlet(r, self.k, Rn) if self.oseen else stokeslet(r, Rn)
            D, dD = stokes_doublet(r, Rn)
            if Rm is not None:
                # the partner of direction e_j is R e_j at the mirrored source
                G, dG, P, dP = G * Rm, dG * Rm[:, None], P * Rm, dP * Rm[:, None]
                D, dD = D * Rm, dD * Rm[:, None]
            V += torch.cat([G.permute(0, 2, 1, 3).reshape(M, 3, -1),
                            D.permute(0, 2, 1, 3).reshape(M, 3, -1)], -1)
            dV += torch.cat([dG.permute(0, 2, 4, 1, 3).reshape(M, 3, 3, -1),
                             dD.permute(0, 2, 4, 1, 3).reshape(M, 3, 3, -1)], -1)
            mono = 1.0 / Rn
            dmono = -r / Rn[..., None] ** 3
            if self.oseen:
                Pd = -self.u_inf * D[..., 0, :].reshape(M, -1)          # p = -U u_x
                dPd = -self.u_inf * dD[..., 0, :, :].permute(0, 3, 1, 2).reshape(M, 3, -1)
            else:
                Pd = torch.zeros(M, 3 * S, **kw)
                dPd = torch.zeros(M, 3, 3 * S, **kw)
            dPc = dP.permute(0, 3, 1, 2).reshape(M, 3, -1)
            Pv += torch.cat([self.nu * P.reshape(M, -1), Pd, mono, P.reshape(M, -1) / 2], -1)
            dPv += torch.cat([self.nu * dPc, dPd, dmono.permute(0, 2, 1), dPc / 2], -1)
        Zp = torch.zeros(M, 3, 4 * S, **kw)
        val = torch.cat([V, Zp], -1)
        grad = torch.cat([dV, Zp[:, :, None].expand(M, 3, 3, 4 * S)], -1)
        stokes = torch.zeros(M, 3, 10 * S, **kw)
        stokes[..., 6 * S:] = dPv[..., 6 * S:]
        if self.oseen:
            stokes[..., :6 * S] = -self.u_inf * dV[:, :, 0]
        return dict(val=val, grad=grad, p=Pv, pgrad=dPv, stokes=stokes)

    def plain(self, coef: torch.Tensor) -> dict:
        """Strengths per source with mirror partners expanded: ``pos`` (S', 3),
        ``force``, ``doublet``, ``dipole`` (S', 3) and ``monopole`` (S',)."""
        S = self.n_sources
        st = coef[:3 * S].reshape(S, 3)
        db = coef[3 * S:6 * S].reshape(S, 3)
        mo = coef[6 * S:7 * S]
        dp = coef[7 * S:].reshape(S, 3)
        out = dict(pos=self.sources, force=st, doublet=db, monopole=mo, dipole=dp)
        if self.mirror_axis is not None:
            Rm = self._reflection(self.sources)
            out = dict(pos=torch.cat([self.sources, self.sources * Rm]),
                       force=torch.cat([st, st * Rm]), doublet=torch.cat([db, db * Rm]),
                       monopole=torch.cat([mo, mo]), dipole=torch.cat([dp, dp * Rm]))
        return out


# ----------------------------------------------------------------------
# Placing the sources
# ----------------------------------------------------------------------


def seat_sources(
    body_sdf: Callable[[torch.Tensor], torch.Tensor],
    surface: torch.Tensor,
    depth: float,
    *,
    max_march: Optional[float] = None,
    n_march: int = 49,
    fraction: float = 0.7,
    mirror_axis: Optional[int] = None,
    min_offset: float = 0.0,
) -> torch.Tensor:
    """Push surface points into the body to serve as MFS source points.

    Each point moves along the inward normal by ``depth``, capped at ``fraction``
    of the local half-thickness (found by marching inward to the deepest point),
    so a source never lands near the far wall of a thin part.  Where even that is
    not safely inside (thin, curved parts), the deepest point on the ray is used.

    Parameters
    ----------
    body_sdf : callable
        Signed distance of the **body**, negative inside the solid.  For a fluid
        :class:`~fastlsq.SDFDomain` ``dom``, pass ``lambda x: -dom(x)``.
    surface : (S, 3) tensor
        Points on the body surface, e.g. from :func:`~fastlsq.project_to_boundary`.
    depth : float
        Target distance below the surface.  A few times the spacing of the source
        points is typical.
    mirror_axis, min_offset : optional
        Keep ``x[mirror_axis] >= min_offset`` so a source never coincides with its
        mirror image.
    """
    from fastlsq.geometry import outward_normal

    n = outward_normal(body_sdf, surface)
    ts = torch.linspace(0.0, max_march or 4.0 * depth, n_march, device=surface.device, dtype=surface.dtype)
    psi = torch.stack([body_sdf(surface - t * n).reshape(-1) for t in ts], 1)       # (S, T)
    half = (-psi.min(dim=1).values).clamp_min(1e-3 * depth)
    d = torch.minimum(torch.full_like(half, depth), fraction * half)
    y = surface - d[:, None] * n
    bad = body_sdf(y).reshape(-1) > -0.3 * d
    if bool(bad.any()):
        t_deep = ts[psi.argmin(dim=1)]
        y[bad] = surface[bad] - t_deep[bad, None] * n[bad]
    if mirror_axis is not None:
        y[:, mirror_axis] = y[:, mirror_axis].clamp_min(min_offset)
    return y
