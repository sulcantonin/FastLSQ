# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Geometry samplers for generating collocation and boundary points.

Two families live here:

* **Analytic samplers** (:func:`sample_box`, :func:`sample_ball`, ...) for the
  handful of shapes with a direct sampling formula.
* **Signed-distance (SDF) geometry** (:class:`SDFDomain` and friends) for
  everything else.  A domain is specified by a *membership oracle* -- any
  callable ``ψ(x)`` that is negative inside, zero on the boundary and positive
  outside.  Interior points come from rejection sampling on ``ψ < 0``, boundary
  points from projecting onto ``ψ = 0``, and outward normals from ``∇ψ/‖∇ψ‖``,
  which is exactly what a Neumann or Robin boundary operator needs.

  Because the solver is meshless, this is the whole story for complex geometry:
  no mesh generation, and the domain only ever has to answer "is this point
  inside?".  CSG composition (``|``, ``&``, ``-``) builds non-convex and
  multiply-connected domains from primitives.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Optional, Callable, Sequence, Tuple, Union

from fastlsq.device import get_device

SDFT = Callable[[torch.Tensor], torch.Tensor]


# ======================================================================
# Generic samplers
# ======================================================================

def sample_box(
    n: int,
    dim: int,
    *,
    bounds: Optional[Tuple[float, float]] = None,
    device=None,
) -> torch.Tensor:
    """Sample uniformly from a hypercube [a, b]^dim.

    Parameters
    ----------
    n : int
        Number of points.
    dim : int
        Spatial dimension.
    bounds : tuple[float, float], optional
        (min, max) bounds. Default: (0.0, 1.0).
    device : torch.device

    Returns
    -------
    x : Tensor, shape (n, dim)
    """
    if bounds is None:
        bounds = (0.0, 1.0)
    a, b = bounds
    return torch.rand(n, dim, device=device or get_device()) * (b - a) + a


def sample_ball(
    n: int,
    dim: int,
    *,
    radius: float = 1.0,
    center: Optional[torch.Tensor] = None,
    device=None,
) -> torch.Tensor:
    """Sample uniformly from a ball (uniform in volume, not on surface).

    Parameters
    ----------
    n : int
        Number of points.
    dim : int
        Spatial dimension.
    radius : float
        Ball radius.
    center : Tensor, shape (dim,), optional
        Center point. Default: origin.
    device : torch.device

    Returns
    -------
    x : Tensor, shape (n, dim)
    """
    # Sample from unit ball: uniform direction, then scale by r^(1/dim)
    x = torch.randn(n, dim, device=device or get_device())
    x = x / torch.norm(x, dim=1, keepdim=True)
    r = torch.rand(n, 1, device=device or get_device()) ** (1.0 / dim)
    x = x * r * radius

    if center is not None:
        x = x + center.unsqueeze(0)
    return x


def sample_sphere(
    n: int,
    dim: int,
    *,
    radius: float = 1.0,
    center: Optional[torch.Tensor] = None,
    device=None,
) -> torch.Tensor:
    """Sample uniformly from a sphere surface.

    Parameters
    ----------
    n : int
        Number of points.
    dim : int
        Spatial dimension.
    radius : float
        Sphere radius.
    center : Tensor, shape (dim,), optional
        Center point. Default: origin.
    device : torch.device

    Returns
    -------
    x : Tensor, shape (n, dim)
    """
    x = torch.randn(n, dim, device=device or get_device())
    x = x / torch.norm(x, dim=1, keepdim=True) * radius

    if center is not None:
        x = x + center.unsqueeze(0)
    return x


def sample_interval(
    n: int,
    *,
    a: float = 0.0,
    b: float = 1.0,
    device=None,
) -> torch.Tensor:
    """Sample uniformly from an interval [a, b].

    Parameters
    ----------
    n : int
        Number of points.
    a, b : float
        Interval bounds.
    device : torch.device

    Returns
    -------
    x : Tensor, shape (n, 1)
    """
    return torch.rand(n, 1, device=device or get_device()) * (b - a) + a


def sample_boundary_box(
    n: int,
    dim: int,
    *,
    bounds: Optional[Tuple[float, float]] = None,
    device=None,
) -> torch.Tensor:
    """Sample uniformly from the boundary of a hypercube.

    Parameters
    ----------
    n : int
        Number of points.
    dim : int
        Spatial dimension.
    bounds : tuple[float, float], optional
        (min, max) bounds. Default: (0.0, 1.0).
    device : torch.device

    Returns
    -------
    x : Tensor, shape (n, dim)
    """
    if bounds is None:
        bounds = (0.0, 1.0)
    a, b = bounds

    n_per_face = n // (2 * dim)
    points = []

    for d in range(dim):
        # Face at x_d = a
        x = torch.rand(n_per_face, dim, device=device or get_device()) * (b - a) + a
        x[:, d] = a
        points.append(x)

        # Face at x_d = b
        x = torch.rand(n_per_face, dim, device=device or get_device()) * (b - a) + a
        x[:, d] = b
        points.append(x)

    # Fill remainder randomly
    remainder = n - len(points) * n_per_face
    if remainder > 0:
        x = torch.rand(remainder, dim, device=device or get_device()) * (b - a) + a
        face_idx = torch.randint(0, 2 * dim, (remainder,), device=device or get_device())
        dim_idx = face_idx // 2
        val = (face_idx % 2) * (b - a) + a
        x[torch.arange(remainder, device=device or get_device()), dim_idx] = val
        points.append(x)

    return torch.cat(points, dim=0)


# ======================================================================
# Signed-distance geometry: membership oracle -> points, boundary, normals
# ======================================================================

def _as_bounds(
    bounds: Union[Tuple[float, float], Sequence[Sequence[float]], torch.Tensor],
    dim: Optional[int] = None,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Normalise a bounds spec to a ``(2, d)`` tensor of ``[lo; hi]`` rows.

    Accepts a single ``(lo, hi)`` pair applied to every axis (``dim`` required),
    or a per-axis sequence ``[(lo0, hi0), (lo1, hi1), ...]``.

    Disambiguation is **by type**, because a ``(2, 2)`` input is otherwise
    genuinely ambiguous in 2-D: a ``Tensor`` is taken to be the canonical
    ``[lo; hi]`` form (what this function itself returns, so re-normalising is
    idempotent), while a list/tuple/array is taken to be a per-axis sequence of
    pairs.  Getting this wrong silently produces an empty sampling box, so the
    result is validated rather than trusted.
    """
    device = device or get_device()
    dtype = dtype or torch.get_default_dtype()
    was_tensor = isinstance(bounds, torch.Tensor)
    t = torch.as_tensor(bounds, device=device, dtype=dtype)

    if t.shape == (2,):
        if dim is None:
            raise ValueError(
                "bounds given as a single (lo, hi) pair requires dim= to be set"
            )
        out = t.reshape(2, 1).expand(2, dim).contiguous()
    elif t.dim() == 2 and was_tensor and t.shape[0] == 2:
        out = t.contiguous()               # canonical [lo; hi]
    elif t.dim() == 2 and t.shape[1] == 2:
        out = t.t().contiguous()           # [(lo,hi)] per axis -> (2, d)
    elif t.dim() == 2 and t.shape[0] == 2:
        out = t.contiguous()
    else:
        raise ValueError(
            f"bounds must be (lo, hi) or a per-axis sequence of pairs; "
            f"got shape {tuple(t.shape)}"
        )

    if not (out[1] > out[0]).all():
        raise ValueError(
            f"bounds has a non-positive extent: lo={out[0].tolist()}, "
            f"hi={out[1].tolist()}. Expected per-axis (lo, hi) pairs, e.g. "
            f"[(-1.0, 1.0), (-1.0, 1.0)]."
        )
    return out


def _value_and_grad(psi: SDFT, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate ``ψ(x)`` -> ``(M,)`` and ``∇ψ(x)`` -> ``(M, d)`` via autograd.

    ``psi`` therefore only has to be written once, as a plain torch expression;
    no hand-derived gradient is required.  Any ``ψ`` built from torch ops works,
    including the ``min``/``max`` of CSG composition (differentiable a.e.).
    """
    xg = x.detach().requires_grad_(True)
    val = psi(xg).reshape(-1)
    (grad,) = torch.autograd.grad(val.sum(), xg, create_graph=False)
    return val.detach(), grad.detach()


def sample_sdf(
    psi: SDFT,
    n: int,
    bounds: Union[Tuple[float, float], Sequence[Sequence[float]], torch.Tensor],
    *,
    dim: Optional[int] = None,
    batch: Optional[int] = None,
    max_rounds: int = 1000,
    device=None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Rejection-sample ``n`` points uniformly from the interior ``{ψ < 0}``.

    Proposals are drawn uniformly from the bounding box and kept where ``ψ < 0``.
    The result is exactly uniform on the domain (rejection sampling is unbiased);
    the only cost of a loose bounding box is throughput.

    Parameters
    ----------
    psi : callable
        Membership oracle: ``(M, d) -> (M,)``, negative strictly inside.
        Only its **sign** is used here, so any implicit function works -- it need
        not be a true distance function.
    n : int
        Number of interior points to return.
    bounds : (lo, hi) or sequence of per-axis (lo, hi)
        Bounding box enclosing the domain.  Must actually contain it: points
        outside the box are never proposed, so a too-tight box silently truncates
        the domain.
    dim : int, optional
        Spatial dimension; required only when ``bounds`` is a single pair.
    batch : int, optional
        Proposals per round.  Default: adaptive, starting at ``4n``.
    max_rounds : int
        Safety cap, so an empty or vanishingly small domain raises instead of
        looping forever.

    Returns
    -------
    x : Tensor, shape (n, d)

    Raises
    ------
    RuntimeError
        If ``n`` points could not be collected within ``max_rounds``, reporting
        the observed acceptance rate -- almost always a wrong-sign ``ψ`` or a
        bounding box that does not enclose the domain.
    """
    device = device or get_device()
    lo_hi = _as_bounds(bounds, dim=dim, device=device)
    d = lo_hi.shape[1]
    lo, hi = lo_hi[0], lo_hi[1]

    batch = batch or max(4 * n, 1024)
    kept: list[torch.Tensor] = []
    n_kept = n_proposed = 0
    # Abort early on a domain that never accepts (wrong-sign psi, or a box that
    # misses the domain) instead of grinding through max_rounds of huge batches.
    dry_rounds = 0

    for _ in range(max_rounds):
        u = torch.rand(batch, d, device=device, dtype=lo.dtype, generator=generator)
        cand = lo + u * (hi - lo)
        inside = cand[psi(cand).reshape(-1) < 0]
        n_proposed += batch
        if inside.numel():
            kept.append(inside)
            n_kept += inside.shape[0]
            dry_rounds = 0
        else:
            dry_rounds += 1
        if n_kept >= n:
            return torch.cat(kept, dim=0)[:n]
        if dry_rounds >= 8 and n_kept == 0:
            break
        # Re-aim the batch size at the shortfall using the measured rate.
        rate = max(n_kept / n_proposed, 1e-6)
        batch = int(min(max((n - n_kept) / rate * 1.5, 1024), 1 << 20))

    raise RuntimeError(
        f"sample_sdf: collected only {n_kept}/{n} points from {n_proposed} proposals "
        f"(acceptance rate {n_kept / max(n_proposed, 1):.2e}) in bounds "
        f"lo={lo.tolist()}, hi={hi.tolist()}. Check that psi is negative inside the "
        f"domain and that bounds enclose it."
    )


def project_to_boundary(
    psi: SDFT,
    x: torch.Tensor,
    *,
    n_iter: int = 24,
    tol: float = 1e-10,
    grad_eps: float = 1e-12,
    n_backtrack: int = 12,
) -> torch.Tensor:
    """Project points onto the zero level set ``{ψ = 0}``.

    Each step is the Newton correction along the normal::

        x ← x − ψ(x) ∇ψ(x) / ‖∇ψ(x)‖²

    For a **true** signed-distance function ``‖∇ψ‖ ≡ 1``, so this reduces to the
    textbook ``x ↦ x − ψ(x)∇ψ(x)`` and converges in a single step for a planar
    boundary.  The ``1/‖∇ψ‖²`` normalisation is what makes it work for a general
    implicit function whose gradient is not unit -- e.g. the polar
    ``ψ = r − R(1 + a cos kθ)`` of :func:`sdf_flower`, or any CSG composite --
    where the unnormalised step would over- or under-shoot badly.

    Steps are **damped**: a full Newton step that fails to reduce ``|ψ|`` is
    halved (per point, up to ``n_backtrack`` times) before being taken.  Undamped
    Newton diverges on exactly the cases this exists for -- ``sdf_flower``'s
    ``‖∇ψ‖`` spans 1 to ~10, and a seed near the origin gets thrown far outside
    by a full step.  Damping costs only extra oracle calls (no autograd) and
    makes convergence from arbitrary interior seeds reliable.

    Parameters
    ----------
    psi : callable
        Membership oracle, ``(M, d) -> (M,)``.
    x : Tensor, shape (M, d)
        Seed points, typically from :func:`sample_sdf` or the bounding box.
    n_iter : int
        Maximum Newton steps.  Stops early once ``max|ψ| < tol``.
    tol : float
        Convergence threshold on ``|ψ|``.
    grad_eps : float
        Steps are skipped where ``‖∇ψ‖²`` falls below this, which happens at
        medial-axis / cusp points where the normal is undefined.
    n_backtrack : int
        Maximum halvings per step.

    Returns
    -------
    x_b : Tensor, shape (M, d)
        Points on the boundary.  Convergence is not guaranteed from every seed --
        a point seeded exactly on a medial axis or cusp has no well-defined
        nearest boundary point.  Check with ``psi(x_b).abs().max()`` and filter;
        :func:`sample_boundary_sdf` does this for you.
    """
    xb = x.detach().clone()
    val, grad = _value_and_grad(psi, xb)

    for _ in range(n_iter):
        if val.abs().max() < tol:
            break
        gn2 = (grad ** 2).sum(dim=1)
        direction = torch.where(
            (gn2 > grad_eps).unsqueeze(1),
            (val / gn2.clamp_min(grad_eps)).unsqueeze(1) * grad,
            torch.zeros_like(grad),
        )

        # Backtrack per point until the step actually decreases |psi|.
        t = torch.ones_like(val).unsqueeze(1)
        cand = xb - direction
        cval = psi(cand).reshape(-1)
        for _ in range(n_backtrack):
            worse = (cval.abs() > val.abs()) & (val.abs() > tol)
            if not bool(worse.any()):
                break
            t = torch.where(worse.unsqueeze(1), t * 0.5, t)
            cand = xb - t * direction
            cval = psi(cand).reshape(-1)

        # Accept only where |psi| actually decreased, so the iteration is
        # monotone per point; a point whose backtracking was exhausted keeps its
        # previous position rather than taking a step known to be worse.
        improved = (cval.abs() <= val.abs()).unsqueeze(1)
        xb = torch.where(improved, cand, xb)
        val, grad = _value_and_grad(psi, xb)
    return xb


def _fd_grad(psi: SDFT, x: torch.Tensor, h: float) -> torch.Tensor:
    """Central finite-difference gradient of ``ψ`` -> ``(M, d)``."""
    g = torch.empty_like(x)
    for k in range(x.shape[1]):
        e = torch.zeros_like(x)
        e[:, k] = h
        g[:, k] = (psi(x + e).reshape(-1) - psi(x - e).reshape(-1)) / (2.0 * h)
    return g


def outward_normal(
    psi: SDFT, x: torch.Tensor, *, grad_eps: float = 1e-12, fd_step: float = 1e-6
) -> torch.Tensor:
    """Unit outward normal ``n = ∇ψ/‖∇ψ‖`` at ``x``, shape ``(M, d)``.

    ``ψ`` increases outward by definition, so ``∇ψ`` points out of the domain.
    This is the field a Neumann condition ``∂u/∂n = g`` needs: with a basis
    gradient ``∇φ`` of shape ``(M, d, N)``, the boundary block is
    ``(n.unsqueeze(-1) * grad_phi).sum(1)``, an ``(M, N)`` design matrix -- see
    :meth:`SDFDomain.neumann_rows`.

    Autograd is used first, with a **central finite-difference fallback**
    wherever it returns a degenerate (near-zero) gradient.  The fallback is not a
    nicety: a distance built as ``sqrt(min_k d_k²)`` -- which is how
    :func:`sdf_polygon`, and hence :func:`sdf_tokamak`, is defined -- must floor
    the squared distance before the square root to keep it finite, and a floor
    contributes **exactly zero gradient** below its threshold.  On the boundary
    the squared distance *is* below that threshold, so autograd alone returns a
    zero normal precisely where the normal is wanted.  Central differences
    evaluate ``ψ`` off the surface instead, and are exact for a piecewise-linear
    ``ψ`` such as a polygon.

    Normals are exact wherever ``ψ`` is differentiable.  At a cusp or reentrant
    corner (an L-shape's inner corner, the medial axis of a CSG ``max``) the
    normal genuinely does not exist; there the value is whichever branch was
    picked up, and a still-degenerate gradient yields a zero vector rather than a
    division blow-up.
    """
    _, grad = _value_and_grad(psi, x)
    degenerate = grad.norm(dim=1) <= grad_eps
    if bool(degenerate.any()):
        grad = grad.clone()
        grad[degenerate] = _fd_grad(psi, x[degenerate], fd_step)

    norm = grad.norm(dim=1, keepdim=True)
    return torch.where(norm > grad_eps, grad / norm.clamp_min(grad_eps), torch.zeros_like(grad))


def sample_boundary_sdf(
    psi: SDFT,
    n: int,
    bounds: Union[Tuple[float, float], Sequence[Sequence[float]], torch.Tensor],
    *,
    dim: Optional[int] = None,
    oversample: int = 4,
    tol: float = 1e-8,
    n_iter: int = 12,
    device=None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample ``n`` points on the boundary ``{ψ = 0}`` by projecting box points.

    Draws ``oversample * n`` uniform proposals from the bounding box, projects
    them with :func:`project_to_boundary`, discards any that failed to converge
    (``|ψ| > tol``) or left the box, and returns ``n`` of the survivors.

    Distribution caveat
    -------------------
    The result is **not** uniform with respect to arc length / surface measure.
    It is the pushforward of the uniform box measure under the projection, which
    over-weights boundary regions facing a large volume of the box (convex bulges)
    and under-weights concave pockets.  For collocation this is normally
    harmless -- the boundary rows only need adequate coverage, and the residual is
    weighted per row anyway -- but do not use these points as quadrature nodes for
    a surface integral without reweighting.  For the exact uniform surface measure
    on a sphere, use :func:`sample_sphere`.
    """
    device = device or get_device()
    lo_hi = _as_bounds(bounds, dim=dim, device=device)
    d = lo_hi.shape[1]
    lo, hi = lo_hi[0], lo_hi[1]

    kept: list[torch.Tensor] = []
    n_kept = 0
    for _ in range(64):
        m = max(oversample * (n - n_kept), 1024)
        u = torch.rand(m, d, device=device, dtype=lo.dtype, generator=generator)
        seeds = lo + u * (hi - lo)
        proj = project_to_boundary(psi, seeds, n_iter=n_iter, tol=tol)
        ok = psi(proj).reshape(-1).abs() < tol
        ok &= ((proj >= lo) & (proj <= hi)).all(dim=1)
        good = proj[ok]
        if good.numel():
            kept.append(good)
            n_kept += good.shape[0]
        if n_kept >= n:
            return torch.cat(kept, dim=0)[:n]

    raise RuntimeError(
        f"sample_boundary_sdf: collected only {n_kept}/{n} boundary points. "
        f"Check that psi has a zero level set inside bounds."
    )


# ======================================================================
# Custom sampler wrapper
# ======================================================================

def custom_sampler(
    sampler_fn: Callable[[int], torch.Tensor],
    n: int,
) -> torch.Tensor:
    """Wrap a custom sampling function.

    Parameters
    ----------
    sampler_fn : callable
        Function that takes n (int) and returns Tensor shape (n, dim).
    n : int

    Returns
    -------
    x : Tensor
    """
    return sampler_fn(n)


# ======================================================================
# SDF primitives and CSG composition
# ======================================================================

def sdf_ball(radius: float = 1.0, center: Optional[Sequence[float]] = None) -> SDFT:
    """Exact SDF of a ball / disk: ``‖x − c‖ − r``.  Any dimension."""

    def psi(x: torch.Tensor) -> torch.Tensor:
        c = 0.0 if center is None else torch.as_tensor(
            center, device=x.device, dtype=x.dtype
        )
        return (x - c).norm(dim=1) - radius

    return psi


def sdf_disk(radius: float = 1.0, center: Optional[Sequence[float]] = None) -> SDFT:
    """Exact SDF of a 2-D disk.  Alias of :func:`sdf_ball` for readability."""
    return sdf_ball(radius, center)


def sdf_box(lo: Sequence[float], hi: Sequence[float]) -> SDFT:
    """Exact SDF of an axis-aligned box ``[lo, hi]``.  Any dimension.

    Uses the standard exact form ``‖max(q,0)‖ + min(max_k q_k, 0)`` with
    ``q = |x − c| − h``, which is correct both outside (distance to the nearest
    face/edge/corner) and inside (negative distance to the nearest face).
    """

    def psi(x: torch.Tensor) -> torch.Tensor:
        lo_t = torch.as_tensor(lo, device=x.device, dtype=x.dtype)
        hi_t = torch.as_tensor(hi, device=x.device, dtype=x.dtype)
        c, h = (lo_t + hi_t) / 2, (hi_t - lo_t) / 2
        q = (x - c).abs() - h
        outside = q.clamp_min(0.0).norm(dim=1)
        inside = q.max(dim=1).values.clamp_max(0.0)
        return outside + inside

    return psi


def sdf_annulus(
    r_inner: float = 0.5,
    r_outer: float = 1.0,
    center: Optional[Sequence[float]] = None,
) -> SDFT:
    """Exact SDF of an annulus / spherical shell -- a **multiply-connected** domain.

    ``max(r_in − ‖x−c‖, ‖x−c‖ − r_out)``.  The hole is the point: it exercises an
    interior boundary, where the outward normal points *toward* the centre.
    """

    def psi(x: torch.Tensor) -> torch.Tensor:
        c = 0.0 if center is None else torch.as_tensor(
            center, device=x.device, dtype=x.dtype
        )
        r = (x - c).norm(dim=1)
        return torch.maximum(r_inner - r, r - r_outer)

    return psi


def sdf_lshape(size: float = 1.0, cut: float = 0.5) -> SDFT:
    """L-shaped domain: the square ``[0, size]²`` minus the corner ``[cut, size]²``.

    The canonical **reentrant-corner** test.  The solution of a Poisson problem
    here has an ``r^{2/3}`` singularity at the inner corner, which is why the
    L-shape is the standard stress case for any method claiming to handle
    non-convex geometry.
    """
    outer = sdf_box([0.0, 0.0], [size, size])
    notch = sdf_box([cut, cut], [size * 2, size * 2])
    return sdf_difference(outer, notch)


def sdf_flower(
    radius: float = 1.0,
    amplitude: float = 0.3,
    petals: int = 5,
    center: Optional[Sequence[float]] = None,
) -> SDFT:
    """Flower / star domain ``r − R(1 + a cos(kθ))``, a smooth non-convex boundary.

    Deliberately **not** a true distance function: ``‖∇ψ‖ ≠ 1``, so it is the
    case that separates a correct projection from a naive one.  The unnormalised
    step ``x − ψ∇ψ`` mis-steps here, while
    :func:`project_to_boundary`'s ``x − ψ∇ψ/‖∇ψ‖²`` converges.  Requires
    ``amplitude < 1`` for the boundary to stay star-shaped about the centre.
    """
    if not 0.0 <= amplitude < 1.0:
        raise ValueError("sdf_flower: amplitude must lie in [0, 1)")

    def psi(x: torch.Tensor) -> torch.Tensor:
        c = 0.0 if center is None else torch.as_tensor(
            center, device=x.device, dtype=x.dtype
        )
        z = x - c
        r = z.norm(dim=1)
        theta = torch.atan2(z[:, 1], z[:, 0])
        return r - radius * (1.0 + amplitude * torch.cos(petals * theta))

    return psi


def sdf_polygon(vertices: Union[Sequence[Sequence[float]], torch.Tensor]) -> SDFT:
    """Exact SDF of a simple 2-D polygon given its vertices in order.

    Distance is the minimum over edge segments; the sign comes from a crossing
    (winding) test, so it is correct for non-convex polygons.  Orientation of the
    vertex list does not matter.

    This is the general escape hatch for a cross-section known only as a curve --
    measured, traced from CAD, or sampled from a parametrisation (see
    :func:`sdf_tokamak`).  Cost is ``O(M·K)`` for ``K`` edges, which is
    irrelevant next to the solve for any reasonable polygon.
    """
    V0 = torch.as_tensor(vertices, dtype=torch.get_default_dtype())
    if V0.dim() != 2 or V0.shape[1] != 2:
        raise ValueError("sdf_polygon: vertices must have shape (K, 2)")

    def psi(x: torch.Tensor) -> torch.Tensor:
        V = V0.to(device=x.device, dtype=x.dtype)
        Vi = V                                   # (K, 2) edge start
        Vj = torch.roll(V, 1, dims=0)            # (K, 2) edge end
        p = x.unsqueeze(1)                       # (M, 1, 2)
        e = (Vj - Vi).unsqueeze(0)               # (1, K, 2)
        w = p - Vi.unsqueeze(0)                  # (M, K, 2)
        t = ((w * e).sum(-1) / (e * e).sum(-1).clamp_min(1e-30)).clamp(0.0, 1.0)
        b = w - e * t.unsqueeze(-1)              # (M, K, 2)
        d2 = (b * b).sum(-1).min(dim=1).values   # (M,)

        c1 = p[..., 1] >= Vi[:, 1].unsqueeze(0)
        c2 = p[..., 1] < Vj[:, 1].unsqueeze(0)
        c3 = e[..., 0] * w[..., 1] > e[..., 1] * w[..., 0]
        flips = ((c1 & c2 & c3) | (~c1 & ~c2 & ~c3)).sum(dim=1)
        sign = torch.where(flips % 2 == 1, -1.0, 1.0).to(x.dtype)
        return sign * d2.clamp_min(1e-30).sqrt()

    return psi


def sdf_tokamak(
    R0: float = 1.0,
    a: float = 0.4,
    elongation: float = 1.8,
    triangularity: float = 0.4,
    n_points: int = 512,
) -> SDFT:
    """Tokamak poloidal cross-section -- the D-shaped Miller parametrisation.

    The boundary curve is::

        R(θ) = R₀ + a cos(θ + arcsin(δ)·sin θ)
        Z(θ) = κ a sin θ

    with elongation ``κ`` and triangularity ``δ``.  There is no closed-form
    implicit function for this curve, so it is discretised into an ``n_points``
    polygon and handed to :func:`sdf_polygon`, which is exact for the polygon and
    converges to the curve as ``O(1/n_points²)``.

    Defaults follow the MAST-U-scale geometry already used by
    ``examples/grad_shafranov.py`` (``R ∈ [0.6, 1.4]``), so this is a drop-in
    replacement for the rectangular domain used there.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    R = R0 + a * np.cos(theta + np.arcsin(triangularity) * np.sin(theta))
    Z = elongation * a * np.sin(theta)
    return sdf_polygon(np.stack([R, Z], axis=1))


# ----------------------------------------------------------------------
# CSG combinators
# ----------------------------------------------------------------------

def sdf_union(a: SDFT, b: SDFT) -> SDFT:
    """``A ∪ B`` -- pointwise ``min``."""
    return lambda x: torch.minimum(a(x).reshape(-1), b(x).reshape(-1))


def sdf_intersection(a: SDFT, b: SDFT) -> SDFT:
    """``A ∩ B`` -- pointwise ``max``."""
    return lambda x: torch.maximum(a(x).reshape(-1), b(x).reshape(-1))


def sdf_difference(a: SDFT, b: SDFT) -> SDFT:
    """``A \\ B`` -- ``max(ψ_A, −ψ_B)``."""
    return lambda x: torch.maximum(a(x).reshape(-1), -b(x).reshape(-1))


def sdf_complement(a: SDFT) -> SDFT:
    """Complement of ``A`` -- ``−ψ_A``."""
    return lambda x: -a(x).reshape(-1)


# ======================================================================
# SDFDomain: a domain you can sample, project onto, and take normals from
# ======================================================================

class SDFDomain:
    """A domain defined by a membership oracle ``ψ``, with a bounding box.

    Bundles the oracle with everything the solver needs from a geometry::

        >>> dom = SDFDomain.annulus(0.3, 1.0)
        >>> x   = dom.sample(4000)              # interior collocation
        >>> xb  = dom.sample_boundary(600)      # boundary collocation
        >>> nb  = dom.normal(xb)                # outward unit normals
        >>> B   = dom.neumann_rows(basis, xb)   # (M, N) block for ∂u/∂n = g

    Domains compose with set operations, so non-convex and multiply-connected
    geometry is built rather than meshed::

        >>> plate = SDFDomain.disk(1.0) - SDFDomain.disk(0.2, center=(0.4, 0.0))
        >>> both  = SDFDomain.disk(1.0) | SDFDomain.box([1.0, -0.3], [2.0, 0.3])

    Parameters
    ----------
    psi : callable
        ``(M, d) -> (M,)``, negative strictly inside.
    bounds : (lo, hi) or per-axis sequence of pairs
        Bounding box enclosing the domain.
    dim : int, optional
        Required only if ``bounds`` is a single pair.
    name : str, optional
        Label for ``repr``.

    Notes
    -----
    CSG results are valid *implicit functions* (correct sign everywhere) but not
    generally exact distance functions -- ``min``/``max`` of two exact SDFs
    over/under-estimates distance near the seam.  Nothing here depends on
    exactness: sampling uses only the sign, and
    :func:`project_to_boundary` normalises by ``‖∇ψ‖²``.  Only interpret ``ψ``
    itself as a true distance for the un-composed primitives.
    """

    def __init__(
        self,
        psi: SDFT,
        bounds: Union[Tuple[float, float], Sequence[Sequence[float]], torch.Tensor],
        dim: Optional[int] = None,
        name: Optional[str] = None,
    ):
        self.psi = psi
        self.name = name
        # Normalise once, at construction, so a malformed box fails here rather
        # than as a mysterious zero-acceptance hang inside the sampler.
        self._bounds_t = _as_bounds(bounds, dim=dim)

    # ------------------------------------------------------------------
    # Geometry queries
    # ------------------------------------------------------------------

    def bounds(self, device=None, dtype=None) -> torch.Tensor:
        """Bounding box as a ``(2, d)`` tensor ``[lo; hi]``."""
        return self._bounds_t.to(
            device=device or self._bounds_t.device,
            dtype=dtype or self._bounds_t.dtype,
        )

    @property
    def dim(self) -> int:
        return self.bounds().shape[1]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.psi(x).reshape(-1)

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        """Boolean mask ``ψ(x) < 0``, shape ``(M,)``."""
        return self(x) < 0

    def sample(self, n: int, **kw) -> torch.Tensor:
        """Uniform interior points -- see :func:`sample_sdf`."""
        return sample_sdf(self.psi, n, self._bounds_t, **kw)

    def sample_boundary(self, n: int, **kw) -> torch.Tensor:
        """Boundary points -- see :func:`sample_boundary_sdf` (note its distribution caveat)."""
        return sample_boundary_sdf(self.psi, n, self._bounds_t, **kw)

    def project(self, x: torch.Tensor, **kw) -> torch.Tensor:
        """Project onto ``{ψ = 0}`` -- see :func:`project_to_boundary`."""
        return project_to_boundary(self.psi, x, **kw)

    def normal(self, x: torch.Tensor, **kw) -> torch.Tensor:
        """Outward unit normals -- see :func:`outward_normal`."""
        return outward_normal(self.psi, x, **kw)

    # ------------------------------------------------------------------
    # Boundary operator blocks
    # ------------------------------------------------------------------

    def neumann_rows(self, basis, x_b: torch.Tensor, cache=None) -> torch.Tensor:
        """Design-matrix block for ``∂u/∂n`` on the boundary -> ``(M, N)``.

        Contracts the analytic basis gradient against the outward normal::

            ∂φ_j/∂n = n · ∇φ_j

        Exact, like every other operator here -- the normal is the only
        approximated quantity, and it is exact wherever ``ψ`` is differentiable.
        """
        n = self.normal(x_b)                              # (M, d)
        grad = basis.gradient(x_b, cache=cache)           # (M, d, N)
        return (n.unsqueeze(-1) * grad).sum(dim=1)

    def robin_rows(
        self, basis, x_b: torch.Tensor, alpha=1.0, beta=1.0, cache=None
    ) -> torch.Tensor:
        """Design-matrix block for the Robin combination ``α u + β ∂u/∂n`` -> ``(M, N)``.

        ``alpha`` and ``beta`` may be scalars or ``(M, 1)`` tensors for a
        spatially varying condition.
        """
        return alpha * basis.evaluate(x_b, cache=cache) + beta * self.neumann_rows(
            basis, x_b, cache=cache
        )

    # ------------------------------------------------------------------
    # CSG
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_bounds(a: SDFDomain, b: SDFDomain, mode: str) -> torch.Tensor:
        ba, bb = a.bounds(), b.bounds()
        if mode == "union":
            return torch.stack(
                [torch.minimum(ba[0], bb[0]), torch.maximum(ba[1], bb[1])]
            )
        if mode == "isect":
            # A n B is contained in BOTH boxes, so the tighter one is valid and
            # markedly cheaper to sample when the operands differ in size.
            lo = torch.maximum(ba[0], bb[0])
            hi = torch.minimum(ba[1], bb[1])
            if not (hi > lo).all():
                raise ValueError(
                    "SDFDomain intersection is empty: the operands' bounding "
                    f"boxes do not overlap (lo={lo.tolist()}, hi={hi.tolist()})."
                )
            return torch.stack([lo, hi])
        # A \ B is contained in A's box and cannot be tightened in general.
        return ba

    def __or__(self, other: SDFDomain) -> SDFDomain:
        return SDFDomain(
            sdf_union(self.psi, other.psi),
            self._merge_bounds(self, other, "union"),
            name=f"({self.name} | {other.name})",
        )

    def __and__(self, other: SDFDomain) -> SDFDomain:
        return SDFDomain(
            sdf_intersection(self.psi, other.psi),
            self._merge_bounds(self, other, "isect"),
            name=f"({self.name} & {other.name})",
        )

    def __sub__(self, other: SDFDomain) -> SDFDomain:
        return SDFDomain(
            sdf_difference(self.psi, other.psi),
            self._merge_bounds(self, other, "diff"),
            name=f"({self.name} - {other.name})",
        )

    # ------------------------------------------------------------------
    # Built-in domains
    # ------------------------------------------------------------------

    @classmethod
    def ball(cls, radius: float = 1.0, center: Optional[Sequence[float]] = None,
             dim: int = 2, pad: float = 0.0) -> SDFDomain:
        c = np.zeros(dim) if center is None else np.asarray(center, dtype=float)
        b = [(float(c[k] - radius - pad), float(c[k] + radius + pad)) for k in range(dim)]
        return cls(sdf_ball(radius, center), b, name=f"ball(r={radius})")

    @classmethod
    def disk(cls, radius: float = 1.0, center: Optional[Sequence[float]] = None) -> SDFDomain:
        """The unit disk of §2.7 by default."""
        return cls.ball(radius, center, dim=2)

    @classmethod
    def box(cls, lo: Sequence[float], hi: Sequence[float]) -> SDFDomain:
        return cls(sdf_box(lo, hi), list(zip(lo, hi)), name="box")

    @classmethod
    def annulus(cls, r_inner: float = 0.5, r_outer: float = 1.0,
                center: Optional[Sequence[float]] = None, dim: int = 2) -> SDFDomain:
        c = np.zeros(dim) if center is None else np.asarray(center, dtype=float)
        b = [(float(c[k] - r_outer), float(c[k] + r_outer)) for k in range(dim)]
        return cls(sdf_annulus(r_inner, r_outer, center), b,
                   name=f"annulus({r_inner}, {r_outer})")

    @classmethod
    def lshape(cls, size: float = 1.0, cut: float = 0.5) -> SDFDomain:
        return cls(sdf_lshape(size, cut), [(0.0, size), (0.0, size)],
                   name=f"lshape({size}, {cut})")

    @classmethod
    def flower(cls, radius: float = 1.0, amplitude: float = 0.3, petals: int = 5,
               center: Optional[Sequence[float]] = None) -> SDFDomain:
        c = np.zeros(2) if center is None else np.asarray(center, dtype=float)
        rmax = radius * (1.0 + amplitude)
        b = [(float(c[k] - rmax), float(c[k] + rmax)) for k in range(2)]
        return cls(sdf_flower(radius, amplitude, petals, center), b,
                   name=f"flower(k={petals})")

    @classmethod
    def tokamak(cls, R0: float = 1.0, a: float = 0.4, elongation: float = 1.8,
                triangularity: float = 0.4, n_points: int = 512) -> SDFDomain:
        """The tokamak poloidal cross-section of §2.7."""
        zmax = elongation * a
        return cls(
            sdf_tokamak(R0, a, elongation, triangularity, n_points),
            [(R0 - a * 1.05, R0 + a * 1.05), (-zmax * 1.05, zmax * 1.05)],
            name=f"tokamak(R0={R0}, a={a})",
        )

    def __repr__(self) -> str:
        return f"SDFDomain({self.name or 'psi'}, dim={self.dim})"


# ======================================================================
# Helper: get sampler by name
# ======================================================================

def get_sampler(name: str) -> Callable:
    """Get a sampler function by name.

    Parameters
    ----------
    name : str
        One of: 'box', 'ball', 'sphere', 'interval', 'boundary_box'.

    Returns
    -------
    sampler : callable
    """
    samplers = {
        "box": sample_box,
        "ball": sample_ball,
        "sphere": sample_sphere,
        "interval": sample_interval,
        "boundary_box": sample_boundary_box,
    }
    if name not in samplers:
        raise ValueError(f"Unknown sampler: {name}. Choose from {list(samplers.keys())}")
    return samplers[name]
