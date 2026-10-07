# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Stokes flow past a sphere: divergence-free sinusoids + Stokeslets, against the
exact solution.

    u = U - (3a/4)(U/r + x (U.x)/r^3) - (a^3/4)(U/r^3 - 3 x (U.x)/r^5)
    p = -(3/2) nu a (U.x) / r^3

The flow is mirror-symmetric in y, so only y >= 0 is collocated: the velocity
basis, the pressure basis and the singular columns are all paired with their
reflections.  The singular sources sit on an inner sphere of radius 0.7a,
deliberately *not* at the centre, where one Stokeslet and one doublet would
reproduce the answer trivially.  Without them (``--no-sources``) the same
sinusoids stall at tens of percent error: the field's continuation into the
sphere is singular, and a band-limited basis cannot imitate that cheaply.

    python examples/stokes_sphere_mfs.py
"""

import argparse
import math
import time

import torch

import fastlsq as fl

torch.set_default_dtype(torch.float64)

A = 0.5                                    # sphere radius
LO = torch.tensor([-2.0, 0.0, -2.0])       # y >= 0 half of the box
HI = torch.tensor([3.0, 2.0, 2.0])
NU = 1.0


def exact(x):
    r = x.norm(dim=1, keepdim=True)
    U = torch.tensor([1.0, 0.0, 0.0])
    ux = x[:, :1]
    u = U - 0.75 * A * (U / r + x * ux / r ** 3) - 0.25 * A ** 3 * (U / r ** 3 - 3 * x * ux / r ** 5)
    p = -1.5 * NU * A * ux[:, 0] / r[:, 0] ** 3
    return u, p


def hemisphere(n, radius, g, fibonacci=False):
    if fibonacci:
        i = torch.arange(2 * n) + 0.5
        phi = torch.acos(1 - 2 * i / (2 * n))
        th = math.pi * (1 + 5 ** 0.5) * i
        p = torch.stack([torch.cos(th) * torch.sin(phi), torch.sin(th) * torch.sin(phi), torch.cos(phi)], 1)
        return radius * p[p[:, 1] > 0.05]
    d = torch.randn(n, 3, generator=g)
    d[:, 1] = d[:, 1].abs()
    return radius * d / d.norm(dim=1, keepdim=True)


def fluid(n, g):
    """Uniform in the box outside the sphere, plus a shell of points near it."""
    x = LO + (HI - LO) * torch.rand(4 * n, 3, generator=g)
    x = x[x.norm(dim=1) > A][:n]
    m = n // 2
    r = A + 0.8 * torch.rand(m, 1, generator=g) ** 2
    return torch.cat([x, hemisphere(m, 1.0, g) * r])


def box_faces(n, g):
    pts = []
    for ax in range(3):
        for val in (LO[ax], HI[ax]):
            if ax == 1 and val == 0:
                continue                   # the symmetry plane needs no rows
            x = LO + (HI - LO) * torch.rand(n, 3, generator=g)
            x[:, ax] = val
            pts.append(x)
    return torch.cat(pts)


def main(N=600, n_src=300, M=6000, sources=True, seed=0, verbose=True):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed + 1)
    vel = fl.DivergenceFreeBasis.random(3, N, sigma=[1.0, 3.0], mirror_axis=1)
    pre = fl.MirrorBasis(fl.SinusoidalBasis.random(3, N // 2, sigma=2.0), axis=1, parity=+1)
    sing = (fl.StokesSingularities(hemisphere(n_src, 0.7 * A, g, fibonacci=True), nu=NU, mirror_axis=1)
            if sources else None)
    flow = fl.IncompressibleFlow(vel, pre, nu=NU, u_inf=[1.0, 0.0, 0.0], singular=sing)

    xi, xs, xb = fluid(M, g), hemisphere(2000, A, g), box_faces(400, g)
    t0 = time.time()
    res = fl.solve_navier_stokes(flow, xi, [fl.Dirichlet(xs, weight=10.0),
                                            fl.Dirichlet(xb, exact(xb)[0], weight=10.0)],
                                 newton=False)          # Stokes: one linear solve
    t_solve = time.time() - t0

    xt = fluid(4000, g)
    ue, pe = exact(xt)
    u, gu, p = flow.evaluate(xt, res.theta)
    err = float((u - ue).norm() / ue.norm())
    dp = p - pe
    perr = float((dp - dp.mean()).norm() / (pe - pe.mean()).norm())
    div = float((gu[:, 0, 0] + gu[:, 1, 1] + gu[:, 2, 2]).abs().max())
    slip = float(flow.evaluate(hemisphere(2000, A, g), res.theta)[0].norm(dim=1).max())

    # drag on the sphere from the traction, against Stokes' law 6 pi nu a U
    n_q = 4000
    xq = hemisphere(n_q, A, g)
    t = flow.traction(xq, xq / A, res.theta)
    drag = float(2 * t[:, 0].mean() * 2 * math.pi * A ** 2)       # both halves
    drag_exact = 6 * math.pi * NU * A
    if verbose:
        print(f"unknowns {flow.n_unknowns} ({'with' if sources else 'without'} singular columns), "
              f"solve {t_solve:.1f}s")
        print(f"  velocity rel. L2 error {err:.2e}   pressure {perr:.2e}")
        print(f"  max |div u| {div:.1e}   max no-slip error {slip:.1e}")
        print(f"  drag {drag:.5f} vs 6 pi nu a U = {drag_exact:.5f} "
              f"(Monte-Carlo quadrature, {abs(drag / drag_exact - 1):.1e})")
    return dict(err=err, perr=perr, div=div, slip=slip, drag=drag, drag_exact=drag_exact)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=600)
    ap.add_argument("--no-sources", action="store_true")
    a = ap.parse_args()
    main(N=a.N, sources=not a.no_sources)
