# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Solving on complex geometry from a membership oracle alone (no mesh).

Demonstrates the SDF path end to end on four domains of increasing difficulty:

    unit disk      smooth convex, everything known in closed form
    annulus        multiply-connected -- an interior boundary whose outward
                   normal points toward the centre
    L-shape        reentrant corner (the classic non-convex stress case)
    tokamak        D-shaped Miller poloidal cross-section, from a polygon SDF

For each we solve the Dirichlet problem  -Delta u = f  with a manufactured
solution, sampling interior and boundary points from the oracle only.  The last
column shows a Neumann solve on the disk, where the outward normal enters the
boundary operator directly.

Run:  python examples/sdf_domains.py
"""

import numpy as np
import torch

from fastlsq.basis import SinusoidalBasis
from fastlsq.geometry import SDFDomain
from fastlsq.linalg import solve_lstsq

torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------
# Manufactured solution: u = sin(a x) cos(b y),  -Delta u = (a^2+b^2) u
# ----------------------------------------------------------------------
A_K, B_K = 2.0, 1.5


def u_exact(p):
    return (torch.sin(A_K * p[:, 0]) * torch.cos(B_K * p[:, 1])).reshape(-1, 1)


def f_source(p):
    return (A_K ** 2 + B_K ** 2) * u_exact(p)


def solve_dirichlet(dom, n_features=800, sigma=6.0, n_pde=6000, n_bc=1200, w_bc=100.0):
    """One-shot Dirichlet solve on an SDF domain."""
    basis = SinusoidalBasis.random(2, n_features, sigma=sigma)
    x = dom.sample(n_pde)
    xb = dom.sample_boundary(n_bc)

    A = torch.cat([-basis.laplacian(x), w_bc * basis.evaluate(xb)])
    rhs = torch.cat([f_source(x), w_bc * u_exact(xb)])
    beta = solve_lstsq(A, rhs, mu=1e-10)

    xt = dom.sample(4000)
    err = basis.evaluate(xt) @ beta - u_exact(xt)
    return (torch.norm(err) / torch.norm(u_exact(xt))).item()


def solve_neumann(dom, n_features=800, sigma=6.0, n_pde=6000, n_bc=1200, w_bc=50.0):
    """Neumann solve: the outward normal enters the boundary block directly.

    Pure Neumann determines u only up to an additive constant, so the error is
    measured after removing the mean.
    """
    basis = SinusoidalBasis.random(2, n_features, sigma=sigma)
    x = dom.sample(n_pde)
    xb = dom.sample_boundary(n_bc)
    nb = dom.normal(xb)

    # g = n . grad u  for the manufactured u
    gx = A_K * torch.cos(A_K * xb[:, 0]) * torch.cos(B_K * xb[:, 1])
    gy = -B_K * torch.sin(A_K * xb[:, 0]) * torch.sin(B_K * xb[:, 1])
    g = (nb * torch.stack([gx, gy], dim=1)).sum(1, keepdim=True)

    A = torch.cat([-basis.laplacian(x), w_bc * dom.neumann_rows(basis, xb)])
    rhs = torch.cat([f_source(x), w_bc * g])
    beta = solve_lstsq(A, rhs, mu=1e-10)

    xt = dom.sample(4000)
    got = basis.evaluate(xt) @ beta
    ref = u_exact(xt)
    got, ref = got - got.mean(), ref - ref.mean()
    return (torch.norm(got - ref) / torch.norm(ref)).item()


def main():
    torch.manual_seed(0)

    domains = [
        ("unit disk", SDFDomain.disk(1.0)),
        ("annulus (0.3, 1.0)", SDFDomain.annulus(0.3, 1.0)),
        ("L-shape", SDFDomain.lshape(1.0, 0.5)),
        ("flower (5 petals)", SDFDomain.flower(1.0, 0.3, 5)),
        ("tokamak cross-section", SDFDomain.tokamak()),
    ]

    print("Dirichlet  -Delta u = f  on domains given only by a membership oracle")
    print("=" * 62)
    print(f"{'domain':<24} {'rel L2':>12}   {'boundary |psi|':>16}")
    print("-" * 62)
    for name, dom in domains:
        rel = solve_dirichlet(dom)
        resid = dom(dom.sample_boundary(500)).abs().max().item()
        print(f"{name:<24} {rel:12.3e}   {resid:16.2e}")

    print()
    print("Neumann  du/dn = g  (normals from grad(psi)/|grad psi|)")
    print("=" * 62)
    for name, dom in [("unit disk", SDFDomain.disk(1.0)),
                      ("tokamak cross-section", SDFDomain.tokamak())]:
        print(f"{name:<24} {solve_neumann(dom):12.3e}   (mean removed)")

    print()
    print("CSG: domains are built, not meshed")
    print("=" * 62)
    plate = SDFDomain.disk(1.0) - SDFDomain.disk(0.25, center=(0.4, 0.0))
    pts = plate.sample(4000)
    hole_d = (pts - torch.tensor([0.4, 0.0])).norm(dim=1)
    print(f"{'disk minus off-centre hole':<34} "
          f"min distance to hole centre = {hole_d.min():.4f} (radius 0.25)")


if __name__ == "__main__":
    main()
