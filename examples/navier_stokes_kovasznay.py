# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Kovasznay flow: steady Navier-Stokes at Re = 40, solved by Newton's method.

    u = 1 - e^(lam x) cos(2 pi y),   v = lam / (2 pi) e^(lam x) sin(2 pi y),
    p = (1 - e^(2 lam x)) / 2,       lam = Re/2 - sqrt(Re^2/4 + 4 pi^2)

The velocity is a divergence-free sinusoidal basis, so continuity holds exactly
and only the momentum equation is collocated.  Each Newton step is one linear
least-squares solve (rows accumulated as normal equations); the first step is
Stokes flow, and convergence from there is quadratic.

    python examples/navier_stokes_kovasznay.py
"""

import math

import torch

import fastlsq as fl

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

RE = 40.0
LAM = RE / 2 - math.sqrt(RE ** 2 / 4 + 4 * math.pi ** 2)
LO, HI = torch.tensor([-0.5, -0.5]), torch.tensor([1.0, 1.5])


def exact(x):
    e = torch.exp(LAM * x[:, 0])
    c, s = torch.cos(2 * math.pi * x[:, 1]), torch.sin(2 * math.pi * x[:, 1])
    u = torch.stack([1 - e * c, LAM / (2 * math.pi) * e * s], 1)
    return u, 0.5 * (1 - torch.exp(2 * LAM * x[:, 0]))


def boundary(n):
    t = torch.rand(n, 1)
    edges = []
    for ax in range(2):
        for val in (LO[ax], HI[ax]):
            x = LO + (HI - LO) * t
            x[:, ax] = val
            edges.append(x)
    return torch.cat(edges)


def main(n_features=800, n_interior=5000, n_boundary=300):
    vel = fl.DivergenceFreeBasis.random(2, n_features, sigma=[1.0, 3.0, 6.0])
    pre = fl.SinusoidalBasis.random(2, n_features // 2, sigma=3.0)
    flow = fl.IncompressibleFlow(vel, pre, nu=1 / RE)

    x = LO + (HI - LO) * torch.rand(n_interior, 2)
    xb = boundary(n_boundary)
    x0 = torch.tensor([[1.0, 0.25]])                        # pins the pressure level
    res = fl.solve_navier_stokes(
        flow, x,
        [fl.Dirichlet(xb, exact(xb)[0], weight=10.0), fl.PressurePoint(x0, exact(x0)[1], weight=10.0)],
        max_iter=10, tol=1e-7, verbose=True)

    xt = LO + (HI - LO) * torch.rand(4000, 2)
    ue, pe = exact(xt)
    u, gu, p = flow.evaluate(xt, res.theta)
    print(f"unknowns {flow.n_unknowns}, converged {res.converged} in {len(res.history)} solves")
    print(f"velocity rel. L2 error {(u - ue).norm() / ue.norm():.2e}")
    print(f"pressure rel. L2 error {(p - pe).norm() / pe.norm():.2e}")
    print(f"max |div u|            {(gu[:, 0, 0] + gu[:, 1, 1]).abs().max():.1e}")
    print(f"median relative momentum residual {flow.residual(xt, res.theta)['relative'].median():.1e}")
    return res


if __name__ == "__main__":
    main()
