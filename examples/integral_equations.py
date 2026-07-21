# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Integral and integro-differential equations through the standard harness.

Produces the integral-equation results table, in the same shape as the PDE table
and from the same `solve_linear` entry point -- these are `Problem` classes in
`fastlsq.problems`, not bespoke scripts.

Every reference here is a **closed form**, not a reference quadrature:

    Fredholm 2nd kind, K = xy        degenerate-kernel theory: u = f + λcx,
                                     c = ∫yf / (1 − λ/3)
    Fredholm 2nd kind, rank 2        the same, via a 2x2 reduced system
    Volterra 2nd kind                differentiate: u' − λu = f', u(0) = f(0)
    Integro-differential ODE         differentiate: u'' + λu = f'

Note the second-kind equations carry **no boundary rows at all** -- the identity
term makes them well posed on its own. Only the integro-differential ODE, which
has a genuine constant of integration, needs an initial condition.

Run:  PYTHONPATH=. python3 examples/integral_equations.py
"""

import time

import torch

from fastlsq.api import solve_linear
from fastlsq.basis import SinusoidalBasis
from fastlsq.problems import (
    FredholmProductKernel,
    FredholmRank2Kernel,
    IntegroDifferentialODE,
    VolterraSecondKind,
)

torch.set_default_dtype(torch.float64)

CFG = dict(scale=5.0, auto_scale=False, n_blocks=1, hidden_size=300,
           n_pde=2000, n_test=2000, verbose=False)


def main():
    problems = [
        FredholmProductKernel(lam=0.5),
        FredholmProductKernel(lam=2.0),
        FredholmRank2Kernel(lam=0.4),
        VolterraSecondKind(lam=1.5, omega=3.0),
        IntegroDifferentialODE(lam=4.0, u0=0.3),
    ]

    print("Integral equations -- one linear least squares each")
    print("=" * 78)
    print(f"{'problem':<32} {'rel L2':>11} {'grad rel L2':>13} {'bc rows':>8} {'time':>8}")
    print("-" * 78)

    for P in problems:
        torch.manual_seed(0)
        t0 = time.time()
        res = solve_linear(P, **CFG)
        dt = time.time() - t0
        m = res["metrics"]
        _, bcs, _ = P.get_train_data(n_pde=8, n_bc=1)
        n_bc_rows = sum(int(pts.shape[0]) for pts, _ in bcs)
        print(f"{P.name:<32} {m['val_err']:11.3e} {m['grad_err']:13.3e} "
              f"{n_bc_rows:8d} {dt:7.1f}s")

    # The kernel's characteristic values bound the usable lambda: at lambda = 3
    # the xy-kernel equation is singular, and nothing else reports that.
    print()
    print("Degenerate-kernel diagnostics")
    print("=" * 78)
    torch.manual_seed(0)
    basis = SinusoidalBasis.random(1, 200, sigma=5.0)
    P = FredholmProductKernel(lam=0.5)
    lams = P.characteristic_values(basis)
    print(f"  K(x,y) = xy on [0,1]: singular at lambda = "
          f"{lams.real.tolist()}  (analytic: [3.0])")
    print(f"  inner-product quadrature converged to "
          f"{P.kernel.check_quadrature(basis):.2e} under 2x refinement")

    print()
    print("Accuracy as lambda approaches the singular value")
    print("=" * 78)
    for lam in (0.5, 1.5, 2.5, 2.9, 2.99):
        torch.manual_seed(0)
        err = solve_linear(FredholmProductKernel(lam=lam), **CFG)["metrics"]["val_err"]
        print(f"  lambda = {lam:5.2f}   rel L2 = {err:.3e}")


if __name__ == "__main__":
    main()
