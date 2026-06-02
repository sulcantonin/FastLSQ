"""
Example: vector-valued FastLSQ for coupled  (psi, omega)  Stokes flow.
========================================================================

Demonstrates the 0.1.5 vector-valued features API by solving the 2-D
streamfunction-vorticity formulation of Stokes flow inside a unit
square with a manufactured solution.

The coupled system is

    nabla^2 psi  +  omega  =  0                 (definition: omega = -nabla^2 psi)
    nabla^2 omega           =  f                 (vorticity diffusion)

with Dirichlet boundary conditions  psi = psi_exact,  omega = omega_exact.

We use the manufactured pair

    psi_exact   =  sin(pi x) sin(pi y)
    omega_exact = 2 pi^2 sin(pi x) sin(pi y)

so that  nabla^2 psi = -2 pi^2 sin(pi x) sin(pi y) = -omega_exact (check)
and     nabla^2 omega = -4 pi^4 sin(pi x) sin(pi y) =: f.

We solve the 2-by-2 block system in a single LSQ shot using
`VectorFastLSQSolver`.

Run:
    python examples/vector_basis_stream_vorticity.py
"""

import numpy as np
import torch

from fastlsq import VectorFastLSQSolver, solve_lstsq
from fastlsq.utils import device


# ----------------------------------------------------------------------
# Manufactured solution
# ----------------------------------------------------------------------

def psi_exact(x):
    return torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2])


def omega_exact(x):
    return 2 * np.pi ** 2 * psi_exact(x)


def f_source(x):
    return -4 * np.pi ** 4 * psi_exact(x)


# ----------------------------------------------------------------------
# Solve
# ----------------------------------------------------------------------

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # --- 1.  Two-component solver  (psi, omega) ---
    solver = VectorFastLSQSolver(input_dim=2, n_components=2, normalize=True)
    # Same bandwidth for both components -- a list works if you want
    # different bandwidths per component.
    for _ in range(3):
        solver.add_block(hidden_size=400, scale=4.0)
    basis = solver.basis
    print(f"VectorFastLSQSolver:  K = {basis.n_components}, "
          f"N_per_component = {basis.n_features_per_component}, "
          f"N_total = {basis.n_features_total}")

    # --- 2.  Collocation points ---
    n_pde = 4000
    n_bc  = 200
    x_pde = torch.rand(n_pde, 2, device=device)
    rb = torch.rand(n_bc, 1, device=device)
    zb = torch.zeros(n_bc, 1, device=device); ob = torch.ones(n_bc, 1, device=device)
    x_bc = torch.cat([
        torch.cat([zb, rb], 1), torch.cat([ob, rb], 1),
        torch.cat([rb, zb], 1), torch.cat([rb, ob], 1),
    ], 0)

    # --- 3.  Block assembly ---
    #
    # The global unknown is  q = [beta_psi ; beta_omega]  (stacked).
    #
    # PDE row 1:   (nabla^2 phi_psi)  beta_psi  +  phi_omega  beta_omega   =  0
    # PDE row 2:   (nabla^2 phi_omega) beta_omega                          =  f
    #
    # We build each row as a SINGLE wide matrix of shape (M, 2*N) by
    # h-concatenating per-component blocks.  Independent BC rows for
    # psi and omega become block-diagonal.

    b_psi = basis.component(0)
    b_omg = basis.component(1)
    M, N = x_pde.shape[0], basis.n_features_per_component
    Z_MN = torch.zeros(M, N, device=device)

    # PDE row 1
    L1_psi = b_psi.laplacian(x_pde)         # (M, N)
    H1_omg = b_omg.evaluate(x_pde)
    A_pde1 = torch.cat([L1_psi, H1_omg], dim=1)               # (M, 2N)
    b_pde1 = torch.zeros(M, 1, device=device)

    # PDE row 2:  no psi term
    L2_omg = b_omg.laplacian(x_pde)
    A_pde2 = torch.cat([Z_MN, L2_omg], dim=1)
    b_pde2 = f_source(x_pde)

    # Boundary rows: psi = psi_exact on dOmega,  omega = omega_exact on dOmega
    H_bc_psi = b_psi.evaluate(x_bc)
    H_bc_omg = b_omg.evaluate(x_bc)
    nbc = x_bc.shape[0]
    Z_bcN = torch.zeros(nbc, N, device=device)
    A_bc_psi = torch.cat([H_bc_psi, Z_bcN], dim=1)
    A_bc_omg = torch.cat([Z_bcN, H_bc_omg], dim=1)

    w_bc = 100.0
    A = torch.cat([A_pde1, A_pde2,
                   w_bc * A_bc_psi, w_bc * A_bc_omg], 0)
    b = torch.cat([b_pde1, b_pde2,
                   w_bc * psi_exact(x_bc),
                   w_bc * omega_exact(x_bc)], 0)
    print(f"system: {A.shape[0]} eqns x {A.shape[1]} features")

    # --- 4.  One linear LSQ over the WHOLE coupled system ---
    beta_stack = solve_lstsq(A, b, mu=1e-8)
    # Unpack stacked beta into per-component vectors
    betas = basis.unstack_beta(beta_stack)
    solver.beta = betas

    # --- 5.  Error check on a fresh grid ---
    n_test = 5000
    x_test = torch.rand(n_test, 2, device=device)
    pred = solver.predict(x_test)              # (M, 2)
    psi_p, omg_p = pred[:, 0:1], pred[:, 1:2]
    psi_e, omg_e = psi_exact(x_test), omega_exact(x_test)
    err_psi = (psi_p - psi_e).norm() / psi_e.norm()
    err_omg = (omg_p - omg_e).norm() / omg_e.norm()
    print(f"L2 relative error  psi   = {err_psi:.2e}")
    print(f"L2 relative error  omega = {err_omg:.2e}")


if __name__ == "__main__":
    main()
