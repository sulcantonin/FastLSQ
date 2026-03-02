#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Inverse problem: Sparse-sensor localisation of hidden current coils in a
variable-permeability quadrupole magnet.

The forward model solves the 2-D magnetostatic equation

    ∇·(ν(x,y) ∇A_z) = −J(x,y),    (x,y) ∈ [−1,1]²,   A_z = 0 on ∂Ω

where ν = 1/μ_r and μ_r encodes a hyperbolic quadrupole iron geometry.
FastLSQ's SinusoidalBasis provides closed-form evaluation of both the
Laplacian (basis.laplacian) and gradient (basis.gradient), so the
variable-coefficient operator matrix is assembled analytically without
any automatic differentiation.

Adam with central-difference gradients then recovers the hidden coil
position (x_c, y_c) from 8 sparse magnetic-field sensor measurements.
"""

import os
import sys
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from fastlsq.basis import SinusoidalBasis
from fastlsq.utils import device

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_FEAT_PER_BLOCK = 500
SIGMAS = [2.0, 5.0, 8.0]
N_FEAT = N_FEAT_PER_BLOCK * len(SIGMAS)

M_INT = 6000
M_BC = 1500
LAM_BC = 200.0
MU_REG = 1e-8

SIGMA_COIL = 0.10
I_COIL = 35.0

TRUE_XC = 0.45
TRUE_YC = 0.45
INIT_XC = 0.2
INIT_YC = 0.8

N_SENSORS = 8
SENSOR_RADIUS = 0.25

ADAM_LR = 0.04
ADAM_ITERS = 40
FD_EPS = 1e-3
NOISE_LEVEL = 0.01

# Quadrupole iron geometry
MU_MAX = 50.0
R_APERTURE = 0.35
R_INNER_YOKE = 0.70
R_OUTER = 0.85
SIGMOID_K = 40.0


# ---------------------------------------------------------------------------
# Basis construction
# ---------------------------------------------------------------------------

def build_basis(n_per_block: int, sigmas: list) -> SinusoidalBasis:
    """Multi-block SinusoidalBasis with concatenated frequency scales."""
    Ws, bs = [], []
    for sigma in sigmas:
        blk = SinusoidalBasis.random(2, n_per_block, sigma=sigma)
        Ws.append(blk.W)
        bs.append(blk.b)
    W = torch.cat(Ws, dim=1)
    b = torch.cat(bs, dim=1)
    return SinusoidalBasis(W, b, normalize=True)


# ---------------------------------------------------------------------------
# Variable-permeability model (hyperbolic quadrupole iron geometry)
# ---------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))


def spatial_mu(pts: np.ndarray) -> np.ndarray:
    """Relative permeability μ_r(x,y) for a hyperbolic quadrupole magnet.

    Models smooth iron pole faces (hyperbola |x²−y²| = R_aperture²) and a
    solid outer return yoke, both blended via sigmoid transitions.
    """
    x, y = pts[:, 0], pts[:, 1]
    r = np.sqrt(x**2 + y**2)

    val = np.sqrt((x**2 - y**2)**2 + 1e-4)

    pole_mask = (_sigmoid(SIGMOID_K * (val - R_APERTURE**2))
                 * _sigmoid(SIGMOID_K * (R_OUTER - r)))
    yoke_mask = (_sigmoid(SIGMOID_K * (r - R_INNER_YOKE))
                 * _sigmoid(SIGMOID_K * (R_OUTER - r)))

    iron = np.maximum(pole_mask, yoke_mask)
    return 1.0 + (MU_MAX - 1.0) * iron


def nu_and_grad(pts: np.ndarray):
    """Return ν = 1/μ_r and its central-difference spatial gradient."""
    eps = 1e-4

    def nu(p):
        return 1.0 / spatial_mu(p)

    nu0 = nu(pts)

    px_p = pts.copy(); px_p[:, 0] += eps
    px_m = pts.copy(); px_m[:, 0] -= eps
    py_p = pts.copy(); py_p[:, 1] += eps
    py_m = pts.copy(); py_m[:, 1] -= eps

    dnu_dx = (nu(px_p) - nu(px_m)) / (2.0 * eps)
    dnu_dy = (nu(py_p) - nu(py_m)) / (2.0 * eps)
    return nu0, dnu_dx, dnu_dy


# ---------------------------------------------------------------------------
# Variable-μ PDE operator assembly
# ---------------------------------------------------------------------------

def build_variable_mu_operator(basis: SinusoidalBasis,
                                pts_int: torch.Tensor,
                                pts_bc: torch.Tensor):
    """Assemble and Cholesky-prefactor the variable-permeability system.

    Operator:  ν(x)·ΔA_z + ∇ν(x)·∇A_z = −J
    In matrix form: A_pde[i,j] = ν_i · Δφ_j(x_i) + ∇ν_i · ∇φ_j(x_i)

    Both Δφ_j and ∇φ_j are evaluated analytically via FastLSQ's closed-form
    derivative identity (eq. 2 of the paper).
    """
    pts_np = pts_int.cpu().numpy()
    nu0, dnu_dx, dnu_dy = nu_and_grad(pts_np)

    # Closed-form analytical matrices from SinusoidalBasis
    lap_t = basis.laplacian(pts_int)                  # (M_int, N)
    grad_t = basis.gradient(pts_int)                  # (M_int, 2, N)

    lap_np = lap_t.cpu().numpy()
    dphi_dx = grad_t[:, 0, :].cpu().numpy()           # (M_int, N)
    dphi_dy = grad_t[:, 1, :].cpu().numpy()           # (M_int, N)

    # Variable-coefficient PDE matrix (no autodiff required)
    A_pde = (nu0[:, None] * lap_np
             + dnu_dx[:, None] * dphi_dx
             + dnu_dy[:, None] * dphi_dy)             # (M_int, N)

    A_bc = LAM_BC * basis.evaluate(pts_bc).cpu().numpy()  # (M_bc, N)

    A_full = np.vstack([A_pde, A_bc])                 # (M_int + M_bc, N)
    At = A_full.T
    AtA = At @ A_full + MU_REG * np.eye(N_FEAT)
    cho = cho_factor(AtA)

    return cho, At, A_pde


# ---------------------------------------------------------------------------
# Current source
# ---------------------------------------------------------------------------

def current_source(pts: np.ndarray, xc: float, yc: float) -> np.ndarray:
    """Quadrupole-symmetric Gaussian current distribution."""
    x, y = pts[:, 0], pts[:, 1]

    def gauss(x0, y0):
        return np.exp(-((x - x0)**2 + (y - y0)**2) / (2.0 * SIGMA_COIL**2))

    return I_COIL * (gauss(xc, yc) + gauss(-xc, -yc)
                     - gauss(-xc, yc) - gauss(xc, -yc))


# ---------------------------------------------------------------------------
# Forward solve: A_z coefficients for given coil position
# ---------------------------------------------------------------------------

def solve_forward(cho, At: np.ndarray, A_pde: np.ndarray,
                  pts_int_np: np.ndarray,
                  xc: float, yc: float) -> np.ndarray:
    """Solve for spectral coefficients β given coil position."""
    J = current_source(pts_int_np, xc, yc)
    rhs_pde = -J
    rhs_bc = np.zeros(M_BC)
    rhs = np.concatenate([rhs_pde, rhs_bc])
    beta = cho_solve(cho, At @ rhs)
    return beta


# ---------------------------------------------------------------------------
# Field evaluation via analytical gradients
# ---------------------------------------------------------------------------

def evaluate_B(basis: SinusoidalBasis, beta: np.ndarray,
               pts: torch.Tensor):
    """Compute B = curl(A_z) = (∂A_z/∂y, −∂A_z/∂x) at given points."""
    beta_t = torch.tensor(beta, dtype=pts.dtype, device=device).unsqueeze(1)
    grad = basis.gradient(pts)                         # (M, 2, N)
    dAz_dx = (grad[:, 0, :] @ beta_t).cpu().numpy().ravel()
    dAz_dy = (grad[:, 1, :] @ beta_t).cpu().numpy().ravel()
    return dAz_dy, -dAz_dx  # Bx, By


# ---------------------------------------------------------------------------
# Sensor layout and noisy observations
# ---------------------------------------------------------------------------

def make_sensors(n: int = N_SENSORS, radius: float = SENSOR_RADIUS):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return radius * np.column_stack((np.cos(theta), np.sin(theta)))


# ---------------------------------------------------------------------------
# Inverse problem: recover (x_c, y_c) from sensor measurements
# ---------------------------------------------------------------------------

def inverse_solve(basis, cho, At, A_pde, pts_int_np, sensors, obs_Bx, obs_By):
    """Adam with central-difference gradients to recover coil location."""
    sensors_t = torch.tensor(sensors, dtype=torch.get_default_dtype(),
                             device=device)

    def loss_and_beta(xc, yc):
        beta = solve_forward(cho, At, A_pde, pts_int_np, xc, yc)
        Bx, By = evaluate_B(basis, beta, sensors_t)
        loss = float(np.mean((Bx - obs_Bx)**2 + (By - obs_By)**2))
        return loss, beta

    params = np.array([INIT_XC, INIT_YC], dtype=float)
    m, v = np.zeros(2), np.zeros(2)
    beta1, beta2 = 0.9, 0.999

    history = []
    beta_history = []

    for it in range(ADAM_ITERS):
        loss, beta_val = loss_and_beta(*params)
        history.append((float(params[0]), float(params[1]),
                        beta_val.copy(), loss))
        beta_history.append(beta_val.copy())

        grad = np.zeros(2)
        for j in range(2):
            p_plus, p_minus = params.copy(), params.copy()
            p_plus[j] += FD_EPS
            p_minus[j] -= FD_EPS
            grad[j] = (loss_and_beta(*p_plus)[0]
                       - loss_and_beta(*p_minus)[0]) / (2.0 * FD_EPS)

        t = it + 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        params -= ADAM_LR * (m / (1 - beta1**t)) / (
            np.sqrt(v / (1 - beta2**t)) + 1e-8)
        params = np.clip(params, 0.1, 0.9)

        if it % 10 == 0 or it == ADAM_ITERS - 1:
            print(f"  iter {it:3d}  loss={loss:.5f}  "
                  f"xc={params[0]:.4f}  yc={params[1]:.4f}  "
                  f"|g|={np.linalg.norm(grad):.3e}")

    return params[0], params[1], history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_grid(n: int = 100):
    xi = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(xi, xi)
    return X, Y, np.column_stack([X.ravel(), Y.ravel()])


def plot_results(basis, cho, At, A_pde, pts_int_np,
                 sensors, obs_Bx, obs_By,
                 xc_opt, yc_opt, history,
                 out_main=None,
                 out_conv=None):
    """Two-panel summary figure and convergence plot."""
    ng = 100
    X, Y, grid_pts = make_grid(ng)
    grid_t = torch.tensor(grid_pts, dtype=torch.get_default_dtype(),
                          device=device)

    mu_grid = spatial_mu(grid_pts).reshape(ng, ng)

    # Final reconstructed field
    beta_final = solve_forward(cho, At, A_pde, pts_int_np, xc_opt, yc_opt)
    Az_grid = (basis.evaluate(grid_t).cpu().numpy() @ beta_final).reshape(ng, ng)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Left: permeability map with sensors ---
    ax = axes[0]
    c = ax.pcolormesh(X, Y, mu_grid, cmap="YlOrRd", shading="gouraud")
    fig.colorbar(c, ax=ax, label=r"Relative permeability $\mu_r$")
    ax.add_patch(plt.Circle((0, 0), R_APERTURE, color="royalblue",
                             fill=False, linestyle="--", linewidth=1.5,
                             alpha=0.8, label="Aperture"))
    ax.plot(sensors[:, 0], sensors[:, 1], "gD", markersize=7,
            markeredgecolor="k", markeredgewidth=0.5, label="Sensors")
    ax.plot(TRUE_XC, TRUE_YC, "w*", markersize=14,
            markeredgecolor="k", label="True coil")
    signs = [(+1, +1), (-1, -1), (-1, +1), (+1, -1)]
    for sx, sy in signs:
        ax.plot(sx * TRUE_XC, sy * TRUE_YC, "w*", markersize=14,
                markeredgecolor="k")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
    ax.set_title(r"Hyperbolic quadrupole $\mu_r(x,y)$ and sensor layout")
    ax.legend(loc="upper right", fontsize=8)

    # --- Right: reconstructed A_z with coil comparison ---
    ax = axes[1]
    ax.pcolormesh(X, Y, Az_grid, cmap="RdBu_r", shading="gouraud",
                  vmin=-2.0, vmax=2.0)
    ax.contour(X, Y, Az_grid, levels=25, colors="k", alpha=0.45,
               linewidths=0.7)
    ax.contour(X, Y, mu_grid, levels=[25], colors="white",
               linewidths=1.5, alpha=0.7)

    ax.plot(sensors[:, 0], sensors[:, 1], "gD", markersize=7,
            markeredgecolor="k", markeredgewidth=0.5, label="Sensors")
    ax.plot(TRUE_XC, TRUE_YC, "w*", markersize=16, alpha=0.6,
            label="True hidden coil")
    ax.plot(xc_opt, yc_opt, "y*", markersize=14, markeredgecolor="k",
            markeredgewidth=0.7, label=f"Recovered ({xc_opt:.3f}, {yc_opt:.3f})")
    for sx, sy in signs:
        ax.plot(sx * TRUE_XC, sy * TRUE_YC, "w*", markersize=16, alpha=0.6)
        ax.plot(sx * xc_opt, sy * yc_opt, "y*", markersize=14,
                markeredgecolor="k", markeredgewidth=0.7)

    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
    ax.set_title(r"Recovered $A_z$ and coil location from 8 sparse sensors")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_main, dpi=200)
    plt.close(fig)
    print(f"Saved {out_main}")

    # --- Convergence plot ---
    losses = [h[3] for h in history]
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.semilogy(losses, "o-", markersize=4, color="steelblue")
    ax2.set_xlabel("Adam iteration")
    ax2.set_ylabel("Sensor MSE loss")
    ax2.set_title("Inverse coil localisation: optimisation convergence")
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_conv, dpi=150)
    plt.close(fig2)
    print(f"Saved {out_conv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"Device: {device}")
    print(f"Features: {N_FEAT} ({N_FEAT_PER_BLOCK} × {len(SIGMAS)} blocks, "
          f"sigma={SIGMAS})")

    # --- Build multi-block basis ---
    print("\n--- Building SinusoidalBasis ---")
    basis = build_basis(N_FEAT_PER_BLOCK, SIGMAS)

    # --- Sample domain ---
    pts_int = -1.0 + 2.0 * torch.rand(M_INT, 2, device=device)
    bc_x = -1.0 + 2.0 * torch.rand(M_BC, device=device)
    bc_y = torch.where(torch.rand(M_BC, device=device) > 0.5,
                       torch.ones(M_BC, device=device),
                       -torch.ones(M_BC, device=device))
    pts_bc = torch.stack([
        torch.cat([bc_x, bc_y]),
        torch.cat([bc_y, bc_x]),
    ], dim=1)[:M_BC]
    pts_int_np = pts_int.cpu().numpy()

    # --- Assemble variable-μ PDE system ---
    print("\n--- Assembling variable-μ PDE system (analytical operators) ---")
    cho, At, A_pde = build_variable_mu_operator(basis, pts_int, pts_bc)
    print(f"  Interior pts: {M_INT},  BC pts: {M_BC}")
    print(f"  System size: {N_FEAT} × {N_FEAT}")

    # --- Place sensors and generate noisy observations ---
    sensors = make_sensors()
    sensors_t = torch.tensor(sensors, dtype=torch.get_default_dtype(),
                             device=device)

    print(f"\n--- Forward solve (true hidden coil @ "
          f"({TRUE_XC}, {TRUE_YC})) ---")
    true_beta = solve_forward(cho, At, A_pde, pts_int_np, TRUE_XC, TRUE_YC)
    true_Bx, true_By = evaluate_B(basis, true_beta, sensors_t)

    np.random.seed(99)
    noise_x = NOISE_LEVEL * np.max(np.abs(true_Bx)) * np.random.randn(N_SENSORS)
    noise_y = NOISE_LEVEL * np.max(np.abs(true_By)) * np.random.randn(N_SENSORS)
    obs_Bx = true_Bx + noise_x
    obs_By = true_By + noise_y
    print(f"  {N_SENSORS} sensors  |  noise σ = {NOISE_LEVEL*100:.0f}% of peak |B|")

    # --- Inverse optimisation ---
    print(f"\n--- Adam inverse solve ({ADAM_ITERS} iters, lr={ADAM_LR}) ---")
    print(f"  Initial guess: ({INIT_XC}, {INIT_YC}), "
          f"True location: ({TRUE_XC}, {TRUE_YC})")
    xc_opt, yc_opt, history = inverse_solve(
        basis, cho, At, A_pde, pts_int_np, sensors, obs_Bx, obs_By
    )
    init_err = np.sqrt((INIT_XC - TRUE_XC)**2 + (INIT_YC - TRUE_YC)**2)
    final_err = np.sqrt((xc_opt - TRUE_XC)**2 + (yc_opt - TRUE_YC)**2)
    print(f"\n  Recovered coil: ({xc_opt:.4f}, {yc_opt:.4f})")
    print(f"  Initial distance to truth: {init_err:.4f}")
    print(f"  Final   distance to truth: {final_err:.4f}")
    print(f"  Loss reduction: {history[0][3]:.5f} → {history[-1][3]:.5f}")

    # --- Plot ---
    print("\n--- Generating figures ---")
    _here = os.path.dirname(os.path.abspath(__file__))
    _paper_i = os.path.join(_here, "..", "paper", "i")
    os.makedirs(_paper_i, exist_ok=True)
    plot_results(basis, cho, At, A_pde, pts_int_np,
                 sensors, obs_Bx, obs_By,
                 xc_opt, yc_opt, history,
                 out_main=os.path.join(_paper_i, "Inverse_Magnetostatics.pdf"),
                 out_conv=os.path.join(_paper_i, "Inverse_Magnetostatics_Convergence.pdf"))

    print("\nDone.")


if __name__ == "__main__":
    main()
