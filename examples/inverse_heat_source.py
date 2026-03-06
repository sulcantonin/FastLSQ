#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Inverse problem: locating 4 anisotropic Gaussian heat sources.

The forward model solves the space-time heat equation

    u_t - α Δu = Σ_k f_k(x, y; θ_k),   (x,y) ∈ [0,1]², t ∈ [0, T]
    u = 0 on ∂Ω,   u(x,y,0) = 0

for the full 3-D field u(x, y, t).  The sources are steady (time-independent)
anisotropic Gaussians; the temperature field evolves from zero and grows as
heat diffuses outward.  Each sensor is a fixed spatial point that records a
temperature time-series: with 4 sensors × 60 snapshots the problem has
240 observations to constrain 24 unknown source parameters.

Each source f_k is parameterised by θ_k = (x_s, y_s, I, a, b, c) where
the precision matrix is P = [[a,0],[b,c]]^T [[a,0],[b,c]] (anisotropic
Gaussian).  FastLSQ's sinusoidal basis provides closed-form ∂φ/∂t and Δφ,
so the operator matrix A is assembled once and Cholesky-prefactored.  Each
L-BFGS-B function evaluation is then two triangular back-substitutions.
"""

import os
import sys
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Problem configuration
# ---------------------------------------------------------------------------

ALPHA    = 0.05
T_FINAL  = 1.5
N_SOURCES = 4

# Accuracy / iterations
LBFGS_MAXITER = 3000      # more iterations for finer convergence
LBFGS_FTOL    = 1e-16     # function tolerance
LBFGS_GTOL    = 1e-12     # gradient tolerance

# Solver resolution (higher → more accurate forward model; N_FEAT divisible by 3)
N_FEAT   = 2100           # more features for finer field
M_INT    = 10000          # more PDE collocation points
M_BC     = 3000           # boundary collocation
M_IC     = 1500           # initial condition collocation

# Animation output
ANIM_FPS        = 8       # frames per second
ANIM_DPI        = 90      # resolution for GIF
ANIM_FRAME_SKIP = 10      # show every Nth frame

# 4 sources at the four quadrants — each (xs, ys, I, a, b, c)
# b=0: axis-aligned anisotropic Gaussians (more identifiable from sensors)
TRUE_PARAMS = np.array([
    0.22, 0.22,  4.0, 14.0,  0.0,  9.0,   # bottom-left, strong, wide
    0.78, 0.25,  2.5,  8.0,  0.0, 13.0,   # bottom-right, medium, tall
    0.75, 0.75,  3.5, 11.0,  0.0, 11.0,   # top-right, strong, near-circular
    0.25, 0.78,  2.0,  9.0,  0.0,  6.0,   # top-left, weak, elongated
])

# Initial guess: wrong positions (~0.15–0.2 away), wrong intensities, isotropic
INIT_PARAMS = np.array([
    0.42, 0.40,  2.0,  8.0,  0.0,  8.0,   # bottom-left quadrant, off
    0.62, 0.40,  2.0,  8.0,  0.0,  8.0,   # bottom-right quadrant, off
    0.58, 0.60,  2.0,  8.0,  0.0,  8.0,   # top-right quadrant, off
    0.42, 0.60,  2.0,  8.0,  0.0,  8.0,   # top-left quadrant, off
])

# Per-source L-BFGS-B bounds: xs,ys ∈ (0.05,0.95), I>0, a,c>0, b free
_SRC_BOUNDS = [(0.05, 0.95), (0.05, 0.95),
               (0.1, 15.0), (0.1, 50.0), (-50.0, 50.0), (0.1, 50.0)]
BOUNDS = _SRC_BOUNDS * N_SOURCES

SOURCE_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
SOURCE_LABELS = ["Src 1", "Src 2", "Src 3", "Src 4"]

# 4 sensors placed irregularly — one per quadrant so every source has a nearby
# observer, but positions don't coincide with any source centre and are
# deliberately offset (as if placed by a technician at accessible points).
SENSORS = np.array([
    [0.12, 0.15],   # bottom-left quadrant  (Source 1 is at 0.22, 0.22)
    [0.88, 0.18],   # bottom-right quadrant (Source 2 is at 0.78, 0.25)
    [0.82, 0.85],   # top-right quadrant    (Source 3 is at 0.75, 0.75)
    [0.18, 0.92],   # top-left quadrant     (Source 4 is at 0.25, 0.78)
])
SENSOR_TIMES = np.linspace(0.05, T_FINAL, 60)
NOISE_LEVEL  = 0.002


# ---------------------------------------------------------------------------
# FastLSQ Heat Solver (pure NumPy, self-contained)
# ---------------------------------------------------------------------------

class FastLSQHeatSolver:
    """Pre-factored heat-equation solver using sinusoidal random features.

    The operator matrix A (heat equation + BCs + IC) is assembled once
    from closed-form derivatives of the sinusoidal basis and then
    Cholesky-factored.  Subsequent forward solves reduce to a single
    triangular back-substitution, making them trivially embeddable in
    an outer optimization loop.
    """

    def __init__(self, alpha=ALPHA, t_final=T_FINAL,
                 n_feat=N_FEAT, n_blocks=3, sigmas=(2.0, 5.0, 9.0),
                 m_int=M_INT, m_bc=M_BC, m_ic=M_IC,
                 lam_bc=200.0, lam_ic=200.0, mu=1e-6):
        self.alpha   = alpha
        self.t_final = t_final
        self.lam_bc  = lam_bc
        self.lam_ic  = lam_ic
        self.n_feat  = n_feat
        self.mu      = mu

        # Random Fourier features: W has shape (n_feat, 3) = (wx, wy, wt)
        np.random.seed(42)
        n_per = n_feat // n_blocks
        self.W  = np.vstack([np.random.randn(n_per, 3) * s for s in sigmas])
        self.bv = np.concatenate([np.random.uniform(0, 2 * np.pi, n_per)
                                  for _ in sigmas])
        self._isN = 1.0 / np.sqrt(n_feat)   # 1/√N normalization

        # Collocation points
        np.random.seed(100)
        pts = np.random.rand(m_int, 3)
        pts[:, 2] *= t_final
        self.pts_int = pts

        bc = []
        for _ in range(m_bc // 4):
            t, s = np.random.rand() * t_final, np.random.rand()
            bc += [[0, s, t], [1, s, t], [s, 0, t], [s, 1, t]]
        self.pts_bc = np.array(bc[:m_bc])
        self.pts_ic = np.hstack([np.random.rand(m_ic, 2), np.zeros((m_ic, 1))])

        self._build_system()

    # ------------------------------------------------------------------
    # Basis and operator

    def _phi(self, pts):
        """Evaluate basis φ_j(x) = sin(W_j · x + b_j) / √N."""
        return np.sin(pts @ self.W.T + self.bv) * self._isN

    def _Lphi(self, pts):
        """Closed-form heat operator L[φ] = ∂φ/∂t − α Δ_xy φ."""
        z   = pts @ self.W.T + self.bv
        lap = self.W[:, 0]**2 + self.W[:, 1]**2          # |ω_xy|²
        return (self.W[:, 2] * np.cos(z)
                + self.alpha * lap * np.sin(z)) * self._isN

    def _build_system(self):
        A = np.vstack([
            self._Lphi(self.pts_int),
            self.lam_bc * self._phi(self.pts_bc),
            self.lam_ic * self._phi(self.pts_ic),
        ])
        self.A  = A
        self.At = A.T
        self.cho = cho_factor(self.At @ A + self.mu * np.eye(self.n_feat))
        self._rhs_bc = np.zeros(len(self.pts_bc))
        self._rhs_ic = np.zeros(len(self.pts_ic))

    # ------------------------------------------------------------------
    # Single anisotropic Gaussian source + analytical gradients

    @staticmethod
    def _one_source(xy, xs, ys, I, a, b, c):
        """Evaluate f_k and its gradients w.r.t. (xs, ys, I, a, b, c)."""
        dx, dy = xy[:, 0] - xs, xy[:, 1] - ys
        u1     = a * dx
        u2     = b * dx + c * dy
        Q      = u1**2 + u2**2
        K      = (a * c / (2 * np.pi)) * np.exp(-0.5 * Q)
        f      = I * K
        # gradients
        dxs = f * (a * u1 + b * u2)
        dys = f * (c * u2)
        dI  = K
        da  = f * (1.0 / a - u1 * dx)
        db  = f * (-u2 * dx)
        dc  = f * (1.0 / c - u2 * dy)
        return f, np.column_stack([dxs, dys, dI, da, db, dc])

    def source_sum(self, xy, params):
        """Sum of N_SOURCES anisotropic Gaussians; returns (f, df/dparams)."""
        n_pts = xy.shape[0]
        f_tot  = np.zeros(n_pts)
        df_tot = np.zeros((n_pts, 6 * N_SOURCES))
        for k in range(N_SOURCES):
            p = params[k * 6:(k + 1) * 6]
            fk, gk = self._one_source(xy, *p)
            f_tot += fk
            df_tot[:, k * 6:(k + 1) * 6] = gk
        return f_tot, df_tot

    # ------------------------------------------------------------------
    # Forward solve and loss / gradient

    def forward_solve(self, params):
        """PDE solve → (beta, predict_fn) for given source parameters."""
        f, _ = self.source_sum(self.pts_int[:, :2], params)
        rhs   = np.concatenate([f,
                                 self.lam_bc * self._rhs_bc,
                                 self.lam_ic * self._rhs_ic])
        beta  = cho_solve(self.cho, self.At @ rhs)
        return beta

    def predict(self, beta, pts):
        return self._phi(pts) @ beta

    def loss_and_grad(self, params, sensors, sensor_times, s_obs):
        """Sensor MSE loss and its analytical gradient w.r.t. all 24 params."""
        f, df_dp = self.source_sum(self.pts_int[:, :2], params)
        rhs  = np.concatenate([f,
                                self.lam_bc * self._rhs_bc,
                                self.lam_ic * self._rhs_ic])
        beta = cho_solve(self.cho, self.At @ rhs)

        K_val    = len(sensor_times) * len(sensors)
        loss     = 0.0
        g_beta   = np.zeros(self.n_feat)

        for ti, t in enumerate(sensor_times):
            pts_s = np.hstack([sensors, np.full((len(sensors), 1), t)])
            H     = self._phi(pts_s)
            resid = H @ beta - s_obs[ti]
            loss += np.sum(resid**2)
            g_beta += H.T @ resid

        loss   /= K_val
        g_beta *= 2.0 / K_val

        # Backprop through the Cholesky solve: dL/dp = df/dp^T · A^T · (AAt)^{-1} · g
        w      = self.A @ cho_solve(self.cho, g_beta)
        w_pde  = w[:len(self.pts_int)]
        grads  = w_pde @ df_dp
        return loss, grads


# ---------------------------------------------------------------------------
# Run, animate, and save results
# ---------------------------------------------------------------------------

def run(noise_level=NOISE_LEVEL):
    print("=== Inverse Heat Source Localisation: 4 Anisotropic Sources ===")

    solver = FastLSQHeatSolver()

    # ---- Synthetic observations ----------------------------------------
    print("  Generating synthetic sensor data …")
    beta_true = solver.forward_solve(TRUE_PARAMS)
    s_clean   = np.array([
        solver.predict(beta_true,
                       np.hstack([SENSORS, np.full((len(SENSORS), 1), t)]))
        for t in SENSOR_TIMES
    ])
    np.random.seed(77)
    s_obs = s_clean + noise_level * np.random.randn(*s_clean.shape)

    # ---- L-BFGS-B optimization -----------------------------------------
    print("  Running L-BFGS-B (24 parameters) …")
    history = [INIT_PARAMS.copy()]

    def callback(xk):
        history.append(xk.copy())

    result = minimize(
        lambda p: solver.loss_and_grad(p, SENSORS, SENSOR_TIMES, s_obs),
        INIT_PARAMS, method="L-BFGS-B", jac=True,
        bounds=BOUNDS, callback=callback,
        options={"maxiter": LBFGS_MAXITER, "ftol": LBFGS_FTOL, "gtol": LBFGS_GTOL},
    )
    print(f"  Converged: {result.success}  iters={result.nit}  "
          f"loss={result.fun:.3e}")

    opt_params = result.x
    beta_opt   = solver.forward_solve(opt_params)

    # ---- Summary -------------------------------------------------------
    print("\n  Source recovery summary:")
    print(f"  {'':20s} {'True':>8s}  {'Init':>8s}  {'Recovered':>10s}  {'|Error|':>10s}")
    labels = ["xs", "ys", "I", "a", "b", "c"]
    for k in range(N_SOURCES):
        print(f"  --- Source {k+1} ---")
        for j, lbl in enumerate(labels):
            idx  = k * 6 + j
            tr   = TRUE_PARAMS[idx]
            ini  = INIT_PARAMS[idx]
            rec  = opt_params[idx]
            print(f"  {lbl:20s} {tr:8.4f}  {ini:8.4f}  {rec:10.4f}  "
                  f"{abs(rec-tr):10.2e}")

    return solver, beta_true, beta_opt, opt_params, np.array(history), result


def make_figures(solver, beta_true, beta_opt, opt_params, history, result):
    """Save paper PDF figure and optimization GIF."""
    _here   = os.path.dirname(os.path.abspath(__file__))
    _paper  = os.path.join(_here, "..", "paper", "i")
    os.makedirs(_paper, exist_ok=True)

    # Grid for field evaluation
    nx = 100
    xg = np.linspace(0, 1, nx)
    XX, YY = np.meshgrid(xg, xg)
    xy_grid = np.column_stack([XX.ravel(), YY.ravel()])
    t_snap  = 0.9
    pts_snap = np.hstack([xy_grid, np.full((len(xy_grid), 1), t_snap)])

    u_true = solver.predict(beta_true, pts_snap).reshape(nx, nx)
    u_opt  = solver.predict(beta_opt,  pts_snap).reshape(nx, nx)
    vmax   = max(u_true.max(), u_opt.max())

    # ----------------------------------------------------------------
    # Static paper figure: left=true field, centre=recovered, right=paths
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle(
        f"Inverse Heat Source Localisation — 4 Anisotropic Sources  "
        f"(noise={NOISE_LEVEL:.3f}, {result.nit} L-BFGS-B iters)",
        fontsize=11)

    for ax, field, title in zip(
            axes[:2],
            [u_true, u_opt],
            [f"True temperature  $t={t_snap}$",
             f"Recovered temperature  $t={t_snap}$"]):
        im = ax.pcolormesh(XX, YY, field, cmap="inferno",
                           shading="gouraud", vmin=0, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="T")
        ax.plot(SENSORS[:, 0], SENSORS[:, 1], "ws", markersize=5,
                markeredgecolor="k", markeredgewidth=0.6, label="Sensors")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)

    # Mark true centres on both panels
    for k in range(N_SOURCES):
        xs, ys = TRUE_PARAMS[k * 6], TRUE_PARAMS[k * 6 + 1]
        axes[0].plot(xs, ys, "*", color=SOURCE_COLORS[k], markersize=14,
                     markeredgecolor="k", markeredgewidth=0.6,
                     label=SOURCE_LABELS[k])
    axes[0].legend(loc="upper right", fontsize=7, framealpha=0.8)

    # Mark recovered and true centres on second panel
    for k in range(N_SOURCES):
        xs_t, ys_t = TRUE_PARAMS[k * 6],   TRUE_PARAMS[k * 6 + 1]
        xs_r, ys_r = opt_params[k * 6],    opt_params[k * 6 + 1]
        c = SOURCE_COLORS[k]
        axes[1].plot(xs_t, ys_t, "*", color=c, markersize=14,
                     markeredgecolor="k", markeredgewidth=0.6, alpha=0.5)
        axes[1].plot(xs_r, ys_r, "o", color=c, markersize=9,
                     markeredgecolor="k", markeredgewidth=0.7)

    # Optimization trajectories
    ax3 = axes[2]
    traj = history                          # shape (n_iters+1, 24)
    for k in range(N_SOURCES):
        xs_t  = TRUE_PARAMS[k * 6]
        ys_t  = TRUE_PARAMS[k * 6 + 1]
        xs_tr = traj[:, k * 6]
        ys_tr = traj[:, k * 6 + 1]
        c     = SOURCE_COLORS[k]
        ax3.plot(xs_tr, ys_tr, "-", color=c, linewidth=1.5,
                 alpha=0.85, label=SOURCE_LABELS[k])
        ax3.plot(xs_tr[0],  ys_tr[0],  "^", color=c, markersize=9,
                 markeredgecolor="k", markeredgewidth=0.6)
        ax3.plot(xs_tr[-1], ys_tr[-1], "o", color=c, markersize=9,
                 markeredgecolor="k", markeredgewidth=0.7)
        ax3.plot(xs_t, ys_t, "*", color=c, markersize=14,
                 markeredgecolor="k", markeredgewidth=0.6, alpha=0.4)

    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.set_aspect("equal")
    ax3.set_title("L-BFGS-B source trajectories\n"
                  "(▲ start, ● recovered, ★ true)", fontsize=10)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.4)

    pdf_path = os.path.join(_paper, "Inverse_Heat_Source.pdf")
    fig.tight_layout()
    fig.savefig(pdf_path, dpi=150)
    plt.close(fig)
    print(f"  Saved paper figure → {pdf_path}")

    # ----------------------------------------------------------------
    # GIF animation: recovered field, trajectories, distance to ground truth
    # ----------------------------------------------------------------
    gif_path = os.path.join(_here, "..", "misc", "inverse_heat_source.gif")
    frame_indices = np.unique(np.concatenate([
        np.arange(0, len(history), ANIM_FRAME_SKIP),
        [len(history) - 1],
    ]))
    print(f"  Rendering animation ({len(frame_indices)} frames, every {ANIM_FRAME_SKIP}th) …")

    # Precompute distances from ground truth
    dists = np.zeros((len(history), N_SOURCES))
    for i, params in enumerate(history):
        for k in range(N_SOURCES):
            dx = params[k * 6] - TRUE_PARAMS[k * 6]
            dy = params[k * 6 + 1] - TRUE_PARAMS[k * 6 + 1]
            dists[i, k] = np.sqrt(dx**2 + dy**2)

    fig2, (ax_field, ax_traj, ax_dist) = plt.subplots(1, 3, figsize=(14, 4.5))
    fig2.subplots_adjust(wspace=0.22, top=0.88)
    fig2.suptitle("FastLSQ — Inverse Heat Source Localisation (4 sources)",
                  fontsize=12)

    def update(idx):
        ax_field.clear()
        ax_traj.clear()
        ax_dist.clear()
        frame = frame_indices[idx]
        params = history[frame]
        beta_f = solver.forward_solve(params)
        u_f    = solver.predict(beta_f, pts_snap).reshape(nx, nx)

        # Panel 1: Recovered temperature field
        ax_field.pcolormesh(XX, YY, u_f, cmap="inferno", shading="gouraud",
                            vmin=0, vmax=vmax)
        ax_field.plot(SENSORS[:, 0], SENSORS[:, 1], "ws", markersize=5,
                      markeredgecolor="k")
        for k in range(N_SOURCES):
            xs_t = TRUE_PARAMS[k * 6];  ys_t = TRUE_PARAMS[k * 6 + 1]
            ax_field.plot(params[k * 6], params[k * 6 + 1], "o", color=SOURCE_COLORS[k],
                          markersize=9, markeredgecolor="k")
            ax_field.plot(xs_t, ys_t, "*", color=SOURCE_COLORS[k], markersize=12,
                          markeredgecolor="k", alpha=0.35)
        ax_field.set_xlim(0, 1); ax_field.set_ylim(0, 1); ax_field.set_aspect("equal")
        ax_field.set_title(f"Recovered field  t={t_snap}", fontsize=10)
        ax_field.grid(True, linestyle="--", alpha=0.35)

        # Panel 2: L-BFGS-B source trajectories
        for k in range(N_SOURCES):
            xs_t = TRUE_PARAMS[k * 6];  ys_t = TRUE_PARAMS[k * 6 + 1]
            xs_c = history[:frame + 1, k * 6]
            ys_c = history[:frame + 1, k * 6 + 1]
            c    = SOURCE_COLORS[k]
            ax_traj.plot(xs_c, ys_c, "-", color=c, linewidth=1.5, alpha=0.7,
                         label=SOURCE_LABELS[k])
            ax_traj.plot(xs_c[0], ys_c[0], "^", color=c, markersize=9,
                         markeredgecolor="k")
            ax_traj.plot(params[k * 6], params[k * 6 + 1], "o", color=c,
                         markersize=9, markeredgecolor="k")
            ax_traj.plot(xs_t, ys_t, "*", color=c, markersize=12,
                         markeredgecolor="k", alpha=0.4)
        ax_traj.set_xlim(0, 1); ax_traj.set_ylim(0, 1); ax_traj.set_aspect("equal")
        ax_traj.set_title("Source trajectories (▲ start, ● current, ★ true)", fontsize=10)
        ax_traj.legend(loc="upper right", fontsize=7)
        ax_traj.grid(True, linestyle="--", alpha=0.35)

        # Panel 3: Distance from estimated to ground truth
        iters = np.arange(frame + 1)
        for k in range(N_SOURCES):
            ax_dist.plot(iters, dists[:frame + 1, k], color=SOURCE_COLORS[k],
                         linewidth=1.5, label=SOURCE_LABELS[k])
        ax_dist.set_xlim(0, len(history) - 1)
        ax_dist.set_ylim(0, dists.max() * 1.05)
        ax_dist.set_xlabel("Iteration", fontsize=9)
        ax_dist.set_ylabel("Distance to ground truth", fontsize=9)
        ax_dist.set_title("Source position error", fontsize=10)
        ax_dist.legend(loc="upper right", fontsize=7)
        ax_dist.grid(True, linestyle="--", alpha=0.35)
        return []

    ani = animation.FuncAnimation(fig2, update, frames=len(frame_indices),
                                  blit=False, interval=1000 // ANIM_FPS)
    ani.save(gif_path, writer="pillow", fps=ANIM_FPS, dpi=ANIM_DPI)
    plt.close(fig2)
    print(f"  Saved GIF → {gif_path}")


def main():
    solver, beta_true, beta_opt, opt_params, history, result = run()
    make_figures(solver, beta_true, beta_opt, opt_params, history, result)
    print("\nDone.")


if __name__ == "__main__":
    main()
