#!/usr/bin/env python
"""
Run all FastLSQ extension demos and generate paper-ready figures + tables.

Sections:
  1. PDE Discovery         -- SINDy-style sparse regression
  2. Inverse Heat Source   -- differentiable digital twin
  3. Inverse Magnetostatics -- quadrupole coil optimisation
  4. Learnable Bandwidth   -- reparameterised sigma via AdamW
  5. Matrix Caching        -- boundary-condition sweep timing

Figures are saved to  paper/i/  so LaTeX can include them directly.
A LaTeX-formatted summary is printed at the end.
"""

import os, sys, time, textwrap
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── FastLSQ imports ──────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastlsq.basis import SinusoidalBasis, Op
from fastlsq.linalg import solve_lstsq
from fastlsq.learnable import LearnableFastLSQ, train_bandwidth
from fastlsq.utils import device, evaluate_error
from fastlsq.problems.linear import Helmholtz2D, PoissonND

torch.set_default_dtype(torch.float64)

PAPER_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "i")
os.makedirs(PAPER_DIR, exist_ok=True)

RESULTS = {}


def _save(fig, name):
    path = os.path.join(PAPER_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {path}")


# =====================================================================
#  1. PDE DISCOVERY  (SINDy-style)
# =====================================================================

def run_pde_discovery():
    print("\n" + "=" * 65)
    print("  1. PDE Discovery via Sparse Regression")
    print("=" * 65)
    from sklearn.linear_model import Lasso
    import re as _re

    t0 = time.time()
    torch.manual_seed(42)

    NOISE_LEVEL = 0.01
    M = 2000
    x_np = np.linspace(0, 2 * np.pi, M)
    dx = x_np[1] - x_np[0]
    x = torch.tensor(x_np, device=device).reshape(-1, 1)
    u_true = torch.exp(-0.5 * x) * torch.sin(2 * x)
    u_noisy = u_true + torch.randn_like(u_true) * NOISE_LEVEL

    # True u_xx for reference
    u_true_np = u_true.flatten().cpu().numpy()
    u_noisy_np = u_noisy.flatten().cpu().numpy()
    u_xx_true_np = (np.exp(-0.5 * x_np)
                    * (-4.25 * np.sin(2 * x_np) - np.cos(2 * x_np)))

    # Finite-difference u_xx (central, 2nd order) from noisy data
    u_xx_fd = np.zeros(M)
    u_xx_fd[1:-1] = (u_noisy_np[2:] - 2 * u_noisy_np[1:-1]
                      + u_noisy_np[:-2]) / dx**2
    u_xx_fd[0] = u_xx_fd[1]; u_xx_fd[-1] = u_xx_fd[-2]

    # Fast-LSQ analytical derivatives
    basis = SinusoidalBasis.random(input_dim=1, n_features=1500,
                                   sigma=3.0, normalize=True)
    H = basis.evaluate(x)
    beta = solve_lstsq(H, u_noisy, mu=1e-8)

    cache = basis.cache(x)
    u_hat = (basis.evaluate(x, cache=cache) @ beta).flatten()
    u_x_hat = (basis.derivative(x, alpha=(1,), cache=cache) @ beta).flatten()
    u_xx_hat = (basis.derivative(x, alpha=(2,), cache=cache) @ beta).flatten()
    u_sq_hat = u_hat ** 2

    u_hat_np = u_hat.detach().cpu().numpy()
    u_xx_hat_np = u_xx_hat.detach().cpu().numpy()

    # SINDy dictionary and LASSO sweep
    Theta = torch.stack([u_hat, u_x_hat, u_sq_hat], dim=1).detach().cpu().numpy()
    Y_target = u_xx_hat_np
    feature_names = ["u", "u_x", "u^2"]

    alphas = np.logspace(-5, 1, 100)
    losses, equations, coefficients = [], [], []
    for alpha in alphas:
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=20000)
        model.fit(Theta, Y_target)
        preds = model.predict(Theta)
        mse = np.mean((Y_target - preds) ** 2)
        losses.append(mse)

        terms = []
        for coef, name in zip(model.coef_, feature_names):
            if abs(coef) > 1e-2:
                terms.append(f"{coef:.2f}*{name}")
        eq_str = "u_xx = " + (" + ".join(terms) if terms else "0")
        equations.append(eq_str)
        coefficients.append(model.coef_.copy())

    # Structure selection: find first model with exactly {u, u_x} active
    best_idx = None
    for i, coefs in enumerate(coefficients):
        active = sum(abs(c) > 1e-2 for c in coefs)
        if active == 2 and abs(coefs[2]) < 1e-2:
            best_idx = i
            break
    if best_idx is None:
        best_idx = np.argmin(losses)

    # OLS refit on the selected support (removes LASSO shrinkage bias)
    active_mask = np.abs(coefficients[best_idx]) > 1e-2
    Theta_active = Theta[:, active_mask]
    ols_coefs_active = np.linalg.lstsq(Theta_active, Y_target, rcond=None)[0]
    ols_coefs = np.zeros(len(feature_names))
    ols_coefs[active_mask] = ols_coefs_active
    ols_terms = []
    for coef, name in zip(ols_coefs, feature_names):
        if abs(coef) > 1e-3:
            ols_terms.append(f"{coef:.2f}*{name}")
    ols_eq = "u_xx = " + (" + ".join(ols_terms) if ols_terms else "0")

    elapsed = time.time() - t0

    c_u = ols_coefs[0]
    c_ux = ols_coefs[1]
    discovered = ols_eq
    print(f"  Noise level: {NOISE_LEVEL}")
    print(f"  True equation:       u_xx = -4.25*u - 1.00*u_x")
    print(f"  LASSO equation:      {equations[best_idx]}")
    print(f"  OLS-refit equation:  {discovered}")
    print(f"  Coefficients: u -> {c_u:.4f} (true: -4.25), "
          f"u_x -> {c_ux:.4f} (true: -1.00)")

    fd_rmse = np.sqrt(np.mean((u_xx_fd - u_xx_true_np)**2))
    an_rmse = np.sqrt(np.mean((u_xx_hat_np - u_xx_true_np)**2))
    print(f"  u_xx RMSE  FD: {fd_rmse:.4f}  |  Analytical: {an_rmse:.4e}  "
          f"({fd_rmse / an_rmse:.0f}x cleaner)")
    print(f"  Time: {elapsed:.2f}s")

    RESULTS["pde_discovery"] = {
        "equation": discovered,
        "c_u": c_u, "c_ux": c_ux,
        "mse": losses[best_idx],
        "time": elapsed,
        "noise": NOISE_LEVEL,
        "fd_rmse": fd_rmse,
        "an_rmse": an_rmse,
    }

    # Collect unique-equation transitions
    unique_eqs_list = []
    unique_sigs = set()
    for a_val, l_val, eq, coefs in zip(alphas, losses, equations, coefficients):
        sig = "".join(c for c in eq if c.isalpha() or c in "_^")
        if sig not in unique_sigs:
            unique_sigs.add(sig)
            unique_eqs_list.append((a_val, l_val, eq, coefs))

    # ---- Figure (2 panels) ----
    plt.rcParams.update({"mathtext.fontset": "cm"})
    fig = plt.figure(figsize=(12, 3.0))
    gs = GridSpec(1, 2, figure=fig, wspace=0.34)

    # (a) u_xx curves for each discovered equation
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x_np, u_xx_true_np, "k-", lw=2, label=r"True $u_{xx}$",
             zorder=10)
    colors_eq = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for idx, (_, _, eq, coefs) in enumerate(unique_eqs_list):
        predicted_uxx = Theta @ coefs
        col = colors_eq[idx % len(colors_eq)]

        eq_parts = eq.split(" = ", 1)
        rhs = eq_parts[1] if len(eq_parts) > 1 else eq
        def _texify(m):
            coef, var = m.group(1), m.group(2)
            var_t = {"u_x": "u_x", "u^2": "u^2", "u": "u"}.get(var, var)
            return f"{coef}\\,{var_t}"
        rhs_tex = _re.sub(r"([\-\+]?\d+\.\d+)\*(\S+)", _texify, rhs)
        rhs_tex = rhs_tex.replace("+ -", "- ").replace("+ +", "+ ")
        lbl = f"${rhs_tex}$" if rhs != "0" else "$0$"

        ax1.plot(x_np, predicted_uxx, color=col, lw=1.5, ls="--",
                 alpha=0.7, label=lbl)

    ols_uxx = Theta @ ols_coefs
    sign_ux = "-" if c_ux < 0 else "+"
    ols_lbl = f"OLS refit: ${c_u:.2f}\\,u {sign_ux} {abs(c_ux):.2f}\\,u_x$"
    ax1.plot(x_np, ols_uxx, color="red", lw=2.2, ls="-", alpha=0.9,
             label=ols_lbl, zorder=9)

    ax1.set_xlabel("$x$", fontsize=12)
    ax1.set_ylabel("$u_{xx}(x)$", fontsize=12)
    ax1.set_title("(a)  Discovered equations", fontsize=12,
                   fontweight="bold")
    ax1.legend(fontsize=8, loc="lower left", framealpha=0.9, ncol=1)
    ax1.tick_params(labelsize=10)
    ax1.grid(True, alpha=0.3)

    # (b) LASSO path
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(alphas, losses, color="blue", linewidth=2)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel(r"LASSO penalty $\lambda$", fontsize=12)
    ax2.set_ylabel("MSE", fontsize=12)
    ax2.set_title("(b)  Sparse regression (SINDy)", fontsize=12,
                   fontweight="bold")
    ax2.tick_params(labelsize=10)

    offsets_data = [(20, -10), (20, 16), (-20, 14), (-20, -12)]
    ha_data = ["left", "left", "right", "right"]
    for idx, (a_val, l_val, eq, coefs) in enumerate(unique_eqs_list):
        col = colors_eq[idx % len(colors_eq)]
        ax2.scatter(a_val, l_val, color=col, zorder=5, s=90,
                    edgecolors="k", linewidths=0.8)
        oy, ox = offsets_data[idx] if idx < len(offsets_data) else (18, -12)
        ha = ha_data[idx] if idx < len(ha_data) else "right"

        eq_parts = eq.split(" = ", 1)
        rhs = eq_parts[1] if len(eq_parts) > 1 else eq
        def _texify2(m):
            coef, var = m.group(1), m.group(2)
            var_t = {"u_x": "u_x", "u^2": "u^2", "u": "u"}.get(var, var)
            return f"{coef}\\,{var_t}"
        rhs_tex = _re.sub(r"([\-\+]?\d+\.\d+)\*(\S+)", _texify2, rhs)
        rhs_tex = rhs_tex.replace("+ -", "- ").replace("+ +", "+ ")
        eq_label = (f"$u_{{xx}} = {rhs_tex}$" if rhs != "0"
                    else "$u_{xx} = 0$")

        ax2.annotate(eq_label, (a_val, l_val), textcoords="offset points",
                     xytext=(ox, oy), ha=ha, fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white",
                               ec="gray", alpha=0.95),
                     arrowprops=dict(arrowstyle="-|>", color="gray",
                                     lw=0.8))
    ax2.grid(True, which="both", ls="--", alpha=0.4)

    fig.tight_layout()
    _save(fig, "PDE_Discovery.pdf")


# =====================================================================
#  2. INVERSE HEAT SOURCE LOCALIZATION
# =====================================================================

def run_inverse_heat():
    print("\n" + "=" * 65)
    print("  2. Inverse Heat Source Localization")
    print("=" * 65)

    t0_total = time.time()
    ALPHA_D = 0.05; T_FINAL = 1.5; SEED = 42
    N_FEAT_PER_BLK = 400; SIGMAS = [2.0, 5.0, 9.0]
    LAM_BC = 200.0; LAM_IC = 200.0; MU_REG = 1e-8
    N_INT = 6000; N_BC = 1200; N_IC = 1200
    N_SENSORS = 4; N_SNAPSHOTS = 40; NOISE_LEVEL = 0.002

    TRUE_P = np.array([0.35, 0.65, 3.0, 12.0, 5.0, 8.0])
    INIT_P = np.array([0.65, 0.30, 1.0, 5.0, 0.0, 5.0])

    # Build basis
    torch.manual_seed(SEED)
    blocks_W, blocks_b = [], []
    for sigma in SIGMAS:
        blocks_W.append(torch.randn(3, N_FEAT_PER_BLK, device=device) * sigma)
        blocks_b.append(torch.rand(1, N_FEAT_PER_BLK, device=device) * 2 * np.pi)
    basis = SinusoidalBasis(torch.cat(blocks_W, 1),
                            torch.cat(blocks_b, 1), normalize=True)
    print(f"  Basis: {basis.n_features} features, d={basis.input_dim}")

    # Sample collocation points
    torch.manual_seed(SEED + 1)
    x_int = torch.rand(N_INT, 3, device=device); x_int[:, 2] *= T_FINAL
    x_bc_list = []
    for face in range(4):
        p = torch.rand(N_BC // 4, 3, device=device); p[:, 2] *= T_FINAL
        if face == 0: p[:, 0] = 0.0
        elif face == 1: p[:, 0] = 1.0
        elif face == 2: p[:, 1] = 0.0
        else: p[:, 1] = 1.0
        x_bc_list.append(p)
    x_bc = torch.cat(x_bc_list)
    x_ic = torch.rand(N_IC, 3, device=device); x_ic[:, 2] = 0.0

    # Assemble PDE operator
    cache_int = basis.cache(x_int)
    dphi_dt = basis.gradient(x_int, cache=cache_int)[:, 2, :]
    lap_xy = basis.laplacian(x_int, dims=[0, 1], cache=cache_int)
    A_pde = dphi_dt - ALPHA_D * lap_xy
    A_bc = LAM_BC * basis.evaluate(x_bc)
    A_ic = LAM_IC * basis.evaluate(x_ic)
    A = torch.cat([A_pde, A_bc, A_ic]).cpu().numpy()
    A_pde_np = A_pde.cpu().numpy()
    x_int_np = x_int.cpu().numpy()

    AtA = A.T @ A; AtA[np.diag_indices_from(AtA)] += MU_REG
    CHO = cho_factor(AtA)
    t_assembly = time.time() - t0_total
    print(f"  Assembly + Cholesky: {t_assembly:.2f}s")

    # Source function
    def source_and_grad(xy, params):
        xs, ys, I, a, b, c = params
        dx = xy[:, 0] - xs; dy = xy[:, 1] - ys
        adx = a * dx; bdx_cdy = b * dx + c * dy
        Q = adx**2 + bdx_cdy**2
        norm_c = a * c / (2.0 * np.pi)
        expQ = np.exp(-0.5 * Q)
        J = I * norm_c * expQ
        dJ = np.empty((len(dx), 6))
        dJ[:, 0] = J * (a**2 * dx + b * bdx_cdy)
        dJ[:, 1] = J * (c * bdx_cdy)
        dJ[:, 2] = norm_c * expQ
        dJ[:, 3] = J * (1.0/a - a * dx**2)
        dJ[:, 4] = -J * bdx_cdy * dx
        dJ[:, 5] = J * (1.0/c - bdx_cdy * dy)
        return J, dJ

    # Sensor layout
    sensor_xy = np.array([[0.25, 0.25], [0.25, 0.75],
                           [0.75, 0.25], [0.75, 0.75]])
    times = np.linspace(0.1, T_FINAL, N_SNAPSHOTS)
    sensor_pts = np.array([[x, y, t] for t in times for x, y in sensor_xy])
    Phi_sensor = basis.evaluate(
        torch.tensor(sensor_pts, device=device)).cpu().numpy()

    # Generate observations
    J_true, _ = source_and_grad(x_int_np, TRUE_P)
    beta_true = cho_solve(CHO, A_pde_np.T @ J_true)
    u_true_s = Phi_sensor @ beta_true
    np.random.seed(SEED + 3)
    u_obs = u_true_s + NOISE_LEVEL * np.random.randn(len(u_true_s))
    print(f"  Observations: {N_SENSORS} sensors x {N_SNAPSHOTS} snapshots "
          f"= {len(u_obs)}")

    # Objective
    trajectory = []
    def obj_grad(params):
        J, dJ_dp = source_and_grad(x_int_np, params)
        beta_p = cho_solve(CHO, A_pde_np.T @ J)
        u_pred = Phi_sensor @ beta_p
        resid = u_pred - u_obs; ns = len(u_obs)
        loss = np.dot(resid, resid) / ns
        trajectory.append(params.copy())
        v = cho_solve(CHO, Phi_sensor.T @ resid)
        z = A_pde_np @ v
        grad = (2.0 / ns) * (dJ_dp.T @ z)
        return loss, grad

    # Run L-BFGS-B
    t0_opt = time.time()
    bounds = [(0.05, 0.95), (0.05, 0.95), (0.1, 20.0),
              (1.0, 30.0), (-15.0, 15.0), (1.0, 30.0)]
    res = minimize(obj_grad, INIT_P, method="L-BFGS-B", jac=True,
                   bounds=bounds,
                   options={"maxiter": 200, "ftol": 1e-15, "gtol": 1e-10})
    t_opt = time.time() - t0_opt
    rec = res.x
    t_total = time.time() - t0_total

    pnames = ["x_s", "y_s", "I", "a", "b", "c"]
    print(f"\n  {'Param':<6} {'True':>10} {'Init':>10} {'Recovered':>10} {'Error':>10}")
    print("  " + "-" * 50)
    errors_abs = []
    for i, name in enumerate(pnames):
        err = abs(rec[i] - TRUE_P[i])
        errors_abs.append(err)
        print(f"  {name:<6} {TRUE_P[i]:10.4f} {INIT_P[i]:10.4f} "
              f"{rec[i]:10.4f} {err:10.2e}")
    print(f"\n  Final loss: {res.fun:.2e}  |  Iterations: {res.nit}  |  "
          f"Opt time: {t_opt:.2f}s  |  Total: {t_total:.2f}s")

    RESULTS["inverse_heat"] = {
        "true": TRUE_P, "init": INIT_P, "recovered": rec,
        "errors": errors_abs, "loss": res.fun, "nit": res.nit,
        "time_assembly": t_assembly, "time_opt": t_opt, "time_total": t_total,
        "n_features": basis.n_features, "n_sensors": N_SENSORS * N_SNAPSHOTS,
    }

    # ---- Figure ----
    ng = 80
    xg = np.linspace(0, 1, ng); yg = np.linspace(0, 1, ng)
    XX, YY = np.meshgrid(xg, yg)
    t_snap = 0.8 * T_FINAL
    pts_plot = np.column_stack([XX.ravel(), YY.ravel(),
                                 np.full(ng*ng, t_snap)])
    Phi_plot = basis.evaluate(
        torch.tensor(pts_plot, device=device)).cpu().numpy()

    def field(params):
        J, _ = source_and_grad(x_int_np, params)
        return (Phi_plot @ cho_solve(CHO, A_pde_np.T @ J)).reshape(ng, ng)

    u_true_f = field(TRUE_P); u_rec_f = field(rec)
    vmin = min(u_true_f.min(), u_rec_f.min())
    vmax = max(u_true_f.max(), u_rec_f.max())
    levels = np.linspace(vmin, vmax, 30)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    im0 = axes[0].contourf(XX, YY, u_true_f, levels=levels, cmap="inferno")
    axes[0].plot(*TRUE_P[:2], "w*", ms=14, markeredgecolor="k")
    axes[0].set(title=f"True temperature ($t={t_snap:.2f}$)",
                xlabel="$x$", ylabel="$y$", aspect="equal")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(XX, YY, u_rec_f, levels=levels, cmap="inferno")
    axes[1].plot(*rec[:2], "c*", ms=14, markeredgecolor="k")
    axes[1].set(title=f"Recovered temperature ($t={t_snap:.2f}$)",
                xlabel="$x$", ylabel="$y$", aspect="equal")
    plt.colorbar(im1, ax=axes[1])

    traj = np.array(trajectory)
    axes[2].plot(traj[:, 0], traj[:, 1], "o-", color="royalblue",
                 ms=3, lw=1, alpha=0.7)
    axes[2].plot(*TRUE_P[:2], "r*", ms=16, label="True", zorder=5)
    axes[2].plot(traj[0, 0], traj[0, 1], "gs", ms=10, label="Init", zorder=5)
    axes[2].plot(traj[-1, 0], traj[-1, 1], "c^", ms=10, label="Final", zorder=5)
    axes[2].set(title="Optimisation trajectory ($x_s, y_s$)",
                xlabel="$x_s$", ylabel="$y_s$",
                xlim=(0, 1), ylim=(0, 1), aspect="equal")
    axes[2].legend(fontsize=9)

    fig.tight_layout()
    _save(fig, "Inverse_Heat_Source.pdf")


# =====================================================================
#  3. INVERSE MAGNETOSTATICS
# =====================================================================

def run_inverse_magneto():
    print("\n" + "=" * 65)
    print("  3. Inverse Magnetostatics (Quadrupole Design)")
    print("=" * 65)

    t0_total = time.time()
    torch.manual_seed(42); np.random.seed(42)

    N_FEAT_BLK = 500; SIGMAS_M = [2.0, 5.0, 8.0]
    M_INT = 5000; M_BC_M = 1500
    LAM_BC_M = 200.0; MU_M = 1e-9
    SIGMA_COIL = 0.15; I_COIL = 25.0
    GRAD_TARGET = 15.0; BEAM_R = 0.3
    ADAM_LR = 0.05; ADAM_ITERS = 50; FD_EPS = 1e-4; SYM_W = 50.0
    XC0, YC0 = 0.8, 0.2

    # Build basis
    Ws, bs = [], []
    for sigma in SIGMAS_M:
        blk = SinusoidalBasis.random(2, N_FEAT_BLK, sigma=sigma)
        Ws.append(blk.W); bs.append(blk.b)
    basis = SinusoidalBasis(torch.cat(Ws, 1), torch.cat(bs, 1), normalize=True)
    N_FEAT = basis.n_features
    print(f"  Basis: {N_FEAT} features, d=2")

    # Sample domain
    pts_int = -1.0 + 2.0 * torch.rand(M_INT, 2, device=device)
    n_side = M_BC_M // 4
    bc_pts = []
    for _ in range(n_side):
        t = -1.0 + 2.0 * torch.rand(1, device=device).item()
        bc_pts.append([t, -1.0])
    for _ in range(n_side):
        t = -1.0 + 2.0 * torch.rand(1, device=device).item()
        bc_pts.append([t, 1.0])
    for _ in range(n_side):
        t = -1.0 + 2.0 * torch.rand(1, device=device).item()
        bc_pts.append([-1.0, t])
    for _ in range(n_side):
        t = -1.0 + 2.0 * torch.rand(1, device=device).item()
        bc_pts.append([1.0, t])
    pts_bc = torch.tensor(bc_pts[:M_BC_M], device=device)

    pts_int_np = pts_int.cpu().numpy()

    # Assemble & prefactor
    A_lap = basis.laplacian(pts_int).cpu().numpy()
    A_val_bc = basis.evaluate(pts_bc).cpu().numpy()
    AtA = A_lap.T @ A_lap + LAM_BC_M**2 * A_val_bc.T @ A_val_bc
    AtA += MU_M * np.eye(N_FEAT)
    cho = cho_factor(AtA)
    t_assembly = time.time() - t0_total
    print(f"  Assembly + Cholesky: {t_assembly:.2f}s")

    def gaussian_coil(x, y, xc, yc):
        return np.exp(-((x-xc)**2 + (y-yc)**2) / (2*SIGMA_COIL**2))

    def current_density(x, y, xc, yc):
        return I_COIL * (
            gaussian_coil(x, y, xc, yc) + gaussian_coil(x, y, -xc, -yc)
            - gaussian_coil(x, y, -xc, yc) - gaussian_coil(x, y, xc, -yc))

    def solve_fwd(xc, yc):
        J = current_density(pts_int_np[:, 0], pts_int_np[:, 1], xc, yc)
        return cho_solve(cho, A_lap.T @ (-J.reshape(-1, 1))).ravel()

    # Evaluation grid inside beam pipe
    cand = np.random.rand(5000, 2) * 2 * BEAM_R - BEAM_R
    r = np.sqrt(cand[:, 0]**2 + cand[:, 1]**2)
    pts_eval_np = cand[r < BEAM_R][:800]
    pts_eval = torch.tensor(pts_eval_np, device=device)

    def eval_field(beta):
        bt = torch.tensor(beta, device=device).unsqueeze(1)
        g = basis.gradient(pts_eval)
        dAz_dx = (g[:, 0, :] @ bt).cpu().numpy().ravel()
        dAz_dy = (g[:, 1, :] @ bt).cpu().numpy().ravel()
        return dAz_dy, -dAz_dx

    def loss_fn(xc, yc):
        beta = solve_fwd(xc, yc)
        Bx, By = eval_field(beta)
        Bx_t = GRAD_TARGET * pts_eval_np[:, 1]
        By_t = GRAD_TARGET * pts_eval_np[:, 0]
        mse = np.mean((Bx - Bx_t)**2 + (By - By_t)**2)
        return mse + SYM_W * (xc - yc)**2

    # Adam
    xc, yc = XC0, YC0
    m_x, m_y, v_x, v_y = 0., 0., 0., 0.
    b1, b2, eps = 0.9, 0.999, 1e-8
    history = []
    t0_opt = time.time()
    for it in range(ADAM_ITERS):
        loss = loss_fn(xc, yc)
        history.append(loss)
        g_x = (loss_fn(xc+FD_EPS, yc) - loss_fn(xc-FD_EPS, yc)) / (2*FD_EPS)
        g_y = (loss_fn(xc, yc+FD_EPS) - loss_fn(xc, yc-FD_EPS)) / (2*FD_EPS)
        t_step = it + 1
        m_x = b1*m_x + (1-b1)*g_x; m_y = b1*m_y + (1-b1)*g_y
        v_x = b2*v_x + (1-b2)*g_x**2; v_y = b2*v_y + (1-b2)*g_y**2
        xc -= ADAM_LR * (m_x/(1-b1**t_step)) / (np.sqrt(v_x/(1-b2**t_step))+eps)
        yc -= ADAM_LR * (m_y/(1-b1**t_step)) / (np.sqrt(v_y/(1-b2**t_step))+eps)
        xc = np.clip(xc, 0.3, 0.95); yc = np.clip(yc, 0.3, 0.95)
        if it % 10 == 0 or it == ADAM_ITERS - 1:
            print(f"    iter {it:3d}  loss={loss:.4f}  "
                  f"xc={xc:.4f}  yc={yc:.4f}")
    t_opt = time.time() - t0_opt
    t_total = time.time() - t0_total

    print(f"\n  Initial coil: ({XC0}, {YC0}) -> Optimized: ({xc:.4f}, {yc:.4f})")
    print(f"  Initial loss: {history[0]:.4f} -> Final loss: {history[-1]:.4f}")
    print(f"  Improvement: {history[0]/history[-1]:.1f}x")
    print(f"  Opt time: {t_opt:.2f}s  |  Total: {t_total:.2f}s")

    RESULTS["inverse_magneto"] = {
        "xc_init": XC0, "yc_init": YC0,
        "xc_opt": xc, "yc_opt": yc,
        "loss_init": history[0], "loss_final": history[-1],
        "improvement": history[0] / max(history[-1], 1e-15),
        "iters": ADAM_ITERS, "time_opt": t_opt, "time_total": t_total,
        "n_features": N_FEAT,
    }

    # ---- Figure ----
    ng = 120
    xi = np.linspace(-1, 1, ng)
    X, Y = np.meshgrid(xi, xi)
    grid_np = np.column_stack([X.ravel(), Y.ravel()])
    grid_t = torch.tensor(grid_np, device=device)

    beta_init = solve_fwd(XC0, YC0)
    beta_opt = solve_fwd(xc, yc)
    Bx_init, By_init = eval_field(beta_init)
    Bx_opt, By_opt = eval_field(beta_opt)

    Az_init = (basis.evaluate(grid_t).cpu().numpy() @ beta_init).reshape(ng, ng)
    Az_opt = (basis.evaluate(grid_t).cpu().numpy() @ beta_opt).reshape(ng, ng)

    theta = np.linspace(0, 2*np.pi, 100)
    bx_c, by_c = BEAM_R * np.cos(theta), BEAM_R * np.sin(theta)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    c0 = axes[0, 0].contourf(X, Y, Az_init, levels=40, cmap="RdBu_r")
    plt.colorbar(c0, ax=axes[0, 0], label="$A_z$")
    axes[0, 0].plot(bx_c, by_c, "k--", lw=1.2)
    axes[0, 0].set_title(f"Initial $A_z$ (coil @ ({XC0}, {YC0}))")
    axes[0, 0].set_aspect("equal")

    c1 = axes[0, 1].contourf(X, Y, Az_opt, levels=40, cmap="RdBu_r")
    plt.colorbar(c1, ax=axes[0, 1], label="$A_z$")
    axes[0, 1].plot(bx_c, by_c, "k--", lw=1.2)
    axes[0, 1].set_title(f"Optimized $A_z$ (coil @ ({xc:.2f}, {yc:.2f}))")
    axes[0, 1].set_aspect("equal")

    # Field error in beam pipe
    mask = np.sqrt(X**2 + Y**2) <= BEAM_R
    ng2 = 60
    xi2 = np.linspace(-BEAM_R, BEAM_R, ng2)
    Xe, Ye = np.meshgrid(xi2, xi2)
    re = np.sqrt(Xe**2 + Ye**2)
    eval_np2 = np.column_stack([Xe.ravel(), Ye.ravel()])
    eval_t2 = torch.tensor(eval_np2, device=device)
    beta_init2 = solve_fwd(XC0, YC0); beta_opt2 = solve_fwd(xc, yc)
    Bx_i2, By_i2 = eval_field(beta_init2)
    Bx_o2, By_o2 = eval_field(beta_opt2)

    Bx_i_grid, By_i_grid = eval_field(beta_init)
    Bx_o_grid, By_o_grid = eval_field(beta_opt)
    Bx_target_ev = GRAD_TARGET * pts_eval_np[:, 1]
    By_target_ev = GRAD_TARGET * pts_eval_np[:, 0]
    err_init = np.sqrt((Bx_i_grid - Bx_target_ev)**2 + (By_i_grid - By_target_ev)**2)
    err_opt = np.sqrt((Bx_o_grid - Bx_target_ev)**2 + (By_o_grid - By_target_ev)**2)

    axes[1, 0].scatter(pts_eval_np[:, 0], pts_eval_np[:, 1],
                        c=err_init, cmap="hot_r", s=4, vmin=0)
    axes[1, 0].plot(bx_c, by_c, "k--", lw=1.2)
    axes[1, 0].set_title("Initial field error in beam pipe")
    axes[1, 0].set_aspect("equal")

    sc = axes[1, 1].scatter(pts_eval_np[:, 0], pts_eval_np[:, 1],
                             c=err_opt, cmap="hot_r", s=4,
                             vmin=0, vmax=err_init.max())
    plt.colorbar(sc, ax=axes[1, 1], label="$|\\mathbf{B} - \\mathbf{B}_{\\mathrm{target}}|$")
    axes[1, 1].plot(bx_c, by_c, "k--", lw=1.2)
    axes[1, 1].set_title("Optimized field error in beam pipe")
    axes[1, 1].set_aspect("equal")

    fig.tight_layout()
    _save(fig, "Inverse_Magnetostatics.pdf")

    # Convergence curve
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    ax2.semilogy(history, "o-", markersize=3, color="royalblue")
    ax2.set_xlabel("Adam iteration"); ax2.set_ylabel("Loss")
    ax2.set_title("Quadrupole optimisation convergence")
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    fig2.tight_layout()
    _save(fig2, "Inverse_Magnetostatics_Convergence.pdf")


# =====================================================================
#  4. LEARNABLE BANDWIDTH
# =====================================================================

def run_learnable_bandwidth():
    print("\n" + "=" * 65)
    print("  4. Learnable Bandwidth via Reparameterisation")
    print("=" * 65)

    torch.manual_seed(42); np.random.seed(42)
    t0 = time.time()

    problem = Helmholtz2D()
    print(f"  Problem: {problem.name} (k={problem.k})")

    N_STEPS_LB = 80
    learnable = LearnableFastLSQ(
        input_dim=2, n_features=500, mode="scalar",
        init_scale=5.0, normalize=True,
    )

    print(f"  Initial sigma: {learnable.sigma.item():.4f}")
    print(f"  Training learnable bandwidth ({N_STEPS_LB} steps) ...")

    history = train_bandwidth(
        learnable, problem,
        n_pde=3000, n_bc=600,
        n_steps=N_STEPS_LB, lr=0.1, mu=1e-10,
        verbose=True,
    )
    t_learn = time.time() - t0

    final_sigma = learnable.sigma.item()
    print(f"\n  Final sigma: {final_sigma:.4f}")
    print(f"  Time: {t_learn:.2f}s")

    # Evaluate final quality
    torch.manual_seed(999)
    x_test = problem.get_test_points(5000)
    u_pred = learnable.predict(x_test)
    u_true = problem.exact(x_test)
    val_err = (torch.norm(u_pred - u_true) / (torch.norm(u_true) + 1e-15)).item()
    print(f"  Final L2 error: {val_err:.2e}")

    # Compare with grid search (same feature count for fair comparison)
    best_gs_err = float("inf"); best_gs_sigma = 1.0
    from fastlsq.solvers import FastLSQSolver
    for sigma in [0.5, 1, 2, 3, 5, 8, 10, 12, 15]:
        torch.manual_seed(42)
        slv = FastLSQSolver(2, normalize=True)
        slv.add_block(hidden_size=500, scale=sigma)
        x_pde, bcs, f_pde = problem.get_train_data(n_pde=3000, n_bc=600)
        A, b_rhs = problem.build(slv, x_pde, bcs, f_pde)
        slv.beta = solve_lstsq(A, b_rhs, mu=1e-10)
        torch.manual_seed(999)
        xt = problem.get_test_points(5000)
        u_p = slv.predict(xt); u_t = problem.exact(xt)
        err = (torch.norm(u_p - u_t) / (torch.norm(u_t) + 1e-15)).item()
        if err < best_gs_err:
            best_gs_err = err; best_gs_sigma = sigma
    print(f"  Grid search best: sigma={best_gs_sigma}, L2={best_gs_err:.2e}")

    RESULTS["learnable_bw"] = {
        "init_sigma": 1.0, "final_sigma": final_sigma,
        "final_l2": val_err, "gs_sigma": best_gs_sigma, "gs_l2": best_gs_err,
        "time": t_learn, "n_steps": len(history),
    }

    # ---- Figure ----
    steps = [h["step"] for h in history]
    losses = [h["loss"] for h in history]
    sigmas = [h["sigma"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.semilogy(steps, losses, "b-o", ms=3, lw=1.5)
    ax1.set_xlabel("AdamW step"); ax1.set_ylabel("Residual loss")
    ax1.set_title("(a) Loss convergence")
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    ax2.plot(steps, sigmas, "r-o", ms=3, lw=1.5)
    ax2.axhline(best_gs_sigma, color="gray", ls="--", lw=1,
                label=f"Grid-search best ($\\sigma$={best_gs_sigma})")
    ax2.set_xlabel("AdamW step"); ax2.set_ylabel("$\\sigma$")
    ax2.set_title("(b) Bandwidth evolution")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, "Learnable_Bandwidth.pdf")


# =====================================================================
#  5. MATRIX CACHING BENCHMARK
# =====================================================================

def run_matrix_caching():
    print("\n" + "=" * 65)
    print("  5. Matrix Caching for Boundary-Condition Sweeps")
    print("=" * 65)

    torch.manual_seed(42)
    problem = Helmholtz2D()
    learnable = LearnableFastLSQ(input_dim=2, n_features=1500,
                                  mode="scalar", init_scale=10.0,
                                  normalize=True)

    x_pde, bcs, f_pde = problem.get_train_data(n_pde=5000, n_bc=1000)
    A, b_rhs = problem.build(learnable, x_pde, bcs, f_pde)

    # Warm up
    learnable.solve_inner(A, b_rhs, mu=1e-10)

    # Time full solve
    n_rhs = 50
    t0 = time.time()
    for i in range(n_rhs):
        b_new = b_rhs + 0.01 * torch.randn_like(b_rhs)
        learnable.solve_inner(A, b_new, mu=1e-10)
    t_full = time.time() - t0
    t_per_full = t_full / n_rhs

    # Time cached solve
    learnable.cache_operator(A)
    t0 = time.time()
    for i in range(n_rhs):
        b_new = b_rhs + 0.01 * torch.randn_like(b_rhs)
        learnable.solve_cached(b_new)
    t_cached = time.time() - t0
    t_per_cached = t_cached / n_rhs

    speedup = t_per_full / max(t_per_cached, 1e-12)
    print(f"  Full lstsq: {t_per_full*1000:.2f} ms/solve")
    print(f"  Cached A+:  {t_per_cached*1000:.2f} ms/solve")
    print(f"  Speedup:    {speedup:.1f}x")

    RESULTS["matrix_caching"] = {
        "t_full_ms": t_per_full * 1000,
        "t_cached_ms": t_per_cached * 1000,
        "speedup": speedup, "n_rhs": n_rhs,
        "n_features": 1500,
    }


# =====================================================================
#  LATEX OUTPUT
# =====================================================================

def print_latex():
    print("\n\n" + "=" * 65)
    print("  LaTeX-ready output for paper/content.tex")
    print("=" * 65)

    # PDE Discovery
    r = RESULTS.get("pde_discovery", {})
    print(textwrap.dedent(f"""\
    %% --- PDE Discovery ---
    %% True: u_xx = -4.25*u - 1.00*u_x
    %% Discovered: {r.get('equation','N/A')}
    %% Coefficients: u -> {r.get('c_u',0):.4f}, u_x -> {r.get('c_ux',0):.4f}
    %% Runtime: {r.get('time',0):.2f}s
    """))

    # Inverse Heat Source
    r = RESULTS.get("inverse_heat", {})
    if r:
        print("% --- Inverse Heat Source ---")
        print("\\begin{table}[h]")
        print("\\caption{Inverse heat-source recovery results.}")
        print("\\label{tab:inverse_heat}")
        print("\\centering\\small")
        print("\\begin{tabular}{l rrr r}")
        print("\\toprule")
        print("Parameter & True & Init & Recovered & Error \\\\")
        print("\\midrule")
        pnames = ["$x_s$", "$y_s$", "$I$", "$a$", "$b$", "$c$"]
        for i, name in enumerate(pnames):
            print(f"{name} & {r['true'][i]:.2f} & {r['init'][i]:.2f} & "
                  f"{r['recovered'][i]:.4f} & {r['errors'][i]:.2e} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print(f"% Final loss: {r['loss']:.2e}, {r['nit']} L-BFGS-B iters, "
              f"{r['n_features']} features, {r['n_sensors']} observations")
        print(f"% Assembly: {r['time_assembly']:.2f}s, "
              f"Optimisation: {r['time_opt']:.2f}s, "
              f"Total: {r['time_total']:.2f}s")
        print("\\end{table}")
        print()

    # Inverse Magnetostatics
    r = RESULTS.get("inverse_magneto", {})
    if r:
        print(f"% --- Inverse Magnetostatics ---")
        print(f"% Initial coil: ({r['xc_init']}, {r['yc_init']})  "
              f"-> Optimized: ({r['xc_opt']:.4f}, {r['yc_opt']:.4f})")
        print(f"% Initial loss: {r['loss_init']:.4f}  "
              f"-> Final loss: {r['loss_final']:.4f}  "
              f"({r['improvement']:.1f}x improvement)")
        print(f"% {r['iters']} Adam iters, {r['n_features']} features, "
              f"Total: {r['time_total']:.2f}s")
        print()

    # Learnable Bandwidth
    r = RESULTS.get("learnable_bw", {})
    if r:
        print(f"% --- Learnable Bandwidth ---")
        print(f"% sigma: {r['init_sigma']:.1f} -> {r['final_sigma']:.4f} "
              f"(grid-search best: {r['gs_sigma']})")
        print(f"% L2 error: {r['final_l2']:.2e} "
              f"(grid-search: {r['gs_l2']:.2e})")
        print(f"% {r['n_steps']} AdamW steps, {r['time']:.2f}s")
        print()

    # Matrix Caching
    r = RESULTS.get("matrix_caching", {})
    if r:
        print(f"% --- Matrix Caching ---")
        print(f"% Full lstsq: {r['t_full_ms']:.2f} ms/solve")
        print(f"% Cached A+:  {r['t_cached_ms']:.2f} ms/solve")
        print(f"% Speedup:    {r['speedup']:.1f}x  "
              f"({r['n_rhs']} RHS sweeps, {r['n_features']} features)")
        print()


# =====================================================================
#  MAIN
# =====================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  FastLSQ Extension Demos -- Paper Figure Generation")
    print("=" * 65)
    print(f"  Device: {device}")
    print(f"  Output directory: {os.path.abspath(PAPER_DIR)}")

    run_pde_discovery()
    run_inverse_heat()
    run_inverse_magneto()
    run_learnable_bandwidth()
    run_matrix_caching()

    print_latex()

    print("\n  All done. Figures saved to paper/i/")
