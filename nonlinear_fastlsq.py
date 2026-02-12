"""
Newton-Raphson Fast-LSQ: Extending Fast-LSQ to Nonlinear PDEs (v2)
===================================================================

Improvements over v1:
  1. Tikhonov regularization in the Newton lstsq step to tame ill-conditioning
  2. Feature normalization by 1/√N for proper kernel scaling
  3. Continuation / homotopy for advection-dominated problems (Burgers)
  4. Relative convergence criterion based on solution change ‖Δu‖/‖u‖
     instead of raw ‖δβ‖ which is meaningless when β is large

Five nonlinear test problems, all in TRUE SOLVER MODE.
"""

import torch
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
np.random.seed(42)


# ==============================================================================
# 1. SOLVER — with 1/√N normalization
# ==============================================================================

class FastLSQSolver:
    """
    Fast-LSQ: Random Fourier Features with sin activation.

    Features are normalized by 1/√N so that:
      u_N(x) = (1/√N) Σ β_j sin(W_j·x + b_j)

    This ensures the empirical kernel (1/N) Σ φ_j(x)φ_j(x') ≈ k(x,x')
    is properly scaled, keeping β_j ~ O(1) and preventing the massive
    cancellation / ill-conditioning seen when β ~ 10^6-10^8.
    """
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.W_list = []
        self.b_list = []
        self.beta = None
        self._n_features = 0

    def add_block(self, hidden_size=500, scale=1.0):
        W = torch.randn(self.input_dim, hidden_size, device=device) * scale
        b = torch.rand(1, hidden_size, device=device) * 2 * np.pi
        self.W_list.append(W)
        self.b_list.append(b)
        self._n_features += hidden_size

    @property
    def n_features(self):
        return self._n_features

    def get_features(self, x):
        """
        Returns H [M,N], dH [M,D,N], ddH [M,D,N] — all pre-divided by √N.
        """
        Hs, dHs, ddHs = [], [], []
        for W, b in zip(self.W_list, self.b_list):
            Z = x @ W + b                                          # [M, H]
            sin_Z = torch.sin(Z)
            cos_Z = torch.cos(Z)
            Hs.append(sin_Z)
            dHs.append(cos_Z.unsqueeze(1) * W.unsqueeze(0))
            ddHs.append(-sin_Z.unsqueeze(1) * (W**2).unsqueeze(0))

        # Normalize by 1/√N for proper kernel scaling
        norm = np.sqrt(self._n_features)
        H   = torch.cat(Hs, -1)   / norm
        dH  = torch.cat(dHs, -1)  / norm
        ddH = torch.cat(ddHs, -1) / norm
        return H, dH, ddH

    def evaluate(self, x):
        H, _, _ = self.get_features(x)
        return H @ self.beta

    def evaluate_with_grad(self, x):
        H, dH, _ = self.get_features(x)
        u = H @ self.beta
        grad_u = torch.einsum('idh,ho->id', dH, self.beta)
        return u, grad_u

    def evaluate_with_laplacian(self, x):
        H, dH, ddH = self.get_features(x)
        u = H @ self.beta
        grad_u = torch.einsum('idh,ho->id', dH, self.beta)
        lap_u = torch.sum(ddH, dim=1) @ self.beta
        return u, grad_u, lap_u


# ==============================================================================
# 2. REGULARIZED LEAST-SQUARES SOLVE
# ==============================================================================

def solve_lstsq(A, b, mu=0.0):
    """
    Solve min ‖Ax - b‖² + μ‖x‖²  (Tikhonov-regularized least squares).

    When μ=0, falls back to standard lstsq.
    When μ>0, forms the normal equations:  (AᵀA + μI)x = Aᵀb
    which is more stable than augmenting the system with √μ I rows.
    """
    if mu <= 0:
        return torch.linalg.lstsq(A, b).solution
    else:
        AtA = A.T @ A
        Atb = A.T @ b
        AtA.diagonal().add_(mu)
        # Use Cholesky for SPD system — faster and more stable than LU
        try:
            L = torch.linalg.cholesky(AtA)
            return torch.cholesky_solve(Atb, L)
        except torch.linalg.LinAlgError:
            # Fallback if Cholesky fails (shouldn't with μ>0, but be safe)
            return torch.linalg.solve(AtA, Atb)


# ==============================================================================
# 3. NEWTON-RAPHSON DRIVER — improved convergence criteria & line search
# ==============================================================================

def newton_solve(solver, problem, x_pde, bcs, f_pde,
                 max_iter=30, tol_res=1e-12, tol_du=1e-13,
                 damping=1.0, mu=1e-10, verbose=True):
    """
    Newton-Raphson iteration with Tikhonov-regularized Fast-LSQ steps.

    Convergence is checked via TWO criteria (both must be small):
      1. Residual norm:     ‖R‖ < tol_res
      2. Relative solution change:  ‖Δu‖ / ‖u‖ < tol_du
         computed at collocation points, NOT via ‖δβ‖ (which is meaningless
         when features are near-linearly-dependent and β is large).

    Args:
        mu: Tikhonov regularization parameter for J δβ = -R solve.
            Even mu=1e-10 dramatically stabilizes the step direction
            without affecting accuracy.
    """
    history = []

    for it in range(max_iter):
        # --- Assemble the Newton system ---
        J, neg_R = problem.build_newton_step(solver, x_pde, bcs, f_pde)
        res_norm = torch.norm(neg_R).item()

        # --- Regularized solve for update ---
        delta_beta = solve_lstsq(J, neg_R, mu=mu)

        # --- Compute solution-level change (meaningful metric) ---
        H_pde, _, _ = solver.get_features(x_pde)
        u_current = H_pde @ solver.beta
        du = H_pde @ delta_beta
        u_norm = torch.norm(u_current).item()
        du_norm = torch.norm(du).item()
        rel_du = du_norm / max(u_norm, 1e-15)

        # --- Backtracking line search on residual norm ---
        alpha = damping
        beta_old = solver.beta.clone()

        for ls_step in range(10):
            solver.beta = beta_old + alpha * delta_beta
            _, new_neg_R = problem.build_newton_step(solver, x_pde, bcs, f_pde)
            new_res = torch.norm(new_neg_R).item()
            # Sufficient decrease: Armijo-like condition
            if new_res < res_norm * (1.0 - 1e-4 * alpha) + 1e-15:
                break
            alpha *= 0.5
        else:
            # Even smallest step didn't decrease — take it anyway
            solver.beta = beta_old + alpha * delta_beta

        history.append({
            'iter': it, 'residual': res_norm,
            'rel_du': rel_du, 'du_norm': du_norm,
            'step_size': alpha
        })

        if verbose:
            print(f"  Newton {it:2d}: |R|={res_norm:.2e}  "
                  f"|Δu|/|u|={rel_du:.2e}  α={alpha:.3f}")

        # --- Convergence check ---
        if res_norm < tol_res and rel_du < tol_du:
            if verbose:
                print(f"  ✓ Converged in {it+1} iterations "
                      f"(|R|={res_norm:.1e}, |Δu|/|u|={rel_du:.1e})")
            break

        # Early stop if residual is tiny even if rel_du is not
        if res_norm < tol_res * 0.01:
            if verbose:
                print(f"  ✓ Residual converged in {it+1} iterations "
                      f"(|R|={res_norm:.1e})")
            break

    return history


def build_solver_with_scale(input_dim, scale, n_blocks=3, hidden=500):
    """Create a fresh FastLSQSolver with given scale."""
    solver = FastLSQSolver(input_dim)
    for _ in range(n_blocks):
        solver.add_block(hidden_size=hidden, scale=scale)
    solver.beta = torch.zeros(solver.n_features, 1, device=device)
    return solver


def get_initial_guess(solver, problem, x_pde, bcs, f_pde, mu=1e-10):
    """Solve the linear part of the PDE as a warm start for Newton."""
    if hasattr(problem, 'build_linear_init'):
        A, b = problem.build_linear_init(solver, x_pde, bcs, f_pde)
        solver.beta = solve_lstsq(A, b, mu=mu)


# ==============================================================================
# 4. CONTINUATION / HOMOTOPY DRIVER
# ==============================================================================

def continuation_solve(solver, problem, x_pde, bcs, f_pde_final,
                       param_name, param_schedule,
                       max_newton_per_step=15, mu=1e-10, verbose=True):
    """
    Solve a sequence of problems with gradually increasing nonlinearity.

    For Burgers:  param_name='nu', schedule=[1.0, 0.5, 0.2, 0.1]
    At each stage, the previous solution is used as the initial guess
    for Newton on the next (harder) parameter value.

    The problem object must support:
      - setattr(problem, param_name, value) to change the parameter
      - problem.source(x) must recompute f for the new parameter
      - problem.build_newton_step(...)
    """
    all_history = []

    for step_idx, param_val in enumerate(param_schedule):
        setattr(problem, param_name, param_val)

        # Recompute source term for the new parameter value
        f_pde = problem.source(x_pde)

        if verbose:
            print(f"\n  --- Continuation step {step_idx+1}/{len(param_schedule)}: "
                  f"{param_name}={param_val} ---")

        # On the first step with beta=0, use linear init
        if step_idx == 0 and torch.norm(solver.beta).item() < 1e-10:
            get_initial_guess(solver, problem, x_pde, bcs, f_pde, mu=mu)

        history = newton_solve(
            solver, problem, x_pde, bcs, f_pde,
            max_iter=max_newton_per_step, mu=mu, verbose=verbose
        )
        all_history.extend(history)

    return all_history


# ==============================================================================
# 5. NONLINEAR PDE PROBLEMS
# ==============================================================================

class NLPoisson2D:
    """
    Nonlinear Poisson with cubic term (2D)
    PDE:   -Δu + u³ = f   on [0,1]²
    BC:    u = g           on ∂Ω
    Exact: u = sin(πx) sin(πy)
    """
    def __init__(self):
        self.name = "NL-Poisson 2D (u³)"
        self.dim = 2
        self.lam_bc = 100.0

    def exact(self, x):
        return torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2])

    def exact_grad(self, x):
        sx = torch.sin(np.pi * x[:, 0:1]); cx = torch.cos(np.pi * x[:, 0:1])
        sy = torch.sin(np.pi * x[:, 1:2]); cy = torch.cos(np.pi * x[:, 1:2])
        return torch.cat([np.pi * cx * sy, np.pi * sx * cy], dim=1)

    def source(self, x):
        u = self.exact(x)
        return 2 * np.pi**2 * u + u**3

    def get_train_data(self, n_pde=5000, n_bc=1000):
        x_pde = torch.rand(n_pde, 2, device=device)
        f_pde = self.source(x_pde)
        n_side = n_bc // 4
        r = lambda n: torch.rand(n, 1, device=device)
        z = lambda n: torch.zeros(n, 1, device=device)
        o = lambda n: torch.ones(n, 1, device=device)
        bcs_pts = [torch.cat([z(n_side), r(n_side)], 1),
                   torch.cat([o(n_side), r(n_side)], 1),
                   torch.cat([r(n_side), z(n_side)], 1),
                   torch.cat([r(n_side), o(n_side)], 1)]
        x_bc = torch.cat(bcs_pts, 0)
        u_bc = self.exact(x_bc)
        return x_pde, [(x_bc, u_bc)], f_pde

    def build_newton_step(self, solver, x_pde, bcs, f_pde):
        H, dH, ddH = solver.get_features(x_pde)
        beta = solver.beta
        lap_feat = torch.sum(ddH, dim=1)

        u_k = H @ beta
        lap_uk = lap_feat @ beta

        R = -lap_uk + u_k**3 - f_pde
        J_pde = -lap_feat + 3 * (u_k**2) * H

        rows_J, rows_b = [J_pde], [-R]
        for (x_bc, u_bc) in bcs:
            H_bc, _, _ = solver.get_features(x_bc)
            rows_J.append(self.lam_bc * H_bc)
            rows_b.append(self.lam_bc * (u_bc - H_bc @ beta))
        return torch.cat(rows_J, 0), torch.cat(rows_b, 0)

    def build_linear_init(self, solver, x_pde, bcs, f_pde):
        H, dH, ddH = solver.get_features(x_pde)
        lap = torch.sum(ddH, dim=1)
        rows_A, rows_b = [-lap], [f_pde]
        for (x_bc, u_bc) in bcs:
            H_bc, _, _ = solver.get_features(x_bc)
            rows_A.append(self.lam_bc * H_bc)
            rows_b.append(self.lam_bc * u_bc)
        return torch.cat(rows_A, 0), torch.cat(rows_b, 0)

    def get_test_points(self, n=5000):
        return torch.rand(n, 2, device=device)


class Bratu2D:
    """
    Bratu equation (2D)
    PDE:   -Δu - λ eᵘ = f   on [0,1]²
    BC:    u = g              on ∂Ω
    Exact: u = sin(πx) sin(πy)
    """
    def __init__(self, lam=1.0):
        self.name = f"Bratu 2D (λ={lam})"
        self.dim = 2
        self.lam = lam
        self.lam_bc = 100.0

    def exact(self, x):
        return torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2])

    def exact_grad(self, x):
        sx = torch.sin(np.pi * x[:, 0:1]); cx = torch.cos(np.pi * x[:, 0:1])
        sy = torch.sin(np.pi * x[:, 1:2]); cy = torch.cos(np.pi * x[:, 1:2])
        return torch.cat([np.pi * cx * sy, np.pi * sx * cy], dim=1)

    def source(self, x):
        u = self.exact(x)
        return 2 * np.pi**2 * u - self.lam * torch.exp(u)

    def get_train_data(self, n_pde=5000, n_bc=1000):
        x_pde = torch.rand(n_pde, 2, device=device)
        f_pde = self.source(x_pde)
        n_side = n_bc // 4
        r = lambda n: torch.rand(n, 1, device=device)
        z = lambda n: torch.zeros(n, 1, device=device)
        o = lambda n: torch.ones(n, 1, device=device)
        bcs_pts = [torch.cat([z(n_side), r(n_side)], 1),
                   torch.cat([o(n_side), r(n_side)], 1),
                   torch.cat([r(n_side), z(n_side)], 1),
                   torch.cat([r(n_side), o(n_side)], 1)]
        x_bc = torch.cat(bcs_pts, 0)
        u_bc = self.exact(x_bc)
        return x_pde, [(x_bc, u_bc)], f_pde

    def build_newton_step(self, solver, x_pde, bcs, f_pde):
        H, dH, ddH = solver.get_features(x_pde)
        beta = solver.beta
        lap_feat = torch.sum(ddH, dim=1)

        u_k = H @ beta
        lap_uk = lap_feat @ beta
        exp_uk = torch.exp(torch.clamp(u_k, max=20.0))  # clamp for safety

        R = -lap_uk - self.lam * exp_uk - f_pde
        J_pde = -lap_feat - self.lam * exp_uk * H

        rows_J, rows_b = [J_pde], [-R]
        for (x_bc, u_bc) in bcs:
            H_bc, _, _ = solver.get_features(x_bc)
            rows_J.append(self.lam_bc * H_bc)
            rows_b.append(self.lam_bc * (u_bc - H_bc @ beta))
        return torch.cat(rows_J, 0), torch.cat(rows_b, 0)

    def build_linear_init(self, solver, x_pde, bcs, f_pde):
        H, dH, ddH = solver.get_features(x_pde)
        lap = torch.sum(ddH, dim=1)
        rows_A, rows_b = [-lap], [f_pde]
        for (x_bc, u_bc) in bcs:
            H_bc, _, _ = solver.get_features(x_bc)
            rows_A.append(self.lam_bc * H_bc)
            rows_b.append(self.lam_bc * u_bc)
        return torch.cat(rows_A, 0), torch.cat(rows_b, 0)

    def get_test_points(self, n=5000):
        return torch.rand(n, 2, device=device)


class SteadyBurgers1D:
    """
    Steady viscous Burgers equation (1D)
    PDE:   u uₓ - ν u_xx = f   on [0, 1]
    BC:    u(0) = u(1) = 0
    Exact: u = sin(2πx)

    Supports continuation: solve at high ν first, then decrease.
    """
    def __init__(self, nu=0.1):
        self.name = f"Steady Burgers 1D (ν={nu})"
        self.dim = 1
        self.nu = nu
        self.nu_target = nu  # remember the target for continuation
        self.lam_bc = 200.0
        self.use_continuation = True
        # Schedule: start viscous (easy), march toward target
        self.continuation_schedule = [1.0, 0.5, 0.2, 0.1]

    def exact(self, x):
        return torch.sin(2 * np.pi * x[:, 0:1])

    def exact_grad(self, x):
        return 2 * np.pi * torch.cos(2 * np.pi * x[:, 0:1])

    def source(self, x):
        """Recomputes f for the CURRENT value of self.nu."""
        u = self.exact(x)
        ux = self.exact_grad(x)
        uxx = -(2 * np.pi)**2 * torch.sin(2 * np.pi * x[:, 0:1])
        return u * ux - self.nu * uxx

    def get_train_data(self, n_pde=3000, n_bc=200):
        x_pde = torch.rand(n_pde, 1, device=device)
        f_pde = self.source(x_pde)
        x_bc = torch.cat([torch.zeros(n_bc // 2, 1, device=device),
                          torch.ones(n_bc // 2, 1, device=device)], 0)
        u_bc = torch.zeros(n_bc, 1, device=device)
        return x_pde, [(x_bc, u_bc)], f_pde

    def build_newton_step(self, solver, x_pde, bcs, f_pde):
        H, dH, ddH = solver.get_features(x_pde)
        beta = solver.beta

        u_k = H @ beta
        ux_feat = dH[:, 0, :]
        ux_k = ux_feat @ beta
        uxx_feat = ddH[:, 0, :]
        uxx_k = uxx_feat @ beta

        R = u_k * ux_k - self.nu * uxx_k - f_pde
        J_pde = u_k * ux_feat + ux_k * H - self.nu * uxx_feat

        rows_J, rows_b = [J_pde], [-R]
        for (x_bc, u_bc) in bcs:
            H_bc, _, _ = solver.get_features(x_bc)
            rows_J.append(self.lam_bc * H_bc)
            rows_b.append(self.lam_bc * (u_bc - H_bc @ beta))
        return torch.cat(rows_J, 0), torch.cat(rows_b, 0)

    def build_linear_init(self, solver, x_pde, bcs, f_pde):
        _, _, ddH = solver.get_features(x_pde)
        uxx_feat = ddH[:, 0, :]
        rows_A, rows_b = [-self.nu * uxx_feat], [f_pde]
        for (x_bc, u_bc) in bcs:
            H_bc, _, _ = solver.get_features(x_bc)
            rows_A.append(self.lam_bc * H_bc)
            rows_b.append(self.lam_bc * u_bc)
        return torch.cat(rows_A, 0), torch.cat(rows_b, 0)

    def get_test_points(self, n=5000):
        return torch.rand(n, 1, device=device)


class NLHelmholtz2D:
    """
    Nonlinear (cubic) Helmholtz equation (2D)
    PDE:   Δu + k²u + α u³ = f   on [0,1]²
    BC:    u = g                   on ∂Ω
    Exact: u = sin(kx) sin(ky)
    """
    def __init__(self, k=3.0, alpha=0.5):
        self.name = f"NL-Helmholtz 2D (k={k})"
        self.dim = 2
        self.k = k
        self.alpha = alpha
        self.lam_bc = 100.0

    def exact(self, x):
        return torch.sin(self.k * x[:, 0:1]) * torch.sin(self.k * x[:, 1:2])

    def exact_grad(self, x):
        k = self.k
        sx = torch.sin(k * x[:, 0:1]); cx = torch.cos(k * x[:, 0:1])
        sy = torch.sin(k * x[:, 1:2]); cy = torch.cos(k * x[:, 1:2])
        return torch.cat([k * cx * sy, k * sx * cy], dim=1)

    def source(self, x):
        u = self.exact(x)
        return -self.k**2 * u + self.alpha * u**3

    def get_train_data(self, n_pde=5000, n_bc=1000):
        x_pde = torch.rand(n_pde, 2, device=device)
        f_pde = self.source(x_pde)
        n_side = n_bc // 4
        r = lambda n: torch.rand(n, 1, device=device)
        z = lambda n: torch.zeros(n, 1, device=device)
        o = lambda n: torch.ones(n, 1, device=device)
        bcs_pts = [torch.cat([z(n_side), r(n_side)], 1),
                   torch.cat([o(n_side), r(n_side)], 1),
                   torch.cat([r(n_side), z(n_side)], 1),
                   torch.cat([r(n_side), o(n_side)], 1)]
        x_bc = torch.cat(bcs_pts, 0)
        u_bc = self.exact(x_bc)
        return x_pde, [(x_bc, u_bc)], f_pde

    def build_newton_step(self, solver, x_pde, bcs, f_pde):
        H, dH, ddH = solver.get_features(x_pde)
        beta = solver.beta
        lap_feat = torch.sum(ddH, dim=1)

        u_k = H @ beta
        lap_uk = lap_feat @ beta

        R = lap_uk + self.k**2 * u_k + self.alpha * u_k**3 - f_pde
        J_pde = lap_feat + self.k**2 * H + 3 * self.alpha * (u_k**2) * H

        rows_J, rows_b = [J_pde], [-R]
        for (x_bc, u_bc) in bcs:
            H_bc, _, _ = solver.get_features(x_bc)
            rows_J.append(self.lam_bc * H_bc)
            rows_b.append(self.lam_bc * (u_bc - H_bc @ beta))
        return torch.cat(rows_J, 0), torch.cat(rows_b, 0)

    def build_linear_init(self, solver, x_pde, bcs, f_pde):
        H, _, ddH = solver.get_features(x_pde)
        lap = torch.sum(ddH, dim=1)
        A_pde = lap + self.k**2 * H
        rows_A, rows_b = [A_pde], [f_pde]
        for (x_bc, u_bc) in bcs:
            H_bc, _, _ = solver.get_features(x_bc)
            rows_A.append(self.lam_bc * H_bc)
            rows_b.append(self.lam_bc * u_bc)
        return torch.cat(rows_A, 0), torch.cat(rows_b, 0)

    def get_test_points(self, n=5000):
        return torch.rand(n, 2, device=device)


class AllenCahn1D:
    """
    Steady Allen-Cahn equation (1D)
    PDE:   ε u_xx + u - u³ = f   on [0, 1]
    BC:    u(0) = 0, u(1) = 0
    Exact: u = sin(πx)
    """
    def __init__(self, eps=0.1):
        self.name = f"Allen-Cahn 1D (ε={eps})"
        self.dim = 1
        self.eps = eps
        self.lam_bc = 200.0

    def exact(self, x):
        return torch.sin(np.pi * x[:, 0:1])

    def exact_grad(self, x):
        return np.pi * torch.cos(np.pi * x[:, 0:1])

    def source(self, x):
        u = self.exact(x)
        uxx = -(np.pi**2) * u
        return self.eps * uxx + u - u**3

    def get_train_data(self, n_pde=3000, n_bc=200):
        x_pde = torch.rand(n_pde, 1, device=device)
        f_pde = self.source(x_pde)
        x_bc = torch.cat([torch.zeros(n_bc // 2, 1, device=device),
                          torch.ones(n_bc // 2, 1, device=device)], 0)
        u_bc = torch.zeros(n_bc, 1, device=device)
        return x_pde, [(x_bc, u_bc)], f_pde

    def build_newton_step(self, solver, x_pde, bcs, f_pde):
        H, dH, ddH = solver.get_features(x_pde)
        beta = solver.beta
        uxx_feat = ddH[:, 0, :]

        u_k = H @ beta
        uxx_k = uxx_feat @ beta

        R = self.eps * uxx_k + u_k - u_k**3 - f_pde
        J_pde = self.eps * uxx_feat + (1 - 3 * u_k**2) * H

        rows_J, rows_b = [J_pde], [-R]
        for (x_bc, u_bc) in bcs:
            H_bc, _, _ = solver.get_features(x_bc)
            rows_J.append(self.lam_bc * H_bc)
            rows_b.append(self.lam_bc * (u_bc - H_bc @ beta))
        return torch.cat(rows_J, 0), torch.cat(rows_b, 0)

    def build_linear_init(self, solver, x_pde, bcs, f_pde):
        H, _, ddH = solver.get_features(x_pde)
        uxx = ddH[:, 0, :]
        rows_A = [self.eps * uxx + H]
        rows_b = [f_pde]
        for (x_bc, u_bc) in bcs:
            H_bc, _, _ = solver.get_features(x_bc)
            rows_A.append(self.lam_bc * H_bc)
            rows_b.append(self.lam_bc * u_bc)
        return torch.cat(rows_A, 0), torch.cat(rows_b, 0)

    def get_test_points(self, n=5000):
        return torch.rand(n, 1, device=device)


# ==============================================================================
# 6. RUNNER
# ==============================================================================

def evaluate_error(solver, problem, n_test=5000):
    """Compute relative L2 errors for value and gradient."""
    torch.manual_seed(999)
    x_test = problem.get_test_points(n_test)
    u_true = problem.exact(x_test)
    grad_true = problem.exact_grad(x_test)
    u_pred, grad_pred = solver.evaluate_with_grad(x_test)

    val_err = (torch.norm(u_pred - u_true) / (torch.norm(u_true) + 1e-15)).item()
    grad_err = (torch.norm(grad_pred - grad_true) /
                (torch.norm(grad_true) + 1e-15)).item()
    return val_err, grad_err


def run_newton_problem(problem, scale, n_blocks=3, hidden=500,
                       max_iter=30, damping=1.0, mu=1e-10, verbose=True):
    """
    Full pipeline: build solver, warm-start, Newton iterate, measure error.
    Handles continuation for problems that request it.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    solver = build_solver_with_scale(problem.dim, scale, n_blocks, hidden)
    data = problem.get_train_data()
    x_pde, bcs, f_pde = data

    t0 = time.time()

    # --- Check if this problem wants continuation ---
    if getattr(problem, 'use_continuation', False):
        schedule = problem.continuation_schedule
        # Make sure the target ν is included at the end
        if schedule[-1] != problem.nu_target:
            schedule = schedule + [problem.nu_target]
        # Filter: only include steps >= target
        schedule = [v for v in schedule if v >= problem.nu_target]

        history = continuation_solve(
            solver, problem, x_pde, bcs, f_pde,
            param_name='nu', param_schedule=schedule,
            max_newton_per_step=max_iter // max(len(schedule), 1) + 5,
            mu=mu, verbose=verbose
        )
        # Restore target parameter for evaluation
        problem.nu = problem.nu_target
    else:
        # Standard Newton
        get_initial_guess(solver, problem, x_pde, bcs, f_pde, mu=mu)
        history = newton_solve(solver, problem, x_pde, bcs, f_pde,
                               max_iter=max_iter, mu=mu,
                               damping=damping, verbose=verbose)

    total_time = time.time() - t0
    val_err, grad_err = evaluate_error(solver, problem)
    n_iters = len(history)

    return val_err, grad_err, total_time, n_iters, history


def grid_search_newton(problem, scales, n_blocks=3, hidden=500,
                       max_iter=30, damping=1.0, mu=1e-10, verbose=False):
    """Grid search over scales for Newton solver."""
    best = {'scale': scales[0], 'val': float('inf'), 'grad': float('inf'),
            'time': 0, 'iters': 0}

    for scale in scales:
        try:
            ve, ge, t, ni, _ = run_newton_problem(
                problem, scale, n_blocks, hidden, max_iter, damping,
                mu=mu, verbose=verbose)
            if np.isnan(ve) or np.isinf(ve):
                ve = 1e10
            if ve < best['val']:
                best = {'scale': scale, 'val': ve, 'grad': ge,
                        'time': t, 'iters': ni}
        except Exception as e:
            if verbose:
                print(f"    Scale {scale:.1f} failed: {e}")
            continue

    return best


def plot_convergence(histories, labels, problem_name, filename):
    """Plot Newton convergence: residual and relative solution change."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for hist, label in zip(histories, labels):
        iters = [h['iter'] for h in hist]
        residuals = [h['residual'] for h in hist]
        rel_dus = [h['rel_du'] for h in hist]

        ax1.semilogy(iters, residuals, '-o', label=label, markersize=4)
        ax2.semilogy(iters, rel_dus, '-o', label=label, markersize=4)

    ax1.set_xlabel('Newton Iteration')
    ax1.set_ylabel('‖Residual‖')
    ax1.set_title(f'{problem_name}: Residual')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Newton Iteration')
    ax2.set_ylabel('‖Δu‖ / ‖u‖  (relative solution change)')
    ax2.set_title(f'{problem_name}: Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {filename}")

# ==============================================================================
# 8. SPECTRAL SENSITIVITY EXTENSIONS
# ==============================================================================

def run_spectral_sweep(problem, scales, mu=1e-10):
    """
    Runs the Newton solver across a range of scales (frequencies)
    to analyze spectral sensitivity.
    """
    val_errors = []
    grad_errors = []
    runtimes = []

    print(f"   -> Sweeping scales: {scales}")

    for scale in scales:
        try:
            # Run quiet (verbose=False) to avoid flooding console
            # The run_newton_problem function handles continuation internally if needed
            ve, ge, t, _, _ = run_newton_problem(
                problem, scale, n_blocks=3, hidden=500,
                max_iter=30, damping=1.0, mu=mu, verbose=False
            )

            # Sanity check for NaNs or Divergence
            if np.isnan(ve) or np.isinf(ve) or ve > 1e5:
                ve = 1.0 # Cap error at 1.0 (100%) for plotting
            if np.isnan(ge) or np.isinf(ge) or ge > 1e5:
                ge = 1.0

            val_errors.append(ve)
            grad_errors.append(ge)
            runtimes.append(t)

        except Exception as e:
            # If solver crashes (singular matrix etc), record high error
            val_errors.append(1.0)
            grad_errors.append(1.0)
            runtimes.append(0.0)

    return val_errors, grad_errors, runtimes

def plot_spectral_sensitivity(problem_name, scales, val_errs, grad_errs):
    """
    Plots Error vs Scale on a Log-Log plot to show Spectral Bias.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(scales, val_errs, 'b-o', label='Value Error ($L_2$)', linewidth=2)
    ax.plot(scales, grad_errs, 'r--s', label='Gradient Error ($L_2$)', linewidth=2)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('Scale $\sigma$ (Bandwidth)', fontsize=12)
    ax.set_ylabel('Relative Error', fontsize=12)
    ax.set_title(f'Spectral Sensitivity: {problem_name}', fontsize=14)

    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend(fontsize=11)

    # Save
    clean_name = problem_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    filename = f"Sensitivity_{clean_name}.pdf"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"   -> 💾 Plot saved: {filename}")


# ==============================================================================
# 8. MAIN
# ==============================================================================

if __name__ == "__main__":

    problems = [
        NLPoisson2D(),
        Bratu2D(lam=1.0),
        SteadyBurgers1D(nu=0.1),
        NLHelmholtz2D(k=3.0, alpha=0.5),
        AllenCahn1D(eps=0.1),
    ]

    scales = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0]
    MU = 1e-10  # Tikhonov parameter (global default)

    print("\n" + "=" * 110)
    print("  NEWTON-FAST-LSQ v2: Tikhonov + 1/√N normalization + continuation + relative convergence")
    print("=" * 110)

    all_results = []

    for problem in problems:
        print(f"\n{'='*70}")
        print(f">>> {problem.name}")
        print(f"{'='*70}")

        # Grid search (quiet)
        best = grid_search_newton(problem, scales, mu=MU, verbose=False)

        print(f"\n  Best scale: σ = {best['scale']}")
        print(f"  Re-running with verbose output...\n")

        # Re-run best with verbose
        ve, ge, t, ni, hist = run_newton_problem(
            problem, best['scale'], mu=MU, verbose=True)

        print()
        print(f"  RESULT: L2={ve:.2e}  |∇|={ge:.2e}  "
              f"iters={ni}  time={t:.3f}s")

        all_results.append({
            'name': problem.name, 'scale': best['scale'],
            'iters': ni, 'time': t, 'val_err': ve, 'grad_err': ge,
            'history': hist
        })

        # Save convergence plot
        clean = problem.name.replace(" ", "_").replace("(", "").replace(")", "")
        clean = clean.replace("=", "").replace("³", "3").replace("ε", "eps")
        clean = clean.replace("λ", "lam").replace("ν", "nu")
        plot_convergence([hist], [f'σ={best["scale"]}'],
                         problem.name, f"Newton_v2_{clean}.pdf")

    # ---- Summary Table ----
    print("\n\n" + "=" * 110)
    print("SUMMARY: Newton-Fast-LSQ v2 on Nonlinear PDEs")
    print("=" * 110)
    print(f"{'PROBLEM':<28} | {'σ':<6} | {'ITERS':<6} | {'TIME (s)':<10} | "
          f"{'VALUE L2':<12} | {'GRAD L2':<12}")
    print("-" * 110)
    for r in all_results:
        print(f"{r['name']:<28} | {r['scale']:<6.1f} | {r['iters']:<6d} | "
              f"{r['time']:<10.4f} | {r['val_err']:.2e}     | {r['grad_err']:.2e}")
    print("=" * 110)

    # ---- Comparison: Newton vs Regression ----
    print("\n\n" + "=" * 110)
    print("COMPARISON: Newton Solver Mode vs Regression Mode")
    print("=" * 110)

    for problem in [NLPoisson2D(), AllenCahn1D(eps=0.1), SteadyBurgers1D(nu=0.1)]:
        print(f"\n>>> {problem.name}")

        # Newton (solver mode)
        best_n = grid_search_newton(problem, scales, mu=MU, verbose=False)
        print(f"  Newton (solver):     L2 = {best_n['val']:.2e}  "
              f"in {best_n['time']:.3f}s  ({best_n['iters']} Newton iters)")

        # Regression mode (cheating: fit u_exact with H β = u)
        best_reg_val = float('inf')
        for scale in scales:
            torch.manual_seed(42)
            solver = build_solver_with_scale(problem.dim, scale)
            x_train = torch.rand(5000, problem.dim, device=device)
            u_train = problem.exact(x_train)
            H, _, _ = solver.get_features(x_train)
            solver.beta = solve_lstsq(H, u_train, mu=MU)
            ve, _ = evaluate_error(solver, problem)
            if ve < best_reg_val:
                best_reg_val = ve

        print(f"  Regression (cheat):  L2 = {best_reg_val:.2e}  "
              f"(uses exact solution as training data)")
