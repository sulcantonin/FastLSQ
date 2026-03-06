#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Comprehensive Benchmark: FastLSQ vs RBF Kansa vs Conventional Solvers
======================================================================

Evaluates ALL 10 PDEs from the paper (5 linear + 5 nonlinear) across three
competing approaches, highlighting FastLSQ's unique combination of accuracy
and generality:

  Method A – FastLSQ (ours)
    Sinusoidal Random Fourier Features (N = 1500 = 3 × 500), closed-form
    PDE operators, one-shot least-squares (linear) or Newton–Raphson
    (nonlinear).  Works for ANY dimension and equation type.

  Method B – RBF Kansa (Hardy Multiquadric, c = 0.5)
    Strictly positive-definite, parameter-robust meshfree method.
    Analytical Laplacian / operator applied pointwise.
    Applicable to 1-D and 2-D problems only (curse of dimensionality
    makes it infeasible for d ≥ 5 at useful accuracy).

  Method C – scikit-fem (P2 Lagrange FEM, pip-installable)
    Industry-standard Galerkin FEM with P2 triangular elements.
    Used for 2-D *elliptic* problems (Helmholtz 2D, NL-Poisson 2D,
    Bratu 2D, NL-Helmholtz 2D) with Newton–Raphson.

  Method D – scipy.integrate.solve_bvp  (MATLAB bvp4c equivalent)
    scipy's adaptive collocation BVP solver.
    Used for 1-D *steady* BVPs (Burgers 1D, Allen-Cahn 1D).

N/A limitations of conventional solvers motivating FastLSQ's design:
  • High-dimensional (Poisson 5D, Heat 5D): FEM scales as O(h^{-d}).
  • Hyperbolic space-time (Wave 1D, Maxwell 2D TM): FEM needs special
    treatment (discontinuous Galerkin, symplectic methods).
  • 1-D BVPs: scikit-fem targets 2-D+ spatial problems.
  • 2-D nonlinear: solve_bvp is a 1-D ODE solver.

PDEs benchmarked
----------------
Linear (Table 1 in paper)
  L1  Poisson 5D            –Δu = f  in [0,1]⁵
  L2  Heat 5D               u_t – κΔu = f  in [0,1]⁵ × [0,1]
  L3  Wave 1D               u_tt – c²u_xx = 0  in [0,1]²  (space-time)
  L4  Helmholtz 2D (k=10)   Δu + k²u = f  in [0,1]²
  L5  Maxwell 2D TM         u_tt – c²(u_xx+u_yy) = 0  in [0,1]³

Nonlinear Newton mode (Table 2 in paper)
  N1  NL-Poisson 2D         –Δu + u³ = f  in [0,1]²
  N2  Bratu 2D              –Δu – λeᵘ = f  in [0,1]²,  λ=1
  N3  Steady Burgers 1D     u u' – ν u'' = f  in [0,1],  ν=0.1
  N4  NL-Helmholtz 2D       Δu + k²u + αu³ = f  in [0,1]²,  k=3
  N5  Allen-Cahn 1D         ε u'' + u – u³ = f  in [0,1],  ε=0.1
"""

import os
import sys
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_bvp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── FastLSQ imports ───────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
torch.set_default_dtype(torch.float64)

from fastlsq.solvers import FastLSQSolver
from fastlsq.linalg import solve_lstsq
from fastlsq.newton import (
    newton_solve, get_initial_guess, continuation_solve,
)
from fastlsq.utils import device
from fastlsq.problems.linear import (
    PoissonND, HeatND, Wave1D, Helmholtz2D, Maxwell2D_TM,
)
from fastlsq.problems.nonlinear import (
    NLPoisson2D, Bratu2D, SteadyBurgers1D, NLHelmholtz2D, AllenCahn1D,
)

# ── scikit-fem imports ────────────────────────────────────────────
from skfem import (
    MeshTri, Basis, ElementTriP2, LinearForm, BilinearForm,
    asm, condense, solve as fem_solve,
)
from skfem.models import laplace as fem_laplace, mass as fem_mass

# ── Setup ─────────────────────────────────────────────────────────
np.random.seed(42)
torch.manual_seed(42)

PAPER_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "i")
os.makedirs(PAPER_DIR, exist_ok=True)

PI = np.pi
NA = float("nan")


# ══════════════════════════════════════════════════════════════════
#  Helper: relative L² error
# ══════════════════════════════════════════════════════════════════

def rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-15))


# ══════════════════════════════════════════════════════════════════
#  FastLSQ solver helpers
# ══════════════════════════════════════════════════════════════════

def _make_solver(dim: int, sigma: float, n_blocks: int = 3, hidden: int = 500):
    """Build a normalised FastLSQSolver with n_blocks blocks at bandwidth sigma."""
    slv = FastLSQSolver(dim, normalize=True)
    for _ in range(n_blocks):
        slv.add_block(hidden_size=hidden, scale=sigma)
    return slv


def fastlsq_linear(problem, sigma: float, n_pde: int = 8000, n_bc: int = 2000):
    """Solve a linear problem with FastLSQ.  Returns (err, time_s)."""
    slv = _make_solver(problem.dim, sigma)
    t0  = time.perf_counter()
    data = problem.get_train_data(n_pde=n_pde, n_bc=n_bc)
    if len(data) == 3:
        x_pde, bcs, f_pde = data
        A, b = problem.build(slv, x_pde, bcs, f_pde)
    else:
        x_pde, bcs = data
        A, b = problem.build(slv, x_pde, bcs)
    slv.beta = solve_lstsq(A, b)
    elapsed = time.perf_counter() - t0

    x_test  = problem.get_test_points(5000)
    u_pred  = slv.predict(x_test).cpu().numpy().ravel()
    u_true  = problem.exact(x_test).cpu().numpy().ravel()
    return rel_l2(u_pred, u_true), elapsed


def fastlsq_newton(problem, sigma: float, n_pde: int = 5000, n_bc: int = 1000,
                   max_iter: int = 30):
    """Solve a nonlinear problem with FastLSQ Newton.  Returns (err, time_s)."""
    slv = _make_solver(problem.dim, sigma)
    t0  = time.perf_counter()
    x_pde, bcs, f_pde = problem.get_train_data(n_pde=n_pde, n_bc=n_bc)

    get_initial_guess(slv, problem, x_pde, bcs, f_pde)

    if getattr(problem, "use_continuation", False):
        schedule = [v for v in problem.continuation_schedule
                    if v >= problem.nu_target]
        continuation_solve(
            slv, problem, x_pde, bcs, f_pde,
            param_name="nu", param_schedule=schedule,
            max_newton_per_step=max_iter // max(len(schedule), 1) + 5,
            verbose=False,
        )
        problem.nu = problem.nu_target
    else:
        newton_solve(slv, problem, x_pde, bcs, f_pde,
                     max_iter=max_iter, verbose=False)

    elapsed = time.perf_counter() - t0
    x_test  = problem.get_test_points(5000)
    u_pred  = slv.predict(x_test).cpu().numpy().ravel()
    u_true  = problem.exact(x_test).cpu().numpy().ravel()
    return rel_l2(u_pred, u_true), elapsed


# ══════════════════════════════════════════════════════════════════
#  RBF Kansa solver (Hardy Multiquadric, dimension-agnostic)
# ══════════════════════════════════════════════════════════════════
#
#  φ_j(x) = sqrt(c² + |x − c_j|²)
#  Δφ_j   = (d·c² + (d−1)·r²) / φ_j³          (r² = |x − c_j|²)
#  ∂φ_j/∂x_k = (x_k − c_jk) / φ_j
#  ∂²φ_j/∂x_k² = 1/φ_j − (x_k − c_jk)²/φ_j³

class MQRBFSolver:
    """Kansa collocation with Hardy Multiquadric RBF."""

    def __init__(self, d: int, n_centers: int = 1500, c: float = 0.5,
                 lam: float = 100.0, mu: float = 1e-8, seed: int = 7):
        self.d   = d
        self.c   = c
        self.lam = lam
        self.mu  = mu
        rng = np.random.default_rng(seed)

        # Interior and boundary collocation points (4:1 ratio)
        n_int = (n_centers * 4) // 5
        n_bc  = n_centers - n_int
        self.x_int = rng.random((n_int * 4, d))   # 4× oversampled
        self.x_bc  = self._boundary(d, n_bc * 4, rng)

        # RBF centres (kept separate for stability)
        self.centers = np.vstack([rng.random((n_int, d)),
                                  self._boundary(d, n_bc, rng)])
        self.N       = self.centers.shape[0]

        # Pre-compute matrices
        self.Phi_int  = self._phi(self.x_int)     # shape (4n_int, N)
        self.L_int    = self._lap(self.x_int)     # Δ operator
        self.Phi_bc   = self._phi(self.x_bc)      # shape (4n_bc, N)

    @staticmethod
    def _boundary(d: int, n: int, rng) -> np.ndarray:
        if d == 1:
            h = n // 2
            return np.vstack([np.zeros((h, 1)), np.ones((n - h, 1))])
        n_face = n // (2 * d)
        pts = []
        for dim in range(d):
            for val in (0.0, 1.0):
                p = rng.random((n_face, d))
                p[:, dim] = val
                pts.append(p)
        return np.vstack(pts)[:n]

    def _r2(self, x: np.ndarray) -> np.ndarray:
        diff = x[:, None, :] - self.centers[None, :, :]   # (M, N, d)
        return np.sum(diff ** 2, axis=2)                   # (M, N)

    def _phi(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(self.c ** 2 + self._r2(x))

    def _lap(self, x: np.ndarray) -> np.ndarray:
        """Laplacian: (d·c² + (d−1)·r²) / φ³"""
        r2  = self._r2(x)
        phi = np.sqrt(self.c ** 2 + r2)
        return (self.d * self.c ** 2 + (self.d - 1) * r2) / phi ** 3

    def phi_test(self, x_test: np.ndarray) -> np.ndarray:
        """Evaluate φ at arbitrary test points."""
        r2 = np.sum((x_test[:, None, :] - self.centers[None, :, :]) ** 2, axis=2)
        return np.sqrt(self.c ** 2 + r2)

    def lstsq(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Tikhonov-regularised normal equations: (AᵀA + μI)x = Aᵀb."""
        AtA = A.T @ A
        AtA.flat[:: A.shape[1] + 1] += self.mu
        return np.linalg.solve(AtA, A.T @ b)


# ══════════════════════════════════════════════════════════════════
#  scikit-fem P2 FEM helpers
# ══════════════════════════════════════════════════════════════════

def _fem_basis(refine: int = 4):
    mesh = MeshTri.init_sqsymmetric().refined(refine)
    return Basis(mesh, ElementTriP2())


def fem_linear(f_fn, exact_fn, k2: float = 0.0, refine: int = 6):
    """FEM solve for  (Δ + k²)u = f  with Dirichlet BCs.

    For k2 = 0 this reduces to  Δu = f  (Poisson), using K·u = f_vec.
    For k2 ≠ 0 (Helmholtz), the weak form is  –∫∇u·∇v + k²∫u·v = ∫f·v,
    giving the system  (k²M – K)·u = f_vec.

    Returns (err, time_s, n_dof).
    """
    t0 = time.perf_counter()
    b  = _fem_basis(refine)
    K  = asm(fem_laplace, b)
    M  = asm(fem_mass, b)
    K_op = (k2 * M - K) if k2 != 0.0 else K

    @LinearForm
    def f_form(v, w):
        return f_fn(w.x[0], w.x[1]) * v

    f_vec = asm(f_form, b)
    x_bc  = exact_fn(b.doflocs[0], b.doflocs[1])
    D     = b.get_dofs().all()
    K_c, f_c, u0, I = condense(K_op, f_vec, x=x_bc, D=D)
    u0[I] = fem_solve(K_c, f_c)
    elapsed = time.perf_counter() - t0

    u_exact = exact_fn(b.doflocs[0], b.doflocs[1])
    return rel_l2(u0, u_exact), elapsed, b.N


def fem_newton(f_fn, exact_fn, nl_r_fn, nl_j_fn,
               k2: float = 0.0, refine: int = 5,
               max_iter: int = 50, tol: float = 1e-14):
    """Newton–FEM for  L·u + N(u) = f  where L = (Δ+k²) or −Δ.

    nl_r_fn(u_h, x, y) → N(u_h)           (nonlinear residual per point)
    nl_j_fn(u_h, x, y) → ∂N/∂u at u_h    (Jacobian coefficient)

    Returns (err, time_s, n_dof).
    """
    t0 = time.perf_counter()
    b  = _fem_basis(refine)
    K  = asm(fem_laplace, b)
    M  = asm(fem_mass, b)
    K_lin = (k2 * M - K) if k2 != 0.0 else K   # linear operator

    @LinearForm
    def f_form(v, w):
        return f_fn(w.x[0], w.x[1]) * v

    f_vec = asm(f_form, b)
    D     = b.get_dofs().all()
    x_bc  = exact_fn(b.doflocs[0], b.doflocs[1])

    # Warm start: solve the linearised (no nonlinear term) problem
    K_c, f_c, u0, I = condense(K_lin, f_vec, x=x_bc, D=D)
    u0[I] = fem_solve(K_c, f_c)
    u = u0.copy()

    for _ in range(max_iter):
        u_q = b.interpolate(u)

        @LinearForm
        def res_nl(v, w):
            return nl_r_fn(w["u_h"].value, w.x[0], w.x[1]) * v

        @BilinearForm
        def jac_nl(u_t, v, w):
            return nl_j_fn(w["u_h"].value, w.x[0], w.x[1]) * u_t * v

        R_nl = asm(res_nl, b, u_h=u_q)
        J_nl = asm(jac_nl, b, u_h=u_q)

        R  = K_lin @ u + R_nl - f_vec
        J  = K_lin + J_nl
        J_c, R_c, du, I2 = condense(J, -R, x=np.zeros_like(u), D=D)
        du[I2] = fem_solve(J_c, R_c)
        u += du
        if np.linalg.norm(du[I2]) < tol:
            break

    elapsed = time.perf_counter() - t0
    u_exact = exact_fn(b.doflocs[0], b.doflocs[1])
    return rel_l2(u, u_exact), elapsed, b.N


# ══════════════════════════════════════════════════════════════════
#  scipy.integrate.solve_bvp helper
# ══════════════════════════════════════════════════════════════════

def bvp_solve(ode_fn, bc_fn, exact_fn, tol: float = 1e-10, max_nodes: int = 100000):
    """Wrap scipy.integrate.solve_bvp.  Returns (err, time_s, n_nodes)."""
    t0    = time.perf_counter()
    x0    = np.linspace(0.0, 1.0, 50)
    y0    = np.zeros((2, x0.size))
    sol   = solve_bvp(ode_fn, bc_fn, x0, y0, tol=tol, max_nodes=max_nodes)
    elapsed = time.perf_counter() - t0
    x_t   = np.linspace(0.0, 1.0, 5000)
    u_pred = sol.sol(x_t)[0]
    u_true = exact_fn(x_t)
    return rel_l2(u_pred, u_true), elapsed, sol.x.size


# ══════════════════════════════════════════════════════════════════
#  Per-problem exact solutions and sources
# ══════════════════════════════════════════════════════════════════

# ── Helmholtz 2D  (Δu + k²u = f, u = sin(kx)sin(ky), k=10) ───────
K_HELM = 10.0

def helm_exact(x, y):   return np.sin(K_HELM * x) * np.sin(K_HELM * y)
def helm_src(x, y):     return -(K_HELM ** 2) * helm_exact(x, y)

# ── NL-Poisson 2D  (–Δu + u³ = f, u = sin(πx)sin(πy)) ────────────

def nlp_exact(x, y):
    return np.sin(PI * x) * np.sin(PI * y)
def nlp_src(x, y):
    u = nlp_exact(x, y)
    return 2 * PI ** 2 * u + u ** 3

# ── Bratu 2D  (–Δu – λeᵘ = f, u = sin(πx)sin(πy), λ=1) ──────────
LAM_BRATU = 1.0

def bratu_exact(x, y):  return np.sin(PI * x) * np.sin(PI * y)
def bratu_src(x, y):
    u = bratu_exact(x, y)
    return 2 * PI ** 2 * u - LAM_BRATU * np.exp(u)

# ── NL-Helmholtz 2D  (Δu + k²u + αu³ = f, u = sin(kx)sin(ky), k=3)
K_NLH = 3.0;  A_NLH = 0.5

def nlh_exact(x, y):    return np.sin(K_NLH * x) * np.sin(K_NLH * y)
def nlh_src(x, y):
    u = nlh_exact(x, y)
    return -(K_NLH ** 2) * u + A_NLH * u ** 3

# ── Burgers 1D  (u u' – ν u'' = f, u = sin(2πx), ν=0.1) ──────────
NU_BURG = 0.1

def burg_exact(x):    return np.sin(2 * PI * x)
def burg_ux(x):       return 2 * PI * np.cos(2 * PI * x)
def burg_uxx(x):      return -(2 * PI) ** 2 * np.sin(2 * PI * x)
def burg_src(x):      return burg_exact(x) * burg_ux(x) - NU_BURG * burg_uxx(x)

# ── Allen-Cahn 1D  (ε u'' + u – u³ = f, u = sin(πx), ε=0.1) ──────
EPS_AC = 0.1

def ac_exact(x):   return np.sin(PI * x)
def ac_src(x):
    u = ac_exact(x)
    return EPS_AC * (-(PI ** 2) * u) + u - u ** 3


# ══════════════════════════════════════════════════════════════════
#  Sigma values (paper Tables 1 and 2)
# ══════════════════════════════════════════════════════════════════

SIGMA = {
    "Poisson 5D":       5.0,   # paper Table 1 optimal
    "Heat 5D":          1.0,   # tuned for 6-D space-time
    "Wave 1D":         20.0,   # needs high σ for 8π max frequency
    "Helmholtz 2D":    10.0,   # matches k=10 frequency
    "Maxwell 2D TM":    5.0,   # paper Table 1 optimal
    "NL-Poisson 2D":    3.0,   # paper Table 2
    "Bratu 2D":         2.0,   # paper Table 2
    "Burgers 1D":      10.0,   # paper Table 2 (continuation)
    "NL-Helmholtz 2D":  5.0,   # paper Table 2
    "Allen-Cahn 1D":    3.0,   # paper Table 2
}

RESULTS: dict = {}


# ══════════════════════════════════════════════════════════════════
#  Orchestrator
# ══════════════════════════════════════════════════════════════════

def _record(name, key, fn):
    """Run fn(), catch exceptions, and store the result."""
    if fn is None:
        RESULTS[name][key] = (NA, NA, "N/A")
        print(f"    {key:6s}: N/A")
        return
    try:
        out = fn()
        e, t = out[:2]
        dof  = out[2] if len(out) > 2 else "—"
        RESULTS[name][key] = (e, t, dof)
        print(f"    {key:6s}: err={e:.2e}  t={t:.2f}s  dof={dof}")
    except Exception as ex:
        RESULTS[name][key] = (NA, NA, f"ERR:{ex}")
        print(f"    {key:6s}: ERROR {ex}")


def run_all():
    print("=" * 70)
    print("  FastLSQ vs RBF Kansa vs scikit-fem P2 / scipy.solve_bvp")
    print("  ALL 10 PDEs from the paper  (N = 1500 features for FastLSQ/RBF)")
    print("=" * 70)

    # ── L1: Poisson 5D ──────────────────────────────────────────────
    print("\n  [Poisson 5D]")
    RESULTS["Poisson 5D"] = {}
    prob = PoissonND()
    rbf  = MQRBFSolver(d=5, n_centers=1500)

    def _rbf_p5d():
        t0 = time.perf_counter()
        # Source: –Δu = Σ (π/2)² sin(πxᵢ/2) = (5π²/4) Σ sin(πxᵢ/2)
        f_int = (PI ** 2 / 4.0) * np.sum(np.sin(PI / 2 * rbf.x_int), axis=1)
        # Boundary values at collocation BC points
        u_bc  = np.sum(np.sin(PI / 2 * rbf.x_bc), axis=1)
        A = np.vstack([-rbf.L_int,
                       rbf.lam * rbf.Phi_bc])
        b = np.concatenate([f_int, rbf.lam * u_bc])
        alpha = rbf.lstsq(A, b)
        elapsed = time.perf_counter() - t0
        x_t    = np.random.rand(2000, 5)
        u_pred = rbf.phi_test(x_t) @ alpha
        u_true = np.sum(np.sin(PI / 2 * x_t), axis=1)
        return rel_l2(u_pred, u_true), elapsed

    _record("Poisson 5D", "flsq",
            lambda: fastlsq_linear(prob, SIGMA["Poisson 5D"]))
    _record("Poisson 5D", "rbf",  _rbf_p5d)
    _record("Poisson 5D", "conv", None)

    # ── L2: Heat 5D ─────────────────────────────────────────────────
    print("\n  [Heat 5D]")
    RESULTS["Heat 5D"] = {}
    prob = HeatND()
    _record("Heat 5D", "flsq",
            lambda: fastlsq_linear(prob, SIGMA["Heat 5D"]))
    _record("Heat 5D", "rbf",  None)   # 6-D space-time; boundary data complex
    _record("Heat 5D", "conv", None)

    # ── L3: Wave 1D ─────────────────────────────────────────────────
    print("\n  [Wave 1D]")
    RESULTS["Wave 1D"] = {}
    prob = Wave1D()
    _record("Wave 1D", "flsq",
            lambda: fastlsq_linear(prob, SIGMA["Wave 1D"]))
    _record("Wave 1D", "rbf",  None)   # hyperbolic space-time
    _record("Wave 1D", "conv", None)

    # ── L4: Helmholtz 2D ────────────────────────────────────────────
    print("\n  [Helmholtz 2D]")
    RESULTS["Helmholtz 2D"] = {}
    prob     = Helmholtz2D()
    rbf_helm = MQRBFSolver(d=2, n_centers=1500)

    def _rbf_helm():
        t0 = time.perf_counter()
        A_pde = rbf_helm.L_int + K_HELM ** 2 * rbf_helm.Phi_int
        f_int = helm_src(rbf_helm.x_int[:, 0], rbf_helm.x_int[:, 1])
        u_bc  = helm_exact(rbf_helm.x_bc[:, 0], rbf_helm.x_bc[:, 1])
        A = np.vstack([A_pde, rbf_helm.lam * rbf_helm.Phi_bc])
        b = np.concatenate([f_int, rbf_helm.lam * u_bc])
        alpha = rbf_helm.lstsq(A, b)
        elapsed = time.perf_counter() - t0
        x_t = np.random.rand(5000, 2)
        return rel_l2(rbf_helm.phi_test(x_t) @ alpha,
                      helm_exact(x_t[:, 0], x_t[:, 1])), elapsed

    # P2 FEM: refine=6 gives ~64 subdivisions for k=10 (high-frequency Helmholtz)
    _record("Helmholtz 2D", "flsq",
            lambda: fastlsq_linear(prob, SIGMA["Helmholtz 2D"]))
    _record("Helmholtz 2D", "rbf",  _rbf_helm)
    _record("Helmholtz 2D", "conv",
            lambda: fem_linear(helm_src, helm_exact, k2=K_HELM ** 2, refine=6))

    # ── L5: Maxwell 2D TM ───────────────────────────────────────────
    print("\n  [Maxwell 2D TM]")
    RESULTS["Maxwell 2D TM"] = {}
    prob = Maxwell2D_TM()
    _record("Maxwell 2D TM", "flsq",
            lambda: fastlsq_linear(prob, SIGMA["Maxwell 2D TM"],
                                   n_pde=5000, n_bc=1000))
    _record("Maxwell 2D TM", "rbf",  None)
    _record("Maxwell 2D TM", "conv", None)

    # ── N1: NL-Poisson 2D ───────────────────────────────────────────
    print("\n  [NL-Poisson 2D]")
    RESULTS["NL-Poisson 2D"] = {}
    prob    = NLPoisson2D()
    rbf_nlp = MQRBFSolver(d=2, n_centers=900, mu=1e-6)

    def _rbf_nlp():
        t0   = time.perf_counter()
        H    = rbf_nlp.Phi_int;  Lp = -rbf_nlp.L_int;  H_bc = rbf_nlp.Phi_bc
        f    = nlp_src(rbf_nlp.x_int[:, 0], rbf_nlp.x_int[:, 1])
        u_bc = nlp_exact(rbf_nlp.x_bc[:, 0], rbf_nlp.x_bc[:, 1])
        lam  = rbf_nlp.lam
        # Warm start: linear solve
        A0 = np.vstack([Lp, lam * H_bc])
        b0 = np.concatenate([
            2 * PI**2 * nlp_exact(rbf_nlp.x_int[:,0], rbf_nlp.x_int[:,1]),
            lam * u_bc])
        alpha = rbf_nlp.lstsq(A0, b0)
        for _ in range(30):
            u_c = H @ alpha
            R   = Lp @ alpha + u_c ** 3 - f
            R_b = lam * (H_bc @ alpha - u_bc)
            J   = np.vstack([Lp + 3 * u_c[:, None] * H, lam * H_bc])
            rhs = -np.concatenate([R, R_b])
            d   = rbf_nlp.lstsq(J, rhs)
            # Backtracking line search
            res0 = np.linalg.norm(rhs)
            step = 1.0
            for _ in range(8):
                a_try = alpha + step * d
                R_try = Lp @ a_try + (H @ a_try) ** 3 - f
                R_b_try = lam * (H_bc @ a_try - u_bc)
                if np.linalg.norm(np.concatenate([R_try, R_b_try])) < res0:
                    break
                step *= 0.5
            alpha += step * d
            if np.linalg.norm(d) / (np.linalg.norm(alpha) + 1e-12) < 1e-9:
                break
        elapsed = time.perf_counter() - t0
        x_t = np.random.rand(5000, 2)
        return rel_l2(rbf_nlp.phi_test(x_t) @ alpha,
                      nlp_exact(x_t[:, 0], x_t[:, 1])), elapsed

    _record("NL-Poisson 2D", "flsq",
            lambda: fastlsq_newton(prob, SIGMA["NL-Poisson 2D"]))
    _record("NL-Poisson 2D", "rbf",  _rbf_nlp)
    _record("NL-Poisson 2D", "conv",
            lambda: fem_newton(
                f_fn=nlp_src, exact_fn=nlp_exact,
                nl_r_fn=lambda u, x, y: u ** 3,
                nl_j_fn=lambda u, x, y: 3 * u ** 2,
                refine=5))

    # ── N2: Bratu 2D ────────────────────────────────────────────────
    print("\n  [Bratu 2D]")
    RESULTS["Bratu 2D"] = {}
    prob     = Bratu2D()
    rbf_bra  = MQRBFSolver(d=2, n_centers=900, mu=1e-6)

    def _rbf_bratu():
        t0   = time.perf_counter()
        H    = rbf_bra.Phi_int;  Lp = -rbf_bra.L_int;  H_bc = rbf_bra.Phi_bc
        f    = bratu_src(rbf_bra.x_int[:, 0], rbf_bra.x_int[:, 1])
        u_bc = bratu_exact(rbf_bra.x_bc[:, 0], rbf_bra.x_bc[:, 1])
        lam  = rbf_bra.lam
        A0   = np.vstack([Lp, lam * H_bc])
        b0   = np.concatenate([f, lam * u_bc])
        alpha = rbf_bra.lstsq(A0, b0)
        for _ in range(30):
            u_c = H @ alpha
            eu  = np.exp(np.clip(u_c, -30, 30))
            R   = Lp @ alpha - LAM_BRATU * eu - f
            R_b = lam * (H_bc @ alpha - u_bc)
            J   = np.vstack([Lp - LAM_BRATU * eu[:, None] * H, lam * H_bc])
            rhs = -np.concatenate([R, R_b])
            d   = rbf_bra.lstsq(J, rhs)
            res0 = np.linalg.norm(rhs)
            step = 1.0
            for _ in range(8):
                a_try  = alpha + step * d
                eu_try = np.exp(np.clip(H @ a_try, -30, 30))
                R_t    = Lp @ a_try - LAM_BRATU * eu_try - f
                R_b_t  = lam * (H_bc @ a_try - u_bc)
                if np.linalg.norm(np.concatenate([R_t, R_b_t])) < res0:
                    break
                step *= 0.5
            alpha += step * d
            if np.linalg.norm(d) / (np.linalg.norm(alpha) + 1e-12) < 1e-9:
                break
        elapsed = time.perf_counter() - t0
        x_t = np.random.rand(5000, 2)
        return rel_l2(rbf_bra.phi_test(x_t) @ alpha,
                      bratu_exact(x_t[:, 0], x_t[:, 1])), elapsed

    def _fem_bratu_nl_r(u, x, y):
        return -LAM_BRATU * np.exp(np.clip(u, -30, 30))

    def _fem_bratu_nl_j(u, x, y):
        return -LAM_BRATU * np.exp(np.clip(u, -30, 30))

    _record("Bratu 2D", "flsq",
            lambda: fastlsq_newton(prob, SIGMA["Bratu 2D"]))
    _record("Bratu 2D", "rbf",  _rbf_bratu)
    _record("Bratu 2D", "conv",
            lambda: fem_newton(
                f_fn=bratu_src, exact_fn=bratu_exact,
                nl_r_fn=_fem_bratu_nl_r,
                nl_j_fn=_fem_bratu_nl_j,
                refine=5))

    # ── N3: Steady Burgers 1D ───────────────────────────────────────
    print("\n  [Burgers 1D]")
    RESULTS["Burgers 1D"] = {}
    prob    = SteadyBurgers1D()
    rbf_bg  = MQRBFSolver(d=1, n_centers=250, mu=1e-6)

    def _rbf_burgers():
        t0   = time.perf_counter()
        H    = rbf_bg.Phi_int;  Lp1D = -rbf_bg.L_int;  H_bc = rbf_bg.Phi_bc
        # ∂φ/∂x for 1-D
        diff  = rbf_bg.x_int[:, None, 0] - rbf_bg.centers[None, :, 0]  # (M,N)
        r2    = diff ** 2
        phi   = np.sqrt(rbf_bg.c ** 2 + r2)
        dPhi  = diff / phi
        u_bc  = np.zeros(len(rbf_bg.x_bc))
        lam   = rbf_bg.lam
        alpha = np.zeros(rbf_bg.N)
        for nu_s in [1.0, 0.5, 0.2, NU_BURG]:
            f_s = (burg_exact(rbf_bg.x_int[:, 0]) * burg_ux(rbf_bg.x_int[:, 0])
                   - nu_s * burg_uxx(rbf_bg.x_int[:, 0]))
            for _ in range(30):
                u_c  = H @ alpha
                ux_c = dPhi @ alpha
                R    = u_c * ux_c - nu_s * (Lp1D @ alpha) - f_s
                R_b  = lam * (H_bc @ alpha - u_bc)
                J    = np.vstack([
                    u_c[:, None] * dPhi + ux_c[:, None] * H - nu_s * Lp1D,
                    lam * H_bc])
                d    = rbf_bg.lstsq(J, -np.concatenate([R, R_b]))
                alpha += d
                if np.linalg.norm(d) / (np.linalg.norm(alpha) + 1e-12) < 1e-11:
                    break
        elapsed = time.perf_counter() - t0
        x_t = np.linspace(0, 1, 5000).reshape(-1, 1)
        return rel_l2(rbf_bg.phi_test(x_t) @ alpha, burg_exact(x_t[:, 0])), elapsed

    def _bvp_burgers():
        def ode(x, y):
            f = burg_src(x)
            return np.vstack([y[1], (y[0] * y[1] - f) / NU_BURG])
        def bc(ya, yb):
            return np.array([ya[0], yb[0]])
        return bvp_solve(ode, bc, burg_exact)

    _record("Burgers 1D", "flsq",
            lambda: fastlsq_newton(prob, SIGMA["Burgers 1D"],
                                   n_pde=3000, n_bc=200))
    _record("Burgers 1D", "rbf",  _rbf_burgers)
    _record("Burgers 1D", "conv", _bvp_burgers)

    # ── N4: NL-Helmholtz 2D ─────────────────────────────────────────
    print("\n  [NL-Helmholtz 2D]")
    RESULTS["NL-Helmholtz 2D"] = {}
    prob    = NLHelmholtz2D()
    rbf_nlh = MQRBFSolver(d=2, n_centers=900, mu=1e-6)

    def _rbf_nlhelmholtz():
        t0   = time.perf_counter()
        H    = rbf_nlh.Phi_int;  L = rbf_nlh.L_int;  H_bc = rbf_nlh.Phi_bc
        f    = nlh_src(rbf_nlh.x_int[:, 0], rbf_nlh.x_int[:, 1])
        u_bc = nlh_exact(rbf_nlh.x_bc[:, 0], rbf_nlh.x_bc[:, 1])
        lam  = rbf_nlh.lam
        A_lin = L + K_NLH ** 2 * H
        A0    = np.vstack([A_lin, lam * H_bc])
        b0    = np.concatenate([f, lam * u_bc])
        alpha = rbf_nlh.lstsq(A0, b0)
        for _ in range(25):
            u_c = H @ alpha
            R   = L @ alpha + K_NLH ** 2 * (H @ alpha) + A_NLH * u_c ** 3 - f
            R_b = lam * (H_bc @ alpha - u_bc)
            J   = np.vstack([
                A_lin + 3 * A_NLH * u_c[:, None] * H,
                lam * H_bc])
            d   = rbf_nlh.lstsq(J, -np.concatenate([R, R_b]))
            alpha += d
            if np.linalg.norm(d) / (np.linalg.norm(alpha) + 1e-12) < 1e-10:
                break
        elapsed = time.perf_counter() - t0
        x_t = np.random.rand(5000, 2)
        return rel_l2(rbf_nlh.phi_test(x_t) @ alpha,
                      nlh_exact(x_t[:, 0], x_t[:, 1])), elapsed

    # NL-Helmholtz: linear part = (Δ+k²)u, use k²M–K; nonlinear = αu³
    _record("NL-Helmholtz 2D", "flsq",
            lambda: fastlsq_newton(prob, SIGMA["NL-Helmholtz 2D"]))
    _record("NL-Helmholtz 2D", "rbf",  _rbf_nlhelmholtz)
    _record("NL-Helmholtz 2D", "conv",
            lambda: fem_newton(
                f_fn=nlh_src, exact_fn=nlh_exact,
                nl_r_fn=lambda u, x, y: A_NLH * u ** 3,
                nl_j_fn=lambda u, x, y: 3 * A_NLH * u ** 2,
                k2=K_NLH ** 2, refine=5))

    # ── N5: Allen-Cahn 1D ───────────────────────────────────────────
    print("\n  [Allen-Cahn 1D]")
    RESULTS["Allen-Cahn 1D"] = {}
    prob    = AllenCahn1D()
    rbf_ac  = MQRBFSolver(d=1, n_centers=250, mu=1e-6)

    def _rbf_ac():
        t0   = time.perf_counter()
        H    = rbf_ac.Phi_int;  L1D = rbf_ac.L_int;  H_bc = rbf_ac.Phi_bc
        f    = ac_src(rbf_ac.x_int[:, 0])
        u_bc = np.zeros(len(rbf_ac.x_bc))
        lam  = rbf_ac.lam
        A0   = np.vstack([EPS_AC * L1D + H, lam * H_bc])
        b0   = np.concatenate([f, lam * u_bc])
        alpha = rbf_ac.lstsq(A0, b0)
        for _ in range(25):
            u_c = H @ alpha
            R   = EPS_AC * (L1D @ alpha) + u_c - u_c ** 3 - f
            R_b = lam * (H_bc @ alpha - u_bc)
            J   = np.vstack([
                EPS_AC * L1D + (1 - 3 * u_c ** 2)[:, None] * H,
                lam * H_bc])
            d   = rbf_ac.lstsq(J, -np.concatenate([R, R_b]))
            alpha += d
            if np.linalg.norm(d) / (np.linalg.norm(alpha) + 1e-12) < 1e-10:
                break
        elapsed = time.perf_counter() - t0
        x_t = np.linspace(0, 1, 5000).reshape(-1, 1)
        return rel_l2(rbf_ac.phi_test(x_t) @ alpha, ac_exact(x_t[:, 0])), elapsed

    def _bvp_ac():
        def ode(x, y):
            f = ac_src(x)
            return np.vstack([y[1], -(y[0] - y[0] ** 3 - f) / EPS_AC])
        def bc(ya, yb):
            return np.array([ya[0], yb[0]])
        return bvp_solve(ode, bc, ac_exact)

    _record("Allen-Cahn 1D", "flsq",
            lambda: fastlsq_newton(prob, SIGMA["Allen-Cahn 1D"],
                                   n_pde=3000, n_bc=200))
    _record("Allen-Cahn 1D", "rbf",  _rbf_ac)
    _record("Allen-Cahn 1D", "conv", _bvp_ac)

    # ── Summary ─────────────────────────────────────────────────────
    _print_table()
    _plot()
    print("\nDone.")


# ══════════════════════════════════════════════════════════════════
#  Pretty table
# ══════════════════════════════════════════════════════════════════

PROBLEM_ORDER = [
    ("Poisson 5D",      "L",  "P5D"),
    ("Heat 5D",         "L",  "H5D"),
    ("Wave 1D",         "L",  "W1D"),
    ("Helmholtz 2D",    "L",  "Helm"),
    ("Maxwell 2D TM",   "L",  "Max"),
    ("NL-Poisson 2D",   "NL", "NLP"),
    ("Bratu 2D",        "NL", "Bratu"),
    ("Burgers 1D",      "NL", "Burg"),
    ("NL-Helmholtz 2D", "NL", "NLH"),
    ("Allen-Cahn 1D",   "NL", "AC"),
]

COL_LABELS = {
    "flsq": "FastLSQ",
    "rbf":  "RBF MQ",
    "conv": "scikit-fem P2 / solve_bvp",
}


def _fmt(e, t, dof):
    if np.isnan(e):
        return "N/A"
    return f"{e:.1e} ({t:.1f}s)"


def _print_table():
    print("\n\n" + "=" * 82)
    print("  BENCHMARK RESULTS  (rel. L² error + CPU time)")
    print("  scikit-fem: 2-D elliptic only; solve_bvp: 1-D BVP only")
    print("=" * 82)
    hdr = f"  {'Problem':<22} {'FastLSQ':>20}  {'RBF MQ':>20}  {'scikit-fem/bvp':>22}"
    print(hdr)
    print("  " + "─" * 78)
    for name, kind, _ in PROBLEM_ORDER:
        if name not in RESULTS:
            continue
        row = RESULTS[name]
        print(f"  {name:<22}"
              f" {_fmt(*row['flsq']):>20}"
              f"  {_fmt(*row['rbf']):>20}"
              f"  {_fmt(*row['conv']):>22}")
    print()

    # LaTeX
    print("  LaTeX rows:")
    print("  " + "─" * 78)
    for name, kind, _ in PROBLEM_ORDER:
        if name not in RESULTS:
            continue
        row = RESULTS[name]
        errs = [row[k][0] for k in ("flsq", "rbf", "conv")]
        valid = [(i, e) for i, e in enumerate(errs) if not np.isnan(e)]
        best  = min(valid, key=lambda x: x[1])[0] if valid else -1
        cells = []
        for i, k in enumerate(("flsq", "rbf", "conv")):
            e, t, _ = row[k]
            if np.isnan(e):
                cells.append("---")
            else:
                s = f"{e:.1e}"
                cells.append(f"\\mathbf{{{s}}}" if i == best else s)
        print(f"  {name:<25} & " + " & ".join(cells) + " \\\\")


# ══════════════════════════════════════════════════════════════════
#  Figure
# ══════════════════════════════════════════════════════════════════

def _plot():
    names  = [n for n, _, _ in PROBLEM_ORDER if n in RESULTS]
    labels = [a for n, _, a in PROBLEM_ORDER if n in RESULTS]
    n      = len(names)
    x      = np.arange(n)
    w      = 0.26
    keys   = ["flsq", "rbf", "conv"]
    colors = ["#1565C0", "#E65100", "#2E7D32"]
    offs   = [-w, 0, w]

    fig, (ax_e, ax_grid) = plt.subplots(
        1, 2, figsize=(17, 5.5), constrained_layout=True)

    # ─ Error bars ───────────────────────────────────────────────────
    for key, color, off in zip(keys, colors, offs):
        errs = []
        for nm in names:
            e = RESULTS[nm][key][0] if nm in RESULTS else NA
            errs.append(e)
        e_plot = [e if not np.isnan(e) else 0 for e in errs]
        bars = ax_e.bar(x + off, e_plot, w, label=COL_LABELS[key],
                        color=color, alpha=0.82, edgecolor="k", linewidth=0.4)
        for bar, e in zip(bars, errs):
            if np.isnan(e):
                bar.set_hatch("///")
                bar.set_facecolor("#BDBDBD")
                bar.set_edgecolor("#757575")

    ax_e.set_yscale("log")
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax_e.set_ylabel(r"Relative $L^2$ error", fontsize=10)
    ax_e.set_title("Accuracy (hatched bars = not applicable)",
                   fontsize=10, fontweight="bold")
    ax_e.legend(fontsize=8, framealpha=0.9, loc="upper right")
    ax_e.grid(True, axis="y", ls="--", alpha=0.4)

    # Shade linear vs nonlinear background
    lin_idx = [i for i, (nm, k, _) in enumerate(PROBLEM_ORDER)
               if k == "L" and nm in RESULTS]
    nl_idx  = [i for i, (nm, k, _) in enumerate(PROBLEM_ORDER)
               if k == "NL" and nm in RESULTS]
    if lin_idx:
        ax_e.axvspan(lin_idx[0] - 0.45, lin_idx[-1] + 0.45,
                     alpha=0.06, color="#1565C0", zorder=0)
        ax_e.text(np.mean(lin_idx), ax_e.get_ylim()[0] * 1.2,
                  "Linear PDEs", ha="center", fontsize=8,
                  color="#1565C0", style="italic")
    if nl_idx:
        ax_e.axvspan(nl_idx[0] - 0.45, nl_idx[-1] + 0.45,
                     alpha=0.06, color="#E65100", zorder=0)
        ax_e.text(np.mean(nl_idx), ax_e.get_ylim()[0] * 1.2,
                  "Nonlinear PDEs", ha="center", fontsize=8,
                  color="#E65100", style="italic")

    # ─ Applicability grid ───────────────────────────────────────────
    avail = np.array([
        [not np.isnan(RESULTS[nm][k][0]) if nm in RESULTS else False
         for k in keys]
        for nm in names], dtype=float)

    ax_grid.imshow(avail.T, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax_grid.set_xticks(range(n))
    ax_grid.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax_grid.set_yticks([0, 1, 2])
    ax_grid.set_yticklabels(["FastLSQ", "RBF MQ", "scikit-fem / solve_bvp"],
                             fontsize=9)
    ax_grid.set_title("Applicability  (✓ = solved)", fontsize=10, fontweight="bold")
    for i in range(n):
        for j in range(3):
            sym  = "✓" if avail[i, j] else "—"
            col  = "#1B5E20" if avail[i, j] else "#9E9E9E"
            ax_grid.text(i, j, sym, ha="center", va="center",
                         fontsize=12, color=col, fontweight="bold")

    fig.suptitle(
        "FastLSQ vs RBF Kansa vs scikit-fem P2 / scipy.solve_bvp  ·  "
        "All 10 PDEs from the paper",
        fontsize=11, fontweight="bold")

    outpath = os.path.join(PAPER_DIR, "Benchmark_Comparison.pdf")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved → {outpath}")


if __name__ == "__main__":
    run_all()
