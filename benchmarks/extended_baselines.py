#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License.

"""Extended baseline comparisons for the JCP submission.

Adds four pieces of evidence the original benchmark_comparison.py doesn't:

    (1) PIELM (tanh, one-shot) on the same 10 PDEs as FastLSQ.
    (2) RBF Kansa shape-parameter sweep on the four problems where RBF runs.
    (3) Vanilla PINN (PyTorch, 10k Adam iters) on 3 representative problems.
    (4) Chebyshev pseudospectral collocation on Burgers 1D (smooth 1-D BVP).

Each block writes a row to benchmarks/results/extended_baselines.csv that the
JCP paper Table can consume.
"""

from __future__ import annotations

import os
import sys
import time
import csv
import math
import numpy as np
import torch
from pathlib import Path
from scipy.optimize import fsolve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastlsq import FastLSQSolver, PIELMSolver, solve_lstsq  # noqa: E402
from fastlsq.problems.linear import (  # noqa: E402
    PoissonND, HeatND, Wave1D, Helmholtz2D, Maxwell2D_TM,
)
from fastlsq.problems.nonlinear import (  # noqa: E402
    NLPoisson2D, Bratu2D, SteadyBurgers1D, NLHelmholtz2D, AllenCahn1D,
)

# Use the helper Newton from FastLSQ to keep PIELM nonlinear fair
from fastlsq.newton import newton_solve, get_initial_guess, continuation_solve  # noqa: E402

torch.set_default_dtype(torch.float32)
PI = math.pi
NA = float("nan")
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "extended_baselines.csv"


def rel_l2(pred, true):
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-15))


def _csv_write(rows):
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["problem", "method", "config", "rel_l2", "time_s", "dof", "notes"])
        for r in rows:
            w.writerow(r)


# ====================================================================
# (1) PIELM on all 10 PDEs
# ====================================================================

SIGMA = {
    "Poisson 5D":      5.0,
    "Heat 5D":         1.0,
    "Wave 1D":        20.0,
    "Helmholtz 2D":   10.0,
    "Maxwell 2D TM":   5.0,
    "NL-Poisson 2D":   3.0,
    "Bratu 2D":        2.0,
    "Burgers 1D":     10.0,
    "NL-Helmholtz 2D": 5.0,
    "Allen-Cahn 1D":   3.0,
}


def _build_pielm(d, sigma, n_blocks=3, hidden=500):
    slv = PIELMSolver(d)
    for _ in range(n_blocks):
        slv.add_block(hidden_size=hidden, scale=sigma)
    return slv


def pielm_linear(problem, sigma, n_pde=8000, n_bc=2000):
    slv = _build_pielm(problem.dim, sigma)
    t0 = time.perf_counter()
    data = problem.get_train_data(n_pde=n_pde, n_bc=n_bc)
    if len(data) == 3:
        x_pde, bcs, f_pde = data
        A, b = problem.build(slv, x_pde, bcs, f_pde)
    else:
        x_pde, bcs = data
        A, b = problem.build(slv, x_pde, bcs)
    slv.beta = solve_lstsq(A, b)
    dt = time.perf_counter() - t0
    x_t = problem.get_test_points(5000)
    return rel_l2(slv.predict(x_t).cpu().numpy().ravel(),
                  problem.exact(x_t).cpu().numpy().ravel()), dt


def pielm_newton(problem, sigma, n_pde=5000, n_bc=1000, max_iter=30):
    slv = _build_pielm(problem.dim, sigma)
    t0 = time.perf_counter()
    x_pde, bcs, f_pde = problem.get_train_data(n_pde=n_pde, n_bc=n_bc)
    get_initial_guess(slv, problem, x_pde, bcs, f_pde)
    if getattr(problem, "use_continuation", False):
        sched = [v for v in problem.continuation_schedule if v >= problem.nu_target]
        continuation_solve(
            slv, problem, x_pde, bcs, f_pde,
            param_name="nu", param_schedule=sched,
            max_newton_per_step=max_iter // max(len(sched), 1) + 5, verbose=False,
        )
        problem.nu = problem.nu_target
    else:
        newton_solve(slv, problem, x_pde, bcs, f_pde, max_iter=max_iter, verbose=False)
    dt = time.perf_counter() - t0
    x_t = problem.get_test_points(5000)
    return rel_l2(slv.predict(x_t).cpu().numpy().ravel(),
                  problem.exact(x_t).cpu().numpy().ravel()), dt


PIELM_SIGMAS = [1.0, 2.0, 3.0, 5.0]
# Fair note: tanh saturates beyond sigma ~ 5, so a uniform PIELM sweep at
# {1, 2, 3, 5} -> best avoids the unfair comparison of giving PIELM
# FastLSQ's optimal high-frequency sigma where tanh degenerates.


def run_pielm_block():
    print("\n=== (1) PIELM on 10 PDEs (sigma sweep, matched N=1500, same pipeline) ===")
    rows = []
    linear_problems = [
        ("Poisson 5D", PoissonND, False),
        ("Heat 5D",    HeatND,    False),
        ("Wave 1D",    Wave1D,    False),
        ("Helmholtz 2D", Helmholtz2D, False),
        ("Maxwell 2D TM", Maxwell2D_TM, False),
        ("NL-Poisson 2D", NLPoisson2D, True),
        ("Bratu 2D",      Bratu2D, True),
        ("Burgers 1D",    SteadyBurgers1D, True),
        ("NL-Helmholtz 2D", NLHelmholtz2D, True),
        ("Allen-Cahn 1D", AllenCahn1D, True),
    ]
    for name, klass, is_nonlin in linear_problems:
        best = (None, math.inf, None)
        print(f"  [{name}]")
        for sig in PIELM_SIGMAS:
            try:
                prob = klass()
                fn = pielm_newton if is_nonlin else pielm_linear
                err, dt = fn(prob, sig)
                print(f"      sigma={sig}  err={err:.2e}  t={dt:.2f}s")
                rows.append([name, "PIELM", f"sigma={sig} N=1500",
                             err, dt, "—", ""])
                if err == err and err < best[1]:    # err==err: skip NaN
                    best = (sig, err, dt)
            except Exception as e:
                print(f"      sigma={sig}  FAIL {type(e).__name__}: {e}")
                rows.append([name, "PIELM", f"sigma={sig} N=1500",
                             NA, NA, "—", f"ERROR {type(e).__name__}"])
        if best[0] is not None:
            print(f"      best: sigma={best[0]} err={best[1]:.2e} t={best[2]:.2f}s")
            rows.append([name, "PIELM-best", f"sigma={best[0]}",
                         best[1], best[2], "—", "tuned best of sweep"])
    _csv_write(rows)


# ====================================================================
# (2) RBF c-sweep: 4 problems, c in {0.1, 0.3, 0.5, 1.0, 2.0}
# ====================================================================

C_SWEEP = [0.1, 0.3, 0.5, 1.0, 2.0]


def _mq_rbf_helm(c, n_centers=1500, lam=100.0, mu=1e-8, seed=7):
    K = 10.0
    rng = np.random.default_rng(seed)
    n_int = (n_centers * 4) // 5
    n_bc = n_centers - n_int
    x_int = rng.random((n_int * 4, 2))
    # boundary: 4 edges
    pts = []
    n_face = (n_bc * 4) // 4
    for dim in range(2):
        for val in (0.0, 1.0):
            p = rng.random((n_face, 2)); p[:, dim] = val
            pts.append(p)
    x_bc = np.vstack(pts)
    centers = np.vstack([rng.random((n_int, 2)), x_bc[:n_bc]])
    N = centers.shape[0]
    def phi(x):
        r2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        return np.sqrt(c ** 2 + r2)
    def lap(x):
        r2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        p = np.sqrt(c ** 2 + r2)
        return (2 * c ** 2 + 1 * r2) / p ** 3
    t0 = time.perf_counter()
    A_pde = lap(x_int) + K ** 2 * phi(x_int)
    f = -(K ** 2) * np.sin(K * x_int[:, 0]) * np.sin(K * x_int[:, 1])
    u_bc = np.sin(K * x_bc[:, 0]) * np.sin(K * x_bc[:, 1])
    A = np.vstack([A_pde, lam * phi(x_bc)])
    b = np.concatenate([f, lam * u_bc])
    AtA = A.T @ A + mu * np.eye(N)
    alpha = np.linalg.solve(AtA, A.T @ b)
    dt = time.perf_counter() - t0
    x_t = np.random.rand(5000, 2)
    err = rel_l2(phi(x_t) @ alpha,
                 np.sin(K * x_t[:, 0]) * np.sin(K * x_t[:, 1]))
    return err, dt


def _mq_rbf_nlpoisson(c, n_centers=900, lam=100.0, mu=1e-6, seed=7,
                     max_newton=30):
    rng = np.random.default_rng(seed)
    n_int = (n_centers * 4) // 5
    n_bc = n_centers - n_int
    x_int = rng.random((n_int * 4, 2))
    pts = []
    n_face = (n_bc * 4) // 4
    for dim in range(2):
        for val in (0.0, 1.0):
            p = rng.random((n_face, 2)); p[:, dim] = val
            pts.append(p)
    x_bc = np.vstack(pts)
    centers = np.vstack([rng.random((n_int, 2)), x_bc[:n_bc]])
    N = centers.shape[0]
    def phi(x):
        r2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        return np.sqrt(c ** 2 + r2)
    def lap(x):
        r2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        p = np.sqrt(c ** 2 + r2)
        return (2 * c ** 2 + 1 * r2) / p ** 3
    H = phi(x_int)
    H_bc = phi(x_bc)
    Lp = -lap(x_int)
    u_ex = np.sin(PI * x_int[:, 0]) * np.sin(PI * x_int[:, 1])
    f = 2 * PI ** 2 * u_ex + u_ex ** 3
    u_bc = np.sin(PI * x_bc[:, 0]) * np.sin(PI * x_bc[:, 1])
    t0 = time.perf_counter()
    A0 = np.vstack([Lp, lam * H_bc])
    b0 = np.concatenate([2 * PI ** 2 * u_ex, lam * u_bc])
    AtA0 = A0.T @ A0 + mu * np.eye(N)
    alpha = np.linalg.solve(AtA0, A0.T @ b0)
    for _ in range(max_newton):
        u_c = H @ alpha
        R = Lp @ alpha + u_c ** 3 - f
        R_b = lam * (H_bc @ alpha - u_bc)
        J = np.vstack([Lp + 3 * u_c[:, None] * H, lam * H_bc])
        rhs = -np.concatenate([R, R_b])
        AtA = J.T @ J + mu * np.eye(N)
        d = np.linalg.solve(AtA, J.T @ rhs)
        res0 = np.linalg.norm(rhs); step = 1.0
        for _ in range(8):
            a_try = alpha + step * d
            R_t = Lp @ a_try + (H @ a_try) ** 3 - f
            R_b_t = lam * (H_bc @ a_try - u_bc)
            if np.linalg.norm(np.concatenate([R_t, R_b_t])) < res0:
                break
            step *= 0.5
        alpha += step * d
        if np.linalg.norm(d) / (np.linalg.norm(alpha) + 1e-12) < 1e-9:
            break
    dt = time.perf_counter() - t0
    x_t = np.random.rand(5000, 2)
    err = rel_l2(phi(x_t) @ alpha,
                 np.sin(PI * x_t[:, 0]) * np.sin(PI * x_t[:, 1]))
    return err, dt


def _mq_rbf_poisson5d(c, n_centers=1500, lam=100.0, mu=1e-8, seed=7):
    rng = np.random.default_rng(seed)
    n_int = (n_centers * 4) // 5
    n_bc = n_centers - n_int
    x_int = rng.random((n_int * 4, 5))
    # boundary
    pts = []
    n_face = (n_bc * 4) // 10
    for dim in range(5):
        for val in (0.0, 1.0):
            p = rng.random((n_face, 5)); p[:, dim] = val
            pts.append(p)
    x_bc = np.vstack(pts)
    centers = np.vstack([rng.random((n_int, 5)), x_bc[:n_bc]])
    N = centers.shape[0]
    def phi(x):
        r2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        return np.sqrt(c ** 2 + r2)
    def lap(x):
        r2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        p = np.sqrt(c ** 2 + r2)
        return (5 * c ** 2 + 4 * r2) / p ** 3
    t0 = time.perf_counter()
    f_int = (PI ** 2 / 4.0) * np.sum(np.sin(PI / 2 * x_int), axis=1)
    u_bc = np.sum(np.sin(PI / 2 * x_bc), axis=1)
    A = np.vstack([-lap(x_int), lam * phi(x_bc)])
    b = np.concatenate([f_int, lam * u_bc])
    AtA = A.T @ A + mu * np.eye(N)
    alpha = np.linalg.solve(AtA, A.T @ b)
    dt = time.perf_counter() - t0
    x_t = np.random.rand(2000, 5)
    err = rel_l2(phi(x_t) @ alpha, np.sum(np.sin(PI / 2 * x_t), axis=1))
    return err, dt


def _mq_rbf_bratu(c, n_centers=900, lam=100.0, mu=1e-6, seed=7, max_newton=30):
    rng = np.random.default_rng(seed)
    n_int = (n_centers * 4) // 5
    n_bc = n_centers - n_int
    x_int = rng.random((n_int * 4, 2))
    pts = []
    n_face = (n_bc * 4) // 4
    for dim in range(2):
        for val in (0.0, 1.0):
            p = rng.random((n_face, 2)); p[:, dim] = val
            pts.append(p)
    x_bc = np.vstack(pts)
    centers = np.vstack([rng.random((n_int, 2)), x_bc[:n_bc]])
    N = centers.shape[0]
    def phi(x):
        r2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        return np.sqrt(c ** 2 + r2)
    def lap(x):
        r2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        p = np.sqrt(c ** 2 + r2)
        return (2 * c ** 2 + 1 * r2) / p ** 3
    LAM = 1.0
    H = phi(x_int); H_bc = phi(x_bc); Lp = -lap(x_int)
    u_ex = np.sin(PI * x_int[:, 0]) * np.sin(PI * x_int[:, 1])
    f = 2 * PI ** 2 * u_ex - LAM * np.exp(u_ex)
    u_bc = np.sin(PI * x_bc[:, 0]) * np.sin(PI * x_bc[:, 1])
    t0 = time.perf_counter()
    A0 = np.vstack([Lp, lam * H_bc])
    b0 = np.concatenate([2 * PI ** 2 * u_ex, lam * u_bc])
    alpha = np.linalg.solve(A0.T @ A0 + mu * np.eye(N), A0.T @ b0)
    for _ in range(max_newton):
        u_c = H @ alpha
        R = Lp @ alpha - LAM * np.exp(u_c) - f
        R_b = lam * (H_bc @ alpha - u_bc)
        J = np.vstack([Lp - LAM * np.exp(u_c)[:, None] * H, lam * H_bc])
        rhs = -np.concatenate([R, R_b])
        d = np.linalg.solve(J.T @ J + mu * np.eye(N), J.T @ rhs)
        res0 = np.linalg.norm(rhs); step = 1.0
        for _ in range(8):
            a_t = alpha + step * d
            R_t = Lp @ a_t - LAM * np.exp(H @ a_t) - f
            R_b_t = lam * (H_bc @ a_t - u_bc)
            if np.linalg.norm(np.concatenate([R_t, R_b_t])) < res0:
                break
            step *= 0.5
        alpha += step * d
        if np.linalg.norm(d) / (np.linalg.norm(alpha) + 1e-12) < 1e-9:
            break
    dt = time.perf_counter() - t0
    x_t = np.random.rand(5000, 2)
    err = rel_l2(phi(x_t) @ alpha,
                 np.sin(PI * x_t[:, 0]) * np.sin(PI * x_t[:, 1]))
    return err, dt


def run_rbf_csweep():
    print("\n=== (2) RBF Kansa shape-parameter sweep ===")
    rows = []
    sweeps = [
        ("Poisson 5D",    _mq_rbf_poisson5d),
        ("Helmholtz 2D",  _mq_rbf_helm),
        ("NL-Poisson 2D", _mq_rbf_nlpoisson),
        ("Bratu 2D",      _mq_rbf_bratu),
    ]
    for name, fn in sweeps:
        best = (None, math.inf, None)
        print(f"  [{name}]")
        for c in C_SWEEP:
            try:
                err, dt = fn(c)
                print(f"      c={c:.2f}  err={err:.2e}  t={dt:.2f}s")
                rows.append([name, "RBF", f"c={c:.2f}", err, dt, "—", ""])
                if err < best[1]:
                    best = (c, err, dt)
            except Exception as e:
                print(f"      c={c:.2f}  FAIL {e}")
                rows.append([name, "RBF", f"c={c:.2f}", NA, NA, "—",
                             f"ERROR {type(e).__name__}"])
        if best[0] is not None:
            print(f"      best: c={best[0]} err={best[1]:.2e} t={best[2]:.2f}s")
            rows.append([name, "RBF-best", f"c={best[0]}", best[1], best[2], "—",
                         "tuned best of sweep"])
    _csv_write(rows)


# ====================================================================
# (3) Vanilla PINN on 3 problems
# ====================================================================

class PINN(torch.nn.Module):
    def __init__(self, d_in, hidden=64, depth=4):
        super().__init__()
        layers = [torch.nn.Linear(d_in, hidden), torch.nn.Tanh()]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(hidden, hidden), torch.nn.Tanh()]
        layers += [torch.nn.Linear(hidden, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _pinn_helmholtz(n_iter=10000, lr=1e-3, lam=100.0, n_pde=4000, n_bc=400, seed=0):
    torch.manual_seed(seed)
    K = 10.0
    net = PINN(2)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        x = torch.rand(n_pde, 2, requires_grad=True)
        x_bc = torch.rand(n_bc, 2)
        # snap bc to boundary
        idx = torch.randint(0, 4, (n_bc,))
        dim_b = idx % 2
        val_b = (idx // 2).float()
        x_bc = x_bc.clone()
        x_bc[torch.arange(n_bc), dim_b] = val_b
        u = net(x)
        grads = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(grads[:, 0].sum(), x, create_graph=True)[0][:, 0]
        u_yy = torch.autograd.grad(grads[:, 1].sum(), x, create_graph=True)[0][:, 1]
        f = -(K ** 2) * torch.sin(K * x[:, 0]) * torch.sin(K * x[:, 1])
        res = u_xx + u_yy + K ** 2 * u - f
        u_bc_pred = net(x_bc)
        u_bc_true = torch.sin(K * x_bc[:, 0]) * torch.sin(K * x_bc[:, 1])
        loss = (res ** 2).mean() + lam * ((u_bc_pred - u_bc_true) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    dt = time.perf_counter() - t0
    x_t = torch.rand(5000, 2)
    with torch.no_grad():
        pred = net(x_t).numpy()
    true = (torch.sin(K * x_t[:, 0]) * torch.sin(K * x_t[:, 1])).numpy()
    return rel_l2(pred, true), dt


def _pinn_nlpoisson(n_iter=10000, lr=1e-3, lam=100.0, n_pde=4000, n_bc=400, seed=0):
    torch.manual_seed(seed)
    net = PINN(2)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        x = torch.rand(n_pde, 2, requires_grad=True)
        x_bc = torch.rand(n_bc, 2)
        idx = torch.randint(0, 4, (n_bc,))
        dim_b = idx % 2
        val_b = (idx // 2).float()
        x_bc = x_bc.clone()
        x_bc[torch.arange(n_bc), dim_b] = val_b
        u = net(x)
        grads = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(grads[:, 0].sum(), x, create_graph=True)[0][:, 0]
        u_yy = torch.autograd.grad(grads[:, 1].sum(), x, create_graph=True)[0][:, 1]
        u_ex = torch.sin(PI * x[:, 0]) * torch.sin(PI * x[:, 1])
        f = 2 * PI ** 2 * u_ex + u_ex ** 3
        res = -(u_xx + u_yy) + u ** 3 - f
        u_bc_pred = net(x_bc)
        u_bc_true = torch.sin(PI * x_bc[:, 0]) * torch.sin(PI * x_bc[:, 1])
        loss = (res ** 2).mean() + lam * ((u_bc_pred - u_bc_true) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    dt = time.perf_counter() - t0
    x_t = torch.rand(5000, 2)
    with torch.no_grad():
        pred = net(x_t).numpy()
    true = (torch.sin(PI * x_t[:, 0]) * torch.sin(PI * x_t[:, 1])).numpy()
    return rel_l2(pred, true), dt


def _pinn_burgers(n_iter=10000, lr=1e-3, lam=100.0, n_pde=2000, n_bc=20, seed=0):
    torch.manual_seed(seed)
    NU = 0.1
    net = PINN(1)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        x = torch.rand(n_pde, 1, requires_grad=True)
        x_bc = torch.tensor([[0.0], [1.0]] * (n_bc // 2 if n_bc > 1 else 1),
                            dtype=torch.float32)
        u = net(x)
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_ex = torch.sin(2 * PI * x[:, 0])
        ux_ex = 2 * PI * torch.cos(2 * PI * x[:, 0])
        uxx_ex = -((2 * PI) ** 2) * torch.sin(2 * PI * x[:, 0])
        f = u_ex * ux_ex - NU * uxx_ex
        res = u * u_x.squeeze(-1) - NU * u_xx.squeeze(-1) - f
        u_bc_pred = net(x_bc)
        u_bc_true = torch.sin(2 * PI * x_bc[:, 0])
        loss = (res ** 2).mean() + lam * ((u_bc_pred - u_bc_true) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    dt = time.perf_counter() - t0
    x_t = torch.linspace(0, 1, 5000).reshape(-1, 1)
    with torch.no_grad():
        pred = net(x_t).numpy()
    true = torch.sin(2 * PI * x_t[:, 0]).numpy()
    return rel_l2(pred, true), dt


def run_pinn_block():
    print("\n=== (3) Vanilla PINN (4-layer MLP, 64 units, 10k Adam iters) ===")
    rows = []
    runs = [
        ("Helmholtz 2D",  _pinn_helmholtz),
        ("NL-Poisson 2D", _pinn_nlpoisson),
        ("Burgers 1D",    _pinn_burgers),
    ]
    for name, fn in runs:
        try:
            err, dt = fn()
            print(f"  [{name}] PINN   err={err:.2e} t={dt:.1f}s")
            rows.append([name, "PINN", "MLP-4-64 tanh, 10k Adam, lam=100",
                         err, dt, "—", ""])
        except Exception as e:
            print(f"  [{name}] PINN   FAIL {e}")
            rows.append([name, "PINN", "MLP-4-64 tanh, 10k Adam",
                         NA, NA, "—", f"ERROR {type(e).__name__}"])
    _csv_write(rows)


# ====================================================================
# (4) Chebyshev pseudospectral on Burgers 1D BVP
# ====================================================================

def _cheb_diff(N):
    """Trefethen's Chebyshev differentiation matrix on [-1, 1]."""
    if N == 0:
        return np.zeros((1, 1)), np.array([1.0])
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = (np.r_[2.0, np.ones(N - 1), 2.0]) * (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))
    return D, x


def cheb_burgers(N=64, nu=0.1, max_newton=40, tol=1e-12):
    D, xi = _cheb_diff(N)
    # Map [-1, 1] -> [0, 1]:  x = (xi + 1)/2,  d/dx = 2 d/dxi
    x = 0.5 * (xi + 1.0)
    D1 = 2.0 * D
    D2 = D1 @ D1
    u_ex = np.sin(2 * PI * x)
    ux_ex = 2 * PI * np.cos(2 * PI * x)
    uxx_ex = -((2 * PI) ** 2) * np.sin(2 * PI * x)
    f = u_ex * ux_ex - nu * uxx_ex
    # Interior nodes: 1..N-1
    interior = np.arange(1, N)
    u = u_ex.copy()  # warm start with exact (could be anything smooth)
    u = 0.5 * np.sin(2 * PI * x)  # cold start away from exact
    u[0] = 0.0; u[-1] = 0.0   # boundary
    t0 = time.perf_counter()
    for _ in range(max_newton):
        ux = D1 @ u
        uxx = D2 @ u
        R = u * ux - nu * uxx - f
        # restrict to interior
        R_int = R[interior]
        # Jacobian: d/du(u u_x) = diag(u_x) + u D1; nonlinear part of u*u_x
        J = (np.diag(ux) + np.diag(u) @ D1) - nu * D2
        J_int = J[np.ix_(interior, interior)]
        du = np.linalg.solve(J_int, -R_int)
        u[interior] += du
        u[0] = 0.0; u[-1] = 0.0
        if np.linalg.norm(du) < tol:
            break
    dt = time.perf_counter() - t0
    return rel_l2(u, u_ex), dt, N + 1


def run_cheb_block():
    print("\n=== (4) Chebyshev pseudospectral on Burgers 1D ===")
    rows = []
    for N in [16, 32, 64, 128]:
        try:
            err, dt, dof = cheb_burgers(N=N)
            print(f"  [Burgers 1D] Cheb N={N:>3}  err={err:.2e}  t={dt*1000:.1f}ms  dof={dof}")
            rows.append(["Burgers 1D", "Chebyshev", f"N={N}", err, dt, dof, ""])
        except Exception as e:
            print(f"  [Burgers 1D] Cheb N={N}  FAIL {e}")
            rows.append(["Burgers 1D", "Chebyshev", f"N={N}", NA, NA, N + 1,
                         f"ERROR {type(e).__name__}"])
    _csv_write(rows)


# ====================================================================
# Main
# ====================================================================

def main():
    if CSV_PATH.exists():
        CSV_PATH.unlink()
    run_pielm_block()
    run_rbf_csweep()
    run_pinn_block()
    run_cheb_block()
    print(f"\n[done] CSV written to {CSV_PATH}")


if __name__ == "__main__":
    main()
