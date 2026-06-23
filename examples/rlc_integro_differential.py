#!/usr/bin/env python3
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
The textbook integro-differential equation: a driven series RLC circuit, in one shot.

Kirchhoff's voltage law around a series RLC loop is the canonical integro-differential
equation of first-year circuit theory -- the current appears differentiated (inductor),
undifferentiated (resistor), *and* integrated (the capacitor stores the accumulated charge
q(t) = ∫_0^t i ds):

        L i'(t) + R i(t) + (1/C) ∫_0^t i(s) ds = V(t),     i(0) = 0.

FastLSQ assembles all three terms -- derivative, value, and running integral -- into a
single linear-least-squares design matrix and solves for i(t) in ONE shot.  There is no
time stepping: the closed-form Volterra integral of every Fourier feature is exact, so the
capacitor term is just another column block.

We drive the loop at resonance (Ω = ω_0 = 1/√(LC)), where the underdamped transient and the
forced response superpose into a decaying-then-ringing current -- a genuinely "complex"
waveform.  The one-shot solution is validated against a fine RK4 integration of the
equivalent 2nd-order ODE (q' = i, L i' = V - R i - q/C); because there is no marching, the
one-shot solve carries no accumulated phase error.

Operator:
        L_op = L · Op.partial(0, 1, d=1)                       # L i'
             + R · Op.identity(d=1)                            # R i
             + (1/C) · IntegralOperator.volterra(0, 0.0, d=1)  # (1/C) ∫_0^t i

Usage:  python rlc_integro_differential.py
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastlsq import SinusoidalBasis, Op, IntegralOperator, solve_lstsq
from fastlsq.geometry import sample_box


def rk4_reference(L, R, C, V_fn, T, n=20001):
    """Fine RK4 integration of the equivalent state-space ODE, from rest.

    State y = [q, i] with q' = i and i' = (V - R i - q/C) / L.  Returns (t, i, q).
    """
    t = np.linspace(0.0, T, n)
    h = t[1] - t[0]
    y = np.zeros((n, 2))  # [q, i], starting from rest

    def rhs(tt, yy):
        q, i = yy
        return np.array([i, (V_fn(tt) - R * i - q / C) / L])

    for k in range(n - 1):
        tk, yk = t[k], y[k]
        k1 = rhs(tk, yk)
        k2 = rhs(tk + 0.5 * h, yk + 0.5 * h * k1)
        k3 = rhs(tk + 0.5 * h, yk + 0.5 * h * k2)
        k4 = rhs(tk + h, yk + h * k3)
        y[k + 1] = yk + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y[:, 1], y[:, 0]


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    # ------------------------------------------------------------------
    # Physical circuit (underdamped, driven at resonance)
    # ------------------------------------------------------------------
    L, R, C = 1.0, 0.8, 1.0 / 64.0       # H, Ω, F
    w0 = 1.0 / np.sqrt(L * C)            # natural frequency  = 8 rad/s
    alpha = R / (2.0 * L)               # damping rate       = 0.4
    Omega = w0                          # drive at resonance
    V0 = 5.0
    T = 6.0
    V_fn_np = lambda t: V0 * np.sin(Omega * t)
    V_fn_t = lambda t: V0 * torch.sin(Omega * t)
    print(f"RLC: L={L}, R={R}, C={C:.5f}  ->  w0={w0:.2f} rad/s, alpha={alpha:.2f}, Q={w0/(2*alpha):.1f}")

    # ------------------------------------------------------------------
    # Reference: fine RK4 of the equivalent ODE (ground truth)
    # ------------------------------------------------------------------
    t_ref, i_ref, q_ref = rk4_reference(L, R, C, V_fn_np, T)

    # ------------------------------------------------------------------
    # Basis + collocation on [0, T]
    # ------------------------------------------------------------------
    basis = SinusoidalBasis.random(input_dim=1, n_features=1600, sigma=5.0, normalize=False)
    t_col = sample_box(6000, 1, bounds=(0.0, T))
    t_ic = torch.zeros(1, 1)

    # ------------------------------------------------------------------
    # One unified integro-differential operator  L i' + R i + (1/C) ∫_0^t i
    # ------------------------------------------------------------------
    op = (L * Op.partial(0, 1, d=1)
          + R * Op.identity(d=1)
          + (1.0 / C) * IntegralOperator.volterra(dim=0, lower=0.0, d=1))
    print("Operator:", op)

    # Assemble [PDE rows ; weighted IC row]  =  [V ; weighted i0]  and solve once.
    W_IC = 100.0
    A = torch.cat([op.apply(basis, t_col), W_IC * basis.evaluate(t_ic)])
    b = torch.cat([V_fn_t(t_col),          W_IC * torch.zeros(1, 1)])
    beta = solve_lstsq(A, b, mu=1e-10)

    # ------------------------------------------------------------------
    # Accuracy vs the RK4 reference
    # ------------------------------------------------------------------
    tt = torch.from_numpy(t_ref).reshape(-1, 1)
    i_pred = (basis.evaluate(tt) @ beta).squeeze().numpy()
    i_true = i_ref
    val_err = np.linalg.norm(i_pred - i_true) / (np.linalg.norm(i_true) + 1e-15)

    # Capacitor charge as a *closed-form* running integral of the recovered current.
    V_run = IntegralOperator.volterra(dim=0, lower=0.0, d=1)
    q_pred = (V_run.apply(basis, tt) @ beta).squeeze().numpy()
    q_err = np.linalg.norm(q_pred - q_ref) / (np.linalg.norm(q_ref) + 1e-15)

    print(f"  current  i(t)  rel-L2 vs RK4 : {val_err:.2e}")
    print(f"  charge   q(t)=∫i  rel-L2     : {q_err:.2e}")

    # ------------------------------------------------------------------
    # Plot:  current vs RK4, charge vs RK4, pointwise residual
    # ------------------------------------------------------------------
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 4))
    ax0.plot(t_ref, i_true, "k-", lw=2.5, label="RK4 reference")
    ax0.plot(t_ref, i_pred, "C1--", lw=1.6, label="FastLSQ (one-shot)")
    ax0.set_title("current  i(t)")
    ax0.set_xlabel("t [s]"); ax0.set_ylabel("i(t) [A]"); ax0.legend()

    ax1.plot(t_ref, q_ref, "k-", lw=2.5, label="RK4 charge")
    ax1.plot(t_ref, q_pred, "C2--", lw=1.6, label="∫₀ᵗ i  (closed form)")
    ax1.set_title("capacitor charge  q(t) = ∫₀ᵗ i ds")
    ax1.set_xlabel("t [s]"); ax1.set_ylabel("q(t) [C]"); ax1.legend()

    ax2.semilogy(t_ref, np.abs(i_pred - i_true) + 1e-18, "C3-")
    ax2.set_title(f"pointwise |error|  (rel-L2 = {val_err:.1e})")
    ax2.set_xlabel("t [s]"); ax2.set_ylabel("|i_pred - i_RK4|")

    fig.suptitle("Series RLC: L i' + R i + (1/C)∫₀ᵗ i = V(t),  driven at resonance", y=1.02)
    plt.tight_layout()
    out = "rlc_integro_differential.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")


if __name__ == "__main__":
    main()
