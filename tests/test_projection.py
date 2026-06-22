# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Verification of the closed-form *projection* (Radon) operator on a Gaussian-windowed
basis -- the line/hyperplane-integral class that the single-axis ``IntegralOperator``
cannot express.

A tomographic measurement is a projection onto a generally non-axis-aligned hyperplane,
``p(u) = ∫ f(z) δ(c·z − u) dz`` (a Fredholm equation of the first kind).  For the
Gaussian-windowed (Gabor) basis the Gaussian × plane-wave hyperplane integral is analytic,
so ``ProjectionOperator.apply`` assembles the design matrix in closed form, with no
quadrature.  These tests make that a *tested* property:

* the closed form equals a Gauss--Hermite quadrature of the (d−1)-dim slice integral to
  machine precision, in d = 2, 3, 4 (the GATE);
* the assembly is differentiable in the projection direction ``c`` (the optics):
  autodiff matches finite differences (needed for differentiable experiment design);
* ``evaluate`` matches the hand-written windowed-feature formula and shapes are right;
* ``from_transport`` realises the tomography convention ``c = Mᵀe``;
* a windowed field is recovered from its projections at several directions in one LSQ.

Run with ``pytest`` or directly as a script.
"""

import math
import itertools

import numpy as np
import torch

from fastlsq import (
    GaussianWindowedBasis, ProjectionOperator, solve_lstsq,
)

torch.set_default_dtype(torch.float64)
DT = torch.float64
PI = np.pi


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _random_windowed_basis(d, n_features, seed, sigma=1.0):
    """A windowed basis with a non-trivial (random SPD) window, package W=(d,N)."""
    g = torch.Generator().manual_seed(seed)
    mean = torch.randn(d, generator=g, dtype=DT)
    A = torch.randn(d, d, generator=g, dtype=DT)
    cov = A @ A.t() + d * torch.eye(d, dtype=DT)   # SPD
    T = torch.linalg.cholesky(cov)                 # lower-Cholesky factor
    W = torch.randn(d, n_features, generator=g, dtype=DT) * sigma
    b = torch.rand(1, n_features, generator=g, dtype=DT) * 2 * PI
    return GaussianWindowedBasis(W, b, mean, T)


def _projection_quadrature(op, basis, u, nq=40):
    """Reference: the (d−1)-dim slice integral by Gauss--Hermite product quadrature.

    Mirrors the closed form ``ProjectionOperator.apply`` but evaluates the hyperplane
    integral numerically.  Change variables to the whitened frame ζ = T⁻¹(z−mean): the
    delta pins the component along ``q = Tᵀc`` to ζ_∥ = (u−u₀)·q/‖q‖², leaving a Gaussian
    integral over the (d−1)-dim complement q^⊥, which Gauss--Hermite integrates.  This
    path need not be differentiable, so a QR complement and ``float()`` casts are fine.
    """
    c = op.c
    q = basis.T.t() @ c
    qn2 = float(q @ q)
    qn = math.sqrt(qn2)
    d = basis.d
    u0 = float(c @ basis.mean)
    jac = float(torch.det(basis.T).abs()) / qn

    # orthonormal complement Q of q (columns span q^⊥): complete QR of [q | 0 ...]
    Araw = torch.zeros(d, d, dtype=DT)
    Araw[:, 0] = q
    Qf, _ = torch.linalg.qr(Araw, mode="complete")
    Q = Qf[:, 1:]                                  # (d, d−1)

    t, w = np.polynomial.hermite.hermgauss(nq)
    t = torch.as_tensor(t, dtype=DT) * math.sqrt(2)   # ∫ e^{-y²/2} g dy = √2 Σ w_i g(√2 t_i)
    w = torch.as_tensor(w, dtype=DT) * math.sqrt(2)
    dm1 = d - 1
    if dm1 == 0:                                    # d = 1: no complement (not used here)
        S = torch.zeros(1, 0, dtype=DT)
        Wp = torch.ones(1, dtype=DT)
    else:
        grids_t = torch.meshgrid(*([t] * dm1), indexing="ij")
        S = torch.stack([g.reshape(-1) for g in grids_t], dim=1)        # (nq^{d−1}, d−1)
        grids_w = torch.meshgrid(*([w] * dm1), indexing="ij")
        Wp = torch.stack([g.reshape(-1) for g in grids_w], dim=1).prod(dim=1)   # (nq^{d−1},)

    base = S @ Q.t()                               # (nq^{d−1}, d), points in q^⊥
    rows = []
    for uj in u.reshape(-1).tolist():
        zeta = base + (uj - u0) * (q / qn2)        # add the pinned parallel component
        sinmat = torch.sin(zeta @ basis.W + basis.b)               # (nq^{d−1}, N)
        weights = Wp * jac * math.exp(-(uj - u0) ** 2 / (2.0 * qn2))
        rows.append(weights @ sinmat)              # (N,)
    return torch.stack(rows, dim=0)                # (M, N)


# ----------------------------------------------------------------------
# GATE: closed form == Gauss--Hermite quadrature, d = 2, 3, 4
# ----------------------------------------------------------------------

def test_closed_form_matches_quadrature():
    """The quadrature-free projection rows equal a Gauss--Hermite slice integral to
    machine precision, across d = 2, 3, 4 -- the design gate."""
    for d in (2, 3, 4):
        basis = _random_windowed_basis(d, n_features=30, seed=100 + d, sigma=1.0)
        c = torch.randn(d, generator=torch.Generator().manual_seed(7 + d), dtype=DT)
        op = ProjectionOperator(c)
        u = torch.linspace(-1.5, 1.5, 11, dtype=DT)

        A_cf = op.apply(basis, u)
        A_q = _projection_quadrature(op, basis, u, nq=40)

        assert A_cf.shape == (u.numel(), basis.n_features)
        max_abs = (A_cf - A_q).abs().max().item()
        rel = (torch.norm(A_cf - A_q) / torch.norm(A_q)).item()
        assert max_abs < 1e-9, f"d={d}: closed-form vs quadrature max|diff|={max_abs:.2e}"
        assert rel < 1e-9, f"d={d}: closed-form vs quadrature rel-L2={rel:.2e}"


# ----------------------------------------------------------------------
# GATE: differentiability in the optics  (autodiff == finite difference)
# ----------------------------------------------------------------------

def test_autodiff_matches_finite_difference_in_c():
    """The assembled rows are differentiable in the direction ``c`` (the optics):
    autograd of a scalar functional wrt ``c`` matches central finite differences.

    This is the property differentiable experiment design relies on; it is what the
    rotation-invariant |ω|² (no QR) and the no-``float()`` design protect."""
    d, N, M = 3, 40, 9
    basis = _random_windowed_basis(d, n_features=N, seed=321, sigma=1.0)
    beta = torch.randn(N, 1, generator=torch.Generator().manual_seed(5), dtype=DT)
    u = torch.linspace(-1.0, 1.0, M, dtype=DT)
    c0 = torch.randn(d, generator=torch.Generator().manual_seed(9), dtype=DT)

    def functional(cc):
        # smooth scalar of the projection rows; depends on c through the whole closed form
        A = ProjectionOperator(cc).apply(basis, u)         # (M, N)
        return ((A @ beta) ** 2).sum()

    c = c0.clone().requires_grad_(True)
    functional(c).backward()
    g_ad = c.grad.detach().clone()

    eps = 1e-6
    g_fd = torch.zeros(d, dtype=DT)
    for i in range(d):
        cp = c0.clone(); cp[i] += eps
        cm = c0.clone(); cm[i] -= eps
        g_fd[i] = (functional(cp) - functional(cm)) / (2 * eps)

    assert torch.isfinite(g_ad).all()
    rel = (torch.norm(g_ad - g_fd) / torch.norm(g_fd)).item()
    assert rel < 1e-6, f"autodiff vs FD rel = {rel:.2e}  (ad={g_ad}, fd={g_fd})"


# ----------------------------------------------------------------------
# basis value + operator shapes
# ----------------------------------------------------------------------

def test_value_matches_formula_and_shapes():
    """``evaluate`` equals exp(−½‖ζ‖²)·sin(ζ@W+b) by hand, and ``apply`` returns (M,N)
    for detector coords given as either (M,) or (M,1)."""
    d, N, M = 3, 16, 20
    basis = _random_windowed_basis(d, n_features=N, seed=11)
    z = torch.randn(M, d, dtype=DT)

    zeta = (z - basis.mean) @ basis.Tinv.t()
    hand = torch.exp(-0.5 * (zeta ** 2).sum(1, keepdim=True)) * torch.sin(zeta @ basis.W + basis.b)
    got = basis.evaluate(z)
    assert got.shape == (M, N)
    assert torch.allclose(got, hand, atol=1e-12, rtol=1e-10)
    assert torch.allclose(basis.value(z), got, atol=1e-12)   # prototype-name alias

    # window is bounded by 1 and decays away from the centre
    assert torch.all(torch.exp(-0.5 * (zeta ** 2).sum(1)) <= 1.0 + 1e-12)

    op = ProjectionOperator(torch.randn(d, dtype=DT))
    u_flat = torch.linspace(-1, 1, 7, dtype=DT)
    A1 = op.apply(basis, u_flat)
    A2 = op.apply(basis, u_flat.reshape(-1, 1))    # (M,1) accepted too
    assert A1.shape == (7, N)
    assert torch.allclose(A1, A2, atol=1e-12)


def test_from_transport_convention():
    """``from_transport(M, e)`` builds the read-out direction c = Mᵀe."""
    d = 3
    g = torch.Generator().manual_seed(2)
    Mmat = torch.randn(d, d, generator=g, dtype=DT)
    e = torch.randn(d, generator=g, dtype=DT)
    op = ProjectionOperator.from_transport(Mmat, e)
    assert torch.allclose(op.c, Mmat.t() @ e, atol=1e-12)

    # transport+readout: measured coord e·(M z) equals (Mᵀe)·z = c·z
    z = torch.randn(5, d, dtype=DT)
    assert torch.allclose((Mmat @ z.t()).t() @ e, z @ op.c, atol=1e-12)


def test_from_transport_is_differentiable_in_optics():
    """Gradients flow back to the transport matrix M (the optics) through the projection."""
    d, N = 2, 24
    basis = _random_windowed_basis(d, n_features=N, seed=44)
    Mmat = torch.eye(d, dtype=DT).clone().requires_grad_(True)
    e = torch.tensor([1.0, 0.0], dtype=DT)
    u = torch.linspace(-1, 1, 6, dtype=DT)
    ProjectionOperator.from_transport(Mmat, e).apply(basis, u).pow(2).sum().backward()
    assert Mmat.grad is not None and torch.isfinite(Mmat.grad).all()


# ----------------------------------------------------------------------
# end-to-end: recover a windowed field from its projections (synthetic gate)
# ----------------------------------------------------------------------

def test_reconstruct_field_from_projections():
    """One-shot LSQ recovers the coefficients of a windowed field from its projections
    taken at several directions (a synthetic mirror of the real-data tomography gate)."""
    torch.manual_seed(0)
    d, N = 2, 200

    # ground-truth field lives in the SAME windowed basis -> exact target exists
    z_support = torch.randn(4000, d, dtype=DT) * torch.tensor([1.0, 0.6]) + torch.tensor([0.2, -0.1])
    basis = GaussianWindowedBasis.from_data(z_support, n_features=N, sigma=1.3)
    beta_true = torch.randn(N, 1, dtype=DT) / math.sqrt(N)

    # projection directions sweeping a half-circle (tomographic angles)
    angles = torch.linspace(0.0, PI, 12, dtype=DT)[:-1]
    u_grid = torch.linspace(-4.0, 4.0, 80, dtype=DT)

    A_blocks, p_blocks = [], []
    for th in angles:
        c = torch.stack([torch.cos(th), torch.sin(th)])
        Aop = ProjectionOperator(c).apply(basis, u_grid)     # (len(u), N)
        A_blocks.append(Aop)
        p_blocks.append(Aop @ beta_true)                     # synthetic measurement
    A = torch.cat(A_blocks, dim=0)
    p = torch.cat(p_blocks, dim=0)

    beta_hat = solve_lstsq(A, p, mu=1e-10)

    # reconstruct the field on a grid and compare to the truth
    gx = torch.linspace(-2.5, 2.5, 40, dtype=DT)
    GX, GY = torch.meshgrid(gx, gx, indexing="ij")
    zt = torch.stack([GX.reshape(-1), GY.reshape(-1)], dim=1)
    f_true = basis.evaluate(zt) @ beta_true
    f_hat = basis.evaluate(zt) @ beta_hat
    rel = (torch.norm(f_hat - f_true) / torch.norm(f_true)).item()
    assert rel < 1e-3, f"field reconstruction rel-L2 = {rel:.2e}"


if __name__ == "__main__":
    test_closed_form_matches_quadrature()
    test_autodiff_matches_finite_difference_in_c()
    test_value_matches_formula_and_shapes()
    test_from_transport_convention()
    test_from_transport_is_differentiable_in_optics()
    test_reconstruct_field_from_projections()
    print("ALL PROJECTION TESTS PASSED")
