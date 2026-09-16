# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Property tests: every closed form against an independent reference, at RANDOM
dimensions, feature counts, bandwidths and orders.

The rest of the suite pins fixed cases.  This file re-draws the problem on every run
(seeded per case, so a failure is reproducible from the parameters in the test id) and
checks each closed form against something that does not share its derivation:

    derivatives (to order 4, mixed included)   autograd
    antiderivatives and definite integrals     Gauss-Legendre quadrature
    iterated (Cauchy) integrals                repeated quadrature
    multi-axis integrals                       tensor-product quadrature
    symbol operators (even AND odd orders)     autograd for even integer orders,
                                               the analytic sin/cos action otherwise
    separable-kernel inner products            Gauss-Legendre quadrature
    degenerate eigenvalues                     the analytic characteristic value
    augmented-basis images                     the same references, on both blocks
    SDF normals and Robin rows                 autograd on the SDF

Added after the September 2026 review, together with the cache-invalidation test at the
bottom: the kernel inner products used to be cached on ``id(basis)``, which a new basis
at a recycled address silently hit.
"""

import math

import numpy as np
import pytest
import torch

from fastlsq.augment import AugmentedBasis, PolynomialColumns
from fastlsq.basis import SinusoidalBasis
from fastlsq.kernels import SeparableKernelOperator, degenerate_eigenvalues

torch.set_default_dtype(torch.float64)

TOL = 1e-10


def _basis(seed, d, n=8, sigma=None):
    torch.manual_seed(seed)
    sigma = sigma if sigma is not None else float(np.random.default_rng(seed).uniform(0.5, 8.0))
    return SinusoidalBasis.random(d, n, sigma=sigma, normalize=False), sigma


def _rand_x(seed, d, m=7):
    g = torch.Generator().manual_seed(seed + 991)
    return torch.rand(m, d, generator=g)


def _autograd_derivative(basis, x, alpha, v):
    """D^alpha (basis.evaluate(x) @ v) by repeated autograd."""
    x = x.clone().requires_grad_(True)
    u = basis.evaluate(x) @ v
    for k, order in enumerate(alpha):
        for _ in range(order):
            u = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, k:k + 1]
    return u.detach()


def _gauss_legendre(n, a, b):
    t, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * t + 0.5 * (a + b), 0.5 * (b - a) * w


# --------------------------------------------------------------------------- derivatives
@pytest.mark.parametrize("seed", range(6))
def test_derivatives_match_autograd(seed):
    rng = np.random.default_rng(seed)
    d = int(rng.integers(1, 4))
    basis, _ = _basis(seed, d, n=int(rng.integers(4, 12)))
    x = _rand_x(seed, d)
    v = torch.randn(basis.n_features, 1, generator=torch.Generator().manual_seed(seed))
    for _ in range(4):
        alpha = [0] * d
        for _ in range(int(rng.integers(1, 5))):             # total order 1..4
            alpha[int(rng.integers(0, d))] += 1
        if sum(alpha) > 4:
            continue
        got = basis.derivative(x, tuple(alpha)) @ v
        ref = _autograd_derivative(basis, x, alpha, v)
        assert torch.allclose(got, ref, atol=TOL, rtol=1e-8), (alpha, d)


@pytest.mark.parametrize("seed", range(4))
def test_gradient_laplacian_hessian_match_autograd(seed):
    rng = np.random.default_rng(seed + 100)
    d = int(rng.integers(1, 4))
    basis, _ = _basis(seed + 100, d)
    x = _rand_x(seed + 100, d)
    v = torch.randn(basis.n_features, 1, generator=torch.Generator().manual_seed(seed))
    xg = x.clone().requires_grad_(True)
    u = basis.evaluate(xg) @ v
    g = torch.autograd.grad(u.sum(), xg, create_graph=True)[0]
    hd = torch.stack([torch.autograd.grad(g[:, k].sum(), xg, create_graph=True)[0][:, k]
                      for k in range(d)], dim=1)
    assert torch.allclose((basis.gradient(x) @ v).squeeze(-1), g.detach(), atol=TOL)
    assert torch.allclose((basis.hessian_diag(x) @ v).squeeze(-1), hd.detach(), atol=TOL)
    assert torch.allclose((basis.laplacian(x) @ v).squeeze(-1), hd.sum(1).detach(), atol=TOL)


# --------------------------------------------------------------------------- integrals
@pytest.mark.parametrize("seed", range(5))
def test_definite_integral_matches_quadrature(seed):
    rng = np.random.default_rng(seed + 200)
    d = int(rng.integers(1, 4))
    dim = int(rng.integers(0, d))
    basis, _ = _basis(seed + 200, d, sigma=float(rng.uniform(0.5, 25.0)))
    x = _rand_x(seed + 200, d, m=5)
    v = torch.randn(basis.n_features, 1, generator=torch.Generator().manual_seed(seed))
    lower = float(rng.uniform(-0.5, 0.0))
    got = (basis.definite_integral(x, dim, lower) @ v).squeeze(-1)
    ref = []
    for row in x:                                            # integrate up to x[dim]
        t, w = _gauss_legendre(200, lower, float(row[dim]))
        pts = row.repeat(len(t), 1).clone()
        pts[:, dim] = torch.tensor(t)
        ref.append(float(torch.tensor(w) @ (basis.evaluate(pts) @ v).squeeze(-1)))
    assert torch.allclose(got, torch.tensor(ref), atol=1e-9)


@pytest.mark.parametrize("order", [1, 2, 3, 5, 8])
def test_iterated_integral_matches_repeated_quadrature(order):
    """Cauchy repeated integral: F_n(x) = 1/(n-1)! int_lo^x (x-t)^{n-1} f(t) dt."""
    seed = 300 + order
    basis, _ = _basis(seed, 1, n=6, sigma=3.0)
    v = torch.randn(basis.n_features, 1, generator=torch.Generator().manual_seed(seed))
    lower = 0.0
    x = torch.rand(4, 1, generator=torch.Generator().manual_seed(seed)) * 0.9 + 0.05
    got = (basis.iterated_integral(x, 0, lower, order=order) @ v).squeeze(-1)
    ref = []
    for row in x:
        t, w = _gauss_legendre(300, lower, float(row[0]))
        f = (basis.evaluate(torch.tensor(t).reshape(-1, 1)) @ v).squeeze(-1).numpy()
        kern = (float(row[0]) - t) ** (order - 1) / math.factorial(order - 1)
        ref.append(float((w * kern) @ f))
    assert torch.allclose(got, torch.tensor(ref), atol=1e-9)


@pytest.mark.parametrize("seed", range(3))
def test_multi_integral_matches_tensor_quadrature(seed):
    d = 2
    basis, _ = _basis(seed + 400, d, n=6, sigma=2.5)
    v = torch.randn(basis.n_features, 1, generator=torch.Generator().manual_seed(seed))
    x = torch.rand(3, d, generator=torch.Generator().manual_seed(seed)) * 0.8 + 0.1
    got = (basis.multi_integral(x, [0, 1], [0.0, 0.0]) @ v).squeeze(-1)
    ref = []
    for row in x:
        a, wa = _gauss_legendre(80, 0.0, float(row[0]))
        c, wc = _gauss_legendre(80, 0.0, float(row[1]))
        X, Y = np.meshgrid(a, c, indexing="ij")
        W = np.outer(wa, wc)
        pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1))
        vals = (basis.evaluate(pts) @ v).reshape(80, 80).numpy()
        ref.append(float((W * vals).sum()))
    assert torch.allclose(got, torch.tensor(ref), atol=1e-9)


# --------------------------------------------------------------------------- symbols
@pytest.mark.parametrize("power", [1, 2, 3, 4])
def test_symbol_even_powers_match_laplacian(power):
    """|W|^{2p} is (-Lap)^p, so the even integer orders check against autograd."""
    d = 2
    basis, _ = _basis(500 + power, d, n=6, sigma=2.0)
    v = torch.randn(basis.n_features, 1, generator=torch.Generator().manual_seed(power))
    x = _rand_x(500 + power, d, m=5)
    got = basis.symbol(x, lambda W: ((W * W).sum(dim=0, keepdim=True)) ** power) @ v
    # (-Lap)^p applied p times by autograd, independently of the symbol machinery
    ref = None
    xg = x.clone().requires_grad_(True)
    u = basis.evaluate(xg) @ v
    for _ in range(power):
        g = torch.autograd.grad(u.sum(), xg, create_graph=True)[0]
        lap = sum(torch.autograd.grad(g[:, k].sum(), xg, create_graph=True)[0][:, k:k + 1]
                  for k in range(d))
        u = -lap
    ref = u.detach()
    assert torch.allclose(got, ref, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("s", [0.25, 0.5, 0.75, 1.5])
def test_symbol_fractional_orders_are_diagonal(s):
    """A real radial symbol rescales each sine column; odd/fractional orders included."""
    d = 2
    basis, _ = _basis(600, d, n=6, sigma=2.0)
    x = _rand_x(600, d, m=5)
    got = basis.symbol(x, lambda W: ((W * W).sum(dim=0, keepdim=True)) ** s)
    W2 = (basis.W * basis.W).sum(dim=0, keepdim=True)
    assert torch.allclose(got, basis.evaluate(x) * W2 ** s, atol=TOL)


def test_symbol_accepts_scalar():
    basis, _ = _basis(601, 2, n=5)
    x = _rand_x(601, 2, m=4)
    got = basis.symbol(x, torch.tensor(3.0))
    assert torch.allclose(got, 3.0 * basis.evaluate(x), atol=TOL)


# --------------------------------------------------------------------------- kernels
@pytest.mark.parametrize("seed", range(3))
def test_separable_inner_products_match_quadrature(seed):
    basis, _ = _basis(700 + seed, 1, n=8, sigma=float(2.0 + seed))
    kernel = SeparableKernelOperator([lambda x: x[:, 0]], [lambda y: y[:, 0]], 0.0, 1.0, d=1)
    C = kernel.inner_products(basis)                          # (1, N) = int y phi_j(y)
    t, w = _gauss_legendre(400, 0.0, 1.0)
    pts = torch.tensor(t).reshape(-1, 1)
    ref = (torch.tensor(w * t) @ basis.evaluate(pts)).reshape(1, -1)
    assert torch.allclose(C, ref, atol=1e-11)


def test_degenerate_eigenvalue_is_analytic():
    basis, _ = _basis(701, 1, n=8, sigma=5.0)
    kernel = SeparableKernelOperator([lambda x: x[:, 0]], [lambda y: y[:, 0]], 0.0, 1.0, d=1)
    lams = degenerate_eigenvalues(kernel, basis).real
    assert abs(float(lams[0]) - 3.0) < 1e-10                  # K = xy on [0,1]


def test_inner_products_cache_follows_the_basis():
    """The cache must not survive an in-place change of the basis it was built for.

    Regression test for the September 2026 review: the key was ``(id(basis), n_features)``,
    so mutating W in place (or handing over a NEW basis that landed at the same address)
    returned the previous inner products with no warning.
    """
    kernel = SeparableKernelOperator([lambda x: x[:, 0]], [lambda y: y[:, 0]], 0.0, 1.0, d=1)
    basis, _ = _basis(702, 1, n=8, sigma=5.0)
    C1 = kernel.inner_products(basis).clone()
    with torch.no_grad():
        basis.W.mul_(2.0)                                     # same object, new numbers
    C2 = kernel.inner_products(basis)
    assert not torch.allclose(C1, C2, atol=1e-12), "stale inner products after an in-place edit"
    t, w = _gauss_legendre(400, 0.0, 1.0)
    pts = torch.tensor(t).reshape(-1, 1)
    ref = (torch.tensor(w * t) @ basis.evaluate(pts)).reshape(1, -1)
    assert torch.allclose(C2, ref, atol=1e-11)
    assert torch.allclose(kernel.inner_products(basis), C2, atol=0)   # still caches


# --------------------------------------------------------------------------- augmented
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_augmented_basis_operators(degree):
    """Every operator of the augmented basis, on both blocks, against the references."""
    d = 2
    basis, _ = _basis(800 + degree, d, n=6, sigma=2.0)
    aug = AugmentedBasis(basis, PolynomialColumns(degree=degree, dim=d))
    x = torch.rand(5, d, generator=torch.Generator().manual_seed(degree)) * 0.8 + 0.1
    v = torch.randn(aug.n_features, 1, generator=torch.Generator().manual_seed(degree))

    xg = x.clone().requires_grad_(True)
    u = aug.evaluate(xg) @ v
    g = torch.autograd.grad(u.sum(), xg, create_graph=True)[0]
    hd = torch.stack([torch.autograd.grad(g[:, k].sum(), xg, create_graph=True)[0][:, k]
                      for k in range(d)], dim=1)
    assert torch.allclose((aug.gradient(x) @ v).squeeze(-1), g.detach(), atol=1e-9)
    assert torch.allclose((aug.hessian_diag(x) @ v).squeeze(-1), hd.detach(), atol=1e-9)
    assert torch.allclose((aug.laplacian(x) @ v).squeeze(-1), hd.sum(1).detach(), atol=1e-9)

    vel = torch.tensor([0.3, -0.7])
    adv = (vel[None, :] * g.detach()).sum(1)
    assert torch.allclose((aug.advection(x, vel) @ v).squeeze(-1), adv, atol=1e-9)

    bh = 0.0                                                  # biharmonic by autograd
    for i in range(d):
        ui = torch.autograd.grad(g[:, i].sum(), xg, create_graph=True)[0][:, i]
        for j in range(d):
            gj = torch.autograd.grad(ui.sum(), xg, create_graph=True)[0][:, j]
            bh = bh + torch.autograd.grad(gj.sum(), xg, create_graph=True)[0][:, j]
    assert torch.allclose((aug.biharmonic(x) @ v).squeeze(-1), bh.detach(), atol=1e-7)

    mi = (aug.multi_integral(x, [0, 1], [0.0, 0.0]) @ v).squeeze(-1)
    ref = []
    for row in x:
        a, wa = _gauss_legendre(60, 0.0, float(row[0]))
        c, wc = _gauss_legendre(60, 0.0, float(row[1]))
        X, Y = np.meshgrid(a, c, indexing="ij")
        W = np.outer(wa, wc)
        pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1))
        ref.append(float((W * (aug.evaluate(pts) @ v).reshape(60, 60).numpy()).sum()))
    assert torch.allclose(mi, torch.tensor(ref), atol=1e-9)


def test_multi_integral_operator_accepts_augmented_basis():
    """Regression: MultiIntegralOperator.apply(aug, x) used to raise AttributeError."""
    from fastlsq.basis import MultiIntegralOperator
    d = 2
    basis, _ = _basis(801, d, n=6, sigma=2.0)
    aug = AugmentedBasis(basis, PolynomialColumns(degree=2, dim=d))
    x = torch.rand(4, d, generator=torch.Generator().manual_seed(1))
    out = MultiIntegralOperator([0, 1], [0.0, 0.0], d=d).apply(aug, x)
    assert out.shape == (4, aug.n_features)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- geometry
@pytest.mark.parametrize("name", ["disk", "flower", "annulus"])
def test_sdf_normals_match_autograd(name):
    """The outward normal is grad(psi)/|grad(psi)| -- check it against autograd on psi."""
    from fastlsq.geometry import SDFDomain
    torch.manual_seed(900)
    dom = getattr(SDFDomain, name)()
    pts = dom.sample_boundary(24)
    n = dom.normal(pts)
    x = pts.clone().requires_grad_(True)
    g = torch.autograd.grad(dom.psi(x).sum(), x)[0]
    g = g / g.norm(dim=1, keepdim=True)
    assert torch.allclose(n, g, atol=1e-6)


@pytest.mark.parametrize("name", ["disk", "flower"])
def test_robin_rows_match_the_closed_form_combination(name):
    """robin_rows(alpha, beta) must equal alpha * phi + beta * n . grad phi."""
    from fastlsq.geometry import SDFDomain
    torch.manual_seed(901)
    dom = getattr(SDFDomain, name)()
    basis, _ = _basis(901, 2, n=8, sigma=3.0)
    xb = dom.sample_boundary(12)
    a, b = 0.7, 1.3
    got = dom.robin_rows(basis, xb, alpha=a, beta=b)
    n = dom.normal(xb)
    ref = a * basis.evaluate(xb) + b * (n[:, :, None] * basis.gradient(xb)).sum(dim=1)
    assert torch.allclose(got, ref, atol=1e-10)
