# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License.

"""Incompressible flow: divergence-free and mirror bases, Stokes singular columns,
chunked normal equations, Newton rows, and two solves against exact solutions."""

import math

import pytest
import torch

import fastlsq as fl
from fastlsq.singular import oseenlet, stokes_doublet, stokeslet


@pytest.fixture(autouse=True)
def float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(old)


def fd_jacobian(f, x, h=1e-6):
    """Central differences of f: (M, d) -> (M, ...) along each axis -> (M, ..., d)."""
    cols = []
    for m in range(x.shape[1]):
        e = torch.zeros_like(x)
        e[:, m] = h
        cols.append((f(x + e) - f(x - e)) / (2 * h))
    return torch.stack(cols, -1)


# ----------------------------------------------------------------------
# DivergenceFreeBasis
# ----------------------------------------------------------------------

@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("axis", [None, 1])
def test_divergence_free_exact(d, axis):
    B = fl.DivergenceFreeBasis.random(d, 80, sigma=[1.0, 4.0], mirror_axis=axis)
    assert B.n_unknowns == (1 if d == 2 else 2) * 80
    x = torch.randn(50, d)
    val, grad, lap = B.blocks(x)
    assert val.shape == (50, d, B.n_unknowns)
    assert grad.shape == (50, d, d, B.n_unknowns)
    div = grad.diagonal(dim1=1, dim2=2).sum(-1)              # (M, P), per column
    assert div.abs().max() < 1e-12


@pytest.mark.parametrize("d", [2, 3])
def test_divergence_free_derivatives(d):
    B = fl.DivergenceFreeBasis.random(d, 40, sigma=2.0, mirror_axis=0)
    x = torch.randn(20, d)
    _, grad, lap = B.blocks(x)
    torch.testing.assert_close(grad, fd_jacobian(B.evaluate, x).permute(0, 1, 3, 2), atol=1e-7, rtol=0)
    lap_fd = fd_jacobian(B.gradient, x, h=1e-4).diagonal(dim1=2, dim2=4).sum(-1)
    torch.testing.assert_close(lap, lap_fd, atol=1e-5, rtol=0)


def test_divergence_free_mirror_and_plain():
    B = fl.DivergenceFreeBasis.random(3, 60, sigma=2.0, mirror_axis=1)
    c = torch.randn(B.n_unknowns)
    x = torch.randn(30, 3)
    R = torch.tensor([1.0, -1.0, 1.0])
    torch.testing.assert_close(B.predict(x * R, c), B.predict(x, c) * R)      # u(Rx) = R u(x)
    W, b, C = B.plain(c)
    torch.testing.assert_close(torch.sin(x @ W.T + b) @ C, B.predict(x, c))


# ----------------------------------------------------------------------
# MirrorBasis
# ----------------------------------------------------------------------

@pytest.mark.parametrize("parity", [1, -1])
def test_mirror_basis(parity):
    half = fl.SinusoidalBasis.random(3, 50, sigma=2.0)
    M = fl.MirrorBasis(half, axis=2, parity=parity)
    x = torch.randn(25, 3)
    Rx = x * torch.tensor([1.0, 1.0, -1.0])
    torch.testing.assert_close(M.evaluate(Rx), parity * M.evaluate(x))
    torch.testing.assert_close(M.evaluate(x), half.evaluate(x) + parity * half.evaluate(Rx))
    # works with any operator
    lap = fl.Op.laplacian(d=3).apply(M, x)
    torch.testing.assert_close(lap, M.laplacian(x))
    torch.testing.assert_close(M.gradient(x), fd_jacobian(M.evaluate, x).transpose(1, 2), atol=1e-7, rtol=0)
    W, b, sign = M.plain()
    beta = torch.randn(50)
    torch.testing.assert_close(torch.sin(x @ W + b) @ (sign * beta.repeat(2)) * half._inv_norm,
                               M.evaluate(x) @ beta)


# ----------------------------------------------------------------------
# Singular solutions
# ----------------------------------------------------------------------

def _points_away(n, src, rmin=0.3):
    x = torch.randn(4 * n, 3)
    x = x[(x[:, None] - src[None]).norm(dim=-1).min(dim=1).values > rmin][:n]
    return x


@pytest.mark.parametrize("kind", ["stokeslet", "doublet", "oseenlet"])
def test_closed_form_gradients(kind):
    x = _points_away(30, torch.zeros(1, 3))
    k = 0.7
    if kind == "stokeslet":
        f = lambda r: stokeslet(r)[0]
        dG = stokeslet(x)[1]
    elif kind == "doublet":
        f = lambda r: stokes_doublet(r)[0]
        dG = stokes_doublet(x)[1]
    else:
        f = lambda r: oseenlet(r, k)[0]
        dG = oseenlet(x, k)[1]
    torch.testing.assert_close(dG, fd_jacobian(f, x), atol=1e-7, rtol=0)


@pytest.mark.parametrize("oseen", [False, True])
@pytest.mark.parametrize("axis", [None, 1])
def test_singular_columns_solve_stokes(oseen, axis):
    """Every column satisfies div u = 0, and its claimed `stokes` block equals
    -nu lap u + grad p computed by differentiating its closed-form gradient."""
    nu, U = 0.05, 1.0
    src = torch.tensor([[0.0, 0.2, 0.0], [0.3, 0.1, -0.1]])
    sing = fl.StokesSingularities(src, nu=nu, mirror_axis=axis, oseen=oseen, u_inf=U)
    allsrc = torch.cat([src, src * torch.tensor([1.0, -1.0, 1.0])])
    x = _points_away(25, allsrc, 0.4)
    B = sing.blocks(x)
    P = sing.n_unknowns
    assert B["val"].shape == (25, 3, P) and B["grad"].shape == (25, 3, 3, P)
    assert B["grad"].diagonal(dim1=1, dim2=2).sum(-1).abs().max() < 1e-9
    torch.testing.assert_close(B["grad"], fd_jacobian(lambda y: sing.blocks(y)["val"], x).permute(0, 1, 3, 2),
                               atol=1e-6, rtol=0)
    torch.testing.assert_close(B["pgrad"], fd_jacobian(lambda y: sing.blocks(y)["p"], x).transpose(1, 2),
                               atol=1e-6, rtol=0)
    lap = fd_jacobian(lambda y: sing.blocks(y)["grad"], x, h=1e-4).diagonal(dim1=2, dim2=4).sum(-1)
    stokes = -nu * lap + B["pgrad"]
    torch.testing.assert_close(stokes, B["stokes"], atol=1e-5, rtol=0)


def test_seat_sources_sphere():
    a, depth = 0.5, 0.1
    body = fl.sdf_ball(a)
    d = torch.randn(200, 3)
    surf = a * d / d.norm(dim=1, keepdim=True)
    y = fl.seat_sources(body, surf, depth)
    torch.testing.assert_close(y.norm(dim=1), torch.full((200,), a - depth), atol=1e-9, rtol=0)
    # a thin plate: never pushed past 70% of the half-thickness
    plate = fl.sdf_box([-1.0, -1.0, -0.05], [1.0, 1.0, 0.05])
    top = torch.rand(50, 3) - 0.5
    top[:, 2] = 0.05
    y = fl.seat_sources(plate, top, depth=0.2)
    assert (y[:, 2] >= 0.05 - 0.7 * 0.05 - 1e-9).all() and (y[:, 2] < 0.05).all()


# ----------------------------------------------------------------------
# NormalEquations
# ----------------------------------------------------------------------

def test_normal_equations_chunked_matches_qr():
    A = torch.randn(600, 40) * torch.logspace(0, 3, 40)        # badly scaled columns
    b = torch.randn(600)
    w = torch.rand(600) + 0.5
    ne = fl.NormalEquations(40)
    for i in range(0, 600, 128):
        ne.add(A[i:i + 128], b[i:i + 128], w[i:i + 128])
    x = ne.solve()
    x_ref = torch.linalg.lstsq(A * w[:, None], (b * w)[:, None]).solution[:, 0]
    torch.testing.assert_close(x, x_ref, atol=1e-9, rtol=1e-9)
    assert ne.n_rows == 600
    c = ne.copy()
    c.add(A[:10], b[:10])
    assert c.n_rows == 610 and ne.n_rows == 600
    assert not torch.equal(c.G, ne.G)


def test_normal_equations_rank_deficient():
    A = torch.randn(100, 5)
    A = torch.cat([A, A[:, :1]], 1)                            # duplicated column
    b = torch.randn(100)
    ne = fl.NormalEquations(6)
    ne.add(A, b)
    x = ne.solve()
    assert torch.isfinite(x).all() and x.abs().max() < 10
    best = torch.linalg.lstsq(A, b[:, None]).solution[:, 0]
    assert float((A @ x - b).norm()) < float((A @ best - b).norm()) * (1 + 1e-8)
    # a negative-definite perturbation forces the ridge up
    ne.G -= 2.0 * torch.diag(torch.diagonal(ne.G))
    with pytest.raises(torch.linalg.LinAlgError):
        ne.solve(max_tries=3)


# ----------------------------------------------------------------------
# IncompressibleFlow rows
# ----------------------------------------------------------------------

def _flow3d(sources=True, axis=1):
    vel = fl.DivergenceFreeBasis.random(3, 60, sigma=[1.0, 3.0], mirror_axis=axis)
    pb = fl.SinusoidalBasis.random(3, 40, sigma=2.0)
    pre = pb if axis is None else fl.MirrorBasis(pb, axis=axis)
    sing = (fl.StokesSingularities(torch.tensor([[0.0, 0.1, 0.0], [0.2, 0.15, 0.05]]), nu=0.1, mirror_axis=axis)
            if sources else None)
    return fl.IncompressibleFlow(vel, pre, nu=0.1, u_inf=[1.0, 0.0, 0.0], singular=sing)


@pytest.mark.parametrize("sources", [False, True])
def test_newton_rows_are_the_linearisation(sources):
    flow = _flow3d(sources)
    x = torch.rand(40, 3) + torch.tensor([0.6, 0.6, 0.6])
    th0 = 0.1 * torch.randn(flow.n_unknowns)
    U, GU, _ = flow.evaluate(x, th0)
    A, b = flow.momentum_rows(x, U, GU)

    def nonlinear(th):
        return flow.residual(x, th)["res"].T.reshape(-1)

    torch.testing.assert_close(A @ th0 - b, nonlinear(th0), atol=1e-10, rtol=0)
    dth = torch.randn(flow.n_unknowns)
    errs = []
    for eps in (1e-2, 1e-3):
        th = th0 + eps * dth
        errs.append(float((nonlinear(th) - (A @ th - b)).norm()))
    assert errs[1] < errs[0] / 50                                # O(eps^2)


def test_boundary_rows():
    flow = _flow3d()
    x = torch.rand(30, 3) + 0.5
    th = 0.1 * torch.randn(flow.n_unknowns)
    u, gu, p = flow.evaluate(x, th)
    A, b = flow.dirichlet_rows(x, torch.tensor([0.3, 0.0, -0.2]))
    torch.testing.assert_close(A @ th - b, (u - torch.tensor([0.3, 0.0, -0.2])).T.reshape(-1))
    # traction-free outlet with normal +x, written out by hand
    A, b = flow.traction_rows(x, 0)
    nu = flow.nu
    hand = torch.stack([-p + 2 * nu * gu[:, 0, 0], nu * (gu[:, 1, 0] + gu[:, 0, 1]),
                        nu * (gu[:, 2, 0] + gu[:, 0, 2])])
    torch.testing.assert_close(A @ th - b, hand.reshape(-1))
    torch.testing.assert_close(flow.traction(x, 0, th), hand.T)
    # free slip on a plane z = const: u_z = 0, du_x/dz = du_y/dz = 0 (and a zero row)
    A, b = flow.slip_rows(x, 2)
    r = A @ th - b
    torch.testing.assert_close(r[:30], u[:, 2])
    torch.testing.assert_close(r[30:].reshape(3, 30), torch.stack([gu[:, 0, 2], gu[:, 1, 2], 0 * u[:, 0]]))
    # pressure gauge
    A, b = flow.pressure_rows(x, 0.25)
    torch.testing.assert_close(A @ th - b, p - 0.25)


@pytest.mark.parametrize("axis", [None, 1])
def test_plain_export(axis):
    flow = _flow3d(axis=axis)
    th = torch.randn(flow.n_unknowns)
    x = torch.rand(20, 3) + 0.5
    u, _, p = flow.evaluate(x, th)
    e = flow.plain(th)
    u_rff = e["u_inf"] + torch.sin(x @ e["W"].T + e["b"]) @ e["C"]
    p_rff = torch.sin(x @ e["Wp"].T + e["bp"]) @ e["Cp"]
    s = e["singular"]
    sing = fl.StokesSingularities(s["pos"], nu=flow.nu)            # mirror already expanded
    coef = torch.cat([s["force"].reshape(-1), s["doublet"].reshape(-1), s["monopole"], s["dipole"].reshape(-1)])
    Bs = sing.blocks(x)
    torch.testing.assert_close(u_rff + Bs["val"] @ coef, u)
    torch.testing.assert_close(p_rff + Bs["p"] @ coef, p)


# ----------------------------------------------------------------------
# Solves against exact solutions
# ----------------------------------------------------------------------

def test_kovasznay_newton():
    Re = 40.0
    lam = Re / 2 - math.sqrt(Re ** 2 / 4 + 4 * math.pi ** 2)

    def exact(x):
        e = torch.exp(lam * x[:, 0])
        c, s = torch.cos(2 * math.pi * x[:, 1]), torch.sin(2 * math.pi * x[:, 1])
        return torch.stack([1 - e * c, lam / (2 * math.pi) * e * s], 1), 0.5 * (1 - torch.exp(2 * lam * x[:, 0]))

    lo, hi = torch.tensor([-0.5, -0.5]), torch.tensor([1.0, 1.5])
    vel = fl.DivergenceFreeBasis.random(2, 400, sigma=[1.0, 3.0, 6.0])
    pre = fl.SinusoidalBasis.random(2, 200, sigma=3.0)
    flow = fl.IncompressibleFlow(vel, pre, nu=1 / Re)
    x = lo + (hi - lo) * torch.rand(3000, 2)
    t = torch.rand(200)
    xb = torch.cat([torch.stack([lo[0] + 0 * t, lo[1] + 2 * t], 1), torch.stack([hi[0] + 0 * t, lo[1] + 2 * t], 1),
                    torch.stack([lo[0] + 1.5 * t, lo[1] + 0 * t], 1), torch.stack([lo[0] + 1.5 * t, hi[1] + 0 * t], 1)])
    x0 = torch.tensor([[1.0, 0.25]])
    res = fl.solve_navier_stokes(flow, x, [fl.Dirichlet(xb, exact(xb)[0], weight=10.0),
                                           fl.PressurePoint(x0, exact(x0)[1], weight=10.0)],
                                 max_iter=8, tol=1e-6)
    assert res.converged
    steps = [h["step"] for h in res.history]
    assert steps[3] < steps[2] ** 1.6 and steps[4] < steps[3] ** 1.6     # quadratic convergence
    xt = lo + (hi - lo) * torch.rand(2000, 2)
    ue, pe = exact(xt)
    u, gu, p = flow.evaluate(xt, res.theta)
    assert float((u - ue).norm() / ue.norm()) < 1e-4
    assert float((p - pe).norm() / pe.norm()) < 1e-3
    assert float((gu[:, 0, 0] + gu[:, 1, 1]).abs().max()) < 1e-12
    torch.testing.assert_close(res.velocity(xt), u)


def test_stokes_sphere_with_singular_columns():
    a, nu = 0.5, 1.0
    lo, hi = torch.tensor([-2.0, 0.0, -2.0]), torch.tensor([3.0, 2.0, 2.0])

    def exact(x):
        r = x.norm(dim=1, keepdim=True)
        U = torch.tensor([1.0, 0.0, 0.0])
        ux = x[:, :1]
        return U - 0.75 * a * (U / r + x * ux / r ** 3) - 0.25 * a ** 3 * (U / r ** 3 - 3 * x * ux / r ** 5)

    def half_sphere(n, radius):
        d = torch.randn(n, 3)
        d[:, 1] = d[:, 1].abs()
        return radius * d / d.norm(dim=1, keepdim=True)

    def fluid(n):
        x = lo + (hi - lo) * torch.rand(4 * n, 3)
        x = x[x.norm(dim=1) > a][:n]
        return torch.cat([x, half_sphere(n // 2, 1.0) * (a + 0.8 * torch.rand(n // 2, 1) ** 2)])

    faces = []
    for ax in range(3):
        for val in (lo[ax], hi[ax]):
            if not (ax == 1 and val == 0):
                f = lo + (hi - lo) * torch.rand(200, 3)
                f[:, ax] = val
                faces.append(f)
    xb = torch.cat(faces)
    src = half_sphere(80, 0.7 * a)
    src[:, 1] = src[:, 1].clamp_min(0.02)
    vel = fl.DivergenceFreeBasis.random(3, 300, sigma=[1.0, 3.0], mirror_axis=1)
    pre = fl.MirrorBasis(fl.SinusoidalBasis.random(3, 150, sigma=2.0), axis=1)
    flow = fl.IncompressibleFlow(vel, pre, nu=nu, u_inf=[1.0, 0.0, 0.0],
                                 singular=fl.StokesSingularities(src, nu=nu, mirror_axis=1))
    res = fl.solve_navier_stokes(flow, fluid(3000), [fl.Dirichlet(half_sphere(1000, a), weight=10.0),
                                                     fl.Dirichlet(xb, exact(xb), weight=10.0)], newton=False)
    xt = fluid(2000)
    ue = exact(xt)
    assert float((flow.evaluate(xt, res.theta)[0] - ue).norm() / ue.norm()) < 2e-3
    xq = half_sphere(4000, a)
    drag = float(2 * flow.traction(xq, xq / a, res.theta)[:, 0].mean() * 2 * math.pi * a ** 2)
    assert abs(drag / (6 * math.pi * nu * a) - 1) < 1e-2


def test_sdf_grid_ball():
    h, n = 0.05, 41
    ax = torch.arange(n) * h - 1.0
    P = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1)
    psi = fl.sdf_grid(P.norm(dim=-1) - 0.6, [-1.0] * 3, h)
    x = torch.randn(300, 3)
    x = 0.6 * x / x.norm(dim=1, keepdim=True) * (1 + 0.3 * (torch.rand(300, 1) - 0.5))
    assert (psi(x) - (x.norm(dim=1) - 0.6)).abs().max() < 2e-3
    far = torch.tensor([[3.0, 0.0, 0.0]])
    assert abs(float(psi(far)) - 2.4) < 1e-9
    xb = fl.project_to_boundary(psi, x)
    assert (xb.norm(dim=1) - 0.6).abs().max() < 2e-3
