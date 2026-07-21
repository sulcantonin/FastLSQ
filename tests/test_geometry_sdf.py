# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Tests for SDF / membership-oracle geometry.

Checks the three things the solver actually needs from a domain -- interior
points, boundary points, outward normals -- against closed-form references, plus
the failure modes that used to be silent (a degenerate bounding box, an
unconvergeable projection).
"""

import math

import numpy as np
import pytest
import torch

from fastlsq.basis import SinusoidalBasis
from fastlsq.geometry import (
    SDFDomain,
    outward_normal,
    project_to_boundary,
    sample_sdf,
    sdf_ball,
    sdf_box,
    sdf_flower,
    sdf_polygon,
    _value_and_grad,
)

torch.set_default_dtype(torch.float64)


# ======================================================================
# Interior sampling
# ======================================================================

def test_disk_sampling_is_inside_and_uniform():
    """Rejection sampling is unbiased: the radial CDF of a uniform disk is r^2."""
    torch.manual_seed(0)
    dom = SDFDomain.disk(1.0)
    x = dom.sample(20000)
    r = x.norm(dim=1)
    assert (r <= 1.0).all()
    # P(r < t) = t^2 for a uniform disk.
    for t in (0.25, 0.5, 0.75):
        assert abs((r < t).double().mean().item() - t ** 2) < 0.02


def test_lshape_excludes_the_notch():
    """The reentrant corner is genuinely cut out, and the area fraction is 3/4."""
    torch.manual_seed(1)
    dom = SDFDomain.lshape(1.0, 0.5)
    x = dom.sample(8000)
    assert not ((x[:, 0] > 0.5) & (x[:, 1] > 0.5)).any()
    frac = dom.contains(torch.rand(40000, 2)).double().mean().item()
    assert abs(frac - 0.75) < 0.01


def test_annulus_excludes_the_hole():
    """A multiply-connected domain keeps its hole empty."""
    torch.manual_seed(2)
    dom = SDFDomain.annulus(0.3, 1.0)
    r = dom.sample(8000).norm(dim=1)
    assert (r >= 0.3).all() and (r <= 1.0).all()


def test_tokamak_cross_section_is_d_shaped():
    """The Miller cross-section spans the expected R range and is up-down symmetric."""
    torch.manual_seed(3)
    dom = SDFDomain.tokamak(R0=1.0, a=0.4, elongation=1.8, triangularity=0.4)
    x = dom.sample(6000)
    assert dom.contains(x).all()
    assert 0.59 < x[:, 0].min() < 0.65 and 1.35 < x[:, 0].max() < 1.41
    # Half-height is the elongated minor radius, kappa * a.
    assert abs(x[:, 1].abs().max().item() - 1.8 * 0.4) < 0.02
    # Elongation (kappa > 1) makes the full height exceed the width 2a.
    width = (x[:, 0].max() - x[:, 0].min()).item()
    height = 2 * x[:, 1].abs().max().item()
    assert height > width
    assert abs(x[:, 1].mean().item()) < 0.05


def test_csg_difference_respects_the_hole():
    """A CSG domain samples correctly -- and its bounding box survives composition.

    Regression guard: a (2, d) bounds tensor produced by CSG was once re-parsed
    as per-axis pairs and transposed, collapsing the box to zero width so that
    nothing was ever accepted and the sampler spun until max_rounds.
    """
    torch.manual_seed(4)
    dom = SDFDomain.disk(1.0) - SDFDomain.disk(0.25, center=(0.4, 0.0))
    x = dom.sample(4000)
    assert (x.norm(dim=1) <= 1.0).all()
    assert ((x - torch.tensor([0.4, 0.0])).norm(dim=1) > 0.25).all()
    lo, hi = dom.bounds()
    assert torch.allclose(lo, torch.tensor([-1.0, -1.0]))
    assert torch.allclose(hi, torch.tensor([1.0, 1.0]))


def test_csg_union_grows_the_bounding_box():
    dom = SDFDomain.disk(1.0) | SDFDomain.box([1.0, -0.3], [2.0, 0.3])
    lo, hi = dom.bounds()
    assert torch.allclose(lo, torch.tensor([-1.0, -1.0]))
    assert torch.allclose(hi, torch.tensor([2.0, 1.0]))


def test_csg_intersection_tightens_the_bounding_box():
    """A n B is inside both boxes, so the overlap is a valid and much cheaper box."""
    dom = SDFDomain.disk(1.0) & SDFDomain.box([0.0, 0.0], [0.6, 0.6])
    lo, hi = dom.bounds()
    assert torch.allclose(lo, torch.tensor([0.0, 0.0]))
    assert torch.allclose(hi, torch.tensor([0.6, 0.6]))
    torch.manual_seed(15)
    x = dom.sample(2000)
    assert dom.contains(x).all()
    assert (x >= 0.0).all() and (x <= 0.6).all()


def test_csg_intersection_of_disjoint_domains_raises():
    with pytest.raises(ValueError, match="intersection is empty"):
        SDFDomain.disk(1.0) & SDFDomain.box([5.0, 5.0], [6.0, 6.0])


def test_degenerate_bounds_raise_instead_of_hanging():
    with pytest.raises(ValueError, match="non-positive extent"):
        SDFDomain(sdf_ball(1.0), [(0.0, 0.0), (-1.0, 1.0)])


def test_empty_domain_raises_promptly():
    """A box that misses the domain reports the acceptance rate, it does not hang."""
    with pytest.raises(RuntimeError, match="acceptance rate"):
        sample_sdf(sdf_ball(0.1), 100, [(5.0, 6.0), (5.0, 6.0)], max_rounds=12)


# ======================================================================
# Boundary projection and normals
# ======================================================================

def test_disk_boundary_and_normals_are_exact():
    """For the disk both are known in closed form: |x| = 1 and n = x/|x|."""
    torch.manual_seed(5)
    dom = SDFDomain.disk(1.0)
    xb = dom.sample_boundary(800)
    assert (xb.norm(dim=1) - 1.0).abs().max() < 1e-12
    n = dom.normal(xb)
    assert torch.allclose(n, xb / xb.norm(dim=1, keepdim=True), atol=1e-12)
    assert (n.norm(dim=1) - 1.0).abs().max() < 1e-14


def test_annulus_inner_normal_points_inward():
    """On an interior boundary the outward normal points toward the centre."""
    torch.manual_seed(6)
    dom = SDFDomain.annulus(0.3, 1.0)
    xb = dom.sample_boundary(600)
    r = xb.norm(dim=1)
    radial = (dom.normal(xb) * xb / r.unsqueeze(1)).sum(dim=1)
    assert radial[r < 0.5].mean() < -0.999      # inner: inward
    assert radial[r > 0.5].mean() > 0.999       # outer: outward


def test_flower_is_not_a_distance_function_but_still_projects():
    """The case that separates a correct projection from a naive one.

    sdf_flower has ||grad psi|| spanning ~1 to ~10, so the textbook
    x - psi*grad(psi) step (valid only when ||grad psi|| == 1) mis-steps.  The
    normalised, damped iteration still lands on the true curve
    r = R(1 + a cos(k theta)).
    """
    torch.manual_seed(7)
    R, amp, k = 1.0, 0.3, 5
    dom = SDFDomain.flower(R, amp, k)
    seeds = dom.sample(500)

    _, g = _value_and_grad(sdf_flower(R, amp, k), seeds)
    gn = g.norm(dim=1)
    assert gn.max() > 3.0, "flower should NOT be a unit-gradient SDF"

    xb = dom.sample_boundary(400)
    theta = torch.atan2(xb[:, 1], xb[:, 0])
    r = xb.norm(dim=1)
    assert (r - R * (1 + amp * torch.cos(k * theta))).abs().max() < 1e-7


def test_projection_converges_from_most_interior_seeds():
    """Damped Newton converges from essentially anywhere in a non-SDF domain.

    Seeds within r < 0.2 of the flower's centre are genuinely degenerate (no
    unique nearest boundary point, and the angular term dominates the gradient),
    so a small failure fraction is expected and is filtered by
    sample_boundary_sdf rather than hidden.
    """
    torch.manual_seed(8)
    dom = SDFDomain.flower(1.0, 0.3, 5)
    seeds = dom.sample(2000)
    res = dom(dom.project(seeds)).abs()
    assert (res < 1e-10).double().mean() > 0.98


def test_box_normals_are_axis_aligned():
    torch.manual_seed(9)
    dom = SDFDomain.box([0.0, 0.0], [1.0, 1.0])
    xb = dom.sample_boundary(500)
    n = dom.normal(xb)
    # Every face normal is a signed unit basis vector.
    assert (n.abs().max(dim=1).values - 1.0).abs().max() < 1e-9
    assert n.abs().min(dim=1).values.max() < 1e-9


def test_polygon_sdf_matches_analytic_square():
    """The polygon SDF reproduces the exact box SDF on a square."""
    torch.manual_seed(10)
    verts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    poly, box = sdf_polygon(verts), sdf_box([0.0, 0.0], [1.0, 1.0])
    x = torch.rand(2000, 2) * 3.0 - 1.0
    assert (poly(x) - box(x)).abs().max() < 1e-12


def test_polygon_normals_are_nonzero_on_the_boundary():
    """Regression: a sqrt(min d^2) SDF yields a ZERO autograd normal on the boundary.

    sdf_polygon must floor the squared distance before the square root, and a
    floor has exactly zero gradient below its threshold -- which is where every
    boundary point sits.  outward_normal therefore falls back to central
    differences.  Without that fallback every polygon normal came back as the
    zero vector, and a Neumann solve on the tokamak was off by 32%.
    """
    torch.manual_seed(14)
    dom = SDFDomain.tokamak()
    xb = dom.sample_boundary(400)
    n = dom.normal(xb)
    assert (n.norm(dim=1) - 1.0).abs().max() < 1e-6, "normals must be unit, not zero"

    # A normal must be perpendicular to the local boundary: stepping along it
    # changes psi at unit rate, stepping across it does not.
    h = 1e-5
    along = (dom(xb + h * n) - dom(xb - h * n)) / (2 * h)
    assert (along - 1.0).abs().max() < 1e-3
    tangent = torch.stack([-n[:, 1], n[:, 0]], dim=1)
    across = (dom(xb + h * tangent) - dom(xb - h * tangent)) / (2 * h)
    assert across.abs().max() < 1e-3


def test_polygon_sign_is_orientation_independent():
    verts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    cw, ccw = sdf_polygon(verts), sdf_polygon(verts[::-1])
    x = torch.rand(500, 2) * 3.0 - 1.0
    assert (cw(x) - ccw(x)).abs().max() < 1e-12


# ======================================================================
# Boundary operator blocks
# ======================================================================

def test_neumann_rows_match_finite_differences():
    """dphi/dn assembled from the analytic gradient equals a directional FD."""
    torch.manual_seed(11)
    dom = SDFDomain.disk(1.0)
    basis = SinusoidalBasis.random(2, 32, sigma=2.0)
    xb = dom.sample_boundary(40)
    n = dom.normal(xb)

    got = dom.neumann_rows(basis, xb)
    h = 1e-6
    fd = (basis.evaluate(xb + h * n) - basis.evaluate(xb - h * n)) / (2 * h)
    assert (got - fd).abs().max() < 1e-6


def test_robin_rows_combine_value_and_flux():
    torch.manual_seed(12)
    dom = SDFDomain.disk(1.0)
    basis = SinusoidalBasis.random(2, 24, sigma=2.0)
    xb = dom.sample_boundary(30)
    a, b = 2.0, 0.5
    got = dom.robin_rows(basis, xb, alpha=a, beta=b)
    ref = a * basis.evaluate(xb) + b * dom.neumann_rows(basis, xb)
    assert torch.allclose(got, ref, atol=1e-14)


def test_neumann_poisson_on_the_disk():
    """End-to-end: -Delta u = f with du/dn = g on a disk, solved in one shot.

    Manufactured u = x^2 - y^2 (harmonic, so f = 0) with the exact Neumann data
    read off the analytic gradient.  Pure Neumann fixes u only up to a constant,
    so the comparison is made after removing the mean.
    """
    from fastlsq.linalg import solve_lstsq

    torch.manual_seed(13)
    dom = SDFDomain.disk(1.0)
    basis = SinusoidalBasis.random(2, 400, sigma=3.0)

    x = dom.sample(3000)
    xb = dom.sample_boundary(600)
    nb = dom.normal(xb)

    def u_exact(p):
        return (p[:, 0] ** 2 - p[:, 1] ** 2).reshape(-1, 1)

    # grad u = (2x, -2y); g = n . grad u
    g = (nb * torch.stack([2 * xb[:, 0], -2 * xb[:, 1]], dim=1)).sum(1, keepdim=True)

    w = 50.0
    A = torch.cat([-basis.laplacian(x), w * dom.neumann_rows(basis, xb)])
    rhs = torch.cat([torch.zeros(x.shape[0], 1), w * g])
    beta = solve_lstsq(A, rhs, mu=1e-10)

    xt = dom.sample(2000)
    got = basis.evaluate(xt) @ beta
    ref = u_exact(xt)
    got = got - got.mean()
    ref = ref - ref.mean()
    rel = (torch.norm(got - ref) / torch.norm(ref)).item()
    assert rel < 5e-3, f"Neumann Poisson on disk rel-L2 = {rel:.2e}"
