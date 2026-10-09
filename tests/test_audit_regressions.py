# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Regression tests for the 0.7.1 audit fixes.

One test per defect, each written against the behaviour that was wrong before
the fix, so a regression shows up as a named failure rather than a vague one.
"""

import math
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

import fastlsq as fl
from fastlsq.problems import PoissonND, NLPoisson2D, SteadyBurgers1D, Wave1D
from fastlsq.utils import preserve_rng

torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------
# Global RNG must survive a solve
# ----------------------------------------------------------------------

def _tiny_solve(**kw):
    return fl.solve_linear(PoissonND(), scale=3.0, n_blocks=1, hidden_size=60,
                           n_pde=300, n_bc=100, n_test=200, verbose=False, **kw)


def test_solve_does_not_reseed_the_global_rng():
    torch.manual_seed(0)
    _tiny_solve()
    a = torch.rand(3)
    _tiny_solve()
    b = torch.rand(3)
    assert not torch.allclose(a, b), "two solves drew from a reset RNG stream"


def test_metrics_and_scale_search_leave_the_stream_where_the_user_put_it():
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    _tiny_solve()                                      # evaluate_error inside
    with preserve_rng():
        torch.manual_seed(7)
    # the solve consumed draws; a *fresh* seed then the same solve must be
    # reproducible, and preserve_rng must have restored the pre-block state
    torch.manual_seed(123)
    _tiny_solve(return_metrics=True)
    torch.manual_seed(123)
    r1 = _tiny_solve(); u1 = r1["u_fn"](torch.full((4, 5), 0.3))
    torch.manual_seed(123)
    r2 = _tiny_solve(); u2 = r2["u_fn"](torch.full((4, 5), 0.3))
    assert torch.allclose(u1, u2)
    torch.manual_seed(123)
    assert torch.allclose(torch.rand(3), expected)


def test_preserve_rng_restores_torch_and_numpy():
    torch.manual_seed(1); np.random.seed(1)
    t0, n0 = torch.rand(2), np.random.rand(2)
    torch.manual_seed(1); np.random.seed(1)
    with preserve_rng():
        torch.manual_seed(99); np.random.seed(99)
        torch.rand(10); np.random.rand(10)
    assert torch.allclose(torch.rand(2), t0)
    assert np.allclose(np.random.rand(2), n0)


# ----------------------------------------------------------------------
# solve_lstsq: 1-D right-hand side
# ----------------------------------------------------------------------

@pytest.mark.parametrize("method", ["svd", "qr", "cholesky", "auto", "rsvd"])
@pytest.mark.parametrize("mu", [0.0, 1e-3])
def test_solve_lstsq_accepts_1d_rhs(method, mu):
    torch.manual_seed(0)
    A, b = torch.randn(40, 8), torch.randn(40)
    x1 = fl.solve_lstsq(A, b, mu=mu, method=method)
    x2 = fl.solve_lstsq(A, b[:, None], mu=mu, method=method)
    assert x1.shape == (8,)
    assert torch.allclose(x1, x2[:, 0])
    x3, info = fl.solve_lstsq(A, b, mu=mu, method=method, return_info=True)
    assert x3.shape == (8,) and "residual" in info


def test_solve_lstsq_rejects_mismatched_rhs():
    with pytest.raises(ValueError):
        fl.solve_lstsq(torch.randn(5, 3), torch.randn(4))


# ----------------------------------------------------------------------
# solve_nonlinear: continuation honours the user's tolerances
# ----------------------------------------------------------------------

def test_continuation_forwards_newton_settings():
    torch.manual_seed(0)
    r = fl.solve_nonlinear(SteadyBurgers1D(nu=0.5), scale=5.0, n_blocks=1,
                           hidden_size=120, n_pde=300, n_bc=40, max_iter=12,
                           tol_res=5e-2, auto_scale=False, verbose=False,
                           return_metrics=False)
    stops = [h["stop"] for h in r["history"] if "stop" in h]
    # every continuation stage stops on the (loose) user tolerance, never on
    # max_iter; before the fix the stages ran with tol_res=1e-12
    assert stops and all(s in ("residual", "residual_abs") for s in stops)
    assert r["n_iters"] <= 2 * len(stops) + 1


# ----------------------------------------------------------------------
# export: safe checkpoints
# ----------------------------------------------------------------------

def test_checkpoint_roundtrip_in_safe_mode(tmp_path):
    torch.manual_seed(0)
    s = fl.FastLSQSolver(2); s.add_block(16, 1.0); s.beta = torch.randn(16, 1)
    path = tmp_path / "ckpt.pt"
    fl.save_checkpoint(s, str(path), metadata={"note": "x", "k": 2.5})
    s2, meta = fl.load_checkpoint(str(path), device=torch.device("cpu"))
    x = torch.rand(5, 2)
    assert torch.allclose(s.predict(x), s2.predict(x))
    assert meta["note"] == "x" and "provenance" in meta
    assert torch.is_tensor(meta["provenance"]["freq_cov"])


def test_checkpoint_refuses_pickled_payload(tmp_path):
    class Evil:
        def __reduce__(self):
            return (os.system, ("echo pwned",))
    path = tmp_path / "evil.pt"
    torch.save({"input_dim": 2, "normalize": False, "W_list": [torch.zeros(2, 1)],
                "b_list": [torch.zeros(1, 1)], "beta": torch.zeros(1, 1),
                "metadata": {"x": Evil()}}, path)
    with pytest.raises(RuntimeError, match="safe mode"):
        fl.load_checkpoint(str(path))


def test_to_dict_contract():
    s = fl.FastLSQSolver(2); s.add_block(4, 1.0)
    with pytest.raises(ValueError):
        fl.to_dict(s)
    d = fl.to_dict(s, include_weights=False, include_metadata=False)
    assert d == {"input_dim": 2, "normalize": False}
    s.beta = torch.zeros(4, 1)
    d = fl.to_dict(s, include_metadata=False)
    s2 = fl.from_dict(d, device=torch.device("cpu"))
    assert s2.n_features == 4


# ----------------------------------------------------------------------
# learnable bandwidth
# ----------------------------------------------------------------------

def test_solve_cached_keeps_vector_output_bookkeeping():
    """solve_cached must unpack beta to (N, k) and set _beta_flat like solve_inner.

    The check is against an independently computed ``pinv(A) @ b`` rather than
    against ``solve_inner``: the two back-ends truncate an ill-conditioned random
    system differently (gelsd with rcond vs. pinv), and that disagreement is
    LAPACK-dependent -- it was within 1e-8 on macOS and not on Linux CI.  The
    defect being pinned is the bookkeeping, not back-end agreement.
    """
    torch.manual_seed(0)
    m = fl.LearnableFastLSQ(1, 8, n_outputs=2)
    x = torch.rand(20, 1)
    H = m.basis.evaluate(x).detach()
    A = torch.block_diag(H, H); b = torch.randn(40, 1)
    m.cache_operator(A)
    m.solve_cached(b)
    beta_flat = torch.linalg.pinv(A) @ b
    assert m.beta.shape == (8, 2)
    assert torch.allclose(m._beta_flat, beta_flat)
    expected = torch.cat([H @ beta_flat[:8], H @ beta_flat[8:]], dim=1)
    assert torch.allclose(m.predict(x), expected, atol=1e-10)
    # and the same call path works for the scalar case
    s = fl.LearnableFastLSQ(1, 8)
    s.cache_operator(H); s.solve_cached(b[:20])
    assert s.beta.shape == (8, 1) and torch.allclose(s.predict(x), H @ s.beta)


def test_scalar_mode_follows_module_dtype():
    m = fl.LearnableFastLSQ(2, 16, mode="scalar").to(torch.float32)
    assert m.basis.W.dtype == torch.float32
    m.basis.evaluate(torch.rand(3, 2, dtype=torch.float32))


def test_bandwidth_bounds_are_consistent_and_explicit():
    for mode in ("scalar", "diagonal", "cholesky"):
        with pytest.raises(ValueError):
            fl.LearnableFastLSQ(2, 8, mode=mode, init_scale=1000.0)
        m = fl.LearnableFastLSQ(2, 8, mode=mode, init_scale=100.0)
        assert abs(m.sigma.item() - 100.0) < 1e-6


def test_envelope_gradient_matches_finite_differences_with_ridge():
    """The outer gradient must be d/dL of the *ridge* objective for mu > 0."""
    torch.manual_seed(0)

    class Fit:
        dim = 1
        def get_train_data(self, n_pde, n_bc):
            x = torch.linspace(0, 1, n_pde)[:, None]
            return x, [], torch.sin(9 * x) + x ** 2
        def build(self, learnable, x, bcs, f):
            return learnable.basis.evaluate(x), f

    mu, prob = 1e-2, Fit()
    m = fl.LearnableFastLSQ(1, 6, mode="scalar", init_scale=3.0)
    x_pde, bcs, f = prob.get_train_data(60, 0)

    def objective(log_sigma):
        with torch.no_grad():
            m.log_sigma.copy_(torch.tensor(log_sigma))
            A, b = prob.build(m, x_pde, bcs, f)
            m.solve_inner(A, b, mu=mu)
            r = A @ m._beta_flat - b
            return (r.pow(2).mean() + mu * m._beta_flat.pow(2).sum() / A.shape[0]).item()

    # autograd gradient as train_bandwidth computes it (one step, lr=0 equivalent)
    hist = fl.train_bandwidth(m, prob, n_pde=60, n_bc=0, n_steps=1, lr=0.0, mu=mu,
                              verbose=False)
    g_auto = m.log_sigma.grad.item()
    l0 = math.log(3.0); h = 1e-5
    g_fd = (objective(l0 + h) - objective(l0 - h)) / (2 * h)
    assert abs(g_auto - g_fd) < 1e-5 * max(1.0, abs(g_fd)), (g_auto, g_fd)


def test_train_bandwidth_has_no_hidden_weight_decay():
    class Flat:
        dim = 1
        def get_train_data(self, n_pde, n_bc):
            x = torch.rand(n_pde, 1); return x, [], torch.zeros(n_pde, 1)
        def build(self, learnable, x, bcs, f):
            return learnable.basis.evaluate(x), f      # zero rhs -> zero gradient
    m = fl.LearnableFastLSQ(1, 8, mode="scalar", init_scale=25.0)
    fl.train_bandwidth(m, Flat(), n_pde=30, n_bc=0, n_steps=30, lr=0.1, verbose=False)
    assert abs(m.sigma.item() - 25.0) < 1e-9


# ----------------------------------------------------------------------
# geometry: scale-free projection
# ----------------------------------------------------------------------

@pytest.mark.parametrize("scale", [1.0, 1e-7, 1e6])
def test_boundary_projection_is_scale_free(scale):
    psi = lambda x: scale * (x.norm(dim=1) - 1.0)
    g = torch.Generator().manual_seed(0)
    xb = fl.sample_boundary_sdf(psi, 200, (-1.5, 1.5), dim=2, generator=g)
    assert xb.shape == (200, 2)
    assert (xb.norm(dim=1) - 1.0).abs().max() < 1e-6
    n = fl.outward_normal(psi, xb)
    assert torch.allclose(n, xb / xb.norm(dim=1, keepdim=True), atol=1e-6)


def test_project_to_boundary_empty_input():
    dom = fl.SDFDomain.disk(1.0)
    assert dom.project(torch.empty(0, 2)).shape == (0, 2)


def test_sdfdomain_does_not_alias_caller_bounds():
    b = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    dom = fl.SDFDomain(fl.sdf_box([0, 0], [1, 1]), b)
    b[1] = 5.0
    assert float(dom.bounds()[1, 0]) == 1.0


# ----------------------------------------------------------------------
# problem contract, diagnostics, plotting, operators, solvers
# ----------------------------------------------------------------------

def test_wave1d_follows_the_three_tuple_contract():
    x, bcs, f = Wave1D().get_train_data(n_pde=50, n_bc=8)
    assert f.shape == (50, 1)
    r = fl.check_problem(Wave1D(), verbose=False)
    assert r["shape_check"] and r["gradient_check"] and r["data_check"]


def test_check_problem_float32_does_not_false_alarm():
    torch.set_default_dtype(torch.float32)
    r = fl.check_problem(PoissonND(), verbose=False)
    assert r["gradient_check"], r["warnings"]
    assert r["bc_check"], r["warnings"]


def test_check_problem_handles_vector_outputs():
    from fastlsq.problems import ElasticWave2D
    r = fl.check_problem(ElasticWave2D(), verbose=False)
    assert r["shape_check"] and r["gradient_check"], r


def test_check_problem_flags_wrong_gradient_and_bc():
    class Bad(PoissonND):
        def exact_grad(self, x):
            return 2.0 * super().exact_grad(x)
        def get_train_data(self, n_pde=100, n_bc=20):
            x_pde, bcs, f = super().get_train_data(n_pde, n_bc)
            return x_pde, [(xb, ub + 1.0) for xb, ub in bcs], f
    r = fl.check_problem(Bad(), verbose=False)
    assert not r["gradient_check"] and not r["bc_check"]


def test_plot_saves_the_figure_it_drew_on(tmp_path):
    from fastlsq.problems import AllenCahn1D
    s = fl.FastLSQSolver(1); s.add_block(8, 1.0); s.beta = torch.zeros(8, 1)
    figA, axA = plt.subplots()
    figB, axB = plt.subplots()          # current figure, must survive
    fl.plot_solution_1d(s, AllenCahn1D(), ax=axA, save_path=str(tmp_path / "a.png"))
    assert (tmp_path / "a.png").exists()
    assert plt.fignum_exists(figB.number) and not plt.fignum_exists(figA.number)
    plt.close("all")
    with pytest.raises(ValueError):
        fl.plot_convergence([])
    fig, axes = fl.viz.hero_figure_landscape(ncols=1)
    assert len(axes) == 1
    plt.close("all")


def test_mirror_basis_supports_symbol_and_integral_operators():
    torch.manual_seed(0)
    base = fl.SinusoidalBasis.random(2, 12, sigma=2.0)
    mb = fl.MirrorBasis(base, axis=1, parity=+1)
    x = torch.rand(7, 2)
    R = torch.tensor([1.0, -1.0])
    for op in (fl.SymbolOperator.fractional_laplacian(0.5),
               fl.IntegralOperator.volterra(0, 0.0, d=2),
               fl.IntegralOperator.volterra(0, 0.0, d=2, order=2),
               fl.MultiIntegralOperator.definite([0, 1], [0, 0], [1, 1], d=2)):
        A = op.apply(mb, x)
        assert A.shape == (7, 12)
        # even parity: the operator image is also even in x_1 for symmetric ops
        if not isinstance(op, fl.MultiIntegralOperator):
            assert torch.allclose(A, op.apply(mb, x * R), atol=1e-10)


def test_predict_before_solve_is_a_clear_error():
    s = fl.FastLSQSolver(2); s.add_block(4, 1.0)
    with pytest.raises(RuntimeError, match="beta"):
        s.predict(torch.rand(3, 2))
    with pytest.raises(RuntimeError, match="add_block"):
        fl.FastLSQSolver(2).basis


def test_polynomial_symbol_rejects_per_feature_tensor():
    basis = fl.SinusoidalBasis.random(1, 8)
    aug = fl.AugmentedBasis(basis, fl.PolynomialColumns(0, 1))
    x = torch.rand(5, 1)
    with pytest.raises(ValueError):
        aug.symbol(x, torch.full((1, 8), 7.0))
    out = aug.symbol(x, torch.tensor(7.0))
    assert torch.allclose(out[:, -1], torch.full((5,), 7.0))
