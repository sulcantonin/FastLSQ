# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Tests for 0.4.1: solve diagnostics, device-correct timing, full-Sigma
constructor, LearnableFastLSQ.freeze, and checkpoint provenance."""

import torch
import numpy as np
import pytest

import fastlsq
from fastlsq import (
    solve_lstsq, solve_linear, SinusoidalBasis, LearnableFastLSQ,
    FastLSQSolver, save_checkpoint, load_checkpoint,
)
from fastlsq.problems.linear import PoissonND


# ----------------------------------------------------------------------
# Tier 1.1 -- solve_lstsq(return_info=True)
# ----------------------------------------------------------------------

def _ls_system(m=200, n=60, k=2, seed=0):
    torch.manual_seed(seed)
    A = torch.randn(m, n)
    b = torch.randn(m, k)
    return A, b


def test_return_info_default_unchanged():
    """Without return_info the return value is just x (back-compat)."""
    A, b = _ls_system()
    x = solve_lstsq(A, b)
    assert isinstance(x, torch.Tensor)
    assert x.shape == (A.shape[1], b.shape[1])


def test_return_info_tuple_and_keys():
    A, b = _ls_system()
    x, info = solve_lstsq(A, b, return_info=True)
    assert set(info) == {"t_solve", "rank_used", "residual", "cond_estimate"}
    assert info["t_solve"] >= 0.0
    assert 0 < info["rank_used"] <= A.shape[1]
    assert info["residual"] >= 0.0
    assert info["cond_estimate"] >= 1.0


def test_return_info_x_matches_plain_solve():
    """The solution must be identical whether or not info is requested."""
    A, b = _ls_system()
    x_plain = solve_lstsq(A, b)
    x_info, _ = solve_lstsq(A, b, return_info=True)
    assert torch.allclose(x_plain, x_info, atol=1e-10)


@pytest.mark.parametrize("method", ["auto", "svd", "qr", "cholesky", "rsvd"])
def test_return_info_all_methods(method):
    A, b = _ls_system()
    x, info = solve_lstsq(A, b, method=method, return_info=True)
    assert x.shape == (A.shape[1], b.shape[1])
    assert 0 < info["rank_used"] <= A.shape[1]


def test_rank_used_detects_deficiency():
    """A rank-deficient A should report rank_used < N."""
    torch.manual_seed(1)
    base = torch.randn(200, 30)
    A = torch.cat([base, base], dim=1)  # 60 cols, true rank 30
    b = torch.randn(200, 1)
    _, info = solve_lstsq(A, b, method="svd", return_info=True)
    assert info["rank_used"] == 30


# ----------------------------------------------------------------------
# Tier 1.3 -- fastlsq.benchmark.time_solve
# ----------------------------------------------------------------------

def test_time_solve_returns_float_floor():
    A, b = _ls_system()
    t = fastlsq.benchmark.time_solve(lambda: solve_lstsq(A, b), reps=3, warmup=1)
    assert isinstance(t, float)
    assert t >= 0.0


def test_time_solve_return_all_stats():
    A, b = _ls_system()
    stats = fastlsq.benchmark.time_solve(
        lambda: solve_lstsq(A, b), reps=4, warmup=1, return_all=True
    )
    assert set(stats) >= {"min", "median", "mean", "std", "times", "device"}
    assert len(stats["times"]) == 4
    assert stats["min"] <= stats["mean"] + 1e-12


def test_synchronize_is_callable():
    fastlsq.benchmark.synchronize("cpu")  # no-op, must not raise


# ----------------------------------------------------------------------
# Tier 1.2 -- phased breakdown in solve_linear metrics
# ----------------------------------------------------------------------

def test_solve_linear_phased_metrics():
    torch.set_default_dtype(torch.float32)
    problem = PoissonND()
    result = solve_linear(
        problem, scale=5.0, n_blocks=2, hidden_size=200,
        n_pde=1000, n_bc=200, verbose=False,
    )
    m = result["metrics"]
    for key in ("scale_search_s", "assemble_s", "solve_s",
                "rank_used", "residual", "cond_estimate"):
        assert key in m, f"missing metric {key}"
    # scale given explicitly -> no search time
    assert m["scale_search_s"] == 0.0
    assert m["assemble_s"] >= 0.0 and m["solve_s"] >= 0.0
    assert 0 < m["rank_used"]


# ----------------------------------------------------------------------
# Tier 2.1 -- SinusoidalBasis.random_covariance
# ----------------------------------------------------------------------

def test_random_covariance_matches_target_sigma():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    Sigma = torch.tensor([[4.0, 1.0], [1.0, 2.0]])
    basis = SinusoidalBasis.random_covariance(2, 200_000, Sigma=Sigma)
    emp = (basis.W @ basis.W.T) / basis.W.shape[1]
    assert torch.allclose(emp, Sigma, rtol=0.05, atol=0.05)


def test_random_covariance_L_equiv_sigma_shape():
    torch.set_default_dtype(torch.float64)
    L = torch.tensor([[2.0, 0.0], [0.5, 1.3]])
    basis = SinusoidalBasis.random_covariance(2, 500, L=L)
    assert basis.W.shape == (2, 500)
    assert basis.b.shape == (1, 500)


def test_random_covariance_requires_exactly_one():
    with pytest.raises(ValueError):
        SinusoidalBasis.random_covariance(2, 10)
    with pytest.raises(ValueError):
        SinusoidalBasis.random_covariance(
            2, 10, Sigma=torch.eye(2), L=torch.eye(2)
        )


# ----------------------------------------------------------------------
# Tier 2.2 -- LearnableFastLSQ.freeze
# ----------------------------------------------------------------------

def test_freeze_returns_detached_basis():
    torch.set_default_dtype(torch.float64)
    model = LearnableFastLSQ(input_dim=2, n_features=128, mode="cholesky")
    frozen = model.freeze()
    assert isinstance(frozen, SinusoidalBasis)
    assert not frozen.W.requires_grad
    # weights match the live (with-grad) basis at freeze time
    assert torch.allclose(frozen.W, model.basis.W.detach())
    assert frozen._inv_norm == model.basis._inv_norm


# ----------------------------------------------------------------------
# Tier 3 -- checkpoint provenance
# ----------------------------------------------------------------------

def test_checkpoint_records_provenance(tmp_path):
    torch.set_default_dtype(torch.float64)
    solver = FastLSQSolver(input_dim=2)
    solver.add_block(hidden_size=32, scale=3.0)
    solver.beta = torch.zeros(solver.n_features, 1)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(solver, str(path))
    _, metadata = load_checkpoint(str(path))

    assert "provenance" in metadata
    prov = metadata["provenance"]
    assert prov["fastlsq_version"] == fastlsq.__version__
    assert prov["input_dim"] == 2
    assert "device" in prov and "dtype" in prov
    assert "freq_std" in prov and prov["freq_std"].shape == (2,)


def test_checkpoint_user_metadata_preserved(tmp_path):
    torch.set_default_dtype(torch.float64)
    solver = FastLSQSolver(input_dim=1)
    solver.add_block(hidden_size=8, scale=1.0)
    solver.beta = torch.zeros(solver.n_features, 1)

    path = tmp_path / "ckpt2.pt"
    save_checkpoint(solver, str(path), metadata={"note": "hello"})
    _, metadata = load_checkpoint(str(path))
    assert metadata["note"] == "hello"
    assert "provenance" in metadata  # auto-added alongside user keys
