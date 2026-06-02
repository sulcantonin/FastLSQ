# Copyright (c) 2026 Antonin Sulc -- MIT.
"""Vector-valued u: R^d -> R^k via the block-stacked LSQ solve
(fastlsq.block + the rank-revealing solve_lstsq + unpack_beta).

Guards the merge of the vector-output work with the rank-revealing solver: the
block-stacked system must solve, both components recovered, and method="auto"
must agree with the explicit truncated-SVD.
"""
import math

import torch

torch.set_default_dtype(torch.float64)

from fastlsq import SinusoidalBasis, solve_lstsq
from fastlsq.block import block_concat, pack_beta, unpack_beta


def test_pack_unpack_roundtrip():
    torch.manual_seed(0)
    beta = torch.randn(50, 3)
    flat = pack_beta(beta)
    assert flat.shape == (150, 1)
    assert torch.allclose(unpack_beta(flat, 50, 3), beta)


def test_block_concat_infers_zero_blocks():
    A = torch.randn(4, 5)
    B = torch.randn(4, 5)
    M = block_concat([[A, None], [None, B]])
    assert M.shape == (8, 10)
    assert torch.allclose(M[:4, :5], A)
    assert torch.allclose(M[4:, 5:], B)
    assert torch.count_nonzero(M[:4, 5:]) == 0
    assert torch.count_nonzero(M[4:, :5]) == 0


def test_vector_u_block_lstsq_recovers_components():
    """A k=2 block-stacked least-squares solve recovers both components."""
    torch.manual_seed(0)
    N, M, k = 200, 400, 2
    x = torch.rand(M, 1)
    basis = SinusoidalBasis(torch.randn(1, N) * 5,
                            torch.rand(1, N) * 2 * math.pi, normalize=False)
    H = basis.evaluate(x)
    u0 = torch.sin(2 * math.pi * x)
    u1 = torch.cos(3 * math.pi * x)
    A = block_concat([[H, None], [None, H]])     # (2M, 2N) block-diagonal
    b = torch.cat([u0, u1], dim=0)               # (2M, 1) component-major
    beta = unpack_beta(solve_lstsq(A, b), N, k)  # rank-revealing -> (N, k)
    assert beta.shape == (N, k)
    pred = basis.evaluate(x) @ beta              # (M, k)
    assert (torch.norm(pred[:, 0:1] - u0) / torch.norm(u0)).item() < 1e-8
    assert (torch.norm(pred[:, 1:2] - u1) / torch.norm(u1)).item() < 1e-8


def test_block_solve_methods_agree():
    """auto and explicit truncated-SVD agree on the block-stacked system."""
    torch.manual_seed(1)
    N, M = 120, 300
    x = torch.rand(M, 1)
    basis = SinusoidalBasis(torch.randn(1, N) * 4,
                            torch.rand(1, N) * 2 * math.pi, normalize=False)
    H = basis.evaluate(x)
    A = block_concat([[H, None], [None, H]])
    b = torch.randn(2 * M, 1)
    assert torch.allclose(solve_lstsq(A, b, method="auto"),
                          solve_lstsq(A, b, method="svd"), atol=1e-8)
