# Copyright (c) 2026 Antonin Sulc -- MIT.
"""Tests for the anisotropic Sigma = L L^T learner (LearnableFastLSQ +
train_bandwidth) and the solve_linear(learn_sigma=...) high-level path."""
import math

import pytest
import torch

torch.set_default_dtype(torch.float64)

from fastlsq import solve_linear
from fastlsq.learnable import LearnableFastLSQ, train_bandwidth
from fastlsq.geometry import sample_box, sample_boundary_box

PI = math.pi


class AnisoPoisson:
    """-Delta u = f on (0,1)^2, u = sin(12 pi x) sin(pi y): fast in x, slow in y.

    Isotropic features waste capacity on high y-frequencies the solution does not
    have, so a learned anisotropic Sigma should help a lot at fixed feature count.
    """
    dim = 2
    m, n = 12, 1
    lam = 100.0

    def exact(self, x):
        return torch.sin(self.m * PI * x[:, 0:1]) * torch.sin(self.n * PI * x[:, 1:2])

    def exact_grad(self, x):
        gx = self.m * PI * torch.cos(self.m * PI * x[:, 0:1]) * torch.sin(self.n * PI * x[:, 1:2])
        gy = self.n * PI * torch.sin(self.m * PI * x[:, 0:1]) * torch.cos(self.n * PI * x[:, 1:2])
        return torch.cat([gx, gy], 1)

    def source(self, x):
        return (self.m ** 2 + self.n ** 2) * PI ** 2 * self.exact(x)

    def get_test_points(self, k):
        return sample_box(k, 2)

    def get_train_data(self, n_pde, n_bc):
        x = sample_box(n_pde, 2)
        xb = sample_boundary_box(n_bc, 2)
        return x, xb, self.source(x)

    def build(self, solver, x, bcs, f):
        B = solver.basis
        A = torch.cat([-B.laplacian(x), self.lam * B.evaluate(bcs)], 0)
        b = torch.cat([f, self.lam * self.exact(bcs)], 0)
        return A, b


def _rel(a, t):
    return (torch.norm(a - t) / torch.norm(t)).item()


@pytest.mark.parametrize("mode", ["diagonal", "cholesky"])
def test_learn_anisotropic_bandwidth(mode):
    torch.manual_seed(0)
    prob = AnisoPoisson()
    xt = prob.get_test_points(2000)
    ut = prob.exact(xt)

    lm = LearnableFastLSQ(2, n_features=500, mode=mode, init_scale=25.0, normalize=False)
    x, bc, f = prob.get_train_data(3000, 800)
    A, b = prob.build(lm, x, bc, f)
    lm.solve_inner(A, b)
    e0 = _rel(lm.predict(xt), ut)                       # isotropic baseline

    train_bandwidth(lm, prob, n_pde=3000, n_bc=800, n_steps=150, lr=0.1, verbose=False)
    e1 = _rel(lm.predict(xt), ut)

    assert torch.isfinite(torch.tensor(e1)), f"{mode}: learner produced non-finite error"
    assert e1 < e0 / 10.0, f"{mode}: {e0:.2e} -> {e1:.2e} (expected >=10x improvement)"


def test_fit_beats_isotropic_solve():
    prob = AnisoPoisson()
    xt = prob.get_test_points(2000)
    ut = prob.exact(xt)
    torch.manual_seed(0)
    iso = solve_linear(prob, scale=25.0, auto_scale=False, n_blocks=1,
                       hidden_size=500, n_pde=3000, n_bc=800, n_test=2000, verbose=False)
    torch.manual_seed(0)
    lm = LearnableFastLSQ(2, n_features=500, mode="cholesky", init_scale=25.0,
                          normalize=False).fit(prob, n_pde=3000, n_bc=800,
                                               n_steps=150, verbose=False)
    learned_err = _rel(lm.predict(xt), ut)
    assert learned_err < iso["metrics"]["val_err"]


def test_covariance_is_psd():
    torch.manual_seed(0)
    lm = LearnableFastLSQ(3, n_features=64, mode="cholesky", init_scale=2.0)
    Sigma = lm.covariance
    eig = torch.linalg.eigvalsh(Sigma)
    assert (eig > 0).all(), "Sigma = L L^T must stay positive-definite"
