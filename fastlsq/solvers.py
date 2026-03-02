# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Solver classes for FastLSQ.

FastLSQSolver  -- Random Fourier Features with sin activation.  Exposes
                  a ``SinusoidalBasis`` via the ``.basis`` property.
PIELMSolver    -- Physics-Informed Extreme Learning Machine with tanh
                  activation (baseline).  Exposes a ``FeatureBasis``
                  adapter via the same ``.basis`` property.
"""

import torch
import numpy as np

from fastlsq.utils import device
from fastlsq.basis import SinusoidalBasis, FeatureBasis


# ======================================================================
# FastLSQ Solver
# ======================================================================

class FastLSQSolver:
    """Random Fourier Features with sin activation.

    Parameters
    ----------
    input_dim : int
        Spatial / spatio-temporal dimensionality of the input.
    normalize : bool, optional
        If True, all features are divided by sqrt(N).
    """

    def __init__(self, input_dim, normalize=False):
        self.input_dim = input_dim
        self.normalize = normalize
        self.W_list: list[torch.Tensor] = []
        self.b_list: list[torch.Tensor] = []
        self.beta: torch.Tensor | None = None
        self._n_features = 0
        self._basis: SinusoidalBasis | None = None

    def add_block(self, hidden_size=500, scale=1.0):
        """Append a block of random Fourier features."""
        W = torch.randn(self.input_dim, hidden_size, device=device)

        if isinstance(scale, (list, np.ndarray)):
            s = torch.tensor(scale, device=device, dtype=W.dtype).unsqueeze(1)
            W = W * s
        else:
            W = W * scale

        b = torch.rand(1, hidden_size, device=device) * 2 * np.pi
        self.W_list.append(W)
        self.b_list.append(b)
        self._n_features += hidden_size
        self._basis = None

    @property
    def n_features(self):
        return self._n_features

    @property
    def basis(self) -> SinusoidalBasis:
        """The underlying analytical derivative engine."""
        if self._basis is None:
            W = torch.cat(self.W_list, dim=1)
            b = torch.cat(self.b_list, dim=1)
            self._basis = SinusoidalBasis(W, b, normalize=self.normalize)
        return self._basis

    def predict(self, x):
        """Evaluate u_N(x)."""
        return self.basis.evaluate(x) @ self.beta

    def predict_with_grad(self, x):
        """Evaluate u_N(x) and its gradient."""
        cache = self.basis.cache(x)
        u = self.basis.evaluate(x, cache=cache) @ self.beta
        grad_u = torch.einsum("idh,ho->id", self.basis.gradient(x, cache=cache), self.beta)
        return u, grad_u

    def predict_with_laplacian(self, x):
        """Evaluate u_N(x), gradient, and Laplacian."""
        cache = self.basis.cache(x)
        u = self.basis.evaluate(x, cache=cache) @ self.beta
        grad_u = torch.einsum("idh,ho->id", self.basis.gradient(x, cache=cache), self.beta)
        lap_u = self.basis.laplacian(x, cache=cache) @ self.beta
        return u, grad_u, lap_u

    evaluate = predict
    evaluate_with_grad = predict_with_grad
    evaluate_with_laplacian = predict_with_laplacian


# ======================================================================
# PIELM Solver (baseline)
# ======================================================================

class PIELMSolver:
    """Physics-Informed Extreme Learning Machine with tanh activation."""

    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.W_list: list[torch.Tensor] = []
        self.b_list: list[torch.Tensor] = []
        self.beta: torch.Tensor | None = None
        self._n_features = 0
        self._basis: FeatureBasis | None = None

    def add_block(self, hidden_size=500, scale=1.0):
        W_base = torch.rand(self.input_dim, hidden_size, device=device) * 2 - 1

        if isinstance(scale, (list, np.ndarray)):
            s = torch.tensor(scale, device=device, dtype=W_base.dtype).unsqueeze(1)
            W = W_base * s
            b_scale = max(scale)
        else:
            W = W_base * scale
            b_scale = scale

        b = (torch.rand(1, hidden_size, device=device) * 2 - 1) * b_scale
        self.W_list.append(W)
        self.b_list.append(b)
        self._n_features += hidden_size
        self._basis = None

    @property
    def n_features(self):
        return self._n_features

    def _get_features(self, x):
        Hs, dHs, ddHs = [], [], []
        for W, b in zip(self.W_list, self.b_list):
            Z = x @ W + b
            H = torch.tanh(Z)
            sech2 = 1 - H ** 2
            dH = sech2.unsqueeze(1) * W.unsqueeze(0)
            ddH = (-2 * H * sech2).unsqueeze(1) * (W ** 2).unsqueeze(0)
            Hs.append(H)
            dHs.append(dH)
            ddHs.append(ddH)

        return torch.cat(Hs, -1), torch.cat(dHs, -1), torch.cat(ddHs, -1)

    @property
    def basis(self) -> FeatureBasis:
        """FeatureBasis adapter wrapping the tanh features."""
        if self._basis is None:
            self._basis = FeatureBasis(
                self._get_features, self._n_features, self.input_dim
            )
        return self._basis

    def predict(self, x):
        return self.basis.evaluate(x) @ self.beta

    def predict_with_grad(self, x):
        cache = self.basis.cache(x)
        u = self.basis.evaluate(x, cache=cache) @ self.beta
        grad_u = torch.einsum("idh,ho->id", self.basis.gradient(x, cache=cache), self.beta)
        return u, grad_u
