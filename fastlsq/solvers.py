# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Solver classes for FastLSQ.

FastLSQSolver  -- Random Fourier Features with sin activation and exact
                  analytical first- and second-order derivatives.
PIELMSolver    -- Physics-Informed Extreme Learning Machine with tanh
                  activation (baseline comparison).
"""

import torch
import numpy as np

from fastlsq.utils import device


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
        If True, all features are divided by sqrt(N) so that the empirical
        kernel is properly scaled.  Recommended for Newton-based nonlinear
        solves where beta must stay O(1).
    """

    def __init__(self, input_dim, normalize=False):
        self.input_dim = input_dim
        self.normalize = normalize
        self.W_list: list[torch.Tensor] = []
        self.b_list: list[torch.Tensor] = []
        self.beta: torch.Tensor | None = None
        self._n_features = 0

    # ------------------------------------------------------------------
    # Feature blocks
    # ------------------------------------------------------------------

    def add_block(self, hidden_size=500, scale=1.0):
        """Append a block of random Fourier features.

        Parameters
        ----------
        hidden_size : int
            Number of features in this block.
        scale : float or list[float]
            Bandwidth parameter.  A list enables *anisotropic* scaling
            (one value per input dimension).
        """
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

    @property
    def n_features(self):
        """Total number of features across all blocks."""
        return self._n_features

    # ------------------------------------------------------------------
    # Feature evaluation
    # ------------------------------------------------------------------

    def get_features(self, x):
        """Compute features, first derivatives, and diagonal Hessian.

        Parameters
        ----------
        x : Tensor, shape (M, D)

        Returns
        -------
        H   : Tensor, shape (M, N)      -- feature values
        dH  : Tensor, shape (M, D, N)   -- d(feature)/dx_i
        ddH : Tensor, shape (M, D, N)   -- d^2(feature)/dx_i^2
        """
        Hs, dHs, ddHs = [], [], []
        for W, b in zip(self.W_list, self.b_list):
            Z = x @ W + b
            sin_Z = torch.sin(Z)
            cos_Z = torch.cos(Z)
            Hs.append(sin_Z)
            dHs.append(cos_Z.unsqueeze(1) * W.unsqueeze(0))
            ddHs.append(-sin_Z.unsqueeze(1) * (W ** 2).unsqueeze(0))

        H = torch.cat(Hs, -1)
        dH = torch.cat(dHs, -1)
        ddH = torch.cat(ddHs, -1)

        if self.normalize:
            norm = np.sqrt(self._n_features)
            H = H / norm
            dH = dH / norm
            ddH = ddH / norm

        return H, dH, ddH

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, x):
        """Evaluate u_N(x)."""
        H, _, _ = self.get_features(x)
        return H @ self.beta

    def predict_with_grad(self, x):
        """Evaluate u_N(x) and its gradient.

        Returns
        -------
        u      : Tensor, shape (M, 1)
        grad_u : Tensor, shape (M, D)
        """
        H, dH, _ = self.get_features(x)
        u = H @ self.beta
        grad_u = torch.einsum("idh,ho->id", dH, self.beta)
        return u, grad_u

    def predict_with_laplacian(self, x):
        """Evaluate u_N(x), gradient, and Laplacian.

        Returns
        -------
        u      : Tensor, shape (M, 1)
        grad_u : Tensor, shape (M, D)
        lap_u  : Tensor, shape (M, 1)
        """
        H, dH, ddH = self.get_features(x)
        u = H @ self.beta
        grad_u = torch.einsum("idh,ho->id", dH, self.beta)
        lap_u = torch.sum(ddH, dim=1) @ self.beta
        return u, grad_u, lap_u

    # Aliases used by the nonlinear Newton driver
    evaluate = predict
    evaluate_with_grad = predict_with_grad
    evaluate_with_laplacian = predict_with_laplacian


# ======================================================================
# PIELM Solver (baseline)
# ======================================================================

class PIELMSolver:
    """Physics-Informed Extreme Learning Machine with tanh activation.

    Used as a baseline comparison for the sin-based Fast-LSQ solver.
    """

    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.W_list: list[torch.Tensor] = []
        self.b_list: list[torch.Tensor] = []
        self.beta: torch.Tensor | None = None
        self._n_features = 0

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

    @property
    def n_features(self):
        return self._n_features

    def get_features(self, x):
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

    def predict(self, x):
        H, _, _ = self.get_features(x)
        return H @ self.beta

    def predict_with_grad(self, x):
        H, dH, _ = self.get_features(x)
        u = H @ self.beta
        grad_u = torch.einsum("idh,ho->id", dH, self.beta)
        return u, grad_u
