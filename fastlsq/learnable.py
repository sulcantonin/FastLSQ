# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Learnable bandwidth solver for FastLSQ.

Provides `LearnableFastLSQ`, a PyTorch nn.Module that wraps the Fast-LSQ
one-shot solver inside a differentiable outer loop.  The bandwidth parameter
(scalar sigma or full Cholesky factor L) is optimised via AdamW while the
inner linear coefficients beta are solved exactly at each step.

Key ideas
---------
* **Reparameterisation trick** -- base weights W_hat ~ N(0, I) are frozen
  once; actual weights are W = L @ W_hat where L is a learnable lower-
  triangular matrix (or a scalar multiple of the identity).
* **Inner exact solve** -- for each L, the PDE matrix A(L) is assembled
  analytically via the cyclic derivative formula and beta*(L) = A(L)^+ b
  is computed in one shot.  Gradients flow back through `torch.linalg.lstsq`.
* **Matrix caching** -- when the PDE operator and geometry stay fixed but
  the source / boundary data change, the pseudo-inverse A^+ can be cached
  and reused for O(MN) per new right-hand side.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from fastlsq.utils import device
from fastlsq.basis import SinusoidalBasis


class LearnableFastLSQ(nn.Module):
    """Differentiable random-feature PDE solver with learnable bandwidth.

    Parameters
    ----------
    input_dim : int
        Spatial / spatio-temporal dimension *d*.
    n_features : int
        Total number of random Fourier features *N*.
    mode : {"scalar", "diagonal", "cholesky"}
        * ``"scalar"``   -- single learnable sigma (isotropic).
        * ``"diagonal"`` -- per-dimension learnable scales (axis-aligned).
        * ``"cholesky"`` -- full learnable Cholesky factor L (anisotropic).
    init_scale : float
        Initial value for sigma (scalar mode) or diagonal of L.
    normalize : bool
        Apply 1/sqrt(N) normalization to features.
    """

    def __init__(
        self,
        input_dim: int,
        n_features: int = 1500,
        mode: str = "scalar",
        init_scale: float = 1.0,
        normalize: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features
        self.mode = mode
        self._normalize = normalize

        # Frozen base weights: W_hat ~ N(0, I_d),  b ~ U(0, 2*pi)
        self.register_buffer(
            "W_hat", torch.randn(input_dim, n_features, device=device)
        )
        self.register_buffer(
            "b", torch.rand(1, n_features, device=device) * 2 * np.pi
        )

        # Learnable bandwidth parameters
        if mode == "scalar":
            self.log_sigma = nn.Parameter(
                torch.tensor(np.log(init_scale), device=device)
            )
        elif mode == "diagonal":
            self.log_diag = nn.Parameter(
                torch.full((input_dim,), np.log(init_scale), device=device)
            )
        elif mode == "cholesky":
            L_init = torch.eye(input_dim, device=device) * init_scale
            self.L_raw = nn.Parameter(L_init)
        else:
            raise ValueError(f"Unknown mode {mode!r}")

        self.beta: Optional[torch.Tensor] = None
        self._cached_pinv: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Effective weight matrix W(L) = L @ W_hat
    # ------------------------------------------------------------------

    def _effective_L(self) -> torch.Tensor:
        """Return the d x d scaling / Cholesky matrix."""
        if self.mode == "scalar":
            sigma = self.log_sigma.exp()
            return sigma * torch.eye(self.input_dim, device=device)
        elif self.mode == "diagonal":
            return torch.diag(self.log_diag.exp())
        else:
            return torch.tril(self.L_raw)

    @property
    def sigma(self) -> torch.Tensor:
        """Current effective bandwidth (scalar summary)."""
        L = self._effective_L()
        return torch.sqrt(torch.diagonal(L @ L.T).mean())

    @property
    def covariance(self) -> torch.Tensor:
        """Current covariance matrix Sigma = L L^T."""
        L = self._effective_L()
        return L @ L.T

    def _W(self) -> torch.Tensor:
        """Compute W = L @ W_hat (d x N)."""
        return self._effective_L() @ self.W_hat

    # ------------------------------------------------------------------
    # SinusoidalBasis (reconstructed each forward pass)
    # ------------------------------------------------------------------

    @property
    def basis(self) -> SinusoidalBasis:
        """A ``SinusoidalBasis`` with the current reparameterised weights.

        Reconstructed on each access so that gradients flow through L.
        """
        return SinusoidalBasis(self._W(), self.b, normalize=self._normalize)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        H = self.basis.evaluate(x)
        return H @ self.beta

    def predict_with_grad(self, x: torch.Tensor):
        b = self.basis
        cache = b.cache(x)
        H = b.evaluate(x, cache=cache)
        dH = b.gradient(x, cache=cache)
        u = H @ self.beta
        grad_u = torch.einsum("idh,ho->id", dH, self.beta)
        return u, grad_u

    # ------------------------------------------------------------------
    # One-step exact solve (inner loop)
    # ------------------------------------------------------------------

    def solve_inner(self, A: torch.Tensor, b: torch.Tensor, mu: float = 0.0):
        """Solve beta* = argmin ||A beta - b||^2 + mu ||beta||^2.

        Uses `torch.linalg.lstsq` so that gradients flow back to L.
        """
        if mu > 0:
            N = A.shape[1]
            I_reg = torch.eye(N, device=A.device, dtype=A.dtype) * np.sqrt(mu)
            A_aug = torch.cat([A, I_reg], dim=0)
            b_aug = torch.cat([b, torch.zeros(N, 1, device=A.device)], dim=0)
            self.beta = torch.linalg.lstsq(A_aug, b_aug).solution
        else:
            self.beta = torch.linalg.lstsq(A, b).solution
        return self.beta

    # ------------------------------------------------------------------
    # Matrix caching for boundary-condition sweeps
    # ------------------------------------------------------------------

    def cache_operator(self, A: torch.Tensor):
        """Pre-compute and cache A^+ for fast boundary-condition sweeps.

        After calling this, `solve_cached(b_new)` costs only O(N*M).
        """
        self._cached_pinv = torch.linalg.pinv(A)

    def solve_cached(self, b: torch.Tensor) -> torch.Tensor:
        """Solve beta = A^+ b using the cached pseudo-inverse."""
        if self._cached_pinv is None:
            raise RuntimeError("Call cache_operator(A) first.")
        self.beta = self._cached_pinv @ b
        return self.beta

    def clear_cache(self):
        self._cached_pinv = None


# ======================================================================
# Training loop
# ======================================================================

def train_bandwidth(
    learnable: LearnableFastLSQ,
    problem,
    *,
    n_pde: int = 5000,
    n_bc: int = 1000,
    n_steps: int = 100,
    lr: float = 1e-2,
    mu: float = 1e-10,
    verbose: bool = True,
) -> list[dict]:
    """Hybrid training: inner exact solve + outer AdamW on bandwidth.

    At each step:
      1. Assemble PDE matrix A(L) analytically.
      2. Solve beta*(L) = A(L)^+ b  exactly.
      3. Compute loss = ||A(L) beta*(L) - b||^2.
      4. Backprop through the lstsq to update L via AdamW.

    Parameters
    ----------
    learnable : LearnableFastLSQ
    problem : PDE problem with `get_train_data`, `build` or `build_newton_step`.
    n_pde, n_bc : int
        Collocation point counts.
    n_steps : int
        Number of outer optimisation steps.
    lr : float
        Learning rate for AdamW.
    mu : float
        Tikhonov regularisation inside the inner solve.
    verbose : bool

    Returns
    -------
    history : list[dict]
        Per-step loss and sigma values.
    """
    optimizer = torch.optim.AdamW(learnable.parameters(), lr=lr)
    history = []

    x_pde, bcs, f_pde = problem.get_train_data(n_pde=n_pde, n_bc=n_bc)

    for step in range(n_steps):
        optimizer.zero_grad()

        A, b_rhs = problem.build(learnable, x_pde, bcs, f_pde)
        learnable.solve_inner(A, b_rhs, mu=mu)

        residual = A @ learnable.beta - b_rhs
        loss = torch.sum(residual ** 2)

        loss.backward()
        optimizer.step()

        info = {
            "step": step,
            "loss": loss.item(),
            "sigma": learnable.sigma.item(),
        }
        if learnable.mode == "cholesky":
            info["cov_diag"] = torch.diagonal(learnable.covariance).detach().cpu().tolist()
        history.append(info)

        if verbose and step % max(1, n_steps // 20) == 0:
            print(f"  Step {step:4d}: loss={loss.item():.4e}  "
                  f"sigma={info['sigma']:.4f}")

    return history
