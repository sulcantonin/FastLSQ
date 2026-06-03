# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Learnable bandwidth and parametric PDE solvers for FastLSQ.

Provides:
* **LearnableFastLSQ** -- bandwidth (sigma / Cholesky L) optimised via AdamW.
* **LearnableParametricPDE** -- PDE operator coefficients (e.g. k, c) as
  nn.Parameters, pluggable into the prebuilt linear solve and AdamW.

Key ideas
---------
* **Reparameterisation trick** -- base weights W_hat ~ N(0, I) are frozen
  once; actual weights are W = L @ W_hat where L is a learnable lower-
  triangular matrix (or a scalar multiple of the identity).
* **Inner exact solve** -- for each L, the PDE matrix A(L) is assembled
  analytically via the cyclic derivative formula and beta*(L) = A(L)^+ b
  is computed in one shot.  Gradients flow back through `torch.linalg.lstsq`.
* **Learnable operator coefficients** -- Op accepts nn.Parameter in scalar
  multiplication, e.g. ``Op.laplacian(d=2) + k**2 * Op.identity(d=2)`` with
  ``k = nn.Parameter(...)``.  Build the operator inside forward() so it
  uses current parameter values; gradients flow through the lstsq to k.
* **Matrix caching** -- when the PDE operator and geometry stay fixed but
  the source / boundary data change, the pseudo-inverse A^+ can be cached
  and reused for O(MN) per new right-hand side.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from fastlsq.device import get_device
from fastlsq.basis import SinusoidalBasis
from fastlsq.block import unpack_beta


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
        n_outputs: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.mode = mode
        self._normalize = normalize

        # Frozen base weights: W_hat ~ N(0, I_d),  b ~ U(0, 2*pi)
        self.register_buffer(
            "W_hat", torch.randn(input_dim, n_features, device=get_device())
        )
        self.register_buffer(
            "b", torch.rand(1, n_features, device=get_device()) * 2 * np.pi
        )

        # Learnable bandwidth parameters
        if mode == "scalar":
            self.log_sigma = nn.Parameter(
                torch.tensor(np.log(init_scale), device=get_device())
            )
        elif mode == "diagonal":
            self.log_diag = nn.Parameter(
                torch.full((input_dim,), np.log(init_scale), device=get_device())
            )
        elif mode == "cholesky":
            # exp(diag) keeps Sigma = L L^T positive-definite and learns at the
            # same multiplicative rate as the diagonal mode; isotropic start.
            L_init = torch.zeros(input_dim, input_dim, device=get_device())
            L_init.diagonal().fill_(float(np.log(init_scale)))
            self.L_raw = nn.Parameter(L_init)
        else:
            raise ValueError(f"Unknown mode {mode!r}")

        self.beta: Optional[torch.Tensor] = None
        # Flat (Nk, 1) coefficient vector kept alongside the (N, k) shaped
        # `self.beta` so that block-structured residual losses
        # ``A @ beta_flat - b`` work without re-packing.
        self._beta_flat: Optional[torch.Tensor] = None
        self._cached_pinv: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Effective weight matrix W(L) = L @ W_hat
    # ------------------------------------------------------------------

    def _effective_L(self) -> torch.Tensor:
        """Return the d x d scaling / Cholesky matrix."""
        if self.mode == "scalar":
            sigma = self.log_sigma.exp()
            return sigma * torch.eye(self.input_dim, device=get_device())
        elif self.mode == "diagonal":
            return torch.diag(self.log_diag.clamp(-3.0, 6.5).exp())
        else:  # cholesky: free strictly-lower part + log-positive diagonal
            off = torch.tril(self.L_raw, diagonal=-1)
            diag = torch.diagonal(self.L_raw).clamp(-3.0, 6.5).exp()
            return off + torch.diag(diag)

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
        grad_u = torch.einsum("idh,hk->idk", dH, self.beta)
        if grad_u.shape[-1] == 1:
            grad_u = grad_u.squeeze(-1)
        return u, grad_u

    # ------------------------------------------------------------------
    # One-step exact solve (inner loop)
    # ------------------------------------------------------------------

    def solve_inner(self, A: torch.Tensor, b: torch.Tensor, mu: float = 0.0,
                    rcond: float = 1e-12):
        """Differentiable rank-revealing inner solve.

        Solves ``beta* = argmin ||A beta - b||^2 + mu ||beta||^2`` through the
        SVD-based ``gelsd`` least-squares driver with ``rcond`` truncation, so
        gradients still flow back to ``L`` *and* the solve is stable when ``A``
        is rank-deficient.  (The ``rcond`` cut suppresses the near-null space,
        and ``gelsd``'s backward uses the stable pseudoinverse formula rather
        than per-singular-vector derivatives -- which is what keeps the outer
        AdamW loop's gradients finite.  A plain ``torch.linalg.lstsq`` *without*
        ``rcond`` is what amplifies the null space.)

        For ``n_outputs > 1`` the system is block-stacked: the flat solution is
        kept as ``self._beta_flat`` (shape-compatible with ``A``) for residual
        losses, while ``self.beta`` is reshaped to ``(N, k)`` for prediction.
        """
        if mu and mu > 0.0:
            n = A.shape[-1]
            A_aug = torch.cat([A, (mu ** 0.5) * torch.eye(n, dtype=A.dtype, device=A.device)], dim=0)
            b_aug = torch.cat([b, torch.zeros(n, b.shape[-1], dtype=b.dtype, device=b.device)], dim=0)
            beta_flat = torch.linalg.lstsq(A_aug, b_aug, rcond=rcond, driver="gelsd").solution
        else:
            beta_flat = torch.linalg.lstsq(A, b, rcond=rcond, driver="gelsd").solution
        self._beta_flat = beta_flat
        if self.n_outputs > 1:
            self.beta = unpack_beta(beta_flat, self.n_features, self.n_outputs)
        else:
            self.beta = beta_flat
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

    # ------------------------------------------------------------------
    # High-level fit: learn the bandwidth / covariance on a problem
    # ------------------------------------------------------------------

    def fit(self, problem, *, n_pde: int = 5000, n_bc: int = 1000,
            n_steps: int = 120, lr: float = 0.1, mu: float = 0.0,
            verbose: bool = True):
        """Learn the bandwidth / covariance on ``problem`` and solve.

        Convenience wrapper around :func:`train_bandwidth`; the training
        hyper-parameters live here, not on the one-shot ``solve_linear``.
        Returns ``self`` so calls can be chained::

            u = LearnableFastLSQ(d, N, mode="cholesky").fit(problem).predict(x)
        """
        train_bandwidth(self, problem, n_pde=n_pde, n_bc=n_bc,
                        n_steps=n_steps, lr=lr, mu=mu, verbose=verbose)
        return self


# ======================================================================
# Training loop
# ======================================================================

def train_bandwidth(
    learnable: LearnableFastLSQ,
    problem,
    *,
    n_pde: int = 5000,
    n_bc: int = 1000,
    n_steps: int = 120,
    lr: float = 0.1,
    mu: float = 0.0,
    rcond: float = 1e-12,
    clip_grad: float = 10.0,
    verbose: bool = True,
) -> list[dict]:
    """Hybrid training: differentiable inner solve + outer AdamW on the bandwidth.

    At each step the PDE matrix ``A(L)`` is assembled, ``beta*(L)`` is solved by a
    **rank-revealing** (truncated-SVD) inner solve, and the outer loss
    ``||A beta* - b||^2`` is backpropagated to ``L``.  The loop is robust:
    gradients are clipped, the best iterate is retained, and a failed inner SVD
    stops training gracefully.  Defaults (``mu=0``, ``lr=0.1``) match the
    validated diagonal/cholesky configuration.

    On return, ``learnable`` holds the best-found ``L`` and a fresh ``beta``.

    Parameters
    ----------
    learnable : LearnableFastLSQ
    problem : object with ``get_train_data(n_pde, n_bc)`` and ``build(learnable, x, bcs, f)``.
    n_pde, n_bc, n_steps, lr, mu, rcond, clip_grad, verbose : see above.

    Returns
    -------
    history : list[dict]  -- per-step loss / sigma (and covariance diagonal for cholesky).
    """
    optimizer = torch.optim.AdamW(learnable.parameters(), lr=lr)
    history = []
    x_pde, bcs, f_pde = problem.get_train_data(n_pde=n_pde, n_bc=n_bc)

    best_loss = float("inf")
    best_params = None

    for step in range(n_steps):
        optimizer.zero_grad()
        A, b_rhs = problem.build(learnable, x_pde, bcs, f_pde)
        try:
            learnable.solve_inner(A, b_rhs, mu=mu, rcond=rcond)
        except torch.linalg.LinAlgError:
            if verbose:
                print(f"  Step {step:4d}: inner SVD failed -- stopping.")
            break
        # _beta_flat is block-stacked and shape-compatible with A (for n_outputs>1);
        # learnable.beta may be reshaped to (N, k), so the loss uses _beta_flat.
        loss = torch.mean((A @ learnable._beta_flat - b_rhs) ** 2)
        if not torch.isfinite(loss):
            if verbose:
                print(f"  Step {step:4d}: non-finite loss -- stopping.")
            break
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in learnable.parameters()):
            if verbose:
                print(f"  Step {step:4d}: non-finite gradient -- stopping.")
            break
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(learnable.parameters(), clip_grad)
        optimizer.step()

        l = loss.item()
        if l < best_loss:
            best_loss = l
            best_params = {n: p.detach().clone() for n, p in learnable.named_parameters()}

        info = {"step": step, "loss": l, "sigma": learnable.sigma.item()}
        if learnable.mode == "cholesky":
            info["cov_diag"] = torch.diagonal(learnable.covariance).detach().cpu().tolist()
        history.append(info)
        if verbose and step % max(1, n_steps // 20) == 0:
            print(f"  Step {step:4d}: loss={l:.4e}  sigma={info['sigma']:.4f}")

    # restore the best iterate and re-solve beta at it
    if best_params is not None:
        with torch.no_grad():
            for n, p in learnable.named_parameters():
                p.copy_(best_params[n])
            try:
                A, b_rhs = problem.build(learnable, x_pde, bcs, f_pde)
                learnable.solve_inner(A, b_rhs, mu=mu, rcond=rcond)
            except torch.linalg.LinAlgError:
                pass

    return history
