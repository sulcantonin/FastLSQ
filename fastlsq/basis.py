# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Sinusoidal basis and symbolic differential operators -- the foundation of FastLSQ.

The core insight: for sinusoidal features φ_j(x) = sin(W_j^T x + b_j),
any derivative of any order admits an exact closed-form expression:

    D^α φ_j(x) = (∏_k W_{jk}^{α_k}) · Φ_{|α| mod 4}(W_j^T x + b_j)

where Φ_0 = sin, Φ_1 = cos, Φ_2 = −sin, Φ_3 = −cos.

This module makes that identity the central, first-class abstraction:

* ``SinusoidalBasis`` -- evaluates arbitrary-order derivatives in O(1).
* ``BasisCache``      -- pre-computes sin(Z)/cos(Z) once, reuses across
                         multiple derivative evaluations at the same points.
* ``DiffOperator``    -- symbolic representation of linear differential
                         operators that compose via +, −, scalar *.
                         Aliased as ``Op`` for concise usage.

Example
-------
>>> from fastlsq.basis import SinusoidalBasis, Op
>>> basis = SinusoidalBasis.random(input_dim=2, n_features=1000, sigma=3.0)
>>> x = torch.rand(5000, 2)
>>>
>>> # Arbitrary mixed partial via multi-index
>>> d3_dxdy2 = basis.derivative(x, alpha=(1, 2))
>>>
>>> # Compose operators symbolically
>>> k = 10.0
>>> helmholtz = Op.laplacian(d=2) + k**2 * Op.identity(d=2)
>>> A_pde = helmholtz.apply(basis, x)          # (5000, 1000)
>>>
>>> # Learnable coefficients: plug nn.Parameter for AdamW optimisation
>>> k = torch.nn.Parameter(torch.tensor(10.0))
>>> helmholtz = Op.laplacian(d=2) + k**2 * Op.identity(d=2)
>>> A_pde = helmholtz.apply(basis, x)         # differentiable w.r.t. k
>>>
>>> # Or use fast-path methods directly
>>> A_pde_fast = basis.laplacian(x) + k**2 * basis.evaluate(x)  # same result
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Optional, Sequence, Union

from fastlsq.utils import device

# Type for operator term coefficients: scalars or learnable tensors (nn.Parameter)
CoeffT = Union[float, int, torch.Tensor]


# ======================================================================
# BasisCache: avoid redundant sin/cos computation
# ======================================================================

class BasisCache:
    """Pre-computed quantities for a fixed (basis, x) pair.

    Computes Z = x @ W + b once and lazily evaluates sin(Z) and cos(Z)
    on first access, then caches them for reuse across multiple
    derivative evaluations.
    """

    __slots__ = ("Z", "_sin", "_cos")

    def __init__(self, Z: torch.Tensor):
        self.Z = Z
        self._sin: Optional[torch.Tensor] = None
        self._cos: Optional[torch.Tensor] = None

    @property
    def sin_Z(self) -> torch.Tensor:
        if self._sin is None:
            self._sin = torch.sin(self.Z)
        return self._sin

    @property
    def cos_Z(self) -> torch.Tensor:
        if self._cos is None:
            self._cos = torch.cos(self.Z)
        return self._cos

    def phase(self, order: int) -> torch.Tensor:
        """Evaluate Φ_{order mod 4}(Z)."""
        r = order % 4
        if r == 0:
            return self.sin_Z
        elif r == 1:
            return self.cos_Z
        elif r == 2:
            return -self.sin_Z
        else:
            return -self.cos_Z


# ======================================================================
# SinusoidalBasis: the core analytical derivative engine
# ======================================================================

class SinusoidalBasis:
    """Analytical derivative engine for sinusoidal random Fourier features.

    Given frozen weights W ∈ R^{d×N} and biases b ∈ R^{1×N}, provides O(1)
    evaluation of any derivative D^α sin(W^T x + b) via the cyclic identity:

        D^α φ_j(x) = (∏_k W_{jk}^{α_k}) · Φ_{|α| mod 4}(W_j^T x + b_j)

    Parameters
    ----------
    W : Tensor, shape (d, N)
        Feature weight matrix (columns are frequency vectors).
    b : Tensor, shape (1, N)
        Bias / phase vector.
    normalize : bool
        If True, all outputs are scaled by 1/√N.
    """

    def __init__(self, W: torch.Tensor, b: torch.Tensor, normalize: bool = True):
        self.W = W
        self.b = b
        self._inv_norm = 1.0 / np.sqrt(W.shape[1]) if normalize else 1.0

    @property
    def input_dim(self) -> int:
        return self.W.shape[0]

    @property
    def n_features(self) -> int:
        return self.W.shape[1]

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def random(
        cls,
        input_dim: int,
        n_features: int,
        sigma: float = 1.0,
        normalize: bool = True,
    ) -> SinusoidalBasis:
        """Create a basis with isotropic Gaussian weights."""
        W = torch.randn(input_dim, n_features, device=device) * sigma
        b = torch.rand(1, n_features, device=device) * 2 * np.pi
        return cls(W, b, normalize=normalize)

    @classmethod
    def random_anisotropic(
        cls,
        input_dim: int,
        n_features: int,
        sigma: Union[list, np.ndarray, torch.Tensor],
        normalize: bool = True,
    ) -> SinusoidalBasis:
        """Create a basis with per-dimension bandwidths."""
        W = torch.randn(input_dim, n_features, device=device)
        s = torch.as_tensor(sigma, device=device, dtype=W.dtype).reshape(-1, 1)
        W = W * s
        b = torch.rand(1, n_features, device=device) * 2 * np.pi
        return cls(W, b, normalize=normalize)

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def cache(self, x: torch.Tensor) -> BasisCache:
        """Create a cache for the given collocation points."""
        return BasisCache(x @ self.W + self.b)

    # ------------------------------------------------------------------
    # Core: arbitrary-order derivative via cyclic identity
    # ------------------------------------------------------------------

    def derivative(
        self,
        x: torch.Tensor,
        alpha: tuple[int, ...],
        cache: Optional[BasisCache] = None,
    ) -> torch.Tensor:
        """Evaluate D^α φ_j(x) for all features j.

        Parameters
        ----------
        x : Tensor, shape (M, d)
        alpha : tuple of d non-negative integers
            Multi-index specifying the derivative order per dimension.
            E.g. (2, 0) = ∂²/∂x₀², (1, 1) = ∂²/∂x₀∂x₁.
        cache : BasisCache, optional
            Reuse pre-computed sin/cos from a previous call.

        Returns
        -------
        Tensor, shape (M, N)
        """
        if cache is None:
            cache = self.cache(x)

        order = sum(alpha)
        base = cache.phase(order)  # (M, N)

        # Monomial prefactor: ∏_k W_k^{α_k}
        prefactor = torch.ones(
            1, self.n_features, device=self.W.device, dtype=self.W.dtype
        )
        for k, a_k in enumerate(alpha):
            if a_k > 0:
                prefactor = prefactor * (self.W[k : k + 1, :] ** a_k)

        return (prefactor * base) * self._inv_norm

    # ------------------------------------------------------------------
    # Convenience: 0th order (basis values)
    # ------------------------------------------------------------------

    def evaluate(
        self, x: torch.Tensor, cache: Optional[BasisCache] = None
    ) -> torch.Tensor:
        """φ_j(x) = sin(W_j^T x + b_j), shape (M, N)."""
        if cache is None:
            cache = self.cache(x)
        return cache.sin_Z * self._inv_norm

    # ------------------------------------------------------------------
    # Convenience: first derivatives (gradient)
    # ------------------------------------------------------------------

    def gradient(
        self, x: torch.Tensor, cache: Optional[BasisCache] = None
    ) -> torch.Tensor:
        """∇φ_j(x), shape (M, d, N).

        Component k is ∂φ_j/∂x_k = W_{jk} cos(Z_j).
        """
        if cache is None:
            cache = self.cache(x)
        return (cache.cos_Z.unsqueeze(1) * self.W.unsqueeze(0)) * self._inv_norm

    # ------------------------------------------------------------------
    # Convenience: diagonal Hessian
    # ------------------------------------------------------------------

    def hessian_diag(
        self, x: torch.Tensor, cache: Optional[BasisCache] = None
    ) -> torch.Tensor:
        """∂²φ_j/∂x_k², shape (M, d, N).

        Component k is −W_{jk}² sin(Z_j).
        """
        if cache is None:
            cache = self.cache(x)
        return (
            -cache.sin_Z.unsqueeze(1) * (self.W ** 2).unsqueeze(0)
        ) * self._inv_norm

    # ------------------------------------------------------------------
    # Fast-path PDE operators (exploit eigenfunction structure)
    # ------------------------------------------------------------------

    def laplacian(
        self,
        x: torch.Tensor,
        dims: Optional[Sequence[int]] = None,
        cache: Optional[BasisCache] = None,
    ) -> torch.Tensor:
        """Δφ_j(x) = −‖W_j‖² sin(Z_j), shape (M, N).

        Parameters
        ----------
        dims : sequence of int, optional
            Restrict the Laplacian to these dimensions (e.g. spatial only).
            Default: all dimensions.
        """
        if cache is None:
            cache = self.cache(x)
        if dims is None:
            w_sq_sum = torch.sum(self.W ** 2, dim=0, keepdim=True)
        else:
            w_sq_sum = torch.sum(self.W[dims, :] ** 2, dim=0, keepdim=True)
        return (-w_sq_sum * cache.sin_Z) * self._inv_norm

    def biharmonic(
        self,
        x: torch.Tensor,
        dims: Optional[Sequence[int]] = None,
        cache: Optional[BasisCache] = None,
    ) -> torch.Tensor:
        """Δ²φ_j(x) = ‖W_j‖⁴ sin(Z_j), shape (M, N)."""
        if cache is None:
            cache = self.cache(x)
        if dims is None:
            w_sq_sum = torch.sum(self.W ** 2, dim=0, keepdim=True)
        else:
            w_sq_sum = torch.sum(self.W[dims, :] ** 2, dim=0, keepdim=True)
        return (w_sq_sum ** 2 * cache.sin_Z) * self._inv_norm

    def advection(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        cache: Optional[BasisCache] = None,
    ) -> torch.Tensor:
        """(v · ∇)φ_j(x) = (v · W_j) cos(Z_j), shape (M, N).

        v can be a constant vector (d,) or spatially varying (M, d).
        """
        if cache is None:
            cache = self.cache(x)
        if v.dim() == 1:
            vdotW = (v.unsqueeze(1) * self.W).sum(dim=0, keepdim=True)
        else:
            vdotW = torch.einsum("md,dn->mn", v, self.W)
        return (vdotW * cache.cos_Z) * self._inv_norm

    # ------------------------------------------------------------------
    # General operator evaluation from symbolic specification
    # ------------------------------------------------------------------

    def operator(
        self,
        x: torch.Tensor,
        terms: list[tuple[CoeffT, Union[tuple, dict]]],
        cache: Optional[BasisCache] = None,
    ) -> torch.Tensor:
        """Evaluate a linear differential operator L = Σ_i c_i D^{α_i}.

        Parameters
        ----------
        x : Tensor (M, d)
        terms : list of (coeff, alpha)
            coeff: scalar coefficient (float) or learnable tensor (e.g. nn.Parameter).
                   Tensors enable gradients to flow through operator coefficients.
            alpha: multi-index as a tuple of length d, or a dict
                   {dim: order} (unspecified dims default to 0).
        cache : BasisCache, optional

        Returns
        -------
        Tensor (M, N)

        Example
        -------
        Helmholtz in 2D: Δu + k²u

        >>> basis.operator(x, [
        ...     (1.0, (2, 0)),   # u_xx
        ...     (1.0, (0, 2)),   # u_yy
        ...     (k**2, (0, 0)),  # k² u (k can be nn.Parameter)
        ... ])
        """
        if cache is None:
            cache = self.cache(x)

        d = self.input_dim
        result = torch.zeros(
            x.shape[0], self.n_features, device=self.W.device, dtype=self.W.dtype
        )
        for coeff, alpha in terms:
            if isinstance(alpha, dict):
                alpha_list = [0] * d
                for dim, order in alpha.items():
                    alpha_list[dim] = order
                alpha = tuple(alpha_list)
            result = result + coeff * self.derivative(x, alpha, cache=cache)
        return result



# ======================================================================
# DiffOperator: composable symbolic differential operators
# ======================================================================

class DiffOperator:
    """Symbolic linear differential operator: L = Σ_i c_i D^{α_i}.

    Composable via arithmetic: ``Op.laplacian(2) + k**2 * Op.identity(2)``
    produces the Helmholtz operator.

    Evaluate on a ``SinusoidalBasis`` with ``.apply(basis, x)``.
    """

    def __init__(
        self,
        terms: Optional[list[tuple[CoeffT, tuple[int, ...]]]] = None,
    ):
        self.terms: list[tuple[CoeffT, tuple[int, ...]]] = terms or []

    # ------------------------------------------------------------------
    # Arithmetic composition
    # ------------------------------------------------------------------

    def __add__(self, other: DiffOperator) -> DiffOperator:
        if not isinstance(other, DiffOperator):
            return NotImplemented
        return DiffOperator(self.terms + other.terms)

    def __sub__(self, other: DiffOperator) -> DiffOperator:
        return self + (-other)

    def __neg__(self) -> DiffOperator:
        return DiffOperator([(-c, a) for c, a in self.terms])

    def __mul__(self, scalar: CoeffT) -> DiffOperator:
        """Scale operator by a scalar or learnable tensor (e.g. nn.Parameter)."""
        return DiffOperator([(c * scalar, a) for c, a in self.terms])

    def __rmul__(self, scalar: CoeffT) -> DiffOperator:
        """Scale operator by a scalar or learnable tensor (e.g. nn.Parameter)."""
        return self.__mul__(scalar)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def apply(
        self,
        basis: SinusoidalBasis,
        x: torch.Tensor,
        cache: Optional[BasisCache] = None,
    ) -> torch.Tensor:
        """Evaluate this operator on a sinusoidal basis at points x.

        Returns
        -------
        Tensor, shape (M, N)
        """
        return basis.operator(x, self.terms, cache=cache)

    # ------------------------------------------------------------------
    # Factory methods for common operators
    # ------------------------------------------------------------------

    @classmethod
    def partial(cls, dim: int, order: int, d: int) -> DiffOperator:
        """∂^order / ∂x_{dim}^order."""
        alpha = tuple(order if k == dim else 0 for k in range(d))
        return cls([(1.0, alpha)])

    @classmethod
    def identity(cls, d: int) -> DiffOperator:
        """Identity operator (zeroth-order derivative)."""
        return cls([(1.0, tuple([0] * d))])

    @classmethod
    def laplacian(cls, d: int, dims: Optional[Sequence[int]] = None) -> DiffOperator:
        """Laplacian Δ = Σ_k ∂²/∂x_k².

        Parameters
        ----------
        d : int
            Total number of dimensions in the multi-index.
        dims : sequence of int, optional
            Restrict to these dimensions (e.g. spatial only in a space-time
            problem).  Default: all d dimensions.
        """
        if dims is None:
            dims = range(d)
        terms = []
        for k in dims:
            alpha = tuple(2 if i == k else 0 for i in range(d))
            terms.append((1.0, alpha))
        return cls(terms)

    @classmethod
    def biharmonic(cls, d: int, dims: Optional[Sequence[int]] = None) -> DiffOperator:
        """Biharmonic Δ² = Σ_{i,j} ∂⁴/∂x_i²∂x_j²."""
        if dims is None:
            dims = range(d)
        terms = []
        for i in dims:
            for j in dims:
                alpha = [0] * d
                alpha[i] += 2
                alpha[j] += 2
                terms.append((1.0, tuple(alpha)))
        return cls(terms)

    @classmethod
    def gradient_component(cls, dim: int, d: int) -> DiffOperator:
        """∂/∂x_{dim}."""
        return cls.partial(dim, 1, d)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if not self.terms:
            return "DiffOperator(0)"
        parts = []
        for c, a in self.terms:
            if isinstance(c, torch.Tensor):
                c_str = f"{c.item():g}" if c.numel() == 1 else "tensor(...)"
            else:
                c_str = f"{c:g}"
            if all(ak == 0 for ak in a):
                parts.append(f"{c_str}*I")
            else:
                idx = ",".join(str(ak) for ak in a)
                parts.append(f"{c_str}*D({idx})")
        return "DiffOperator(" + " + ".join(parts) + ")"


# Shorthand alias
Op = DiffOperator


# ======================================================================
# FeatureBasis: adapter for non-sinusoidal solvers (e.g. PIELM / tanh)
# ======================================================================

class FeatureCache:
    """Cache storing pre-computed (H, dH, ddH) for a generic feature basis."""

    __slots__ = ("H", "dH", "ddH")

    def __init__(self, H: torch.Tensor, dH: torch.Tensor, ddH: torch.Tensor):
        self.H = H
        self.dH = dH
        self.ddH = ddH


class FeatureBasis:
    """Adapter that wraps a ``get_features()`` callable into the basis API.

    This lets non-sinusoidal solvers (e.g. PIELMSolver with tanh) expose
    the same ``evaluate / gradient / hessian_diag / laplacian`` interface
    so that problem classes work identically regardless of activation.

    Parameters
    ----------
    get_features_fn : callable(x) -> (H, dH, ddH)
    n_features : int
    input_dim : int
    """

    def __init__(self, get_features_fn, n_features: int, input_dim: int):
        self._get_features = get_features_fn
        self._n_features = n_features
        self._input_dim = input_dim

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def input_dim(self) -> int:
        return self._input_dim

    def cache(self, x: torch.Tensor) -> FeatureCache:
        H, dH, ddH = self._get_features(x)
        return FeatureCache(H, dH, ddH)

    def evaluate(
        self, x: torch.Tensor, cache: Optional[FeatureCache] = None
    ) -> torch.Tensor:
        if cache is not None:
            return cache.H
        H, _, _ = self._get_features(x)
        return H

    def gradient(
        self, x: torch.Tensor, cache: Optional[FeatureCache] = None
    ) -> torch.Tensor:
        if cache is not None:
            return cache.dH
        _, dH, _ = self._get_features(x)
        return dH

    def hessian_diag(
        self, x: torch.Tensor, cache: Optional[FeatureCache] = None
    ) -> torch.Tensor:
        if cache is not None:
            return cache.ddH
        _, _, ddH = self._get_features(x)
        return ddH

    def laplacian(
        self,
        x: torch.Tensor,
        dims: Optional[Sequence[int]] = None,
        cache: Optional[FeatureCache] = None,
    ) -> torch.Tensor:
        ddH = self.hessian_diag(x, cache)
        if dims is None:
            return torch.sum(ddH, dim=1)
        return torch.sum(ddH[:, list(dims), :], dim=1)
