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

The very same identity runs *backwards*: integration is differentiation of
*negative* order.  Since ∫ sin(w x + b) dx = −(1/w) cos(w x + b), an
antiderivative is just a term with a negative multi-index entry and a
reciprocal prefactor 1/w.  So the calculus is closed in *both* directions:

* ``DiffOperator.antiderivative`` -- indefinite ∫ as a negative-order partial.
* ``IntegralOperator``            -- definite / running (Volterra) integrals
                                     with limits, evaluated in closed form.
* ``IntegroDifferentialOperator`` -- the common roof under which differential
                                     and integral terms compose into one
                                     linear-in-coefficients design matrix.

That turns "exact closed-form *derivatives* → linear least squares" into
"exact closed-form *calculus*": integro-differential and integral equations
are assembled and solved in exactly the same one-shot LSQ.

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

from fastlsq.device import get_device

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
    dc_eps : float
        DC guard for *integration* (negative-order derivatives).  A feature whose
        frequency along an integrated axis satisfies ``|W_{jk}| <= dc_eps`` has no
        sinusoidal antiderivative (its primitive is a ramp that leaves the basis),
        so its column is zeroed instead of dividing by ~0.  Only affects negative
        orders; ordinary derivatives are untouched.
    """

    def __init__(
        self,
        W: torch.Tensor,
        b: torch.Tensor,
        normalize: bool = True,
        dc_eps: float = 1e-8,
    ):
        self.W = W
        self.b = b
        self._inv_norm = 1.0 / np.sqrt(W.shape[1]) if normalize else 1.0
        self._dc_eps = dc_eps

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
        dc_eps: float = 1e-8,
    ) -> SinusoidalBasis:
        """Create a basis with isotropic Gaussian weights."""
        W = torch.randn(input_dim, n_features, device=get_device()) * sigma
        b = torch.rand(1, n_features, device=get_device()) * 2 * np.pi
        return cls(W, b, normalize=normalize, dc_eps=dc_eps)

    @classmethod
    def random_anisotropic(
        cls,
        input_dim: int,
        n_features: int,
        sigma: Union[list, np.ndarray, torch.Tensor],
        normalize: bool = True,
        dc_eps: float = 1e-8,
    ) -> SinusoidalBasis:
        """Create a basis with per-dimension bandwidths."""
        W = torch.randn(input_dim, n_features, device=get_device())
        s = torch.as_tensor(sigma, device=get_device(), dtype=W.dtype).reshape(-1, 1)
        W = W * s
        b = torch.rand(1, n_features, device=get_device()) * 2 * np.pi
        return cls(W, b, normalize=normalize, dc_eps=dc_eps)

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def cache(self, x: torch.Tensor) -> BasisCache:
        """Create a cache for the given collocation points."""
        # Accept inputs in any dtype/device (e.g. float32 from user code) and
        # promote to the basis's own dtype/device so ``x @ self.W`` never trips
        # a float32-vs-float64 mismatch.
        if x.dtype != self.W.dtype or x.device != self.W.device:
            x = x.to(dtype=self.W.dtype, device=self.W.device)
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
        alpha : tuple of d integers
            Multi-index specifying the derivative order per dimension.
            E.g. (2, 0) = ∂²/∂x₀², (1, 1) = ∂²/∂x₀∂x₁.

            *Negative* entries denote (indefinite) **integration**: the same cyclic
            identity run backwards, with a reciprocal prefactor.  E.g. (-1, 0) is the
            antiderivative ∫·dx₀ = −cos(Z)/W₀, and mixed signs like (-1, 1) give
            ∂/∂x₁ ∫·dx₀ = (W₁/W₀) sin(Z).  See :meth:`DiffOperator.antiderivative`.
            The arbitrary constant of integration leaves the sinusoidal family and is
            *not* represented here; features with ``|W_{jk}| <= dc_eps`` along an
            integrated axis are zeroed (see ``dc_eps``).
        cache : BasisCache, optional
            Reuse pre-computed sin/cos from a previous call.

        Returns
        -------
        Tensor, shape (M, N)
        """
        if cache is None:
            cache = self.cache(x)

        order = sum(alpha)
        base = cache.phase(order)  # (M, N); phase() handles order % 4 for negative orders

        # Monomial prefactor: ∏_k W_k^{α_k}.  Positive α_k multiply by W_k (derivative);
        # negative α_k divide (integration), DC-guarded against ~0 frequencies.
        prefactor = torch.ones(
            1, self.n_features, device=self.W.device, dtype=self.W.dtype
        )
        for k, a_k in enumerate(alpha):
            if a_k == 0:
                continue
            Wk = self.W[k : k + 1, :]
            if a_k > 0:
                prefactor = prefactor * (Wk ** a_k)
            else:
                safe = Wk.abs() > self._dc_eps
                Wk_safe = torch.where(safe, Wk, torch.ones_like(Wk))
                prefactor = prefactor * torch.where(
                    safe, Wk_safe ** a_k, torch.zeros_like(Wk)
                )

        return (prefactor * base) * self._inv_norm

    # ------------------------------------------------------------------
    # Definite / running (Volterra) integral along one axis
    # ------------------------------------------------------------------

    def definite_integral(
        self,
        x: torch.Tensor,
        dim: int,
        lower: float,
        upper: Optional[float] = None,
        cache: Optional[BasisCache] = None,
    ) -> torch.Tensor:
        """∫ φ_j d x_{dim} along one axis, in closed form, shape (M, N).

        ``upper=None`` gives the **Volterra** (running) integral with variable upper
        limit ``x[:, dim]``; a numeric ``upper`` gives the **definite** integral over
        ``[lower, upper]`` along ``dim`` (the other coordinates are held at ``x``).

        Unlike a standalone antiderivative, the ``1/W`` prefactor *cancels* in the
        difference F(hi) − F(lo), so this is evaluated via the numerically stable
        identity (exact, and finite even for near-DC features ``W_{dim,j} → 0``):

            ∫_lo^hi sin(Z) d x_dim = Δ · sin((Z_hi+Z_lo)/2) · sinc(W_dim·Δ / 2π),

        with Δ = hi − lo and ``torch.sinc(t) = sin(πt)/(πt)``.
        """
        if x.dtype != self.W.dtype or x.device != self.W.device:
            x = x.to(dtype=self.W.dtype, device=self.W.device)

        Wd = self.W[dim : dim + 1, :]  # (1, N)

        x_lo = x.clone()
        x_lo[:, dim] = lower
        Z_lo = self.cache(x_lo).Z  # (M, N)

        if upper is None:  # Volterra: upper limit is x_dim itself
            Z_hi = cache.Z if cache is not None else self.cache(x).Z
            t_hi = x[:, dim : dim + 1]  # (M, 1)
        else:
            x_hi = x.clone()
            x_hi[:, dim] = upper
            Z_hi = self.cache(x_hi).Z
            t_hi = torch.full(
                (x.shape[0], 1), float(upper), device=self.W.device, dtype=self.W.dtype
            )

        delta = t_hi - float(lower)  # (M, 1)
        half_sum = 0.5 * (Z_hi + Z_lo)
        sinc_arg = (Wd * delta) / (2.0 * np.pi)  # (M, N)
        out = delta * torch.sin(half_sum) * torch.sinc(sinc_arg)
        return out * self._inv_norm

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
            D = self.derivative(x, alpha, cache=cache)
            if callable(coeff) and not isinstance(coeff, torch.Tensor):
                # Variable-coefficient term c(x) * D^alpha phi.
                c_vals = coeff(x)
                if c_vals.dim() == 1:
                    c_vals = c_vals.unsqueeze(-1)
                result = result + c_vals * D
            else:
                result = result + coeff * D
        return result



# ======================================================================
# GaussianWindowedBasis: windowed-Fourier (Gabor) basis for projection ops
# ======================================================================

class GaussianWindowedBasis:
    """Gaussian-windowed sinusoidal (Gabor) features for closed-form projection.

    Each feature is a plane wave multiplied by a fixed Gaussian window in a
    *whitened* frame::

        ψ_j(z) = exp(−‖ζ‖²/2) · sin(W_j·ζ + b_j),      ζ = T⁻¹ (z − mean)

    where ``mean`` (d,) is the window centre and ``T`` (d, d) is the lower-Cholesky
    factor of the prior covariance (so ``ζ`` is the whitened coordinate).

    The window is a **fixed prior** -- set once from the data's second moments,
    *not* trained.  It is what makes the line / hyperplane integral of every
    feature *converge* and admit a closed form (the projection of a bare unbounded
    sinusoid over an infinite hyperplane diverges; the Gaussian makes it integrable
    and analytic -- see :class:`ProjectionOperator`).  The coefficients on the
    features stay linear, so fitting is still **one linear least squares**.

    This is the windowed-Fourier (Gabor) member of the basis family, distinct from
    the bare :class:`SinusoidalBasis`.  The Gaussian envelope changes the derivative
    algebra (envelope×sine derivatives are still closed form, but no longer the bare
    cyclic identity ``D^α sin = (∏W^{α_k}) Φ_{|α| mod 4}``), so this class is
    deliberately scoped to **value** (:meth:`evaluate`) and **projection** (via
    :class:`ProjectionOperator`); it does *not* expose the full
    :class:`DiffOperator` calculus.

    Convention matches :class:`SinusoidalBasis`: ``W`` has shape ``(d, N)`` (one
    column per feature) and values are ``sin(ζ @ W + b)`` with ``b`` of shape
    ``(1, N)``.

    Parameters
    ----------
    W : Tensor, shape (d, N)
        Whitened-frame frequency vectors (columns are features).
    b : Tensor, shape (1, N)
        Phase / bias vector.
    mean : Tensor, shape (d,)
        Window centre in data coordinates.
    T : Tensor, shape (d, d)
        Lower-Cholesky factor of the prior covariance; whitening is ζ = T⁻¹(z−mean).
    """

    def __init__(
        self,
        W: torch.Tensor,
        b: torch.Tensor,
        mean: torch.Tensor,
        T: torch.Tensor,
    ):
        self.W = W
        self.b = b
        self.mean = mean
        self.T = T
        self.Tinv = torch.linalg.inv(T)

    @property
    def d(self) -> int:
        return self.W.shape[0]

    @property
    def input_dim(self) -> int:
        return self.W.shape[0]

    @property
    def n(self) -> int:
        return self.W.shape[1]

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
        mean: Optional[torch.Tensor] = None,
        T: Optional[torch.Tensor] = None,
    ) -> GaussianWindowedBasis:
        """Random whitened-frame frequencies; identity window unless ``mean``/``T`` given."""
        dev = get_device()
        W = torch.randn(input_dim, n_features, device=dev) * sigma
        b = torch.rand(1, n_features, device=dev) * 2 * np.pi
        if mean is None:
            mean = torch.zeros(input_dim, device=dev, dtype=W.dtype)
        else:
            mean = torch.as_tensor(mean, device=dev, dtype=W.dtype)
        if T is None:
            T = torch.eye(input_dim, device=dev, dtype=W.dtype)
        else:
            T = torch.as_tensor(T, device=dev, dtype=W.dtype)
        return cls(W, b, mean, T)

    @classmethod
    def from_data(
        cls,
        z: torch.Tensor,
        n_features: int,
        sigma: float = 1.0,
        eps: float = 1e-9,
    ) -> GaussianWindowedBasis:
        """Set the fixed Gaussian window from data second moments.

        ``mean`` and ``T`` (lower-Cholesky of the empirical covariance, ridged by
        ``eps`` for safety) are estimated from the support points ``z`` of shape
        ``(M, d)``; frequencies are drawn isotropically in the whitened frame.  This
        realises the "window = fixed prior set from the data's moments" design and
        leaves the feature coefficients linear.
        """
        z = torch.as_tensor(z)
        d = z.shape[1]
        mean = z.mean(0)
        zc = z - mean
        cov = (zc.t() @ zc) / max(z.shape[0] - 1, 1)
        cov = cov + eps * torch.eye(d, device=z.device, dtype=z.dtype)
        T = torch.linalg.cholesky(cov)
        W = torch.randn(d, n_features, device=z.device, dtype=z.dtype) * sigma
        b = torch.rand(1, n_features, device=z.device, dtype=z.dtype) * 2 * np.pi
        return cls(W, b, mean, T)

    # ------------------------------------------------------------------
    # Value
    # ------------------------------------------------------------------

    def evaluate(
        self, z: torch.Tensor, cache: Optional[object] = None
    ) -> torch.Tensor:
        """ψ_j(z) = exp(−‖ζ‖²/2)·sin(ζ @ W + b), shape (M, N), ζ = T⁻¹(z−mean)."""
        if z.dtype != self.W.dtype or z.device != self.W.device:
            z = z.to(dtype=self.W.dtype, device=self.W.device)
        zeta = (z - self.mean) @ self.Tinv.t()
        window = torch.exp(-0.5 * (zeta ** 2).sum(1, keepdim=True))
        return window * torch.sin(zeta @ self.W + self.b)

    def value(self, z: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`evaluate` (matches the downstream prototype name)."""
        return self.evaluate(z)



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

    def __add__(self, other):
        if isinstance(other, DiffOperator):
            return DiffOperator(self.terms + other.terms)
        if isinstance(other, (IntegralOperator, IntegroDifferentialOperator)):
            # Mixing differential and integral terms -> integro-differential operator.
            return IntegroDifferentialOperator([(1.0, self)]) + other
        return NotImplemented

    def __sub__(self, other: DiffOperator) -> DiffOperator:
        return self + (-other)

    def __neg__(self) -> DiffOperator:
        new_terms = []
        for c, a in self.terms:
            if callable(c) and not isinstance(c, torch.Tensor):
                fn = c
                new_terms.append((lambda x, fn=fn: -fn(x), a))
            else:
                new_terms.append((-c, a))
        return DiffOperator(new_terms)

    def __mul__(self, scalar: CoeffT) -> DiffOperator:
        """Scale operator by a scalar or learnable tensor (e.g. nn.Parameter)."""
        new_terms = []
        for c, a in self.terms:
            if callable(c) and not isinstance(c, torch.Tensor):
                fn = c
                new_terms.append((lambda x, fn=fn, s=scalar: s * fn(x), a))
            else:
                new_terms.append((c * scalar, a))
        return DiffOperator(new_terms)

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
    def antiderivative(cls, dim: int, order: int, d: int) -> DiffOperator:
        """Indefinite ∫…∫ along axis ``dim`` (``order`` times) — the negative-order partial.

        Built on the same cyclic identity as :meth:`partial`, run backwards:
        ``∫ sin(w x + b) dx = −(1/w) cos(w x + b)`` is a derivative term with a *negative*
        multi-index entry and a reciprocal prefactor 1/w.  Composes with derivative terms
        (``+``, ``−``, scalar ``*``) into integro-differential operators, e.g.
        ``Op.partial(0, 1, 1) - k * Op.antiderivative(0, 1, 1)``.

        The arbitrary constant(s) of integration leave the sinusoidal family and are *not*
        represented here; pin them with explicit low-order polynomial columns or a boundary
        row.  For *definite* / running (Volterra) integrals with limits, see
        :class:`IntegralOperator`.
        """
        if order < 1:
            raise ValueError("antiderivative order must be a positive integer")
        alpha = tuple(-order if k == dim else 0 for k in range(d))
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

    @classmethod
    def field(cls, coeff_fn, alpha) -> DiffOperator:
        """Variable-coefficient term c(x) · D^alpha.

        Parameters
        ----------
        coeff_fn : callable
            Maps ``x`` of shape ``(M, d)`` to a tensor of shape ``(M, 1)`` or ``(M,)``.
        alpha : tuple of int
            Multi-index of the derivative.
        """
        return cls([(coeff_fn, tuple(alpha))])

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
            elif callable(c):
                c_str = "c(x)"
            else:
                c_str = f"{c:g}"
            if all(ak == 0 for ak in a):
                parts.append(f"{c_str}*I")
            else:
                idx = ",".join(str(ak) for ak in a)
                parts.append(f"{c_str}*D({idx})")
        return "DiffOperator(" + " + ".join(parts) + ")"


# ======================================================================
# IntegralOperator: definite / running (Volterra) integrals with limits
# ======================================================================

class IntegralOperator:
    """Definite or running (Volterra) integral along one axis, in closed form.

    Unlike :meth:`DiffOperator.antiderivative` (indefinite, pointwise), these carry
    integration *limits*, so they are a separate, limit-bearing operator:

    >>> V = IntegralOperator.volterra(dim=0, lower=0.0, d=1)         # (Vu)(x)=∫_0^x u dt
    >>> Iab = IntegralOperator.definite(dim=0, lower=0.0, upper=1.0, d=1)  # ∫_0^1 u dt

    ``apply(basis, x)`` returns an (M, N) design matrix: for Volterra, the running
    integral up to each ``x``; for a partial definite integral in d>1, the matrix
    marginalised along ``dim`` (other coordinates held at ``x``).  Composes with
    :class:`DiffOperator` via ``+ - *`` into an :class:`IntegroDifferentialOperator`,
    so an integro-differential / integral equation is one linear-least-squares block:

    >>> from fastlsq import Op
    >>> L = Op.partial(0, 1, d=1) + IntegralOperator.volterra(dim=0, lower=0.0, d=1)
    >>> A = L.apply(basis, x)        # u'(x) + ∫_0^x u dt, shape (M, N)
    """

    def __init__(
        self,
        dim: int,
        d: int,
        lower: float,
        upper: Optional[float] = None,
        order: int = 1,
    ):
        if order < 1:
            raise ValueError("integration order must be a positive integer")
        self.dim = dim
        self.d = d
        self.lower = lower
        self.upper = upper  # None => Volterra (variable upper limit x_dim)
        self.order = order

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def volterra(cls, dim: int, lower: float, d: int, order: int = 1) -> IntegralOperator:
        """Running integral ∫_{lower}^{x_dim} · d x_dim (Volterra)."""
        return cls(dim=dim, d=d, lower=lower, upper=None, order=order)

    @classmethod
    def definite(
        cls, dim: int, lower: float, upper: float, d: int, order: int = 1
    ) -> IntegralOperator:
        """Definite integral ∫_{lower}^{upper} · d x_dim."""
        return cls(dim=dim, d=d, lower=lower, upper=upper, order=order)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def apply(
        self,
        basis: SinusoidalBasis,
        x: torch.Tensor,
        cache: Optional[BasisCache] = None,
    ) -> torch.Tensor:
        """Evaluate the integral operator on ``basis`` at points ``x`` -> (M, N)."""
        if self.order == 1:
            # Stable closed form (no 1/W division); handles near-DC features exactly.
            return basis.definite_integral(
                x, self.dim, self.lower, upper=self.upper, cache=cache
            )
        # order >= 2: difference of (DC-guarded) higher antiderivatives.
        alpha = tuple(-self.order if k == self.dim else 0 for k in range(self.d))
        F_lo = basis.derivative(self._shift(x, self.lower), alpha)
        if self.upper is None:
            F_hi = basis.derivative(x, alpha, cache=cache)
        else:
            F_hi = basis.derivative(self._shift(x, self.upper), alpha)
        return F_hi - F_lo

    def _shift(self, x: torch.Tensor, value: float) -> torch.Tensor:
        """Copy of ``x`` with the integration axis pinned to ``value``."""
        xs = x.clone()
        xs[:, self.dim] = value
        return xs

    # ------------------------------------------------------------------
    # Arithmetic composition -> IntegroDifferentialOperator
    # ------------------------------------------------------------------

    def __add__(self, other):
        return IntegroDifferentialOperator([(1.0, self)]).__add__(other)

    def __radd__(self, other):
        return IntegroDifferentialOperator([(1.0, self)]).__radd__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __neg__(self):
        return IntegroDifferentialOperator([(-1.0, self)])

    def __mul__(self, scalar: CoeffT):
        return IntegroDifferentialOperator([(scalar, self)])

    __rmul__ = __mul__

    def __repr__(self) -> str:
        o = "" if self.order == 1 else f", order={self.order}"
        if self.upper is None:
            return f"IntegralOperator(∫_{self.lower}^x_{self.dim} d x_{self.dim}{o})"
        return f"IntegralOperator(∫_{self.lower}^{self.upper} d x_{self.dim}{o})"


# ======================================================================
# IntegroDifferentialOperator: the common roof for differential + integral terms
# ======================================================================

class IntegroDifferentialOperator:
    """Linear combination of heterogeneous operator terms, each evaluating to (M, N).

    The unifying abstraction under which :class:`DiffOperator` (differential) and
    :class:`IntegralOperator` (integral) terms compose into a single design matrix:

    >>> from fastlsq import Op, IntegralOperator
    >>> L = Op.partial(0, 1, d=1) + k * IntegralOperator.volterra(dim=0, lower=0.0, d=1)
    >>> A = L.apply(basis, x)          # one linear-least-squares block, shape (M, N)

    Terms are pairs ``(coeff, op)`` where ``op`` exposes ``.apply(basis, x, cache)`` and
    ``coeff`` is a scalar **or** a learnable tensor (e.g. ``nn.Parameter``) — gradients flow
    through integro-differential coefficients exactly as for :class:`DiffOperator`.
    """

    def __init__(self, terms):
        self.terms = list(terms)

    @staticmethod
    def _as_terms(other):
        if isinstance(other, IntegroDifferentialOperator):
            return list(other.terms)
        if isinstance(other, (DiffOperator, IntegralOperator)):
            return [(1.0, other)]
        return None

    def apply(
        self,
        basis: SinusoidalBasis,
        x: torch.Tensor,
        cache: Optional[BasisCache] = None,
    ) -> torch.Tensor:
        """Evaluate Σ_i coeff_i · op_i(basis, x) -> (M, N), sharing one cache for x."""
        if cache is None:
            cache = basis.cache(x)
        result = None
        for coeff, op in self.terms:
            M = op.apply(basis, x, cache=cache)
            term = coeff * M
            result = term if result is None else result + term
        return result

    def __add__(self, other):
        wrapped = self._as_terms(other)
        if wrapped is None:
            return NotImplemented
        return IntegroDifferentialOperator(self.terms + wrapped)

    def __radd__(self, other):
        wrapped = self._as_terms(other)
        if wrapped is None:
            return NotImplemented
        return IntegroDifferentialOperator(wrapped + self.terms)

    def __sub__(self, other):
        return self.__add__(-other)

    def __neg__(self):
        return IntegroDifferentialOperator([(-c, op) for c, op in self.terms])

    def __mul__(self, scalar: CoeffT):
        return IntegroDifferentialOperator([(c * scalar, op) for c, op in self.terms])

    __rmul__ = __mul__

    def __repr__(self) -> str:
        return "IntegroDifferentialOperator(" + " + ".join(
            f"{c}*{op!r}" if not isinstance(c, torch.Tensor) else f"tensor*{op!r}"
            for c, op in self.terms
        ) + ")"


# Shorthand alias
Op = DiffOperator


# ======================================================================
# ProjectionOperator: closed-form Radon / line-integral (projection) operator
# ======================================================================

class ProjectionOperator:
    """Closed-form projection (Radon) operator for a :class:`GaussianWindowedBasis`.

    Implements the line / hyperplane integral -- a Fredholm equation of the first
    kind::

        (P f)(u) = ∫ f(z) δ(c·z − u) dz

    i.e. the integral of ``f`` over the hyperplane ``{z : c·z = u}`` with normal
    ``c`` (beam phase-space tomography, CT, Abel inversion).  This is the
    *projection / Radon* class of the operator taxonomy: a projection onto a
    generally **non-axis-aligned** hyperplane, which the single-axis
    :class:`IntegralOperator` cannot express.

    For a Gaussian-windowed basis it assembles with **no quadrature** -- the
    Gaussian × plane-wave hyperplane integral is analytic.  In the whitened frame,
    with ``q = Tᵀc``, ``σ_u² = ‖q‖²``, ``u₀ = c·mean`` and ``jac = |det T| / ‖q‖``::

        (P ψ_j)(u) = jac · (2π)^((d−1)/2) · exp(−‖ω_j‖²/2)
                     · exp(−(u − u₀)² / (2 σ_u²)) · sin(α_j u + φ_j)
        α_j     = (W_j·q) / ‖q‖²
        ‖ω_j‖²  = ‖W_j‖² − (W_j·q̂)²            (q̂ = q/‖q‖;  across-slice energy)
        φ_j     = b_j − α_j u₀

    ``apply(basis, x)`` returns an ``(M, N)`` design matrix at the scalar detector
    coordinates ``x = u``, so a windowed-tomography fit is one linear least squares.

    Differentiability in ``c``
    --------------------------
    Differentiability of the assembled rows in the direction ``c`` (the *optics*) is
    a first-class requirement -- the downstream use is differentiable experiment
    design (autodiff ``d(posterior)/d(optics)`` to choose the next measurement).
    Two things would silently break it, and are deliberately avoided here:

    * the across-slice energy is computed from the **rotation-invariant** form
      ``‖ω_j‖² = ‖W_j‖² − (W_j·q̂)²`` -- mathematically identical to ``‖QᵀW_j‖²`` for
      an orthonormal complement ``Q`` of ``q``, but with **no** QR (a QR complement
      is not differentiable in ``c``);
    * every quantity (``u₀``, ``jac``, ``σ_u²``, ``α``, ...) stays a tensor -- no
      ``float()`` / ``.item()`` casts that would detach the graph.

    So ``autograd`` flows from the rows back to ``c`` (verified ``autodiff ==
    finite-difference`` to ~5e-9), as differentiable optics design needs.

    Scope / honesty
    ---------------
    * The closed form holds **only** for the Gaussian-windowed basis: the Gaussian ×
      plane-wave integral is analytic, but other windows (compact / polynomial)
      generally are **not**.  So the scope is *Gaussian-windowed* tomographic /
      line-integral operators, not "any projection".
    * The window is a *fixed prior* (set from data moments), required for
      convergence -- not a tuned hyperparameter.
    * This is a different analytic-kernel mechanism from the Fourier-symbol
      (convolution / fractional) operators already in the package; it is the
      **projection / Radon (line/hyperplane integral)** class.
    * No novelty is claimed over ELM / RBF-for-integral-equations prior art; the
      distinctive parts are *quadrature-free* closed-form projection rows,
      differentiability in the optics, and one unified linear-least-squares solve.

    It is standalone (it needs the windowed basis, so it does not compose into
    :class:`IntegroDifferentialOperator`), but mirrors the ``apply(basis, x, cache)``
    operator signature.

    Parameters
    ----------
    c : Tensor, shape (d,)
        Hyperplane normal / read-out direction.  May require grad (the optics).
    """

    def __init__(self, c: torch.Tensor):
        self.c = c

    @classmethod
    def from_transport(cls, M: torch.Tensor, e: torch.Tensor) -> ProjectionOperator:
        """Projection along read-out axis ``e`` after transport ``M``: ``c = Mᵀe``.

        The tomography convention: a phase-space coordinate is transported by a
        (linear) optics map ``M`` and read on axis ``e``, so the measured coordinate
        is ``e·(M z) = (Mᵀe)·z``.  ``M`` (and hence ``c``) is differentiable, so
        gradients flow back to the optics.
        """
        return cls(M.t() @ e)

    def apply(
        self,
        basis: GaussianWindowedBasis,
        x: torch.Tensor,
        cache: Optional[object] = None,
    ) -> torch.Tensor:
        """Assemble the ``(M, N)`` projection design matrix at detector coords ``x``.

        ``x`` is the scalar projection coordinate ``u`` per measurement, shape
        ``(M,)`` or ``(M, 1)``.  ``cache`` is accepted for operator-API symmetry and
        ignored (the closed form recomputes from ``c``).  No quadrature error.
        """
        c = self.c
        if c.dtype != basis.W.dtype or c.device != basis.W.device:
            c = c.to(dtype=basis.W.dtype, device=basis.W.device)

        u = x
        if u.dim() == 2 and u.shape[1] == 1:
            u = u[:, 0]
        elif u.dim() != 1:
            raise ValueError(
                "ProjectionOperator expects detector coords of shape (M,) or (M, 1)"
            )
        if u.dtype != basis.W.dtype or u.device != basis.W.device:
            u = u.to(dtype=basis.W.dtype, device=basis.W.device)

        q = basis.T.t() @ c                          # (d,)
        qn2 = q @ q                                  # ()   = σ_u²
        qn = torch.sqrt(qn2)                         # ()   = ‖q‖
        u0 = c @ basis.mean                          # ()
        jac = torch.det(basis.T).abs() / qn          # ()
        b = basis.b.reshape(-1)                      # (N,)
        alpha = (q @ basis.W) / qn2                  # (N,)
        # |ω|², rotation-invariant (== ‖QᵀW‖²) and differentiable in c -- NOT via QR.
        perp2 = (basis.W ** 2).sum(0) - ((q / qn) @ basis.W) ** 2   # (N,)
        phase = b - alpha * u0                       # (N,)
        amp = (
            jac
            * (2.0 * np.pi) ** ((basis.d - 1) / 2)
            * torch.exp(-0.5 * perp2.clamp_min(0.0))
        )                                            # (N,)
        env_u = torch.exp(-0.5 * (u - u0) ** 2 / qn2)   # (M,)
        return env_u[:, None] * (amp * torch.sin(u[:, None] * alpha + phase))


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
