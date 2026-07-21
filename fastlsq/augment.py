# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Polynomial / DC augmentation columns for the design matrix.

A sinusoidal feature bank cannot represent a constant: every ``sin(W·x + b)``
with ``W ≠ 0`` integrates to a sinusoid, so the *arbitrary constant* of an
indefinite integral, and the DC component of a solution, both leave the family.
The same holds for the linear ramp left by a second antiderivative, and so on.

Two ways out.  The one every shipped example currently uses is a **weighted
boundary row**: pin the solution at a point and let least squares infer the
offset.  That works, but it couples the offset to the boundary weight, and it
cannot express an offset the operator genuinely annihilates.

The other -- and what §2.3 means by "pinned by an explicit polynomial column" --
is to *widen the basis*: append columns ``1, x, x², …`` whose operator image is
known exactly, so the offset is a fitted coefficient rather than a soft penalty.
That is what this module provides::

    >>> from fastlsq import AugmentedBasis, PolynomialColumns
    >>> aug = AugmentedBasis(basis, PolynomialColumns(degree=1, dim=1))
    >>> A   = op.apply(aug, x)      # (M, N + n_poly) -- every operator, unchanged

:class:`AugmentedBasis` implements the same duck-typed protocol as
:class:`~fastlsq.basis.SinusoidalBasis`, so it drops into
:class:`~fastlsq.basis.DiffOperator`, :class:`~fastlsq.basis.IntegralOperator`,
:class:`~fastlsq.basis.SymbolOperator` and the solvers without any of them
knowing.  Coefficients come back stacked ``[β_features; β_poly]``.

The columns carry **exact** operator images -- analytic monomial derivatives and
antiderivatives, not a zero-derivative stub -- so an augmented column is as
correct under ``D^α`` and under an integral operator as the features are.
"""

from __future__ import annotations

import itertools
import math

import torch
import numpy as np
from typing import Optional, Sequence, Union

from fastlsq.basis import BasisCache, CoeffT, SinusoidalBasis


# ======================================================================
# AugmentedCache
# ======================================================================

class AugmentedCache:
    """Cache for an :class:`AugmentedBasis`: the feature cache plus the points.

    The polynomial block needs ``x`` itself (not ``Z = xW + b``), so the two
    halves cache different things.  Operators only ever hand a cache back to the
    basis that made it, so carrying both is safe.
    """

    __slots__ = ("inner", "x")

    def __init__(self, inner: BasisCache, x: torch.Tensor):
        self.inner = inner
        self.x = x


# ======================================================================
# PolynomialColumns
# ======================================================================

class PolynomialColumns:
    """Monomial columns ``x^β`` for ``|β| <= degree``, with exact operator images.

    Parameters
    ----------
    degree : int
        Maximum **total** degree.  ``degree=0`` is the single DC column (the
        constant), which is what pins one integration constant; ``degree=1`` adds
        the linear ramp needed by a second antiderivative, and so on.
    dim : int
        Spatial dimension ``d`` of the problem.
    dims : sequence of int, optional
        Restrict the monomials to vary only along these axes (others get
        exponent 0).  Useful in space-time problems where only the time axis
        needs a ramp.  Default: all axes.

    Attributes
    ----------
    exponents : Tensor, shape (n_cols, d)
        The multi-indices ``β``, in graded order starting with the constant.

    Notes
    -----
    Adding columns necessarily raises the condition number -- a constant column
    is nearly parallel to any low-frequency feature.  Keep ``degree`` as low as
    the operator's null space demands (usually 0 or 1); the rank-revealing solver
    absorbs the rest.
    """

    def __init__(self, degree: int = 0, dim: int = 1, dims: Optional[Sequence[int]] = None):
        if degree < 0:
            raise ValueError("degree must be >= 0")
        self.degree = degree
        self.dim = dim
        axes = list(range(dim)) if dims is None else list(dims)
        self.dims = axes

        rows = []
        for total in range(degree + 1):
            for combo in itertools.combinations_with_replacement(axes, total):
                beta = [0] * dim
                for k in combo:
                    beta[k] += 1
                rows.append(beta)
        # De-duplicate while preserving graded order.
        seen, uniq = set(), []
        for r in rows:
            t = tuple(r)
            if t not in seen:
                seen.add(t)
                uniq.append(r)
        self.exponents = torch.tensor(uniq, dtype=torch.long)

    @property
    def n_columns(self) -> int:
        return self.exponents.shape[0]

    def __repr__(self) -> str:
        return f"PolynomialColumns(degree={self.degree}, dim={self.dim}, n={self.n_columns})"

    # ------------------------------------------------------------------
    # Exact operator images
    # ------------------------------------------------------------------

    def _monomials(self, x: torch.Tensor, exps: torch.Tensor) -> torch.Tensor:
        """``∏_k x_k^{e_k}`` for each column -> (M, n_cols)."""
        e = exps.to(x.device)
        # (M, 1, d) ** (1, n, d) -> (M, n, d), then product over d.
        p = x.unsqueeze(1) ** e.unsqueeze(0).to(x.dtype)
        return p.prod(dim=-1)

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Column values -> (M, n_cols)."""
        return self._monomials(x, self.exponents)

    def derivative(self, x: torch.Tensor, alpha: Sequence[int]) -> torch.Tensor:
        """``D^α x^β`` -> (M, n_cols), exact, for positive *or* negative ``α``.

        Uses ``D^α x^β = [β!/(β−α)!] x^{β−α}`` per axis, which is the falling
        factorial for a derivative and the reciprocal rising factorial for an
        antiderivative -- e.g. ``α = −1`` gives ``x^{β+1}/(β+1)``, the correct
        indefinite integral.  A derivative that kills the monomial
        (``α_k > β_k``) yields a zero column, as it must.
        """
        beta = self.exponents.to(x.device)                       # (n, d)
        a = torch.as_tensor(list(alpha), device=x.device, dtype=torch.long)  # (d,)
        new = beta - a.unsqueeze(0)                              # (n, d)

        alive = (new >= 0).all(dim=1)                            # (n,)
        safe = new.clamp_min(0)

        # coefficient = prod_k Gamma(beta_k+1) / Gamma(new_k+1)
        lg = torch.lgamma((beta + 1).to(x.dtype)) - torch.lgamma((safe + 1).to(x.dtype))
        coeff = torch.exp(lg.sum(dim=1))                         # (n,)
        coeff = torch.where(alive, coeff, torch.zeros_like(coeff))

        vals = self._monomials(x, safe)
        return vals * coeff.unsqueeze(0)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """``∇ x^β`` -> (M, d, n_cols)."""
        cols = []
        for k in range(self.dim):
            a = [0] * self.dim
            a[k] = 1
            cols.append(self.derivative(x, a))
        return torch.stack(cols, dim=1)

    def iterated_integral(
        self,
        x: torch.Tensor,
        dim: int,
        lower: float,
        upper: Optional[float] = None,
        order: int = 1,
    ) -> torch.Tensor:
        """``n``-fold iterated integral along ``dim``, all lower limits at ``lower``.

        Matches :meth:`SinusoidalBasis.iterated_integral`: the Cauchy repeated
        integral, *not* ``F_n(hi) − F_n(lo)``.  Expanding the monomial about the
        lower limit, ``t^p = Σ_j C(p,j) lo^{p−j} (t−lo)^j``, and using
        ``∫^{(n)}_{lo} (t−lo)^j = (x−lo)^{j+n} · j!/(j+n)!`` gives the closed form

            F_n(x) = Σ_{j=0}^{p} C(p,j) lo^{p−j} (x−lo)^{j+n} j!/(j+n)!

        exactly, with no quadrature.  Other axes keep their monomial factors
        evaluated at ``x``.
        """
        beta = self.exponents.to(x.device)                       # (n, d)
        u = (upper if upper is not None else x[:, dim])
        if not torch.is_tensor(u):
            u = torch.full((x.shape[0],), float(u), device=x.device, dtype=x.dtype)
        du = (u - lower).unsqueeze(1)                            # (M, 1)

        # Factors from the axes that are NOT integrated.
        other = beta.clone()
        other[:, dim] = 0
        rest = self._monomials(x, other)                         # (M, n)

        p = beta[:, dim]                                         # (n,)
        out = torch.zeros(x.shape[0], self.n_columns, device=x.device, dtype=x.dtype)
        pmax = int(p.max().item()) if self.n_columns else 0
        for j in range(pmax + 1):
            mask = (p >= j).to(x.dtype)                          # (n,)
            # C(p, j) * lower^(p-j) * j!/(j+n)!
            comb = torch.exp(
                torch.lgamma((p + 1).to(x.dtype))
                - torch.lgamma(torch.tensor(float(j + 1), device=x.device, dtype=x.dtype))
                - torch.lgamma((p - j).clamp_min(0).to(x.dtype) + 1)
            )
            lo_pow = torch.tensor(float(lower), device=x.device, dtype=x.dtype) ** (
                (p - j).clamp_min(0).to(x.dtype)
            )
            scale = math.factorial(j) / math.factorial(j + order)
            out = out + (mask * comb * lo_pow * scale).unsqueeze(0) * du ** (j + order)
        return out * rest

    def definite_integral(
        self,
        x: torch.Tensor,
        dim: int,
        lower: float,
        upper: Optional[float] = None,
    ) -> torch.Tensor:
        """Single integral along ``dim`` -- ``order=1`` of :meth:`iterated_integral`."""
        return self.iterated_integral(x, dim, lower, upper=upper, order=1)

    def symbol(self, x: torch.Tensor, m) -> torch.Tensor:
        """Fourier multiplier applied to the columns -> (M, n_cols).

        Only the **constant** column has a defined image under a general symbol:
        it is the zero-frequency plane wave, so ``L·1 = m(0)``.  A non-constant
        monomial is not in ``L¹ + L²`` and its multiplier image is a distribution,
        not a function -- there is no honest value to return, so this raises.

        Practically this is rarely a limitation: augmentation exists to pin the
        null space of *differential and integral* operators.  If you need a
        polynomial column alongside a nonlocal operator, restrict to
        ``degree=0``.
        """
        if self.degree > 0:
            raise NotImplementedError(
                "SymbolOperator on polynomial columns of degree > 0 is not defined: "
                "a non-constant monomial has no function-valued Fourier-multiplier "
                "image. Use PolynomialColumns(degree=0) (the DC column) with "
                "symbol operators, or pin the offset with a boundary row."
            )
        zero = torch.zeros(self.dim, 1, device=x.device, dtype=x.dtype)
        m_val = m(zero) if callable(m) and not isinstance(m, torch.Tensor) else m
        m_val = torch.as_tensor(m_val, device=x.device).reshape(-1)[:1]
        if m_val.is_complex():
            # An operator that maps real functions to real functions has a
            # Hermitian symbol, m(-xi) = conj(m(xi)), so m(0) is necessarily
            # real; any imaginary part is round-off and is dropped.
            m_val = m_val.real
        return m_val.to(x.dtype).reshape(1, 1).expand(x.shape[0], self.n_columns)


# ======================================================================
# AugmentedBasis
# ======================================================================

class AugmentedBasis:
    """A :class:`~fastlsq.basis.SinusoidalBasis` widened by explicit extra columns.

    Implements the same protocol as the wrapped basis, so every operator and
    solver works unchanged; each method concatenates the feature block with the
    augmentation block along the column axis::

        [ op(features)  |  op(columns) ]     shape (M, N + n_cols)

    Parameters
    ----------
    basis : SinusoidalBasis
        The feature bank.
    columns : PolynomialColumns
        The augmentation block.

    Examples
    --------
    Pin the integration constant of an indefinite integral::

        >>> aug = AugmentedBasis(basis, PolynomialColumns(degree=0, dim=1))
        >>> L   = Op.partial(0, 1, d=1) - 2.0 * Op.antiderivative(0, 1, d=1)
        >>> A   = L.apply(aug, x)          # (M, N+1)

    Split the fitted coefficients back out with :meth:`split`.
    """

    def __init__(self, basis: SinusoidalBasis, columns: PolynomialColumns):
        if basis.input_dim != columns.dim:
            raise ValueError(
                f"dimension mismatch: basis.input_dim={basis.input_dim} but "
                f"columns.dim={columns.dim}"
            )
        self.basis = basis
        self.columns = columns

    # ------------------------------------------------------------------
    # Protocol: shape / caching
    # ------------------------------------------------------------------

    @property
    def input_dim(self) -> int:
        return self.basis.input_dim

    @property
    def n_features(self) -> int:
        return self.basis.n_features + self.columns.n_columns

    @property
    def W(self) -> torch.Tensor:
        return self.basis.W

    @property
    def b(self) -> torch.Tensor:
        return self.basis.b

    def cache(self, x: torch.Tensor) -> AugmentedCache:
        if x.dtype != self.basis.W.dtype or x.device != self.basis.W.device:
            x = x.to(dtype=self.basis.W.dtype, device=self.basis.W.device)
        return AugmentedCache(self.basis.cache(x), x)

    @staticmethod
    def _parts(cache):
        """Unpack an AugmentedCache, tolerating None."""
        if cache is None:
            return None, None
        if isinstance(cache, AugmentedCache):
            return cache.inner, cache.x
        return cache, None

    def _x(self, x: torch.Tensor, cache) -> torch.Tensor:
        _, cx = self._parts(cache)
        xx = cx if cx is not None else x
        if xx.dtype != self.basis.W.dtype or xx.device != self.basis.W.device:
            xx = xx.to(dtype=self.basis.W.dtype, device=self.basis.W.device)
        return xx

    # ------------------------------------------------------------------
    # Protocol: operator images
    # ------------------------------------------------------------------

    def evaluate(self, x: torch.Tensor, cache=None) -> torch.Tensor:
        inner, _ = self._parts(cache)
        return torch.cat(
            [self.basis.evaluate(x, cache=inner), self.columns.evaluate(self._x(x, cache))],
            dim=1,
        )

    def derivative(self, x: torch.Tensor, alpha, cache=None) -> torch.Tensor:
        inner, _ = self._parts(cache)
        return torch.cat(
            [
                self.basis.derivative(x, alpha, cache=inner),
                self.columns.derivative(self._x(x, cache), alpha),
            ],
            dim=1,
        )

    def gradient(self, x: torch.Tensor, cache=None) -> torch.Tensor:
        inner, _ = self._parts(cache)
        return torch.cat(
            [self.basis.gradient(x, cache=inner), self.columns.gradient(self._x(x, cache))],
            dim=2,
        )

    def laplacian(self, x: torch.Tensor, dims=None, cache=None) -> torch.Tensor:
        inner, _ = self._parts(cache)
        xx = self._x(x, cache)
        axes = range(self.input_dim) if dims is None else dims
        poly = None
        for k in axes:
            a = [0] * self.input_dim
            a[k] = 2
            term = self.columns.derivative(xx, a)
            poly = term if poly is None else poly + term
        return torch.cat([self.basis.laplacian(x, dims=dims, cache=inner), poly], dim=1)

    def operator(self, x: torch.Tensor, terms, cache=None) -> torch.Tensor:
        """Mirror of :meth:`SinusoidalBasis.operator`, applied to both blocks."""
        inner, _ = self._parts(cache)
        xx = self._x(x, cache)
        feat = self.basis.operator(x, terms, cache=inner)

        d = self.input_dim
        poly = torch.zeros(
            xx.shape[0], self.columns.n_columns, device=xx.device, dtype=xx.dtype
        )
        for coeff, alpha in terms:
            if isinstance(alpha, dict):
                al = [0] * d
                for k, o in alpha.items():
                    al[k] = o
                alpha = tuple(al)
            D = self.columns.derivative(xx, alpha)
            if callable(coeff) and not isinstance(coeff, torch.Tensor):
                c = coeff(xx)
                if c.dim() == 1:
                    c = c.unsqueeze(-1)
                poly = poly + c * D
            else:
                poly = poly + coeff * D
        return torch.cat([feat, poly], dim=1)

    def definite_integral(self, x, dim, lower, upper=None, cache=None) -> torch.Tensor:
        inner, _ = self._parts(cache)
        return torch.cat(
            [
                self.basis.definite_integral(x, dim, lower, upper=upper, cache=inner),
                self.columns.definite_integral(self._x(x, cache), dim, lower, upper=upper),
            ],
            dim=1,
        )

    def iterated_integral(self, x, dim, lower, upper=None, order=1, cache=None) -> torch.Tensor:
        inner, _ = self._parts(cache)
        return torch.cat(
            [
                self.basis.iterated_integral(
                    x, dim, lower, upper=upper, order=order, cache=inner
                ),
                self.columns.iterated_integral(
                    self._x(x, cache), dim, lower, upper=upper, order=order
                ),
            ],
            dim=1,
        )

    def symbol(self, x, m, cache=None) -> torch.Tensor:
        inner, _ = self._parts(cache)
        return torch.cat(
            [
                self.basis.symbol(x, m, cache=inner),
                self.columns.symbol(self._x(x, cache), m),
            ],
            dim=1,
        )

    # ------------------------------------------------------------------
    # Coefficient bookkeeping
    # ------------------------------------------------------------------

    def split(self, beta: torch.Tensor):
        """Split fitted coefficients into ``(feature_part, polynomial_part)``."""
        n = self.basis.n_features
        return beta[:n], beta[n:]

    def __repr__(self) -> str:
        return (
            f"AugmentedBasis({self.basis.n_features} features + "
            f"{self.columns.n_columns} columns)"
        )
