# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Mirror-symmetric bases: exact parity under reflection across a coordinate plane.

A reflected sinusoid is still a sinusoid,

    sin(W . (R x) + b) = sin((R W) . x + b),        R = diag(1, .., -1, .., 1),

so pairing every feature with its reflection costs nothing in closed form: the
pair is two :class:`SinusoidalBasis` banks, ``(W, b)`` and ``(R W, b)``, added
(even parity) or subtracted (odd parity) column by column.  Every derivative is
the same cyclic identity applied to each bank.

Use it when the solution has a known symmetry -- flow past a body that is
symmetric about ``y = 0``, say -- and collocate only half the domain.  The
symmetry is then exact, not learned, and the half-domain needs half the
features for the same resolution.

    >>> half = SinusoidalBasis.random(3, 800, sigma=3.0)
    >>> p = MirrorBasis(half, axis=1, parity=+1)      # p(x, -y, z) = p(x, y, z)
    >>> A = Op.laplacian(d=3).apply(p, x)             # (M, 800), works with any Op
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from fastlsq.basis import SinusoidalBasis


def reflection(d: int, axis: int, device=None, dtype=None) -> torch.Tensor:
    """The diagonal of R: ones with -1 at ``axis``."""
    r = torch.ones(d, device=device, dtype=dtype or torch.get_default_dtype())
    r[axis] = -1.0
    return r


class _MirrorCache:
    __slots__ = ("direct", "mirror")

    def __init__(self, direct, mirror):
        self.direct = direct
        self.mirror = mirror


class MirrorBasis:
    """``phi_j(x) + parity * phi_j(R x)`` for a sinusoidal basis ``phi``.

    Parameters
    ----------
    basis : SinusoidalBasis
        The half bank; its normalisation convention carries over.
    axis : int
        The coordinate reflected (``axis=1`` reflects ``y``).
    parity : +1 or -1
        +1 gives functions even in ``x_axis``, -1 odd ones (which vanish on the
        mirror plane).

    Implements the full duck-typed basis protocol (``evaluate``, ``derivative``,
    ``gradient``, ``laplacian``, ``biharmonic``, ``advection``, ``hessian_diag``,
    ``operator``, ``symbol``, ``definite_integral``, ``multi_integral``,
    ``iterated_integral``, ``cache``), so every operator class --
    :class:`~fastlsq.basis.DiffOperator`, :class:`~fastlsq.basis.SymbolOperator`,
    :class:`~fastlsq.basis.IntegralOperator`,
    :class:`~fastlsq.basis.MultiIntegralOperator` -- and the solvers accept it
    unchanged.  Each is exact: the mirror bank is itself a bank of plane waves
    with frequencies ``R W``, so applying a linear operator bank by bank is the
    operator applied to the pair.
    """

    def __init__(self, basis: SinusoidalBasis, axis: int, parity: int = 1):
        if parity not in (1, -1):
            raise ValueError("parity must be +1 or -1")
        if not 0 <= axis < basis.input_dim:
            raise ValueError(f"axis {axis} out of range for input_dim {basis.input_dim}")
        self.basis = basis
        self.axis = axis
        self.parity = parity
        R = reflection(basis.input_dim, axis, device=basis.W.device, dtype=basis.W.dtype)
        self.mirror = SinusoidalBasis(R[:, None] * basis.W, basis.b,
                                      normalize=basis._inv_norm != 1.0)
        self.mirror._inv_norm = basis._inv_norm

    @property
    def input_dim(self) -> int:
        return self.basis.input_dim

    @property
    def n_features(self) -> int:
        return self.basis.n_features

    @property
    def W(self) -> torch.Tensor:
        return self.basis.W

    @property
    def b(self) -> torch.Tensor:
        return self.basis.b

    def cache(self, x: torch.Tensor) -> _MirrorCache:
        return _MirrorCache(self.basis.cache(x), self.mirror.cache(x))

    def _pair(self, name, x, *args, cache=None, **kw):
        c = cache if isinstance(cache, _MirrorCache) else self.cache(x)
        a = getattr(self.basis, name)(x, *args, cache=c.direct, **kw)
        m = getattr(self.mirror, name)(x, *args, cache=c.mirror, **kw)
        return a + m if self.parity == 1 else a - m

    def evaluate(self, x, cache=None):
        return self._pair("evaluate", x, cache=cache)

    def derivative(self, x, alpha, cache=None):
        return self._pair("derivative", x, alpha, cache=cache)

    def gradient(self, x, cache=None):
        return self._pair("gradient", x, cache=cache)

    def hessian_diag(self, x, cache=None):
        return self._pair("hessian_diag", x, cache=cache)

    def laplacian(self, x, dims: Optional[Sequence[int]] = None, cache=None):
        return self._pair("laplacian", x, dims=dims, cache=cache)

    def operator(self, x, terms, cache=None):
        return self._pair("operator", x, terms, cache=cache)

    def biharmonic(self, x, dims: Optional[Sequence[int]] = None, cache=None):
        return self._pair("biharmonic", x, dims=dims, cache=cache)

    def advection(self, x, v, cache=None):
        return self._pair("advection", x, v, cache=cache)

    def symbol(self, x, m, cache=None):
        return self._pair("symbol", x, m, cache=cache)

    def definite_integral(self, x, dim, lower, upper=None, cache=None):
        return self._pair("definite_integral", x, dim, lower, upper=upper, cache=cache)

    def multi_integral(self, x, dims, lowers, uppers=None, cache=None):
        return self._pair("multi_integral", x, dims, lowers, uppers=uppers, cache=cache)

    def iterated_integral(self, x, dim, lower, upper=None, order=1, cache=None):
        return self._pair("iterated_integral", x, dim, lower, upper=upper,
                          order=order, cache=cache)

    def plain(self):
        """The full (unpaired) bank: ``W`` (d, 2N), ``b`` (1, 2N) and the column
        signs, so that ``predict = sin(x @ W + b) @ (sign * [beta; beta])``."""
        W = torch.cat([self.basis.W, self.mirror.W], 1)
        b = torch.cat([self.basis.b, self.basis.b], 1)
        sign = torch.cat([torch.ones(self.n_features), self.parity * torch.ones(self.n_features)])
        return W, b, sign.to(W)
