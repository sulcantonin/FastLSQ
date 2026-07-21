# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Separable (degenerate) integral kernels and Fredholm equations.

A kernel is **separable** (classically, *degenerate*) when it is a finite sum of
products of a function of ``x`` and a function of ``y``::

    K(x, y) = Σ_{m=1}^{R} g_m(x) h_m(y)

That structure collapses the integral operator::

    (Ku)(x) = ∫_Ω K(x, y) u(y) dy = Σ_m g_m(x) · ∫_Ω h_m(y) u(y) dy

so acting on the basis needs only the ``R × N`` matrix of inner products
``C_{mj} = ∫ h_m φ_j``, computed **once** and independent of the collocation
points.  Assembly is then ``A = G(x) @ C`` with ``G`` of shape ``(M, R)`` -- a
rank-``R`` factorisation rather than an ``M × M`` kernel evaluation.

This is what makes the classical theory of degenerate kernels constructive here:
a Fredholm equation of the second kind

    u(x) − λ ∫_Ω K(x, y) u(y) dy = f(x)

is assembled as ``Op.identity(d) − λ·K`` and solved in the same single linear
least squares as every other operator in the package -- see
:func:`fredholm_second_kind`.

Where the quadrature is
-----------------------
The inner products ``C`` are the one place quadrature enters, and it is a
*precomputation*, not a per-row cost: tensor-product Gauss-Legendre over the
integration box, exact for polynomial integrands of degree ``≤ 2·n_quad − 1``
and spectrally convergent for smooth ones.  It must nevertheless resolve the
oscillation of the features, so ``n_quad`` should exceed a few nodes per
wavelength of the largest ``‖W‖`` -- :meth:`SeparableKernelOperator.check_quadrature`
reports the achieved convergence, and analytic inner products can be supplied
directly with :meth:`SeparableKernelOperator.from_inner_products`.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Callable, Optional, Sequence, Union

from fastlsq.basis import CoeffT, DiffOperator, IntegroDifferentialOperator

FnT = Callable[[torch.Tensor], torch.Tensor]


def _tensor_gauss_legendre(lower, upper, d: int, n_quad: int, device, dtype):
    """Tensor-product Gauss-Legendre nodes/weights on a box -> ((Q, d), (Q,))."""
    lo = np.atleast_1d(np.asarray(lower, dtype=float))
    hi = np.atleast_1d(np.asarray(upper, dtype=float))
    if lo.size == 1:
        lo = np.repeat(lo, d)
    if hi.size == 1:
        hi = np.repeat(hi, d)
    if lo.size != d or hi.size != d:
        raise ValueError("lower/upper must be scalars or have one entry per axis")

    t, w = np.polynomial.legendre.leggauss(n_quad)
    axes_n, axes_w = [], []
    for k in range(d):
        axes_n.append(0.5 * (hi[k] - lo[k]) * t + 0.5 * (hi[k] + lo[k]))
        axes_w.append(0.5 * (hi[k] - lo[k]) * w)

    grids = np.meshgrid(*axes_n, indexing="ij")
    nodes = np.stack([g.reshape(-1) for g in grids], axis=1)
    wgrids = np.meshgrid(*axes_w, indexing="ij")
    weights = np.prod(np.stack([g.reshape(-1) for g in wgrids], axis=1), axis=1)

    return (
        torch.as_tensor(nodes, device=device, dtype=dtype),
        torch.as_tensor(weights, device=device, dtype=dtype),
    )


class SeparableKernelOperator:
    """Integral operator with a separable kernel ``K(x,y) = Σ_m g_m(x) h_m(y)``.

    Parameters
    ----------
    g_terms : sequence of callable
        ``g_m``, each mapping ``(M, d) -> (M,)`` or ``(M, 1)``.
    h_terms : sequence of callable
        ``h_m``, each mapping ``(Q, d) -> (Q,)`` or ``(Q, 1)``.  Must have the
        same length as ``g_terms``.
    lower, upper : float or sequence of float
        Integration box ``Ω``.  Scalars apply to every axis.
    d : int
        Spatial dimension.
    n_quad : int
        Gauss-Legendre nodes **per axis** for the inner products (so ``Q =
        n_quad**d``).  Only used the first time :meth:`apply` sees a basis.

    Examples
    --------
    The classic degenerate kernel ``K(x,y) = x y`` on ``[0,1]``::

        >>> K = SeparableKernelOperator([lambda x: x[:, 0]],
        ...                             [lambda y: y[:, 0]],
        ...                             lower=0.0, upper=1.0, d=1)
        >>> A = K.apply(basis, x)              # (M, N), rank <= 1

    Notes
    -----
    The assembled block has rank at most ``R = len(g_terms)``, by construction.
    That is the point -- a degenerate kernel carries only ``R`` degrees of
    freedom no matter how many collocation points are used -- but it also means
    ``K`` alone can never be solved for ``u``; it is a *compact* operator.  Use
    it in a second-kind equation (``I − λK``, see :func:`fredholm_second_kind`),
    which is well posed, rather than a first-kind one, which is not.
    """

    def __init__(
        self,
        g_terms: Sequence[FnT],
        h_terms: Sequence[FnT],
        lower: Union[float, Sequence[float]],
        upper: Union[float, Sequence[float]],
        d: int = 1,
        n_quad: int = 200,
    ):
        if len(g_terms) != len(h_terms):
            raise ValueError(
                f"g_terms and h_terms must have equal length; got "
                f"{len(g_terms)} and {len(h_terms)}"
            )
        self.g_terms = list(g_terms)
        self.h_terms = list(h_terms)
        self.lower = lower
        self.upper = upper
        self.d = d
        self.n_quad = n_quad
        self._C: Optional[torch.Tensor] = None
        self._C_key = None

    @property
    def rank(self) -> int:
        """Number of separable terms ``R`` -- an upper bound on the block's rank."""
        return len(self.g_terms)

    # ------------------------------------------------------------------
    # Inner products
    # ------------------------------------------------------------------

    @classmethod
    def from_inner_products(
        cls, g_terms: Sequence[FnT], C: torch.Tensor
    ) -> SeparableKernelOperator:
        """Build from **analytic** inner products, skipping quadrature entirely.

        ``C`` has shape ``(R, N)`` with ``C[m, j] = ∫ h_m φ_j``.  Use when the
        inner products are known in closed form for the basis at hand.
        """
        obj = cls(g_terms, [None] * len(g_terms), 0.0, 1.0, d=1, n_quad=0)
        obj._C = C
        obj._C_key = "analytic"
        return obj

    def inner_products(self, basis) -> torch.Tensor:
        """``C[m, j] = ∫_Ω h_m(y) φ_j(y) dy`` -> ``(R, N)``, computed once per basis."""
        key = (id(basis), basis.n_features)
        if self._C is not None and self._C_key in ("analytic", key):
            return self._C

        nodes, w = _tensor_gauss_legendre(
            self.lower, self.upper, self.d, self.n_quad,
            basis.W.device, basis.W.dtype,
        )
        phi = basis.evaluate(nodes)                       # (Q, N)
        H = torch.stack([h(nodes).reshape(-1) for h in self.h_terms])  # (R, Q)
        self._C = (H * w) @ phi                           # (R, N)
        self._C_key = key
        return self._C

    def check_quadrature(self, basis, refine: int = 2) -> float:
        """Relative change in ``C`` when the quadrature is refined ``refine``x.

        A small value means the inner products are converged for this basis; a
        large one means ``n_quad`` does not resolve the feature oscillation and
        should be raised.  Returns the relative Frobenius difference.
        """
        coarse = self.inner_products(basis).clone()
        saved_C, saved_key, saved_n = self._C, self._C_key, self.n_quad
        self._C, self._C_key, self.n_quad = None, None, self.n_quad * refine
        try:
            fine = self.inner_products(basis)
            rel = (torch.norm(fine - coarse) / torch.norm(fine)).item()
        finally:
            self._C, self._C_key, self.n_quad = saved_C, saved_key, saved_n
        return rel

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def apply(self, basis, x: torch.Tensor, cache=None) -> torch.Tensor:
        """Assemble ``(Kφ_j)(x)`` -> ``(M, N)`` as the rank-``R`` product ``G @ C``."""
        if x.dtype != basis.W.dtype or x.device != basis.W.device:
            x = x.to(dtype=basis.W.dtype, device=basis.W.device)
        C = self.inner_products(basis)                    # (R, N)
        G = torch.stack([g(x).reshape(-1) for g in self.g_terms], dim=1)  # (M, R)
        return G @ C

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
        return f"SeparableKernelOperator(rank={self.rank}, d={self.d})"


# ======================================================================
# Fredholm equations
# ======================================================================

def fredholm_second_kind(
    kernel: SeparableKernelOperator, lam: CoeffT, d: int
) -> IntegroDifferentialOperator:
    """Operator for ``u(x) − λ ∫ K(x,y) u(y) dy = f(x)`` (Fredholm, second kind).

    Simply ``I − λK``, returned as a composed operator so it assembles into one
    design matrix::

        >>> L = fredholm_second_kind(K, lam=0.5, d=1)
        >>> A = L.apply(basis, x)
        >>> beta = solve_lstsq(A, f(x))

    Second-kind equations are well posed for ``λ`` away from the kernel's
    eigenvalues; the identity dominates and the compact ``K`` perturbs it.  For a
    degenerate kernel of rank ``R`` there are at most ``R`` such eigenvalues, and
    :func:`degenerate_eigenvalues` computes them so a near-singular ``λ`` can be
    detected rather than silently producing a garbage fit.

    ``lam`` may be an ``nn.Parameter``, making the equation differentiable in the
    coupling strength.
    """
    return DiffOperator.identity(d) - lam * kernel


def degenerate_eigenvalues(kernel: SeparableKernelOperator, basis) -> torch.Tensor:
    """Characteristic values of a degenerate kernel, as seen by ``basis``.

    For ``K = Σ g_m h_m`` the eigenvalue problem ``φ = λ K φ`` reduces to the
    ``R × R`` matrix ``S_{mn} = ∫ h_m g_n``; the equation ``u − λKu = f`` is
    singular exactly at ``λ = 1/μ`` for ``μ`` an eigenvalue of ``S``.  Returns
    those ``1/μ`` (complex, as ``S`` need not be symmetric).

    Use it to sanity-check a chosen ``λ``: a value close to one of these makes
    the second-kind problem ill posed, which shows up as a large residual rather
    than an obvious error.
    """
    C = kernel.inner_products(basis)                      # (R, N) = ∫ h_m φ_j
    # S[m, n] = ∫ h_m g_n, obtained by expanding g_n on the same quadrature.
    nodes, w = _tensor_gauss_legendre(
        kernel.lower, kernel.upper, kernel.d, kernel.n_quad,
        basis.W.device, basis.W.dtype,
    )
    H = torch.stack([h(nodes).reshape(-1) for h in kernel.h_terms])   # (R, Q)
    G = torch.stack([g(nodes).reshape(-1) for g in kernel.g_terms])   # (R, Q)
    S = (H * w) @ G.t()                                   # (R, R)
    mu = torch.linalg.eigvals(S)
    return 1.0 / mu
