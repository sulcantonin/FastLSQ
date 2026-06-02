# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Vector-valued sinusoidal basis -- 0.1.5 addition.

Use this when the unknown is a **vector field**

    u(x)  =  (u_1(x),  u_2(x),  ...,  u_K(x))

such as

* incompressible NS primitive variables           (u, v, p)
* streamfunction-vorticity formulation             (psi, omega)
* multi-species advection-diffusion                (c_1, ..., c_K)
* coupled Maxwell components, MHD, etc.

A :class:`VectorBasis` is *K independent* random-Fourier-feature
``SinusoidalBasis`` instances bundled into one object.  Each component
has its OWN random weights so the components can have different
smoothness / bandwidth (for instance pressure usually smoother than
velocity).  Output tensors stack the component axis as the second
dimension:

    evaluate(x)            -> (M, K, N)
    gradient(x)            -> (M, K, d, N)
    laplacian(x)           -> (M, K, N)
    derivative(x, alpha)   -> (M, K, N)

A :class:`VectorFastLSQSolver` wraps a list of scalar
``FastLSQSolver``s and exposes ``basis``, ``add_block``, ``predict``
just like the scalar version, but in the vector setting.

Example -- streamfunction-vorticity NS for a single (psi, omega) pair:

>>> from fastlsq import VectorFastLSQSolver, Op
>>> solver = VectorFastLSQSolver(input_dim=2, n_components=2)
>>> for _ in range(3):
...     solver.add_block(hidden_size=500, scale=2.0)
>>> basis = solver.basis            # VectorBasis with 2 components
>>>
>>> # In the coupled-system LSQ you typically index each component
>>> # individually and stack rows manually:
>>> H_psi   = basis.component(0).evaluate(x_pde)        # (M, N)
>>> Lap_psi = basis.component(0).laplacian(x_pde)
>>> H_omg   = basis.component(1).evaluate(x_pde)
>>>
>>> # row 1:  omega + nabla^2 psi  =  0       (definition of vorticity)
>>> A_row_1 = torch.cat([Lap_psi,  H_omg], dim=1)
>>> # row 2:  -nu nabla^2 omega + (psi_y) d_x omega - (psi_x) d_y omega = 0
>>> # ... assemble similarly ...
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import torch

from fastlsq.basis import SinusoidalBasis
from fastlsq.solvers import FastLSQSolver
from fastlsq.utils import device


# ======================================================================
# VectorBasis -- K stacked SinusoidalBasis instances
# ======================================================================


class VectorBasis:
    """K-component random-Fourier-feature basis.

    Internally a list of :class:`SinusoidalBasis` (one per output
    component) that all share the same ``input_dim``.  Per-component
    weight matrices ``W`` and biases ``b`` are independent so
    components can have different bandwidths.

    Parameters
    ----------
    components : sequence of SinusoidalBasis
        At least one.  All must have the same ``input_dim``.  They
        do not need to have the same ``n_features``, but if they do
        the stacked output methods (`evaluate`, `gradient`, etc.) can
        produce a single dense tensor.

    Attributes
    ----------
    components : list[SinusoidalBasis]
    n_components : int
    input_dim : int
    n_features_per_component : int -- assumes all components match
    n_features_total : int        -- sum across components
    """

    def __init__(self, components: Sequence[SinusoidalBasis]):
        comps = list(components)
        if not comps:
            raise ValueError("VectorBasis needs at least one component")
        dims = {c.input_dim for c in comps}
        if len(dims) > 1:
            raise ValueError(
                f"all components must share input_dim, got {sorted(dims)}"
            )
        self.components: list[SinusoidalBasis] = comps

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def random(
        cls,
        input_dim: int,
        n_features: int,
        sigma: float = 1.0,
        n_components: int = 2,
        normalize: bool = True,
        sigmas: Optional[Sequence[float]] = None,
    ) -> "VectorBasis":
        """Make a VectorBasis with K independent isotropic random bases.

        Parameters
        ----------
        sigmas : optional, length ``n_components``
            Per-component bandwidth.  If given, ``sigma`` is ignored.
            Useful when components have different natural smoothness
            (e.g. pressure smoother than velocity).
        """
        if sigmas is None:
            sigmas = [sigma] * n_components
        if len(sigmas) != n_components:
            raise ValueError(
                f"len(sigmas)={len(sigmas)} != n_components={n_components}"
            )
        comps = [
            SinusoidalBasis.random(
                input_dim=input_dim,
                n_features=n_features,
                sigma=s,
                normalize=normalize,
            )
            for s in sigmas
        ]
        return cls(comps)

    @classmethod
    def from_solvers(
        cls, solvers: Sequence[FastLSQSolver]
    ) -> "VectorBasis":
        """Bundle ``solver.basis`` of each scalar solver into a VectorBasis."""
        return cls([s.basis for s in solvers])

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def n_components(self) -> int:
        return len(self.components)

    @property
    def input_dim(self) -> int:
        return self.components[0].input_dim

    @property
    def n_features_per_component(self) -> int:
        ns = {c.n_features for c in self.components}
        if len(ns) > 1:
            raise RuntimeError(
                "components have different feature counts; use "
                "`component(k).n_features` instead"
            )
        return self.components[0].n_features

    @property
    def n_features_total(self) -> int:
        return sum(c.n_features for c in self.components)

    def component(self, k: int) -> SinusoidalBasis:
        """The k-th underlying scalar basis."""
        return self.components[k]

    # ------------------------------------------------------------------
    # Stacked evaluators -- only valid when all components share n_features
    # ------------------------------------------------------------------

    def _stack(self, fn_name: str, *args, **kwargs) -> torch.Tensor:
        outs = [getattr(c, fn_name)(*args, **kwargs) for c in self.components]
        return torch.stack(outs, dim=1)

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """phi_{k,j}(x) for all components k and features j.
        Shape: (M, K, N).
        """
        return self._stack("evaluate", x)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """grad phi_{k,j}(x).  Shape: (M, K, d, N)."""
        return self._stack("gradient", x)

    def laplacian(
        self,
        x: torch.Tensor,
        dims: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Laplacian per component.  Shape: (M, K, N)."""
        return self._stack("laplacian", x, dims=dims)

    def hessian_diag(self, x: torch.Tensor) -> torch.Tensor:
        """Diagonal Hessian per component.  Shape: (M, K, d, N)."""
        return self._stack("hessian_diag", x)

    def derivative(
        self, x: torch.Tensor, alpha: tuple
    ) -> torch.Tensor:
        """Arbitrary mixed partial per component.  Shape: (M, K, N)."""
        return self._stack("derivative", x, alpha)

    # ------------------------------------------------------------------
    # Block-diagonal assembly helpers
    # ------------------------------------------------------------------

    def block_diag_evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Block-diagonal evaluation matrix of shape (K*M, K*N).

        Equivalent to ``torch.block_diag(phi_0(x), phi_1(x), ...)``,
        i.e. row block k uses only component-k features.  Multiply by
        the stacked coefficient vector ``beta_stack`` of shape
        ``(K*N, 1)`` to get the stacked predictions.  Useful when the
        coupled-PDE rows are *independent* per component (no
        cross-coupling).
        """
        return torch.block_diag(*[c.evaluate(x) for c in self.components])

    def block_diag_laplacian(
        self, x: torch.Tensor, dims: Optional[Sequence[int]] = None
    ) -> torch.Tensor:
        """Block-diagonal Laplacian, (K*M, K*N)."""
        return torch.block_diag(
            *[c.laplacian(x, dims=dims) for c in self.components]
        )

    def block_diag_derivative(
        self, x: torch.Tensor, alpha: tuple
    ) -> torch.Tensor:
        """Block-diagonal D^alpha, (K*M, K*N)."""
        return torch.block_diag(
            *[c.derivative(x, alpha) for c in self.components]
        )

    # ------------------------------------------------------------------
    # Coefficient packing utilities
    # ------------------------------------------------------------------

    def stack_betas(self, betas: Sequence[torch.Tensor]) -> torch.Tensor:
        """Stack per-component betas (each (N_k, 1)) into a single
        column of shape (sum_k N_k, 1)."""
        if len(betas) != self.n_components:
            raise ValueError(
                f"got {len(betas)} betas for {self.n_components} components"
            )
        return torch.cat([b.reshape(-1, 1) for b in betas], dim=0)

    def unstack_beta(
        self, beta: torch.Tensor
    ) -> list[torch.Tensor]:
        """Split a stacked coefficient vector back into per-component pieces."""
        sizes = [c.n_features for c in self.components]
        if beta.numel() != sum(sizes):
            raise ValueError(
                f"beta has {beta.numel()} entries; expected {sum(sizes)}"
            )
        flat = beta.reshape(-1)
        out = []
        offset = 0
        for n in sizes:
            out.append(flat[offset : offset + n].reshape(-1, 1))
            offset += n
        return out

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        x: torch.Tensor,
        betas: Union[Sequence[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate the multi-component prediction u(x) of shape (M, K).

        `betas` may be a list of per-component (N_k, 1) tensors, a
        stacked (sum N_k, 1) column, or a (N_per, K) matrix when all
        components share n_features.
        """
        # normalise to list of per-component (N_k, 1)
        if isinstance(betas, (list, tuple)):
            beta_list = list(betas)
        else:
            b = torch.as_tensor(betas)
            if b.ndim == 2 and b.shape[1] == self.n_components and \
               b.shape[0] == self.components[0].n_features:
                beta_list = [b[:, k : k + 1] for k in range(self.n_components)]
            else:
                beta_list = self.unstack_beta(b)
        outs = [c.evaluate(x) @ bk for c, bk in zip(self.components, beta_list)]
        return torch.cat(outs, dim=1)


# ======================================================================
# VectorFastLSQSolver -- bundles K scalar FastLSQ solvers
# ======================================================================


class VectorFastLSQSolver:
    """K-component analogue of :class:`FastLSQSolver`.

    Adds the same set of blocks to every component (so the K bases
    share an architecture but each gets independent random draws --
    the per-component bandwidth can still be tuned by passing a list
    of scales to :meth:`add_block`).

    The user-visible coefficient ``beta`` may be stored as either:

    * a list of K per-component vectors ``[(N, 1), ...]``,
    * a single stacked column ``(K*N, 1)``,
    * a matrix ``(N, K)`` -- one column per component.

    :meth:`predict` accepts any of these shapes.
    """

    def __init__(
        self, input_dim: int, n_components: int, normalize: bool = False
    ):
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        self.input_dim = input_dim
        self.normalize = normalize
        self._n_components = n_components
        self._solvers: list[FastLSQSolver] = [
            FastLSQSolver(input_dim, normalize=normalize)
            for _ in range(n_components)
        ]
        self._vector_basis: Optional[VectorBasis] = None
        self.beta = None  # populated by the user after solving

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def add_block(
        self,
        hidden_size: int = 500,
        scale: Union[float, Sequence[float]] = 1.0,
    ):
        """Append a feature block to every component.

        If `scale` is a list/tuple of length ``n_components``, each
        component is scaled independently.  Otherwise the same scale
        is shared.
        """
        if isinstance(scale, (list, tuple)) and \
           len(scale) == self._n_components and \
           not isinstance(scale[0], (list, tuple, np.ndarray)):
            scales = list(scale)
        else:
            scales = [scale] * self._n_components
        for s, sc in zip(self._solvers, scales):
            s.add_block(hidden_size=hidden_size, scale=sc)
        self._vector_basis = None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def n_components(self) -> int:
        return self._n_components

    @property
    def n_features_per_component(self) -> int:
        return self._solvers[0].n_features

    @property
    def n_features_total(self) -> int:
        return self.n_features_per_component * self._n_components

    @property
    def basis(self) -> VectorBasis:
        """The :class:`VectorBasis` bundling all component bases."""
        if self._vector_basis is None:
            self._vector_basis = VectorBasis([s.basis for s in self._solvers])
        return self._vector_basis

    def component_solver(self, k: int) -> FastLSQSolver:
        """The underlying scalar solver for component k.  Useful when
        you want to call ``predict_with_grad`` or other scalar-only
        methods on one component in isolation."""
        return self._solvers[k]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """(M, K) -- per-component predictions."""
        if self.beta is None:
            raise RuntimeError(
                "solver.beta is not set; assemble + solve first"
            )
        return self.basis.predict(x, self.beta)
