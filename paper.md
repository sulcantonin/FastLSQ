---
title: 'FastLSQ: A Python framework for one-shot PDE solving via Fourier features with exact analytical derivatives'
tags:
  - Python
  - PyTorch
  - partial differential equations
  - integral equations
  - Fourier features
  - least squares
  - scientific machine learning
  - physics-informed machine learning
authors:
  - name: Antonin Sulc
    orcid: 0000-0001-7767-778X
    corresponding: true
    affiliation: 1
affiliations:
  - name: Lawrence Berkeley National Laboratory, Berkeley, CA, United States
    index: 1
    ror: 02jbv0t02
date: 29 August 2026
bibliography: paper.bib
---

# Summary

Partial differential equations (PDEs) and integral equations underpin
quantitative models across physics and engineering. `FastLSQ` is a Python
framework that solves them by representing the unknown solution in a basis of
random sinusoidal (Fourier) features whose derivatives *and* integrals are
available in closed form. Because every feature is a plane wave, a linear
differential, integral, integro-differential, or nonlocal operator maps the
basis to another explicit matrix, and a linear PDE reduces to a **single
least-squares solve** — with no mesh, no automatic differentiation, and no
iterative network training. Nonlinear problems are handled by Newton–Raphson
iteration over the same assembly. The random-feature system is typically
rank-deficient, so the solve is routed through an automatically selected,
backward-stable least-squares back-end (a Cholesky fast-path, Householder QR,
and a rank-revealing SVD fallback) that runs on CPU, CUDA, or Apple-MPS via
`PyTorch` [@paszke2019pytorch]. The framework unifies differential operators,
single- and multi-axis integrals, Fredholm/Volterra integral equations,
Fourier-multiplier (fractional/nonlocal) operators, meshless signed-distance
geometry, and vector-valued systems behind one consistent, composable API, and
because coefficients may be `PyTorch` parameters, PDE coefficients and basis
bandwidth can be learned by gradient descent for inverse problems.

# Statement of need

Classical discretisations (finite elements, finite differences, finite volumes)
require a mesh, which becomes costly to generate for complex geometries and
scales poorly with dimension. Physics-informed neural networks (PINNs)
[@raissi2019pinn; @lu2021deepxde] remove the mesh and are highly flexible, but
they recast solving into a nonconvex optimisation that is trained with
gradient descent and automatic differentiation, which is comparatively slow and
can be difficult to converge. Random-feature and extreme-learning-machine
approaches [@huang2006elm; @dwivedi2020pielm; @chen2022rfm] recover speed and
convexity by *fixing* the features and solving a linear system, but existing
implementations tend to target a narrow set of operators and expose few of the
numerical safeguards that the resulting (severely rank-deficient) least-squares
problems demand.

`FastLSQ` addresses this gap. Its central abstraction, `SinusoidalBasis`,
computes any mixed partial derivative of any order in closed form and in `O(1)`
per term via a cyclic identity, so no computational graph or automatic
differentiation is built. The same identity runs backwards — integration is
differentiation of negative order — which closes the calculus in both
directions and lets differential, integral, and integro-differential terms be
assembled into one linear-in-coefficients design matrix. A dedicated,
rank-revealing least-squares layer makes the one-shot solve numerically
trustworthy on the near-degenerate feature matrix rather than leaving accuracy
on the floor. On top of this foundation, the package provides Fourier-multiplier
operators for nonlocal problems (e.g. the fractional Laplacian)
[@lischke2020fractional], degenerate-kernel machinery for Fredholm equations of
the second kind [@atkinson1997integral], meshless geometry from signed-distance
functions, first-class vector-valued solutions, and learnable parameters for
inverse problems. `FastLSQ` targets researchers and students in scientific
computing and scientific machine learning who need a fast, meshless, and
extensible tool that spans a broad operator taxonomy through a single
least-squares interface. A companion preprint develops the method and its
evaluation in detail [@sulc2026fastlsq].

# State of the field

Several families of methods solve PDEs without a conforming mesh. PINNs
[@raissi2019pinn] and libraries such as `DeepXDE` [@lu2021deepxde] parameterise
the solution with a neural network and minimise the PDE residual by gradient
descent, relying on automatic differentiation for the differential operators;
they are general and mature but pay the cost of nonconvex training. Extreme
learning machines [@huang2006elm] and their physics-informed variant, PIELM
[@dwivedi2020pielm], fix a single hidden layer of random features and solve the
resulting linear system directly, and the random feature method [@chen2022rfm]
develops this idea into a systematic PDE solver; these share `FastLSQ`'s
one-shot, least-squares philosophy. Random Fourier features [@rahimi2007random]
provide the theoretical footing for the sinusoidal basis, while classical
spectral methods [@trefethen2000spectral] exploit the same exact
differentiation of trigonometric modes on structured grids.

`FastLSQ` differs from these in three ways. First, it treats *exact analytical
calculus* on the basis — arbitrary-order derivatives, definite and running
integrals, and diagonal Fourier multipliers — as the first-class,
operator-agnostic abstraction, so differential, integral, integro-differential,
nonlocal, and integral-equation problems are all assembled and solved through
one code path. Second, it pairs the one-shot solve with an automatically routed,
backward-stable, rank-revealing least-squares back-end [@golub2013matrix] built
for the rank-deficient random-feature matrix, rather than a plain normal-equation
or `lstsq` call. Third, it is meshless by construction: any signed-distance (or
membership-oracle) function yields interior points, boundary points, and outward
normals for Neumann/Robin conditions, with constructive-solid-geometry
composition for non-convex and multiply-connected domains.

# Software design

`FastLSQ` is written in Python on top of `PyTorch` [@paszke2019pytorch] and
follows three design principles: a small set of composable abstractions, a
consistent problem interface, and device/dtype-aware numerics. The foundation is
`SinusoidalBasis`, which evaluates features and their exact derivatives and
caches the shared `sin`/`cos` evaluations across operators. Linear operators are
represented symbolically — `Op` (aliased `DiffOperator`) for differential terms,
`IntegralOperator`/`MultiIntegralOperator` for integrals, `SymbolOperator` for
Fourier multipliers, and `SeparableKernelOperator` for degenerate kernels — and
compose with ordinary `+`, `-`, and scalar `*`, each producing an explicit
design-matrix block through a uniform `apply(basis, x)` method. A problem is any
object that supplies training and test points and a `build` method that stacks
these blocks into a system `A\beta = b`; `solve_linear` and `solve_nonlinear`
then dispatch to the solver and the rank-revealing least-squares layer
(`solve_lstsq`), which selects among Cholesky, QR, SVD, and randomized-SVD
back-ends and transparently falls back to CPU where a device lacks a robust
factorisation. Because operator coefficients and the basis bandwidth may be
`PyTorch` `nn.Parameter` objects, gradients flow through the pre-factored solve,
enabling learnable PDE coefficients and inverse problems with the same
machinery. Vector-valued systems reuse the scalar path through block-assembly
helpers, and the package ships plotting utilities, diagnostics, checkpointing,
and a suite of benchmark problems with closed-form solutions.

The resulting interface stays close to the mathematics: a linear problem is a
single call, and any operator is built by composing primitives.

```python
import torch
from fastlsq import solve_linear, Op
from fastlsq.basis import SinusoidalBasis
from fastlsq.problems.linear import Helmholtz2D

# High-level: solve a linear PDE in a single least-squares call -- no mesh,
# no automatic differentiation, no iterative training
result = solve_linear(Helmholtz2D(), scale=5.0)
u = result["u_fn"]                                    # solution as a callable u(x)

# Low-level: assemble any operator symbolically on the exact-derivative basis
basis = SinusoidalBasis.random(input_dim=2, n_features=1500, sigma=5.0)
x = torch.rand(4000, 2)
k = 10.0
helmholtz = Op.laplacian(d=2) + k**2 * Op.identity(d=2)   # compose with +, -, *
A = helmholtz.apply(basis, x)                             # explicit (M, N) matrix
```

The `examples/` directory and the project README collect runnable tutorials for
each problem class.

# Capabilities

The same `solve_linear`/`solve_nonlinear` interface spans a broad taxonomy of
problems beyond scalar linear PDEs.

**Nonlinear PDEs** are solved by damped Newton–Raphson iteration with Tikhonov
regularisation and homotopy continuation, reusing the exact-derivative assembly
at each step.

**Vector-valued and coupled systems.** Solutions $u:\mathbb{R}^d \to
\mathbb{R}^k$ are first-class. A problem declares $k$ outputs and assembles a
block-stacked operator, so coupled systems — linear elasticity, Stokes flow, the
Maxwell vector potential — are written by stacking their equations as operator
rows and their unknown fields as coefficient blocks. The two-dimensional Stokes
system for $(u, v, p)$, for instance, stacks the momentum balances
$-\Delta u + \partial_x p = f_x$ and $-\Delta v + \partial_y p = f_y$ with the
incompressibility constraint $\partial_x u + \partial_y v = 0$; the helper
`block_concat` assembles the block matrix and `solver.predict(x)` returns all
$k$ components at once. Scalar problems are simply the $k = 1$ case.

**Integral and integro-differential equations.** Because integration is exact on
the basis, a Fredholm equation of the second kind
$u(x) - \lambda\!\int K(x, y)\,u(y)\,dy = f(x)$ is assembled as $I - \lambda K$
and solved in the *same* single least squares — needing no boundary rows, since
the identity term already makes it well posed [@atkinson1997integral]. A
separable (degenerate) kernel $K(x, y) = \sum_m g_m(x)\,h_m(y)$ collapses the
operator to a rank-$R$ product whose inner products are precomputed once, and
built-in diagnostics report the characteristic values $\lambda$ at which the
equation is singular and whether the quadrature resolves the basis. Volterra
(running-integral) equations, multi-axis definite and running integrals
(area/volume functionals and memory terms), and mixed integro-differential
operators compose in the same way.

**Nonlocal operators, geometry, and inverse problems.** Fourier-multiplier
operators treat nonlocal problems such as the fractional Laplacian $(-\Delta)^s$
[@lischke2020fractional] exactly and without quadrature; signed-distance
functions provide meshless collocation on non-convex and multiply-connected
domains; and because operator coefficients and the basis bandwidth may be
learnable parameters, the framework supports PDE-constrained inverse problems.

# Mathematics

For a sinusoidal feature $\varphi_j(x) = \sin(W_j^\top x + b_j)$ with frequency
$W_j \in \mathbb{R}^d$ and phase $b_j$, every mixed partial derivative has the
closed form

$$
D^{\alpha}\varphi_j(x)
= \left(\prod_{k=1}^{d} W_{jk}^{\alpha_k}\right)\,
  \Phi_{|\alpha|\bmod 4}\!\left(W_j^\top x + b_j\right),
$$

where $\alpha$ is a multi-index, $|\alpha| = \sum_k \alpha_k$, and
$(\Phi_0,\Phi_1,\Phi_2,\Phi_3) = (\sin,\cos,-\sin,-\cos)$ cycles with the
derivative order. Writing the solution as $u(x) \approx \sum_j \beta_j
\varphi_j(x)$, any linear operator $\mathcal{L}$ acts term by term,
$(\mathcal{L}u)(x) = \sum_j \beta_j\,(\mathcal{L}\varphi_j)(x)$, so collocating
the PDE and its boundary conditions at sample points assembles a matrix $A$ and
right-hand side $b$, and the solution coefficients solve the least-squares
problem

$$
\beta^\star = \arg\min_{\beta}\; \lVert A\beta - b \rVert_2^2 + \mu\,\lVert \beta \rVert_2^2,
$$

with the Tikhonov ridge entering through the stable augmentation $[A;\,
\sqrt{\mu}\,I]$ rather than the condition-squaring normal equations. Because each
feature is a plane wave, a Fourier multiplier acts diagonally,
$\mathcal{L}\,e^{i\xi\cdot x} = m(\xi)\,e^{i\xi\cdot x}$ — for the fractional
Laplacian $m(\xi) = |\xi|^{2s}$ — so nonlocal operators reduce to an exact,
quadrature-free per-column rescale of the basis.

# Research impact statement

`FastLSQ` is a newly released package, so its significance is stated in terms of
readiness and reproducibility rather than accumulated citations. The method and
its evaluation are developed in a companion preprint [@sulc2026fastlsq], and the
software is openly available under the MIT license on the Python Package Index
(`pip install fastlsq`). It ships with more than twenty benchmark problems —
linear, nonlinear, vector-valued, and integral-equation — that carry closed-form
solutions, so every capability is validated against an exact reference, in
several cases to near machine precision, with reproducible `examples/` scripts
and an automated test suite. By spanning differential, integral, nonlocal, and
integral-equation operators through one meshless least-squares interface that
runs unmodified on CPU, CUDA, and Apple-MPS hardware, `FastLSQ` lowers the
barrier to one-shot, exact-derivative PDE solving in scientific computing and
scientific machine learning. An early presentation of the method was given at
Lawrence Berkeley National Laboratory.

# AI usage disclosure

Generative AI tools, including Claude Code (as reflected in the project's
version-control history), were used to assist with software implementation,
refactoring, and documentation, and with drafting this manuscript. AI-assisted
output was verified by the author against the package's automated test suite and
its benchmark problems with closed-form solutions, whose errors are checked
against exact references; the underlying method, all design decisions, and the
final text are the author's own, and the author takes full responsibility for the
correctness of the software and this manuscript.

# Acknowledgements

<!-- Author action required: JOSS asks that you acknowledge any financial support.
     Replace the bracketed line below with your actual funding (program name and
     grant/contract number) and any individual acknowledgements, then delete this
     comment. Work at Lawrence Berkeley National Laboratory is commonly supported
     by the U.S. Department of Energy, Office of Science, under Contract
     No. DE-AC02-05CH11231. -->
The author acknowledges the support of [funding source and grant/contract number].

# References
