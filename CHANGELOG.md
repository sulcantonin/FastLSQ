# Changelog

All notable changes to FastLSQ will be documented in this file.

## [Unreleased]

### Added

- **Learnable operator coefficients**: `Op` now accepts `nn.Parameter` (and tensors) as coefficients in scalar multiplication. Use `k = nn.Parameter(...)` with `Op.laplacian(d=2) + k**2 * Op.identity(d=2)` and optimise via AdamW; gradients flow through the prebuilt linear solve. See `examples/learnable_helmholtz.py`.

## [0.2.0] - 2026-03-01

### Added

#### SinusoidalBasis -- analytical derivative engine (new foundation)
- `SinusoidalBasis` class: evaluates arbitrary-order mixed partial derivatives
  of sinusoidal features in O(1) via the cyclic derivative identity
- `BasisCache`: pre-computes sin(Z)/cos(Z) once, reuses across all derivative
  evaluations at the same points
- `DiffOperator` / `Op`: symbolic linear differential operators that compose
  via `+`, `-`, scalar `*`.  Factory methods: `Op.laplacian()`,
  `Op.partial()`, `Op.identity()`, `Op.biharmonic()`, `Op.gradient_component()`
- `FeatureBasis`: adapter wrapping non-sinusoidal solvers (e.g. PIELMSolver
  with tanh) into the same basis interface

#### Learnable bandwidth
- `LearnableFastLSQ`: PyTorch `nn.Module` with learnable bandwidth via
  reparameterisation trick (scalar, diagonal, or full Cholesky modes)
- `train_bandwidth()`: hybrid training loop (inner exact solve + outer AdamW
  on bandwidth parameters)

#### PDE discovery example
- `examples/pde_discovery.py`: sparse regression (SINDy-style) using
  analytical derivatives from `SinusoidalBasis` as the dictionary

### Changed

#### Architecture: basis-centric API (breaking)
- **`solver.basis`** is now the single entry point for all feature and
  derivative computations.  Every solver exposes a `.basis` property
  (`SinusoidalBasis` for FastLSQ, `FeatureBasis` for PIELM).
- All problem classes (`linear.py`, `nonlinear.py`, `regression.py`) rewritten
  to use `solver.basis.evaluate()`, `.gradient()`, `.hessian_diag()`,
  `.laplacian()`, and `.cache()` instead of tuple unpacking.
- `FastLSQSolver.predict()`, `.predict_with_grad()`, `.predict_with_laplacian()`
  delegate directly to `self.basis`.
- `PIELMSolver` now exposes `.basis` (via `FeatureBasis` adapter) so problem
  classes work identically for both solver types.
- `newton.py`: uses `solver.basis.evaluate()` for convergence metrics.

#### Examples rewritten
- `add_your_own_pde.py`: shows `Op`-based PDE definition only
- `custom_features.py`: shows cosine features via phase shift only

### Removed
- `get_features()` method removed from public API (all solvers, all problems)
- `SinusoidalBasis.get_features()` legacy tuple interface removed
- `CustomFeatureSolver` subclass pattern removed
- Legacy `build()` patterns using `(H, dH, ddH)` tuple unpacking

### Fixed
- Missing `sample_ball` import in `tests/test_basic.py`


## [0.1.0] - 2026-02-12

### Added

#### High-level API
- `solve_linear()` - One-line function to solve linear PDEs
- `solve_nonlinear()` - One-line function to solve nonlinear PDEs via Newton-Raphson
- Automatic scale selection via `auto_select_scale()`

#### Plotting & Visualization
- `plot_solution_1d()` - Plot 1D solutions with exact comparison
- `plot_solution_2d_slice()` - Plot 2D solutions as 1D slices
- `plot_solution_2d_contour()` - Contour plots for 2D solutions
- `plot_convergence()` - Newton iteration convergence plots
- `plot_spectral_sensitivity()` - Error vs scale analysis

#### Geometry & Sampling
- `sample_box()` - Uniform sampling from hypercubes
- `sample_ball()` - Uniform sampling from balls
- `sample_sphere()` - Uniform sampling from sphere surfaces
- `sample_interval()` - 1D interval sampling
- `sample_boundary_box()` - Boundary point generation for boxes
- `get_sampler()` - Get sampler by name

#### Diagnostics & Error Handling
- `check_problem()` - Validate problem definitions (shapes, gradients, data)
- `check_solver_conditioning()` - Check linear system conditioning
- `suggest_scale()` - Heuristic scale suggestions

#### Export & Interoperability
- `to_numpy()` - Convert predictions to NumPy arrays
- `to_dict()` / `from_dict()` - Serialize/deserialize solver state
- `save_checkpoint()` / `load_checkpoint()` - Save/load solver checkpoints
- `FastLSQModule` - PyTorch Lightning integration (optional)

#### Documentation & Examples
- `examples/tutorial_basic.py` - Basic linear PDE tutorial
- `examples/tutorial_nonlinear.py` - Nonlinear PDE tutorial
- `examples/add_your_own_pde.py` - Guide for adding custom PDEs
- `examples/custom_features.py` - Extensibility example
- Comprehensive README with usage examples

#### Infrastructure
- `benchmarks/run_all.py` - Reproducible benchmark suite
- `tests/test_basic.py` - Basic test suite
- `.github/workflows/ci.yml` - GitHub Actions CI/CD
- `CHANGELOG.md` - This file

### Changed

- Restructured codebase from 2 monolithic files into a proper Python package
- Unified `FastLSQSolver` with optional normalization parameter
- Removed code duplication (Bratu, NL-Helmholtz regression inherit from nonlinear versions)
- Removed non-scientific content (emojis, casual language)

### Fixed

- Fixed missing `solve_lstsq` import in linear solver (was only defined in nonlinear file)
- Improved error messages and diagnostics
