# Changelog

All notable changes to FastLSQ will be documented in this file.

## [0.2.3] - 2026-06-04

### Added

- **Householder-QR least-squares back-end** `solve_lstsq(..., method="qr")`:
  backward-stable at `cond(A)` (ridge applied via the `[A; sqrt(mu) I]`
  augmentation, not the normal equations), giving SVD-grade accuracy (~1e-14 on
  the Helmholtz random-feature benchmark) at QR cost -- cheaper than `"svd"` and
  far more accurate than the normal-equations `"cholesky"` (no `cond(A)`
  squaring, no required ridge). Assumes the system is numerically full column
  rank; use `"svd"` for a rank-deficient `A`. The default `"auto"` keeps its
  rank-revealing SVD fallback (these feature matrices are typically
  rank-deficient), so QR is opt-in via `method="qr"`.
- **`solve_linear(..., method=...)`**: the linear solve back-end is now
  selectable from the high-level API (`"auto"`, `"qr"`, `"svd"`, `"cholesky"`,
  `"rsvd"`; defaults to `"auto"`).

### Changed

- **N-scaled collocation defaults.** `solve_linear` and `solve_nonlinear` now
  default `n_pde`/`n_bc` to `None` and derive them from the feature count
  (`n_pde = max(3000, 3 * n_blocks * hidden_size)`, `n_bc = max(800, n_pde // 5)`),
  replacing the fixed `10000`/`2000` (and `5000`/`1000`) over-sampling that was
  ~6x the default feature count. Faster for the default configuration; passing
  explicit `n_pde`/`n_bc` still overrides.

## [0.2.2] - 2026-06-03

### Fixed

- **Learnable bandwidth now trains.** `LearnableFastLSQ.solve_inner` replaced the
  backprop-through-`torch.linalg.svd` inner solve (which returned NaN gradients
  w.r.t. the bandwidth on the clustered singular values of random-feature
  matrices) with the SVD-based `gelsd` rank-revealing least-squares driver, so
  `train_bandwidth` / `fit` no longer stall at step 0.
- **Default-solve accuracy.** Tightened the `_auto_solve` Cholesky-acceptance
  probe from `rcond**0.5` to `rcond**0.25`, so `method="auto"` falls back to SVD
  before the normal-equations Cholesky loses half its float64 digits
  (cond(A) ~ 1e7 previously returned a ~1e-3-accurate answer).
- **Newton convergence and robustness.** The stop test now combines a *relative*
  residual criterion (`res_norm < tol_res * R0`) with the relative solution
  change (`||Δu||/||u|| < tol_du`); the previous unreachable absolute residual
  tolerance forced every nonlinear solve to run the full `max_iter`. The
  backtracking line search keeps the previous iterate when no step satisfies
  Armijo instead of committing a worse point. `solve_nonlinear` default
  tolerances loosened to `tol_res=1e-8`, `tol_du=1e-10`.
- **Continuation guard.** `solve_nonlinear` no longer raises `TypeError` when a
  problem sets `use_continuation=True` without a `nu_target`.
- **Regression problems solvable via the public API.** Their `get_train_data`
  now accepts the `n_pde`/`n_bc` signature used by `solve_linear`,
  `auto_select_scale`, and `check_problem` (was `n_samples`, raising
  `TypeError`); `auto_select_scale` now raises when every trial fails instead of
  silently returning the first scale.
- **Float32 inputs.** `SinusoidalBasis.cache` promotes inputs to the basis
  dtype/device, so float32 collocation points no longer raise `float != double`.
- **Checkpoint reload.** `load_checkpoint` passes `weights_only=False`, fixing
  `UnpicklingError` on torch >= 2.6 (checkpoints store NumPy arrays).
- **Vector per-component scale.** `VectorFastLSQSolver.add_block` accepts a NumPy
  array of per-component bandwidths (previously list/tuple only, silently
  misread as per-dimension).
- **ElasticWave2D operator.** Scaled the spatial and cross terms by `t_max²`
  (time normalisation), consistent with `Wave2D_MS`.

### Changed

- Problem modules (`nonlinear.py`, `regression.py`) resolve the device via the
  live `get_device()` rather than an import-time snapshot.
- Packaging: the source distribution no longer ships the `misc/` images (the
  sdist was ~14 MB); project URLs point to `github.com/sulcantonin/FastLSQ`;
  README images use absolute URLs so they render on PyPI.
  `examples/orbit_hill.py` solves via rank-revealing `lstsq` rather than a
  normal-equations Cholesky.

## [0.2.1] - 2026-06-02

### Added

- **Device abstraction** (`fastlsq/device.py`): `resolve_device`, `set_device`,
  `get_device`, `device_info` for CPU / CUDA / Apple-MPS. dtype-aware -- MPS is
  auto-selected only for float32 (it has no float64), so the default float64
  high-accuracy regime stays on CPU/CUDA. Override with `set_device(...)` or the
  `FASTLSQ_DEVICE` environment variable; internal tensor creation respects the
  active device at call time.
- **Pluggable linear solver** `solve_lstsq(..., method=...)`:
  - `"svd"` -- rank-revealing truncated SVD (LAPACK `gelsd` fast path on CPU);
  - `"cholesky"` -- fast normal-equations solve for well-conditioned systems;
  - `"rsvd"` -- torch-native randomized SVD (`O(MNk)`) for strongly low-rank `A`;
  - `"auto"` (default) -- Cholesky with a cheap conditioning probe, falling back
    to SVD when ill-conditioned (recovers the fast path without losing accuracy).
  MPS factorizations run on CPU (no robust `svd`/`lstsq` there) and move back.
- **Working anisotropic Sigma = L Lᵀ learner**: `LearnableFastLSQ` (diagonal &
  cholesky modes) now converges -- `solve_inner` uses a differentiable
  *rank-revealing* solve, the Cholesky factor is log-parameterized (clamped,
  positive-definite), and `train_bandwidth` is robust (gradient clipping,
  best-iterate restore, graceful SVD/gradient-failure handling). Chainable
  `LearnableFastLSQ.fit(problem, ...)` for one-line learn-then-predict.
- **Vector-valued solutions (`u: ℝᵈ → ℝᵏ`)**: first-class support for coupled and
  decoupled multi-output PDEs via the new `fastlsq.block` module. Problems opt in
  with `self.n_outputs = k`; `solver.beta` is `(N, k)` and `solver.predict(x)`
  returns `(M, k)` (scalar `k=1` is bit-for-bit unchanged). `block_concat`
  assembles a nested list of blocks (`None` = zero block); `pack_beta` /
  `unpack_beta` convert between `(N, k)` and the block-stacked `(N*k, 1)` solve.
  The block-stacked LSQ is solved by the rank-revealing solver, and the
  Σ-learner computes its loss on the flat `_beta_flat` so it stays correct for
  `k>1`.
- **Learnable operator coefficients**: `Op` accepts `nn.Parameter` (and tensors)
  as coefficients, e.g. `Op.laplacian(d=2) + k**2 * Op.identity(d=2)` with
  `k = nn.Parameter(...)`; gradients flow through the prebuilt linear solve.

### Changed

- `solve_lstsq` defaults to the rank-revealing / `auto` solve instead of forming
  the normal equations -- several orders of magnitude more accurate on the
  rank-deficient random-feature systems (at a higher, still one-shot, cost).
- `solve_linear` is one-shot: the bandwidth-learning hyper-parameters live on
  `LearnableFastLSQ.fit()` / `train_bandwidth`, not the solve signature.
- Packaging: `requires-python` lowered to `>=3.9` (+3.9 classifier); description
  updated.

### Fixed

- The previously dead `LearnableFastLSQ(mode="cholesky")` path, which diverged
  from an unstable inner `torch.linalg.lstsq` and an unconstrained `L`.

## [0.1.5] - 2026-05-25

### Added

- **Vector-valued features**: new `VectorBasis` and `VectorFastLSQSolver`
  (`fastlsq/vector.py`) for solving coupled systems where the unknown is a
  vector field `u(x) = (u_1, ..., u_K)`.  Use cases include
  streamfunction-vorticity NS `(psi, omega)`, incompressible NS primitive
  variables `(u, v, p)`, multi-species transport, MHD, etc.

  - `VectorBasis.random(input_dim, n_features, sigma, n_components, sigmas=...)`
    creates K independent random-Fourier `SinusoidalBasis`es; per-component
    bandwidth can be tuned via `sigmas`.
  - Stacked evaluators: `evaluate(x) -> (M, K, N)`, `gradient -> (M, K, d, N)`,
    `laplacian`, `hessian_diag`, `derivative(alpha)`.
  - Block-diagonal assembly helpers (`block_diag_evaluate`,
    `block_diag_laplacian`, `block_diag_derivative`) for systems whose
    rows are independent per component.
  - Coefficient packing utilities (`stack_betas`, `unstack_beta`,
    `predict`) accept per-component lists, stacked columns, or
    `(N, K)` matrices interchangeably.
  - `VectorFastLSQSolver(input_dim, n_components, normalize=True)` is the
    multi-component counterpart of `FastLSQSolver`; `add_block(scale=...)`
    accepts either a scalar or a list of K scalars for per-component
    bandwidth.
  - `component(k)` / `component_solver(k)` give direct access to the k-th
    scalar basis / solver for ad-hoc per-component work.

### Other

- Synchronised `pyproject.toml`, `fastlsq.__version__`, and CHANGELOG (the
  package source had drifted to `__version__ = "0.1.0"` against
  `pyproject.toml = "0.1.4"`; both now read `"0.1.5"`).

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
