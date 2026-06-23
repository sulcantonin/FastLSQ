# Changelog

All notable changes to FastLSQ will be documented in this file.

## [0.4.1] - 2026-06-23

### Added

- **Solve-time diagnostics — `solve_lstsq(..., return_info=True)`.** Returns
  `(x, info)` with `info = {t_solve, rank_used, residual, cond_estimate}`. `t_solve`
  is the **device-synced wall-time of the solve step alone** (singular values for
  the rank/cond diagnostics are computed *outside* the timed region), `rank_used` is
  the rank-revealing effective numerical rank, and `cond_estimate` is `s_max/s_min`
  over the retained subspace. The default `return_info=False` path is unchanged.
- **Phased breakdown in `solve_linear` metrics.** `metrics` now reports
  `scale_search_s`, `assemble_s`, `solve_s` (plus `rank_used`, `residual`,
  `cond_estimate`) so the headline time is no longer dominated by the auto-scale
  search — the reported solve time is a number the library hands you, not one a
  benchmark reconstructs by hand.
- **`fastlsq.benchmark.time_solve(fn, reps, warmup, device)`** — a device-correct
  timing primitive (`synchronize` bracketing + warm-up + min-of-reps) returning the
  reproducible solve-time *floor* in seconds (`return_all=True` for full stats). Also
  exposes `fastlsq.benchmark.synchronize`.
- **`SinusoidalBasis.random_covariance(d, N, Sigma=… | L=…)`** — fixed full-`Sigma`
  constructor (`W = L @ W_hat`, `Sigma = L Lᵀ`), the symmetric counterpart to
  `random` / `random_anisotropic`.
- **`LearnableFastLSQ.freeze() -> SinusoidalBasis`** — freezes the learned bandwidth
  into a plain, detached basis for one clean, timed one-shot deployment solve (the
  `.basis` property still reconstructs-with-grad each access).
- **Checkpoint provenance.** `save_checkpoint` auto-records a `provenance` block
  (library version, device, dtype, timestamp, and the realized scale / `Sigma`) into
  the metadata.

## [0.4.0] - 2026-06-22

### Added

- **Closed-form projection (Radon) operator for windowed bases.** The integral
  operator class now covers the *projection / Radon* (line/hyperplane-integral)
  family, not just the single-axis integrals of 0.3.0. A tomographic measurement is
  a projection onto a generally **non-axis-aligned** hyperplane,
  `p(u) = ∫ f(z) δ(c·z − u) dz` (a Fredholm equation of the first kind) — beam
  phase-space tomography, CT, Abel inversion — which the single-axis `IntegralOperator`
  cannot express.
- **`GaussianWindowedBasis`** — the windowed-Fourier (Gabor) member of the basis
  family, `ψ_j(z) = exp(−‖ζ‖²/2)·sin(W_j·ζ + b_j)` with `ζ = T⁻¹(z − mean)`. The
  Gaussian window is a **fixed prior** (set once from the data's second moments via
  `GaussianWindowedBasis.from_data`, *not* trained): the projection of a bare unbounded
  sinusoid over an infinite hyperplane diverges, and the window makes the hyperplane
  integral integrable and analytic. Coefficients stay linear, so a fit is still one
  linear least squares. The Gaussian envelope changes the derivative algebra, so this
  class is deliberately scoped to **value** (`evaluate`) and **projection** — it does
  not claim the full `DiffOperator` calculus of the bare `SinusoidalBasis`.
- **`ProjectionOperator`** — assembles the `(M, N)` projection design matrix in
  **closed form, with no quadrature**: in the whitened frame, with `q = Tᵀc`,
  `σ_u² = ‖q‖²`, `u₀ = c·mean`, `jac = |det T|/‖q‖`,
  `(P ψ_j)(u) = jac·(2π)^((d−1)/2)·exp(−‖ω_j‖²/2)·exp(−(u−u₀)²/(2σ_u²))·sin(α_j u + φ_j)`,
  with `α_j = (W_j·q)/‖q‖²`, `‖ω_j‖² = ‖W_j‖² − (W_j·q̂)²`, `φ_j = b_j − α_j u₀`. The
  rows are **differentiable in the direction `c` (the optics)**, for differentiable
  experiment design (autodiff `d(posterior)/d(optics)`): the across-slice energy uses
  the rotation-invariant `‖W‖²−(W·q̂)²` (no QR complement) and every quantity stays a
  tensor (no `float()`/`.item()` casts), so autograd flows to `c` (verified
  `autodiff == finite-difference` to ~5e-9). `from_transport(M, e)` builds the
  tomography convention `c = Mᵀe` (transport by optics `M`, read on axis `e`). The
  operator mirrors the `apply(basis, x, cache)` signature but is standalone (it needs
  the windowed basis, so it does not compose into `IntegroDifferentialOperator`). Both
  classes are exported from `fastlsq`.
- **Tests / example.** `tests/test_projection.py` asserts the closed form equals a
  Gauss--Hermite quadrature of the slice integral to machine precision in d = 2, 3, 4
  (≤4e-13 observed), that autodiff of the rows wrt `c` matches finite differences, and
  that a windowed field is recovered from its projections at several directions in one
  LSQ. `examples/inverse/tomography_projection.py` demonstrates the full reconstruction
  plus the differentiable-optics gradient.

### Scope (honest)

- The closed-form projection assembly works **only** for the Gaussian-windowed basis
  (the Gaussian × plane-wave hyperplane integral is analytic); other windows
  (compact / polynomial) generally are **not** closed form. Scope = *Gaussian-windowed*
  tomographic / line-integral operators, not "any projection".
- The window is a *fixed prior* (set from data moments), required for convergence — not
  a tuned hyperparameter.
- This is a **different** analytic-kernel mechanism from the Fourier-symbol
  (convolution / fractional) class — it is the projection / Radon (line/hyperplane
  integral) class.
- No novelty is claimed over ELM / RBF-for-integral-equations prior art; the distinctive
  parts are quadrature-free closed-form projection rows, differentiability in the optics,
  and the unified operator algebra.

## [0.3.0] - 2026-06-21

### Added

- **Exact closed-form integral & integro-differential operators.** The cyclic
  derivative identity `D^α sin(W·x+b) = (∏ W^{α_k}) Φ_{|α| mod 4}` runs *backwards* --
  integration is differentiation of *negative* order, `∫ sin(wx+b) dx = −(1/w)cos(wx+b)`.
  `SinusoidalBasis.derivative` now accepts **signed** multi-indices (negative entries =
  indefinite integration), with a DC guard (`dc_eps`) that zeros features whose frequency
  along an integrated axis is ~0 (their primitive is a ramp that leaves the sinusoidal
  family). `DiffOperator.antiderivative(dim, order, d)` exposes this as a factory, so
  `Op.partial(0,1,1) - k*Op.antiderivative(0,1,1)` composes integro-differential operators.
- **`IntegralOperator`** -- definite and running (Volterra) integrals with limits,
  via the new `SinusoidalBasis.definite_integral`. Evaluated with a numerically stable
  `sinc` identity (no `1/w` division), so the running integral of a near-DC feature is
  exact rather than singular. Factories `IntegralOperator.volterra(...)` /
  `IntegralOperator.definite(...)`.
- **`IntegroDifferentialOperator`** -- the common roof under which differential and
  integral terms compose (`+`, `−`, scalar/`nn.Parameter` `*`) into one `(M, N)`
  linear-least-squares design matrix; coefficients stay differentiable so learnable
  integral-term coefficients train through the solve. All three exported from `fastlsq`.
- **Examples** `examples/integro_differential_demo.py` (one-shot forward solve of
  `u'(x) + ∫_0^x u ds = f`, rel-L2 ~3e-11) and
  `examples/inverse/inverse_memory_kernel.py` (recover an unknown memory strength λ from
  noisy data via AdamW through the differentiable solve). New `tests/test_integral.py`
  asserts ∫-then-∂ round-trips to identity, Volterra/definite match quadrature, the DC
  guard stays finite, and gradients flow to learnable coefficients.

## [0.2.6] - 2026-06-09

### Changed

- **Faster `_auto_solve` fallback for ill-conditioned systems.** When the
  Cholesky path fails on CPU with no ridge (`mu = 0`), `_auto_solve` now goes
  straight to the rank-deficient-safe SVD solve (LAPACK `gelsd`) instead of
  first attempting Householder QR with the blow-up guard -- on CPU `gelsd` is
  faster than QR, so the detour only added a full extra factorization. The
  QR-then-SVD path is unchanged for ridge solves and non-CPU devices.

## [0.2.5] - 2026-06-04

### Fixed

- **`Wave2D_MS` solves via `solve_linear`.** The long-time anisotropic wave
  returned relative value error 1.0 in every configuration because its
  `t_max = 100` time normalisation packed ~87 temporal cycles into `tau ∈ [0,1]`:
  the PDE's second time-derivative amplifies the random-feature *representation*
  error by `Omega²` (`Omega = pi·sqrt(1+a2)·t_max`), so the one-shot
  least-squares collocation cannot resolve the oscillation -- even 8000 features
  with near-hard boundary constraints stay at rel-err 1.0, because the best
  representable solution itself carries a huge PDE residual. Reducing `t_max` to
  `4` (~3.5 cycles) and matching the anisotropic temporal feature bandwidth to
  `Omega` (`scale_multipliers = [1, 1, 7]`) recovers the solution to ~3e-4 at
  900 features (`scale = 3`); the exactly-consistent `t_max²`-scaled operator is
  unchanged. Added to the `tests/test_benchmarks_inverse.py` linear smoke test.
  Resolves the `Wave2D_MS` [0.2.4] known issue.
- **`ElasticWave2D` solves via the block-stacked vector path.** The coupled
  2-output elastic-wave problem now declares `n_outputs = 2`, assembles its
  operator in block-stacked form (`A ∈ ℝ^{Mk×Nk}`, `b ∈ ℝ^{Mk×1}`) via
  `block_concat`, and gains the `exact_grad` Jacobian (shape `(M, d, k)`, time
  axis chain-ruled by `t_max`) that the error metric requires. `unpack_beta` now
  recovers a `(N, 2)` `beta`, so `solve_linear(ElasticWave2D(), scale=5.0)`
  recovers both components (relative value error ~7e-3 at the default
  resolution) instead of failing to unpack the vector solution. Added to the
  `tests/test_benchmarks_inverse.py` linear smoke test. Resolves the
  `ElasticWave2D` [0.2.4] known issue; the `t_max²` operator scaling from
  [0.2.2] (consistent with `Wave2D_MS`) is preserved.

## [0.2.4] - 2026-06-04

### Added

- **Benchmark + inverse-problem test suite** (`tests/test_benchmarks_inverse.py`):
  12 deterministic smoke tests (~11 s) that solve the linear (`PoissonND`,
  `HeatND`, `Wave1D`, `Helmholtz2D`, `Maxwell2D_TM`) and nonlinear
  (`NLPoisson2D`, `Bratu2D`, `SteadyBurgers1D`, `NLHelmholtz2D`, `AllenCahn1D`)
  benchmark equations through the public `solve_linear` / `solve_nonlinear` API,
  plus two inverse pipelines -- Gaussian source-position recovery (forward solve
  + L-BFGS) and SINDy-style PDE discovery via analytical derivatives --
  exercising the 0.2.3 QR / N-scaled-collocation solver path end to end.

### Known issues

- `Wave2D_MS` does not solve via `solve_linear` (relative error 1.0 in every
  configuration tested) -- a pre-existing problem-definition gap, independent of
  the solver work, excluded from the new smoke test pending a fix. *(Fixed in
  [0.2.5]: `t_max` reduced 100 -> 4 so the normalised-time oscillation
  (~3.5 vs ~87 cycles) is resolvable; now covered by the smoke test.)*
- `ElasticWave2D` -- a 2-output vector problem whose `exact()` returns `(N, 2)`
  -- never sets `n_outputs`, so the scalar API cannot unpack it; also excluded
  here. *(Fixed in [0.2.5]: it now uses the block-stacked vector path and
  is covered by the smoke test.)*

## [0.2.3] - 2026-06-04

### Added

- **Householder-QR least-squares back-end** `solve_lstsq(..., method="qr")`:
  backward-stable at `cond(A)` (ridge applied via the `[A; sqrt(mu) I]`
  augmentation, not the normal equations), giving SVD-grade accuracy (~1e-14 on
  the Helmholtz random-feature benchmark) at QR cost -- and, on the
  rank-deficient CPU/no-ridge path, faster than the `gelsd` `"svd"` driver too,
  while far more accurate than the normal-equations `"cholesky"` (no `cond(A)`
  squaring, no required ridge). Assumes the system is numerically full column
  rank; `"svd"` remains the rank-deficient-safe reference.
- **`solve_linear(..., method=...)`**: the linear solve back-end is now
  selectable from the high-level API (`"auto"`, `"qr"`, `"svd"`, `"cholesky"`,
  `"rsvd"`; defaults to `"auto"`).

### Changed

- **`method="auto"` now tries QR before SVD.** After the Cholesky conditioning
  probe rejects the fast path, `auto` uses the faster, more accurate QR solve and
  falls back to the rank-revealing SVD only when QR's solution blows up
  (`||x|| / (1 + ||b||)` above a generous guard). Real PDE systems measure
  `<= 0.3` and keep QR; genuinely rank-deficient *inconsistent* systems (e.g. a
  random RHS) measure ~3e14 and route to SVD. Net: the default solve is faster
  and at least as accurate on real problems, with minimum-norm SVD preserved
  exactly where it is needed.
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
