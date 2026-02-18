# Changelog

All notable changes to FastLSQ will be documented in this file.

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
