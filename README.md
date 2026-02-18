# FastLSQ

<p align="center">
  <img src="misc/fastlsq_teaser.png" alt="FastLSQ method overview" width="800"/>
</p>

**Solving PDEs in one shot via Fourier features with exact analytical derivatives.**

FastLSQ is a lightweight PDE solver that uses Random Fourier Features with
`sin` activation and closed-form first- and second-order derivatives.
Linear PDEs are solved in a single least-squares step; nonlinear PDEs are
solved via Newton-Raphson iteration with Tikhonov regularisation,
1/sqrt(N) feature normalisation, and continuation/homotopy.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Or simply install the dependencies manually:

```bash
pip install torch numpy matplotlib
```

## Quick start

### Solve a linear PDE in one line

```python
from fastlsq import solve_linear
from fastlsq.problems.linear import PoissonND

problem = PoissonND()
result = solve_linear(problem, scale=5.0)  # Auto-selects scale if None

u_fn = result["u_fn"]
print(f"Value error: {result['metrics']['val_err']:.2e}")
```

### Solve a nonlinear PDE

```python
from fastlsq import solve_nonlinear
from fastlsq.problems.nonlinear import NLPoisson2D

problem = NLPoisson2D()
result = solve_nonlinear(problem, max_iter=30)

print(f"Converged in {result['n_iters']} iterations")
print(f"Value error: {result['metrics']['val_err']:.2e}")
```

### Plot solutions

```python
from fastlsq.plotting import plot_solution_2d_contour, plot_convergence

# Plot 2D solution
plot_solution_2d_contour(result["solver"], problem, save_path="solution.png")

# Plot Newton convergence
plot_convergence(result["history"], problem_name=problem.name, save_path="convergence.png")
```

### Check your problem definition

```python
from fastlsq.diagnostics import check_problem

check_problem(problem)  # Validates shapes, gradients, data consistency
```

### Benchmarks

```bash
# Linear PDE benchmark (Fast-LSQ vs PIELM)
python examples/run_linear.py

# Nonlinear PDE benchmark (Newton-Raphson)
python examples/run_nonlinear.py
```

## Features

- **High-level API**: Solve PDEs in one line with `solve_linear()` and `solve_nonlinear()`
- **Auto-tuning**: Automatic scale selection via grid search
- **Built-in plotting**: Solution visualization, convergence plots, spectral sensitivity
- **Geometry samplers**: Box, ball, sphere, interval, custom samplers
- **Diagnostics**: Problem validation, conditioning checks, error detection
- **Export utilities**: NumPy conversion, checkpoint saving/loading
- **PyTorch Lightning**: Integration for training loops
- **20+ benchmark problems**: Linear, nonlinear, and regression-mode PDEs

## Method overview

1. **Feature construction.** Given collocation points x, compute
   `H = sin(x W + b)` together with exact derivatives
   `dH = cos(x W + b) * W` and `ddH = -sin(x W + b) * W^2`.

2. **Linear solve.** Assemble the PDE operator in feature space:
   `A beta = b`, and solve via least squares (optionally Tikhonov-regularised).

3. **Newton iteration (nonlinear).** Linearise the PDE residual around the
   current iterate, solve `J delta_beta = -R` with backtracking line search,
   and repeat until convergence.

## Adding your own PDE

Create a problem class with these methods:

```python
class MyProblem:
    def __init__(self):
        self.name = "My PDE"
        self.dim = 2  # Spatial dimension

    def exact(self, x):
        """Analytical solution u(x)."""
        return torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2])

    def exact_grad(self, x):
        """Gradient of exact solution."""
        # ... compute gradient analytically
        return grad_u

    def get_train_data(self, n_pde=5000, n_bc=1000):
        """Return (x_pde, bcs, f_pde) for training."""
        # ... sample collocation and boundary points
        return x_pde, bcs, f_pde

    def build(self, solver, x_pde, bcs, f_pde):
        """Assemble linear system A beta = b."""
        # ... build system matrix
        return A, b

    def get_test_points(self, n=5000):
        """Random test points for evaluation."""
        return torch.rand(n, self.dim)
```

Then solve it:

```python
problem = MyProblem()
result = solve_linear(problem)
```

## Paper

The full preprint is available on [arXiv](https://arxiv.org/abs/2602.10541)

## Citing this work

If you use FastLSQ in your research, please cite:

```bibtex
@misc{sulc2026solvingpdesshotfourier,
      title={Solving PDEs in One Shot via Fourier Features with Exact Analytical Derivatives},
      author={Antonin Sulc},
      year={2026},
      eprint={2602.10541},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2602.10541},
}
```

## License

This project is licensed under the MIT License -- see [LICENSE](LICENSE) for details.
