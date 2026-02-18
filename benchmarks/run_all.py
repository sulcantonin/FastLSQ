#!/usr/bin/env python
# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Comprehensive benchmark suite for FastLSQ.

Runs all problems and saves results to CSV for reproducibility.
"""

import torch
import numpy as np
import pandas as pd
import time
from pathlib import Path

from fastlsq import solve_linear, solve_nonlinear
from fastlsq.problems.linear import (
    PoissonND, HeatND, Wave1D, Wave2D_MS, Helmholtz2D, Maxwell2D_TM,
)
from fastlsq.problems.nonlinear import (
    NLPoisson2D, Bratu2D, SteadyBurgers1D, NLHelmholtz2D, AllenCahn1D,
)
from fastlsq.problems.regression import (
    Burgers1D_Regression, KdV_Regression, ReactionDiffusion_Regression,
    SineGordon_Regression, KleinGordon_Regression, GrayScott_Pulse,
    NavierStokes2D_Kovasznay, Bratu2D_Regression, NLHelmholtz2D_Regression,
)

# Setup
torch.set_default_dtype(torch.float32)
output_dir = Path("benchmarks/results")
output_dir.mkdir(parents=True, exist_ok=True)


def run_linear_benchmarks():
    """Run all linear/regression benchmarks."""
    problems = [
        PoissonND(), HeatND(), Wave1D(), Wave2D_MS(), Helmholtz2D(),
        Maxwell2D_TM(), Burgers1D_Regression(), KdV_Regression(),
        ReactionDiffusion_Regression(), SineGordon_Regression(),
        KleinGordon_Regression(), GrayScott_Pulse(),
        NavierStokes2D_Kovasznay(), Bratu2D_Regression(),
        NLHelmholtz2D_Regression(),
    ]

    results = []
    for problem in problems:
        print(f"\n{'='*60}")
        print(f"Running: {problem.name}")
        print(f"{'='*60}")

        try:
            t0 = time.time()
            result = solve_linear(
                problem,
                scale=None,  # Auto-select
                n_blocks=3,
                hidden_size=500,
                verbose=False,
            )
            runtime = time.time() - t0

            results.append({
                "problem": problem.name,
                "type": "linear",
                "dim": problem.dim,
                "scale": result["scale"],
                "val_err": result["metrics"]["val_err"],
                "grad_err": result["metrics"]["grad_err"],
                "runtime": runtime,
            })
            print(f"  ✓ Error: {result['metrics']['val_err']:.2e}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results.append({
                "problem": problem.name,
                "type": "linear",
                "dim": problem.dim,
                "scale": None,
                "val_err": np.nan,
                "grad_err": np.nan,
                "runtime": np.nan,
            })

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "linear_benchmarks.csv", index=False)
    print(f"\n✓ Saved results to {output_dir / 'linear_benchmarks.csv'}")
    return df


def run_nonlinear_benchmarks():
    """Run all nonlinear benchmarks."""
    problems = [
        NLPoisson2D(),
        Bratu2D(lam=1.0),
        SteadyBurgers1D(nu=0.1),
        NLHelmholtz2D(k=3.0, alpha=0.5),
        AllenCahn1D(eps=0.1),
    ]

    torch.set_default_dtype(torch.float64)  # Nonlinear needs higher precision

    results = []
    for problem in problems:
        print(f"\n{'='*60}")
        print(f"Running: {problem.name}")
        print(f"{'='*60}")

        try:
            t0 = time.time()
            result = solve_nonlinear(
                problem,
                scale=None,  # Auto-select
                n_blocks=3,
                hidden_size=500,
                max_iter=30,
                verbose=False,
            )
            runtime = time.time() - t0

            results.append({
                "problem": problem.name,
                "type": "nonlinear",
                "dim": problem.dim,
                "scale": result["scale"],
                "val_err": result["metrics"]["val_err"],
                "grad_err": result["metrics"]["grad_err"],
                "n_iters": result["n_iters"],
                "runtime": runtime,
            })
            print(f"  ✓ Error: {result['metrics']['val_err']:.2e}, "
                  f"Iterations: {result['n_iters']}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results.append({
                "problem": problem.name,
                "type": "nonlinear",
                "dim": problem.dim,
                "scale": None,
                "val_err": np.nan,
                "grad_err": np.nan,
                "n_iters": np.nan,
                "runtime": np.nan,
            })

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "nonlinear_benchmarks.csv", index=False)
    print(f"\n✓ Saved results to {output_dir / 'nonlinear_benchmarks.csv'}")
    return df


if __name__ == "__main__":
    print("FastLSQ Benchmark Suite")
    print("=" * 60)

    linear_df = run_linear_benchmarks()
    nonlinear_df = run_nonlinear_benchmarks()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Linear problems:    {len(linear_df)}")
    print(f"Nonlinear problems: {len(nonlinear_df)}")
    print(f"\nResults saved to: {output_dir}/")
