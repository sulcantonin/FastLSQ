# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Shared utilities: device configuration, evaluation helpers."""

import torch
import numpy as np

# ---------------------------------------------------------------------------
# Device configuration
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup(dtype=torch.float64, seed=42):
    """Set default dtype, seed RNGs, and print device info."""
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Device : {device}")
    print(f"Dtype  : {dtype}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_error(solver, problem, n_test=5000):
    """Compute relative L2 errors for function value and gradient.

    Returns
    -------
    val_err : float
        Relative L2 error of the predicted solution.
    grad_err : float
        Relative L2 error of the predicted gradient.
    """
    torch.manual_seed(999)
    x_test = problem.get_test_points(n_test)
    u_true = problem.exact(x_test)
    grad_true = problem.exact_grad(x_test)
    u_pred, grad_pred = solver.predict_with_grad(x_test)

    val_err = (torch.norm(u_pred - u_true) / (torch.norm(u_true) + 1e-15)).item()
    grad_err = (torch.norm(grad_pred - grad_true) /
                (torch.norm(grad_true) + 1e-15)).item()
    return val_err, grad_err
