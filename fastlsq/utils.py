# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Shared utilities: device configuration, evaluation helpers."""

import contextlib

import torch
import numpy as np

# ---------------------------------------------------------------------------
# Device configuration
# ---------------------------------------------------------------------------
# Device selection lives in fastlsq.device (CPU/CUDA/Apple-MPS, dtype-aware).
# Internal code calls fastlsq.device.get_device() so it respects runtime
# set_device(); ``device`` below is kept as a back-compat import-time snapshot.
from fastlsq.device import (  # noqa: E402,F401
    resolve_device, get_device, set_device, device_info,
)

device = get_device()


@contextlib.contextmanager
def preserve_rng(device=None):
    """Save and restore the global torch and NumPy RNG state around a block.

    FastLSQ seeds its own *internal* draws (the fixed test set of
    :func:`evaluate_error`, the per-trial seeds of
    :func:`~fastlsq.tuning.auto_select_scale`) for reproducible diagnostics.
    Doing that with ``torch.manual_seed`` alone would also reset the *caller's*
    stream, so two consecutive ``solve_linear`` calls would draw identical random
    features and a user's own ``torch.rand`` afterwards would become deterministic.
    Wrapping those draws in this context keeps the library's seeding invisible to
    the caller.

    The CPU generator, the NumPy global generator and the generator of ``device``
    (the active FastLSQ device by default; CUDA or MPS) are restored.  Other CUDA
    devices are not touched.
    """
    dev = device or get_device()
    cpu_state = torch.get_rng_state()
    np_state = np.random.get_state()
    acc_state = None
    if dev.type == "cuda" and torch.cuda.is_available():
        acc_state = torch.cuda.get_rng_state(dev)
    elif dev.type == "mps" and hasattr(getattr(torch, "mps", None), "get_rng_state"):
        acc_state = torch.mps.get_rng_state()
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        np.random.set_state(np_state)
        if acc_state is not None:
            if dev.type == "cuda":
                torch.cuda.set_rng_state(acc_state, dev)
            else:
                torch.mps.set_rng_state(acc_state)


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

    The test set is drawn from a fixed seed so that errors are comparable across
    calls; the caller's global RNG state is restored afterwards (see
    :func:`preserve_rng`).

    Returns
    -------
    val_err : float
        Relative L2 error of the predicted solution.
    grad_err : float
        Relative L2 error of the predicted gradient.
    """
    with preserve_rng():
        torch.manual_seed(999)
        x_test = problem.get_test_points(n_test)
    u_true = problem.exact(x_test)
    grad_true = problem.exact_grad(x_test)
    u_pred, grad_pred = solver.predict_with_grad(x_test)

    val_err = (torch.norm(u_pred - u_true) / (torch.norm(u_true) + 1e-15)).item()
    grad_err = (torch.norm(grad_pred - grad_true) /
                (torch.norm(grad_true) + 1e-15)).item()
    return val_err, grad_err
