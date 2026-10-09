# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Export utilities for FastLSQ solutions (NumPy, VTK, etc.)."""

import time

import torch
import numpy as np
from typing import Optional, Union, Dict, Any

from fastlsq.solvers import FastLSQSolver
from fastlsq.device import get_device


def _provenance(solver: FastLSQSolver) -> Dict[str, Any]:
    """Auto-recorded provenance for a saved model: library version, device, dtype,
    and the realized frequency bandwidth (scale / Sigma) the model actually uses."""
    import fastlsq  # lazy: avoids a circular import at module load

    prov: Dict[str, Any] = {
        "fastlsq_version": getattr(fastlsq, "__version__", None),
        "created": time.time(),
        "input_dim": solver.input_dim,
        "n_features": solver.n_features,
    }
    W = torch.cat(solver.W_list, dim=1) if solver.W_list else None
    ref = W if W is not None else solver.beta
    if ref is not None:
        prov["device"] = str(ref.device)
        prov["dtype"] = str(ref.dtype).replace("torch.", "")
    if W is not None:
        # Realized frequency second moment Sigma = (W Wᵀ)/N and per-axis scale.
        # Stored as CPU tensors (not NumPy arrays) so the checkpoint stays
        # loadable under torch.load's safe weights_only=True default.
        freq_cov = (W @ W.transpose(-2, -1)) / W.shape[1]
        prov["freq_cov"] = freq_cov.detach().cpu().clone()
        prov["freq_std"] = torch.sqrt(
            torch.diagonal(freq_cov).clamp_min(0.0)
        ).detach().cpu().clone()
    return prov


def to_numpy(
    solver: FastLSQSolver,
    x: Union[torch.Tensor, np.ndarray],
    *,
    return_gradient: bool = False,
    return_laplacian: bool = False,
) -> Union[np.ndarray, tuple]:
    """Convert FastLSQ predictions to NumPy arrays.

    Parameters
    ----------
    solver : FastLSQSolver
    x : Tensor or ndarray
        Input points, shape (n, dim).
    return_gradient : bool
        Also return gradient.
    return_laplacian : bool
        Also return Laplacian.

    Returns
    -------
    u : ndarray, shape (n, 1)
    grad_u : ndarray, shape (n, dim), optional
    lap_u : ndarray, shape (n, 1), optional
    """
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, device=solver.beta.device, dtype=solver.beta.dtype)

    if return_laplacian:
        u, grad_u, lap_u = solver.predict_with_laplacian(x)
        return (
            u.cpu().numpy(),
            grad_u.cpu().numpy(),
            lap_u.cpu().numpy(),
        )
    elif return_gradient:
        u, grad_u = solver.predict_with_grad(x)
        return u.cpu().numpy(), grad_u.cpu().numpy()
    else:
        u = solver.predict(x)
        return u.cpu().numpy()


def to_dict(
    solver: FastLSQSolver,
    *,
    include_weights: bool = True,
    include_metadata: bool = True,
) -> Dict[str, Any]:
    """Export solver state to a dictionary (for serialization).

    The structural fields ``input_dim`` and ``normalize`` are always written:
    :func:`from_dict` cannot rebuild a solver without them.  Weights are stored
    as **CPU tensors** (not NumPy arrays), so that a checkpoint written by
    :func:`save_checkpoint` loads under ``torch.load``'s safe
    ``weights_only=True`` default.  Use ``{k: v.numpy() ...}`` yourself if you
    need arrays.

    Parameters
    ----------
    solver : FastLSQSolver
    include_weights : bool
        Include W_list, b_list, beta.  Requires a solved solver (``beta`` set).
    include_metadata : bool
        Include the derived ``n_features`` field.

    Returns
    -------
    state : dict
    """
    state = {"input_dim": solver.input_dim, "normalize": solver.normalize}
    if include_metadata:
        state["n_features"] = solver.n_features
    if include_weights:
        if solver.beta is None:
            raise ValueError(
                "to_dict: solver.beta is None (no solve has been run); pass "
                "include_weights=False to export only the structure.")
        state["W_list"] = [w.detach().cpu().clone() for w in solver.W_list]
        state["b_list"] = [b.detach().cpu().clone() for b in solver.b_list]
        state["beta"] = solver.beta.detach().cpu().clone()
    return state


def from_dict(
    state: Dict[str, Any],
    *,
    device: Optional[torch.device] = None,
) -> FastLSQSolver:
    """Reconstruct solver from dictionary.

    Parameters
    ----------
    state : dict
        State dictionary from `to_dict()` (tensors or NumPy arrays; files written
        by older versions stored arrays).
    device : torch.device, optional
        Device to load onto.  Defaults to the active FastLSQ device
        (:func:`fastlsq.get_device`), so ``set_device()`` is honoured.

    Returns
    -------
    solver : FastLSQSolver
    """
    if device is None:
        device = get_device()

    solver = FastLSQSolver(
        state["input_dim"],
        normalize=state.get("normalize", False),
    )

    W_list = [torch.as_tensor(w).to(device) for w in state["W_list"]]
    b_list = [torch.as_tensor(b).to(device) for b in state["b_list"]]

    for W, b in zip(W_list, b_list):
        solver.W_list.append(W)
        solver.b_list.append(b)
        solver._n_features += W.shape[1]

    solver.beta = torch.as_tensor(state["beta"]).to(device)

    return solver


def save_checkpoint(
    solver: FastLSQSolver,
    path: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save solver checkpoint to file.

    Parameters
    ----------
    solver : FastLSQSolver
    path : str
        File path (.pt or .pth extension recommended).
    metadata : dict, optional
        Additional metadata to save.  A ``provenance`` block (library version,
        device, dtype, timestamp, and realized scale / Sigma) is auto-recorded
        unless the caller supplies its own ``provenance`` key.  Keep the values
        to plain Python types and tensors: the file is loaded with
        ``weights_only=True``, which rejects arbitrary pickled objects.
    """
    state = to_dict(solver, include_weights=True, include_metadata=True)
    meta = dict(metadata) if metadata else {}
    meta.setdefault("provenance", _provenance(solver))
    state["metadata"] = meta
    torch.save(state, path)


def load_checkpoint(
    path: str,
    *,
    device: Optional[torch.device] = None,
    allow_pickle: bool = False,
) -> tuple[FastLSQSolver, Optional[Dict[str, Any]]]:
    """Load solver checkpoint from file.

    The file is read with ``torch.load(weights_only=True)``, which only
    unpickles tensors and plain containers and therefore cannot execute code
    embedded in a crafted file.  Checkpoints written by FastLSQ < 0.7.1 stored
    NumPy arrays, which that mode rejects; pass ``allow_pickle=True`` to read
    such a file **only if you trust its origin** (it is then loaded with
    ``weights_only=False``, i.e. a full unpickle).

    Parameters
    ----------
    path : str
        File path.
    device : torch.device, optional
        Defaults to the active FastLSQ device.
    allow_pickle : bool
        Permit a full unpickle for legacy (pre-0.7.1) files from a trusted source.

    Returns
    -------
    solver : FastLSQSolver
    metadata : dict, optional
    """
    device = device or get_device()
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:  # torch raises UnpicklingError / RuntimeError here
        if not allow_pickle:
            raise RuntimeError(
                f"load_checkpoint: {path!r} is not loadable in safe mode "
                "(weights_only=True). If it was written by FastLSQ < 0.7.1 and "
                "you trust where it came from, pass allow_pickle=True."
            ) from exc
        state = torch.load(path, map_location=device, weights_only=False)
    metadata = state.pop("metadata", None)
    solver = from_dict(state, device=device)
    return solver, metadata
