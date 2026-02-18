# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Export utilities for FastLSQ solutions (NumPy, VTK, etc.)."""

import torch
import numpy as np
from typing import Optional, Union, Dict, Any

from fastlsq.solvers import FastLSQSolver


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

    Parameters
    ----------
    solver : FastLSQSolver
    include_weights : bool
        Include W_list, b_list, beta.
    include_metadata : bool
        Include input_dim, normalize, n_features.

    Returns
    -------
    state : dict
    """
    state = {}
    if include_metadata:
        state["input_dim"] = solver.input_dim
        state["normalize"] = solver.normalize
        state["n_features"] = solver.n_features
    if include_weights:
        state["W_list"] = [w.cpu().numpy() for w in solver.W_list]
        state["b_list"] = [b.cpu().numpy() for b in solver.b_list]
        state["beta"] = solver.beta.cpu().numpy()
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
        State dictionary from `to_dict()`.
    device : torch.device, optional
        Device to load onto.

    Returns
    -------
    solver : FastLSQSolver
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    solver = FastLSQSolver(
        state["input_dim"],
        normalize=state.get("normalize", False),
    )

    W_list = [torch.tensor(w, device=device) for w in state["W_list"]]
    b_list = [torch.tensor(b, device=device) for b in state["b_list"]]

    for W, b in zip(W_list, b_list):
        solver.W_list.append(W)
        solver.b_list.append(b)
        solver._n_features += W.shape[1]

    solver.beta = torch.tensor(state["beta"], device=device)

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
        Additional metadata to save.
    """
    state = to_dict(solver, include_weights=True, include_metadata=True)
    if metadata:
        state["metadata"] = metadata
    torch.save(state, path)


def load_checkpoint(
    path: str,
    *,
    device: Optional[torch.device] = None,
) -> tuple[FastLSQSolver, Optional[Dict[str, Any]]]:
    """Load solver checkpoint from file.

    Parameters
    ----------
    path : str
        File path.
    device : torch.device, optional

    Returns
    -------
    solver : FastLSQSolver
    metadata : dict, optional
    """
    state = torch.load(path, map_location=device)
    metadata = state.pop("metadata", None)
    solver = from_dict(state, device=device)
    return solver, metadata
