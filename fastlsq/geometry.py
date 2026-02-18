# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Geometry samplers for generating collocation and boundary points."""

import torch
import numpy as np
from typing import Optional, Callable, Tuple, Union

from fastlsq.utils import device


# ======================================================================
# Generic samplers
# ======================================================================

def sample_box(
    n: int,
    dim: int,
    *,
    bounds: Optional[Tuple[float, float]] = None,
    device=device,
) -> torch.Tensor:
    """Sample uniformly from a hypercube [a, b]^dim.

    Parameters
    ----------
    n : int
        Number of points.
    dim : int
        Spatial dimension.
    bounds : tuple[float, float], optional
        (min, max) bounds. Default: (0.0, 1.0).
    device : torch.device

    Returns
    -------
    x : Tensor, shape (n, dim)
    """
    if bounds is None:
        bounds = (0.0, 1.0)
    a, b = bounds
    return torch.rand(n, dim, device=device) * (b - a) + a


def sample_ball(
    n: int,
    dim: int,
    *,
    radius: float = 1.0,
    center: Optional[torch.Tensor] = None,
    device=device,
) -> torch.Tensor:
    """Sample uniformly from a ball (uniform in volume, not on surface).

    Parameters
    ----------
    n : int
        Number of points.
    dim : int
        Spatial dimension.
    radius : float
        Ball radius.
    center : Tensor, shape (dim,), optional
        Center point. Default: origin.
    device : torch.device

    Returns
    -------
    x : Tensor, shape (n, dim)
    """
    # Sample from unit ball: uniform direction, then scale by r^(1/dim)
    x = torch.randn(n, dim, device=device)
    x = x / torch.norm(x, dim=1, keepdim=True)
    r = torch.rand(n, 1, device=device) ** (1.0 / dim)
    x = x * r * radius

    if center is not None:
        x = x + center.unsqueeze(0)
    return x


def sample_sphere(
    n: int,
    dim: int,
    *,
    radius: float = 1.0,
    center: Optional[torch.Tensor] = None,
    device=device,
) -> torch.Tensor:
    """Sample uniformly from a sphere surface.

    Parameters
    ----------
    n : int
        Number of points.
    dim : int
        Spatial dimension.
    radius : float
        Sphere radius.
    center : Tensor, shape (dim,), optional
        Center point. Default: origin.
    device : torch.device

    Returns
    -------
    x : Tensor, shape (n, dim)
    """
    x = torch.randn(n, dim, device=device)
    x = x / torch.norm(x, dim=1, keepdim=True) * radius

    if center is not None:
        x = x + center.unsqueeze(0)
    return x


def sample_interval(
    n: int,
    *,
    a: float = 0.0,
    b: float = 1.0,
    device=device,
) -> torch.Tensor:
    """Sample uniformly from an interval [a, b].

    Parameters
    ----------
    n : int
        Number of points.
    a, b : float
        Interval bounds.
    device : torch.device

    Returns
    -------
    x : Tensor, shape (n, 1)
    """
    return torch.rand(n, 1, device=device) * (b - a) + a


def sample_boundary_box(
    n: int,
    dim: int,
    *,
    bounds: Optional[Tuple[float, float]] = None,
    device=device,
) -> torch.Tensor:
    """Sample uniformly from the boundary of a hypercube.

    Parameters
    ----------
    n : int
        Number of points.
    dim : int
        Spatial dimension.
    bounds : tuple[float, float], optional
        (min, max) bounds. Default: (0.0, 1.0).
    device : torch.device

    Returns
    -------
    x : Tensor, shape (n, dim)
    """
    if bounds is None:
        bounds = (0.0, 1.0)
    a, b = bounds

    n_per_face = n // (2 * dim)
    points = []

    for d in range(dim):
        # Face at x_d = a
        x = torch.rand(n_per_face, dim, device=device) * (b - a) + a
        x[:, d] = a
        points.append(x)

        # Face at x_d = b
        x = torch.rand(n_per_face, dim, device=device) * (b - a) + a
        x[:, d] = b
        points.append(x)

    # Fill remainder randomly
    remainder = n - len(points) * n_per_face
    if remainder > 0:
        x = torch.rand(remainder, dim, device=device) * (b - a) + a
        face_idx = torch.randint(0, 2 * dim, (remainder,), device=device)
        dim_idx = face_idx // 2
        val = (face_idx % 2) * (b - a) + a
        x[torch.arange(remainder, device=device), dim_idx] = val
        points.append(x)

    return torch.cat(points, dim=0)


# ======================================================================
# Custom sampler wrapper
# ======================================================================

def custom_sampler(
    sampler_fn: Callable[[int], torch.Tensor],
    n: int,
) -> torch.Tensor:
    """Wrap a custom sampling function.

    Parameters
    ----------
    sampler_fn : callable
        Function that takes n (int) and returns Tensor shape (n, dim).
    n : int

    Returns
    -------
    x : Tensor
    """
    return sampler_fn(n)


# ======================================================================
# Helper: get sampler by name
# ======================================================================

def get_sampler(name: str) -> Callable:
    """Get a sampler function by name.

    Parameters
    ----------
    name : str
        One of: 'box', 'ball', 'sphere', 'interval', 'boundary_box'.

    Returns
    -------
    sampler : callable
    """
    samplers = {
        "box": sample_box,
        "ball": sample_ball,
        "sphere": sample_sphere,
        "interval": sample_interval,
        "boundary_box": sample_boundary_box,
    }
    if name not in samplers:
        raise ValueError(f"Unknown sampler: {name}. Choose from {list(samplers.keys())}")
    return samplers[name]
