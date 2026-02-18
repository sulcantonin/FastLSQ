# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Built-in plotting utilities for FastLSQ solutions and diagnostics."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Any

from fastlsq.solvers import FastLSQSolver
from fastlsq.utils import device


# ======================================================================
# Solution plotting
# ======================================================================

def plot_solution_1d(
    solver: FastLSQSolver,
    problem,
    *,
    x_min: float = 0.0,
    x_max: float = 1.0,
    n_points: int = 1000,
    plot_exact: bool = True,
    plot_gradient: bool = False,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    **kwargs,
) -> plt.Axes:
    """Plot 1D solution comparison (predicted vs exact).

    Parameters
    ----------
    solver : FastLSQSolver
    problem : object
        Must have `exact(x)` and optionally `exact_grad(x)`.
    x_min, x_max : float
        Domain bounds.
    n_points : int
        Number of evaluation points.
    plot_exact : bool
        Overlay exact solution.
    plot_gradient : bool
        Also plot gradient comparison.
    ax : matplotlib.Axes, optional
        Axes to plot on (creates new figure if None).
    title : str, optional
        Plot title.
    save_path : str, optional
        If provided, save figure to this path.
    **kwargs
        Passed to plt.plot().

    Returns
    -------
    ax : matplotlib.Axes
    """
    x_plot = torch.linspace(x_min, x_max, n_points, device=device).unsqueeze(1)
    u_pred = solver.predict(x_plot).cpu().numpy()
    u_exact = problem.exact(x_plot).cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x_plot.cpu().numpy(), u_pred, label="FastLSQ", linewidth=2, **kwargs)
    if plot_exact:
        ax.plot(x_plot.cpu().numpy(), u_exact, "--", label="Exact", linewidth=2, alpha=0.7)

    if plot_gradient:
        grad_pred, grad_exact = solver.predict_with_grad(x_plot)[1], problem.exact_grad(x_plot)
        ax2 = ax.twinx()
        ax2.plot(x_plot.cpu().numpy(), grad_pred.cpu().numpy(), ":", label="Grad (pred)", alpha=0.6)
        ax2.plot(x_plot.cpu().numpy(), grad_exact.cpu().numpy(), ":", label="Grad (exact)", alpha=0.6)
        ax2.set_ylabel("Gradient", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return ax


def plot_solution_2d_slice(
    solver: FastLSQSolver,
    problem,
    *,
    dim: int = 0,
    slice_val: float = 0.5,
    x_min: float = 0.0,
    x_max: float = 1.0,
    n_points: int = 100,
    plot_exact: bool = True,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    **kwargs,
) -> plt.Axes:
    """Plot 2D solution as a 1D slice.

    Parameters
    ----------
    solver : FastLSQSolver
    problem : object
        Must have `exact(x)`.
    dim : int
        Which dimension to slice along (0 or 1).
    slice_val : float
        Value of the other dimension at which to slice.
    x_min, x_max : float
        Domain bounds for the sliced dimension.
    n_points : int
        Number of evaluation points.
    plot_exact : bool
        Overlay exact solution.
    ax : matplotlib.Axes, optional
    title : str, optional
    save_path : str, optional
    **kwargs
        Passed to plt.plot().

    Returns
    -------
    ax : matplotlib.Axes
    """
    x_plot = torch.linspace(x_min, x_max, n_points, device=device).unsqueeze(1)
    if dim == 0:
        x_full = torch.cat([x_plot, slice_val * torch.ones_like(x_plot)], dim=1)
        xlabel = "x"
    else:
        x_full = torch.cat([slice_val * torch.ones_like(x_plot), x_plot], dim=1)
        xlabel = "y"

    u_pred = solver.predict(x_full).cpu().numpy()
    u_exact = problem.exact(x_full).cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x_plot.cpu().numpy(), u_pred, label="FastLSQ", linewidth=2, **kwargs)
    if plot_exact:
        ax.plot(x_plot.cpu().numpy(), u_exact, "--", label="Exact", linewidth=2, alpha=0.7)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"u({xlabel})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return ax


def plot_solution_2d_contour(
    solver: FastLSQSolver,
    problem,
    *,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    n_points: int = 100,
    plot_exact: bool = True,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    **kwargs,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Plot 2D solution as contour plots (predicted vs exact).

    Parameters
    ----------
    solver : FastLSQSolver
    problem : object
        Must have `exact(x)`.
    x_min, x_max, y_min, y_max : float
        Domain bounds.
    n_points : int
        Grid resolution (n_points x n_points).
    plot_exact : bool
        Show exact solution comparison.
    figsize : tuple
        Figure size.
    save_path : str, optional
    **kwargs
        Passed to plt.contourf().

    Returns
    -------
    fig : matplotlib.Figure
    axes : tuple of matplotlib.Axes
    """
    x = np.linspace(x_min, x_max, n_points)
    y = np.linspace(y_min, y_max, n_points)
    X, Y = np.meshgrid(x, y)
    xy = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), device=device)

    u_pred = solver.predict(xy).cpu().numpy().reshape(X.shape)
    u_exact = problem.exact(xy).cpu().numpy().reshape(X.shape)

    if plot_exact:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0] // 2, figsize[1]))
        ax2 = None

    im1 = ax1.contourf(X, Y, u_pred, levels=20, **kwargs)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("FastLSQ Prediction")
    plt.colorbar(im1, ax=ax1)

    if plot_exact and ax2 is not None:
        im2 = ax2.contourf(X, Y, u_exact, levels=20, **kwargs)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title("Exact Solution")
        plt.colorbar(im2, ax=ax2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return fig, (ax1, ax2) if plot_exact else (ax1,)


# ======================================================================
# Convergence plotting
# ======================================================================

def plot_convergence(
    history: List[Dict[str, Any]],
    *,
    labels: Optional[List[str]] = None,
    problem_name: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Plot Newton convergence history (residual and relative solution change).

    Parameters
    ----------
    history : list[dict]
        List of iteration dicts with keys: 'iter', 'residual', 'rel_du', etc.
    labels : list[str], optional
        Labels for multiple histories (if history is a list of lists).
    problem_name : str, optional
        Title prefix.
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    fig : matplotlib.Figure
    axes : tuple of matplotlib.Axes
    """
    # Handle single history vs list of histories
    if isinstance(history[0], dict):
        histories = [history]
        if labels is None:
            labels = [None]
    else:
        histories = history
        if labels is None:
            labels = [None] * len(histories)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for hist, label in zip(histories, labels):
        iters = [h["iter"] for h in hist]
        residuals = [h["residual"] for h in hist]
        rel_dus = [h.get("rel_du", 0.0) for h in hist]

        ax1.semilogy(iters, residuals, "-o", label=label, markersize=4)
        ax2.semilogy(iters, rel_dus, "-o", label=label, markersize=4)

    ax1.set_xlabel("Newton Iteration")
    ax1.set_ylabel("Residual norm")
    ax1.set_title(f"{problem_name + ': ' if problem_name else ''}Residual")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Newton Iteration")
    ax2.set_ylabel("Relative solution change")
    ax2.set_title(f"{problem_name + ': ' if problem_name else ''}Convergence")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return fig, (ax1, ax2)


def plot_spectral_sensitivity(
    scales: List[float],
    val_errors: List[float],
    grad_errors: Optional[List[float]] = None,
    *,
    problem_name: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot error vs scale (spectral sensitivity analysis).

    Parameters
    ----------
    scales : list[float]
    val_errors : list[float]
    grad_errors : list[float], optional
    problem_name : str, optional
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    fig : matplotlib.Figure
    ax : matplotlib.Axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(scales, val_errors, "b-o", label=r"Value Error ($L_2$)", linewidth=2)
    if grad_errors is not None:
        ax.plot(scales, grad_errors, "r--s", label=r"Gradient Error ($L_2$)", linewidth=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Scale $\sigma$ (Bandwidth)", fontsize=12)
    ax.set_ylabel("Relative Error", fontsize=12)
    if problem_name:
        ax.set_title(f"Spectral Sensitivity: {problem_name}", fontsize=14)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return fig, ax
