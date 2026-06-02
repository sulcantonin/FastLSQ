# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Cinematic visualisation helpers for FastLSQ showcase demos.

Clean white scientific-paper style: pure white background, dark navy
chips, saturated palettes for field visualization, and dark
high-contrast comet trails for particles.  All ten ``code_*`` demos
render through this module to share a single visual identity.

Separate from :mod:`fastlsq.plotting`, which targets diagnostic
1D/2D slices for the benchmark suite.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba, to_rgb


# ======================================================================
# Style constants -- clean scientific paper style
# ======================================================================

BG_LIGHT = "#ffffff"        # pure white canvas
PANEL_BG = "#fafafc"        # very subtle off-white for chips
TXT_DARK = "#1a1a2e"        # deep navy ink
TXT_BODY = "#2d2d40"        # body text
TXT_DIM = "#5d5d7d"         # captions / metrics
TXT_RULE = "#cfd0d8"        # chip borders

ACCENT_HOT = "#c0392b"      # crimson
ACCENT_COOL = "#1f3a93"     # deep blue
ACCENT_GOLD = "#b8860b"     # dark goldenrod
ACCENT_GREEN = "#1f7a4d"    # forest green

# Particle colour rotation for inverse-problem demos (well-separated
# on white background, all dark / saturated).
PARTICLE_PALETTE = [
    "#c0392b",   # crimson
    "#1f3a93",   # deep blue
    "#1f7a4d",   # forest green
    "#b8860b",   # gold
    "#8e44ad",   # purple
    "#d35400",   # burnt orange
]

# Back-compat aliases (older demo code uses these names)
BG_DARK = BG_LIGHT          # alias kept for source backwards-compat
TXT_GOLD = ACCENT_HOT       # alias: where the dark style used gold accents, now crimson
TXT_BRIGHT = TXT_DARK

FONT_TITLE = 14
FONT_EQ = 10
FONT_CHIP = 10
FONT_METRIC = 9

CHIP_BBOX = dict(facecolor="#ffffff", edgecolor=TXT_RULE,
                 boxstyle="round,pad=0.45", linewidth=0.7)


# ======================================================================
# Palettes -- all white-anchored, sequential or diverging
# ======================================================================

def _cmap(name: str, stops: Sequence[str]) -> LinearSegmentedColormap:
    cm = LinearSegmentedColormap.from_list(name, list(stops))
    cm.set_bad(BG_LIGHT, alpha=1.0)
    return cm


PALETTES = {
    # Sequential: white -> warm fire (low values nearly invisible on
    # white, high values pop).
    "thermal": _cmap("fastlsq_thermal",
                     ["#ffffff", "#ffeec5", "#fdc06f", "#f47b3a",
                      "#c0392b", "#7a1018", "#2a0508"]),
    # Sequential: white -> plasma magenta-purple.
    "plasma":  _cmap("fastlsq_plasma",
                     ["#ffffff", "#f6dafd", "#e394d8", "#c95bb1",
                      "#9b297a", "#5a1361", "#1a0935"]),
    # Diverging: deep blue -> white -> deep red.  For signed fields
    # (vorticity, wave amplitudes, error maps).
    "wave":    _cmap("fastlsq_wave",
                     ["#1a1a4e", "#1f3a93", "#5a8cd6", "#bcd3f0",
                      "#ffffff",
                      "#f0b8b8", "#d65a5a", "#c0392b", "#4e1a1a"]),
    # Sequential: white -> cool quantum.
    "quantum": _cmap("fastlsq_quantum",
                     ["#ffffff", "#cde9f0", "#75c0d4", "#2b8aa6",
                      "#1f3a93", "#0c1b58", "#040820"]),
    # Sequential: white -> mirage / eikonal warm.
    "eikonal": _cmap("fastlsq_eikonal",
                     ["#ffffff", "#fdedb5", "#f7c459", "#e58527",
                      "#a82a6d", "#491b6d", "#0d0a2a"]),
    # Sequential: white -> fluid teal-navy.
    "fluid":   _cmap("fastlsq_fluid",
                     ["#ffffff", "#cfe9df", "#6abd9a", "#239867",
                      "#0e6d5c", "#0a3a4d", "#04141f"]),
}


# ======================================================================
# Figure factory
# ======================================================================

def hero_figure(size: Tuple[float, float] = (8.0, 8.0),
                dpi: int = 180,
                aspect: str = "equal",
                facecolor: str = BG_LIGHT) -> Tuple[plt.Figure, plt.Axes]:
    """Square white-canvas figure with no spines/ticks/labels."""
    fig, ax = plt.subplots(figsize=size, dpi=dpi,
                           facecolor=facecolor,
                           constrained_layout=True)
    ax.set_facecolor(facecolor)
    if aspect:
        ax.set_aspect(aspect)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    return fig, ax


def hero_figure_landscape(size: Tuple[float, float] = (13.0, 7.5),
                          dpi: int = 180,
                          ncols: int = 2,
                          width_ratios: Optional[Sequence[float]] = None,
                          facecolor: str = BG_LIGHT
                          ) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    if width_ratios is None:
        width_ratios = [1.6, 1.0]
    fig, axes = plt.subplots(1, ncols, figsize=size, dpi=dpi,
                             facecolor=facecolor,
                             gridspec_kw=dict(width_ratios=width_ratios),
                             constrained_layout=True)
    for ax in axes:
        ax.set_facecolor(facecolor)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(False)
    return fig, axes


# ======================================================================
# Annotation chips
# ======================================================================

def add_title(ax: plt.Axes, text: str, equation: Optional[str] = None,
              loc: Tuple[float, float] = (0.018, 0.955)) -> None:
    """Bold dark-navy title chip in the upper-left corner."""
    ax.text(loc[0], loc[1], text,
            transform=ax.transAxes, color=TXT_DARK,
            fontsize=FONT_TITLE, fontweight="bold", zorder=20,
            ha="left", va="top", bbox=CHIP_BBOX)
    if equation:
        ax.text(loc[0] + 0.005, loc[1] - 0.065, equation,
                transform=ax.transAxes, color=TXT_DIM,
                fontsize=FONT_EQ, style="italic", zorder=20,
                ha="left", va="top")


def add_metrics(ax: plt.Axes,
                n_features: Optional[int] = None,
                m_points: Optional[int] = None,
                solve_time_s: Optional[float] = None,
                rel_l2: Optional[float] = None,
                extra: Optional[str] = None,
                loc: Tuple[float, float] = (0.018, 0.018)) -> None:
    parts = []
    if n_features is not None:
        parts.append(f"N = {n_features:,}")
    if m_points is not None:
        parts.append(f"M = {m_points:,}")
    if solve_time_s is not None:
        parts.append(f"t = {solve_time_s:.2f} s")
    if rel_l2 is not None:
        parts.append(f"rel L2 = {rel_l2:.1e}")
    if extra:
        parts.append(extra)
    text = "   ".join(parts)
    ax.text(loc[0], loc[1], text,
            transform=ax.transAxes, color=TXT_DIM,
            fontsize=FONT_METRIC, family="monospace", zorder=20,
            ha="left", va="bottom", bbox=CHIP_BBOX)


def add_paramchip(ax: plt.Axes, label: str, value: float,
                  unit: Optional[str] = None, fmt: str = "{:.2f}",
                  loc: Tuple[float, float] = (0.982, 0.955)) -> None:
    s = f"{label} = {fmt.format(value)}"
    if unit:
        s += f" {unit}"
    ax.text(loc[0], loc[1], s,
            transform=ax.transAxes, color=TXT_DARK,
            fontsize=FONT_CHIP, fontweight="bold", zorder=20,
            ha="right", va="top", bbox=CHIP_BBOX)


def add_caption(ax: plt.Axes, text: str,
                loc: Tuple[float, float] = (0.982, 0.018)) -> None:
    ax.text(loc[0], loc[1], text,
            transform=ax.transAxes, color=TXT_DIM,
            fontsize=FONT_METRIC, zorder=20,
            ha="right", va="bottom", bbox=CHIP_BBOX)


def add_timebar(ax: plt.Axes, frac: float,
                color: str = TXT_DARK,
                height: float = 0.006) -> None:
    """Thin progress bar across the bottom inside ax. frac in [0,1]."""
    ax.add_patch(plt.Rectangle((0.0, 0.0), 1.0, height,
                               transform=ax.transAxes,
                               facecolor=TXT_RULE, edgecolor="none",
                               zorder=19))
    ax.add_patch(plt.Rectangle((0.0, 0.0), float(np.clip(frac, 0, 1)), height,
                               transform=ax.transAxes,
                               facecolor=color, edgecolor="none",
                               zorder=20))


# ======================================================================
# Comet-trail streamers
# ======================================================================

@dataclass
class CometStreamers:
    """Particle pool with comet trails rendered via LineCollection.

    For the scientific white style, trails use saturated dark colours
    so they read against the light background.
    """
    trail_len: int = 14
    max_life: int = 60
    color: str = ACCENT_HOT
    head_size: float = 12.0
    line_width_max: float = 2.0
    line_width_min: float = 0.2

    pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    vel: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    age: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    hist: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 2), np.float32))

    def spawn(self, pos: np.ndarray, vel: np.ndarray) -> None:
        pos = np.asarray(pos, dtype=np.float32).reshape(-1, 2)
        vel = np.asarray(vel, dtype=np.float32).reshape(-1, 2)
        n_new = pos.shape[0]
        if n_new == 0:
            return
        self.pos = np.concatenate([self.pos, pos], axis=0)
        self.vel = np.concatenate([self.vel, vel], axis=0)
        self.age = np.concatenate([self.age, np.zeros(n_new, np.int32)], axis=0)

    def advect(self, dt: float,
               field_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
               bounds: Optional[Tuple[float, float, float, float]] = None
               ) -> None:
        if self.pos.shape[0] == 0:
            return
        if field_fn is not None:
            self.vel = np.asarray(field_fn(self.pos), dtype=np.float32).reshape(-1, 2)
        self.pos = self.pos + dt * self.vel
        self.age = self.age + 1
        keep = self.age < self.max_life
        if bounds is not None:
            xmin, xmax, ymin, ymax = bounds
            keep &= (self.pos[:, 0] >= xmin) & (self.pos[:, 0] <= xmax)
            keep &= (self.pos[:, 1] >= ymin) & (self.pos[:, 1] <= ymax)
        self.pos = self.pos[keep]
        self.vel = self.vel[keep]
        self.age = self.age[keep]

    def _build_history(self, dt: float) -> np.ndarray:
        n = self.pos.shape[0]
        T = self.trail_len
        hist = np.empty((T, n, 2), dtype=np.float32)
        hist[-1] = self.pos
        for back in range(1, T):
            hist[T - 1 - back] = self.pos - back * dt * self.vel
        return hist

    def draw(self, ax: plt.Axes, dt: float,
             color: Optional[str] = None,
             head: bool = True, glow: bool = False,
             zorder: int = 8) -> None:
        if self.pos.shape[0] == 0:
            return
        c = color or self.color
        hist = self._build_history(dt)
        T = self.trail_len
        segs = np.empty((T - 1, hist.shape[1], 2, 2), dtype=np.float32)
        segs[..., 0, :] = hist[:-1]
        segs[..., 1, :] = hist[1:]
        segs_flat = segs.reshape(-1, 2, 2)
        frac_t = (np.arange(T - 1, dtype=np.float32) + 1) / T
        age_frac = np.maximum(0.0, 1.0 - self.age / self.max_life)
        rgba = np.tile(np.array(to_rgba(c), dtype=np.float32)[None, None, :],
                       (T - 1, hist.shape[1], 1))
        rgba[..., 3] = (frac_t[:, None] ** 1.4) * age_frac[None, :] * 0.85
        lw = (self.line_width_min
              + (self.line_width_max - self.line_width_min) * frac_t ** 2)
        lw_full = lw[:, None] * np.ones((T - 1, hist.shape[1]), dtype=np.float32)
        lc = LineCollection(segs_flat,
                            colors=rgba.reshape(-1, 4),
                            linewidths=lw_full.reshape(-1),
                            capstyle="round", zorder=zorder)
        ax.add_collection(lc)
        if head:
            ax.scatter(self.pos[:, 0], self.pos[:, 1],
                       s=self.head_size, c=c, marker="o", linewidth=0,
                       alpha=0.95, zorder=zorder + 1)
        if glow:
            ax.scatter(self.pos[:, 0], self.pos[:, 1],
                       s=self.head_size * 3.0,
                       c=[(*to_rgb(c), 0.12)],
                       marker="o", linewidth=0, zorder=zorder + 0.5)


def emit_isotropic_2d(n: int, rng: np.random.Generator) -> np.ndarray:
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.stack([np.cos(th), np.sin(th)], axis=1).astype(np.float32)


def emit_dipole_2d(n: int, axis: np.ndarray, beta: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Sample n unit vectors from the 2D far-field synchrotron angular
    distribution for a charge moving along ``axis`` with speed parameter
    ``beta`` = v/c.  The Doppler-beamed pattern f(theta) ∝ 1/(1 - β cosθ)³
    is inverted via grid CDF."""
    axis = np.asarray(axis, dtype=np.float32).reshape(2)
    nrm = float(np.hypot(axis[0], axis[1])) + 1e-12
    ux, uy = axis / nrm
    px, py = -uy, ux
    th_grid = np.linspace(-np.pi, np.pi, 2048)
    pdf = 1.0 / np.maximum(1e-6, (1.0 - beta * np.cos(th_grid)) ** 3)
    cdf = np.cumsum(pdf); cdf /= cdf[-1]
    u = rng.uniform(0.0, 1.0, n)
    th = np.interp(u, cdf, th_grid)
    cx = np.cos(th); sx = np.sin(th)
    vx = ux * cx + px * sx
    vy = uy * cx + py * sx
    return np.stack([vx, vy], axis=1).astype(np.float32)


# ======================================================================
# Pre-render verification
# ======================================================================

def verify_before_render(check_fn: Callable[[], float],
                         tol: float,
                         label: str = "verification") -> float:
    val = float(check_fn())
    if val > tol:
        raise RuntimeError(
            f"{label} FAILED: rel L2 = {val:.3e} > tol {tol:.0e}")
    print(f"  {label} ok: rel L2 = {val:.3e} (<= {tol:.0e})")
    return val


# ======================================================================
# Animation writer
# ======================================================================

def save_animation(fig: plt.Figure,
                   update_fn: Callable[[int], Iterable],
                   n_frames: int,
                   fps: int,
                   out_path: str,
                   dpi: Optional[int] = None,
                   writer: Optional[str] = None) -> None:
    """Render the animation and save. Auto-detects .gif vs .mp4 from suffix.

    Defaults are sized for the social-media (~15 MB) sweet spot.  GIF at
    80 dpi (palette-indexed), MP4 at 120 dpi (H.264).  Pass ``dpi``
    explicitly to override."""
    suffix = os.path.splitext(out_path)[1].lower()
    if writer is None:
        writer = "pillow" if suffix == ".gif" else "ffmpeg"
    if dpi is None:
        if suffix == ".gif":
            dpi = 80
        elif suffix in (".mp4", ".mov", ".m4v"):
            dpi = 120
        else:
            dpi = int(fig.get_dpi())
    print(f"  rendering {n_frames} frames @ {fps} fps  ->  {out_path}  (dpi={dpi})")
    t0 = time.time()
    ani = animation.FuncAnimation(fig, update_fn, frames=n_frames, blit=False)
    ani.save(out_path, writer=writer, fps=fps, dpi=dpi)
    plt.close(fig)
    print(f"  saved in {time.time() - t0:.1f} s")


def save_hero_still(fig: plt.Figure, out_path: str,
                    dpi: Optional[int] = None) -> None:
    if dpi is None:
        dpi = max(180, int(fig.get_dpi()))
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.08)
    print(f"  still saved -> {out_path}")


# ======================================================================
# Convergence inset
# ======================================================================

def add_convergence_inset(fig: plt.Figure, ax: plt.Axes,
                          history: Sequence[float],
                          *,
                          loc: Tuple[float, float, float, float]
                          = (0.66, 0.04, 0.31, 0.18),
                          ylabel: str = "loss",
                          color: str = ACCENT_HOT) -> plt.Axes:
    """Inset convergence plot — white-bg scientific style."""
    inax = fig.add_axes(loc, facecolor="#ffffff")
    inax.set_yscale("log")
    inax.plot(np.arange(1, len(history) + 1), history,
              color=color, linewidth=1.4)
    inax.tick_params(colors=TXT_DIM, labelsize=7,
                     length=2, width=0.5)
    for s in ("top", "right"):
        inax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        inax.spines[s].set_color(TXT_DIM)
        inax.spines[s].set_linewidth(0.5)
    inax.set_xlabel("iter", color=TXT_DIM, fontsize=7)
    inax.set_ylabel(ylabel, color=TXT_DIM, fontsize=7)
    inax.grid(True, color=TXT_RULE, linewidth=0.4)
    return inax


__all__ = [
    "BG_LIGHT", "PANEL_BG", "TXT_DARK", "TXT_BODY", "TXT_DIM", "TXT_RULE",
    "ACCENT_HOT", "ACCENT_COOL", "ACCENT_GOLD", "ACCENT_GREEN",
    "PARTICLE_PALETTE",
    "BG_DARK", "TXT_GOLD", "TXT_BRIGHT",  # back-compat aliases
    "FONT_TITLE", "FONT_EQ", "FONT_CHIP", "FONT_METRIC",
    "PALETTES",
    "hero_figure", "hero_figure_landscape",
    "add_title", "add_metrics", "add_paramchip", "add_caption", "add_timebar",
    "CometStreamers", "emit_isotropic_2d", "emit_dipole_2d",
    "verify_before_render",
    "save_animation", "save_hero_still",
    "add_convergence_inset",
]
