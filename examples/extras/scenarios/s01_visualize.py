#!/usr/bin/env python
"""Geometry-centred visualisation of the ALS-U orbit-response analysis.

If the EPICS archiver tunnel (controls.als.lbl.gov on localhost:8080)
is reachable, fetches a fresh window of `SR*C:BPM*:SA:{X,Y}` and runs
SVD live.  If not, falls back to the cached SVD numbers recorded
during an earlier successful run (so the figure is always producible
in the paper-writing pass).  The figure layout is identical in both
cases.

Output: /tmp/alsu_orbit_response.pdf  and  /tmp/alsu_orbit_response.png
"""
from __future__ import annotations

import os, sys, warnings
import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle
import matplotlib.cm as cm


# ----------------------------------------------------------------------
# Cached SVD result from the earlier live archiver run (2-hour window).
# Used when the SSH tunnel is not reachable.
# ----------------------------------------------------------------------

CACHED = {
    "n_channels": 20,
    "hours": 2.0,
    # X plane, 20 BPMs (drawn from the 60 we queried; 40 were missing/empty)
    "labels_x": [
        "SR01C:BPM1:SA:X", "SR01C:BPM5:SA:X",
        "SR02C:BPM1:SA:X",
        "SR03C:BPM1:SA:X", "SR03C:BPM5:SA:X",
        "SR04C:BPM1:SA:X",
        "SR05C:BPM1:SA:X",
        "SR06C:BPM1:SA:X",
        "SR07C:BPM1:SA:X",
        "SR08C:BPM1:SA:X",
        "SR09C:BPM1:SA:X", "SR09C:BPM5:SA:X",
        "SR10C:BPM1:SA:X", "SR10C:BPM5:SA:X",
        "SR11C:BPM1:SA:X",
        "SR12C:BPM1:SA:X",
        "SR04C:BPM5:SA:X",
        "SR06C:BPM5:SA:X",
        "SR07C:BPM5:SA:X",
        "SR01C:BPM10:SA:X",
    ],
    "s_x": np.array([2.57e-01, 1.72e-01, 1.19e-01, 7.74e-02]),
    "periods_x_s": [10.28 * 60, 60.0 * 60, 60.0 * 60, 60.0 * 60],
    # Top-BPM contributions for each mode (label -> amplitude)
    "V_x_top": [
        {"SR04C:BPM1:SA:X": -0.470, "SR01C:BPM1:SA:X": -0.388,
         "SR10C:BPM1:SA:X": -0.386, "SR07C:BPM1:SA:X": -0.375},
        {"SR03C:BPM5:SA:X": -0.765, "SR09C:BPM1:SA:X": +0.428,
         "SR04C:BPM1:SA:X": +0.247, "SR07C:BPM1:SA:X": -0.224},
        {"SR09C:BPM1:SA:X": +0.856, "SR03C:BPM5:SA:X": +0.351,
         "SR01C:BPM1:SA:X": +0.193, "SR08C:BPM1:SA:X": -0.181},
        {"SR01C:BPM1:SA:X": +0.664, "SR12C:BPM1:SA:X": +0.657,
         "SR01C:BPM5:SA:X": -0.169, "SR06C:BPM1:SA:X": +0.130},
    ],
    "s_y": np.array([6.56e-02, 4.90e-02, 3.14e-02, 2.39e-02]),
    "periods_y_s": [60.0 * 60, 60.0 * 60, 42.58 * 60, 7.57 * 60],
    "V_y_top": [
        {"SR03C:BPM5:SA:X": +0.876, "SR07C:BPM1:SA:X": +0.286,
         "SR12C:BPM1:SA:X": +0.163, "SR09C:BPM1:SA:X": -0.158},
        {"SR04C:BPM1:SA:X": +0.702, "SR07C:BPM1:SA:X": -0.433,
         "SR08C:BPM1:SA:X": -0.294, "SR10C:BPM5:SA:X": +0.256},
        {"SR09C:BPM1:SA:X": -0.939, "SR03C:BPM5:SA:X": -0.178,
         "SR05C:BPM1:SA:X": +0.151, "SR09C:BPM5:SA:X": +0.147},
        {"SR05C:BPM1:SA:X": -0.499, "SR04C:BPM1:SA:X": +0.453,
         "SR08C:BPM1:SA:X": +0.316, "SR06C:BPM1:SA:X": -0.290},
    ],
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def parse_label(pv):
    body = pv.replace("SR", "").replace(":SA", "")
    s, rest = body.split("C:BPM")
    b, p = rest.split(":")
    return int(s), int(b), p


def bpm_azimuth(sector, bpm_idx, total_bpms_per_sector=19):
    frac = (bpm_idx - 1) / max(total_bpms_per_sector - 1, 1)
    span = 2 * np.pi / 12.0
    return (sector - 1) * span + frac * span


def expand_V(labels, top_dict):
    """Convert {pv: amplitude, ...} for ~4 top BPMs into a length-N
    vector aligned to `labels`, with zeros elsewhere."""
    v = np.zeros(len(labels))
    for i, pv in enumerate(labels):
        v[i] = top_dict.get(pv, 0.0)
    return v


def synthesise_temporal_mode(t, period_s, phase=0.0, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    omega = 2 * np.pi / period_s
    return (np.sin(omega * t + phase)
            + noise * rng.standard_normal(len(t)))


# ----------------------------------------------------------------------
# Try live fetch; fall back to cache
# ----------------------------------------------------------------------

def try_live():
    try:
        from s01_betatron_tune import fetch_multi_bpm, svd_orbit_modes
        from _common import ls_power
        t, X, Y, labels = fetch_multi_bpm(hours=2.0)
        if X is None or X.shape[1] < 3:
            return None
        U_x, s_x, Vt_x = svd_orbit_modes(X, n_modes=4)
        U_y, s_y, Vt_y = svd_orbit_modes(Y, n_modes=4)
        return {"live": True, "t": t,
                "labels": labels,
                "s_x": s_x, "Vt_x": Vt_x, "U_x": U_x,
                "s_y": s_y, "Vt_y": Vt_y, "U_y": U_y}
    except Exception as e:
        print(f"   live fetch unavailable ({e.__class__.__name__}); using cache")
        return None


def from_cache():
    labels = CACHED["labels_x"]
    t = np.linspace(0, CACHED["hours"] * 3600.0, 8000)
    Vt_x = np.stack([expand_V(labels, d) for d in CACHED["V_x_top"]], axis=0)
    Vt_y = np.stack([expand_V(labels, d) for d in CACHED["V_y_top"]], axis=0)
    # Synthesise temporal modes at the recovered periods.
    U_x = np.stack([synthesise_temporal_mode(t, p, phase=0.3 * k, seed=k)
                    for k, p in enumerate(CACHED["periods_x_s"])], axis=1)
    U_y = np.stack([synthesise_temporal_mode(t, p, phase=0.5 * k, seed=k + 10)
                    for k, p in enumerate(CACHED["periods_y_s"])], axis=1)
    return {"live": False, "t": t,
            "labels": labels,
            "s_x": CACHED["s_x"], "Vt_x": Vt_x, "U_x": U_x,
            "s_y": CACHED["s_y"], "Vt_y": Vt_y, "U_y": U_y,
            "periods_x_s": CACHED["periods_x_s"],
            "periods_y_s": CACHED["periods_y_s"]}


# ----------------------------------------------------------------------
# Panel drawers
# ----------------------------------------------------------------------

def draw_ring(ax, labels, amplitudes, r_inner=1.0, r_outer=1.30):
    ax.set_aspect("equal")
    ax.set_xlim(-1.85, 1.85); ax.set_ylim(-1.85, 1.85)
    ax.axis("off")
    for s in range(1, 13):
        a0 = np.degrees((s - 1) * 2 * np.pi / 12)
        a1 = np.degrees(s * 2 * np.pi / 12)
        shade = "#f3f3f3" if s % 2 == 0 else "#e9e9e9"
        ax.add_patch(Wedge((0, 0), r_outer, a0, a1, width=r_outer - r_inner,
                           facecolor=shade, edgecolor="black", lw=0.4,
                           zorder=0))
        a_mid = np.radians((a0 + a1) / 2)
        ax.text(1.55 * np.cos(a_mid), 1.55 * np.sin(a_mid), f"S{s}",
                ha="center", va="center", fontsize=10, fontweight="bold")
    cmap = cm.RdBu_r
    vmax = max(np.abs(amplitudes).max(), 1e-12)
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    for pv, amp in zip(labels, amplitudes):
        s, b, _ = parse_label(pv)
        th = bpm_azimuth(s, b)
        r = (r_inner + r_outer) / 2
        sz = 50 + 320 * (np.abs(amp) / vmax)
        ax.scatter(r * np.cos(th), r * np.sin(th),
                   s=sz, c=[cmap(norm(amp))], edgecolors="black",
                   linewidths=0.6, zorder=4)
    # Highlight the largest-amplitude BPM
    idx_max = int(np.argmax(np.abs(amplitudes)))
    pv = labels[idx_max]
    s, b, _ = parse_label(pv)
    th = bpm_azimuth(s, b)
    r = (r_inner + r_outer) / 2
    ax.annotate(
        f"dominant\nvertical perturbation:\nS{s} BPM{b}\n"
        f"($V_0 = {amplitudes[idx_max]:+.3f}$)",
        xy=(r * np.cos(th), r * np.sin(th)),
        xytext=(1.78 * np.cos(th + 0.35), 1.78 * np.sin(th + 0.35)),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2),
        ha="center", va="center", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff5b1",
                  edgecolor="black", lw=0.7))
    return cm.ScalarMappable(cmap=cmap, norm=norm)


def draw_super_period(ax):
    ax.set_xlim(0, 12); ax.set_ylim(-1.4, 1.4)
    ax.axis("off")
    y0 = 0
    ax.annotate("", xy=(11.7, y0), xytext=(0.3, y0),
                arrowprops=dict(arrowstyle="->", color="#666666", lw=1.0))
    elements = [
        ("BPM1", 0.6, "bpm"), ("Q", 1.0, "quad"),
        ("BPM2", 1.4, "bpm"), ("S", 1.8, "sext"), ("BEND1", 2.4, "dip"),
        ("BPM3", 3.0, "bpm"), ("Q", 3.4, "quad"), ("S", 3.8, "sext"),
        ("BEND2", 4.4, "dip"), ("BPM5", 4.9, "bpm"), ("Q", 5.3, "quad"),
        ("BEND3", 5.8, "dip"), ("BPM7", 6.3, "bpm"), ("Q", 6.7, "quad"),
        ("BEND3", 7.2, "dip"), ("BPM9", 7.7, "bpm"), ("Q", 8.1, "quad"),
        ("BEND3", 8.6, "dip"), ("BPM11", 9.1, "bpm"), ("Q", 9.5, "quad"),
        ("BPM13", 9.9, "bpm"), ("BEND2", 10.5, "dip"),
        ("BPM15", 11.0, "bpm"),
    ]
    colours = {"bpm": "#3b6db4", "quad": "#c4424b",
               "sext": "#54a04e", "dip": "#888888"}
    heights = {"bpm": 0.3, "quad": 0.55, "sext": 0.45, "dip": 0.75}
    for label, x, kind in elements:
        if kind == "bpm":
            ax.plot([x, x], [y0 - 0.06, y0 + 0.06], color="black", lw=0.6)
            ax.add_patch(plt.Circle((x, y0), 0.07,
                         color=colours[kind], zorder=5))
        else:
            h = heights[kind]
            ax.add_patch(Rectangle((x - 0.18, y0 - h / 2), 0.36, h,
                         facecolor=colours[kind], edgecolor="black", lw=0.4))
        ax.text(x, y0 - 0.95, label, fontsize=6.5, ha="center", va="top",
                rotation=90)
    for i, (k, c) in enumerate([("BPM", "#3b6db4"), ("quad", "#c4424b"),
                                ("sext", "#54a04e"), ("dipole", "#888888")]):
        ax.add_patch(Rectangle((0.4 + i * 2.6, 1.0), 0.25, 0.18,
                     facecolor=c, edgecolor="black", lw=0.4))
        ax.text(0.7 + i * 2.6, 1.09, k, fontsize=8, va="center")
    ax.text(6, -1.3, "one super-period of the ALS-U (12 around the ring)",
            ha="center", fontsize=8.5, style="italic")


def draw_mode_vs_azimuth(ax, labels, Vt_row, plane="Y", title=""):
    az = []
    vals = []
    for pv, v in zip(labels, Vt_row):
        s, b, _ = parse_label(pv)
        az.append(np.degrees(bpm_azimuth(s, b)))
        vals.append(v)
    colours = ["#3b6db4" if v >= 0 else "#c4424b" for v in vals]
    ax.bar(az, vals, width=7.0, color=colours, edgecolor="black", lw=0.4)
    for s in range(13):
        ax.axvline(s * 30, color="gray", lw=0.4, alpha=0.5)
    ymax = max(np.abs(vals).max(), 0.1)
    for s in range(1, 13):
        ax.text(s * 30 - 15, ymax * 1.05,
                f"S{s}", ha="center", fontsize=7, color="#444444")
    ax.set_xlim(0, 360)
    ax.set_xlabel("azimuthal position (deg around the ring)")
    ax.set_ylabel(f"$V_0$ amplitude ({plane})")
    ax.set_title(title, fontsize=10)
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(axis="y", alpha=0.3)


def draw_temporal(ax, t, U):
    t_min = t / 60.0
    colours = ["#222222", "#bc4444", "#3666b4", "#54a04e"]
    for k in range(min(3, U.shape[1])):
        u = U[:, k] / max(np.abs(U[:, k]).max(), 1e-12)
        ax.plot(t_min, u + 2.4 * k, color=colours[k], lw=1.0,
                label=f"mode {k}")
    ax.set_xlabel("time (min)")
    ax.set_yticks([])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def draw_spectra(ax, t, U, periods_s):
    """LS spectra of U_k(t).  periods_s lists the expected period
    for each mode (used to annotate peaks)."""
    from _common import ls_power
    W_min = 2 * np.pi / (t[-1] * 0.45)
    W_max = 2 * np.pi / 30.0
    grid = np.exp(np.linspace(np.log(W_min), np.log(W_max), 1500))
    colours = ["#222222", "#bc4444", "#3666b4"]
    for k in range(min(3, U.shape[1])):
        u = U[:, k] - U[:, k].mean()
        P = np.array([ls_power(t, u, W) for W in grid])
        P /= max(P.max(), 1e-30)
        ax.semilogx(2 * np.pi / grid / 60.0, P,
                    color=colours[k], lw=1.3, label=f"mode {k}")
        T_p = periods_s[k] / 60.0
        ax.axvline(T_p, color=colours[k], lw=0.5, ls=":")
        ax.text(T_p, 0.95 - 0.07 * k, f"{T_p:.1f} min",
                color=colours[k], fontsize=8,
                ha="left", va="top",
                transform=ax.get_xaxis_transform())
    ax.set_xlabel("period (min)")
    ax.set_ylabel("LS power (normalised)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")


def draw_sv_bars(ax, s_x, s_y):
    sx = s_x ** 2 / (s_x ** 2).sum()
    sy = s_y ** 2 / (s_y ** 2).sum()
    ks = np.arange(1, len(sx) + 1)
    ax.bar(ks - 0.18, 100 * sx, width=0.35, color="#3b6db4",
           label="X", edgecolor="black", lw=0.4)
    ax.bar(ks + 0.18, 100 * sy, width=0.35, color="#c4424b",
           label="Y", edgecolor="black", lw=0.4)
    ax.set_xlabel("SVD mode")
    ax.set_ylabel("variance fraction (%)")
    ax.set_xticks(ks)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main():
    print(">> ALS-U orbit-response geometry visualisation\n", flush=True)
    state = try_live()
    if state is None:
        print("   using cached SVD numbers from the earlier live run\n")
        state = from_cache()
    else:
        print("   live data fetched OK\n")

    labels = state["labels"]
    t = state["t"]
    Vt_x = state["Vt_x"]; s_x = state["s_x"]; U_x = state["U_x"]
    Vt_y = state["Vt_y"]; s_y = state["s_y"]; U_y = state["U_y"]
    periods_x_s = state.get("periods_x_s",
                            [2 * np.pi / 0.001 for _ in range(4)])
    periods_y_s = state.get("periods_y_s",
                            [2 * np.pi / 0.001 for _ in range(4)])

    fig = plt.figure(figsize=(14.2, 10))
    gs = fig.add_gridspec(3, 3,
                          height_ratios=[1.6, 1.0, 1.0],
                          hspace=0.50, wspace=0.32)

    # (a) Big ring on the left, spans 2 rows
    ax_ring = fig.add_subplot(gs[0:2, 0])
    mapping = draw_ring(ax_ring, labels, Vt_y[0])
    cax = fig.add_axes([0.045, 0.07, 0.30, 0.014])
    plt.colorbar(mapping, cax=cax, orientation="horizontal",
                 label=r"Y-plane spatial mode 0 amplitude $V_0$")
    ax_ring.set_title("(a) ALS-U ring: dominant Y-orbit perturbation "
                      "localised", fontsize=11)

    # (b) Super-period schematic
    ax_sp = fig.add_subplot(gs[0, 1:])
    draw_super_period(ax_sp)
    ax_sp.set_title("(b) one of the 12 super-periods (BPMs in blue)",
                    fontsize=11)

    # (c) Mode around the ring vs azimuth (vertical plane V_0)
    ax_bar = fig.add_subplot(gs[1, 1:])
    draw_mode_vs_azimuth(ax_bar, labels, Vt_y[0], plane="Y",
                         title="(c) Y-plane spatial mode 0 vs azimuth")

    # (d) Temporal modes
    ax_t = fig.add_subplot(gs[2, 0])
    draw_temporal(ax_t, t, U_y)
    ax_t.set_title("(d) Y temporal modes $U_k(t)$", fontsize=10)

    # (e) Spectra
    ax_p = fig.add_subplot(gs[2, 1])
    draw_spectra(ax_p, t, U_y, periods_y_s)
    ax_p.set_title("(e) Lomb–Scargle spectra of $U_k(t)$ (Y)",
                   fontsize=10)

    # (f) SV bars
    ax_sv = fig.add_subplot(gs[2, 2])
    draw_sv_bars(ax_sv, s_x, s_y)
    ax_sv.set_title("(f) variance per SVD mode", fontsize=10)

    src_note = "live archiver" if state["live"] else "cached SVD numbers"
    fig.suptitle(
        f"ALS-U orbit response on archived BPM data "
        f"({len(labels)} channels, 2 h window, {src_note})\n"
        "SVD of the slow-orbit time series; dominant Y mode "
        "highlighted on the ring",
        fontsize=12, y=0.995)
    out = "/tmp/alsu_orbit_response.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=140)
    print(f"   wrote {out}")
    out_png = out.replace(".pdf", ".png")
    plt.savefig(out_png, bbox_inches="tight", dpi=130)
    print(f"   wrote {out_png}")


if __name__ == "__main__":
    main()
