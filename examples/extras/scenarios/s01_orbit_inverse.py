#!/usr/bin/env python
"""Closed-orbit-response inverse on real ALS-U BPM data.

PDE: linearised Hill equation for the closed orbit,
        x''(s) + K(s) x(s) = sum_j theta_j delta(s - s_j),
with K(s) read directly off the parsed ALS-U v21 4-raft Superbend
MATLAB lattice via pyAT.  The Green's function of this PDE for a
dipole kick at location s_j produces, at any BPM at s_i, the closed-
orbit excursion
   R_{ij} = sqrt(beta(s_i) beta(s_j)) / (2 sin(pi Q))
           * cos(pi Q - |mu(s_i) - mu(s_j)|),
where beta(s), mu(s) and Q are the Twiss parameters of the parsed
lattice (obtained from pyAT's linopt4 evaluator).

Given live archived slow-orbit data x(s_i, t) we compute the dominant
spatial mode V_0 by SVD, then *project* V_0 onto every column of R to
identify the lattice location s_j whose Green's function best
explains the observed pattern.  This localises the perturbation to
a specific azimuthal position on the ring.

This is the §3 Loop-C symbolic-prior-inverse pattern (recover a
source field via the PDE Green's function) applied to live ALS-U
data, using the actual machine lattice as the PDE coefficient field.

Outputs:
   /tmp/alsu_orbit_inverse.pdf   (and .png)
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
from matplotlib.patches import Wedge, Rectangle, Circle
import matplotlib.cm as cm

from _alsu_lattice import parse_elements, build_one_superperiod
from s01_betatron_tune import fetch_multi_bpm, svd_orbit_modes
import at


LATTICE_PATH = ("/Users/asulc/PycharmProjects/signal_to_vector/lattice/"
                "ALS_U_v21_4raft_SB_updtBPM.m")


# ----------------------------------------------------------------------
# Build the lattice and extract optics at each BPM in the full 12-SP ring
# ----------------------------------------------------------------------

def alsu_optics():
    """Return (s_positions, beta_x, mu_x, beta_y, mu_y, Q_x, Q_y,
    bpm_labels) for every BPM marker in the *full* 12-superperiod ring.
    Phases mu accumulate continuously around the ring."""
    elems = parse_elements(LATTICE_PATH)
    sp = build_one_superperiod(elems, variant="SUP")
    ring = at.Lattice(sp, energy=2.0e9, periodicity=12, name="ALS-U SUP")
    ring.disable_6d()
    bpm_idx = [i for i, e in enumerate(ring) if e.FamName.startswith("BPM")]
    ld0, rd, eld = at.get_optics(ring, refpts=bpm_idx, method=at.linopt4)[:3]
    # Per-SP optics at the n_bpm_per_sp BPMs in one super-period.
    beta_x_sp = np.array([row["beta"][0] for row in eld])
    beta_y_sp = np.array([row["beta"][1] for row in eld])
    mu_x_sp   = np.array([row["mu"][0]   for row in eld])
    mu_y_sp   = np.array([row["mu"][1]   for row in eld])
    s_sp      = np.array([row["s_pos"]   for row in eld])
    # SP-fractional tune from rd (it's the phase advance / 2pi of the
    # one-superperiod map but the Lattice has periodicity=12, so rd
    # returns the *per-SP* phase already as Q_sp).
    Q_x_sp = float(rd["tune"][0])
    Q_y_sp = float(rd["tune"][1])
    Q_x_tot = 12 * Q_x_sp
    Q_y_tot = 12 * Q_y_sp
    # Build the full 228-BPM list by replicating each SP 12 times.
    # Phase at BPM i in sector k is mu_x_sp[i] + (k-1) * 2 pi Q_x_sp.
    # (beta is periodic so beta[i] is the same in every sector.)
    sectors = np.arange(1, 13)
    S, I = np.meshgrid(sectors, np.arange(len(eld)), indexing="ij")
    BX = np.tile(beta_x_sp, len(sectors))
    BY = np.tile(beta_y_sp, len(sectors))
    MX = np.tile(mu_x_sp, len(sectors)) \
         + (S.flatten() - 1) * 2 * np.pi * Q_x_sp
    MY = np.tile(mu_y_sp, len(sectors)) \
         + (S.flatten() - 1) * 2 * np.pi * Q_y_sp
    s_full = np.tile(s_sp, len(sectors)) \
             + (S.flatten() - 1) * s_sp[-1]
    labels = [f"SR{int(s):02d}C:BPM{i+1}"
              for s, i in zip(S.flatten(), I.flatten())]
    return {"s": s_full, "beta_x": BX, "mu_x": MX,
            "beta_y": BY, "mu_y": MY,
            "Q_x": Q_x_tot, "Q_y": Q_y_tot,
            "labels": labels, "n_bpm_sp": len(eld)}


# ----------------------------------------------------------------------
# Closed-orbit response matrix (Green's function for Hill)
# ----------------------------------------------------------------------

def response_matrix(beta_i, mu_i, beta_j, mu_j, Q):
    """R[i,j] = orbit at BPM i due to a unit dipole kick at point j.
    Hill-equation Green's function for an isolated kick in a periodic
    ring with tune Q."""
    bi = np.asarray(beta_i)[:, None]
    bj = np.asarray(beta_j)[None, :]
    mi = np.asarray(mu_i)[:, None]
    mj = np.asarray(mu_j)[None, :]
    # |mu_i - mu_j| reduced mod 2 pi
    dmu = np.abs(mi - mj)
    dmu = np.minimum(dmu, 2 * np.pi * Q - dmu)
    R = np.sqrt(bi * bj) / (2 * np.sin(np.pi * Q)) \
        * np.cos(np.pi * Q - dmu)
    return R


# ----------------------------------------------------------------------
# Match measured spatial mode against response columns
# ----------------------------------------------------------------------

def match_kick_locations(V0_meas, labels_meas, opt, plane="y"):
    """Project V0_meas (length n_BPMs measured) onto every column of
    the response matrix R[meas-BPMs, ALL-BPMs].  Return ranked list
    (idx_in_full, score, label_full)."""
    beta_full = opt["beta_y" if plane == "y" else "beta_x"]
    mu_full   = opt["mu_y"   if plane == "y" else "mu_x"]
    Q         = opt["Q_y"    if plane == "y" else "Q_x"]
    labels_full = opt["labels"]
    # Match measured labels (like "SR03C:BPM5:SA:X") to indices in the
    # full lattice BPM list (like "SR03C:BPM5").
    name_to_idx = {n: k for k, n in enumerate(labels_full)}
    rows = []
    for L in labels_meas:
        L_short = L.replace(":SA:X", "").replace(":SA:Y", "")
        if L_short not in name_to_idx:
            continue
        rows.append((L, name_to_idx[L_short]))
    if not rows:
        return None, None
    meas_idx = np.array([r[1] for r in rows])
    # Build response matrix: rows = measured BPMs, columns = all BPMs (candidate kicks)
    R = response_matrix(beta_full[meas_idx], mu_full[meas_idx],
                        beta_full, mu_full, Q)
    # Normalise each column then project the measured mode onto each.
    col_norms = np.linalg.norm(R, axis=0)
    col_norms[col_norms < 1e-12] = 1.0
    R_n = R / col_norms[None, :]
    V_n = V0_meas / max(np.linalg.norm(V0_meas), 1e-12)
    scores = R_n.T @ V_n          # length = n_kick-candidates
    order = np.argsort(-np.abs(scores))
    ranked = [(int(j), float(scores[j]), labels_full[j]) for j in order[:20]]
    return ranked, (R, R_n, scores, meas_idx)


# ----------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------

def draw_ring_with_source(ax, opt, V0_meas, ranked,
                          plane="Y"):
    ax.set_aspect("equal")
    ax.set_xlim(-1.95, 1.95); ax.set_ylim(-1.95, 1.95)
    ax.axis("off")
    r_inner, r_outer = 1.0, 1.35
    for s in range(1, 13):
        a0 = np.degrees((s - 1) * 2 * np.pi / 12)
        a1 = np.degrees(s * 2 * np.pi / 12)
        shade = "#f3f3f3" if s % 2 == 0 else "#e9e9e9"
        ax.add_patch(Wedge((0, 0), r_outer, a0, a1,
                           width=r_outer - r_inner,
                           facecolor=shade, edgecolor="black", lw=0.4,
                           zorder=0))
        a_mid = np.radians((a0 + a1) / 2)
        ax.text(1.65 * np.cos(a_mid), 1.65 * np.sin(a_mid), f"S{s}",
                ha="center", va="center", fontsize=10, fontweight="bold")
    # Convert lattice s position to azimuth on the drawn ring
    s_lat = opt["s"]
    C = s_lat[-1]  # full ring circumference (approximate)
    theta_all = 2 * np.pi * (s_lat / C)
    # Draw all 228 candidate BPM positions as small dots
    r_mid = (r_inner + r_outer) / 2
    for th in theta_all:
        ax.plot(r_mid * np.cos(th), r_mid * np.sin(th), ".",
                color="#cccccc", markersize=2.5, zorder=2)
    # Highlight the kick localisation: top-N ranked candidates
    cmap = cm.RdBu_r
    top = ranked[:8]
    abs_top = max(abs(t[1]) for t in top)
    for rank, (j, score, lab) in enumerate(top):
        th = 2 * np.pi * (s_lat[j] / C)
        size = 60 + 350 * (abs(score) / abs_top)
        ax.scatter(r_mid * np.cos(th), r_mid * np.sin(th),
                   s=size, c=[cmap(0.5 + 0.5 * score / abs_top)],
                   edgecolors="black", linewidths=0.8, zorder=5)
    # Best-match arrow
    j_best, s_best, lab_best = top[0]
    th_b = 2 * np.pi * (s_lat[j_best] / C)
    ax.annotate(
        f"best-match kick:\n{lab_best}\n"
        f"score = {s_best:+.3f}",
        xy=(r_mid * np.cos(th_b), r_mid * np.sin(th_b)),
        xytext=(1.80 * np.cos(th_b + 0.42), 1.80 * np.sin(th_b + 0.42)),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4),
        ha="center", va="center", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff5b1",
                  edgecolor="black", lw=0.7))
    ax.set_title(f"({plane}) closed-orbit-response inverse: "
                 "kick localisation on the ring", fontsize=11)


def draw_match(ax, V0_meas, R_n_col_best, labels_meas):
    """Side-by-side: measured V_0 (data) vs predicted Green's function
    (lattice)."""
    n = len(V0_meas)
    idx = np.arange(n)
    V_meas = V0_meas / np.linalg.norm(V0_meas)
    V_pred = R_n_col_best
    ax.plot(idx, V_meas, "o-", color="black", label="measured $V_0$",
            lw=1.0, markersize=4)
    ax.plot(idx, V_pred, "s--", color="#bc4444",
            label="predicted Green's func", lw=1.0, markersize=4)
    ax.set_xticks(idx)
    short = [l.replace("SR", "S").replace("C:BPM", "-B").replace(":SA:X", "")
             .replace(":SA:Y", "") for l in labels_meas]
    ax.set_xticklabels(short, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("normalised amplitude")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main():
    print(">> ALS-U closed-orbit-response inverse on real BPM data\n",
          flush=True)
    opt = alsu_optics()
    print(f"   lattice: total tunes Q_x = {opt['Q_x']:.4f},  "
          f"Q_y = {opt['Q_y']:.4f}")
    print(f"   {len(opt['labels'])} BPM positions in the full ring")
    print(f"   beta_x range = [{opt['beta_x'].min():.2f}, "
          f"{opt['beta_x'].max():.2f}] m,  "
          f"beta_y range = [{opt['beta_y'].min():.2f}, "
          f"{opt['beta_y'].max():.2f}] m\n")
    print("   fetching live BPM SA data ...", flush=True)
    t, X_mat, Y_mat, labels = fetch_multi_bpm(hours=2.0)
    if X_mat is None or X_mat.shape[1] < 3:
        print("   live fetch returned no usable data; aborting "
              "(re-open the SSH tunnel and retry)")
        return
    U_y, s_y, Vt_y = svd_orbit_modes(Y_mat, n_modes=4)
    U_x, s_x, Vt_x = svd_orbit_modes(X_mat, n_modes=4)
    print(f"   Y-plane variance fractions: "
          f"{[f'{100*v:.1f}%' for v in s_y**2 / (s_y**2).sum()]}")
    # Match Y-plane V_0 against the lattice response columns.
    V0y = Vt_y[0]
    ranked_y, mats_y = match_kick_locations(V0y, labels, opt, plane="y")
    if ranked_y is None:
        print("   no matched BPMs between measurement and lattice list")
        return
    print(f"\n   top-5 kick-location matches (Y plane):")
    for j, score, lab in ranked_y[:5]:
        print(f"     {lab:18s}  score = {score:+.4f}")
    # Match X plane too.
    V0x = Vt_x[0]
    ranked_x, mats_x = match_kick_locations(V0x, labels, opt, plane="x")
    print(f"\n   top-5 kick-location matches (X plane):")
    for j, score, lab in ranked_x[:5]:
        print(f"     {lab:18s}  score = {score:+.4f}")

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0],
                          hspace=0.35, wspace=0.30)
    # Big ring (top-left)
    ax_ring = fig.add_subplot(gs[0, 0])
    draw_ring_with_source(ax_ring, opt, V0y, ranked_y, plane="Y")
    # Beta function as a function of azimuth (top-right)
    ax_beta = fig.add_subplot(gs[0, 1])
    C = opt["s"][-1]
    az = 360.0 * (opt["s"][:opt["n_bpm_sp"]] / opt["s"][opt["n_bpm_sp"]-1]) / 12.0
    ax_beta.plot(az, opt["beta_x"][:opt["n_bpm_sp"]],
                 "o-", color="#3b6db4", lw=1.0, label=r"$\beta_x$")
    ax_beta.plot(az, opt["beta_y"][:opt["n_bpm_sp"]],
                 "s-", color="#c4424b", lw=1.0, label=r"$\beta_y$")
    ax_beta.set_xlabel("azimuth within one super-period (deg)")
    ax_beta.set_ylabel(r"$\beta(s)$ [m]")
    ax_beta.set_title(r"(b) Twiss $\beta_{x,y}(s)$ from parsed lattice")
    ax_beta.legend(fontsize=9); ax_beta.grid(alpha=0.3)
    # Measured V_0 vs predicted (bottom-left)
    ax_m = fig.add_subplot(gs[1, 0])
    R, R_n, scores, meas_idx = mats_y
    j_best = ranked_y[0][0]
    V_pred = R_n[:, j_best]
    sign = np.sign(np.dot(V0y / np.linalg.norm(V0y), V_pred)) or 1.0
    draw_match(ax_m, V0y, sign * V_pred, labels)
    ax_m.set_title(f"(c) Y mode 0 vs Green's function at best-match j = "
                   f"{ranked_y[0][2]}", fontsize=10)
    # Score landscape around the ring (bottom-right)
    ax_sc = fig.add_subplot(gs[1, 1])
    s_lat_full = opt["s"]
    C = s_lat_full[-1]
    ax_sc.plot(360 * s_lat_full / C, np.abs(scores), color="#222222", lw=0.8)
    # Mark sector boundaries
    for s in range(1, 13):
        ax_sc.axvline(s * 30, color="gray", lw=0.4, alpha=0.4)
    j_best = ranked_y[0][0]
    th_best = 360 * s_lat_full[j_best] / C
    ax_sc.axvline(th_best, color="#bc4444", lw=1.2, ls="--",
                  label=f"best j @ {ranked_y[0][2]}")
    ax_sc.set_xlabel("azimuth around the ring (deg)")
    ax_sc.set_ylabel(r"$|V_0 \cdot R[:,j]|$ projection score")
    ax_sc.set_title("(d) match score vs kick-location azimuth", fontsize=10)
    ax_sc.legend(fontsize=9, loc="upper right")
    ax_sc.grid(alpha=0.3)
    fig.suptitle("ALS-U closed-orbit-response PDE inverse on live BPM data\n"
                 "(parsed lattice → Hill-equation Green's function → "
                 "magnet-location identification)", fontsize=12, y=0.995)
    out = "/tmp/alsu_orbit_inverse.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=140)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=130)
    print(f"\n   wrote {out}")


if __name__ == "__main__":
    main()
