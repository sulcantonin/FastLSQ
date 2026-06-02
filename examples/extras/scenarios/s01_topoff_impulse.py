#!/usr/bin/env python
"""Top-off-impulse calibration of the ALS-U orbit-response inverse.

Every ~60 seconds the storage ring is refilled by the top-off injection
kicker.  The kicker fires at a fixed lattice azimuth (the injection
septum, in sector~1 or 2) for a few hundred nanoseconds; the orbit
shifts by an amount set by the open-loop Hill-equation Green's function;
the slow-orbit feedback (SOFB) then re-zeros the orbit on its closed-
loop time constant (a few hundred milliseconds).  In the brief gap
between "kick has happened" and "SOFB has corrected", the archived
slow-acquisition BPM samples expose the *open-loop* response of the
ring to a known impulse at a known azimuth.

This script:
  1. fetches SR:DCCT over a multi-hour window and detects top-off events
     by thresholding the time-derivative,
  2. fetches multi-BPM SR{ss}C:BPM{m}:SA:X over the same window,
  3. for each detected event, computes the open-loop orbit step
       delta x_i  =  mean( x_i(t_k + 0.1..0.4 s) )
                  -  mean( x_i(t_k - 2.0..-0.2 s) ),
  4. stacks the per-event steps to suppress noise,
  5. projects the averaged delta x onto every column of the lattice
     Green's function (from the parsed ALS-U lattice), and reports the
     calibration score.

Output: /tmp/alsu_topoff_calibration.pdf
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
from matplotlib.patches import Wedge, Circle

from s01_betatron_tune import _ensure_daq, _series_to_arrays
from s01_orbit_inverse import alsu_optics, response_matrix


# ----------------------------------------------------------------------
# Top-off event detection
# ----------------------------------------------------------------------

def detect_topoff_events(t, dcct, min_step_mA=0.05, refractory_s=20.0):
    """Return event timestamps where DCCT jumps upward.

    Parameters
    ----------
    t          : (N,)  time in seconds since some origin
    dcct       : (N,)  beam current in mA
    min_step_mA: minimum positive jump amplitude to count as an event
    refractory_s: minimum gap between consecutive events
    """
    dI = np.diff(dcct)
    dt = np.diff(t)
    rate = dI / np.maximum(dt, 1e-6)        # mA/s
    # Mark candidate samples where the rate exceeds an injection threshold.
    # Use rate (mA/s) rather than absolute step so cadence variations are OK.
    threshold_rate = min_step_mA / 0.5      # 0.05 mA jump in <= 0.5 s
    candidates = np.where(rate > threshold_rate)[0]
    events = []
    last_t = -np.inf
    for k in candidates:
        if t[k] - last_t < refractory_s: continue
        events.append(t[k])
        last_t = t[k]
    return np.asarray(events)


# ----------------------------------------------------------------------
# Per-event orbit step extraction
# ----------------------------------------------------------------------

def stack_orbit_steps(t_bpm, X_mat, events,
                      pre=(-2.0, -0.2),  post=(0.1, 0.4),
                      long_after=(2.0, 6.0)):
    """For each event time t_k, compute
        delta x_i^{(k)} = mean(post window) - mean(pre window)
    using the archived 10-Hz SA data.  Also return the long-after
    average for sanity (should be ~0 since SOFB has compensated)."""
    deltas, deltas_la = [], []
    for tk in events:
        pre_mask  = (t_bpm > tk + pre[0])  & (t_bpm <= tk + pre[1])
        post_mask = (t_bpm > tk + post[0]) & (t_bpm <= tk + post[1])
        la_mask   = (t_bpm > tk + long_after[0]) & (t_bpm <= tk + long_after[1])
        if pre_mask.sum() < 3 or post_mask.sum() < 1:
            continue
        d  = X_mat[post_mask].mean(0) - X_mat[pre_mask].mean(0)
        deltas.append(d)
        if la_mask.sum() >= 3:
            deltas_la.append(X_mat[la_mask].mean(0)
                             - X_mat[pre_mask].mean(0))
    deltas    = np.asarray(deltas)        if deltas    else None
    deltas_la = np.asarray(deltas_la)     if deltas_la else None
    return deltas, deltas_la


# ----------------------------------------------------------------------
# Match against lattice Green's function (same as orbit_inverse.py)
# ----------------------------------------------------------------------

def best_match_kick(delta_x, bpm_labels, opt, plane="x"):
    beta_full = opt[f"beta_{plane}"]
    mu_full   = opt[f"mu_{plane}"]
    Q         = opt[f"Q_{plane}"]
    name_to_idx = {n: k for k, n in enumerate(opt["labels"])}
    rows = []
    for L in bpm_labels:
        L_short = L.replace(":SA:X", "").replace(":SA:Y", "")
        if L_short in name_to_idx:
            rows.append((L, name_to_idx[L_short]))
    if not rows:
        return None
    meas_idx = np.array([r[1] for r in rows])
    R = response_matrix(beta_full[meas_idx], mu_full[meas_idx],
                        beta_full, mu_full, Q)
    cn = np.linalg.norm(R, axis=0); cn[cn < 1e-12] = 1.0
    R_n = R / cn[None, :]
    Vn = delta_x / max(np.linalg.norm(delta_x), 1e-12)
    scores = R_n.T @ Vn
    order = np.argsort(-np.abs(scores))
    top = [(int(j), float(scores[j]), opt["labels"][j])
           for j in order[:15]]
    return top, scores, opt["s"]


# ----------------------------------------------------------------------
# Fetch helpers
# ----------------------------------------------------------------------

PV_DCCT = "SR:DCCT"
SECTORS_BPMS = [(s, n) for s in range(1, 13) for n in (1, 5, 10, 14, 18)]


def fetch_dcct_and_bpms(hours=6.0):
    get_pv = _ensure_daq()
    from datetime import datetime, timedelta
    end = datetime.utcnow(); start = end - timedelta(hours=hours)
    # DCCT
    print(f"   fetching {PV_DCCT} ({hours:.1f} h) ...", flush=True)
    resp, st = get_pv(PV_DCCT, start=start, end=end)
    if st != 200:
        raise RuntimeError(f"DCCT fetch failed (HTTP {st})")
    t_dc, v_dc = _series_to_arrays(resp)
    print(f"     {len(t_dc)} samples")
    # BPMs
    bpm_data = {}
    pv_list = [f"SR{s:02d}C:BPM{n}:SA:X" for s, n in SECTORS_BPMS]
    print(f"   fetching {len(pv_list)} BPM PVs ...", flush=True)
    for i, pv in enumerate(pv_list):
        try:
            r, ss = get_pv(pv, start=start, end=end)
            if ss != 200: continue
            tp, vp = _series_to_arrays(r)
            if tp is not None and len(tp) > 100:
                bpm_data[pv] = (tp, vp)
        except Exception:
            continue
        if (i + 1) % 20 == 0:
            print(f"     {i+1}/{len(pv_list)} fetched, "
                  f"{len(bpm_data)} populated")
    return t_dc, v_dc, bpm_data


def align_bpms(bpm_data):
    """Interpolate each BPM onto a common grid."""
    densest = max(bpm_data.values(), key=lambda kv: len(kv[0]))
    t_grid = densest[0]
    labels = []; cols = []
    for pv, (tp, vp) in bpm_data.items():
        try:
            v_i = np.interp(t_grid, tp, vp)
        except Exception:
            continue
        labels.append(pv); cols.append(v_i)
    X = np.stack(cols, axis=1) if cols else None
    return t_grid, X, labels


# ----------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------

def draw_ring_score(ax, opt, top, scores):
    ax.set_aspect("equal")
    ax.set_xlim(-1.9, 1.9); ax.set_ylim(-1.9, 1.9); ax.axis("off")
    r_in, r_out = 1.0, 1.35
    for s in range(1, 13):
        a0 = np.degrees((s - 1) * 2 * np.pi / 12)
        a1 = np.degrees(s * 2 * np.pi / 12)
        ax.add_patch(Wedge((0, 0), r_out, a0, a1, width=r_out - r_in,
                           facecolor="#eeeeee" if s % 2 else "#dddddd",
                           edgecolor="black", lw=0.4, zorder=0))
        a_mid = np.radians((a0 + a1) / 2)
        ax.text(1.55 * np.cos(a_mid), 1.55 * np.sin(a_mid), f"S{s}",
                ha="center", va="center", fontsize=10, fontweight="bold")
    s_lat = opt["s"]; C = s_lat[-1]
    theta_all = 2 * np.pi * (s_lat / C)
    r_mid = (r_in + r_out) / 2
    for th in theta_all:
        ax.plot(r_mid * np.cos(th), r_mid * np.sin(th), ".",
                color="#cccccc", markersize=2.0, zorder=2)
    # Top matches sized + coloured by score
    smax = max(abs(s) for _, s, _ in top[:8])
    for j, sc, lab in top[:8]:
        th = 2 * np.pi * (s_lat[j] / C)
        ax.scatter(r_mid * np.cos(th), r_mid * np.sin(th),
                   s=60 + 380 * (abs(sc) / smax),
                   c=[plt.cm.RdBu_r(0.5 + 0.5 * sc / smax)],
                   edgecolors="black", linewidths=0.8, zorder=5)
    j_best, sc_best, lab_best = top[0]
    th_b = 2 * np.pi * (s_lat[j_best] / C)
    ax.annotate(f"best match:\n{lab_best}\nscore = {sc_best:+.3f}",
                xy=(r_mid * np.cos(th_b), r_mid * np.sin(th_b)),
                xytext=(1.85 * np.cos(th_b + 0.35),
                        1.85 * np.sin(th_b + 0.35)),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4),
                ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#fff5b1", edgecolor="black", lw=0.7))


def make_figure(t_dc, v_dc, events, t_bpm, X, labels, opt,
                deltas, mean_delta, deltas_la, top, scores,
                out_pdf="/tmp/alsu_topoff_calibration.pdf"):
    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.3, 1.0],
                          hspace=0.40, wspace=0.30)
    # (a) DCCT trace + detected events
    ax = fig.add_subplot(gs[0, :])
    ax.plot((t_dc - t_dc[0]) / 60.0, v_dc, lw=0.6, color="#222222")
    if events.size:
        for tk in events:
            ax.axvline((tk - t_dc[0]) / 60.0,
                       color="#bc4444", lw=0.4, alpha=0.55)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("SR:DCCT (mA)")
    ax.set_title(f"(a) beam current and detected top-off events "
                 f"(n = {events.size})", fontsize=10)
    ax.grid(alpha=0.3)
    # (b) ring with kick localisation
    ax = fig.add_subplot(gs[1, 0])
    draw_ring_score(ax, opt, top, scores)
    ax.set_title("(b) ring location of best-match kick", fontsize=10)
    # (c) per-event orbit steps (heatmap of delta_x_i^(k))
    ax = fig.add_subplot(gs[1, 1])
    vmax = np.percentile(np.abs(deltas), 95) if deltas is not None else 1
    im = ax.imshow(deltas.T, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax,
                   extent=[0, deltas.shape[0], len(labels), 0])
    ax.set_xlabel("event index")
    ax.set_ylabel("BPM channel")
    ax.set_title("(c) per-event orbit step $\\Delta x_i^{(k)}$",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="m")
    # (d) score landscape
    ax = fig.add_subplot(gs[1, 2])
    C = opt["s"][-1]
    ax.plot(360 * opt["s"] / C, np.abs(scores), color="#222222", lw=0.8)
    j_best = top[0][0]
    ax.axvline(360 * opt["s"][j_best] / C,
               color="#bc4444", lw=1.4, ls="--",
               label=f"best @ {top[0][2]}")
    for s in range(1, 13):
        ax.axvline(s * 30, color="gray", lw=0.4, alpha=0.4)
    ax.set_xlabel("azimuth (deg)")
    ax.set_ylabel(r"$|\Delta\bar x \cdot R(:,j)|$")
    ax.set_title("(d) projection-score landscape", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # (e) mean delta_x compared to predicted at best j
    ax = fig.add_subplot(gs[2, :])
    n = len(labels); idx = np.arange(n)
    Vn = mean_delta / np.linalg.norm(mean_delta)
    name_to_idx = {n: k for k, n in enumerate(opt["labels"])}
    meas_idx = np.array([name_to_idx[L.replace(":SA:X", "")
                         .replace(":SA:Y", "")]
                         for L in labels
                         if L.replace(":SA:X","").replace(":SA:Y","")
                         in name_to_idx])
    R_col = response_matrix(opt["beta_x"][meas_idx], opt["mu_x"][meas_idx],
                            np.array([opt["beta_x"][top[0][0]]]),
                            np.array([opt["mu_x"][top[0][0]]]),
                            opt["Q_x"]).ravel()
    R_col /= max(np.linalg.norm(R_col), 1e-12)
    sign = np.sign(Vn @ R_col) or 1.0
    ax.plot(idx, Vn, "o-", color="black", lw=1.0,
            label="averaged orbit step $\\bar{\\Delta x}/\\|\\bar{\\Delta x}\\|$")
    ax.plot(idx, sign * R_col, "s--", color="#bc4444", lw=1.0,
            label=f"lattice Green's function at j = {top[0][2]}")
    ax.set_xticks(idx)
    short = [L.replace("SR", "S").replace("C:BPM", "-B").replace(":SA:X", "")
             for L in labels]
    ax.set_xticklabels(short, rotation=60, ha="right", fontsize=7)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("normalised amplitude")
    ax.set_title(
        f"(e) measured impulse response vs lattice prediction\n"
        f"correlation = {Vn @ (sign * R_col):.4f},  "
        f"n = {events.size} stacked events",
        fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle("Top-off injection as a periodic known impulse: "
                 "calibration of the ALS-U orbit-response inverse on "
                 "live archived data", fontsize=12, y=0.995)
    plt.savefig(out_pdf, bbox_inches="tight", dpi=140)
    plt.savefig(out_pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=130)
    print(f"\n   wrote {out_pdf}")


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main():
    print(">> Top-off impulse calibration of ALS-U orbit inverse\n",
          flush=True)
    opt = alsu_optics()
    print(f"   lattice tunes (parsed):  Q_x = {opt['Q_x']:.4f},  "
          f"Q_y = {opt['Q_y']:.4f}\n")
    t_dc, v_dc, bpm_raw = fetch_dcct_and_bpms(hours=6.0)
    print(f"\n   DCCT samples: {len(t_dc)}")
    events = detect_topoff_events(t_dc, v_dc)
    print(f"   detected top-off events: {events.size}")
    if events.size == 0:
        print("   no events detected; aborting")
        return
    t_bpm, X, labels = align_bpms(bpm_raw)
    if X is None:
        print("   no BPM data; aborting"); return
    print(f"   BPM matrix:  {X.shape[0]} samples x {X.shape[1]} channels")
    deltas, deltas_la = stack_orbit_steps(t_bpm, X, events)
    if deltas is None or deltas.shape[0] < 4:
        print("   too few clean events; aborting"); return
    print(f"   {deltas.shape[0]} clean event windows extracted")
    mean_delta = np.mean(deltas, axis=0)
    se = np.std(deltas, axis=0) / np.sqrt(deltas.shape[0])
    snr = np.abs(mean_delta) / (se + 1e-12)
    print(f"   mean orbit step  range = [{mean_delta.min()*1e6:+.2f}, "
          f"{mean_delta.max()*1e6:+.2f}] um")
    print(f"   per-BPM SNR (|mean|/SE) range = [{snr.min():.2f}, "
          f"{snr.max():.2f}]")
    top, scores, s_lat = best_match_kick(mean_delta, labels, opt,
                                          plane="x")
    print(f"\n   top-5 kick locations from lattice Green's function:")
    for j, sc, lab in top[:5]:
        print(f"     {lab:24s} score = {sc:+.4f}")
    # Correlation against best match
    name_to_idx = {n: k for k, n in enumerate(opt["labels"])}
    meas_idx = np.array([name_to_idx[L.replace(":SA:X", "")
                         .replace(":SA:Y", "")]
                         for L in labels
                         if L.replace(":SA:X","").replace(":SA:Y","")
                         in name_to_idx])
    R_col = response_matrix(opt["beta_x"][meas_idx], opt["mu_x"][meas_idx],
                            np.array([opt["beta_x"][top[0][0]]]),
                            np.array([opt["mu_x"][top[0][0]]]),
                            opt["Q_x"]).ravel()
    R_col /= max(np.linalg.norm(R_col), 1e-12)
    Vn = mean_delta / max(np.linalg.norm(mean_delta), 1e-12)
    sign = np.sign(Vn @ R_col) or 1.0
    corr = float(Vn @ (sign * R_col))
    print(f"\n   correlation(stacked step, Green's function at best j) "
          f"= {corr:+.4f}")
    print(f"   ({'CALIBRATED' if abs(corr) > 0.7 else 'AMBIGUOUS'})")
    make_figure(t_dc, v_dc, events, t_bpm, X, labels, opt,
                deltas, mean_delta, deltas_la, top, scores)


if __name__ == "__main__":
    main()
