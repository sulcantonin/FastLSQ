#!/usr/bin/env python
"""Passive LOCO calibration of the ALS-U closed-orbit response matrix.

Fetches both the corrector-magnet setpoint readbacks (SR{ss}C:HCM{n}:MRV)
and the slow-orbit BPM positions (SR{ss}C:BPM{m}:SA:X) over a multi-hour
archived window from the EPICS archiver appliance via daq.get_pv.
Then computes two response matrices:

  R_meas[i, j]  = Cov(x_BPM_i, theta_HCM_j) / Var(theta_HCM_j)
                  -- the *measured* response matrix from passive
                     correlated drift of the SOFB system.

  R_pred[i, j]  = lattice Green's function (Hill equation) computed
                  analytically from beta_x, mu_x, and Q_x extracted
                  from the parsed ALS-U lattice via pyAT.

The comparison validates (or invalidates) the inverse-problem pipeline
of s01_orbit_inverse.py against a controlled known-physics
ground-truth: every corrector setpoint that the SOFB moves IS the kick
theta_j at lattice position s_j, and the orbit response should follow
the Green's function.

Outputs /tmp/alsu_passive_loco.pdf with three panels:
  (a) R_meas heatmap, (b) R_pred heatmap, (c) per-pair residual
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

from s01_betatron_tune import _ensure_daq, _series_to_arrays
from s01_orbit_inverse import alsu_optics, response_matrix


SECTORS = tuple(range(1, 13))
HCM_INDICES = (1, 4)        # HCM1, HCM4 per sector
BPM_INDICES = (1, 5, 10, 14, 18)


def build_pv_lists():
    corr_pvs = [f"SR{s:02d}C:HCM{n}:MRV"
                for s in SECTORS for n in HCM_INDICES]
    bpm_pvs  = [f"SR{s:02d}C:BPM{n}:SA:X"
                for s in SECTORS for n in BPM_INDICES]
    return corr_pvs, bpm_pvs


def fetch_one(pv, start, end, get_pv):
    try:
        resp, st = get_pv(pv, start=start, end=end)
        if st != 200:
            return None
        t, v = _series_to_arrays(resp)
        if t is None or len(t) < 64:
            return None
        return (t, v)
    except Exception:
        return None


def fetch_passive_loco(hours=6.0, n_workers=1):
    """Pull all corrector and BPM scalars over the last `hours` hours.
    Returns (t_grid, C[t, n_c], B[t, n_b], corr_labels, bpm_labels)."""
    get_pv = _ensure_daq()
    from datetime import datetime, timedelta
    end = datetime.utcnow(); start = end - timedelta(hours=hours)
    corr_pvs, bpm_pvs = build_pv_lists()
    all_pvs = corr_pvs + bpm_pvs
    print(f"   fetching {len(all_pvs)} PVs over {hours:.1f} h ...",
          flush=True)
    raw = {}
    for i, pv in enumerate(all_pvs):
        r = fetch_one(pv, start, end, get_pv)
        if r is not None:
            raw[pv] = r
        if (i + 1) % 20 == 0:
            print(f"     {i+1}/{len(all_pvs)} done, "
                  f"{len(raw)} populated", flush=True)
    # Build aligned matrices
    populated_corrs = [pv for pv in corr_pvs if pv in raw]
    populated_bpms  = [pv for pv in bpm_pvs  if pv in raw]
    print(f"   corrector channels populated: {len(populated_corrs)}/{len(corr_pvs)}")
    print(f"   BPM channels populated:       {len(populated_bpms)}/{len(bpm_pvs)}")
    if not populated_corrs or not populated_bpms:
        return None, None, None, [], []
    # Densest series sets the grid
    densest = max(populated_corrs + populated_bpms,
                  key=lambda pv: len(raw[pv][0]))
    t_grid = raw[densest][0]
    if len(t_grid) > 6000:
        idx = np.linspace(0, len(t_grid) - 1, 6000).astype(int)
        t_grid = t_grid[idx]
    def assemble(labels):
        cols = []
        for pv in labels:
            tp, vp = raw[pv]
            try:
                v_i = np.interp(t_grid, tp, vp)
            except Exception:
                continue
            cols.append(v_i)
        return np.stack(cols, axis=1)
    C = assemble(populated_corrs)
    B = assemble(populated_bpms)
    return t_grid, C, B, populated_corrs, populated_bpms


def measured_response(C, B, mu_reg=1e-6):
    """Solve B = C @ R^T  =>  R[i,j] = response of BPM i to corr j.
    Add ridge mu_reg to handle collinearity (SOFB correctors are
    correlated).  Returns R[n_BPMs, n_corrs]."""
    Cc = C - C.mean(0, keepdims=True)
    Bc = B - B.mean(0, keepdims=True)
    # Ridge: (C^T C + mu I) R = C^T B
    CtC = Cc.T @ Cc
    n = CtC.shape[0]
    rhs = Cc.T @ Bc
    R_T = np.linalg.solve(CtC + mu_reg * np.linalg.norm(CtC) * np.eye(n),
                          rhs)
    return R_T.T


def predicted_response(corr_labels, bpm_labels, opt, plane="x"):
    """Lattice Green's function R[BPM, corrector] using the same
    convention as response_matrix in s01_orbit_inverse."""
    name_to_idx = {n: k for k, n in enumerate(opt["labels"])}
    # Map each label "SR04C:BPM5:SA:X" to lattice index "SR04C:BPM5"
    bpm_idx = [name_to_idx.get(l.replace(":SA:X", "").replace(":SA:Y", ""))
               for l in bpm_labels]
    # Map each corrector "SR04C:HCM1:MRV" to its closest lattice point.
    # We approximate the corrector's azimuthal position by the BPM that
    # sits closest to it in the same sector (BPM(2) ~ HCM1, BPM(15) ~ HCM4).
    HCM_TO_BPM = {1: 2, 4: 15}  # rough placement per super-period
    corr_idx = []
    for L in corr_labels:
        s = int(L[2:4]); n = int(L.split("HCM")[1].split(":")[0])
        bpm_proxy = f"SR{s:02d}C:BPM{HCM_TO_BPM.get(n, 1)}"
        corr_idx.append(name_to_idx.get(bpm_proxy))
    # Build response submatrix.
    bi = np.array([k for k in bpm_idx  if k is not None])
    ci = np.array([k for k in corr_idx if k is not None])
    bpm_ok = [(L, k) for L, k in zip(bpm_labels, bpm_idx) if k is not None]
    corr_ok = [(L, k) for L, k in zip(corr_labels, corr_idx) if k is not None]
    R = response_matrix(opt["beta_x"][bi], opt["mu_x"][bi],
                        opt["beta_x"][ci], opt["mu_x"][ci],
                        opt["Q_x"])
    return R, [b[0] for b in bpm_ok], [c[0] for c in corr_ok]


def compare_response(R_meas, R_pred):
    """Return scalar comparison metrics."""
    # Normalise both to remove scale convention difference
    M = R_meas / max(np.linalg.norm(R_meas), 1e-30)
    P = R_pred / max(np.linalg.norm(R_pred), 1e-30)
    # Allow sign flip (overall amperage-to-radian factor is unknown)
    if np.sum(M * P) < 0:
        M = -M
    corr = float(np.sum(M * P) / max(
        np.linalg.norm(M) * np.linalg.norm(P), 1e-30))
    res_rms = float(np.linalg.norm(M - P) / max(np.linalg.norm(P), 1e-30))
    return corr, res_rms, M, P


def short(pv):
    return pv.replace("SR", "S").replace("C:HCM", "-H") \
             .replace("C:BPM", "-B").replace(":MRV", "") \
             .replace(":SA:X", "")


def make_figure(R_meas_n, R_pred_n, corr_labels, bpm_labels,
                corr, res_rms, out_pdf):
    fig = plt.figure(figsize=(13, 5.5))
    gs = fig.add_gridspec(1, 3, wspace=0.40)
    vmax = max(np.abs(R_meas_n).max(), np.abs(R_pred_n).max()) * 1.05
    titles = [
        f"(a) $R_\\mathrm{{meas}}$ from passive correlated drift",
        f"(b) $R_\\mathrm{{pred}}$ from parsed lattice Green's function",
        f"(c) residual  $R_\\mathrm{{meas}} - R_\\mathrm{{pred}}$\n"
        f"correlation = {corr:+.3f}, "
        f"$\\|\\Delta\\| / \\|R\\| = {res_rms:.3f}$"]
    panels = [R_meas_n, R_pred_n, R_meas_n - R_pred_n]
    for k, (P, title) in enumerate(zip(panels, titles)):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(P, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(np.arange(len(corr_labels)))
        ax.set_xticklabels([short(c) for c in corr_labels],
                           rotation=70, fontsize=6)
        ax.set_yticks(np.arange(len(bpm_labels)))
        ax.set_yticklabels([short(b) for b in bpm_labels], fontsize=6)
        ax.set_xlabel("corrector (HCM)")
        if k == 0: ax.set_ylabel("BPM")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.suptitle("Passive LOCO calibration of the ALS-U orbit-response "
                 "Green's function on live archived data",
                 fontsize=12, y=0.995)
    plt.savefig(out_pdf, bbox_inches="tight", dpi=140)
    plt.savefig(out_pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=130)
    print(f"\n   wrote {out_pdf}")


def main():
    print(">> Passive LOCO: measured vs lattice-predicted response\n",
          flush=True)
    opt = alsu_optics()
    print(f"   parsed lattice: Q_x = {opt['Q_x']:.4f}, "
          f"Q_y = {opt['Q_y']:.4f}, {len(opt['labels'])} BPM positions\n")
    t, C, B, corr_labels, bpm_labels = fetch_passive_loco(hours=6.0)
    if C is None:
        print("   no usable data; aborting")
        return
    print(f"\n   {C.shape[0]} aligned samples,  "
          f"{C.shape[1]} correctors x {B.shape[1]} BPMs")
    print(f"   corrector RMS: {C.std(0).mean():.3e} A,  "
          f"BPM RMS: {B.std(0).mean()*1000:.3f} um")
    # Measured response matrix
    R_meas = measured_response(C, B)
    print(f"   R_meas shape: {R_meas.shape}")
    # Lattice-predicted response matrix
    R_pred, bpm_ok, corr_ok = predicted_response(corr_labels, bpm_labels, opt)
    print(f"   R_pred shape: {R_pred.shape}")
    if R_pred.shape != R_meas.shape:
        # Restrict R_meas to matched labels
        bi = [bpm_labels.index(b) for b in bpm_ok]
        ci = [corr_labels.index(c) for c in corr_ok]
        R_meas = R_meas[np.ix_(bi, ci)]
        bpm_labels = bpm_ok; corr_labels = corr_ok
    corr_val, res_rms, M, P = compare_response(R_meas, R_pred)
    print(f"\n   correlation(R_meas, R_pred) = {corr_val:+.4f}")
    print(f"   relative residual           = {res_rms:.4f}")
    print(f"   ({'STRONG' if corr_val > 0.7 else 'WEAK'} match: "
          f"calibration {'succeeds' if corr_val > 0.7 else 'is ambiguous'})")
    make_figure(M, P, corr_labels, bpm_labels, corr_val, res_rms,
                "/tmp/alsu_passive_loco.pdf")


if __name__ == "__main__":
    main()
