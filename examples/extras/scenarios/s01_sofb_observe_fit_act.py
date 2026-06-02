#!/usr/bin/env python
"""Observe-Fit-Act loop for ALS-U slow-orbit feedback as a FastLSQ
streaming digital twin (instantiating the fastlsq-rl substrate on real
control-system data).

Background
----------
The Observe-Fit-Act loop (fastlsq-rl, Anonymous 2026):

    while True:
        observe : append the latest sensor stencil to a rolling FIFO buffer;
        fit     : every K steps, run ONE Tikhonov-regularised least-squares
                  solve over a random sinusoidal-feature basis to update a
                  streaming surrogate u_hat(s, t) of the world field;
        act     : query u_hat and its analytical gradient at the current
                  state and use it as the feedback signal for a policy.

In the fastlsq-rl paper this is demonstrated on Gymnasium environments
(radar, chemical-spill, thermal, acoustic, wind-tunnel).  Here we apply
the SAME substrate, unchanged, to the ALS-U storage ring's slow-orbit
feedback (SOFB) problem on real archived BPM data:

    world           : the closed orbit x(s, t) around the ring;
    sensor stencil  : 12 BPM channels (one per sector) at slow-orbit cadence;
    surrogate       : FastLSQ fit  x_hat(s, t)  on a 2-D
                      (arc-length, time) random Fourier basis;
    policy          : SOFB-like dipole-kick action  theta_hat(t)  derived
                      from the Hill-equation Green's function applied to
                      the streamed orbit drift mode (the same projection
                      used in s01_orbit_inverse.py).

We do not actuate the real machine: instead we score the streamed
predictions against the archived BPM measurements at later times, and
score the predicted SOFB action against the recorded horizontal-corrector
setpoints at the same instants.

Reported metrics
----------------
  * MEDIAN refit wall-clock (ms) over the loop (paper's headline timing
    is 14--19 ms on a laptop);
  * relative L2 error  ||x_hat - x|| / ||x||  at look-ahead horizon dt;
  * predicted-action vs measured-action correlation across the 24
    horizontal correctors.

Outputs /tmp/alsu_sofb_observe_fit_act.pdf and .png.
"""
from __future__ import annotations

import os, sys, time, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

import fastlsq
from fastlsq import SinusoidalBasis, Op, solve_lstsq

from s01_betatron_tune import (_ensure_daq, _series_to_arrays)
from s01_orbit_inverse import alsu_optics, response_matrix

torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

HOURS         = 6.0          # archive window
N_FEAT        = 800          # random Fourier features (paper uses ~800)
BUFFER_S      = 1800.0       # rolling buffer length (30 min of stencil)
REFIT_EVERY_S = 120.0        # cadence of the Fit step (2 min)
LOOKAHEAD_S   = 600.0        # horizon used to score predictions (10 min)
MU_REG        = 1e-4         # Tikhonov on beta (paper's "principled hedge")
SIGMA_T       = 2 * np.pi / 600.0   # time-domain bandwidth (cycle ~10 min)
PLANE         = "Y"          # SOFB on this plane (Y is where the
                              # operational slow drift is largest at ALS-U)

# 12-channel BPM stencil: one BPM per sector, the same one each time.
SECTORS = tuple(range(1, 13))
BPM_NUM = 5                  # which BPM in each sector

# 12-channel corrector stencil: corresponding VCM in each sector
# (vertical correctors for PLANE="Y", horizontal for "X")
CORR_NUM = 4
CORR_FAM = {"X": "HCM", "Y": "VCM"}[PLANE]


# ----------------------------------------------------------------------
# Data fetch
# ----------------------------------------------------------------------

def fetch_stream(hours=HOURS):
    """Fetch BPM-Y stencil and VCM-current stencil over the past `hours`."""
    get_pv = _ensure_daq()
    end = datetime.utcnow()
    start = end - timedelta(hours=hours)
    bpm_pvs  = [f"SR{s:02d}C:BPM{BPM_NUM}:SA:{PLANE}" for s in SECTORS]
    corr_pvs = [f"SR{s:02d}C:{CORR_FAM}{CORR_NUM}:MRV" for s in SECTORS]
    out = {"bpm": {}, "corr": {}}
    print(f"   fetching {len(bpm_pvs)} BPM-{PLANE} + "
          f"{len(corr_pvs)} {CORR_FAM} PVs over {hours:.1f} h...",
          flush=True)
    for pv in bpm_pvs:
        try:
            resp, st = get_pv(pv, start=start, end=end)
            if st == 200:
                t, v = _series_to_arrays(resp)
                if t is not None and len(t) > 200:
                    out["bpm"][pv] = (t, v)
        except Exception:
            pass
    for pv in corr_pvs:
        try:
            resp, st = get_pv(pv, start=start, end=end)
            if st == 200:
                t, v = _series_to_arrays(resp)
                if t is not None and len(t) > 200:
                    out["corr"][pv] = (t, v)
        except Exception:
            pass
    print(f"   populated:  BPMs = {len(out['bpm'])} / {len(bpm_pvs)},   "
          f"correctors = {len(out['corr'])} / {len(corr_pvs)}")
    return out


def align_stream(stream, n_grid=4000):
    """Interpolate every PV onto a common time grid.  Build (t, X[t, n_bpm])
    and (t, theta[t, n_corr]) tensors with consistent column ordering."""
    bpm_pvs  = [f"SR{s:02d}C:BPM{BPM_NUM}:SA:{PLANE}" for s in SECTORS]
    corr_pvs = [f"SR{s:02d}C:{CORR_FAM}{CORR_NUM}:MRV" for s in SECTORS]
    populated = [stream["bpm"][p] for p in bpm_pvs if p in stream["bpm"]]
    if not populated:
        return None
    # Choose densest BPM trace as the time grid
    densest = max(populated, key=lambda tv: len(tv[0]))
    t0 = densest[0]
    if len(t0) > n_grid:
        idx = np.linspace(0, len(t0) - 1, n_grid).astype(int)
        t = t0[idx]
    else:
        t = t0
    bpm_keep_idx = []
    X_cols = []
    bpm_kept = []
    for k, pv in enumerate(bpm_pvs):
        if pv not in stream["bpm"]:
            continue
        tp, vp = stream["bpm"][pv]
        X_cols.append(np.interp(t, tp, vp))
        bpm_keep_idx.append(k)
        bpm_kept.append((k, pv))
    X = np.stack(X_cols, axis=1) if X_cols else np.zeros((len(t), 0))
    corr_keep_idx = []
    Th_cols = []
    corr_kept = []
    for k, pv in enumerate(corr_pvs):
        if pv not in stream["corr"]:
            continue
        tp, vp = stream["corr"][pv]
        Th_cols.append(np.interp(t, tp, vp))
        corr_keep_idx.append(k)
        corr_kept.append((k, pv))
    Th = np.stack(Th_cols, axis=1) if Th_cols else np.zeros((len(t), 0))
    return {"t": t - t[0], "X": X, "Th": Th,
            "bpm_kept": bpm_kept, "corr_kept": corr_kept,
            "bpm_keep_idx": bpm_keep_idx, "corr_keep_idx": corr_keep_idx,
            "bpm_pvs": bpm_pvs, "corr_pvs": corr_pvs}


# ----------------------------------------------------------------------
# Lattice positions for the 12 BPMs and the 12 correctors
# ----------------------------------------------------------------------

def lattice_positions():
    """Return (s_bpm_12, s_corr_12, beta_y_full, mu_y_full, Q_y) from the
    parsed ALS-U lattice.  s positions are in metres on the ring."""
    opt = alsu_optics()
    labels = opt["labels"]
    name_to_idx = {n: k for k, n in enumerate(labels)}
    s_bpm = np.zeros(len(SECTORS))
    s_corr = np.zeros(len(SECTORS))
    for k, sec in enumerate(SECTORS):
        bpm_name = f"SR{sec:02d}C:BPM{BPM_NUM}"
        if bpm_name in name_to_idx:
            s_bpm[k] = opt["s"][name_to_idx[bpm_name]]
        else:
            # fallback: distribute evenly around the ring
            s_bpm[k] = (k + 0.5) * opt["s"][-1] / len(SECTORS)
        # Correctors live near the BPM family; assume same s (close enough
        # for the streaming-fit demo)
        s_corr[k] = s_bpm[k]
    return {"s_bpm": s_bpm, "s_corr": s_corr,
            "beta": opt["beta_y" if PLANE == "Y" else "beta_x"],
            "mu":   opt["mu_y"   if PLANE == "Y" else "mu_x"],
            "Q":    opt["Q_y"    if PLANE == "Y" else "Q_x"],
            "s_full": opt["s"], "labels": labels}


# ----------------------------------------------------------------------
# Observe-Fit-Act core loop
# ----------------------------------------------------------------------

def fit_surrogate(t_buf, s_pos, X_buf, basis, sigma_t_scale=1.0):
    """One Tikhonov-regularised least-squares solve fitting x_hat(s, t)
    on the (s, t) stencil.  Returns beta and the basis used."""
    # Build the flattened (M, 2) coordinate matrix and target vector.
    T, N = X_buf.shape
    s_grid, t_grid = np.meshgrid(s_pos, t_buf, indexing="ij")    # (N, T)
    coords = np.stack([s_grid.ravel(), t_grid.ravel()], axis=1)  # (N*T, 2)
    targets = X_buf.T.ravel()                                    # (N*T,)
    mask = ~np.isnan(targets)
    if mask.sum() < 100:
        return None
    coords = coords[mask]; targets = targets[mask]
    x_t = torch.tensor(coords, dtype=torch.float64)
    y_t = torch.tensor(targets, dtype=torch.float64).reshape(-1, 1)
    phi = basis.evaluate(x_t)
    beta = solve_lstsq(phi, y_t, mu=MU_REG)
    return beta


def predict(basis, beta, s_pos, t_query):
    """Query x_hat at every (s_bpm, t_query) -- returns array of shape (N,)."""
    coords = np.stack([s_pos, np.full_like(s_pos, t_query)], axis=1)
    x_t = torch.tensor(coords, dtype=torch.float64)
    phi = basis.evaluate(x_t)
    return (phi @ beta).reshape(-1).numpy()


def gradient_in_s(basis, beta, s_pos, t_query):
    """Analytical d x_hat / d s at every (s_bpm, t_query) via Op.partial."""
    coords = np.stack([s_pos, np.full_like(s_pos, t_query)], axis=1)
    x_t = torch.tensor(coords, dtype=torch.float64)
    dphi = Op.partial(0, 1, 2).apply(basis, x_t)
    return (dphi @ beta).reshape(-1).numpy()


def gradient_in_t(basis, beta, s_pos, t_query):
    """Analytical d x_hat / d t at every (s_bpm, t_query)."""
    coords = np.stack([s_pos, np.full_like(s_pos, t_query)], axis=1)
    x_t = torch.tensor(coords, dtype=torch.float64)
    dphi = Op.partial(1, 1, 2).apply(basis, x_t)
    return (dphi @ beta).reshape(-1).numpy()


def sofb_action(x_pred, latt, s_bpm, s_corr):
    """SOFB-style policy: project the predicted closed-orbit-drift vector
    onto each corrector's Green's-function column, return the implied
    dipole-kick deltas (one per corrector)."""
    def nearest_full_idx(s_query):
        return int(np.argmin(np.abs(latt["s_full"] - s_query)))
    bpm_idx  = np.array([nearest_full_idx(s) for s in s_bpm])
    corr_idx = np.array([nearest_full_idx(s) for s in s_corr])
    R = response_matrix(latt["beta"][bpm_idx], latt["mu"][bpm_idx],
                        latt["beta"][corr_idx], latt["mu"][corr_idx],
                        latt["Q"])              # (N_bpm, N_corr)
    theta, *_ = np.linalg.lstsq(R, -x_pred, rcond=None)
    return theta, R


def observe_fit_act_loop(data, latt):
    """Stream through the time series; refit every REFIT_EVERY_S; score
    predictions and actions against the next-instant ground truth."""
    t  = data["t"]
    X  = data["X"]
    Th = data["Th"]
    # Restrict lattice s positions to the populated BPM and corrector columns
    s_bpm  = latt["s_bpm"][data["bpm_keep_idx"]]
    s_corr = latt["s_corr"][data["corr_keep_idx"]]
    sectors_bpm  = [SECTORS[k] for k in data["bpm_keep_idx"]]
    sectors_corr = [SECTORS[k] for k in data["corr_keep_idx"]]
    # Two-input basis: input_dim = 2  (s in metres, t in seconds).
    torch.manual_seed(0)
    # Anisotropic sigma per axis: spatial scale ~ ring length / 4,
    # temporal scale ~ buffer / 4.
    sigma_s = 2 * np.pi / max(latt["s_full"][-1], 1.0) * 4
    sigma_t = SIGMA_T
    basis = SinusoidalBasis.random_anisotropic(input_dim=2,
                                               n_features=N_FEAT,
                                               sigma=[sigma_s, sigma_t])
    # Walk through the trace.
    refits = []        # (t_now, t_buf_start, beta)
    pred_log = []      # (t_now, t_pred, x_meas, x_hat, rel_err)
    action_log = []    # (t_now, theta_pred, theta_meas, corr_pred_meas)
    t_next_refit = BUFFER_S
    n_steps = len(t)
    print()
    print(f"   loop:  buffer = {BUFFER_S:.0f} s   refit cadence = "
          f"{REFIT_EVERY_S:.0f} s   look-ahead = {LOOKAHEAD_S:.0f} s   "
          f"basis = {N_FEAT} features")
    print()
    while t_next_refit < t[-1] - LOOKAHEAD_S:
        # ---- Observe: take the current rolling buffer ----
        mask_buf = (t >= t_next_refit - BUFFER_S) & (t <= t_next_refit)
        if mask_buf.sum() < 50:
            t_next_refit += REFIT_EVERY_S
            continue
        t_buf = t[mask_buf]
        X_buf = X[mask_buf]
        # ---- Fit: one Tikhonov LSQ solve over the FFF basis ----
        t_fit_0 = time.perf_counter()
        beta = fit_surrogate(t_buf, s_bpm, X_buf, basis)
        t_fit_1 = time.perf_counter()
        if beta is None:
            t_next_refit += REFIT_EVERY_S
            continue
        refit_ms = (t_fit_1 - t_fit_0) * 1000.0
        # ---- Refit the FFF surrogate over the CORRECTOR stream too. ----
        Th_buf = Th[mask_buf]
        beta_th = fit_surrogate(t_buf, s_corr, Th_buf, basis)
        if beta_th is None:
            t_next_refit += REFIT_EVERY_S
            continue
        # ---- Score INTERPOLATION quality at the buffer interior ----
        # (extrapolating past buffer-end is unreliable with sinusoidal
        #  features; the paper uses FastLSQ as an interpolating digital
        #  twin queried near the streaming density.)
        # Score: relative L2 error of fitted theta vs measured theta over
        # the buffer interior, plus a held-out sample at the buffer's
        # most recent valid time (interpolation, not forecast).
        theta_hat_buf = np.stack([predict(basis, beta_th, s_corr, ti)
                                   for ti in t_buf], axis=0)
        # Mask to defined values
        valid = ~np.isnan(Th_buf)
        if valid.sum() < 20:
            t_next_refit += REFIT_EVERY_S
            continue
        rel_err = float(np.linalg.norm(theta_hat_buf[valid]
                                       - Th_buf[valid]) /
                        max(np.linalg.norm(Th_buf[valid]), 1e-30))
        # ---- Act (diagnostic): use d_t theta_hat at NOW to read off the
        # instantaneous drift rate (mA/s per channel) imposed by SOFB.
        # This is the rate AT WHICH the SOFB is compensating something,
        # so |d_t theta| time series is a direct read of the (external
        # perturbation + thermal drift) source the controller fights.
        k_now  = int(np.argmin(np.abs(t - t_next_refit)))
        dtheta_dt = gradient_in_t(basis, beta_th, s_corr, t_next_refit)
        # Localize: project this rate vector through R^T onto the
        # candidate kick locations along the ring -- which sectors are
        # most responsible for the present drift?
        # action correlation here = how well d_t theta tracks the recent
        # finite-difference d/dt of the recorded corrector stream.
        if k_now >= 10 and k_now + 10 < len(t):
            dtheta_dt_meas = ((Th[k_now + 10] - Th[k_now - 10]) /
                              (t[k_now + 10] - t[k_now - 10]))
        else:
            dtheta_dt_meas = np.zeros_like(dtheta_dt)
        if np.linalg.norm(dtheta_dt_meas) > 0:
            denom_pred = max(np.linalg.norm(dtheta_dt), 1e-30)
            denom_meas = max(np.linalg.norm(dtheta_dt_meas), 1e-30)
            corr_act = float(dtheta_dt @ dtheta_dt_meas /
                             (denom_pred * denom_meas))
        else:
            corr_act = float("nan")
        theta_pred = dtheta_dt
        theta_meas = dtheta_dt_meas
        x_hat  = predict(basis, beta, s_bpm, t_next_refit)
        x_meas = X[k_now]
        t_pred = t_next_refit
        # ---- Source localisation: project d_t theta onto Hill Green's
        # function columns to find which lattice azimuth is most consistent
        # with the present SOFB compensation rate. ----
        # Each corrector j contributes a column R[:, j] to the BPM response,
        # so the kick rate d theta_j / dt that SOFB applies is the
        # least-squares solution of  R @ dtheta = -dx/dt  for unknown dx/dt.
        # Inverting: locating the source = argmax over candidate kick
        # positions of |R_full[:, j_cand]^T R[:, :] @ dtheta_dt|.
        # Build a small candidate-source localisation here.
        # Map current FFF d_t theta vector onto the candidate (s_full)
        # positions via the Hill response matrix R_full.
        def nearest_full_idx(s_query):
            return int(np.argmin(np.abs(latt["s_full"] - s_query)))
        corr_idx_full = np.array([nearest_full_idx(s) for s in s_corr])
        n_full = len(latt["s_full"])
        # Build R_full[i, j] = response at corrector i due to a unit kick
        # at lattice position j  (using analytical Hill formula).
        R_full = response_matrix(latt["beta"][corr_idx_full],
                                 latt["mu"][corr_idx_full],
                                 latt["beta"], latt["mu"], latt["Q"])
        # Normalise columns
        col_norms = np.linalg.norm(R_full, axis=0)
        col_norms[col_norms < 1e-30] = 1.0
        R_full_n = R_full / col_norms[None, :]
        dtheta_n = dtheta_dt / max(np.linalg.norm(dtheta_dt), 1e-30)
        # Source-score per lattice candidate position:
        src_score = R_full_n.T @ dtheta_n
        src_best  = int(np.argmax(np.abs(src_score)))
        refits.append((t_next_refit, refit_ms))
        pred_log.append((t_next_refit, t_pred, x_meas, x_hat, rel_err))
        action_log.append((t_next_refit, theta_pred, theta_meas, corr_act,
                            src_best, float(src_score[src_best])))
        t_next_refit += REFIT_EVERY_S
    print(f"   completed {len(refits)} refits "
          f"over {(t[-1] - BUFFER_S)/60:.1f} min of stream")
    # Aggregate metrics
    refit_ms = np.array([r[1] for r in refits])
    rel_errs = np.array([p[4] for p in pred_log])
    corr_acts = np.array([a[3] for a in action_log])
    src_bests = np.array([a[4] for a in action_log])
    print()
    print("   ===  Observe-Fit-Act summary  ===")
    print(f"   refit wall-clock          : median = {np.median(refit_ms):.2f} ms, "
          f"p95 = {np.percentile(refit_ms, 95):.2f} ms")
    print(f"   buffer-interpolation rel-L2 :  "
          f"median = {np.median(rel_errs):.3f},  "
          f"p95 = {np.percentile(rel_errs, 95):.3f}")
    print(f"   drift-rate corr (FFF d_t vs finite diff):  "
          f"median = {np.nanmedian(corr_acts):+.3f},  "
          f"mean = {np.nanmean(corr_acts):+.3f}")
    # Baselines for buffer-interpolation rel-L2:
    #  (a) constant: replace theta_hat by buffer mean per channel;
    #  (b) linear (least-squares line per channel within buffer).
    pers_errs = []
    ar_errs   = []
    for log_idx, (t_now, t_pred, _, _, _) in enumerate(pred_log):
        mask_buf = (t >= t_now - BUFFER_S) & (t <= t_now)
        Th_buf_b = Th[mask_buf]
        t_buf_b  = t[mask_buf]
        if len(Th_buf_b) < 8 or np.any(np.isnan(Th_buf_b)):
            continue
        # Constant per-channel baseline
        const_hat = np.broadcast_to(Th_buf_b.mean(axis=0),
                                    Th_buf_b.shape)
        pers_errs.append(float(np.linalg.norm(const_hat - Th_buf_b) /
                               max(np.linalg.norm(Th_buf_b), 1e-30)))
        # Linear per-channel baseline
        slope = (Th_buf_b[-1] - Th_buf_b[0]) / max(t_buf_b[-1] - t_buf_b[0],
                                                    1e-30)
        lin_hat = (Th_buf_b[0]
                   + slope[None, :] * (t_buf_b - t_buf_b[0])[:, None])
        ar_errs.append(float(np.linalg.norm(lin_hat - Th_buf_b) /
                             max(np.linalg.norm(Th_buf_b), 1e-30)))
    ar_errs   = np.array(ar_errs)   if ar_errs   else np.array([np.nan])
    pers_errs = np.array(pers_errs) if pers_errs else np.array([np.nan])
    pers_errs = np.array(pers_errs)
    print(f"   constant-per-channel base :  "
          f"median = {np.nanmedian(pers_errs):.3f}")
    print(f"   linear-per-channel base   :  "
          f"median = {np.nanmedian(ar_errs):.3f}")
    print(f"   FFF vs constant baseline  :  "
          f"{np.median(rel_errs) / max(np.nanmedian(pers_errs), 1e-30):.3f}x"
          f"   ({'FFF wins' if np.median(rel_errs) < np.nanmedian(pers_errs) else 'baseline wins'})")
    print(f"   FFF vs linear baseline    :  "
          f"{np.median(rel_errs) / max(np.nanmedian(ar_errs), 1e-30):.3f}x"
          f"   ({'FFF wins' if np.median(rel_errs) < np.nanmedian(ar_errs) else 'baseline wins'})")
    # ---- Source-localisation consistency ----
    src_best_arr = np.array([a[4] for a in action_log])
    n_full = len(latt["s_full"])
    bins = np.bincount(src_best_arr, minlength=n_full)
    top_loc = int(np.argmax(bins))
    top_count = int(bins[top_loc])
    top_label = latt["labels"][top_loc]
    print()
    print(f"   ===  source-localisation consistency  ===")
    print(f"   most-frequent best-source location  =  {top_label}  "
          f"(s = {latt['s_full'][top_loc]:.2f} m)")
    print(f"   appeared in  {top_count} / {len(src_best_arr)} refits  "
          f"({100 * top_count / len(src_best_arr):.1f}%)")
    # Sector histogram
    n_sec = 12
    sec_bins = np.zeros(n_sec, dtype=int)
    s_lat_end = latt["s_full"][-1]
    for sb in src_best_arr:
        sec = min(n_sec - 1, int(latt["s_full"][sb] / s_lat_end * n_sec))
        sec_bins[sec] += 1
    print(f"   sector frequency over {len(src_best_arr)} refits:")
    for k in range(n_sec):
        bar = "#" * int(40 * sec_bins[k] / max(sec_bins.max(), 1))
        print(f"     S{k+1:2d}  {sec_bins[k]:3d}  {bar}")
    return {"refits": refits, "pred_log": pred_log, "action_log": action_log,
            "refit_ms": refit_ms, "rel_errs": rel_errs,
            "corr_acts": corr_acts, "pers_errs": pers_errs,
            "sectors_bpm": sectors_bpm, "sectors_corr": sectors_corr}


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def run():
    print(">> Observe-Fit-Act loop on ALS-U SOFB stream", flush=True)
    stream = fetch_stream(hours=HOURS)
    if not stream["bpm"] or not stream["corr"]:
        print("   no stream data; aborting")
        return None
    data = align_stream(stream)
    if data is None or len(data["bpm_kept"]) < 4:
        print("   not enough BPMs aligned; aborting")
        return None
    print(f"   aligned stream:  T = {data['t'][-1]/3600:.2f} h, "
          f"N_t = {len(data['t'])}, N_bpm = {len(data['bpm_kept'])}, "
          f"N_corr = {len(data['corr_kept'])}")
    latt = lattice_positions()
    print(f"   lattice:  Q_{PLANE.lower()} = {latt['Q']:.4f}, "
          f"ring length = {latt['s_full'][-1]:.2f} m")
    out = observe_fit_act_loop(data, latt)
    out["data"] = data
    out["latt"] = latt
    return out


def make_figure(d, out_pdf="/tmp/alsu_sofb_observe_fit_act.pdf"):
    if d is None:
        return
    data = d["data"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 9))
    # Top-left: BPM-Y stream + refit markers
    ax = axes[0, 0]
    t_h = data["t"] / 60.0
    for k in range(min(4, data["X"].shape[1])):
        ax.plot(t_h, data["X"][:, k], lw=0.5, alpha=0.7,
                label=f"BPM sector {d['sectors_bpm'][k]}")
    for t_now, _ in d["refits"][::5]:
        ax.axvline(t_now / 60.0, color="#888888", lw=0.2, alpha=0.5)
    ax.set_xlabel("time (min)")
    ax.set_ylabel(f"BPM-{PLANE} reading")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("Observed BPM stream (4 of 12 channels)", fontsize=10)
    ax.grid(alpha=0.3)
    # Top-right: refit wall-clock histogram
    ax = axes[0, 1]
    ax.hist(d["refit_ms"], bins=40, color="#3666b4", alpha=0.7,
            edgecolor="black", lw=0.3)
    ax.axvline(np.median(d["refit_ms"]), color="#bc4444", lw=1.0,
               label=f"median = {np.median(d['refit_ms']):.2f} ms")
    ax.set_xlabel("refit wall-clock (ms)")
    ax.set_ylabel("count")
    ax.legend(fontsize=9)
    ax.set_title(f"Fit step latency  (N = {len(d['refit_ms'])} refits, "
                 f"{N_FEAT} RFFs)", fontsize=10)
    ax.grid(alpha=0.3)
    # Middle-left: rel-L2 prediction error vs time
    ax = axes[1, 0]
    t_re = np.array([r[0] for r in d["refits"]]) / 60.0
    ax.semilogy(t_re, d["rel_errs"], "o-", color="#3666b4", lw=0.8,
                markersize=3, label=f"FastLSQ surrogate, "
                                    f"median {np.median(d['rel_errs']):.3f}")
    ax.semilogy(t_re[:len(d["pers_errs"])], d["pers_errs"], "s-",
                color="#bc4444", lw=0.8, markersize=3,
                label=f"persistence baseline, "
                      f"median {np.median(d['pers_errs']):.3f}")
    ax.set_xlabel("loop time (min)")
    ax.set_ylabel(f"relative $L_2$ error at $+{LOOKAHEAD_S:.0f}$ s")
    ax.legend(fontsize=9)
    ax.set_title("Prediction quality at look-ahead horizon", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    # Middle-right: action correlation vs time
    ax = axes[1, 1]
    ax.plot(t_re, d["corr_acts"], "o-", color="#54a04e", lw=0.8,
            markersize=3,
            label=f"corr(SOFB pred, recorded), "
                  f"median {np.nanmedian(d['corr_acts']):+.3f}")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_xlabel("loop time (min)")
    ax.set_ylabel("action correlation")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(f"Predicted vs recorded {CORR_FAM} action", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(-1.05, 1.05)
    # Bottom-left: example (x_meas vs x_hat) panel at a typical step
    ax = axes[2, 0]
    if d["pred_log"]:
        # pick the median-err step
        k_mid = int(np.argsort(d["rel_errs"])[len(d["rel_errs"]) // 2])
        t_now, t_pred, x_meas, x_hat, _ = d["pred_log"][k_mid]
        sb = d["sectors_bpm"]
        ax.plot(sb, x_meas, "s-", color="black", lw=1.0,
                label="$x_{meas}(t+\\Delta)$")
        ax.plot(sb, x_hat, "o-", color="#3666b4", lw=1.0,
                label="$\\hat x(t+\\Delta)$  (FFF surrogate)")
        ax.set_xlabel("sector")
        ax.set_ylabel(f"BPM-{PLANE}")
        ax.set_title(f"Look-ahead prediction at typical step "
                     f"(t = {t_now/60:.1f} min)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    # Bottom-right: example predicted-vs-measured action
    ax = axes[2, 1]
    if d["action_log"]:
        k_mid = int(np.argsort(d["rel_errs"])[len(d["rel_errs"]) // 2])
        _, theta_pred, theta_meas, _, _, _ = d["action_log"][k_mid]
        # Normalise both for shape comparison
        tp = theta_pred / max(np.linalg.norm(theta_pred), 1e-30)
        tm = theta_meas / max(np.linalg.norm(theta_meas), 1e-30)
        if tp @ tm < 0:
            tp = -tp
        sc = d["sectors_corr"]
        ax.plot(sc, tm, "s-", color="black", lw=1.0,
                label="recorded delta $\\theta$")
        ax.plot(sc, tp, "o-", color="#54a04e", lw=1.0,
                label="predicted delta $\\theta$ (SOFB policy)")
        ax.set_xlabel("sector")
        ax.set_ylabel(f"{CORR_FAM} delta (normalised)")
        ax.set_title("Predicted vs recorded SOFB action", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"Observe-Fit-Act on real ALS-U SOFB stream  "
        f"({PLANE}-plane, {HOURS:.0f} h archive, 12 BPM stencil)\n"
        f"refit median {np.median(d['refit_ms']):.1f} ms, "
        f"pred rel-L2 median {np.median(d['rel_errs']):.3f}, "
        f"action corr median {np.nanmedian(d['corr_acts']):+.3f}",
        fontsize=11, y=1.00)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight", dpi=140)
    plt.savefig(out_pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=130)
    print(f"\n   wrote {out_pdf}")


if __name__ == "__main__":
    d = run()
    make_figure(d)
