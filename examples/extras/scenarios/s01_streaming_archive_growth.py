#!/usr/bin/env python
"""Streaming-archive Observe-Fit-Act learning curve on the ALS-U SOFB.

Why this script exists
----------------------
The earlier ``s01_sofb_observe_fit_act.py`` runs a fixed-buffer rolling
refit over an archived window; it shows that one Tikhonov solve over a
random-Fourier basis fits each window in ~15 ms (matching the
fastlsq-rl paper's headline timing), but the "exploration" element of
the original Observe-Fit-Act loop --- the fact that the agent's actions
*change* what data arrives next --- is absent.  Without an actuator the
loop is purely passive streaming inference.

This script handles the half of the gap that does not require an
actuator: **as more archive data accumulates over wall-clock time, does
the FFF surrogate of the SOFB stream get better at predicting the next
chunk?**  Concretely:

  * We download a long archive window (12 h) once.
  * We replay it as if we were online: at simulated wall-clock time T,
    only the data with archive timestamp <= T is "known".
  * The loop refits the FFF surrogate on the cumulative buffer, then
    SCORES it on the immediately-following chunk that the next refit
    would consume.
  * We trace median rel-L2 on the *next* chunk vs. cumulative buffer
    length, producing a learning curve.

The natural drivers of new information in the archive --- Earth tides
(M2/S2/N2/K1/O1), Touschek-lifetime drift, top-off injection pattern,
diurnal thermal cycle --- play the role of the exploration policy.  The
agent isn't choosing actions; physics is.

Outputs /tmp/alsu_streaming_archive_growth.pdf and .png.
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

from s01_betatron_tune import _ensure_daq, _series_to_arrays

torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

HOURS         = 12.0
N_FEAT        = 800
EVAL_CHUNK_S  = 600.0            # 10-minute next-chunk evaluation horizon
REFIT_EVERY_S = 600.0            # one refit per chunk
PLANE         = "Y"
SECTORS       = tuple(range(1, 13))
CORR_NUM      = 4
CORR_FAM      = {"X": "HCM", "Y": "VCM"}[PLANE]
MU_REG        = 1e-4


# ----------------------------------------------------------------------
# Data fetch (one-shot, then replayed online)
# ----------------------------------------------------------------------

def fetch_corrector_stream(hours=HOURS):
    get_pv = _ensure_daq()
    end = datetime.utcnow()
    start = end - timedelta(hours=hours)
    corr_pvs = [f"SR{s:02d}C:{CORR_FAM}{CORR_NUM}:MRV" for s in SECTORS]
    out = {}
    print(f"   fetching {len(corr_pvs)} {CORR_FAM} PVs over {hours:.1f} h...",
          flush=True)
    for pv in corr_pvs:
        try:
            resp, st = get_pv(pv, start=start, end=end)
            if st == 200:
                t, v = _series_to_arrays(resp)
                if t is not None and len(t) > 200:
                    out[pv] = (t, v)
        except Exception:
            pass
    print(f"   populated correctors: {len(out)} / {len(corr_pvs)}")
    return out, corr_pvs


def align(stream, corr_pvs, n_grid=8000):
    populated = [stream[p] for p in corr_pvs if p in stream]
    if not populated:
        return None
    densest = max(populated, key=lambda tv: len(tv[0]))
    t0 = densest[0]
    if len(t0) > n_grid:
        idx = np.linspace(0, len(t0) - 1, n_grid).astype(int)
        t = t0[idx]
    else:
        t = t0
    keep_idx = []
    Th_cols = []
    for k, pv in enumerate(corr_pvs):
        if pv not in stream:
            continue
        tp, vp = stream[pv]
        Th_cols.append(np.interp(t, tp, vp))
        keep_idx.append(k)
    Th = np.stack(Th_cols, axis=1) if Th_cols else np.zeros((len(t), 0))
    s_corr = np.array([SECTORS[k] for k in keep_idx], dtype=float)
    s_corr = s_corr / float(max(SECTORS)) * 200.0   # arbitrary spatial scale
    return {"t": t - t[0], "Th": Th, "keep_idx": keep_idx, "s_corr": s_corr}


# ----------------------------------------------------------------------
# Streaming-growth learning-curve loop
# ----------------------------------------------------------------------

def fit_streaming(t_buf, s_corr, Th_buf, basis):
    T = t_buf
    N = len(s_corr)
    s_grid, t_grid = np.meshgrid(s_corr, T, indexing="ij")
    coords = np.stack([s_grid.ravel(), t_grid.ravel()], axis=1)
    targets = Th_buf.T.ravel()
    mask = ~np.isnan(targets)
    if mask.sum() < 200:
        return None
    coords = coords[mask]; targets = targets[mask]
    x_t = torch.tensor(coords, dtype=torch.float64)
    y_t = torch.tensor(targets, dtype=torch.float64).reshape(-1, 1)
    phi = basis.evaluate(x_t)
    return solve_lstsq(phi, y_t, mu=MU_REG)


def predict_block(basis, beta, s_corr, t_query):
    """Predict the full Theta at every (s_corr, t_query) for an array of
    times t_query: returns (M_t, N_corr) array."""
    M_t = len(t_query)
    N   = len(s_corr)
    s_grid, t_grid = np.meshgrid(s_corr, t_query, indexing="ij")
    coords = np.stack([s_grid.ravel(), t_grid.ravel()], axis=1)
    x_t = torch.tensor(coords, dtype=torch.float64)
    phi = basis.evaluate(x_t)
    pred = (phi @ beta).reshape(N, M_t).numpy().T
    return pred


def streaming_loop(data):
    t   = data["t"]
    Th  = data["Th"]
    s_c = data["s_corr"]
    torch.manual_seed(0)
    T_total = float(t[-1])
    sigma_s = 2 * np.pi / max(s_c.max(), 1.0) * 4
    sigma_t = 2 * np.pi / max(T_total * 0.2, 1.0)
    basis = SinusoidalBasis.random_anisotropic(input_dim=2,
                                               n_features=N_FEAT,
                                               sigma=[sigma_s, sigma_t])
    # The "online" loop: at each step we treat data up to t_now as known
    # and score predictions on the next chunk [t_now, t_now + EVAL_CHUNK_S].
    t_now = REFIT_EVERY_S
    log = []
    print()
    print("   loop:  cumulative buffer grows by "
          f"{REFIT_EVERY_S/60:.1f} min per refit,  "
          f"eval horizon = {EVAL_CHUNK_S/60:.1f} min")
    print()
    print(f"   {'cum_h':>8s}  {'fit_ms':>8s}  {'eval_rel_L2':>12s}  "
          f"{'const_rel_L2':>13s}  {'lin_rel_L2':>11s}")
    while t_now < T_total - EVAL_CHUNK_S:
        # ---- Observe: all archive data up to t_now ----
        mask_buf  = t <= t_now
        mask_eval = (t > t_now) & (t <= t_now + EVAL_CHUNK_S)
        if mask_buf.sum() < 64 or mask_eval.sum() < 8:
            t_now += REFIT_EVERY_S; continue
        t_buf  = t[mask_buf]
        Th_buf = Th[mask_buf]
        t_eval  = t[mask_eval]
        Th_eval = Th[mask_eval]
        # ---- Fit: one Tikhonov solve over the growing buffer ----
        t0 = time.perf_counter()
        beta = fit_streaming(t_buf, s_c, Th_buf, basis)
        t1 = time.perf_counter()
        if beta is None:
            t_now += REFIT_EVERY_S; continue
        fit_ms = (t1 - t0) * 1000.0
        # ---- Score: predict the next chunk and compare ----
        Th_pred = predict_block(basis, beta, s_c, t_eval)
        denom = max(np.linalg.norm(Th_eval), 1e-30)
        eval_rel = float(np.linalg.norm(Th_pred - Th_eval) / denom)
        # Baselines on the same eval chunk:
        # constant = repeat the last-observed value per channel
        const_pred = np.broadcast_to(Th_buf[-1], Th_eval.shape)
        const_rel = float(np.linalg.norm(const_pred - Th_eval) / denom)
        # linear = extrapolate last-window slope per channel
        if len(Th_buf) > 8:
            slope = (Th_buf[-1] - Th_buf[-8]) / max(t_buf[-1] - t_buf[-8],
                                                     1e-30)
            lin_pred = (Th_buf[-1] + slope[None, :] *
                        (t_eval - t_buf[-1])[:, None])
            lin_rel = float(np.linalg.norm(lin_pred - Th_eval) / denom)
        else:
            lin_rel = float("nan")
        log.append({"t_now": t_now, "fit_ms": fit_ms,
                    "buf_h": t_now / 3600.0,
                    "buf_n": int(mask_buf.sum()),
                    "eval_rel": eval_rel, "const_rel": const_rel,
                    "lin_rel": lin_rel})
        print(f"   {t_now/3600.0:8.2f}  {fit_ms:8.1f}  "
              f"{eval_rel:12.3e}  {const_rel:13.3e}  {lin_rel:11.3e}")
        t_now += REFIT_EVERY_S
    return log


def run():
    print(">> Streaming-archive growth study on real ALS-U SOFB stream",
          flush=True)
    stream, pvs = fetch_corrector_stream(hours=HOURS)
    if not stream:
        print("   no archive data fetched; aborting")
        return None
    data = align(stream, pvs)
    if data is None:
        print("   alignment failed; aborting")
        return None
    print(f"   aligned: T = {data['t'][-1]/3600:.2f} h, "
          f"N_t = {len(data['t'])}, N_corr = {data['Th'].shape[1]}")
    log = streaming_loop(data)
    if not log:
        print("   loop produced no data; aborting")
        return None
    # Summary
    bufs   = np.array([r["buf_h"]   for r in log])
    fffs   = np.array([r["eval_rel"]  for r in log])
    cons   = np.array([r["const_rel"] for r in log])
    lins   = np.array([r["lin_rel"]   for r in log])
    fit_ms = np.array([r["fit_ms"]    for r in log])
    print()
    print("   ===  summary  ===")
    print(f"   refit ms          : median {np.median(fit_ms):.1f},   "
          f"p95 {np.percentile(fit_ms, 95):.1f}")
    print(f"   median next-chunk rel-L2:")
    print(f"     FFF cumulative  : {np.median(fffs):.3f}")
    print(f"     last-value      : {np.median(cons):.3f}")
    print(f"     slope-extrap    : {np.nanmedian(lins):.3f}")
    # learning-curve trend: ratio (FFF rel-L2 at first 1/4 of run)
    #                       /   (FFF rel-L2 at last  1/4 of run)
    q = max(1, len(log) // 4)
    early = np.median(fffs[:q])
    late  = np.median(fffs[-q:])
    print()
    print(f"   FFF early-buffer error (first {q} refits)  =  {early:.3e}")
    print(f"   FFF late-buffer  error (last  {q} refits)  =  {late:.3e}")
    print(f"   ratio (early / late)                       =  "
          f"{early/max(late, 1e-30):.2f}x"
          f"   ({'improvement' if late < early else 'no improvement'})")
    return {"data": data, "log": log,
            "bufs": bufs, "fffs": fffs,
            "cons": cons, "lins": lins, "fit_ms": fit_ms,
            "early": early, "late": late}


def make_figure(d, out_pdf="/tmp/alsu_streaming_archive_growth.pdf"):
    if d is None:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                             gridspec_kw=dict(height_ratios=[1, 0.6]))
    ax = axes[0]
    ax.semilogy(d["bufs"], d["fffs"], "o-", color="#3666b4", lw=1.0,
                markersize=4, label="FastLSQ cumulative-buffer")
    ax.semilogy(d["bufs"], d["cons"], "s-", color="#bc4444", lw=0.8,
                markersize=3, label="last-value baseline")
    ax.semilogy(d["bufs"], d["lins"], "^-", color="#54a04e", lw=0.8,
                markersize=3, label="slope-extrap baseline")
    ax.set_xlabel("cumulative buffer (hours of archive data)")
    ax.set_ylabel(f"next-chunk rel-$L_2$  ({EVAL_CHUNK_S/60:.0f} min horizon)")
    ax.set_title(f"Streaming-archive learning curve on ALS-U {CORR_FAM} stream\n"
                 f"FFF early error {d['early']:.2e}  $\\to$  "
                 f"late error {d['late']:.2e}   "
                 f"(ratio {d['early']/max(d['late'],1e-30):.2f}$\\times$);  "
                 f"refit median {np.median(d['fit_ms']):.0f} ms",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax = axes[1]
    ax.plot(d["bufs"], d["fit_ms"], "o-", color="#888888", lw=0.8,
            markersize=3)
    ax.axhline(np.median(d["fit_ms"]), color="#3666b4", lw=0.8, ls="--",
               label=f"median {np.median(d['fit_ms']):.0f} ms")
    ax.set_xlabel("cumulative buffer (hours)")
    ax.set_ylabel("refit wall-clock (ms)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight", dpi=140)
    plt.savefig(out_pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=130)
    print(f"   wrote {out_pdf}")


if __name__ == "__main__":
    d = run()
    make_figure(d)
