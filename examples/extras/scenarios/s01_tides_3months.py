#!/usr/bin/env python
"""Earth-tide signature in ALS-U corrector setpoints over Feb/Mar/Apr 2026.

The Earth's crust deforms by tens of nanometers under lunar and solar
gravitational tides; that deformation moves the storage-ring floor and
therefore the ring's reference orbit; the slow-orbit feedback (SOFB)
compensates by moving the corrector magnets.  The SOFB therefore
"records" the suppressed open-loop orbit drift in its corrector
setpoint strengths.  Over a month-long window this signature should
contain the standard astronomical tidal constituents:
   M2 = 12.4206 h   (principal lunar semidiurnal)
   S2 = 12.0000 h   (principal solar semidiurnal)
   N2 = 12.6583 h   (larger elliptic lunar)
   K1 = 23.9345 h   (luni-solar diurnal)
   O1 = 25.8193 h   (principal lunar diurnal)

We pull the 24 horizontal-corrector setpoint time series for each of
Feb, Mar, and Apr 2026 via the archiver in 60-second snapshots every
30 minutes (so each month produces ~1440 sample-clusters per channel,
plenty for tidal-band resolution and small enough to fetch).  Each
channel's time series is fitted on a Fast Fourier Features basis;
Lomb-Scargle periodograms are computed per channel and averaged
across all 24 correctors per month to suppress per-magnet noise.

If the tidal peaks appear at the same azimuth-independent frequencies
in all three months, that confirms a real astronomical (rather than
operational) origin.

Outputs /tmp/alsu_tides_3months.pdf and .png.
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


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

from datetime import datetime, timedelta

# Three four-week windows aligned to the ALS-U operating schedule.
MONTHS = [
    ("Feb 2026", datetime(2026, 2, 1), datetime(2026, 3, 1)),
    ("Mar 2026", datetime(2026, 3, 1), datetime(2026, 4, 1)),
    ("Apr 2026", datetime(2026, 4, 1), datetime(2026, 5, 1)),
]

# 24 horizontal corrector readbacks (HCM1, HCM4 per sector).
CORR_PVS = [f"SR{s:02d}C:HCM{n}:MRV"
            for s in range(1, 13) for n in (1, 4)]

# Tidal constituents we will look for (period in hours).
TIDES = [
    ("M2", 12.4206012),
    ("S2", 12.0000000),
    ("N2", 12.6583480),
    ("K1", 23.9344696),
    ("O1", 25.8193387),
]


# ----------------------------------------------------------------------
# Snapshot-mode fetch
# ----------------------------------------------------------------------

def fetch_snapshots(pv, start, end, interval_min=30, dur_s=60):
    """Return (t_secs, vals) by querying the archiver in short snapshot
    windows at regular intervals, then concatenating.  This avoids
    pulling the full 10 Hz time series over a 4-week window."""
    get_pv = _ensure_daq()
    t_all, v_all = [], []
    t = start
    delta = timedelta(seconds=dur_s)
    step = timedelta(minutes=interval_min)
    while t < end:
        try:
            resp, st = get_pv(pv, start=t, end=t + delta)
            if st == 200:
                ts, vs = _series_to_arrays(resp)
                if ts is not None and len(ts) > 0:
                    t_all.append(ts + (t - start).total_seconds())
                    v_all.append(vs)
        except Exception:
            pass
        t += step
    if not t_all:
        return None, None
    return np.concatenate(t_all), np.concatenate(v_all)


# ----------------------------------------------------------------------
# Spectrum on one channel
# ----------------------------------------------------------------------

def ls_spectrum(t_s, v, period_grid_h):
    """Lomb-Scargle power at each period in `period_grid_h` (hours)."""
    v_c = v - v.mean()
    powers = []
    for T_h in period_grid_h:
        W = 2 * np.pi / (T_h * 3600.0)        # rad/s, t in seconds
        sw = np.sin(W * t_s); cw = np.cos(W * t_s)
        ds = float(sw @ sw); dc = float(cw @ cw); cs = float(sw @ cw)
        det = ds * dc - cs * cs
        if det <= 0:
            powers.append(0.0); continue
        ys = float(sw @ v_c); yc = float(cw @ v_c)
        a = (dc * ys - cs * yc) / det
        b = (ds * yc - cs * ys) / det
        powers.append(a * ys + b * yc)
    return np.asarray(powers)


# ----------------------------------------------------------------------
# Per-month analysis
# ----------------------------------------------------------------------

def analyse_month(name, start, end, period_grid_h):
    print(f"\n=== {name}  ({start.date()} -- {end.date()}) ===",
          flush=True)
    print(f"   fetching {len(CORR_PVS)} correctors in snapshot mode "
          f"(30 min cadence x 60 s snapshots) ...", flush=True)
    channel_data = {}
    for i, pv in enumerate(CORR_PVS):
        try:
            t_s, v = fetch_snapshots(pv, start, end,
                                     interval_min=30, dur_s=60)
        except Exception as e:
            t_s = v = None
        if t_s is not None and len(t_s) > 200:
            channel_data[pv] = (t_s, v)
        if (i + 1) % 8 == 0:
            print(f"     {i+1}/{len(CORR_PVS)} fetched, "
                  f"{len(channel_data)} populated", flush=True)
    print(f"   populated channels: {len(channel_data)}")
    if not channel_data:
        return None
    per_channel = []
    for pv, (t_s, v) in channel_data.items():
        P = ls_spectrum(t_s, v, period_grid_h)
        P_norm = P / max(P.max(), 1e-30)
        per_channel.append((pv, P_norm))
    # Average across channels (after each is normalised to peak 1)
    P_avg = np.mean([P for _, P in per_channel], axis=0)
    # Peak at each named tide
    print(f"   spectral peaks at known tidal frequencies:")
    peak_info = []
    for tide, T_known in TIDES:
        # Find peak within ±5% of expected
        rel = np.abs(period_grid_h - T_known) / T_known
        within = np.where(rel < 0.05)[0]
        if len(within) == 0:
            peak_info.append((tide, T_known, np.nan, 0.0))
            continue
        k = within[np.argmax(P_avg[within])]
        T_rec = period_grid_h[k]
        amp = P_avg[k]
        peak_info.append((tide, T_known, T_rec, amp))
        print(f"     {tide}  T_known = {T_known:7.4f} h    "
              f"T_recovered = {T_rec:7.4f} h    "
              f"amp = {amp:.3f}    "
              f"rel err = {(T_rec - T_known) / T_known:+.2e}")
    return {"period_h": period_grid_h, "P_avg": P_avg,
            "per_channel": per_channel,
            "peak_info": peak_info,
            "n_channels": len(channel_data)}


# ----------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------

def make_figure(results, period_grid_h, out_pdf):
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 1, hspace=0.05)
    colours = ["#3b6db4", "#54a04e", "#c4424b"]
    for k, (res, c) in enumerate(zip(results, colours)):
        if res is None: continue
        ax = fig.add_subplot(gs[k])
        ax.semilogy(period_grid_h, res["P_avg"],
                    color=c, lw=1.2,
                    label=res["name"] + f"  ({res['n_channels']} ch.)")
        # Mark known tides
        ymax = res["P_avg"].max()
        for tide, T_known in TIDES:
            ax.axvline(T_known, color="black", lw=0.4, ls=":")
            if k == 0:
                ax.text(T_known, ymax * 1.3, tide, ha="center",
                        fontsize=8, color="black")
        # Mark recovered peaks
        for tide, T_known, T_rec, amp in res["peak_info"]:
            if not np.isnan(T_rec):
                ax.plot(T_rec, amp, "o", color=c, markersize=6,
                        markeredgecolor="black", markeredgewidth=0.4)
        ax.set_xlim(period_grid_h.min(), period_grid_h.max())
        ax.set_ylim(1e-3, 3.0)
        ax.set_ylabel("avg LS power\n(normalised)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10, loc="upper right")
        if k < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("period (hours)")
    fig.suptitle("Earth-tide signature in ALS-U horizontal-corrector "
                 "setpoints over three monthly windows\n"
                 "(snapshot-mode fetch, 30-min cadence x 60-s windows, "
                 "average LS spectrum across 24 channels)",
                 fontsize=11, y=0.99)
    plt.savefig(out_pdf, bbox_inches="tight", dpi=140)
    plt.savefig(out_pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=130)
    print(f"\n   wrote {out_pdf}")


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main():
    print(">> Three-month Earth-tide search in ALS-U corrector setpoints",
          flush=True)
    # Period grid: 6 h to 50 h, log-spaced for resolution near 12-26 h.
    period_grid_h = np.exp(np.linspace(np.log(6), np.log(50), 2000))
    results = []
    for name, start, end in MONTHS:
        r = analyse_month(name, start, end, period_grid_h)
        if r is not None:
            r["name"] = name
        results.append(r)
    out = "/tmp/alsu_tides_3months.pdf"
    make_figure(results, period_grid_h, out)
    # Cross-month summary
    print(f"\n   CROSS-MONTH consistency check (rel err at each tide):")
    for tide, T_known in TIDES:
        recovered = []
        for r in results:
            if r is None: continue
            for t, _, T_rec, amp in r["peak_info"]:
                if t == tide and not np.isnan(T_rec):
                    recovered.append((r["name"], T_rec, amp))
        print(f"     {tide}  (T_known = {T_known:7.4f} h):")
        for name, T_rec, amp in recovered:
            print(f"       {name:10s} -> {T_rec:7.4f} h "
                  f"(rel err {(T_rec - T_known)/T_known:+.2e}, amp {amp:.2f})")


if __name__ == "__main__":
    main()
