#!/usr/bin/env python
"""Scenario 01 -- accelerator-physics signals on the ALS-U.

Three modes:

  * --mode lattice  (default): the betatron tune from a tracking
    simulation of the real ALS-U v21 4-raft Superbend lattice via
    pyAT.  Kicked particle, 4096 turns through one superperiod,
    turn-by-turn position at a BPM marker, FFF + LS recovery,
    comparison against the analytical one-turn-matrix tune.
    This is the "what does the lattice say" path.

  * --mode archive: archived *scalar* PVs over a long time window
    via the ALS controls archiver appliance (daq.get_pv()).  We
    fetch `SR:DCCT` (beam current) and `SR01C:BPM1:SA:X` (slow-orbit
    scalar BPM position) over the last 24 hours, then the FFF + LS
    pipeline recovers the top-off injection cadence from the DCCT
    signature and the dominant orbit-drift frequencies from the SA
    BPM time series.  This is the "what does the live machine
    actually do day-to-day" path, and it only uses archived
    scalars (turn-by-turn waveforms `:wfr:` are not archived
    on a routine basis).

  * --mode both: runs lattice then archive.

PV map for --mode archive:

    beam current scalar:           SR:DCCT
    slow-orbit BPM position (x):   SR01C:BPM1:SA:X
    slow-orbit BPM position (y):   SR01C:BPM1:SA:Y
    live tune readback (Qx):       IGPF:TFBX:SRAM:PEAKTUNE1
    live tune readback (Qy):       IGPF:TFBY:SRAM:PEAKTUNE1

The betatron tune itself requires turn-by-turn (revolution-rate)
position data, which the archiver does not keep.  We therefore
*do not* try to recover the tune from --mode archive; the tune
demonstration lives in --mode lattice on the parsed lattice.
"""
from __future__ import annotations

import os, sys, argparse, warnings
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

from _common import fit_1d_fff, ls_periodogram_peaks, spectral_summary

LATTICE_PATH = (
    "/Users/asulc/PycharmProjects/signal_to_vector/lattice/"
    "ALS_U_v21_4raft_SB_updtBPM.m"
)

# Live ALS-U scalar PVs identified in pv_names_list.txt.
PV_DCCT       = "SR:DCCT"
PV_SA_X       = "SR01C:BPM1:SA:X"
PV_SA_Y       = "SR01C:BPM1:SA:Y"
PV_TUNE_X     = "IGPF:TFBX:SRAM:PEAKTUNE1"
PV_TUNE_Y     = "IGPF:TFBY:SRAM:PEAKTUNE1"


# ----------------------------------------------------------------------
# Tracking-mode: pyAT
# ----------------------------------------------------------------------

def track_alsu(n_turns=4096, kick_x_mm=1.0, kick_y_mm=0.5):
    """Build the ALS-U lattice, find the closed orbit, kick the beam,
    and record turn-by-turn position at the first BPM marker.

    Returns (x_n, y_n, Q_x_design, Q_y_design, n_bpms).
    """
    from _alsu_lattice import parse_elements, build_one_superperiod
    import at
    elems = parse_elements(LATTICE_PATH)
    sp = build_one_superperiod(elems, variant="SUP")
    ring = at.Lattice(sp, energy=2.0e9, periodicity=12,
                      name="ALS-U v21 SUP")
    ring.disable_6d()

    # Analytical one-turn matrix tune for the single superperiod
    M44, *_ = at.find_m44(ring)
    tx = (M44[0, 0] + M44[1, 1]) / 2
    ty = (M44[2, 2] + M44[3, 3]) / 2
    Qx_sp = np.arccos(tx) / (2 * np.pi)
    Qy_sp = np.arccos(ty) / (2 * np.pi)

    # BPM marker indices (within the one-superperiod sequence we track).
    bpm_idx = [i for i, e in enumerate(ring) if e.FamName.startswith("BPM")]
    n_bpms = len(bpm_idx)

    # Kick the particle and track for N turns.
    r0 = np.zeros((6, 1))
    r0[0, 0] = kick_x_mm * 1e-3
    r0[2, 0] = kick_y_mm * 1e-3
    out = np.asarray(at.lattice_pass(ring, r0, nturns=n_turns,
                                     refpts=bpm_idx[:1]))
    # out shape: (6, n_particles=1, n_refpts=1, n_turns)
    x_n = out[0, 0, 0, :]
    y_n = out[2, 0, 0, :]
    return x_n, y_n, Qx_sp, Qy_sp, n_bpms


# ----------------------------------------------------------------------
# Archive mode: pull scalar PVs over a long window
# ----------------------------------------------------------------------

def _ensure_daq():
    """Import daq.get_pv from the user's als3 utility package."""
    try:
        sys.path.insert(0, "/Users/asulc/PycharmProjects/als3/utils")
        from daq import get_pv
        return get_pv
    except Exception as e:
        raise RuntimeError(
            f"cannot import als3.utils.daq ({e}); "
            "archive mode requires the SSH tunnel to controls.als.lbl.gov "
            "on localhost:8080"
        )


def _series_to_arrays(resp):
    """Convert a daq.get_pv response into (t_seconds_since_start, value)."""
    series = resp[0] if isinstance(resp, list) else resp
    pts = series.get("data", []) if isinstance(series, dict) else []
    if not pts:
        return None, None
    secs = np.array([p.get("secs", 0) + p.get("nanos", 0) * 1e-9
                     for p in pts], dtype=float)
    vals = np.array([p.get("val", 0.0) for p in pts], dtype=float)
    t0 = secs[0]
    return secs - t0, vals


def fetch_archive_scalars(hours=24.0):
    """Pull SR:DCCT, SR01C:BPM1:SA:X, SA:Y over the last `hours` hours."""
    get_pv = _ensure_daq()
    from datetime import datetime, timedelta
    end = datetime.utcnow()
    start = end - timedelta(hours=hours)
    out = {}
    for label, pv in (("dcct", PV_DCCT),
                      ("sa_x", PV_SA_X),
                      ("sa_y", PV_SA_Y),
                      ("qx_live", PV_TUNE_X),
                      ("qy_live", PV_TUNE_Y)):
        resp, status = get_pv(pv, start=start, end=end)
        if status != 200:
            out[label] = (None, None, f"HTTP {status}")
            continue
        t, v = _series_to_arrays(resp)
        out[label] = (t, v, None)
    return out


# ----------------------------------------------------------------------
# Multi-BPM mode: spatial-mode decomposition of orbit drift
# ----------------------------------------------------------------------

# Spatially-spread BPM subset: 5 BPMs per sector x 12 sectors -> 60 BPM channels per plane.
DEFAULT_BPMS = (1, 5, 10, 14, 18)
DEFAULT_SECTORS = tuple(range(1, 13))


def build_bpm_pv_list(sectors=DEFAULT_SECTORS, bpms=DEFAULT_BPMS,
                      planes=("X", "Y")):
    pvs = []
    for s in sectors:
        for b in bpms:
            for p in planes:
                pvs.append(f"SR{s:02d}C:BPM{b}:SA:{p}")
    return pvs


def fetch_multi_bpm(hours=2.0, sectors=DEFAULT_SECTORS, bpms=DEFAULT_BPMS,
                    n_workers=16):
    """Fetch SA scalars for many BPMs in parallel via daq.download_pvs_from_csv.
    Returns aligned (t, X[t, n], Y[t, n], bpm_labels)."""
    get_pv = _ensure_daq()
    from datetime import datetime, timedelta
    from multiprocessing import Pool
    end = datetime.utcnow()
    start = end - timedelta(hours=hours)
    pvs_x = build_bpm_pv_list(sectors, bpms, planes=("X",))
    pvs_y = build_bpm_pv_list(sectors, bpms, planes=("Y",))
    all_pvs = pvs_x + pvs_y
    print(f"   fetching {len(all_pvs)} PVs in parallel ({n_workers} workers)...",
          flush=True)
    # Serial fallback if parallel breaks; daq.get_pv is the unit call.
    results = {}
    for pv in all_pvs:
        try:
            resp, st = get_pv(pv, start=start, end=end)
            if st != 200:
                results[pv] = None; continue
            t, v = _series_to_arrays(resp)
            results[pv] = (t, v) if t is not None else None
        except Exception:
            results[pv] = None
    return _align(results, pvs_x, pvs_y, sectors, bpms)


def _align(results, pvs_x, pvs_y, sectors, bpms):
    """Interpolate every series onto a common time grid.  Drop BPMs with
    no data; warn if many missing."""
    # Find the densest series to set the grid
    populated = [(p, results[p]) for p in (pvs_x + pvs_y) if results.get(p)]
    if not populated:
        return None, None, None, []
    densest = max(populated, key=lambda kv: len(kv[1][0]))
    t_grid = densest[1][0]
    if len(t_grid) > 8000:
        idx = np.linspace(0, len(t_grid) - 1, 8000).astype(int)
        t_grid = t_grid[idx]
    labels = []
    def build_matrix(pv_list):
        cols = []
        for pv in pv_list:
            r = results.get(pv)
            if r is None: continue
            t_p, v_p = r
            if len(t_p) < 64: continue
            try:
                v_i = np.interp(t_grid, t_p, v_p)
            except Exception:
                continue
            cols.append((pv, v_i))
        return cols
    cols_x = build_matrix(pvs_x); cols_y = build_matrix(pvs_y)
    print(f"   aligned to {len(t_grid)} time samples on common grid")
    print(f"   X channels populated: {len(cols_x)}/{len(pvs_x)}")
    print(f"   Y channels populated: {len(cols_y)}/{len(pvs_y)}")
    # Use the X-channel BPMs as the canonical channel order; their labels are
    # parsed from the PV name.
    X_mat = np.stack([c[1] for c in cols_x], axis=1) if cols_x else None
    Y_mat = np.stack([c[1] for c in cols_y], axis=1) if cols_y else None
    labels = [c[0] for c in cols_x]
    return t_grid, X_mat, Y_mat, labels


def svd_orbit_modes(M, n_modes=5):
    """Centre each channel; SVD; return temporal modes U[:, k], singular
    values sigma_k, spatial patterns Vt[k, :]."""
    Mc = M - M.mean(axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(Mc, full_matrices=False)
    return U[:, :n_modes], s[:n_modes], Vt[:n_modes, :]


def spectral_per_mode(t, U_modes, search_min_s=20.0, search_max_s=3600.0):
    """For each temporal mode column of U, fit on FFF and find the
    dominant LS-periodogram peak."""
    results = []
    for k in range(U_modes.shape[1]):
        u = U_modes[:, k]
        u_c = u - u.mean()
        sigma_W = 2 * np.pi / search_min_s
        basis, beta, rmse, _ = fit_1d_fff(t, u_c, sigma_W,
                                          n_features=800, mu_reg=1e-6)
        W_min = 2 * np.pi / search_max_s
        W_max = 2 * np.pi / search_min_s
        peaks = ls_periodogram_peaks(t, u_c, W_min, W_max,
                                     n_grid=2000, n_peaks=3,
                                     suppress_log_frac=0.05)
        if peaks:
            W0, _ = peaks[0]
            results.append({"period_s": 2 * np.pi / W0,
                            "rmse": rmse, "peaks": peaks})
        else:
            results.append({"period_s": float("nan"),
                            "rmse": rmse, "peaks": []})
    return results


def localize_mode(Vt_row, labels, top_k=4):
    """Identify the BPMs (and hence the sector) where a spatial mode
    pattern has the largest amplitude."""
    amps = np.abs(Vt_row)
    order = np.argsort(-amps)
    return [(labels[i], float(Vt_row[i])) for i in order[:top_k]]


# ----------------------------------------------------------------------
# FFF + LS recovery (shared between modes)
# ----------------------------------------------------------------------

def recover_tune(z_n, label):
    n = np.arange(len(z_n), dtype=float)
    z_c = z_n - z_n.mean()
    sigma_W = 2 * np.pi * 0.5
    basis, beta, rmse, _ = fit_1d_fff(n, z_c, sigma_W,
                                      n_features=1500, mu_reg=1e-12)
    W_min = 2 * np.pi * 0.01
    W_max = 2 * np.pi * 0.49
    peaks = ls_periodogram_peaks(n, z_c, W_min, W_max,
                                 n_grid=8000, n_peaks=3,
                                 suppress_log_frac=0.05)
    Q = peaks[0][0] / (2 * np.pi) if peaks else float("nan")
    s = spectral_summary(basis.W.numpy(), beta.numpy(), k_top=3)
    return Q, rmse, s


def run_lattice_mode():
    print(">> Scenario 01a: betatron tune from ALS-U lattice tracking",
          flush=True)
    x_n, y_n, Qx_sp, Qy_sp, n_bpms = track_alsu()
    print(f"   ALS-U v21 SUP lattice parsed from .m file")
    print(f"   {n_bpms} BPM markers in one super-period")
    print(f"   {len(x_n)} turns recorded\n")
    print(f"   Analytical one-turn-map tunes per super-period:")
    print(f"     Qx_sp = {Qx_sp:.5f},  Qy_sp = {Qy_sp:.5f}\n")
    Qx_rec, rmse_x, sx = recover_tune(x_n, "x")
    Qy_rec, rmse_y, sy = recover_tune(y_n, "y")
    print(f"   FFF recovery:")
    print(f"     Qx_rec = {Qx_rec:.5f}  (rmse {rmse_x:.2e},  K_99 "
          f"{sx['K_target']}/{sx['N']},  rel err "
          f"{abs(Qx_rec-Qx_sp)/Qx_sp:.2e})")
    print(f"     Qy_rec = {Qy_rec:.5f}  (rmse {rmse_y:.2e},  K_99 "
          f"{sy['K_target']}/{sy['N']},  rel err "
          f"{abs(Qy_rec-Qy_sp)/Qy_sp:.2e})\n")


def analyse_scalar(t, v, label, period_min_s, period_max_s,
                   expected_periods=()):
    """FFF + LS on a scalar archive series."""
    if t is None or len(v) < 32:
        print(f"   {label}: insufficient samples")
        return
    v_c = v - v.mean()
    sd = v_c.std()
    print(f"   {label}: {len(t)} samples over {t[-1]/3600:.2f} h, "
          f"std = {sd:.3e}")
    sigma_W = 2 * np.pi / period_min_s
    basis, beta, rmse, _ = fit_1d_fff(t, v_c, sigma_W,
                                      n_features=800, mu_reg=1e-6)
    W_min = 2 * np.pi / period_max_s
    W_max = 2 * np.pi / period_min_s
    peaks = ls_periodogram_peaks(t, v_c, W_min, W_max,
                                 n_grid=4000, n_peaks=5,
                                 suppress_log_frac=0.05)
    print(f"     {'W (rad/s)':12s} {'period':20s} {'tag':20s}")
    for W, _ in peaks:
        T = 2 * np.pi / W
        T_str = (f"{T/3600:.2f} h" if T > 7200
                 else (f"{T/60:.2f} min" if T > 120
                       else f"{T:.2f} s"))
        tag = ""
        for name, T0, tol in expected_periods:
            if abs(T - T0) / T0 < tol:
                tag = name; break
        print(f"     {W:<12.6f} {T_str:20s} {tag}")
    s = spectral_summary(basis.W.numpy(), beta.numpy(), k_top=3)
    print(f"     top-3 energy = {s['energy_K_top']:.3f},  "
          f"K_99 = {s['K_target']}/{s['N']}")


def run_archive_mode(hours=24.0):
    print(f">> Scenario 01b: archived scalar PVs over the last {hours:.0f} h",
          flush=True)
    try:
        data = fetch_archive_scalars(hours=hours)
    except Exception as e:
        print(f"   archive fetch failed: {e}")
        return
    # DCCT -> top-off injection cadence (expected ~30s at ALS, may differ at ALS-U)
    t_d, v_d, err_d = data["dcct"]
    if err_d is None:
        analyse_scalar(t_d, v_d, "SR:DCCT",
                       period_min_s=10.0, period_max_s=3600.0,
                       expected_periods=[("top-off ~ 30 s", 30.0, 0.5)])
    else:
        print(f"   SR:DCCT: {err_d}")
    # SA BPM X -> orbit-drift spectrum
    t_x, v_x, err_x = data["sa_x"]
    if err_x is None:
        analyse_scalar(t_x, v_x, "SR01C:BPM1:SA:X",
                       period_min_s=1.0, period_max_s=hours * 3600.0,
                       expected_periods=[("diurnal", 86400.0, 0.2),
                                         ("hourly", 3600.0, 0.2)])
    else:
        print(f"   SR01C:BPM1:SA:X: {err_x}")
    # Tune readback drift
    t_q, v_q, err_q = data["qx_live"]
    if err_q is None:
        print(f"   IGPF:TFBX:SRAM:PEAKTUNE1: {len(v_q)} samples,  "
              f"mean = {v_q.mean():.5f},  std = {v_q.std():.3e}")


def run_multi_bpm_mode(hours=2.0):
    print(f">> Scenario 01c: SVD spatial modes from {hours:.0f} h of all "
          f"SR*C:BPM*:SA:{{X,Y}}", flush=True)
    try:
        t, X_mat, Y_mat, labels = fetch_multi_bpm(hours=hours)
    except Exception as e:
        print(f"   multi-BPM fetch failed: {e}")
        return
    if X_mat is None:
        print("   no channels populated; aborting")
        return
    for plane_name, M in (("horizontal X", X_mat), ("vertical Y", Y_mat)):
        if M is None or M.shape[1] < 3: continue
        print(f"\n   ---- {plane_name} plane: SVD on "
              f"[{M.shape[0]} time x {M.shape[1]} BPMs] ----")
        U, s, Vt = svd_orbit_modes(M, n_modes=4)
        # Variance fraction per mode
        s2 = s ** 2
        var_frac = s2 / s2.sum()
        cum = np.cumsum(var_frac)
        print(f"     SVD singular values: "
              f"{' '.join(f'{x:.2e}' for x in s)}")
        print(f"     variance fraction:   "
              f"{' '.join(f'{100*v:.1f}%' for v in var_frac)}")
        print(f"     cumulative:          "
              f"{' '.join(f'{100*c:.1f}%' for c in cum)}")
        # FFF on each temporal mode
        spec = spectral_per_mode(t, U)
        for k, r in enumerate(spec):
            T = r["period_s"]
            T_str = (f"{T/3600:.2f} h" if T > 7200
                     else (f"{T/60:.2f} min" if T > 120
                           else f"{T:.2f} s"))
            top = localize_mode(Vt[k], labels, top_k=4)
            top_str = ", ".join(f"{n.replace('SR','S').replace(':SA:',':').replace(':BPM','-B'):s}={a:+.3f}" for n, a in top)
            print(f"     mode {k}  var={100*var_frac[k]:4.1f}%  "
                  f"period={T_str:<10s}  top BPMs: {top_str}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("lattice", "archive", "multi-bpm", "all"),
                    default="lattice",
                    help="lattice = pyAT tracking of ALS-U; "
                         "archive = scalars from one BPM + DCCT; "
                         "multi-bpm = SVD modes from many BPMs; "
                         "all = run all three")
    ap.add_argument("--hours", type=float, default=2.0,
                    help="window for archive / multi-bpm")
    args = ap.parse_args()

    if args.mode in ("lattice", "all"):
        run_lattice_mode()
    if args.mode in ("archive", "all"):
        run_archive_mode(hours=args.hours)
    if args.mode in ("multi-bpm", "all"):
        run_multi_bpm_mode(hours=args.hours)


if __name__ == "__main__":
    main()
