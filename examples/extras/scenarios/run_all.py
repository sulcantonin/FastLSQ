#!/usr/bin/env python
"""Run every scenario in the FFF cross-domain catalogue and emit a
single summary table.

Each scenario auto-downloads its public data (or generates synthetic
data where no public dataset is appropriate); after every run we
collect the headline metric and report it.
"""
from __future__ import annotations
import io, contextlib, time, importlib, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# (id, module-name, short label)
SCENARIOS = [
    ("01", "s01_betatron_tune",       "Accelerator: betatron tune"),
    ("02", "s02_plasma_wakefield",    "Accelerator: plasma wakefield"),
    ("03", "s03_synchrobetatron",     "Accelerator: synchro-betatron"),
    ("04", "s04_sunspots",            "Solar: sunspot cycles"),
    ("05", "s05_helioseismology",     "Solar: p-mode helioseismology"),
    ("06", "s06_tides",               "Earth: NOAA tides"),
    ("07", "s07_iers_earth_rotation", "Earth: IERS rotation"),
    ("08", "s08_mauna_loa_co2",       "Earth: Mauna Loa CO2"),
    ("09", "s09_enso_qbo",            "Earth: ENSO + QBO"),
    ("10", "s10_pulsar_timing",       "Astronomy: pulsar timing"),
    ("11", "s11_modal_analysis",      "Engineering: modal analysis"),
    ("12", "s12_mems_resonator",      "Engineering: MEMS Duffing"),
    ("13", "s13_variable_stars_kepler", "Astronomy: Kepler Cepheid"),
    ("14", "s14_eeg",                 "Neuroscience: EEG alpha"),
    ("15", "s15_circadian",           "Biology: circadian"),
]


def silent_run(mod_name):
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        try:
            importlib.import_module(mod_name).main()
            ok = True; err = ""
        except Exception as e:
            ok = False; err = str(e)
    dt = time.perf_counter() - t0
    return ok, err, dt, buf.getvalue()


def main():
    print("=" * 76)
    print("Fast Fourier Features cross-domain scenarios -- run all")
    print("=" * 76, flush=True)
    rows = []
    for sid, mod, label in SCENARIOS:
        print(f"\n[{sid}] {label} ...", flush=True)
        ok, err, dt, out = silent_run(mod)
        status = "OK" if ok else f"FAIL: {err.splitlines()[0][:80]}"
        rows.append((sid, label, status, dt))
        print(f"     {status}    ({dt:.2f}s)")
    print("\n" + "=" * 76)
    print(" SUMMARY")
    print("=" * 76)
    print(f"  {'id':4s} {'scenario':36s} {'time (s)':10s} status")
    for sid, label, status, dt in rows:
        print(f"  {sid:4s} {label:36s} {dt:<10.2f} {status}")
    n_ok = sum(1 for r in rows if r[2] == "OK")
    print(f"\n  {n_ok}/{len(rows)} scenarios completed without error.")


if __name__ == "__main__":
    main()
