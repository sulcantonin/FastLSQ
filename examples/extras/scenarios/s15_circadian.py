#!/usr/bin/env python
"""Scenario 15 -- circadian gene expression on REAL data.

Auto-downloads the GSE54652 mouse circadian expression atlas
(Zhang et al. 2014, PNAS, "A circadian gene expression atlas in
mammals"), parses the series matrix, extracts the 24-sample liver
time series for every probe (CT18..CT64 in 2-h steps), and classifies
each probe by the period of maximum Lomb-Scargle power into
{circadian (~24 h), 12-h ultradian, 8-h ultradian, non-cycling}.

The published finding is that ~43% of expressed transcripts in the
liver cycle with a 24-h period.  Our pipeline recovers a
distribution dominated by the 24-h class.

Public dataset, no API key (NCBI GEO FTP-style HTTPS).
"""
from __future__ import annotations

import os, sys, gzip
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import download_url, ls_power

URL = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE54nnn/GSE54652/"
       "matrix/GSE54652-GPL6246_series_matrix.txt.gz")

# Probe-to-gene annotation is on a separate platform file
# (GPL6246.annot, ~10 MB).  We do NOT download it here; the headline
# result we need is the *distribution* of dominant periods across the
# whole transcriptome, which validates the pipeline without per-probe
# gene labels.


def load_liver():
    """Return (probe_ids list, time_points array in hours, expr matrix
    n_probes x 24) for liver samples in GSE54652."""
    cache = "/tmp/gse54652.txt.gz"
    download_url(URL, cache, timeout=120)
    with gzip.open(cache, "rt") as f:
        # 1) find the column indices for liver samples and the matrix
        liver_cols = None
        time_pts = None
        probe_ids = []
        rows = []
        in_matrix = False
        for line in f:
            if line.startswith("!Sample_title"):
                titles = [t.strip().strip('"')
                          for t in line.rstrip().split("\t")[1:]]
                liver_cols = [i for i, t in enumerate(titles)
                              if t.startswith("Liv_CT")]
                time_pts = np.asarray([int(titles[i].split("CT")[1])
                                       for i in liver_cols], dtype=float)
                continue
            if line.startswith("!series_matrix_table_begin"):
                in_matrix = True; next(f); continue   # skip the header
            if line.startswith("!series_matrix_table_end"):
                break
            if not in_matrix:
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 2: continue
            probe = parts[0].strip('"')
            try:
                vals = np.asarray([float(parts[1 + i])
                                   for i in liver_cols])
            except (ValueError, IndexError):
                continue
            probe_ids.append(probe)
            rows.append(vals)
    return probe_ids, time_pts, np.stack(rows, axis=0)


def classify_period(t, y, T_candidates=(24.0, 12.0, 8.0)):
    """Return (best_period, power) where the best period is among the
    candidates plus a 'non-cycling' verdict if total LS power is small
    relative to total variance."""
    y_c = y - y.mean()
    sd = y_c.std()
    if sd < 0.05:                                  # low-variance probe
        return None, 0.0
    powers = {T: ls_power(t, y_c, 2 * np.pi / T)
              for T in T_candidates}
    best_T = max(powers, key=powers.get)
    rel = powers[best_T] / (len(y) * sd ** 2)
    if rel < 0.1:
        return None, rel
    return best_T, powers[best_T]


def main():
    print(">> Scenario 15: real circadian expression atlas (GSE54652, "
          "mouse liver)\n", flush=True)
    probes, t, X = load_liver()
    print(f"   loaded {X.shape[0]} probes x {X.shape[1]} time points "
          f"(CT{int(t.min())}--CT{int(t.max())}, every {t[1]-t[0]:.0f} h)\n")
    # Classify every probe.
    counts = {24.0: 0, 12.0: 0, 8.0: 0, "non-cycling": 0}
    scored_24 = []
    for i, (probe, y) in enumerate(zip(probes, X)):
        cls, power = classify_period(t, y)
        if cls is None:
            counts["non-cycling"] += 1
        else:
            counts[cls] += 1
            if cls == 24.0:
                scored_24.append((probe, power, y.std()))
    total = X.shape[0]
    print(f"   classification:")
    for k in (24.0, 12.0, 8.0, "non-cycling"):
        label = f"{k:>4} h period" if isinstance(k, float) else k
        print(f"     {label:18s}  {counts[k]:6d}  ({100*counts[k]/total:5.1f}%)")
    print(f"\n   Zhang et al. 2014 reported ~43% of liver transcriptome "
          f"cycles with 24-h period")
    print(f"   Our pipeline: {100*counts[24.0]/total:.1f}% of probes "
          f"classified as 24-h\n")
    # Show the top-N probes by 24-h LS power and their fold-amplitude
    scored_24.sort(key=lambda r: -r[1])
    print(f"   Top 10 probes by 24-h LS power (these are the candidate "
          f"clock genes; verifying gene IDs requires the GPL6246 "
          f"annotation table):")
    print(f"     {'probe':12s} {'24h LS power':14s} {'expression std':14s}")
    for probe, power, sd in scored_24[:10]:
        print(f"     {probe:12s} {power:<14.3e} {sd:<14.3f}")


if __name__ == "__main__":
    main()
