#!/usr/bin/env python
"""Scenario 12 -- harmonic resonator characterisation on REAL audio.

A struck tuning fork is a physical Duffing-like resonator: a thin
metal bar with a sharply defined fundamental, weak harmonic content,
and a slow exponential ringdown set by the metal's Q.  The pipeline
that recovers (omega_0, Q, A_3/A_1) on a synthetic MEMS oscillator
applies verbatim to the audio waveform.

Auto-downloads a CC-licensed 440 Hz tuning-fork recording from
Wikimedia Commons (no API key) and decodes it via ffmpeg
(standard on macOS / Linux audio toolchains).

If ffmpeg or the OGG file is unavailable the script falls back to
the synthetic Duffing trace; the headline numbers below come from
the real audio.
"""
from __future__ import annotations

import os, sys, struct, subprocess, wave
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import download_url, fit_1d_fff, ls_periodogram_peaks, \
                    ls_power, spectral_summary

OGG_URL = "https://upload.wikimedia.org/wikipedia/commons/7/70/Tuning-fork-440Hz.ogg"
F0_KNOWN_HZ = 440.0


def load_audio():
    """Download the OGG and convert to mono 22 kHz WAV via ffmpeg.
    Returns (sample-rate, np.ndarray of samples in [-1, 1])."""
    ogg = "/tmp/tuningfork.ogg"
    wav = "/tmp/tuningfork.wav"
    download_url(OGG_URL, ogg, timeout=30)
    if not os.path.exists(wav) or os.path.getsize(wav) < 1000:
        subprocess.run(["ffmpeg", "-y", "-i", ogg, "-ac", "1",
                        "-ar", "22050", "-acodec", "pcm_s16le", wav],
                       check=True, capture_output=True)
    with wave.open(wav, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    return sr, samples


def synth_fallback():
    """Synthetic Duffing ringdown (same as the previous version)."""
    omega0 = 2 * np.pi * 1e6
    Q = 5e3; alpha = 0.05 * omega0 ** 2
    n = 500000
    dt = 1e-8
    t = np.arange(n) * dt
    x = np.zeros(n); v = np.zeros(n); x[0] = 1.0
    gamma = omega0 / Q
    a_prev = -gamma * v[0] - omega0 ** 2 * x[0] - alpha * x[0] ** 3
    for k in range(n - 1):
        x[k+1] = x[k] + v[k] * dt + 0.5 * a_prev * dt * dt
        a_new = (-gamma * (v[k] + 0.5 * a_prev * dt)
                 - omega0 ** 2 * x[k+1] - alpha * x[k+1] ** 3)
        v[k+1] = v[k] + 0.5 * (a_prev + a_new) * dt
        a_prev = a_new
    return 1.0 / dt, x


def main():
    print(">> Scenario 12: tuning-fork resonator from REAL audio\n",
          flush=True)
    try:
        sr, x = load_audio()
        src = "Wikimedia tuning fork 440 Hz (CC-licensed)"
        F0_ref = F0_KNOWN_HZ
    except Exception as e:
        print(f"   audio path unavailable ({e}); falling back to synthetic\n")
        sr, x = synth_fallback()
        src = "synthetic Duffing"
        F0_ref = 1e6
    t = np.arange(len(x)) / sr
    print(f"   source: {src}")
    print(f"   {len(x)} samples,  sr = {sr/1000:.1f} kHz,  "
          f"duration = {t[-1]:.2f} s\n")

    # Skip the attack transient: start at 0.5 s into the recording
    i0 = int(0.5 * sr)
    # Take a 3-second window (long enough for clean ringdown decay)
    i1 = min(i0 + int(3.0 * sr), len(x))
    t_w = t[i0:i1] - t[i0]; x_w = x[i0:i1]
    x_w = x_w - x_w.mean()
    print(f"   analysis window: 0.5 s -> {0.5 + (i1-i0)/sr:.2f} s "
          f"({i1-i0} samples)")

    # FFF fit centred at the expected fundamental
    sigma_W = 2 * (2 * np.pi * F0_ref)
    basis, beta, rmse, _ = fit_1d_fff(t_w, x_w, sigma_W,
                                      n_features=2000, mu_reg=1e-10)
    print(f"   FFF fit RMSE = {rmse:.3e}  (std = {x_w.std():.3e})\n")

    # LS periodogram around the fundamental and its harmonics
    W_min = 2 * np.pi * 0.5 * F0_ref
    W_max = 2 * np.pi * 5.0 * F0_ref
    peaks = ls_periodogram_peaks(t_w, x_w, W_min, W_max,
                                 n_grid=8000, n_peaks=5,
                                 suppress_log_frac=0.05)
    print("   Top peaks:")
    print(f"     {'f (Hz)':12s} {'tag':18s} {'rel err':10s}")
    f0_recovered = None
    for W, _ in peaks:
        f = W / (2 * np.pi)
        tag = ""; rel = ""
        if f0_recovered is None:
            f0_recovered = f; tag = "fundamental"
            rel = f"{(f - F0_ref) / F0_ref:+.2e}"
        elif abs(f / f0_recovered - 2.0) < 0.05:
            tag = "2nd harmonic"
        elif abs(f / f0_recovered - 3.0) < 0.05:
            tag = "3rd harmonic"
        print(f"     {f:<12.4f} {tag:18s} {rel}")
    # Q-factor from amplitude decay of the envelope
    envelope = np.abs(x[i0:i1])
    # Local maxima
    mask = (envelope[1:-1] > envelope[:-2]) & (envelope[1:-1] > envelope[2:])
    t_pk = t_w[1:-1][mask]; e_pk = envelope[1:-1][mask]
    threshold = e_pk.max() * 1e-2
    keep = (e_pk > threshold) & (t_pk > 0.1) & (t_pk < t_w[-1] - 0.1)
    Q_recovered = float("nan")
    if keep.sum() > 20:
        slope, _ = np.polyfit(t_pk[keep], np.log(e_pk[keep]), 1)
        if slope < 0:
            Q_recovered = -2 * np.pi * f0_recovered / (2 * slope)
    print(f"\n   Recovered f_0 = {f0_recovered:.4f} Hz   "
          f"(known {F0_ref:.1f} Hz)")
    print(f"   Q from amplitude decay envelope = {Q_recovered:.1f}")
    # Harmonic content
    omega0 = 2 * np.pi * f0_recovered
    P1 = ls_power(t_w, x_w, omega0)
    P2 = ls_power(t_w, x_w, 2 * omega0)
    P3 = ls_power(t_w, x_w, 3 * omega0)
    A21 = float(np.sqrt(max(P2 / max(P1, 1e-30), 0.0)))
    A31 = float(np.sqrt(max(P3 / max(P1, 1e-30), 0.0)))
    print(f"   Harmonic amplitude ratios: A2/A1 = {A21:.4f},  A3/A1 = {A31:.4f}")
    s = spectral_summary(basis.W.numpy(), beta.numpy(), k_top=3)
    print(f"\n   Expansion: top-3 energy = {s['energy_K_top']:.3f}, "
          f"K_99 = {s['K_target']}/{s['N']}")


if __name__ == "__main__":
    main()
