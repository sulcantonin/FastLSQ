# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Device-correct timing primitives for FastLSQ.

Naively wall-clock-timing a CUDA solve is wrong: kernels are launched
asynchronously, so ``t1 - t0`` measures launch overhead, not compute.  Without a
``synchronize`` bracket every GPU number is meaningless (and irreproducible).
:func:`time_solve` adds the missing primitive -- ``synchronize`` bracketing plus
warm-up and a min-of-reps reduction -- so the reported solve time is the
reproducible compute *floor* on any device (CPU / CUDA / Apple-MPS).

    import fastlsq as fl
    t = fl.benchmark.time_solve(lambda: fl.solve_lstsq(A, b))   # seconds, floor
"""

import time

import torch

from fastlsq.device import get_device


def synchronize(device=None):
    """Block until all queued work on ``device`` has finished.

    No-op on CPU; calls ``torch.cuda.synchronize`` on CUDA and
    ``torch.mps.synchronize`` on Apple-MPS (when available).  ``device=None``
    uses the active FastLSQ device.
    """
    device = get_device() if device is None else torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        sync = getattr(getattr(torch, "mps", None), "synchronize", None)
        if sync is not None:
            sync()


def time_solve(fn, reps=10, warmup=2, device=None, return_all=False):
    """Device-correct timing floor for a callable.

    Runs ``warmup`` unmeasured calls (allocator / autotune warm-up), then ``reps``
    measured calls, each bracketed by :func:`synchronize` so CUDA/MPS asynchronous
    execution is fully accounted for.  Returns the **minimum** elapsed time in
    seconds -- the reproducible solve-time floor -- which is the right summary for
    a deterministic op whose variance is pure system noise.

    Parameters
    ----------
    fn : callable
        Zero-argument callable performing the work to time, e.g.
        ``lambda: solve_lstsq(A, b)``.
    reps : int
        Number of measured repetitions (>= 1).
    warmup : int
        Number of unmeasured warm-up repetitions.
    device : str | torch.device | None
        Device to synchronize around each call.  ``None`` uses the active FastLSQ
        device.
    return_all : bool
        If True, return a stats dict
        ``{"min", "median", "mean", "std", "times", "reps", "warmup", "device"}``
        instead of the bare floor.

    Returns
    -------
    float
        The minimum solve time in seconds (``return_all=False``).
    dict
        Full timing statistics (``return_all=True``).
    """
    device = get_device() if device is None else torch.device(device)

    for _ in range(max(0, warmup)):
        fn()
    synchronize(device)

    times = []
    for _ in range(max(1, reps)):
        synchronize(device)
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        times.append(time.perf_counter() - t0)

    t = torch.tensor(times, dtype=torch.float64)
    floor = float(t.min())
    if not return_all:
        return floor
    return {
        "min": floor,
        "median": float(t.median()),
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=False)),
        "times": times,
        "reps": len(times),
        "warmup": max(0, warmup),
        "device": str(device),
    }
