# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Device selection for FastLSQ: CPU / CUDA / Apple-MPS, dtype-aware.

FastLSQ's high-accuracy regime needs float64, but Apple's MPS backend does
**not** support float64 -- so MPS is auto-selected only for float32 work and is
never chosen when the active dtype is float64.

Configure at runtime::

    import fastlsq as fl
    fl.set_device("cuda")        # or "mps", "cpu", or None (auto)
    fl.get_device()              # -> torch.device

or via the ``FASTLSQ_DEVICE`` environment variable.  ``set_device`` also sets the
torch *default* device so that tensors created without an explicit ``device=``
(e.g. inside problem definitions) land on the same device.
"""

import os
import warnings

import torch

_FLOAT64 = (torch.float64, torch.double)


def _cuda_ok() -> bool:
    return torch.cuda.is_available()


def _mps_ok() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and mps.is_available()


def resolve_device(prefer=None, dtype=None) -> torch.device:
    """Resolve a ``torch.device``.

    Parameters
    ----------
    prefer : {'cuda','mps','cpu'} | torch.device | None
        Explicit request (falls back to the ``FASTLSQ_DEVICE`` env var).  If the
        requested accelerator is unavailable -- or MPS is requested with a
        float64 dtype -- a warning is issued and CPU is returned.
    dtype : torch.dtype | None
        The dtype the device must support.  ``None`` uses the current default
        dtype.  MPS is excluded whenever this is float64.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    is_double = dtype in _FLOAT64

    want = prefer if prefer is not None else os.environ.get("FASTLSQ_DEVICE", "").strip()
    if isinstance(want, torch.device):
        want = str(want)
    want = (want or "").lower()

    if want:
        if want.startswith("cuda"):
            if _cuda_ok():
                return torch.device(want)
            warnings.warn("FastLSQ: CUDA requested but unavailable; using CPU.")
            return torch.device("cpu")
        if want.startswith("mps"):
            if _mps_ok() and not is_double:
                return torch.device("mps")
            if _mps_ok() and is_double:
                warnings.warn("FastLSQ: MPS requested but the active dtype is "
                              "float64, which MPS does not support; using CPU. "
                              "Set float32 first to use the Apple GPU.")
            elif not _mps_ok():
                warnings.warn("FastLSQ: MPS requested but unavailable; using CPU.")
            return torch.device("cpu")
        return torch.device("cpu")

    # ---- auto ----
    if _cuda_ok():
        return torch.device("cuda")
    if _mps_ok() and not is_double:
        return torch.device("mps")
    return torch.device("cpu")


# FastLSQ targets the float64 high-accuracy regime by default (its ~1e-12
# results require it).  Set it at import so that auto device-selection excludes
# Apple-MPS (no float64 there) unless the user deliberately switches to float32.
torch.set_default_dtype(torch.float64)
_DEVICE = resolve_device()


def get_device() -> torch.device:
    """The active FastLSQ device."""
    return _DEVICE


def set_device(dev=None, dtype=None, set_torch_default=True) -> torch.device:
    """Set the active device.

    Parameters
    ----------
    dev : {'cuda','mps','cpu'} | torch.device | None
        ``None`` auto-resolves (CUDA > MPS(float32) > CPU).
    dtype : torch.dtype | None
        Used to keep MPS out for float64.
    set_torch_default : bool
        Also set the torch default device so tensors created without an explicit
        ``device=`` follow along (recommended for consistency).
    """
    global _DEVICE
    _DEVICE = resolve_device(prefer=dev, dtype=dtype)
    if set_torch_default and hasattr(torch, "set_default_device"):
        torch.set_default_device(_DEVICE)
    return _DEVICE


def device_info() -> dict:
    """A small dict describing the active device (handy for logging)."""
    d = _DEVICE
    info = {"device": str(d), "default_dtype": str(torch.get_default_dtype()).replace("torch.", "")}
    if d.type == "cuda":
        i = d.index if d.index is not None else torch.cuda.current_device()
        info["name"] = torch.cuda.get_device_name(i)
        info["total_mem_GB"] = round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2)
    elif d.type == "mps":
        info["name"] = "Apple MPS (float32 only)"
    else:
        info["name"] = "CPU"
    return info
