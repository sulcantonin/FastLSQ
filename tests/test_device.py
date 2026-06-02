# Copyright (c) 2026 Antonin Sulc -- MIT.
"""Tests for the dtype-aware device abstraction (CPU / CUDA / Apple-MPS)."""
import torch

import fastlsq
from fastlsq.device import resolve_device, set_device, get_device, device_info


def test_float64_never_resolves_to_mps():
    # MPS has no float64 -> must fall back to CPU/CUDA for the high-accuracy path.
    assert resolve_device(prefer="mps", dtype=torch.float64).type != "mps"
    assert resolve_device(dtype=torch.float64).type in ("cpu", "cuda")


def test_float32_allows_mps_when_present():
    d = resolve_device(prefer="mps", dtype=torch.float32)
    if torch.backends.mps.is_available():
        assert d.type == "mps"
    else:
        assert d.type == "cpu"


def test_explicit_cpu():
    assert resolve_device(prefer="cpu").type == "cpu"


def test_set_get_device_roundtrip():
    orig = get_device()
    try:
        set_device("cpu")
        assert get_device().type == "cpu"
    finally:
        set_device(orig)


def test_device_info_has_expected_keys():
    info = device_info()
    assert {"device", "name", "default_dtype"} <= set(info)


def test_device_api_is_public():
    for name in ("resolve_device", "set_device", "get_device", "device_info"):
        assert hasattr(fastlsq, name)
