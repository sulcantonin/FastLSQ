# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License.

"""Smoke tests for the stealth-navigation world-model demo."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import stealth_navigation as sn  # noqa: E402


def test_radar_field_shape_and_finiteness():
    em = sn.make_emitters(np.random.default_rng(0))
    pts = np.random.default_rng(1).random((64, 2))
    u = sn.radar_field(pts, em)
    assert u.shape == (64,)
    assert np.isfinite(u).all()


def test_surrogate_gradient_matches_finite_differences():
    """The selling point: grad of the surrogate is closed-form and correct."""
    basis = sn.build_basis()
    wm = sn.WorldModel(basis)
    rng = np.random.default_rng(0)
    X = rng.random((400, 2))
    y = sn.radar_field(X, sn.make_emitters(rng))
    wm.refit(X, y)

    p = np.array([[0.5, 0.5]])
    g = wm.gradient(p)[0]
    h = 1e-6
    fd = np.array([
        (wm.predict(p + np.array([[h, 0.0]]))[0] - wm.predict(p - np.array([[h, 0.0]]))[0]) / (2 * h),
        (wm.predict(p + np.array([[0.0, h]]))[0] - wm.predict(p - np.array([[0.0, h]]))[0]) / (2 * h),
    ])
    assert np.allclose(g, fd, rtol=1e-4, atol=1e-4), f"analytic {g} vs fd {fd}"


@pytest.mark.parametrize("w_stealth,should_be_detected", [(0.0, True), (sn.W_STEALTH, False)])
def test_gradient_steering_avoids_the_detector(w_stealth, should_be_detected):
    """Driving straight trips the detector; steering on the analytic gradient does not."""
    r = sn.run_episode(verbose=False, w_stealth=w_stealth)
    assert r["detected"] is should_be_detected
    if not should_be_detected:
        assert r["reached"], "stealth run should still reach the goal"
        assert r["peak_u2"] < sn.TAU
