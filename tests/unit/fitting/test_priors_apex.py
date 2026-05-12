"""Tests for dominant apex detection."""

from __future__ import annotations

import numpy as np
from scipy.stats import skewnorm

from chromhandler.fitting.priors import detect_dominant_apex
from chromhandler.fitting.skew_normal import cp_to_dp


def test_dominant_apex_on_single_peak() -> None:
    t = np.arange(2.5, 2.9, 0.001)
    xi, omega, alpha = (float(x) for x in cp_to_dp(2.7, 0.03, 0.0))  # type: ignore[arg-type]
    s = skewnorm.pdf(t, alpha, loc=xi, scale=omega)
    apex_loc, apex_height = detect_dominant_apex(t, s, 2.5, 2.9)
    assert abs(apex_loc - 2.7) < 0.005 and apex_height > 0.0


def test_dominant_apex_picks_taller() -> None:
    t = np.arange(2.5, 2.9, 0.001)
    xi1, om1, a1 = (float(x) for x in cp_to_dp(2.65, 0.02, 0.0))  # type: ignore[arg-type]
    xi2, om2, a2 = (float(x) for x in cp_to_dp(2.75, 0.02, 0.0))  # type: ignore[arg-type]
    s = 1.0 * skewnorm.pdf(t, a1, loc=xi1, scale=om1) + 0.3 * skewnorm.pdf(
        t, a2, loc=xi2, scale=om2
    )
    apex_loc, _ = detect_dominant_apex(t, s, 2.5, 2.9)
    assert abs(apex_loc - 2.65) < 0.01


def test_dominant_apex_on_noise_returns_argmax() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(2.5, 2.9, 0.001)
    s = np.asarray(rng.normal(0.0, 1.0, size=t.shape), dtype=np.float64)
    apex_loc, apex_height = detect_dominant_apex(t, s, 2.5, 2.9)
    assert 2.5 <= apex_loc <= 2.9 and np.isfinite(apex_height)
