"""Property tests for the pure-math skew-normal layer."""

from __future__ import annotations

import math

from scipy.stats import skewnorm

from chromhandler.fitting import skew_normal as sn


def test_gamma1_max_matches_half_normal_limit():
    """GAMMA1_MAX equals the skewness of the half-normal (a -> inf limit)."""
    expected: float = float(skewnorm.stats(a=1e6, moments="s"))  # type: ignore[arg-type]
    assert math.isclose(sn.GAMMA1_MAX, expected, rel_tol=1e-6)


def test_gamma1_max_closed_form():
    """GAMMA1_MAX matches the closed-form expression in spec §2.2."""
    b = math.sqrt(2.0 / math.pi)
    expected = ((4.0 - math.pi) / 2.0) * b**3 / (1.0 - 2.0 / math.pi) ** 1.5
    assert math.isclose(sn.GAMMA1_MAX, expected, rel_tol=1e-12)
