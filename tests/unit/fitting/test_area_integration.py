"""Regression test for review item 7: signed (not rectified) area integration.

``_trapezoid_in_window`` feeds the LogNormal area-prior median. Clipping the
baseline-subtracted signal with ``max(.,0)`` before integrating rectifies
the noise and biases the area up by ``~sigma/sqrt(2pi) * window_width`` —
spurious for weak/absent windows, ~few % for real peaks. Integration must be
signed.
"""
import numpy as np

from chromhandler.fitting.priors import _trapezoid_in_window


def test_trapezoid_in_window_is_signed_not_rectified():
    t = np.arange(0.0, 2.0, 0.01)
    # Two full periods of a sine over [0, 2]: signed integral is ~0, but the
    # rectified integral (max(.,0)) is ~0.64 — a clear discriminator.
    s = np.sin(2.0 * np.pi * t)

    area = _trapezoid_in_window(t, s, 0.0, 2.0)
    assert abs(area) < 0.05, f"expected signed ~0, got {area}"

    rectified = float(np.trapezoid(np.maximum(s, 0.0), t))
    assert rectified > 0.5, "sanity: rectifying WOULD bias high"


def test_trapezoid_in_window_recovers_known_signed_area():
    t = np.linspace(0.0, 1.0, 1001)
    s = 3.0 * np.ones_like(t)  # constant 3 over unit window -> area 3
    area = _trapezoid_in_window(t, s, 0.0, 1.0)
    np.testing.assert_allclose(area, 3.0, rtol=1e-6)
