"""Mathematical property tests for the split-normal PDF."""
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

# These imports will fail until Task 2 implements the functions
from chromhandler.fitting._legacy_model import log_split_normal_pdf, split_normal_pdf

_HWHM_FACTOR = math.sqrt(2.0 * math.log(2.0))


def _eval(apex: float, sl: float, sr: float, x_arr: np.ndarray) -> np.ndarray:
    """Evaluate split_normal_pdf with scalar params on a 1-D x array."""
    x_2d = jnp.array(x_arr[None, :])          # [1, n]
    ap = jnp.array([[apex]], dtype=jnp.float32)
    sl_ = jnp.array([[sl]], dtype=jnp.float32)
    sr_ = jnp.array([[sr]], dtype=jnp.float32)
    pdf = split_normal_pdf(x_2d, ap, sl_, sr_)  # [1, 1, n]
    return np.array(pdf[0, 0, :])


def test_symmetric_integrates_to_one():
    """Symmetric split-normal (sl == sr) integrates to 1."""
    x = np.linspace(-10.0, 10.0, 50_000)
    dx = x[1] - x[0]
    pdf = _eval(0.0, 1.0, 1.0, x)
    assert abs(pdf.sum() * dx - 1.0) < 0.005


def test_asymmetric_integrates_to_one():
    """Asymmetric split-normal (sl != sr) integrates to 1."""
    x = np.linspace(-10.0, 10.0, 50_000)
    dx = x[1] - x[0]
    pdf = _eval(0.0, 0.4, 1.2, x)
    assert abs(pdf.sum() * dx - 1.0) < 0.005


def test_mode_exactly_at_apex():
    """PDF maximum must be at the specified apex."""
    x = np.linspace(-3.0, 5.0, 100_000)
    apex = 1.3
    pdf = _eval(apex, 0.6, 1.1, x)
    peak_x = float(x[np.argmax(pdf)])
    assert abs(peak_x - apex) < 0.001


def test_density_continuous_at_apex():
    """Density must be continuous at apex (no jump)."""
    apex_val = 0.7
    sl, sr = 0.5, 0.9
    eps = 1e-4
    x = np.array([apex_val - eps, apex_val, apex_val + eps])
    pdf = _eval(apex_val, sl, sr, x)
    # All three values should differ by < 0.1 %
    assert abs(pdf[0] - pdf[1]) / pdf[1] < 0.001
    assert abs(pdf[2] - pdf[1]) / pdf[1] < 0.001


def test_left_hwhm_matches_sl():
    """Left HWHM of the PDF should equal sl * sqrt(2*ln2)."""
    apex_val, sl_val, sr_val = 0.0, 0.5, 1.0
    x = np.linspace(-5.0, 5.0, 200_000)
    pdf = _eval(apex_val, sl_val, sr_val, x)
    peak_val = pdf.max()
    half_max = 0.5 * peak_val
    # Left side: x < apex
    left_x = x[x < apex_val]
    left_pdf = pdf[x < apex_val]
    left_cross = float(np.interp(half_max, left_pdf, left_x))
    measured = apex_val - left_cross
    expected = sl_val * _HWHM_FACTOR
    assert abs(measured - expected) / expected < 0.005


def test_right_hwhm_matches_sr():
    """Right HWHM of the PDF should equal sr * sqrt(2*ln2)."""
    apex_val, sl_val, sr_val = 0.0, 0.5, 1.0
    x = np.linspace(-5.0, 5.0, 200_000)
    pdf = _eval(apex_val, sl_val, sr_val, x)
    peak_val = pdf.max()
    half_max = 0.5 * peak_val
    right_x = x[x >= apex_val]
    right_pdf = pdf[x >= apex_val]
    right_cross = float(np.interp(half_max, right_pdf[::-1], right_x[::-1]))
    measured = right_cross - apex_val
    expected = sr_val * _HWHM_FACTOR
    assert abs(measured - expected) / expected < 0.005


def test_log_pdf_matches_pdf():
    """`log_split_normal_pdf` is consistent with `split_normal_pdf`."""
    x = np.linspace(-3.0, 3.0, 500)
    x_2d = jnp.array(x[None, :])
    ap = jnp.array([[0.2]], dtype=jnp.float32)
    sl_ = jnp.array([[0.7]], dtype=jnp.float32)
    sr_ = jnp.array([[1.3]], dtype=jnp.float32)
    log_p = np.array(log_split_normal_pdf(x_2d, ap, sl_, sr_)[0, 0, :])
    p = np.array(split_normal_pdf(x_2d, ap, sl_, sr_)[0, 0, :])
    np.testing.assert_allclose(np.exp(log_p), p, rtol=1e-5)
