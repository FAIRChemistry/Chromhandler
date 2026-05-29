"""Tests for the prior-construction layer in ``chromhandler.fitting.priors``.

The critical correctness contract is that ``compute_window_features`` returns
CP parameters ``(mu, width, skew)`` that match the model's CP parameterisation,
i.e. that recovering features from a synthetic SN density with known CP truth
returns those CP values — not the apex (mode) and Gaussian-FWHM-converted
width that an earlier implementation produced.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.prepared_dataset import prepare_dataset
from chromhandler.fitting.priors import PriorConfig, build_priors, compute_window_features
from chromhandler.fitting.skew_normal import (
    GAMMA1_MAX,
    cp_from_peak_features,
    cp_to_dp,
    density_cp,
    fwhm_dp,
    hwhm_ratio_dp,
    mode_dp,
)


@pytest.mark.parametrize(
    "skew_true",
    [
        -0.95 * GAMMA1_MAX,
        -0.50,
        -0.20,
        0.0,
        0.20,
        0.50,
        0.95 * GAMMA1_MAX,
    ],
)
def test_cp_from_peak_features_inverts_skew_normal(skew_true: float) -> None:
    """Round-trip: build CP truth, derive apex+FWHM+HWHM-ratio, recover CP.

    The three measured quantities are computed directly from the SN math
    helpers, so this test isolates :func:`cp_from_peak_features` from any
    signal-processing noise (no smoothing, no discretisation).
    """
    mu_true = 5.0
    sigma_true = 0.2
    xi, omega, alpha = cp_to_dp(
        jnp.asarray(mu_true),
        jnp.asarray(sigma_true),
        jnp.asarray(skew_true),
    )
    apex = float(mode_dp(xi, omega, alpha))
    fwhm = float(fwhm_dp(np.asarray(xi), np.asarray(omega), np.asarray(alpha)))
    hwhm_ratio = float(
        hwhm_ratio_dp(np.asarray(xi), np.asarray(omega), np.asarray(alpha))
    )

    mu, sigma, skew = cp_from_peak_features(apex, fwhm, hwhm_ratio)

    # Precision is bounded by the asymmetry-to-skew table interpolation
    # (~1e-3 in skew at table grid stride alpha=0.25), which propagates to
    # roughly atol=0.01*sigma_true on mu. Far below the pre-fix bias
    # (~0.65*sigma_true at skew_true=0.85) so the test still catches the
    # original bug if reintroduced.
    np.testing.assert_allclose(mu, mu_true, atol=0.01 * sigma_true)
    np.testing.assert_allclose(sigma, sigma_true, rtol=0.01)
    np.testing.assert_allclose(skew, skew_true, atol=0.01)


@pytest.mark.parametrize(
    "skew_true",
    [-0.85, -0.50, -0.20, 0.0, 0.20, 0.50, 0.85],
)
def test_compute_window_features_recovers_cp_on_clean_synthetic_peak(
    skew_true: float,
) -> None:
    """Full pipeline: synthetic SN density → compute_window_features → CP truth.

    Runs the noise-free Savitzky-Golay smoothing path on a dense sampling
    of a known SN density, then asserts that the returned ``(mu, width,
    skew)`` match the true CP parameters to within tight bias bounds.

    Window is set to plus/minus 3*sigma around the true mean — close to
    what a real PeakAnnotation specifies. ``width_low_n_points_per_fwhm``
    defaults to 8, so 401 points across 1.2 minutes is ample resolution.
    """
    mu_true = 5.0
    sigma_true = 0.2
    window_low = mu_true - 3.0 * sigma_true
    window_high = mu_true + 3.0 * sigma_true
    t = np.linspace(window_low, window_high, 401)
    density = np.asarray(
        density_cp(
            jnp.asarray(t),
            jnp.asarray(mu_true),
            jnp.asarray(sigma_true),
            jnp.asarray(skew_true),
        )
    )
    signal = density * 1000.0  # arbitrary scale; FWHM/asymmetry are scale-free

    feat = compute_window_features(t, signal, window_low, window_high)
    assert feat is not None

    # mu: pre-fix this was off by mode_dp - mu_true (~0.65*sigma at
    # skew_true=0.85); post-fix it tracks the true mean to within
    # ~0.015*sigma. Threshold is loose enough to absorb smoothing-induced
    # apex jitter, tight enough to catch the original apex/mode bug.
    np.testing.assert_allclose(feat.mu, mu_true, atol=0.05 * sigma_true)
    # width: pre-fix this was off by ~12% at skew_true=0.85 (Gaussian
    # FWHM-to-sigma rule applied to SN FWHM); post-fix it matches CP sigma
    # to within ~1%.
    np.testing.assert_allclose(feat.width, sigma_true, rtol=0.03)
    # skew: the asymmetry inversion was internally consistent already;
    # discretisation + smoothing add a small bias but it stays small.
    np.testing.assert_allclose(feat.skew, skew_true, atol=0.02)


def test_compute_window_features_returns_none_when_too_few_points() -> None:
    """Below the smoothing window count returns None instead of raising."""
    t = np.linspace(0.0, 1.0, 4)
    s = np.zeros_like(t)
    feat = compute_window_features(t, s, 0.0, 1.0, smoothing_window=5)
    assert feat is None


def _toy_area_dataset():
    """2 traces: trace 0 has a clear Gaussian peak (supported); trace 1 is
    flat baseline (unsupported). Small noise so the noise floor is realistic."""
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 10.0, 0.05)
    peak = 100.0 * np.exp(-0.5 * ((t - 5.0) / 0.2) ** 2)
    s0 = peak + 1.0 + rng.normal(0.0, 0.5, t.shape)
    s1 = 1.0 + rng.normal(0.0, 0.5, t.shape)
    peak_anns = [PeakAnnotation(molecule_id="x", rt_min=4.0, rt_max=6.0, mode="single")]
    base_anns = [
        BaselineAnnotation(rt_min=0.0, rt_max=1.0),
        BaselineAnnotation(rt_min=9.0, rt_max=10.0),
    ]
    return prepare_dataset([t, t], [s0, s1], peak_anns, base_anns)


def test_area_prior_is_lognormal_positive_with_fixed_scale():
    ds = _toy_area_dataset()
    priors = build_priors(ds, PriorConfig(signal_threshold=10.0))
    p = priors[0]
    assert np.all(p.area_loc_per_trace > 0.0)
    assert p.area_log_scale == 1.0
    assert bool(p.has_support_per_trace[0])
    assert not bool(p.has_support_per_trace[1])
    assert p.area_loc_per_trace[0] > p.area_loc_per_trace[1]
    assert 30.0 < p.area_loc_per_trace[0] < 70.0


def test_area_sigma_log_is_configurable():
    ds = _toy_area_dataset()
    priors = build_priors(ds, PriorConfig(signal_threshold=10.0, area_sigma_log=0.5))
    assert priors[0].area_log_scale == 0.5
