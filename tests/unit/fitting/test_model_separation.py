"""Tests that artefact separation prior respects window-geometry bounds."""
from __future__ import annotations

from typing import Any

import jax
import numpy as np
import numpyro

from chromhandler.fitting import model
from chromhandler.fitting.types import ModelHyperparams
from tests.unit.fitting.conftest import w_min_from_dt

numpyro.enable_x64()


def _minimal_model_inputs(
    *,
    apex_loc: float = 3.0,
    window_lo: float = 2.85,
    window_hi: float = 3.15,
    artefact_side: int = 1,          # +1 = right artefact
    w_left_loc: float = 0.05,
    w_right_loc: float = 0.05,
    trace_shift_scale: float = 0.005,
    n_trace: int = 3,
    n_time: int = 50,
    area_art_shared: float = 10.0,
) -> dict[str, Any]:
    """Minimal model inputs for a single artefact_doublet peak."""
    import jax.numpy as jnp

    x = jnp.linspace(window_lo - 0.1, window_hi + 0.1, n_time)
    x = jnp.tile(x, (n_trace, 1))

    # Geometry-derived bounds (mirror priors.py).  dt comes from the actual
    # x-axis (which extends ±0.1 beyond the peak window); w_max is bounded by
    # the annotated peak window.
    dt = float((window_hi + 0.1 - (window_lo - 0.1)) / (n_time - 1))
    w_min_val = w_min_from_dt(dt)
    w_max_val = (window_hi - window_lo) / 4.0

    return {
        "x": x,
        "y": None,
        "hyperparams": ModelHyperparams(),
        "peak_mode_code": jnp.array([1], dtype=jnp.int32),       # artefact_doublet
        "artefact_side": jnp.array([artefact_side], dtype=jnp.int32),
        "artefact_peak_index": jnp.array([0], dtype=jnp.int32),
        "free_peak_index": jnp.array([], dtype=jnp.int32),
        "nonfree_idx": jnp.array([0], dtype=jnp.int32),
        "nonfree_position": jnp.array([0], dtype=jnp.int32),
        "apex_loc": jnp.array([apex_loc], dtype=jnp.float32),
        "trace_shift_scale": jnp.array(trace_shift_scale, dtype=jnp.float32),
        "apex_offset_scale": jnp.array([0.005], dtype=jnp.float32),
        "w_left_loc": jnp.array([w_left_loc], dtype=jnp.float32),
        "w_left_scale": jnp.array([0.01], dtype=jnp.float32),
        "w_right_loc": jnp.array([w_right_loc], dtype=jnp.float32),
        "w_right_scale": jnp.array([0.01], dtype=jnp.float32),
        "w_min": jnp.array([w_min_val], dtype=jnp.float32),
        "w_max": jnp.array([w_max_val], dtype=jnp.float32),
        "dt": jnp.array([dt], dtype=jnp.float32),
        "n_valid": jnp.array([float(n_trace)], dtype=jnp.float32),
        "area_gaussian_pt": jnp.ones((n_trace, 1), dtype=jnp.float32) * 100.0,
        "area_art_shared": jnp.array([area_art_shared], dtype=jnp.float32),
        "window_lo": jnp.array([window_lo], dtype=jnp.float32),
        "window_hi": jnp.array([window_hi], dtype=jnp.float32),
        "baseline_intercept_loc": jnp.zeros(n_trace, dtype=jnp.float32),
        "baseline_intercept_scale": jnp.ones(n_trace, dtype=jnp.float32) * 100.0,
        "baseline_slope_loc": jnp.zeros(n_trace, dtype=jnp.float32),
        "baseline_slope_scale": jnp.ones(n_trace, dtype=jnp.float32) * 10.0,
        "sigma_y_prior_loc": jnp.ones(n_trace, dtype=jnp.float32) * 50.0,
    }


def _sample_prior(inputs: dict[str, Any], n_samples: int = 500) -> dict[str, Any]:
    """Run prior predictive and return all samples."""
    from numpyro.infer import Predictive

    predictive = Predictive(model.model, num_samples=n_samples)
    rng_key = jax.random.PRNGKey(42)
    return predictive(rng_key, **inputs)


def _sample_separation(inputs: dict[str, Any], n_samples: int = 500) -> np.ndarray[Any, Any]:
    """Run prior predictive and return log_separation_artefact samples."""
    samples = _sample_prior(inputs, n_samples)
    return np.asarray(samples["log_separation_artefact"])  # [n_samples, n_artefact]


def test_separation_right_artefact_within_window():
    """Sampled separation must not push the artefact apex past window_hi."""
    apex_loc = 3.0
    window_hi = 3.15
    trace_shift_scale = 0.005
    inputs = _minimal_model_inputs(
        apex_loc=apex_loc,
        window_hi=window_hi,
        artefact_side=1,  # right
        trace_shift_scale=trace_shift_scale,
    )
    sep_samples = np.exp(_sample_separation(inputs))  # [n_samples, 1]
    expected_max = window_hi - apex_loc - trace_shift_scale
    assert np.all(sep_samples[:, 0] < expected_max), (
        f"separation {sep_samples.max():.4f} exceeded sep_max={expected_max:.4f}"
    )


def test_separation_left_artefact_within_window():
    """Sampled separation must not push the artefact apex below window_lo."""
    apex_loc = 3.31
    window_lo = 3.15
    trace_shift_scale = 0.005
    inputs = _minimal_model_inputs(
        apex_loc=apex_loc,
        window_lo=window_lo,
        artefact_side=-1,  # left
        trace_shift_scale=trace_shift_scale,
    )
    sep_samples = np.exp(_sample_separation(inputs))
    expected_max = apex_loc - window_lo - trace_shift_scale
    assert np.all(sep_samples[:, 0] < expected_max), (
        f"separation {sep_samples.max():.4f} exceeded sep_max={expected_max:.4f}"
    )


def test_separation_above_sampled_artefact_hwhm():
    """Sampled separation must be ≥ sampled artefact HWHM (identifiability) per sample.

    Under the new rule sep_min = exp(log_w_art), so each draw's separation is
    bounded below by that same draw's artefact HWHM (modulo the room/2 clamp).
    """
    w_left = 0.05
    w_right = 0.07
    apex_loc = 3.0
    window_hi = 3.15
    trace_shift_scale = 0.005
    inputs = _minimal_model_inputs(
        w_left_loc=w_left,
        w_right_loc=w_right,
        apex_loc=apex_loc,
        window_hi=window_hi,
        trace_shift_scale=trace_shift_scale,
        artefact_side=1,
    )
    samples = _sample_prior(inputs)
    sep_samples = np.exp(np.asarray(samples["log_separation_artefact"]))[:, 0]
    w_art_samples = np.exp(np.asarray(samples["log_w_art"]))[:, 0]
    room = window_hi - apex_loc - trace_shift_scale
    effective_min = np.minimum(w_art_samples, room * 0.5)
    assert np.all(sep_samples >= effective_min - 1e-6), (
        f"{np.sum(sep_samples < effective_min)} / {sep_samples.size} samples "
        f"fell below sampled artefact HWHM"
    )


def test_artefact_width_is_symmetric():
    """Artefact second component must have sl == sr (symmetric split-normal).

    Default artefact_side=+1 (right), so artefact is the right component.
    Check that sl_r == sr_r for peak index 0.
    """
    inputs = _minimal_model_inputs()
    samples = _sample_prior(inputs)
    derived = model.compute_derived_quantities(
        samples, inputs, inputs["hyperparams"]
    )
    sl_art_vals = np.asarray(derived["sl_r"])[:, :, 0]
    sr_art_vals = np.asarray(derived["sr_r"])[:, :, 0]
    np.testing.assert_allclose(sl_art_vals, sr_art_vals, rtol=1e-5)


def test_artefact_median_narrower_than_primary():
    """Artefact prior centre (geomean of w_min and primary) must be < primary FWHM.

    With the principled prior, the centre is ``sqrt(w_min * w_primary)`` which
    is always < ``w_primary`` whenever ``w_min < w_primary``.  The prior can
    still be wide, so we test the posterior median rather than a tail fraction.
    """
    inputs = _minimal_model_inputs()
    samples = _sample_prior(inputs)
    log_w_art = np.asarray(samples["log_w_art"])  # [n_samples, n_artefact]
    w_art = np.exp(log_w_art[:, 0])
    w_primary_mean = 0.5 * (
        float(inputs["w_left_loc"][0]) + float(inputs["w_right_loc"][0])
    )
    assert float(np.median(w_art)) < w_primary_mean, (
        f"Median artefact width {np.median(w_art):.5f} not below primary {w_primary_mean:.5f}"
    )


def test_artefact_area_varies_across_traces():
    """Per-trace artefact areas must differ — hierarchical prior gives non-zero spread."""
    n_trace = 5
    inputs = _minimal_model_inputs(n_trace=n_trace)
    samples = _sample_prior(inputs, n_samples=200)
    # log_area_art_raw shape: [n_samples, n_trace, n_artefact]
    raw = np.asarray(samples["log_area_art_raw"])
    assert raw.shape == (200, n_trace, 1), f"unexpected shape {raw.shape}"
    # Across traces (axis=1), each sample's per-trace offsets must have non-zero spread
    trace_std = raw.std(axis=1)  # [n_samples, n_artefact]
    assert np.all(trace_std > 0), "per-trace raw offsets have zero spread — hierarchy collapsed"


def test_artefact_area_mean_near_prior_centre():
    """Population mean log_area_art_mean must be centred on the input prior."""
    area_art_shared = 10.0
    inputs = _minimal_model_inputs(area_art_shared=area_art_shared)
    samples = _sample_prior(inputs, n_samples=500)
    mean_samples = np.asarray(samples["log_area_art_mean"])  # [n_samples, n_artefact]
    expected_center = np.log(area_art_shared)
    actual_center = float(mean_samples[:, 0].mean())
    assert abs(actual_center - expected_center) < 0.15, (
        f"log_area_art_mean drifted from prior centre: {actual_center:.3f} vs {expected_center:.3f}"
    )
