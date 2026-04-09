"""Tests for per-peak apex offset (independent peak position jitter)."""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpyro  # noqa: F401  # type: ignore[import]
import pytest  # noqa: F401  # type: ignore[import]
from numpyro.infer import Predictive

from chromhandler.fitting import model
from chromhandler.fitting.types import ModelHyperparams


def _minimal_inputs(
    apex_loc: float = 3.0,
    apex_offset_scale: float = 0.005,
    trace_shift_scale: float = 0.01,
    n_trace: int = 4,
    n_time: int = 30,
) -> dict[str, Any]:
    """Minimal single-peak model inputs with apex_offset_scale."""
    x = jnp.tile(jnp.linspace(2.8, 3.2, n_time)[None, :], (n_trace, 1))
    return {
        "x": x,
        "y": None,
        "hyperparams": ModelHyperparams(),
        "peak_mode_code": jnp.array([0], dtype=jnp.int32),
        "artefact_side": jnp.array([0], dtype=jnp.int32),
        "artefact_peak_index": jnp.zeros((0,), dtype=jnp.int32),
        "free_peak_index": jnp.zeros((0,), dtype=jnp.int32),
        "nonfree_idx": jnp.array([0], dtype=jnp.int32),
        "nonfree_position": jnp.array([0], dtype=jnp.int32),
        "apex_loc": jnp.array([apex_loc], dtype=jnp.float32),
        "apex_offset_scale": jnp.array([apex_offset_scale], dtype=jnp.float32),
        "trace_shift_scale": jnp.array(trace_shift_scale, dtype=jnp.float32),
        "w_left_loc": jnp.array([0.04], dtype=jnp.float32),
        "w_left_scale": jnp.array([0.005], dtype=jnp.float32),
        "w_right_loc": jnp.array([0.04], dtype=jnp.float32),
        "w_right_scale": jnp.array([0.005], dtype=jnp.float32),
        "area_gaussian_pt": jnp.ones((n_trace, 1), dtype=jnp.float32) * 500.0,
        "area_art_shared": jnp.zeros((0,), dtype=jnp.float32),
        "snr_per_trace": jnp.ones((n_trace, 1), dtype=jnp.float32) * 20.0,
        "window_lo": jnp.array([2.8], dtype=jnp.float32),
        "window_hi": jnp.array([3.2], dtype=jnp.float32),
        "baseline_intercept_loc": jnp.zeros(n_trace, dtype=jnp.float32),
        "baseline_intercept_scale": jnp.ones(n_trace, dtype=jnp.float32) * 50.0,
        "baseline_slope_loc": jnp.zeros(n_trace, dtype=jnp.float32),
        "baseline_slope_scale": jnp.ones(n_trace, dtype=jnp.float32) * 10.0,
        "sigma_y_prior_loc": jnp.ones(n_trace, dtype=jnp.float32) * 20.0,
    }


def test_apex_offset_raw_sampled():
    """apex_offset_raw must appear in prior predictive samples."""
    inputs = _minimal_inputs()
    predictive = Predictive(model.model, num_samples=10)
    samples = predictive(jax.random.PRNGKey(0), **inputs)
    assert "apex_offset_raw" in samples, "apex_offset_raw not in samples"


def test_apex_offset_raw_shape():
    """apex_offset_raw shape must be [n_samples, n_trace, n_peak]."""
    n_trace, n_peak = 4, 1
    inputs = _minimal_inputs(n_trace=n_trace)
    predictive = Predictive(model.model, num_samples=20)
    samples = predictive(jax.random.PRNGKey(1), **inputs)
    arr = samples["apex_offset_raw"]
    assert arr.shape == (20, n_trace, n_peak), f"wrong shape: {arr.shape}"


def test_zero_scale_means_no_offset():
    """With apex_offset_scale=0, apex_offset_raw samples don't affect the signal."""
    n_trace = 4
    inputs_zero = _minimal_inputs(n_trace=n_trace, apex_offset_scale=0.0)
    inputs_nonzero = _minimal_inputs(n_trace=n_trace, apex_offset_scale=0.1)
    predictive = Predictive(model.model, num_samples=50)
    rng = jax.random.PRNGKey(2)
    s_zero = predictive(rng, **inputs_zero)
    s_nonzero = predictive(rng, **inputs_nonzero)
    # With zero scale, apex_offset_raw has no effect — variance in signal should be lower
    var_zero = float(jnp.var(s_zero["y"]))
    var_nonzero = float(jnp.var(s_nonzero["y"]))
    assert var_nonzero > var_zero, (
        f"nonzero apex_offset_scale should increase signal variance: "
        f"var_zero={var_zero:.1f}, var_nonzero={var_nonzero:.1f}"
    )


def test_apex_offset_independent_per_peak():
    """With two peaks, apex_offset_raw[:,0] and apex_offset_raw[:,1] are independent."""
    n_trace = 8
    x = jnp.tile(jnp.linspace(2.7, 3.5, 40)[None, :], (n_trace, 1))
    # Two single peaks
    inputs = _minimal_inputs(n_trace=n_trace)
    # Override to 2 peaks
    inputs["peak_mode_code"] = jnp.array([0, 0], dtype=jnp.int32)
    inputs["artefact_side"] = jnp.array([0, 0], dtype=jnp.int32)
    inputs["nonfree_idx"] = jnp.array([0, 1], dtype=jnp.int32)
    inputs["nonfree_position"] = jnp.array([0, 1], dtype=jnp.int32)
    inputs["apex_loc"] = jnp.array([2.9, 3.2], dtype=jnp.float32)
    inputs["apex_offset_scale"] = jnp.array([0.01, 0.01], dtype=jnp.float32)
    inputs["w_left_loc"] = jnp.array([0.04, 0.04], dtype=jnp.float32)
    inputs["w_left_scale"] = jnp.array([0.005, 0.005], dtype=jnp.float32)
    inputs["w_right_loc"] = jnp.array([0.04, 0.04], dtype=jnp.float32)
    inputs["w_right_scale"] = jnp.array([0.005, 0.005], dtype=jnp.float32)
    inputs["area_gaussian_pt"] = jnp.ones((n_trace, 2), dtype=jnp.float32) * 500.0
    inputs["snr_per_trace"] = jnp.ones((n_trace, 2), dtype=jnp.float32) * 20.0
    inputs["window_lo"] = jnp.array([2.7, 3.1], dtype=jnp.float32)
    inputs["window_hi"] = jnp.array([3.1, 3.5], dtype=jnp.float32)
    inputs["x"] = x

    predictive = Predictive(model.model, num_samples=200)
    samples = predictive(jax.random.PRNGKey(3), **inputs)
    off = samples["apex_offset_raw"]  # [200, n_trace, 2]
    # Correlation between peak 0 and peak 1 offsets should be low (< 0.3)
    off0 = off[:, :, 0].ravel()
    off1 = off[:, :, 1].ravel()
    corr = float(jnp.corrcoef(off0, off1)[0, 1])
    assert abs(corr) < 0.3, f"peaks not independent: corr={corr:.3f}"
