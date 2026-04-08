"""Tests that artefact separation prior respects window-geometry bounds."""
from __future__ import annotations

import jax
import numpy as np
import numpyro

from chromhandler.fitting import model
from chromhandler.fitting.types import ModelHyperparams

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
) -> dict:
    """Minimal model inputs for a single artefact_doublet peak."""
    import jax.numpy as jnp

    x = jnp.linspace(window_lo - 0.1, window_hi + 0.1, n_time)
    x = jnp.tile(x, (n_trace, 1))

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
        "w_left_loc": jnp.array([w_left_loc], dtype=jnp.float32),
        "w_left_scale": jnp.array([0.01], dtype=jnp.float32),
        "w_right_loc": jnp.array([w_right_loc], dtype=jnp.float32),
        "w_right_scale": jnp.array([0.01], dtype=jnp.float32),
        "area_gaussian_pt": jnp.ones((n_trace, 1), dtype=jnp.float32) * 100.0,
        "area_art_shared": jnp.array([10.0], dtype=jnp.float32),
        "snr_per_trace": jnp.ones((n_trace, 1), dtype=jnp.float32) * 10.0,
        "window_lo": jnp.array([window_lo], dtype=jnp.float32),
        "window_hi": jnp.array([window_hi], dtype=jnp.float32),
        "baseline_intercept_loc": jnp.zeros(n_trace, dtype=jnp.float32),
        "baseline_intercept_scale": jnp.ones(n_trace, dtype=jnp.float32) * 100.0,
        "baseline_slope_loc": jnp.zeros(n_trace, dtype=jnp.float32),
        "baseline_slope_scale": jnp.ones(n_trace, dtype=jnp.float32) * 10.0,
        "sigma_y_prior_loc": jnp.ones(n_trace, dtype=jnp.float32) * 50.0,
    }


def _sample_separation(inputs: dict, n_samples: int = 500) -> np.ndarray:
    """Run prior predictive and return log_separation_artefact samples."""
    from numpyro.infer import Predictive

    predictive = Predictive(model.model, num_samples=n_samples)
    rng_key = jax.random.PRNGKey(42)
    samples = predictive(rng_key, **inputs)
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


def test_separation_above_min():
    """Sampled separation must be above sep_min = art_sep_min_w_mult * min(w_left, w_right)."""
    hp = ModelHyperparams()
    w_left = 0.05
    w_right = 0.07
    inputs = _minimal_model_inputs(w_left_loc=w_left, w_right_loc=w_right)
    sep_samples = np.exp(_sample_separation(inputs))
    expected_min = hp.art_sep_min_w_mult * min(w_left, w_right)
    assert np.all(sep_samples[:, 0] > expected_min), (
        f"separation {sep_samples.min():.6f} below sep_min={expected_min:.6f}"
    )
