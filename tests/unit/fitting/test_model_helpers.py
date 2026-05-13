"""Tests for model.py private helpers."""

from __future__ import annotations

import numpy as np
import pytest

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.model import (
    SAMPLED_LEFT_PER_TRACE,
    SAMPLED_LEFT_SHARED,
    SAMPLED_TRACE_NUISANCE,
    _baseline_contribution,
    _compute_baseline_se,
    _left_component_contribution,
    _validate_single_mode_only,
)
from chromhandler.fitting.prepared_dataset import prepare_dataset
from chromhandler.fitting.priors import PriorConfig, build_priors


def _toy_dataset(n_trace: int = 3):
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 5.0, 0.01)
    times = [t.copy() for _ in range(n_trace)]
    signals = [10.0 + 0.5 * t + rng.normal(0.0, 0.1, size=t.shape) for _ in range(n_trace)]
    peaks = [PeakAnnotation(molecule_id="A", rt_min=2.0, rt_max=3.0, mode="single")]
    bases = [BaselineAnnotation(rt_min=0.5, rt_max=1.5),
             BaselineAnnotation(rt_min=3.5, rt_max=4.5)]
    return prepare_dataset(times, signals, peaks, bases)  # type: ignore[arg-type]


def test_sample_name_constants_present() -> None:
    assert "mu_anchor_left" in SAMPLED_LEFT_SHARED
    assert "log_sigma_left" in SAMPLED_LEFT_SHARED
    assert "gamma1_left" in SAMPLED_LEFT_SHARED
    assert "log_A_left" in SAMPLED_LEFT_PER_TRACE
    assert "trace_shift" in SAMPLED_TRACE_NUISANCE
    assert "baseline_intercept" in SAMPLED_TRACE_NUISANCE
    assert "baseline_slope" in SAMPLED_TRACE_NUISANCE


def test_validate_single_mode_passes_on_single() -> None:
    ds = _toy_dataset()
    priors = build_priors(ds, config=PriorConfig())
    _validate_single_mode_only(priors)  # no raise


def test_validate_single_mode_raises_on_doublet() -> None:
    # Hand-construct a doublet prior by replace
    import dataclasses

    ds = _toy_dataset()
    p = build_priors(ds, config=PriorConfig())[0]
    p_doublet = dataclasses.replace(
        p,
        n_components=2,
        Delta_loc=0.05, Delta_scale=0.005, Delta_low=0.003, Delta_high=0.125,
        log_sigma_right_loc=p.log_sigma_left_loc, log_sigma_right_scale=p.log_sigma_left_scale,
        log_sigma_right_low=p.log_sigma_left_low, log_sigma_right_high=p.log_sigma_left_high,
        gamma1_right_loc=p.gamma1_left_loc, gamma1_right_scale=p.gamma1_left_scale,
        log_A_right_loc_per_trace=p.log_A_left_loc_per_trace, log_A_right_scale=p.log_A_left_scale,
    )
    with pytest.raises(NotImplementedError, match="single"):
        _validate_single_mode_only([p_doublet])


def test_compute_baseline_se_returns_per_trace() -> None:
    ds = _toy_dataset(n_trace=3)
    intercept_se, slope_se = _compute_baseline_se(ds)
    assert intercept_se.shape == (3,)
    assert slope_se.shape == (3,)
    assert np.all(intercept_se >= 0)
    assert np.all(slope_se >= 0)


def test_baseline_contribution_shape() -> None:
    ds = _toy_dataset(n_trace=3)
    intercept = np.zeros(3)
    slope = np.zeros(3)
    bc = _baseline_contribution(ds.time, intercept, slope)
    assert bc.shape == ds.time.shape
    np.testing.assert_array_equal(bc, np.zeros_like(ds.time))


def test_left_component_contribution_shape_and_finite() -> None:
    ds = _toy_dataset(n_trace=3)
    n_peak = 1
    mu_anchor = np.array([2.5])
    trace_shift = np.zeros(3)
    log_sigma = np.array([np.log(0.05)])
    gamma1 = np.array([0.0])
    log_A = np.full((3, n_peak), np.log(10.0))
    out = _left_component_contribution(ds.time, mu_anchor, trace_shift, log_sigma, gamma1, log_A)
    assert out.shape == ds.time.shape
    assert np.all(np.isfinite(out))
    # Signal should be positive everywhere a peak can exist
    assert float(out.max()) > 0
