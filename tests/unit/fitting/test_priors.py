"""Unit tests for prior geometry and baseline estimation (no MCMC).

Extracted from:
  - tests/fitting/test_better_model.py

Content: Prior creation, validation, window bounds (geometry and baseline only).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.baseline import BaselinePriors, estimate_baseline
from chromhandler.fitting.priors import GeometricPeakPriors


@pytest.mark.unit
def test_geometric_peak_priors_single_mode() -> None:
    """GeometricPeakPriors can be created with single mode."""
    priors = GeometricPeakPriors(
        mode="single",
        apex_loc=5.0,
        apex_scale=0.05,
        sigma_loc=0.03,
        sigma_scale=0.005,
        alpha_loc=0.15,
        alpha_scale=0.03,
        main_area_per_trace=np.array([100.0, 95.0], dtype=float),
        total_area_per_trace=np.array([102.0, 97.0], dtype=float),
        artefact_shoulder_area_loc=0.0,
        window_lo=4.8,
        window_hi=5.2,
        n_valid_traces=2,
    )
    assert priors.mode == "single"
    assert priors.apex_loc == 5.0
    assert priors.window_lo == 4.8
    assert priors.window_hi == 5.2


@pytest.mark.unit
def test_geometric_peak_priors_artefact_mode() -> None:
    """GeometricPeakPriors can be created with artefact_doublet mode."""
    priors = GeometricPeakPriors(
        mode="artefact_doublet",
        apex_loc=2.5,
        apex_scale=0.02,
        sigma_loc=0.04,
        sigma_scale=0.006,
        alpha_loc=0.20,
        alpha_scale=0.04,
        main_area_per_trace=np.array([80.0], dtype=float),
        total_area_per_trace=np.array([110.0], dtype=float),
        artefact_shoulder_area_loc=1.5,
        window_lo=2.3,
        window_hi=2.85,
        n_valid_traces=1,
    )
    assert priors.mode == "artefact_doublet"
    assert priors.artefact_shoulder_area_loc == 1.5


@pytest.mark.unit
def test_geometric_peak_priors_free_doublet_mode() -> None:
    """GeometricPeakPriors can be created with free_doublet mode."""
    priors = GeometricPeakPriors(
        mode="free_doublet",
        apex_loc=4.2,
        apex_scale=0.03,
        sigma_loc=0.06,
        sigma_scale=0.008,
        alpha_loc=0.25,
        alpha_scale=0.04,
        main_area_per_trace=np.array([150.0, 145.0], dtype=float),
        total_area_per_trace=np.array([155.0, 148.0], dtype=float),
        artefact_shoulder_area_loc=0.0,
        window_lo=3.9,
        window_hi=4.5,
        n_valid_traces=2,
    )
    assert priors.mode == "free_doublet"
    assert priors.n_valid_traces == 2


@pytest.mark.unit
def test_geometric_peak_priors_window_bounds() -> None:
    """GeometricPeakPriors window bounds are stored correctly."""
    priors = GeometricPeakPriors(
        mode="single",
        apex_loc=5.0,
        apex_scale=0.05,
        sigma_loc=0.03,
        sigma_scale=0.005,
        alpha_loc=0.15,
        alpha_scale=0.03,
        main_area_per_trace=np.array([100.0], dtype=float),
        total_area_per_trace=np.array([102.0], dtype=float),
        artefact_shoulder_area_loc=0.0,
        window_lo=4.8,
        window_hi=5.2,
        n_valid_traces=1,
    )
    assert priors.window_hi > priors.window_lo
    width = priors.window_hi - priors.window_lo
    assert abs(width - 0.4) < 1e-9


@pytest.mark.unit
def test_geometric_peak_priors_apex_scales_positive() -> None:
    """GeometricPeakPriors apex_scale must be positive."""
    priors = GeometricPeakPriors(
        mode="single",
        apex_loc=5.0,
        apex_scale=0.05,
        sigma_loc=0.03,
        sigma_scale=0.005,
        alpha_loc=0.15,
        alpha_scale=0.03,
        main_area_per_trace=np.array([100.0], dtype=float),
        total_area_per_trace=np.array([102.0], dtype=float),
        artefact_shoulder_area_loc=0.0,
        window_lo=4.8,
        window_hi=5.2,
        n_valid_traces=1,
    )
    assert priors.apex_scale > 0.0
    assert priors.sigma_scale > 0.0


@pytest.mark.unit
def test_geometric_peak_priors_area_arrays() -> None:
    """GeometricPeakPriors main_area_per_trace is correct shape."""
    trace_areas = np.array([100.0, 95.0, 105.0], dtype=float)
    priors = GeometricPeakPriors(
        mode="single",
        apex_loc=5.0,
        apex_scale=0.05,
        sigma_loc=0.03,
        sigma_scale=0.005,
        alpha_loc=0.15,
        alpha_scale=0.03,
        main_area_per_trace=trace_areas,
        total_area_per_trace=trace_areas + 2.0,
        artefact_shoulder_area_loc=0.0,
        window_lo=4.8,
        window_hi=5.2,
        n_valid_traces=3,
    )
    assert priors.main_area_per_trace.shape == (3,)
    assert np.all(priors.main_area_per_trace > 0.0)


@pytest.mark.unit
def test_baseline_priors_creation() -> None:
    """BaselinePriors can be created with per-trace intercept/slope."""
    intercept = jnp.array([0.1, 0.12, 0.08], dtype=float)
    slope = jnp.array([0.001, 0.0012, 0.0009], dtype=float)
    intercept_scale = jnp.array([0.05, 0.06, 0.04], dtype=float)
    slope_scale = jnp.array([0.0005, 0.0006, 0.0004], dtype=float)

    bp = BaselinePriors(
        intercept=intercept,
        slope=slope,
        intercept_scale=intercept_scale,
        slope_scale=slope_scale,
    )
    assert bp.intercept.shape == (3,)
    assert bp.slope.shape == (3,)
    assert bp.intercept_scale.shape == (3,)
    assert bp.slope_scale.shape == (3,)


@pytest.mark.unit
def test_baseline_priors_positive_scales() -> None:
    """BaselinePriors scales should be positive."""
    intercept = jnp.array([0.1], dtype=float)
    slope = jnp.array([0.001], dtype=float)
    intercept_scale = jnp.array([0.05], dtype=float)
    slope_scale = jnp.array([0.0005], dtype=float)

    bp = BaselinePriors(
        intercept=intercept,
        slope=slope,
        intercept_scale=intercept_scale,
        slope_scale=slope_scale,
    )
    assert float(bp.intercept_scale[0]) > 0.0
    assert float(bp.slope_scale[0]) > 0.0


@pytest.mark.unit
def test_estimate_baseline_with_simple_data() -> None:
    """estimate_baseline can process simple synthetic data."""
    # Create simple synthetic data: flat signal with low noise
    time = jnp.asarray(
        np.broadcast_to(
            np.linspace(0.0, 6.0, 100, dtype=float)[None, :], (2, 100)
        ).copy()
    )
    signal = jnp.ones((2, 100), dtype=float) * 0.5  # flat baseline

    peaks = [
        PeakAnnotation(molecule_id="mol", rt_min=1.0, rt_max=2.0, mode="single")
    ]
    baselines = [BaselineAnnotation(rt_min=0.0, rt_max=0.5)]

    bp = estimate_baseline(time, signal, peaks=peaks, baselines=baselines)

    assert bp.intercept.shape == (2,)
    assert bp.slope.shape == (2,)
    assert np.all(np.isfinite(bp.intercept))
    assert np.all(np.isfinite(bp.slope))


@pytest.mark.unit
def test_estimate_baseline_shapes() -> None:
    """estimate_baseline returns correct shapes for n_trace."""
    time = jnp.asarray(
        np.broadcast_to(
            np.linspace(0.0, 6.0, 100, dtype=float)[None, :], (3, 100)
        ).copy()
    )
    signal = jnp.ones((3, 100), dtype=float) * 0.5

    peaks = [
        PeakAnnotation(molecule_id="mol", rt_min=1.0, rt_max=2.0, mode="single")
    ]

    bp = estimate_baseline(time, signal, peaks=peaks)

    assert bp.intercept.shape == (3,)
    assert bp.slope.shape == (3,)
    assert bp.intercept_scale.shape == (3,)
    assert bp.slope_scale.shape == (3,)


@pytest.mark.unit
def test_baseline_priors_immutable() -> None:
    """BaselinePriors is frozen (immutable)."""
    bp = BaselinePriors(
        intercept=jnp.array([0.1], dtype=float),
        slope=jnp.array([0.001], dtype=float),
        intercept_scale=jnp.array([0.05], dtype=float),
        slope_scale=jnp.array([0.0005], dtype=float),
    )
    with pytest.raises(AttributeError):
        bp.intercept = jnp.array([0.2], dtype=float)  # type: ignore


@pytest.mark.unit
def test_geometric_peak_priors_immutable() -> None:
    """GeometricPeakPriors is frozen (immutable)."""
    priors = GeometricPeakPriors(
        mode="single",
        apex_loc=5.0,
        apex_scale=0.05,
        sigma_loc=0.03,
        sigma_scale=0.005,
        alpha_loc=0.15,
        alpha_scale=0.03,
        main_area_per_trace=np.array([100.0], dtype=float),
        total_area_per_trace=np.array([102.0], dtype=float),
        artefact_shoulder_area_loc=0.0,
        window_lo=4.8,
        window_hi=5.2,
        n_valid_traces=1,
    )
    with pytest.raises(AttributeError):
        priors.apex_loc = 6.0  # type: ignore


@pytest.mark.unit
def test_geometric_peak_priors_n_components() -> None:
    """GeometricPeakPriors.n_components matches mode."""
    single_priors = GeometricPeakPriors(
        mode="single",
        apex_loc=5.0,
        apex_scale=0.05,
        sigma_loc=0.03,
        sigma_scale=0.005,
        alpha_loc=0.15,
        alpha_scale=0.03,
        main_area_per_trace=np.array([100.0], dtype=float),
        total_area_per_trace=np.array([102.0], dtype=float),
        artefact_shoulder_area_loc=0.0,
        window_lo=4.8,
        window_hi=5.2,
        n_valid_traces=1,
    )
    assert single_priors.n_components == 1
