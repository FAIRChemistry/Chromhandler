"""Unit tests for estimate_baseline per-trace sigma_noise floors."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chromhandler.annotations import PeakAnnotation
from chromhandler.fitting.baseline import estimate_baseline


def _single_peak(rt_min: float = 4.0, rt_max: float = 6.0) -> PeakAnnotation:
    return PeakAnnotation(molecule_id="m", rt_min=rt_min, rt_max=rt_max)


@pytest.mark.unit
def test_estimate_baseline_requires_sigma_noise_kwarg() -> None:
    """estimate_baseline's sigma_noise kwarg is required (no default)."""
    t = np.linspace(0.0, 10.0, 200)
    time = jnp.asarray(np.tile(t, (2, 1)))
    signal = jnp.asarray(np.full((2, 200), 100.0))
    with pytest.raises(TypeError, match="sigma_noise"):
        estimate_baseline(time, signal, peaks=[_single_peak()])  # type: ignore[call-arg]


@pytest.mark.unit
def test_estimate_baseline_floors_use_per_trace_sigma_noise() -> None:
    """With a degenerate flat-signal fit, the floor equals sigma_noise per trace."""
    # Every peak-window edge is identical, so OLS SE collapses and the floor kicks in.
    t = np.linspace(0.0, 10.0, 200)
    time = jnp.asarray(np.tile(t, (2, 1)))
    signal = jnp.asarray(np.full((2, 200), 100.0))
    sigma_noise = jnp.asarray([0.5, 1.5])
    time_span = float(t[-1] - t[0])  # 10.0

    priors = estimate_baseline(
        time, signal, peaks=[_single_peak()], sigma_noise=sigma_noise
    )

    # Intercept floor == sigma_noise (per trace).
    np.testing.assert_allclose(np.asarray(priors.intercept_scale), [0.5, 1.5], atol=1e-6)
    # Slope floor == sigma_noise / time_span.
    np.testing.assert_allclose(
        np.asarray(priors.slope_scale), [0.5 / time_span, 1.5 / time_span], atol=1e-6
    )


@pytest.mark.unit
def test_estimate_baseline_floors_tolerate_zero_time_span() -> None:
    """If a trace's time span is zero, slope floor falls back to sigma_noise."""
    t = np.full(200, 2.0)
    time = jnp.asarray(np.tile(t, (1, 1)))
    signal = jnp.asarray(np.full((1, 200), 100.0))
    sigma_noise = jnp.asarray([0.7])

    priors = estimate_baseline(
        time, signal, peaks=[_single_peak(rt_min=1.5, rt_max=2.5)], sigma_noise=sigma_noise
    )
    np.testing.assert_allclose(np.asarray(priors.slope_scale), [0.7], atol=1e-6)


@pytest.mark.unit
def test_estimate_baseline_scales_above_floor_when_ols_is_informative() -> None:
    """When anchors give an informative OLS fit, scales can exceed the floor."""
    rng = np.random.default_rng(0)
    n = 2000
    t = np.linspace(0.0, 10.0, n)
    signal_np = np.stack(
        [
            100.0 + 2.0 * t + rng.normal(0.0, 0.5, size=n),
            100.0 + 2.0 * t + rng.normal(0.0, 0.5, size=n),
        ]
    )
    time = jnp.asarray(np.tile(t, (2, 1)))
    signal = jnp.asarray(signal_np)
    # Pretend sigma_noise is tiny so floors cannot dominate.
    sigma_noise = jnp.asarray([1e-6, 1e-6])

    priors = estimate_baseline(
        time, signal, peaks=[_single_peak()], sigma_noise=sigma_noise
    )

    # With 2000 anchors of noise=0.5, OLS SE for intercept is well above 1e-6.
    assert float(priors.intercept_scale[0]) > 1e-3
    assert float(priors.slope_scale[0]) > 1e-6
