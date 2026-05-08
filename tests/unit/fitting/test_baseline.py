"""Tests for chromhandler.fitting.baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from chromhandler.annotations import BaselineAnnotation

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TestEstimateBaselines:
    """Per-trace OLS baseline estimation from user-annotated regions."""

    def test_constant_baseline(self) -> None:
        from chromhandler.fitting.baseline import estimate_baselines

        time = np.linspace(0.0, 5.0, 501).reshape(1, -1)
        signal = np.full_like(time, 7.5)
        regions = [BaselineAnnotation(rt_min=0.5, rt_max=1.0)]
        intercept, slope = estimate_baselines(time, signal, regions)
        np.testing.assert_allclose(intercept, [7.5], atol=1e-9)
        np.testing.assert_allclose(slope, [0.0], atol=1e-9)

    def test_linear_baseline_recovered(self) -> None:
        from chromhandler.fitting.baseline import estimate_baselines

        time = np.linspace(0.0, 5.0, 501).reshape(1, -1)
        true_intercept, true_slope = 2.0, 0.3
        signal = true_intercept + true_slope * time
        regions = [
            BaselineAnnotation(rt_min=0.5, rt_max=1.0),
            BaselineAnnotation(rt_min=4.0, rt_max=4.5),
        ]
        intercept, slope = estimate_baselines(time, signal, regions)
        np.testing.assert_allclose(intercept, [true_intercept], atol=1e-6)
        np.testing.assert_allclose(slope, [true_slope], atol=1e-6)

    def test_per_trace_independence(self) -> None:
        from chromhandler.fitting.baseline import estimate_baselines

        rng = np.random.default_rng(0)
        time = np.tile(np.linspace(0.0, 5.0, 501), (3, 1))
        true_intercepts = np.array([1.0, 2.0, 3.0])
        true_slopes = np.array([0.0, 0.1, -0.2])
        noise: NDArray[np.float64] = rng.standard_normal(time.shape)
        signal_raw = true_intercepts[:, None] + true_slopes[:, None] * time + 0.01 * noise
        signal: NDArray[np.float64] = np.asarray(signal_raw, dtype=np.float64)
        regions = [
            BaselineAnnotation(rt_min=0.5, rt_max=1.0),
            BaselineAnnotation(rt_min=4.0, rt_max=4.5),
        ]
        intercept, slope = estimate_baselines(time, signal, regions)
        np.testing.assert_allclose(intercept, true_intercepts, atol=0.05)
        np.testing.assert_allclose(slope, true_slopes, atol=0.02)

    def test_nan_padded_trace_handled(self) -> None:
        from chromhandler.fitting.baseline import estimate_baselines

        time = np.full((1, 600), np.nan)
        signal = np.full((1, 600), np.nan)
        time[0, :501] = np.linspace(0.0, 5.0, 501)
        signal[0, :501] = 1.5  # constant baseline
        regions = [BaselineAnnotation(rt_min=0.5, rt_max=1.0)]
        intercept, slope = estimate_baselines(time, signal, regions)
        np.testing.assert_allclose(intercept, [1.5], atol=1e-9)
        np.testing.assert_allclose(slope, [0.0], atol=1e-9)

    def test_too_few_baseline_points_raises(self) -> None:
        from chromhandler.fitting.baseline import estimate_baselines

        time = np.linspace(0.0, 5.0, 11).reshape(1, -1)
        signal = np.zeros_like(time)
        regions = [BaselineAnnotation(rt_min=0.0, rt_max=0.05)]
        with pytest.raises(ValueError, match="too few"):
            estimate_baselines(time, signal, regions)

    def test_no_regions_raises(self) -> None:
        from chromhandler.fitting.baseline import estimate_baselines

        time = np.linspace(0.0, 5.0, 501).reshape(1, -1)
        signal = np.zeros_like(time)
        with pytest.raises(ValueError, match="at least one"):
            estimate_baselines(time, signal, [])
