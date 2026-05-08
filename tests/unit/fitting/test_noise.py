"""Tests for chromhandler.fitting.noise."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from chromhandler.annotations import BaselineAnnotation

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TestEstimateNoisePerTrace:
    """MAD-based per-trace noise std from baseline residuals."""

    def test_recovers_known_noise(self) -> None:
        from chromhandler.fitting.noise import estimate_noise_per_trace

        rng = np.random.default_rng(0)
        time = np.linspace(0.0, 5.0, 5001).reshape(1, -1)
        true_sigma = 0.05
        noise_draw: NDArray[np.float64] = rng.standard_normal(time.shape)
        signal: NDArray[np.float64] = np.asarray(
            1.0 + 0.2 * time + true_sigma * noise_draw, dtype=np.float64
        )
        regions = [
            BaselineAnnotation(rt_min=0.5, rt_max=1.0),
            BaselineAnnotation(rt_min=4.0, rt_max=4.5),
        ]
        intercept = np.array([1.0])
        slope = np.array([0.2])
        noise = estimate_noise_per_trace(time, signal, regions, intercept, slope)
        assert noise.shape == (1,)
        np.testing.assert_allclose(noise, [true_sigma], rtol=0.10)

    def test_per_trace_independence(self) -> None:
        from chromhandler.fitting.noise import estimate_noise_per_trace

        rng = np.random.default_rng(1)
        time = np.tile(np.linspace(0.0, 5.0, 5001), (3, 1))
        true_sigmas = np.array([0.01, 0.05, 0.20])
        noise_draw: NDArray[np.float64] = rng.standard_normal(time.shape)
        signal: NDArray[np.float64] = np.asarray(
            true_sigmas[:, None] * noise_draw, dtype=np.float64
        )
        regions = [BaselineAnnotation(rt_min=0.5, rt_max=4.5)]
        intercept = np.zeros(3)
        slope = np.zeros(3)
        noise = estimate_noise_per_trace(time, signal, regions, intercept, slope)
        np.testing.assert_allclose(noise, true_sigmas, rtol=0.10)

    def test_nan_padding_ignored(self) -> None:
        from chromhandler.fitting.noise import estimate_noise_per_trace

        rng = np.random.default_rng(2)
        time = np.full((1, 6000), np.nan)
        signal = np.full((1, 6000), np.nan)
        time[0, :5001] = np.linspace(0.0, 5.0, 5001)
        signal[0, :5001] = 0.05 * rng.standard_normal(5001)
        regions = [BaselineAnnotation(rt_min=0.5, rt_max=4.5)]
        intercept = np.zeros(1)
        slope = np.zeros(1)
        noise = estimate_noise_per_trace(time, signal, regions, intercept, slope)
        np.testing.assert_allclose(noise, [0.05], rtol=0.10)

    def test_robust_to_outliers(self) -> None:
        from chromhandler.fitting.noise import estimate_noise_per_trace

        rng = np.random.default_rng(3)
        time = np.linspace(0.0, 5.0, 5001).reshape(1, -1)
        noise_draw: NDArray[np.float64] = rng.standard_normal(time.shape)
        signal: NDArray[np.float64] = np.asarray(0.05 * noise_draw, dtype=np.float64)
        # Inject 1% extreme outliers
        signal[0, ::100] += 5.0
        regions = [BaselineAnnotation(rt_min=0.5, rt_max=4.5)]
        intercept = np.zeros(1)
        slope = np.zeros(1)
        noise = estimate_noise_per_trace(time, signal, regions, intercept, slope)
        # MAD-based estimate should still be near 0.05 despite outliers
        np.testing.assert_allclose(noise, [0.05], rtol=0.20)

    def test_no_regions_raises(self) -> None:
        from chromhandler.fitting.noise import estimate_noise_per_trace

        time = np.linspace(0.0, 5.0, 501).reshape(1, -1)
        signal = np.zeros_like(time)
        with pytest.raises(ValueError, match="at least one"):
            estimate_noise_per_trace(
                time, signal, [], np.zeros(1), np.zeros(1)
            )
