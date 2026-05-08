"""Tests for chromhandler.fitting.preprocessing."""

from __future__ import annotations

import numpy as np
import pytest


class TestPadToCommonAxis:
    """Padding variable-length traces to a rectangular array."""

    def test_equal_lengths_no_padding(self) -> None:
        from chromhandler.fitting.preprocessing import pad_to_common_axis

        times = [np.array([0.0, 0.1, 0.2]), np.array([0.0, 0.1, 0.2])]
        signals = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
        t, s = pad_to_common_axis(times, signals)
        assert t.shape == (2, 3)
        assert s.shape == (2, 3)
        np.testing.assert_array_equal(s[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(s[1], [4.0, 5.0, 6.0])

    def test_short_trace_padded_with_nan(self) -> None:
        from chromhandler.fitting.preprocessing import pad_to_common_axis

        times = [np.array([0.0, 0.1, 0.2, 0.3]), np.array([0.0, 0.1])]
        signals = [np.array([1.0, 2.0, 3.0, 4.0]), np.array([5.0, 6.0])]
        t, s = pad_to_common_axis(times, signals)
        assert t.shape == (2, 4)
        assert s.shape == (2, 4)
        assert np.isnan(t[1, 2:]).all()
        assert np.isnan(s[1, 2:]).all()
        np.testing.assert_array_equal(s[1, :2], [5.0, 6.0])

    def test_mismatched_time_signal_lengths_raises(self) -> None:
        from chromhandler.fitting.preprocessing import pad_to_common_axis

        times = [np.array([0.0, 0.1])]
        signals = [np.array([1.0, 2.0, 3.0])]
        with pytest.raises(ValueError, match="length"):
            pad_to_common_axis(times, signals)

    def test_unequal_outer_lengths_raises(self) -> None:
        from chromhandler.fitting.preprocessing import pad_to_common_axis

        times = [np.array([0.0, 0.1])]
        signals: list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]] = []
        with pytest.raises(ValueError, match="same number"):
            pad_to_common_axis(times, signals)


class TestComputeDtPerTrace:
    """Median sampling interval per trace."""

    def test_uniform_grid(self) -> None:
        from chromhandler.fitting.preprocessing import compute_dt_per_trace

        time = np.array([[0.0, 0.1, 0.2, 0.3], [0.0, 0.1, 0.2, 0.3]])
        dt = compute_dt_per_trace(time)
        np.testing.assert_allclose(dt, [0.1, 0.1])

    def test_nan_padding_ignored(self) -> None:
        from chromhandler.fitting.preprocessing import compute_dt_per_trace

        time = np.array([[0.0, 0.1, 0.2, 0.3], [0.0, 0.1, np.nan, np.nan]])
        dt = compute_dt_per_trace(time)
        np.testing.assert_allclose(dt, [0.1, 0.1])

    def test_irregular_grid_uses_median(self) -> None:
        from chromhandler.fitting.preprocessing import compute_dt_per_trace

        time = np.array([[0.0, 0.1, 0.2, 0.3, 0.5]])
        dt = compute_dt_per_trace(time)
        np.testing.assert_allclose(dt, [0.1])  # median of [0.1, 0.1, 0.1, 0.2]


class TestComputeGlobalDt:
    """Global dt = median of per-trace dt values."""

    def test_simple(self) -> None:
        from chromhandler.fitting.preprocessing import compute_global_dt

        assert compute_global_dt(np.array([0.1, 0.1, 0.1])) == 0.1

    def test_uses_median(self) -> None:
        from chromhandler.fitting.preprocessing import compute_global_dt

        assert compute_global_dt(np.array([0.1, 0.1, 0.5])) == 0.1

    def test_ignores_nan_from_single_sample_traces(self) -> None:
        from chromhandler.fitting.preprocessing import compute_global_dt

        # A trace with only one sample contributes NaN to dt_per_trace
        # (no diff defined); the global value should still come from
        # the well-defined traces.
        assert compute_global_dt(np.array([np.nan, 0.1, 0.1])) == 0.1
