"""Tests for chromhandler.fitting.prepared_dataset."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation


class TestPreparedDatasetConstruction:
    """The PreparedDataset frozen dataclass."""

    def test_fields_present(self) -> None:
        from chromhandler.fitting.prepared_dataset import PreparedDataset

        time = np.zeros((2, 10))
        signal = np.zeros((2, 10))
        ds = PreparedDataset(
            time=time,
            signal=signal,
            valid_mask=np.ones((2, 10), dtype=bool),
            dt_per_trace=np.full(2, 0.1),
            dt_global=0.1,
            n_trace=2,
            peak_annotations=[],
            baseline_annotations=[],
            baseline_intercept=np.zeros(2),
            baseline_slope=np.zeros(2),
            noise_per_trace=np.full(2, 0.01),
            trace_ids=("trace_0", "trace_1"),
        )
        assert ds.n_trace == 2
        assert ds.dt_global == 0.1
        np.testing.assert_array_equal(ds.time, time)

    def test_is_frozen(self) -> None:
        from chromhandler.fitting.prepared_dataset import PreparedDataset

        ds = PreparedDataset(
            time=np.zeros((1, 5)),
            signal=np.zeros((1, 5)),
            valid_mask=np.ones((1, 5), dtype=bool),
            dt_per_trace=np.full(1, 0.1),
            dt_global=0.1,
            n_trace=1,
            peak_annotations=[],
            baseline_annotations=[],
            baseline_intercept=np.zeros(1),
            baseline_slope=np.zeros(1),
            noise_per_trace=np.full(1, 0.01),
            trace_ids=("trace_0",),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            ds.n_trace = 99  # type: ignore[misc]


class TestPrepareDataset:
    """End-to-end orchestrator."""

    def test_simple_pipeline_runs(self) -> None:
        from chromhandler.fitting.prepared_dataset import prepare_dataset

        rng = np.random.default_rng(0)
        time_grid = np.linspace(0.0, 5.0, 501)
        true_sigma = 0.02
        baseline = 1.0 + 0.1 * time_grid
        signals = [
            baseline + true_sigma * rng.standard_normal(time_grid.size)
            for _ in range(3)
        ]
        times = [time_grid for _ in range(3)]
        peaks = [PeakAnnotation(molecule_id="x", rt_min=2.0, rt_max=3.0)]
        baselines = [
            BaselineAnnotation(rt_min=0.5, rt_max=1.5),
            BaselineAnnotation(rt_min=3.5, rt_max=4.5),
        ]

        ds = prepare_dataset(times, signals, peaks, baselines)

        assert ds.n_trace == 3
        # Pruning must drop columns outside (peak or baseline) windows.
        # Windows cover [0.5,1.5] + [2.0,3.0] + [3.5,4.5] = 3.0 of 5.0 total
        # range at dt=0.01 -> ~303 columns (incl. endpoints), << padded 501.
        assert ds.time.shape[1] < 501
        assert ds.time.shape == (3, 303)
        assert ds.signal.shape == (3, 303)
        assert ds.valid_mask.shape == (3, 303)
        assert ds.valid_mask.all()  # every kept column has valid data
        np.testing.assert_allclose(ds.dt_global, 0.01, rtol=1e-3)
        np.testing.assert_allclose(ds.baseline_intercept, [1.0] * 3, atol=0.05)
        np.testing.assert_allclose(ds.baseline_slope, [0.1] * 3, atol=0.05)
        np.testing.assert_allclose(ds.noise_per_trace, [true_sigma] * 3, rtol=0.20)
        assert ds.peak_annotations == peaks
        assert ds.baseline_annotations == baselines

    def test_baseline_in_peak_window_raises(self) -> None:
        from chromhandler.fitting.prepared_dataset import prepare_dataset

        time_grid = np.linspace(0.0, 5.0, 501)
        signals = [np.ones_like(time_grid)]
        times = [time_grid]
        peaks = [PeakAnnotation(molecule_id="x", rt_min=2.0, rt_max=3.0)]
        baselines = [BaselineAnnotation(rt_min=2.5, rt_max=3.5)]

        with pytest.raises(ValueError, match="overlaps peak"):
            prepare_dataset(times, signals, peaks, baselines)

    def test_variable_length_traces_padded(self) -> None:
        from chromhandler.fitting.prepared_dataset import prepare_dataset

        rng = np.random.default_rng(0)
        long_t = np.linspace(0.0, 5.0, 501)
        short_t = np.linspace(0.0, 4.0, 401)
        long_s = 1.0 + 0.01 * rng.standard_normal(long_t.size)
        short_s = 1.0 + 0.01 * rng.standard_normal(short_t.size)
        times = [long_t, short_t]
        signals = [long_s, short_s]
        peaks = [PeakAnnotation(molecule_id="x", rt_min=2.0, rt_max=3.0)]
        baselines = [BaselineAnnotation(rt_min=0.5, rt_max=1.5)]

        ds = prepare_dataset(times, signals, peaks, baselines)

        # Two windows ([0.5,1.5] + [2.0,3.0]) cover 2.0 of the 5.0 padded range.
        assert ds.time.shape[1] < 501
        assert ds.time.shape == (2, 202)
        assert ds.signal.shape == (2, 202)
        # Both windows lie within trace 1's range (≤4.0), so no padded NaN
        # survives after pruning.
        assert not np.isnan(ds.signal).any()
        assert ds.valid_mask.all()


def _make_inputs_with_n_traces(n_trace: int = 3):
    t = np.arange(2.5, 3.6, 0.01)
    times = [t.copy() for _ in range(n_trace)]
    signals = [
        np.full_like(t, 100.0) + 10.0 * np.exp(-((t - 2.8) ** 2) / 0.02)
        for _ in range(n_trace)
    ]
    peak_anns = [PeakAnnotation(molecule_id="A", rt_min=2.6, rt_max=3.0)]
    base_anns = [
        BaselineAnnotation(rt_min=2.55, rt_max=2.58),
        BaselineAnnotation(rt_min=3.50, rt_max=3.55),
    ]
    return times, signals, peak_anns, base_anns


def test_prepared_dataset_trace_ids_default() -> None:
    from chromhandler.fitting.prepared_dataset import prepare_dataset

    times, signals, peak_anns, base_anns = _make_inputs_with_n_traces(3)
    ds = prepare_dataset(times, signals, peak_anns, base_anns)
    assert ds.trace_ids == ("trace_0", "trace_1", "trace_2")


def test_prepared_dataset_trace_ids_propagates() -> None:
    from chromhandler.fitting.prepared_dataset import prepare_dataset

    times, signals, peak_anns, base_anns = _make_inputs_with_n_traces(3)
    ds = prepare_dataset(
        times, signals, peak_anns, base_anns,
        trace_ids=["s1/c1", "s1/c2", "s2/c1"],
    )
    assert ds.trace_ids == ("s1/c1", "s1/c2", "s2/c1")


def test_prepared_dataset_trace_ids_length_mismatch_raises() -> None:
    from chromhandler.fitting.prepared_dataset import prepare_dataset

    times, signals, peak_anns, base_anns = _make_inputs_with_n_traces(3)
    with pytest.raises(ValueError, match="trace_ids"):
        prepare_dataset(
            times, signals, peak_anns, base_anns,
            trace_ids=["only_one"],  # 1 entry, but 3 traces
        )


def test_prepare_dataset_raises_on_undefined_sampling_interval() -> None:
    """All traces with a single sample -> no diff -> dt_global is NaN.
    Must raise a clear error about the sampling interval rather than
    propagate NaN into the model's warp scale (review item 8)."""
    from chromhandler.fitting.prepared_dataset import prepare_dataset

    times = [np.array([1.0]), np.array([1.0])]
    signals = [np.array([5.0]), np.array([6.0])]
    peak_anns = [PeakAnnotation(molecule_id="x", rt_min=0.9, rt_max=1.1, mode="single")]
    base_anns = [BaselineAnnotation(rt_min=0.0, rt_max=0.5)]
    with pytest.raises(ValueError, match=r"dt_global|sampling interval|finite time"):
        prepare_dataset(times, signals, peak_anns, base_anns)
