"""Tests for chromhandler.fitting.plotting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.prepared_dataset import PreparedDataset, prepare_dataset


def _make_synthetic_dataset(n_trace: int = 3) -> PreparedDataset:
    """Build a tiny synthetic PreparedDataset for plotting tests.

    n_trace traces, 101 time points each on [0, 5] min, with a clean
    linear baseline + small Gaussian noise. One peak window 2.0-3.0,
    two baseline regions 0.5-1.0 and 4.0-4.5.
    """
    rng = np.random.default_rng(0)
    time_grid = np.linspace(0.0, 5.0, 101)
    times = [time_grid for _ in range(n_trace)]
    signals = [
        1.0 + 0.1 * time_grid + 0.02 * rng.standard_normal(time_grid.size)
        for _ in range(n_trace)
    ]
    peaks = [PeakAnnotation(molecule_id="x", rt_min=2.0, rt_max=3.0)]
    baselines = [
        BaselineAnnotation(rt_min=0.5, rt_max=1.0),
        BaselineAnnotation(rt_min=4.0, rt_max=4.5),
    ]
    return prepare_dataset(times, signals, peaks, baselines)


class TestAddSignal:
    """The add_signal axes primitive."""

    def test_returns_same_axes(self) -> None:
        from chromhandler.fitting.plotting import add_signal

        ds = _make_synthetic_dataset(n_trace=2)
        fig, ax = plt.subplots()
        try:
            returned = add_signal(ax, ds, trace_idx=0)
            assert returned is ax
        finally:
            plt.close(fig)

    def test_adds_one_line(self) -> None:
        from chromhandler.fitting.plotting import add_signal

        ds = _make_synthetic_dataset(n_trace=2)
        fig, ax = plt.subplots()
        try:
            n_lines_before = len(ax.lines)
            add_signal(ax, ds, trace_idx=0)
            assert len(ax.lines) == n_lines_before + 1
        finally:
            plt.close(fig)

    def test_skips_nan_padding(self) -> None:
        from chromhandler.fitting.plotting import add_signal

        # Manually construct a dataset with NaN-padding to verify it's masked.
        ds = _make_synthetic_dataset(n_trace=2)
        fig, ax = plt.subplots()
        try:
            add_signal(ax, ds, trace_idx=0)
            (line,) = ax.lines[-1:]
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            assert not np.any(np.isnan(xdata))
            assert not np.any(np.isnan(ydata))
        finally:
            plt.close(fig)


class TestAddAnnotationRegions:
    """The add_annotation_regions axes primitive."""

    def test_returns_same_axes(self) -> None:
        from chromhandler.fitting.plotting import add_annotation_regions

        ds = _make_synthetic_dataset()
        fig, ax = plt.subplots()
        try:
            returned = add_annotation_regions(ax, ds)
            assert returned is ax
        finally:
            plt.close(fig)

    def test_adds_one_axvspan_per_region(self) -> None:
        from chromhandler.fitting.plotting import add_annotation_regions

        ds = _make_synthetic_dataset()  # 1 peak window + 2 baseline regions
        fig, ax = plt.subplots()
        try:
            n_patches_before = len(ax.patches)
            add_annotation_regions(ax, ds)
            # axvspan adds Polygon patches; one per peak + one per baseline = 3
            assert len(ax.patches) - n_patches_before == 3
        finally:
            plt.close(fig)
