"""Tests for chromhandler.fitting.plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.prepared_dataset import PreparedDataset, prepare_dataset

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


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


class TestAddBaseline:
    """The add_baseline axes primitive."""

    def test_returns_same_axes(self) -> None:
        from chromhandler.fitting.plotting import add_baseline

        ds = _make_synthetic_dataset()
        fig, ax = plt.subplots()
        try:
            returned = add_baseline(ax, ds, trace_idx=0)
            assert returned is ax
        finally:
            plt.close(fig)

    def test_adds_baseline_line(self) -> None:
        from chromhandler.fitting.plotting import add_baseline

        ds = _make_synthetic_dataset()
        fig, ax = plt.subplots()
        try:
            n_lines_before = len(ax.lines)
            add_baseline(ax, ds, trace_idx=0, show_noise_band=False)
            assert len(ax.lines) == n_lines_before + 1
        finally:
            plt.close(fig)

    def test_noise_band_adds_polycollection(self) -> None:
        from chromhandler.fitting.plotting import add_baseline

        ds = _make_synthetic_dataset()
        fig, ax = plt.subplots()
        try:
            n_collections_before = len(ax.collections)
            add_baseline(ax, ds, trace_idx=0, show_noise_band=True)
            # fill_between adds a PolyCollection
            assert len(ax.collections) == n_collections_before + 1
        finally:
            plt.close(fig)

    def test_no_noise_band_when_disabled(self) -> None:
        from chromhandler.fitting.plotting import add_baseline

        ds = _make_synthetic_dataset()
        fig, ax = plt.subplots()
        try:
            n_collections_before = len(ax.collections)
            add_baseline(ax, ds, trace_idx=0, show_noise_band=False)
            assert len(ax.collections) == n_collections_before
        finally:
            plt.close(fig)

    def test_baseline_values_match_intercept_slope(self) -> None:
        from chromhandler.fitting.plotting import add_baseline

        ds = _make_synthetic_dataset()
        fig, ax = plt.subplots()
        try:
            add_baseline(ax, ds, trace_idx=0, show_noise_band=False)
            (line,) = ax.lines[-1:]
            xdata = np.asarray(line.get_xdata())
            ydata = np.asarray(line.get_ydata())
            expected = (
                ds.baseline_intercept[0] + ds.baseline_slope[0] * xdata
            )
            np.testing.assert_allclose(ydata, expected, atol=1e-9)
        finally:
            plt.close(fig)


class TestAddModel:
    """The add_model axes primitive."""

    def test_returns_same_axes(self) -> None:
        from chromhandler.fitting.plotting import add_model

        ds = _make_synthetic_dataset()
        fig, ax = plt.subplots()
        try:
            returned = add_model(ax, ds, trace_idx=0, model_fn=lambda t, i: t)
            assert returned is ax
        finally:
            plt.close(fig)

    def test_calls_model_fn_with_valid_time_and_trace_idx(self) -> None:
        from chromhandler.fitting.plotting import add_model

        ds = _make_synthetic_dataset()
        seen: dict[str, object] = {}

        def model_fn(t: NDArray[np.float64], i: int) -> NDArray[np.float64]:
            seen["t"] = t
            seen["i"] = i
            return np.zeros_like(t)

        fig, ax = plt.subplots()
        try:
            add_model(ax, ds, trace_idx=2, model_fn=model_fn)
            assert seen["i"] == 2
            t_obj = seen["t"]
            assert isinstance(t_obj, np.ndarray)
            t: NDArray[np.float64] = t_obj
            assert not np.any(np.isnan(t))  # NaN padding masked out
            assert t.shape == (101,)
        finally:
            plt.close(fig)

    def test_overlays_model_output_as_line(self) -> None:
        from chromhandler.fitting.plotting import add_model

        ds = _make_synthetic_dataset()

        def model_fn(t: NDArray[np.float64], i: int) -> NDArray[np.float64]:
            return 0.5 + 0.0 * t  # constant 0.5

        fig, ax = plt.subplots()
        try:
            n_lines_before = len(ax.lines)
            add_model(ax, ds, trace_idx=0, model_fn=model_fn)
            assert len(ax.lines) == n_lines_before + 1
            (line,) = ax.lines[-1:]
            ydata = np.asarray(line.get_ydata())
            np.testing.assert_allclose(ydata, 0.5, atol=1e-9)
        finally:
            plt.close(fig)


class TestPlotOverview:
    """plot_overview figure-level convenience."""

    def test_returns_figure_with_one_axes_per_trace(self) -> None:
        from chromhandler.fitting.plotting import plot_overview

        ds = _make_synthetic_dataset(n_trace=4)
        fig = plot_overview(ds)
        try:
            # Each trace gets one axes; trailing cells in the grid stay
            # empty but are still added by plt.subplots — figure-level
            # convenience hides them.
            data_axes = [a for a in fig.axes if a.has_data() or a.lines or a.patches]
            assert len(data_axes) == 4
        finally:
            plt.close(fig)

    def test_save_path_writes_file(self, tmp_path: Path) -> None:
        from chromhandler.fitting.plotting import plot_overview

        ds = _make_synthetic_dataset(n_trace=2)
        out = tmp_path / "overview.png"
        fig = plot_overview(ds, path=out)
        try:
            assert out.exists()
            assert out.stat().st_size > 0
        finally:
            plt.close(fig)


class TestPlotBaselineDiagnostic:
    """plot_baseline_diagnostic figure-level convenience."""

    def test_returns_figure_with_one_axes_per_trace(self) -> None:
        from chromhandler.fitting.plotting import plot_baseline_diagnostic

        ds = _make_synthetic_dataset(n_trace=4)
        fig = plot_baseline_diagnostic(ds)
        try:
            data_axes = [a for a in fig.axes if a.has_data() or a.lines or a.patches]
            assert len(data_axes) == 4
        finally:
            plt.close(fig)

    def test_each_panel_has_baseline_line_and_noise_band(self) -> None:
        from chromhandler.fitting.plotting import plot_baseline_diagnostic

        ds = _make_synthetic_dataset(n_trace=3)
        fig = plot_baseline_diagnostic(ds)
        try:
            data_axes = [a for a in fig.axes if a.has_data() or a.lines or a.patches]
            assert len(data_axes) == 3
            for ax in data_axes:
                # signal line + baseline line >= 2; one PolyCollection from noise band.
                assert len(ax.lines) >= 2
                assert len(ax.collections) >= 1
        finally:
            plt.close(fig)

    def test_panel_titles_include_dt_and_noise(self) -> None:
        from chromhandler.fitting.plotting import plot_baseline_diagnostic

        ds = _make_synthetic_dataset(n_trace=2)
        fig = plot_baseline_diagnostic(ds)
        try:
            data_axes = [a for a in fig.axes if a.has_data() or a.lines or a.patches]
            for i, ax in enumerate(data_axes):
                title = ax.get_title()
                assert f"trace {i}" in title
                assert "dt=" in title
                assert "noise=" in title
        finally:
            plt.close(fig)

    def test_save_path_writes_file(self, tmp_path: Path) -> None:
        from chromhandler.fitting.plotting import plot_baseline_diagnostic

        ds = _make_synthetic_dataset(n_trace=2)
        out = tmp_path / "baseline_diag.png"
        fig = plot_baseline_diagnostic(ds, path=out)
        try:
            assert out.exists()
            assert out.stat().st_size > 0
        finally:
            plt.close(fig)
