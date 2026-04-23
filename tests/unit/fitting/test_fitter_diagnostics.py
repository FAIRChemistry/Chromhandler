"""Unit tests for Fitter diagnostic output methods.

Tests save_summary(), plot_traces(), plot_fit_peaks(), and plot_fit_combined() without MCMC.
Fake ArviZ InferenceData built via az.from_dict() — no mocks.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import arviz as az
import matplotlib
import numpy as np
import numpy.typing as npt
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting import Fitter

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gaussian(
    x: npt.NDArray[np.float64], apex: float, sigma: float, area: float
) -> npt.NDArray[np.float64]:
    return area / (sigma * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((x - apex) / sigma) ** 2)  # type: ignore[return-value]


def _make_fitter(n_trace: int = 3, n_time: int = 150) -> Fitter:
    x = np.linspace(2.5, 3.5, n_time)
    rng = np.random.default_rng(7)
    signal = np.stack([
        _gaussian(x, 3.0, 0.05, 120.0) + rng.normal(0, 0.3, n_time)
        for _ in range(n_trace)
    ])
    time = np.tile(x, (n_trace, 1))
    fitter = Fitter(time, signal)
    assert fitter.trace_sigma_noise.shape == (fitter.n_traces,)
    assert np.all(fitter.trace_sigma_noise > 0.0)
    fitter.add_baseline_annotation(BaselineAnnotation(rt_min=2.5, rt_max=2.62))
    fitter.add_baseline_annotation(BaselineAnnotation(rt_min=3.38, rt_max=3.5))
    fitter.add_peak_annotation(PeakAnnotation(molecule_id="A", rt_min=2.7, rt_max=3.3, mode="single"))
    return fitter


def _inject_posterior(fitter: Fitter) -> None:
    """Inject a minimal real ArviZ InferenceData (no MCMC needed)."""
    rng = np.random.default_rng(0)
    n_chains, n_draws = 2, 50
    fitter._posterior = az.from_dict(
        posterior={
            "apex": rng.normal(size=(n_chains, n_draws, 1)),
            "sigma_y": np.abs(rng.normal(size=(n_chains, n_draws, 1))),
        }
    )


# ---------------------------------------------------------------------------
# save_summary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_save_summary_raises_before_fit() -> None:
    fitter = _make_fitter()
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        fitter.save_summary("/tmp/summary.txt")


@pytest.mark.unit
def test_save_summary_creates_file(tmp_path: Path) -> None:
    fitter = _make_fitter()
    _inject_posterior(fitter)
    out = tmp_path / "summary.txt"
    fitter.save_summary(out)
    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.unit
def test_save_summary_content_has_mean_column(tmp_path: Path) -> None:
    fitter = _make_fitter()
    _inject_posterior(fitter)
    out = tmp_path / "summary.txt"
    fitter.save_summary(out)
    content = out.read_text()
    assert "mean" in content


# ---------------------------------------------------------------------------
# plot_traces
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plot_traces_raises_before_fit() -> None:
    fitter = _make_fitter()
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        fitter.plot_traces()


@pytest.mark.unit
def test_plot_traces_returns_figure() -> None:
    fitter = _make_fitter()
    _inject_posterior(fitter)
    fig = fitter.plot_traces()
    assert isinstance(fig, plt.Figure)  # type: ignore[reportPrivateImportUsage]
    plt.close(fig)


@pytest.mark.unit
def test_plot_traces_saves_png(tmp_path: Path) -> None:
    fitter = _make_fitter()
    _inject_posterior(fitter)
    out = tmp_path / "traces.png"
    fig = fitter.plot_traces(path=out)
    assert out.exists()
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_fit_peaks / plot_fit_combined (Fitter methods)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plot_fit_peaks_scatter_only_before_fit() -> None:
    """plot_fit_peaks works without a posterior — scatter-only mode."""
    fitter = _make_fitter()
    fig, axes = fitter.plot_fit_peaks()
    assert isinstance(fig, plt.Figure)  # type: ignore[reportPrivateImportUsage]
    assert axes.shape == (fitter.n_traces, 1)
    plt.close(fig)


@pytest.mark.unit
def test_plot_fit_combined_scatter_only_before_fit() -> None:
    """plot_fit_combined works without a posterior — scatter-only mode."""
    fitter = _make_fitter()
    fig, axes = fitter.plot_fit_combined()
    assert isinstance(fig, plt.Figure)  # type: ignore[reportPrivateImportUsage]
    assert axes.shape == (fitter.n_traces, 1)
    plt.close(fig)


@pytest.mark.unit
def test_plot_fit_peaks_saves_png(tmp_path: Path) -> None:
    fitter = _make_fitter()
    out = tmp_path / "fit_peaks.png"
    fig, _ = fitter.plot_fit_peaks(path=out)
    assert out.exists()
    plt.close(fig)


@pytest.mark.unit
def test_plot_fit_combined_saves_png(tmp_path: Path) -> None:
    fitter = _make_fitter()
    out = tmp_path / "fit_combined.png"
    fig, _ = fitter.plot_fit_combined(path=out)
    assert out.exists()
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_geometric_diagnostic
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plot_geometric_diagnostic_runs_without_posterior() -> None:
    """plot_geometric_diagnostic is a pre-fit tool — no posterior required."""
    fitter = _make_fitter()
    fig, axes, outliers = fitter.plot_geometric_diagnostic()
    assert isinstance(fig, plt.Figure)  # type: ignore[reportPrivateImportUsage]
    assert axes.shape == (1, len(fitter.peaks))
    assert isinstance(outliers, list)
    plt.close(fig)


@pytest.mark.unit
def test_plot_geometric_diagnostic_saves_png(tmp_path: Path) -> None:
    fitter = _make_fitter()
    out = tmp_path / "geom.png"
    fig, _, _ = fitter.plot_geometric_diagnostic(path=out)
    assert out.exists()
    plt.close(fig)


@pytest.mark.unit
def test_plot_geometric_diagnostic_flags_synthetic_outlier() -> None:
    """Injecting a 2x-broader peak in one trace flags that trace as an outlier."""
    from chromhandler.fitting import Fitter

    n_trace, n_time = 6, 300
    x = np.linspace(2.5, 3.5, n_time)
    rng = np.random.default_rng(1)
    signal = np.stack([
        _gaussian(x, 3.0, 0.05, 120.0) + rng.normal(0, 0.1, n_time)
        for _ in range(n_trace - 1)
    ])
    wide = (_gaussian(x, 3.0, 0.12, 120.0) + rng.normal(0, 0.1, n_time))[None, :]
    signal = np.concatenate([signal, wide], axis=0)
    time = np.tile(x, (n_trace, 1))

    fitter = Fitter(time, signal)
    assert fitter.trace_sigma_noise.shape == (fitter.n_traces,)
    assert np.all(fitter.trace_sigma_noise > 0.0)
    fitter.add_baseline_annotation(BaselineAnnotation(rt_min=2.5, rt_max=2.62))
    fitter.add_baseline_annotation(BaselineAnnotation(rt_min=3.38, rt_max=3.5))
    fitter.add_peak_annotation(PeakAnnotation(molecule_id="A", rt_min=2.7, rt_max=3.3, mode="single"))

    fig, _, outliers = fitter.plot_geometric_diagnostic(k_mad=2.0)
    assert n_trace - 1 in outliers
    plt.close(fig)
