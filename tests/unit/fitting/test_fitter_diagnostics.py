"""Unit tests for Fitter diagnostic output methods.

Tests save_summary(), plot_traces(), and plot_fit() without MCMC.
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
# plot_fit (Fitter method)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plot_fit_scatter_only_before_fit() -> None:
    """plot_fit works without a posterior — scatter-only mode."""
    fitter = _make_fitter()
    fig, axes = fitter.plot_fit()
    assert isinstance(fig, plt.Figure)  # type: ignore[reportPrivateImportUsage]
    # 1 peak, no combined column → axes shape [n_traces, 1]
    assert axes.shape == (fitter.n_traces, 1)
    plt.close(fig)


@pytest.mark.unit
def test_plot_fit_saves_png(tmp_path: Path) -> None:
    fitter = _make_fitter()
    out = tmp_path / "fit.png"
    fig, _ = fitter.plot_fit(path=out)
    assert out.exists()
    plt.close(fig)
