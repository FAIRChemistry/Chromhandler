"""Tests for FitResult class construction + save."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for tests

from typing import TYPE_CHECKING, Any

import arviz as az
import numpy as np
from scipy.stats import skewnorm

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.fitter import FitResult
from chromhandler.fitting.model import ModelConfig, run_mcmc
from chromhandler.fitting.prepared_dataset import prepare_dataset
from chromhandler.fitting.priors import PriorConfig, build_priors

if TYPE_CHECKING:
    from pathlib import Path


def _result_fixture() -> FitResult:
    rng = np.random.default_rng(0)
    t = np.arange(2.5, 3.6, 0.001)
    times = [t.copy() for _ in range(3)]
    signals: list[np.ndarray[Any, np.dtype[np.float64]]] = [
        amp * np.asarray(skewnorm.pdf(t, 0.0, loc=3.0, scale=0.025)) + 5.0
        + rng.normal(0.0, 0.5, size=t.shape)
        for amp in (100.0, 60.0, 30.0)
    ]
    peaks = [PeakAnnotation(molecule_id="A", rt_min=2.85, rt_max=3.15, mode="single")]
    bases = [
        BaselineAnnotation(rt_min=2.50, rt_max=2.52),
        BaselineAnnotation(rt_min=3.55, rt_max=3.58),
    ]
    ds = prepare_dataset(times, signals, peaks, bases)
    priors = build_priors(ds, config=PriorConfig())
    config = ModelConfig(num_warmup=30, num_samples=30, num_chains=2, seed=0)
    idata = run_mcmc(ds, priors, config)
    return FitResult(idata=idata, dataset=ds, priors=priors, model_config=config)


def test_fitresult_construction() -> None:
    result = _result_fixture()
    assert isinstance(result.idata, az.InferenceData)
    assert result.dataset.n_trace == 3
    assert len(result.priors) == 1
    assert result.model_config.seed == 0


def test_fitresult_save_and_load(tmp_path: Path) -> None:
    result = _result_fixture()
    out_path = tmp_path / "result.nc"
    result.save(out_path)
    assert out_path.exists()
    # Roundtrip through ArviZ
    reloaded = az.from_netcdf(out_path)
    assert hasattr(reloaded, "posterior")
    assert "mu_anchor_left" in reloaded.posterior.data_vars  # type: ignore[attr-defined]


def test_summary_returns_dataframe() -> None:
    result = _result_fixture()
    df = result.summary()
    import pandas as pd
    assert isinstance(df, pd.DataFrame)
    assert "mean" in df.columns
    assert "r_hat" in df.columns
    # Sanity: mu_anchor_left should be in the table
    assert any("mu_anchor_left" in str(idx) for idx in df.index)


def test_diagnostics_returns_dict() -> None:
    result = _result_fixture()
    d = result.diagnostics()
    assert isinstance(d, dict)
    assert "fit_healthy" in d
    assert "r_hat_max" in d


def test_plot_traces_returns_figure() -> None:
    result = _result_fixture()
    fig = result.plot_traces()
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_prior_overlay_returns_figure() -> None:
    result = _result_fixture()
    fig = result.plot_prior_overlay()
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)
