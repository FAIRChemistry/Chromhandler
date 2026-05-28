"""Tests for FitResult class construction + save."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for tests

from typing import TYPE_CHECKING, Any

import arviz as az
import numpy as np
import pytest
from scipy.stats import skewnorm

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.fitter import FitResult
from chromhandler.fitting.model import ModelConfig, run_mcmc
from chromhandler.fitting.prepared_dataset import prepare_dataset
from chromhandler.fitting.priors import PriorConfig, build_priors

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def fit_result() -> FitResult:
    """Module-scoped fitted result; MCMC runs once and is reused across tests.

    Realistic SNR with per-trace mu/sigma jitter: synthetic peaks too clean
    collapse the posterior so tightly that ArviZ's per-chain KDE in
    plot_traces overflows (the bandwidth division explodes when each
    chain's range is far below the implicit Scott's-rule bandwidth).
    noise_std=5 on peak amplitudes (100, 60, 30) gives SNR ~20-6, in
    range of real chromatography.

    Note: tests that mutate ``result.idata`` (plot_fit / plot_prior_predictive
    add posterior_predictive / prior_predictive groups lazily) share state.
    Those tests assert on the final-state result, not on the absence of the
    lazy-cache groups beforehand.
    """
    rng = np.random.default_rng(0)
    t = np.arange(2.5, 3.6, 0.001)
    times = [t.copy() for _ in range(3)]
    mu_per_trace = [3.000, 3.005, 2.998]
    sigma_per_trace = [0.025, 0.027, 0.024]
    signals: list[np.ndarray[Any, np.dtype[np.float64]]] = [
        amp * np.asarray(skewnorm.pdf(t, 0.0, loc=mu, scale=sg)) + 5.0
        + rng.normal(0.0, 5.0, size=t.shape)
        for amp, mu, sg in zip(
            (100.0, 60.0, 30.0), mu_per_trace, sigma_per_trace, strict=True,
        )
    ]
    peaks = [PeakAnnotation(molecule_id="A", rt_min=2.85, rt_max=3.15, mode="single")]
    bases = [
        BaselineAnnotation(rt_min=2.50, rt_max=2.52),
        BaselineAnnotation(rt_min=3.55, rt_max=3.58),
    ]
    ds = prepare_dataset(times, signals, peaks, bases)
    priors = build_priors(ds, config=PriorConfig())
    config = ModelConfig(num_warmup=200, num_samples=200, num_chains=2, seed=1)
    idata = run_mcmc(ds, priors, config)
    return FitResult(idata=idata, dataset=ds, priors=priors, model_config=config)


def test_fitresult_construction(fit_result: FitResult) -> None:
    assert isinstance(fit_result.idata, az.InferenceData)
    assert fit_result.dataset.n_trace == 3
    assert len(fit_result.priors) == 1
    assert fit_result.model_config.seed == 1


def test_fitresult_save_and_load(fit_result: FitResult, tmp_path: Path) -> None:
    out_path = tmp_path / "result.nc"
    fit_result.save(out_path)
    assert out_path.exists()
    # Roundtrip through ArviZ
    reloaded = az.from_netcdf(out_path)
    assert hasattr(reloaded, "posterior")
    assert "mu" in reloaded.posterior.data_vars  # type: ignore[attr-defined]


def test_summary_returns_dataframe(fit_result: FitResult) -> None:
    df = fit_result.summary()
    import pandas as pd
    assert isinstance(df, pd.DataFrame)
    assert "mean" in df.columns
    assert "r_hat" in df.columns
    # Sanity: mu should be in the table
    assert any("mu" in str(idx) for idx in df.index)


def test_diagnostics_returns_dict(fit_result: FitResult) -> None:
    d = fit_result.diagnostics()
    assert isinstance(d, dict)
    assert "fit_healthy" in d
    assert "r_hat_max" in d


def test_plot_traces_returns_figure(fit_result: FitResult) -> None:
    fig = fit_result.plot_traces()
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_prior_overlay_returns_figure(fit_result: FitResult) -> None:
    fig = fit_result.plot_prior_overlay()
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_fit_returns_figure_and_caches(fit_result: FitResult) -> None:
    fig = fit_result.plot_fit()
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)
    # plot_fit ensures posterior_predictive is present (lazy cache; harmless
    # if a prior test already populated it on the shared module fixture).
    assert hasattr(fit_result.idata, "posterior_predictive")
    # Layout must be n_trace x n_peak after the per-(trace, peak) grid
    # refactor. The fixture has 3 traces x 1 peak.
    assert len(fig.axes) == 3 * 1


def test_plot_prior_predictive_returns_figure_and_caches(fit_result: FitResult) -> None:
    fig = fit_result.plot_prior_predictive()
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)
    assert hasattr(fit_result.idata, "prior_predictive")
    # Layout must be n_trace x n_peak after the per-(trace, peak) grid
    # refactor. The fixture has 3 traces x 1 peak.
    assert len(fig.axes) == 3 * 1
