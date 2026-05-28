"""Tests for the fit() entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import skewnorm

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting import FitResult, ModelConfig, fit
from chromhandler.fitting.prepared_dataset import prepare_dataset

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _toy_dataset():
    rng = np.random.default_rng(0)
    t = np.arange(2.5, 3.6, 0.001)
    times: list[NDArray[np.float64]] = [t.copy() for _ in range(3)]
    signals: list[NDArray[np.float64]] = [
        (amp * skewnorm.pdf(t, 0.0, loc=3.0, scale=0.025) + 5.0
        + rng.normal(0.0, 0.5, size=t.shape)).astype(np.float64)
        for amp in (100.0, 60.0, 30.0)
    ]
    peaks = [PeakAnnotation(molecule_id="A", rt_min=2.85, rt_max=3.15, mode="single")]
    bases = [BaselineAnnotation(rt_min=2.50, rt_max=2.52),
             BaselineAnnotation(rt_min=3.55, rt_max=3.58)]
    return prepare_dataset(times, signals, peaks, bases)


def test_fit_returns_fitresult() -> None:
    ds = _toy_dataset()
    result = fit(ds, model_config=ModelConfig(num_warmup=30, num_samples=30, num_chains=2))
    assert isinstance(result, FitResult)
    assert result.dataset is ds


def test_fit_with_default_configs() -> None:
    ds = _toy_dataset()
    # Use small config for speed
    result = fit(ds, model_config=ModelConfig(num_warmup=20, num_samples=20, num_chains=2))
    assert "mu_anchor" in result.idata.posterior.data_vars  # type: ignore[attr-defined]
