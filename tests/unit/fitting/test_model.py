"""Tests for the NumPyro model() function."""

from __future__ import annotations

import jax
import numpy as np
import numpyro
from scipy.stats import skewnorm

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.model import ModelConfig, model
from chromhandler.fitting.prepared_dataset import prepare_dataset
from chromhandler.fitting.priors import PriorConfig, build_priors


def _toy_setup(n_trace: int = 3):
    rng = np.random.default_rng(0)
    t = np.arange(2.5, 3.6, 0.001)
    times: list[np.ndarray[tuple[int], np.dtype[np.float64]]] = [t.copy() for _ in range(n_trace)]
    signals: list[np.ndarray[tuple[int], np.dtype[np.float64]]] = []
    for amp in np.linspace(100.0, 40.0, n_trace):
        s: np.ndarray[tuple[int], np.dtype[np.float64]] = amp * skewnorm.pdf(t, 0.0, loc=3.0, scale=0.025)
        noise: np.ndarray[tuple[int], np.dtype[np.float64]] = np.asarray(rng.normal(0.0, 0.5, size=t.shape))  # type: ignore[reportUnknownArgumentType]
        s = s + 5.0 + noise
        signals.append(s)
    peaks = [PeakAnnotation(molecule_id="A", rt_min=2.85, rt_max=3.15, mode="single")]
    bases = [BaselineAnnotation(rt_min=2.50, rt_max=2.52),
             BaselineAnnotation(rt_min=3.55, rt_max=3.58)]
    ds = prepare_dataset(times, signals, peaks, bases)
    priors = build_priors(ds, config=PriorConfig())
    return ds, priors


def test_model_prior_predictive_runs_and_has_right_shape() -> None:
    ds, priors = _toy_setup(n_trace=3)
    config = ModelConfig(num_warmup=1, num_samples=1, num_chains=1)
    predictive = numpyro.infer.Predictive(model, num_samples=2)
    rng_key = jax.random.PRNGKey(0)
    samples = predictive(rng_key, ds, priors, config)
    # All expected sites present
    assert "mu_anchor_left" in samples
    assert "log_sigma_left" in samples
    assert "gamma1_left" in samples
    assert "log_A_left" in samples
    assert "trace_shift" in samples
    assert "baseline_intercept" in samples
    assert "baseline_slope" in samples
    assert "obs" in samples
    # Shapes
    assert samples["mu_anchor_left"].shape == (2, 1)            # (n_samples, n_peak)
    assert samples["log_A_left"].shape == (2, ds.n_trace, 1)    # (n_samples, n_trace, n_peak)
    assert samples["trace_shift"].shape == (2, ds.n_trace)
    assert samples["obs"].shape == (2, ds.n_trace, ds.time.shape[1])


def test_model_obs_values_are_finite_under_prior() -> None:
    ds, priors = _toy_setup(n_trace=3)
    config = ModelConfig(num_warmup=1, num_samples=1, num_chains=1)
    predictive = numpyro.infer.Predictive(model, num_samples=5)
    rng_key = jax.random.PRNGKey(0)
    samples = predictive(rng_key, ds, priors, config)
    # `obs` from prior predictive can be large but should be finite
    assert np.all(np.isfinite(np.asarray(samples["obs"])))
