"""Tests for posterior.py helpers."""

from __future__ import annotations

from typing import Any

import arviz as az
import numpy as np

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.model import ModelConfig, run_mcmc
from chromhandler.fitting.posterior import (
    compute_posterior_predictive,
    compute_prior_predictive,
    derived_areas,
    diagnostics,
)
from chromhandler.fitting.prepared_dataset import prepare_dataset
from chromhandler.fitting.priors import PriorConfig, build_priors


def _idata_fixture():
    rng = np.random.default_rng(0)
    from scipy.stats import skewnorm
    t = np.arange(2.5, 3.6, 0.001)
    times = [t.copy() for _ in range(3)]
    signals: list[np.ndarray[Any, np.dtype[np.float64]]] = [
        amp * np.asarray(skewnorm.pdf(t, 0.0, loc=3.0, scale=0.025)) + 5.0
        + rng.normal(0.0, 0.5, size=t.shape)
        for amp in (100.0, 60.0, 30.0)
    ]
    peaks = [PeakAnnotation(molecule_id="A", rt_min=2.85, rt_max=3.15, mode="single")]
    bases = [BaselineAnnotation(rt_min=2.50, rt_max=2.52),
             BaselineAnnotation(rt_min=3.55, rt_max=3.58)]
    ds = prepare_dataset(times, signals, peaks, bases)
    priors = build_priors(ds, config=PriorConfig())
    config = ModelConfig(num_warmup=30, num_samples=30, num_chains=2, seed=0)
    idata = run_mcmc(ds, priors, config)
    return idata, ds, priors, config


def test_compute_posterior_predictive_adds_group() -> None:
    idata, ds, priors, config = _idata_fixture()
    out = compute_posterior_predictive(idata, ds, priors, config)
    assert isinstance(out, az.InferenceData)
    assert hasattr(out, "posterior_predictive")


def test_compute_prior_predictive_adds_group() -> None:
    """The prior predictive must sample from the actual model priors —
    the band must have non-zero variance and must NOT be byte-equal to
    the observed signal (that would be the obs= conditioning bug)."""
    idata, ds, priors, config = _idata_fixture()
    # Snapshot priors to assert non-mutation.
    snap_mu_loc = priors[0].mu_left_loc
    snap_log_A_scale = priors[0].log_A_left_scale
    snap_log_A_loc = np.asarray(priors[0].log_A_left_loc_per_trace).copy()

    out = compute_prior_predictive(idata, ds, priors, config)
    assert isinstance(out, az.InferenceData)
    assert hasattr(out, "prior")
    assert hasattr(out, "prior_predictive")

    # The bug we are guarding against: every draw == observed signal.
    pp_obs = np.asarray(out.prior_predictive["obs"])  # type: ignore[attr-defined]  # (1, draws, trace, time)
    assert pp_obs.var(axis=1).max() > 0.0, (
        "prior predictive has zero variance across draws — obs= is still "
        "conditioned somewhere"
    )
    flat = pp_obs[0]  # (draws, trace, time)
    matches_data = np.array([
        np.allclose(flat[i], np.asarray(ds.signal), equal_nan=True)
        for i in range(flat.shape[0])
    ])
    assert not matches_data.any(), (
        "at least one prior-predictive draw is bit-equal to dataset.signal"
    )

    # priors_list must not be mutated by compute_prior_predictive.
    assert priors[0].mu_left_loc == snap_mu_loc
    assert priors[0].log_A_left_scale == snap_log_A_scale
    assert np.array_equal(
        np.asarray(priors[0].log_A_left_loc_per_trace), snap_log_A_loc,
    )


def test_derived_areas_shape() -> None:
    idata, ds, priors, _config = _idata_fixture()
    areas = derived_areas(idata)
    # [chain, draw, trace, peak]
    assert areas.ndim == 4
    assert areas.shape[2] == ds.n_trace
    assert areas.shape[3] == len(priors)
    assert np.all(areas > 0)  # exp(log_A) > 0


def test_diagnostics_keys_and_types() -> None:
    idata, *_ = _idata_fixture()
    d = diagnostics(idata)
    assert set(d.keys()) >= {
        "r_hat_max", "r_hat_max_param",
        "ess_min_bulk", "ess_min_param",
        "n_divergent", "fit_healthy",
    }
    assert isinstance(d["r_hat_max"], float)
    assert isinstance(d["fit_healthy"], bool)
    assert isinstance(d["n_divergent"], int)
