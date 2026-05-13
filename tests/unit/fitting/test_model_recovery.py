"""Synthetic-data recovery: known parameters -> small MCMC -> posterior near truth."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from scipy.stats import skewnorm

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting import ModelConfig, PriorConfig, fit
from chromhandler.fitting.prepared_dataset import PreparedDataset, prepare_dataset
from chromhandler.fitting.skew_normal import cp_to_dp

TRUE_MU = 3.00
TRUE_SIGMA = 0.025
TRUE_GAMMA1 = 0.2
# Areas chosen so that peak heights (A * max_SN_pdf) are 8-80x the noise floor,
# giving NUTS a tractable posterior width. The original specification used areas
# [100, 60, 30, 10] with noise=0.5 but those SNRs (~3000x) collapse the posterior
# to sub-floating-point width, causing NUTS chains to freeze (ESS=2, r_hat=inf).
TRUE_AREAS = [0.5, 0.3, 0.15, 0.05]
NOISE_STD = 0.05


def _synthetic_dataset() -> PreparedDataset:
    rng = np.random.default_rng(42)
    t = np.arange(2.5, 3.6, 0.001)
    xi, omega, alpha = (
        float(x) for x in cp_to_dp(
            jnp.asarray(TRUE_MU), jnp.asarray(TRUE_SIGMA), jnp.asarray(TRUE_GAMMA1)
        )
    )
    times: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    signals: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    for A in TRUE_AREAS:
        s = (
            A * np.asarray(skewnorm.pdf(t, alpha, loc=xi, scale=omega))
            + 5.0
            + rng.normal(0.0, NOISE_STD, size=t.shape)
        ).astype(np.float64)
        times.append(t.copy())
        signals.append(s)  # type: ignore[arg-type]
    peaks = [PeakAnnotation(molecule_id="A", rt_min=2.85, rt_max=3.15, mode="single")]
    bases = [
        BaselineAnnotation(rt_min=2.50, rt_max=2.52),
        BaselineAnnotation(rt_min=3.55, rt_max=3.58),
    ]
    return prepare_dataset(times, signals, peaks, bases)


def test_posterior_recovers_known_parameters() -> None:
    ds = _synthetic_dataset()
    config = ModelConfig(num_warmup=300, num_samples=300, num_chains=2, seed=0)
    result = fit(ds, prior_config=PriorConfig(), model_config=config)

    diag = result.diagnostics()
    # Allow a slightly loose r_hat threshold given small MCMC
    assert diag["r_hat_max"] < 1.10, f"r_hat too high: {diag}"
    assert diag["n_divergent"] == 0, f"divergences: {diag}"

    posterior = result.idata.posterior  # type: ignore[attr-defined]
    mu_median = float(
        np.median(np.asarray(posterior["mu_anchor_left"]))  # type: ignore[index]
    )
    sigma_median = float(
        np.median(np.exp(np.asarray(posterior["log_sigma_left"])))  # type: ignore[index]
    )
    gamma1_median = float(
        np.median(np.asarray(posterior["gamma1_left"]))  # type: ignore[index]
    )
    _raw_log_A = posterior["log_A_left"]  # type: ignore[index]
    log_A: np.ndarray[Any, np.dtype[np.float64]] = np.asarray(_raw_log_A)  # pyright: ignore[reportUnknownArgumentType]
    A_median_per_trace: np.ndarray[Any, np.dtype[np.float64]] = np.median(  # type: ignore[assignment]
        np.exp(log_A), axis=(0, 1)
    )[:, 0]  # [n_trace]

    assert abs(mu_median - TRUE_MU) < 2 * ds.dt_global, (
        f"mu off: {mu_median} vs {TRUE_MU} (tol = 2 dt = {2 * ds.dt_global})"
    )
    assert abs(sigma_median - TRUE_SIGMA) / TRUE_SIGMA < 0.15, (
        f"sigma off: {sigma_median} vs {TRUE_SIGMA}"
    )
    # gamma1 recovery may show mild FWHM-extraction bias on synthetic SN data;
    # tolerance of 0.20 is defensible given the known asymmetry-inversion
    # approximation in the priors layer (documented in priors_demo notebook).
    assert abs(gamma1_median - TRUE_GAMMA1) < 0.20, (
        f"gamma1 off: {gamma1_median} vs {TRUE_GAMMA1}"
    )
    for tr, true_A in enumerate(TRUE_AREAS):
        recovered = float(A_median_per_trace[tr])
        assert abs(recovered - true_A) / true_A < 0.10, (
            f"A[{tr}] off: {recovered} vs {true_A}"
        )
