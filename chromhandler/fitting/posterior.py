"""Derived quantities and predictive sampling on top of an ArviZ InferenceData.

Pure functions — no plotting. All inputs/outputs are arviz.InferenceData
or numpy arrays. Called from FitResult methods to compute predictive
samples lazily and to extract diagnostics dicts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import arviz
import jax
import numpy as np
import numpyro

from chromhandler.fitting.model import ModelConfig, model

if TYPE_CHECKING:
    from chromhandler.fitting.prepared_dataset import PreparedDataset
    from chromhandler.fitting.priors import SkewNormalPriors


def compute_posterior_predictive(
    idata: arviz.InferenceData,
    dataset: PreparedDataset,
    priors_list: list[SkewNormalPriors],
    config: ModelConfig,
) -> arviz.InferenceData:
    """Sample posterior predictive `obs` and add a `posterior_predictive` group.

    Mutates the passed InferenceData and returns it.
    """
    # Extract posterior samples flattened across (chain, draw)
    posterior = idata.posterior  # type: ignore[attr-defined]
    n_chain = int(posterior.sizes["chain"])  # type: ignore[union-attr]
    n_draw = int(posterior.sizes["draw"])  # type: ignore[union-attr]
    # Build a flat posterior dict for Predictive
    flat_posterior = {
        name: np.asarray(posterior[name]).reshape(  # type: ignore[index]
            (n_chain * n_draw, *np.asarray(posterior[name]).shape[2:])  # type: ignore[index]
        )
        for name in posterior.data_vars  # type: ignore[union-attr]
    }
    predictive = numpyro.infer.Predictive(
        model, posterior_samples=flat_posterior, return_sites=["obs"],
    )
    rng_key = jax.random.PRNGKey(config.seed + 1)
    samples = predictive(rng_key, dataset, priors_list, config)
    obs = np.asarray(samples["obs"]).reshape(
        (n_chain, n_draw, dataset.n_trace, dataset.time.shape[1])
    )
    # Build posterior_predictive group manually
    coords = {
        "chain": np.arange(n_chain),
        "draw": np.arange(n_draw),
        "trace": np.arange(dataset.n_trace),
        "time_idx": np.arange(dataset.time.shape[1]),
    }
    pp = arviz.from_dict(
        posterior_predictive={"obs": obs},
        coords=coords,
        dims={"obs": ["chain", "draw", "trace", "time_idx"]},
    )
    idata.extend(pp)
    return idata


def _weakly_informative_priors(
    dataset: PreparedDataset,
    priors_list: list[SkewNormalPriors],
) -> list[SkewNormalPriors]:
    """Construct weakly-informative versions of the input priors for the
    prior-predictive visualisation.

    The data-derived priors that ``build_priors`` produces are tight — each
    per-trace amplitude prior is centred on that trace's *measured* area —
    so sampling from them gives a prior predictive band that trivially
    overlaps the data, which is uninformative as a check.

    The "canonical" prior predictive should reflect what the model believes
    *before* conditioning on the per-trace data. We rebuild each
    :class:`SkewNormalPriors` with:

    - ``mu`` centred at the annotation-window centre, scale = window_width / 4
    - ``log_sigma`` at the geometric centre of its bounds, scale = bounds-range / 4
    - ``gamma1`` centred at 0, scale 0.3 (covers typical chromatographic shapes)
    - ``log_A`` anchored at the *median* across measured trace areas — a
      magnitude reference — with scale ``log(10)`` (factor-of-10 uncertainty
      in either direction), shared across all traces (no per-trace lock-in)

    Window bounds (mu_low/high, log_sigma_low/high) are kept unchanged so the
    truncation still respects the user's annotation geometry.
    """
    import dataclasses

    weak: list[SkewNormalPriors] = []
    for p in priors_list:
        window_width = p.mu_left_high - p.mu_left_low
        log_sigma_range = p.log_sigma_left_high - p.log_sigma_left_low

        # Magnitude anchor: median of the measured per-trace log areas.
        # Wide scale (log(10) ≈ 2.30) makes the prior predictive span ~2 orders
        # of magnitude in amplitude.
        log_A_anchor = float(np.median(p.log_A_left_loc_per_trace))
        weak_log_A = np.full(dataset.n_trace, log_A_anchor, dtype=np.float64)

        weak.append(dataclasses.replace(
            p,
            mu_left_loc=(p.mu_left_low + p.mu_left_high) / 2,
            mu_left_scale=window_width / 4,
            log_sigma_left_loc=(p.log_sigma_left_low + p.log_sigma_left_high) / 2,
            log_sigma_left_scale=log_sigma_range / 4,
            gamma1_left_loc=0.0,
            gamma1_left_scale=0.3,
            log_A_left_loc_per_trace=weak_log_A,
            log_A_left_scale=float(np.log(10.0)),
        ))
    return weak


def compute_prior_predictive(
    idata: arviz.InferenceData,
    dataset: PreparedDataset,
    priors_list: list[SkewNormalPriors],
    config: ModelConfig,
) -> arviz.InferenceData:
    """Sample from a weakly-informative prior, run model forward; adds
    ``prior`` + ``prior_predictive`` groups to ``idata``.

    The "prior" used here is NOT the data-derived ``priors_list`` (which
    would produce a predictive band trivially overlapping the data, because
    per-trace amplitude priors are centred on per-trace measurements).
    Instead, we substitute a weakly-informative version that reflects what
    the model believes before conditioning on data — see
    :func:`_weakly_informative_priors`. This is the canonical Bayesian
    prior-predictive check: "given my priors, what data would my model
    generate?"
    """
    weak_priors = _weakly_informative_priors(dataset, priors_list)
    predictive = numpyro.infer.Predictive(model, num_samples=config.prior_predictive_n_samples)
    rng_key = jax.random.PRNGKey(config.seed + 2)
    samples = predictive(rng_key, dataset, weak_priors, config)

    n_samples = config.prior_predictive_n_samples
    n_trace = dataset.n_trace
    n_time = dataset.time.shape[1]

    obs = np.asarray(samples["obs"]).reshape((1, n_samples, n_trace, n_time))
    coords: dict[str, Any] = {
        "chain": [0],
        "draw": np.arange(n_samples),
        "trace": np.arange(n_trace),
        "time_idx": np.arange(n_time),
    }
    prior_dict = {
        name: np.asarray(samples[name]).reshape(
            (1, n_samples, *np.asarray(samples[name]).shape[1:])
        )
        for name in samples
        if name != "obs"
    }
    pp = arviz.from_dict(
        prior=prior_dict,
        prior_predictive={"obs": obs},
        coords=coords,
        dims={"obs": ["chain", "draw", "trace", "time_idx"]},
    )
    idata.extend(pp)
    return idata


def derived_areas(idata: arviz.InferenceData) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Per-(chain, draw, trace, peak) posterior areas = exp(log_A_left).

    Returns:
        Array shape ``[n_chain, n_draw, n_trace, n_peak]``.
    """
    log_A: np.ndarray[Any, np.dtype[np.float64]] = np.asarray(idata.posterior["log_A_left"])  # type: ignore[attr-defined]
    return np.exp(log_A)  # type: ignore[return-value]


def diagnostics(idata: arviz.InferenceData) -> dict[str, Any]:
    """Quick "did MCMC converge?" summary.

    Returns:
        {
            "r_hat_max": float, "r_hat_max_param": str,
            "ess_min_bulk": float, "ess_min_param": str,
            "n_divergent": int,
            "n_samples_total": int,
            "fit_healthy": bool,
        }
    """
    summary = arviz.summary(idata, kind="diagnostics")
    r_hat = summary["r_hat"]
    ess_bulk = summary["ess_bulk"]

    r_hat_max = float(r_hat.max())
    r_hat_max_param = str(r_hat.idxmax())
    ess_min_bulk = float(ess_bulk.min())
    ess_min_param = str(ess_bulk.idxmin())

    n_divergent = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:  # type: ignore[attr-defined]
        n_divergent = int(np.asarray(idata.sample_stats["diverging"]).sum())  # type: ignore[attr-defined]

    n_chain = int(idata.posterior.sizes["chain"])  # type: ignore[attr-defined]
    n_draw = int(idata.posterior.sizes["draw"])  # type: ignore[attr-defined]
    n_samples_total = n_chain * n_draw

    # NaN-guard r_hat: ArviZ returns NaN when within-chain variance is zero
    # (degenerate high-SNR posterior — chains lock on the same value). Such
    # a fit is not pathological, so treat NaN as "healthy" on this axis.
    rhat_ok = bool(np.isnan(r_hat_max) or r_hat_max < 1.01)
    fit_healthy = bool(rhat_ok and ess_min_bulk > 400 and n_divergent == 0)

    return {
        "r_hat_max": r_hat_max,
        "r_hat_max_param": r_hat_max_param,
        "ess_min_bulk": ess_min_bulk,
        "ess_min_param": ess_min_param,
        "n_divergent": n_divergent,
        "n_samples_total": n_samples_total,
        "fit_healthy": fit_healthy,
    }
