"""NumPyro Bayesian model for the single-peak skew-normal fitter.

One mixture component per :class:`~chromhandler.annotations.PeakAnnotation`.

All sample sites are unit-scale ``Normal(0, 1)`` (non-centred, suffix
``_raw``); the physically meaningful quantities (``mu``, ``width``,
``skew``, ``area``, ``time_shift``, ``time_stretch``, ``noise``, ...) are
exposed as ``numpyro.deterministic`` sites in their natural (unconstrained)
space. Log-space parameterisations of ``width`` and ``noise`` stay inside
the model body — they are implementation, not interface.

Hard ``TruncatedNormal`` bounds are replaced by soft priors plus, where
a physical constraint is real, a smooth bijector:

- ``mu``, ``width`` : ``width`` is sampled in log-space (LogNormal,
  non-centred) and exposed in natural space; ``mu`` is an unconstrained
  ``Normal``. The likelihood + a positive area prior identify them.
- ``skew`` : has a real physical bound ``|skew| < GAMMA1_MAX`` from
  skew-normal math (max skewness of any SN equals that of the half-normal);
  enforced via ``tanh`` so the boundary is smooth.
- ``area`` : positivity enforced via ``softplus`` of the non-centred draw,
  not by truncation.
- ``noise`` : sampled in log-space, exposed in natural space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import arviz
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from chromhandler.fitting.skew_normal import GAMMA1_MAX, density_cp

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from chromhandler.fitting.prepared_dataset import PreparedDataset
    from chromhandler.fitting.priors import SkewNormalPriors


def _compute_baseline_se(
    dataset: PreparedDataset,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-trace OLS standard errors for the baseline intercept and slope.

    Returns ``(intercept_se, slope_se)``, both shape ``[n_trace]``.
    """
    n_trace = dataset.n_trace
    intercept_se = np.zeros(n_trace, dtype=np.float64)
    slope_se = np.zeros(n_trace, dtype=np.float64)
    for tr in range(n_trace):
        t = dataset.time[tr]
        s = dataset.signal[tr]
        baseline_mask = np.zeros_like(t, dtype=bool)
        for ba in dataset.baseline_annotations:
            baseline_mask |= ((t >= ba.rt_min) & (t <= ba.rt_max) & np.isfinite(s))
        if baseline_mask.sum() < 3:
            intercept_se[tr] = float(dataset.noise_per_trace[tr])
            slope_se[tr] = float(dataset.noise_per_trace[tr])
            continue
        t_b = t[baseline_mask]
        s_b = s[baseline_mask]
        X = np.column_stack([np.ones_like(t_b), t_b])
        beta, *_ = np.linalg.lstsq(X, s_b, rcond=None)
        residuals = s_b - X @ beta
        sigma2 = float(np.sum(residuals**2) / max(t_b.size - 2, 1))  # type: ignore[arg-type]
        try:
            cov = sigma2 * np.linalg.inv(X.T @ X)
            intercept_se[tr] = float(np.sqrt(max(cov[0, 0], 0.0)))
            slope_se[tr] = float(np.sqrt(max(cov[1, 1], 0.0)))
        except np.linalg.LinAlgError:
            intercept_se[tr] = float(dataset.noise_per_trace[tr])
            slope_se[tr] = float(dataset.noise_per_trace[tr])
    return intercept_se, slope_se


@dataclass(frozen=True)
class ModelConfig:
    """User-facing configuration for the NumPyro fit."""

    # --- HMC / NUTS settings ---
    num_warmup: int = 500
    num_samples: int = 500
    num_chains: int = 4
    target_accept_prob: float = 0.9
    max_tree_depth: int = 10
    seed: int = 0

    # --- Model-layer priors (per-trace, not per-peak) ---
    baseline_intercept_se_floor: float = 1.0
    baseline_slope_se_floor: float = 0.01

    # --- Effective noise prior ---
    # noise[trace] is LogNormal: log(noise) ~ Normal(log(baseline_noise),
    # log_noise_scale). Scalar additive Gaussian likelihood per trace.
    # The hyperprior anchor is the data-derived baseline RMS; the data can
    # inflate it to absorb model-form mismatch. scale=2.0 on the log axis
    # allows up to ~7x inflation per 1 sigma -- weakly informative.
    log_noise_scale: float = 2.0

    # --- Per-trace linear time-axis warp: ---
    #   t' = time_shift[trace] + time_stretch[trace] * t
    # time_shift (additive, time units): anchored at ~5*dt -- typical
    # injection-timing offset scale. time_stretch (multiplicative,
    # dimensionless): anchored near 1 with ~1% deviation -- typical HPLC
    # column drift. Both have sum-to-zero centring per trace, which breaks
    # the global mu <-> warp degeneracy.
    warp_shift_scale_dt_multiplier: float = 5.0
    warp_stretch_scale: float = 0.01

    # --- Prior predictive ---
    prior_predictive_n_samples: int = 200


def model(
    dataset: PreparedDataset,
    priors_list: list[SkewNormalPriors],
    config: ModelConfig,
) -> None:
    """Non-centred, unit-scale skew-normal model.

    All ``numpyro.sample`` sites are ``Normal(0, 1)`` (suffix ``_raw``).
    The physical quantities are exposed as ``numpyro.deterministic``
    sites so downstream summary/plotting code can read them unchanged.
    Log-space parameterisations of ``width`` and ``noise`` are internal:
    only the natural-space sites are exposed.

    Deterministic sites:
        - ``mu[peak]``              = mu_loc + mu_scale * mu_raw
        - ``width[peak]``           = exp(log_width_loc + log_width_scale * width_raw)
        - ``skew[peak]``            = GAMMA1_MAX * tanh((skew_loc + skew_scale * skew_raw) / GAMMA1_MAX)
        - ``area[trace, peak]``     = softplus(area_loc + area_scale * area_raw)
        - ``time_shift[trace]``     = shift_scale * time_shift_raw, centred so mean == 0
        - ``time_stretch[trace]``   = exp(stretch_scale * time_stretch_raw), centred so mean(log) == 0
        - ``mu_warped[trace, peak]``    = (mu[peak] - time_shift[trace]) / time_stretch[trace]
        - ``width_warped[trace, peak]`` = width[peak] / time_stretch[trace]
        - ``baseline_intercept[trace]`` / ``baseline_slope[trace]``
        - ``noise[trace]``          = exp(log(baseline_noise) + log_noise_scale * noise_raw)
    """
    n_trace = dataset.n_trace
    n_peak = len(priors_list)
    dt_global = float(dataset.dt_global)

    # === Shared per-peak shape priors (non-centred Normal(0,1)) ===
    mu_loc = jnp.asarray([p.mu_loc for p in priors_list])
    mu_scale = jnp.asarray([p.mu_scale for p in priors_list])
    mu_raw = numpyro.sample("mu_raw", dist.Normal(jnp.zeros(n_peak), 1.0))
    mu = numpyro.deterministic("mu", mu_loc + mu_scale * mu_raw)

    # width: sampled LogNormal (non-centred in log-space), exposed in
    # natural space. The log-space parameterisation is implementation,
    # not interface — downstream code reads ``width`` directly.
    log_width_loc = jnp.asarray([p.log_width_loc for p in priors_list])
    log_width_scale = jnp.asarray([p.log_width_scale for p in priors_list])
    width_raw = numpyro.sample(
        "width_raw", dist.Normal(jnp.zeros(n_peak), 1.0),
    )
    width = numpyro.deterministic(
        "width", jnp.exp(log_width_loc + log_width_scale * width_raw),
    )

    # skew: real physical bound |skew| < GAMMA1_MAX (skew-normal math).
    # Smooth bijector instead of TruncatedNormal wall.
    skew_loc = jnp.asarray([p.skew_loc for p in priors_list])
    skew_scale = jnp.asarray([p.skew_scale for p in priors_list])
    skew_max = float(GAMMA1_MAX)
    skew_raw = numpyro.sample("skew_raw", dist.Normal(jnp.zeros(n_peak), 1.0))
    skew_unconstrained = skew_loc + skew_scale * skew_raw
    skew = numpyro.deterministic(
        "skew", skew_max * jnp.tanh(skew_unconstrained / skew_max),
    )

    # === Per-(trace, peak) area: softplus(non-centred Normal) ===
    # Positivity enforced smoothly; no truncation wall. For supported
    # traces (area_loc >> area_scale) softplus is essentially identity.
    # For unsupported traces (area_loc=0) softplus collapses to ~0 but
    # the likelihood can pull area>0 without fighting a vanishing prior.
    area_loc = jnp.asarray(
        np.stack([p.area_loc_per_trace for p in priors_list], axis=1)
    )  # [n_trace, n_peak]
    area_scale = jnp.asarray(
        np.stack([p.area_scale_per_trace for p in priors_list], axis=1)
    )  # [n_trace, n_peak]
    area_raw = numpyro.sample(
        "area_raw", dist.Normal(jnp.zeros((n_trace, n_peak)), 1.0),
    )
    area = numpyro.deterministic(
        "area", jax.nn.softplus(area_loc + area_scale * area_raw),
    )

    intercept_se, slope_se = _compute_baseline_se(dataset)
    intercept_se_eff = np.maximum(intercept_se, config.baseline_intercept_se_floor)
    slope_se_eff = np.maximum(slope_se, config.baseline_slope_se_floor)
    baseline_intercept_loc = jnp.asarray(dataset.baseline_intercept)
    baseline_slope_loc = jnp.asarray(dataset.baseline_slope)
    baseline_intercept_raw = numpyro.sample(
        "baseline_intercept_raw", dist.Normal(jnp.zeros(n_trace), 1.0),
    )
    baseline_intercept = numpyro.deterministic(
        "baseline_intercept",
        baseline_intercept_loc + jnp.asarray(intercept_se_eff) * baseline_intercept_raw,
    )
    baseline_slope_raw = numpyro.sample(
        "baseline_slope_raw", dist.Normal(jnp.zeros(n_trace), 1.0),
    )
    baseline_slope = numpyro.deterministic(
        "baseline_slope",
        baseline_slope_loc + jnp.asarray(slope_se_eff) * baseline_slope_raw,
    )

    # === Per-trace additive noise (LogNormal, non-centred in log-space) ===
    # noise[trace] is anchored at the data-derived baseline RMS; the data
    # can shift it to absorb model-form mismatch within the scale prior.
    # Sampled in log-space internally; exposed in natural space.
    log_noise_loc = jnp.log(jnp.asarray(dataset.noise_per_trace))
    noise_raw = numpyro.sample(
        "noise_raw", dist.Normal(jnp.zeros(n_trace), 1.0),
    )
    noise = numpyro.deterministic(
        "noise", jnp.exp(log_noise_loc + config.log_noise_scale * noise_raw),
    )

    # === Per-trace linear time-axis warp ===
    # t' = time_shift[trace] + time_stretch[trace] * t. Captures additive
    # offsets (time_shift) and proportional column drift (time_stretch).
    # Both follow the non-centred + sum-to-zero pattern: the centring
    # breaks the global mu<->warp degeneracy (translating mu by epsilon is
    # equivalent to all time_shift[trace] += epsilon; scaling mu by k is
    # equivalent to all time_stretch[trace] /= k).
    shift_scale = config.warp_shift_scale_dt_multiplier * dt_global
    time_shift_raw = numpyro.sample(
        "time_shift_raw", dist.Normal(jnp.zeros(n_trace), 1.0),
    )
    _shift = shift_scale * time_shift_raw
    time_shift = numpyro.deterministic("time_shift", _shift - jnp.mean(_shift))

    time_stretch_raw = numpyro.sample(
        "time_stretch_raw", dist.Normal(jnp.zeros(n_trace), 1.0),
    )
    _log_stretch = config.warp_stretch_scale * time_stretch_raw
    log_stretch_centred = _log_stretch - jnp.mean(_log_stretch)
    time_stretch = numpyro.deterministic("time_stretch", jnp.exp(log_stretch_centred))

    # === Predicted signal ===
    # Warped per-(trace, peak) shape derived from the linear warp.
    # Physics: when the time axis stretches by time_stretch, peak position
    # scales as (mu - time_shift)/time_stretch and peak width scales as
    # width/time_stretch. skew is dimensionless (skewness coefficient)
    # and does not transform.
    mu_warped = numpyro.deterministic(
        "mu_warped",
        (mu[None, :] - time_shift[:, None]) / time_stretch[:, None],
    )  # [n_trace, n_peak]
    width_warped = numpyro.deterministic(
        "width_warped", width[None, :] / time_stretch[:, None],
    )  # [n_trace, n_peak]

    time_arr = jnp.asarray(dataset.time)
    baseline = baseline_intercept[:, None] + baseline_slope[:, None] * time_arr

    # Vectorise the per-peak density into one fused JIT kernel (instead of
    # n_peak separate density_cp calls inside a Python loop). Broadcasting:
    #   time:   [n_trace, 1,      n_time]
    #   mu:     [n_trace, n_peak, 1     ]
    #   width:  [n_trace, n_peak, 1     ]
    #   skew:   [1,       n_peak, 1     ]
    # → dens_all shape [n_trace, n_peak, n_time], then weighted-sum over peaks.
    dens_all = density_cp(
        time_arr[:, None, :],                 # type: ignore[arg-type]
        mu_warped[:, :, None],                # type: ignore[arg-type]
        width_warped[:, :, None],             # type: ignore[arg-type]
        skew[None, :, None],                  # type: ignore[arg-type]
    )
    peak_contrib = jnp.sum(area[:, :, None] * dens_all, axis=1)  # [n_trace, n_time]
    predicted = baseline + peak_contrib

    # === Likelihood (NaN-masked additive Gaussian) ===
    # Sanitise any NaN/Inf in `predicted` so Normal never sees an invalid
    # location. Masked positions are zeroed out of the likelihood so the
    # substitution is loss-free.
    predicted = jnp.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0)
    with numpyro.handlers.mask(mask=jnp.asarray(dataset.valid_mask)):
        numpyro.sample("obs", dist.Normal(predicted, noise[:, None]))


def run_mcmc(
    dataset: PreparedDataset,
    priors_list: list[SkewNormalPriors],
    config: ModelConfig,
) -> arviz.InferenceData:
    """Run NUTS sampling and return an ArviZ InferenceData."""
    conditioned_model = numpyro.handlers.condition(
        model, data={"obs": jnp.asarray(dataset.signal)},
    )
    kernel = numpyro.infer.NUTS(
        conditioned_model,
        target_accept_prob=config.target_accept_prob,
        max_tree_depth=config.max_tree_depth,
    )
    mcmc = numpyro.infer.MCMC(
        kernel,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
        progress_bar=True,
    )
    mcmc.run(
        jax.random.PRNGKey(config.seed),
        dataset, priors_list, config,
    )
    return arviz.from_numpyro(mcmc)  # type: ignore[return-value]
