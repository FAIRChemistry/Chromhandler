"""NumPyro Bayesian model for the single-peak skew-normal fitter.

One mixture component per :class:`~chromhandler.annotations.PeakAnnotation`.

All sample sites are unit-scale ``Normal(0, 1)`` (non-centred); the
physically meaningful quantities (``mu_anchor``, ``log_sigma``, ``gamma1``,
``A``, ...) are recovered via deterministic transforms. Hard
``TruncatedNormal`` bounds are replaced by soft priors plus, where a
physical constraint is real, a smooth bijector:

- ``mu``, ``log_sigma`` : unconstrained ``Normal`` priors. The likelihood
  + a positive area prior identify them; no walls needed.
- ``gamma1`` : has a real physical bound ``|gamma1| < GAMMA1_MAX`` from
  skew-normal math; enforced via ``tanh`` so the boundary is smooth.
- ``A`` : positivity enforced via ``softplus`` of the non-centred draw,
  not by truncation.
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


SAMPLED_PARAMETER_NAMES: tuple[str, ...] = (
    "mu_anchor", "log_sigma", "gamma1", "A",
    "a", "b",
    "mu_eff", "sigma_eff",
    "baseline_intercept", "baseline_slope", "log_noise",
)


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
    # log_noise[trace] ~ Normal(log(baseline_noise[trace]), log_noise_scale).
    # Scalar additive Gaussian likelihood per trace. The hyperprior anchor
    # is the data-derived baseline RMS; the data can inflate it to absorb
    # model-form mismatch. scale=2.0 on the log axis allows up to ~7x
    # inflation per 1 sigma -- weakly informative.
    log_noise_scale: float = 2.0

    # --- Per-trace linear time-axis warp: t' = a[trace] + b[trace] * t ---
    # a (additive, time units): anchored at ~5*dt -- typical injection-timing
    # offset scale. b (multiplicative, dimensionless): anchored near 1 with
    # ~1% deviation -- typical HPLC column drift. Both have sum-to-zero
    # centring per trace, which breaks the global anchor<->warp degeneracy.
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

    Deterministic sites:
        - ``mu_anchor[peak]``       = mu_loc + mu_scale * mu_anchor_raw
        - ``log_sigma[peak]``       = log_sigma_loc + log_sigma_scale * log_sigma_raw
        - ``gamma1[peak]``          = GAMMA1_MAX * tanh((gamma1_loc + gamma1_scale * gamma1_raw) / GAMMA1_MAX)
        - ``A[trace, peak]``        = softplus(area_loc + area_scale * A_raw)
        - ``a[trace]``              = shift_scale * a_raw, centred so mean(a) == 0
        - ``b[trace]``              = exp(log_b_raw * stretch_scale), centred so mean(log b) == 0
        - ``mu_eff[trace, peak]``   = (mu_anchor[peak] - a[trace]) / b[trace]
        - ``sigma_eff[trace, peak]`` = sigma[peak] / b[trace]
        - ``baseline_intercept[trace]`` / ``baseline_slope[trace]``
        - ``log_noise[trace]``      = log(baseline_noise) + log_noise_scale * log_noise_raw
    """
    n_trace = dataset.n_trace
    n_peak = len(priors_list)
    dt_global = float(dataset.dt_global)

    # === Shared per-peak shape priors (non-centred Normal(0,1)) ===
    mu_loc = jnp.asarray([p.mu_loc for p in priors_list])
    mu_scale = jnp.asarray([p.mu_scale for p in priors_list])
    mu_raw = numpyro.sample("mu_anchor_raw", dist.Normal(jnp.zeros(n_peak), 1.0))
    mu_anchor = numpyro.deterministic("mu_anchor", mu_loc + mu_scale * mu_raw)

    log_sigma_loc = jnp.asarray([p.log_sigma_loc for p in priors_list])
    log_sigma_scale = jnp.asarray([p.log_sigma_scale for p in priors_list])
    log_sigma_raw = numpyro.sample(
        "log_sigma_raw", dist.Normal(jnp.zeros(n_peak), 1.0),
    )
    log_sigma = numpyro.deterministic(
        "log_sigma", log_sigma_loc + log_sigma_scale * log_sigma_raw,
    )

    # gamma1: real physical bound |gamma1| < GAMMA1_MAX (skew-normal math).
    # Smooth bijector instead of TruncatedNormal wall.
    gamma1_loc = jnp.asarray([p.gamma1_loc for p in priors_list])
    gamma1_scale = jnp.asarray([p.gamma1_scale for p in priors_list])
    gamma1_max = float(GAMMA1_MAX)
    gamma1_raw = numpyro.sample("gamma1_raw", dist.Normal(jnp.zeros(n_peak), 1.0))
    gamma1_unconstrained = gamma1_loc + gamma1_scale * gamma1_raw
    gamma1 = numpyro.deterministic(
        "gamma1", gamma1_max * jnp.tanh(gamma1_unconstrained / gamma1_max),
    )

    # === Per-(trace, peak) area: softplus(non-centred Normal) ===
    # Positivity enforced smoothly; no truncation wall. For supported
    # traces (area_loc >> area_scale) softplus is essentially identity.
    # For unsupported traces (area_loc=0) softplus collapses to ~0 but
    # the likelihood can pull A>0 without fighting a vanishing prior.
    area_loc = jnp.asarray(
        np.stack([p.area_loc_per_trace for p in priors_list], axis=1)
    )  # [n_trace, n_peak]
    area_scale = jnp.asarray(
        np.stack([p.area_scale_per_trace for p in priors_list], axis=1)
    )  # [n_trace, n_peak]
    A_raw = numpyro.sample(
        "A_raw", dist.Normal(jnp.zeros((n_trace, n_peak)), 1.0),
    )
    A = numpyro.deterministic("A", jax.nn.softplus(area_loc + area_scale * A_raw))

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

    # === Per-trace additive noise (non-centred LogNormal in log-space) ===
    # log_noise[t] anchored at the data-derived baseline RMS; the data
    # can shift it to absorb model-form mismatch within the scale prior.
    log_noise_loc = jnp.log(jnp.asarray(dataset.noise_per_trace))
    log_noise_raw = numpyro.sample(
        "log_noise_raw", dist.Normal(jnp.zeros(n_trace), 1.0),
    )
    log_noise = numpyro.deterministic(
        "log_noise", log_noise_loc + config.log_noise_scale * log_noise_raw,
    )
    noise = jnp.exp(log_noise)

    # === Per-trace linear time-axis warp ===
    # t' = a[trace] + b[trace] * t. Captures additive offsets (a) and
    # proportional column drift (b). Both follow the non-centred +
    # sum-to-zero pattern: the centring breaks the global anchor<->warp
    # degeneracy (translating mu_anchor by epsilon is equivalent to
    # all a[trace] += epsilon; scaling mu_anchor by k is equivalent
    # to all b[trace] /= k).
    shift_scale = config.warp_shift_scale_dt_multiplier * dt_global
    a_raw = numpyro.sample(
        "a_raw", dist.Normal(jnp.zeros(n_trace), 1.0),
    )
    _a = shift_scale * a_raw
    a = numpyro.deterministic("a", _a - jnp.mean(_a))

    log_b_raw = numpyro.sample(
        "log_b_raw", dist.Normal(jnp.zeros(n_trace), 1.0),
    )
    _log_b = config.warp_stretch_scale * log_b_raw
    log_b = _log_b - jnp.mean(_log_b)
    b = numpyro.deterministic("b", jnp.exp(log_b))

    # === Predicted signal ===
    # Effective per-(trace, peak) shape derived from the linear warp.
    # Physics: when the time axis stretches by b, peak position scales
    # as (mu - a)/b and peak width scales as sigma/b. gamma1 is
    # dimensionless (skewness coefficient) and does not transform.
    sigma = jnp.exp(log_sigma)
    mu_eff = numpyro.deterministic(
        "mu_eff", (mu_anchor[None, :] - a[:, None]) / b[:, None],
    )  # [n_trace, n_peak]
    sigma_eff = numpyro.deterministic(
        "sigma_eff", sigma[None, :] / b[:, None],
    )  # [n_trace, n_peak]

    time_arr = jnp.asarray(dataset.time)
    baseline = baseline_intercept[:, None] + baseline_slope[:, None] * time_arr

    # Vectorise the per-peak density into one fused JIT kernel (instead of
    # n_peak separate density_cp calls inside a Python loop). Broadcasting:
    #   time:   [n_trace, 1,      n_time]
    #   mu:     [n_trace, n_peak, 1     ]
    #   sigma:  [n_trace, n_peak, 1     ]
    #   gamma1: [1,       n_peak, 1     ]
    # → dens_all shape [n_trace, n_peak, n_time], then weighted-sum over peaks.
    dens_all = density_cp(
        time_arr[:, None, :],                 # type: ignore[arg-type]
        mu_eff[:, :, None],                   # type: ignore[arg-type]
        sigma_eff[:, :, None],                # type: ignore[arg-type]
        gamma1[None, :, None],                # type: ignore[arg-type]
    )
    peak_contrib = jnp.sum(A[:, :, None] * dens_all, axis=1)  # [n_trace, n_time]
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
