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
    "sigma_shift", "mu_shift",
    "tau_log_sigma", "tau_gamma1",
    "log_sigma_dev", "gamma1_dev",
    "log_sigma_eff", "gamma1_eff",
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

    # --- Per-trace shape-deviation hyperprior ---
    # tau_log_sigma[peak] ~ HalfNormal(fraction * priors[peak].log_sigma_scale)
    # tau_gamma1[peak]    ~ HalfNormal(fraction * priors[peak].gamma1_scale)
    # Anchors the per-(trace, peak) shape-deviation hyperpriors at a
    # fraction of the within-trace prior uncertainty. fraction=0.5 says
    # "across-trace variation is at most ~half of within-trace prior
    # uncertainty by default; the data can pull tau larger if needed".
    shape_dev_hyperprior_fraction: float = 0.5

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
        - ``tau_log_sigma[peak]``   = log_sigma_anchor * HalfNormal(1)
        - ``tau_gamma1[peak]``      = gamma1_anchor * HalfNormal(1)
        - ``log_sigma_dev[trace, peak]`` = tau_log_sigma * dev_raw,
          centred so sum_traces(log_sigma_dev[:, peak]) == 0
        - ``gamma1_dev[trace, peak]``    = tau_gamma1 * dev_raw,
          same per-peak centring
        - ``log_sigma_eff[trace, peak]`` = log_sigma[peak] + log_sigma_dev
        - ``gamma1_eff[trace, peak]``    = tanh-bounded gamma1 + deviation
        - ``A[trace, peak]``        = softplus(area_loc + area_scale * A_raw)
        - ``sigma_shift``           = dt_global * exp(log_sigma_shift_raw)
        - ``mu_shift[trace, peak]`` = sigma_shift * mu_shift_raw, centred so
          sum_traces(mu_shift[:, peak]) == 0 (breaks anchor-shift degeneracy)
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
    # Register the population-level deterministic site; the value itself is
    # consumed downstream only via gamma1_eff, so we don't keep a local.
    numpyro.deterministic(
        "gamma1", gamma1_max * jnp.tanh(gamma1_unconstrained / gamma1_max),
    )

    # === Per-peak hyperpriors on shape-deviation magnitude ===
    # tau_*[peak] anchored at fraction * priors[peak].<param>_scale.
    # Non-centred via HalfNormal: tau_raw ~ HalfNormal(1) anchored by
    # multiplication, equivalent to tau ~ HalfNormal(anchor).
    log_sigma_anchor = config.shape_dev_hyperprior_fraction * jnp.asarray(
        [p.log_sigma_scale for p in priors_list]
    )  # [n_peak]
    gamma1_anchor = config.shape_dev_hyperprior_fraction * jnp.asarray(
        [p.gamma1_scale for p in priors_list]
    )  # [n_peak]
    tau_log_sigma_raw = numpyro.sample(
        "tau_log_sigma_raw", dist.HalfNormal(jnp.ones(n_peak)),
    )
    tau_log_sigma = numpyro.deterministic(
        "tau_log_sigma", log_sigma_anchor * tau_log_sigma_raw,
    )
    tau_gamma1_raw = numpyro.sample(
        "tau_gamma1_raw", dist.HalfNormal(jnp.ones(n_peak)),
    )
    tau_gamma1 = numpyro.deterministic(
        "tau_gamma1", gamma1_anchor * tau_gamma1_raw,
    )

    # === Per-(trace, peak) shape deviations (non-centred, sum-to-zero) ===
    # Centring per peak breaks the same anchor<->dev degeneracy we broke
    # for mu_shift: the chain cannot trade a constant epsilon between
    # log_sigma[peak] and all log_sigma_dev[:, peak] without changing
    # the likelihood, so the per-peak sum is forced to zero.
    log_sigma_dev_raw = numpyro.sample(
        "log_sigma_dev_raw",
        dist.Normal(jnp.zeros((n_trace, n_peak)), 1.0),
    )
    _lsd = tau_log_sigma * log_sigma_dev_raw  # broadcasts to [n_trace, n_peak]
    log_sigma_dev = numpyro.deterministic(
        "log_sigma_dev", _lsd - jnp.mean(_lsd, axis=0, keepdims=True),
    )
    gamma1_dev_raw = numpyro.sample(
        "gamma1_dev_raw",
        dist.Normal(jnp.zeros((n_trace, n_peak)), 1.0),
    )
    _g1d = tau_gamma1 * gamma1_dev_raw
    gamma1_dev = numpyro.deterministic(
        "gamma1_dev", _g1d - jnp.mean(_g1d, axis=0, keepdims=True),
    )

    # === Effective per-(trace, peak) shape ===
    # log_sigma: deviation added in log-space.
    log_sigma_eff = numpyro.deterministic(
        "log_sigma_eff", log_sigma[None, :] + log_sigma_dev,
    )
    # gamma1: deviation added in UNCONSTRAINED space, then tanh-bound
    # so the per-trace value stays within (-GAMMA1_MAX, GAMMA1_MAX).
    gamma1_unconstrained_eff = gamma1_unconstrained[None, :] + gamma1_dev
    gamma1_eff = numpyro.deterministic(
        "gamma1_eff",
        gamma1_max * jnp.tanh(gamma1_unconstrained_eff / gamma1_max),
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

    # === Per-(trace, peak) retention shift (hierarchical) ===
    # Hyperprior on the shift scale, anchored at dt (sampling resolution).
    # LogNormal(log dt, 1) is weakly informative: 95% prior CI ~ [dt/7, 7*dt].
    # Hierarchical shrinkage: small shifts pull sigma_shift down, which
    # pulls all per-(trace, peak) shifts toward zero. Large data-driven
    # shifts grow sigma_shift to let them through.
    log_sigma_shift_raw = numpyro.sample(
        "log_sigma_shift_raw", dist.Normal(0.0, 1.0),
    )
    sigma_shift = numpyro.deterministic(
        "sigma_shift", dt_global * jnp.exp(log_sigma_shift_raw),
    )
    mu_shift_raw = numpyro.sample(
        "mu_shift_raw", dist.Normal(jnp.zeros((n_trace, n_peak)), 1.0),
    )
    # Apply per-peak sum-to-zero centring to break the
    # mu_anchor[peak] <-> mu_shift[:, peak] translation degeneracy.
    # The constraint is exact and the implied per-shift prior is the
    # conditional Normal(0, sigma_shift) given sum=0, which differs from
    # the unconstrained prior by a factor ~ sqrt(1 - 1/n_trace) ≈ 1 for
    # n_trace >> 1.
    _ms = sigma_shift * mu_shift_raw
    mu_shift = numpyro.deterministic(
        "mu_shift", _ms - jnp.mean(_ms, axis=0, keepdims=True),
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

    # === Predicted signal ===
    # sigma and gamma1 are now per-(trace, peak) effective values.
    sigma_eff = jnp.exp(log_sigma_eff)            # [n_trace, n_peak]
    mu = mu_anchor[None, :] + mu_shift            # [n_trace, n_peak]

    time_arr = jnp.asarray(dataset.time)
    baseline = baseline_intercept[:, None] + baseline_slope[:, None] * time_arr

    peak_contrib = jnp.zeros_like(time_arr)
    for peak in range(n_peak):
        dens = density_cp(
            time_arr,
            mu[:, peak : peak + 1],                       # type: ignore[arg-type]
            sigma_eff[:, peak : peak + 1],                # type: ignore[arg-type]
            gamma1_eff[:, peak : peak + 1],               # type: ignore[arg-type]
        )
        peak_contrib = peak_contrib + A[:, peak : peak + 1] * dens
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
