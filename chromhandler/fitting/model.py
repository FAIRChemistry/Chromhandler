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
from typing import TYPE_CHECKING, Any

import arviz
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from chromhandler.fitting.skew_normal import GAMMA1_MAX, density_cp

if TYPE_CHECKING:
    from chromhandler.fitting.prepared_dataset import PreparedDataset
    from chromhandler.fitting.priors import SkewNormalPriors


def marginal_baseline_loglik(
    signal: jnp.ndarray,
    peak_contrib: jnp.ndarray,
    time: jnp.ndarray,
    valid_mask: jnp.ndarray,
    noise: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Flat-prior analytic marginalisation of the per-trace linear baseline.

    Integrates out ``baseline = a + b·t`` (independent per trace) under an
    improper-flat prior. Returns ``(loglik_per_trace, intercept_hat,
    slope_hat)`` where the loglik is the marginal Gaussian log-density of
    the peak-subtracted residual (up to a parameter-free additive constant),
    and the hats are the Rao-Blackwellised conditional-mean baseline in
    ORIGINAL (uncentred) coordinates.

    Float32-safe: uses the direct-residual form (form residuals after
    removing the fitted line, then sum squares) rather than the
    large-minus-large projection identity ``rss - Sr²/n - Str²/Stt``.
    The direct form keeps operands at noise² magnitude and avoids
    catastrophic cancellation when the baseline is large relative to noise.
    """
    w = valid_mask.astype(time.dtype)
    n = jnp.sum(w, axis=1)
    n_safe = jnp.maximum(n, 1.0)
    t_clean = jnp.where(valid_mask, time, 0.0)
    t_mean = jnp.sum(w * t_clean, axis=1) / n_safe
    tc = jnp.where(valid_mask, time - t_mean[:, None], 0.0)
    Stt = jnp.sum(tc * tc, axis=1)
    Stt_safe = jnp.maximum(Stt, 1e-30)

    r = jnp.where(valid_mask, jnp.nan_to_num(signal) - peak_contrib, 0.0)
    a_hat = jnp.sum(r, axis=1) / n_safe            # centred-coord intercept
    b_hat = jnp.sum(tc * r, axis=1) / Stt_safe     # slope

    # Direct-residual form: form the residual AFTER removing the fitted line
    # (magnitude ~noise), then sum squares. Avoids the large-minus-large
    # cancellation of rss - Sr^2/n - Str^2/Stt, so it stays accurate in float32.
    resid = jnp.where(valid_mask, r - (a_hat[:, None] + b_hat[:, None] * tc), 0.0)
    rss_perp = jnp.sum(resid * resid, axis=1)

    sigma2 = noise**2
    dof = jnp.maximum(n - 2.0, 0.0)
    loglik = -0.5 * dof * jnp.log(2.0 * jnp.pi * sigma2) - rss_perp / (2.0 * sigma2)
    loglik = jnp.where(n >= 2.0, loglik, 0.0)

    slope_hat = b_hat
    intercept_hat = a_hat - slope_hat * t_mean
    return loglik, intercept_hat, slope_hat



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


def _latent_block(
    dataset: PreparedDataset,
    priors_list: list[SkewNormalPriors],
    config: ModelConfig,
) -> dict[str, Any]:
    """Sample all latent peak/noise/warp sites and return peak_contrib + noise.

    Shared by ``model`` (which marginalises the baseline) and
    ``predictive_model`` (which draws the baseline from its conditional).
    Registers the same ``numpyro.deterministic`` sites as before so
    downstream summary/plot code is unchanged.
    """
    n_trace = dataset.n_trace
    n_peak = len(priors_list)
    dt_global = float(dataset.dt_global)

    mu_loc = jnp.asarray([p.mu_loc for p in priors_list])
    mu_scale = jnp.asarray([p.mu_scale for p in priors_list])
    mu_raw = numpyro.sample("mu_raw", dist.Normal(jnp.zeros(n_peak), 1.0))
    mu = numpyro.deterministic("mu", mu_loc + mu_scale * mu_raw)

    log_width_loc = jnp.log(jnp.asarray([p.width_loc for p in priors_list]))
    width_log_scale = jnp.asarray([p.width_log_scale for p in priors_list])
    width_raw = numpyro.sample("width_raw", dist.Normal(jnp.zeros(n_peak), 1.0))
    width = numpyro.deterministic(
        "width", jnp.exp(log_width_loc + width_log_scale * width_raw)
    )

    skew_loc = jnp.asarray([p.skew_loc for p in priors_list])
    skew_scale = jnp.asarray([p.skew_scale for p in priors_list])
    skew_max = float(GAMMA1_MAX)
    skew_raw = numpyro.sample("skew_raw", dist.Normal(jnp.zeros(n_peak), 1.0))
    skew = numpyro.deterministic(
        "skew", skew_max * jnp.tanh((skew_loc + skew_scale * skew_raw) / skew_max)
    )

    area_loc = jnp.asarray(np.stack([p.area_loc_per_trace for p in priors_list], axis=1))
    area_scale = jnp.asarray(np.stack([p.area_scale_per_trace for p in priors_list], axis=1))
    area_raw = numpyro.sample("area_raw", dist.Normal(jnp.zeros((n_trace, n_peak)), 1.0))
    area = numpyro.deterministic("area", jax.nn.softplus(area_loc + area_scale * area_raw))

    log_noise_loc = jnp.log(jnp.asarray(dataset.noise_per_trace))
    noise_raw = numpyro.sample("noise_raw", dist.Normal(jnp.zeros(n_trace), 1.0))
    noise = numpyro.deterministic(
        "noise", jnp.exp(log_noise_loc + config.log_noise_scale * noise_raw)
    )

    shift_scale = config.warp_shift_scale_dt_multiplier * dt_global
    time_shift_raw = numpyro.sample("time_shift_raw", dist.Normal(jnp.zeros(n_trace), 1.0))
    _shift = shift_scale * time_shift_raw
    time_shift = numpyro.deterministic("time_shift", _shift - jnp.mean(_shift))

    time_stretch_raw = numpyro.sample("time_stretch_raw", dist.Normal(jnp.zeros(n_trace), 1.0))
    _log_stretch = config.warp_stretch_scale * time_stretch_raw
    time_stretch = numpyro.deterministic(
        "time_stretch", jnp.exp(_log_stretch - jnp.mean(_log_stretch))
    )

    mu_warped = numpyro.deterministic(
        "mu_warped", (mu[None, :] - time_shift[:, None]) / time_stretch[:, None]
    )
    width_warped = numpyro.deterministic(
        "width_warped", width[None, :] / time_stretch[:, None]
    )

    time_arr = jnp.asarray(dataset.time)
    dens_all = density_cp(
        time_arr[:, None, :],                 # type: ignore[arg-type]
        mu_warped[:, :, None],                # type: ignore[arg-type]
        width_warped[:, :, None],             # type: ignore[arg-type]
        skew[None, :, None],                  # type: ignore[arg-type]
    )
    peak_contrib = jnp.sum(area[:, :, None] * dens_all, axis=1)  # [n_trace, n_time]
    return {"peak_contrib": peak_contrib, "noise": noise}


def model(
    dataset: PreparedDataset,
    priors_list: list[SkewNormalPriors],
    config: ModelConfig,
) -> None:
    """Non-centred, unit-scale skew-normal model with analytic baseline marginalisation.

    All ``numpyro.sample`` sites are ``Normal(0, 1)`` (suffix ``_raw``).
    The physical quantities are exposed as ``numpyro.deterministic``
    sites so downstream summary/plotting code can read them unchanged.
    Log-space parameterisations of ``width`` and ``noise`` are internal:
    only the natural-space sites are exposed.

    The per-trace linear baseline ``a + b·t`` is integrated out analytically
    under an improper-flat prior (Rao-Blackwellisation). The marginal
    log-likelihood is injected via ``numpyro.factor``; there is no ``obs``
    sample site. The Rao-Blackwellised conditional-mean baseline is exposed
    as deterministic sites for reporting and posterior predictive use.

    Deterministic sites:
        - ``mu[peak]``              = mu_loc + mu_scale * mu_raw
        - ``width[peak]``           = exp(log(width_loc) + width_log_scale * width_raw)
        - ``skew[peak]``            = GAMMA1_MAX * tanh((skew_loc + skew_scale * skew_raw) / GAMMA1_MAX)
        - ``area[trace, peak]``     = softplus(area_loc + area_scale * area_raw)
        - ``time_shift[trace]``     = shift_scale * time_shift_raw, centred so mean == 0
        - ``time_stretch[trace]``   = exp(stretch_scale * time_stretch_raw), centred so mean(log) == 0
        - ``mu_warped[trace, peak]``    = (mu[peak] - time_shift[trace]) / time_stretch[trace]
        - ``width_warped[trace, peak]`` = width[peak] / time_stretch[trace]
        - ``baseline_intercept[trace]`` = analytic conditional-mean intercept (not sampled)
        - ``baseline_slope[trace]``     = analytic conditional-mean slope (not sampled)
        - ``noise[trace]``          = exp(log(baseline_noise) + log_noise_scale * noise_raw)
    """
    block = _latent_block(dataset, priors_list, config)
    peak_contrib = block["peak_contrib"]
    noise = block["noise"]

    loglik, intercept_hat, slope_hat = marginal_baseline_loglik(
        jnp.asarray(dataset.signal),
        peak_contrib,
        jnp.asarray(dataset.time),
        jnp.asarray(dataset.valid_mask),
        noise,
    )
    # Rao-Blackwellised baseline (conditional mean) exposed for reporting.
    numpyro.deterministic("baseline_intercept", intercept_hat)
    numpyro.deterministic("baseline_slope", slope_hat)
    numpyro.factor("obs_marginal", jnp.sum(loglik))


def predictive_model(
    dataset: PreparedDataset,
    priors_list: list[SkewNormalPriors],
    config: ModelConfig,
) -> None:
    """Generative twin of ``model`` for prior/posterior predictive sampling.

    Samples the same latent sites (so posterior samples substitute cleanly),
    draws the baseline from its conditional given the observed data, then
    samples ``obs``. For prior predictive the conditional is taken against
    the real data (a data-anchored prior predictive) so the band sits at a
    sensible level despite the improper flat baseline prior — viz only.
    """
    block = _latent_block(dataset, priors_list, config)
    peak_contrib = block["peak_contrib"]
    noise = block["noise"]

    time = jnp.asarray(dataset.time)
    valid_mask = jnp.asarray(dataset.valid_mask)
    w = valid_mask.astype(time.dtype)
    n = jnp.maximum(jnp.sum(w, axis=1), 1.0)
    t_clean = jnp.where(valid_mask, time, 0.0)
    t_mean = jnp.sum(w * t_clean, axis=1) / n
    tc = jnp.where(valid_mask, time - t_mean[:, None], 0.0)
    Stt = jnp.maximum(jnp.sum(tc * tc, axis=1), 1e-30)

    r = jnp.where(valid_mask, jnp.nan_to_num(jnp.asarray(dataset.signal)) - peak_contrib, 0.0)
    a_hat = jnp.sum(r, axis=1) / n              # centred intercept
    b_hat = jnp.sum(tc * r, axis=1) / Stt
    n_trace = dataset.n_trace
    eps = numpyro.sample("baseline_raw", dist.Normal(jnp.zeros((n_trace, 2)), 1.0))
    a_c = a_hat + jnp.sqrt(noise**2 / n) * eps[:, 0]
    b_c = b_hat + jnp.sqrt(noise**2 / Stt) * eps[:, 1]
    baseline = a_c[:, None] + b_c[:, None] * tc

    predicted = baseline + peak_contrib
    predicted = jnp.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0)
    with numpyro.handlers.mask(mask=valid_mask):
        numpyro.sample("obs", dist.Normal(predicted, noise[:, None]))


def run_mcmc(
    dataset: PreparedDataset,
    priors_list: list[SkewNormalPriors],
    config: ModelConfig,
) -> arviz.InferenceData:
    """Run NUTS sampling and return an ArviZ InferenceData."""
    kernel = numpyro.infer.NUTS(
        model,
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
