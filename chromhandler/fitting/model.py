"""NumPyro Bayesian model for the skew-normal peak fitter.

Single-mode peaks only at present. Doublet support is a documented
extension — see TODO(doublet) markers throughout this module and the
"Doublet extension hooks" section of the design spec
(``docs/superpowers/specs/2026-05-12-fitter-integration-design.md``).
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

# --- Sample-site name constants (TODO(doublet): populate SAMPLED_RIGHT_* below) ---
SAMPLED_LEFT_SHARED: tuple[str, ...] = ("mu_anchor_left", "log_sigma_left", "gamma1_left")
SAMPLED_LEFT_PER_TRACE: tuple[str, ...] = ("log_A_left",)
SAMPLED_TRACE_NUISANCE: tuple[str, ...] = (
    "trace_shift", "baseline_intercept", "baseline_slope",
)
SAMPLED_RIGHT_SHARED: tuple[str, ...] = ()        # TODO(doublet)
SAMPLED_RIGHT_PER_TRACE: tuple[str, ...] = ()     # TODO(doublet)


def _validate_single_mode_only(priors_list: list[SkewNormalPriors]) -> None:  # type: ignore[reportUnusedFunction]
    """Raise if any peak in priors_list has n_components > 1.

    Hoisted out of model() so the JIT-compiled hot path is clean.
    """
    doublet = [i for i, p in enumerate(priors_list) if p.n_components == 2]
    if doublet:
        raise NotImplementedError(
            f"model.py supports n_components=1 (single) peaks only. "
            f"Doublet peaks at indices {doublet}. Doublet support is a "
            f"documented future extension — see model.py module docstring "
            f"and `# TODO(doublet)` markers."
        )


def _compute_baseline_se(  # type: ignore[reportUnusedFunction]
    dataset: PreparedDataset,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-trace OLS standard errors for the baseline intercept and slope.

    Computed from the residuals of the baseline OLS fit on each trace's
    annotated baseline regions. Returns ``(intercept_se, slope_se)``,
    both shape ``[n_trace]``.

    Used by ``model()`` to set the Normal priors on baseline parameters.
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
            # Fall back to noise std as a wide-but-finite SE.
            intercept_se[tr] = float(dataset.noise_per_trace[tr])
            slope_se[tr] = float(dataset.noise_per_trace[tr])
            continue
        t_b = t[baseline_mask]
        s_b = s[baseline_mask]
        # OLS via lstsq with design matrix [1, t]
        X = np.column_stack([np.ones_like(t_b), t_b])
        beta, *_ = np.linalg.lstsq(X, s_b, rcond=None)
        residuals = s_b - X @ beta
        # Standard OLS covariance
        sigma2 = float(np.sum(residuals**2) / max(t_b.size - 2, 1))  # type: ignore[arg-type]
        try:
            cov = sigma2 * np.linalg.inv(X.T @ X)
            intercept_se[tr] = float(np.sqrt(max(cov[0, 0], 0.0)))
            slope_se[tr] = float(np.sqrt(max(cov[1, 1], 0.0)))
        except np.linalg.LinAlgError:
            intercept_se[tr] = float(dataset.noise_per_trace[tr])
            slope_se[tr] = float(dataset.noise_per_trace[tr])
    return intercept_se, slope_se


def _baseline_contribution(  # type: ignore[reportUnusedFunction]
    time: NDArray[np.float64],
    intercept: NDArray[np.float64],
    slope: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-trace baseline = intercept + slope * t. Shape [n_trace, n_time]."""
    return intercept[:, None] + slope[:, None] * time


def _left_component_contribution(  # type: ignore[reportUnusedFunction]
    time: NDArray[np.float64],
    mu_anchor: NDArray[np.float64],
    trace_shift: NDArray[np.float64],
    log_sigma: NDArray[np.float64],
    gamma1: NDArray[np.float64],
    log_A: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sum of left-component skew-normal densities per (trace, time).

    Args:
        time: [n_trace, n_time]
        mu_anchor: [n_peak]
        trace_shift: [n_trace]
        log_sigma: [n_peak]
        gamma1: [n_peak]
        log_A: [n_trace, n_peak]

    Returns:
        Predicted signal [n_trace, n_time].
    """
    n_trace, n_time = time.shape
    n_peak = mu_anchor.shape[0]
    sigma = np.exp(log_sigma)
    # mu[trace, peak] = mu_anchor[peak] + trace_shift[trace]
    mu = mu_anchor[None, :] + trace_shift[:, None]    # [n_trace, n_peak]
    A = np.exp(log_A)                                  # [n_trace, n_peak]

    out = np.zeros((n_trace, n_time), dtype=np.float64)
    for peak in range(n_peak):
        # density_cp accepts vectorised inputs; here we evaluate per-peak
        # over all (trace, time) at once.
        density = np.asarray(density_cp(
            time,                                      # type: ignore[arg-type]  # [n_trace, n_time]
            mu[:, peak:peak + 1],                      # type: ignore[arg-type]  # [n_trace, 1]
            sigma[peak],
            gamma1[peak],
        ))
        out = out + A[:, peak:peak + 1] * density
    return out


@dataclass(frozen=True)
class ModelConfig:
    """User-facing configuration for the NumPyro fit.

    Tuned defaults for fast development iteration on chromatographic data.
    Override fields directly when constructing for publication-quality runs.
    """

    # --- HMC / NUTS settings ---
    num_warmup: int = 500
    num_samples: int = 500
    num_chains: int = 4
    target_accept_prob: float = 0.9
    max_tree_depth: int = 10
    seed: int = 0

    # --- Model-layer priors (per-trace, not per-peak) ---
    trace_shift_scale_dt_multiplier: float = 5.0
    """drift_scale = N * dt_global. trace_shift ~ Normal(0, drift_scale)."""

    baseline_intercept_se_floor: float = 1.0
    """Minimum SE for the baseline intercept prior (signal units)."""

    baseline_slope_se_floor: float = 0.01
    """Minimum SE for the baseline slope prior (signal units per minute)."""

    # --- Prior predictive ---
    prior_predictive_n_samples: int = 200
    """Number of prior samples used to compute prior predictive band."""


def model(
    dataset: PreparedDataset,
    priors_list: list[SkewNormalPriors],
    config: ModelConfig,
) -> None:
    """NumPyro Bayesian model for the skew-normal peak fitter.

    Single-mode peaks only. ``run_mcmc`` calls
    ``_validate_single_mode_only`` before invoking this function.

    Sample sites (single mode):
        - mu_anchor_left[peak]
        - log_sigma_left[peak]
        - gamma1_left[peak]
        - log_A_left[trace, peak]
        - trace_shift[trace]
        - baseline_intercept[trace]
        - baseline_slope[trace]
        - obs (likelihood, NaN-masked)

    TODO(doublet): when adding doublet support,
        - sample Delta[doublet_peak], log_sigma_right[doublet_peak],
          gamma1_right[doublet_peak], log_A_right[trace, doublet_peak]
        - add right-component contribution to predicted
        - remove the _validate_single_mode_only call from run_mcmc
    """
    n_trace = dataset.n_trace
    n_peak = len(priors_list)
    dt_global = float(dataset.dt_global)

    # === Left-component shared shape priors ===
    mu_loc = jnp.asarray([p.mu_left_loc for p in priors_list])
    mu_scale = jnp.asarray([p.mu_left_scale for p in priors_list])
    mu_low = jnp.asarray([p.mu_left_low for p in priors_list])
    mu_high = jnp.asarray([p.mu_left_high for p in priors_list])
    mu_anchor_left = numpyro.sample(
        "mu_anchor_left",
        dist.TruncatedNormal(loc=mu_loc, scale=mu_scale, low=mu_low, high=mu_high),
    )  # [n_peak]

    log_sigma_loc = jnp.asarray([p.log_sigma_left_loc for p in priors_list])
    log_sigma_scale = jnp.asarray([p.log_sigma_left_scale for p in priors_list])
    log_sigma_low = jnp.asarray([p.log_sigma_left_low for p in priors_list])
    log_sigma_high = jnp.asarray([p.log_sigma_left_high for p in priors_list])
    log_sigma_left = numpyro.sample(
        "log_sigma_left",
        dist.TruncatedNormal(
            loc=log_sigma_loc,
            scale=log_sigma_scale,
            low=log_sigma_low,
            high=log_sigma_high,
        ),
    )

    gamma1_loc = jnp.asarray([p.gamma1_left_loc for p in priors_list])
    gamma1_scale = jnp.asarray([p.gamma1_left_scale for p in priors_list])
    gamma1_bound = 0.99 * float(GAMMA1_MAX)
    gamma1_left = numpyro.sample(
        "gamma1_left",
        dist.TruncatedNormal(
            loc=gamma1_loc,
            scale=gamma1_scale,
            low=-gamma1_bound,
            high=gamma1_bound,
        ),
    )

    # === Per-trace amplitude: Normal(loc_per_trace, scale) ===
    log_A_loc = jnp.asarray(
        np.stack([p.log_A_left_loc_per_trace for p in priors_list], axis=1)
    )  # [n_trace, n_peak]
    log_A_scale = jnp.asarray([p.log_A_left_scale for p in priors_list])  # [n_peak]
    log_A_left = numpyro.sample(
        "log_A_left",
        dist.Normal(loc=log_A_loc, scale=log_A_scale[None, :]),
    )

    # === Per-trace nuisance ===
    drift_scale = config.trace_shift_scale_dt_multiplier * dt_global
    trace_shift = numpyro.sample(
        "trace_shift",
        dist.Normal(loc=jnp.zeros(n_trace), scale=drift_scale),
    )

    intercept_se, slope_se = _compute_baseline_se(dataset)
    intercept_se_eff = np.maximum(intercept_se, config.baseline_intercept_se_floor)
    slope_se_eff = np.maximum(slope_se, config.baseline_slope_se_floor)
    baseline_intercept = numpyro.sample(
        "baseline_intercept",
        dist.Normal(
            loc=jnp.asarray(dataset.baseline_intercept),
            scale=jnp.asarray(intercept_se_eff),
        ),
    )
    baseline_slope = numpyro.sample(
        "baseline_slope",
        dist.Normal(
            loc=jnp.asarray(dataset.baseline_slope),
            scale=jnp.asarray(slope_se_eff),
        ),
    )

    # === DOUBLET EXTENSION HOOK ===
    # TODO(doublet): sample right-component params here:
    #   Delta, log_sigma_right, gamma1_right, log_A_right
    # and add right_contrib = ... to `predicted` below.

    # === Predicted signal (JAX-native; do NOT call _left_component_contribution) ===
    sigma_left = jnp.exp(log_sigma_left)
    A_left = jnp.exp(log_A_left)
    mu = mu_anchor_left[None, :] + trace_shift[:, None]  # [n_trace, n_peak]

    time_arr = jnp.asarray(dataset.time)
    baseline = baseline_intercept[:, None] + baseline_slope[:, None] * time_arr

    left_contrib = jnp.zeros_like(time_arr)
    for peak in range(n_peak):
        dens = density_cp(
            time_arr,
            mu[:, peak : peak + 1],  # type: ignore[arg-type]
            sigma_left[peak],
            gamma1_left[peak],  # type: ignore[arg-type]
        )
        left_contrib = left_contrib + A_left[:, peak : peak + 1] * dens
    # TODO(doublet): + right_contrib
    predicted = baseline + left_contrib

    # === Likelihood (NaN-masked) ===
    # Note: the `"obs"` site is UNCONDITIONED here. The observation is
    # applied externally via `numpyro.handlers.condition` in `run_mcmc`.
    # This keeps `numpyro.infer.Predictive(model, ...)` honest — both
    # prior and posterior predictive sample from the likelihood instead
    # of short-circuiting on the `obs=` argument.
    noise = jnp.asarray(dataset.noise_per_trace)
    with numpyro.handlers.mask(mask=jnp.asarray(dataset.valid_mask)):
        numpyro.sample(
            "obs",
            dist.Normal(predicted, noise[:, None]),
        )


def run_mcmc(
    dataset: PreparedDataset,
    priors_list: list[SkewNormalPriors],
    config: ModelConfig,
) -> arviz.InferenceData:
    """Run NUTS sampling and return an ArviZ InferenceData.

    The observation is applied to the unconditioned `model` via
    ``numpyro.handlers.condition(model, data={"obs": dataset.signal})``
    so that `model` itself stays a pure generative program. This is the
    idiomatic numpyro pattern and makes prior/posterior predictive work
    correctly without per-call workarounds.

    Args:
        dataset: PreparedDataset to fit.
        priors_list: One SkewNormalPriors per peak annotation.
        config: ModelConfig with HMC settings.

    Returns:
        arviz.InferenceData with `posterior` and `observed_data` groups.

    Raises:
        NotImplementedError: If any prior has n_components > 1.
    """
    _validate_single_mode_only(priors_list)

    conditioned_model = numpyro.handlers.condition(
        model, data={"obs": jnp.asarray(dataset.signal)},
    )
    kernel = numpyro.infer.NUTS(
        conditioned_model,
        target_accept_prob=config.target_accept_prob,
        max_tree_depth=config.max_tree_depth,
        init_strategy=numpyro.infer.init_to_median(num_samples=20),
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
