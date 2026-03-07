"""Bi-skew-normal peak model for chromatographic fitting.

Each logical peak is represented by two skew-normal components (main + shoulder).
For non-shoulder peaks, the second component is deterministically disabled in the
likelihood via zero area and identical location to the main component.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import log_ndtr

numpyro.set_host_device_count(8)


_SQRT_2_OVER_PI = jnp.sqrt(2.0 / jnp.pi)
_ALPHA_ABS_MAX = 3.5
_ALPHA_LOC_CLIP_FRAC = 0.98


def log_skew_normal_pdf(
    x: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    alpha: jnp.ndarray,
) -> jnp.ndarray:
    """Compute numerically stable skew-normal log-density."""
    sigma_safe = jnp.maximum(sigma, 1e-6)
    z = (x[..., None, :] - mu[..., :, None]) / sigma_safe[..., :, None]
    log_phi = -0.5 * z**2 - 0.5 * jnp.log(2.0 * jnp.pi)
    return (
        jnp.log(2.0)
        - jnp.log(sigma_safe)[..., :, None]
        + log_phi
        + log_ndtr(alpha[..., :, None] * z)
    )


def skew_normal_pdf(
    x: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    alpha: jnp.ndarray,
) -> jnp.ndarray:
    """Skew-normal density values."""
    return jnp.exp(log_skew_normal_pdf(x, mu, sigma, alpha))


def skew_mixture_area(
    x: jnp.ndarray,
    A: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    alpha: jnp.ndarray,
) -> jnp.ndarray:
    """Area-scaled skew-normal mixture signal."""
    return jnp.sum(skew_normal_pdf(x, mu, sigma, alpha) * A[..., :, None], axis=-2)


def skew_components_area(
    x: jnp.ndarray,
    A: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    alpha: jnp.ndarray,
) -> jnp.ndarray:
    """Area-scaled per-component curves."""
    return skew_normal_pdf(x, mu, sigma, alpha) * A[..., :, None]


def _flatten_peak_components(values: jnp.ndarray) -> jnp.ndarray:
    """Flatten ``[..., n_peak, 2]`` to ``[..., n_peak * 2]``."""
    array = jnp.asarray(values)
    if array.ndim < 2 or int(array.shape[-1]) != 2:
        raise ValueError(f"Expected [..., n_peak, 2], got {array.shape}.")
    return array.reshape(array.shape[:-2] + (array.shape[-2] * array.shape[-1],))


def model(
    x: jnp.ndarray,  # [n_trace, n_time]
    y: jnp.ndarray | None,  # [n_trace, n_time] or None for prior predictive
    mu_lo: jnp.ndarray,  # [n_peak]
    mu_hi: jnp.ndarray,  # [n_peak]
    shoulder_side: jnp.ndarray,  # [n_peak] in {-1, 0, +1}
    shoulder_peak_index: jnp.ndarray,  # [n_shoulder_peak] subset of peak indices
    A_init: jnp.ndarray,  # [n_trace, n_peak, 2]
    mu_center_loc: jnp.ndarray,  # [n_peak]
    mu_center_scale: jnp.ndarray,  # [n_peak]
    separation_low: jnp.ndarray,  # [n_peak]
    separation_high: jnp.ndarray,  # [n_peak]
    sigma_prior_loc: jnp.ndarray,  # [n_peak, 2]
    sigma_prior_scale: jnp.ndarray,  # [n_peak, 2]
    alpha_prior_loc: jnp.ndarray,  # [n_peak, 2]
    alpha_prior_scale: jnp.ndarray,  # [n_peak, 2]
    area_total_loc: jnp.ndarray,  # [n_peak]
    area_split_alpha: jnp.ndarray,  # [n_peak]
    area_split_beta: jnp.ndarray,  # [n_peak]
    baseline_intercept_loc: jnp.ndarray,  # [n_trace]
    baseline_intercept_scale: jnp.ndarray,  # [n_trace]
    baseline_slope_loc: jnp.ndarray,  # [n_trace]
    baseline_slope_scale: jnp.ndarray,  # [n_trace]
    sigma_y_prior_loc: jnp.ndarray,  # [n_trace]
    peak_mask: jnp.ndarray | None = None,  # [n_trace, n_time]
) -> None:
    """Bi-skew-normal peak model with direct per-trace linear baseline priors."""
    n_trace, _ = x.shape
    n_peak = int(mu_lo.shape[0])
    shoulder_side_v = jnp.asarray(shoulder_side, dtype=jnp.float32).reshape(-1)
    shoulder_enabled = shoulder_side_v != 0.0
    shoulder_peak_index = jnp.asarray(shoulder_peak_index, dtype=jnp.int32).reshape(-1)
    n_shoulder_peak = int(shoulder_peak_index.shape[0])

    finite_mask = (
        jnp.ones((n_trace, x.shape[1]), dtype=bool) if y is None else jnp.isfinite(y)
    )
    likelihood_mask = (
        (peak_mask & finite_mask) if peak_mask is not None else finite_mask
    )
    y_obs = None if y is None else jnp.where(finite_mask, y, 0.0)

    # --- sigma [n_peak, 2]: LogNormal prior, shared across traces ---
    sigma_loc_safe = jnp.maximum(sigma_prior_loc, 1e-8)
    tau_sigma = jnp.sqrt(
        jnp.log1p((jnp.maximum(sigma_prior_scale, 0.0) / sigma_loc_safe) ** 2)
    )
    # Sample sigma: main component for all peaks, shoulder only for shoulder peaks.
    # Avoids wasted dimensions and improves posterior geometry for non-shoulder peaks.
    log_sigma_main = numpyro.sample(
        "log_sigma_main", dist.Normal(jnp.log(sigma_loc_safe[:, 0]), tau_sigma[:, 0])
    )  # [n_peak]
    sigma_main = jnp.exp(log_sigma_main)  # [n_peak]

    numpyro.deterministic("sigma_main", sigma_main)

    if n_shoulder_peak > 0:
        log_sigma_shoulder = numpyro.sample(
            "log_sigma_shoulder",
            dist.Normal(
                jnp.log(sigma_loc_safe[shoulder_peak_index, 1]),
                tau_sigma[shoulder_peak_index, 1],
            ),
        )  # [n_shoulder_peak]
        sigma_shoulder = jnp.exp(log_sigma_shoulder)
        numpyro.deterministic("sigma_shoulder", sigma_shoulder)
        sigma_pair = (
            jnp.zeros((n_peak, 2), dtype=x.dtype)
            .at[:, 0]
            .set(sigma_main)
            .at[shoulder_peak_index, 1]
            .set(sigma_shoulder)
        )
    else:
        sigma_pair = jnp.stack(
            [sigma_main, jnp.zeros_like(sigma_main)], axis=-1
        )  # [n_peak, 2]

    sigma_flat = jnp.broadcast_to(
        _flatten_peak_components(sigma_pair[None, :, :]), (n_trace, 2 * n_peak)
    )
    numpyro.deterministic("sigma", sigma_flat)

    # --- alpha [n_peak, 2]: smoothly bounded to (-alpha_max, +alpha_max) ---
    # Sample main component for all peaks, shoulder only for shoulder peaks.
    # Avoids wasted dimensions and improves posterior geometry.
    # We keep the latent geometry unconstrained (Normal), then map through
    # sigmoid + affine. This prevents impossible tails beyond +/- alpha_max
    # without hard truncation.
    alpha_unit_main = jnp.clip(
        alpha_prior_loc[:, 0] / _ALPHA_ABS_MAX,
        -_ALPHA_LOC_CLIP_FRAC,
        _ALPHA_LOC_CLIP_FRAC,
    )
    alpha_prob_main = 0.5 * (alpha_unit_main + 1.0)
    alpha_raw_loc_main = jnp.log(alpha_prob_main) - jnp.log1p(-alpha_prob_main)
    local_jacobian_main = (
        2.0
        * _ALPHA_ABS_MAX
        * jnp.maximum(alpha_prob_main * (1.0 - alpha_prob_main), 1e-3)
    )
    alpha_raw_scale_main = jnp.maximum(
        alpha_prior_scale[:, 0] / local_jacobian_main, 1e-3
    )
    alpha_main = numpyro.sample(
        "alpha_main",
        dist.TransformedDistribution(
            dist.Normal(alpha_raw_loc_main, alpha_raw_scale_main),
            [
                dist.transforms.SigmoidTransform(),
                dist.transforms.AffineTransform(
                    loc=-_ALPHA_ABS_MAX, scale=2.0 * _ALPHA_ABS_MAX
                ),
            ],
        ),
    )  # [n_peak]

    if n_shoulder_peak > 0:
        alpha_unit_shoulder = jnp.clip(
            alpha_prior_loc[shoulder_peak_index, 1] / _ALPHA_ABS_MAX,
            -_ALPHA_LOC_CLIP_FRAC,
            _ALPHA_LOC_CLIP_FRAC,
        )
        alpha_prob_shoulder = 0.5 * (alpha_unit_shoulder + 1.0)
        alpha_raw_loc_shoulder = jnp.log(alpha_prob_shoulder) - jnp.log1p(
            -alpha_prob_shoulder
        )
        local_jacobian_shoulder = (
            2.0
            * _ALPHA_ABS_MAX
            * jnp.maximum(alpha_prob_shoulder * (1.0 - alpha_prob_shoulder), 1e-3)
        )
        alpha_raw_scale_shoulder = jnp.maximum(
            alpha_prior_scale[shoulder_peak_index, 1] / local_jacobian_shoulder, 1e-3
        )
        alpha_shoulder = numpyro.sample(
            "alpha_shoulder",
            dist.TransformedDistribution(
                dist.Normal(alpha_raw_loc_shoulder, alpha_raw_scale_shoulder),
                [
                    dist.transforms.SigmoidTransform(),
                    dist.transforms.AffineTransform(
                        loc=-_ALPHA_ABS_MAX, scale=2.0 * _ALPHA_ABS_MAX
                    ),
                ],
            ),
        )  # [n_shoulder_peak]
        alpha_pair = (
            jnp.zeros((n_peak, 2), dtype=x.dtype)
            .at[:, 0]
            .set(alpha_main)
            .at[shoulder_peak_index, 1]
            .set(alpha_shoulder)
        )
    else:
        alpha_pair = jnp.stack(
            [alpha_main, jnp.zeros_like(alpha_main)], axis=-1
        )  # [n_peak, 2]

    # Store unflattened alpha_pair as deterministic for posterior extraction: [n_draw, n_peak, 2]
    numpyro.deterministic("alpha", alpha_pair)

    # Flatten for model computation: [n_trace, 2*n_peak]
    alpha_flat = jnp.broadcast_to(
        _flatten_peak_components(alpha_pair[None, :, :]), (n_trace, 2 * n_peak)
    )

    # --- mode-center / trace offsets: non-centered parameterization (no hard clipping) ---
    window_span = jnp.maximum(mu_hi - mu_lo, 1e-4)
    # `mu_center` is intentionally sampled as a center-of-modes parameter (shared hierarchical prior).
    mu_center = numpyro.sample(
        "mu_center", dist.Normal(mu_center_loc, jnp.maximum(mu_center_scale, 1e-6))
    )
    # Increased trace scale: 0.5 × mu_center_scale (capped at 0.5 × window) allows substantial
    # per-trace variation (e.g., ±0.01 min in 0.2 min window). Non-centered parameterization
    # (offset from center) improves MCMC efficiency vs direct TruncatedNormal sampling.
    mu_trace_scale = jnp.clip(0.5 * mu_center_scale, 1e-4, 0.5 * window_span)
    mu_trace_offset = numpyro.sample(
        "mu_trace_offset",
        dist.Normal(0.0, jnp.maximum(mu_trace_scale, 1e-6)).expand((n_trace, n_peak)),
    )
    center_mode_trace = mu_center[None, :] + mu_trace_offset  # [n_trace, n_peak]

    sep_low = jnp.maximum(separation_low, 0.0)
    sep_high = jnp.maximum(separation_high, sep_low + 1e-6)
    # Only sample separation for shoulder peaks — non-shoulder separation is
    # deterministically 0.  Sampling on a near-degenerate [0, 1e-6] interval
    # for non-shoulder peaks causes immediate divergences.
    # Use Uniform(low, high) — avoids the boundary-stacking that occurs when a
    # TruncatedNormal is centred far from where the data wants to be.
    if n_shoulder_peak > 0:
        sep_raw = numpyro.sample(
            "separation",
            dist.Uniform(
                low=sep_low[shoulder_peak_index],
                high=sep_high[shoulder_peak_index],
            ),
        )  # [n_shoulder_peak]
        separation = (
            jnp.zeros((n_peak,), dtype=x.dtype).at[shoulder_peak_index].set(sep_raw)
        )
    else:
        separation = jnp.zeros((n_peak,), dtype=x.dtype)

    offset = 0.5 * separation[None, :] * shoulder_side_v[None, :]
    mode_main = center_mode_trace - offset
    mode_shoulder = center_mode_trace + offset
    mode_pair = jnp.stack([mode_main, mode_shoulder], axis=-1)  # [n_trace, n_peak, 2]

    # Note: Hard clipping removed. TruncatedNormal prior on center_mode_trace ensures bounds are
    # respected, allowing smooth gradient flow through MCMC. Separation offsets can exceed initial
    # bounds, which is acceptable since the likelihood will penalize poor fits.
    numpyro.deterministic("mode", _flatten_peak_components(mode_pair))

    # Derive skew-normal location from sampled mode + shape parameters.
    delta_pair = alpha_pair / jnp.sqrt(1.0 + alpha_pair**2)  # [n_peak, 2]
    mode_shift_pair = sigma_pair * delta_pair * _SQRT_2_OVER_PI  # [n_peak, 2]
    mu_pair = mode_pair - mode_shift_pair[None, :, :]
    mu = numpyro.deterministic("mu", _flatten_peak_components(mu_pair))

    # --- Area parameterization: total area + split ---
    # LogNormal prior: median = area_total_loc, CV = 0.3
    # sigma_log = sqrt(log(1 + CV²)) = sqrt(log(1.09)) ≈ 0.294
    # Prevents the "vanishing peak" degeneracy of Uniform(0, high): when A_total→0
    # the peak vanishes from the likelihood, gradients on μ/σ/α collapse to zero,
    # and NUTS gets stuck. LogNormal strongly penalises collapse toward zero.
    area_loc_safe = jnp.maximum(area_total_loc, 1e-8)  # [n_peak]
    _AREA_LOG_SIGMA = jnp.sqrt(jnp.log(1.0 + 0.3**2))  # ≈ 0.294, CV=30%
    A_total = numpyro.sample(
        "A_total",
        dist.LogNormal(jnp.log(area_loc_safe), _AREA_LOG_SIGMA).expand(
            (n_trace, n_peak)
        ),
    )

    if n_shoulder_peak > 0:
        area_split_shoulder = numpyro.sample(
            "area_split_shoulder",
            dist.Uniform(0.0, 1.0).expand((n_trace, n_shoulder_peak)),
        )
        area_split_eff = jnp.ones((n_trace, n_peak), dtype=x.dtype)
        area_split_eff = area_split_eff.at[:, shoulder_peak_index].set(
            area_split_shoulder
        )
    else:
        area_split_eff = jnp.ones((n_trace, n_peak), dtype=x.dtype)
    numpyro.deterministic("area_split", area_split_eff)

    A_main = A_total * area_split_eff
    A_shoulder = A_total * (1.0 - area_split_eff) * shoulder_enabled[None, :]
    A_pair = jnp.stack([A_main, A_shoulder], axis=-1)  # [n_trace, n_peak, 2]
    A = numpyro.deterministic("A", _flatten_peak_components(A_pair))
    numpyro.deterministic("area", jnp.sum(A, axis=-1))  # [n_trace]

    # --- Baseline: per-trace normal priors from OLS estimates/SE ---
    baseline_intercept = numpyro.sample(
        "baseline_intercept",
        dist.Normal(
            baseline_intercept_loc,
            jnp.maximum(baseline_intercept_scale, 1e-6),
        ),
    )
    baseline_slope = numpyro.sample(
        "baseline_slope",
        dist.Normal(
            baseline_slope_loc,
            jnp.maximum(baseline_slope_scale, 1e-8),
        ),
    )
    baseline_curve = numpyro.deterministic(
        "baseline_curve",
        baseline_intercept[:, None] + baseline_slope[:, None] * x,
    )

    # --- Likelihood ---
    peak_signal = skew_mixture_area(x, A, mu, sigma_flat, alpha_flat)
    mu_y = numpyro.deterministic("mu_y", peak_signal + baseline_curve)

    noise_guess = jnp.maximum(sigma_y_prior_loc, 1e-6)
    sigma_y = numpyro.sample(
        "sigma_y", dist.LogNormal(jnp.log(jnp.maximum(noise_guess, 1e-6)), 0.5)
    )
    numpyro.sample(
        "y",
        dist.Normal(mu_y, sigma_y[:, None]).mask(likelihood_mask),
        obs=y_obs,
    )


__all__ = [
    "log_skew_normal_pdf",
    "model",
    "SAMPLED_PARAMETER_NAMES",
    "skew_components_area",
    "skew_mixture_area",
    "skew_normal_pdf",
]


SAMPLED_PARAMETER_NAMES = (
    "log_sigma",
    "alpha",
    "mu_center",
    "mu_trace_offset",
    "separation",
    "A_total",
    "area_split_shoulder",  # sampled Beta; "area_split" is the deterministic wrapper
    "baseline_intercept",
    "baseline_slope",
    "sigma_y",
)
