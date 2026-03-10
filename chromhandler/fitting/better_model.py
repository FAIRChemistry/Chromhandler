"""Mode-aware skew-normal chromatographic peak model.

Supports three peak modes:

- ``single``: one component per logical peak window.
- ``artefact_doublet``: dominant component plus a signed artefact component
  with shared cross-trace shoulder area and a tightly coupled second sigma
  ``sigma_artefact_second``.
- ``free_doublet``: true two-component peak with midpoint-centred retention
  time, per-trace separation, per-trace total area, a nuisance split parameter,
  a tightly coupled second sigma ``sigma_free_second``, and an
  independently sampled bounded second alpha ``alpha_free_second``.

The model samples latent parameters in mode-specific parameterizations, then
exports generic deterministic per-component arrays ``mu``, ``sigma``,
``alpha``, and ``A`` so downstream code can reconstruct components without
knowing which branch produced them.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import log_ndtr

numpyro.set_host_device_count(8)

# Mode → location shift: delta × sqrt(2/π), delta = alpha / sqrt(1 + alpha²)
_SQRT_2_OVER_PI: float = float(jnp.sqrt(2.0 / jnp.pi))
_ALPHA_MAX: float = 2.5  # canonical hard bound via tanh transform
_ALPHA_BOUND_EPS: float = 1e-4
_RAW_ALPHA_SCALE_FLOOR: float = 1e-4

# Area prior: LogNormal sigma for per-trace primary/total areas
_AREA_LOG_SIGMA: float = 0.4
# Shared shoulder area prior: LogNormal sigma (30% CV) — allows slow column drift
# while strongly enforcing the constant-artefact constraint across traces.
_SH_AREA_LOG_SIGMA: float = 0.3


# ---------------------------------------------------------------------------
# Skew-normal math
# ---------------------------------------------------------------------------


def log_skew_normal_pdf(
    x: jnp.ndarray,  # [n_trace, n_time]
    mu: jnp.ndarray,  # [n_trace, n_comp]  — skew-normal location param ξ (NOT mode)
    sigma: jnp.ndarray,  # [n_comp]   — shared across traces
    alpha: jnp.ndarray,  # [n_comp]   — shared across traces
) -> jnp.ndarray:
    """Numerically stable log skew-normal density.

    Parameters ``mu`` must be the skew-normal **location parameter ξ**, not the
    component modes. The model converts sampled component modes internally via
    ``ξ = mode − σ·δ·√(2/π)`` before calling this density.

    Returns
    -------
    jnp.ndarray
        Shape ``[n_trace, n_comp, n_time]``.
    """
    sigma_s = jnp.maximum(sigma, 1e-6)  # [n_comp]
    z = (x[:, None, :] - mu[:, :, None]) / sigma_s[
        None, :, None
    ]  # [n_trace, n_comp, n_time]
    return (
        jnp.log(2.0)
        - jnp.log(sigma_s)[None, :, None]
        - 0.5 * z**2
        - 0.5 * jnp.log(2.0 * jnp.pi)
        + log_ndtr(alpha[None, :, None] * z)
    )


def skew_normal_pdf(
    x: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    alpha: jnp.ndarray,
) -> jnp.ndarray:
    """Skew-normal density. Same shape convention as ``log_skew_normal_pdf``."""
    return jnp.exp(log_skew_normal_pdf(x, mu, sigma, alpha))


def mixture_signal(
    x: jnp.ndarray,  # [n_trace, n_time]
    mu: jnp.ndarray,  # [n_trace, n_comp]
    sigma: jnp.ndarray,  # [n_comp]
    alpha: jnp.ndarray,  # [n_comp]
    A: jnp.ndarray,  # [n_trace, n_comp]
) -> jnp.ndarray:
    """Area-scaled skew-normal mixture, summed over all components.

    Returns
    -------
    jnp.ndarray
        Shape ``[n_trace, n_time]``.
    """
    pdf = skew_normal_pdf(x, mu, sigma, alpha)  # [n_trace, n_comp, n_time]
    return jnp.sum(A[:, :, None] * pdf, axis=1)  # [n_trace, n_time]


def _bounded_alpha_prior_to_raw(
    alpha_loc: jnp.ndarray,
    alpha_scale: jnp.ndarray,
    *,
    alpha_max: float = _ALPHA_MAX,
    bound_eps: float = _ALPHA_BOUND_EPS,
    scale_floor: float = _RAW_ALPHA_SCALE_FLOOR,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map bounded alpha priors to raw-space Normal parameters.

    The model samples ``alpha_raw`` and transforms with
    ``alpha = alpha_max * tanh(alpha_raw)``. This helper converts
    bounded-space ``(alpha_loc, alpha_scale)`` to raw-space Normal
    parameters via the inverse transform and a delta-method scale.
    """
    alpha_loc_arr = jnp.asarray(alpha_loc, dtype=jnp.float32)
    alpha_scale_arr = jnp.asarray(alpha_scale, dtype=jnp.float32)
    alpha_max_safe = max(float(alpha_max), 1e-6)
    loc_clipped = jnp.clip(
        alpha_loc_arr,
        -alpha_max_safe + bound_eps,
        alpha_max_safe - bound_eps,
    )
    raw_loc = jnp.arctanh(loc_clipped / alpha_max_safe)
    derivative = alpha_max_safe * (1.0 - (loc_clipped / alpha_max_safe) ** 2)
    raw_scale = alpha_scale_arr / jnp.maximum(derivative, scale_floor)
    raw_scale = jnp.maximum(raw_scale, scale_floor)
    return raw_loc, raw_scale


def _lognormal_params_from_linear(
    loc: jnp.ndarray,
    scale: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map linear-space location/scale summaries to LogNormal parameters."""
    loc_safe = jnp.maximum(jnp.asarray(loc, dtype=jnp.float32), 1e-6)
    scale_safe = jnp.maximum(jnp.asarray(scale, dtype=jnp.float32), 1e-6)
    log_var = jnp.log1p((scale_safe / loc_safe) ** 2)
    log_scale = jnp.maximum(jnp.sqrt(log_var), 1e-4)
    log_loc = jnp.log(loc_safe) - 0.5 * log_var
    return log_loc, log_scale


# ---------------------------------------------------------------------------
# NumPyro model
# ---------------------------------------------------------------------------

_MODE_SINGLE = 0
_MODE_ARTEFACT_DOUBLET = 1
_MODE_FREE_DOUBLET = 2


def model(
    x: jnp.ndarray,  # [n_trace, n_time]
    y: jnp.ndarray | None,  # [n_trace, n_time]  or  None (prior predictive)
    # --- peak structure ---
    peak_mode_code: jnp.ndarray,  # [n_peak]  int: 0=single, 1=artefact_doublet, 2=free_doublet
    artefact_side: jnp.ndarray,  # [n_peak]  int: -1=left, 0=none, +1=right
    artefact_peak_index: jnp.ndarray,  # [n_artefact]  indices into peaks
    free_peak_index: jnp.ndarray,  # [n_free]  indices into peaks
    nonfree_peak_index: jnp.ndarray,  # [n_nonfree] indices into peaks
    # --- peak priors (from geometric_priors_to_arrays) ---
    mu_center_loc: jnp.ndarray,  # [n_peak]
    mu_center_scale: jnp.ndarray,  # [n_peak]
    sigma_loc: jnp.ndarray,  # [n_peak]  FWHM-derived sigma prior centres
    sigma_scale: jnp.ndarray,  # [n_peak]  FWHM-derived sigma prior scales
    alpha_loc: jnp.ndarray,  # [n_peak]  FWHM-derived alpha prior centres
    alpha_scale: jnp.ndarray,  # [n_peak]  FWHM-derived alpha prior scales
    main_area_per_trace: jnp.ndarray,  # [n_trace, n_peak]  per-trace Gaussian primary-component area
    total_area_per_trace: jnp.ndarray,  # [n_trace, n_peak]  per-trace total free-doublet area
    artefact_shoulder_area_prior: jnp.ndarray,  # [n_artefact]  shared artefact shoulder area prior centres
    # --- baseline priors (from estimate_baseline) ---
    baseline_intercept_loc: jnp.ndarray,  # [n_trace]
    baseline_intercept_scale: jnp.ndarray,  # [n_trace]
    baseline_slope_loc: jnp.ndarray,  # [n_trace]
    baseline_slope_scale: jnp.ndarray,  # [n_trace]
    # --- noise prior ---
    sigma_y_prior_loc: jnp.ndarray,  # [n_trace]
) -> None:
    """Bayesian skew-normal peak model supporting single and doublet modes."""
    n_trace, _ = x.shape
    n_peak = int(mu_center_loc.shape[0])
    n_artefact = int(artefact_peak_index.shape[0])
    n_free = int(free_peak_index.shape[0])
    n_nonfree = int(nonfree_peak_index.shape[0])
    n_comp = 2 * n_peak  # main + shoulder slot for every peak

    mode_code = jnp.asarray(peak_mode_code, dtype=jnp.int32)
    artefact_side_v = jnp.asarray(artefact_side, dtype=jnp.float32)
    artefact_idx = jnp.asarray(artefact_peak_index, dtype=jnp.int32)
    free_idx = jnp.asarray(free_peak_index, dtype=jnp.int32)
    nonfree_idx = jnp.asarray(nonfree_peak_index, dtype=jnp.int32)
    free_mask = mode_code == _MODE_FREE_DOUBLET

    # ------------------------------------------------------------------ sigma
    sigma_loc_safe = jnp.maximum(jnp.asarray(sigma_loc, dtype=jnp.float32), 1e-6)
    sigma_scale_safe = jnp.maximum(jnp.asarray(sigma_scale, dtype=jnp.float32), 1e-6)
    sigma_prior_loc = jnp.where(free_mask, 0.5 * sigma_loc_safe, sigma_loc_safe)
    sigma_prior_scale = jnp.where(free_mask, 0.5 * sigma_scale_safe, sigma_scale_safe)
    log_sigma_loc, log_sigma_scale = _lognormal_params_from_linear(
        sigma_prior_loc,
        sigma_prior_scale,
    )
    log_sigma_peak = numpyro.sample(
        "log_sigma_peak", dist.Normal(log_sigma_loc, log_sigma_scale)
    )  # [n_peak]
    sigma_peak = numpyro.deterministic("sigma_peak", jnp.exp(log_sigma_peak))

    sigma_component = jnp.stack([sigma_peak, sigma_peak], axis=-1)  # [n_peak, 2]
    if n_artefact > 0:
        log_sigma_artefact_second = numpyro.sample(
            "log_sigma_artefact_second",
            dist.Normal(log_sigma_peak[artefact_idx], 0.15),
        )
        sigma_artefact_second = numpyro.deterministic(
            "sigma_artefact_second",
            jnp.exp(log_sigma_artefact_second),
        )
        sigma_component = sigma_component.at[artefact_idx, 1].set(sigma_artefact_second)
    if n_free > 0:
        log_sigma_free_second = numpyro.sample(
            "log_sigma_free_second",
            dist.Normal(log_sigma_peak[free_idx], 0.05),
        )
        sigma_free_second = numpyro.deterministic(
            "sigma_free_second",
            jnp.exp(log_sigma_free_second),
        )
        sigma_component = sigma_component.at[free_idx, 1].set(sigma_free_second)
    sigma_flat = sigma_component.reshape(n_comp)
    numpyro.deterministic("sigma", sigma_flat)

    # ------------------------------------------------------------------ alpha
    alpha_raw_loc, alpha_raw_scale = _bounded_alpha_prior_to_raw(
        alpha_loc,
        jnp.maximum(jnp.asarray(alpha_scale, dtype=jnp.float32), 1e-6),
    )
    alpha_raw_peak = numpyro.sample(
        "alpha_raw_peak", dist.Normal(alpha_raw_loc, alpha_raw_scale)
    )  # [n_peak]
    alpha_peak = numpyro.deterministic(
        "alpha_peak", _ALPHA_MAX * jnp.tanh(alpha_raw_peak)
    )  # [n_peak]
    alpha_component = jnp.stack([alpha_peak, alpha_peak], axis=-1)
    if n_free > 0:
        alpha_raw_free_second = numpyro.sample(
            "alpha_raw_free_second",
            dist.Normal(alpha_raw_loc[free_idx], alpha_raw_scale[free_idx]),
        )
        alpha_free_second = numpyro.deterministic(
            "alpha_free_second",
            _ALPHA_MAX * jnp.tanh(alpha_raw_free_second),
        )
        alpha_component = alpha_component.at[free_idx, 1].set(alpha_free_second)
    alpha_flat = alpha_component.reshape(n_comp)
    numpyro.deterministic("alpha", alpha_flat)

    # ------------------------------------------------------------------ center
    mu_scale_safe = jnp.maximum(jnp.asarray(mu_center_scale), 1e-6)
    center_per_trace = numpyro.sample(
        "center_per_trace",
        dist.Normal(mu_center_loc, mu_scale_safe).expand([n_trace, n_peak]),
    )  # [n_trace, n_peak]

    # ------------------------------------------------------------------ component modes
    mode_component = jnp.stack([center_per_trace, center_per_trace], axis=-1)
    if n_artefact > 0:
        artefact_sep_loc = jnp.maximum(2.0 * sigma_loc_safe[artefact_idx], 1e-6)
        separation_artefact = numpyro.sample(
            "separation_artefact",
            dist.LogNormal(jnp.log(artefact_sep_loc), 0.05),
        )
        mode_component = mode_component.at[:, artefact_idx, 1].set(
            center_per_trace[:, artefact_idx]
            + separation_artefact[None, :] * artefact_side_v[artefact_idx][None, :]
        )
    if n_free > 0:
        u_sep_free = numpyro.sample(
            "u_sep_free",
            dist.Beta(2.0, 2.0).expand([n_trace, n_free]),
        )
        separation_free = numpyro.deterministic(
            "separation_free",
            3.0 * sigma_loc_safe[free_idx] * u_sep_free,
        )
        free_center = center_per_trace[:, free_idx]
        half_sep = 0.5 * separation_free
        mode_component = mode_component.at[:, free_idx, 0].set(free_center - half_sep)
        mode_component = mode_component.at[:, free_idx, 1].set(free_center + half_sep)

    # Convert mode → skew-normal location parameter ξ:
    #   ξ = mode − σ·δ·√(2/π),   δ = α / √(1 + α²)
    delta_component = alpha_component / jnp.sqrt(1.0 + alpha_component**2)
    mode_shift = sigma_component * delta_component * _SQRT_2_OVER_PI
    mu_component = mode_component - mode_shift[None, :, :]
    mu_flat = mu_component.reshape(n_trace, n_comp)
    numpyro.deterministic("mu", mu_flat)

    # ------------------------------------------------------------------ area
    area_component = jnp.zeros((n_trace, n_peak, 2), dtype=x.dtype)
    area_total = jnp.zeros((n_trace, n_peak), dtype=x.dtype)

    if n_nonfree > 0:
        primary_area_safe = jnp.maximum(
            jnp.asarray(main_area_per_trace, dtype=jnp.float32)[:, nonfree_idx],
            1e-8,
        )
        A_primary = numpyro.sample(
            "A_primary",
            dist.LogNormal(jnp.log(primary_area_safe), _AREA_LOG_SIGMA),
        )
        area_component = area_component.at[:, nonfree_idx, 0].set(A_primary)
        area_total = area_total.at[:, nonfree_idx].set(A_primary)

    if n_artefact > 0:
        artefact_shoulder_area_safe = jnp.maximum(
            jnp.asarray(artefact_shoulder_area_prior, dtype=jnp.float32),
            1e-8,
        )
        A_artefact_shared = numpyro.sample(
            "A_artefact_shared",
            dist.LogNormal(jnp.log(artefact_shoulder_area_safe), _SH_AREA_LOG_SIGMA),
        )
        A_artefact = jnp.broadcast_to(
            A_artefact_shared[None, :],
            (n_trace, n_artefact),
        )
        area_component = area_component.at[:, artefact_idx, 1].set(A_artefact)
        area_total = area_total.at[:, artefact_idx].add(A_artefact)

    if n_free > 0:
        total_area_safe = jnp.maximum(
            jnp.asarray(total_area_per_trace, dtype=jnp.float32)[:, free_idx],
            1e-8,
        )
        A_total_free = numpyro.sample(
            "A_total_free",
            dist.LogNormal(jnp.log(total_area_safe), _AREA_LOG_SIGMA),
        )
        w_free = numpyro.sample(
            "w_free",
            dist.Beta(2.0, 2.0).expand((n_trace, n_free)),
        )
        A_free_1 = A_total_free * w_free
        A_free_2 = A_total_free * (1.0 - w_free)
        area_component = area_component.at[:, free_idx, 0].set(A_free_1)
        area_component = area_component.at[:, free_idx, 1].set(A_free_2)
        area_total = area_total.at[:, free_idx].set(A_total_free)

    numpyro.deterministic("A_total", area_total)
    A_flat = area_component.reshape(n_trace, n_comp)
    numpyro.deterministic("A", A_flat)

    # ------------------------------------------------------------------ baseline
    baseline_intercept = numpyro.sample(
        "baseline_intercept",
        dist.Normal(
            baseline_intercept_loc, jnp.maximum(baseline_intercept_scale, 1e-6)
        ),
    )  # [n_trace]
    baseline_slope = numpyro.sample(
        "baseline_slope",
        dist.Normal(baseline_slope_loc, jnp.maximum(baseline_slope_scale, 1e-8)),
    )  # [n_trace]
    baseline = baseline_intercept[:, None] + baseline_slope[:, None] * x
    numpyro.deterministic("baseline_curve", baseline)

    # ------------------------------------------------------------------ likelihood
    mu_y = numpyro.deterministic(
        "mu_y", mixture_signal(x, mu_flat, sigma_flat, alpha_flat, A_flat) + baseline
    )
    sigma_y = numpyro.sample(
        "sigma_y",
        dist.LogNormal(jnp.log(jnp.maximum(sigma_y_prior_loc, 1e-6)), 0.5),
    )  # [n_trace]

    if y is not None:
        finite_mask = jnp.isfinite(y)
        numpyro.sample(
            "y",
            dist.Normal(mu_y, sigma_y[:, None]).mask(finite_mask),
            obs=jnp.where(finite_mask, y, 0.0),
        )


# ---------------------------------------------------------------------------
# Summary parameter names (for ArviZ / posterior extraction)
# ---------------------------------------------------------------------------

SUMMARY_PARAMETER_NAMES = (
    "sigma_peak",
    "sigma_artefact_second",
    "sigma_free_second",
    "alpha_peak",
    "alpha_free_second",
    "center_per_trace",
    "A_primary",
    "A_artefact_shared",
    "A_total_free",
    "separation_artefact",
    "separation_free",
    "baseline_intercept",
    "baseline_slope",
    "sigma_y",
)
