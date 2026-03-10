"""Mode-aware skew-normal chromatographic peak model.

Supports three peak modes:

- ``single``: one component per logical peak window.
- ``artefact_doublet``: dominant component plus a signed artefact component.
- ``free_doublet``: true two-component peak with midpoint-centred retention
  time, per-trace separation, per-trace total area, and a free area split.

The model samples a small set of mode-specific primitive latents and assembles
one canonical deterministic left/right state for every logical peak:

- ``apex_l`` / ``apex_r``
- ``separation``
- ``xi_l`` / ``xi_r``
- ``sigma_l`` / ``sigma_r``
- ``alpha_l`` / ``alpha_r``
- ``area_l`` / ``area_r`` / ``area_total``

Left/right ordering is always by retention time, not by dominant-vs-secondary
component role.
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
    xi: jnp.ndarray,  # [n_trace, n_comp]  — skew-normal location param ξ (NOT mode)
    sigma: jnp.ndarray,  # [n_trace, n_comp]
    alpha: jnp.ndarray,  # [n_trace, n_comp]
) -> jnp.ndarray:
    """Numerically stable log skew-normal density.

    Parameters ``xi`` must be the skew-normal **location parameter ξ**, not the
    component modes. The model converts sampled component modes internally via
    ``ξ = mode − σ·δ·√(2/π)`` before calling this density.

    Returns
    -------
    jnp.ndarray
        Shape ``[n_trace, n_comp, n_time]``.
    """
    sigma_s = jnp.maximum(sigma, 1e-6)  # [n_trace, n_comp]
    z = (x[:, None, :] - xi[:, :, None]) / sigma_s[:, :, None]
    return (
        jnp.log(2.0)
        - jnp.log(sigma_s)[:, :, None]
        - 0.5 * z**2
        - 0.5 * jnp.log(2.0 * jnp.pi)
        + log_ndtr(alpha[:, :, None] * z)
    )


def skew_normal_pdf(
    x: jnp.ndarray,
    xi: jnp.ndarray,
    sigma: jnp.ndarray,
    alpha: jnp.ndarray,
) -> jnp.ndarray:
    """Skew-normal density. Same shape convention as ``log_skew_normal_pdf``."""
    return jnp.exp(log_skew_normal_pdf(x, xi, sigma, alpha))


def mixture_signal(
    x: jnp.ndarray,  # [n_trace, n_time]
    xi: jnp.ndarray,  # [n_trace, n_comp]
    sigma: jnp.ndarray,  # [n_trace, n_comp]
    alpha: jnp.ndarray,  # [n_trace, n_comp]
    area: jnp.ndarray,  # [n_trace, n_comp]
) -> jnp.ndarray:
    """Area-scaled skew-normal mixture, summed over all components.

    Returns
    -------
    jnp.ndarray
        Shape ``[n_trace, n_time]``.
    """
    pdf = skew_normal_pdf(x, xi, sigma, alpha)  # [n_trace, n_comp, n_time]
    return jnp.sum(area[:, :, None] * pdf, axis=1)  # [n_trace, n_time]


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


def _broadcast_peak_to_traces(
    values: jnp.ndarray,
    n_trace: int,
) -> jnp.ndarray:
    """Broadcast a peak-level vector to a dense per-trace matrix."""
    values_arr = jnp.asarray(values, dtype=jnp.float32).reshape(1, -1)
    return jnp.broadcast_to(values_arr, (n_trace, values_arr.shape[1]))


def _apex_to_xi(
    apex: jnp.ndarray,
    sigma: jnp.ndarray,
    alpha: jnp.ndarray,
) -> jnp.ndarray:
    """Convert skew-normal mode locations to the location parameter ``xi``."""
    delta = alpha / jnp.sqrt(1.0 + alpha**2)
    return apex - sigma * delta * _SQRT_2_OVER_PI


def _stack_left_right(
    left: jnp.ndarray,
    right: jnp.ndarray,
) -> jnp.ndarray:
    """Flatten left/right peak matrices to the mixture component axis."""
    return jnp.stack([left, right], axis=-1).reshape(left.shape[0], -1)


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
    apex_anchor_loc: jnp.ndarray,  # [n_peak]
    apex_anchor_scale: jnp.ndarray,  # [n_peak]
    sigma_loc: jnp.ndarray,  # [n_peak]  FWHM-derived sigma prior centres
    sigma_scale: jnp.ndarray,  # [n_peak]  FWHM-derived sigma prior scales
    alpha_loc: jnp.ndarray,  # [n_peak]  FWHM-derived alpha prior centres
    alpha_scale: jnp.ndarray,  # [n_peak]  FWHM-derived alpha prior scales
    dominant_area_loc_per_trace: jnp.ndarray,  # [n_trace, n_peak]  per-trace dominant-component area prior
    area_total_loc_per_trace: jnp.ndarray,  # [n_trace, n_peak]  per-trace total free-doublet area prior
    artefact_area_loc_shared: jnp.ndarray,  # [n_artefact]  shared artefact area prior centres
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
    n_peak = int(apex_anchor_loc.shape[0])
    n_artefact = int(artefact_peak_index.shape[0])
    n_free = int(free_peak_index.shape[0])
    n_nonfree = int(nonfree_peak_index.shape[0])
    n_comp = 2 * n_peak

    mode_code = jnp.asarray(peak_mode_code, dtype=jnp.int32)
    artefact_side_v = jnp.asarray(artefact_side, dtype=jnp.float32)
    artefact_idx = jnp.asarray(artefact_peak_index, dtype=jnp.int32)
    free_idx = jnp.asarray(free_peak_index, dtype=jnp.int32)
    nonfree_idx = jnp.asarray(nonfree_peak_index, dtype=jnp.int32)
    free_mask = mode_code == _MODE_FREE_DOUBLET
    nonfree_position = jnp.cumsum((mode_code != _MODE_FREE_DOUBLET).astype(jnp.int32)) - 1

    # ------------------------------------------------------------------ primitive shape latents
    sigma_loc_safe = jnp.maximum(jnp.asarray(sigma_loc, dtype=jnp.float32), 1e-6)
    sigma_scale_safe = jnp.maximum(jnp.asarray(sigma_scale, dtype=jnp.float32), 1e-6)
    sigma_prior_loc = jnp.where(free_mask, 0.5 * sigma_loc_safe, sigma_loc_safe)
    sigma_prior_scale = jnp.where(free_mask, 0.5 * sigma_scale_safe, sigma_scale_safe)
    log_sigma_loc, log_sigma_scale = _lognormal_params_from_linear(
        sigma_prior_loc,
        sigma_prior_scale,
    )
    log_sigma_base = numpyro.sample(
        "log_sigma_base", dist.Normal(log_sigma_loc, log_sigma_scale)
    )  # [n_peak]
    sigma_base = numpyro.deterministic("sigma_base", jnp.exp(log_sigma_base))
    if n_artefact > 0:
        log_sigma_r_artefact = numpyro.sample(
            "log_sigma_r_artefact",
            dist.Normal(log_sigma_base[artefact_idx], 0.15),
        )
        sigma_r_artefact = numpyro.deterministic(
            "sigma_r_artefact",
            jnp.exp(log_sigma_r_artefact),
        )
    if n_free > 0:
        log_sigma_r_free = numpyro.sample(
            "log_sigma_r_free",
            dist.Normal(log_sigma_base[free_idx], 0.05),
        )
        sigma_r_free = numpyro.deterministic(
            "sigma_r_free",
            jnp.exp(log_sigma_r_free),
        )

    # ------------------------------------------------------------------ primitive skew latents
    alpha_raw_loc, alpha_raw_scale = _bounded_alpha_prior_to_raw(
        alpha_loc,
        jnp.maximum(jnp.asarray(alpha_scale, dtype=jnp.float32), 1e-6),
    )
    alpha_raw_base = numpyro.sample(
        "alpha_raw_base", dist.Normal(alpha_raw_loc, alpha_raw_scale)
    )  # [n_peak]
    alpha_base = numpyro.deterministic(
        "alpha_base", _ALPHA_MAX * jnp.tanh(alpha_raw_base)
    )  # [n_peak]
    if n_free > 0:
        alpha_raw_r_free = numpyro.sample(
            "alpha_raw_r_free",
            dist.Normal(alpha_raw_loc[free_idx], alpha_raw_scale[free_idx]),
        )
        alpha_r_free = numpyro.deterministic(
            "alpha_r_free",
            _ALPHA_MAX * jnp.tanh(alpha_raw_r_free),
        )

    # ------------------------------------------------------------------ primitive position / area latents
    apex_anchor_scale_safe = jnp.maximum(jnp.asarray(apex_anchor_scale), 1e-6)
    if n_nonfree > 0:
        apex_dominant_per_trace = numpyro.sample(
            "apex_dominant_per_trace",
            dist.Normal(
                apex_anchor_loc[nonfree_idx],
                apex_anchor_scale_safe[nonfree_idx],
            ).expand([n_trace, n_nonfree]),
        )
        dominant_area_safe = jnp.maximum(
            jnp.asarray(dominant_area_loc_per_trace, dtype=jnp.float32)[:, nonfree_idx],
            1e-8,
        )
        area_dominant = numpyro.sample(
            "area_dominant",
            dist.LogNormal(jnp.log(dominant_area_safe), _AREA_LOG_SIGMA),
        )
    if n_artefact > 0:
        artefact_sep_loc = jnp.maximum(2.0 * sigma_loc_safe[artefact_idx], 1e-6)
        separation_artefact = numpyro.sample(
            "separation_artefact",
            dist.LogNormal(jnp.log(artefact_sep_loc), 0.05),
        )
        artefact_area_safe = jnp.maximum(
            jnp.asarray(artefact_area_loc_shared, dtype=jnp.float32),
            1e-8,
        )
        area_artefact_shared = numpyro.sample(
            "area_artefact_shared",
            dist.LogNormal(jnp.log(artefact_area_safe), _SH_AREA_LOG_SIGMA),
        )
    if n_free > 0:
        apex_center_per_trace = numpyro.sample(
            "apex_center_per_trace",
            dist.Normal(
                apex_anchor_loc[free_idx],
                apex_anchor_scale_safe[free_idx],
            ).expand([n_trace, n_free]),
        )
        u_separation_free = numpyro.sample(
            "u_separation_free",
            dist.Beta(2.0, 2.0).expand([n_trace, n_free]),
        )
        area_total_safe = jnp.maximum(
            jnp.asarray(area_total_loc_per_trace, dtype=jnp.float32)[:, free_idx],
            1e-8,
        )
        area_total_free = numpyro.sample(
            "area_total_free",
            dist.LogNormal(jnp.log(area_total_safe), _AREA_LOG_SIGMA),
        )
        area_frac_left_free = numpyro.sample(
            "area_frac_left_free",
            dist.Beta(2.0, 2.0).expand((n_trace, n_free)),
        )

    # ------------------------------------------------------------------ canonical left/right assembly
    apex_l = jnp.zeros((n_trace, n_peak), dtype=jnp.float32)
    apex_r = jnp.zeros((n_trace, n_peak), dtype=jnp.float32)
    sigma_l = jnp.zeros((n_trace, n_peak), dtype=jnp.float32)
    sigma_r = jnp.zeros((n_trace, n_peak), dtype=jnp.float32)
    alpha_l = jnp.zeros((n_trace, n_peak), dtype=jnp.float32)
    alpha_r = jnp.zeros((n_trace, n_peak), dtype=jnp.float32)
    area_l = jnp.zeros((n_trace, n_peak), dtype=jnp.float32)
    area_r = jnp.zeros((n_trace, n_peak), dtype=jnp.float32)
    separation = jnp.zeros((n_trace, n_peak), dtype=jnp.float32)

    if n_nonfree > 0:
        sigma_nonfree = _broadcast_peak_to_traces(sigma_base[nonfree_idx], n_trace)
        alpha_nonfree = _broadcast_peak_to_traces(alpha_base[nonfree_idx], n_trace)
        apex_l = apex_l.at[:, nonfree_idx].set(apex_dominant_per_trace)
        apex_r = apex_r.at[:, nonfree_idx].set(apex_dominant_per_trace)
        sigma_l = sigma_l.at[:, nonfree_idx].set(sigma_nonfree)
        sigma_r = sigma_r.at[:, nonfree_idx].set(sigma_nonfree)
        alpha_l = alpha_l.at[:, nonfree_idx].set(alpha_nonfree)
        alpha_r = alpha_r.at[:, nonfree_idx].set(alpha_nonfree)
        area_l = area_l.at[:, nonfree_idx].set(area_dominant)

    if n_artefact > 0:
        artefact_nonfree_idx = nonfree_position[artefact_idx]
        apex_dom_art = apex_dominant_per_trace[:, artefact_nonfree_idx]
        sigma_dom_art = _broadcast_peak_to_traces(sigma_base[artefact_idx], n_trace)
        sigma_art = _broadcast_peak_to_traces(sigma_r_artefact, n_trace)
        alpha_dom_art = _broadcast_peak_to_traces(alpha_base[artefact_idx], n_trace)
        area_dom_art = area_dominant[:, artefact_nonfree_idx]
        area_art = _broadcast_peak_to_traces(area_artefact_shared, n_trace)
        separation_art = _broadcast_peak_to_traces(separation_artefact, n_trace)
        artefact_left = artefact_side_v[artefact_idx] < 0.0

        apex_l_art = jnp.where(
            artefact_left[None, :],
            apex_dom_art - separation_art,
            apex_dom_art,
        )
        apex_r_art = jnp.where(
            artefact_left[None, :],
            apex_dom_art,
            apex_dom_art + separation_art,
        )
        sigma_l_art = jnp.where(artefact_left[None, :], sigma_art, sigma_dom_art)
        sigma_r_art = jnp.where(artefact_left[None, :], sigma_dom_art, sigma_art)
        area_l_art = jnp.where(artefact_left[None, :], area_art, area_dom_art)
        area_r_art = jnp.where(artefact_left[None, :], area_dom_art, area_art)

        apex_l = apex_l.at[:, artefact_idx].set(apex_l_art)
        apex_r = apex_r.at[:, artefact_idx].set(apex_r_art)
        sigma_l = sigma_l.at[:, artefact_idx].set(sigma_l_art)
        sigma_r = sigma_r.at[:, artefact_idx].set(sigma_r_art)
        alpha_l = alpha_l.at[:, artefact_idx].set(alpha_dom_art)
        alpha_r = alpha_r.at[:, artefact_idx].set(alpha_dom_art)
        area_l = area_l.at[:, artefact_idx].set(area_l_art)
        area_r = area_r.at[:, artefact_idx].set(area_r_art)
        separation = separation.at[:, artefact_idx].set(separation_art)

    if n_free > 0:
        separation_free = 3.0 * sigma_loc_safe[free_idx][None, :] * u_separation_free
        sigma_l_free = _broadcast_peak_to_traces(sigma_base[free_idx], n_trace)
        sigma_r_free_b = _broadcast_peak_to_traces(sigma_r_free, n_trace)
        alpha_l_free = _broadcast_peak_to_traces(alpha_base[free_idx], n_trace)
        alpha_r_free_b = _broadcast_peak_to_traces(alpha_r_free, n_trace)
        apex_l_free = apex_center_per_trace - 0.5 * separation_free
        apex_r_free = apex_center_per_trace + 0.5 * separation_free
        area_l_free = area_total_free * area_frac_left_free
        area_r_free = area_total_free * (1.0 - area_frac_left_free)

        apex_l = apex_l.at[:, free_idx].set(apex_l_free)
        apex_r = apex_r.at[:, free_idx].set(apex_r_free)
        sigma_l = sigma_l.at[:, free_idx].set(sigma_l_free)
        sigma_r = sigma_r.at[:, free_idx].set(sigma_r_free_b)
        alpha_l = alpha_l.at[:, free_idx].set(alpha_l_free)
        alpha_r = alpha_r.at[:, free_idx].set(alpha_r_free_b)
        area_l = area_l.at[:, free_idx].set(area_l_free)
        area_r = area_r.at[:, free_idx].set(area_r_free)
        separation = separation.at[:, free_idx].set(separation_free)

    area_total = numpyro.deterministic("area_total", area_l + area_r)
    numpyro.deterministic("apex_l", apex_l)
    numpyro.deterministic("apex_r", apex_r)
    numpyro.deterministic("separation", separation)
    numpyro.deterministic("sigma_l", sigma_l)
    numpyro.deterministic("sigma_r", sigma_r)
    numpyro.deterministic("alpha_l", alpha_l)
    numpyro.deterministic("alpha_r", alpha_r)
    numpyro.deterministic("area_l", area_l)
    numpyro.deterministic("area_r", area_r)
    xi_l = numpyro.deterministic("xi_l", _apex_to_xi(apex_l, sigma_l, alpha_l))
    xi_r = numpyro.deterministic("xi_r", _apex_to_xi(apex_r, sigma_r, alpha_r))

    xi_flat = _stack_left_right(xi_l, xi_r)
    sigma_flat = _stack_left_right(sigma_l, sigma_r)
    alpha_flat = _stack_left_right(alpha_l, alpha_r)
    area_flat = _stack_left_right(area_l, area_r)

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
        "mu_y", mixture_signal(x, xi_flat, sigma_flat, alpha_flat, area_flat) + baseline
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
    "sigma_base",
    "sigma_r_artefact",
    "sigma_r_free",
    "alpha_base",
    "alpha_r_free",
    "apex_dominant_per_trace",
    "apex_center_per_trace",
    "area_dominant",
    "area_artefact_shared",
    "area_total_free",
    "separation",
    "area_total",
    "baseline_intercept",
    "baseline_slope",
    "sigma_y",
)
