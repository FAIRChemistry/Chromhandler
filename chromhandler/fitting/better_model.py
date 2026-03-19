"""Mode-aware skew-normal chromatographic peak model.

Supports three peak modes:

- ``single``: one component per logical peak window.
- ``artefact_doublet``: dominant component plus a signed artefact component.
- ``free_doublet``: true two-component peak with a shared apex-derived
  reference location, hierarchically pooled separation, per-trace total area,
  and a free area split.

The model samples a small set of mode-specific primitive latents and assembles
one canonical deterministic left/right state for every logical peak:

- ``trace_shift`` / ``apex_residual`` / ``apex``
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
from jax.scipy.special import expit, log_ndtr

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

# Free-doublet separation: sigmoid-bounded hierarchy
_FREE_SEP_MIN_SIGMA_MULT: float = 0.5  # sep_min = mult × sigma_loc
_FREE_SEP_TYPICAL_SIGMA_MULT: float = 1.5  # target typical = mult × sigma_loc
_FREE_SEP_MAX_WINDOW_FRAC: float = 0.5  # sep_max = frac × window_width
_FREE_SEP_TRACE_SCALE_PRIOR: float = (
    0.5  # HalfNormal scale for per-peak trace variation
)

# Artefact area hierarchy
_ARTEFACT_AREA_TRACE_LOG_SCALE: float = 0.15  # ~15% CV per-trace variation


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


def _bounded_separation_prior_to_raw(
    target: jnp.ndarray,
    target_scale: jnp.ndarray,
    sep_min: jnp.ndarray,
    sep_max: jnp.ndarray,
    *,
    bound_eps: float = 1e-4,
    scale_floor: float = 1e-4,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map bounded separation priors to sigmoid raw-space Normal parameters.

    Analogous to ``_bounded_alpha_prior_to_raw`` but uses the logistic
    sigmoid instead of tanh.  Given a target separation in
    ``[sep_min, sep_max]``, returns ``(raw_loc, raw_scale)`` such that
    ``sep_min + (sep_max - sep_min) * sigmoid(raw)`` recovers the target.
    """
    sep_range = jnp.maximum(sep_max - sep_min, 1e-8)
    frac = jnp.clip(
        (jnp.asarray(target, dtype=jnp.float32) - sep_min) / sep_range,
        bound_eps,
        1.0 - bound_eps,
    )
    raw_loc = jnp.log(frac) - jnp.log1p(-frac)  # logit
    derivative = sep_range * frac * (1.0 - frac)  # delta method
    raw_scale = jnp.asarray(target_scale, dtype=jnp.float32) / jnp.maximum(
        derivative, scale_floor
    )
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
    free_fixed_local_index: jnp.ndarray,  # [n_free_fixed]  positions within n_free axis, vary_separation=False
    free_vary_local_index: jnp.ndarray,  # [n_free_vary]   positions within n_free axis, vary_separation=True
    # --- peak priors (from geometric_priors_to_arrays) ---
    apex_loc: jnp.ndarray,  # [n_peak]
    apex_scale: jnp.ndarray,  # [n_peak]
    trace_shift_scale: jnp.ndarray,  # scalar
    sigma_loc: jnp.ndarray,  # [n_peak]  FWHM-derived sigma prior centres
    sigma_scale: jnp.ndarray,  # [n_peak]  FWHM-derived sigma prior scales
    alpha_loc: jnp.ndarray,  # [n_peak]  FWHM-derived alpha prior centres
    alpha_scale: jnp.ndarray,  # [n_peak]  FWHM-derived alpha prior scales
    dominant_area_loc_per_trace: jnp.ndarray,  # [n_trace, n_peak]  per-trace dominant-component area prior
    area_total_loc_per_trace: jnp.ndarray,  # [n_trace, n_peak]  per-trace total free-doublet area prior
    artefact_area_loc_shared: jnp.ndarray,  # [n_artefact]  shared artefact area prior centres
    # --- peak window bounds (from geometric_priors_to_arrays) ---
    window_lo: jnp.ndarray,  # [n_peak]  peak window lower bounds (minutes)
    window_hi: jnp.ndarray,  # [n_peak]  peak window upper bounds (minutes)
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
    n_peak = int(apex_loc.shape[0])
    n_artefact = int(artefact_peak_index.shape[0])
    n_free = int(free_peak_index.shape[0])
    n_nonfree = int(nonfree_peak_index.shape[0])
    n_free_vary = int(free_vary_local_index.shape[0])

    mode_code = jnp.asarray(peak_mode_code, dtype=jnp.int32)
    artefact_side_v = jnp.asarray(artefact_side, dtype=jnp.float32)
    artefact_idx = jnp.asarray(artefact_peak_index, dtype=jnp.int32)
    free_idx = jnp.asarray(free_peak_index, dtype=jnp.int32)
    nonfree_idx = jnp.asarray(nonfree_peak_index, dtype=jnp.int32)
    free_vary_local = jnp.asarray(free_vary_local_index, dtype=jnp.int32)
    free_mask = mode_code == _MODE_FREE_DOUBLET
    nonfree_position = (
        jnp.cumsum((mode_code != _MODE_FREE_DOUBLET).astype(jnp.int32)) - 1
    )
    apex_loc_arr = jnp.asarray(apex_loc, dtype=jnp.float32)
    apex_scale_safe = jnp.maximum(jnp.asarray(apex_scale, dtype=jnp.float32), 1e-6)
    trace_shift_scale_safe = jnp.maximum(
        jnp.asarray(trace_shift_scale, dtype=jnp.float32),
        1e-6,
    )

    # ------------------------------------------------------------------ primitive shape latents
    sigma_loc_safe = jnp.maximum(jnp.asarray(sigma_loc, dtype=jnp.float32), 1e-6)
    sigma_scale_safe = jnp.maximum(jnp.asarray(sigma_scale, dtype=jnp.float32), 1e-6)
    # For free doublets each component is ~half the window width, so halve the reference.
    sigma_prior_loc = jnp.where(free_mask, 0.5 * sigma_loc_safe, sigma_loc_safe)
    # LogUniform: sample log(sigma) ~ Uniform(log(0.5*ref), log(2.0*ref)).
    # Hard bounds prevent sigma from escaping to implausible values under VI.
    log_sigma_lo = jnp.log(0.5 * sigma_prior_loc)
    log_sigma_hi = jnp.log(2.0 * sigma_prior_loc)
    log_sigma_base = numpyro.sample(
        "log_sigma_base", dist.Uniform(log_sigma_lo, log_sigma_hi)
    )  # [n_peak]
    sigma_base = numpyro.deterministic("sigma_base", jnp.exp(log_sigma_base))
    if n_artefact > 0:
        # LogUniform bounded by the same reference as the dominant sigma for that peak.
        # Prevents artefact sigma from expanding to absorb baseline curvature.
        art_ref = sigma_prior_loc[artefact_idx]  # already sigma_loc_safe for nonfree peaks
        log_sigma_r_artefact = numpyro.sample(
            "log_sigma_r_artefact",
            dist.Uniform(jnp.log(0.5 * art_ref), jnp.log(2.0 * art_ref)),
        )
        sigma_r_artefact = numpyro.deterministic(
            "sigma_r_artefact",
            jnp.exp(log_sigma_r_artefact),
        )
    if n_free > 0:
        free_ref = sigma_prior_loc[free_idx]  # 0.5 * sigma_loc for free peaks
        log_sigma_r_free = numpyro.sample(
            "log_sigma_r_free",
            dist.Uniform(jnp.log(0.5 * free_ref), jnp.log(2.0 * free_ref)),
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
    trace_shift_raw = numpyro.sample(
        "trace_shift_raw",
        dist.Normal(0.0, 1.0).expand([n_trace]),
    )
    trace_shift = numpyro.deterministic(
        "trace_shift",
        trace_shift_scale_safe * (trace_shift_raw - jnp.mean(trace_shift_raw)),
    )
    apex_residual_raw = numpyro.sample(
        "apex_residual_raw",
        dist.Normal(0.0, 1.0).expand([n_trace, n_peak]),
    )
    apex_residual = numpyro.deterministic(
        "apex_residual",
        apex_residual_raw * apex_scale_safe[None, :],
    )
    apex = numpyro.deterministic(
        "apex",
        apex_loc_arr[None, :] + trace_shift[:, None] + apex_residual,
    )

    if n_nonfree > 0:
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
        area_artefact_typical = numpyro.sample(
            "area_artefact_typical",
            dist.LogNormal(jnp.log(artefact_area_safe), _SH_AREA_LOG_SIGMA),
        )
        area_artefact_trace_offset = numpyro.sample(
            "area_artefact_trace_offset",
            dist.Normal(0.0, 1.0).expand([n_trace, n_artefact]),
        )
        area_artefact = numpyro.deterministic(
            "area_artefact",
            area_artefact_typical[None, :]
            * jnp.exp(_ARTEFACT_AREA_TRACE_LOG_SCALE * area_artefact_trace_offset),
        )
    if n_free > 0:
        # --- Geometric bounds (all free peaks) ---
        sep_min = numpyro.deterministic(
            "separation_free_min",
            _FREE_SEP_MIN_SIGMA_MULT * sigma_loc_safe[free_idx],
        )
        window_width_free = (
            jnp.asarray(window_hi, dtype=jnp.float32)[free_idx]
            - jnp.asarray(window_lo, dtype=jnp.float32)[free_idx]
        )
        sep_max = numpyro.deterministic(
            "separation_free_max",
            _FREE_SEP_MAX_WINDOW_FRAC * window_width_free,
        )
        sep_range = sep_max - sep_min

        # --- Population-level typical (all free peaks, sigmoid-bounded) ---
        sep_target = _FREE_SEP_TYPICAL_SIGMA_MULT * sigma_loc_safe[free_idx]
        sep_target_scale = _FREE_SEP_TYPICAL_SIGMA_MULT * sigma_scale_safe[free_idx]
        sep_typical_raw_loc, sep_typical_raw_scale = _bounded_separation_prior_to_raw(
            sep_target,
            sep_target_scale,
            sep_min,
            sep_max,
        )
        sep_typical_raw = numpyro.sample(
            "sep_typical_raw",
            dist.Normal(sep_typical_raw_loc, sep_typical_raw_scale),
        )
        sep_typical = sep_min + sep_range * expit(sep_typical_raw)
        numpyro.deterministic("separation_free_typical", sep_typical)

        # --- Assemble separation_free [n_trace, n_free] ---
        # Start: all peaks at their typical separation (correct for fixed peaks)
        separation_free_arr = jnp.broadcast_to(sep_typical[None, :], (n_trace, n_free))

        # Varying peaks: override with per-trace sigmoid-bounded values
        if n_free_vary > 0:
            sep_trace_scale = numpyro.sample(
                "sep_trace_scale",
                dist.HalfNormal(_FREE_SEP_TRACE_SCALE_PRIOR).expand([n_free_vary]),
            )
            separation_free_trace_offset = numpyro.sample(
                "separation_free_trace_offset",
                dist.Normal(0.0, 1.0).expand([n_trace, n_free_vary]),
            )
            sep_raw_vary = (
                sep_typical_raw[None, free_vary_local]
                + sep_trace_scale[None, :] * separation_free_trace_offset
            )
            sep_vary = sep_min[None, free_vary_local] + sep_range[
                None, free_vary_local
            ] * expit(sep_raw_vary)
            separation_free_arr = separation_free_arr.at[:, free_vary_local].set(
                sep_vary
            )

        separation_free = numpyro.deterministic("separation_free", separation_free_arr)

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
        apex_nonfree = apex[:, nonfree_idx]
        apex_l = apex_l.at[:, nonfree_idx].set(apex_nonfree)
        apex_r = apex_r.at[:, nonfree_idx].set(apex_nonfree)
        sigma_l = sigma_l.at[:, nonfree_idx].set(sigma_nonfree)
        sigma_r = sigma_r.at[:, nonfree_idx].set(sigma_nonfree)
        alpha_l = alpha_l.at[:, nonfree_idx].set(alpha_nonfree)
        alpha_r = alpha_r.at[:, nonfree_idx].set(alpha_nonfree)
        area_l = area_l.at[:, nonfree_idx].set(area_dominant)

    if n_artefact > 0:
        artefact_nonfree_idx = nonfree_position[artefact_idx]
        apex_art = apex[:, artefact_idx]
        sigma_dom_art = _broadcast_peak_to_traces(sigma_base[artefact_idx], n_trace)
        sigma_art = _broadcast_peak_to_traces(sigma_r_artefact, n_trace)
        alpha_dom_art = _broadcast_peak_to_traces(alpha_base[artefact_idx], n_trace)
        area_dom_art = area_dominant[:, artefact_nonfree_idx]
        area_art = area_artefact
        separation_art = _broadcast_peak_to_traces(separation_artefact, n_trace)
        artefact_left = artefact_side_v[artefact_idx] < 0.0

        apex_l_art = jnp.where(
            artefact_left[None, :],
            apex_art - separation_art,
            apex_art,
        )
        apex_r_art = jnp.where(
            artefact_left[None, :],
            apex_art,
            apex_art + separation_art,
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
        sigma_l_free = _broadcast_peak_to_traces(sigma_base[free_idx], n_trace)
        sigma_r_free_b = _broadcast_peak_to_traces(sigma_r_free, n_trace)
        alpha_l_free = _broadcast_peak_to_traces(alpha_base[free_idx], n_trace)
        alpha_r_free_b = _broadcast_peak_to_traces(alpha_r_free, n_trace)
        apex_free = apex[:, free_idx]
        apex_l_free = apex_free - 0.5 * separation_free
        apex_r_free = apex_free + 0.5 * separation_free
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

    numpyro.deterministic("area_total", area_l + area_r)
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
    # Centre x at the peak-window midpoint so that baseline_intercept is the
    # baseline level *within the observed region* rather than an extrapolation
    # to x = 0.  Without centring, intercept and slope share a near-perfect
    # anti-correlation ridge (a change of Δb₁ in slope requires
    # Δb₀ ≈ −x_mid·Δb₁ ≈ −3·Δb₁ to keep the baseline constant), making both
    # parameters essentially unidentifiable from windowed data.
    x_mid = 0.5 * (jnp.min(window_lo) + jnp.max(window_hi))  # scalar

    # Transform the caller's x = 0 prior to the x = x_mid basis:
    #   E[b₀ + b₁·x_mid]   = b₀_loc + b₁_loc·x_mid
    #   Var[b₀ + b₁·x_mid] ≈ Var(b₀) + x_mid²·Var(b₁)
    # (conservative: ignores the negative b₀–b₁ covariance in the linear fit,
    # so the prior is slightly wider than optimal — harmless, the likelihood
    # tightens it).
    baseline_mid_loc = baseline_intercept_loc + baseline_slope_loc * x_mid
    baseline_mid_scale = jnp.sqrt(
        jnp.maximum(baseline_intercept_scale, 1e-6) ** 2
        + (x_mid * jnp.maximum(baseline_slope_scale, 1e-8)) ** 2
    )
    baseline_intercept = numpyro.sample(
        "baseline_intercept",
        dist.Normal(baseline_mid_loc, jnp.maximum(baseline_mid_scale, 1e-6)),
    )  # [n_trace] — baseline level at x_mid (directly observable, identifiable)

    # Hierarchical slope: pool across traces so that traces with weak signal
    # borrow information from their neighbours.  Non-centered parameterisation
    # is numerically stable for both MCMC and VI.
    #
    # Hyperprior mean: centred at the mean of the OLS slope estimates.
    # Hyperprior scale: covers the spread of OLS estimates + average per-trace uncertainty.
    slope_pop_loc = jnp.mean(jnp.asarray(baseline_slope_loc, dtype=jnp.float32))
    slope_pop_scale_prior = jnp.maximum(
        jnp.std(jnp.asarray(baseline_slope_loc, dtype=jnp.float32))
        + jnp.mean(jnp.asarray(baseline_slope_scale, dtype=jnp.float32)),
        1e-6,
    )
    baseline_slope_pop_mean = numpyro.sample(
        "baseline_slope_pop_mean",
        dist.Normal(slope_pop_loc, slope_pop_scale_prior),
    )  # scalar — population-average slope
    # How much individual traces vary around the population mean.
    # Prior: HalfNormal with scale = mean per-trace OLS uncertainty.
    slope_variation_prior = jnp.maximum(
        jnp.mean(jnp.asarray(baseline_slope_scale, dtype=jnp.float32)),
        1e-6,
    )
    baseline_slope_pop_scale = numpyro.sample(
        "baseline_slope_pop_scale",
        dist.HalfNormal(slope_variation_prior),
    )  # scalar ≥ 0
    # Per-trace slopes: non-centered; shape [n_trace].
    # Note: NOT mean-subtracted (unlike trace_shift_raw) — there is no
    # sum-to-zero identifiability constraint on slopes; pop_mean absorbs
    # the overall level.
    baseline_slope_raw = numpyro.sample(
        "baseline_slope_raw",
        dist.Normal(0.0, 1.0).expand([n_trace]),
    )
    baseline_slope = numpyro.deterministic(
        "baseline_slope",
        baseline_slope_pop_mean + baseline_slope_pop_scale * baseline_slope_raw,
    )  # [n_trace]

    baseline = baseline_intercept[:, None] + baseline_slope[:, None] * (x - x_mid)
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
    "trace_shift",
    "apex",
    "separation_free_typical",
    "sep_trace_scale",
    "separation",
    "sigma_base",
    "sigma_r_artefact",
    "sigma_r_free",
    "alpha_base",
    "alpha_r_free",
    "area_l",
    "area_r",
    "area_total",
    "area_artefact_typical",
    "baseline_intercept",
    "baseline_slope",
    "baseline_slope_pop_mean",
    "baseline_slope_pop_scale",
    "sigma_y",
)

TRACE_PARAMETER_NAMES = SUMMARY_PARAMETER_NAMES
