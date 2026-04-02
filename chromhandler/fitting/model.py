"""Mode-aware skew-normal chromatographic peak model.

Supports three peak modes:

- ``single``: one component per logical peak window.
- ``artefact_doublet``: dominant component plus an artefact component.
- ``free_doublet``: true two-component peak with a free area split.

Parameterization: the model samples ``(log_w_left, log_w_right)`` — log of the
left and right half-widths at half-maximum (HWHM).  These are orthogonal and
directly observable, giving NUTS well-conditioned geometry.  The
``(sigma, alpha)`` skew-normal parameters are recovered deterministically via
the Gaussian-HWHM approximation inside the model.

Location geometry: ``apex[t, p] = apex_loc[p] + trace_shift[t]``.

The model keeps all intermediate arrays as plain local variables to minimise
HMC leapfrog overhead.  Call :func:`compute_derived_quantities` after sampling
to reconstruct them from the raw posterior samples.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import log_ndtr

from .types import ModelHyperparams  # noqa: TC001 — used at runtime in model() signature

numpyro.set_host_device_count(8)

_SQRT_2_OVER_PI: float = float(jnp.sqrt(2.0 / jnp.pi))
_HWHM_FACTOR: float = float(jnp.sqrt(2.0 * jnp.log(2.0)))


# ---------------------------------------------------------------------------
# Half-width → (sigma, alpha) conversion
# ---------------------------------------------------------------------------


def _halfwidths_to_shape(
    log_w_left: jax.Array,
    log_w_right: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Convert log half-widths to (sigma, alpha). Works on any batch shape."""
    w_left = jnp.exp(log_w_left)
    w_right = jnp.exp(log_w_right)
    s_left = w_left / _HWHM_FACTOR
    s_right = w_right / _HWHM_FACTOR
    sigma = jnp.sqrt(0.5 * (s_left**2 + s_right**2))
    denom = jnp.maximum(s_right + s_left, 1e-12)
    delta = jnp.clip((s_right - s_left) / denom, -0.95, 0.95)
    alpha = delta / jnp.sqrt(jnp.maximum(1.0 - delta**2, 1e-8))
    return sigma, alpha


# ---------------------------------------------------------------------------
# Skew-normal math
# ---------------------------------------------------------------------------


def log_skew_normal_pdf(
    x: jax.Array,      # [n_trace, n_time]
    xi: jax.Array,     # [n_trace, n_comp]  — skew-normal location param (NOT mode)
    sigma: jax.Array,  # [n_trace, n_comp]
    alpha: jax.Array,  # [n_trace, n_comp]
) -> jax.Array:
    """Numerically stable log skew-normal density.

    Returns shape ``[n_trace, n_comp, n_time]``.
    """
    sigma_s = jnp.maximum(sigma, 1e-6)
    z = (x[:, None, :] - xi[:, :, None]) / sigma_s[:, :, None]
    return (
        jnp.log(2.0)
        - jnp.log(sigma_s)[:, :, None]
        - 0.5 * z**2
        - 0.5 * jnp.log(2.0 * jnp.pi)
        + log_ndtr(alpha[:, :, None] * z)
    )


def skew_normal_pdf(
    x: jax.Array,
    xi: jax.Array,
    sigma: jax.Array,
    alpha: jax.Array,
) -> jax.Array:
    """Skew-normal density. Same shape convention as ``log_skew_normal_pdf``."""
    return jnp.exp(log_skew_normal_pdf(x, xi, sigma, alpha))


def mixture_signal(
    x: jax.Array,      # [n_trace, n_time]
    xi: jax.Array,     # [n_trace, n_comp]
    sigma: jax.Array,  # [n_trace, n_comp]
    alpha: jax.Array,  # [n_trace, n_comp]
    area: jax.Array,   # [n_trace, n_comp]
) -> jax.Array:
    """Area-scaled skew-normal mixture, summed over components.

    Returns shape ``[n_trace, n_time]``.
    """
    pdf = skew_normal_pdf(x, xi, sigma, alpha)  # [n_trace, n_comp, n_time]
    return jnp.sum(area[:, :, None] * pdf, axis=1)


# ---------------------------------------------------------------------------
# Assembly helpers
# ---------------------------------------------------------------------------


def _apex_to_xi(
    apex: jax.Array,
    sigma: jax.Array,
    alpha: jax.Array,
) -> jax.Array:
    """Convert skew-normal mode to the location parameter xi."""
    delta = alpha / jnp.sqrt(1.0 + alpha**2)
    return apex - sigma * delta * _SQRT_2_OVER_PI


def _stack_left_right(left: Any, right: Any) -> jax.Array:
    """Flatten left/right peak matrices to the mixture component axis."""
    left_arr = jnp.asarray(left)
    return jnp.stack([left_arr, jnp.asarray(right)], axis=-1).reshape(left_arr.shape[0], -1)


_LRArrays = tuple[
    jax.Array,  # apex_l
    jax.Array,  # apex_r
    jax.Array,  # sigma_l
    jax.Array,  # sigma_r
    jax.Array,  # alpha_l
    jax.Array,  # alpha_r
    jax.Array,  # area_l
    jax.Array,  # area_r
]


def _assemble_nonfree(
    lr: _LRArrays,
    apex: Any,        # [n_trace, n_peak]
    sigma_base: Any,  # [n_peak]
    alpha_base: Any,  # [n_peak]
    area_dominant: Any,  # [n_trace, n_nonfree]
    nonfree_idx: Any,    # [n_nonfree]
    n_trace: int,
) -> _LRArrays:
    """Fill left/right matrices for single and artefact_doublet (nonfree) peaks."""
    apex_l, apex_r, sigma_l, sigma_r, alpha_l, alpha_r, area_l, area_r = lr
    sigma_nf = jnp.broadcast_to(sigma_base[None, nonfree_idx], (n_trace, nonfree_idx.shape[0]))
    alpha_nf = jnp.broadcast_to(alpha_base[None, nonfree_idx], (n_trace, nonfree_idx.shape[0]))
    apex_nf = apex[:, nonfree_idx]
    apex_l = apex_l.at[:, nonfree_idx].set(apex_nf)
    apex_r = apex_r.at[:, nonfree_idx].set(apex_nf)
    sigma_l = sigma_l.at[:, nonfree_idx].set(sigma_nf)
    sigma_r = sigma_r.at[:, nonfree_idx].set(sigma_nf)
    alpha_l = alpha_l.at[:, nonfree_idx].set(alpha_nf)
    alpha_r = alpha_r.at[:, nonfree_idx].set(alpha_nf)
    area_l = area_l.at[:, nonfree_idx].set(area_dominant)
    return apex_l, apex_r, sigma_l, sigma_r, alpha_l, alpha_r, area_l, area_r


def _assemble_artefact(
    lr: _LRArrays,
    apex: Any,              # [n_trace, n_peak]
    sigma_base: Any,        # [n_peak]
    alpha_base: Any,        # [n_peak]
    sigma_art: Any,         # [n_artefact]  — artefact-component sigma
    alpha_art: Any,         # [n_artefact]  — artefact-component alpha
    area_dominant: Any,     # [n_trace, n_nonfree]
    area_artefact: Any,     # [n_trace, n_artefact]
    separation_artefact: Any,   # [n_artefact]
    artefact_idx: Any,          # [n_artefact]
    nonfree_position: Any,      # [n_peak]
    artefact_side_v: Any,       # [n_peak]  float: -1=left, 0=none, +1=right
    n_trace: int,
) -> _LRArrays:
    """Overwrite artefact_doublet columns with two-component geometry."""
    apex_l, apex_r, sigma_l, sigma_r, alpha_l, alpha_r, area_l, area_r = lr

    artefact_nonfree_idx = nonfree_position[artefact_idx]
    n_art = artefact_idx.shape[0]
    apex_art = apex[:, artefact_idx]
    sigma_dom = jnp.broadcast_to(sigma_base[None, artefact_idx], (n_trace, n_art))
    s_art = jnp.broadcast_to(sigma_art[None, :], (n_trace, n_art))
    alpha_dom = jnp.broadcast_to(alpha_base[None, artefact_idx], (n_trace, n_art))
    a_art = jnp.broadcast_to(alpha_art[None, :], (n_trace, n_art))
    area_dom = area_dominant[:, artefact_nonfree_idx]
    sep = jnp.broadcast_to(separation_artefact[None, :], (n_trace, n_art))

    art_left = artefact_side_v[artefact_idx] < 0.0  # [n_artefact]
    apex_l_art = jnp.where(art_left[None, :], apex_art - sep, apex_art)
    apex_r_art = jnp.where(art_left[None, :], apex_art, apex_art + sep)
    sigma_l_art = jnp.where(art_left[None, :], s_art, sigma_dom)
    sigma_r_art = jnp.where(art_left[None, :], sigma_dom, s_art)
    alpha_l_art = jnp.where(art_left[None, :], a_art, alpha_dom)
    alpha_r_art = jnp.where(art_left[None, :], alpha_dom, a_art)
    area_l_art = jnp.where(art_left[None, :], area_artefact, area_dom)
    area_r_art = jnp.where(art_left[None, :], area_dom, area_artefact)

    apex_l = apex_l.at[:, artefact_idx].set(apex_l_art)
    apex_r = apex_r.at[:, artefact_idx].set(apex_r_art)
    sigma_l = sigma_l.at[:, artefact_idx].set(sigma_l_art)
    sigma_r = sigma_r.at[:, artefact_idx].set(sigma_r_art)
    alpha_l = alpha_l.at[:, artefact_idx].set(alpha_l_art)
    alpha_r = alpha_r.at[:, artefact_idx].set(alpha_r_art)
    area_l = area_l.at[:, artefact_idx].set(area_l_art)
    area_r = area_r.at[:, artefact_idx].set(area_r_art)
    return apex_l, apex_r, sigma_l, sigma_r, alpha_l, alpha_r, area_l, area_r


def _assemble_free(
    lr: _LRArrays,
    apex: Any,               # [n_trace, n_peak]
    sigma_base: Any,         # [n_peak]
    alpha_base: Any,         # [n_peak]
    sigma_r_free: Any,       # [n_free]
    alpha_r_free: Any,       # [n_free]
    area_total_free: Any,    # [n_trace, n_free]
    area_frac_left_free: Any,  # [n_trace, n_free]
    separation_free: Any,    # [n_free]
    free_idx: Any,           # [n_free]
    n_trace: int,
) -> _LRArrays:
    """Fill free_doublet columns with two-component geometry."""
    apex_l, apex_r, sigma_l, sigma_r, alpha_l, alpha_r, area_l, area_r = lr
    n_free = free_idx.shape[0]

    sl = jnp.broadcast_to(sigma_base[None, free_idx], (n_trace, n_free))
    sr = jnp.broadcast_to(sigma_r_free[None, :], (n_trace, n_free))
    al = jnp.broadcast_to(alpha_base[None, free_idx], (n_trace, n_free))
    ar = jnp.broadcast_to(alpha_r_free[None, :], (n_trace, n_free))
    apex_free = apex[:, free_idx]
    apex_l_free = apex_free - 0.5 * separation_free[None, :]
    apex_r_free = apex_free + 0.5 * separation_free[None, :]
    area_l_free = area_total_free * area_frac_left_free
    area_r_free = area_total_free * (1.0 - area_frac_left_free)

    apex_l = apex_l.at[:, free_idx].set(apex_l_free)
    apex_r = apex_r.at[:, free_idx].set(apex_r_free)
    sigma_l = sigma_l.at[:, free_idx].set(sl)
    sigma_r = sigma_r.at[:, free_idx].set(sr)
    alpha_l = alpha_l.at[:, free_idx].set(al)
    alpha_r = alpha_r.at[:, free_idx].set(ar)
    area_l = area_l.at[:, free_idx].set(area_l_free)
    area_r = area_r.at[:, free_idx].set(area_r_free)
    return apex_l, apex_r, sigma_l, sigma_r, alpha_l, alpha_r, area_l, area_r


# ---------------------------------------------------------------------------
# NumPyro model
# ---------------------------------------------------------------------------

_MODE_SINGLE = 0
_MODE_ARTEFACT_DOUBLET = 1
_MODE_FREE_DOUBLET = 2


def model(
    x: jax.Array,           # [n_trace, n_time]
    y: jax.Array | None,    # [n_trace, n_time] or None (prior predictive)
    # --- hyperparameters ---
    hyperparams: ModelHyperparams,
    # --- peak structure ---
    peak_mode_code: jax.Array,      # [n_peak]
    artefact_side: jax.Array,       # [n_peak]  int: -1=left, 0=none, +1=right
    artefact_peak_index: jax.Array, # [n_artefact]
    free_peak_index: jax.Array,     # [n_free]
    nonfree_idx: jax.Array,         # [n_nonfree]
    nonfree_position: jax.Array,    # [n_peak]
    # --- peak priors ---
    apex_loc: jax.Array,            # [n_peak]
    trace_shift_scale: jax.Array,   # scalar
    w_left_loc: jax.Array,          # [n_peak]
    w_left_scale: jax.Array,        # [n_peak]
    w_right_loc: jax.Array,         # [n_peak]
    w_right_scale: jax.Array,       # [n_peak]
    area_gaussian_pt: jax.Array,    # [n_trace, n_peak]
    area_art_shared: jax.Array,     # [n_artefact]
    snr_per_trace: jax.Array,       # [n_trace, n_peak]
    # --- peak window bounds ---
    window_lo: jax.Array,           # [n_peak]
    window_hi: jax.Array,           # [n_peak]
    # --- baseline priors ---
    baseline_intercept_loc: jax.Array,    # [n_trace]
    baseline_intercept_scale: jax.Array,  # [n_trace]
    baseline_slope_loc: jax.Array,        # [n_trace]
    baseline_slope_scale: jax.Array,      # [n_trace]
    # --- noise prior ---
    sigma_y_prior_loc: jax.Array,   # [n_trace]
) -> None:
    """Bayesian skew-normal peak model using (log_w_left, log_w_right) parameterization.

    All input arrays are pre-validated float32 by ``Fitter._prepare_model_inputs()``.
    """
    # 1. Shape constants
    n_trace, _ = x.shape
    n_peak = int(apex_loc.shape[0])
    n_artefact = int(artefact_peak_index.shape[0])
    n_free = int(free_peak_index.shape[0])
    n_nonfree = int(nonfree_idx.shape[0])

    artefact_side_v = artefact_side.astype(jnp.float32)
    artefact_idx = artefact_peak_index
    free_idx = free_peak_index

    # 2. S/N-dependent area spread [n_trace, n_peak]
    hp = hyperparams
    snr_frac = jnp.clip(
        (snr_per_trace - hp.area_snr_threshold_low)
        / (hp.area_snr_threshold_high - hp.area_snr_threshold_low),
        0.0, 1.0,
    )
    area_log_sigma = (
        hp.area_log_sigma_low_snr
        - snr_frac * (hp.area_log_sigma_low_snr - hp.area_log_sigma_high_snr)
    )  # [n_trace, n_peak]

    # 3. Primary half-width priors (one per peak, shared across traces)
    w_left_log_scale = jnp.maximum(
        w_left_scale / jnp.maximum(w_left_loc, 1e-9), hp.w_prior_log_scale
    )
    w_right_log_scale = jnp.maximum(
        w_right_scale / jnp.maximum(w_right_loc, 1e-9), hp.w_prior_log_scale
    )
    log_w_left = numpyro.sample(
        "log_w_left", dist.Normal(jnp.log(w_left_loc), w_left_log_scale)
    )  # [n_peak]
    log_w_right = numpyro.sample(
        "log_w_right", dist.Normal(jnp.log(w_right_loc), w_right_log_scale)
    )  # [n_peak]
    sigma_base, alpha_base = _halfwidths_to_shape(  # type: ignore[arg-type]
        log_w_left, log_w_right
    )  # [n_peak]

    # 4. Doublet second-component shape (artefact + free combined)
    #    doublet order: artefact peaks first, free peaks after
    n_doublet = n_artefact + n_free
    sigma_art: jax.Array = jnp.zeros((0,), dtype=jnp.float32)
    alpha_art: jax.Array = jnp.zeros((0,), dtype=jnp.float32)
    sigma_r_free: jax.Array = jnp.zeros((0,), dtype=jnp.float32)
    alpha_r_free: jax.Array = jnp.zeros((0,), dtype=jnp.float32)

    if n_doublet > 0:
        # Build doublet index and prior centres at Python level
        doublet_idx_np = np.concatenate([  # type: ignore[arg-type]
            np.asarray(artefact_peak_index), np.asarray(free_peak_index)
        ]).astype(np.int32)
        doublet_idx = jnp.asarray(doublet_idx_np)

        # Prior: 0.6 * observed half-widths, wide log-scale
        log_w_left_2 = numpyro.sample(
            "log_w_left_2",
            dist.Normal(jnp.log(0.6 * w_left_loc[doublet_idx]), 0.5),
        )  # [n_doublet]
        log_w_right_2 = numpyro.sample(
            "log_w_right_2",
            dist.Normal(jnp.log(0.6 * w_right_loc[doublet_idx]), 0.5),
        )  # [n_doublet]
        sigma_2, alpha_2 = _halfwidths_to_shape(  # type: ignore[arg-type]
            log_w_left_2, log_w_right_2
        )

        sigma_art = sigma_2[:n_artefact]
        alpha_art = alpha_2[:n_artefact]
        sigma_r_free = sigma_2[n_artefact:]
        alpha_r_free = alpha_2[n_artefact:]

    # 5. Separation priors
    separation_artefact: jax.Array = jnp.zeros((0,), dtype=jnp.float32)
    separation_free: jax.Array = jnp.zeros((0,), dtype=jnp.float32)

    if n_artefact > 0:
        sep_loc_art = w_left_loc[artefact_idx]  # ~1 half-width
        log_separation_artefact = numpyro.sample(
            "log_separation_artefact",
            dist.Normal(jnp.log(sep_loc_art), 0.6),
        )  # [n_artefact]
        separation_artefact = jnp.exp(log_separation_artefact)  # type: ignore[assignment]

    if n_free > 0:
        sep_loc_free = 2.0 * w_left_loc[free_idx]
        log_separation_free = numpyro.sample(
            "log_separation_free",
            dist.Normal(jnp.log(sep_loc_free), hp.free_sep_log_sigma),
        )  # [n_free]
        separation_free = jnp.exp(log_separation_free)  # type: ignore[assignment]

    # 6. Per-trace parameters
    x_mid = 0.5 * (jnp.min(window_lo) + jnp.max(window_hi))
    baseline_mid_loc = baseline_intercept_loc + baseline_slope_loc * x_mid
    baseline_mid_scale = jnp.sqrt(
        baseline_intercept_scale**2 + (x_mid * baseline_slope_scale) ** 2
    )

    with numpyro.plate("traces", n_trace):
        trace_shift_raw = numpyro.sample("trace_shift_raw", dist.Normal(0.0, 1.0))
        baseline_intercept = numpyro.sample(
            "baseline_intercept", dist.Normal(baseline_mid_loc, baseline_mid_scale)
        )
        baseline_slope = numpyro.sample(
            "baseline_slope", dist.Normal(baseline_slope_loc, baseline_slope_scale)
        )
        sigma_y = numpyro.sample(
            "sigma_y", dist.LogNormal(jnp.log(sigma_y_prior_loc), 0.5)
        )

    trace_shift = trace_shift_scale * (trace_shift_raw - jnp.mean(trace_shift_raw))
    apex = apex_loc[None, :] + trace_shift[:, None]  # [n_trace, n_peak]

    # 7. Per-trace area sampling
    # NOTE: plates are given unique names and nested to avoid NumPyro misidentifying
    # the trace vs. peak dimensions when pmap-vectorising over chains.
    area_dominant: jax.Array = jnp.zeros((n_trace, 0), dtype=jnp.float32)
    area_artefact: jax.Array = jnp.zeros((n_trace, 0), dtype=jnp.float32)
    area_total_free: jax.Array = jnp.zeros((n_trace, 0), dtype=jnp.float32)
    area_frac_left_free: jax.Array = jnp.zeros((n_trace, 0), dtype=jnp.float32)

    if n_nonfree > 0:
        with numpyro.plate("traces_nonfree", n_trace, dim=-2):
            with numpyro.plate("nonfree_peaks", n_nonfree, dim=-1):
                area_dominant = numpyro.sample(  # type: ignore[assignment]
                    "area_dominant",
                    dist.LogNormal(
                        jnp.log(jnp.maximum(area_gaussian_pt[:, nonfree_idx], 1e-6)),
                        area_log_sigma[:, nonfree_idx],
                    ),
                )  # [n_trace, n_nonfree]

    if n_artefact > 0:
        # Artefact peaks are systematic (column chemistry); area is constant across
        # traces.  Sampling a single shared value per artefact type is sufficient
        # and avoids the per-trace offset plate that caused plate-name conflicts.
        area_artefact_typical = numpyro.sample(
            "area_artefact_typical",
            dist.LogNormal(jnp.log(jnp.maximum(area_art_shared, 1e-6)), hp.area_art_log_sigma),
        )  # [n_artefact]
        area_artefact = jnp.broadcast_to(  # type: ignore[assignment]
            area_artefact_typical[None, :], (n_trace, n_artefact)
        )  # [n_trace, n_artefact]

    if n_free > 0:
        with numpyro.plate("traces_free", n_trace, dim=-2):
            with numpyro.plate("free_peaks", n_free, dim=-1):
                area_total_free = numpyro.sample(  # type: ignore[assignment]
                    "area_total_free",
                    dist.LogNormal(
                        jnp.log(jnp.maximum(area_gaussian_pt[:, free_idx], 1e-6)),
                        area_log_sigma[:, free_idx],
                    ),
                )  # [n_trace, n_free]
                area_frac_left_free = numpyro.sample(  # type: ignore[assignment]
                    "area_frac_left_free",
                    dist.Beta(2.0, 2.0),
                )  # [n_trace, n_free]

    # 8. Left/right canonical assembly
    lr: _LRArrays = (
        jnp.zeros((n_trace, n_peak), dtype=jnp.float32),
        jnp.zeros((n_trace, n_peak), dtype=jnp.float32),
        jnp.zeros((n_trace, n_peak), dtype=jnp.float32),
        jnp.zeros((n_trace, n_peak), dtype=jnp.float32),
        jnp.zeros((n_trace, n_peak), dtype=jnp.float32),
        jnp.zeros((n_trace, n_peak), dtype=jnp.float32),
        jnp.zeros((n_trace, n_peak), dtype=jnp.float32),
        jnp.zeros((n_trace, n_peak), dtype=jnp.float32),
    )
    if n_nonfree > 0:
        lr = _assemble_nonfree(lr, apex, sigma_base, alpha_base, area_dominant, nonfree_idx, n_trace)
    if n_artefact > 0:
        lr = _assemble_artefact(
            lr, apex, sigma_base, alpha_base, sigma_art, alpha_art,
            area_dominant, area_artefact, separation_artefact,
            artefact_idx, nonfree_position, artefact_side_v, n_trace,
        )
    if n_free > 0:
        lr = _assemble_free(
            lr, apex, sigma_base, alpha_base, sigma_r_free, alpha_r_free,
            area_total_free, area_frac_left_free, separation_free, free_idx, n_trace,
        )

    apex_l, apex_r, sigma_l, sigma_r, alpha_l, alpha_r, area_l, area_r = lr

    xi_l = _apex_to_xi(apex_l, sigma_l, alpha_l)
    xi_r = _apex_to_xi(apex_r, sigma_r, alpha_r)

    xi_flat = _stack_left_right(xi_l, xi_r)
    sigma_flat = _stack_left_right(sigma_l, sigma_r)
    alpha_flat = _stack_left_right(alpha_l, alpha_r)
    area_flat = _stack_left_right(area_l, area_r)

    # 9. Baseline and likelihood
    baseline = baseline_intercept[:, None] + baseline_slope[:, None] * (x - x_mid)
    mu_y = mixture_signal(x, xi_flat, sigma_flat, alpha_flat, area_flat) + baseline
    if y is not None:
        finite_mask = jnp.isfinite(y)
        numpyro.sample(
            "y",
            dist.Normal(mu_y, sigma_y[:, None]).mask(finite_mask),
            obs=jnp.where(finite_mask, y, 0.0),
        )


# ---------------------------------------------------------------------------
# Post-sampling derived quantity reconstruction
# ---------------------------------------------------------------------------


def compute_derived_quantities(
    samples: dict[str, Any],
    model_inputs: dict[str, Any],
    hyperparams: ModelHyperparams,
) -> dict[str, Any]:
    """Reconstruct deterministic-style derived arrays from raw posterior samples.

    Parameters
    ----------
    samples:
        Dict returned by ``mcmc.get_samples(group_by_chain=False)``.
        Shape: ``[n_total, ...]`` per entry.
    model_inputs:
        Same dict passed to ``mcmc.run(**model_inputs)``.
    hyperparams:
        Same :class:`ModelHyperparams` used during fitting.
    """
    artefact_idx = jnp.asarray(model_inputs["artefact_peak_index"], dtype=jnp.int32)
    free_idx = jnp.asarray(model_inputs["free_peak_index"], dtype=jnp.int32)
    nonfree_idx_in = jnp.asarray(model_inputs["nonfree_idx"], dtype=jnp.int32)
    nonfree_position = jnp.asarray(model_inputs["nonfree_position"], dtype=jnp.int32)
    artefact_side_v = jnp.asarray(model_inputs["artefact_side"], dtype=jnp.float32)
    apex_loc_arr = jnp.asarray(model_inputs["apex_loc"], dtype=jnp.float32)
    trace_shift_scale = float(jnp.asarray(model_inputs["trace_shift_scale"]).max())
    window_lo_arr = jnp.asarray(model_inputs["window_lo"], dtype=jnp.float32)
    window_hi_arr = jnp.asarray(model_inputs["window_hi"], dtype=jnp.float32)
    w_left_loc_arr = jnp.asarray(model_inputs["w_left_loc"], dtype=jnp.float32)

    n_peak = int(apex_loc_arr.shape[0])
    n_artefact = int(artefact_idx.shape[0])
    n_free = int(free_idx.shape[0])
    n_nonfree = int(nonfree_idx_in.shape[0])
    n_doublet = n_artefact + n_free

    log_w_left = jnp.asarray(samples["log_w_left"])    # [n_total, n_peak]
    log_w_right = jnp.asarray(samples["log_w_right"])  # [n_total, n_peak]
    trace_shift_raw = jnp.asarray(samples["trace_shift_raw"])  # [n_total, n_trace]

    n_total, n_trace = trace_shift_raw.shape

    sigma_base, alpha_base = _halfwidths_to_shape(log_w_left, log_w_right)  # [n_total, n_peak]
    trace_shift = trace_shift_scale * (
        trace_shift_raw - trace_shift_raw.mean(axis=1, keepdims=True)
    )  # [n_total, n_trace]
    apex = apex_loc_arr[None, None, :] + trace_shift[:, :, None]  # [n_total, n_trace, n_peak]

    # Second-component shapes
    sigma_art: jax.Array = jnp.zeros((n_total, 0), dtype=jnp.float32)
    alpha_art: jax.Array = jnp.zeros((n_total, 0), dtype=jnp.float32)
    sigma_r_free: jax.Array = jnp.zeros((n_total, 0), dtype=jnp.float32)
    alpha_r_free: jax.Array = jnp.zeros((n_total, 0), dtype=jnp.float32)
    separation_artefact: jax.Array = jnp.zeros((n_total, 0), dtype=jnp.float32)
    separation_free: jax.Array = jnp.zeros((n_total, 0), dtype=jnp.float32)

    if n_doublet > 0:
        log_w_left_2 = jnp.asarray(samples["log_w_left_2"])   # [n_total, n_doublet]
        log_w_right_2 = jnp.asarray(samples["log_w_right_2"])
        sigma_2, alpha_2 = _halfwidths_to_shape(log_w_left_2, log_w_right_2)
        sigma_art = sigma_2[:, :n_artefact]
        alpha_art = alpha_2[:, :n_artefact]
        sigma_r_free = sigma_2[:, n_artefact:]
        alpha_r_free = alpha_2[:, n_artefact:]

    if n_artefact > 0:
        separation_artefact = jnp.exp(
            jnp.asarray(samples["log_separation_artefact"])
        )  # [n_total, n_artefact]

        area_artefact_typical = jnp.asarray(samples["area_artefact_typical"])
        # Artefact areas are constant across traces (no per-trace offset in model).
        area_artefact = jnp.broadcast_to(
            area_artefact_typical[:, None, :], (n_total, n_trace, n_artefact)
        )  # [n_total, n_trace, n_artefact]
    else:
        area_artefact = jnp.zeros((n_total, n_trace, 0), dtype=jnp.float32)

    if n_free > 0:
        sep_loc_free = 2.0 * w_left_loc_arr[free_idx]
        sep_range = jnp.maximum(
            hyperparams.art_sep_max_window_frac * (window_hi_arr[free_idx] - window_lo_arr[free_idx])
            - sep_loc_free,
            1e-8,
        )
        _ = sep_range  # kept for potential future use
        separation_free = jnp.exp(
            jnp.asarray(samples["log_separation_free"])
        )  # [n_total, n_free]

    baseline_slope = jnp.asarray(samples["baseline_slope"])  # [n_total, n_trace]

    # Left/right assembly: [n_total, n_trace, n_peak]
    apex_l: jax.Array = jnp.zeros((n_total, n_trace, n_peak), dtype=jnp.float32)
    apex_r: jax.Array = jnp.zeros((n_total, n_trace, n_peak), dtype=jnp.float32)
    sigma_l: jax.Array = jnp.zeros((n_total, n_trace, n_peak), dtype=jnp.float32)
    sigma_r: jax.Array = jnp.zeros((n_total, n_trace, n_peak), dtype=jnp.float32)
    alpha_l: jax.Array = jnp.zeros((n_total, n_trace, n_peak), dtype=jnp.float32)
    alpha_r: jax.Array = jnp.zeros((n_total, n_trace, n_peak), dtype=jnp.float32)
    area_l: jax.Array = jnp.zeros((n_total, n_trace, n_peak), dtype=jnp.float32)
    area_r: jax.Array = jnp.zeros((n_total, n_trace, n_peak), dtype=jnp.float32)
    separation_out: jax.Array = jnp.zeros((n_total, n_trace, n_peak), dtype=jnp.float32)

    if n_nonfree > 0:
        area_dominant = jnp.asarray(samples["area_dominant"])  # [n_total, n_trace, n_nonfree]
        sb_nf = jnp.broadcast_to(
            sigma_base[:, None, nonfree_idx_in], (n_total, n_trace, n_nonfree)
        )
        ab_nf = jnp.broadcast_to(
            alpha_base[:, None, nonfree_idx_in], (n_total, n_trace, n_nonfree)
        )
        apex_l = apex_l.at[:, :, nonfree_idx_in].set(apex[:, :, nonfree_idx_in])
        apex_r = apex_r.at[:, :, nonfree_idx_in].set(apex[:, :, nonfree_idx_in])
        sigma_l = sigma_l.at[:, :, nonfree_idx_in].set(sb_nf)
        sigma_r = sigma_r.at[:, :, nonfree_idx_in].set(sb_nf)
        alpha_l = alpha_l.at[:, :, nonfree_idx_in].set(ab_nf)
        alpha_r = alpha_r.at[:, :, nonfree_idx_in].set(ab_nf)
        area_l = area_l.at[:, :, nonfree_idx_in].set(area_dominant)

    if n_artefact > 0:
        artefact_nonfree_idx = nonfree_position[artefact_idx]
        n_art = int(artefact_idx.shape[0])
        apex_art_s = apex[:, :, artefact_idx]
        sb_art = jnp.broadcast_to(sigma_base[:, None, artefact_idx], (n_total, n_trace, n_art))
        s_art_bc = jnp.broadcast_to(sigma_art[:, None, :], (n_total, n_trace, n_art))
        ab_art = jnp.broadcast_to(alpha_base[:, None, artefact_idx], (n_total, n_trace, n_art))
        a_art_bc = jnp.broadcast_to(alpha_art[:, None, :], (n_total, n_trace, n_art))
        area_dom_art = jnp.asarray(samples["area_dominant"])[:, :, artefact_nonfree_idx]
        sep_art_bc = jnp.broadcast_to(separation_artefact[:, None, :], (n_total, n_trace, n_art))

        art_left = artefact_side_v[artefact_idx] < 0.0
        apex_l_art = jnp.where(art_left[None, None, :], apex_art_s - sep_art_bc, apex_art_s)
        apex_r_art = jnp.where(art_left[None, None, :], apex_art_s, apex_art_s + sep_art_bc)
        sigma_l_art = jnp.where(art_left[None, None, :], s_art_bc, sb_art)
        sigma_r_art = jnp.where(art_left[None, None, :], sb_art, s_art_bc)
        alpha_l_art = jnp.where(art_left[None, None, :], a_art_bc, ab_art)
        alpha_r_art = jnp.where(art_left[None, None, :], ab_art, a_art_bc)
        area_l_art = jnp.where(art_left[None, None, :], area_artefact, area_dom_art)
        area_r_art = jnp.where(art_left[None, None, :], area_dom_art, area_artefact)

        apex_l = apex_l.at[:, :, artefact_idx].set(apex_l_art)
        apex_r = apex_r.at[:, :, artefact_idx].set(apex_r_art)
        sigma_l = sigma_l.at[:, :, artefact_idx].set(sigma_l_art)
        sigma_r = sigma_r.at[:, :, artefact_idx].set(sigma_r_art)
        alpha_l = alpha_l.at[:, :, artefact_idx].set(alpha_l_art)
        alpha_r = alpha_r.at[:, :, artefact_idx].set(alpha_r_art)
        area_l = area_l.at[:, :, artefact_idx].set(area_l_art)
        area_r = area_r.at[:, :, artefact_idx].set(area_r_art)
        separation_out = separation_out.at[:, :, artefact_idx].set(sep_art_bc)

    if n_free > 0:
        n_free_int = int(free_idx.shape[0])
        sb_free = jnp.broadcast_to(sigma_base[:, None, free_idx], (n_total, n_trace, n_free_int))
        sr_free_bc = jnp.broadcast_to(sigma_r_free[:, None, :], (n_total, n_trace, n_free_int))
        ab_free = jnp.broadcast_to(alpha_base[:, None, free_idx], (n_total, n_trace, n_free_int))
        ar_free_bc = jnp.broadcast_to(alpha_r_free[:, None, :], (n_total, n_trace, n_free_int))
        area_total_free_s = jnp.asarray(samples["area_total_free"])
        area_frac_left = jnp.asarray(samples["area_frac_left_free"])
        apex_free = apex[:, :, free_idx]
        sep_free_bc = jnp.broadcast_to(separation_free[:, None, :], (n_total, n_trace, n_free_int))

        apex_l_free = apex_free - 0.5 * sep_free_bc
        apex_r_free = apex_free + 0.5 * sep_free_bc
        area_l_free = area_total_free_s * area_frac_left
        area_r_free = area_total_free_s * (1.0 - area_frac_left)

        apex_l = apex_l.at[:, :, free_idx].set(apex_l_free)
        apex_r = apex_r.at[:, :, free_idx].set(apex_r_free)
        sigma_l = sigma_l.at[:, :, free_idx].set(sb_free)
        sigma_r = sigma_r.at[:, :, free_idx].set(sr_free_bc)
        alpha_l = alpha_l.at[:, :, free_idx].set(ab_free)
        alpha_r = alpha_r.at[:, :, free_idx].set(ar_free_bc)
        area_l = area_l.at[:, :, free_idx].set(area_l_free)
        area_r = area_r.at[:, :, free_idx].set(area_r_free)
        separation_out = separation_out.at[:, :, free_idx].set(sep_free_bc)

    delta_l = alpha_l / jnp.sqrt(1.0 + alpha_l**2)
    xi_l = apex_l - sigma_l * delta_l * _SQRT_2_OVER_PI
    delta_r = alpha_r / jnp.sqrt(1.0 + alpha_r**2)
    xi_r = apex_r - sigma_r * delta_r * _SQRT_2_OVER_PI

    return {
        "sigma_base": sigma_base,
        "alpha_base": alpha_base,
        "trace_shift": trace_shift,
        "apex": apex,
        "baseline_slope": baseline_slope,
        "apex_l": apex_l,
        "apex_r": apex_r,
        "sigma_l": sigma_l,
        "sigma_r": sigma_r,
        "alpha_l": alpha_l,
        "alpha_r": alpha_r,
        "area_l": area_l,
        "area_r": area_r,
        "area_total": area_l + area_r,
        "xi_l": xi_l,
        "xi_r": xi_r,
        "separation": separation_out,
    }


# ---------------------------------------------------------------------------
# Summary parameter names (for ArviZ / posterior extraction)
# ---------------------------------------------------------------------------

SUMMARY_PARAMETER_NAMES = (
    "trace_shift",
    "apex",
    "log_w_left",
    "log_w_right",
    "log_w_left_2",
    "log_w_right_2",
    "log_separation_artefact",
    "log_separation_free",
    "area_l",
    "area_r",
    "area_total",
    "area_artefact_typical",
    "baseline_intercept",
    "baseline_slope",
    "sigma_y",
)

TRACE_PARAMETER_NAMES = SUMMARY_PARAMETER_NAMES
