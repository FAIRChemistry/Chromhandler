"""Split-normal chromatographic peak model.

Supports three peak modes:

- ``single``: one component per logical peak window.
- ``artefact_doublet``: dominant component plus an artefact component.
- ``free_doublet``: true two-component peak with a free area split.

Parameterization: the model samples ``(log_w_left, log_w_right)`` — log of the
left and right half-widths at half-maximum (HWHM).  These are orthogonal and
directly observable, giving NUTS well-conditioned geometry.  The half-sigmas
``(sl, sr)`` are recovered deterministically via ``sl = HWHM_left / sqrt(2*ln2)``.

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

from .types import ModelHyperparams  # noqa: TC001 — used at runtime in model() signature

numpyro.set_host_device_count(8)

_HWHM_FACTOR: float = float(jnp.sqrt(2.0 * jnp.log(2.0)))


# ---------------------------------------------------------------------------
# Half-width → (s_left, s_right) conversion
# ---------------------------------------------------------------------------


def _halfwidths_to_split(
    log_w_left: jax.Array,
    log_w_right: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Convert log HWHM to (s_left, s_right): half-sigma for each side.

    s = HWHM / sqrt(2*ln2).  Works on any batch shape.
    """
    s_left = jnp.exp(log_w_left) / _HWHM_FACTOR
    s_right = jnp.exp(log_w_right) / _HWHM_FACTOR
    return s_left, s_right


# ---------------------------------------------------------------------------
# Split-normal (bi-normal) PDF
# ---------------------------------------------------------------------------


def log_split_normal_pdf(
    x: jax.Array,     # [n_trace, n_time]
    apex: jax.Array,  # [n_trace, n_comp]  — exact mode location
    sl: jax.Array,    # [n_trace, n_comp]  — left half-sigma
    sr: jax.Array,    # [n_trace, n_comp]  — right half-sigma
) -> jax.Array:       # [n_trace, n_comp, n_time]
    """Numerically stable log split-normal density.

    The mode is exactly at ``apex``.  Normalisation constant is
    ``2 / (sl + sr)`` — identical from both sides, so the density is
    continuous at the mode.

    Returns shape ``[n_trace, n_comp, n_time]``.
    """
    sl_s = jnp.maximum(sl, 1e-6)
    sr_s = jnp.maximum(sr, 1e-6)
    left_mask = x[:, None, :] <= apex[:, :, None]
    sigma = jnp.where(left_mask, sl_s[:, :, None], sr_s[:, :, None])
    z = (x[:, None, :] - apex[:, :, None]) / sigma
    log_norm = (
        jnp.log(2.0)
        - jnp.log(sl_s + sr_s)[:, :, None]
        - 0.5 * jnp.log(2.0 * jnp.pi)
    )
    return log_norm - 0.5 * z**2


def split_normal_pdf(
    x: jax.Array,
    apex: jax.Array,
    sl: jax.Array,
    sr: jax.Array,
) -> jax.Array:
    """Split-normal density. Same shape convention as ``log_split_normal_pdf``."""
    return jnp.exp(log_split_normal_pdf(x, apex, sl, sr))


def mixture_signal(
    x: jax.Array,     # [n_trace, n_time]
    apex: jax.Array,  # [n_trace, n_comp]
    sl: jax.Array,    # [n_trace, n_comp]
    sr: jax.Array,    # [n_trace, n_comp]
    area: jax.Array,  # [n_trace, n_comp]
) -> jax.Array:
    """Area-scaled split-normal mixture, summed over components.

    Returns shape ``[n_trace, n_time]``.
    """
    pdf = split_normal_pdf(x, apex, sl, sr)       # [n_trace, n_comp, n_time]
    return jnp.sum(area[:, :, None] * pdf, axis=1)


# ---------------------------------------------------------------------------
# Assembly helpers
# ---------------------------------------------------------------------------


def _stack_left_right(left: Any, right: Any) -> jax.Array:
    """Flatten left/right peak matrices to the mixture component axis."""
    left_arr = jnp.asarray(left)
    return jnp.stack([left_arr, jnp.asarray(right)], axis=-1).reshape(left_arr.shape[0], -1)


_CompArrays = tuple[
    jax.Array,  # apex_l  — left  component mode location  [n_trace, n_peak]
    jax.Array,  # apex_r  — right component mode location  [n_trace, n_peak]
    jax.Array,  # sl_l    — left  component left  half-sigma
    jax.Array,  # sl_r    — right component left  half-sigma
    jax.Array,  # sr_l    — left  component right half-sigma
    jax.Array,  # sr_r    — right component right half-sigma
    jax.Array,  # area_l
    jax.Array,  # area_r
]


def _assemble_nonfree(
    lr: _CompArrays,
    apex: Any,          # [n_trace, n_peak]
    sl_base: Any,       # [n_peak]
    sr_base: Any,       # [n_peak]
    area_dominant: Any, # [n_trace, n_nonfree]
    nonfree_idx: Any,   # [n_nonfree]
    n_trace: int,
) -> _CompArrays:
    """Fill left/right matrices for single and artefact_doublet (nonfree) peaks."""
    apex_l, apex_r, sl_l, sl_r, sr_l, sr_r, area_l, area_r = lr
    sl_nf = jnp.broadcast_to(sl_base[None, nonfree_idx], (n_trace, nonfree_idx.shape[0]))
    sr_nf = jnp.broadcast_to(sr_base[None, nonfree_idx], (n_trace, nonfree_idx.shape[0]))
    apex_nf = apex[:, nonfree_idx]
    apex_l = apex_l.at[:, nonfree_idx].set(apex_nf)
    apex_r = apex_r.at[:, nonfree_idx].set(apex_nf)
    sl_l = sl_l.at[:, nonfree_idx].set(sl_nf)
    sl_r = sl_r.at[:, nonfree_idx].set(sl_nf)
    sr_l = sr_l.at[:, nonfree_idx].set(sr_nf)
    sr_r = sr_r.at[:, nonfree_idx].set(sr_nf)
    area_l = area_l.at[:, nonfree_idx].set(area_dominant)
    return apex_l, apex_r, sl_l, sl_r, sr_l, sr_r, area_l, area_r


def _assemble_artefact(
    lr: _CompArrays,
    apex: Any,                  # [n_trace, n_peak]
    sl_base: Any,               # [n_peak]
    sr_base: Any,               # [n_peak]
    sl_art: Any,                # [n_artefact]
    sr_art: Any,                # [n_artefact]
    area_dominant: Any,         # [n_trace, n_nonfree]
    area_artefact: Any,         # [n_trace, n_artefact]
    separation_artefact: Any,   # [n_artefact]
    artefact_idx: Any,          # [n_artefact]
    nonfree_position: Any,      # [n_peak]
    artefact_side_v: Any,       # [n_peak] float: -1=left, 0=none, +1=right
    n_trace: int,
) -> _CompArrays:
    """Overwrite artefact_doublet columns with two-component geometry."""
    apex_l, apex_r, sl_l, sl_r, sr_l, sr_r, area_l, area_r = lr

    artefact_nonfree_idx = nonfree_position[artefact_idx]
    n_art = artefact_idx.shape[0]
    apex_art = apex[:, artefact_idx]
    sl_dom = jnp.broadcast_to(sl_base[None, artefact_idx], (n_trace, n_art))
    sr_dom = jnp.broadcast_to(sr_base[None, artefact_idx], (n_trace, n_art))
    sl_a = jnp.broadcast_to(sl_art[None, :], (n_trace, n_art))
    sr_a = jnp.broadcast_to(sr_art[None, :], (n_trace, n_art))
    area_dom = area_dominant[:, artefact_nonfree_idx]
    sep = jnp.broadcast_to(separation_artefact[None, :], (n_trace, n_art))

    art_left = artefact_side_v[artefact_idx] < 0.0
    apex_l_art = jnp.where(art_left[None, :], apex_art - sep, apex_art)
    apex_r_art = jnp.where(art_left[None, :], apex_art, apex_art + sep)
    sl_l_art = jnp.where(art_left[None, :], sl_a, sl_dom)
    sl_r_art = jnp.where(art_left[None, :], sl_dom, sl_a)
    sr_l_art = jnp.where(art_left[None, :], sr_a, sr_dom)
    sr_r_art = jnp.where(art_left[None, :], sr_dom, sr_a)
    area_l_art = jnp.where(art_left[None, :], area_artefact, area_dom)
    area_r_art = jnp.where(art_left[None, :], area_dom, area_artefact)

    apex_l = apex_l.at[:, artefact_idx].set(apex_l_art)
    apex_r = apex_r.at[:, artefact_idx].set(apex_r_art)
    sl_l = sl_l.at[:, artefact_idx].set(sl_l_art)
    sl_r = sl_r.at[:, artefact_idx].set(sl_r_art)
    sr_l = sr_l.at[:, artefact_idx].set(sr_l_art)
    sr_r = sr_r.at[:, artefact_idx].set(sr_r_art)
    area_l = area_l.at[:, artefact_idx].set(area_l_art)
    area_r = area_r.at[:, artefact_idx].set(area_r_art)
    return apex_l, apex_r, sl_l, sl_r, sr_l, sr_r, area_l, area_r


def _assemble_free(
    lr: _CompArrays,
    apex: Any,              # [n_trace, n_peak]
    sl_base: Any,           # [n_peak]
    sr_base: Any,           # [n_peak]
    sl_r_free: Any,         # [n_free] — left half-sigma for right free component
    sr_r_free: Any,         # [n_free] — right half-sigma for right free component
    area_total_free: Any,   # [n_trace, n_free]
    area_frac_left_free: Any,   # [n_trace, n_free]
    separation_free: Any,   # [n_free]
    free_idx: Any,          # [n_free]
    n_trace: int,
) -> _CompArrays:
    """Fill free_doublet columns with two-component geometry."""
    apex_l, apex_r, sl_l, sl_r, sr_l, sr_r, area_l, area_r = lr
    n_free = free_idx.shape[0]

    sl_dom = jnp.broadcast_to(sl_base[None, free_idx], (n_trace, n_free))
    sr_dom = jnp.broadcast_to(sr_base[None, free_idx], (n_trace, n_free))
    sl_rf = jnp.broadcast_to(sl_r_free[None, :], (n_trace, n_free))
    sr_rf = jnp.broadcast_to(sr_r_free[None, :], (n_trace, n_free))
    apex_free = apex[:, free_idx]
    apex_l_free = apex_free - 0.5 * separation_free[None, :]
    apex_r_free = apex_free + 0.5 * separation_free[None, :]
    area_l_free = area_total_free * area_frac_left_free
    area_r_free = area_total_free * (1.0 - area_frac_left_free)

    apex_l = apex_l.at[:, free_idx].set(apex_l_free)
    apex_r = apex_r.at[:, free_idx].set(apex_r_free)
    sl_l = sl_l.at[:, free_idx].set(sl_dom)
    sl_r = sl_r.at[:, free_idx].set(sl_rf)
    sr_l = sr_l.at[:, free_idx].set(sr_dom)
    sr_r = sr_r.at[:, free_idx].set(sr_rf)
    area_l = area_l.at[:, free_idx].set(area_l_free)
    area_r = area_r.at[:, free_idx].set(area_r_free)
    return apex_l, apex_r, sl_l, sl_r, sr_l, sr_r, area_l, area_r


# ---------------------------------------------------------------------------
# NumPyro model
# ---------------------------------------------------------------------------

_MODE_SINGLE = 0
_MODE_ARTEFACT_DOUBLET = 1
_MODE_FREE_DOUBLET = 2


def model(
    x: jax.Array,  # [n_trace, n_time]
    y: jax.Array | None,  # [n_trace, n_time] or None (prior predictive)
    # --- hyperparameters ---
    hyperparams: ModelHyperparams,
    # --- peak structure ---
    peak_mode_code: jax.Array,  # [n_peak]
    artefact_side: jax.Array,  # [n_peak]  int: -1=left, 0=none, +1=right
    artefact_peak_index: jax.Array,  # [n_artefact]
    free_peak_index: jax.Array,  # [n_free]
    nonfree_idx: jax.Array,  # [n_nonfree]
    nonfree_position: jax.Array,  # [n_peak]
    # --- peak priors ---
    apex_loc: jax.Array,  # [n_peak]
    trace_shift_scale: jax.Array,  # scalar
    apex_offset_scale: jax.Array,  # [n_peak]  — per-peak residual jitter std
    w_left_loc: jax.Array,  # [n_peak]
    w_left_scale: jax.Array,  # [n_peak]
    w_right_loc: jax.Array,  # [n_peak]
    w_right_scale: jax.Array,  # [n_peak]
    area_gaussian_pt: jax.Array,  # [n_trace, n_peak]
    area_art_shared: jax.Array,  # [n_artefact]
    snr_per_trace: jax.Array,  # [n_trace, n_peak]
    # --- peak window bounds ---
    window_lo: jax.Array,  # [n_peak]
    window_hi: jax.Array,  # [n_peak]
    # --- baseline priors ---
    baseline_intercept_loc: jax.Array,  # [n_trace]
    baseline_intercept_scale: jax.Array,  # [n_trace]
    baseline_slope_loc: jax.Array,  # [n_trace]
    baseline_slope_scale: jax.Array,  # [n_trace]
    # --- noise prior ---
    sigma_y_prior_loc: jax.Array,  # [n_trace]
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
        0.0,
        1.0,
    )
    area_log_sigma = hp.area_log_sigma_low_snr - snr_frac * (
        hp.area_log_sigma_low_snr - hp.area_log_sigma_high_snr
    )  # [n_trace, n_peak]

    # 3. Primary half-width priors (one per peak, shared across traces)
    w_left_log_scale = jnp.maximum(w_left_scale / jnp.maximum(w_left_loc, 1e-9), hp.w_prior_log_scale)
    w_right_log_scale = jnp.maximum(w_right_scale / jnp.maximum(w_right_loc, 1e-9), hp.w_prior_log_scale)
    log_w_left = numpyro.sample("log_w_left", dist.Normal(jnp.log(w_left_loc), w_left_log_scale))  # [n_peak]
    log_w_right = numpyro.sample(
        "log_w_right", dist.Normal(jnp.log(w_right_loc), w_right_log_scale)
    )  # [n_peak]
    sl_base, sr_base = _halfwidths_to_split(log_w_left, log_w_right)  # type: ignore[arg-type]  # [n_peak]

    # 4. Doublet second-component shape (artefact + free combined)
    #    doublet order: artefact peaks first, free peaks after
    n_doublet = n_artefact + n_free
    sl_art: jax.Array = jnp.zeros((0,), dtype=jnp.float32)
    sr_art: jax.Array = jnp.zeros((0,), dtype=jnp.float32)
    sl_r_free: jax.Array = jnp.zeros((0,), dtype=jnp.float32)
    sr_r_free: jax.Array = jnp.zeros((0,), dtype=jnp.float32)

    if n_doublet > 0:
        # Build doublet index and prior centres at Python level
        doublet_idx_np = np.concatenate(
            [  # type: ignore[arg-type]
                np.asarray(artefact_peak_index),
                np.asarray(free_peak_index),
            ]
        ).astype(np.int32)
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
        sl_2, sr_2 = _halfwidths_to_split(log_w_left_2, log_w_right_2)  # type: ignore[arg-type]

        sl_art = sl_2[:n_artefact]
        sr_art = sr_2[:n_artefact]
        sl_r_free = sl_2[n_artefact:]
        sr_r_free = sr_2[n_artefact:]

    # 5. Separation priors
    separation_artefact: jax.Array = jnp.zeros((0,), dtype=jnp.float32)
    separation_free: jax.Array = jnp.zeros((0,), dtype=jnp.float32)

    if n_artefact > 0:
        # Window room on the artefact side, minus trace_shift_scale safety margin.
        # artefact_side[artefact_idx]: +1 = right, -1 = left
        art_side = artefact_side[artefact_idx].astype(jnp.float32)  # [n_artefact]
        room = jnp.where(
            art_side > 0,
            window_hi[artefact_idx] - apex_loc[artefact_idx] - trace_shift_scale,
            apex_loc[artefact_idx] - window_lo[artefact_idx] - trace_shift_scale,
        )  # [n_artefact]
        sep_min = hp.art_sep_min_w_mult * jnp.minimum(
            w_left_loc[artefact_idx], w_right_loc[artefact_idx]
        )  # [n_artefact]
        room = jnp.maximum(room, sep_min * 2.0)
        log_separation_artefact = numpyro.sample(
            "log_separation_artefact",
            dist.Uniform(jnp.log(sep_min), jnp.log(room)),
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
    baseline_mid_scale = jnp.sqrt(baseline_intercept_scale**2 + (x_mid * baseline_slope_scale) ** 2)

    with numpyro.plate("traces", n_trace):
        trace_shift_raw = numpyro.sample("trace_shift_raw", dist.Normal(0.0, 1.0))
        baseline_intercept = numpyro.sample(
            "baseline_intercept", dist.Normal(baseline_mid_loc, baseline_mid_scale)
        )
        baseline_slope = numpyro.sample(
            "baseline_slope", dist.Normal(baseline_slope_loc, baseline_slope_scale)
        )
        sigma_y = numpyro.sample("sigma_y", dist.LogNormal(jnp.log(sigma_y_prior_loc), 0.5))

    trace_shift = trace_shift_scale * (trace_shift_raw - jnp.mean(trace_shift_raw))

    # Per-peak independent apex offset (non-centered parameterization)
    with numpyro.plate("traces_apex", n_trace, dim=-2):
        with numpyro.plate("peaks_apex", n_peak, dim=-1):
            apex_offset_raw = numpyro.sample("apex_offset_raw", dist.Normal(0.0, 1.0))
    # [n_trace, n_peak]
    apex_offset = apex_offset_scale[None, :] * apex_offset_raw
    apex = apex_loc[None, :] + trace_shift[:, None] + apex_offset  # [n_trace, n_peak]

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
    lr: _CompArrays = (
        jnp.zeros((n_trace, n_peak), dtype=jnp.float32),  # apex_l
        jnp.zeros((n_trace, n_peak), dtype=jnp.float32),  # apex_r
        jnp.zeros((n_trace, n_peak), dtype=jnp.float32),  # sl_l
        jnp.zeros((n_trace, n_peak), dtype=jnp.float32),  # sl_r
        jnp.zeros((n_trace, n_peak), dtype=jnp.float32),  # sr_l
        jnp.zeros((n_trace, n_peak), dtype=jnp.float32),  # sr_r
        jnp.zeros((n_trace, n_peak), dtype=jnp.float32),  # area_l
        jnp.zeros((n_trace, n_peak), dtype=jnp.float32),  # area_r
    )
    if n_nonfree > 0:
        lr = _assemble_nonfree(lr, apex, sl_base, sr_base, area_dominant, nonfree_idx, n_trace)
    if n_artefact > 0:
        lr = _assemble_artefact(
            lr,
            apex,
            sl_base,
            sr_base,
            sl_art,
            sr_art,
            area_dominant,
            area_artefact,
            separation_artefact,
            artefact_idx,
            nonfree_position,
            artefact_side_v,
            n_trace,
        )
    if n_free > 0:
        lr = _assemble_free(
            lr,
            apex,
            sl_base,
            sr_base,
            sl_r_free,
            sr_r_free,
            area_total_free,
            area_frac_left_free,
            separation_free,
            free_idx,
            n_trace,
        )

    apex_l, apex_r, sl_l, sl_r, sr_l, sr_r, area_l, area_r = lr

    apex_flat = _stack_left_right(apex_l, apex_r)
    sl_flat = _stack_left_right(sl_l, sl_r)
    sr_flat = _stack_left_right(sr_l, sr_r)
    area_flat = _stack_left_right(area_l, area_r)

    # 9. Baseline and likelihood
    baseline = baseline_intercept[:, None] + baseline_slope[:, None] * (x - x_mid)
    mu_y = mixture_signal(x, apex_flat, sl_flat, sr_flat, area_flat) + baseline
    if y is not None:
        finite_mask = jnp.isfinite(y)
        numpyro.sample(
            "y",
            dist.Normal(mu_y, sigma_y[:, None]).mask(finite_mask),
            obs=jnp.where(finite_mask, y, 0.0),
        )
    else:
        numpyro.sample("y", dist.Normal(mu_y, sigma_y[:, None]))


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
    apex_offset_scale_arr = jnp.asarray(model_inputs["apex_offset_scale"], dtype=jnp.float32)
    trace_shift_scale = float(jnp.asarray(model_inputs["trace_shift_scale"]).max())
    n_peak = int(apex_loc_arr.shape[0])
    n_artefact = int(artefact_idx.shape[0])
    n_free = int(free_idx.shape[0])
    n_nonfree = int(nonfree_idx_in.shape[0])
    n_doublet = n_artefact + n_free

    log_w_left = jnp.asarray(samples["log_w_left"])  # [n_total, n_peak]
    log_w_right = jnp.asarray(samples["log_w_right"])  # [n_total, n_peak]
    trace_shift_raw = jnp.asarray(samples["trace_shift_raw"])  # [n_total, n_trace]

    n_total, n_trace = trace_shift_raw.shape

    sl_base, sr_base = _halfwidths_to_split(log_w_left, log_w_right)  # [n_total, n_peak]
    trace_shift = trace_shift_scale * (
        trace_shift_raw - trace_shift_raw.mean(axis=1, keepdims=True)
    )  # [n_total, n_trace]
    apex_offset_raw = jnp.asarray(samples["apex_offset_raw"])  # [n_total, n_trace, n_peak]
    apex_offset = apex_offset_scale_arr[None, None, :] * apex_offset_raw
    apex = apex_loc_arr[None, None, :] + trace_shift[:, :, None] + apex_offset  # [n_total, n_trace, n_peak]

    # Second-component shapes
    sl_art: jax.Array = jnp.zeros((n_total, 0), dtype=jnp.float32)
    sr_art: jax.Array = jnp.zeros((n_total, 0), dtype=jnp.float32)
    sl_r_free: jax.Array = jnp.zeros((n_total, 0), dtype=jnp.float32)
    sr_r_free: jax.Array = jnp.zeros((n_total, 0), dtype=jnp.float32)
    separation_artefact: jax.Array = jnp.zeros((n_total, 0), dtype=jnp.float32)
    separation_free: jax.Array = jnp.zeros((n_total, 0), dtype=jnp.float32)

    if n_doublet > 0:
        log_w_left_2 = jnp.asarray(samples["log_w_left_2"])  # [n_total, n_doublet]
        log_w_right_2 = jnp.asarray(samples["log_w_right_2"])
        sl_2, sr_2 = _halfwidths_to_split(log_w_left_2, log_w_right_2)
        sl_art = sl_2[:, :n_artefact]
        sr_art = sr_2[:, :n_artefact]
        sl_r_free = sl_2[:, n_artefact:]
        sr_r_free = sr_2[:, n_artefact:]

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
        separation_free = jnp.exp(jnp.asarray(samples["log_separation_free"]))  # [n_total, n_free]

    baseline_slope = jnp.asarray(samples["baseline_slope"])  # [n_total, n_trace]

    # Left/right assembly: [n_total, n_trace, n_peak]
    apex_l: jax.Array = jnp.zeros((n_total, n_trace, n_peak), dtype=jnp.float32)
    apex_r: jax.Array = jnp.zeros((n_total, n_trace, n_peak), dtype=jnp.float32)
    sl_l: jax.Array = jnp.zeros((n_total, n_trace, n_peak), dtype=jnp.float32)
    sl_r: jax.Array = jnp.zeros((n_total, n_trace, n_peak), dtype=jnp.float32)
    sr_l: jax.Array = jnp.zeros((n_total, n_trace, n_peak), dtype=jnp.float32)
    sr_r: jax.Array = jnp.zeros((n_total, n_trace, n_peak), dtype=jnp.float32)
    area_l: jax.Array = jnp.zeros((n_total, n_trace, n_peak), dtype=jnp.float32)
    area_r: jax.Array = jnp.zeros((n_total, n_trace, n_peak), dtype=jnp.float32)
    separation_out: jax.Array = jnp.zeros((n_total, n_trace, n_peak), dtype=jnp.float32)

    if n_nonfree > 0:
        area_dominant = jnp.asarray(samples["area_dominant"])  # [n_total, n_trace, n_nonfree]
        sl_nf = jnp.broadcast_to(sl_base[:, None, nonfree_idx_in], (n_total, n_trace, n_nonfree))
        sr_nf = jnp.broadcast_to(sr_base[:, None, nonfree_idx_in], (n_total, n_trace, n_nonfree))
        apex_l = apex_l.at[:, :, nonfree_idx_in].set(apex[:, :, nonfree_idx_in])
        apex_r = apex_r.at[:, :, nonfree_idx_in].set(apex[:, :, nonfree_idx_in])
        sl_l = sl_l.at[:, :, nonfree_idx_in].set(sl_nf)
        sl_r = sl_r.at[:, :, nonfree_idx_in].set(sl_nf)
        sr_l = sr_l.at[:, :, nonfree_idx_in].set(sr_nf)
        sr_r = sr_r.at[:, :, nonfree_idx_in].set(sr_nf)
        area_l = area_l.at[:, :, nonfree_idx_in].set(area_dominant)

    if n_artefact > 0:
        artefact_nonfree_idx = nonfree_position[artefact_idx]
        n_art = int(artefact_idx.shape[0])
        apex_art_s = apex[:, :, artefact_idx]
        sl_dom = jnp.broadcast_to(sl_base[:, None, artefact_idx], (n_total, n_trace, n_art))
        sr_dom = jnp.broadcast_to(sr_base[:, None, artefact_idx], (n_total, n_trace, n_art))
        sl_a_bc = jnp.broadcast_to(sl_art[:, None, :], (n_total, n_trace, n_art))
        sr_a_bc = jnp.broadcast_to(sr_art[:, None, :], (n_total, n_trace, n_art))
        area_dom_art = jnp.asarray(samples["area_dominant"])[:, :, artefact_nonfree_idx]
        sep_art_bc = jnp.broadcast_to(separation_artefact[:, None, :], (n_total, n_trace, n_art))

        art_left = artefact_side_v[artefact_idx] < 0.0
        apex_l_art = jnp.where(art_left[None, None, :], apex_art_s - sep_art_bc, apex_art_s)
        apex_r_art = jnp.where(art_left[None, None, :], apex_art_s, apex_art_s + sep_art_bc)
        sl_l_art = jnp.where(art_left[None, None, :], sl_a_bc, sl_dom)
        sl_r_art = jnp.where(art_left[None, None, :], sl_dom, sl_a_bc)
        sr_l_art = jnp.where(art_left[None, None, :], sr_a_bc, sr_dom)
        sr_r_art = jnp.where(art_left[None, None, :], sr_dom, sr_a_bc)
        area_l_art = jnp.where(art_left[None, None, :], area_artefact, area_dom_art)
        area_r_art = jnp.where(art_left[None, None, :], area_dom_art, area_artefact)

        apex_l = apex_l.at[:, :, artefact_idx].set(apex_l_art)
        apex_r = apex_r.at[:, :, artefact_idx].set(apex_r_art)
        sl_l = sl_l.at[:, :, artefact_idx].set(sl_l_art)
        sl_r = sl_r.at[:, :, artefact_idx].set(sl_r_art)
        sr_l = sr_l.at[:, :, artefact_idx].set(sr_l_art)
        sr_r = sr_r.at[:, :, artefact_idx].set(sr_r_art)
        area_l = area_l.at[:, :, artefact_idx].set(area_l_art)
        area_r = area_r.at[:, :, artefact_idx].set(area_r_art)
        separation_out = separation_out.at[:, :, artefact_idx].set(sep_art_bc)

    if n_free > 0:
        n_free_int = int(free_idx.shape[0])
        sl_dom_free = jnp.broadcast_to(sl_base[:, None, free_idx], (n_total, n_trace, n_free_int))
        sr_dom_free = jnp.broadcast_to(sr_base[:, None, free_idx], (n_total, n_trace, n_free_int))
        sl_rf_bc = jnp.broadcast_to(sl_r_free[:, None, :], (n_total, n_trace, n_free_int))
        sr_rf_bc = jnp.broadcast_to(sr_r_free[:, None, :], (n_total, n_trace, n_free_int))
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
        sl_l = sl_l.at[:, :, free_idx].set(sl_dom_free)
        sl_r = sl_r.at[:, :, free_idx].set(sl_rf_bc)
        sr_l = sr_l.at[:, :, free_idx].set(sr_dom_free)
        sr_r = sr_r.at[:, :, free_idx].set(sr_rf_bc)
        area_l = area_l.at[:, :, free_idx].set(area_l_free)
        area_r = area_r.at[:, :, free_idx].set(area_r_free)
        separation_out = separation_out.at[:, :, free_idx].set(sep_free_bc)

    return {
        "sl_base": sl_base,
        "sr_base": sr_base,
        "trace_shift": trace_shift,
        "apex": apex,
        "baseline_slope": baseline_slope,
        "apex_l": apex_l,
        "apex_r": apex_r,
        "sl_l": sl_l,
        "sl_r": sl_r,
        "sr_l": sr_l,
        "sr_r": sr_r,
        "area_l": area_l,
        "area_r": area_r,
        "area_total": area_l + area_r,
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
