"""Split-normal chromatographic peak model.

Supports three peak modes:

- ``single``: one component per logical peak window.
- ``artefact_doublet``: dominant component plus an artefact component.
- ``free_doublet``: true two-component peak with a free area split.

Parameterization: the model samples ``(log_w_left, log_w_right)`` — log of the
left and right half-widths at half-maximum (HWHM).  These are orthogonal and
directly observable, giving NUTS well-conditioned geometry.  The half-sigmas
``(sigma_left, sigma_right)`` are recovered deterministically via
``sigma = HWHM / sqrt(2*ln2)``.

Location geometry: ``apex[t, p] = apex_loc[p] + trace_shift[t]``.

The model keeps all intermediate arrays as plain local variables to minimise
HMC leapfrog overhead.  Call :func:`compute_derived_quantities` after sampling
to reconstruct them from the raw posterior samples.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .types import ModelHyperparams  # noqa: TC001 — used at runtime in model() signature

numpyro.set_host_device_count(8)

_HWHM_FACTOR: float = float(jnp.sqrt(2.0 * jnp.log(2.0)))


def _w_log_scale(
    w_scale: jax.Array,
    w_loc: jax.Array,
    dt: jax.Array,
    n_valid: jax.Array,
) -> jax.Array:
    """Unified data-derived log-scale for every width prior.

    Takes the maximum of three principled lower bounds:

    - ``w_scale / w_loc``      — cross-trace FWHM variability (empirical).
    - ``dt / w_loc``           — Nyquist precision (can't know FWHM tighter than one sample period).
    - ``1 / sqrt(n_valid)``    — statistical pooling precision (mean of ``n_valid`` measurements).

    No free hyperparameters.  Works elementwise on ``[n_peak]``-shaped arrays
    or on sliced subsets (e.g. ``w_loc[artefact_idx]``).
    """
    w_loc_safe = jnp.maximum(w_loc, 1e-9)
    n_valid_safe = jnp.maximum(n_valid, 1.0)
    return jnp.maximum(
        jnp.maximum(w_scale / w_loc_safe, dt / w_loc_safe),
        1.0 / jnp.sqrt(n_valid_safe),
    )


# ---------------------------------------------------------------------------
# Half-width → (sigma_left, sigma_right) conversion
# ---------------------------------------------------------------------------


def _halfwidths_to_split(
    log_w_left: jax.Array,
    log_w_right: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Convert log HWHM to (sigma_left, sigma_right): half-sigma for each side of the mode.

    sigma = HWHM / sqrt(2*ln2).  Works on any batch shape.
    """
    sigma_left = jnp.exp(log_w_left) / _HWHM_FACTOR
    sigma_right = jnp.exp(log_w_right) / _HWHM_FACTOR
    return sigma_left, sigma_right


# ---------------------------------------------------------------------------
# Split-normal (bi-normal) PDF
# ---------------------------------------------------------------------------


def log_split_normal_pdf(
    x: jax.Array,  # [n_trace, n_time]
    apex: jax.Array,  # [n_trace, n_comp]  — exact mode location
    sigma_left: jax.Array,  # [n_trace, n_comp]  — half-sigma on the left side of the mode
    sigma_right: jax.Array,  # [n_trace, n_comp]  — half-sigma on the right side of the mode
) -> jax.Array:  # [n_trace, n_comp, n_time]
    """Numerically stable log split-normal density.

    The mode is exactly at ``apex``.  Normalisation constant is
    ``2 / (sigma_left + sigma_right)`` — identical from both sides, so the
    density is continuous at the mode.

    Returns shape ``[n_trace, n_comp, n_time]``.
    """
    sigma_left_safe = jnp.maximum(sigma_left, 1e-6)
    sigma_right_safe = jnp.maximum(sigma_right, 1e-6)
    left_mask = x[:, None, :] <= apex[:, :, None]
    sigma = jnp.where(
        left_mask, sigma_left_safe[:, :, None], sigma_right_safe[:, :, None]
    )
    z = (x[:, None, :] - apex[:, :, None]) / sigma
    log_norm = (
        jnp.log(2.0)
        - jnp.log(sigma_left_safe + sigma_right_safe)[:, :, None]
        - 0.5 * jnp.log(2.0 * jnp.pi)
    )
    return log_norm - 0.5 * z**2


def split_normal_pdf(
    x: jax.Array,
    apex: jax.Array,
    sigma_left: jax.Array,
    sigma_right: jax.Array,
) -> jax.Array:
    """Split-normal density. Same shape convention as ``log_split_normal_pdf``."""
    return jnp.exp(log_split_normal_pdf(x, apex, sigma_left, sigma_right))


def mixture_signal(
    x: jax.Array,  # [n_trace, n_time]
    apex: jax.Array,  # [n_trace, n_comp]
    sigma_left: jax.Array,  # [n_trace, n_comp]
    sigma_right: jax.Array,  # [n_trace, n_comp]
    area: jax.Array,  # [n_trace, n_comp]
) -> jax.Array:
    """Area-scaled split-normal mixture, summed over components.

    Returns shape ``[n_trace, n_time]``.
    """
    pdf = split_normal_pdf(x, apex, sigma_left, sigma_right)  # [n_trace, n_comp, n_time]
    return jnp.sum(area[:, :, None] * pdf, axis=1)


# ---------------------------------------------------------------------------
# Assembly helpers
# ---------------------------------------------------------------------------


def _stack_left_right(left: Any, right: Any) -> jax.Array:
    """Flatten left/right peak matrices to the mixture component axis."""
    left_arr = jnp.asarray(left)
    return jnp.stack([left_arr, jnp.asarray(right)], axis=-1).reshape(left_arr.shape[0], -1)


@dataclass(frozen=True)
class SplitGeometry:
    """Per-component split-normal geometry, shape ``[..., n_trace, n_peak]``.

    Each split-normal peak has a left and a right component (for doublets;
    singles store identical values in both).  Each component in turn has two
    half-sigmas — one for the left side of its mode, one for the right —
    encoding split-normal asymmetry.  That's two orthogonal left/right axes,
    which the field names spell out rather than hiding in positional letters.
    """

    apex_left:            jax.Array  # mode location of the left  component
    apex_right:           jax.Array  # mode location of the right component
    sigma_left_of_left:   jax.Array  # left  half-sigma of the left  component
    sigma_left_of_right:  jax.Array  # left  half-sigma of the right component
    sigma_right_of_left:  jax.Array  # right half-sigma of the left  component
    sigma_right_of_right: jax.Array  # right half-sigma of the right component
    area_left:            jax.Array
    area_right:           jax.Array

    def flatten_to_mixture(
        self,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Interleave left/right components along the mixture-component axis.

        Returns ``(apex, sigma_left, sigma_right, area)`` each shaped
        ``[n_trace, 2*n_peak]``, ready for :func:`mixture_signal`.
        """
        return (
            _stack_left_right(self.apex_left, self.apex_right),
            _stack_left_right(self.sigma_left_of_left, self.sigma_left_of_right),
            _stack_left_right(self.sigma_right_of_left, self.sigma_right_of_right),
            _stack_left_right(self.area_left, self.area_right),
        )

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> SplitGeometry:
        """Build a zero-initialised geometry with every field set to ``jnp.zeros(shape)``."""
        z = jnp.zeros(shape, dtype=jnp.float32)
        return cls(z, z, z, z, z, z, z, z)


def _broadcast_to_trace_axis(
    per_peak: Any,  # [..., n_peak_slice]
    target_shape: tuple[int, ...],  # [..., n_trace, n_peak_slice]
) -> jax.Array:
    """Insert a trace axis at ``axis=-2`` and broadcast to ``target_shape``.

    Works for any leading-batch shape: ``per_peak`` of shape ``[n_slice]``
    broadcasts to ``[n_trace, n_slice]``; ``[n_total, n_slice]`` broadcasts
    to ``[n_total, n_trace, n_slice]``.  Single source of truth shared by
    :func:`model` (2D geometry) and :func:`compute_derived_quantities` (3D).
    """
    return jnp.broadcast_to(jnp.expand_dims(per_peak, axis=-2), target_shape)


def _assemble_nonfree(
    geom: SplitGeometry,
    apex: Any,  # [..., n_trace, n_peak]
    primary_sigma_left: Any,  # [..., n_peak]
    primary_sigma_right: Any,  # [..., n_peak]
    area_dominant: Any,  # [..., n_trace, n_nonfree]
    nonfree_idx: Any,  # [n_nonfree]
) -> SplitGeometry:
    """Fill left/right components for single and artefact_doublet (nonfree) peaks.

    Singles share the same apex and sigmas across both "components" — the
    left-component and right-component values coincide, collapsing the pair
    to a single split-normal peak.  Only ``area_left`` is written;
    ``area_right`` is intentionally left at zero so the right "component"
    contributes nothing to the mixture signal for singles.  Artefact doublets
    overwrite both areas later in :func:`_assemble_artefact`.
    """
    target_shape = (*geom.apex_left.shape[:-1], nonfree_idx.shape[0])
    primary_sigma_left_bcast = _broadcast_to_trace_axis(
        primary_sigma_left[..., nonfree_idx], target_shape
    )
    primary_sigma_right_bcast = _broadcast_to_trace_axis(
        primary_sigma_right[..., nonfree_idx], target_shape
    )
    apex_nonfree = apex[..., nonfree_idx]
    return replace(
        geom,
        apex_left=geom.apex_left.at[..., nonfree_idx].set(apex_nonfree),
        apex_right=geom.apex_right.at[..., nonfree_idx].set(apex_nonfree),
        sigma_left_of_left=geom.sigma_left_of_left.at[..., nonfree_idx].set(primary_sigma_left_bcast),
        sigma_left_of_right=geom.sigma_left_of_right.at[..., nonfree_idx].set(primary_sigma_left_bcast),
        sigma_right_of_left=geom.sigma_right_of_left.at[..., nonfree_idx].set(primary_sigma_right_bcast),
        sigma_right_of_right=geom.sigma_right_of_right.at[..., nonfree_idx].set(primary_sigma_right_bcast),
        area_left=geom.area_left.at[..., nonfree_idx].set(area_dominant),
    )


def _assemble_artefact(
    geom: SplitGeometry,
    apex: Any,  # [..., n_trace, n_peak]
    primary_sigma_left: Any,  # [..., n_peak]
    primary_sigma_right: Any,  # [..., n_peak]
    artefact_sigma_left: Any,  # [..., n_artefact]
    artefact_sigma_right: Any,  # [..., n_artefact]
    area_dominant: Any,  # [..., n_trace, n_nonfree]
    area_artefact: Any,  # [..., n_trace, n_artefact]
    separation_artefact: Any,  # [..., n_artefact]
    artefact_idx: Any,  # [n_artefact]
    nonfree_position: Any,  # [n_peak]
    artefact_side_sign: Any,  # [n_peak] float: -1=left, 0=none, +1=right
) -> SplitGeometry:
    """Overwrite artefact_doublet columns with two-component geometry.

    Primary on one side, artefact on the other — ``artefact_side_sign`` selects
    which.  Artefact sigmas are symmetric (left-half-sigma == right-half-sigma);
    the primary keeps its measured asymmetry.
    """
    target_shape = (*geom.apex_left.shape[:-1], artefact_idx.shape[0])
    apex_artefact = apex[..., artefact_idx]
    primary_sigma_left_bcast = _broadcast_to_trace_axis(
        primary_sigma_left[..., artefact_idx], target_shape
    )
    primary_sigma_right_bcast = _broadcast_to_trace_axis(
        primary_sigma_right[..., artefact_idx], target_shape
    )
    artefact_sigma_left_bcast = _broadcast_to_trace_axis(artefact_sigma_left, target_shape)
    artefact_sigma_right_bcast = _broadcast_to_trace_axis(artefact_sigma_right, target_shape)
    area_primary_at_artefacts = area_dominant[..., nonfree_position[artefact_idx]]
    separation_bcast = _broadcast_to_trace_axis(separation_artefact, target_shape)

    artefact_is_left = jnp.broadcast_to(
        artefact_side_sign[artefact_idx] < 0.0, target_shape
    )
    new_apex_left = jnp.where(artefact_is_left, apex_artefact - separation_bcast, apex_artefact)
    new_apex_right = jnp.where(artefact_is_left, apex_artefact, apex_artefact + separation_bcast)
    new_sigma_left_of_left = jnp.where(
        artefact_is_left, artefact_sigma_left_bcast, primary_sigma_left_bcast
    )
    new_sigma_left_of_right = jnp.where(
        artefact_is_left, primary_sigma_left_bcast, artefact_sigma_left_bcast
    )
    new_sigma_right_of_left = jnp.where(
        artefact_is_left, artefact_sigma_right_bcast, primary_sigma_right_bcast
    )
    new_sigma_right_of_right = jnp.where(
        artefact_is_left, primary_sigma_right_bcast, artefact_sigma_right_bcast
    )
    new_area_left = jnp.where(artefact_is_left, area_artefact, area_primary_at_artefacts)
    new_area_right = jnp.where(artefact_is_left, area_primary_at_artefacts, area_artefact)

    return replace(
        geom,
        apex_left=geom.apex_left.at[..., artefact_idx].set(new_apex_left),
        apex_right=geom.apex_right.at[..., artefact_idx].set(new_apex_right),
        sigma_left_of_left=geom.sigma_left_of_left.at[..., artefact_idx].set(new_sigma_left_of_left),
        sigma_left_of_right=geom.sigma_left_of_right.at[..., artefact_idx].set(new_sigma_left_of_right),
        sigma_right_of_left=geom.sigma_right_of_left.at[..., artefact_idx].set(new_sigma_right_of_left),
        sigma_right_of_right=geom.sigma_right_of_right.at[..., artefact_idx].set(new_sigma_right_of_right),
        area_left=geom.area_left.at[..., artefact_idx].set(new_area_left),
        area_right=geom.area_right.at[..., artefact_idx].set(new_area_right),
    )


def _assemble_free(
    geom: SplitGeometry,
    apex: Any,  # [..., n_trace, n_peak]
    primary_sigma_left: Any,  # [..., n_peak]
    primary_sigma_right: Any,  # [..., n_peak]
    free_second_sigma_left: Any,  # [..., n_free] — left  half-sigma of the right (2nd) free component
    free_second_sigma_right: Any,  # [..., n_free] — right half-sigma of the right (2nd) free component
    area_total_free: Any,  # [..., n_trace, n_free]
    area_frac_left_free: Any,  # [..., n_trace, n_free]
    separation_free: Any,  # [..., n_free]
    free_idx: Any,  # [n_free]
) -> SplitGeometry:
    """Fill free_doublet columns with two-component geometry.

    Primary keeps its left-component slot; the 2nd (right) component carries
    ``free_second_sigma_*`` — a separately-sampled pair of half-sigmas.
    """
    target_shape = (*geom.apex_left.shape[:-1], free_idx.shape[0])
    primary_sigma_left_bcast = _broadcast_to_trace_axis(
        primary_sigma_left[..., free_idx], target_shape
    )
    primary_sigma_right_bcast = _broadcast_to_trace_axis(
        primary_sigma_right[..., free_idx], target_shape
    )
    free_second_sigma_left_bcast = _broadcast_to_trace_axis(free_second_sigma_left, target_shape)
    free_second_sigma_right_bcast = _broadcast_to_trace_axis(free_second_sigma_right, target_shape)
    apex_free = apex[..., free_idx]
    half_sep = 0.5 * _broadcast_to_trace_axis(separation_free, target_shape)

    return replace(
        geom,
        apex_left=geom.apex_left.at[..., free_idx].set(apex_free - half_sep),
        apex_right=geom.apex_right.at[..., free_idx].set(apex_free + half_sep),
        sigma_left_of_left=geom.sigma_left_of_left.at[..., free_idx].set(primary_sigma_left_bcast),
        sigma_left_of_right=geom.sigma_left_of_right.at[..., free_idx].set(free_second_sigma_left_bcast),
        sigma_right_of_left=geom.sigma_right_of_left.at[..., free_idx].set(primary_sigma_right_bcast),
        sigma_right_of_right=geom.sigma_right_of_right.at[..., free_idx].set(free_second_sigma_right_bcast),
        area_left=geom.area_left.at[..., free_idx].set(area_total_free * area_frac_left_free),
        area_right=geom.area_right.at[..., free_idx].set(area_total_free * (1.0 - area_frac_left_free)),
    )


# ---------------------------------------------------------------------------
# Prior-sampling sub-functions
#
# Each helper owns one coherent chunk of the prior and returns narrowly-typed
# outputs.  ``numpyro.sample(...)`` call strings are part of the model's
# external API — tests, visualize.py, and fitter.py all read posterior samples
# by these names — so they stay byte-identical.
# ---------------------------------------------------------------------------


class _PerTraceParams(NamedTuple):
    """Per-trace scalars sampled together in the ``"traces"`` plate.

    Field types are ``Any`` because ``numpyro.sample`` returns ``ArrayLike``,
    not ``jax.Array`` — declaring the fields narrower forces spurious casts.
    """

    trace_shift_raw: Any   # [n_trace]  — unit-Normal, scaled later
    baseline_intercept: Any
    baseline_slope: Any
    sigma_y: Any


def _sample_log_w(
    name: str,
    log_centre: jax.Array,
    w_scale: jax.Array,
    w_loc_for_scale: jax.Array,
    w_min: jax.Array,
    w_max: jax.Array,
    dt: jax.Array,
    n_valid: jax.Array,
) -> jax.Array:
    """Sample one log-HWHM from a truncated-normal on ``[log w_min, log w_max]``.

    Centre and scale are computed upstream; this helper owns the uniform
    truncation rule and the ``_w_log_scale`` combination so the five width
    priors in this module share a single shape.
    """
    return numpyro.sample(  # type: ignore[return-value]
        name,
        dist.TruncatedNormal(
            log_centre,
            _w_log_scale(w_scale, w_loc_for_scale, dt, n_valid),
            low=jnp.log(w_min),
            high=jnp.log(w_max),
        ),
    )


def _sample_primary_widths(
    w_left_loc: jax.Array,
    w_left_scale: jax.Array,
    w_right_loc: jax.Array,
    w_right_scale: jax.Array,
    w_min: jax.Array,
    w_max: jax.Array,
    dt: jax.Array,
    n_valid: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Sample the primary component's left/right HWHM priors.

    Truncated to [w_min, w_max] — Nyquist floor and window-size ceiling.
    Returns ``(primary_sigma_left, primary_sigma_right)``, both shape ``[n_peak]``.
    """
    log_w_left = _sample_log_w(
        "log_w_left", jnp.log(w_left_loc), w_left_scale, w_left_loc,
        w_min, w_max, dt, n_valid,
    )
    log_w_right = _sample_log_w(
        "log_w_right", jnp.log(w_right_loc), w_right_scale, w_right_loc,
        w_min, w_max, dt, n_valid,
    )
    return _halfwidths_to_split(log_w_left, log_w_right)


def _sample_artefact_width(
    w_left_loc: jax.Array,
    w_left_scale: jax.Array,
    w_right_loc: jax.Array,
    w_right_scale: jax.Array,
    w_min: jax.Array,
    w_max: jax.Array,
    dt: jax.Array,
    n_valid: jax.Array,
    artefact_idx: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sample the artefact component's (symmetric) HWHM prior.

    HWHM centre: geometric mean of ``w_min`` and the primary envelope FWHM —
    the max-entropy choice for a log-scale parameter bounded in
    ``[w_min, w_primary]``, consistent across resolved / overlapping cases.

    Returns ``(sigma_left, sigma_right, log_w_art)``.  The two sigmas are
    identical (split-normal is symmetric for artefacts) but returned separately
    to match the downstream interface.
    """
    w_mean_primary = 0.5 * (w_left_loc[artefact_idx] + w_right_loc[artefact_idx])
    w_art_centre = jnp.sqrt(w_min[artefact_idx] * w_mean_primary)
    w_scale_art = 0.5 * (w_left_scale[artefact_idx] + w_right_scale[artefact_idx])
    log_w_art = _sample_log_w(
        "log_w_art", jnp.log(w_art_centre), w_scale_art, w_mean_primary,
        w_min[artefact_idx], w_max[artefact_idx], dt[artefact_idx], n_valid[artefact_idx],
    )
    sigma_art = jnp.exp(log_w_art) / _HWHM_FACTOR
    return sigma_art, sigma_art, log_w_art


def _sample_free_second_widths(
    w_left_loc: jax.Array,
    w_left_scale: jax.Array,
    w_right_loc: jax.Array,
    w_right_scale: jax.Array,
    w_min: jax.Array,
    w_max: jax.Array,
    dt: jax.Array,
    n_valid: jax.Array,
    free_idx: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Sample the 2nd (right) component HWHMs of a free doublet.

    Centre at ``sqrt(w_min * w_loc)`` — same auxiliary-component rule as the
    artefact — and truncate to ``[w_min, w_max]`` like the primary.
    """
    w_left_loc_free = w_left_loc[free_idx]
    w_right_loc_free = w_right_loc[free_idx]
    w_min_free = w_min[free_idx]
    w_max_free = w_max[free_idx]
    dt_free = dt[free_idx]
    n_valid_free = n_valid[free_idx]
    log_w_left_2 = _sample_log_w(
        "log_w_left_2", jnp.log(jnp.sqrt(w_min_free * w_left_loc_free)),
        w_left_scale[free_idx], w_left_loc_free,
        w_min_free, w_max_free, dt_free, n_valid_free,
    )
    log_w_right_2 = _sample_log_w(
        "log_w_right_2", jnp.log(jnp.sqrt(w_min_free * w_right_loc_free)),
        w_right_scale[free_idx], w_right_loc_free,
        w_min_free, w_max_free, dt_free, n_valid_free,
    )
    return _halfwidths_to_split(log_w_left_2, log_w_right_2)


def _sample_artefact_separation(
    log_w_art: jax.Array,
    artefact_side: jax.Array,
    apex_loc: jax.Array,
    window_lo: jax.Array,
    window_hi: jax.Array,
    trace_shift_scale: jax.Array,
    artefact_idx: jax.Array,
) -> jax.Array:
    """Sample artefact apex separation, bounded by window room and HWHM.

    Identifiability lower bound ``sep_min = exp(log_w_art)`` — the artefact is
    unresolvable if it sits closer to the primary than its own HWHM.  Clamped
    so ``sep_min`` can't force the apex outside the window.

    Returns ``separation_artefact`` shape ``[n_artefact]``.
    """
    art_side = artefact_side[artefact_idx].astype(jnp.float32)
    room = jnp.where(
        art_side > 0,
        window_hi[artefact_idx] - apex_loc[artefact_idx] - trace_shift_scale,
        apex_loc[artefact_idx] - window_lo[artefact_idx] - trace_shift_scale,
    )
    sep_min = jnp.minimum(jnp.exp(log_w_art), room * 0.5)
    log_separation_artefact = numpyro.sample(
        "log_separation_artefact",
        dist.Uniform(jnp.log(sep_min), jnp.log(room)),
    )
    return jnp.exp(log_separation_artefact)  # type: ignore[return-value]


def _sample_free_separation(
    w_left_loc: jax.Array,
    free_idx: jax.Array,
    free_sep_log_sigma: float,
) -> jax.Array:
    """Sample free-doublet apex separation around twice the primary left HWHM."""
    sep_loc_free = 2.0 * w_left_loc[free_idx]
    log_separation_free = numpyro.sample(
        "log_separation_free",
        dist.Normal(jnp.log(sep_loc_free), free_sep_log_sigma),
    )
    return jnp.exp(log_separation_free)  # type: ignore[return-value]


def _sample_per_trace_params(
    n_trace: int,
    baseline_mid_loc: jax.Array,
    baseline_mid_scale: jax.Array,
    baseline_slope_loc: jax.Array,
    baseline_slope_scale: jax.Array,
    sigma_y_prior_loc: jax.Array,
) -> _PerTraceParams:
    """Plate-sample the per-trace scalars (shift, baseline, noise)."""
    with numpyro.plate("traces", n_trace):
        trace_shift_raw = numpyro.sample("trace_shift_raw", dist.Normal(0.0, 1.0))
        baseline_intercept = numpyro.sample(
            "baseline_intercept", dist.Normal(baseline_mid_loc, baseline_mid_scale)
        )
        baseline_slope = numpyro.sample(
            "baseline_slope", dist.Normal(baseline_slope_loc, baseline_slope_scale)
        )
        sigma_y = numpyro.sample("sigma_y", dist.LogNormal(jnp.log(sigma_y_prior_loc), 0.5))
    return _PerTraceParams(trace_shift_raw, baseline_intercept, baseline_slope, sigma_y)


def _sample_apex_offsets(
    n_trace: int,
    n_peak: int,
    apex_offset_scale: jax.Array,
) -> jax.Array:
    """Non-centred per-peak apex jitter, returned scaled.

    Shape ``[n_trace, n_peak]``.
    """
    with numpyro.plate("traces_apex", n_trace, dim=-2):
        with numpyro.plate("peaks_apex", n_peak, dim=-1):
            apex_offset_raw = numpyro.sample("apex_offset_raw", dist.Normal(0.0, 1.0))
    return apex_offset_scale[None, :] * apex_offset_raw


def _sample_nonfree_areas(
    n_trace: int,
    n_nonfree: int,
    area_gaussian_pt_nonfree: jax.Array,
    area_log_sigma: float,
) -> jax.Array:
    """Plate-sample dominant-component areas for single & artefact_doublet peaks."""
    with numpyro.plate("traces_nonfree", n_trace, dim=-2):
        with numpyro.plate("nonfree_peaks", n_nonfree, dim=-1):
            area_dominant = numpyro.sample(
                "area_dominant",
                dist.LogNormal(
                    jnp.log(jnp.maximum(area_gaussian_pt_nonfree, 1e-6)),
                    area_log_sigma,
                ),
            )
    return area_dominant  # type: ignore[return-value]


def _sample_artefact_areas(
    n_trace: int,
    n_artefact: int,
    area_art_shared: jax.Array,
    area_art_log_sigma: float,
    area_art_trace_log_scale: float,
) -> jax.Array:
    """Hierarchical artefact area: shared mean + non-centred per-trace offset.

    Non-centred parameterisation keeps NUTS geometry well-conditioned when
    traces are consistent (``area_art_trace_log_scale`` small → offsets weakly
    identified from data, but raw ~ Normal(0, 1) is always well-conditioned).

    Returns ``area_artefact`` shape ``[n_trace, n_artefact]``.
    """
    log_area_art_mean = numpyro.sample(
        "log_area_art_mean",
        dist.Normal(jnp.log(jnp.maximum(area_art_shared, 1e-6)), area_art_log_sigma),
    )
    with numpyro.plate("traces_art", n_trace, dim=-2):
        with numpyro.plate("artefact_peaks_art", n_artefact, dim=-1):
            log_area_art_raw = numpyro.sample("log_area_art_raw", dist.Normal(0.0, 1.0))
    return jnp.exp(log_area_art_mean[None, :] + area_art_trace_log_scale * log_area_art_raw)


def _sample_free_areas(
    n_trace: int,
    n_free: int,
    area_gaussian_pt_free: jax.Array,
    area_log_sigma: float,
) -> tuple[jax.Array, jax.Array]:
    """Plate-sample total area and left-fraction for free-doublet peaks."""
    with numpyro.plate("traces_free", n_trace, dim=-2):
        with numpyro.plate("free_peaks", n_free, dim=-1):
            area_total_free = numpyro.sample(
                "area_total_free",
                dist.LogNormal(
                    jnp.log(jnp.maximum(area_gaussian_pt_free, 1e-6)),
                    area_log_sigma,
                ),
            )
            area_frac_left_free = numpyro.sample(
                "area_frac_left_free",
                dist.Beta(2.0, 2.0),
            )
    return area_total_free, area_frac_left_free  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# NumPyro model
# ---------------------------------------------------------------------------

_MODE_SINGLE = 0
_MODE_ARTEFACT_DOUBLET = 1
_MODE_FREE_DOUBLET = 2


def model(
    # --- Data ---
    x: jax.Array,  # [n_trace, n_time]
    y: jax.Array | None,  # [n_trace, n_time] or None (prior predictive)
    # --- Hyperparameters ---
    hyperparams: ModelHyperparams,
    # --- Peak structure indices ---
    peak_mode_code: jax.Array,  # [n_peak]
    artefact_side: jax.Array,  # [n_peak]  int: -1=left, 0=none, +1=right
    artefact_peak_index: jax.Array,  # [n_artefact]
    free_peak_index: jax.Array,  # [n_free]
    nonfree_idx: jax.Array,  # [n_nonfree]
    nonfree_position: jax.Array,  # [n_peak]
    # --- Location priors ---
    apex_loc: jax.Array,  # [n_peak]
    trace_shift_scale: jax.Array,  # scalar
    apex_offset_scale: jax.Array,  # [n_peak]  — per-peak residual jitter std
    # --- Width priors ---
    w_left_loc: jax.Array,  # [n_peak]
    w_left_scale: jax.Array,  # [n_peak]
    w_right_loc: jax.Array,  # [n_peak]
    w_right_scale: jax.Array,  # [n_peak]
    w_min: jax.Array,  # [n_peak]  — Nyquist-like HWHM lower bound (per peak window)
    w_max: jax.Array,  # [n_peak]  — geometry-derived HWHM upper bound (per peak window)
    dt: jax.Array,  # [n_peak]  — median sampling interval per peak window
    n_valid: jax.Array,  # [n_peak]  — valid-trace count per peak window
    # --- Area priors ---
    area_gaussian_pt: jax.Array,  # [n_trace, n_peak]
    area_art_shared: jax.Array,  # [n_artefact]
    # --- Window bounds ---
    window_lo: jax.Array,  # [n_peak]
    window_hi: jax.Array,  # [n_peak]
    # --- Baseline priors ---
    baseline_intercept_loc: jax.Array,  # [n_trace]
    baseline_intercept_scale: jax.Array,  # [n_trace]
    baseline_slope_loc: jax.Array,  # [n_trace]
    baseline_slope_scale: jax.Array,  # [n_trace]
    # --- Noise prior ---
    sigma_y_prior_loc: jax.Array,  # [n_trace]
) -> None:
    """Bayesian skew-normal peak model using (log_w_left, log_w_right) parameterization.

    All input arrays are pre-validated float32 by ``Fitter._prepare_model_inputs()``.
    """
    # --- Shape constants ---
    n_trace, _ = x.shape
    n_peak = int(apex_loc.shape[0])
    n_artefact = int(artefact_peak_index.shape[0])
    n_free = int(free_peak_index.shape[0])
    n_nonfree = int(nonfree_idx.shape[0])

    artefact_side_sign = artefact_side.astype(jnp.float32)
    artefact_idx = artefact_peak_index
    free_idx = free_peak_index
    hp = hyperparams

    # --- Width priors ---
    primary_sigma_left, primary_sigma_right = _sample_primary_widths(
        w_left_loc, w_left_scale, w_right_loc, w_right_scale,
        w_min, w_max, dt, n_valid,
    )  # [n_peak], [n_peak]

    # Placeholder zero-arrays let pyright see every name is bound before the
    # `_assemble_*` calls read them; each guard below overwrites the relevant
    # subset for its peak mode.
    artefact_sigma_left: jax.Array = jnp.zeros((0,), dtype=jnp.float32)
    artefact_sigma_right: jax.Array = jnp.zeros((0,), dtype=jnp.float32)
    free_second_sigma_left: jax.Array = jnp.zeros((0,), dtype=jnp.float32)
    free_second_sigma_right: jax.Array = jnp.zeros((0,), dtype=jnp.float32)
    log_w_art: jax.Array = jnp.zeros((0,), dtype=jnp.float32)
    separation_artefact: jax.Array = jnp.zeros((0,), dtype=jnp.float32)
    separation_free: jax.Array = jnp.zeros((0,), dtype=jnp.float32)

    if n_artefact > 0:
        artefact_sigma_left, artefact_sigma_right, log_w_art = _sample_artefact_width(
            w_left_loc, w_left_scale, w_right_loc, w_right_scale,
            w_min, w_max, dt, n_valid, artefact_idx,
        )
        separation_artefact = _sample_artefact_separation(
            log_w_art, artefact_side, apex_loc,
            window_lo, window_hi, trace_shift_scale, artefact_idx,
        )

    if n_free > 0:
        free_second_sigma_left, free_second_sigma_right = _sample_free_second_widths(
            w_left_loc, w_left_scale, w_right_loc, w_right_scale,
            w_min, w_max, dt, n_valid, free_idx,
        )
        separation_free = _sample_free_separation(
            w_left_loc, free_idx, hp.free_sep_log_sigma,
        )

    # --- Per-trace parameters ---
    window_midpoint_x = 0.5 * (jnp.min(window_lo) + jnp.max(window_hi))
    baseline_mid_loc = baseline_intercept_loc + baseline_slope_loc * window_midpoint_x
    baseline_mid_scale = jnp.sqrt(
        baseline_intercept_scale**2 + (window_midpoint_x * baseline_slope_scale) ** 2
    )

    trace_params = _sample_per_trace_params(
        n_trace, baseline_mid_loc, baseline_mid_scale,
        baseline_slope_loc, baseline_slope_scale, sigma_y_prior_loc,
    )
    trace_shift = trace_shift_scale * (
        trace_params.trace_shift_raw - jnp.mean(trace_params.trace_shift_raw)
    )
    apex_offset = _sample_apex_offsets(n_trace, n_peak, apex_offset_scale)
    apex = apex_loc[None, :] + trace_shift[:, None] + apex_offset  # [n_trace, n_peak]

    # --- Area priors ---
    area_dominant: jax.Array = jnp.zeros((n_trace, 0), dtype=jnp.float32)
    area_artefact: jax.Array = jnp.zeros((n_trace, 0), dtype=jnp.float32)
    area_total_free: jax.Array = jnp.zeros((n_trace, 0), dtype=jnp.float32)
    area_frac_left_free: jax.Array = jnp.zeros((n_trace, 0), dtype=jnp.float32)

    if n_nonfree > 0:
        area_dominant = _sample_nonfree_areas(
            n_trace, n_nonfree, area_gaussian_pt[:, nonfree_idx], hp.area_log_sigma,
        )
    if n_artefact > 0:
        area_artefact = _sample_artefact_areas(
            n_trace, n_artefact, area_art_shared,
            hp.area_art_log_sigma, hp.area_art_trace_log_scale,
        )
    if n_free > 0:
        area_total_free, area_frac_left_free = _sample_free_areas(
            n_trace, n_free, area_gaussian_pt[:, free_idx], hp.area_log_sigma,
        )

    # --- Left/right canonical assembly ---
    geom = SplitGeometry.zeros((n_trace, n_peak))
    if n_nonfree > 0:
        geom = _assemble_nonfree(
            geom, apex, primary_sigma_left, primary_sigma_right,
            area_dominant, nonfree_idx,
        )
    if n_artefact > 0:
        geom = _assemble_artefact(
            geom, apex, primary_sigma_left, primary_sigma_right,
            artefact_sigma_left, artefact_sigma_right,
            area_dominant, area_artefact, separation_artefact,
            artefact_idx, nonfree_position, artefact_side_sign,
        )
    if n_free > 0:
        geom = _assemble_free(
            geom, apex, primary_sigma_left, primary_sigma_right,
            free_second_sigma_left, free_second_sigma_right,
            area_total_free, area_frac_left_free, separation_free,
            free_idx,
        )

    # --- Baseline and likelihood ---
    apex_flat, sigma_left_flat, sigma_right_flat, area_flat = geom.flatten_to_mixture()
    baseline = (
        trace_params.baseline_intercept[:, None]
        + trace_params.baseline_slope[:, None] * (x - window_midpoint_x)
    )
    mu_y = mixture_signal(x, apex_flat, sigma_left_flat, sigma_right_flat, area_flat) + baseline
    if y is not None:
        finite_mask = jnp.isfinite(y)
        numpyro.sample(
            "y",
            dist.Normal(mu_y, trace_params.sigma_y[:, None]).mask(finite_mask),
            obs=jnp.where(finite_mask, y, 0.0),
        )
    else:
        numpyro.sample("y", dist.Normal(mu_y, trace_params.sigma_y[:, None]))


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
    artefact_side_sign = jnp.asarray(model_inputs["artefact_side"], dtype=jnp.float32)
    apex_loc_arr = jnp.asarray(model_inputs["apex_loc"], dtype=jnp.float32)
    apex_offset_scale_arr = jnp.asarray(model_inputs["apex_offset_scale"], dtype=jnp.float32)
    trace_shift_scale = float(jnp.asarray(model_inputs["trace_shift_scale"]).max())
    n_peak = int(apex_loc_arr.shape[0])
    n_artefact = int(artefact_idx.shape[0])
    n_free = int(free_idx.shape[0])
    n_nonfree = int(nonfree_idx_in.shape[0])
    log_w_left = jnp.asarray(samples["log_w_left"])  # [n_total, n_peak]
    log_w_right = jnp.asarray(samples["log_w_right"])  # [n_total, n_peak]
    trace_shift_raw = jnp.asarray(samples["trace_shift_raw"])  # [n_total, n_trace]

    n_total, n_trace = trace_shift_raw.shape

    # [n_total, n_peak]
    primary_sigma_left, primary_sigma_right = _halfwidths_to_split(log_w_left, log_w_right)
    trace_shift = trace_shift_scale * (
        trace_shift_raw - trace_shift_raw.mean(axis=1, keepdims=True)
    )  # [n_total, n_trace]
    apex_offset_raw = jnp.asarray(samples["apex_offset_raw"])  # [n_total, n_trace, n_peak]
    apex_offset = apex_offset_scale_arr[None, None, :] * apex_offset_raw
    apex = apex_loc_arr[None, None, :] + trace_shift[:, :, None] + apex_offset  # [n_total, n_trace, n_peak]

    artefact_sigma_left: jax.Array = jnp.zeros((n_total, 0), dtype=jnp.float32)
    artefact_sigma_right: jax.Array = jnp.zeros((n_total, 0), dtype=jnp.float32)
    free_second_sigma_left: jax.Array = jnp.zeros((n_total, 0), dtype=jnp.float32)
    free_second_sigma_right: jax.Array = jnp.zeros((n_total, 0), dtype=jnp.float32)
    separation_artefact: jax.Array = jnp.zeros((n_total, 0), dtype=jnp.float32)
    separation_free: jax.Array = jnp.zeros((n_total, 0), dtype=jnp.float32)

    if n_artefact > 0:
        log_w_art = jnp.asarray(samples["log_w_art"])  # [n_total, n_artefact]
        w_art = jnp.exp(log_w_art) / _HWHM_FACTOR
        artefact_sigma_left = w_art
        artefact_sigma_right = w_art

    if n_free > 0:
        log_w_left_2 = jnp.asarray(samples["log_w_left_2"])  # [n_total, n_free]
        log_w_right_2 = jnp.asarray(samples["log_w_right_2"])
        free_second_sigma_left, free_second_sigma_right = _halfwidths_to_split(
            log_w_left_2, log_w_right_2
        )

    area_dominant = (
        jnp.asarray(samples["area_dominant"])
        if n_nonfree > 0
        else jnp.zeros((n_total, n_trace, 0), dtype=jnp.float32)
    )  # [n_total, n_trace, n_nonfree]

    if n_artefact > 0:
        separation_artefact = jnp.exp(
            jnp.asarray(samples["log_separation_artefact"])
        )  # [n_total, n_artefact]

        log_area_art_mean = jnp.asarray(samples["log_area_art_mean"])  # [n_total, n_artefact]
        log_area_art_raw = jnp.asarray(samples["log_area_art_raw"])  # [n_total, n_trace, n_artefact]
        area_artefact = jnp.exp(
            log_area_art_mean[:, None, :] + hyperparams.area_art_trace_log_scale * log_area_art_raw
        )  # [n_total, n_trace, n_artefact]
    else:
        area_artefact = jnp.zeros((n_total, n_trace, 0), dtype=jnp.float32)

    if n_free > 0:
        separation_free = jnp.exp(jnp.asarray(samples["log_separation_free"]))  # [n_total, n_free]
        area_total_free = jnp.asarray(samples["area_total_free"])  # [n_total, n_trace, n_free]
        area_frac_left_free = jnp.asarray(samples["area_frac_left_free"])
    else:
        area_total_free = jnp.zeros((n_total, n_trace, 0), dtype=jnp.float32)
        area_frac_left_free = jnp.zeros((n_total, n_trace, 0), dtype=jnp.float32)

    baseline_slope = jnp.asarray(samples["baseline_slope"])  # [n_total, n_trace]

    geom = SplitGeometry.zeros((n_total, n_trace, n_peak))
    if n_nonfree > 0:
        geom = _assemble_nonfree(
            geom, apex, primary_sigma_left, primary_sigma_right,
            area_dominant, nonfree_idx_in,
        )
    if n_artefact > 0:
        geom = _assemble_artefact(
            geom, apex, primary_sigma_left, primary_sigma_right,
            artefact_sigma_left, artefact_sigma_right,
            area_dominant, area_artefact, separation_artefact,
            artefact_idx, nonfree_position, artefact_side_sign,
        )
    if n_free > 0:
        geom = _assemble_free(
            geom, apex, primary_sigma_left, primary_sigma_right,
            free_second_sigma_left, free_second_sigma_right,
            area_total_free, area_frac_left_free, separation_free,
            free_idx,
        )

    # Per-peak separation array — CDQ-specific diagnostic, not part of the mixture signal.
    separation_out = jnp.zeros((n_total, n_trace, n_peak), dtype=jnp.float32)
    if n_artefact > 0:
        separation_out = separation_out.at[..., artefact_idx].set(
            _broadcast_to_trace_axis(
                separation_artefact, (n_total, n_trace, artefact_idx.shape[0])
            )
        )
    if n_free > 0:
        separation_out = separation_out.at[..., free_idx].set(
            _broadcast_to_trace_axis(
                separation_free, (n_total, n_trace, free_idx.shape[0])
            )
        )

    return {
        # Dict keys are an external API (fitter.py + tests read them) — preserve byte-for-byte.
        "sl_base": primary_sigma_left,
        "sr_base": primary_sigma_right,
        "trace_shift": trace_shift,
        "apex": apex,
        "baseline_slope": baseline_slope,
        "apex_l": geom.apex_left,
        "apex_r": geom.apex_right,
        "sl_l": geom.sigma_left_of_left,
        "sl_r": geom.sigma_left_of_right,
        "sr_l": geom.sigma_right_of_left,
        "sr_r": geom.sigma_right_of_right,
        "area_l": geom.area_left,
        "area_r": geom.area_right,
        "area_total": geom.area_left + geom.area_right,
        "separation": separation_out,
    }


# ---------------------------------------------------------------------------
# Posterior variable filtering
# ---------------------------------------------------------------------------

# Variables that are never interpretable on their own and should be excluded
# from ArviZ summary tables and trace plots.  Everything else in the posterior
# is shown automatically — no allowlist to maintain.
INTERNAL_POSTERIOR_VARS: frozenset[str] = frozenset({
    # Non-centred parameterisation raw unit-Normal samples
    "trace_shift_raw",
    "apex_offset_raw",
    "log_area_art_raw",
    # Intermediate geometric arrays (half-sigma per component, per-side apex)
    "sl_base",
    "sr_base",
    "apex_l",
    "apex_r",
    "sl_l",
    "sl_r",
    "sr_l",
    "sr_r",
})


# ---------------------------------------------------------------------------
# User-facing configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """User-facing configuration for the NumPyro fit.

    Tuned defaults for fast development iteration on chromatographic data.
    Override fields directly when constructing for publication-quality runs.
    """

    # --- HMC / NUTS settings ---
    num_warmup: int = 500
    """Number of NUTS warmup samples per chain."""

    num_samples: int = 500
    """Number of NUTS post-warmup samples per chain."""

    num_chains: int = 4
    """Number of parallel NUTS chains."""

    target_accept_prob: float = 0.9
    """Target acceptance probability for the NUTS step-size adaptor."""

    max_tree_depth: int = 10
    """Maximum tree depth for the NUTS integrator."""

    seed: int = 0
    """Random seed for the NUTS sampler."""

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
