"""Linear baseline estimation for chromatographic peak fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from chromhandler.annotations import BaselineAnnotation, PeakAnnotation

_DEFAULT_PERCENTILE: Final = 15.0
_DEFAULT_EDGE_FRACTION: Final = 0.20
_MIN_EDGE_POINTS: Final = 6


@dataclass(frozen=True)
class BaselinePriors:
    """Per-trace linear baseline priors for the Bayesian model."""

    intercept: jax.Array  # [n_trace] location
    slope: jax.Array  # [n_trace] location
    intercept_scale: jax.Array  # [n_trace]
    slope_scale: jax.Array  # [n_trace]


def estimate_baseline(
    time: jax.Array,
    signal: jax.Array,
    *,
    peaks: list[PeakAnnotation],
    sigma_noise: jax.Array,
    baselines: list[BaselineAnnotation] | None = None,
    edge_fraction: float = _DEFAULT_EDGE_FRACTION,
    percentile: float = _DEFAULT_PERCENTILE,
) -> BaselinePriors:
    """Estimate per-trace linear baseline priors.

    Anchor points are collected from:

    - **Explicit baseline sections** — all finite points within each
      :class:`~.data.BaselineAnnotation` region.
    - **Peak window edges** — bottom ``percentile``% of the left+right
      ``edge_fraction`` of each :class:`~.data.PeakAnnotation` window.
      Falls back to bottom ``percentile``% of the full window when the
      window is too narrow for edge extraction.

    A per-trace OLS line is then fitted through the anchor points.
    Prior scales are derived from the OLS standard errors, floored at a
    per-trace physical scale: intercept floor = ``sigma_noise``, slope
    floor = ``sigma_noise / time_span`` (falling back to ``sigma_noise``
    when ``time_span <= 0``).

    Args:
        time:          ``[n_trace, n_time]`` retention-time axis.
        signal:        ``[n_trace, n_time]`` signal matrix.
        peaks:         Peak window annotations.
        sigma_noise:   ``[n_trace]`` per-trace noise estimate (DER_SNR).
        baselines:     Optional explicit baseline region annotations.
        edge_fraction: Fraction of each window to use for edge anchors.
        percentile:    Signal percentile threshold for anchor selection.

    Returns:
        :class:`BaselinePriors` with per-trace intercept/slope and scales.
    """
    if time.ndim != 2 or signal.ndim != 2:
        raise ValueError("time and signal must be 2-D [n_trace, n_time].")
    if time.shape != signal.shape:
        raise ValueError("time and signal shape mismatch.")
    sigma_noise = jnp.asarray(sigma_noise)
    if sigma_noise.shape != (time.shape[0],):
        raise ValueError(
            f"sigma_noise must have shape [n_trace]={time.shape[0]}, got {sigma_noise.shape}."
        )
    if not (0.0 < float(percentile) <= 100.0):
        raise ValueError("percentile must satisfy 0 < percentile <= 100.")
    if not (0.0 < float(edge_fraction) <= 0.5):
        raise ValueError("edge_fraction must satisfy 0 < edge_fraction <= 0.5.")

    baseline_regions = [] if baselines is None else baselines

    anchor_mask = _select_anchors(
        time,
        signal,
        peaks=peaks,
        baselines=baseline_regions,
        edge_fraction=float(edge_fraction),
        percentile=float(percentile),
    )
    intercept, slope, se_intercept, se_slope = _fit_line(time, signal, anchor_mask)

    # Per-trace time span for slope floor. Fall back to sigma_noise when span <= 0.
    time_span = time[:, -1] - time[:, 0]
    slope_floor = jnp.where(
        time_span > 0.0,
        sigma_noise / jnp.where(time_span > 0.0, time_span, 1.0),
        sigma_noise,
    )

    intercept_scale = _scale_from_se(se_intercept, floor=sigma_noise)
    slope_scale = _scale_from_se(se_slope, floor=slope_floor)

    return BaselinePriors(
        intercept=intercept,
        slope=slope,
        intercept_scale=intercept_scale,
        slope_scale=slope_scale,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _select_anchors(
    time: jax.Array,
    signal: jax.Array,
    *,
    peaks: list[PeakAnnotation],
    baselines: list[BaselineAnnotation],
    edge_fraction: float,
    percentile: float,
) -> jax.Array:
    """Build a per-trace boolean mask of baseline anchor points.

    Explicit baseline sections contribute all their finite points.
    Peak windows contribute low-percentile edge points (or full-window
    percentile fallback when the window is too narrow for edge extraction).
    """
    finite = jnp.isfinite(time) & jnp.isfinite(signal)
    selected = jnp.zeros_like(finite, dtype=bool)
    q = percentile / 100.0

    # Explicit baseline sections: use all finite points.
    for bl in baselines:
        in_region = (time >= float(bl.rt_min)) & (time <= float(bl.rt_max)) & finite
        selected = selected | in_region

    # Peak windows: low-percentile edge points with full-window fallback.
    for pk in peaks:
        in_window = (time >= float(pk.rt_min)) & (time <= float(pk.rt_max)) & finite
        left_edge, right_edge = _window_edge_masks(
            in_window,
            edge_fraction=edge_fraction,
            min_edge_points=_MIN_EDGE_POINTS,
        )
        edge_mask = left_edge | right_edge

        left_threshold = jnp.nanquantile(jnp.where(left_edge, signal, jnp.nan), q, axis=1)
        right_threshold = jnp.nanquantile(jnp.where(right_edge, signal, jnp.nan), q, axis=1)
        left_threshold = jnp.where(jnp.isfinite(left_threshold), left_threshold, -jnp.inf)
        right_threshold = jnp.where(jnp.isfinite(right_threshold), right_threshold, -jnp.inf)

        edge_selected = (left_edge & (signal <= left_threshold[:, None])) | (
            right_edge & (signal <= right_threshold[:, None])
        )

        # Fallback: full-window percentile when edges are too sparse.
        full_threshold = jnp.nanquantile(jnp.where(in_window, signal, jnp.nan), q, axis=1)
        full_threshold = jnp.where(jnp.isfinite(full_threshold), full_threshold, -jnp.inf)
        full_selected = in_window & (signal <= full_threshold[:, None])

        has_edges = jnp.any(edge_mask, axis=1)
        selected = selected | jnp.where(has_edges[:, None], edge_selected, full_selected)

    # Ensure ≥2 anchor points per trace; fall back to all annotated points.
    all_annotated = jnp.zeros_like(finite, dtype=bool)
    for pk in peaks:
        all_annotated = all_annotated | ((time >= float(pk.rt_min)) & (time <= float(pk.rt_max)) & finite)
    for bl in baselines:
        all_annotated = all_annotated | ((time >= float(bl.rt_min)) & (time <= float(bl.rt_max)) & finite)

    anchor_count = jnp.sum(selected, axis=1)
    selected = jnp.where(anchor_count[:, None] >= 2, selected, all_annotated)
    anchor_count = jnp.sum(selected, axis=1)
    selected = jnp.where(anchor_count[:, None] >= 2, selected, finite)
    return selected


def _window_edge_masks(
    in_window: jax.Array,
    *,
    edge_fraction: float,
    min_edge_points: int,
) -> tuple[jax.Array, jax.Array]:
    """Return left/right edge masks for each trace in a peak window."""
    n_time = in_window.shape[1]
    indices = jnp.arange(n_time, dtype=jnp.int32)[None, :]
    present = jnp.any(in_window, axis=1)
    count = jnp.sum(in_window, axis=1).astype(jnp.int32)

    first = jnp.min(jnp.where(in_window, indices, n_time), axis=1)
    last = jnp.max(jnp.where(in_window, indices, -1), axis=1)

    edge_len = jnp.ceil(edge_fraction * count.astype(jnp.float32)).astype(jnp.int32)
    edge_len = jnp.maximum(edge_len, int(min_edge_points))
    edge_len = jnp.minimum(edge_len, jnp.maximum(count // 2, 1))

    left = in_window & (indices <= (first + edge_len - 1)[:, None]) & present[:, None]
    right = in_window & (indices >= (last - edge_len + 1)[:, None]) & present[:, None]
    return left, right


def _fit_line(
    time: jax.Array,
    signal: jax.Array,
    mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """OLS linear fit per trace through masked points.

    Returns:
        ``(intercept, slope, se_intercept, se_slope)`` each ``[n_trace]``.

    Example:
        >>> import jax.numpy as jnp
        >>> t = jnp.linspace(0, 1, 100)[None, :]
        >>> y = 2.0 * t + 500.0
        >>> intercept, slope, *_ = _fit_line(t, y, jnp.ones_like(t, dtype=bool))
        >>> round(float(slope[0]), 1)
        2.0
    """
    t = jnp.where(mask, time, jnp.nan)
    y = jnp.where(mask, signal, jnp.nan)

    xm = jnp.nanmean(t, axis=1, keepdims=True)
    ym = jnp.nanmean(y, axis=1, keepdims=True)
    sxx = jnp.nansum((t - xm) ** 2, axis=1)
    sxy = jnp.nansum((t - xm) * (y - ym), axis=1)

    slope = jnp.where(sxx > 1e-12, sxy / sxx, 0.0)
    intercept = jnp.nan_to_num(ym[..., 0], nan=0.0) - slope * jnp.nan_to_num(xm[..., 0], nan=0.0)

    # OLS standard errors
    n = jnp.sum(mask, axis=1).astype(jnp.float32)
    x_mean = jnp.nanmean(t, axis=1)
    sxx_se = jnp.nansum((t - x_mean[:, None]) ** 2, axis=1)
    sse = jnp.nansum((y - intercept[:, None] - slope[:, None] * t) ** 2, axis=1)
    sigma2 = sse / jnp.maximum(n - 2.0, 1.0)
    valid = (n > 2.0) & jnp.isfinite(sigma2) & (sxx_se > 1e-12)

    se_slope = jnp.where(valid, jnp.sqrt(jnp.maximum(sigma2 / jnp.maximum(sxx_se, 1e-12), 0.0)), jnp.nan)
    se_intercept = jnp.where(
        valid,
        jnp.sqrt(
            jnp.maximum(
                sigma2 * (1.0 / jnp.maximum(n, 1.0) + x_mean**2 / jnp.maximum(sxx_se, 1e-12)),
                0.0,
            )
        ),
        jnp.nan,
    )
    return intercept, slope, se_intercept, se_slope


def _scale_from_se(se: jax.Array, *, floor: jax.Array) -> jax.Array:
    """Use OLS standard errors as prior scales, with a per-trace floor.

    Args:
        se:    ``[n_trace]`` standard errors (may contain NaN for degenerate fits).
        floor: ``[n_trace]`` per-trace minimum scale, strictly positive.
    """
    return jnp.where(
        jnp.isfinite(se) & (se > 0.0),
        jnp.maximum(se, floor),
        floor,
    )
