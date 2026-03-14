"""Linear baseline estimation for chromatographic peak fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import jax.numpy as jnp

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation

_DEFAULT_PERCENTILE: Final = 5.0
_DEFAULT_EDGE_FRACTION: Final = 0.20
_MIN_EDGE_POINTS: Final = 6
_MIN_INTERCEPT_SCALE: Final = 1.0
_MIN_SLOPE_SCALE: Final = 1e-3
_PRIOR_SE_MULTIPLIER: Final = 2.5
_GLOBAL_SCALE_CAP: Final = 2.0


@dataclass(frozen=True)
class BaselinePriors:
    """Per-trace linear baseline priors for the Bayesian model."""

    intercept: jnp.ndarray  # [n_trace] location
    slope: jnp.ndarray  # [n_trace] location
    intercept_scale: jnp.ndarray  # [n_trace]
    slope_scale: jnp.ndarray  # [n_trace]


def estimate_baseline(
    time: jnp.ndarray,
    signal: jnp.ndarray,
    *,
    peaks: list[PeakAnnotation],
    baselines: list[BaselineAnnotation] = (),
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
    Prior scales are derived from the OLS standard errors, capped at
    twice the across-trace robust spread.

    Args:
        time:          ``[n_trace, n_time]`` retention-time axis.
        signal:        ``[n_trace, n_time]`` signal matrix.
        peaks:         Peak window annotations.
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
    if not (0.0 < float(percentile) <= 100.0):
        raise ValueError("percentile must satisfy 0 < percentile <= 100.")
    if not (0.0 < float(edge_fraction) <= 0.5):
        raise ValueError("edge_fraction must satisfy 0 < edge_fraction <= 0.5.")

    anchor_mask = _select_anchors(
        time,
        signal,
        peaks=peaks,
        baselines=baselines,
        edge_fraction=float(edge_fraction),
        percentile=float(percentile),
    )
    intercept, slope, se_intercept, se_slope = _fit_line(time, signal, anchor_mask)

    global_intercept_scale = _robust_scale(intercept, floor=_MIN_INTERCEPT_SCALE)
    global_slope_scale = _robust_scale(slope, floor=_MIN_SLOPE_SCALE)

    intercept_scale = jnp.clip(
        _scale_from_se(se_intercept, floor=_MIN_INTERCEPT_SCALE),
        _MIN_INTERCEPT_SCALE,
        _GLOBAL_SCALE_CAP * global_intercept_scale,
    )
    slope_scale = jnp.clip(
        _scale_from_se(se_slope, floor=_MIN_SLOPE_SCALE),
        _MIN_SLOPE_SCALE,
        _GLOBAL_SCALE_CAP * global_slope_scale,
    )

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
    time: jnp.ndarray,
    signal: jnp.ndarray,
    *,
    peaks: list[PeakAnnotation],
    baselines: list[BaselineAnnotation],
    edge_fraction: float,
    percentile: float,
) -> jnp.ndarray:
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

        left_threshold = jnp.nanquantile(
            jnp.where(left_edge, signal, jnp.nan), q, axis=1
        )
        right_threshold = jnp.nanquantile(
            jnp.where(right_edge, signal, jnp.nan), q, axis=1
        )
        left_threshold = jnp.where(
            jnp.isfinite(left_threshold), left_threshold, -jnp.inf
        )
        right_threshold = jnp.where(
            jnp.isfinite(right_threshold), right_threshold, -jnp.inf
        )

        edge_selected = (left_edge & (signal <= left_threshold[:, None])) | (
            right_edge & (signal <= right_threshold[:, None])
        )

        # Fallback: full-window percentile when edges are too sparse.
        full_threshold = jnp.nanquantile(
            jnp.where(in_window, signal, jnp.nan), q, axis=1
        )
        full_threshold = jnp.where(
            jnp.isfinite(full_threshold), full_threshold, -jnp.inf
        )
        full_selected = in_window & (signal <= full_threshold[:, None])

        has_edges = jnp.any(edge_mask, axis=1)
        selected = selected | jnp.where(
            has_edges[:, None], edge_selected, full_selected
        )

    # Ensure ≥2 anchor points per trace; fall back to all annotated points.
    all_annotated = jnp.zeros_like(finite, dtype=bool)
    for pk in peaks:
        all_annotated = all_annotated | (
            (time >= float(pk.rt_min)) & (time <= float(pk.rt_max)) & finite
        )
    for bl in baselines:
        all_annotated = all_annotated | (
            (time >= float(bl.rt_min)) & (time <= float(bl.rt_max)) & finite
        )

    anchor_count = jnp.sum(selected, axis=1)
    selected = jnp.where(anchor_count[:, None] >= 2, selected, all_annotated)
    anchor_count = jnp.sum(selected, axis=1)
    selected = jnp.where(anchor_count[:, None] >= 2, selected, finite)
    return selected


def _window_edge_masks(
    in_window: jnp.ndarray,
    *,
    edge_fraction: float,
    min_edge_points: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
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
    time: jnp.ndarray,
    signal: jnp.ndarray,
    mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    intercept = jnp.nan_to_num(ym[..., 0], nan=0.0) - slope * jnp.nan_to_num(
        xm[..., 0], nan=0.0
    )

    # OLS standard errors
    n = jnp.sum(mask, axis=1).astype(jnp.float32)
    x_mean = jnp.nanmean(t, axis=1)
    sxx_se = jnp.nansum((t - x_mean[:, None]) ** 2, axis=1)
    sse = jnp.nansum((y - intercept[:, None] - slope[:, None] * t) ** 2, axis=1)
    sigma2 = sse / jnp.maximum(n - 2.0, 1.0)
    valid = (n > 2.0) & jnp.isfinite(sigma2) & (sxx_se > 1e-12)

    se_slope = jnp.where(
        valid, jnp.sqrt(jnp.maximum(sigma2 / jnp.maximum(sxx_se, 1e-12), 0.0)), jnp.nan
    )
    se_intercept = jnp.where(
        valid,
        jnp.sqrt(
            jnp.maximum(
                sigma2
                * (1.0 / jnp.maximum(n, 1.0) + x_mean**2 / jnp.maximum(sxx_se, 1e-12)),
                0.0,
            )
        ),
        jnp.nan,
    )
    return intercept, slope, se_intercept, se_slope


def _scale_from_se(se: jnp.ndarray, *, floor: float) -> jnp.ndarray:
    """Convert OLS standard errors to prior scales."""
    raw = _PRIOR_SE_MULTIPLIER * se
    finite = raw[jnp.isfinite(raw) & (raw > 0.0)]
    fallback = (
        max(float(jnp.nanmedian(finite)), float(floor))
        if int(finite.size) > 0
        else float(floor)
    )
    return jnp.where(
        jnp.isfinite(raw) & (raw > 0.0),
        jnp.maximum(raw, floor),
        jnp.asarray(fallback, dtype=raw.dtype),
    )


def _robust_scale(values: jnp.ndarray, *, floor: float) -> jnp.ndarray:
    """Robust across-trace spread (MAD-based) with floor."""
    finite = values[jnp.isfinite(values)]
    if int(finite.size) <= 1:
        return jnp.asarray(floor, dtype=values.dtype)
    med = jnp.median(finite)
    mad = jnp.median(jnp.abs(finite - med))
    return jnp.asarray(max(float(1.4826 * mad), float(floor)), dtype=values.dtype)


__all__ = ["BaselinePriors", "estimate_baseline"]
