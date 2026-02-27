"""Moment-based diagnostics for chromatographic peak windows.

This module provides robust, baseline-aware summary metrics that can be used to
screen peak quality and derive tighter integration bounds inside broad user
windows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class PeakMomentMetrics:
    """Moment-derived diagnostics for one peak in one chromatogram trace.

    Attributes:
        window_low: Lower bound of the analyzed window.
        window_high: Upper bound of the analyzed window.
        area: Baseline-corrected positive area in the window.
        apex_time: Time of the maximum corrected intensity.
        apex_height: Maximum corrected intensity.
        centroid: First-moment center of mass.
        sigma: Standard deviation from the second central moment.
        skewness: Standardized third central moment.
        centroid_apex_z: Normalized centroid shift ``(centroid - apex_time) / sigma``.
        tail_ratio: Right/left tail area ratio around the apex.
        log_tail_ratio: Natural log of ``tail_ratio``.
        left_sigma: Left-side width around centroid.
        right_sigma: Right-side width around centroid.
        start_time: Cumulative-area lower quantile bound inside the window.
        end_time: Cumulative-area upper quantile bound inside the window.
        baseline_slope: Slope of local linear baseline in the window.
        baseline_intercept: Intercept of local linear baseline in the window.
    """

    window_low: float
    window_high: float
    area: float
    apex_time: float
    apex_height: float
    centroid: float
    sigma: float
    skewness: float
    centroid_apex_z: float
    tail_ratio: float
    log_tail_ratio: float
    left_sigma: float
    right_sigma: float
    start_time: float
    end_time: float
    baseline_slope: float
    baseline_intercept: float


@dataclass(frozen=True)
class PeakPriorHints:
    """Robust prior hints for one logical peak from moment metrics."""

    mu_loc: float
    mu_scale: float
    sigma_loc: float
    sigma_scale: float
    alpha_loc: float
    alpha_scale: float
    area_loc: float
    area_scale: float
    trace_count: int


def _nan_metrics(window_low: float, window_high: float) -> PeakMomentMetrics:
    """Return a metrics object filled with NaN values."""
    return PeakMomentMetrics(
        window_low=float(window_low),
        window_high=float(window_high),
        area=np.nan,
        apex_time=np.nan,
        apex_height=np.nan,
        centroid=np.nan,
        sigma=np.nan,
        skewness=np.nan,
        centroid_apex_z=np.nan,
        tail_ratio=np.nan,
        log_tail_ratio=np.nan,
        left_sigma=np.nan,
        right_sigma=np.nan,
        start_time=np.nan,
        end_time=np.nan,
        baseline_slope=np.nan,
        baseline_intercept=np.nan,
    )


def _safe_trapezoid(y_values: np.ndarray, x_values: np.ndarray) -> float:
    """Compute trapezoidal integral with basic size guards."""
    if y_values.size < 2 or x_values.size < 2:
        return 0.0
    return float(np.trapezoid(y_values, x_values))


def _cumulative_area_curve(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return cumulative area sampled at x-grid points and total area."""
    if x_values.size == 0:
        return np.zeros(0, dtype=float), 0.0
    if x_values.size == 1:
        return np.zeros(1, dtype=float), 0.0

    segment_area = (
        0.5 * (y_values[:-1] + y_values[1:]) * np.maximum(np.diff(x_values), 0.0)
    )
    cumulative = np.concatenate(
        [np.zeros(1, dtype=float), np.cumsum(segment_area, dtype=float)]
    )
    total_area = float(cumulative[-1])
    return cumulative, total_area


def compute_peak_moment_metrics(
    x_values: np.ndarray,
    y_values: np.ndarray,
    window_low: float,
    window_high: float,
    start_quantile: float = 0.005,
    end_quantile: float = 0.995,
    tail_window_sigma: float = 2.0,
    baseline_slope: float = 0.0,
    baseline_intercept: float = 0.0,
) -> PeakMomentMetrics:
    """Compute moment-based diagnostics within a user-defined peak window.

    Args:
        x_values: 1D retention-time values.
        y_values: 1D signal values.
        window_low: Lower retention-time bound.
        window_high: Upper retention-time bound.
        start_quantile: Lower cumulative-area quantile for tight start bound.
        end_quantile: Upper cumulative-area quantile for tight end bound.
        tail_window_sigma: Tail window radius as multiples of ``sigma``.

    Returns:
        Moment diagnostics for one trace/window pair.
    """
    if window_high <= window_low:
        return _nan_metrics(window_low, window_high)
    if not (0.0 < start_quantile < end_quantile < 1.0):
        raise ValueError("Require 0 < start_quantile < end_quantile < 1.")
    if tail_window_sigma <= 0.0:
        raise ValueError("tail_window_sigma must be > 0.")

    x_array = np.asarray(x_values, dtype=float).reshape(-1)
    y_array = np.asarray(y_values, dtype=float).reshape(-1)
    finite_mask = np.isfinite(x_array) & np.isfinite(y_array)
    window_mask = (
        finite_mask & (x_array >= float(window_low)) & (x_array <= float(window_high))
    )

    if int(np.sum(window_mask)) < 3:
        return _nan_metrics(window_low, window_high)

    x_window = x_array[window_mask]
    y_window = y_array[window_mask]
    order = np.argsort(x_window)
    x_window = x_window[order]
    y_window = y_window[order]

    baseline = float(baseline_slope) * x_window + float(baseline_intercept)
    y_corrected = np.clip(y_window - baseline, a_min=0.0, a_max=None)
    area = _safe_trapezoid(y_corrected, x_window)

    if area <= 1e-12:
        shifted = y_corrected - float(np.nanmin(y_corrected))
        y_corrected = np.clip(shifted, a_min=0.0, a_max=None)
        area = _safe_trapezoid(y_corrected, x_window)

    if area <= 1e-12:
        return _nan_metrics(window_low, window_high)

    apex_index = int(np.argmax(y_corrected))
    apex_time = float(x_window[apex_index])
    apex_height = float(y_corrected[apex_index])

    centroid = float(_safe_trapezoid(x_window * y_corrected, x_window) / area)
    second_moment = float(
        _safe_trapezoid(((x_window - centroid) ** 2) * y_corrected, x_window) / area
    )
    sigma = float(np.sqrt(max(second_moment, 1e-12)))

    third_moment = float(
        _safe_trapezoid(((x_window - centroid) ** 3) * y_corrected, x_window) / area
    )
    skewness = float(third_moment / (sigma**3 + 1e-12))
    centroid_apex_z = float((centroid - apex_time) / max(sigma, 1e-12))

    cumulative_area, total_area = _cumulative_area_curve(x_window, y_corrected)
    if total_area <= 1e-12:
        return _nan_metrics(window_low, window_high)
    normalized_cumulative = cumulative_area / total_area
    start_time = float(np.interp(start_quantile, normalized_cumulative, x_window))
    end_time = float(np.interp(end_quantile, normalized_cumulative, x_window))

    left_mask = x_window <= centroid
    right_mask = x_window >= centroid
    left_area = _safe_trapezoid(y_corrected[left_mask], x_window[left_mask])
    right_area = _safe_trapezoid(y_corrected[right_mask], x_window[right_mask])

    if left_area > 1e-12:
        left_second = _safe_trapezoid(
            ((x_window[left_mask] - centroid) ** 2) * y_corrected[left_mask],
            x_window[left_mask],
        )
        left_sigma = float(np.sqrt(max(left_second / left_area, 1e-12)))
    else:
        left_sigma = np.nan

    if right_area > 1e-12:
        right_second = _safe_trapezoid(
            ((x_window[right_mask] - centroid) ** 2) * y_corrected[right_mask],
            x_window[right_mask],
        )
        right_sigma = float(np.sqrt(max(right_second / right_area, 1e-12)))
    else:
        right_sigma = np.nan

    tail_radius = float(tail_window_sigma * max(sigma, 1e-12))
    left_tail_mask = (x_window >= (apex_time - tail_radius)) & (x_window <= apex_time)
    right_tail_mask = (x_window >= apex_time) & (x_window <= (apex_time + tail_radius))
    left_tail_area = _safe_trapezoid(
        y_corrected[left_tail_mask], x_window[left_tail_mask]
    )
    right_tail_area = _safe_trapezoid(
        y_corrected[right_tail_mask], x_window[right_tail_mask]
    )
    tail_ratio = float((right_tail_area + 1e-12) / (left_tail_area + 1e-12))
    log_tail_ratio = float(np.log(tail_ratio))

    return PeakMomentMetrics(
        window_low=float(window_low),
        window_high=float(window_high),
        area=float(area),
        apex_time=apex_time,
        apex_height=apex_height,
        centroid=centroid,
        sigma=sigma,
        skewness=skewness,
        centroid_apex_z=centroid_apex_z,
        tail_ratio=tail_ratio,
        log_tail_ratio=log_tail_ratio,
        left_sigma=float(left_sigma),
        right_sigma=float(right_sigma),
        start_time=start_time,
        end_time=end_time,
        baseline_slope=float(baseline_slope),
        baseline_intercept=float(baseline_intercept),
    )


def compute_peak_moment_metrics_batch(
    x_matrix: np.ndarray,
    y_matrix: np.ndarray,
    window_low: float,
    window_high: float,
    start_quantile: float = 0.005,
    end_quantile: float = 0.995,
    tail_window_sigma: float = 2.0,
    baseline_slopes: float | np.ndarray | None = None,
    baseline_intercepts: float | np.ndarray | None = None,
) -> list[PeakMomentMetrics]:
    """Compute moment diagnostics for all traces of one logical peak window.

    Args:
        x_matrix: 2D time matrix with shape ``[num_traces, num_points]``.
        y_matrix: 2D signal matrix with same shape as ``x_matrix``.
        window_low: Lower retention-time bound.
        window_high: Upper retention-time bound.
        start_quantile: Lower cumulative-area quantile for tight start bound.
        end_quantile: Upper cumulative-area quantile for tight end bound.
        tail_window_sigma: Tail window radius as multiples of ``sigma``.

    Returns:
        List of per-trace metrics.
    """
    x_array = np.asarray(x_matrix, dtype=float)
    y_array = np.asarray(y_matrix, dtype=float)
    if x_array.ndim != 2 or y_array.ndim != 2:
        raise ValueError("x_matrix and y_matrix must both be 2D.")
    if x_array.shape != y_array.shape:
        raise ValueError("x_matrix and y_matrix must have matching shape.")

    num_traces = int(x_array.shape[0])
    if baseline_slopes is None:
        slope_array = np.zeros((num_traces,), dtype=float)
    else:
        slope_raw = np.asarray(baseline_slopes, dtype=float)
        if slope_raw.ndim == 0:
            slope_array = np.full((num_traces,), float(slope_raw), dtype=float)
        elif slope_raw.shape == (num_traces,):
            slope_array = slope_raw
        else:
            raise ValueError("baseline_slopes must be scalar or shape [num_traces].")

    if baseline_intercepts is None:
        intercept_array = np.zeros((num_traces,), dtype=float)
    else:
        intercept_raw = np.asarray(baseline_intercepts, dtype=float)
        if intercept_raw.ndim == 0:
            intercept_array = np.full((num_traces,), float(intercept_raw), dtype=float)
        elif intercept_raw.shape == (num_traces,):
            intercept_array = intercept_raw
        else:
            raise ValueError(
                "baseline_intercepts must be scalar or shape [num_traces]."
            )

    metrics: list[PeakMomentMetrics] = []
    for trace_index in range(num_traces):
        metrics.append(
            compute_peak_moment_metrics(
                x_values=x_array[trace_index],
                y_values=y_array[trace_index],
                window_low=window_low,
                window_high=window_high,
                start_quantile=start_quantile,
                end_quantile=end_quantile,
                tail_window_sigma=tail_window_sigma,
                baseline_slope=float(slope_array[trace_index]),
                baseline_intercept=float(intercept_array[trace_index]),
            )
        )
    return metrics


def compute_peak_moment_metrics_from_peak_masks(
    x_matrix: np.ndarray,
    y_matrix: np.ndarray,
    peak_masks: np.ndarray,
    baseline_slopes: np.ndarray,
    baseline_intercepts: np.ndarray,
    start_quantile: float = 0.005,
    end_quantile: float = 0.995,
    tail_window_sigma: float = 2.0,
) -> list[list[PeakMomentMetrics]]:
    """Compute per-trace moment metrics for each peak mask."""
    x_array = np.asarray(x_matrix, dtype=float)
    y_array = np.asarray(y_matrix, dtype=float)
    mask_array = np.asarray(peak_masks, dtype=bool)
    slope_array = np.asarray(baseline_slopes, dtype=float).reshape(-1)
    intercept_array = np.asarray(baseline_intercepts, dtype=float).reshape(-1)

    if x_array.ndim != 2 or y_array.ndim != 2:
        raise ValueError("x_matrix and y_matrix must both be 2D.")
    if x_array.shape != y_array.shape:
        raise ValueError("x_matrix and y_matrix must have matching shape.")
    if mask_array.ndim != 3:
        raise ValueError(
            "peak_masks must be 3D with shape [num_peaks, num_traces, num_points]."
        )
    if mask_array.shape[1:] != x_array.shape:
        raise ValueError(
            "peak_masks shape must match x/y matrix shape on [num_traces, num_points]."
        )
    if slope_array.shape != (x_array.shape[0],):
        raise ValueError("baseline_slopes must have shape [num_traces].")
    if intercept_array.shape != (x_array.shape[0],):
        raise ValueError("baseline_intercepts must have shape [num_traces].")

    per_peak: list[list[PeakMomentMetrics]] = []
    for peak_index in range(mask_array.shape[0]):
        per_trace: list[PeakMomentMetrics] = []
        for trace_index in range(x_array.shape[0]):
            mask = mask_array[peak_index, trace_index]
            finite = np.isfinite(x_array[trace_index]) & np.isfinite(
                y_array[trace_index]
            )
            active = mask & finite
            if int(np.sum(active)) < 3:
                if np.any(active):
                    x_active = x_array[trace_index, active]
                    per_trace.append(
                        _nan_metrics(float(np.min(x_active)), float(np.max(x_active)))
                    )
                else:
                    per_trace.append(_nan_metrics(np.nan, np.nan))
                continue

            x_active = x_array[trace_index, active]
            y_active = y_array[trace_index, active]
            per_trace.append(
                compute_peak_moment_metrics(
                    x_values=x_active,
                    y_values=y_active,
                    window_low=float(np.min(x_active)),
                    window_high=float(np.max(x_active)),
                    start_quantile=start_quantile,
                    end_quantile=end_quantile,
                    tail_window_sigma=tail_window_sigma,
                    baseline_slope=float(slope_array[trace_index]),
                    baseline_intercept=float(intercept_array[trace_index]),
                )
            )
        per_peak.append(per_trace)
    return per_peak


def _robust_location_scale(
    values: np.ndarray, scale_floor: float = 1e-6
) -> tuple[float, float]:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return np.nan, np.nan
    location = float(np.median(finite_values))
    mad = float(np.median(np.abs(finite_values - location)))
    scale = float(max(1.4826 * mad, scale_floor))
    return location, scale


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if int(np.sum(valid)) == 0:
        return np.nan
    v = values[valid]
    w = weights[valid]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cumulative = np.cumsum(w)
    cutoff = 0.5 * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, cutoff, side="left"))
    index = int(np.clip(index, 0, v.size - 1))
    return float(v[index])


def _weighted_robust_location_scale(
    values: np.ndarray,
    weights: np.ndarray,
    scale_floor: float = 1e-6,
) -> tuple[float, float]:
    values_array = np.asarray(values, dtype=float).reshape(-1)
    weights_array = np.asarray(weights, dtype=float).reshape(-1)
    if values_array.shape != weights_array.shape:
        return _robust_location_scale(values_array, scale_floor=scale_floor)

    valid = (
        np.isfinite(values_array) & np.isfinite(weights_array) & (weights_array > 0.0)
    )
    if int(np.sum(valid)) == 0:
        return _robust_location_scale(values_array, scale_floor=scale_floor)

    v = values_array[valid]
    w = weights_array[valid]
    location = _weighted_median(v, w)
    if not np.isfinite(location):
        return _robust_location_scale(values_array, scale_floor=scale_floor)

    abs_dev = np.abs(v - location)
    mad = _weighted_median(abs_dev, w)
    if not np.isfinite(mad):
        mad = float(np.nanmedian(abs_dev))
    scale = float(max(1.4826 * float(mad), scale_floor))
    return float(location), scale


def _skewness_from_alpha(alpha: float) -> float:
    delta = float(alpha) / np.sqrt(1.0 + float(alpha) ** 2)
    beta = delta * np.sqrt(2.0 / np.pi)
    denominator = max((1.0 - (2.0 * delta * delta / np.pi)) ** 1.5, 1e-12)
    numerator = ((4.0 - np.pi) / 2.0) * (beta**3)
    return float(numerator / denominator)


def alpha_from_skewness(skewness: float) -> float:
    """Map standardized skewness to skew-normal alpha via monotonic bisection."""
    if not np.isfinite(skewness):
        return np.nan
    target = float(np.clip(skewness, -0.995, 0.995))
    lower = -50.0
    upper = 50.0
    for _ in range(80):
        mid = 0.5 * (lower + upper)
        if _skewness_from_alpha(mid) < target:
            lower = mid
        else:
            upper = mid
    return float(0.5 * (lower + upper))


def estimate_skew_normal_prior_hints(
    metrics_by_peak: Sequence[Sequence[PeakMomentMetrics]],
) -> list[PeakPriorHints]:
    """Convert moment metrics into robust prior hints for skew-normal mixtures."""
    hints: list[PeakPriorHints] = []
    for peak_metrics in metrics_by_peak:
        if len(peak_metrics) == 0:
            hints.append(
                PeakPriorHints(
                    mu_loc=np.nan,
                    mu_scale=np.nan,
                    sigma_loc=np.nan,
                    sigma_scale=np.nan,
                    alpha_loc=np.nan,
                    alpha_scale=np.nan,
                    area_loc=np.nan,
                    area_scale=np.nan,
                    trace_count=0,
                )
            )
            continue

        arrays = metrics_list_to_arrays(peak_metrics)
        apex_time_values = arrays.get("apex_time", np.array([], dtype=float))
        centroid_values = arrays.get("centroid", np.array([], dtype=float))
        sigma_values = arrays.get("sigma", np.array([], dtype=float))
        left_sigma_values = arrays.get("left_sigma", np.array([], dtype=float))
        right_sigma_values = arrays.get("right_sigma", np.array([], dtype=float))
        area_values = arrays.get("area", np.array([], dtype=float))
        skewness_values = arrays.get("skewness", np.array([], dtype=float))
        apex_height_values = arrays.get("apex_height", np.array([], dtype=float))
        alpha_values = np.asarray(
            [alpha_from_skewness(value) for value in skewness_values], dtype=float
        )
        # Soft-limit extreme skew proposals from broad windows so shape priors
        # stay focused on dominant peak mass instead of long low-amplitude tails.
        alpha_soft_cap = 2.5
        alpha_values = alpha_soft_cap * np.tanh(alpha_values / alpha_soft_cap)
        delta_values = alpha_values / np.sqrt(1.0 + alpha_values**2)
        variance_factor = 1.0 - (2.0 * delta_values * delta_values / np.pi)
        omega_values = np.full_like(sigma_values, np.nan, dtype=float)
        valid_sigma = (
            np.isfinite(sigma_values)
            & (sigma_values > 1e-12)
            & np.isfinite(variance_factor)
            & (variance_factor > 1e-8)
        )
        omega_values[valid_sigma] = sigma_values[valid_sigma] / np.sqrt(
            variance_factor[valid_sigma]
        )
        mu_values_from_centroid = (
            centroid_values - omega_values * delta_values * np.sqrt(2.0 / np.pi)
        )
        mu_values_from_centroid = np.where(
            np.isfinite(mu_values_from_centroid),
            mu_values_from_centroid,
            centroid_values,
        )
        # Main-peak center prior: use apex-time candidates and prominence-weighting
        # to avoid shoulder-driven centroid shifts.
        if apex_time_values.shape == mu_values_from_centroid.shape:
            mu_values = np.asarray(apex_time_values, dtype=float)
            mu_values = np.where(
                np.isfinite(mu_values), mu_values, mu_values_from_centroid
            )
        else:
            mu_values = mu_values_from_centroid

        base_weights = np.ones_like(mu_values, dtype=float)
        if apex_height_values.shape == mu_values.shape:
            apex_positive = np.clip(apex_height_values, a_min=0.0, a_max=None)
            finite_apex = apex_positive[
                np.isfinite(apex_positive) & (apex_positive > 0.0)
            ]
            if finite_apex.size > 0:
                apex_reference = float(np.median(finite_apex))
                apex_reference = max(apex_reference, 1e-12)
                apex_norm = np.clip(
                    apex_positive / apex_reference, a_min=0.0, a_max=None
                )

                # Strongly favor dominant traces: low-apex traces (often missing-peak
                # or artifact-dominated) get near-zero influence on shape priors.
                base_weights = apex_norm**2
                base_weights = np.where(apex_norm >= 0.25, base_weights, 0.0)
                base_weights = np.minimum(base_weights, 25.0)

        # Intentionally keep weighting based on apex height only.
        # Do not apply additional prominence scaling against mean window height.

        mu_loc_seed, mu_scale_seed = _robust_location_scale(mu_values, scale_floor=1e-4)
        if np.isfinite(mu_loc_seed):
            center_band = max(1.25 * float(mu_scale_seed), 1e-4)
            centrality = np.exp(-0.5 * ((mu_values - mu_loc_seed) / center_band) ** 2)
            shape_weights = base_weights * centrality
        else:
            shape_weights = base_weights

        mu_loc, mu_scale = _weighted_robust_location_scale(
            mu_values, shape_weights, scale_floor=1e-4
        )
        sigma_model_values = np.where(
            np.isfinite(omega_values) & (omega_values > 1e-12),
            omega_values,
            sigma_values,
        )
        if (
            left_sigma_values.shape == sigma_values.shape
            and right_sigma_values.shape == sigma_values.shape
        ):
            # Use only the smallest side-width per trace to bias priors toward
            # the main narrow peak core instead of shoulder-inflated widths.
            sigma_smallest_values = np.fmin(left_sigma_values, right_sigma_values)
        else:
            sigma_smallest_values = np.full_like(
                sigma_model_values, np.nan, dtype=float
            )

        valid_smallest_sigma = np.isfinite(sigma_smallest_values) & (
            sigma_smallest_values > 1e-12
        )
        if int(np.sum(valid_smallest_sigma)) > 0:
            sigma_used = sigma_smallest_values[valid_smallest_sigma]
        else:
            valid_sigma_model = np.isfinite(sigma_model_values) & (
                sigma_model_values > 1e-12
            )
            sigma_used = sigma_model_values[valid_sigma_model]

        if sigma_used.size > 0:
            sigma_loc = float(np.mean(sigma_used))
            if sigma_used.size > 1:
                sigma_scale = float(np.std(sigma_used, ddof=1))
            else:
                sigma_scale = max(0.1 * sigma_loc, 1e-4)
            sigma_scale = float(max(sigma_scale, 1e-4))
        else:
            sigma_loc = np.nan
            sigma_scale = np.nan
        alpha_loc, alpha_scale = _weighted_robust_location_scale(
            alpha_values, shape_weights, scale_floor=0.1
        )
        area_loc, area_scale = _weighted_robust_location_scale(
            area_values, base_weights, scale_floor=1e-6
        )

        if np.isfinite(sigma_loc):
            sigma_loc = float(max(sigma_loc, 1e-6))
        if np.isfinite(area_loc):
            area_loc = float(max(area_loc, 1e-8))

        finite_mu_count = int(np.sum(np.isfinite(mu_values)))
        hints.append(
            PeakPriorHints(
                mu_loc=mu_loc,
                mu_scale=mu_scale,
                sigma_loc=sigma_loc,
                sigma_scale=sigma_scale,
                alpha_loc=alpha_loc,
                alpha_scale=alpha_scale,
                area_loc=area_loc,
                area_scale=area_scale,
                trace_count=finite_mu_count,
            )
        )

    return hints


def metrics_list_to_arrays(
    metrics_list: Sequence[PeakMomentMetrics],
) -> dict[str, np.ndarray]:
    """Convert a list of dataclass metrics into a dict of float arrays."""
    if len(metrics_list) == 0:
        return {}

    keys = list(asdict(metrics_list[0]).keys())
    stacked: dict[str, np.ndarray] = {}
    for key in keys:
        stacked[key] = np.asarray(
            [getattr(metric, key) for metric in metrics_list], dtype=float
        )
    return stacked


def summarize_metrics(metrics: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    """Compute quick robust summaries for each metric array."""
    summary: dict[str, dict[str, Any]] = {}
    for key, values in metrics.items():
        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            summary[key] = {"count": 0, "median": np.nan, "mad": np.nan}
            continue
        median_value = float(np.median(finite_values))
        mad_value = float(np.median(np.abs(finite_values - median_value)))
        summary[key] = {
            "count": int(finite_values.size),
            "median": median_value,
            "mad": mad_value,
        }
    return summary


__all__ = [
    "PeakPriorHints",
    "PeakMomentMetrics",
    "alpha_from_skewness",
    "compute_peak_moment_metrics",
    "compute_peak_moment_metrics_batch",
    "compute_peak_moment_metrics_from_peak_masks",
    "estimate_skew_normal_prior_hints",
    "metrics_list_to_arrays",
    "summarize_metrics",
]
