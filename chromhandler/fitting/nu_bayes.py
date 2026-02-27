from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_value

from .baseline import BaselineEstimate
from .data import BaselineAnnotation, PeakAnnotation, baseline_to_mask, peaks_to_mask
from .moments import (
    PeakMomentMetrics,
    PeakPriorHints,
    alpha_from_skewness,
    compute_peak_moment_metrics_from_peak_masks,
    estimate_skew_normal_prior_hints,
    metrics_list_to_arrays,
    summarize_metrics,
)
from .peak_models import (
    SAMPLED_PARAMETER_NAMES,
    skew_mixture_area,
    skew_normal_pdf,
)
from .peak_models import (
    model as peak_model,
)
from .shift import ShiftAlignmentResult, align_chromatogram_shifts

numpyro.set_host_device_count(8)


def _interpolate_crossing(
    x0: float, y0: float, x1: float, y1: float, level: float
) -> float:
    """Linearly interpolate crossing x for y=level between two samples."""
    denominator = float(y1 - y0)
    if not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        return float(0.5 * (x0 + x1))
    t = float((level - y0) / denominator)
    t = float(np.clip(t, 0.0, 1.0))
    return float(x0 + t * (x1 - x0))


def _compute_normalized_fwhm(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    half_level: float = 0.5,
) -> dict[str, float | np.ndarray]:
    """Compute FWHM from a baseline-corrected peak trace."""
    x_array = np.asarray(x_values, dtype=float).reshape(-1)
    y_array = np.asarray(y_values, dtype=float).reshape(-1)
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    if int(np.sum(finite)) < 3:
        return {
            "x": np.asarray([], dtype=float),
            "y_norm": np.asarray([], dtype=float),
            "apex_time": np.nan,
            "apex_height": np.nan,
            "left_time": np.nan,
            "right_time": np.nan,
            "fwhm": np.nan,
            "half_level": float(half_level),
        }

    x_sorted = x_array[finite]
    y_sorted = y_array[finite]
    order = np.argsort(x_sorted)
    x_sorted = x_sorted[order]
    y_sorted = y_sorted[order]
    y_positive = np.clip(y_sorted, a_min=0.0, a_max=None)

    apex_index = int(np.argmax(y_positive))
    apex_height = float(y_positive[apex_index])
    apex_time = float(x_sorted[apex_index])
    if not np.isfinite(apex_height) or apex_height <= 1e-12:
        return {
            "x": x_sorted,
            "y_norm": np.zeros_like(x_sorted, dtype=float),
            "apex_time": apex_time,
            "apex_height": apex_height,
            "left_time": np.nan,
            "right_time": np.nan,
            "fwhm": np.nan,
            "half_level": float(half_level),
        }

    y_norm = y_positive / apex_height
    level = float(np.clip(half_level, 1e-6, 1.0 - 1e-6))

    left_time = np.nan
    for index in range(apex_index - 1, -1, -1):
        y0 = float(y_norm[index])
        y1 = float(y_norm[index + 1])
        if y0 <= level <= y1 and y1 > y0:
            left_time = _interpolate_crossing(
                float(x_sorted[index]),
                y0,
                float(x_sorted[index + 1]),
                y1,
                level,
            )
            break

    right_time = np.nan
    for index in range(apex_index, y_norm.size - 1):
        y0 = float(y_norm[index])
        y1 = float(y_norm[index + 1])
        if y0 >= level >= y1 and y0 > y1:
            right_time = _interpolate_crossing(
                float(x_sorted[index]),
                y0,
                float(x_sorted[index + 1]),
                y1,
                level,
            )
            break

    if np.isfinite(left_time) and np.isfinite(right_time) and right_time > left_time:
        fwhm = float(right_time - left_time)
    else:
        fwhm = np.nan

    return {
        "x": x_sorted,
        "y_norm": y_norm,
        "apex_time": apex_time,
        "apex_height": apex_height,
        "left_time": float(left_time),
        "right_time": float(right_time),
        "fwhm": float(fwhm),
        "half_level": level,
    }


def _robust_location_scale(
    values: np.ndarray,
    *,
    scale_floor: float = 1e-6,
) -> tuple[float, float]:
    """Return robust location/scale using median and MAD."""
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.nan, np.nan
    location = float(np.median(finite))
    mad = float(np.median(np.abs(finite - location)))
    scale = float(max(1.4826 * mad, scale_floor))
    return location, scale


def _mad_apex_gate(
    apex_times: np.ndarray,
    *,
    n_mad: float = 2.0,
) -> dict[str, float | np.ndarray]:
    """Build a robust inclusion gate around apex times."""
    apex_array = np.asarray(apex_times, dtype=float).reshape(-1)
    finite = np.isfinite(apex_array)
    keep = np.zeros_like(apex_array, dtype=bool)
    if int(np.sum(finite)) == 0:
        return {
            "keep_mask": keep,
            "center": np.nan,
            "scale": np.nan,
            "low": np.nan,
            "high": np.nan,
            "n_mad": float(n_mad),
        }

    location_raw, scale_raw = _robust_location_scale(
        apex_array[finite], scale_floor=1e-6
    )
    if not np.isfinite(scale_raw) or scale_raw <= 1e-12:
        keep[finite] = True
        return {
            "keep_mask": keep,
            "center": float(location_raw),
            "scale": float(scale_raw),
            "low": float(location_raw),
            "high": float(location_raw),
            "n_mad": float(n_mad),
        }

    threshold = float(n_mad * scale_raw)
    keep_initial = finite & (np.abs(apex_array - location_raw) <= threshold)
    if int(np.sum(keep_initial)) == 0:
        keep_initial = finite.copy()

    location_refined, scale_refined = _robust_location_scale(
        apex_array[keep_initial], scale_floor=1e-6
    )
    if not np.isfinite(scale_refined) or scale_refined <= 1e-12:
        keep = keep_initial
        low = float(location_refined)
        high = float(location_refined)
    else:
        refined_threshold = float(n_mad * scale_refined)
        keep = finite & (np.abs(apex_array - location_refined) <= refined_threshold)
        low = float(location_refined - refined_threshold)
        high = float(location_refined + refined_threshold)

    return {
        "keep_mask": keep,
        "center": float(location_refined),
        "scale": float(scale_refined),
        "low": low,
        "high": high,
        "n_mad": float(n_mad),
    }


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Return weighted median for finite positive-weight entries."""
    values_array = np.asarray(values, dtype=float).reshape(-1)
    weights_array = np.asarray(weights, dtype=float).reshape(-1)
    valid = (
        np.isfinite(values_array) & np.isfinite(weights_array) & (weights_array > 0.0)
    )
    if int(np.sum(valid)) == 0:
        return np.nan
    v = values_array[valid]
    w = weights_array[valid]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cumulative = np.cumsum(w, dtype=float)
    cutoff = 0.5 * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, cutoff, side="left"))
    index = int(np.clip(index, 0, v.size - 1))
    return float(v[index])


def _weighted_robust_location_scale(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    scale_floor: float = 1e-6,
) -> tuple[float, float]:
    """Return weighted robust location/scale using weighted MAD."""
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
    return float(location), float(scale)


def _skew_mode_offsets(alpha_values: np.ndarray) -> np.ndarray:
    """Return standardized skew-normal mode offsets for alpha values."""
    alpha_array = np.asarray(alpha_values, dtype=float)
    flat = alpha_array.reshape(-1)
    offsets = np.full_like(flat, np.nan, dtype=float)
    valid = np.isfinite(flat)
    if int(np.sum(valid)) == 0:
        return offsets.reshape(alpha_array.shape)

    grid = np.linspace(-8.0, 8.0, 2049, dtype=float)
    alpha_valid = np.asarray(flat[valid], dtype=np.float32)
    pdf = np.asarray(
        skew_normal_pdf(
            jnp.asarray(grid, dtype=jnp.float32),
            jnp.zeros((alpha_valid.shape[0],), dtype=jnp.float32),
            jnp.ones((alpha_valid.shape[0],), dtype=jnp.float32),
            jnp.asarray(alpha_valid, dtype=jnp.float32),
        ),
        dtype=float,
    )
    if pdf.ndim == 1:
        pdf = pdf[None, :]
    max_index = np.argmax(pdf, axis=1)
    offsets_valid = grid[max_index]
    offsets[np.flatnonzero(valid)] = offsets_valid
    return offsets.reshape(alpha_array.shape)


class Fitter:
    def __init__(
        self,
        time: jnp.ndarray,  # shape (n_chromatograms, n_timepoints)
        signal: jnp.ndarray,  # shape (n_chromatograms, n_timepoints)
        *,
        peaks: list[PeakAnnotation],
        baselines: list[BaselineAnnotation],
    ) -> None:
        self.time = time
        self.signal = signal
        self.peaks = list(peaks)
        self.baselines = list(baselines)
        self.aligned_time = self.time
        self.shift_result: ShiftAlignmentResult | None = None
        self.shift_samples = jnp.zeros((self.signal.shape[0]))
        self.shift_time = jnp.zeros((self.signal.shape[0]))
        self.alignment_mask: jnp.ndarray | None = None
        self.peak_moment_metrics: list[list[PeakMomentMetrics]] = []
        self.peak_prior_hints: list[PeakPriorHints] = []
        self._baseline_estimate_cache: dict[bool, list[BaselineEstimate]] = {}
        self.mu_init = jnp.zeros((self.signal.shape[0], 0), dtype=jnp.float32)
        self.sigma_init = jnp.zeros((self.signal.shape[0], 0), dtype=jnp.float32)
        self.A_init = jnp.zeros((self.signal.shape[0], 0), dtype=jnp.float32)
        self.alpha_init = jnp.zeros((self.signal.shape[0], 0), dtype=jnp.float32)
        self.model_inputs: dict[str, Any] | None = None
        self.samples: dict[str, jnp.ndarray] | None = None
        self.mcmc: Any = None
        self.idata: Any = None
        self.sigma_min, self.sigma_max = self._default_sigma_bounds()
        self.alpha_prior_sd = 1.0
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        if self.time.ndim != 2 or self.signal.ndim != 2:
            raise ValueError(
                "time and signal must be 2D with shape (n_chromatograms, n_timepoints)"
            )
        if self.time.shape != self.signal.shape:
            raise ValueError("time and signal must have the same shape")
        for peak in self.peaks:
            if not isinstance(peak, PeakAnnotation):
                raise TypeError(
                    "All entries in `peaks` must be PeakAnnotation instances."
                )
        for baseline in self.baselines:
            if not isinstance(baseline, BaselineAnnotation):
                raise TypeError(
                    "All entries in `baselines` must be BaselineAnnotation instances."
                )

    def _time_axis(self, *, use_aligned_time: bool) -> jnp.ndarray:
        return self.aligned_time if use_aligned_time else self.time

    def _peak_masks_for_time(self, time_axis: jnp.ndarray) -> jnp.ndarray:
        if len(self.peaks) == 0:
            return jnp.zeros((0, time_axis.shape[0], time_axis.shape[1]), dtype=bool)
        return peaks_to_mask(self.peaks, time_axis)

    def _baseline_mask_for_time(self, time_axis: jnp.ndarray) -> jnp.ndarray:
        return baseline_to_mask(self.baselines, time_axis)

    def get_peak_masks(self, *, use_aligned_time: bool = False) -> jnp.ndarray:
        return self._peak_masks_for_time(
            self._time_axis(use_aligned_time=use_aligned_time)
        )

    def get_baseline_mask(self, *, use_aligned_time: bool = False) -> jnp.ndarray:
        return self._baseline_mask_for_time(
            self._time_axis(use_aligned_time=use_aligned_time)
        )

    @property
    def peak_masks(self) -> jnp.ndarray:
        """Peak masks on the original time axis."""
        return self.get_peak_masks(use_aligned_time=False)

    @property
    def baseline_mask(self) -> jnp.ndarray:
        """Baseline mask on the original time axis."""
        return self.get_baseline_mask(use_aligned_time=False)

    def _compute_baseline_estimates(
        self, *, use_aligned_time: bool
    ) -> list[BaselineEstimate]:
        anchor_payload = self._collect_peak_edge_baseline_anchors(
            use_aligned_time=use_aligned_time
        )
        y_anchor = np.asarray(anchor_payload["y_anchor"], dtype=float)
        estimates: list[BaselineEstimate] = []
        for trace_index in range(y_anchor.shape[0]):
            anchor_value = (
                float(y_anchor[trace_index, 0])
                if y_anchor.ndim == 2 and y_anchor.shape[1] > 0
                else np.nan
            )
            if np.isfinite(anchor_value):
                intercept = anchor_value
            else:
                signal_row = np.asarray(self.signal[trace_index], dtype=float)
                finite_signal = signal_row[np.isfinite(signal_row)]
                intercept = (
                    float(np.nanmin(finite_signal)) if finite_signal.size > 0 else 0.0
                )
            estimates.append(
                BaselineEstimate(slope=0.0, intercept=intercept, r2=np.nan)
            )
        return estimates

    def _collect_peak_edge_baseline_anchors(
        self, *, use_aligned_time: bool
    ) -> dict[str, np.ndarray]:
        """Collect one anchor per trace for intercept-only baseline estimates.

        The anchor is the minimum signal value inside the union of all peak
        windows. If a trace has no finite samples in peak windows, falls back
        to the global trace minimum over finite points.
        """
        time_axis = self._time_axis(use_aligned_time=use_aligned_time)
        if len(self.peaks) == 0:
            raise ValueError(
                "Baseline estimation requires peak annotations. "
                "Define at least one peak window."
            )

        time_np = np.asarray(time_axis, dtype=float)
        signal_np = np.asarray(self.signal, dtype=float)
        peak_masks = np.asarray(
            self.get_peak_masks(use_aligned_time=use_aligned_time), dtype=bool
        )
        if peak_masks.shape[0] == 0:
            raise ValueError(
                "Baseline estimation requires at least one peak mask window."
            )
        peak_union_mask = np.any(peak_masks, axis=0)
        n_trace = int(time_np.shape[0])
        n_time = int(time_np.shape[1])
        n_anchor = 1

        x_anchor = np.full((n_trace, n_anchor), np.nan, dtype=float)
        y_anchor = np.full((n_trace, n_anchor), np.nan, dtype=float)
        anchor_indices = np.full((n_trace, n_anchor), -1, dtype=int)

        for trace_index in range(n_trace):
            trace_time = time_np[trace_index]
            trace_signal = signal_np[trace_index]
            finite = np.isfinite(trace_time) & np.isfinite(trace_signal)
            if not np.any(finite):
                continue

            in_window = finite & peak_union_mask[trace_index]
            search_mask = in_window if np.any(in_window) else finite
            candidate_indices = np.flatnonzero(search_mask)
            if candidate_indices.size == 0:
                continue
            min_local = int(np.argmin(trace_signal[candidate_indices]))
            min_global = int(candidate_indices[min_local])
            anchor_indices[trace_index, 0] = min_global
            x_anchor[trace_index, 0] = trace_time[min_global]
            y_anchor[trace_index, 0] = trace_signal[min_global]

        return {
            "x_anchor": x_anchor,
            "y_anchor": y_anchor,
            "anchor_indices": anchor_indices,
            "n_time": np.asarray([n_time], dtype=int),
        }

    def _baseline_intercept_prior(
        self, *, use_aligned_time: bool
    ) -> tuple[np.ndarray, float, float]:
        """Return per-trace intercept anchors and global mean/std prior stats."""
        baseline_estimates = self.get_baseline_estimates(
            use_aligned_time=use_aligned_time
        )
        intercept_raw = np.asarray(
            [float(estimate.intercept) for estimate in baseline_estimates], dtype=float
        )
        finite = intercept_raw[np.isfinite(intercept_raw)]
        if finite.size == 0:
            prior_mean = 0.0
            prior_std = 1.0
        else:
            prior_mean = float(np.mean(finite))
            if finite.size > 1:
                prior_std = float(np.std(finite, ddof=1))
            else:
                prior_std = max(abs(prior_mean) * 0.05, 1.0)
            prior_std = max(prior_std, 1e-3)

        intercept_anchor = np.where(
            np.isfinite(intercept_raw), intercept_raw, prior_mean
        )
        return intercept_anchor.astype(float), float(prior_mean), float(prior_std)

    def get_baseline_estimates(
        self, *, use_aligned_time: bool = False
    ) -> list[BaselineEstimate]:
        key = bool(use_aligned_time)
        if key not in self._baseline_estimate_cache:
            self._baseline_estimate_cache[key] = self._compute_baseline_estimates(
                use_aligned_time=use_aligned_time
            )
        return self._baseline_estimate_cache[key]

    @property
    def baseline_estimates(self) -> list[BaselineEstimate]:
        """Baseline estimates on the original time axis."""
        return self.get_baseline_estimates(use_aligned_time=False)

    def _median_time_step_per_chromatogram(self) -> jnp.ndarray:
        """Return per-trace median time step."""
        diffs = jnp.abs(jnp.diff(self.time, axis=1))
        finite_diffs = jnp.where(jnp.isfinite(diffs), diffs, jnp.nan)
        median_step = jnp.nanmedian(finite_diffs, axis=1)
        return jnp.where(jnp.isfinite(median_step), median_step, 0.0)

    def _default_sigma_bounds(self) -> tuple[float, float]:
        diffs = np.abs(np.diff(np.asarray(self.time, dtype=float), axis=1))
        finite_diffs = diffs[np.isfinite(diffs)]
        if finite_diffs.size == 0:
            median_step = 1e-3
        else:
            median_step = float(np.nanmedian(finite_diffs))
        median_step = max(median_step, 1e-6)

        # Use peak-window spans as primary scale (sigma roughly span/6 for
        # a near-Gaussian shape that fits inside a user window), and keep
        # dt-based guards only as lower-resolution floors.
        spans = np.asarray(
            [
                float(peak.high) - float(peak.low)
                for peak in self.peaks
                if np.isfinite(float(peak.low))
                and np.isfinite(float(peak.high))
                and float(peak.high) > float(peak.low)
            ],
            dtype=float,
        )
        if spans.size > 0:
            span_median = float(np.nanmedian(spans))
            span_max = float(np.nanmax(spans))
            sigma_reference = max(span_median / 6.0, 1e-4)
            sigma_min = max(2.0 * median_step, 0.10 * sigma_reference, 1e-4)
            sigma_max = max(2.5 * sigma_reference, 0.50 * span_max, 5.0 * sigma_min)
        else:
            sigma_min = max(2.0 * median_step, 1e-4)
            sigma_max = max(120.0 * median_step, 5.0 * sigma_min)

        return sigma_min, sigma_max

    def _sorted_peaks(self) -> list[PeakAnnotation]:
        if len(self.peaks) == 0:
            return []
        centers = np.asarray(
            [0.5 * (float(peak.low) + float(peak.high)) for peak in self.peaks],
            dtype=float,
        )
        order = np.argsort(centers)
        return [self.peaks[int(index)] for index in order]

    def baseline_corrected_signal(
        self, *, use_aligned_time: bool = True
    ) -> jnp.ndarray:
        time_axis = self._time_axis(use_aligned_time=use_aligned_time)
        baseline_estimates = self.get_baseline_estimates(
            use_aligned_time=use_aligned_time
        )
        slopes = jnp.asarray(
            [float(estimate.slope) for estimate in baseline_estimates],
            dtype=jnp.float32,
        )
        intercepts = jnp.asarray(
            [float(estimate.intercept) for estimate in baseline_estimates],
            dtype=jnp.float32,
        )
        baseline = slopes[:, None] * time_axis + intercepts[:, None]
        corrected = jnp.asarray(self.signal, dtype=jnp.float32) - baseline
        finite_mask = jnp.isfinite(corrected) & jnp.isfinite(time_axis)
        return jnp.where(finite_mask, corrected, jnp.nan)

    def _build_component_metadata(self) -> dict[str, Any]:
        sorted_peaks = self._sorted_peaks()
        if len(sorted_peaks) == 0:
            raise ValueError("At least one peak annotation is required for fitting.")

        mu_lo: list[float] = []
        mu_hi: list[float] = []
        logical_mu_lo: list[float] = []
        logical_mu_hi: list[float] = []
        logical_main_component_index: list[int] = []
        logical_shoulder_component_index: list[int] = []
        logical_shoulder_side: list[int] = []
        component_to_logical_index: list[int] = []
        component_include_in_total_area: list[bool] = []

        component_index = 0
        for logical_index, peak in enumerate(sorted_peaks):
            logical_mu_lo.append(float(peak.low))
            logical_mu_hi.append(float(peak.high))

            main_index = component_index
            component_index += 1
            mu_lo.append(float(peak.low))
            mu_hi.append(float(peak.high))
            component_to_logical_index.append(logical_index)
            component_include_in_total_area.append(True)

            if peak.shoulder is None:
                shoulder_index = -1
                shoulder_side_code = 0
            else:
                shoulder_index = component_index
                component_index += 1
                mu_lo.append(float(peak.low))
                mu_hi.append(float(peak.high))
                component_to_logical_index.append(logical_index)
                component_include_in_total_area.append(not bool(peak.exclude_shoulder))
                shoulder_side_code = -1 if peak.shoulder == "left" else 1

            logical_main_component_index.append(main_index)
            logical_shoulder_component_index.append(shoulder_index)
            logical_shoulder_side.append(shoulder_side_code)

        return {
            "sorted_peaks": sorted_peaks,
            "mu_lo": jnp.asarray(mu_lo, dtype=jnp.float32),
            "mu_hi": jnp.asarray(mu_hi, dtype=jnp.float32),
            "logical_mu_lo": jnp.asarray(logical_mu_lo, dtype=jnp.float32),
            "logical_mu_hi": jnp.asarray(logical_mu_hi, dtype=jnp.float32),
            "logical_main_component_index": jnp.asarray(
                logical_main_component_index, dtype=jnp.int32
            ),
            "logical_shoulder_component_index": jnp.asarray(
                logical_shoulder_component_index, dtype=jnp.int32
            ),
            "logical_shoulder_side": jnp.asarray(
                logical_shoulder_side, dtype=jnp.int32
            ),
            "component_to_logical_index": jnp.asarray(
                component_to_logical_index, dtype=jnp.int32
            ),
            "component_include_in_total_area": jnp.asarray(
                component_include_in_total_area, dtype=bool
            ),
        }

    def _moment_metric_to_skew_guess(
        self,
        metric: PeakMomentMetrics,
        *,
        low: float,
        high: float,
        area_fallback: float,
    ) -> tuple[float, float, float, float]:
        span = max(float(high) - float(low), 1e-4)
        mu_fallback = 0.5 * (float(low) + float(high))
        sigma_fallback = max(span / 6.0, 1e-4)

        area = float(metric.area)
        centroid = float(metric.centroid)
        sigma_moment = float(metric.sigma)
        skewness = float(metric.skewness)

        alpha = 0.0
        if np.isfinite(skewness):
            alpha = float(alpha_from_skewness(skewness))

        if np.isfinite(sigma_moment) and sigma_moment > 1e-12:
            delta = alpha / np.sqrt(1.0 + alpha**2)
            variance_factor = max(1.0 - (2.0 * delta * delta / np.pi), 1e-8)
            sigma = sigma_moment / np.sqrt(variance_factor)
        else:
            sigma = sigma_fallback
        sigma = float(np.clip(sigma, 1e-4, max(span, 1e-4)))

        if np.isfinite(centroid):
            delta = alpha / np.sqrt(1.0 + alpha**2)
            mu = centroid - sigma * delta * np.sqrt(2.0 / np.pi)
        else:
            mu = mu_fallback
        mu = float(np.clip(mu, float(low), float(high)))

        if not np.isfinite(area) or area <= 1e-12:
            area = float(area_fallback)
        area = max(float(area), 1e-8)
        return area, mu, sigma, alpha

    def _build_component_initializers_from_moments(
        self,
        metrics_by_peak: list[list[PeakMomentMetrics]],
        metadata: dict[str, Any],
        *,
        time_axis: jnp.ndarray,
        signal_corrected: jnp.ndarray,
        peak_masks: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        sorted_peaks: list[PeakAnnotation] = metadata["sorted_peaks"]
        n_traces = int(self.signal.shape[0])
        n_components = int(np.asarray(metadata["mu_lo"]).shape[0])
        if len(metrics_by_peak) != len(sorted_peaks):
            raise ValueError(
                "Moment metric peak count does not match peak annotations. "
                f"metrics={len(metrics_by_peak)} annotations={len(sorted_peaks)}"
            )
        time_matrix = np.asarray(time_axis, dtype=float)
        signal_matrix = np.asarray(signal_corrected, dtype=float)
        peak_mask_matrix = np.asarray(peak_masks, dtype=bool)
        if peak_mask_matrix.shape[0] != len(sorted_peaks):
            raise ValueError(
                "Peak mask logical-peak count does not match peak annotations. "
                f"mask_peaks={peak_mask_matrix.shape[0]} annotations={len(sorted_peaks)}"
            )
        if peak_mask_matrix.shape[1:] != time_matrix.shape:
            raise ValueError(
                "Peak mask trace/time shape does not match time axis. "
                f"mask_shape={peak_mask_matrix.shape[1:]} time_shape={time_matrix.shape}"
            )

        mu_lo = np.asarray(metadata["mu_lo"], dtype=float)
        mu_hi = np.asarray(metadata["mu_hi"], dtype=float)
        main_index = np.asarray(metadata["logical_main_component_index"], dtype=int)
        shoulder_index = np.asarray(
            metadata["logical_shoulder_component_index"], dtype=int
        )
        shoulder_side = np.asarray(metadata["logical_shoulder_side"], dtype=int)
        sqrt_two_pi = float(np.sqrt(2.0 * np.pi))

        mu_init = np.broadcast_to(
            0.5 * (mu_lo + mu_hi), (n_traces, n_components)
        ).copy()
        sigma_init = np.broadcast_to(
            np.maximum((mu_hi - mu_lo) / 6.0, 1e-4), (n_traces, n_components)
        ).copy()
        A_init = np.full((n_traces, n_components), 1e-3, dtype=float)
        alpha_init = np.zeros((n_traces, n_components), dtype=float)

        for logical_index, peak in enumerate(sorted_peaks):
            peak_metrics = metrics_by_peak[logical_index]
            valid_peak_areas = np.asarray(
                [
                    float(item.area)
                    for item in peak_metrics
                    if np.isfinite(float(item.area)) and float(item.area) > 1e-12
                ],
                dtype=float,
            )
            if valid_peak_areas.size > 0:
                area_fallback = max(0.05 * float(np.nanmedian(valid_peak_areas)), 1e-8)
            elif logical_index < len(self.peak_prior_hints) and np.isfinite(
                float(self.peak_prior_hints[logical_index].area_loc)
            ):
                area_fallback = max(
                    0.05 * float(self.peak_prior_hints[logical_index].area_loc),
                    1e-8,
                )
            else:
                area_fallback = 1e-3

            m_idx = int(main_index[logical_index])
            s_idx = int(shoulder_index[logical_index])
            side = int(shoulder_side[logical_index])
            mu_trace_guess = np.full((n_traces,), np.nan, dtype=float)
            sigma_trace_guess = np.full((n_traces,), np.nan, dtype=float)

            # First pass: full-window moment guesses for mu/sigma/alpha.
            for trace_index in range(n_traces):
                metric = peak_metrics[trace_index]
                _, mu_guess, sigma_guess, alpha_guess = (
                    self._moment_metric_to_skew_guess(
                        metric,
                        low=float(peak.low),
                        high=float(peak.high),
                        area_fallback=area_fallback,
                    )
                )
                mu_trace_guess[trace_index] = mu_guess
                sigma_trace_guess[trace_index] = sigma_guess

                if s_idx < 0:
                    mu_init[trace_index, m_idx] = mu_guess
                    sigma_init[trace_index, m_idx] = sigma_guess
                    alpha_init[trace_index, m_idx] = alpha_guess
                    continue

                span = max(float(peak.high) - float(peak.low), 1e-4)
                offset = max(0.08 * span, min(0.25 * span, 0.8 * sigma_guess))
                direction = 1.0 if side > 0 else -1.0
                main_mu = np.clip(
                    mu_guess - 0.5 * direction * offset,
                    float(peak.low),
                    float(peak.high),
                )
                shoulder_mu = np.clip(
                    mu_guess + 0.5 * direction * offset,
                    float(peak.low),
                    float(peak.high),
                )

                mu_init[trace_index, m_idx] = main_mu
                mu_init[trace_index, s_idx] = shoulder_mu
                sigma_init[trace_index, m_idx] = sigma_guess
                sigma_init[trace_index, s_idx] = max(0.75 * sigma_guess, 1e-4)
                alpha_init[trace_index, m_idx] = alpha_guess
                alpha_init[trace_index, s_idx] = 0.5 * alpha_guess

            # Second pass: refine area using mean mu/sigma and local window baseline.
            hint = (
                self.peak_prior_hints[logical_index]
                if logical_index < len(self.peak_prior_hints)
                else None
            )
            mu_hint = np.nan if hint is None else float(hint.mu_loc)
            sigma_hint = np.nan if hint is None else float(hint.sigma_loc)
            mu_candidates = mu_trace_guess[np.isfinite(mu_trace_guess)]
            sigma_candidates = sigma_trace_guess[
                np.isfinite(sigma_trace_guess) & (sigma_trace_guess > 0.0)
            ]
            if mu_candidates.size > 0:
                mean_mu = float(np.nanmedian(mu_candidates))
            elif np.isfinite(mu_hint):
                mean_mu = mu_hint
            else:
                mean_mu = 0.5 * (float(peak.low) + float(peak.high))
            mean_mu = float(np.clip(mean_mu, float(peak.low), float(peak.high)))

            if sigma_candidates.size > 0:
                mean_sigma = float(np.nanmedian(sigma_candidates))
            elif np.isfinite(sigma_hint) and sigma_hint > 0.0:
                mean_sigma = sigma_hint
            else:
                mean_sigma = max((float(peak.high) - float(peak.low)) / 6.0, 1e-4)
            mean_sigma = float(
                np.clip(
                    mean_sigma,
                    max(float(self.sigma_min), 1e-4),
                    max(float(self.sigma_max), 1.5 * float(self.sigma_min)),
                )
            )

            for trace_index in range(n_traces):
                trace_mask = peak_mask_matrix[logical_index, trace_index]
                trace_time = time_matrix[trace_index]
                trace_signal = signal_matrix[trace_index]
                finite_window = (
                    trace_mask & np.isfinite(trace_time) & np.isfinite(trace_signal)
                )
                # If evidence is weak/negative after local detrending, keep a tiny
                # non-zero floor so the model can still recover if data demands it.
                min_area_guess = max(0.02 * float(area_fallback), 1e-8)
                area_guess = min_area_guess
                if np.any(finite_window):
                    candidate_idx = np.flatnonzero(finite_window)
                    nearest_idx = int(
                        candidate_idx[
                            int(np.argmin(np.abs(trace_time[candidate_idx] - mean_mu)))
                        ]
                    )
                    x_mu = float(trace_time[nearest_idx])
                    y_mu = float(trace_signal[nearest_idx])

                    # Build a local linear baseline from window edges only.
                    edge_count = int(max(1, min(4, candidate_idx.size // 4)))
                    left_idx = candidate_idx[:edge_count]
                    right_idx = candidate_idx[-edge_count:]
                    left_x = float(np.nanmedian(trace_time[left_idx]))
                    right_x = float(np.nanmedian(trace_time[right_idx]))
                    left_y = float(np.nanmedian(trace_signal[left_idx]))
                    right_y = float(np.nanmedian(trace_signal[right_idx]))

                    if (
                        np.isfinite(left_x)
                        and np.isfinite(right_x)
                        and np.isfinite(left_y)
                        and np.isfinite(right_y)
                        and right_x > left_x
                    ):
                        local_slope = (right_y - left_y) / (right_x - left_x)
                        baseline_at_mu = left_y + local_slope * (x_mu - left_x)
                    elif np.isfinite(left_y) and np.isfinite(right_y):
                        baseline_at_mu = 0.5 * (left_y + right_y)
                    elif np.isfinite(left_y):
                        baseline_at_mu = left_y
                    elif np.isfinite(right_y):
                        baseline_at_mu = right_y
                    else:
                        baseline_at_mu = 0.0

                    apex_above_local_baseline = max(y_mu - baseline_at_mu, 0.0)
                    refined_area = apex_above_local_baseline * mean_sigma * sqrt_two_pi
                    if np.isfinite(refined_area) and refined_area > 1e-12:
                        area_guess = float(refined_area)

                area_guess = max(float(area_guess), 1e-8)
                if s_idx < 0:
                    A_init[trace_index, m_idx] = area_guess
                else:
                    A_init[trace_index, m_idx] = 0.85 * area_guess
                    A_init[trace_index, s_idx] = 0.15 * area_guess

        mu_init = np.clip(mu_init, mu_lo[None, :], mu_hi[None, :])
        sigma_init = np.clip(
            sigma_init,
            max(float(self.sigma_min), 1e-4),
            max(float(self.sigma_max), 1.5 * float(self.sigma_min)),
        )
        A_init = np.maximum(A_init, 1e-8)
        return (
            jnp.asarray(mu_init, dtype=jnp.float32),
            jnp.asarray(sigma_init, dtype=jnp.float32),
            jnp.asarray(A_init, dtype=jnp.float32),
            jnp.asarray(alpha_init, dtype=jnp.float32),
        )

    def _build_component_initializers_from_fwhm(
        self,
        metadata: dict[str, Any],
        *,
        use_aligned_time: bool,
        time_axis: jnp.ndarray,
        signal_corrected: jnp.ndarray,
        peak_masks: jnp.ndarray,
        half_level: float = 0.5,
        apex_gate_n_mad: float = 2.0,
        alpha_soft_cap: float = 2.5,
    ) -> tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, list[PeakPriorHints]
    ]:
        """Build skew-normal initializers and pooled priors from gated FWHM traces."""
        sorted_peaks: list[PeakAnnotation] = metadata["sorted_peaks"]
        n_traces = int(self.signal.shape[0])
        n_components = int(np.asarray(metadata["mu_lo"]).shape[0])
        n_logical = len(sorted_peaks)

        time_matrix = np.asarray(time_axis, dtype=float)
        signal_matrix = np.asarray(signal_corrected, dtype=float)
        peak_mask_matrix = np.asarray(peak_masks, dtype=bool)
        if peak_mask_matrix.shape[0] != n_logical:
            raise ValueError(
                "Peak mask logical-peak count does not match peak annotations. "
                f"mask_peaks={peak_mask_matrix.shape[0]} annotations={n_logical}"
            )
        if peak_mask_matrix.shape[1:] != time_matrix.shape:
            raise ValueError(
                "Peak mask trace/time shape does not match time axis. "
                f"mask_shape={peak_mask_matrix.shape[1:]} time_shape={time_matrix.shape}"
            )

        mu_lo = np.asarray(metadata["mu_lo"], dtype=float)
        mu_hi = np.asarray(metadata["mu_hi"], dtype=float)
        main_index = np.asarray(metadata["logical_main_component_index"], dtype=int)
        shoulder_index = np.asarray(
            metadata["logical_shoulder_component_index"], dtype=int
        )
        shoulder_side = np.asarray(metadata["logical_shoulder_side"], dtype=int)
        gaussian_hwhm_factor = float(np.sqrt(2.0 * np.log(2.0)))
        sqrt_two_pi = float(np.sqrt(2.0 * np.pi))

        mu_init = np.broadcast_to(
            0.5 * (mu_lo + mu_hi), (n_traces, n_components)
        ).copy()
        sigma_init = np.broadcast_to(
            np.maximum((mu_hi - mu_lo) / 6.0, 1e-4), (n_traces, n_components)
        ).copy()
        A_init = np.full((n_traces, n_components), 1e-8, dtype=float)
        alpha_init = np.zeros((n_traces, n_components), dtype=float)

        fwhm_payload = self.compute_peak_fwhm(
            use_aligned_time=use_aligned_time,
            half_level=half_level,
            apply_apex_gate=True,
            apex_gate_n_mad=apex_gate_n_mad,
        )
        valid_trace = np.asarray(fwhm_payload["valid_trace"], dtype=bool)
        gate_keep = np.asarray(fwhm_payload["gate_keep"], dtype=bool)
        apex_time_all = np.asarray(fwhm_payload["apex_time_all"], dtype=float)
        apex_height_all = np.asarray(fwhm_payload["apex_height_all"], dtype=float)
        left_time_all = np.asarray(fwhm_payload["left_time_all"], dtype=float)
        right_time_all = np.asarray(fwhm_payload["right_time_all"], dtype=float)

        prior_hints: list[PeakPriorHints] = []
        for logical_index, peak in enumerate(sorted_peaks):
            low = float(peak.low)
            high = float(peak.high)
            span = max(high - low, 1e-4)
            m_idx = int(main_index[logical_index])
            s_idx = int(shoulder_index[logical_index])
            side = int(shoulder_side[logical_index])

            mode_trace = np.asarray(apex_time_all[:, logical_index], dtype=float)
            apex_height_trace = np.asarray(
                apex_height_all[:, logical_index], dtype=float
            )
            left_trace = np.asarray(left_time_all[:, logical_index], dtype=float)
            right_trace = np.asarray(right_time_all[:, logical_index], dtype=float)

            w_left = mode_trace - left_trace
            w_right = right_trace - mode_trace
            width_valid = (
                valid_trace[:, logical_index]
                & np.isfinite(mode_trace)
                & np.isfinite(apex_height_trace)
                & np.isfinite(w_left)
                & np.isfinite(w_right)
                & (w_left > 1e-8)
                & (w_right > 1e-8)
                & (apex_height_trace > 0.0)
            )
            keep_width = gate_keep[:, logical_index] & width_valid
            if int(np.sum(keep_width)) == 0:
                keep_width = width_valid.copy()

            sigma_left_trace = np.full((n_traces,), np.nan, dtype=float)
            sigma_right_trace = np.full((n_traces,), np.nan, dtype=float)
            sigma_trace = np.full((n_traces,), np.nan, dtype=float)
            alpha_trace = np.full((n_traces,), np.nan, dtype=float)
            xi_trace = np.full((n_traces,), np.nan, dtype=float)
            area_trace = np.full((n_traces,), 1e-8, dtype=float)

            if int(np.sum(keep_width)) > 0:
                sigma_left_trace[keep_width] = w_left[keep_width] / gaussian_hwhm_factor
                sigma_right_trace[keep_width] = (
                    w_right[keep_width] / gaussian_hwhm_factor
                )
                sigma_trace[keep_width] = np.sqrt(
                    0.5
                    * (
                        sigma_left_trace[keep_width] ** 2
                        + sigma_right_trace[keep_width] ** 2
                    )
                )
                sigma_trace[keep_width] = np.clip(
                    sigma_trace[keep_width],
                    max(float(self.sigma_min), 1e-4),
                    max(float(self.sigma_max), 1.5 * float(self.sigma_min)),
                )
                delta = (
                    sigma_right_trace[keep_width] - sigma_left_trace[keep_width]
                ) / (sigma_right_trace[keep_width] + sigma_left_trace[keep_width])
                delta = np.clip(delta, -0.95, 0.95)
                alpha_raw = delta / np.sqrt(np.maximum(1.0 - delta**2, 1e-8))
                alpha_trace[keep_width] = float(alpha_soft_cap) * np.tanh(
                    alpha_raw / max(float(alpha_soft_cap), 1e-6)
                )
                mode_offsets = _skew_mode_offsets(alpha_trace[keep_width])
                xi_trace[keep_width] = mode_trace[keep_width] - (
                    sigma_trace[keep_width] * mode_offsets
                )
                xi_trace[keep_width] = np.clip(xi_trace[keep_width], low, high)

            if int(np.sum(keep_width)) > 0:
                apex_kept = np.clip(
                    apex_height_trace[keep_width], a_min=0.0, a_max=None
                )
                positive_kept = apex_kept[np.isfinite(apex_kept) & (apex_kept > 0.0)]
                if positive_kept.size > 0:
                    apex_reference = max(float(np.nanmedian(positive_kept)), 1e-12)
                    weights = np.clip((apex_kept / apex_reference) ** 2, 0.0, 25.0)
                    weights = np.where(apex_kept >= 0.25 * apex_reference, weights, 0.0)
                else:
                    weights = np.ones((int(np.sum(keep_width)),), dtype=float)
            else:
                weights = np.asarray([], dtype=float)

            xi_kept = xi_trace[keep_width]
            sigma_kept = sigma_trace[keep_width]
            alpha_kept = alpha_trace[keep_width]
            mode_kept = mode_trace[keep_width]

            mu_loc, mu_scale = _weighted_robust_location_scale(
                xi_kept, weights, scale_floor=1e-4
            )
            sigma_loc, sigma_scale = _weighted_robust_location_scale(
                sigma_kept, weights, scale_floor=1e-4
            )
            alpha_loc, alpha_scale = _weighted_robust_location_scale(
                alpha_kept, weights, scale_floor=1e-3
            )
            mode_loc, _ = _weighted_robust_location_scale(
                mode_kept, weights, scale_floor=1e-4
            )

            if not np.isfinite(mu_loc):
                mu_loc = 0.5 * (low + high)
            if not np.isfinite(mu_scale) or mu_scale <= 0.0:
                mu_scale = max(0.02 * span, 1e-4)
            if not np.isfinite(sigma_loc) or sigma_loc <= 0.0:
                sigma_loc = max(span / 6.0, 1e-4)
            if not np.isfinite(sigma_scale) or sigma_scale <= 0.0:
                sigma_scale = max(0.2 * sigma_loc, 1e-4)
            if not np.isfinite(alpha_loc):
                alpha_loc = 0.0
            if not np.isfinite(alpha_scale) or alpha_scale <= 0.0:
                alpha_scale = max(float(self.alpha_prior_sd), 1e-3)

            mu_loc = float(np.clip(mu_loc, low, high))
            sigma_loc = float(
                np.clip(
                    sigma_loc,
                    max(float(self.sigma_min), 1e-4),
                    max(float(self.sigma_max), 1.5 * float(self.sigma_min)),
                )
            )
            sigma_scale = float(max(sigma_scale, 1e-4))
            alpha_scale = float(max(alpha_scale, 1e-3))

            if not np.isfinite(mode_loc):
                mode_offset_ref = float(_skew_mode_offsets(np.asarray([alpha_loc]))[0])
                mode_loc = float(mu_loc + sigma_loc * mode_offset_ref)
            mode_loc = float(np.clip(mode_loc, low, high))

            for trace_index in range(n_traces):
                if keep_width[trace_index]:
                    xi_value = float(xi_trace[trace_index])
                    sigma_value = float(sigma_trace[trace_index])
                    alpha_value = float(alpha_trace[trace_index])
                    target_mode = float(mode_trace[trace_index])
                else:
                    xi_value = float(mu_loc)
                    sigma_value = float(sigma_loc)
                    alpha_value = float(alpha_loc)
                    target_mode = float(mode_loc)

                if not np.isfinite(xi_value):
                    xi_value = float(mu_loc)
                if not np.isfinite(sigma_value) or sigma_value <= 1e-8:
                    sigma_value = float(sigma_loc)
                if not np.isfinite(alpha_value):
                    alpha_value = float(alpha_loc)
                if not np.isfinite(target_mode):
                    target_mode = float(mode_loc)

                sigma_value = float(
                    np.clip(
                        sigma_value,
                        max(float(self.sigma_min), 1e-4),
                        max(float(self.sigma_max), 1.5 * float(self.sigma_min)),
                    )
                )
                xi_value = float(np.clip(xi_value, low, high))
                target_mode = float(np.clip(target_mode, low, high))

                active = (
                    peak_mask_matrix[logical_index, trace_index]
                    & np.isfinite(time_matrix[trace_index])
                    & np.isfinite(signal_matrix[trace_index])
                )
                if int(np.sum(active)) > 0:
                    candidate_idx = np.flatnonzero(active)
                else:
                    in_window = (
                        np.isfinite(time_matrix[trace_index])
                        & np.isfinite(signal_matrix[trace_index])
                        & (time_matrix[trace_index] >= low)
                        & (time_matrix[trace_index] <= high)
                    )
                    candidate_idx = np.flatnonzero(in_window)

                height = 0.0
                if candidate_idx.size > 0:
                    nearest_idx = int(
                        candidate_idx[
                            int(
                                np.argmin(
                                    np.abs(
                                        time_matrix[trace_index, candidate_idx]
                                        - target_mode
                                    )
                                )
                            )
                        ]
                    )
                    height = max(float(signal_matrix[trace_index, nearest_idx]), 0.0)

                pdf_at_target = float(
                    np.asarray(
                        skew_normal_pdf(
                            jnp.asarray([target_mode], dtype=jnp.float32),
                            jnp.asarray([xi_value], dtype=jnp.float32),
                            jnp.asarray([sigma_value], dtype=jnp.float32),
                            jnp.asarray([alpha_value], dtype=jnp.float32),
                        )[0, 0],
                        dtype=float,
                    )
                )
                if np.isfinite(pdf_at_target) and pdf_at_target > 1e-12:
                    area_value = height / pdf_at_target
                else:
                    area_value = height * sigma_value * sqrt_two_pi
                area_value = max(float(area_value), 1e-8)
                area_trace[trace_index] = area_value

                if s_idx < 0:
                    mu_init[trace_index, m_idx] = xi_value
                    sigma_init[trace_index, m_idx] = sigma_value
                    alpha_init[trace_index, m_idx] = alpha_value
                    A_init[trace_index, m_idx] = area_value
                else:
                    offset = max(0.08 * span, min(0.25 * span, 0.8 * sigma_value))
                    direction = 1.0 if side > 0 else -1.0
                    main_mu = float(
                        np.clip(xi_value - 0.5 * direction * offset, low, high)
                    )
                    shoulder_mu = float(
                        np.clip(xi_value + 0.5 * direction * offset, low, high)
                    )
                    mu_init[trace_index, m_idx] = main_mu
                    mu_init[trace_index, s_idx] = shoulder_mu
                    sigma_init[trace_index, m_idx] = sigma_value
                    sigma_init[trace_index, s_idx] = max(0.75 * sigma_value, 1e-4)
                    alpha_init[trace_index, m_idx] = alpha_value
                    alpha_init[trace_index, s_idx] = 0.5 * alpha_value
                    A_init[trace_index, m_idx] = 0.85 * area_value
                    A_init[trace_index, s_idx] = 0.15 * area_value

            area_kept = area_trace[keep_width]
            area_loc, area_scale = _weighted_robust_location_scale(
                area_kept, weights, scale_floor=1e-6
            )
            if not np.isfinite(area_loc):
                area_loc = float(np.nanmedian(area_trace[np.isfinite(area_trace)]))
            if not np.isfinite(area_scale) or area_scale <= 0.0:
                area_scale = max(0.25 * float(area_loc), 1e-6)
            area_loc = float(max(area_loc, 1e-8))
            area_scale = float(max(area_scale, 1e-6))

            prior_hints.append(
                PeakPriorHints(
                    mu_loc=float(mu_loc),
                    mu_scale=float(mu_scale),
                    sigma_loc=float(sigma_loc),
                    sigma_scale=float(sigma_scale),
                    alpha_loc=float(alpha_loc),
                    alpha_scale=float(alpha_scale),
                    area_loc=float(area_loc),
                    area_scale=float(area_scale),
                    trace_count=int(np.sum(keep_width)),
                )
            )

        mu_init = np.clip(mu_init, mu_lo[None, :], mu_hi[None, :])
        sigma_init = np.clip(
            sigma_init,
            max(float(self.sigma_min), 1e-4),
            max(float(self.sigma_max), 1.5 * float(self.sigma_min)),
        )
        A_init = np.maximum(A_init, 1e-8)
        return (
            jnp.asarray(mu_init, dtype=jnp.float32),
            jnp.asarray(sigma_init, dtype=jnp.float32),
            jnp.asarray(A_init, dtype=jnp.float32),
            jnp.asarray(alpha_init, dtype=jnp.float32),
            prior_hints,
        )

    def _build_model_inputs(
        self,
        *,
        use_aligned_time: bool = True,
        start_quantile: float = 0.005,
        end_quantile: float = 0.995,
        tail_window_sigma: float = 2.0,
    ) -> dict[str, Any]:
        metadata = self._build_component_metadata()
        time_axis = self._time_axis(use_aligned_time=use_aligned_time)
        signal_corrected = self.baseline_corrected_signal(
            use_aligned_time=use_aligned_time
        )
        peak_masks = self.get_peak_masks(use_aligned_time=use_aligned_time)
        mu_init, sigma_init, A_init, alpha_init, prior_hints = (
            self._build_component_initializers_from_fwhm(
                metadata,
                use_aligned_time=use_aligned_time,
                time_axis=time_axis,
                signal_corrected=signal_corrected,
                peak_masks=peak_masks,
                half_level=0.5,
                apex_gate_n_mad=2.0,
                alpha_soft_cap=2.5,
            )
        )
        self.peak_prior_hints = prior_hints
        self.mu_init = mu_init
        self.sigma_init = sigma_init
        self.A_init = A_init
        self.alpha_init = alpha_init

        intercept_anchor, intercept_prior_mean, intercept_prior_scale = (
            self._baseline_intercept_prior(use_aligned_time=use_aligned_time)
        )
        peak_mask = (
            jnp.any(peak_masks, axis=0)
            if peak_masks.shape[0] > 0
            else jnp.zeros(time_axis.shape, dtype=bool)
        )
        peak_mask_arg: jnp.ndarray | None = (
            peak_mask if bool(jnp.any(peak_mask)) else None
        )
        logical_mu_lo = np.asarray(metadata["logical_mu_lo"], dtype=float).reshape(-1)
        logical_mu_hi = np.asarray(metadata["logical_mu_hi"], dtype=float).reshape(-1)
        logical_span = np.maximum(logical_mu_hi - logical_mu_lo, 1e-4)
        n_logical = logical_mu_lo.size
        logical_main_component_index = np.asarray(
            metadata["logical_main_component_index"], dtype=int
        ).reshape(-1)

        mu_prior_loc = np.empty((n_logical,), dtype=float)
        mu_prior_scale = np.empty((n_logical,), dtype=float)
        for logical_index in range(n_logical):
            hint = (
                self.peak_prior_hints[logical_index]
                if logical_index < len(self.peak_prior_hints)
                else None
            )
            loc_hint = np.nan if hint is None else float(hint.mu_loc)
            scale_hint = np.nan if hint is None else float(hint.mu_scale)

            loc_fallback = 0.5 * (
                logical_mu_lo[logical_index] + logical_mu_hi[logical_index]
            )
            scale_fallback = max(0.02 * logical_span[logical_index], 1e-4)

            loc_value = loc_hint if np.isfinite(loc_hint) else loc_fallback
            scale_value = (
                scale_hint
                if np.isfinite(scale_hint) and scale_hint > 0.0
                else scale_fallback
            )
            mu_prior_loc[logical_index] = np.clip(
                loc_value, logical_mu_lo[logical_index], logical_mu_hi[logical_index]
            )
            mu_prior_scale[logical_index] = max(scale_value, 1e-6)

        sigma_min = float(self.sigma_min)
        sigma_max = float(self.sigma_max)
        sigma_prior_loc = np.empty((n_logical,), dtype=float)
        sigma_prior_scale = np.empty((n_logical,), dtype=float)
        sigma_init_np = np.asarray(sigma_init, dtype=float)
        alpha_init_np = np.asarray(alpha_init, dtype=float)
        for logical_index in range(n_logical):
            hint = (
                self.peak_prior_hints[logical_index]
                if logical_index < len(self.peak_prior_hints)
                else None
            )
            loc_hint = np.nan if hint is None else float(hint.sigma_loc)
            scale_hint = np.nan if hint is None else float(hint.sigma_scale)

            main_component = int(logical_main_component_index[logical_index])
            if 0 <= main_component < sigma_init_np.shape[1]:
                init_column = sigma_init_np[:, main_component]
            else:
                init_column = np.asarray([], dtype=float)
            finite_init = init_column[np.isfinite(init_column) & (init_column > 0.0)]
            if finite_init.size > 0:
                loc_fallback = float(np.nanmedian(finite_init))
            else:
                loc_fallback = max(logical_span[logical_index] / 6.0, 1e-4)

            scale_fallback = max(0.2 * loc_fallback, 1e-4)
            loc_value = (
                loc_hint if np.isfinite(loc_hint) and loc_hint > 0.0 else loc_fallback
            )
            scale_value = (
                scale_hint
                if np.isfinite(scale_hint) and scale_hint > 0.0
                else scale_fallback
            )
            sigma_prior_loc[logical_index] = np.clip(
                loc_value, sigma_min + 1e-6, sigma_max - 1e-6
            )
            sigma_prior_scale[logical_index] = max(scale_value, 1e-6)

        alpha_prior_loc = np.empty((n_logical,), dtype=float)
        alpha_prior_scale = np.empty((n_logical,), dtype=float)
        for logical_index in range(n_logical):
            hint = (
                self.peak_prior_hints[logical_index]
                if logical_index < len(self.peak_prior_hints)
                else None
            )
            alpha_loc_hint = np.nan if hint is None else float(hint.alpha_loc)
            alpha_scale_hint = np.nan if hint is None else float(hint.alpha_scale)

            main_component = int(logical_main_component_index[logical_index])
            if 0 <= main_component < alpha_init_np.shape[1]:
                alpha_init_col = alpha_init_np[:, main_component]
            else:
                alpha_init_col = np.asarray([], dtype=float)
            alpha_init_finite = alpha_init_col[np.isfinite(alpha_init_col)]
            if alpha_init_finite.size > 0:
                alpha_loc_fallback = float(np.nanmedian(alpha_init_finite))
                alpha_mad = float(
                    np.nanmedian(np.abs(alpha_init_finite - alpha_loc_fallback))
                )
                alpha_scale_fallback = max(
                    1.4826 * alpha_mad,
                    float(self.alpha_prior_sd),
                    1e-3,
                )
            else:
                alpha_loc_fallback = 0.0
                alpha_scale_fallback = max(float(self.alpha_prior_sd), 1e-3)
            alpha_loc_value = (
                alpha_loc_hint if np.isfinite(alpha_loc_hint) else alpha_loc_fallback
            )
            alpha_scale_value = (
                alpha_scale_hint
                if np.isfinite(alpha_scale_hint) and alpha_scale_hint > 0.0
                else alpha_scale_fallback
            )
            alpha_prior_loc[logical_index] = alpha_loc_value
            alpha_prior_scale[logical_index] = max(alpha_scale_value, 1e-3)

        model_inputs = {
            "x": jnp.asarray(time_axis, dtype=jnp.float32),
            "y": jnp.asarray(self.signal, dtype=jnp.float32),
            "mu_lo": jnp.asarray(metadata["mu_lo"], dtype=jnp.float32),
            "mu_hi": jnp.asarray(metadata["mu_hi"], dtype=jnp.float32),
            "sigma_min": float(self.sigma_min),
            "sigma_max": float(self.sigma_max),
            "logical_mu_lo": jnp.asarray(metadata["logical_mu_lo"], dtype=jnp.float32),
            "logical_mu_hi": jnp.asarray(metadata["logical_mu_hi"], dtype=jnp.float32),
            "logical_main_component_index": jnp.asarray(
                metadata["logical_main_component_index"], dtype=jnp.int32
            ),
            "logical_shoulder_component_index": jnp.asarray(
                metadata["logical_shoulder_component_index"], dtype=jnp.int32
            ),
            "logical_shoulder_side": jnp.asarray(
                metadata["logical_shoulder_side"], dtype=jnp.int32
            ),
            "component_to_logical_index": jnp.asarray(
                metadata["component_to_logical_index"], dtype=jnp.int32
            ),
            "component_include_in_total_area": jnp.asarray(
                metadata["component_include_in_total_area"], dtype=bool
            ),
            "mu_init": mu_init,
            "sigma_init": sigma_init,
            "A_init": A_init,
            "alpha_init": alpha_init,
            "mu_prior_loc": jnp.asarray(mu_prior_loc, dtype=jnp.float32),
            "mu_prior_scale": jnp.asarray(mu_prior_scale, dtype=jnp.float32),
            "sigma_prior_loc": jnp.asarray(sigma_prior_loc, dtype=jnp.float32),
            "sigma_prior_scale": jnp.asarray(sigma_prior_scale, dtype=jnp.float32),
            "alpha_prior_loc": jnp.asarray(alpha_prior_loc, dtype=jnp.float32),
            "alpha_prior_scale": jnp.asarray(alpha_prior_scale, dtype=jnp.float32),
            "peak_mask": peak_mask_arg,
            "intercept_anchor": jnp.asarray(intercept_anchor, dtype=jnp.float32),
            "intercept_prior_mean": float(intercept_prior_mean),
            "intercept_prior_scale": float(intercept_prior_scale),
            "alpha_prior_sd": float(self.alpha_prior_sd),
        }
        self.model_inputs = model_inputs
        return model_inputs

    def _build_init_values_for_nuts(
        self, model_inputs: dict[str, Any]
    ) -> dict[str, Any]:
        mu_init = np.asarray(model_inputs["mu_init"], dtype=float)
        sigma_init = np.asarray(model_inputs["sigma_init"], dtype=float)
        A_init = np.asarray(model_inputs["A_init"], dtype=float)
        main_index = np.asarray(model_inputs["logical_main_component_index"], dtype=int)
        shoulder_index = np.asarray(
            model_inputs["logical_shoulder_component_index"], dtype=int
        )
        has_shoulder = shoulder_index >= 0
        shoulder_safe = np.where(has_shoulder, shoulder_index, 0)

        mu_main = mu_init[:, main_index]
        mu_shoulder = mu_init[:, shoulder_safe]
        mu_center = mu_main + 0.5 * has_shoulder[None, :] * (mu_shoulder - mu_main)
        if "mu_prior_loc" in model_inputs:
            mu_prior_loc = np.asarray(
                model_inputs["mu_prior_loc"], dtype=float
            ).reshape(-1)
            if mu_prior_loc.size == mu_center.shape[1]:
                mu_center = np.broadcast_to(
                    mu_prior_loc[None, :], mu_center.shape
                ).copy()
        mu_center_shared = np.nanmedian(mu_center, axis=0)

        separation_trace = np.abs(mu_shoulder - mu_main)
        separation = np.nanmedian(separation_trace, axis=0)

        logical_mu_lo = np.asarray(model_inputs["logical_mu_lo"], dtype=float)
        logical_mu_hi = np.asarray(model_inputs["logical_mu_hi"], dtype=float)
        logical_span = np.maximum(logical_mu_hi - logical_mu_lo, 1e-4)
        separation_floor = 0.05 * logical_span
        separation = np.where(
            has_shoulder, np.maximum(separation, separation_floor), 1e-4
        )

        sigma_min = float(model_inputs["sigma_min"])
        sigma_max = float(model_inputs["sigma_max"])
        sigma_safe = np.clip(sigma_init, sigma_min, sigma_max)
        n_logical = int(np.asarray(main_index, dtype=int).shape[0])
        sigma_main = sigma_safe[:, main_index]
        sigma_logical_seed = np.nanmedian(sigma_main, axis=0)
        if "sigma_prior_loc" in model_inputs:
            sigma_prior_loc = np.asarray(model_inputs["sigma_prior_loc"], dtype=float)
            sigma_seed = None
            if sigma_prior_loc.ndim == 1 and sigma_prior_loc.size == n_logical:
                sigma_seed = sigma_prior_loc.copy()
            elif sigma_prior_loc.ndim == 2 and sigma_prior_loc.shape[1] == n_logical:
                sigma_seed = np.nanmedian(sigma_prior_loc, axis=0)
            if sigma_seed is not None:
                sigma_logical_seed = np.where(
                    np.isfinite(sigma_seed) & (sigma_seed > 0.0),
                    sigma_seed,
                    sigma_logical_seed,
                )
        sigma_logical_seed = np.clip(sigma_logical_seed, sigma_min, sigma_max)
        log_sigma = np.log(sigma_logical_seed)
        log_sigma_low = np.log(sigma_min)
        log_sigma_high = np.log(sigma_max)
        log_sigma_eps = 1e-6
        if log_sigma_high > log_sigma_low + 2.0 * log_sigma_eps:
            log_sigma = np.clip(
                log_sigma,
                log_sigma_low + log_sigma_eps,
                log_sigma_high - log_sigma_eps,
            )
        alpha = np.zeros((n_logical,), dtype=float)
        if "alpha_prior_loc" in model_inputs:
            alpha_prior_loc = np.asarray(model_inputs["alpha_prior_loc"], dtype=float)
            if alpha_prior_loc.ndim == 1 and alpha_prior_loc.size == alpha.size:
                alpha = np.where(np.isfinite(alpha_prior_loc), alpha_prior_loc, alpha)
            elif alpha_prior_loc.ndim == 2 and alpha_prior_loc.shape[1] == alpha.size:
                alpha_seed = np.nanmedian(alpha_prior_loc, axis=0)
                alpha = np.where(np.isfinite(alpha_seed), alpha_seed, alpha)
        area_low = np.full_like(A_init, 1e-8, dtype=float)
        area_high = np.maximum(
            2.0 * np.asarray(model_inputs["A_init"], dtype=float), area_low + 1e-8
        )
        area_span = np.maximum(area_high - area_low, 1e-12)
        area_eps = 1e-6 * area_span
        area = np.clip(
            np.asarray(A_init, dtype=float),
            area_low + area_eps,
            area_high - area_eps,
        )

        mu_center_low = logical_mu_lo
        mu_center_high = logical_mu_hi
        mu_center_span = np.maximum(mu_center_high - mu_center_low, 1e-6)
        mu_center_eps = 1e-6 * mu_center_span
        mu_center_shared = np.clip(
            mu_center_shared,
            mu_center_low + mu_center_eps,
            mu_center_high - mu_center_eps,
        )
        if "mu_prior_scale" in model_inputs:
            mu_prior_scale = np.asarray(
                model_inputs["mu_prior_scale"], dtype=float
            ).reshape(-1)
            if mu_prior_scale.size != n_logical:
                mu_prior_scale = np.broadcast_to(
                    np.nanmedian(mu_prior_scale), (n_logical,)
                )
        else:
            mu_prior_scale = 0.02 * logical_span
        mu_prior_scale = np.where(
            np.isfinite(mu_prior_scale) & (mu_prior_scale > 0.0),
            mu_prior_scale,
            0.02 * logical_span,
        )
        mu_trace_offset = np.zeros((mu_init.shape[0], n_logical), dtype=float)

        y_matrix = np.asarray(model_inputs["y"], dtype=float)
        sigma_y = np.nanstd(y_matrix, axis=1)
        sigma_y = np.where(np.isfinite(sigma_y), sigma_y, 1.0)
        sigma_y = np.maximum(sigma_y, 1e-3)

        intercept_anchor = np.asarray(
            model_inputs.get("intercept_anchor", np.zeros((y_matrix.shape[0],))),
            dtype=float,
        ).reshape(-1)
        if intercept_anchor.size != y_matrix.shape[0]:
            intercept_anchor = np.broadcast_to(
                np.nanmedian(intercept_anchor), (y_matrix.shape[0],)
            )
        return {
            "log_sigma": jnp.asarray(log_sigma, dtype=jnp.float32),
            "alpha": jnp.asarray(alpha, dtype=jnp.float32),
            "mu_center": jnp.asarray(mu_center_shared, dtype=jnp.float32),
            "mu_trace_offset": jnp.asarray(mu_trace_offset, dtype=jnp.float32),
            "separation": jnp.asarray(separation, dtype=jnp.float32),
            "A": jnp.asarray(area, dtype=jnp.float32),
            "sigma_y": jnp.asarray(sigma_y, dtype=jnp.float32),
            "baseline_intercept": jnp.asarray(
                np.clip(intercept_anchor, -499.0, 499.0), dtype=jnp.float32
            ),
        }

    def fit(
        self,
        *,
        use_aligned_time: bool = True,
        start_quantile: float = 0.005,
        end_quantile: float = 0.995,
        tail_window_sigma: float = 2.0,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        num_chains: int = 8,
        seed: int = 42,
        progress_bar: bool = True,
    ) -> Fitter:
        model_inputs = self._build_model_inputs(
            use_aligned_time=use_aligned_time,
            start_quantile=start_quantile,
            end_quantile=end_quantile,
            tail_window_sigma=tail_window_sigma,
        )
        init_values = self._build_init_values_for_nuts(model_inputs)

        self.mcmc = MCMC(
            NUTS(peak_model, init_strategy=init_to_value(values=init_values)),
            num_warmup=int(num_warmup),
            num_samples=int(num_samples),
            num_chains=int(num_chains),
            progress_bar=bool(progress_bar),
            chain_method="parallel" if int(num_chains) > 1 else "sequential",
        )
        self.mcmc.run(jax.random.PRNGKey(int(seed)), **model_inputs)
        self.samples = self.mcmc.get_samples()
        try:
            import arviz as az
        except Exception:
            self.idata = None
        else:
            self.idata = az.from_numpyro(self.mcmc)
        return self

    def predict_mean(self, *, use_aligned_time: bool = True) -> jnp.ndarray:
        if self.samples is None:
            raise RuntimeError("Call fit() before predict_mean().")

        x_values = self._time_axis(use_aligned_time=use_aligned_time)
        A = jnp.asarray(self.samples["A"], dtype=jnp.float32)
        mu = jnp.asarray(self.samples["mu"], dtype=jnp.float32)
        sigma = jnp.asarray(self.samples["sigma"], dtype=jnp.float32)
        alpha = jnp.asarray(
            self._posterior_alpha_component_draws(
                n_draw=int(A.shape[0]),
                n_chrom=int(A.shape[1]),
                n_component=int(A.shape[2]),
            ),
            dtype=jnp.float32,
        )
        draws = skew_mixture_area(x_values, A, mu, sigma, alpha)
        return jnp.mean(draws, axis=0)

    def save_arviz_summary_txt(
        self,
        save_path: str = "arviz_summary.txt",
        *,
        var_names: list[str] | None = None,
        round_to: int = 3,
    ) -> str:
        """Save posterior summary as plain text.

        Default behavior (``var_names=None``): summarizes all sampled model
        parameters (excluding observed data and deterministic-only sites).
        """
        if self.mcmc is None:
            raise RuntimeError("Call fit() before save_arviz_summary_txt().")

        if self.idata is None:
            try:
                import arviz as az
            except Exception as exc:
                raise RuntimeError(
                    "ArviZ is required to save summary text. Install `arviz`."
                ) from exc
            self.idata = az.from_numpyro(self.mcmc)
        else:
            import arviz as az

        posterior_vars = list(self.idata.posterior.data_vars)
        if var_names is None:
            requested_vars = [
                name for name in SAMPLED_PARAMETER_NAMES if name in posterior_vars
            ]
            if not requested_vars:
                observed_vars = (
                    set(self.idata.observed_data.data_vars)
                    if getattr(self.idata, "observed_data", None) is not None
                    else set()
                )
                requested_vars = [
                    name for name in posterior_vars if name not in observed_vars
                ]
        else:
            requested_vars = list(var_names)
        summary_vars = [name for name in requested_vars if name in posterior_vars]
        if not summary_vars:
            raise ValueError(
                "No valid posterior variables available for summary. "
                f"Requested: {requested_vars}"
            )

        summary_df = az.summary(self.idata, var_names=summary_vars, round_to=round_to)
        with open(save_path, "w", encoding="utf-8") as handle:
            handle.write(summary_df.to_string())
            handle.write("\n")
        return save_path

    def _ensure_idata(self) -> Any:
        if self.idata is not None:
            return self.idata
        if self.mcmc is None:
            raise RuntimeError("Call fit() before using ArviZ plots/summary.")
        try:
            import arviz as az
        except Exception as exc:
            raise RuntimeError(
                "ArviZ is required for posterior diagnostics. Install `arviz`."
            ) from exc
        self.idata = az.from_numpyro(self.mcmc)
        return self.idata

    def _posterior_baseline_intercept_draws(
        self,
        *,
        n_draw: int,
        n_chrom: int,
    ) -> np.ndarray:
        """Return baseline-intercept draws with shape ``[draw, chromatogram]``."""
        if self.samples is None:
            raise RuntimeError("Call fit() before extracting posterior baseline draws.")

        if "baseline_intercept" in self.samples:
            baseline = np.asarray(self.samples["baseline_intercept"], dtype=float)
        elif "baseline_intercept_delta" in self.samples:
            delta = np.asarray(self.samples["baseline_intercept_delta"], dtype=float)
            if self.model_inputs is None:
                raise RuntimeError(
                    "Model metadata is unavailable. Run fit() before plotting."
                )
            intercept_anchor = np.asarray(
                self.model_inputs.get("intercept_anchor", np.zeros((n_chrom,))),
                dtype=float,
            ).reshape(-1)
            if intercept_anchor.size != n_chrom:
                fallback = float(np.nanmedian(intercept_anchor))
                intercept_anchor = np.full((n_chrom,), fallback, dtype=float)
            baseline = intercept_anchor[None, :] - delta
        else:
            baseline = np.zeros((n_draw, n_chrom), dtype=float)

        if baseline.ndim == 1:
            if baseline.size == n_chrom:
                baseline = np.broadcast_to(baseline[None, :], (n_draw, n_chrom))
            elif baseline.size == n_draw and n_chrom == 1:
                baseline = baseline[:, None]
            else:
                raise ValueError(
                    "Baseline posterior has unsupported shape "
                    f"{baseline.shape}; expected [draw, chromatogram]."
                )
        if baseline.ndim != 2:
            raise ValueError(
                f"Baseline posterior has unsupported rank {baseline.ndim}; expected 2."
            )
        if baseline.shape[0] != n_draw:
            if baseline.shape[0] == 1:
                baseline = np.broadcast_to(baseline, (n_draw, baseline.shape[1]))
            else:
                raise ValueError(
                    "Baseline draw axis mismatch: "
                    f"expected {n_draw}, got {baseline.shape[0]}."
                )
        if baseline.shape[1] != n_chrom:
            if baseline.shape[1] == 1:
                baseline = np.broadcast_to(baseline, (baseline.shape[0], n_chrom))
            else:
                raise ValueError(
                    "Baseline chromatogram axis mismatch: "
                    f"expected {n_chrom}, got {baseline.shape[1]}."
                )
        return baseline

    def _posterior_alpha_component_draws(
        self,
        *,
        n_draw: int,
        n_chrom: int,
        n_component: int,
    ) -> np.ndarray:
        """Return alpha draws expanded to ``[draw, chromatogram, component]``."""
        if self.samples is None:
            raise RuntimeError("Call fit() before extracting posterior alpha draws.")

        if "alpha_component" in self.samples:
            alpha_values = np.asarray(self.samples["alpha_component"], dtype=float)
        elif "alpha" in self.samples:
            alpha_values = np.asarray(self.samples["alpha"], dtype=float)
        else:
            raise ValueError("Posterior samples do not contain `alpha`.")

        if alpha_values.ndim == 3:
            alpha_component = alpha_values
        elif alpha_values.ndim == 2:
            if self.model_inputs is None:
                raise RuntimeError(
                    "Model metadata is unavailable. Run fit() before plotting."
                )
            component_to_logical = np.asarray(
                self.model_inputs["component_to_logical_index"], dtype=int
            ).reshape(-1)
            if component_to_logical.size != n_component:
                raise ValueError(
                    "Component mapping size mismatch while expanding alpha: "
                    f"expected {n_component}, got {component_to_logical.size}."
                )
            if int(np.max(component_to_logical, initial=-1)) >= int(
                alpha_values.shape[1]
            ):
                raise ValueError(
                    "Logical alpha axis is too small for component mapping."
                )
            alpha_by_component = alpha_values[:, component_to_logical]
            alpha_component = np.broadcast_to(
                alpha_by_component[:, None, :],
                (alpha_values.shape[0], n_chrom, n_component),
            )
        else:
            raise ValueError(
                "Posterior alpha has unsupported rank "
                f"{alpha_values.ndim}; expected 2 or 3."
            )

        if alpha_component.shape[0] != n_draw:
            if alpha_component.shape[0] == 1:
                alpha_component = np.broadcast_to(
                    alpha_component,
                    (n_draw, alpha_component.shape[1], alpha_component.shape[2]),
                )
            else:
                raise ValueError(
                    "Alpha draw axis mismatch: "
                    f"expected {n_draw}, got {alpha_component.shape[0]}."
                )
        if alpha_component.shape[1] != n_chrom:
            if alpha_component.shape[1] == 1:
                alpha_component = np.broadcast_to(
                    alpha_component,
                    (alpha_component.shape[0], n_chrom, alpha_component.shape[2]),
                )
            else:
                raise ValueError(
                    "Alpha chromatogram axis mismatch: "
                    f"expected {n_chrom}, got {alpha_component.shape[1]}."
                )
        if alpha_component.shape[2] != n_component:
            raise ValueError(
                "Alpha component axis mismatch: "
                f"expected {n_component}, got {alpha_component.shape[2]}."
            )
        return alpha_component

    def plot_arviz_trace(
        self,
        *,
        save_path: str = "nu_bayes_trace.png",
        var_names: list[str] | None = None,
        compact: bool = True,
        combined: bool = False,
        dpi: int = 150,
    ) -> str:
        """Save an ArviZ trace plot for posterior diagnostics."""
        import arviz as az

        idata = self._ensure_idata()
        az.plot_trace(
            idata,
            var_names=var_names,
            compact=compact,
            combined=combined,
        )  # type: ignore[arg-type]
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return save_path

    def plot_arviz_pair(
        self,
        *,
        save_path: str = "nu_bayes_pair.png",
        var_names: list[str] | None = None,
        chromatogram_index: int | None = None,
        component_index: int | None = None,
        kind: str = "kde",
        marginals: bool = False,
        divergences: bool = False,
        point_estimate: str | None = "median",
        figsize: tuple[float, float] | None = None,
        max_subplots: int = 200,
        dpi: int = 80,
    ) -> str:
        """Save an ArviZ pair plot for posterior diagnostics.

        Defaults to ``A``, ``mu``, ``sigma``, ``alpha``,
        ``baseline_intercept``, and ``sigma_y`` when present in the posterior.

        Notes:
            - ``chromatogram_index`` applies to variables with a first
              non-sampling dimension indexing chromatograms (e.g. ``A``, ``mu``,
              ``sigma``, ``alpha``, ``sigma_y``).
            - ``component_index`` applies to variables with a second
              non-sampling dimension indexing components (e.g. ``A``, ``mu``,
              ``sigma``, ``alpha``).
            - ArviZ limits rendered subplots via ``plot.max_subplots``.
              Use ``max_subplots`` to avoid truncation for high-dimensional
              variables.
        """
        import arviz as az

        idata = self._ensure_idata()
        posterior_vars = list(idata.posterior.data_vars)
        requested_vars = (
            [
                "A",
                "mu",
                "sigma",
                "alpha",
                "baseline_intercept",
                "sigma_y",
            ]
            if var_names is None
            else list(var_names)
        )
        pair_vars = [name for name in requested_vars if name in posterior_vars]
        if not pair_vars:
            raise ValueError(
                "No valid posterior variables available for pair plot. "
                f"Requested: {requested_vars}"
            )

        coords: dict[str, int] = {}
        for name in pair_vars:
            var = idata.posterior[name]
            dims = [dim for dim in var.dims if dim not in ("chain", "draw")]
            if not dims:
                continue

            if chromatogram_index is not None:
                dim_name = dims[0]
                dim_size = int(var.sizes[dim_name])
                index = int(chromatogram_index)
                if index < 0 or index >= dim_size:
                    raise ValueError(
                        f"chromatogram_index={index} out of range for "
                        f"{name}.{dim_name} with size {dim_size}."
                    )
                coords[dim_name] = index

            if component_index is not None and len(dims) >= 2:
                dim_name = dims[1]
                dim_size = int(var.sizes[dim_name])
                index = int(component_index)
                if index < 0 or index >= dim_size:
                    raise ValueError(
                        f"component_index={index} out of range for "
                        f"{name}.{dim_name} with size {dim_size}."
                    )
                coords[dim_name] = index

        scalar_variable_count = 0
        for name in pair_vars:
            var = idata.posterior[name]
            dims = [dim for dim in var.dims if dim not in ("chain", "draw")]
            contribution = 1
            for dim_name in dims:
                if dim_name in coords:
                    continue
                contribution *= int(var.sizes[dim_name])
            scalar_variable_count += int(contribution)

        # ArviZ truncates silently when plot.max_subplots is too low.
        # Guard against that explicitly using a conservative panel bound.
        required_panels = int(max(1, scalar_variable_count * scalar_variable_count))
        if required_panels > int(max_subplots):
            raise ValueError(
                "Pair plot request would be truncated by ArviZ: "
                f"{scalar_variable_count} scalar variables imply up to "
                f"{required_panels} panels, "
                f"max_subplots={int(max_subplots)}. "
                "Reduce variables via var_names/chromatogram_index/component_index "
                "or raise max_subplots (may be slow)."
            )

        kind_resolved = kind
        marginals_resolved = bool(marginals)
        if kind == "kde" and scalar_variable_count > 24:
            kind_resolved = "scatter"
            marginals_resolved = False

        with az.rc_context({"plot.max_subplots": int(max_subplots)}):
            axes = az.plot_pair(
                idata,
                var_names=pair_vars,
                coords=coords or None,
                kind=kind_resolved,
                marginals=marginals_resolved,
                divergences=divergences,
                point_estimate=point_estimate,
                figsize=figsize,
            )
        if isinstance(axes, np.ndarray):
            figure = np.ravel(axes)[0].figure
        elif hasattr(axes, "figure"):
            figure = axes.figure
        else:
            figure = plt.gcf()
        figure.tight_layout()
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return save_path

    def _build_prior_guided_moment_guess(
        self,
        *,
        use_aligned_time: bool = True,
        start_quantile: float = 0.005,
        end_quantile: float = 0.995,
        tail_window_sigma: float = 2.0,
    ) -> dict[str, np.ndarray]:
        """Build deterministic prior-style peak guesses from current initializers."""
        model_inputs = self._build_model_inputs(
            use_aligned_time=use_aligned_time,
            start_quantile=start_quantile,
            end_quantile=end_quantile,
            tail_window_sigma=tail_window_sigma,
        )

        time = np.asarray(
            self._time_axis(use_aligned_time=use_aligned_time), dtype=float
        )
        signal_corrected = np.asarray(
            self.baseline_corrected_signal(use_aligned_time=use_aligned_time),
            dtype=float,
        )
        peak_masks = np.asarray(
            self.get_peak_masks(use_aligned_time=use_aligned_time), dtype=bool
        )
        if peak_masks.shape[0] == 0:
            raise ValueError("No peak masks available for moment plotting.")

        mu_prior_loc = np.asarray(model_inputs["mu_prior_loc"], dtype=float).reshape(-1)
        mu_prior_scale = np.asarray(
            model_inputs.get("mu_prior_scale", np.full_like(mu_prior_loc, np.nan)),
            dtype=float,
        ).reshape(-1)
        main_component_index = np.asarray(
            model_inputs["logical_main_component_index"], dtype=int
        ).reshape(-1)
        n_peak = int(peak_masks.shape[0])
        n_trace = int(time.shape[0])
        if (
            mu_prior_loc.size != n_peak
            or mu_prior_scale.size != n_peak
            or main_component_index.size != n_peak
        ):
            raise ValueError(
                "Prior/init/mask peak count mismatch. "
                f"mu_prior={mu_prior_loc.size}, mu_scale={mu_prior_scale.size}, "
                f"main_index={main_component_index.size}, masks={n_peak}"
            )

        mu_init = np.asarray(model_inputs["mu_init"], dtype=float)
        sigma_init = np.asarray(model_inputs["sigma_init"], dtype=float)
        alpha_init = np.asarray(model_inputs["alpha_init"], dtype=float)
        area_init = np.asarray(model_inputs["A_init"], dtype=float)
        mu_guess = np.asarray(mu_init[:, main_component_index], dtype=float)
        sigma_guess = np.asarray(sigma_init[:, main_component_index], dtype=float)
        alpha_guess = np.asarray(alpha_init[:, main_component_index], dtype=float)
        area_guess = np.asarray(area_init[:, main_component_index], dtype=float)
        if (
            mu_guess.shape != (n_trace, n_peak)
            or sigma_guess.shape != (n_trace, n_peak)
            or alpha_guess.shape != (n_trace, n_peak)
            or area_guess.shape != (n_trace, n_peak)
        ):
            raise ValueError(
                "Main-component initializer shapes do not match expected "
                f"(n_trace={n_trace}, n_peak={n_peak})."
            )
        area_guess = np.maximum(area_guess, 1e-8)
        return {
            "time": time,
            "signal_corrected": signal_corrected,
            "peak_masks": peak_masks,
            "mu_guess": mu_guess,
            "mu_prior_loc": mu_prior_loc,
            "mu_prior_scale": mu_prior_scale,
            "sigma_guess": sigma_guess,
            "alpha_guess": alpha_guess,
            "area_guess": area_guess,
        }

    def compute_peak_fwhm(
        self,
        *,
        use_aligned_time: bool = True,
        half_level: float = 0.5,
        apply_apex_gate: bool = True,
        apex_gate_n_mad: float = 2.0,
    ) -> dict[str, np.ndarray]:
        """Compute per-trace, per-peak FWHM on baseline-corrected masked data."""
        if not (0.0 < float(half_level) < 1.0):
            raise ValueError("half_level must satisfy 0 < half_level < 1.")
        if float(apex_gate_n_mad) <= 0.0:
            raise ValueError("apex_gate_n_mad must be > 0.")

        time = np.asarray(
            self._time_axis(use_aligned_time=use_aligned_time), dtype=float
        )
        signal_corrected = np.asarray(
            self.baseline_corrected_signal(use_aligned_time=use_aligned_time),
            dtype=float,
        )
        peak_masks = np.asarray(
            self.get_peak_masks(use_aligned_time=use_aligned_time), dtype=bool
        )
        if peak_masks.shape[0] == 0:
            raise ValueError("No peak masks available for FWHM computation.")

        n_peak = int(peak_masks.shape[0])
        n_trace = int(time.shape[0])
        apex_time_all = np.full((n_trace, n_peak), np.nan, dtype=float)
        apex_height_all = np.full((n_trace, n_peak), np.nan, dtype=float)
        left_time_all = np.full((n_trace, n_peak), np.nan, dtype=float)
        right_time_all = np.full((n_trace, n_peak), np.nan, dtype=float)
        fwhm_all = np.full((n_trace, n_peak), np.nan, dtype=float)
        valid_trace = np.zeros((n_trace, n_peak), dtype=bool)
        gate_keep = np.zeros((n_trace, n_peak), dtype=bool)
        gate_center = np.full((n_peak,), np.nan, dtype=float)
        gate_scale = np.full((n_peak,), np.nan, dtype=float)
        gate_low = np.full((n_peak,), np.nan, dtype=float)
        gate_high = np.full((n_peak,), np.nan, dtype=float)

        # Pass 1: gather per-trace apex/FWHM candidates.
        for peak_index in range(n_peak):
            for trace_index in range(n_trace):
                mask = peak_masks[peak_index, trace_index]
                finite = np.isfinite(time[trace_index]) & np.isfinite(
                    signal_corrected[trace_index]
                )
                active = mask & finite
                if int(np.sum(active)) < 3:
                    continue

                payload = _compute_normalized_fwhm(
                    time[trace_index, active],
                    signal_corrected[trace_index, active],
                    half_level=half_level,
                )
                x_curve = np.asarray(payload["x"], dtype=float)
                y_norm = np.asarray(payload["y_norm"], dtype=float)
                if x_curve.size < 3 or y_norm.size < 3:
                    continue
                apex_height_value = float(payload["apex_height"])
                if not np.isfinite(apex_height_value) or apex_height_value <= 1e-12:
                    continue
                valid_trace[trace_index, peak_index] = True
                apex_time_all[trace_index, peak_index] = float(payload["apex_time"])
                apex_height_all[trace_index, peak_index] = apex_height_value
                left_time_all[trace_index, peak_index] = float(payload["left_time"])
                right_time_all[trace_index, peak_index] = float(payload["right_time"])
                fwhm_all[trace_index, peak_index] = float(payload["fwhm"])

        # Pass 2: robust apex-time gate per peak.
        for peak_index in range(n_peak):
            apex_candidates = apex_time_all[:, peak_index]
            finite_candidates = (
                np.isfinite(apex_candidates) & valid_trace[:, peak_index]
            )
            if int(np.sum(finite_candidates)) == 0:
                continue

            if apply_apex_gate:
                gate = _mad_apex_gate(
                    apex_candidates[finite_candidates], n_mad=float(apex_gate_n_mad)
                )
                keep_sub = np.asarray(gate["keep_mask"], dtype=bool)
                keep_indices = np.flatnonzero(finite_candidates)
                keep_full = np.zeros((n_trace,), dtype=bool)
                keep_full[keep_indices] = keep_sub
                gate_keep[:, peak_index] = keep_full
                gate_center[peak_index] = float(gate["center"])
                gate_scale[peak_index] = float(gate["scale"])
                gate_low[peak_index] = float(gate["low"])
                gate_high[peak_index] = float(gate["high"])
            else:
                gate_keep[:, peak_index] = finite_candidates
                center, scale = _robust_location_scale(
                    apex_candidates[finite_candidates], scale_floor=1e-6
                )
                gate_center[peak_index] = float(center)
                gate_scale[peak_index] = float(scale)
                gate_low[peak_index] = np.nan
                gate_high[peak_index] = np.nan

        # Keep only gated traces in final outputs.
        fwhm = np.full((n_trace, n_peak), np.nan, dtype=float)
        apex_time = np.full((n_trace, n_peak), np.nan, dtype=float)
        apex_height = np.full((n_trace, n_peak), np.nan, dtype=float)
        left_time = np.full((n_trace, n_peak), np.nan, dtype=float)
        right_time = np.full((n_trace, n_peak), np.nan, dtype=float)
        use_mask = gate_keep & valid_trace
        fwhm[use_mask] = fwhm_all[use_mask]
        apex_time[use_mask] = apex_time_all[use_mask]
        apex_height[use_mask] = apex_height_all[use_mask]
        left_time[use_mask] = left_time_all[use_mask]
        right_time[use_mask] = right_time_all[use_mask]

        return {
            "fwhm": fwhm,
            "apex_time": apex_time,
            "apex_height": apex_height,
            "left_time": left_time,
            "right_time": right_time,
            "half_level": np.asarray(float(half_level), dtype=float),
            "valid_trace": valid_trace,
            "gate_keep": gate_keep,
            "gate_reject": valid_trace & (~gate_keep),
            "gate_center": gate_center,
            "gate_scale": gate_scale,
            "gate_low": gate_low,
            "gate_high": gate_high,
            "apex_time_all": apex_time_all,
            "apex_height_all": apex_height_all,
            "left_time_all": left_time_all,
            "right_time_all": right_time_all,
            "fwhm_all": fwhm_all,
            "apex_gate_n_mad": np.asarray(float(apex_gate_n_mad), dtype=float),
        }

    def plot_peak_fwhm(
        self,
        *,
        use_aligned_time: bool = True,
        half_level: float = 0.5,
        apply_apex_gate: bool = True,
        apex_gate_n_mad: float = 2.0,
        save_path: str = "nu_bayes_peak_fwhm.png",
        column_mode: str = "peak",
        chromatogram_indices: list[int] | None = None,
        peak_indices: list[int] | None = None,
        normalize_position: bool = True,
        data_alpha: float = 0.45,
        data_size: float = 8.0,
        line_width: float = 1.5,
        dpi: int = 150,
    ) -> str:
        """Plot normalized peak profiles with FWHM crossings.

        The y-axis is normalized by local apex height in each trace/peak mask.
        """
        if column_mode not in {"chromatogram", "peak"}:
            raise ValueError("column_mode must be 'chromatogram' or 'peak'.")
        if not (0.0 < float(half_level) < 1.0):
            raise ValueError("half_level must satisfy 0 < half_level < 1.")
        if float(apex_gate_n_mad) <= 0.0:
            raise ValueError("apex_gate_n_mad must be > 0.")
        if len(self.peaks) == 0:
            raise ValueError("No peaks are defined.")

        fwhm_payload = self.compute_peak_fwhm(
            use_aligned_time=use_aligned_time,
            half_level=half_level,
            apply_apex_gate=apply_apex_gate,
            apex_gate_n_mad=apex_gate_n_mad,
        )
        gate_keep = np.asarray(fwhm_payload["gate_keep"], dtype=bool)
        valid_trace = np.asarray(fwhm_payload["valid_trace"], dtype=bool)

        time = np.asarray(
            self._time_axis(use_aligned_time=use_aligned_time), dtype=float
        )
        signal_corrected = np.asarray(
            self.baseline_corrected_signal(use_aligned_time=use_aligned_time),
            dtype=float,
        )
        peak_masks = np.asarray(
            self.get_peak_masks(use_aligned_time=use_aligned_time), dtype=bool
        )
        if peak_masks.shape[0] == 0:
            raise ValueError("No peak masks available for FWHM plotting.")

        n_chrom = int(time.shape[0])
        n_peak = int(peak_masks.shape[0])
        chrom_sel = (
            list(range(n_chrom))
            if chromatogram_indices is None
            else [int(index) for index in chromatogram_indices]
        )
        peak_sel = (
            list(range(n_peak))
            if peak_indices is None
            else [int(index) for index in peak_indices]
        )
        if not chrom_sel:
            raise ValueError("chromatogram_indices resolves to an empty selection.")
        if not peak_sel:
            raise ValueError("peak_indices resolves to an empty selection.")
        if min(chrom_sel) < 0 or max(chrom_sel) >= n_chrom:
            raise ValueError(f"chromatogram_indices out of range [0, {n_chrom - 1}].")
        if min(peak_sel) < 0 or max(peak_sel) >= n_peak:
            raise ValueError(f"peak_indices out of range [0, {n_peak - 1}].")
        if column_mode == "chromatogram":
            n_rows = len(peak_sel)
            n_cols = len(chrom_sel)
            row_labels = [f"Peak {peak_index + 1}" for peak_index in peak_sel]
            col_labels = [f"Trace {chrom_index + 1}" for chrom_index in chrom_sel]
        else:
            n_rows = len(chrom_sel)
            n_cols = len(peak_sel)
            row_labels = [f"Trace {chrom_index + 1}" for chrom_index in chrom_sel]
            col_labels = [f"Peak {peak_index + 1}" for peak_index in peak_sel]

        figure, axes = plt.subplots(
            n_rows,
            n_cols,
            squeeze=False,
            figsize=(3.8 * n_cols, 2.6 * n_rows),
            constrained_layout=True,
        )

        for row_index in range(n_rows):
            for col_index in range(n_cols):
                ax = axes[row_index, col_index]
                if column_mode == "chromatogram":
                    peak_index = peak_sel[row_index]
                    chrom_index = chrom_sel[col_index]
                else:
                    chrom_index = chrom_sel[row_index]
                    peak_index = peak_sel[col_index]

                mask = peak_masks[peak_index, chrom_index]
                finite = np.isfinite(time[chrom_index]) & np.isfinite(
                    signal_corrected[chrom_index]
                )
                active = mask & finite
                if int(np.sum(active)) < 3:
                    ax.text(
                        0.5,
                        0.5,
                        "insufficient mask points",
                        ha="center",
                        va="center",
                        fontsize=8,
                        transform=ax.transAxes,
                    )
                    ax.grid(True, alpha=0.2)
                    continue

                payload = _compute_normalized_fwhm(
                    time[chrom_index, active],
                    signal_corrected[chrom_index, active],
                    half_level=half_level,
                )
                x_curve = np.asarray(payload["x"], dtype=float)
                y_norm = np.asarray(payload["y_norm"], dtype=float)
                if x_curve.size < 3:
                    ax.text(
                        0.5,
                        0.5,
                        "invalid normalized peak",
                        ha="center",
                        va="center",
                        fontsize=8,
                        transform=ax.transAxes,
                    )
                    ax.grid(True, alpha=0.2)
                    continue

                apex_time = float(payload["apex_time"])
                left_time = float(payload["left_time"])
                right_time = float(payload["right_time"])
                fwhm = float(payload["fwhm"])
                is_valid = bool(valid_trace[chrom_index, peak_index])
                is_kept = (
                    bool(gate_keep[chrom_index, peak_index]) if is_valid else False
                )

                if normalize_position and np.isfinite(apex_time):
                    x_plot = x_curve - apex_time
                    apex_plot = 0.0
                    left_plot = (
                        left_time - apex_time if np.isfinite(left_time) else np.nan
                    )
                    right_plot = (
                        right_time - apex_time if np.isfinite(right_time) else np.nan
                    )
                else:
                    x_plot = x_curve
                    apex_plot = apex_time
                    left_plot = left_time
                    right_plot = right_time

                trace_color = "tab:blue" if is_kept else "0.55"
                trace_alpha = data_alpha if is_kept else min(0.35, data_alpha)
                ax.scatter(
                    x_plot,
                    y_norm,
                    s=data_size,
                    alpha=trace_alpha,
                    color="0.35",
                    linewidths=0,
                )
                ax.plot(
                    x_plot,
                    y_norm,
                    color=trace_color,
                    linewidth=line_width,
                )
                ax.axhline(
                    float(half_level), color="tab:orange", linestyle="--", linewidth=1.0
                )
                if np.isfinite(apex_plot):
                    ax.axvline(apex_plot, color="0.25", linestyle=":", linewidth=1.0)
                if np.isfinite(left_plot):
                    ax.axvline(
                        left_plot, color="tab:green", linestyle="--", linewidth=0.9
                    )
                if np.isfinite(right_plot):
                    ax.axvline(
                        right_plot, color="tab:green", linestyle="--", linewidth=0.9
                    )
                if is_kept and np.isfinite(left_plot) and np.isfinite(right_plot):
                    ax.hlines(
                        float(half_level),
                        left_plot,
                        right_plot,
                        colors="tab:red",
                        linewidth=1.8,
                    )
                if is_kept and np.isfinite(fwhm):
                    ax.text(
                        0.02,
                        0.98,
                        f"FWHM={fwhm:.4f} min",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8,
                        color="tab:red",
                    )
                elif is_valid and apply_apex_gate and not is_kept:
                    ax.text(
                        0.02,
                        0.98,
                        "rejected by apex gate",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8,
                        color="0.35",
                    )

                ax.set_ylim(-0.05, 1.05)
                ax.grid(True, alpha=0.2)
                if row_index == 0:
                    ax.set_title(col_labels[col_index])
                if col_index == 0:
                    ax.set_ylabel(row_labels[row_index] + "\nNorm. signal")
                if row_index == (n_rows - 1):
                    ax.set_xlabel(
                        "Time rel. to apex [min]"
                        if normalize_position
                        else "Time [min]"
                    )

        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return save_path

    def plot_peak_apex_gate(
        self,
        *,
        use_aligned_time: bool = True,
        apply_apex_gate: bool = True,
        apex_gate_n_mad: float = 2.0,
        save_path: str = "nu_bayes_peak_apex_gate.png",
        peak_indices: list[int] | None = None,
        chromatogram_indices: list[int] | None = None,
        data_size: float = 16.0,
        dpi: int = 150,
    ) -> str:
        """Plot per-peak apex-time gating diagnostics."""
        if len(self.peaks) == 0:
            raise ValueError("No peaks are defined.")
        if float(apex_gate_n_mad) <= 0.0:
            raise ValueError("apex_gate_n_mad must be > 0.")

        payload = self.compute_peak_fwhm(
            use_aligned_time=use_aligned_time,
            half_level=0.5,
            apply_apex_gate=apply_apex_gate,
            apex_gate_n_mad=apex_gate_n_mad,
        )
        apex_time_all = np.asarray(payload["apex_time_all"], dtype=float)
        valid_trace = np.asarray(payload["valid_trace"], dtype=bool)
        gate_keep = np.asarray(payload["gate_keep"], dtype=bool)
        gate_center = np.asarray(payload["gate_center"], dtype=float).reshape(-1)
        gate_low = np.asarray(payload["gate_low"], dtype=float).reshape(-1)
        gate_high = np.asarray(payload["gate_high"], dtype=float).reshape(-1)

        n_chrom = int(apex_time_all.shape[0])
        n_peak = int(apex_time_all.shape[1])
        peak_sel = (
            list(range(n_peak))
            if peak_indices is None
            else [int(index) for index in peak_indices]
        )
        chrom_sel = (
            list(range(n_chrom))
            if chromatogram_indices is None
            else [int(index) for index in chromatogram_indices]
        )
        if not peak_sel:
            raise ValueError("peak_indices resolves to an empty selection.")
        if not chrom_sel:
            raise ValueError("chromatogram_indices resolves to an empty selection.")
        if min(peak_sel) < 0 or max(peak_sel) >= n_peak:
            raise ValueError(f"peak_indices out of range [0, {n_peak - 1}].")
        if min(chrom_sel) < 0 or max(chrom_sel) >= n_chrom:
            raise ValueError(f"chromatogram_indices out of range [0, {n_chrom - 1}].")

        n_cols = len(peak_sel)
        figure, axes = plt.subplots(
            1,
            n_cols,
            squeeze=False,
            figsize=(4.0 * n_cols, 3.1),
            constrained_layout=True,
        )
        x_trace = np.asarray(chrom_sel, dtype=int)

        for col_index, peak_index in enumerate(peak_sel):
            ax = axes[0, col_index]
            apex_values = apex_time_all[x_trace, peak_index]
            valid = valid_trace[x_trace, peak_index]
            keep = gate_keep[x_trace, peak_index] & valid
            reject = valid & (~keep)

            if np.any(keep):
                ax.scatter(
                    x_trace[keep] + 1,
                    apex_values[keep],
                    s=data_size,
                    alpha=0.9,
                    color="tab:blue",
                    linewidths=0,
                    label="kept",
                )
            if np.any(reject):
                ax.scatter(
                    x_trace[reject] + 1,
                    apex_values[reject],
                    s=data_size,
                    alpha=0.9,
                    color="tab:red",
                    linewidths=0,
                    label="rejected",
                )

            center = float(gate_center[peak_index])
            low = float(gate_low[peak_index])
            high = float(gate_high[peak_index])
            if np.isfinite(center):
                ax.axhline(center, color="0.25", linestyle="--", linewidth=1.0)
            if apply_apex_gate and np.isfinite(low) and np.isfinite(high):
                ax.axhline(low, color="tab:orange", linestyle=":", linewidth=1.0)
                ax.axhline(high, color="tab:orange", linestyle=":", linewidth=1.0)

            n_valid = int(np.sum(valid))
            n_keep = int(np.sum(keep))
            ax.set_title(f"Peak {peak_index + 1} ({n_keep}/{n_valid} kept)")
            ax.set_xlabel("Trace index")
            if col_index == 0:
                ax.set_ylabel("Apex time [min]")
            ax.grid(True, alpha=0.2)
            if col_index == 0 and (np.any(keep) or np.any(reject)):
                ax.legend(loc="best", fontsize=8, frameon=False)

        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return save_path

    def plot_moment_peak_fits(
        self,
        *,
        use_aligned_time: bool = True,
        start_quantile: float = 0.005,
        end_quantile: float = 0.995,
        tail_window_sigma: float = 2.0,
        save_path: str = "nu_bayes_moment_peak_fits.png",
        column_mode: str = "peak",
        chromatogram_indices: list[int] | None = None,
        peak_indices: list[int] | None = None,
        data_alpha: float = 0.4,
        data_size: float = 8.0,
        line_width: float = 1.5,
        dpi: int = 150,
    ) -> str:
        """Plot prior-guided moment curves against masked data.

        The layout, masking, and selection logic mirror
        :meth:`plot_posterior_peak_fits`.
        """
        if column_mode not in {"chromatogram", "peak"}:
            raise ValueError("column_mode must be 'chromatogram' or 'peak'.")
        if len(self.peaks) == 0:
            raise ValueError("No peaks are defined.")

        guess_payload = self._build_prior_guided_moment_guess(
            use_aligned_time=use_aligned_time,
            start_quantile=start_quantile,
            end_quantile=end_quantile,
            tail_window_sigma=tail_window_sigma,
        )
        time = guess_payload["time"]
        signal_corrected = guess_payload["signal_corrected"]
        peak_masks = guess_payload["peak_masks"]
        mu_guess_matrix = guess_payload["mu_guess"]
        mu_prior_loc = np.asarray(guess_payload["mu_prior_loc"], dtype=float).reshape(
            -1
        )
        mu_prior_scale = np.asarray(
            guess_payload["mu_prior_scale"], dtype=float
        ).reshape(-1)
        sigma_guess_matrix = guess_payload["sigma_guess"]
        alpha_guess_matrix = guess_payload["alpha_guess"]
        area_guess_matrix = guess_payload["area_guess"]
        if peak_masks.shape[0] == 0:
            raise ValueError("No peak masks available for moment plotting.")

        n_chrom = int(time.shape[0])
        n_peak = int(peak_masks.shape[0])
        if len(self.peaks) != n_peak:
            raise ValueError(
                "Peak annotation count does not match mask peak count. "
                f"annotations={len(self.peaks)} masks={n_peak}"
            )

        chrom_sel = (
            list(range(n_chrom))
            if chromatogram_indices is None
            else [int(index) for index in chromatogram_indices]
        )
        peak_sel = (
            list(range(n_peak))
            if peak_indices is None
            else [int(index) for index in peak_indices]
        )
        if not chrom_sel:
            raise ValueError("chromatogram_indices resolves to an empty selection.")
        if not peak_sel:
            raise ValueError("peak_indices resolves to an empty selection.")
        if min(chrom_sel) < 0 or max(chrom_sel) >= n_chrom:
            raise ValueError(f"chromatogram_indices out of range [0, {n_chrom - 1}].")
        if min(peak_sel) < 0 or max(peak_sel) >= n_peak:
            raise ValueError(f"peak_indices out of range [0, {n_peak - 1}].")
        if column_mode == "chromatogram":
            n_rows = len(peak_sel)
            n_cols = len(chrom_sel)
            row_labels = [f"Peak {peak_index + 1}" for peak_index in peak_sel]
            col_labels = [f"Trace {chrom_index + 1}" for chrom_index in chrom_sel]
        else:
            n_rows = len(chrom_sel)
            n_cols = len(peak_sel)
            row_labels = [f"Trace {chrom_index + 1}" for chrom_index in chrom_sel]
            col_labels = [f"Peak {peak_index + 1}" for peak_index in peak_sel]

        figure, axes = plt.subplots(
            n_rows,
            n_cols,
            squeeze=False,
            figsize=(3.6 * n_cols, 2.6 * n_rows),
            constrained_layout=True,
        )

        for row_index in range(n_rows):
            for col_index in range(n_cols):
                ax = axes[row_index, col_index]
                if column_mode == "chromatogram":
                    peak_index = peak_sel[row_index]
                    chrom_index = chrom_sel[col_index]
                else:
                    chrom_index = chrom_sel[row_index]
                    peak_index = peak_sel[col_index]

                mu_center = (
                    float(mu_prior_loc[peak_index])
                    if peak_index < mu_prior_loc.size
                    else np.nan
                )
                mu_std = (
                    float(mu_prior_scale[peak_index])
                    if peak_index < mu_prior_scale.size
                    else np.nan
                )
                if np.isfinite(mu_center):
                    if np.isfinite(mu_std) and mu_std > 0.0:
                        ax.axvspan(
                            mu_center - mu_std,
                            mu_center + mu_std,
                            color="tab:orange",
                            alpha=0.2,
                            linewidth=0.0,
                        )
                    ax.axvline(
                        mu_center,
                        color="tab:orange",
                        linestyle="--",
                        linewidth=max(0.9, 0.8 * line_width),
                    )
                    ax.text(
                        mu_center,
                        0.98,
                        f"{mu_center:.4f}",
                        transform=ax.get_xaxis_transform(),
                        ha="left",
                        va="top",
                        rotation=90,
                        fontsize=7,
                        color="tab:orange",
                    )

                mask = peak_masks[peak_index, chrom_index]
                finite_mask = np.isfinite(time[chrom_index]) & np.isfinite(
                    signal_corrected[chrom_index]
                )
                active = mask & finite_mask
                if int(np.sum(active)) < 3:
                    ax.text(
                        0.5,
                        0.5,
                        "insufficient mask points",
                        ha="center",
                        va="center",
                        fontsize=8,
                        transform=ax.transAxes,
                    )
                    ax.grid(True, alpha=0.2)
                    continue

                x_active = np.asarray(time[chrom_index, active], dtype=float)
                y_active = np.asarray(
                    signal_corrected[chrom_index, active], dtype=float
                )
                order = np.argsort(x_active)
                x_active = x_active[order]
                y_active = y_active[order]

                area_guess = float(area_guess_matrix[chrom_index, peak_index])
                mu_guess = float(mu_guess_matrix[chrom_index, peak_index])
                sigma_guess = float(sigma_guess_matrix[chrom_index, peak_index])
                alpha_guess = float(alpha_guess_matrix[chrom_index, peak_index])
                if not (
                    np.isfinite(area_guess)
                    and np.isfinite(mu_guess)
                    and np.isfinite(sigma_guess)
                    and np.isfinite(alpha_guess)
                    and sigma_guess > 1e-12
                ):
                    ax.text(
                        0.5,
                        0.5,
                        "invalid moment guess",
                        ha="center",
                        va="center",
                        fontsize=8,
                        transform=ax.transAxes,
                    )
                    ax.grid(True, alpha=0.2)
                    continue

                pdf = np.asarray(
                    skew_normal_pdf(
                        jnp.asarray(x_active, dtype=jnp.float32),
                        jnp.asarray([mu_guess], dtype=jnp.float32),
                        jnp.asarray([sigma_guess], dtype=jnp.float32),
                        jnp.asarray([alpha_guess], dtype=jnp.float32),
                    )[0],
                    dtype=float,
                )
                y_guess = area_guess * pdf

                ax.scatter(
                    x_active,
                    y_active,
                    s=data_size,
                    alpha=data_alpha,
                    color="0.35",
                    linewidths=0,
                )
                ax.plot(
                    x_active,
                    y_guess,
                    color="tab:blue",
                    linewidth=line_width,
                )
                ax.grid(True, alpha=0.2)

                if row_index == 0:
                    ax.set_title(col_labels[col_index])
                if col_index == 0:
                    ax.set_ylabel(row_labels[row_index])
                if row_index == (n_rows - 1):
                    ax.set_xlabel("Time [min]")

        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return save_path

    def plot_moment_peak_residuals(
        self,
        *,
        use_aligned_time: bool = True,
        start_quantile: float = 0.005,
        end_quantile: float = 0.995,
        tail_window_sigma: float = 2.0,
        save_path: str = "nu_bayes_moment_peak_residuals.png",
        column_mode: str = "peak",
        chromatogram_indices: list[int] | None = None,
        peak_indices: list[int] | None = None,
        data_alpha: float = 0.4,
        data_size: float = 8.0,
        line_width: float = 1.5,
        dpi: int = 150,
    ) -> str:
        """Plot residuals ``y - prior_guided_moment_guess`` with fit-like layout."""
        if column_mode not in {"chromatogram", "peak"}:
            raise ValueError("column_mode must be 'chromatogram' or 'peak'.")
        if len(self.peaks) == 0:
            raise ValueError("No peaks are defined.")

        guess_payload = self._build_prior_guided_moment_guess(
            use_aligned_time=use_aligned_time,
            start_quantile=start_quantile,
            end_quantile=end_quantile,
            tail_window_sigma=tail_window_sigma,
        )
        time = guess_payload["time"]
        signal_corrected = guess_payload["signal_corrected"]
        peak_masks = guess_payload["peak_masks"]
        mu_guess_matrix = guess_payload["mu_guess"]
        sigma_guess_matrix = guess_payload["sigma_guess"]
        alpha_guess_matrix = guess_payload["alpha_guess"]
        area_guess_matrix = guess_payload["area_guess"]
        if peak_masks.shape[0] == 0:
            raise ValueError("No peak masks available for moment plotting.")

        n_chrom = int(time.shape[0])
        n_peak = int(peak_masks.shape[0])
        if len(self.peaks) != n_peak:
            raise ValueError(
                "Peak annotation count does not match mask peak count. "
                f"annotations={len(self.peaks)} masks={n_peak}"
            )

        chrom_sel = (
            list(range(n_chrom))
            if chromatogram_indices is None
            else [int(index) for index in chromatogram_indices]
        )
        peak_sel = (
            list(range(n_peak))
            if peak_indices is None
            else [int(index) for index in peak_indices]
        )
        if not chrom_sel:
            raise ValueError("chromatogram_indices resolves to an empty selection.")
        if not peak_sel:
            raise ValueError("peak_indices resolves to an empty selection.")
        if min(chrom_sel) < 0 or max(chrom_sel) >= n_chrom:
            raise ValueError(f"chromatogram_indices out of range [0, {n_chrom - 1}].")
        if min(peak_sel) < 0 or max(peak_sel) >= n_peak:
            raise ValueError(f"peak_indices out of range [0, {n_peak - 1}].")

        if column_mode == "chromatogram":
            n_rows = len(peak_sel)
            n_cols = len(chrom_sel)
            row_labels = [f"Peak {peak_index + 1}" for peak_index in peak_sel]
            col_labels = [f"Trace {chrom_index + 1}" for chrom_index in chrom_sel]
        else:
            n_rows = len(chrom_sel)
            n_cols = len(peak_sel)
            row_labels = [f"Trace {chrom_index + 1}" for chrom_index in chrom_sel]
            col_labels = [f"Peak {peak_index + 1}" for peak_index in peak_sel]

        figure, axes = plt.subplots(
            n_rows,
            n_cols,
            squeeze=False,
            figsize=(3.6 * n_cols, 2.6 * n_rows),
            constrained_layout=True,
        )

        for row_index in range(n_rows):
            for col_index in range(n_cols):
                ax = axes[row_index, col_index]
                if column_mode == "chromatogram":
                    peak_index = peak_sel[row_index]
                    chrom_index = chrom_sel[col_index]
                else:
                    chrom_index = chrom_sel[row_index]
                    peak_index = peak_sel[col_index]

                mask = peak_masks[peak_index, chrom_index]
                finite_mask = np.isfinite(time[chrom_index]) & np.isfinite(
                    signal_corrected[chrom_index]
                )
                active = mask & finite_mask
                if int(np.sum(active)) < 3:
                    ax.text(
                        0.5,
                        0.5,
                        "insufficient mask points",
                        ha="center",
                        va="center",
                        fontsize=8,
                        transform=ax.transAxes,
                    )
                    ax.grid(True, alpha=0.2)
                    continue

                x_active = np.asarray(time[chrom_index, active], dtype=float)
                y_active = np.asarray(
                    signal_corrected[chrom_index, active], dtype=float
                )
                order = np.argsort(x_active)
                x_active = x_active[order]
                y_active = y_active[order]

                area_guess = float(area_guess_matrix[chrom_index, peak_index])
                mu_guess = float(mu_guess_matrix[chrom_index, peak_index])
                sigma_guess = float(sigma_guess_matrix[chrom_index, peak_index])
                alpha_guess = float(alpha_guess_matrix[chrom_index, peak_index])
                if not (
                    np.isfinite(area_guess)
                    and np.isfinite(mu_guess)
                    and np.isfinite(sigma_guess)
                    and np.isfinite(alpha_guess)
                    and sigma_guess > 1e-12
                ):
                    ax.text(
                        0.5,
                        0.5,
                        "invalid moment guess",
                        ha="center",
                        va="center",
                        fontsize=8,
                        transform=ax.transAxes,
                    )
                    ax.grid(True, alpha=0.2)
                    continue

                pdf = np.asarray(
                    skew_normal_pdf(
                        jnp.asarray(x_active, dtype=jnp.float32),
                        jnp.asarray([mu_guess], dtype=jnp.float32),
                        jnp.asarray([sigma_guess], dtype=jnp.float32),
                        jnp.asarray([alpha_guess], dtype=jnp.float32),
                    )[0],
                    dtype=float,
                )
                y_guess = area_guess * pdf
                residual = y_active - y_guess

                ax.scatter(
                    x_active,
                    residual,
                    s=data_size,
                    alpha=data_alpha,
                    color="0.35",
                    linewidths=0,
                )
                ax.plot(
                    x_active,
                    residual,
                    color="tab:blue",
                    linewidth=line_width,
                )
                ax.axhline(0.0, color="0.2", linewidth=0.8, alpha=0.7)
                ax.grid(True, alpha=0.2)

                if row_index == 0:
                    ax.set_title(col_labels[col_index])
                if col_index == 0:
                    ax.set_ylabel(f"{row_labels[row_index]}\nResidual")
                if row_index == (n_rows - 1):
                    ax.set_xlabel("Time [min]")

        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return save_path

    def plot_moment_peak_residuals_column_sum(
        self,
        *,
        use_aligned_time: bool = True,
        start_quantile: float = 0.005,
        end_quantile: float = 0.995,
        tail_window_sigma: float = 2.0,
        save_path: str = "nu_bayes_moment_peak_residuals_column_mean.png",
        column_mode: str = "peak",
        chromatogram_indices: list[int] | None = None,
        peak_indices: list[int] | None = None,
        aggregation: str = "mean",
        min_count_per_timepoint: int = 1,
        line_width: float = 1.8,
        dpi: int = 150,
    ) -> str:
        """Plot per-column residual trend by direct index-wise aggregation.

        Residuals are aggregated at matching time-array indices (no interpolation).
        """
        if column_mode not in {"chromatogram", "peak"}:
            raise ValueError("column_mode must be 'chromatogram' or 'peak'.")
        if aggregation not in {"mean", "sum"}:
            raise ValueError("aggregation must be 'mean' or 'sum'.")
        if len(self.peaks) == 0:
            raise ValueError("No peaks are defined.")
        min_count_per_timepoint = max(int(min_count_per_timepoint), 1)

        guess_payload = self._build_prior_guided_moment_guess(
            use_aligned_time=use_aligned_time,
            start_quantile=start_quantile,
            end_quantile=end_quantile,
            tail_window_sigma=tail_window_sigma,
        )
        time = guess_payload["time"]
        signal_corrected = guess_payload["signal_corrected"]
        peak_masks = guess_payload["peak_masks"]
        mu_guess_matrix = guess_payload["mu_guess"]
        sigma_guess_matrix = guess_payload["sigma_guess"]
        alpha_guess_matrix = guess_payload["alpha_guess"]
        area_guess_matrix = guess_payload["area_guess"]
        if peak_masks.shape[0] == 0:
            raise ValueError("No peak masks available for moment plotting.")

        n_chrom = int(time.shape[0])
        n_peak = int(peak_masks.shape[0])
        chrom_sel = (
            list(range(n_chrom))
            if chromatogram_indices is None
            else [int(index) for index in chromatogram_indices]
        )
        peak_sel = (
            list(range(n_peak))
            if peak_indices is None
            else [int(index) for index in peak_indices]
        )
        if not chrom_sel:
            raise ValueError("chromatogram_indices resolves to an empty selection.")
        if not peak_sel:
            raise ValueError("peak_indices resolves to an empty selection.")
        if min(chrom_sel) < 0 or max(chrom_sel) >= n_chrom:
            raise ValueError(f"chromatogram_indices out of range [0, {n_chrom - 1}].")
        if min(peak_sel) < 0 or max(peak_sel) >= n_peak:
            raise ValueError(f"peak_indices out of range [0, {n_peak - 1}].")

        if column_mode == "chromatogram":
            n_rows = len(peak_sel)
            n_cols = len(chrom_sel)
            col_labels = [f"Trace {chrom_index + 1}" for chrom_index in chrom_sel]
        else:
            n_rows = len(chrom_sel)
            n_cols = len(peak_sel)
            col_labels = [f"Peak {peak_index + 1}" for peak_index in peak_sel]

        figure, axes = plt.subplots(
            1,
            n_cols,
            squeeze=False,
            figsize=(4.0 * n_cols, 3.0),
            constrained_layout=True,
        )

        for col_index in range(n_cols):
            ax = axes[0, col_index]
            n_timepoint = int(time.shape[1])
            residual_sum = np.zeros((n_timepoint,), dtype=float)
            x_sum = np.zeros((n_timepoint,), dtype=float)
            count = np.zeros((n_timepoint,), dtype=int)

            for row_index in range(n_rows):
                if column_mode == "chromatogram":
                    peak_index = peak_sel[row_index]
                    chrom_index = chrom_sel[col_index]
                else:
                    chrom_index = chrom_sel[row_index]
                    peak_index = peak_sel[col_index]

                mask = peak_masks[peak_index, chrom_index]
                finite_mask = np.isfinite(time[chrom_index]) & np.isfinite(
                    signal_corrected[chrom_index]
                )
                active = mask & finite_mask
                if int(np.sum(active)) < 3:
                    continue

                active_indices = np.flatnonzero(active)
                x_active = np.asarray(time[chrom_index, active_indices], dtype=float)
                y_active = np.asarray(
                    signal_corrected[chrom_index, active_indices], dtype=float
                )

                area_guess = float(area_guess_matrix[chrom_index, peak_index])
                mu_guess = float(mu_guess_matrix[chrom_index, peak_index])
                sigma_guess = float(sigma_guess_matrix[chrom_index, peak_index])
                alpha_guess = float(alpha_guess_matrix[chrom_index, peak_index])
                if not (
                    np.isfinite(area_guess)
                    and np.isfinite(mu_guess)
                    and np.isfinite(sigma_guess)
                    and np.isfinite(alpha_guess)
                    and sigma_guess > 1e-12
                ):
                    continue

                pdf = np.asarray(
                    skew_normal_pdf(
                        jnp.asarray(x_active, dtype=jnp.float32),
                        jnp.asarray([mu_guess], dtype=jnp.float32),
                        jnp.asarray([sigma_guess], dtype=jnp.float32),
                        jnp.asarray([alpha_guess], dtype=jnp.float32),
                    )[0],
                    dtype=float,
                )
                residual = y_active - area_guess * pdf
                finite_residual = np.isfinite(x_active) & np.isfinite(residual)
                if int(np.sum(finite_residual)) < 3:
                    continue
                idx = active_indices[finite_residual]
                residual_sum[idx] += residual[finite_residual]
                x_sum[idx] += x_active[finite_residual]
                count[idx] += 1

            has_data = count >= min_count_per_timepoint
            if not np.any(has_data):
                ax.text(
                    0.5,
                    0.5,
                    "insufficient residual points",
                    ha="center",
                    va="center",
                    fontsize=8,
                    transform=ax.transAxes,
                )
                ax.axhline(0.0, color="0.2", linewidth=0.8, alpha=0.7)
                ax.set_title(col_labels[col_index])
                ax.grid(True, alpha=0.2)
                continue

            centers = x_sum[has_data] / np.maximum(count[has_data], 1)
            if aggregation == "mean":
                values = residual_sum[has_data] / np.maximum(count[has_data], 1)
            else:
                values = residual_sum[has_data]
            order = np.argsort(centers)
            centers = centers[order]
            values = values[order]

            ax.plot(
                centers,
                values,
                color="tab:blue",
                linewidth=line_width,
            )
            ax.axhline(0.0, color="0.2", linewidth=0.8, alpha=0.7)
            ax.set_title(col_labels[col_index])
            ax.set_xlabel("Time [min]")
            if col_index == 0:
                ax.set_ylabel(
                    "Residual mean" if aggregation == "mean" else "Residual sum"
                )
            ax.grid(True, alpha=0.2)

        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return save_path

    def plot_posterior_peak_fits(
        self,
        *,
        use_aligned_time: bool = True,
        save_path: str = "nu_bayes_posterior_peak_fits.png",
        column_mode: str = "peak",
        chromatogram_indices: list[int] | None = None,
        peak_indices: list[int] | None = None,
        data_alpha: float = 0.4,
        data_size: float = 8.0,
        line_width: float = 1.5,
        hdi_alpha: float = 0.22,
        dpi: int = 150,
    ) -> str:
        """Plot mask-restricted posterior fit with median and 95% interval.

        Uses raw observed signal and includes the inferred baseline intercept.

        Args:
            use_aligned_time: If True, use aligned time axis.
            save_path: Output image file path.
            column_mode: Layout mode:
                ``"peak"`` -> rows are chromatograms, columns are peaks.
                If more than one peak is selected, an extra ``"All peaks"``
                column is appended per row spanning the contiguous x-range
                from the leftmost to rightmost selected peak window.
                ``"chromatogram"`` -> rows are peaks, columns are chromatograms.
            chromatogram_indices: Optional subset of chromatograms to show.
            peak_indices: Optional subset of peaks to show.
            data_alpha: Scatter alpha for observed data.
            data_size: Scatter marker size for observed data.
            line_width: Width of posterior median line.
            hdi_alpha: Fill alpha for the 95% interval band.
            dpi: Figure save DPI.
        """
        if self.samples is None:
            raise RuntimeError("Call fit() before plotting posterior peak fits.")
        if self.model_inputs is None:
            raise RuntimeError(
                "Model metadata is unavailable. Run fit() before plotting."
            )
        if column_mode not in {"chromatogram", "peak"}:
            raise ValueError("column_mode must be 'chromatogram' or 'peak'.")

        time = np.asarray(
            self._time_axis(use_aligned_time=use_aligned_time), dtype=float
        )
        signal_observed = np.asarray(self.signal, dtype=float)
        peak_masks = np.asarray(
            self.get_peak_masks(use_aligned_time=use_aligned_time), dtype=bool
        )
        if peak_masks.shape[0] == 0:
            raise ValueError("No peak masks available for posterior plotting.")

        n_chrom = int(time.shape[0])
        n_peak = int(peak_masks.shape[0])
        chrom_sel = (
            list(range(n_chrom))
            if chromatogram_indices is None
            else [int(index) for index in chromatogram_indices]
        )
        peak_sel = (
            list(range(n_peak))
            if peak_indices is None
            else [int(index) for index in peak_indices]
        )
        if not chrom_sel:
            raise ValueError("chromatogram_indices resolves to an empty selection.")
        if not peak_sel:
            raise ValueError("peak_indices resolves to an empty selection.")
        if min(chrom_sel) < 0 or max(chrom_sel) >= n_chrom:
            raise ValueError(f"chromatogram_indices out of range [0, {n_chrom - 1}].")
        if min(peak_sel) < 0 or max(peak_sel) >= n_peak:
            raise ValueError(f"peak_indices out of range [0, {n_peak - 1}].")
        peak_sel_array = np.asarray(peak_sel, dtype=int)
        include_full_window_column = column_mode == "peak" and len(peak_sel) > 1

        component_to_logical = np.asarray(
            self.model_inputs["component_to_logical_index"], dtype=int
        )
        A = np.asarray(self.samples["A"], dtype=float)
        mu = np.asarray(self.samples["mu"], dtype=float)
        sigma = np.asarray(self.samples["sigma"], dtype=float)
        alpha = self._posterior_alpha_component_draws(
            n_draw=int(A.shape[0]),
            n_chrom=int(A.shape[1]),
            n_component=int(A.shape[2]),
        )
        if (
            A.ndim != 3
            or mu.shape != A.shape
            or sigma.shape != A.shape
            or alpha.shape != A.shape
        ):
            raise ValueError(
                "Posterior sample arrays must have shape [draw, chromatogram, component]."
            )
        if A.shape[1] != n_chrom:
            raise ValueError(
                "Posterior sample chromatogram axis does not match current data shape."
            )
        baseline_draws = self._posterior_baseline_intercept_draws(
            n_draw=int(A.shape[0]),
            n_chrom=n_chrom,
        )

        if column_mode == "chromatogram":
            n_rows = len(peak_sel)
            n_cols = len(chrom_sel)
            row_labels = [f"Peak {peak_index + 1}" for peak_index in peak_sel]
            col_labels = [f"Trace {chrom_index + 1}" for chrom_index in chrom_sel]
        else:
            n_rows = len(chrom_sel)
            n_cols = len(peak_sel) + (1 if include_full_window_column else 0)
            row_labels = [f"Trace {chrom_index + 1}" for chrom_index in chrom_sel]
            col_labels = [f"Peak {peak_index + 1}" for peak_index in peak_sel]
            if include_full_window_column:
                col_labels.append("All peaks")

        figure, axes = plt.subplots(
            n_rows,
            n_cols,
            squeeze=False,
            figsize=(3.6 * n_cols, 2.6 * n_rows),
            constrained_layout=True,
        )

        for row_index in range(n_rows):
            for col_index in range(n_cols):
                ax = axes[row_index, col_index]
                if column_mode == "chromatogram":
                    peak_index = peak_sel[row_index]
                    chrom_index = chrom_sel[col_index]
                    finite_mask = np.isfinite(time[chrom_index]) & np.isfinite(
                        signal_observed[chrom_index]
                    )
                    active = peak_masks[peak_index, chrom_index] & finite_mask
                    component_index = np.where(component_to_logical == peak_index)[0]
                else:
                    chrom_index = chrom_sel[row_index]
                    finite_mask = np.isfinite(time[chrom_index]) & np.isfinite(
                        signal_observed[chrom_index]
                    )
                    if include_full_window_column and col_index == len(peak_sel):
                        union_mask = np.any(
                            peak_masks[peak_sel_array, chrom_index], axis=0
                        )
                        window_points = union_mask & finite_mask
                        if int(np.sum(window_points)) >= 2:
                            x_window = np.asarray(
                                time[chrom_index, window_points], dtype=float
                            )
                            x_low = float(np.nanmin(x_window))
                            x_high = float(np.nanmax(x_window))
                            active = (
                                finite_mask
                                & (time[chrom_index] >= x_low)
                                & (time[chrom_index] <= x_high)
                            )
                        else:
                            active = window_points
                        component_index = np.where(
                            np.isin(component_to_logical, peak_sel_array)
                        )[0]
                    else:
                        peak_index = peak_sel[col_index]
                        active = peak_masks[peak_index, chrom_index] & finite_mask
                        component_index = np.where(component_to_logical == peak_index)[
                            0
                        ]
                if int(np.sum(active)) < 3:
                    ax.text(
                        0.5,
                        0.5,
                        "insufficient mask points",
                        ha="center",
                        va="center",
                        fontsize=8,
                        transform=ax.transAxes,
                    )
                    ax.grid(True, alpha=0.2)
                    continue

                x_active = np.asarray(time[chrom_index, active], dtype=float)
                y_active = np.asarray(signal_observed[chrom_index, active], dtype=float)
                order = np.argsort(x_active)
                x_active = x_active[order]
                y_active = y_active[order]

                if component_index.size == 0:
                    ax.text(
                        0.5,
                        0.5,
                        "no mapped component",
                        ha="center",
                        va="center",
                        fontsize=8,
                        transform=ax.transAxes,
                    )
                    ax.grid(True, alpha=0.2)
                    continue

                A_draw = A[:, chrom_index, :][:, component_index]
                mu_draw = mu[:, chrom_index, :][:, component_index]
                sigma_draw = sigma[:, chrom_index, :][:, component_index]
                alpha_draw = alpha[:, chrom_index, :][:, component_index]

                pdf = np.asarray(
                    skew_normal_pdf(
                        jnp.asarray(x_active, dtype=jnp.float32),
                        jnp.asarray(mu_draw, dtype=jnp.float32),
                        jnp.asarray(sigma_draw, dtype=jnp.float32),
                        jnp.asarray(alpha_draw, dtype=jnp.float32),
                    ),
                    dtype=float,
                )
                peak_draws = np.sum(A_draw[:, :, None] * pdf, axis=1)
                baseline_active = baseline_draws[:, chrom_index][:, None]
                total_draws = peak_draws + baseline_active
                y_median = np.nanmedian(total_draws, axis=0)
                y_low = np.nanquantile(total_draws, 0.025, axis=0)
                y_high = np.nanquantile(total_draws, 0.975, axis=0)
                baseline_median = float(np.nanmedian(baseline_active[:, 0]))
                baseline_low = float(np.nanquantile(baseline_active[:, 0], 0.025))
                baseline_high = float(np.nanquantile(baseline_active[:, 0], 0.975))
                baseline_line = np.full_like(x_active, baseline_median, dtype=float)
                baseline_low_line = np.full_like(x_active, baseline_low, dtype=float)
                baseline_high_line = np.full_like(x_active, baseline_high, dtype=float)

                ax.scatter(
                    x_active,
                    y_active,
                    s=data_size,
                    alpha=data_alpha,
                    color="0.35",
                    linewidths=0,
                )
                ax.plot(
                    x_active,
                    y_median,
                    color="tab:blue",
                    linewidth=line_width,
                )
                ax.fill_between(
                    x_active,
                    y_low,
                    y_high,
                    color="tab:blue",
                    alpha=hdi_alpha,
                )
                ax.plot(
                    x_active,
                    baseline_line,
                    color="tab:orange",
                    linestyle="--",
                    linewidth=max(1.0, 0.9 * line_width),
                )
                ax.fill_between(
                    x_active,
                    baseline_low_line,
                    baseline_high_line,
                    color="tab:orange",
                    alpha=min(0.6 * hdi_alpha, 0.18),
                )
                ax.grid(True, alpha=0.2)

                if row_index == 0:
                    ax.set_title(col_labels[col_index])
                if col_index == 0:
                    ax.set_ylabel(row_labels[row_index])
                if row_index == (n_rows - 1):
                    ax.set_xlabel("Time [min]")

        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return save_path

    def plot_posterior_peak_residuals(
        self,
        *,
        use_aligned_time: bool = True,
        save_path: str = "nu_bayes_posterior_peak_residuals.png",
        column_mode: str = "peak",
        chromatogram_indices: list[int] | None = None,
        peak_indices: list[int] | None = None,
        data_alpha: float = 0.4,
        data_size: float = 8.0,
        line_width: float = 1.5,
        hdi_alpha: float = 0.22,
        dpi: int = 150,
    ) -> str:
        """Plot mask-restricted residuals ``y - median_posterior_fit``.

        The layout, masking, and selection logic match
        :meth:`plot_posterior_peak_fits` and includes baseline in the fit.
        """
        if self.samples is None:
            raise RuntimeError("Call fit() before plotting posterior peak residuals.")
        if self.model_inputs is None:
            raise RuntimeError(
                "Model metadata is unavailable. Run fit() before plotting."
            )
        if column_mode not in {"chromatogram", "peak"}:
            raise ValueError("column_mode must be 'chromatogram' or 'peak'.")

        time = np.asarray(
            self._time_axis(use_aligned_time=use_aligned_time), dtype=float
        )
        signal_observed = np.asarray(self.signal, dtype=float)
        peak_masks = np.asarray(
            self.get_peak_masks(use_aligned_time=use_aligned_time), dtype=bool
        )
        if peak_masks.shape[0] == 0:
            raise ValueError("No peak masks available for posterior plotting.")

        n_chrom = int(time.shape[0])
        n_peak = int(peak_masks.shape[0])
        chrom_sel = (
            list(range(n_chrom))
            if chromatogram_indices is None
            else [int(index) for index in chromatogram_indices]
        )
        peak_sel = (
            list(range(n_peak))
            if peak_indices is None
            else [int(index) for index in peak_indices]
        )
        if not chrom_sel:
            raise ValueError("chromatogram_indices resolves to an empty selection.")
        if not peak_sel:
            raise ValueError("peak_indices resolves to an empty selection.")
        if min(chrom_sel) < 0 or max(chrom_sel) >= n_chrom:
            raise ValueError(f"chromatogram_indices out of range [0, {n_chrom - 1}].")
        if min(peak_sel) < 0 or max(peak_sel) >= n_peak:
            raise ValueError(f"peak_indices out of range [0, {n_peak - 1}].")

        component_to_logical = np.asarray(
            self.model_inputs["component_to_logical_index"], dtype=int
        )
        A = np.asarray(self.samples["A"], dtype=float)
        mu = np.asarray(self.samples["mu"], dtype=float)
        sigma = np.asarray(self.samples["sigma"], dtype=float)
        alpha = self._posterior_alpha_component_draws(
            n_draw=int(A.shape[0]),
            n_chrom=int(A.shape[1]),
            n_component=int(A.shape[2]),
        )
        if (
            A.ndim != 3
            or mu.shape != A.shape
            or sigma.shape != A.shape
            or alpha.shape != A.shape
        ):
            raise ValueError(
                "Posterior sample arrays must have shape [draw, chromatogram, component]."
            )
        if A.shape[1] != n_chrom:
            raise ValueError(
                "Posterior sample chromatogram axis does not match current data shape."
            )
        baseline_draws = self._posterior_baseline_intercept_draws(
            n_draw=int(A.shape[0]),
            n_chrom=n_chrom,
        )

        if column_mode == "chromatogram":
            n_rows = len(peak_sel)
            n_cols = len(chrom_sel)
            row_labels = [f"Peak {peak_index + 1}" for peak_index in peak_sel]
            col_labels = [f"Trace {chrom_index + 1}" for chrom_index in chrom_sel]
        else:
            n_rows = len(chrom_sel)
            n_cols = len(peak_sel)
            row_labels = [f"Trace {chrom_index + 1}" for chrom_index in chrom_sel]
            col_labels = [f"Peak {peak_index + 1}" for peak_index in peak_sel]

        figure, axes = plt.subplots(
            n_rows,
            n_cols,
            squeeze=False,
            figsize=(3.6 * n_cols, 2.6 * n_rows),
            constrained_layout=True,
        )

        for row_index in range(n_rows):
            for col_index in range(n_cols):
                ax = axes[row_index, col_index]
                if column_mode == "chromatogram":
                    peak_index = peak_sel[row_index]
                    chrom_index = chrom_sel[col_index]
                else:
                    chrom_index = chrom_sel[row_index]
                    peak_index = peak_sel[col_index]

                mask = peak_masks[peak_index, chrom_index]
                finite_mask = np.isfinite(time[chrom_index]) & np.isfinite(
                    signal_observed[chrom_index]
                )
                active = mask & finite_mask
                if int(np.sum(active)) < 3:
                    ax.text(
                        0.5,
                        0.5,
                        "insufficient mask points",
                        ha="center",
                        va="center",
                        fontsize=8,
                        transform=ax.transAxes,
                    )
                    ax.grid(True, alpha=0.2)
                    continue

                x_active = np.asarray(time[chrom_index, active], dtype=float)
                y_active = np.asarray(signal_observed[chrom_index, active], dtype=float)
                order = np.argsort(x_active)
                x_active = x_active[order]
                y_active = y_active[order]

                component_index = np.where(component_to_logical == peak_index)[0]
                if component_index.size == 0:
                    ax.text(
                        0.5,
                        0.5,
                        "no mapped component",
                        ha="center",
                        va="center",
                        fontsize=8,
                        transform=ax.transAxes,
                    )
                    ax.grid(True, alpha=0.2)
                    continue

                A_draw = A[:, chrom_index, :][:, component_index]
                mu_draw = mu[:, chrom_index, :][:, component_index]
                sigma_draw = sigma[:, chrom_index, :][:, component_index]
                alpha_draw = alpha[:, chrom_index, :][:, component_index]

                pdf = np.asarray(
                    skew_normal_pdf(
                        jnp.asarray(x_active, dtype=jnp.float32),
                        jnp.asarray(mu_draw, dtype=jnp.float32),
                        jnp.asarray(sigma_draw, dtype=jnp.float32),
                        jnp.asarray(alpha_draw, dtype=jnp.float32),
                    ),
                    dtype=float,
                )
                peak_draws = np.sum(A_draw[:, :, None] * pdf, axis=1)
                baseline_active = baseline_draws[:, chrom_index][:, None]
                total_draws = peak_draws + baseline_active

                residual_draws = y_active[None, :] - total_draws
                residual_median = np.nanmedian(residual_draws, axis=0)
                residual_low = np.nanquantile(residual_draws, 0.025, axis=0)
                residual_high = np.nanquantile(residual_draws, 0.975, axis=0)

                ax.scatter(
                    x_active,
                    residual_median,
                    s=data_size,
                    alpha=data_alpha,
                    color="0.35",
                    linewidths=0,
                )
                ax.plot(
                    x_active,
                    residual_median,
                    color="tab:blue",
                    linewidth=line_width,
                )
                ax.fill_between(
                    x_active,
                    residual_low,
                    residual_high,
                    color="tab:blue",
                    alpha=hdi_alpha,
                )
                ax.axhline(0.0, color="0.2", linewidth=0.8, alpha=0.7)
                ax.grid(True, alpha=0.2)

                if row_index == 0:
                    ax.set_title(col_labels[col_index])
                if col_index == 0:
                    ax.set_ylabel(f"{row_labels[row_index]}\nResidual")
                if row_index == (n_rows - 1):
                    ax.set_xlabel("Time [min]")

        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return save_path

    def align_peaks(
        self,
        *,
        use_peak_mask: bool = True,
        peak_indices: list[int] | None = None,
        lr: float = 1e-2,
        n_steps: int = 500,
        center_weight: float = 1e3,
        max_shift_samples: float | None = None,
        enforce_zero_mean: bool = True,
        return_history: bool = False,
        verbose: bool = True,
    ) -> ShiftAlignmentResult:
        """Align chromatograms with one shift per chromatogram.

        Args:
            use_peak_mask: If True, alignment only uses points in selected peak mask.
            peak_indices: Peak indices to include when `use_peak_mask=True`.
                If None, all peaks are used.
            lr: Optimizer learning rate.
            n_steps: Number of optimizer steps.
            center_weight: Penalty for non-zero mean shift.
            max_shift_samples: Optional clipping bound for shift values.
            enforce_zero_mean: Recenter shifts to zero mean after each optimizer step.
            return_history: If True, include optimization loss history.
            verbose: If True, print alignment diagnostics.
        """
        peak_masks = self.get_peak_masks(use_aligned_time=False)
        alignment_mask: jnp.ndarray | None = None
        if use_peak_mask:
            if peak_masks.shape[0] == 0:
                raise ValueError(
                    "use_peak_mask=True but no peaks are available in `peaks`."
                )
            if peak_indices is None:
                selected_peak_masks = peak_masks
            else:
                idx = jnp.asarray(peak_indices, dtype=jnp.int32)
                if idx.ndim != 1 or idx.size == 0:
                    raise ValueError("peak_indices must be a non-empty 1D list.")
                if int(jnp.min(idx)) < 0 or int(jnp.max(idx)) >= peak_masks.shape[0]:
                    raise ValueError(
                        f"peak_indices out of range [0, {peak_masks.shape[0] - 1}]"
                    )
                selected_peak_masks = peak_masks[idx]
            alignment_mask = jnp.any(selected_peak_masks, axis=0)

        result = align_chromatogram_shifts(
            signal=self.signal,
            mask=alignment_mask,
            lr=lr,
            n_steps=n_steps,
            center_weight=center_weight,
            max_shift_samples=max_shift_samples,
            enforce_zero_mean=enforce_zero_mean,
            return_history=return_history,
        )

        self.shift_result = result
        self.shift_samples = jnp.asarray(result.shifts_samples, dtype=jnp.float32)
        self.alignment_mask = alignment_mask

        time_step = self._median_time_step_per_chromatogram()
        self.shift_time = self.shift_samples * time_step
        self.aligned_time = self.time + self.shift_time[:, None]
        self._baseline_estimate_cache.clear()
        self.model_inputs = None
        self.samples = None
        self.mcmc = None
        self.idata = None

        if verbose:
            print("\n[align] Single-stage chromatogram alignment")
            print(f"  chromatograms: {self.signal.shape[0]}")
            print(f"  points: {self.signal.shape[1]}")
            print(
                "  mask: "
                + (
                    f"peak-based ({int(jnp.sum(alignment_mask))} points)"
                    if alignment_mask is not None
                    else "all points"
                )
            )
            print(f"  loss initial: {result.loss_initial:.6e}")
            print(f"  loss final:   {result.loss_final:.6e}")
            print("  shifts:")
            for idx in range(int(self.shift_samples.shape[0])):
                print(
                    f"    chrom {idx:2d}: "
                    f"shift_samples={float(self.shift_samples[idx]):+.5f}, "
                    f"shift_time={float(self.shift_time[idx]):+.6f}"
                )

        return result

    def compute_peak_moment_metrics(
        self,
        *,
        use_aligned_time: bool = True,
        start_quantile: float = 0.005,
        end_quantile: float = 0.995,
        tail_window_sigma: float = 2.0,
    ) -> dict[str, Any]:
        """Compute per-peak moment diagnostics from `peak_masks` using fitted baseline."""
        time = self._time_axis(use_aligned_time=use_aligned_time)
        peak_masks = self.get_peak_masks(use_aligned_time=use_aligned_time)
        baseline_estimates = self.get_baseline_estimates(
            use_aligned_time=use_aligned_time
        )
        baseline_slopes = np.asarray(
            [float(estimate.slope) for estimate in baseline_estimates], dtype=float
        )
        baseline_intercepts = np.asarray(
            [float(estimate.intercept) for estimate in baseline_estimates],
            dtype=float,
        )

        metrics_by_peak = compute_peak_moment_metrics_from_peak_masks(
            x_matrix=np.asarray(time, dtype=float),
            y_matrix=np.asarray(self.signal, dtype=float),
            peak_masks=np.asarray(peak_masks, dtype=bool),
            baseline_slopes=baseline_slopes,
            baseline_intercepts=baseline_intercepts,
            start_quantile=float(start_quantile),
            end_quantile=float(end_quantile),
            tail_window_sigma=float(tail_window_sigma),
        )
        prior_hints = estimate_skew_normal_prior_hints(metrics_by_peak)
        arrays_by_peak = [
            metrics_list_to_arrays(metrics) for metrics in metrics_by_peak
        ]
        summaries_by_peak = [summarize_metrics(metrics) for metrics in arrays_by_peak]

        self.peak_moment_metrics = metrics_by_peak
        self.peak_prior_hints = prior_hints

        return {
            "metrics_by_peak": metrics_by_peak,
            "arrays_by_peak": arrays_by_peak,
            "summaries_by_peak": summaries_by_peak,
            "prior_hints": prior_hints,
        }

    def plot(
        self,
        baseline: bool = False,
        use_aligned_time: bool = False,
        save_path: str | None = None,
        single_axis: bool = False,
    ) -> None:
        time = self._time_axis(use_aligned_time=use_aligned_time)
        peak_masks = np.asarray(
            self.get_peak_masks(use_aligned_time=use_aligned_time), dtype=bool
        )
        baseline_mask = np.asarray(
            self.get_baseline_mask(use_aligned_time=use_aligned_time), dtype=bool
        )
        baseline_estimates = self.get_baseline_estimates(
            use_aligned_time=use_aligned_time
        )
        n_chromatograms = self.time.shape[0]
        cmap = plt.get_cmap("viridis")
        colors = [cmap(i / max(n_chromatograms - 1, 1)) for i in range(n_chromatograms)]
        if single_axis:
            _, single_ax = plt.subplots(1, 1, figsize=(10, 4.5))
            axes = [single_ax]
            target_axes = [single_ax for _ in range(n_chromatograms)]
        else:
            _, axes = plt.subplots(
                n_chromatograms, 1, sharex=True, figsize=(10, 2.5 * n_chromatograms)
            )
            if n_chromatograms == 1:
                axes = [axes]
            target_axes = list(axes)

        for i, ax in enumerate(target_axes):
            time_i = np.asarray(time[i], dtype=np.float32)
            signal_i = np.asarray(self.signal[i], dtype=np.float32)
            baseline_mask_i = baseline_mask[i]
            if peak_masks.shape[0] > 0:
                peak_mask_i = np.any(peak_masks[:, i, :], axis=0)
            else:
                peak_mask_i = np.zeros_like(baseline_mask_i, dtype=bool)

            line_mask = baseline_mask_i | peak_mask_i
            finite_mask = np.isfinite(time_i) & np.isfinite(signal_i)
            line_mask &= finite_mask
            outside_mask = (~(baseline_mask_i | peak_mask_i)) & finite_mask

            signal_line = np.where(line_mask, signal_i, np.nan)
            ax.plot(
                time_i,
                signal_line,
                color=colors[i],
                linewidth=1,
                label=f"Chromatogram {i + 1}" if single_axis else None,
            )
            if np.any(outside_mask):
                ax.scatter(
                    time_i[outside_mask],
                    signal_i[outside_mask],
                    color=colors[i],
                    s=4,
                    alpha=0.5,
                    linewidths=0,
                )
            if not single_axis:
                ax.set_ylabel(f"Chromatogram {i + 1}")
            if baseline:
                y_baseline = (
                    baseline_estimates[i].slope * time_i
                    + baseline_estimates[i].intercept
                )
                ax.plot(
                    time_i,
                    y_baseline,
                    color=colors[i],
                    linewidth=1,
                    linestyle=":",
                )
        if single_axis:
            axes[0].set_ylabel("Signal")
            axes[0].set_xlabel("Time [min]")
            if n_chromatograms <= 12:
                axes[0].legend(loc="best", fontsize=8, frameon=False)
        else:
            axes[-1].set_xlabel("Time [min]")

        if baseline:
            for axis in axes:
                axis.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))
                axis.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
                axis.tick_params(axis="x", which="minor", bottom=True, length=3)
                axis.tick_params(axis="y", which="minor", left=True, length=3)
                axis.grid(True, which="both", alpha=0.2)

        plt.tight_layout()
        out_name = save_path
        if out_name is None:
            out_name = (
                "chromatograms_aligned.png" if use_aligned_time else "chromatograms.png"
            )
        plt.savefig(out_name)
        plt.close()

    def plot_baseline_anchor_fit(
        self,
        *,
        use_aligned_time: bool = True,
        surrounding_points: int = 6,
        save_path: str = "baseline_peak_edge_fit.png",
        dpi: int = 150,
    ) -> str:
        """Plot baseline-anchor diagnostics per chromatogram.

        Shows:
            - surrounding points near anchor locations as scatter
            - anchor points used for baseline estimation as scatter
            - fitted baseline line
        """
        surrounding_points = max(int(surrounding_points), 0)
        time = np.asarray(
            self._time_axis(use_aligned_time=use_aligned_time), dtype=float
        )
        signal = np.asarray(self.signal, dtype=float)
        baseline_estimates = self.get_baseline_estimates(
            use_aligned_time=use_aligned_time
        )
        anchor_payload = self._collect_peak_edge_baseline_anchors(
            use_aligned_time=use_aligned_time
        )
        x_anchor = np.asarray(anchor_payload["x_anchor"], dtype=float)
        y_anchor = np.asarray(anchor_payload["y_anchor"], dtype=float)
        anchor_indices = np.asarray(anchor_payload["anchor_indices"], dtype=int)

        n_trace = int(signal.shape[0])
        figure, axes = plt.subplots(
            n_trace,
            1,
            sharex=True,
            squeeze=False,
            figsize=(10, 2.6 * n_trace),
            constrained_layout=True,
        )
        axes_1d = axes[:, 0]

        for trace_index, ax in enumerate(axes_1d):
            trace_time = time[trace_index]
            trace_signal = signal[trace_index]
            finite = np.isfinite(trace_time) & np.isfinite(trace_signal)
            if not np.any(finite):
                ax.text(
                    0.5,
                    0.5,
                    "no finite points",
                    ha="center",
                    va="center",
                    fontsize=8,
                    transform=ax.transAxes,
                )
                ax.grid(True, alpha=0.2)
                continue

            valid_anchor_idx = anchor_indices[trace_index]
            valid_anchor_idx = valid_anchor_idx[valid_anchor_idx >= 0]

            surrounding_idx: np.ndarray
            if valid_anchor_idx.size == 0:
                surrounding_idx = np.asarray([], dtype=int)
            else:
                neighborhoods: list[np.ndarray] = []
                for idx in valid_anchor_idx:
                    start = max(0, int(idx) - surrounding_points)
                    stop = min(int(trace_time.size), int(idx) + surrounding_points + 1)
                    neighborhoods.append(np.arange(start, stop, dtype=int))
                surrounding_idx = (
                    np.unique(np.concatenate(neighborhoods))
                    if neighborhoods
                    else np.asarray([], dtype=int)
                )
                surrounding_idx = surrounding_idx[
                    np.isfinite(trace_time[surrounding_idx])
                    & np.isfinite(trace_signal[surrounding_idx])
                ]

            if surrounding_idx.size > 0:
                ax.scatter(
                    trace_time[surrounding_idx],
                    trace_signal[surrounding_idx],
                    s=8,
                    alpha=0.45,
                    color="0.45",
                    linewidths=0,
                    label="surrounding",
                )

            anchor_valid = np.isfinite(x_anchor[trace_index]) & np.isfinite(
                y_anchor[trace_index]
            )
            if np.any(anchor_valid):
                ax.scatter(
                    x_anchor[trace_index, anchor_valid],
                    y_anchor[trace_index, anchor_valid],
                    s=24,
                    alpha=0.95,
                    color="tab:red",
                    linewidths=0,
                    label="anchor used",
                )

            x_line = trace_time[finite]
            order = np.argsort(x_line)
            x_line = x_line[order]
            slope = float(baseline_estimates[trace_index].slope)
            intercept = float(baseline_estimates[trace_index].intercept)
            y_line = slope * x_line + intercept
            ax.plot(
                x_line, y_line, color="tab:blue", linewidth=1.4, label="baseline fit"
            )

            ax.set_ylabel(f"Trace {trace_index + 1}")
            ax.grid(True, alpha=0.2)
            if trace_index == 0:
                ax.legend(loc="best", fontsize=8, frameon=False)

        axes_1d[-1].set_xlabel("Time [min]")
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return save_path

    def plot_peak_moment_initial_guess(
        self,
        *,
        use_aligned_time: bool = True,
        start_quantile: float = 0.005,
        end_quantile: float = 0.995,
        tail_window_sigma: float = 2.0,
        save_path: str | None = None,
    ) -> dict[str, Any]:
        """Plot moment-derived skew-normal initial guesses against measured peak data."""
        moment_data = self.compute_peak_moment_metrics(
            use_aligned_time=use_aligned_time,
            start_quantile=start_quantile,
            end_quantile=end_quantile,
            tail_window_sigma=tail_window_sigma,
        )
        metrics_by_peak = moment_data["metrics_by_peak"]
        n_peaks = len(metrics_by_peak)
        if n_peaks == 0:
            raise ValueError(
                "Cannot plot initial guesses because no peaks are defined."
            )

        time = np.asarray(
            self._time_axis(use_aligned_time=use_aligned_time), dtype=float
        )
        signal = np.asarray(self.signal, dtype=float)
        peak_masks = np.asarray(
            self.get_peak_masks(use_aligned_time=use_aligned_time), dtype=bool
        )
        n_chromatograms = int(signal.shape[0])
        baseline_estimates = self.get_baseline_estimates(
            use_aligned_time=use_aligned_time
        )

        baseline_slopes = np.asarray(
            [float(estimate.slope) for estimate in baseline_estimates], dtype=float
        )
        baseline_intercepts = np.asarray(
            [float(estimate.intercept) for estimate in baseline_estimates],
            dtype=float,
        )

        cmap = plt.get_cmap("viridis")
        vmax = max(n_chromatograms - 1, 1)

        figure, axes = plt.subplots(
            n_chromatograms,
            n_peaks,
            figsize=(4.8 * n_peaks, 2.1 * n_chromatograms),
            squeeze=False,
            constrained_layout=True,
        )

        fit_payload = [
            {
                "params_by_trace": np.full((n_chromatograms, 4), np.nan, dtype=float),
                "mask_counts": np.zeros((n_chromatograms,), dtype=int),
            }
            for _ in range(n_peaks)
        ]

        window_labels: list[str] = []
        for peak_idx in range(n_peaks):
            window_low = np.asarray(
                [float(metric.window_low) for metric in metrics_by_peak[peak_idx]],
                dtype=float,
            )
            window_high = np.asarray(
                [float(metric.window_high) for metric in metrics_by_peak[peak_idx]],
                dtype=float,
            )
            low_median = float(np.nanmedian(window_low))
            high_median = float(np.nanmedian(window_high))
            window_labels.append(
                f"Peak {peak_idx + 1} [{low_median:.4f}, {high_median:.4f}]"
            )

        for trace_idx in range(n_chromatograms):
            color = cmap(trace_idx / vmax if vmax > 0 else 0.0)
            for peak_idx in range(n_peaks):
                ax = axes[trace_idx, peak_idx]
                trace_mask = peak_masks[peak_idx, trace_idx]
                finite_mask = np.isfinite(time[trace_idx]) & np.isfinite(
                    signal[trace_idx]
                )
                active = trace_mask & finite_mask
                if int(np.sum(active)) >= 3:
                    x_trace = time[trace_idx, active]
                    y_trace = signal[trace_idx, active]
                    baseline_trace = (
                        baseline_slopes[trace_idx] * x_trace
                        + baseline_intercepts[trace_idx]
                    )
                    y_corrected = np.clip(
                        y_trace - baseline_trace, a_min=0.0, a_max=None
                    )
                    if np.any(np.isfinite(y_corrected)):
                        fit_payload[peak_idx]["mask_counts"][trace_idx] = int(
                            x_trace.size
                        )
                        ax.scatter(
                            x_trace,
                            y_corrected,
                            color=color,
                            s=8,
                            alpha=0.35,
                            linewidths=0,
                        )

                        metric = metrics_by_peak[peak_idx][trace_idx]
                        area = float(metric.area)
                        centroid = float(metric.centroid)
                        sigma_moment = float(metric.sigma)
                        alpha = float(alpha_from_skewness(float(metric.skewness)))
                        delta = alpha / np.sqrt(1.0 + alpha**2)
                        variance_factor = max(
                            1.0 - (2.0 * delta * delta / np.pi),
                            1e-8,
                        )
                        sigma = sigma_moment / np.sqrt(variance_factor)
                        mu = centroid - sigma * delta * np.sqrt(2.0 / np.pi)
                        if (
                            np.isfinite(area)
                            and np.isfinite(mu)
                            and np.isfinite(sigma)
                            and np.isfinite(alpha)
                            and area > 1e-12
                            and sigma > 1e-8
                        ):
                            x_dense = np.linspace(
                                float(np.min(x_trace)), float(np.max(x_trace)), 220
                            )
                            pdf = np.asarray(
                                skew_normal_pdf(
                                    jnp.asarray(x_dense, dtype=jnp.float32),
                                    jnp.asarray([mu], dtype=jnp.float32),
                                    jnp.asarray([sigma], dtype=jnp.float32),
                                    jnp.asarray([alpha], dtype=jnp.float32),
                                )[0],
                                dtype=float,
                            )
                            guess = area * pdf
                            ax.plot(
                                x_dense,
                                guess,
                                color=color,
                                linewidth=1.0,
                                alpha=0.85,
                            )
                            fit_payload[peak_idx]["params_by_trace"][trace_idx] = (
                                np.array([area, mu, sigma, alpha], dtype=float)
                            )

                if trace_idx == 0:
                    ax.set_title(window_labels[peak_idx])
                if peak_idx == 0:
                    ax.set_ylabel(f"Trace {trace_idx + 1}\nSignal (baseline corrected)")
                if trace_idx == (n_chromatograms - 1):
                    ax.set_xlabel("Time [min]")
                ax.grid(True, alpha=0.2)

        out_name = (
            save_path
            if save_path is not None
            else "peak_moment_initial_guess_overlay.png"
        )
        figure.savefig(out_name)
        plt.close(figure)

        return {
            "moment_data": moment_data,
            "fit_payload": fit_payload,
            "save_path": out_name,
        }


if __name__ == "__main__":
    from rich import print

    from .data import (
        BaselineAnnotation,
        PeakAnnotation,
    )

    arr = jnp.load("/Users/max/code/sahh-kinetics-hplc/chromatograms.npy").reshape(
        -1, 3000
    )[:70, :1000]
    time = jnp.load("/Users/max/code/sahh-kinetics-hplc/times.npy").reshape(-1, 3000)[
        :70, :1000
    ]
    sample_names = jnp.load("/Users/max/code/sahh-kinetics-hplc/folder_names.npy")
    chromatogram_names = jnp.load("/Users/max/code/sahh-kinetics-hplc/sample_names.npy")

    baselines = [BaselineAnnotation(low=0, high=1), BaselineAnnotation(low=4, high=6)]

    peaks = [
        PeakAnnotation(name="peak1", low=2.6, high=2.83),
        PeakAnnotation(name="peak2", low=2.9, high=3.18),
        PeakAnnotation(name="peak3", low=3.18, high=3.45),
    ]

    print("Initializing fitter...")
    fitter = Fitter(
        time,
        arr,
        baselines=baselines,
        peaks=peaks,
    )

    print("Plotting initial chromatograms...")
    fitter.plot(baseline=True, save_path="initial_chromatograms.png", single_axis=True)

    print("Aligning peaks...")
    fitter.align_peaks()

    print("Plotting aligned chromatograms...")
    fitter.plot(
        baseline=True,
        use_aligned_time=True,
        save_path="aligned_chromatograms.png",
        single_axis=True,
    )
    print("Plotting baseline anchor diagnostics...")
    baseline_diag_path = fitter.plot_baseline_anchor_fit(
        use_aligned_time=True,
        surrounding_points=8,
        save_path="baseline_peak_edge_fit.png",
    )
    print(f"Saved baseline anchor diagnostics: {baseline_diag_path}")

    print(fitter.compute_peak_moment_metrics())

    print("Plotting moment diagnostics...")
    moment_fit_path = fitter.plot_moment_peak_fits(
        use_aligned_time=True,
        save_path="nu_bayes_moment_peak_fits.png",
        column_mode="peak",
    )
    print(f"Saved moment peak fits: {moment_fit_path}")
    apex_gate_path = fitter.plot_peak_apex_gate(
        use_aligned_time=True,
        apply_apex_gate=True,
        apex_gate_n_mad=2.0,
        save_path="nu_bayes_peak_apex_gate.png",
    )
    print(f"Saved apex gate diagnostics: {apex_gate_path}")
    fwhm_path = fitter.plot_peak_fwhm(
        use_aligned_time=True,
        apply_apex_gate=True,
        apex_gate_n_mad=2.0,
        save_path="nu_bayes_peak_fwhm.png",
        column_mode="peak",
        normalize_position=True,
    )
    print(f"Saved normalized FWHM plot: {fwhm_path}")
    moment_residual_path = fitter.plot_moment_peak_residuals(
        use_aligned_time=True,
        save_path="nu_bayes_moment_peak_residuals.png",
        column_mode="peak",
    )
    print(f"Saved moment peak residuals: {moment_residual_path}")
    moment_residual_mean_path = fitter.plot_moment_peak_residuals_column_sum(
        use_aligned_time=True,
        save_path="nu_bayes_moment_peak_residuals_column_mean.png",
        column_mode="peak",
        aggregation="mean",
    )
    print(f"Saved moment residual column means: {moment_residual_mean_path}")
    print("Fitting...")

    fitter.fit()
    summary_path = fitter.save_arviz_summary_txt("nu_bayes_arviz_summary.txt")
    print(f"Saved ArviZ summary: {summary_path}")
    trace_path = fitter.plot_arviz_trace(
        save_path="nu_bayes_trace.png",
        var_names=["A", "mu", "sigma", "alpha", "baseline_intercept", "sigma_y"],
    )
    print(f"Saved ArviZ trace: {trace_path}")
    posterior_fit_path = fitter.plot_posterior_peak_fits(
        use_aligned_time=True,
        save_path="nu_bayes_posterior_peak_fits.png",
        column_mode="peak",
    )
    print(f"Saved posterior peak fits: {posterior_fit_path}")
    residual_path = fitter.plot_posterior_peak_residuals(
        use_aligned_time=True,
        save_path="nu_bayes_posterior_peak_residuals.png",
        column_mode="peak",
    )
    print(f"Saved posterior peak residuals: {residual_path}")

    pair_path = fitter.plot_arviz_pair(
        save_path="nu_bayes_pair.png",
        var_names=[
            "A",
            "mu",
            "sigma",
            "alpha",
            "baseline_intercept",
            "sigma_y",
        ],
        max_subplots=10000,
    )
    print(f"Saved ArviZ pair plot: {pair_path}")
