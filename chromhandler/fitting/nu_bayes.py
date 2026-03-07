from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS

from .baseline import BaselinePriors, estimate_baseline
from .data import (
    BaselineAnnotation,
    PeakAnnotation,
    PeakPriorHints,
    baseline_to_mask,
    peaks_to_mask,
)
from .peak_models import (
    SAMPLED_PARAMETER_NAMES,
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
        self.peak_prior_hints: list[PeakPriorHints] = []
        self._baseline_priors_cache: dict[bool, BaselinePriors] = {}
        self.mu_init = jnp.zeros((self.signal.shape[0], 0), dtype=jnp.float32)
        self.sigma_init = jnp.zeros((self.signal.shape[0], 0), dtype=jnp.float32)
        self.A_init = jnp.zeros((self.signal.shape[0], 0), dtype=jnp.float32)
        self.alpha_init = jnp.zeros((self.signal.shape[0], 0), dtype=jnp.float32)
        self.model_inputs: dict[str, Any] | None = None
        self.samples: dict[str, jnp.ndarray] | None = None
        self.mcmc: Any = None
        self.idata: Any = None
        self._fwhm_diag: list[dict] | None = (
            None  # populated by _build_component_initializers
        )
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

    def _compute_baseline_priors(self, *, use_aligned_time: bool) -> BaselinePriors:
        """Fit OLS linear baseline through peak-window anchor points."""
        time_axis = self._time_axis(use_aligned_time=use_aligned_time)
        return estimate_baseline(
            time_axis,
            self.signal,
            peaks=self.peaks,
            baselines=self.baselines,
        )

    def get_baseline_priors(self, *, use_aligned_time: bool = False) -> BaselinePriors:
        """Per-trace linear baseline priors, cached per time-axis variant."""
        key = bool(use_aligned_time)
        if key not in self._baseline_priors_cache:
            self._baseline_priors_cache[key] = self._compute_baseline_priors(
                use_aligned_time=use_aligned_time
            )
        return self._baseline_priors_cache[key]

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
        priors = self.get_baseline_priors(use_aligned_time=use_aligned_time)
        baseline = (
            jnp.asarray(priors.intercept, dtype=jnp.float32)[:, None]
            + jnp.asarray(priors.slope, dtype=jnp.float32)[:, None] * time_axis
        )
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
                component_include_in_total_area.append(True)
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

    def _build_component_initializers(
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
        _fwhm_diag_list: list[dict] = []
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

            # --- Shoulder separation estimation ---
            # Three-tier approach: bimodal apex → HWHM excess → geometric fallback.
            sep_est_for_prior = 0.0
            _tier_used = 0
            _t1_n_cand = 0
            _t1_sh_apexes: np.ndarray = np.array([], dtype=float)
            _t1_raw_sep = 0.0
            _t2_sigma_ref = 0.0
            _t2_gauss_hwhm = 0.0
            _t2_median_hwhm = 0.0
            _t2_excess = 0.0
            _t3_sigma_fb = 0.0
            _t3_geo = 0.0
            if side != 0 and int(np.sum(keep_width)) > 0:
                T_main_val = mode_loc  # robust median of in-gate apex times

                # Tier 1: out-of-gate traces on the shoulder side
                out_of_gate_valid = width_valid & ~gate_keep[:, logical_index]
                if side > 0:
                    cand_sh = (
                        out_of_gate_valid
                        & np.isfinite(mode_trace)
                        & (mode_trace > T_main_val)
                    )
                else:
                    cand_sh = (
                        out_of_gate_valid
                        & np.isfinite(mode_trace)
                        & (mode_trace < T_main_val)
                    )
                _t1_n_cand = int(np.sum(cand_sh))
                if _t1_n_cand >= 2:
                    sh_apexes = mode_trace[cand_sh]
                    sh_apexes = sh_apexes[np.isfinite(sh_apexes)]
                    _t1_sh_apexes = sh_apexes.copy()
                    if sh_apexes.size >= 1:
                        raw_sep = abs(T_main_val - float(np.median(sh_apexes)))
                        _t1_raw_sep = raw_sep
                        if raw_sep > 0.003 * span:
                            sep_est_for_prior = raw_sep
                            _tier_used = 1

                # Tier 2: excess HWHM on the shoulder side.
                # Use the UNAFFECTED (opposite) side sigma as reference so that
                # natural peak asymmetry doesn't mask the shoulder:
                #   left shoulder → right side is clean → use sigma_right as ref
                #   right shoulder → left side is clean → use sigma_left as ref
                if sep_est_for_prior < 1e-6:
                    if side > 0:
                        sigma_far_vals = sigma_left_trace[keep_width]
                    else:
                        sigma_far_vals = sigma_right_trace[keep_width]
                    sigma_far_vals = sigma_far_vals[
                        np.isfinite(sigma_far_vals) & (sigma_far_vals > 0)
                    ]
                    sigma_ref = (
                        float(np.nanmedian(sigma_far_vals))
                        if sigma_far_vals.size > 0
                        else float(sigma_loc)
                    )
                    _t2_sigma_ref = sigma_ref
                    gauss_hwhm = sigma_ref * gaussian_hwhm_factor
                    _t2_gauss_hwhm = gauss_hwhm
                    hwhm_side = (w_right if side > 0 else w_left)[keep_width]
                    hwhm_side = hwhm_side[np.isfinite(hwhm_side) & (hwhm_side > 0)]
                    if hwhm_side.size > 0:
                        excess = float(np.nanmedian(hwhm_side)) - gauss_hwhm
                        _t2_median_hwhm = float(np.nanmedian(hwhm_side))
                        _t2_excess = excess
                        if excess > 1e-6:
                            sep_est_for_prior = max(1.5 * excess, 0.005 * span)
                            _tier_used = 2

                # Tier 3: geometric fallback — at least 2σ to avoid shoulder
                # landing inside the main peak body where subtraction collapses.
                if sep_est_for_prior < 1e-6:
                    sigma_all = sigma_trace[
                        np.isfinite(sigma_trace) & (sigma_trace > 0)
                    ]
                    sigma_fb = (
                        float(np.nanmedian(sigma_all))
                        if sigma_all.size > 0
                        else float(sigma_loc)
                    )
                    _t3_sigma_fb = sigma_fb
                    if side > 0:
                        geo = 0.15 * (high - T_main_val)
                        sep_est_for_prior = max(geo, 2.0 * sigma_fb, 0.01 * span)
                    else:
                        geo = 0.15 * (T_main_val - low)
                        sep_est_for_prior = max(geo, 2.0 * sigma_fb, 0.01 * span)
                    _t3_geo = geo
                    _tier_used = 3

                sep_est_for_prior = float(
                    np.clip(sep_est_for_prior, 0.002 * span, 0.60 * span)
                )

            # --- Second pass: data-driven shoulder sep_est from residuals ---
            # Pre-compute the main-component area for every trace so that
            # _compute_main_residuals_for_window can build accurate residuals
            # before the per-trace init loop runs.
            _pass2: dict = {"n_valid": 0, "sep_est": None}
            if side != 0:
                _main_area_pre = np.full((n_traces,), 1e-8, dtype=float)
                for _t in range(n_traces):
                    _xi = (
                        float(xi_trace[_t])
                        if keep_width[_t] and np.isfinite(xi_trace[_t])
                        else float(mu_loc)
                    )
                    _sig = (
                        float(sigma_trace[_t])
                        if keep_width[_t] and np.isfinite(sigma_trace[_t])
                        else float(sigma_loc)
                    )
                    _alp = (
                        float(alpha_trace[_t])
                        if keep_width[_t] and np.isfinite(alpha_trace[_t])
                        else float(alpha_loc)
                    )
                    _tgt = (
                        float(mode_trace[_t])
                        if keep_width[_t] and np.isfinite(mode_trace[_t])
                        else float(mode_loc)
                    )
                    _active = (
                        peak_mask_matrix[logical_index, _t]
                        & np.isfinite(time_matrix[_t])
                        & np.isfinite(signal_matrix[_t])
                    )
                    _cand = np.flatnonzero(_active)
                    if _cand.size == 0:
                        _in_win = (
                            np.isfinite(time_matrix[_t])
                            & np.isfinite(signal_matrix[_t])
                            & (time_matrix[_t] >= low)
                            & (time_matrix[_t] <= high)
                        )
                        _cand = np.flatnonzero(_in_win)
                    _h = 0.0
                    if _cand.size > 0:
                        _ni = int(
                            _cand[int(np.argmin(np.abs(time_matrix[_t, _cand] - _tgt)))]
                        )
                        _h = max(float(signal_matrix[_t, _ni]), 0.0)
                    _pdf_pre = float(
                        np.asarray(
                            skew_normal_pdf(
                                jnp.asarray([_tgt], dtype=jnp.float32),
                                jnp.asarray([_xi], dtype=jnp.float32),
                                jnp.asarray([_sig], dtype=jnp.float32),
                                jnp.asarray([_alp], dtype=jnp.float32),
                            )[0, 0],
                            dtype=float,
                        )
                    )
                    _main_area_pre[_t] = max(
                        _h / _pdf_pre
                        if np.isfinite(_pdf_pre) and _pdf_pre > 1e-12
                        else _h * _sig * sqrt_two_pi,
                        1e-8,
                    )
                _d_pass2 = {
                    "T_main": mode_loc,
                    "sigma_loc": sigma_loc,
                    "mu_loc": mu_loc,
                    "alpha_loc": alpha_loc,
                    "low": low,
                    "high": high,
                    "side": side,
                    "xi_trace": xi_trace,
                    "sigma_trace": sigma_trace,
                    "alpha_trace": alpha_trace,
                    "main_area_per_trace": _main_area_pre,
                }
                _pass2 = self._second_pass_shoulder_fwhm(
                    _d_pass2,
                    time_matrix,
                    signal_matrix,
                )
                if _pass2.get("n_valid", 0) >= 3 and _pass2.get("sep_est") is not None:
                    _sep2 = float(
                        np.clip(
                            max(float(_pass2["sep_est"]), 2.0 * sigma_loc),
                            0.002 * span,
                            0.60 * span,
                        )
                    )
                    sep_est_for_prior = _sep2
                    _tier_used = 0  # sentinel: "pass2"

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
                    # Use data-driven separation estimate when available
                    if sep_est_for_prior > 1e-6:
                        offset = sep_est_for_prior
                    else:
                        offset = max(0.08 * span, min(0.25 * span, 0.8 * sigma_value))
                    direction = 1.0 if side > 0 else -1.0
                    # main_mu: location parameter (xi) of main component
                    main_mu = float(np.clip(xi_value, low, high))
                    # shoulder_mode_pos: the MODE of the shoulder component,
                    # offset from the main peak MODE (target_mode), not from xi
                    shoulder_mode_pos = float(
                        np.clip(target_mode + direction * offset, low, high)
                    )

                    # --- Shoulder component parameters ---
                    sigma_sh = max(0.75 * sigma_value, 1e-4)
                    alpha_sh = 0.5 * alpha_value
                    mode_offset_sh = float(
                        _skew_mode_offsets(np.asarray([alpha_sh]))[0]
                    )
                    # xi_sh: location parameter whose mode = shoulder_mode_pos
                    xi_sh = float(
                        np.clip(
                            shoulder_mode_pos - sigma_sh * mode_offset_sh, low, high
                        )
                    )

                    # Signal height at shoulder mode position
                    if candidate_idx.size > 0:
                        near_sh = int(
                            candidate_idx[
                                int(
                                    np.argmin(
                                        np.abs(
                                            time_matrix[trace_index, candidate_idx]
                                            - shoulder_mode_pos
                                        )
                                    )
                                )
                            ]
                        )
                        h_sh = max(float(signal_matrix[trace_index, near_sh]), 0.0)
                    else:
                        h_sh = 0.0

                    # Net shoulder height: subtract main component's contribution
                    pdf_main_at_sh = float(
                        np.asarray(
                            skew_normal_pdf(
                                jnp.asarray([shoulder_mode_pos], dtype=jnp.float32),
                                jnp.asarray([xi_value], dtype=jnp.float32),
                                jnp.asarray([sigma_value], dtype=jnp.float32),
                                jnp.asarray([alpha_value], dtype=jnp.float32),
                            )[0, 0],
                            dtype=float,
                        )
                    )
                    h_sh_net = h_sh - area_value * pdf_main_at_sh

                    # Shoulder area from net height when the shoulder pokes out
                    # clearly above the main tail. When buried (small sep or
                    # compressed tail side), fall back to Beta(3,1) prior mean:
                    # main_fraction = 0.75 → shoulder = 25% of total = main / 3.
                    if h_sh_net > 0.02 * max(h_sh, 1e-12):
                        pdf_sh_at_mode = float(
                            np.asarray(
                                skew_normal_pdf(
                                    jnp.asarray([shoulder_mode_pos], dtype=jnp.float32),
                                    jnp.asarray([xi_sh], dtype=jnp.float32),
                                    jnp.asarray([sigma_sh], dtype=jnp.float32),
                                    jnp.asarray([alpha_sh], dtype=jnp.float32),
                                )[0, 0],
                                dtype=float,
                            )
                        )
                        if np.isfinite(pdf_sh_at_mode) and pdf_sh_at_mode > 1e-12:
                            area_sh = h_sh_net / pdf_sh_at_mode
                        else:
                            area_sh = h_sh_net * sigma_sh * sqrt_two_pi
                    else:
                        # Shoulder buried in main's tail → Beta(3,1) prior mean
                        area_sh = area_value / 3.0
                    area_sh = max(float(area_sh), 1e-8)

                    mu_init[trace_index, m_idx] = main_mu
                    # Store xi_sh (location parameter) so prior plot displays
                    # shoulder mode at shoulder_mode_pos = xi_sh + sigma_sh*offset
                    mu_init[trace_index, s_idx] = xi_sh
                    sigma_init[trace_index, m_idx] = sigma_value
                    sigma_init[trace_index, s_idx] = sigma_sh
                    alpha_init[trace_index, m_idx] = alpha_value
                    alpha_init[trace_index, s_idx] = alpha_sh
                    A_init[trace_index, m_idx] = area_value
                    A_init[trace_index, s_idx] = area_sh
                    # Update total area for prior_hints computation
                    area_trace[trace_index] = area_value + area_sh

            _fwhm_diag_list.append(
                {
                    "peak_name": getattr(peak, "name", f"peak_{logical_index}"),
                    "logical_index": logical_index,
                    "low": low,
                    "high": high,
                    "side": side,
                    "T_main": mode_loc,
                    "mu_loc": mu_loc,
                    "sigma_loc": sigma_loc,
                    "alpha_loc": alpha_loc,
                    "sep_est": sep_est_for_prior,
                    "tier_used": _tier_used,
                    "t1_n_cand": _t1_n_cand,
                    "t1_sh_apexes": _t1_sh_apexes.copy(),
                    "t1_raw_sep": _t1_raw_sep,
                    "t2_sigma_ref": _t2_sigma_ref,
                    "t2_gauss_hwhm": _t2_gauss_hwhm,
                    "t2_median_hwhm": _t2_median_hwhm,
                    "t2_excess": _t2_excess,
                    "t3_sigma_fb": _t3_sigma_fb,
                    "t3_geo": _t3_geo,
                    "pass2_sep_est": _pass2.get("sep_est"),
                    "pass2_n_valid": _pass2.get("n_valid", 0),
                    "pass2_area_split": _pass2.get("area_split"),
                    "pass2_sh_apex_times": _pass2.get("sh_apex_times", []),
                    "pass2_trace_indices": _pass2.get("trace_indices", []),
                    "pass2_snr_list": _pass2.get("snr_list", []),
                    "mode_trace": mode_trace.copy(),
                    "xi_trace": xi_trace.copy(),
                    "sigma_trace": sigma_trace.copy(),
                    "alpha_trace": alpha_trace.copy(),
                    "area_trace": area_trace.copy(),
                    "w_left": w_left.copy(),
                    "w_right": w_right.copy(),
                    "keep_width": keep_width.copy(),
                    "gate_keep_col": gate_keep[:, logical_index].copy(),
                    "width_valid": width_valid.copy(),
                    "main_area_per_trace": A_init[:, m_idx].copy(),
                }
            )

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
                    sep_est=float(sep_est_for_prior),
                )
            )

        mu_init = np.clip(mu_init, mu_lo[None, :], mu_hi[None, :])
        sigma_init = np.clip(
            sigma_init,
            max(float(self.sigma_min), 1e-4),
            max(float(self.sigma_max), 1.5 * float(self.sigma_min)),
        )
        A_init = np.maximum(A_init, 1e-8)
        self._fwhm_diag = _fwhm_diag_list
        return (
            jnp.asarray(mu_init, dtype=jnp.float32),
            jnp.asarray(sigma_init, dtype=jnp.float32),
            jnp.asarray(A_init, dtype=jnp.float32),
            jnp.asarray(alpha_init, dtype=jnp.float32),
            prior_hints,
        )

    # ------------------------------------------------------------------
    # Second-pass shoulder FWHM helpers
    # ------------------------------------------------------------------

    def _compute_main_residuals_for_window(
        self,
        d: dict,
        t_win: np.ndarray,
        sig_win: np.ndarray,
    ) -> np.ndarray:
        """Subtract per-trace main-component skew-normal fit from windowed signal.

        Parameters
        ----------
        d:
            A ``_fwhm_diag``-compatible dict containing ``xi_trace``,
            ``sigma_trace``, ``alpha_trace``, ``mu_loc``, ``sigma_loc``,
            ``alpha_loc``, and ``main_area_per_trace``.
        t_win:
            1-D time grid for the window, shape ``(n_t_win,)``.
        sig_win:
            Baseline-corrected signal sliced to the window,
            shape ``(n_trace, n_t_win)``.

        Returns
        -------
        numpy.ndarray
            Residuals, shape ``(n_trace, n_t_win)``.  Rows where the main
            component cannot be reconstructed are filled with ``nan``.
        """
        n_trace = sig_win.shape[0]
        residuals = np.full_like(sig_win, np.nan, dtype=float)
        for t in range(n_trace):
            xi_v = (
                float(d["xi_trace"][t])
                if np.isfinite(d["xi_trace"][t])
                else float(d["mu_loc"])
            )
            sig_v = (
                float(d["sigma_trace"][t])
                if np.isfinite(d["sigma_trace"][t])
                else float(d["sigma_loc"])
            )
            alp_v = (
                float(d["alpha_trace"][t])
                if np.isfinite(d["alpha_trace"][t])
                else float(d["alpha_loc"])
            )
            area_main = float(d["main_area_per_trace"][t])
            if area_main < 1e-8 or sig_v < 1e-6:
                continue
            pdf_vals = np.asarray(
                skew_normal_pdf(
                    jnp.asarray(t_win, dtype=jnp.float32),
                    jnp.asarray([xi_v], dtype=jnp.float32),
                    jnp.asarray([sig_v], dtype=jnp.float32),
                    jnp.asarray([alp_v], dtype=jnp.float32),
                )[:, 0],
                dtype=float,
            )
            residuals[t] = sig_win[t] - area_main * pdf_vals
        return residuals

    def _second_pass_shoulder_fwhm(
        self,
        d: dict,
        time_matrix: np.ndarray,
        signal_matrix: np.ndarray,
        *,
        window_sigma: float = 3.0,
        min_n: int = 3,
        min_shoulder_fraction: float = 0.04,
    ) -> dict:
        """Data-driven shoulder separation from FWHM on main-fit residuals.

        Subtracts the main skew-normal reconstruction (Pass-1 parameters) from
        the baseline-corrected signal, cuts to a window on the shoulder side
        (``window_sigma`` × ``sigma_loc`` wide), and runs FWHM analysis on
        the residuals to locate the shoulder apex.

        Parameters
        ----------
        d:
            A dict with keys ``T_main``, ``sigma_loc``, ``mu_loc``,
            ``alpha_loc``, ``low``, ``high``, ``side``, ``xi_trace``,
            ``sigma_trace``, ``alpha_trace``, ``main_area_per_trace``.
        time_matrix:
            ``[n_trace, n_time]`` (or ``[n_time]``) time axis.
        signal_matrix:
            ``[n_trace, n_time]`` baseline-corrected signal.
        window_sigma:
            How many ``sigma_loc`` to extend the window on the shoulder side.
        min_n:
            Minimum number of traces with a valid shoulder FWHM for the
            result to be accepted.
        min_shoulder_fraction:
            Minimum ratio of residual shoulder-peak height to the
            baseline-corrected signal at ``T_main``.  Traces where the
            shoulder residual is smaller than this fraction of the main
            signal are excluded from the apex median.  This filters out
            early traces where the shoulder is below the modeling-noise
            floor (default 0.04 = 4 %).

        Returns
        -------
        dict
            ``sep_est`` (unsigned, mode-to-mode), ``sigma_sh``,
            ``area_split``, ``n_valid``, ``sh_apex_times``,
            ``trace_indices``, ``snr_list``.  When the estimate is
            unreliable ``sep_est`` is ``None`` and ``n_valid < min_n``.
        """
        T_main = float(d["T_main"])
        sigma_loc = float(d["sigma_loc"])
        low = float(d["low"])
        high = float(d["high"])
        side = int(d["side"])
        direction = 1.0 if side > 0 else -1.0
        span = max(high - low, 1e-4)

        # Shoulder-side window: [T_main … T_main + direction × window_sigma × σ]
        win_end = T_main + direction * window_sigma * sigma_loc
        if direction > 0:
            win_lo, win_hi = T_main, win_end
        else:
            win_lo, win_hi = win_end, T_main
        win_lo = float(np.clip(win_lo, low, high))
        win_hi = float(np.clip(win_hi, low, high))
        if win_hi - win_lo < 0.002 * span:
            return {
                "n_valid": 0,
                "sep_est": None,
                "sh_apex_times": [],
                "trace_indices": [],
                "snr_list": [],
            }

        t_ref = time_matrix[0] if time_matrix.ndim == 2 else time_matrix
        window_mask = (t_ref >= win_lo) & (t_ref <= win_hi)
        if int(np.sum(window_mask)) < 3:
            return {
                "n_valid": 0,
                "sep_est": None,
                "sh_apex_times": [],
                "trace_indices": [],
                "snr_list": [],
            }

        t_win = t_ref[window_mask]
        sig_win = signal_matrix[:, window_mask]

        residuals = self._compute_main_residuals_for_window(d, t_win, sig_win)

        gaussian_hwhm_factor = float(np.sqrt(2.0 * np.log(2.0)))
        # Index of T_main in the window — used for SNR reference height
        idx_T_main = int(np.argmin(np.abs(t_win - T_main)))
        sh_apex_times: list[float] = []
        sh_sigmas: list[float] = []
        sh_areas: list[float] = []
        sh_main_areas: list[float] = []
        _snr_list: list[float] = []
        _trace_indices_list: list[int] = []

        for t in range(residuals.shape[0]):
            row = residuals[t]
            if not np.all(np.isfinite(row)):
                continue
            pos = np.clip(row, 0.0, None)
            if float(np.max(pos)) < 1e-8:
                continue
            payload = _compute_normalized_fwhm(t_win, row, half_level=0.5)
            if not np.isfinite(payload["apex_time"]):
                continue
            if not np.isfinite(payload["fwhm"]):
                continue
            apex_t = float(payload["apex_time"])
            # Require shoulder apex on the shoulder side of the main apex
            if direction > 0 and apex_t <= T_main:
                continue
            if direction < 0 and apex_t >= T_main:
                continue
            # SNR filter: residual shoulder peak must be at least min_shoulder_fraction
            # of the baseline-corrected signal at T_main.  This excludes early traces
            # where the shoulder is below the main-peak modeling noise floor, so that
            # the unweighted median is only computed from traces where the shoulder is
            # genuinely visible in the residual.
            h_ref = max(float(sig_win[t, idx_T_main]), 1e-10)
            apex_height_res = float(payload["apex_height"])
            snr = apex_height_res / h_ref
            if snr < min_shoulder_fraction:
                continue  # shoulder too small relative to main — likely modeling noise
            sh_apex_times.append(apex_t)
            sh_sigmas.append(float(payload["fwhm"]) / (2.0 * gaussian_hwhm_factor))
            sh_areas.append(float(np.trapz(pos, t_win)))
            sh_main_areas.append(float(d["main_area_per_trace"][t]))
            _snr_list.append(snr)
            _trace_indices_list.append(t)

        n_valid = len(sh_apex_times)
        if n_valid < min_n:
            return {
                "n_valid": n_valid,
                "sep_est": None,
                "sh_apex_times": sh_apex_times,
                "trace_indices": _trace_indices_list,
                "snr_list": _snr_list,
            }

        # Unweighted median across SNR-filtered traces. Each included trace's shoulder
        # estimate contributes equally.  Early traces where the shoulder is invisible
        # have already been removed by the SNR filter above.
        sep_est = abs(float(np.median(sh_apex_times)) - T_main)
        sigma_sh = float(np.median(sh_sigmas))
        split_ratios = np.array(
            [
                sa / (sa + ma) if (sa + ma) > 0.0 else 0.25
                for sa, ma in zip(sh_areas, sh_main_areas)
            ],
            dtype=float,
        )
        area_split = float(np.median(split_ratios))
        return {
            "sep_est": sep_est,
            "sigma_sh": sigma_sh,
            "area_split": area_split,
            "n_valid": n_valid,
            "sh_apex_times": sh_apex_times,
            "trace_indices": _trace_indices_list,
            "snr_list": _snr_list,
        }

    def _build_model_inputs(
        self,
        *,
        use_aligned_time: bool = True,
    ) -> dict[str, Any]:
        metadata = self._build_component_metadata()
        time_axis = self._time_axis(use_aligned_time=use_aligned_time)
        signal_corrected = self.baseline_corrected_signal(
            use_aligned_time=use_aligned_time
        )
        peak_masks = self.get_peak_masks(use_aligned_time=use_aligned_time)
        mu_init, sigma_init, A_init, alpha_init, prior_hints = (
            self._build_component_initializers(
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

        baseline_priors = self.get_baseline_priors(use_aligned_time=use_aligned_time)
        peak_mask = (
            jnp.any(peak_masks, axis=0)
            if peak_masks.shape[0] > 0
            else jnp.zeros(time_axis.shape, dtype=bool)
        )
        peak_mask_arg: jnp.ndarray | None = (
            peak_mask if bool(jnp.any(peak_mask)) else None
        )

        # Peak-level geometry
        sorted_peaks: list[PeakAnnotation] = metadata["sorted_peaks"]
        n_peak = len(sorted_peaks)
        mu_lo_np = np.asarray(metadata["logical_mu_lo"], dtype=float)  # [n_peak]
        mu_hi_np = np.asarray(metadata["logical_mu_hi"], dtype=float)  # [n_peak]
        logical_span = np.maximum(mu_hi_np - mu_lo_np, 1e-4)
        shoulder_side_np = np.asarray(
            metadata["logical_shoulder_side"], dtype=int
        )  # [n_peak]
        shoulder_enabled = shoulder_side_np != 0
        shoulder_peak_index = np.where(shoulder_enabled)[0].astype(np.int32)
        main_idx = np.asarray(metadata["logical_main_component_index"], dtype=int)
        shoulder_idx_raw = np.asarray(
            metadata["logical_shoulder_component_index"], dtype=int
        )
        shoulder_safe = np.where(shoulder_idx_raw >= 0, shoulder_idx_raw, main_idx)

        # Reshape init arrays: [n_trace, n_component_old] → [n_trace, n_peak, 2]
        # where axis=-1 index 0 = main component, 1 = shoulder component.
        A_init_np = np.asarray(A_init, dtype=float)
        A_init_model = np.stack(
            [
                A_init_np[:, main_idx],
                A_init_np[:, shoulder_safe] * shoulder_enabled[None, :],
            ],
            axis=-1,
        )  # [n_trace, n_peak, 2]
        mu_init_np = np.asarray(mu_init, dtype=float)
        mu_init_model = np.stack(
            [mu_init_np[:, main_idx], mu_init_np[:, shoulder_safe]], axis=-1
        )  # [n_trace, n_peak, 2]
        sigma_init_np = np.asarray(sigma_init, dtype=float)
        sigma_init_model = np.stack(
            [sigma_init_np[:, main_idx], sigma_init_np[:, shoulder_safe]], axis=-1
        )  # [n_trace, n_peak, 2]
        alpha_init_np = np.asarray(alpha_init, dtype=float)
        alpha_init_model = np.stack(
            [alpha_init_np[:, main_idx], alpha_init_np[:, shoulder_safe]], axis=-1
        )  # [n_trace, n_peak, 2]

        # component_to_logical for new flat layout [main0, sh0, main1, sh1, ...]
        component_to_logical = np.repeat(np.arange(n_peak), 2)  # [2*n_peak]

        # Per-peak priors from prior_hints.
        # For shoulder peaks, shift mu_center_loc from the main-peak apex to the
        # midpoint between main and shoulder modes (model parameterisation uses
        # mu_center = midpoint, so anchoring at T_main would allow the shoulder
        # component to drift onto the main peak).
        mu_center_loc_raw = np.array([h.mu_loc for h in prior_hints], dtype=float)
        for _i, _h in enumerate(prior_hints):
            if shoulder_enabled[_i] and _h.sep_est > 1e-6:
                _side = float(shoulder_side_np[_i])
                mu_center_loc_raw[_i] = _h.mu_loc + 0.5 * _side * _h.sep_est
        mu_center_loc = np.clip(mu_center_loc_raw, mu_lo_np, mu_hi_np).astype(float)
        mu_center_scale = np.maximum([h.mu_scale for h in prior_hints], 1e-6).astype(
            float
        )

        sigma_loc_np = np.asarray([h.sigma_loc for h in prior_hints], dtype=float)
        sigma_scale_np = np.asarray([h.sigma_scale for h in prior_hints], dtype=float)
        sigma_prior_loc = np.stack(
            [sigma_loc_np, 0.75 * sigma_loc_np], axis=-1
        )  # [n_peak, 2]
        sigma_prior_scale = np.stack(
            [sigma_scale_np, sigma_scale_np], axis=-1
        )  # [n_peak, 2]

        alpha_loc_np = np.asarray([h.alpha_loc for h in prior_hints], dtype=float)
        alpha_scale_np = np.maximum([h.alpha_scale for h in prior_hints], 1e-3).astype(
            float
        )
        alpha_prior_loc = np.stack(
            [alpha_loc_np, 0.5 * alpha_loc_np], axis=-1
        )  # [n_peak, 2]
        alpha_prior_scale = np.stack(
            [alpha_scale_np, alpha_scale_np], axis=-1
        )  # [n_peak, 2]

        area_total_loc = np.maximum([h.area_loc for h in prior_hints], 1e-8).astype(
            float
        )
        area_total_scale = np.maximum([h.area_scale for h in prior_hints], 1e-6).astype(
            float
        )

        # Beta(1, 1) = Uniform(0, 1) for shoulder area split.
        # No prior preference on how total area is split between main and shoulder.
        # In a kinetic series, area_split must range from ~1 (early, all main) to
        # ~0 (late, all shoulder). Separation parameterisation (mu_center ± separation/2)
        # already prevents component-swap, so the prior uses maximum entropy.
        area_split_alpha = np.where(shoulder_enabled, 1.0, 1.0)
        area_split_beta = np.where(shoulder_enabled, 1.0, 1.0)

        # Separation bounds: data-driven around sep_est when available, else geometric.
        # Tight [0.5×, 2.5×] bounds prevent the component swap (shoulder stealing the
        # main peak) that occurs when the prior is wide and uninformative.
        separation_low = np.zeros(n_peak, dtype=float)
        separation_high = np.full(n_peak, 1e-6, dtype=float)
        separation_est = np.zeros(n_peak, dtype=float)
        for _i, _h in enumerate(prior_hints):
            if shoulder_enabled[_i]:
                _s = _h.sep_est
                if _s > 1e-6:
                    _lo = float(
                        np.clip(
                            0.5 * _s, 0.005 * logical_span[_i], 0.40 * logical_span[_i]
                        )
                    )
                    _hi = float(np.clip(2.5 * _s, _lo + 1e-6, 0.80 * logical_span[_i]))
                else:
                    _lo = 0.04 * float(logical_span[_i])
                    _hi = 0.50 * float(logical_span[_i])
                    _s = 0.5 * (_lo + _hi)
                separation_low[_i] = _lo
                separation_high[_i] = _hi
                separation_est[_i] = _s

        # Per-trace noise prior: signal std in peak windows
        signal_np = np.asarray(self.signal, dtype=float)
        if peak_mask_arg is not None:
            peak_mask_np = np.asarray(peak_mask, dtype=bool)
            masked_signal = np.where(peak_mask_np, signal_np, np.nan)
        else:
            masked_signal = signal_np
        sigma_y_prior_loc = np.nanstd(masked_signal, axis=1)
        sigma_y_prior_loc = np.where(
            np.isfinite(sigma_y_prior_loc) & (sigma_y_prior_loc > 0),
            sigma_y_prior_loc,
            1.0,
        )

        model_inputs = {
            "x": jnp.asarray(time_axis, dtype=jnp.float32),
            "y": jnp.asarray(self.signal, dtype=jnp.float32),
            "mu_lo": jnp.asarray(mu_lo_np, dtype=jnp.float32),
            "mu_hi": jnp.asarray(mu_hi_np, dtype=jnp.float32),
            "shoulder_side": jnp.asarray(shoulder_side_np, dtype=jnp.int32),
            "shoulder_peak_index": jnp.asarray(shoulder_peak_index, dtype=jnp.int32),
            "A_init": jnp.asarray(A_init_model, dtype=jnp.float32),
            "mu_center_loc": jnp.asarray(mu_center_loc, dtype=jnp.float32),
            "mu_center_scale": jnp.asarray(mu_center_scale, dtype=jnp.float32),
            "separation_low": jnp.asarray(separation_low, dtype=jnp.float32),
            "separation_high": jnp.asarray(separation_high, dtype=jnp.float32),
            "sigma_prior_loc": jnp.asarray(sigma_prior_loc, dtype=jnp.float32),
            "sigma_prior_scale": jnp.asarray(sigma_prior_scale, dtype=jnp.float32),
            "alpha_prior_loc": jnp.asarray(alpha_prior_loc, dtype=jnp.float32),
            "alpha_prior_scale": jnp.asarray(alpha_prior_scale, dtype=jnp.float32),
            "area_total_loc": jnp.asarray(area_total_loc, dtype=jnp.float32),
            # area_total_scale kept as metadata for diagnostics only (model uses LogNormal CV=0.3)
            "area_total_scale": jnp.asarray(area_total_scale, dtype=jnp.float32),
            "area_split_alpha": jnp.asarray(area_split_alpha, dtype=jnp.float32),
            "area_split_beta": jnp.asarray(area_split_beta, dtype=jnp.float32),
            "baseline_intercept_loc": jnp.asarray(
                baseline_priors.intercept, dtype=jnp.float32
            ),
            "baseline_intercept_scale": jnp.asarray(
                baseline_priors.intercept_scale, dtype=jnp.float32
            ),
            "baseline_slope_loc": jnp.asarray(baseline_priors.slope, dtype=jnp.float32),
            "baseline_slope_scale": jnp.asarray(
                baseline_priors.slope_scale, dtype=jnp.float32
            ),
            "sigma_y_prior_loc": jnp.asarray(sigma_y_prior_loc, dtype=jnp.float32),
            "peak_mask": peak_mask_arg,
            # Extra metadata used by post-processing/plotting (not passed to numpyro model)
            "component_to_logical_index": jnp.asarray(
                component_to_logical, dtype=jnp.int32
            ),
            "separation_est": jnp.asarray(separation_est, dtype=jnp.float32),
            "mu_init_model": jnp.asarray(mu_init_model, dtype=jnp.float32),
            "sigma_init_model": jnp.asarray(sigma_init_model, dtype=jnp.float32),
            "alpha_init_model": jnp.asarray(alpha_init_model, dtype=jnp.float32),
        }
        self.model_inputs = model_inputs
        return model_inputs

    def _build_init_values_for_nuts(
        self, model_inputs: dict[str, Any]
    ) -> dict[str, Any]:
        n_peak = int(jnp.asarray(model_inputs["mu_lo"]).shape[0])
        shoulder_side = np.asarray(model_inputs["shoulder_side"], dtype=int)  # [n_peak]
        shoulder_enabled = shoulder_side != 0
        shoulder_peak_index = np.asarray(model_inputs["shoulder_peak_index"], dtype=int)
        n_shoulder_peak = int(shoulder_peak_index.shape[0])

        mu_lo = np.asarray(model_inputs["mu_lo"], dtype=float)
        mu_hi = np.asarray(model_inputs["mu_hi"], dtype=float)
        logical_span = np.maximum(mu_hi - mu_lo, 1e-4)

        # log_sigma: [n_peak, 2]
        sigma_prior_loc = np.asarray(model_inputs["sigma_prior_loc"], dtype=float)
        log_sigma = np.log(np.maximum(sigma_prior_loc, 1e-6))

        # alpha: [n_peak, 2]
        alpha_prior_loc = np.asarray(model_inputs["alpha_prior_loc"], dtype=float)
        alpha = np.where(np.isfinite(alpha_prior_loc), alpha_prior_loc, 0.0)

        # mu_center: [n_peak]
        mu_center_loc = np.asarray(model_inputs["mu_center_loc"], dtype=float)
        mu_center = np.clip(
            mu_center_loc, mu_lo + 1e-6 * logical_span, mu_hi - 1e-6 * logical_span
        )

        # mu_trace_offset: [n_trace, n_peak] — start at zero
        n_trace = int(jnp.asarray(model_inputs["x"]).shape[0])
        mu_trace_offset = np.zeros((n_trace, n_peak), dtype=float)

        # A_total: [n_trace, n_peak] — init at trapezoid-based area estimate (sum of components)
        # LogNormal prior has no hard bounds, so any positive value is valid.
        A_init = np.asarray(model_inputs["A_init"], dtype=float)  # [n_trace, n_peak, 2]
        A_total = np.maximum(np.sum(A_init, axis=-1), 1e-8)  # [n_trace, n_peak]

        # baseline_intercept: [n_trace]
        baseline_intercept = np.asarray(
            model_inputs["baseline_intercept_loc"], dtype=float
        )

        # baseline_slope: [n_trace]
        baseline_slope = np.asarray(model_inputs["baseline_slope_loc"], dtype=float)

        # sigma_y: [n_trace]
        sigma_y_prior = np.asarray(model_inputs["sigma_y_prior_loc"], dtype=float)
        sigma_y = np.where(
            np.isfinite(sigma_y_prior) & (sigma_y_prior > 0), sigma_y_prior, 1.0
        )

        init: dict[str, Any] = {
            "log_sigma": jnp.asarray(log_sigma, dtype=jnp.float32),
            "alpha": jnp.asarray(alpha, dtype=jnp.float32),
            "mu_center": jnp.asarray(mu_center, dtype=jnp.float32),
            "mu_trace_offset": jnp.asarray(mu_trace_offset, dtype=jnp.float32),
            "A_total": jnp.asarray(A_total, dtype=jnp.float32),
            "baseline_intercept": jnp.asarray(baseline_intercept, dtype=jnp.float32),
            "baseline_slope": jnp.asarray(baseline_slope, dtype=jnp.float32),
            "sigma_y": jnp.asarray(sigma_y, dtype=jnp.float32),
        }
        if n_shoulder_peak > 0:
            # separation: [n_shoulder_peak] — strictly inside (sep_low, sep_high)
            sep_low = np.asarray(model_inputs["separation_low"], dtype=float)
            sep_high = np.asarray(model_inputs["separation_high"], dtype=float)
            sep_low_sh = sep_low[shoulder_peak_index]
            sep_high_sh = sep_high[shoulder_peak_index]
            sep_span_sh = np.maximum(sep_high_sh - sep_low_sh, 1e-12)
            sep_eps = 1e-4 * sep_span_sh
            # Initialise near the lower bound rather than the midpoint.
            # Initialise at the data-driven sep_est (stored in model_inputs).
            # With tight [0.5×, 2.5×] bounds, sep_est = sep_low + 0.25 * span,
            # so this puts the chain immediately near the correct separation rather
            # than at the geometric lower bound.
            sep_est_all = np.asarray(
                model_inputs.get("separation_est", np.zeros(n_peak)), dtype=float
            )
            sep_est_sh = sep_est_all[shoulder_peak_index]
            sep_init = np.where(
                (sep_est_sh > sep_low_sh + sep_eps)
                & (sep_est_sh < sep_high_sh - sep_eps),
                sep_est_sh,
                sep_low_sh + 0.25 * sep_span_sh,
            )
            sep_init = np.clip(sep_init, sep_low_sh + sep_eps, sep_high_sh - sep_eps)
            init["separation"] = jnp.asarray(sep_init, dtype=jnp.float32)

            # area_split_shoulder: [n_trace, n_shoulder_peak]
            A_main = A_init[:, shoulder_peak_index, 0]
            A_total_sh = A_total[:, shoulder_peak_index]
            area_split_shoulder = np.clip(
                A_main / np.maximum(A_total_sh, 1e-8), 0.05, 0.95
            )
            init["area_split_shoulder"] = jnp.asarray(
                area_split_shoulder, dtype=jnp.float32
            )
        return init

    def print_prior_summary(self, *, use_aligned_time: bool = False) -> None:
        """Print a human-readable summary of all prior parameters to stdout."""
        if self.model_inputs is None:
            mi = self._build_model_inputs(use_aligned_time=use_aligned_time)
        else:
            mi = self.model_inputs

        n_trace = int(jnp.asarray(mi["x"]).shape[0])
        n_peak = int(jnp.asarray(mi["mu_lo"]).shape[0])

        mu_lo = np.asarray(mi["mu_lo"], dtype=float)
        mu_hi = np.asarray(mi["mu_hi"], dtype=float)
        mu_center_loc = np.asarray(mi["mu_center_loc"], dtype=float)
        mu_center_scale = np.asarray(mi["mu_center_scale"], dtype=float)
        sigma_prior_loc = np.asarray(mi["sigma_prior_loc"], dtype=float)
        sigma_prior_scale = np.asarray(mi["sigma_prior_scale"], dtype=float)
        alpha_prior_loc = np.asarray(mi["alpha_prior_loc"], dtype=float)
        alpha_prior_scale = np.asarray(mi["alpha_prior_scale"], dtype=float)
        area_total_loc = np.asarray(mi["area_total_loc"], dtype=float)
        area_total_scale = np.asarray(mi["area_total_scale"], dtype=float)
        sep_low = np.asarray(mi["separation_low"], dtype=float)
        sep_high = np.asarray(mi["separation_high"], dtype=float)
        shoulder_side = np.asarray(mi["shoulder_side"], dtype=int)
        bl_int_loc = np.asarray(mi["baseline_intercept_loc"], dtype=float)
        bl_int_scale = np.asarray(mi["baseline_intercept_scale"], dtype=float)
        bl_sl_loc = np.asarray(mi["baseline_slope_loc"], dtype=float)
        bl_sl_scale = np.asarray(mi["baseline_slope_scale"], dtype=float)
        sigma_y_loc = np.asarray(mi["sigma_y_prior_loc"], dtype=float)

        w = 100
        print(f"\n{'=' * w}")
        print(f"  Prior summary — {n_trace} traces, {n_peak} peaks")
        print(f"{'=' * w}")

        print("\n  Peak priors:")
        print(
            f"  {'#':>2}  {'window':>13}  {'mu_center':>18}  "
            f"{'sigma(main)':>16}  {'alpha(main)':>16}  {'area_total':>18}  shoulder"
        )
        for i in range(n_peak):
            side = {-1: "left", 0: "none", 1: "right"}.get(int(shoulder_side[i]), "?")
            print(
                f"  {i:>2}  [{mu_lo[i]:5.3f},{mu_hi[i]:5.3f}]"
                f"  {mu_center_loc[i]:7.4f} ±{mu_center_scale[i]:.4f}"
                f"  {sigma_prior_loc[i, 0]:.5f} ±{sigma_prior_scale[i, 0]:.5f}"
                f"  {alpha_prior_loc[i, 0]:+.4f} ±{alpha_prior_scale[i, 0]:.4f}"
                f"  {area_total_loc[i]:8.1f} ±{area_total_scale[i]:.1f}"
                f"  {side}"
            )
            if side != "none":
                print(
                    f"  {'':>2}  {'(shoulder)':>13}"
                    f"  {'':>18}"
                    f"  {sigma_prior_loc[i, 1]:.5f} ±{sigma_prior_scale[i, 1]:.5f}"
                    f"  {alpha_prior_loc[i, 1]:+.4f} ±{alpha_prior_scale[i, 1]:.4f}"
                    f"  sep=[{sep_low[i]:.4f}, {sep_high[i]:.4f}]"
                )

        print("\n  Baseline priors (per trace):")
        print(f"  {'#':>4}  {'intercept':>22}  {'slope':>22}  {'sigma_y_prior':>14}")
        for i in range(n_trace):
            print(
                f"  {i:>4}"
                f"  {bl_int_loc[i]:10.3f} ±{bl_int_scale[i]:.3f}"
                f"  {bl_sl_loc[i]:10.4f} ±{bl_sl_scale[i]:.6f}"
                f"  {sigma_y_loc[i]:12.3f}"
            )
        print(f"{'=' * w}\n")

    def fit(
        self,
        *,
        use_aligned_time: bool = True,
        num_warmup: int = 1000,
        num_samples: int = 500,
        num_chains: int = 8,
        seed: int = 42,
        progress_bar: bool = True,
    ) -> Fitter:
        model_inputs = self._build_model_inputs(
            use_aligned_time=use_aligned_time,
        )
        init_values = self._build_init_values_for_nuts(model_inputs)

        self.mcmc = MCMC(
            NUTS(peak_model),  # , init_strategy=init_to_value(values=init_values)),
            num_warmup=int(num_warmup),
            num_samples=int(num_samples),
            num_chains=int(num_chains),
            progress_bar=bool(progress_bar),
            chain_method="parallel" if int(num_chains) > 1 else "sequential",
        )
        # Strip metadata-only keys that are not numpyro model parameters
        _EXTRA_KEYS = {
            "component_to_logical_index",
            "separation_est",
            "mu_init_model",
            "sigma_init_model",
            "alpha_init_model",
            "area_total_scale",  # diagnostic metadata; model uses LogNormal CV=0.3 instead
        }
        model_kwargs = {k: v for k, v in model_inputs.items() if k not in _EXTRA_KEYS}
        self.mcmc.run(jax.random.PRNGKey(int(seed)), **model_kwargs)
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
        # mu_y is the full fitted signal (peaks + baseline) stored as deterministic
        mu_y = jnp.asarray(
            self.samples["mu_y"], dtype=jnp.float32
        )  # [n_draw, n_trace, n_time]
        return jnp.mean(mu_y, axis=0)

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

        if "alpha" not in self.samples:
            raise ValueError("Posterior samples do not contain `alpha`.")

        alpha_raw = np.asarray(
            self.samples["alpha"], dtype=float
        )  # [n_draw, n_peak, 2]
        if alpha_raw.ndim != 3:
            raise ValueError(
                f"Posterior alpha has unexpected rank {alpha_raw.ndim}; expected 3 (draw, peak, 2)."
            )
        n_draw_actual, n_peak, _ = alpha_raw.shape
        # Flatten [n_draw, n_peak, 2] → [n_draw, 2*n_peak] then broadcast to [n_draw, n_chrom, 2*n_peak]
        alpha_flat = alpha_raw.reshape(n_draw_actual, 2 * n_peak)  # [n_draw, 2*n_peak]
        return np.broadcast_to(
            alpha_flat[:, None, :], (n_draw_actual, n_chrom, 2 * n_peak)
        )

    _DEFAULT_TRACE_VARS: tuple[str, ...] = (
        "log_sigma",
        "alpha",
        "mu_center",
        "separation",
        "A_total",
        "area_split_shoulder",
        "baseline_intercept",
        "baseline_slope",
        "sigma_y",
    )

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
        if var_names is None:
            present = set(idata.posterior.data_vars)
            var_names = [v for v in self._DEFAULT_TRACE_VARS if v in present]
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

    def print_fwhm_diagnostics(self) -> None:
        """Print per-peak FWHM sep_est tier breakdown and apex annotation details.

        Call after :meth:`build_model_inputs` (which internally calls
        ``_build_component_initializers``).  For shoulder peaks the output shows
        which tier determined ``sep_est``, the intermediate values considered at
        each tier, and a per-trace apex/HWHM summary so you can see whether
        bimodal apex detection fired and where the shoulder was placed.
        """
        if self._fwhm_diag is None:
            print("No FWHM diagnostics — call build_model_inputs() first.")
            return

        side_label = {0: "none", 1: "right", -1: "left"}
        tier_label = {0: "pass2", 1: "T1-bimodal", 2: "T2-HWHM", 3: "T3-geometric"}

        for d in self._fwhm_diag:
            name = d["peak_name"]
            side = d["side"]
            print(
                f"\n{'=' * 60}\n"
                f"Peak '{name}'  window=[{d['low']:.4f}, {d['high']:.4f}]  "
                f"shoulder={side_label.get(side, '?')}"
            )
            print(
                f"  FWHM fit  T_main={d['T_main']:.4f}  "
                f"sigma={d['sigma_loc']:.5f}  alpha={d['alpha_loc']:+.3f}"
            )

            if side == 0:
                print("  (no shoulder — sep_est not computed)")
                continue

            print(f"  sep_est = {d['sep_est']:.5f}  [{tier_label[d['tier_used']]}]")

            # Pass-2 summary (always shown for shoulder peaks)
            p2_n = d.get("pass2_n_valid", 0)
            p2_sep = d.get("pass2_sep_est")
            p2_as = d.get("pass2_area_split")
            p2_sep_str = f"{p2_sep:.5f}" if p2_sep is not None else "n/a"
            p2_as_str = f"{p2_as:.3f}" if p2_as is not None else "n/a"
            print(
                f"  Pass 2     n_valid={p2_n}  sep_est={p2_sep_str}"
                f"  area_split={p2_as_str}"
                + ("  ← USED" if d["tier_used"] == 0 else "  (fallback to tier)")
            )
            # Per-trace pass-2 breakdown (shows which traces contributed and their SNRs)
            p2_traces = d.get("pass2_trace_indices", [])
            p2_apexes = d.get("pass2_sh_apex_times", [])
            p2_snrs = d.get("pass2_snr_list", [])
            if p2_traces:
                print(
                    "  Pass 2 per-trace contributions (SNR = shoulder/main-at-T_main):"
                )
                for _t, _apex, _snr in zip(p2_traces, p2_apexes, p2_snrs):
                    print(f"    trace {_t:2d}  apex={_apex:.4f}  snr={_snr:.3f}")
            elif p2_n == 0:
                print(
                    "  Pass 2: no traces passed SNR filter (min_shoulder_fraction=0.04)"
                )

            # Tier 1
            n_in = int(np.sum(d["gate_keep_col"] & d["width_valid"]))
            n_out = d["t1_n_cand"]
            print(
                f"\n  Tier 1 – bimodal apex:"
                f"  in-gate={n_in}  out-of-gate shoulder-side={n_out}"
            )
            if n_out >= 2 and d["t1_sh_apexes"].size > 0:
                sh = d["t1_sh_apexes"]
                print(
                    f"    shoulder apexes: min={sh.min():.4f}  "
                    f"med={np.median(sh):.4f}  max={sh.max():.4f}"
                )
                print(f"    raw_sep = {d['t1_raw_sep']:.5f}", end="")
                if d["t1_raw_sep"] > 0.003 * (d["high"] - d["low"]):
                    print("  → FIRED")
                else:
                    print("  → too small (<0.003×span), skipped")
            else:
                print("    <2 shoulder candidates, skipped")

            # Tier 2
            print(
                f"\n  Tier 2 – HWHM excess (opposite-side σ ref):"
                f"  σ_ref={d['t2_sigma_ref']:.5f}  "
                f"gauss_hwhm={d['t2_gauss_hwhm']:.5f}"
            )
            if d["t2_median_hwhm"] > 0:
                print(
                    f"    median hwhm_side={d['t2_median_hwhm']:.5f}  "
                    f"excess={d['t2_excess']:.5f}",
                    end="",
                )
                if d["t2_excess"] > 1e-6:
                    print("  → FIRED")
                else:
                    print("  → excess ≤ 0, skipped")
            else:
                print("    no valid hwhm_side samples")

            # Tier 3
            print(
                f"\n  Tier 3 – geometric fallback:"
                f"  geo={d['t3_geo']:.5f}  2σ={2 * d['t3_sigma_fb']:.5f}",
                end="",
            )
            if d["tier_used"] == 3:
                print("  → FIRED")
            else:
                print("  → (not reached)")

            # Per-trace apex summary
            mt = d["mode_trace"]
            gk = d["gate_keep_col"]
            wv = d["width_valid"]
            in_gate = gk & wv
            out_sh = (~gk) & wv
            print(
                f"\n  Apex annotation across {len(mt)} traces:"
                f"  in-gate={int(np.sum(in_gate))}"
                f"  out-of-gate-valid={int(np.sum(out_sh))}"
            )
            if int(np.sum(in_gate)) > 0:
                mg = mt[in_gate]
                print(
                    f"    in-gate apex:  med={np.nanmedian(mg):.4f}  "
                    f"range=[{np.nanmin(mg):.4f}, {np.nanmax(mg):.4f}]"
                )
            if int(np.sum(out_sh)) > 0:
                mo = mt[out_sh]
                mo = mo[np.isfinite(mo)]
                if mo.size > 0:
                    print(
                        f"    out-of-gate apex: med={np.nanmedian(mo):.4f}  "
                        f"range=[{np.nanmin(mo):.4f}, {np.nanmax(mo):.4f}]"
                    )

            # HWHM asymmetry summary
            wl = d["w_left"][in_gate]
            wr = d["w_right"][in_gate]
            wl = wl[np.isfinite(wl) & (wl > 0)]
            wr = wr[np.isfinite(wr) & (wr > 0)]
            if wl.size > 0 and wr.size > 0:
                print(
                    f"    HWHM left={np.median(wl):.5f}  "
                    f"right={np.median(wr):.5f}  "
                    f"ratio L/R={np.median(wl) / np.median(wr):.3f}"
                )
        print()

    def plot_fwhm_apex_diagnostics(
        self,
        *,
        save_path: str = "nu_bayes_fwhm_apex_diagnostics.png",
        dpi: int = 150,
        use_aligned_time: bool = True,
    ) -> str:
        """Per-peak diagnostic: apex scatter, HWHM asymmetry, and FWHM residual.

        For every peak with a shoulder annotation this produces three rows:

        * **Apex scatter** — trace index vs detected apex time, coloured by
          gate membership (green = in-gate, red = out-of-gate valid,
          grey = invalid).  Vertical lines show T_main and the estimated
          shoulder position.
        * **HWHM left/right** — per-trace left (blue) and right (orange) HWHM
          with the Tier-2 reference ``gauss_hwhm`` overlaid.
        * **FWHM residual** — signal minus the main-peak FWHM reconstruction
          for each trace shown as light curves; the median residual is bold.
          A vertical line marks the shoulder position so you can judge whether
          residual structure lines up.

        Call after :meth:`build_model_inputs`.
        """
        if self._fwhm_diag is None:
            raise RuntimeError("Call build_model_inputs() first.")

        import matplotlib.cm as cm
        import matplotlib.pyplot as plt

        shoulder_peaks = [d for d in self._fwhm_diag if d["side"] != 0]
        if not shoulder_peaks:
            print("No shoulder peaks found — nothing to plot.")
            return save_path

        time_axis = np.asarray(
            self._time_axis(use_aligned_time=use_aligned_time), dtype=float
        )
        signal_np = np.asarray(
            self.baseline_corrected_signal(use_aligned_time=use_aligned_time),
            dtype=float,
        )
        n_trace = signal_np.shape[0]
        n_sh = len(shoulder_peaks)

        fig, axes = plt.subplots(3, n_sh, figsize=(5 * n_sh, 10), squeeze=False)
        fig.suptitle("FWHM Apex Diagnostics", fontsize=12, fontweight="bold")

        cmap = cm.get_cmap("coolwarm").resampled(n_trace)

        for col, d in enumerate(shoulder_peaks):
            name = d["peak_name"]
            side = d["side"]
            low, high = d["low"], d["high"]
            T_main = d["T_main"]
            sep_est = d["sep_est"]
            direction = 1.0 if side > 0 else -1.0
            sh_pos = T_main + direction * sep_est
            tier_lbl = {0: "P2", 1: "T1", 2: "T2", 3: "T3"}[d["tier_used"]]

            mt = d["mode_trace"]
            gk = d["gate_keep_col"]
            wv = d["width_valid"]
            wl = d["w_left"]
            wr = d["w_right"]

            # ── Row 0: apex scatter ─────────────────────────────────────────
            ax0 = axes[0, col]
            trace_idx = np.arange(n_trace)
            for t in trace_idx:
                if not wv[t] or not np.isfinite(mt[t]):
                    color, marker, zorder = "lightgrey", "x", 1
                elif gk[t]:
                    color, marker, zorder = "tab:green", "o", 3
                else:
                    color, marker, zorder = "tab:red", "^", 2
                ax0.scatter(
                    mt[t],
                    t,
                    color=color,
                    marker=marker,
                    s=30,
                    zorder=zorder,
                    linewidths=0.5,
                )
            ax0.axvline(
                T_main, color="green", lw=1.5, ls="-", label=f"T_main={T_main:.4f}"
            )
            ax0.axvline(
                sh_pos,
                color="red",
                lw=1.5,
                ls="--",
                label=f"sh_pos={sh_pos:.4f} [{tier_lbl}]",
            )
            if d["t1_sh_apexes"].size > 0:
                ax0.axvline(
                    float(np.median(d["t1_sh_apexes"])),
                    color="orange",
                    lw=1,
                    ls=":",
                    label=f"T1 sh_med={np.median(d['t1_sh_apexes']):.4f}",
                )
            ax0.set_xlim(low, high)
            ax0.set_xlabel("Apex time [min]")
            ax0.set_ylabel("Trace index")
            ax0.set_title(
                f"Peak '{name}' — apex scatter\nsep_est={sep_est:.4f} via {tier_lbl}"
            )
            ax0.legend(fontsize=7, loc="upper right")

            # ── Row 1: HWHM left/right ─────────────────────────────────────
            ax1 = axes[1, col]
            for t in trace_idx:
                if not wv[t]:
                    continue
                c = cmap(t / max(n_trace - 1, 1))
                mk = "o" if gk[t] else "^"
                ax1.scatter(
                    t,
                    wl[t] if np.isfinite(wl[t]) else np.nan,
                    color="tab:blue",
                    marker=mk,
                    s=20,
                    alpha=0.6,
                )
                ax1.scatter(
                    t,
                    wr[t] if np.isfinite(wr[t]) else np.nan,
                    color="tab:orange",
                    marker=mk,
                    s=20,
                    alpha=0.6,
                )
            if d["t2_gauss_hwhm"] > 0:
                ax1.axhline(
                    d["t2_gauss_hwhm"],
                    color="black",
                    lw=1,
                    ls="--",
                    label=f"T2 ref hwhm={d['t2_gauss_hwhm']:.4f}",
                )
            # dummy handles for legend
            ax1.scatter([], [], color="tab:blue", s=20, label="HWHM left")
            ax1.scatter([], [], color="tab:orange", s=20, label="HWHM right")
            ax1.scatter([], [], color="grey", marker="^", s=20, label="out-of-gate")
            ax1.set_xlabel("Trace index")
            ax1.set_ylabel("HWHM [min]")
            ax1.set_title(
                f"HWHM asymmetry\n"
                f"T2: excess={d['t2_excess']:.5f}  "
                f"σ_ref={d['t2_sigma_ref']:.5f}"
            )
            ax1.legend(fontsize=7)

            # ── Row 2: FWHM residual ───────────────────────────────────────
            ax2 = axes[2, col]
            # time_axis may be 2D [n_trace, n_time] after alignment; use first
            # trace as 1D reference for the window mask (shifts are small).
            t_ref = time_axis[0] if time_axis.ndim == 2 else time_axis
            window_mask = (t_ref >= low) & (t_ref <= high)
            t_win = t_ref[window_mask]
            sig_win = signal_np[:, window_mask]
            residuals_arr = self._compute_main_residuals_for_window(d, t_win, sig_win)
            residuals = []
            for t in trace_idx:
                resid = residuals_arr[t]
                residuals.append(resid)
                if np.all(np.isfinite(resid)):
                    c = cmap(t / max(n_trace - 1, 1))
                    ax2.plot(t_win, resid, color=c, alpha=0.3, lw=0.8)

            resid_stack = np.array(residuals)
            valid_rows = np.all(np.isfinite(resid_stack), axis=1)
            if np.any(valid_rows):
                med_resid = np.nanmedian(resid_stack[valid_rows], axis=0)
                ax2.plot(t_win, med_resid, color="black", lw=2, label="median residual")
            ax2.axvline(
                sh_pos, color="red", lw=1.5, ls="--", label=f"sh_pos={sh_pos:.4f}"
            )
            ax2.axhline(0, color="grey", lw=0.8, ls=":")
            ax2.set_xlim(low, high)
            ax2.set_xlabel("Time [min]")
            ax2.set_ylabel("Signal residual")
            ax2.set_title(
                "FWHM residual (signal − main fit)\nshould show shoulder structure"
            )
            ax2.legend(fontsize=7)

        fig.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_fwhm_prior_summary(
        self,
        *,
        use_aligned_time: bool = True,
        half_level: float = 0.5,
        apex_gate_n_mad: float = 2.0,
        save_path: str = "nu_bayes_fwhm_prior_summary.png",
        dpi: int = 150,
    ) -> str:
        """Plot FWHM-derived prior features across all traces per peak.

        Shows three panels per peak:
        - FWHM scatter (kept vs rejected by apex gate)
        - Left/right HWHM asymmetry (proxy for alpha prior)
        - Inferred sigma scatter (from HWHM) with robust prior median overlay

        This diagnostic links ``compute_peak_fwhm`` outputs to the prior
        parameters printed by :meth:`print_prior_summary`.
        """
        if len(self.peaks) == 0:
            raise ValueError("No peaks are defined.")

        payload = self.compute_peak_fwhm(
            use_aligned_time=use_aligned_time,
            half_level=half_level,
            apply_apex_gate=True,
            apex_gate_n_mad=apex_gate_n_mad,
        )
        apex_time_all = np.asarray(payload["apex_time_all"], dtype=float)
        left_time_all = np.asarray(payload["left_time_all"], dtype=float)
        right_time_all = np.asarray(payload["right_time_all"], dtype=float)
        fwhm_all = np.asarray(payload["fwhm_all"], dtype=float)
        valid_trace = np.asarray(payload["valid_trace"], dtype=bool)
        gate_keep = np.asarray(payload["gate_keep"], dtype=bool)

        n_trace, n_peak = apex_time_all.shape
        gaussian_hwhm_factor = float(np.sqrt(2.0 * np.log(2.0)))

        # Derive sigma/alpha from HWHM for all traces.
        w_left_all = apex_time_all - left_time_all  # [n_trace, n_peak]
        w_right_all = right_time_all - apex_time_all  # [n_trace, n_peak]
        valid_hwhm = (
            valid_trace
            & np.isfinite(w_left_all)
            & (w_left_all > 1e-8)
            & np.isfinite(w_right_all)
            & (w_right_all > 1e-8)
        )
        sigma_from_fwhm = np.where(
            valid_hwhm,
            np.sqrt(
                0.5
                * (
                    (np.where(valid_hwhm, w_left_all, 1.0) / gaussian_hwhm_factor) ** 2
                    + (np.where(valid_hwhm, w_right_all, 1.0) / gaussian_hwhm_factor)
                    ** 2
                )
            ),
            np.nan,
        )
        # Asymmetry ratio: (right-left)/(right+left); positive → right-skewed → alpha>0
        denom = np.where(valid_hwhm, w_left_all + w_right_all, np.nan)
        asymmetry = np.where(
            valid_hwhm, (w_right_all - w_left_all) / np.maximum(denom, 1e-12), np.nan
        )

        # Build per-peak prior estimates for overlay.
        sorted_peaks = self._sorted_peaks()
        mi = self.model_inputs
        has_priors = mi is not None and "sigma_prior_loc" in mi
        if has_priors:
            sigma_prior_loc_np = np.asarray(
                mi["sigma_prior_loc"], dtype=float
            )  # [n_peak, 2]
            shoulder_side_np = np.asarray(mi["shoulder_side"], dtype=int)
        else:
            sigma_prior_loc_np = np.full((n_peak, 2), np.nan)
            shoulder_side_np = np.zeros(n_peak, dtype=int)

        trace_indices = np.arange(n_trace)
        n_cols = n_peak
        n_rows = 3
        figure, axes = plt.subplots(
            n_rows,
            n_cols,
            squeeze=False,
            figsize=(4.0 * n_cols, 2.6 * n_rows),
            constrained_layout=True,
        )

        for peak_idx in range(n_peak):
            keep = gate_keep[:, peak_idx] & valid_hwhm[:, peak_idx]
            reject = valid_hwhm[:, peak_idx] & ~keep

            ax_fwhm = axes[0, peak_idx]
            ax_asym = axes[1, peak_idx]
            ax_sig = axes[2, peak_idx]

            # Row 0: FWHM scatter
            if np.any(keep):
                ax_fwhm.scatter(
                    trace_indices[keep] + 1,
                    fwhm_all[keep, peak_idx],
                    s=18,
                    color="tab:blue",
                    label="kept",
                    linewidths=0,
                )
            if np.any(reject):
                ax_fwhm.scatter(
                    trace_indices[reject] + 1,
                    fwhm_all[reject, peak_idx],
                    s=18,
                    color="0.55",
                    label="rejected",
                    alpha=0.5,
                    linewidths=0,
                )
            if np.any(keep):
                fwhm_median = float(np.nanmedian(fwhm_all[keep, peak_idx]))
                ax_fwhm.axhline(
                    fwhm_median,
                    color="tab:orange",
                    linestyle="--",
                    linewidth=1.0,
                    label=f"median={fwhm_median:.4f}",
                )
            ax_fwhm.set_title(f"Peak {peak_idx + 1}")
            ax_fwhm.set_ylabel("FWHM [min]" if peak_idx == 0 else "")
            ax_fwhm.grid(True, alpha=0.2)
            if peak_idx == 0:
                ax_fwhm.legend(fontsize=7, frameon=False)

            # Row 1: HWHM asymmetry
            if np.any(keep):
                ax_asym.scatter(
                    trace_indices[keep] + 1,
                    asymmetry[keep, peak_idx],
                    s=18,
                    color="tab:blue",
                    linewidths=0,
                )
            if np.any(reject):
                ax_asym.scatter(
                    trace_indices[reject] + 1,
                    asymmetry[reject, peak_idx],
                    s=18,
                    color="0.55",
                    alpha=0.5,
                    linewidths=0,
                )
            ax_asym.axhline(0, color="0.4", linestyle=":", linewidth=0.8)
            side_label = {-1: " (left shoulder)", 0: "", 1: " (right shoulder)"}.get(
                int(shoulder_side_np[peak_idx]), ""
            )
            ax_asym.set_ylabel(f"(R-L)/(R+L){side_label}" if peak_idx == 0 else "")
            ax_asym.grid(True, alpha=0.2)

            # Row 2: sigma from FWHM
            if np.any(keep):
                ax_sig.scatter(
                    trace_indices[keep] + 1,
                    sigma_from_fwhm[keep, peak_idx],
                    s=18,
                    color="tab:blue",
                    linewidths=0,
                )
            if np.any(reject):
                ax_sig.scatter(
                    trace_indices[reject] + 1,
                    sigma_from_fwhm[reject, peak_idx],
                    s=18,
                    color="0.55",
                    alpha=0.5,
                    linewidths=0,
                )
            if has_priors and np.isfinite(sigma_prior_loc_np[peak_idx, 0]):
                ax_sig.axhline(
                    sigma_prior_loc_np[peak_idx, 0],
                    color="tab:orange",
                    linestyle="--",
                    linewidth=1.1,
                    label=f"prior σ_main={sigma_prior_loc_np[peak_idx, 0]:.4f}",
                )
            if (
                has_priors
                and int(shoulder_side_np[peak_idx]) != 0
                and np.isfinite(sigma_prior_loc_np[peak_idx, 1])
            ):
                ax_sig.axhline(
                    sigma_prior_loc_np[peak_idx, 1],
                    color="tab:red",
                    linestyle=":",
                    linewidth=1.1,
                    label=f"prior σ_sh={sigma_prior_loc_np[peak_idx, 1]:.4f}",
                )
            ax_sig.set_ylabel("σ from FWHM [min]" if peak_idx == 0 else "")
            ax_sig.set_xlabel("Trace index")
            ax_sig.grid(True, alpha=0.2)
            if peak_idx == 0:
                ax_sig.legend(fontsize=7, frameon=False)

        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return save_path

    def plot_prior_peak_fits(
        self,
        *,
        use_aligned_time: bool = True,
        save_path: str = "nu_bayes_prior_peak_fits.png",
        column_mode: str = "peak",
        chromatogram_indices: list[int] | None = None,
        peak_indices: list[int] | None = None,
        data_alpha: float = 0.4,
        data_size: float = 8.0,
        line_width: float = 1.5,
        dpi: int = 150,
    ) -> str:
        """Plot FWHM-derived initialisation curves overlaid on observed data.

        Same layout as :meth:`plot_posterior_peak_fits` but shows per-trace
        point-estimate skew-normal curves derived from the FWHM analysis.
        No HDI bands are drawn since these are point estimates, not posterior draws.

        Green dashed = main component, red dashed = shoulder component,
        blue solid = total (all peaks + baseline), orange dashed = baseline.
        Can be called before :meth:`fit`; will build model inputs if needed.
        """
        if self.model_inputs is None:
            self._build_model_inputs(use_aligned_time=use_aligned_time)
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
            raise ValueError("No peak masks available for prior plotting.")

        n_chrom = int(time.shape[0])
        n_peak = int(peak_masks.shape[0])
        chrom_sel = (
            list(range(n_chrom))
            if chromatogram_indices is None
            else [int(i) for i in chromatogram_indices]
        )
        peak_sel = (
            list(range(n_peak))
            if peak_indices is None
            else [int(i) for i in peak_indices]
        )
        peak_sel_array = np.asarray(peak_sel, dtype=int)
        include_full_window_column = column_mode == "peak" and len(peak_sel) > 1

        # [n_chrom, n_peak, 2] arrays: axis -1 index 0 = main, 1 = shoulder
        mu_init_np = np.asarray(self.model_inputs["mu_init_model"], dtype=float)
        sigma_init_np = np.asarray(self.model_inputs["sigma_init_model"], dtype=float)
        alpha_init_np = np.asarray(self.model_inputs["alpha_init_model"], dtype=float)
        A_init_np = np.asarray(self.model_inputs["A_init"], dtype=float)
        baseline_intercept = np.asarray(
            self.model_inputs["baseline_intercept_loc"], dtype=float
        )  # [n_chrom]
        baseline_slope = np.asarray(
            self.model_inputs["baseline_slope_loc"], dtype=float
        )  # [n_chrom]

        if column_mode == "chromatogram":
            n_rows = len(peak_sel)
            n_cols = len(chrom_sel)
            row_labels = [f"Peak {pi + 1}" for pi in peak_sel]
            col_labels = [f"Trace {ci + 1}" for ci in chrom_sel]
        else:
            n_rows = len(chrom_sel)
            n_cols = len(peak_sel) + (1 if include_full_window_column else 0)
            row_labels = [f"Trace {ci + 1}" for ci in chrom_sel]
            col_labels = [f"Peak {pi + 1}" for pi in peak_sel]
            if include_full_window_column:
                col_labels.append("All peaks")

        figure, axes = plt.subplots(
            n_rows,
            n_cols,
            squeeze=False,
            figsize=(3.6 * n_cols, 2.6 * n_rows),
            constrained_layout=True,
        )

        _comp_colors = ["tab:green", "tab:red"]
        _comp_labels = ["main", "shoulder"]

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
                    peaks_for_cell = [peak_index]
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
                            x_window = time[chrom_index, window_points]
                            x_low = float(np.nanmin(x_window))
                            x_high = float(np.nanmax(x_window))
                            active = (
                                finite_mask
                                & (time[chrom_index] >= x_low)
                                & (time[chrom_index] <= x_high)
                            )
                        else:
                            active = window_points
                        peaks_for_cell = list(peak_sel)
                    else:
                        peak_index = peak_sel[col_index]
                        active = peak_masks[peak_index, chrom_index] & finite_mask
                        peaks_for_cell = [peak_index]

                if int(np.sum(active)) < 3:
                    ax.text(
                        0.5,
                        0.5,
                        "insufficient data",
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

                # Gather [n_comp_total, n_time] across all peaks for this cell.
                # Each peak contributes 2 components (main + shoulder).
                mu_c = np.concatenate(
                    [mu_init_np[chrom_index, pi, :] for pi in peaks_for_cell]
                )
                sigma_c = np.concatenate(
                    [sigma_init_np[chrom_index, pi, :] for pi in peaks_for_cell]
                )
                alpha_c = np.concatenate(
                    [alpha_init_np[chrom_index, pi, :] for pi in peaks_for_cell]
                )
                A_c = np.concatenate(
                    [A_init_np[chrom_index, pi, :] for pi in peaks_for_cell]
                )

                pdf = np.asarray(
                    skew_normal_pdf(
                        jnp.asarray(x_active, dtype=jnp.float32),
                        jnp.asarray(mu_c, dtype=jnp.float32),
                        jnp.asarray(sigma_c, dtype=jnp.float32),
                        jnp.asarray(alpha_c, dtype=jnp.float32),
                    ),
                    dtype=float,
                )  # [n_comp, n_time]

                component_curves = A_c[:, None] * pdf  # [n_comp, n_time]
                peak_total = np.sum(component_curves, axis=0)
                bl = (
                    baseline_intercept[chrom_index]
                    + baseline_slope[chrom_index] * x_active
                )
                total_curve = peak_total + bl

                # Observed data
                ax.scatter(
                    x_active,
                    y_active,
                    s=data_size,
                    alpha=data_alpha,
                    color="0.35",
                    linewidths=0,
                )

                # Per-component init curves
                for _ci in range(component_curves.shape[0]):
                    _color = _comp_colors[min(_ci, len(_comp_colors) - 1)]
                    _label = _comp_labels[min(_ci, len(_comp_labels) - 1)]
                    ax.plot(
                        x_active,
                        component_curves[_ci],
                        color=_color,
                        linestyle="--",
                        linewidth=max(0.9, 0.8 * line_width),
                        alpha=0.85,
                        label=_label,
                    )

                # Total (peaks + baseline)
                ax.plot(
                    x_active,
                    total_curve,
                    color="tab:blue",
                    linewidth=line_width,
                    label="total",
                )

                # Baseline
                ax.plot(
                    x_active,
                    bl,
                    color="tab:orange",
                    linestyle="--",
                    linewidth=max(1.0, 0.9 * line_width),
                    label="baseline",
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
        baseline_intercept_draws = np.asarray(
            self.samples["baseline_intercept"], dtype=float
        )  # [n_draw, n_chrom]
        baseline_slope_draws = np.asarray(
            self.samples["baseline_slope"], dtype=float
        )  # [n_draw, n_chrom]

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
                # per-component peak draws (no baseline): [n_draw, n_component, n_time]
                component_draws = A_draw[:, :, None] * pdf
                peak_draws = np.sum(component_draws, axis=1)
                # Baseline curve: [n_draw, n_time_active]
                baseline_active = (
                    baseline_intercept_draws[:, chrom_index, None]
                    + baseline_slope_draws[:, chrom_index, None] * x_active[None, :]
                )
                total_draws = peak_draws + baseline_active
                y_median = np.nanmedian(total_draws, axis=0)
                y_low = np.nanquantile(total_draws, 0.025, axis=0)
                y_high = np.nanquantile(total_draws, 0.975, axis=0)
                baseline_line = np.nanmedian(baseline_active, axis=0)

                # Observed data
                ax.scatter(
                    x_active,
                    y_active,
                    s=data_size,
                    alpha=data_alpha,
                    color="0.35",
                    linewidths=0,
                )

                # --- Per-component raw peak curves (peak signal only, no baseline) ---
                # Plotted UNDER the total so the shoulder is separately visible.
                _comp_colors = ["tab:green", "tab:red"]
                _comp_styles = ["--", "--"]
                _comp_labels = ["main", "shoulder"]
                n_drawn_components = int(component_draws.shape[1])
                for _ci in range(n_drawn_components):
                    _raw = component_draws[:, _ci, :]  # [n_draw, n_time], peak only
                    _median_raw = np.nanmedian(_raw, axis=0)
                    _lo_raw = np.nanquantile(_raw, 0.025, axis=0)
                    _hi_raw = np.nanquantile(_raw, 0.975, axis=0)
                    _color = _comp_colors[min(_ci, len(_comp_colors) - 1)]
                    _ls = _comp_styles[min(_ci, len(_comp_styles) - 1)]
                    _label = _comp_labels[min(_ci, len(_comp_labels) - 1)]
                    ax.plot(
                        x_active,
                        _median_raw,
                        color=_color,
                        linestyle=_ls,
                        linewidth=max(0.9, 0.8 * line_width),
                        alpha=0.85,
                        label=_label,
                    )
                    ax.fill_between(
                        x_active,
                        _lo_raw,
                        _hi_raw,
                        color=_color,
                        alpha=max(0.08, 0.5 * hdi_alpha),
                    )

                # --- Total fit (all peaks + baseline) ---
                ax.plot(
                    x_active,
                    y_median,
                    color="tab:blue",
                    linewidth=line_width,
                    label="total",
                )
                ax.fill_between(
                    x_active,
                    y_low,
                    y_high,
                    color="tab:blue",
                    alpha=hdi_alpha,
                )

                # --- Baseline ---
                ax.plot(
                    x_active,
                    baseline_line,
                    color="tab:orange",
                    linestyle="--",
                    linewidth=max(1.0, 0.9 * line_width),
                    label="baseline",
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
        baseline_intercept_draws = np.asarray(
            self.samples["baseline_intercept"], dtype=float
        )  # [n_draw, n_chrom]
        baseline_slope_draws = np.asarray(
            self.samples["baseline_slope"], dtype=float
        )  # [n_draw, n_chrom]

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
                # Baseline curve: [n_draw, n_time_active]
                baseline_active = (
                    baseline_intercept_draws[:, chrom_index, None]
                    + baseline_slope_draws[:, chrom_index, None] * x_active[None, :]
                )
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
        self._baseline_priors_cache.clear()
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
        baseline_priors = self.get_baseline_priors(use_aligned_time=use_aligned_time)
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
                    baseline_priors.slope[i] * time_i + baseline_priors.intercept[i]
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
        save_path: str = "baseline_anchor_fit.png",
        dpi: int = 150,
    ) -> str:
        """Plot OLS baseline fit per chromatogram (one subplot per trace)."""
        from .baseline import (
            _DEFAULT_EDGE_FRACTION,
            _DEFAULT_PERCENTILE,
            _select_anchors,
        )

        time_axis = self._time_axis(use_aligned_time=use_aligned_time)
        time_np = np.asarray(time_axis, dtype=float)
        signal_np = np.asarray(self.signal, dtype=float)
        priors = self.get_baseline_priors(use_aligned_time=use_aligned_time)
        intercepts = np.asarray(priors.intercept, dtype=float)
        slopes = np.asarray(priors.slope, dtype=float)

        anchor_mask = np.asarray(
            _select_anchors(
                time_axis,
                self.signal,
                peaks=self.peaks,
                baselines=self.baselines,
                edge_fraction=_DEFAULT_EDGE_FRACTION,
                percentile=_DEFAULT_PERCENTILE,
            ),
            dtype=bool,
        )

        n_trace = int(signal_np.shape[0])
        fig, axes = plt.subplots(
            n_trace,
            1,
            sharex=True,
            squeeze=False,
            figsize=(10, 2.5 * n_trace),
            constrained_layout=True,
        )
        cmap = plt.cm.viridis
        colors = [cmap(i / max(n_trace - 1, 1)) for i in range(n_trace)]

        for ti, ax in enumerate(axes[:, 0]):
            t = time_np[ti]
            s = signal_np[ti]
            finite = np.isfinite(t) & np.isfinite(s)
            if not np.any(finite):
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            color = colors[ti]
            # Raw data (light)
            ax.plot(t[finite], s[finite], color=color, alpha=0.3, linewidth=0.8)
            # Anchor points used for OLS fit
            anchors = finite & anchor_mask[ti]
            if np.any(anchors):
                ax.scatter(
                    t[anchors],
                    s[anchors],
                    s=10,
                    color=color,
                    alpha=0.9,
                    linewidths=0,
                    zorder=3,
                    label="anchors" if ti == 0 else None,
                )
            # OLS baseline line
            t_range = t[finite]
            y_line = intercepts[ti] + slopes[ti] * t_range
            order = np.argsort(t_range)
            ax.plot(
                t_range[order],
                y_line[order],
                color="tab:red",
                linewidth=1.4,
                label="OLS baseline" if ti == 0 else None,
            )

            ax.set_ylabel(f"Trace {ti + 1}", fontsize=8)
            ax.grid(True, alpha=0.2)
            if ti == 0:
                ax.legend(loc="best", fontsize=7, frameon=False)

        axes[-1, 0].set_xlabel("Time [min]")
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path


if __name__ == "__main__":
    from rich import print

    from .data import (
        BaselineAnnotation,
        PeakAnnotation,
    )

    arr = jnp.load("/Users/max/code/sahh-kinetics-hplc/chromatograms.npy").reshape(
        -1, 3000
    )[:10, :1000]
    time = jnp.load("/Users/max/code/sahh-kinetics-hplc/times.npy").reshape(-1, 3000)[
        :10, :1000
    ]
    sample_names = jnp.load("/Users/max/code/sahh-kinetics-hplc/folder_names.npy")
    chromatogram_names = jnp.load("/Users/max/code/sahh-kinetics-hplc/sample_names.npy")

    baselines = [BaselineAnnotation(low=0, high=1), BaselineAnnotation(low=4, high=6)]

    peaks = [
        PeakAnnotation(name="peak1", low=2.6, high=2.83),
        PeakAnnotation(name="peak2", low=2.9, high=3.18, shoulder="right"),
        # PeakAnnotation(name="peak3", low=3.18, high=3.45, shoulder=None),
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
        save_path="baseline_anchor_fit.png",
    )
    print(f"Saved baseline anchor diagnostics: {baseline_diag_path}")

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
    fitter.print_prior_summary()
    fitter.print_fwhm_diagnostics()

    apex_diag_path = fitter.plot_fwhm_apex_diagnostics(
        save_path="nu_bayes_fwhm_apex_diagnostics.png",
        use_aligned_time=True,
    )
    print(f"Saved FWHM apex diagnostics: {apex_diag_path}")

    prior_fit_path = fitter.plot_prior_peak_fits(
        use_aligned_time=True,
        save_path="nu_bayes_prior_peak_fits.png",
        column_mode="peak",
    )
    print(f"Saved prior peak fits: {prior_fit_path}")
    assert False
    print("Fitting...")
    fitter.fit()
    summary_path = fitter.save_arviz_summary_txt("nu_bayes_arviz_summary.txt")
    print(f"Saved ArviZ summary: {summary_path}")
    trace_path = fitter.plot_arviz_trace(
        save_path="nu_bayes_trace.png",
        var_names=[
            "A",
            "mu_center",
            "mu_trace_offset",
            "sigma",
            "alpha",
            "area_split_shoulder",
            "separation",
            "baseline_intercept",
            "baseline_slope",
            "sigma_y",
        ],
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
        kind="kde",
        var_names=[
            "mu",
            "separation",
            "alpha",
            "A_total",
            "area_split_shoulder",
            "sigma_y",
        ],
        max_subplots=10000,
    )
    print(f"Saved ArviZ pair plot: {pair_path}")
