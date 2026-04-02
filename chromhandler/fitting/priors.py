"""Window-geometry-based Bayesian priors for chromatographic peak fitting.

All priors are derived from directly observable FWHM half-width geometry:

- **Apex-height-weighted centroid**   → apex_loc, apex_scale
- **Left/right HWHM geometry**       → w_left_loc/scale, w_right_loc/scale
- **Gaussian area from height*sigma** → area_gaussian_pt
- **Trapezoid window integration**    → area_trapz_pt
- **Residual integration**            → area_art_shared (artefact peaks only)
- **Signal-to-noise ratio**           → snr_per_trace (for adaptive area prior width)

Pipeline
--------
1. ``_median_dt``              — robust median sampling interval in a window.
2. ``_height_weighted_apex``   — height-weighted apex centroid across traces.
3. ``_halfwidth_priors``       — height-weighted left/right half-width priors.
4. ``_window_area``            — per-trace baseline-subtracted trapezoid areas.
5. ``build_peak_priors``       — single-pass assembly of all priors per peak.
6. ``geometric_priors_to_arrays`` — convert list of priors to model-ready numpy arrays.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Final

import jax
import jax.numpy as jnp
import numpy as np

from .types import (
    PeakMode,
    peak_component_count,
    peak_is_artefact_mode,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from chromhandler.annotations import PeakAnnotation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FLOAT_MIN: Final = 1e-12

_GAUSSIAN_HWHM_FACTOR: Final = math.sqrt(2.0 * math.log(2.0))
_GAUSSIAN_FWHM_FACTOR: Final = 2.0 * _GAUSSIAN_HWHM_FACTOR
_GAUSSIAN_AREA_FROM_HEIGHT_SIGMA: Final = math.sqrt(2.0 * math.pi)

# Minimum height fraction for apex outlier rejection (fraction of max apex height)
_MIN_APEX_HEIGHT_FRAC: Final = 0.0025
_SINGLE_VALUE_SCALE_FRAC: Final = 0.05


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GeometricPeakPriors:
    """Window-geometry-based priors for one annotated peak window.

    Half-width priors ``w_left_*`` and ``w_right_*`` are the directly measured
    FWHM half-widths (left and right of the apex), aggregated across traces
    with height weighting.  The model samples ``log(w_left)`` and
    ``log(w_right)`` and converts to ``(sigma, alpha)`` deterministically.

    Attributes
    ----------
    apex_loc:
        Apex-height-weighted centroid of the peak across traces [time units].
    apex_scale:
        Height-weighted standard deviation of apex positions across traces.
        Reflects retention-time drift between injections, not peak width.
    w_left_loc:
        Height-weighted mean of the left half-width at half-maximum [time units].
    w_left_scale:
        Height-weighted spread of the left HWHM across traces.
    w_right_loc:
        Height-weighted mean of the right half-width at half-maximum [time units].
    w_right_scale:
        Height-weighted spread of the right HWHM across traces.
    area_gaussian_pt:
        Gaussian-approximation area per trace, shape ``[n_trace]``.
        ``A = apex_height * sigma * sqrt(2*pi)`` where sigma is derived from
        the measured half-widths.  For traces without valid FWHM, falls
        back to the cross-trace median (scaled down).
    area_trapz_pt:
        Total trapezoid-integrated window area per trace, shape ``[n_trace]``.
    area_art_shared:
        Median positive residual area across traces (``max(trapz - gaussian, 0)``).
        ``0.0`` unless ``mode == "artefact_doublet"``.
    snr_per_trace:
        Per-trace signal-to-noise ratio (apex height / noise estimate),
        shape ``[n_trace]``.  Used by the model for adaptive area prior width.
    window_lo:
        Lower bound of the peak window [time units].
    window_hi:
        Upper bound of the peak window [time units].
    n_valid_traces:
        Number of traces that contributed a valid FWHM measurement.
    """

    mode: PeakMode
    apex_loc: float
    apex_scale: float
    w_left_loc: float
    w_left_scale: float
    w_right_loc: float
    w_right_scale: float
    area_gaussian_pt: NDArray[np.float64]
    area_trapz_pt: NDArray[np.float64]
    area_art_shared: float
    snr_per_trace: NDArray[np.float64]
    window_lo: float
    window_hi: float
    n_valid_traces: int

    @property
    def n_components(self) -> int:
        return peak_component_count(self.mode)

    def __repr__(self) -> str:
        art_str = (
            f", art_area={self.area_art_shared:.2e}" if peak_is_artefact_mode(self.mode) else ""
        )
        return (
            f"GeometricPeakPriors("
            f"window=[{self.window_lo:.4f}, {self.window_hi:.4f}], "
            f"mode={self.mode}, "
            f"apex={self.apex_loc:.4f}±{self.apex_scale:.4f}, "
            f"w_left={self.w_left_loc:.5f}±{self.w_left_scale:.5f}, "
            f"w_right={self.w_right_loc:.5f}±{self.w_right_scale:.5f}, "
            f"ncomp={self.n_components}{art_str}, "
            f"n_valid={self.n_valid_traces})"
        )


@dataclasses.dataclass(frozen=True)
class PeakApexTraces:
    """Per-trace FWHM apex data used for trace-shift refinement.

    Attributes
    ----------
    fwhm_apex_trace:
        FWHM-detected apex time per trace and peak window, shape ``[n_trace, n_peak]``.
        ``NaN`` for traces where FWHM detection failed.
    fwhm_valid_trace:
        Boolean mask: ``True`` where a valid FWHM apex was detected,
        shape ``[n_trace, n_peak]``.
    """

    fwhm_apex_trace: NDArray[np.float64]
    fwhm_valid_trace: NDArray[np.bool_]


@dataclasses.dataclass(frozen=True)
class _TraceFwhmGeometry:
    """Per-trace apex/FWHM geometry for one peak window."""

    apex_time: jax.Array
    apex_height: jax.Array
    w_left: jax.Array
    w_right: jax.Array
    height_valid: jax.Array
    fwhm_valid: jax.Array


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _median_dt(x_win: NDArray[np.float64]) -> float:
    """Median sampling interval within a window slice.

    Uses ``np.diff`` on the sorted, finite window points.  Robust to a small
    number of irregular gaps (e.g. from NaN padding at window boundaries).

    Args:
        x_win: 1-D array of time values inside the window.

    Returns:
        Median consecutive spacing [time units].  Falls back to the full
        range divided by the number of points when fewer than 2 points exist.
    """
    x_finite = x_win[np.isfinite(x_win)]
    if x_finite.size < 2:
        span = float(x_finite[-1] - x_finite[0]) if x_finite.size == 2 else 1e-3
        return max(span, _FLOAT_MIN)
    return float(np.median(np.diff(np.sort(x_finite))))


def _weighted_loc(
    values: jnp.ndarray,
    weights: jnp.ndarray,
    valid: jnp.ndarray,
) -> float:
    """Return the weighted location over valid entries.

    Raises
    ------
    ValueError
        If no valid entry remains after masking.
    """
    values_v = jnp.asarray(values, dtype=jnp.float32).reshape(-1)
    weights_v = jnp.asarray(weights, dtype=jnp.float32).reshape(-1)
    valid_v = jnp.asarray(valid, dtype=bool).reshape(-1)

    if values_v.shape != weights_v.shape or values_v.shape != valid_v.shape:
        raise ValueError("values, weights, and valid must share the same shape.")

    n_valid = int(jnp.sum(valid_v))
    if n_valid == 0:
        raise ValueError("No valid values available for weighted location.")

    values_keep = values_v[valid_v]
    weights_keep = weights_v[valid_v]
    weight_sum = float(jnp.sum(weights_keep))
    if not math.isfinite(weight_sum) or weight_sum <= _FLOAT_MIN:
        raise ValueError("Weighted location received non-positive total weight.")

    loc = jnp.sum(weights_keep * values_keep) / weight_sum
    return float(loc)


def _weighted_scale(
    values: jnp.ndarray,
    weights: jnp.ndarray,
    valid: jnp.ndarray,
    loc: float,
    *,
    scale_floor: float = 1e-6,
    single_value_scale_frac: float = _SINGLE_VALUE_SCALE_FRAC,
) -> float:
    """Return the weighted spread over valid entries.

    With a single valid trace, the scale is set to ``5%`` of the observed value
    (subject to ``scale_floor``). With zero valid traces, this raises.
    """
    values_v = jnp.asarray(values, dtype=jnp.float32).reshape(-1)
    weights_v = jnp.asarray(weights, dtype=jnp.float32).reshape(-1)
    valid_v = jnp.asarray(valid, dtype=bool).reshape(-1)

    if values_v.shape != weights_v.shape or values_v.shape != valid_v.shape:
        raise ValueError("values, weights, and valid must share the same shape.")

    n_valid = int(jnp.sum(valid_v))
    if n_valid == 0:
        raise ValueError("No valid values available for weighted scale.")

    values_keep = values_v[valid_v]
    if n_valid == 1:
        single_value = float(values_keep[0])
        return max(single_value_scale_frac * abs(single_value), scale_floor)

    weights_keep = weights_v[valid_v]
    weight_sum = float(jnp.sum(weights_keep))
    if not math.isfinite(weight_sum) or weight_sum <= _FLOAT_MIN:
        raise ValueError("Weighted scale received non-positive total weight.")

    variance = jnp.sum(weights_keep * (values_keep - loc) ** 2) / weight_sum
    return max(float(jnp.sqrt(jnp.maximum(variance, 0.0))), scale_floor)


def _valid_apex_shapes(
    y_win: jnp.ndarray,  # [n_trace, n_win]
    apex_idx: jnp.ndarray,  # [n_trace]
    apex_height: jnp.ndarray,  # [n_trace]
) -> jnp.ndarray:
    """Vectorized monotonic-shape check around the apex.

    Allows 10% overshoot on either side to tolerate noise.
    """
    n_win = int(y_win.shape[1])
    idx = jnp.arange(n_win, dtype=jnp.int32)[None, :]
    upper = apex_height[:, None] * 1.1
    left_mask = idx < apex_idx[:, None]
    right_mask = idx > apex_idx[:, None]
    left_ok = jnp.all(jnp.where(left_mask, y_win <= upper, True), axis=1)
    right_ok = jnp.all(jnp.where(right_mask, y_win <= upper, True), axis=1)
    return left_ok & right_ok


def _height_weighted_apex(
    x_win: NDArray[np.float64],  # [n_win]
    y_win: NDArray[np.float64],  # [n_trace, n_win]  baseline-subtracted
    min_height_frac: float = _MIN_APEX_HEIGHT_FRAC,
) -> tuple[float, float, int]:
    """Height-weighted apex centroid and spread across traces.

    For each trace, the apex is the global maximum inside the window after
    baseline subtraction.  Traces are kept if their apex height exceeds
    ``min_height_frac x max_apex_height`` across all traces.  The centroid
    and spread of the kept apex *times* are then weighted by apex heights.

    Args:
        x_win:           1-D time axis inside the window, shape ``[n_win]``.
        y_win:           Baseline-subtracted signal, shape ``[n_trace, n_win]``.
        min_height_frac: Minimum apex height relative to the tallest trace.

    Returns:
        ``(apex_loc, apex_scale, n_valid)``
    """
    if x_win.size == 0 or y_win.shape[1] == 0:
        raise ValueError("x_win and y_win must have non-zero size")

    x_arr = jnp.asarray(x_win, dtype=jnp.float32)
    y_arr = jnp.asarray(y_win, dtype=jnp.float32)
    n_trace, n_win = y_arr.shape

    y_for_max = jnp.where(jnp.isfinite(y_arr), y_arr, -jnp.inf)
    apex_idx = jnp.argmax(y_for_max, axis=1)  # [n_trace]
    trace_idx = jnp.arange(n_trace, dtype=jnp.int32)
    apex_times = x_arr[apex_idx]  # [n_trace]
    apex_heights = y_arr[trace_idx, apex_idx]  # [n_trace]

    max_height = float(jnp.max(jnp.where(jnp.isfinite(apex_heights), apex_heights, 0.0)))
    height_threshold = max(max_height * min_height_frac, _FLOAT_MIN)

    valid = jnp.isfinite(apex_heights) & (apex_heights >= height_threshold)
    valid = valid & (apex_idx >= 3) & (apex_idx <= n_win - 4)
    valid = valid & _valid_apex_shapes(y_arr, apex_idx, apex_heights)

    n_valid = int(jnp.sum(valid))
    if n_valid == 0:
        raise ValueError("No valid apex heights found")

    apex_loc = _weighted_loc(apex_times, apex_heights, valid)
    dt_floor = _median_dt(x_win)
    apex_scale = _weighted_scale(
        apex_times,
        apex_heights,
        valid,
        apex_loc,
        scale_floor=dt_floor,
    )
    return apex_loc, apex_scale, n_valid


def _robust_mad_scale(
    values: NDArray[np.float64],
    *,
    scale_floor: float = 1e-6,
    single_value_scale_frac: float = _SINGLE_VALUE_SCALE_FRAC,
) -> float:
    """Return a robust scale estimate from the median absolute deviation."""
    values_arr = np.asarray(values, dtype=float)
    values_finite = values_arr[np.isfinite(values_arr)]
    if values_finite.size == 0:
        return float(scale_floor)
    if values_finite.size == 1:
        return max(single_value_scale_frac * abs(float(values_finite[0])), scale_floor)

    center = float(np.median(values_finite))
    mad = float(np.median(np.abs(values_finite - center)))
    return max(1.4826 * mad, scale_floor)


def _trace_fwhm_geometry(
    x_win: NDArray[np.float64],  # [n_win]
    y_win: NDArray[np.float64],  # [n_trace, n_win]
    *,
    level: float = 0.5,
    min_height_frac: float = _MIN_APEX_HEIGHT_FRAC,
) -> _TraceFwhmGeometry:
    """Vectorized per-trace FWHM geometry inside one peak window.

    Returns
    -------
    _TraceFwhmGeometry
        Raw apex measurements plus explicit height- and FWHM-validity masks.
    """
    if not (0.0 < level < 1.0):
        raise ValueError("level must satisfy 0 < level < 1.")

    x_arr = jnp.asarray(x_win, dtype=jnp.float32).reshape(-1)
    y_arr = jnp.asarray(y_win, dtype=jnp.float32)
    if y_arr.ndim != 2 or y_arr.shape[1] != x_arr.shape[0]:
        raise ValueError("x_win and y_win must align as [n_trace, n_win].")

    n_trace, n_win = y_arr.shape
    y_for_max = jnp.where(jnp.isfinite(y_arr), y_arr, -jnp.inf)
    apex_idx = jnp.argmax(y_for_max, axis=1)  # [n_trace]
    trace_idx = jnp.arange(n_trace, dtype=jnp.int32)
    apex_height = y_arr[trace_idx, apex_idx]  # [n_trace]
    apex_time = x_arr[apex_idx]  # [n_trace]

    max_height = float(jnp.max(jnp.where(jnp.isfinite(apex_height), apex_height, 0.0)))
    height_threshold = max(max_height * min_height_frac, _FLOAT_MIN)

    height_valid = jnp.isfinite(apex_height) & (apex_height >= height_threshold)
    geometry_valid = height_valid & (apex_idx >= 3) & (apex_idx <= n_win - 4)
    geometry_valid = geometry_valid & _valid_apex_shapes(y_arr, apex_idx, apex_height)

    height_safe = jnp.where(height_valid, apex_height, 1.0)
    y_norm = y_arr / height_safe[:, None]

    t_idx = jnp.arange(n_win - 1, dtype=jnp.int32)[None, :]
    left_of_apex = t_idx < apex_idx[:, None]
    left_rising = (y_norm[:, :-1] <= level) & (y_norm[:, 1:] > level) & left_of_apex
    right_falling = (y_norm[:, :-1] >= level) & (y_norm[:, 1:] < level) & ~left_of_apex

    has_left = jnp.any(left_rising, axis=1)
    has_right = jnp.any(right_falling, axis=1)

    left_i = (n_win - 2) - jnp.argmax(jnp.flip(left_rising, axis=1), axis=1)
    right_i = jnp.argmax(right_falling, axis=1)
    left_i = jnp.clip(left_i, 0, n_win - 2)
    right_i = jnp.clip(right_i, 0, n_win - 2)

    x_pair = jnp.broadcast_to(x_arr[None, :], (n_trace, n_win))

    def _take2(arr: jnp.ndarray, idx: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        idx2 = idx[:, None]
        v0 = jnp.take_along_axis(arr, idx2, axis=1).squeeze(1)
        v1 = jnp.take_along_axis(arr, idx2 + 1, axis=1).squeeze(1)
        return v0, v1

    x_l0, x_l1 = _take2(x_pair, left_i)
    y_l0, y_l1 = _take2(y_norm, left_i)
    x_r0, x_r1 = _take2(x_pair, right_i)
    y_r0, y_r1 = _take2(y_norm, right_i)

    def _interp(
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        y0: jnp.ndarray,
        y1: jnp.ndarray,
        valid_cross: jnp.ndarray,
    ) -> jnp.ndarray:
        dy = y1 - y0
        dy_safe = jnp.where(jnp.abs(dy) > _FLOAT_MIN, dy, 1.0)
        crossing = x0 + (level - y0) / dy_safe * (x1 - x0)
        return jnp.where(valid_cross & (jnp.abs(dy) > _FLOAT_MIN), crossing, jnp.nan)

    left_time = _interp(x_l0, x_l1, y_l0, y_l1, has_left & geometry_valid)
    right_time = _interp(x_r0, x_r1, y_r0, y_r1, has_right & geometry_valid)

    valid_fwhm = geometry_valid & has_left & has_right
    valid_fwhm = valid_fwhm & jnp.isfinite(left_time) & jnp.isfinite(right_time)
    valid_fwhm = valid_fwhm & (left_time < apex_time) & (right_time > apex_time)

    w_left = jnp.where(valid_fwhm, apex_time - left_time, jnp.nan)
    w_right = jnp.where(valid_fwhm, right_time - apex_time, jnp.nan)
    return _TraceFwhmGeometry(
        apex_time=apex_time,
        apex_height=apex_height,
        w_left=w_left,
        w_right=w_right,
        height_valid=height_valid,
        fwhm_valid=valid_fwhm,
    )


def _halfwidth_priors(
    x_win: NDArray[np.float64],
    y_win: NDArray[np.float64],
    *,
    level: float = 0.5,
    min_height_frac: float = _MIN_APEX_HEIGHT_FRAC,
) -> tuple[float, float, float, float, _TraceFwhmGeometry]:
    """Height-weighted population priors for left/right half-widths.

    Returns ``(w_left_loc, w_left_scale, w_right_loc, w_right_scale, geometry)``.
    Half-widths are the directly measured HWHM values — no conversion to
    ``(sigma, alpha)`` is performed here.
    """
    geometry = _trace_fwhm_geometry(
        x_win,
        y_win,
        level=level,
        min_height_frac=min_height_frac,
    )
    apex_height = jnp.where(geometry.fwhm_valid, geometry.apex_height, jnp.nan)

    w_left_loc = _weighted_loc(geometry.w_left, apex_height, geometry.fwhm_valid)
    w_left_scale = _weighted_scale(
        geometry.w_left,
        apex_height,
        geometry.fwhm_valid,
        w_left_loc,
        scale_floor=1e-6,
    )
    w_right_loc = _weighted_loc(geometry.w_right, apex_height, geometry.fwhm_valid)
    w_right_scale = _weighted_scale(
        geometry.w_right,
        apex_height,
        geometry.fwhm_valid,
        w_right_loc,
        scale_floor=1e-6,
    )
    return w_left_loc, w_left_scale, w_right_loc, w_right_scale, geometry


def _window_area(
    x_win: NDArray[np.float64],  # [n_win]
    y_win: NDArray[np.float64],  # [n_trace, n_win]  baseline-subtracted
) -> NDArray[np.float64]:
    """Per-trace baseline-subtracted trapezoid areas inside a window.

    Negative values (noise below baseline) are clipped to zero before
    integration.

    Args:
        x_win: 1-D time axis inside the window, shape ``[n_win]``.
        y_win: Baseline-subtracted signal, shape ``[n_trace, n_win]``.

    Returns:
        Array of shape ``[n_trace]`` with one trapezoid-integrated area per trace.
    """
    areas = [
        float(np.trapezoid(np.maximum(y_win[t], 0.0), x_win))  # type: ignore[attr-defined]
        for t in range(y_win.shape[0])
    ]
    return np.array(areas, dtype=np.float64)


def _gaussian_area_from_halfwidths(
    geometry: _TraceFwhmGeometry,
) -> jax.Array:
    """Per-trace Gaussian area estimate from FWHM half-widths and apex height.

    ``A = height * sigma * sqrt(2*pi)`` where ``sigma`` is derived from the
    measured left/right half-widths: ``sigma = sqrt(0.5*(w_l² + w_r²)) / HWHM_factor``.

    Returns ``NaN`` for traces where FWHM detection failed.
    """
    s_left = geometry.w_left / _GAUSSIAN_HWHM_FACTOR
    s_right = geometry.w_right / _GAUSSIAN_HWHM_FACTOR
    sigma = jnp.sqrt(0.5 * (s_left**2 + s_right**2))
    area = _GAUSSIAN_AREA_FROM_HEIGHT_SIGMA * jnp.maximum(geometry.apex_height, 0.0) * sigma
    return jnp.where(geometry.fwhm_valid, area, jnp.nan)


def _estimate_snr(
    y_win: NDArray[np.float64],
    apex_height: jax.Array,
) -> NDArray[np.float64]:
    """Per-trace signal-to-noise ratio from first-difference MAD.

    Uses the median absolute deviation of first differences as a robust
    noise estimator (no baseline regions required).
    """
    diffs = np.abs(np.diff(np.asarray(y_win, dtype=float), axis=1))
    # MAD of first differences → noise std (factor 0.7071 = 1/sqrt(2))
    noise_est = float(np.median(diffs)) * 0.7071
    noise_est = max(noise_est, _FLOAT_MIN)
    return np.maximum(np.asarray(apex_height, dtype=float) / noise_est, 0.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_peak_priors(
    peaks: list[PeakAnnotation],
    x: NDArray[np.float64],
    signal: NDArray[np.float64],
    baseline: NDArray[np.float64],
) -> tuple[list[GeometricPeakPriors], PeakApexTraces]:
    """Build window-geometry-based priors and per-trace apex data in one pass.

    Computes FWHM geometry once per peak window, assembling both the
    ``GeometricPeakPriors`` needed by the model and the ``PeakApexTraces``
    needed by :func:`refine_apex_priors_with_trace_shift`.

    Parameters
    ----------
    peaks:
        Annotated peak windows in any order.
        ``peak.mode != "single"`` → double-peak window (2 components).
    x:
        Shared 1-D time axis, shape ``[n_time]``.  Must be strictly increasing.
    signal:
        Raw signal matrix, shape ``[n_trace, n_time]``.
    baseline:
        Estimated baseline matrix from ``baseline.py``, shape ``[n_trace, n_time]``.

    Returns
    -------
    tuple[list[GeometricPeakPriors], PeakApexTraces]
        ``priors`` — one :class:`GeometricPeakPriors` per element of ``peaks``.
        ``apex_traces`` — per-trace FWHM apex data for trace-shift refinement.

    Raises
    ------
    ValueError
        If ``signal`` / ``baseline`` are not 2-D, ``x`` length does not match,
        or a peak window contains no data points.
    """
    x = np.asarray(x, dtype=float).ravel()
    signal = np.asarray(signal, dtype=float)
    baseline = np.asarray(baseline, dtype=float)

    if signal.ndim != 2:
        raise ValueError(f"signal must be 2-D [n_trace, n_time], got shape {signal.shape}.")
    if baseline.ndim != 2:
        raise ValueError(f"baseline must be 2-D [n_trace, n_time], got shape {baseline.shape}.")
    if x.size != signal.shape[1]:
        raise ValueError(f"x length ({x.size}) must match signal.shape[1] ({signal.shape[1]}).")
    if signal.shape != baseline.shape:
        raise ValueError(
            f"signal and baseline must have the same shape, got {signal.shape} vs {baseline.shape}."
        )

    n_trace = int(signal.shape[0])
    n_peak = len(peaks)
    signal_corrected = signal - baseline  # [n_trace, n_time]

    fwhm_apex_trace = np.full((n_trace, n_peak), np.nan, dtype=np.float64)
    fwhm_valid_trace = np.zeros((n_trace, n_peak), dtype=bool)
    priors: list[GeometricPeakPriors] = []

    for peak_idx, peak in enumerate(peaks):
        lo, hi = float(peak.rt_min), float(peak.rt_max)
        mask = (x >= lo) & (x <= hi) & np.isfinite(x)

        if not np.any(mask):
            raise ValueError(
                f"Peak '{peak.molecule_id}' window [{lo:.4f}, {hi:.4f}] contains no finite data points in x."
            )

        x_win = x[mask]  # [n_win]
        y_win = signal_corrected[:, mask]  # [n_trace, n_win]

        # --- Apex position priors ---
        apex_loc, apex_scale, n_valid = _height_weighted_apex(x_win, y_win)

        # --- Half-width shape priors (directly from FWHM geometry) ---
        w_left_loc, w_left_scale, w_right_loc, w_right_scale, geometry = _halfwidth_priors(
            x_win,
            y_win,
        )

        # --- Area priors ---
        # Gaussian area from half-widths (NaN for invalid traces)
        area_gaussian_raw = _gaussian_area_from_halfwidths(geometry)
        # Fill invalid traces with scaled-down cross-trace median
        median_area = float(jnp.nanmedian(area_gaussian_raw))
        median_area = max(median_area, _FLOAT_MIN)
        area_gaussian_pt = np.maximum(
            np.where(np.isfinite(area_gaussian_raw), area_gaussian_raw, median_area * 0.01),
            _FLOAT_MIN,
        )
        # Trapezoid total area
        area_trapz_pt = np.maximum(_window_area(x_win, y_win), _FLOAT_MIN)

        # Artefact residual area
        if peak.mode == "artefact_doublet":
            residual = np.maximum(area_trapz_pt - np.asarray(area_gaussian_pt, dtype=float), 0.0)
            valid_residual = residual[residual > _FLOAT_MIN]
            area_art_shared = float(np.median(valid_residual)) if valid_residual.size > 0 else _FLOAT_MIN
        else:
            area_art_shared = 0.0

        # --- Signal-to-noise ratio ---
        snr_per_trace = _estimate_snr(y_win, geometry.apex_height)

        # --- Per-trace apex data for trace-shift refinement ---
        fwhm_apex_trace[:, peak_idx] = np.asarray(
            jnp.where(geometry.fwhm_valid, geometry.apex_time, jnp.nan),
            dtype=np.float64,
        )
        fwhm_valid_trace[:, peak_idx] = np.asarray(geometry.fwhm_valid, dtype=bool)

        priors.append(
            GeometricPeakPriors(
                mode=peak.mode,
                apex_loc=apex_loc,
                apex_scale=float(apex_scale),
                w_left_loc=w_left_loc,
                w_left_scale=w_left_scale,
                w_right_loc=w_right_loc,
                w_right_scale=w_right_scale,
                area_gaussian_pt=area_gaussian_pt,
                area_trapz_pt=area_trapz_pt,
                area_art_shared=area_art_shared,
                snr_per_trace=snr_per_trace,
                window_lo=lo,
                window_hi=hi,
                n_valid_traces=n_valid,
            )
        )

    return priors, PeakApexTraces(
        fwhm_apex_trace=fwhm_apex_trace,
        fwhm_valid_trace=fwhm_valid_trace,
    )


def refine_apex_priors_with_trace_shift(
    priors: list[GeometricPeakPriors],
    traces: PeakApexTraces,
    *,
    apex_scale_floor: float = 1e-4,
    trace_shift_scale_floor: float = 1e-6,
) -> tuple[list[GeometricPeakPriors], float]:
    """Refine apex scales for the shared trace-shift hierarchy.

    The shared trace shift is estimated from the per-trace median apex
    deviations across peaks with valid FWHM apex measurements. The per-peak
    apex scales are then recomputed from the residual deviations after
    removing that shared trace shift. Peaks without enough residual data fall
    back to the ``apex_scale`` stored in the prior.
    """
    if not priors:
        return [], float(trace_shift_scale_floor)

    fwhm_apex = np.asarray(traces.fwhm_apex_trace, dtype=float)
    fwhm_valid = np.asarray(traces.fwhm_valid_trace, dtype=bool)
    if fwhm_apex.shape[1] != len(priors):
        raise ValueError(
            "refine_apex_priors_with_trace_shift requires one apex_trace column per peak prior."
        )

    apex_loc = np.asarray([p.apex_loc for p in priors], dtype=float)
    legacy_apex_scale = np.asarray([p.apex_scale for p in priors], dtype=float)
    delta = np.where(fwhm_valid, fwhm_apex - apex_loc[None, :], np.nan)

    shift_hat = np.full(delta.shape[0], np.nan, dtype=float)
    for trace_idx in range(delta.shape[0]):
        delta_trace = delta[trace_idx]
        finite_delta = delta_trace[np.isfinite(delta_trace)]
        if finite_delta.size >= 2:
            shift_hat[trace_idx] = float(np.median(finite_delta))  # type: ignore[arg-type]

    trace_shift_scale = _robust_mad_scale(
        shift_hat,
        scale_floor=trace_shift_scale_floor,
    )

    apex_scale_refined = legacy_apex_scale.copy()
    for peak_idx in range(delta.shape[1]):
        valid_residual = np.isfinite(delta[:, peak_idx]) & np.isfinite(shift_hat)
        residual = delta[valid_residual, peak_idx] - shift_hat[valid_residual]
        if residual.size >= 2:
            apex_scale_refined[peak_idx] = _robust_mad_scale(
                residual,
                scale_floor=apex_scale_floor,
            )
        else:
            apex_scale_refined[peak_idx] = max(
                float(legacy_apex_scale[peak_idx]),
                float(apex_scale_floor),
            )

    refined_priors = [
        dataclasses.replace(prior, apex_scale=float(apex_scale_refined[idx]))
        for idx, prior in enumerate(priors)
    ]
    return refined_priors, float(trace_shift_scale)


def geometric_priors_to_arrays(
    priors: list[GeometricPeakPriors],
) -> dict[str, NDArray[np.float64]]:
    """Convert a list of ``GeometricPeakPriors`` to model-ready numpy arrays.

    Parameters
    ----------
    priors:
        Output of ``build_peak_priors``.

    Returns
    -------
    dict with keys:

    - ``apex_loc``           [n_peak]          — apex-weighted centroid.
    - ``apex_scale``         [n_peak]          — local apex spread after removing shared trace drift.
    - ``w_left_loc``         [n_peak]          — left HWHM prior centres.
    - ``w_left_scale``       [n_peak]          — left HWHM prior scales.
    - ``w_right_loc``        [n_peak]          — right HWHM prior centres.
    - ``w_right_scale``      [n_peak]          — right HWHM prior scales.
    - ``window_lo``          [n_peak]          — window lower bounds.
    - ``window_hi``          [n_peak]          — window upper bounds.
    - ``area_gaussian_pt``   [n_trace, n_peak] — per-trace Gaussian area estimates.
    - ``area_trapz_pt``      [n_trace, n_peak] — per-trace trapezoid areas.
    - ``area_art_shared``    [n_artefact]      — shared artefact area prior centres.
    - ``snr_per_trace``      [n_trace, n_peak] — per-trace signal-to-noise ratio.
    """
    return {
        "apex_loc": np.array([p.apex_loc for p in priors], dtype=np.float64),
        "apex_scale": np.array([p.apex_scale for p in priors], dtype=np.float64),
        "w_left_loc": np.array([p.w_left_loc for p in priors], dtype=np.float64),
        "w_left_scale": np.array([p.w_left_scale for p in priors], dtype=np.float64),
        "w_right_loc": np.array([p.w_right_loc for p in priors], dtype=np.float64),
        "w_right_scale": np.array([p.w_right_scale for p in priors], dtype=np.float64),
        "window_lo": np.array([p.window_lo for p in priors], dtype=np.float64),
        "window_hi": np.array([p.window_hi for p in priors], dtype=np.float64),
        "area_gaussian_pt": np.column_stack(
            [p.area_gaussian_pt for p in priors]
        ).astype(np.float64) if priors else np.empty((0, 0), dtype=np.float64),
        "area_trapz_pt": np.column_stack(
            [p.area_trapz_pt for p in priors]
        ).astype(np.float64) if priors else np.empty((0, 0), dtype=np.float64),
        "area_art_shared": np.array(
            [p.area_art_shared for p in priors if p.mode == "artefact_doublet"],
            dtype=np.float64,
        ),
        "snr_per_trace": np.column_stack(
            [p.snr_per_trace for p in priors]
        ).astype(np.float64) if priors else np.empty((0, 0), dtype=np.float64),
    }


def summarise_priors(priors: list[GeometricPeakPriors]) -> str:
    """Return a human-readable summary table of computed priors.

    Parameters
    ----------
    priors:
        Output of ``build_peak_priors``.

    Returns
    -------
    str
        Multi-line table suitable for logging or ``print()``.
    """
    lines = [
        f"{'Peak':>4}  {'mode':>17}  {'window':>18}  {'apex_loc':>8}  {'apex_sc':>8}  "
        f"{'w_L':>8}  {'w_L_sc':>8}  "
        f"{'w_R':>8}  {'w_R_sc':>8}  "
        f"{'art_area':>10}  {'med_snr':>7}  {'nvalid':>6}",
        "-" * 132,
    ]
    for i, p in enumerate(priors):
        art_str = f"{p.area_art_shared:.3e}" if p.mode == "artefact_doublet" else "       ---"
        med_snr = float(np.median(p.snr_per_trace))
        lines.append(
            f"{i:>4}  "
            f"{p.mode:>17}  "
            f"[{p.window_lo:.3f},{p.window_hi:.3f}]  "
            f"{p.apex_loc:>8.4f}  {p.apex_scale:>8.5f}  "
            f"{p.w_left_loc:>8.5f}  {p.w_left_scale:>8.5f}  "
            f"{p.w_right_loc:>8.5f}  {p.w_right_scale:>8.5f}  "
            f"{art_str:>10}  "
            f"{med_snr:>7.1f}  {p.n_valid_traces:>6}"
        )
    return "\n".join(lines)


__all__ = [
    "GeometricPeakPriors",
    "PeakApexTraces",
    "build_peak_priors",
    "geometric_priors_to_arrays",
    "refine_apex_priors_with_trace_shift",
    "summarise_priors",
]
