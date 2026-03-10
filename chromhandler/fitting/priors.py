"""Window-geometry-based Bayesian priors for chromatographic peak fitting.

Combines window geometry, apex statistics, and FWHM-derived shape estimates.
All priors are derived from:

- **Apex-height-weighted centroid**      → mu_loc, mu_scale
- **FWHM half-width geometry**           → sigma_loc, sigma_scale, alpha_loc, alpha_scale
- **Gaussian area from apex × sigma**    → main_area_per_trace
- **Trapezoid total-window integration** → total_area_per_trace
- **Residual integration**               → artefact_shoulder_area_loc (artefact peaks only)

Pipeline
--------
1. ``_median_dt``              — robust median sampling interval in a window.
2. ``_height_weighted_apex``   — height-weighted apex centroid across traces.
3. ``_shape_priors_from_fwhm`` — height-weighted FWHM-based sigma/alpha priors.
4. ``_window_area``            — per-trace baseline-subtracted trapezoid areas.
5. ``_main_peak_approximation`` — per-trace Gaussian main-peak approximation.
6. ``_residual_area``            — per-trace residual area after subtracting Gaussian main area.
7. ``build_geometric_priors``   — assemble all of the above per ``PeakAnnotation``.
8. ``geometric_priors_to_arrays`` — convert list of priors to model-ready numpy arrays.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Final

import jax.numpy as jnp
import numpy as np

from .data import (
    PeakAnnotation,
    PeakMode,
    peak_component_count,
    peak_is_artefact_mode,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FLOAT_MIN: Final = 1e-12

_GAUSSIAN_HWHM_FACTOR: Final = math.sqrt(2.0 * math.log(2.0))
_GAUSSIAN_FWHM_FACTOR: Final = 2.0 * _GAUSSIAN_HWHM_FACTOR
_GAUSSIAN_AREA_FROM_HEIGHT_SIGMA: Final = math.sqrt(2.0 * math.pi)

# Minimum height fraction for apex outlier rejection (fraction of max apex height)
_MIN_APEX_HEIGHT_FRAC: Final = 0.05
_SINGLE_VALUE_SCALE_FRAC: Final = 0.05


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GeometricPeakPriors:
    """Window-geometry-based priors for one annotated peak window.

    Attributes
    ----------
    mu_loc:
        Apex-height-weighted centroid of the peak across traces [time units].
    mu_scale:
        Height-weighted standard deviation of apex positions across traces.
        Reflects retention-time drift between injections, not peak width.
    sigma_loc:
        Height-weighted centre of per-trace FWHM-derived sigma estimates.
    sigma_scale:
        Height-weighted spread of per-trace FWHM-derived sigma estimates.
    alpha_loc:
        Height-weighted centre of per-trace FWHM-derived alpha estimates.
    alpha_scale:
        Height-weighted spread of per-trace FWHM-derived alpha estimates.
    mode:
        Peak fitting mode from the source :class:`PeakAnnotation`.
    main_area_per_trace:
        Gaussian main-area estimate for each trace, shape ``[n_trace]``.

        This is derived from the baseline-corrected apex height and Gaussian
        sigma under a Gaussian approximation:

        ``A_main = apex_height × sigma × sqrt(2π)``.

        Used as the per-trace prior centre for the primary component area in
        the ``single`` and ``artefact_doublet`` model branches.
    total_area_per_trace:
        Total trapezoid-integrated window area per trace, shape ``[n_trace]``.
        Used as the per-trace prior centre for ``A_total_free`` in the
        ``free_doublet`` model branch.
    artefact_shoulder_area_loc:
        Median positive residual area across traces [area units], where the
        residual is ``max(total_trapezoid_area - gaussian_main_area, 0)``.
        ``0.0`` unless ``mode == "artefact_doublet"``. Used as the prior
        centre for the shared artefact shoulder area scalar in the artefact
        branch of the model.
    window_lo:
        Lower bound of the peak window [time units].
    window_hi:
        Upper bound of the peak window [time units].
    n_valid_traces:
        Number of traces that contributed a valid apex to the centroid estimate.
    """

    mode: PeakMode
    mu_loc: float
    mu_scale: float
    sigma_loc: float
    sigma_scale: float
    alpha_loc: float
    alpha_scale: float
    main_area_per_trace: np.ndarray
    total_area_per_trace: np.ndarray
    artefact_shoulder_area_loc: float
    window_lo: float
    window_hi: float
    n_valid_traces: int

    @property
    def n_components(self) -> int:
        return peak_component_count(self.mode)

    def __repr__(self) -> str:
        art_str = (
            f", art_sh_area={self.artefact_shoulder_area_loc:.2e}"
            if peak_is_artefact_mode(self.mode)
            else ""
        )
        return (
            f"GeometricPeakPriors("
            f"window=[{self.window_lo:.4f}, {self.window_hi:.4f}], "
            f"mode={self.mode}, "
            f"mu={self.mu_loc:.4f}±{self.mu_scale:.4f}, "
            f"sigma={self.sigma_loc:.4f}±{self.sigma_scale:.4f}, "
            f"alpha={self.alpha_loc:.3f}±{self.alpha_scale:.3f}, "
            f"ncomp={self.n_components}{art_str}, "
            f"n_valid={self.n_valid_traces})"
        )


@dataclasses.dataclass(frozen=True)
class FwhmShapeDiagnostics:
    """Per-trace FWHM-derived main-peak shape diagnostics across all windows.

    Attributes
    ----------
    mu_trace:
        Detected apex times per trace and peak window, shape ``[n_trace, n_peak]``.
        This is the FWHM-detected apex position used for diagnostics, not the
        posterior ``center_per_trace`` latent from the model.
    apex_height_trace:
        Baseline-corrected apex heights, shape ``[n_trace, n_peak]``.
    sigma_trace:
        Per-trace sigma estimates derived from FWHM half-width geometry,
        shape ``[n_trace, n_peak]``.
    alpha_trace:
        Per-trace alpha estimates derived from FWHM half-width asymmetry,
        shape ``[n_trace, n_peak]``.
    fwhm_trace:
        Per-trace FWHM values, shape ``[n_trace, n_peak]``.
    area_gaussian_trace:
        Per-trace Gaussian main-area estimates. FWHM-valid traces use their
        own per-trace apex and sigma; low-height fallback traces use the
        shared ``mu_loc`` / ``sigma_loc`` approximation.
        shape ``[n_trace, n_peak]``.
    area_total_trace:
        Per-trace total trapezoid-integrated window areas,
        shape ``[n_trace, n_peak]``.
    area_residual_trace:
        Per-trace residual areas used for shoulder estimation,
        shape ``[n_trace, n_peak]``.
    height_valid_trace:
        Boolean mask marking traces whose window apex height passes the
        relative threshold in each peak window, shape ``[n_trace, n_peak]``.
    fwhm_valid_trace:
        Boolean mask marking traces that produced a valid FWHM-based shape
        estimate in each peak window, shape ``[n_trace, n_peak]``.
    approx_center_trace:
        Dense per-trace Gaussian-approximation centres, shape ``[n_trace, n_peak]``.
    approx_height_trace:
        Dense per-trace Gaussian-approximation heights, shape ``[n_trace, n_peak]``.
    approx_sigma_trace:
        Dense per-trace Gaussian-approximation sigma values, shape ``[n_trace, n_peak]``.
    approx_valid_trace:
        Boolean mask marking traces that contribute a Gaussian main-peak
        approximation, shape ``[n_trace, n_peak]``.
    approx_fallback_trace:
        Boolean mask marking traces that use the low-height fallback
        approximation instead of a per-trace FWHM estimate, shape ``[n_trace, n_peak]``.
    """

    mu_trace: np.ndarray
    apex_height_trace: np.ndarray
    sigma_trace: np.ndarray
    alpha_trace: np.ndarray
    fwhm_trace: np.ndarray
    area_gaussian_trace: np.ndarray
    area_total_trace: np.ndarray
    area_residual_trace: np.ndarray
    height_valid_trace: np.ndarray
    fwhm_valid_trace: np.ndarray
    approx_center_trace: np.ndarray
    approx_height_trace: np.ndarray
    approx_sigma_trace: np.ndarray
    approx_valid_trace: np.ndarray
    approx_fallback_trace: np.ndarray


@dataclasses.dataclass(frozen=True)
class _TraceFwhmGeometry:
    """Per-trace apex/FWHM geometry for one peak window."""

    apex_time: jnp.ndarray
    apex_height: jnp.ndarray
    w_left: jnp.ndarray
    w_right: jnp.ndarray
    height_valid: jnp.ndarray
    fwhm_valid: jnp.ndarray


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _median_dt(x_win: np.ndarray) -> float:
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
    x_win: np.ndarray,  # [n_win]
    y_win: np.ndarray,  # [n_trace, n_win]  baseline-subtracted
    min_height_frac: float = _MIN_APEX_HEIGHT_FRAC,
) -> tuple[float, float, int]:
    """Height-weighted apex centroid and spread across traces.

    For each trace, the apex is the global maximum inside the window after
    baseline subtraction.  Traces are kept if their apex height exceeds
    ``min_height_frac × max_apex_height`` across all traces.  The centroid
    and spread of the kept apex *times* are then weighted by apex heights.

    Args:
        x_win:           1-D time axis inside the window, shape ``[n_win]``.
        y_win:           Baseline-subtracted signal, shape ``[n_trace, n_win]``.
        min_height_frac: Minimum apex height relative to the tallest trace.

    Returns:
        ``(mu_loc, mu_scale, n_valid)``
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

    max_height = float(
        jnp.max(jnp.where(jnp.isfinite(apex_heights), apex_heights, 0.0))
    )
    height_threshold = max(max_height * min_height_frac, _FLOAT_MIN)

    valid = jnp.isfinite(apex_heights) & (apex_heights >= height_threshold)
    valid = valid & (apex_idx >= 3) & (apex_idx <= n_win - 4)
    valid = valid & _valid_apex_shapes(y_arr, apex_idx, apex_heights)

    n_valid = int(jnp.sum(valid))
    if n_valid == 0:
        raise ValueError("No valid apex heights found")

    mu_loc = _weighted_loc(apex_times, apex_heights, valid)
    dt_floor = _median_dt(x_win)
    mu_scale = _weighted_scale(
        apex_times,
        apex_heights,
        valid,
        mu_loc,
        scale_floor=dt_floor,
    )
    return mu_loc, mu_scale, n_valid


def _trace_fwhm_geometry(
    x_win: np.ndarray,  # [n_win]
    y_win: np.ndarray,  # [n_trace, n_win]
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


def _fwhm_to_sigma_alpha(
    w_left: jnp.ndarray,
    w_right: jnp.ndarray,
    valid: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert left/right HWHM values to per-trace ``(sigma, alpha)``."""
    w_left_arr = jnp.asarray(w_left, dtype=jnp.float32)
    w_right_arr = jnp.asarray(w_right, dtype=jnp.float32)
    valid_arr = jnp.asarray(valid, dtype=bool)

    sl = jnp.where(valid_arr, w_left_arr / _GAUSSIAN_HWHM_FACTOR, jnp.nan)
    sr = jnp.where(valid_arr, w_right_arr / _GAUSSIAN_HWHM_FACTOR, jnp.nan)
    sigma = jnp.where(valid_arr, jnp.sqrt(0.5 * (sl**2 + sr**2)), jnp.nan)
    delta = (sr - sl) / jnp.maximum(sr + sl, _FLOAT_MIN)
    delta = jnp.where(valid_arr, jnp.clip(delta, -0.95, 0.95), jnp.nan)
    alpha = jnp.where(
        valid_arr,
        delta / jnp.sqrt(jnp.maximum(1.0 - delta**2, 1e-8)),
        jnp.nan,
    )
    return sigma, alpha


def _shape_priors_from_fwhm(
    x_win: np.ndarray,
    y_win: np.ndarray,
    *,
    level: float = 0.5,
    min_height_frac: float = _MIN_APEX_HEIGHT_FRAC,
) -> tuple[float, float, float, float, _TraceFwhmGeometry]:
    """Estimate population ``sigma`` and ``alpha`` priors from per-trace FWHM geometry."""
    geometry = _trace_fwhm_geometry(
        x_win,
        y_win,
        level=level,
        min_height_frac=min_height_frac,
    )
    sigma_trace, alpha_trace = _fwhm_to_sigma_alpha(
        geometry.w_left,
        geometry.w_right,
        geometry.fwhm_valid,
    )
    sigma_trace = jnp.where(geometry.fwhm_valid, sigma_trace, jnp.nan)
    apex_height = jnp.where(geometry.fwhm_valid, geometry.apex_height, jnp.nan)

    sigma_loc = _weighted_loc(sigma_trace, apex_height, geometry.fwhm_valid)
    sigma_scale = _weighted_scale(
        sigma_trace,
        apex_height,
        geometry.fwhm_valid,
        sigma_loc,
        scale_floor=1e-6,
    )
    alpha_loc = _weighted_loc(alpha_trace, apex_height, geometry.fwhm_valid)
    alpha_scale = _weighted_scale(
        alpha_trace,
        apex_height,
        geometry.fwhm_valid,
        alpha_loc,
        scale_floor=1e-3,
    )
    return sigma_loc, sigma_scale, alpha_loc, alpha_scale, geometry


def _window_area(
    x_win: np.ndarray,  # [n_win]
    y_win: np.ndarray,  # [n_trace, n_win]  baseline-subtracted
) -> np.ndarray:
    """Per-trace baseline-subtracted trapezoid areas inside a window.

    Negative values (noise below baseline) are clipped to zero before
    integration.

    Args:
        x_win: 1-D time axis inside the window, shape ``[n_win]``.
        y_win: Baseline-subtracted signal, shape ``[n_trace, n_win]``.

    Returns:
        Array of shape ``[n_trace]`` with one trapezoid-integrated area per trace.
    """
    return np.array(
        [
            float(np.trapz(np.maximum(y_win[t], 0.0), x_win))
            for t in range(y_win.shape[0])
        ]
    )


def _gaussian_area_from_sigma(
    apex_height: jnp.ndarray,
    sigma: jnp.ndarray,
    valid: jnp.ndarray,
) -> jnp.ndarray:
    """Estimate Gaussian peak area from apex height and sigma."""
    apex_height_arr = jnp.asarray(apex_height, dtype=jnp.float32)
    sigma_arr = jnp.asarray(sigma, dtype=jnp.float32)
    valid_arr = jnp.asarray(valid, dtype=bool)

    return jnp.where(
        valid_arr,
        _GAUSSIAN_AREA_FROM_HEIGHT_SIGMA * apex_height_arr * sigma_arr,
        0.0,
    )


def _residual_area(
    total_area: jnp.ndarray,
    area_gaussian: jnp.ndarray,
    valid: jnp.ndarray,
) -> jnp.ndarray:
    """Residual positive area after subtracting the Gaussian main-area estimate."""
    total_area_arr = jnp.asarray(total_area, dtype=jnp.float32)
    area_gaussian_arr = jnp.asarray(area_gaussian, dtype=jnp.float32)
    valid_arr = jnp.asarray(valid, dtype=bool)
    residual = jnp.maximum(total_area_arr - area_gaussian_arr, 0.0)
    return jnp.where(valid_arr, residual, 0.0)


def _main_peak_approximation(
    x_win: np.ndarray,
    y_win: np.ndarray,
    *,
    mu_loc: float,
    sigma_loc: float,
    geometry: _TraceFwhmGeometry,
) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Dense per-trace Gaussian main-peak approximation for one window.

    Valid FWHM traces use their own apex height/time and FWHM-derived sigma.
    Low-height traces fall back to the shared ``mu_loc`` and ``sigma_loc``.
    """
    sigma_trace, _ = _fwhm_to_sigma_alpha(
        geometry.w_left,
        geometry.w_right,
        geometry.fwhm_valid,
    )
    fallback_mask = ~geometry.height_valid

    x_arr = np.asarray(x_win, dtype=float).ravel()
    y_arr = jnp.asarray(y_win, dtype=jnp.float32)
    nearest_idx = int(np.argmin(np.abs(x_arr - float(mu_loc))))
    nearest_height = jnp.maximum(
        jnp.where(jnp.isfinite(y_arr[:, nearest_idx]), y_arr[:, nearest_idx], 0.0),
        0.0,
    )

    approx_center = jnp.where(
        geometry.fwhm_valid,
        geometry.apex_time,
        jnp.where(fallback_mask, float(mu_loc), jnp.nan),
    )
    approx_height = jnp.where(
        geometry.fwhm_valid,
        jnp.maximum(geometry.apex_height, 0.0),
        jnp.where(fallback_mask, nearest_height, jnp.nan),
    )
    approx_sigma = jnp.where(
        geometry.fwhm_valid,
        sigma_trace,
        jnp.where(fallback_mask, float(sigma_loc), jnp.nan),
    )
    approx_valid = geometry.fwhm_valid | fallback_mask
    approx_fallback = (~geometry.fwhm_valid) & fallback_mask
    area_gaussian = _gaussian_area_from_sigma(
        approx_height,
        approx_sigma,
        approx_valid,
    )
    return (
        approx_center,
        approx_height,
        approx_sigma,
        area_gaussian,
        approx_valid,
        approx_fallback,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_fwhm_shape_diagnostics(
    peaks: list[PeakAnnotation],
    x: np.ndarray,
    signal: np.ndarray,
    baseline: np.ndarray,
    *,
    level: float = 0.5,
    min_height_frac: float = _MIN_APEX_HEIGHT_FRAC,
) -> FwhmShapeDiagnostics:
    """Compute per-trace FWHM-derived main-peak shape diagnostics for all windows."""
    x = np.asarray(x, dtype=float).ravel()
    signal = np.asarray(signal, dtype=float)
    baseline = np.asarray(baseline, dtype=float)

    if signal.ndim != 2:
        raise ValueError(
            f"signal must be 2-D [n_trace, n_time], got shape {signal.shape}."
        )
    if baseline.ndim != 2:
        raise ValueError(
            f"baseline must be 2-D [n_trace, n_time], got shape {baseline.shape}."
        )
    if x.size != signal.shape[1]:
        raise ValueError(
            f"x length ({x.size}) must match signal.shape[1] ({signal.shape[1]})."
        )
    if signal.shape != baseline.shape:
        raise ValueError(
            f"signal and baseline must have the same shape, "
            f"got {signal.shape} vs {baseline.shape}."
        )

    n_trace = int(signal.shape[0])
    n_peak = len(peaks)
    signal_corrected = signal - baseline

    mu_trace = np.full((n_trace, n_peak), np.nan, dtype=np.float32)
    apex_height_trace = np.full((n_trace, n_peak), np.nan, dtype=np.float32)
    sigma_trace = np.full((n_trace, n_peak), np.nan, dtype=np.float32)
    alpha_trace = np.full((n_trace, n_peak), np.nan, dtype=np.float32)
    fwhm_trace = np.full((n_trace, n_peak), np.nan, dtype=np.float32)
    area_gaussian_trace = np.zeros((n_trace, n_peak), dtype=np.float32)
    area_total_trace = np.zeros((n_trace, n_peak), dtype=np.float32)
    area_residual_trace = np.zeros((n_trace, n_peak), dtype=np.float32)
    height_valid_trace = np.zeros((n_trace, n_peak), dtype=bool)
    fwhm_valid_trace = np.zeros((n_trace, n_peak), dtype=bool)
    approx_center_trace = np.full((n_trace, n_peak), np.nan, dtype=np.float32)
    approx_height_trace = np.full((n_trace, n_peak), np.nan, dtype=np.float32)
    approx_sigma_trace = np.full((n_trace, n_peak), np.nan, dtype=np.float32)
    approx_valid_trace = np.zeros((n_trace, n_peak), dtype=bool)
    approx_fallback_trace = np.zeros((n_trace, n_peak), dtype=bool)

    for peak_idx, peak in enumerate(peaks):
        lo, hi = float(peak.low), float(peak.high)
        mask = (x >= lo) & (x <= hi) & np.isfinite(x)

        if not np.any(mask):
            raise ValueError(
                f"Peak '{peak.name}' window [{lo:.4f}, {hi:.4f}] "
                f"contains no finite data points in x."
            )

        x_win = x[mask]
        y_win = signal_corrected[:, mask]
        mu_loc_j, _, _ = _height_weighted_apex(
            x_win,
            y_win,
            min_height_frac=min_height_frac,
        )
        sigma_loc_j, _, _, _, geometry_j = _shape_priors_from_fwhm(
            x_win,
            y_win,
            level=level,
            min_height_frac=min_height_frac,
        )
        total_area_j = _window_area(x_win, y_win)
        sigma_j, alpha_j = _fwhm_to_sigma_alpha(
            geometry_j.w_left,
            geometry_j.w_right,
            geometry_j.fwhm_valid,
        )
        fwhm_j = jnp.where(
            geometry_j.fwhm_valid,
            geometry_j.w_left + geometry_j.w_right,
            jnp.nan,
        )
        (
            approx_center_j,
            approx_height_j,
            approx_sigma_j,
            area_gaussian_j,
            approx_valid_j,
            approx_fallback_j,
        ) = _main_peak_approximation(
            x_win,
            y_win,
            mu_loc=mu_loc_j,
            sigma_loc=sigma_loc_j,
            geometry=geometry_j,
        )
        area_residual_j = _residual_area(total_area_j, area_gaussian_j, approx_valid_j)

        mu_trace[:, peak_idx] = np.asarray(
            jnp.where(geometry_j.fwhm_valid, geometry_j.apex_time, jnp.nan),
            dtype=np.float32,
        )
        apex_height_trace[:, peak_idx] = np.asarray(
            jnp.where(geometry_j.fwhm_valid, geometry_j.apex_height, jnp.nan),
            dtype=np.float32,
        )
        sigma_trace[:, peak_idx] = np.asarray(sigma_j, dtype=np.float32)
        alpha_trace[:, peak_idx] = np.asarray(alpha_j, dtype=np.float32)
        fwhm_trace[:, peak_idx] = np.asarray(fwhm_j, dtype=np.float32)
        area_gaussian_trace[:, peak_idx] = np.asarray(area_gaussian_j, dtype=np.float32)
        area_total_trace[:, peak_idx] = np.asarray(total_area_j, dtype=np.float32)
        area_residual_trace[:, peak_idx] = np.asarray(area_residual_j, dtype=np.float32)
        height_valid_trace[:, peak_idx] = np.asarray(
            geometry_j.height_valid, dtype=bool
        )
        fwhm_valid_trace[:, peak_idx] = np.asarray(geometry_j.fwhm_valid, dtype=bool)
        approx_center_trace[:, peak_idx] = np.asarray(approx_center_j, dtype=np.float32)
        approx_height_trace[:, peak_idx] = np.asarray(approx_height_j, dtype=np.float32)
        approx_sigma_trace[:, peak_idx] = np.asarray(approx_sigma_j, dtype=np.float32)
        approx_valid_trace[:, peak_idx] = np.asarray(approx_valid_j, dtype=bool)
        approx_fallback_trace[:, peak_idx] = np.asarray(approx_fallback_j, dtype=bool)

    return FwhmShapeDiagnostics(
        mu_trace=mu_trace,
        apex_height_trace=apex_height_trace,
        sigma_trace=sigma_trace,
        alpha_trace=alpha_trace,
        fwhm_trace=fwhm_trace,
        area_gaussian_trace=area_gaussian_trace,
        area_total_trace=area_total_trace,
        area_residual_trace=area_residual_trace,
        height_valid_trace=height_valid_trace,
        fwhm_valid_trace=fwhm_valid_trace,
        approx_center_trace=approx_center_trace,
        approx_height_trace=approx_height_trace,
        approx_sigma_trace=approx_sigma_trace,
        approx_valid_trace=approx_valid_trace,
        approx_fallback_trace=approx_fallback_trace,
    )


def build_geometric_priors(
    peaks: list[PeakAnnotation],
    x: np.ndarray,
    signal: np.ndarray,
    baseline: np.ndarray,
) -> list[GeometricPeakPriors]:
    """Build window-geometry-based priors for all annotated peak windows.

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
    list[GeometricPeakPriors]
        One entry per element of ``peaks``, preserving the input order.

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
        raise ValueError(
            f"signal must be 2-D [n_trace, n_time], got shape {signal.shape}."
        )
    if baseline.ndim != 2:
        raise ValueError(
            f"baseline must be 2-D [n_trace, n_time], got shape {baseline.shape}."
        )
    if x.size != signal.shape[1]:
        raise ValueError(
            f"x length ({x.size}) must match signal.shape[1] ({signal.shape[1]})."
        )
    if signal.shape != baseline.shape:
        raise ValueError(
            f"signal and baseline must have the same shape, "
            f"got {signal.shape} vs {baseline.shape}."
        )

    signal_corrected = signal - baseline  # [n_trace, n_time]
    results: list[GeometricPeakPriors] = []

    for peak in peaks:
        lo, hi = float(peak.low), float(peak.high)
        mask = (x >= lo) & (x <= hi) & np.isfinite(x)

        if not np.any(mask):
            raise ValueError(
                f"Peak '{peak.name}' window [{lo:.4f}, {hi:.4f}] "
                f"contains no finite data points in x."
            )

        x_win = x[mask]  # [n_win]
        y_win = signal_corrected[:, mask]  # [n_trace, n_win]

        mu_loc, mu_scale, n_valid = _height_weighted_apex(x_win, y_win)
        sigma_loc, sigma_scale, alpha_loc, alpha_scale, geometry = (
            _shape_priors_from_fwhm(
                x_win,
                y_win,
            )
        )
        (
            _,
            _,
            _,
            main_area_pt,
            approx_valid_pt,
            _,
        ) = _main_peak_approximation(
            x_win,
            y_win,
            mu_loc=mu_loc,
            sigma_loc=sigma_loc,
            geometry=geometry,
        )
        total_area_pt = _window_area(x_win, y_win)
        sh_area_pt = _residual_area(total_area_pt, main_area_pt, approx_valid_pt)

        main_area_per_trace = np.maximum(main_area_pt, _FLOAT_MIN)
        total_area_per_trace = np.maximum(total_area_pt, _FLOAT_MIN)
        if peak.mode == "artefact_doublet":
            # Shared artefact area prior: median positive residual across traces.
            valid_sh = sh_area_pt[sh_area_pt > _FLOAT_MIN]
            artefact_shoulder_area_loc = (
                float(np.median(valid_sh)) if valid_sh.size > 0 else _FLOAT_MIN
            )
        else:
            artefact_shoulder_area_loc = 0.0

        results.append(
            GeometricPeakPriors(
                mode=peak.mode,
                mu_loc=mu_loc,
                mu_scale=mu_scale / 4,
                sigma_loc=sigma_loc,
                sigma_scale=sigma_scale,
                alpha_loc=alpha_loc,
                alpha_scale=alpha_scale,
                main_area_per_trace=main_area_per_trace,
                total_area_per_trace=total_area_per_trace,
                artefact_shoulder_area_loc=artefact_shoulder_area_loc,
                window_lo=lo,
                window_hi=hi,
                n_valid_traces=n_valid,
            )
        )

    return results


def geometric_priors_to_arrays(
    priors: list[GeometricPeakPriors],
) -> dict[str, np.ndarray]:
    """Convert a list of ``GeometricPeakPriors`` to model-ready numpy arrays.

    Always includes per-trace area arrays (required by the model).

    Parameters
    ----------
    priors:
        Output of ``build_geometric_priors``.

    Returns
    -------
    dict with keys:

    - ``mu_center_loc``      [n_peak]          — apex-weighted centroid.
    - ``mu_center_scale``    [n_peak]          — centroid spread across traces.
    - ``sigma_loc``          [n_peak]          — FWHM-derived sigma prior centres.
    - ``sigma_scale``        [n_peak]          — FWHM-derived sigma prior scales.
    - ``alpha_loc``          [n_peak]          — FWHM-derived alpha prior centres.
    - ``alpha_scale``        [n_peak]          — FWHM-derived alpha prior scales.
    - ``window_lo``          [n_peak]          — window lower bounds.
    - ``window_hi``          [n_peak]          — window upper bounds.
    - ``main_area_per_trace``       [n_peak, n_trace] — per-trace Gaussian main areas.
    - ``total_area_per_trace``      [n_peak, n_trace] — per-trace total window areas.
    - ``artefact_shoulder_area_prior`` [n_artefact]   — shared artefact shoulder area prior centres.
    """
    return {
        "mu_center_loc": np.array([p.mu_loc for p in priors], dtype=np.float32),
        "mu_center_scale": np.array([p.mu_scale for p in priors], dtype=np.float32),
        "sigma_loc": np.array([p.sigma_loc for p in priors], dtype=np.float32),
        "sigma_scale": np.array([p.sigma_scale for p in priors], dtype=np.float32),
        "alpha_loc": np.array([p.alpha_loc for p in priors], dtype=np.float32),
        "alpha_scale": np.array([p.alpha_scale for p in priors], dtype=np.float32),
        "window_lo": np.array([p.window_lo for p in priors], dtype=np.float32),
        "window_hi": np.array([p.window_hi for p in priors], dtype=np.float32),
        "main_area_per_trace": np.array(
            [p.main_area_per_trace for p in priors], dtype=np.float32
        ),  # [n_peak, n_trace]
        "total_area_per_trace": np.array(
            [p.total_area_per_trace for p in priors], dtype=np.float32
        ),  # [n_peak, n_trace]
        "artefact_shoulder_area_prior": np.array(
            [
                p.artefact_shoulder_area_loc
                for p in priors
                if p.mode == "artefact_doublet"
            ],
            dtype=np.float32,
        ),  # [n_artefact]
    }


def summarise_priors(priors: list[GeometricPeakPriors]) -> str:
    """Return a human-readable summary table of computed priors.

    Parameters
    ----------
    priors:
        Output of ``build_geometric_priors``.

    Returns
    -------
    str
        Multi-line table suitable for logging or ``print()``.
    """
    lines = [
        f"{'Peak':>4}  {'mode':>17}  {'window':>18}  {'mu_loc':>8}  {'mu_scale':>8}  "
        f"{'σ_loc':>7}  {'σ_scale':>8}  "
        f"{'α_loc':>7}  {'α_scale':>8}  {'art_sh':>10}  {'ncomp':>5}  {'nvalid':>6}",
        "-" * 128,
    ]
    for i, p in enumerate(priors):
        sh_area_str = (
            f"{p.artefact_shoulder_area_loc:.3e}"
            if p.mode == "artefact_doublet"
            else "       ---"
        )
        lines.append(
            f"{i:>4}  "
            f"{p.mode:>17}  "
            f"[{p.window_lo:.3f},{p.window_hi:.3f}]  "
            f"{p.mu_loc:>8.4f}  {p.mu_scale:>8.5f}  "
            f"{p.sigma_loc:>7.5f}  {p.sigma_scale:>8.5f}  "
            f"{p.alpha_loc:>7.3f}  {p.alpha_scale:>8.4f}  "
            f"{sh_area_str:>10}  "
            f"{p.n_components:>5}  {p.n_valid_traces:>6}"
        )
    return "\n".join(lines)


__all__ = [
    "FwhmShapeDiagnostics",
    "GeometricPeakPriors",
    "build_geometric_priors",
    "compute_fwhm_shape_diagnostics",
    "geometric_priors_to_arrays",
    "summarise_priors",
]
