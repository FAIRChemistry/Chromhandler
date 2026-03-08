"""Window-geometry-based Bayesian priors for chromatographic peak fitting.

Replaces FWHM-based prior construction. All priors are derived from:

- **Window geometry** (lo, hi)          → sigma bounds
- **Sampling interval** in the window   → sigma_low via the 8-point rule
- **Apex-height-weighted centroid**     → mu_loc, mu_scale
- **Trapezoid baseline integration**    → area_loc, area_cv, area_per_trace
- **Half-window split integration**     → shoulder_area_loc (shoulder peaks only)

No FWHM analysis is required, making this approach robust at low S/N ratios.

Pipeline
--------
1. ``_median_dt``              — robust median sampling interval in a window.
2. ``_sigma_log_bounds``       — ``[log σ_min, log σ_max]`` from sampling + geometry.
3. ``_height_weighted_apex``   — height-weighted apex centroid across traces.
4. ``_window_area``            — median baseline-subtracted trapezoid area + CV.
5. ``_area_split_per_trace``   — per-trace main-peak area fraction for shoulder windows.
6. ``build_geometric_priors``  — assemble all of the above per ``PeakAnnotation``.
7. ``geometric_priors_to_arrays`` — convert list of priors to model-ready numpy arrays.

Sigma parameterisation
----------------------
The companion model samples ``log_sigma ~ Uniform(log_sigma_low, log_sigma_high)``
instead of a LogNormal.  The bounds are set so that:

- ``sigma_low  = (MIN_PEAK_POINTS × dt) × FWHM_TO_SIGMA``
  *(FWHM of at least MIN_PEAK_POINTS data points — minimum resolvable peak)*
- ``sigma_high = window_width / (2 × SIGMA_NSIGMA)``
  *(±SIGMA_NSIGMA × sigma fits within half the window)*

Shoulder area parameterisation
-------------------------------
For shoulder peaks the total window area is split at the main-peak apex into a
per-trace **main** area (``area_per_trace``) and a **shared scalar** shoulder area
(``shoulder_area_loc``).  The split uses half-window trapezoid integration — no
height-ratio lookup at fixed grid positions.

``shoulder_area_loc`` is the median estimated shoulder area across traces and becomes
the prior centre for ``A_sh_shared`` in the companion model.  This encodes the
physical constraint that a chromatographic artefact shoulder has approximately
constant absolute area across all injections, regardless of analyte concentration.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Final

import numpy as np

from .data import PeakAnnotation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FLOAT_MIN: Final = 1e-12

# FWHM → sigma conversion factor: sigma = FWHM / (2 √(2 ln 2))
_FWHM_TO_SIGMA: Final = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))  # ≈ 0.4247

# Minimum number of data points needed to define a peak (sets sigma_low)
_MIN_PEAK_POINTS: Final = 8

# Number of sigma that must fit within half the window (sets sigma_high)
_SIGMA_NSIGMA: Final = 3.0

# Minimum height fraction for apex outlier rejection (fraction of max apex height)
_MIN_APEX_HEIGHT_FRAC: Final = 0.05


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
    log_sigma_low:
        Natural log of ``sigma_min`` — sigma cannot be smaller than what
        yields a peak with fewer than ``MIN_PEAK_POINTS`` data points.
    log_sigma_high:
        Natural log of ``sigma_max`` — sigma cannot be larger than what
        places ±``SIGMA_NSIGMA`` × sigma outside half the window.
    area_loc:
        Median baseline-subtracted trapezoid area of the **full window** across
        valid traces.  Diagnostic only — the model uses per-trace main areas.
    area_cv:
        Coefficient of variation (std / mean) of the total window area.
        Diagnostic only.
    area_per_trace:
        **Main component** baseline-subtracted trapezoid area for each trace,
        shape ``[n_trace]``.

        - Single-peak windows: equals the total window area.
        - Shoulder windows: total area × main-peak fraction (shoulder contribution
          removed via half-window integration at the main-peak apex).

        Used as the per-trace prior centre for ``A_main`` in the model.
    shoulder_area_loc:
        Median estimated absolute shoulder area across traces [area units].
        ``0.0`` for single-peak windows.  Used as the prior centre for the
        shared ``A_sh_shared`` scalar in the model — encodes the constraint
        that a chromatographic artefact shoulder has constant area across all
        injections.
    n_components:
        ``1`` for a single peak; ``2`` for a peak with a shoulder annotation.
    window_lo:
        Lower bound of the peak window [time units].
    window_hi:
        Upper bound of the peak window [time units].
    n_valid_traces:
        Number of traces that contributed a valid apex to the centroid estimate.
    """

    mu_loc: float
    mu_scale: float
    log_sigma_low: float
    log_sigma_high: float
    area_loc: float
    area_cv: float
    area_per_trace: np.ndarray  # [n_trace] MAIN component area per trace
    shoulder_area_loc: float  # shared prior centre for A_sh_shared; 0.0 if no shoulder
    n_components: int
    window_lo: float
    window_hi: float
    n_valid_traces: int

    @property
    def sigma_low(self) -> float:
        """Minimum sigma in time units."""
        return math.exp(self.log_sigma_low)

    @property
    def sigma_high(self) -> float:
        """Maximum sigma in time units."""
        return math.exp(self.log_sigma_high)

    @property
    def sigma_mid(self) -> float:
        """Geometric midpoint of the LogUniform sigma prior."""
        return math.exp(0.5 * (self.log_sigma_low + self.log_sigma_high))

    def __repr__(self) -> str:
        shoulder = "yes" if self.n_components == 2 else "no"
        sh_str = (
            f", sh_area={self.shoulder_area_loc:.2e}" if self.n_components == 2 else ""
        )
        return (
            f"GeometricPeakPriors("
            f"window=[{self.window_lo:.4f}, {self.window_hi:.4f}], "
            f"mu={self.mu_loc:.4f}±{self.mu_scale:.4f}, "
            f"sigma=[{self.sigma_low:.4f}, {self.sigma_high:.4f}], "
            f"area={self.area_loc:.2e} (cv={self.area_cv:.2f}), "
            f"shoulder={shoulder}{sh_str}, "
            f"n_valid={self.n_valid_traces})"
        )


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


def _sigma_log_bounds(lo: float, hi: float, dt: float) -> tuple[float, float]:
    """Compute ``[log σ_min, log σ_max]`` from window geometry and sampling rate.

    The **lower bound** is set by requiring the peak to be resolved by at least
    ``MIN_PEAK_POINTS`` data points.  A FWHM = ``MIN_PEAK_POINTS × dt`` maps to
    ``sigma_min = FWHM × FWHM_TO_SIGMA``.

    The **upper bound** ensures that ±``SIGMA_NSIGMA`` × sigma fits within half
    the window (i.e. ``SIGMA_NSIGMA × sigma_max = window_width / 2``), so the
    complete peak remains inside the annotated region.

    Args:
        lo:  Window lower bound [time units].
        hi:  Window upper bound [time units].
        dt:  Median sampling interval [time units].

    Returns:
        ``(log_sigma_low, log_sigma_high)`` — natural-log bounds.
    """
    span = max(hi - lo, _FLOAT_MIN)
    sigma_min = max(_MIN_PEAK_POINTS * dt * _FWHM_TO_SIGMA, span / 400.0)
    sigma_max = span / (2.0 * _SIGMA_NSIGMA)
    # Guard: sigma_max must be strictly greater than sigma_min
    sigma_max = max(sigma_max, sigma_min * 2.0)
    return math.log(sigma_min), math.log(sigma_max)


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
    lo, hi = float(x_win[0]), float(x_win[-1])
    span = max(hi - lo, _FLOAT_MIN)

    if x_win.size == 0 or y_win.shape[1] == 0:
        return (lo + hi) / 2.0, span / 6.0, 0

    apex_idx = np.argmax(y_win, axis=1)  # [n_trace]
    apex_times = x_win[apex_idx]  # [n_trace]
    apex_heights = y_win[np.arange(y_win.shape[0]), apex_idx]  # [n_trace]

    max_height = (
        float(np.nanmax(apex_heights)) if np.any(np.isfinite(apex_heights)) else 0.0
    )
    threshold = max(max_height * min_height_frac, _FLOAT_MIN)
    valid = np.isfinite(apex_heights) & (apex_heights >= threshold)
    n_valid = int(np.sum(valid))

    if n_valid == 0:
        return (lo + hi) / 2.0, span / 6.0, 0

    w = apex_heights[valid]
    t = apex_times[valid]
    w_sum = float(np.sum(w))

    mu_loc = float(np.sum(w * t) / w_sum)

    if n_valid == 1:
        mu_scale = span / 6.0
    else:
        variance = float(np.sum(w * (t - mu_loc) ** 2) / w_sum)
        mu_scale = max(math.sqrt(max(variance, 0.0)), _FLOAT_MIN)
        mu_scale = max(
            mu_scale,
            float(np.median(np.abs(np.diff(x_win)))) if x_win.size > 1 else _FLOAT_MIN,
        )

    return mu_loc, mu_scale, n_valid


def _window_area(
    x_win: np.ndarray,  # [n_win]
    y_win: np.ndarray,  # [n_trace, n_win]  baseline-subtracted
) -> tuple[float, float, np.ndarray]:
    """Median baseline-subtracted trapezoid area, CV, and per-trace areas.

    Negative values (noise below baseline) are clipped to zero before
    integration.  Traces with zero or non-finite area are excluded from the
    median and CV computation.

    Args:
        x_win: 1-D time axis inside the window, shape ``[n_win]``.
        y_win: Baseline-subtracted signal, shape ``[n_trace, n_win]``.

    Returns:
        ``(area_loc, area_cv, areas)``
    """
    areas = np.array(
        [
            float(np.trapz(np.maximum(y_win[t], 0.0), x_win))
            for t in range(y_win.shape[0])
        ]
    )
    valid = np.isfinite(areas) & (areas > _FLOAT_MIN)

    if not np.any(valid):
        return _FLOAT_MIN, 0.5, areas

    valid_areas = areas[valid]
    area_loc = float(np.median(valid_areas))

    if valid_areas.size <= 1:
        return area_loc, 0.3, areas

    mad = float(np.median(np.abs(valid_areas - area_loc)))
    cv = (mad * 1.4826) / max(area_loc, _FLOAT_MIN)
    return area_loc, float(np.clip(cv, 0.1, 2.0)), areas


def _area_split_per_trace(
    x_win: np.ndarray,  # [n_win]
    y_win: np.ndarray,  # [n_trace, n_win]  baseline-subtracted
    boundary: float,  # mu_loc — apex of the main peak
    shoulder_side: int,  # +1 → shoulder to the RIGHT, -1 → shoulder to the LEFT
) -> np.ndarray:  # [n_trace]  values in [0.05, 0.95]
    """Per-trace fraction of window area belonging to the main peak component.

    Integrates the baseline-corrected signal on each side of the main-peak apex
    (``boundary``).  The main-peak fraction is the area on the opposite side of
    the apex from the shoulder.

    Args:
        x_win:         1-D time axis inside the window, shape ``[n_win]``.
        y_win:         Baseline-subtracted signal, shape ``[n_trace, n_win]``.
        boundary:      Main-peak apex position [time units].
        shoulder_side: +1 (shoulder right) or -1 (shoulder left).

    Returns:
        Array of shape ``[n_trace]`` with values clipped to ``[0.05, 0.95]``.
    """
    mask_left = x_win <= boundary
    mask_right = ~mask_left
    splits = np.empty(y_win.shape[0], dtype=np.float32)

    for t in range(y_win.shape[0]):
        y_t = np.maximum(y_win[t], 0.0)
        a_left = (
            float(np.trapz(y_t[mask_left], x_win[mask_left]))
            if mask_left.any()
            else 0.0
        )
        a_right = (
            float(np.trapz(y_t[mask_right], x_win[mask_right]))
            if mask_right.any()
            else 0.0
        )
        total = a_left + a_right
        if total < 1e-10:
            split = 0.5
        elif shoulder_side == 1:
            split = a_left / total  # shoulder right → main is left side
        else:
            split = a_right / total  # shoulder left → main is right side
        splits[t] = float(np.clip(split, 0.05, 0.95))

    return splits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
        ``peak.shoulder is not None`` → double-peak window (2 components).
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

        dt = _median_dt(x_win)
        log_sigma_low, log_sigma_high = _sigma_log_bounds(lo, hi, dt)
        mu_loc, mu_scale, n_valid = _height_weighted_apex(x_win, y_win)
        area_loc, area_cv, total_area_pt = _window_area(x_win, y_win)

        n_components = 2 if peak.shoulder is not None else 1
        if n_components == 2:
            # Split total window area into main and shoulder components.
            # main_fractions[t] ∈ [0.05, 0.95] → clipped to prevent degenerate areas.
            shoulder_side_int = 1 if peak.shoulder == "right" else -1
            main_fractions = _area_split_per_trace(
                x_win, y_win, mu_loc, shoulder_side_int
            )
            main_area_pt = total_area_pt * main_fractions  # [n_trace]
            sh_area_pt = total_area_pt * (1.0 - main_fractions)  # [n_trace]

            # Shared shoulder area prior: median over traces with positive signal.
            valid_sh = sh_area_pt[sh_area_pt > _FLOAT_MIN]
            shoulder_area_loc = (
                float(np.median(valid_sh)) if valid_sh.size > 0 else _FLOAT_MIN
            )
            area_per_trace_out = np.maximum(main_area_pt, _FLOAT_MIN)
        else:
            shoulder_area_loc = 0.0
            area_per_trace_out = total_area_pt

        results.append(
            GeometricPeakPriors(
                mu_loc=mu_loc,
                mu_scale=mu_scale / 4,
                log_sigma_low=log_sigma_low,
                log_sigma_high=log_sigma_high,
                area_loc=area_loc,
                area_cv=area_cv,
                area_per_trace=area_per_trace_out,
                shoulder_area_loc=shoulder_area_loc,
                n_components=n_components,
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
    - ``log_sigma_low``      [n_peak]          — LogUniform lower bound (log scale).
    - ``log_sigma_high``     [n_peak]          — LogUniform upper bound (log scale).
    - ``window_lo``          [n_peak]          — window lower bounds.
    - ``window_hi``          [n_peak]          — window upper bounds.
    - ``area_per_trace``     [n_peak, n_trace] — per-trace **main** component areas.
      Single-peak rows equal the total window area; shoulder-peak rows have the
      estimated shoulder contribution removed.
    - ``shoulder_area_prior``  [n_shoulder]    — shared shoulder area prior centres.
      One entry per shoulder peak, in annotation order.
    """
    return {
        "mu_center_loc": np.array([p.mu_loc for p in priors], dtype=np.float32),
        "mu_center_scale": np.array([p.mu_scale for p in priors], dtype=np.float32),
        "log_sigma_low": np.array([p.log_sigma_low for p in priors], dtype=np.float32),
        "log_sigma_high": np.array(
            [p.log_sigma_high for p in priors], dtype=np.float32
        ),
        "window_lo": np.array([p.window_lo for p in priors], dtype=np.float32),
        "window_hi": np.array([p.window_hi for p in priors], dtype=np.float32),
        "area_per_trace": np.array(
            [p.area_per_trace for p in priors], dtype=np.float32
        ),  # [n_peak, n_trace]
        "shoulder_area_prior": np.array(
            [p.shoulder_area_loc for p in priors if p.n_components == 2],
            dtype=np.float32,
        ),  # [n_shoulder]
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
        f"{'Peak':>4}  {'window':>18}  {'mu_loc':>8}  {'mu_scale':>8}  "
        f"{'σ_low':>7}  {'σ_high':>7}  {'area':>10}  {'cv':>5}  "
        f"{'sh_area':>10}  {'ncomp':>5}  {'nvalid':>6}",
        "-" * 100,
    ]
    for i, p in enumerate(priors):
        shoulder = " (sh)" if p.n_components == 2 else "     "
        sh_area_str = f"{p.shoulder_area_loc:.3e}" if p.n_components == 2 else "       ---"
        lines.append(
            f"{i:>4}{shoulder}  "
            f"[{p.window_lo:.3f},{p.window_hi:.3f}]  "
            f"{p.mu_loc:>8.4f}  {p.mu_scale:>8.5f}  "
            f"{p.sigma_low:>7.5f}  {p.sigma_high:>7.5f}  "
            f"{p.area_loc:>10.3e}  {p.area_cv:>5.2f}  "
            f"{sh_area_str:>10}  "
            f"{p.n_components:>5}  {p.n_valid_traces:>6}"
        )
    return "\n".join(lines)


__all__ = [
    "GeometricPeakPriors",
    "build_geometric_priors",
    "geometric_priors_to_arrays",
    "summarise_priors",
]
