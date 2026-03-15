"""Modular visualization for window-geometry Bayesian prior fitting.

Provides reusable plotting methods for:
- Prior visualization (loc line + scale shaded region)
- Per-trace, per-peak-window subplots
- Baseline overlay with uncertainty bands
- Signal data as scatter points
"""

from __future__ import annotations

from typing import Literal, Optional

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, ListedColormap, to_rgba
from matplotlib.patches import Patch

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation

# ---------------------------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------------------------


def _validate_hex_colors(colors: list[str], n_peak: int) -> None:
    """Validate that colors is a list of valid hex codes with correct length.

    Parameters
    ----------
    colors : list[str]
        List of hex color codes.
    n_peak : int
        Expected number of peaks/colors.

    Raises
    ------
    ValueError
        If length doesn't match n_peak or any color is not a valid hex code.
    """
    if len(colors) != n_peak:
        raise ValueError(
            f"colors must have length n_peak={n_peak}, got {len(colors)}. "
            "Provide one hex color code per peak."
        )

    for i, color in enumerate(colors):
        if not isinstance(color, str):
            raise ValueError(
                f"colors[{i}] is not a string, got {type(color).__name__}."
            )
        if not color.startswith("#"):
            raise ValueError(
                f"colors[{i}]='{color}' is not a valid hex code. "
                "Use format '#RRGGBB' (e.g., '#FF0000' for red)."
            )
        if len(color) not in (7, 9):  # #RRGGBB or #RRGGBBAA
            raise ValueError(
                f"colors[{i}]='{color}' is not a valid hex code. "
                "Use format '#RRGGBB' (e.g., '#FF0000' for red) or "
                "'#RRGGBBAA' with alpha."
            )


# ---------------------------------------------------------------------------
# Full-trace overview
# ---------------------------------------------------------------------------


def plot_trace_rows(
    time: np.ndarray,
    signal: np.ndarray,
    peaks: list[PeakAnnotation],
    *,
    figsize: Optional[tuple[float, float]] = None,
    t_min: float | None = None,
    t_max: float | None = None,
    trace_color: str = "black",
    trace_linewidth: float = 1.0,
    peak_alpha: float = 0.14,
    show_peak_legend: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot all chromatograms as stacked full-trace rows.

    Each trace is drawn on its own axes across the selected time range. Peak
    windows are shown as semi-transparent vertical spans with distinct
    ``tab10`` colors.
    """
    time_arr = np.asarray(time, dtype=float)
    signal_arr = np.asarray(signal, dtype=float)

    if time_arr.shape != signal_arr.shape:
        raise ValueError(
            f"time and signal must share shape, got {time_arr.shape} vs {signal_arr.shape}."
        )
    if time_arr.ndim != 2:
        raise ValueError("time and signal must be 2-D [n_trace, n_time].")
    if t_min is not None and t_max is not None and float(t_min) >= float(t_max):
        raise ValueError(
            f"plot_trace_rows requires t_min < t_max, got {t_min} and {t_max}."
        )

    n_trace, _ = time_arr.shape
    if figsize is None:
        figsize = (12, max(2.2 * n_trace, 3.0))

    fig, axes = plt.subplots(
        n_trace,
        1,
        figsize=figsize,
        sharex=True,
        squeeze=False,
    )
    axes_col = axes[:, 0]

    peak_colors = [plt.get_cmap("tab10")(i % 10) for i in range(len(peaks))]

    for t, ax in enumerate(axes_col):
        x_trace = time_arr[t]
        y_trace = signal_arr[t]
        finite = np.isfinite(x_trace) & np.isfinite(y_trace)
        if t_min is not None:
            finite &= x_trace >= float(t_min)
        if t_max is not None:
            finite &= x_trace <= float(t_max)

        for peak, color in zip(peaks, peak_colors, strict=False):
            span_lo = (
                peak.rt_min if t_min is None else max(float(peak.rt_min), float(t_min))
            )
            span_hi = (
                peak.rt_max if t_max is None else min(float(peak.rt_max), float(t_max))
            )
            if span_lo >= span_hi:
                continue
            ax.axvspan(span_lo, span_hi, color=color, alpha=peak_alpha, linewidth=0)
            ax.axvline(
                peak.rt_min, color=color, alpha=0.55, linewidth=0.9, linestyle="--"
            )
            ax.axvline(
                peak.rt_max,
                color=color,
                alpha=0.55,
                linewidth=0.9,
                linestyle="--",
            )

        if np.any(finite):
            ax.plot(
                x_trace[finite],
                y_trace[finite],
                color=trace_color,
                linewidth=trace_linewidth,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No finite data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
                color="red",
            )

        ax.set_ylabel(f"T{t}", fontsize=8)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.tick_params(labelsize=8)
        if t_min is not None or t_max is not None:
            ax.set_xlim(
                float(t_min) if t_min is not None else None,
                float(t_max) if t_max is not None else None,
            )

    axes_col[-1].set_xlabel("Time", fontsize=9)

    if show_peak_legend and peaks:
        legend_handles = [
            Patch(
                facecolor=color,
                edgecolor=color,
                alpha=peak_alpha + 0.18,
                label=peak.molecule_id,
            )
            for peak, color in zip(peaks, peak_colors, strict=False)
        ]
        axes_col[0].legend(
            handles=legend_handles, fontsize=8, loc="best", ncol=min(4, len(peaks))
        )

    fig.suptitle(
        f"Chromatogram Overview: {n_trace} traces", fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    return fig, axes_col


# ---------------------------------------------------------------------------
# Prior shading helper
# ---------------------------------------------------------------------------


def add_prior_shading(
    ax: plt.Axes,
    x: np.ndarray,
    loc: np.ndarray,
    scale: np.ndarray,
    *,
    label: str = "",
    color: str = "C0",
    alpha: float = 0.3,
    linewidth: float = 1.5,
) -> None:
    """Plot prior loc as line with scale as shaded region.

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    x : np.ndarray
        X-axis (1-D or broadcast-compatible shape).
    loc : np.ndarray
        Prior location (mean/center line).
    scale : np.ndarray
        Prior scale (std/uncertainty — defines shaded band).
    label : str
        Legend label for the line.
    color : str
        Line and fill color.
    alpha : float
        Alpha for shaded region (default 0.3).
    linewidth : float
        Line width for center line.
    """
    loc_arr = np.asarray(loc)
    scale_arr = np.asarray(scale)

    # Ensure x is 1-D
    x_arr = np.asarray(x).ravel()

    # Plot center line
    ax.plot(x_arr, loc_arr, color=color, linewidth=linewidth, label=label)

    # Plot shaded region ±scale
    upper = loc_arr + scale_arr
    lower = loc_arr - scale_arr
    ax.fill_between(x_arr, lower, upper, color=color, alpha=alpha)


# ---------------------------------------------------------------------------
# Peak window annotation helper
# ---------------------------------------------------------------------------


def add_peak_window_bounds(
    ax: plt.Axes,
    peak: PeakAnnotation,
    *,
    color: str = "red",
    linestyle: str = "--",
    alpha: float = 0.5,
    linewidth: float = 1.0,
) -> None:
    """Add vertical lines marking peak window bounds.

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    peak : PeakAnnotation
        Peak with low/high bounds.
    color : str
        Line color.
    linestyle : str
        Line style ("--", "-", ":", etc.).
    alpha : float
        Line transparency.
    linewidth : float
        Line width.
    """
    ax.axvline(
        peak.rt_min, color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth
    )
    ax.axvline(
        peak.rt_max, color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth
    )


# ---------------------------------------------------------------------------
# Vertical prior helper
# ---------------------------------------------------------------------------


def add_vertical_prior_band(
    ax: plt.Axes,
    loc: float,
    scale: float,
    *,
    label: str = "",
    color: str = "tab:orange",
    alpha: float = 0.15,
    linewidth: float = 1.5,
    linestyle: str = "-",
) -> None:
    """Plot a vertical prior location line with a shaded ±scale band."""
    loc_f = float(loc)
    scale_f = max(float(scale), 0.0)
    ax.axvspan(loc_f - scale_f, loc_f + scale_f, color=color, alpha=alpha)
    ax.axvline(
        loc_f,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )


# ---------------------------------------------------------------------------
# Scatter sizing helper
# ---------------------------------------------------------------------------


def _marker_sizes_from_values(
    values: np.ndarray,
    *,
    size_min: float = 10.0,
    size_max: float = 140.0,
) -> np.ndarray:
    """Map positive values to marker sizes for scatter plots."""
    values_arr = np.asarray(values, dtype=float).reshape(-1)
    sizes = np.full(values_arr.shape, 0.5 * (size_min + size_max), dtype=float)
    finite_pos = values_arr[np.isfinite(values_arr) & (values_arr > 0.0)]
    if finite_pos.size == 0:
        return sizes

    vmin = float(np.min(finite_pos))
    vmax = float(np.max(finite_pos))
    if vmax <= vmin + 1e-12:
        return sizes

    values_clip = np.clip(values_arr, vmin, vmax)
    scaled = (values_clip - vmin) / (vmax - vmin)
    return size_min + scaled * (size_max - size_min)


# ---------------------------------------------------------------------------
# 2-D prior helper
# ---------------------------------------------------------------------------


def add_sigma_alpha_prior_density(
    ax: plt.Axes,
    *,
    alpha_loc: float,
    alpha_scale: float,
    sigma_loc: float,
    sigma_scale: float,
    x_data: Optional[np.ndarray] = None,
    y_data: Optional[np.ndarray] = None,
    cmap: "str | Colormap" = "viridis",
    n_levels: int = 4,
    linecolor: str = "white",
    n_grid: int = 220,
    set_limits: bool = True,
) -> None:
    """Plot a diagonal 2-D Gaussian prior as filled quartile contour bands."""
    alpha_loc_f = float(alpha_loc)
    alpha_scale_f = max(float(alpha_scale), 1e-6)
    sigma_loc_f = float(sigma_loc)
    sigma_scale_f = max(float(sigma_scale), 1e-6)

    x_span = 4.5 * alpha_scale_f
    y_span = 4.5 * sigma_scale_f
    x_min = alpha_loc_f - x_span
    x_max = alpha_loc_f + x_span
    y_min = sigma_loc_f - y_span
    y_max = sigma_loc_f + y_span

    if x_data is not None:
        x_vals = np.asarray(x_data, dtype=float).reshape(-1)
        x_vals = x_vals[np.isfinite(x_vals)]
        if x_vals.size > 0:
            x_pad = max(0.2 * np.ptp(x_vals), 0.1 * alpha_scale_f, 1e-3)
            x_min = min(x_min, float(np.min(x_vals) - x_pad))
            x_max = max(x_max, float(np.max(x_vals) + x_pad))
    if y_data is not None:
        y_vals = np.asarray(y_data, dtype=float).reshape(-1)
        y_vals = y_vals[np.isfinite(y_vals)]
        if y_vals.size > 0:
            y_pad = max(0.2 * np.ptp(y_vals), 0.1 * sigma_scale_f, 1e-4)
            y_min = min(y_min, float(np.min(y_vals) - y_pad))
            y_max = max(y_max, float(np.max(y_vals) + y_pad))

    x_grid = np.linspace(x_min, x_max, n_grid)
    y_grid = np.linspace(y_min, y_max, n_grid)
    xx, yy = np.meshgrid(x_grid, y_grid)

    z_alpha = (xx - alpha_loc_f) / alpha_scale_f
    z_sigma = (yy - sigma_loc_f) / sigma_scale_f
    density = np.exp(-0.5 * (z_alpha**2 + z_sigma**2)) / (
        2.0 * np.pi * alpha_scale_f * sigma_scale_f
    )

    # Chi-square(df=2) quantiles give equal probability mass per band.
    # r^2_p = -2 log(1-p) maps mass fraction to squared Mahalanobis radius.
    mass_levels = np.linspace(
        1.0 / (n_levels + 1), 1.0 - 1.0 / (n_levels + 1), n_levels
    )
    mass_levels[-1] = min(float(mass_levels[-1]), 0.95)
    r2 = -2.0 * np.log(1.0 - mass_levels)
    density_peak = 1.0 / (2.0 * np.pi * alpha_scale_f * sigma_scale_f)
    density_levels = density_peak * np.exp(-0.5 * r2)
    contour_levels = np.sort(
        np.concatenate([density_levels[::-1], np.array([density_peak])])
    )

    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    if isinstance(cmap_obj, ListedColormap):
        colors = [cmap_obj(i / max(n_levels - 1, 1)) for i in range(n_levels)]
    else:
        colors = cmap_obj(np.linspace(0.20, 0.92, n_levels))

    ax.contourf(
        xx,
        yy,
        density,
        levels=contour_levels,
        colors=colors,
        alpha=0.65,
        antialiased=True,
    )
    ax.contour(
        xx,
        yy,
        density,
        levels=contour_levels[:-1],
        colors=linecolor,
        linewidths=0.7,
        alpha=0.75,
    )
    if set_limits:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)


def _check_peak_vector(
    values: Optional[np.ndarray],
    name: str,
    n_peak: int,
) -> Optional[np.ndarray]:
    """Validate an optional per-peak vector argument."""
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.shape[0] != n_peak:
        raise ValueError(f"{name} must have length {n_peak}, got shape {arr.shape}.")
    return arr


def _limits_from_values(
    values: np.ndarray,
    *,
    pad_frac: float,
    pad_floor: float,
) -> tuple[float, float] | None:
    """Return padded limits from a 1-D numeric array, or ``None`` if empty."""
    values_arr = np.asarray(values, dtype=float).reshape(-1)
    finite = values_arr[np.isfinite(values_arr)]
    if finite.size == 0:
        return None

    lo = float(np.min(finite))
    hi = float(np.max(finite))
    span = hi - lo
    pad = max(pad_frac * span, pad_floor) if span > 1e-12 else pad_floor
    return lo - pad, hi + pad


def _merge_limits(
    limits: list[tuple[float, float] | None],
    *,
    default: tuple[float, float],
) -> tuple[float, float]:
    """Merge optional limit pairs into one global visible range."""
    valid_limits = [limit for limit in limits if limit is not None]
    if not valid_limits:
        return default

    lo = min(limit[0] for limit in valid_limits)
    hi = max(limit[1] for limit in valid_limits)
    if hi <= lo:
        lo, hi = default
    return lo, hi


def _resolve_prior_colormap(
    prior_cmap: "str | Colormap" = "viridis",
    prior_colors: Optional[list[str]] = None,
) -> "tuple[Colormap, int]":
    """Return (Colormap, n_levels) from either a colormap name or explicit color list."""
    if prior_colors is not None:
        if len(prior_colors) < 2:
            raise ValueError("prior_colors must contain at least 2 colors.")
        cmap = ListedColormap([to_rgba(c) for c in prior_colors])
        return cmap, len(prior_colors)
    cmap_obj = plt.get_cmap(prior_cmap) if isinstance(prior_cmap, str) else prior_cmap
    return cmap_obj, 4


# ---------------------------------------------------------------------------
# Baseline overlay helper
# ---------------------------------------------------------------------------


def add_baseline_to_axes(
    ax: plt.Axes,
    x: np.ndarray,
    baseline_intercept: float,
    baseline_slope: float,
    intercept_scale: float,
    slope_scale: float,
) -> None:
    """Plot baseline prior on a single axes with uncertainty band.

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    x : np.ndarray
        Time/x-axis (1-D).
    baseline_intercept : float
        Intercept prior loc.
    baseline_slope : float
        Slope prior loc.
    intercept_scale : float
        Intercept prior scale.
    slope_scale : float
        Slope prior scale.
    """
    x_arr = np.asarray(x).ravel()

    # Baseline: y = intercept + slope * x
    baseline_loc = baseline_intercept + baseline_slope * x_arr
    baseline_scale = np.sqrt(intercept_scale**2 + (slope_scale * x_arr) ** 2)

    # Plot baseline with uncertainty
    add_prior_shading(
        ax,
        x_arr,
        baseline_loc,
        baseline_scale,
        label="Baseline prior",
        color="tab:blue",
        alpha=0.5,
        linewidth=1.5,
    )


# ---------------------------------------------------------------------------
# Prior Gaussian helper
# ---------------------------------------------------------------------------


def _gaussian_peak_curve_from_sigma(
    x: np.ndarray,
    center: float,
    height: float,
    sigma: float,
) -> np.ndarray:
    """Gaussian peak approximation from center, height, and sigma."""
    x_arr = np.asarray(x, dtype=float).ravel()
    center_f = float(center)
    height_f = float(height)
    sigma_f = float(sigma)

    if (
        not np.isfinite(center_f)
        or not np.isfinite(height_f)
        or not np.isfinite(sigma_f)
        or height_f <= 0.0
        or sigma_f <= 0.0
    ):
        return np.full_like(x_arr, np.nan, dtype=float)

    z = (x_arr - center_f) / sigma_f
    return height_f * np.exp(-0.5 * z**2)


# ---------------------------------------------------------------------------
# Main visualization: prior traces
# ---------------------------------------------------------------------------


def plot_prior_traces(
    time: np.ndarray,
    signal: np.ndarray,
    peaks: list[PeakAnnotation],
    baseline_intercept: np.ndarray,
    baseline_slope: np.ndarray,
    baseline_intercept_scale: np.ndarray,
    baseline_slope_scale: np.ndarray,
    apex_loc: Optional[np.ndarray] = None,
    apex_scale: Optional[np.ndarray] = None,
    approx_apex_trace: Optional[np.ndarray] = None,
    approx_height_trace: Optional[np.ndarray] = None,
    approx_sigma_trace: Optional[np.ndarray] = None,
    approx_valid_trace: Optional[np.ndarray] = None,
    approx_fallback_trace: Optional[np.ndarray] = None,
    *,
    show_baseline: bool = True,
    show_apex_prior: bool = True,
    show_gaussian_prior_peak: bool = True,
    show_peak_bounds: bool = True,
    figsize: Optional[tuple[float, float]] = None,
    cmap: str = "viridis",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot prior traces: subplots[trace, peak_window].

    Raw signal as gray scatter, baseline prior overlay (optional), peak bounds (optional).

    Parameters
    ----------
    time : np.ndarray
        Time matrix, shape [n_trace, n_time]. May contain NaN padding.
    signal : np.ndarray
        Signal matrix, shape [n_trace, n_time]. May contain NaN padding.
    peaks : list[PeakAnnotation]
        Peak window definitions (low, high bounds).
    baseline_intercept : np.ndarray
        Baseline intercept prior loc, shape [n_trace].
    baseline_slope : np.ndarray
        Baseline slope prior loc, shape [n_trace].
    baseline_intercept_scale : np.ndarray
        Baseline intercept prior scale, shape [n_trace].
    baseline_slope_scale : np.ndarray
        Baseline slope prior scale, shape [n_trace].
    apex_loc : np.ndarray or None
        Peak apex prior locations, shape [n_peak]. If provided, drawn as
        vertical lines in each corresponding peak window.
    apex_scale : np.ndarray or None
        Peak apex prior scales, shape [n_peak]. If provided, drawn as
        vertical shaded bands spanning ``apex_loc ± apex_scale``.
    approx_apex_trace : np.ndarray or None
        Dense Gaussian-approximation apex locations, shape [n_trace, n_peak].
    approx_height_trace : np.ndarray or None
        Dense Gaussian-approximation heights, shape [n_trace, n_peak].
    approx_sigma_trace : np.ndarray or None
        Dense Gaussian-approximation sigmas, shape [n_trace, n_peak].
    approx_valid_trace : np.ndarray or None
        Boolean mask for traces that contribute a Gaussian approximation,
        shape [n_trace, n_peak].
    approx_fallback_trace : np.ndarray or None
        Boolean mask for traces using the low-height fallback approximation,
        shape [n_trace, n_peak].
    show_baseline : bool
        If True, overlay baseline prior with uncertainty band.
    show_apex_prior : bool
        If True and ``apex_loc`` / ``apex_scale`` are provided,
        overlay the apex prior as a vertical line with a shaded band.
    show_gaussian_prior_peak : bool
        If True and all approximation arrays are provided, overlay the
        baseline-plus-Gaussian prior peak approximation.
    show_peak_bounds : bool
        If True, add vertical dashed lines at peak window bounds.
    figsize : tuple or None
        Figure size (default: auto-scale based on grid).
    cmap : str
        Colormap for trace highlighting (not used in simple version, reserved
        for future enhancement).

    Returns
    -------
    fig : plt.Figure
        The figure object.
    axes : np.ndarray
        2-D array of axes, shape [n_trace, n_peak].
    """
    time_arr = np.asarray(time, dtype=float)
    signal_arr = np.asarray(signal, dtype=float)

    n_trace, n_time = time_arr.shape
    n_peak = len(peaks)

    # Default figsize: scale by grid size
    if figsize is None:
        figsize = (4 * n_peak, 3 * n_trace)

    fig, axes = plt.subplots(
        n_trace,
        n_peak,
        figsize=figsize,
        sharex=False,
        sharey=False,
        squeeze=False,
    )

    # Prepare baseline arrays
    b_intercept = np.asarray(baseline_intercept, dtype=float)
    b_slope = np.asarray(baseline_slope, dtype=float)
    b_intercept_scale = np.asarray(baseline_intercept_scale, dtype=float)
    b_slope_scale = np.asarray(baseline_slope_scale, dtype=float)
    apex_loc_arr = None if apex_loc is None else np.asarray(apex_loc, dtype=float)
    apex_scale_arr = None if apex_scale is None else np.asarray(apex_scale, dtype=float)
    approx_apex_arr = (
        None
        if approx_apex_trace is None
        else np.asarray(approx_apex_trace, dtype=float)
    )
    approx_height_arr = (
        None
        if approx_height_trace is None
        else np.asarray(approx_height_trace, dtype=float)
    )
    approx_sigma_arr = (
        None
        if approx_sigma_trace is None
        else np.asarray(approx_sigma_trace, dtype=float)
    )
    approx_valid_arr = (
        None
        if approx_valid_trace is None
        else np.asarray(approx_valid_trace, dtype=bool)
    )
    approx_fallback_arr = (
        None
        if approx_fallback_trace is None
        else np.asarray(approx_fallback_trace, dtype=bool)
    )

    if (apex_loc_arr is None) != (apex_scale_arr is None):
        raise ValueError(
            "apex_loc and apex_scale must either both be provided or both be omitted."
        )
    if apex_loc_arr is not None and apex_loc_arr.shape[0] != n_peak:
        raise ValueError(
            f"apex_loc must have length {n_peak}, got shape {apex_loc_arr.shape}."
        )
    if apex_scale_arr is not None and apex_scale_arr.shape[0] != n_peak:
        raise ValueError(
            f"apex_scale must have length {n_peak}, got shape {apex_scale_arr.shape}."
        )
    gaussian_inputs = [
        approx_apex_arr,
        approx_height_arr,
        approx_sigma_arr,
        approx_valid_arr,
        approx_fallback_arr,
    ]
    has_gaussian_inputs = all(arr is not None for arr in gaussian_inputs)
    if any(arr is not None for arr in gaussian_inputs) and not has_gaussian_inputs:
        raise ValueError(
            "approx_apex_trace, approx_height_trace, approx_sigma_trace, "
            "approx_valid_trace, and approx_fallback_trace must all be provided together."
        )
    if has_gaussian_inputs:
        expected_shape = (n_trace, n_peak)
        for name, arr in (
            ("approx_apex_trace", approx_apex_arr),
            ("approx_height_trace", approx_height_arr),
            ("approx_sigma_trace", approx_sigma_arr),
            ("approx_valid_trace", approx_valid_arr),
            ("approx_fallback_trace", approx_fallback_arr),
        ):
            if arr is not None and arr.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {arr.shape}."
                )

    # Plot each (trace, peak) subplot
    for t in range(n_trace):
        for p, peak in enumerate(peaks):
            ax = axes[t, p]

            # Get data for this trace
            x_trace = time_arr[t, :]
            y_trace = signal_arr[t, :]

            # Extract peak window
            mask = (x_trace >= peak.rt_min) & (x_trace <= peak.rt_max)
            x_window = x_trace[mask]
            y_window = y_trace[mask]

            # Skip if no valid data in window
            if len(x_window) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="red",
                )
                ax.set_title(f"{peak.molecule_id} (trace {t})", fontsize=9)
                continue

            # Plot raw signal as gray scatter
            finite_mask = np.isfinite(x_window) & np.isfinite(y_window)
            ax.scatter(
                x_window[finite_mask],
                y_window[finite_mask],
                s=30,
                alpha=0.5,
                color="gray",
                label="Raw signal",
            )

            # Overlay baseline prior if requested
            if show_baseline:
                add_baseline_to_axes(
                    ax,
                    x_window,
                    b_intercept[t],
                    b_slope[t],
                    b_intercept_scale[t],
                    b_slope_scale[t],
                )

            if show_gaussian_prior_peak and has_gaussian_inputs:
                assert approx_apex_arr is not None
                assert approx_height_arr is not None
                assert approx_sigma_arr is not None
                assert approx_valid_arr is not None
                assert approx_fallback_arr is not None
                if bool(approx_valid_arr[t, p]):
                    peak_curve = _gaussian_peak_curve_from_sigma(
                        x_window,
                        approx_apex_arr[t, p],
                        approx_height_arr[t, p],
                        approx_sigma_arr[t, p],
                    )
                    baseline_center = b_intercept[t] + b_slope[t] * x_window
                    total_curve = baseline_center + peak_curve
                    finite_curve = np.isfinite(total_curve)
                    if np.any(finite_curve):
                        is_fallback = bool(approx_fallback_arr[t, p])
                        ax.plot(
                            x_window[finite_curve],
                            total_curve[finite_curve],
                            color="tab:green",
                            linestyle="--" if is_fallback else "-",
                            linewidth=1.5,
                            label=(
                                "Fallback Gaussian prior peak"
                                if (is_fallback and t == 0 and p == 0)
                                else (
                                    "Gaussian prior peak"
                                    if ((not is_fallback) and t == 0 and p == 0)
                                    else ""
                                )
                            ),
                        )

            if (
                show_apex_prior
                and apex_loc_arr is not None
                and apex_scale_arr is not None
            ):
                add_vertical_prior_band(
                    ax,
                    apex_loc_arr[p],
                    apex_scale_arr[p],
                    label="apex prior" if (t == 0 and p == 0) else "",
                    color="tab:orange",
                    alpha=0.18,
                    linewidth=1.5,
                    linestyle="-",
                )

            # Add peak window bounds if requested
            if show_peak_bounds:
                add_peak_window_bounds(ax, peak, color="red", alpha=0.4, linewidth=1.0)

            # Labels and formatting
            ax.set_title(f"{peak.molecule_id} (trace {t})", fontsize=9)
            if p == 0:
                ax.set_ylabel("Signal", fontsize=8)
            if t == n_trace - 1:
                ax.set_xlabel("Time", fontsize=8)

            ax.grid(True, alpha=0.3, linestyle="--")
            ax.tick_params(labelsize=7)

    # Add legend to first axes
    if n_trace > 0 and n_peak > 0:
        axes[0, 0].legend(fontsize=8, loc="best")

    fig.suptitle(
        f"Prior Traces: {n_trace} traces × {n_peak} peak windows",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()

    return fig, axes


# ---------------------------------------------------------------------------
# FWHM Sigma/Alpha Diagnostics
# ---------------------------------------------------------------------------


def plot_sigma_alpha_scatter(
    peaks: list[PeakAnnotation],
    sigma_trace: np.ndarray,
    alpha_trace: np.ndarray,
    valid_trace: np.ndarray,
    *,
    apex_height_trace: Optional[np.ndarray] = None,
    sigma_loc: Optional[np.ndarray] = None,
    sigma_scale: Optional[np.ndarray] = None,
    alpha_loc: Optional[np.ndarray] = None,
    alpha_scale: Optional[np.ndarray] = None,
    figsize: Optional[tuple[float, float]] = None,
    cmap: str = "viridis",
    prior_colors: Optional[list[str]] = None,
    prior_linecolor: str = "white",
    colorize_by: Literal[None, "sample_id", "subset"] = None,
    sample_ids: Optional[list[str]] = None,
    subset_ids: Optional[list[str]] = None,
    label_fontsize: float = 9,
    title_fontsize: float = 10,
    tick_fontsize: float = 8,
    spine_linewidth: float = 0.8,
    marker_size: float = 40,
    marker_linewidth: float = 0.9,
    transparent: bool = True,
    show_prior_density: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot per-trace FWHM-derived ``sigma`` vs ``alpha`` with 2-D prior density.

    When *colorize_by* is ``"sample_id"``, scatter points are colored by
    *sample_ids* (length n_trace) using the turbo colormap.  When
    ``"subset"``, points are colored by *subset_ids* (length n_trace).
    When ``None``, points use transparent fill with black outline.

    A shared legend is drawn beneath the figure when *colorize_by* is set.
    """
    sigma_arr = np.asarray(sigma_trace, dtype=float)
    alpha_arr = np.asarray(alpha_trace, dtype=float)
    valid_arr = np.asarray(valid_trace, dtype=bool)

    if sigma_arr.shape != alpha_arr.shape or sigma_arr.shape != valid_arr.shape:
        raise ValueError("sigma_trace, alpha_trace, and valid_trace must share shape.")
    if sigma_arr.ndim != 2:
        raise ValueError("sigma_trace must be 2-D [n_trace, n_peak].")

    n_trace, n_peak = sigma_arr.shape
    if len(peaks) != n_peak:
        raise ValueError(
            f"peaks length must match n_peak={n_peak}, got {len(peaks)} peaks."
        )

    if colorize_by == "sample_id":
        if sample_ids is None or len(sample_ids) != n_trace:
            raise ValueError(
                f"sample_ids must have length n_trace={n_trace} when colorize_by='sample_id'."
            )
        labels_for_color = sample_ids
    elif colorize_by == "subset":
        if subset_ids is None or len(subset_ids) != n_trace:
            raise ValueError(
                f"subset_ids must have length n_trace={n_trace} when colorize_by='subset'."
            )
        labels_for_color = subset_ids
    else:
        labels_for_color = None

    # Map unique label -> turbo color when coloring
    label_to_color: Optional[dict[str, tuple[float, float, float, float]]] = None
    if labels_for_color is not None:
        unique_labels = list(dict.fromkeys(str(lab) for lab in labels_for_color))
        n_unique = len(unique_labels)
        turbo = plt.get_cmap("turbo")
        t_vals = np.linspace(0.0, 1.0, n_unique) if n_unique > 1 else np.array([0.5])
        label_to_color = {lab: turbo(t_vals[i]) for i, lab in enumerate(unique_labels)}

    apex_height_arr = None
    if apex_height_trace is not None:
        apex_height_arr = np.asarray(apex_height_trace, dtype=float)
        if apex_height_arr.shape != sigma_arr.shape:
            raise ValueError(
                "apex_height_trace must share shape with sigma_trace if provided."
            )

    sigma_loc_arr = _check_peak_vector(sigma_loc, "sigma_loc", n_peak)
    sigma_scale_arr = _check_peak_vector(sigma_scale, "sigma_scale", n_peak)
    alpha_loc_arr = _check_peak_vector(alpha_loc, "alpha_loc", n_peak)
    alpha_scale_arr = _check_peak_vector(alpha_scale, "alpha_scale", n_peak)

    if (sigma_loc_arr is None) != (sigma_scale_arr is None):
        raise ValueError("sigma_loc and sigma_scale must both be provided or omitted.")
    if (alpha_loc_arr is None) != (alpha_scale_arr is None):
        raise ValueError("alpha_loc and alpha_scale must both be provided or omitted.")

    prior_cmap_obj, prior_n_levels = _resolve_prior_colormap(
        prior_cmap=cmap,
        prior_colors=prior_colors,
    )

    panel_valid_masks: list[np.ndarray] = []
    panel_alpha: list[np.ndarray] = []
    panel_sigma: list[np.ndarray] = []
    x_lim_per_panel: list[tuple[float, float]] = []
    y_lim_per_panel: list[tuple[float, float]] = []

    for peak_idx in range(n_peak):
        valid_peak = (
            valid_arr[:, peak_idx]
            & np.isfinite(sigma_arr[:, peak_idx])
            & np.isfinite(alpha_arr[:, peak_idx])
        )
        alpha_peak = alpha_arr[valid_peak, peak_idx]
        sigma_peak = sigma_arr[valid_peak, peak_idx]
        panel_valid_masks.append(valid_peak)
        panel_alpha.append(alpha_peak)
        panel_sigma.append(sigma_peak)

        x_cands: list[tuple[float, float] | None] = [
            _limits_from_values(alpha_peak, pad_frac=0.20, pad_floor=1e-3)
        ]
        y_cands: list[tuple[float, float] | None] = [
            _limits_from_values(sigma_peak, pad_frac=0.20, pad_floor=1e-4)
        ]

        if (
            alpha_loc_arr is not None
            and alpha_scale_arr is not None
            and sigma_loc_arr is not None
            and sigma_scale_arr is not None
            and np.isfinite(alpha_loc_arr[peak_idx])
            and np.isfinite(alpha_scale_arr[peak_idx])
            and np.isfinite(sigma_loc_arr[peak_idx])
            and np.isfinite(sigma_scale_arr[peak_idx])
        ):
            prior_x_min = float(alpha_loc_arr[peak_idx] - 4 * alpha_scale_arr[peak_idx])
            prior_x_max = float(alpha_loc_arr[peak_idx] + 4 * alpha_scale_arr[peak_idx])
            prior_y_min = float(sigma_loc_arr[peak_idx] - 4 * sigma_scale_arr[peak_idx])
            prior_y_max = float(sigma_loc_arr[peak_idx] + 4 * sigma_scale_arr[peak_idx])
            x_cands.append((prior_x_min, prior_x_max))
            y_cands.append((prior_y_min, prior_y_max))

        x_lim_per_panel.append(_merge_limits(x_cands, default=(-1.0, 1.0)))
        y_lim_per_panel.append(_merge_limits(y_cands, default=(0.0, 1.0)))

    if figsize is None:
        figsize = (4.3 * n_peak, 4.2)

    fig, axes = plt.subplots(
        1,
        n_peak,
        figsize=figsize,
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    bg = "none" if transparent else "white"
    fig.patch.set_facecolor(bg)
    axes = axes.reshape(1, n_peak)
    for ax in axes.flat:
        ax.patch.set_facecolor(bg)

    for peak_idx, peak in enumerate(peaks):
        ax = axes[0, peak_idx]
        valid_peak = panel_valid_masks[peak_idx]
        alpha_peak = panel_alpha[peak_idx]
        sigma_peak = panel_sigma[peak_idx]

        if (
            show_prior_density
            and alpha_loc_arr is not None
            and alpha_scale_arr is not None
            and sigma_loc_arr is not None
            and sigma_scale_arr is not None
            and np.isfinite(alpha_loc_arr[peak_idx])
            and np.isfinite(alpha_scale_arr[peak_idx])
            and np.isfinite(sigma_loc_arr[peak_idx])
            and np.isfinite(sigma_scale_arr[peak_idx])
        ):
            add_sigma_alpha_prior_density(
                ax,
                alpha_loc=float(alpha_loc_arr[peak_idx]),
                alpha_scale=float(alpha_scale_arr[peak_idx]),
                sigma_loc=float(sigma_loc_arr[peak_idx]),
                sigma_scale=float(sigma_scale_arr[peak_idx]),
                x_data=alpha_peak,
                y_data=sigma_peak,
                cmap=prior_cmap_obj,
                n_levels=prior_n_levels,
                linecolor=prior_linecolor,
                set_limits=False,
            )

        if np.any(valid_peak):
            alpha_scatter = 0.2
            trace_indices = np.where(valid_peak)[0]
            if label_to_color is not None and labels_for_color is not None:
                edge_colors = [
                    label_to_color[str(labels_for_color[t])] for t in trace_indices
                ]
                face_colors = [
                    (*edge_colors[i][:3], 0.35) for i in range(len(edge_colors))
                ]
            else:
                edge_colors = (0.15, 0.15, 0.15, 0.9)
                face_colors = (1, 1, 1, alpha_scatter)

            if apex_height_arr is not None:
                heights_peak = apex_height_arr[valid_peak, peak_idx]
                # Scale height-derived sizes proportionally to marker_size baseline
                sizes_peak = _marker_sizes_from_values(heights_peak) * (marker_size / 40)
                ax.scatter(
                    alpha_peak,
                    sigma_peak,
                    s=sizes_peak,
                    facecolors=face_colors,
                    edgecolors=edge_colors,
                    linewidths=marker_linewidth,
                    alpha=None,
                    zorder=4,
                )
            else:
                ax.scatter(
                    alpha_peak,
                    sigma_peak,
                    s=marker_size,
                    facecolors=face_colors,
                    edgecolors=edge_colors,
                    linewidths=marker_linewidth,
                    alpha=None,
                    zorder=4,
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No valid FWHM traces",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=label_fontsize,
                color="0.4",
            )

        ax.set_title(peak.molecule_id, fontsize=title_fontsize)
        ax.set_xlabel(r"$\alpha$ [–]", fontsize=label_fontsize)
        if peak_idx == 0:
            ax.set_ylabel(r"$\sigma$ [min]", fontsize=label_fontsize)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=tick_fontsize, width=spine_linewidth)
        for spine in ax.spines.values():
            spine.set_linewidth(spine_linewidth)
        ax.set_xlim(*x_lim_per_panel[peak_idx])
        ax.set_ylim(*y_lim_per_panel[peak_idx])

    if label_to_color is not None:
        legend_handles = [
            Patch(facecolor=c[:3], edgecolor="black", linewidth=0.9, label=lab)
            for lab, c in label_to_color.items()
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(len(legend_handles), 8),
            frameon=True,
            fontsize=8,
        )

    fig.tight_layout(rect=[0, 0.08, 1, 1] if label_to_color is not None else None)
    return fig, axes


# ---------------------------------------------------------------------------
# HDI (95% Credible Interval) Plotting Helpers
# ---------------------------------------------------------------------------


# def _compute_skew_normal_component(
#     x: jnp.ndarray,
#     xi: jnp.ndarray,
#     sigma: jnp.ndarray,
#     alpha: jnp.ndarray,
#     area: jnp.ndarray,
# ) -> np.ndarray:
#     """Compute area-scaled skew-normal PDF values.

#     Parameters
#     ----------
#     x : jnp.ndarray
#         Time axis [n_window]
#     xi : jnp.ndarray
#         Location parameter [n_total]
#     sigma : jnp.ndarray
#         Scale parameter [n_total]
#     alpha : jnp.ndarray
#         Skewness parameter [n_total]
#     area : jnp.ndarray
#         Area scaling factor [n_total]

#     Returns
#     -------
#     np.ndarray
#         Component signal [n_total, n_window]
#     """
#     x_broad = x[None, :]  # [1, n_window]
#     xi_broad = xi[:, None]  # [n_total, 1]
#     sigma_broad = sigma[:, None]  # [n_total, 1]
#     alpha_broad = alpha[:, None]  # [n_total, 1]

#     # Compute log PDF (sigma normalization already included)
#     sigma_safe = jnp.maximum(sigma_broad, 1e-6)
#     z = (x_broad - xi_broad) / sigma_safe
#     log_pdf = (
#         jnp.log(2.0)
#         - jnp.log(sigma_safe)
#         - 0.5 * z**2
#         - 0.5 * jnp.log(2.0 * jnp.pi)
#         + jnp.log(jax.scipy.special.ndtr(alpha_broad * z))
#     )
#     # Exponentiate directly (sigma normalization already in log_pdf)
#     pdf = jnp.exp(log_pdf)  # [n_total, n_window]
#     component = area[:, None] * pdf  # [n_total, n_window]
#     return np.asarray(component)


def add_hdi_band(
    ax: plt.Axes,
    x: np.ndarray,
    samples_2d: np.ndarray,
    *,
    color: str = "C0",
    alpha: float = 0.3,
    linewidth: float = 1.5,
    label: str = "",
    linestyle: str = "-",
) -> None:
    """Plot posterior median and 95% HDI band from samples over x-axis.

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    x : np.ndarray
        X-axis values (1-D).
    samples_2d : np.ndarray
        Posterior samples [n_draw, n_time] for a signal/component over time.
    color : str
        Line and band color.
    alpha : float
        Alpha for shaded HDI band.
    linewidth : float
        Line width for median.
    label : str
        Legend label.
    linestyle : str
        Line style for median ("-", "--", ":", etc.).
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    samples_arr = np.asarray(samples_2d, dtype=float)

    # Compute credible interval per timepoint
    median = np.percentile(samples_arr, 50, axis=0)
    hdi_low = np.percentile(samples_arr, 2.5, axis=0)
    hdi_high = np.percentile(samples_arr, 97.5, axis=0)

    # Filter finite values
    finite_mask = (
        np.isfinite(x_arr)
        & np.isfinite(median)
        & np.isfinite(hdi_low)
        & np.isfinite(hdi_high)
    )
    x_fin = x_arr[finite_mask]
    median_fin = median[finite_mask]
    hdi_low_fin = hdi_low[finite_mask]
    hdi_high_fin = hdi_high[finite_mask]

    # Plot median line
    ax.plot(
        x_fin,
        median_fin,
        color=color,
        linewidth=linewidth,
        label=label,
        linestyle=linestyle,
    )

    # Plot 95% HDI band
    ax.fill_between(x_fin, hdi_low_fin, hdi_high_fin, color=color, alpha=alpha)


# ---------------------------------------------------------------------------
# Posterior Fit Plots
# ---------------------------------------------------------------------------


def plot_fit(
    time: np.ndarray,
    signal: np.ndarray,
    peaks: list[PeakAnnotation],
    curves: "PosteriorCurves | None",
    *,
    fitted_rows: Optional[np.ndarray] = None,
    baselines: Optional[list[BaselineAnnotation]] = None,
    chromatogram_ids: Optional[list[str]] = None,
    hdi_prob: float = 0.95,
    figsize: Optional[tuple[float, float]] = None,
    colors: Optional[list[str]] = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot raw data and posterior fit curves.

    Columns: one per peak window, plus a combined column when ``n_peak > 1``.
    Each peak-window subplot shows:

    - **Gray scatter** — raw signal within the window.
    - **Solid gray + band** — baseline median + HDI (fitted rows only).
    - **Dotted gray + band** — left component median + HDI (fitted rows only).
    - **Dashed gray + band** — right component median + HDI (fitted rows only).
    - **Blue + band** — total fitted signal median + HDI (fitted rows only).
    - **Red dashed** — peak window bounds.

    Parameters
    ----------
    time : np.ndarray  [n_display, n_time]
        Raw time matrix for traces to display.
    signal : np.ndarray  [n_display, n_time]
        Raw signal matrix.
    peaks : list[PeakAnnotation]
        Peak window definitions from the fitted subset.
    curves : PosteriorCurves or None
        Precomputed posterior curves from
        :meth:`~BetterFitter.posterior_curves`.  ``None`` → scatter-only.
    fitted_rows : np.ndarray or None
        Indices (into the *time/signal* row dimension) of traces that have
        posterior curves.  ``curves.total_median[i]`` corresponds to
        ``fitted_rows[i]``.  When ``None`` all rows with ``curves`` are
        assumed to be fitted.
    baselines : list or None
        Baseline annotations used to extend the combined-column display range.
    chromatogram_ids : list[str] or None
        Per-trace labels for subplot titles.
    hdi_prob : float
        Credible-interval probability; used only for legend labels.
    figsize : tuple or None
        Figure size; auto-scaled when ``None``.
    colors : list[str] or None
        List of hex color codes (e.g., ['#FF5733', '#33FF57']) for the total
        fitted signal line + HDI band, one per peak.  Length must match
        ``n_peak``.  When ``None`` (default), uses blue ('C0') for all peaks.
        The combined column (when n_peak > 1) always uses blue.

    Returns
    -------
    fig : plt.Figure
    axes : np.ndarray  [n_display, n_col]
    """
    # Lazy import to avoid circular dependency at module level

    time_arr = np.asarray(time, dtype=float)
    signal_arr = np.asarray(signal, dtype=float)
    n_display, _ = time_arr.shape
    n_peak = len(peaks)
    has_combined = n_peak > 1
    n_col = n_peak + (1 if has_combined else 0)

    if colors is not None:
        _validate_hex_colors(colors, n_peak)

    if chromatogram_ids is not None and len(chromatogram_ids) != n_display:
        raise ValueError(
            f"chromatogram_ids must have length n_display={n_display}, "
            f"got {len(chromatogram_ids)}."
        )

    # Resolve fitted_rows
    if curves is not None and fitted_rows is None:
        fitted_rows = np.arange(min(n_display, len(curves.trace_indices)))
    elif fitted_rows is None:
        fitted_rows = np.array([], dtype=int)

    # Build map: display_row → index into curves arrays
    row_to_curve: dict[int, int] = {}
    if curves is not None:
        for ci, row in enumerate(fitted_rows):
            row_to_curve[int(row)] = ci

    # Combined column range
    rt_lo = float(min(p.rt_min for p in peaks))
    rt_hi = float(max(p.rt_max for p in peaks))
    if baselines:
        rt_lo = min(rt_lo, float(min(b.rt_min for b in baselines)))
        rt_hi = max(rt_hi, float(max(b.rt_max for b in baselines)))

    hdi_label = f"{int(hdi_prob * 100)}% HDI"

    # ── Figure ──────────────────────────────────────────────────────────────
    if figsize is None:
        figsize = (4 * n_col, 3 * n_display)
    fig, axes = plt.subplots(
        n_display,
        n_col,
        figsize=figsize,
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    fig.patch.set_facecolor("none")
    for ax in axes.flat:
        ax.patch.set_facecolor("none")

    def trace_label(t: int) -> str:
        return str(chromatogram_ids[t]) if chromatogram_ids else f"trace {t}"

    # ── Per-display-trace loop ───────────────────────────────────────────────
    for t in range(n_display):
        x_trace = time_arr[t]
        y_trace = signal_arr[t]
        ci = row_to_curve.get(t)  # None → no posterior for this row
        add_legend = t == 0

        # ── Per-peak-window columns ──────────────────────────────────────────
        for p, peak in enumerate(peaks):
            ax = axes[t, p]

            # Raw scatter masked to this window
            raw_mask = (x_trace >= peak.rt_min) & (x_trace <= peak.rt_max)
            x_raw, y_raw = x_trace[raw_mask], y_trace[raw_mask]
            fin = np.isfinite(x_raw) & np.isfinite(y_raw)
            if fin.any():
                ax.scatter(
                    x_raw[fin],
                    y_raw[fin],
                    s=25,
                    alpha=0.5,
                    color="gray",
                    zorder=1,
                    label="Raw signal" if add_legend and p == 0 else "",
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="red",
                )

            # Posterior curves for this trace (if available)
            if ci is not None and curves is not None:
                win = (curves.x >= peak.rt_min) & (curves.x <= peak.rt_max)
                x_c = curves.x[win]
                if x_c.size > 0:
                    first = add_legend and p == 0
                    # Baseline — solid gray
                    _plot_hdi_line(
                        ax,
                        x_c,
                        curves.baseline_median[ci, win],
                        curves.baseline_lower[ci, win],
                        curves.baseline_upper[ci, win],
                        color="gray",
                        alpha=0.2,
                        linewidth=1.0,
                        linestyle="-",
                        label=f"Baseline ({hdi_label})" if first else "",
                    )
                    # Left component — dotted gray
                    _plot_hdi_line(
                        ax,
                        x_c,
                        curves.comp_l_median[ci, p, win],
                        curves.comp_l_lower[ci, p, win],
                        curves.comp_l_upper[ci, p, win],
                        color="gray",
                        alpha=0.15,
                        linewidth=0.9,
                        linestyle=":",
                        label=f"Left comp. ({hdi_label})" if first else "",
                    )
                    # Right component — dashed gray
                    _plot_hdi_line(
                        ax,
                        x_c,
                        curves.comp_r_median[ci, p, win],
                        curves.comp_r_lower[ci, p, win],
                        curves.comp_r_upper[ci, p, win],
                        color="gray",
                        alpha=0.15,
                        linewidth=0.9,
                        linestyle="--",
                        label=f"Right comp. ({hdi_label})" if first else "",
                    )
                    # Total — blue
                    _plot_hdi_line(
                        ax,
                        x_c,
                        curves.total_median[ci, win],
                        curves.total_lower[ci, win],
                        curves.total_upper[ci, win],
                        color="C0",
                        alpha=0.3,
                        linewidth=1.5,
                        linestyle="-",
                        label=f"Fitted signal ({hdi_label})" if first else "",
                    )

            add_peak_window_bounds(ax, peak, color="red", alpha=0.3, linewidth=0.8)
            ax.set_title(f"{peak.molecule_id} ({trace_label(t)})", fontsize=9)
            if p == 0:
                ax.set_ylabel("Signal", fontsize=8)
            if t == n_display - 1:
                ax.set_xlabel("Time (min)", fontsize=8)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.tick_params(labelsize=7)

        # ── Combined column ──────────────────────────────────────────────────
        if has_combined:
            ax_c = axes[t, n_peak]

            raw_comb = (x_trace >= rt_lo) & (x_trace <= rt_hi)
            x_raw_c, y_raw_c = x_trace[raw_comb], y_trace[raw_comb]
            fin_c = np.isfinite(x_raw_c) & np.isfinite(y_raw_c)
            if fin_c.any():
                ax_c.scatter(
                    x_raw_c[fin_c],
                    y_raw_c[fin_c],
                    s=25,
                    alpha=0.5,
                    color="gray",
                    zorder=1,
                )

            if ci is not None and curves is not None:
                comb = (curves.x >= rt_lo) & (curves.x <= rt_hi)
                x_cc = curves.x[comb]
                if x_cc.size > 0:
                    _plot_hdi_line(
                        ax_c,
                        x_cc,
                        curves.baseline_median[ci, comb],
                        curves.baseline_lower[ci, comb],
                        curves.baseline_upper[ci, comb],
                        color="gray",
                        alpha=0.2,
                        linewidth=1.0,
                        linestyle="-",
                    )
                    _plot_hdi_line(
                        ax_c,
                        x_cc,
                        curves.total_median[ci, comb],
                        curves.total_lower[ci, comb],
                        curves.total_upper[ci, comb],
                        color="C0",
                        alpha=0.3,
                        linewidth=1.5,
                        linestyle="-",
                    )

            ax_c.set_title(f"Combined — {trace_label(t)}", fontsize=9)
            if t == n_display - 1:
                ax_c.set_xlabel("Time (min)", fontsize=8)
            ax_c.grid(True, alpha=0.3, linestyle="--")
            ax_c.tick_params(labelsize=7)

    if n_display > 0 and n_peak > 0 and axes[0, 0].has_data():
        axes[0, 0].legend(fontsize=7, loc="best")

    fig.tight_layout()
    return fig, axes


def _plot_hdi_line(
    ax: plt.Axes,
    x: np.ndarray,
    median: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    color: str = "C0",
    alpha: float = 0.3,
    linewidth: float = 1.5,
    linestyle: str = "-",
    label: str = "",
) -> None:
    """Plot a median line + HDI fill-between band."""
    fin = np.isfinite(x) & np.isfinite(median)
    if not fin.any():
        return
    ax.plot(
        x[fin],
        median[fin],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )
    ax.fill_between(x[fin], lower[fin], upper[fin], color=color, alpha=alpha)


def plot_posterior_predictive(
    time: np.ndarray,
    signal: np.ndarray,
    peaks: list[PeakAnnotation],
    posterior: object,
    *,
    x_posterior: Optional[np.ndarray] = None,
    y_posterior: Optional[np.ndarray] = None,
    chromatogram_ids: Optional[list[str]] = None,
    figsize: Optional[tuple[float, float]] = None,
    baselines: Optional[list] = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Deprecated — use :meth:`~BetterFitter.plot_fit` instead.

    .. deprecated::
        ``plot_posterior_predictive`` is superseded by the
        :meth:`BetterFitter.plot_fit` method which avoids the posterior
        axis alignment bug.  This function remains for backward compatibility
        but will be removed in a future release.
    """
    import warnings

    warnings.warn(
        "plot_posterior_predictive() is deprecated. "
        "Use fitter.plot_fit() instead, which correctly handles the windowed "
        "posterior time axis and supports multi-subset trace selection.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Minimal fallback: scatter-only plot (no posterior overlay)
    time_arr = np.asarray(time, dtype=float)
    signal_arr = np.asarray(signal, dtype=float)
    n_trace = time_arr.shape[0]
    return plot_fit(
        time_arr,
        signal_arr,
        peaks,
        None,  # no curves — scatter only
        baselines=baselines,
        chromatogram_ids=chromatogram_ids,
        figsize=figsize,
    )


# ---------------------------------------------------------------------------
# MCMC Trace Plots
# ---------------------------------------------------------------------------


def plot_trace(
    posterior: object,
    var_names: list[str] | None = None,
    figsize: Optional[tuple[float, float]] = None,
) -> plt.Figure:
    """Plot MCMC trace for all sampled parameters.

    Shows sample values over iterations for convergence diagnostics.

    Parameters
    ----------
    posterior : arviz.InferenceData
        Posterior from ArviZ (result of az.from_numpyro()).
    var_names : list[str] or None
        Parameter names to plot. If None, uses the fitter's default shared
        diagnostics (e.g. ``['trace_shift', 'apex', 'sigma_base']``).
    figsize : tuple or None
        Figure size. If None, auto-scales based on number of variables.

    Returns
    -------
    fig : plt.Figure
        The figure object with trace plots.
    """
    import warnings

    import arviz as az

    available_vars = list(posterior.posterior.data_vars)
    if var_names is None:
        from .better_model import TRACE_PARAMETER_NAMES

        var_names = [name for name in TRACE_PARAMETER_NAMES if name in available_vars]
        if not var_names:
            var_names = available_vars
    else:
        requested = list(var_names)
        var_names = [name for name in requested if name in available_vars]
        missing = [name for name in requested if name not in available_vars]
        if missing:
            warnings.warn(
                "Ignoring unavailable posterior variables in plot_trace: "
                + ", ".join(missing),
                stacklevel=2,
            )
        if not var_names:
            raise ValueError(
                "plot_trace received no available posterior variables to plot. "
                f"Available variables: {', '.join(available_vars)}"
            )

    n_vars = len(var_names)
    if figsize is None:
        # 2 columns, enough rows for all variables
        n_cols = 2
        n_rows = (n_vars + n_cols - 1) // n_cols
        figsize = (12, 3.5 * n_rows)

    # Create trace plot (returns figure directly)
    az.plot_trace(
        posterior,
        var_names=var_names,
        figsize=figsize,
        kind="trace",
    )

    fig = plt.gcf()  # Get current figure
    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Entry point for testing
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import jax.numpy as jnp

    # Load test data
    arr = jnp.load("/Users/max/code/sahh-kinetics-hplc/chromatograms.npy").reshape(
        -1, 3000
    )[:5, :1000]
    time = jnp.load("/Users/max/code/sahh-kinetics-hplc/times.npy").reshape(-1, 3000)[
        :5, :1000
    ]

    # Define peaks and baselines
    from chromhandler.annotations import BaselineAnnotation

    baselines = [
        BaselineAnnotation(rt_min=0, rt_max=1),
        BaselineAnnotation(rt_min=4, rt_max=6),
    ]
    peaks = [
        PeakAnnotation(molecule_id="peak1", rt_min=2.6, rt_max=2.83, mode="single"),
        PeakAnnotation(
            molecule_id="peak2",
            rt_min=2.9,
            rt_max=3.18,
            mode="artefact_doublet",
            artefact_side="right",
        ),
    ]

    # Dummy baseline priors
    n_trace = arr.shape[0]
    baseline_intercept = np.ones(n_trace) * 100.0
    baseline_slope = np.ones(n_trace) * 50.0
    baseline_intercept_scale = np.ones(n_trace) * 20.0
    baseline_slope_scale = np.ones(n_trace) * 5.0

    # Plot with baseline + peak bounds
    fig, axes = plot_prior_traces(
        time,
        arr,
        peaks,
        baseline_intercept,
        baseline_slope,
        baseline_intercept_scale,
        baseline_slope_scale,
        show_baseline=True,
        show_peak_bounds=True,
    )
    plt.savefig("/tmp/prior_traces_with_baseline.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: /tmp/prior_traces_with_baseline.png")

    # Plot without baseline, with peak bounds
    fig, axes = plot_prior_traces(
        time,
        arr,
        peaks,
        baseline_intercept,
        baseline_slope,
        baseline_intercept_scale,
        baseline_slope_scale,
        show_baseline=False,
        show_peak_bounds=True,
    )
    plt.savefig("/tmp/prior_traces_no_baseline.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: /tmp/prior_traces_no_baseline.png")
