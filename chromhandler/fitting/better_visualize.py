"""Modular visualization for window-geometry Bayesian prior fitting.

Provides reusable plotting methods for:
- Prior visualization (loc line + scale shaded region)
- Per-trace, per-peak-window subplots
- Baseline overlay with uncertainty bands
- Signal data as scatter points
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .data import PeakAnnotation, peak_is_artefact_mode, peak_is_doublet_mode

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
        peak.low, color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth
    )
    ax.axvline(
        peak.high, color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth
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
    cmap: str = "viridis",
    n_grid: int = 220,
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

    # For a 2-D Gaussian, chi-square(df=2) quantiles have a closed form:
    # r^2_p = -2 log(1-p). These density levels define quartile-like shells.
    mass_levels = np.array([0.25, 0.50, 0.75, 0.95], dtype=float)
    r2 = -2.0 * np.log(1.0 - mass_levels)
    density_peak = 1.0 / (2.0 * np.pi * alpha_scale_f * sigma_scale_f)
    density_levels = density_peak * np.exp(-0.5 * r2)
    contour_levels = np.sort(
        np.concatenate([density_levels[::-1], np.array([density_peak])])
    )
    colors = plt.get_cmap(cmap)(np.linspace(0.20, 0.92, 4))

    ax.contourf(
        xx,
        yy,
        density,
        levels=contour_levels,
        colors=colors,
        alpha=0.75,
        antialiased=True,
    )
    ax.contour(
        xx,
        yy,
        density,
        levels=contour_levels[:-1],
        colors="white",
        linewidths=0.8,
        alpha=0.8,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


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
    mu_center_loc: Optional[np.ndarray] = None,
    mu_center_scale: Optional[np.ndarray] = None,
    approx_center_trace: Optional[np.ndarray] = None,
    approx_height_trace: Optional[np.ndarray] = None,
    approx_sigma_trace: Optional[np.ndarray] = None,
    approx_valid_trace: Optional[np.ndarray] = None,
    approx_fallback_trace: Optional[np.ndarray] = None,
    *,
    show_baseline: bool = True,
    show_mu_prior: bool = True,
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
    mu_center_loc : np.ndarray or None
        Peak-center prior locations, shape [n_peak]. If provided, drawn as
        vertical lines in each corresponding peak window.
    mu_center_scale : np.ndarray or None
        Peak-center prior scales, shape [n_peak]. If provided, drawn as
        vertical shaded bands spanning ``mu_center_loc ± mu_center_scale``.
    approx_center_trace : np.ndarray or None
        Dense Gaussian-approximation centers, shape [n_trace, n_peak].
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
    show_mu_prior : bool
        If True and ``mu_center_loc`` / ``mu_center_scale`` are provided,
        overlay the center prior as a vertical line with a shaded band.
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
    mu_loc = None if mu_center_loc is None else np.asarray(mu_center_loc, dtype=float)
    mu_scale = (
        None if mu_center_scale is None else np.asarray(mu_center_scale, dtype=float)
    )
    approx_center_arr = (
        None
        if approx_center_trace is None
        else np.asarray(approx_center_trace, dtype=float)
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

    if (mu_loc is None) != (mu_scale is None):
        raise ValueError(
            "mu_center_loc and mu_center_scale must either both be provided or both be omitted."
        )
    if mu_loc is not None and mu_loc.shape[0] != n_peak:
        raise ValueError(
            f"mu_center_loc must have length {n_peak}, got shape {mu_loc.shape}."
        )
    if mu_scale is not None and mu_scale.shape[0] != n_peak:
        raise ValueError(
            f"mu_center_scale must have length {n_peak}, got shape {mu_scale.shape}."
        )
    gaussian_inputs = [
        approx_center_arr,
        approx_height_arr,
        approx_sigma_arr,
        approx_valid_arr,
        approx_fallback_arr,
    ]
    has_gaussian_inputs = all(arr is not None for arr in gaussian_inputs)
    if any(arr is not None for arr in gaussian_inputs) and not has_gaussian_inputs:
        raise ValueError(
            "approx_center_trace, approx_height_trace, approx_sigma_trace, "
            "approx_valid_trace, and approx_fallback_trace must all be provided together."
        )
    if has_gaussian_inputs:
        expected_shape = (n_trace, n_peak)
        for name, arr in (
            ("approx_center_trace", approx_center_arr),
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
            mask = (x_trace >= peak.low) & (x_trace <= peak.high)
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
                ax.set_title(f"{peak.name} (trace {t})", fontsize=9)
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
                assert approx_center_arr is not None
                assert approx_height_arr is not None
                assert approx_sigma_arr is not None
                assert approx_valid_arr is not None
                assert approx_fallback_arr is not None
                if bool(approx_valid_arr[t, p]):
                    peak_curve = _gaussian_peak_curve_from_sigma(
                        x_window,
                        approx_center_arr[t, p],
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

            if show_mu_prior and mu_loc is not None and mu_scale is not None:
                add_vertical_prior_band(
                    ax,
                    mu_loc[p],
                    mu_scale[p],
                    label="mu prior" if (t == 0 and p == 0) else "",
                    color="tab:orange",
                    alpha=0.18,
                    linewidth=1.5,
                    linestyle="-",
                )

            # Add peak window bounds if requested
            if show_peak_bounds:
                add_peak_window_bounds(ax, peak, color="red", alpha=0.4, linewidth=1.0)

            # Labels and formatting
            ax.set_title(f"{peak.name} (trace {t})", fontsize=9)
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
) -> tuple[plt.Figure, np.ndarray]:
    """Plot per-trace FWHM-derived ``sigma`` vs ``alpha`` with 2-D prior density."""
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

    apex_height_arr = None
    if apex_height_trace is not None:
        apex_height_arr = np.asarray(apex_height_trace, dtype=float)
        if apex_height_arr.shape != sigma_arr.shape:
            raise ValueError(
                "apex_height_trace must share shape with sigma_trace if provided."
            )

    def _check_peak_vector(
        values: Optional[np.ndarray], name: str
    ) -> Optional[np.ndarray]:
        if values is None:
            return None
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.shape[0] != n_peak:
            raise ValueError(
                f"{name} must have length {n_peak}, got shape {arr.shape}."
            )
        return arr

    sigma_loc_arr = _check_peak_vector(sigma_loc, "sigma_loc")
    sigma_scale_arr = _check_peak_vector(sigma_scale, "sigma_scale")
    alpha_loc_arr = _check_peak_vector(alpha_loc, "alpha_loc")
    alpha_scale_arr = _check_peak_vector(alpha_scale, "alpha_scale")

    if (sigma_loc_arr is None) != (sigma_scale_arr is None):
        raise ValueError("sigma_loc and sigma_scale must both be provided or omitted.")
    if (alpha_loc_arr is None) != (alpha_scale_arr is None):
        raise ValueError("alpha_loc and alpha_scale must both be provided or omitted.")

    if figsize is None:
        figsize = (4.3 * n_peak, 4.2)

    fig, axes = plt.subplots(
        1,
        n_peak,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.reshape(1, n_peak)

    for peak_idx, peak in enumerate(peaks):
        ax = axes[0, peak_idx]
        valid_peak = (
            valid_arr[:, peak_idx]
            & np.isfinite(sigma_arr[:, peak_idx])
            & np.isfinite(alpha_arr[:, peak_idx])
        )
        alpha_peak = alpha_arr[valid_peak, peak_idx]
        sigma_peak = sigma_arr[valid_peak, peak_idx]

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
            add_sigma_alpha_prior_density(
                ax,
                alpha_loc=float(alpha_loc_arr[peak_idx]),
                alpha_scale=float(alpha_scale_arr[peak_idx]),
                sigma_loc=float(sigma_loc_arr[peak_idx]),
                sigma_scale=float(sigma_scale_arr[peak_idx]),
                x_data=alpha_peak,
                y_data=sigma_peak,
                cmap=cmap,
            )

        if np.any(valid_peak):
            alpha_scatter = 0.2
            if apex_height_arr is not None:
                heights_peak = apex_height_arr[valid_peak, peak_idx]
                sizes_peak = _marker_sizes_from_values(heights_peak)
                # Draw transparent fill (scatter) with opaque edge line
                ax.scatter(
                    alpha_peak,
                    sigma_peak,
                    s=sizes_peak,
                    facecolors=(1, 1, 1, alpha_scatter),  # RGBA tuple for fill
                    edgecolors=(0, 0, 0, 1),  # RGBA tuple for edge (black, opaque)
                    linewidths=0.9,
                    alpha=None,  # Disable alpha param so RGBA is used
                    zorder=4,
                )
            else:
                ax.scatter(
                    alpha_peak,
                    sigma_peak,
                    s=60,
                    facecolors=(1, 1, 1, alpha_scatter),
                    edgecolors=(0, 0, 0, 1),
                    linewidths=0.9,
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
                fontsize=10,
                color="0.4",
            )

        ax.set_title(peak.name, fontsize=10)
        ax.set_xlabel(r"$\alpha$ (–)", fontsize=9)
        if peak_idx == 0:
            ax.set_ylabel(r"$\sigma$ (min)", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=8)

    # Center x-axis on alpha=0 (sharex=True so one call applies to all)
    x_lo, x_hi = axes[0, 0].get_xlim()
    span = max(abs(x_lo), abs(x_hi), 1e-6)
    axes[0, 0].set_xlim(-span, span)

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# HDI (95% Credible Interval) Plotting Helpers
# ---------------------------------------------------------------------------


def _compute_skew_normal_component(
    x: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    alpha: jnp.ndarray,
    area: jnp.ndarray,
) -> np.ndarray:
    """Compute area-scaled skew-normal PDF values.

    Parameters
    ----------
    x : jnp.ndarray
        Time axis [n_window]
    mu : jnp.ndarray
        Location parameter [n_total]
    sigma : jnp.ndarray
        Scale parameter [n_total]
    alpha : jnp.ndarray
        Skewness parameter [n_total]
    area : jnp.ndarray
        Area scaling factor [n_total]

    Returns
    -------
    np.ndarray
        Component signal [n_total, n_window]
    """
    x_broad = x[None, :]  # [1, n_window]
    mu_broad = mu[:, None]  # [n_total, 1]
    sigma_broad = sigma[:, None]  # [n_total, 1]
    alpha_broad = alpha[:, None]  # [n_total, 1]

    z = (x_broad - mu_broad) / sigma_broad
    log_pdf = (
        -0.5 * z**2
        - 0.5 * jnp.log(2.0 * jnp.pi)
        + jnp.log(2.0)
        + jnp.log(jax.scipy.special.ndtr(alpha_broad * z))
    )
    pdf = jnp.exp(log_pdf) / sigma_broad  # [n_total, n_window]
    component = area[:, None] * pdf  # [n_total, n_window]
    return np.asarray(component)


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
# Posterior Predictive Plots
# ---------------------------------------------------------------------------


def plot_posterior_predictive(
    time: np.ndarray,
    signal: np.ndarray,
    peaks: list[PeakAnnotation],
    posterior: object,
    *,
    x_posterior: Optional[np.ndarray] = None,
    y_posterior: Optional[np.ndarray] = None,
    figsize: Optional[tuple[float, float]] = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot posterior predictive: fitted signal + components vs raw data.

    Subplots organized as [trace, peak_window]. Each shows:
    - Raw data (gray scatter, α=0.5)
    - Individual fitted components for doublet modes (gray lines + 95% HDI)
    - Baseline (gray dashed + 95% HDI)
    - Total signal (blue line + blue 95% HDI)
    - Peak window bounds (red dashed lines)

    Parameters
    ----------
    time : np.ndarray
        Time matrix, shape [n_trace, n_time].
    signal : np.ndarray
        Signal matrix, shape [n_trace, n_time] (raw data).
    peaks : list[PeakAnnotation]
        Peak window definitions.
    posterior : arviz.InferenceData
        Posterior from ArviZ (result of `fitter.posterior`).
    x_posterior : np.ndarray or None
        Masked time axis if windowed likelihood was used. Can be either
        `[n_time_posterior]` for a shared axis or `[n_trace, n_time_posterior]`
        for per-trace aligned axes. If None, assumes full likelihood was used.
    y_posterior : np.ndarray or None
        Masked signal matrix if windowed likelihood was used, shape [n_trace, n_time_posterior].
        Only used for display if posterior time differs from input time.
    figsize : tuple or None
        Figure size. If None, auto-scales based on grid.

    Returns
    -------
    fig : plt.Figure
        The figure object.
    axes : np.ndarray
        2-D array of axes, shape [n_trace, n_peak].
    """
    time_arr = np.asarray(time, dtype=float)
    signal_arr = np.asarray(signal, dtype=float)
    x_posterior_arr = (
        None if x_posterior is None else np.asarray(x_posterior, dtype=float)
    )

    n_trace, n_time_full = time_arr.shape
    n_peak = len(peaks)

    if x_posterior_arr is not None:
        if x_posterior_arr.ndim not in (1, 2):
            raise ValueError(
                f"x_posterior must be 1-D or 2-D, got shape {x_posterior_arr.shape}."
            )
        if x_posterior_arr.ndim == 2 and x_posterior_arr.shape[0] != n_trace:
            raise ValueError(
                f"x_posterior trace axis must match time shape {time_arr.shape}, "
                f"got {x_posterior_arr.shape}."
            )

    # Default figsize
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

    # Extract posterior samples (reshape to combine chains + draws)
    mu_y_raw = posterior.posterior[
        "mu_y"
    ].values  # [n_chain, n_draw, n_trace, n_time_posterior]
    baseline_raw = posterior.posterior[
        "baseline_curve"
    ].values  # [n_chain, n_draw, n_trace, n_time_posterior]

    # Flatten chains and draws: [n_chain, n_draw, ...] → [n_total_samples, ...]
    n_chain = mu_y_raw.shape[0]
    n_draw = mu_y_raw.shape[1]
    # Use actual posterior time dimension (may be different from input if windowed likelihood was used)
    n_time_posterior = mu_y_raw.shape[3]
    n_total = n_chain * n_draw
    mu_y_samples = mu_y_raw.reshape(
        n_total, n_trace, n_time_posterior
    )  # [n_total, n_trace, n_time_posterior]
    baseline_samples = baseline_raw.reshape(
        n_total, n_trace, n_time_posterior
    )  # [n_total, n_trace, n_time_posterior]

    # Extract posterior samples for component reconstruction.
    # Use the model's deterministic component parameters directly instead of
    # re-deriving them from intermediate latents in the visualizer.
    component_params: dict[str, np.ndarray] = {}
    has_doublet_peaks = any(peak_is_doublet_mode(peak.mode) for peak in peaks)
    if has_doublet_peaks:
        try:
            component_params["sigma"] = posterior.posterior["sigma"].values.reshape(
                n_total, -1
            )  # [n_total, n_comp]
            component_params["alpha"] = posterior.posterior["alpha"].values.reshape(
                n_total, -1
            )  # [n_total, n_comp]
            component_params["mu"] = posterior.posterior["mu"].values.reshape(
                n_total, n_trace, -1
            )  # [n_total, n_trace, n_comp]
            component_params["A"] = posterior.posterior["A"].values.reshape(
                n_total, n_trace, -1
            )  # [n_total, n_trace, n_comp]
        except KeyError:
            component_params = {}

    # Plot each (trace, peak) subplot
    for t in range(n_trace):
        for p, peak in enumerate(peaks):
            ax = axes[t, p]

            # Get data for this trace
            x_trace = time_arr[t, :]
            y_trace = signal_arr[t, :]

            # Extract peak window from raw data
            mask_full = (x_trace >= peak.low) & (x_trace <= peak.high)
            x_window = x_trace[mask_full]
            y_window = y_trace[mask_full]

            # Skip if no valid data in raw signal
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
                ax.set_title(f"{peak.name} (trace {t})", fontsize=9)
                continue

            # For posterior samples: if windowed likelihood was used (n_time_posterior != n_time_full),
            # we need to use the provided masked time axis. Otherwise use index masking.
            if n_time_posterior == n_time_full:
                # Full likelihood: posterior time matches input time, use index mask directly
                mu_y_window = mu_y_samples[:, t, mask_full]  # [n_draw, n_window]
                baseline_window = baseline_samples[
                    :, t, mask_full
                ]  # [n_draw, n_window]
                x_posterior_axis = x_window  # Use the input time axis
            else:
                # Windowed likelihood: use the provided masked time axis
                if x_posterior_arr is not None:
                    # We have the exact masked time axis. Find which points fall in the peak window.
                    x_posterior_trace = (
                        x_posterior_arr
                        if x_posterior_arr.ndim == 1
                        else x_posterior_arr[t]
                    )
                    mask_posterior = (x_posterior_trace >= peak.low) & (
                        x_posterior_trace <= peak.high
                    )
                    x_posterior_axis = x_posterior_trace[mask_posterior]
                    mu_y_window = mu_y_samples[
                        :, t, mask_posterior
                    ]  # [n_draw, n_window]
                    baseline_window = baseline_samples[
                        :, t, mask_posterior
                    ]  # [n_draw, n_window]
                else:
                    # Masked time axis not provided, but we know posterior is different size.
                    # Plot all available posterior samples with synthetic evenly-spaced time.
                    # This is a fallback for robustness.
                    x_posterior_axis = np.linspace(
                        peak.low, peak.high, n_time_posterior
                    )
                    mu_y_window = mu_y_samples[:, t, :]  # [n_draw, n_time_posterior]
                    baseline_window = baseline_samples[
                        :, t, :
                    ]  # [n_draw, n_time_posterior]

            # Plot raw signal as gray scatter
            finite_mask = np.isfinite(x_window) & np.isfinite(y_window)
            ax.scatter(
                x_window[finite_mask],
                y_window[finite_mask],
                s=30,
                alpha=0.5,
                color="gray",
                label="Raw signal",
                zorder=1,
            )

            # Plot baseline (gray dashed + HDI band)
            add_hdi_band(
                ax,
                x_posterior_axis,
                baseline_window,
                color="gray",
                alpha=0.2,
                linewidth=1,
                linestyle="--",
                label="Baseline (95% HDI)",
            )

            # Plot individual components for any doublet mode using the generic
            # deterministic per-component arrays exported by the model.
            if bool(component_params) and peak_is_doublet_mode(peak.mode):
                try:
                    comp_1_idx = 2 * p
                    comp_2_idx = comp_1_idx + 1

                    sigma_1_flat = component_params["sigma"][:, comp_1_idx]
                    sigma_2_flat = component_params["sigma"][:, comp_2_idx]
                    alpha_1_flat = component_params["alpha"][:, comp_1_idx]
                    alpha_2_flat = component_params["alpha"][:, comp_2_idx]
                    mu_1_flat = component_params["mu"][:, t, comp_1_idx]
                    mu_2_flat = component_params["mu"][:, t, comp_2_idx]
                    area_1_flat = component_params["A"][:, t, comp_1_idx]
                    area_2_flat = component_params["A"][:, t, comp_2_idx]

                    x_plot = jnp.asarray(np.asarray(x_posterior_axis, dtype=float))

                    component_1 = _compute_skew_normal_component(
                        x_plot,
                        jnp.asarray(mu_1_flat),
                        jnp.asarray(sigma_1_flat),
                        jnp.asarray(alpha_1_flat),
                        jnp.asarray(area_1_flat),
                    )
                    component_2 = _compute_skew_normal_component(
                        x_plot,
                        jnp.asarray(mu_2_flat),
                        jnp.asarray(sigma_2_flat),
                        jnp.asarray(alpha_2_flat),
                        jnp.asarray(area_2_flat),
                    )

                    if peak_is_artefact_mode(peak.mode):
                        label_1 = "Main component (95% HDI)"
                        label_2 = "Artefact component (95% HDI)"
                    else:
                        label_1 = "Component 1 (95% HDI)"
                        label_2 = "Component 2 (95% HDI)"

                    add_hdi_band(
                        ax,
                        x_posterior_axis,
                        component_1,
                        color="gray",
                        alpha=0.25,
                        linewidth=1.2,
                        linestyle="-",
                        label=label_1,
                    )

                    add_hdi_band(
                        ax,
                        x_posterior_axis,
                        component_2,
                        color="gray",
                        alpha=0.15,
                        linewidth=1.0,
                        linestyle="--",
                        label=label_2,
                    )
                except (KeyError, IndexError, ValueError, AttributeError):
                    # If component reconstruction fails, skip silently and continue
                    # with the total posterior predictive signal.
                    pass

            # Plot total fitted signal (blue line + HDI band)
            add_hdi_band(
                ax,
                x_posterior_axis,
                mu_y_window,
                color="C0",
                alpha=0.3,
                linewidth=1.5,
                linestyle="-",
                label="Fitted signal (95% HDI)",
            )

            # Add peak window bounds
            add_peak_window_bounds(ax, peak, color="red", alpha=0.3, linewidth=1.0)

            # Labels and formatting
            ax.set_title(f"{peak.name} (trace {t})", fontsize=9)
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
        f"Posterior Predictive: {n_trace} traces × {n_peak} peak windows",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()

    return fig, axes


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
        Parameter names to plot. If None, plots all sampled parameters.
        Can filter to subset for readability (e.g., ['center_per_trace', 'sigma_peak']).
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
    fig.suptitle(
        "MCMC Trace Plots - Convergence Diagnostics",
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )
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
    baselines = [
        PeakAnnotation(name="bl1", low=0, high=1, mode="single"),
        PeakAnnotation(name="bl2", low=4, high=6, mode="single"),
    ]
    peaks = [
        PeakAnnotation(name="peak1", low=2.6, high=2.83, mode="single"),
        PeakAnnotation(
            name="peak2",
            low=2.9,
            high=3.18,
            mode="artefact_doublet",
            shoulder="right",
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
