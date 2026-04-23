"""Posterior fit visualizations for :class:`Fitter`.

Public entry points:

- :func:`plot_fit_peaks` — per-peak posterior grid (rows = traces, cols = peaks).
- :func:`plot_fit_combined` — combined posterior view (rows = traces, one column).
- :func:`plot_geometric_diagnostic` — pre-fit ``(sigma_eff, alpha_asym)``
  scatter with MAD-based cluster-outlier flagging.

Posterior methods drive data directly off an ArviZ ``InferenceData`` and
use :func:`chromhandler.fitting.model.log_split_normal_pdf` as the single
source of truth for the split-normal PDF.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import arviz as az
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import median_abs_deviation

from . import model
from .priors import _trace_fwhm_geometry, fwhm_geometry_to_sigma_alpha

if TYPE_CHECKING:
    from arviz import InferenceData
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from chromhandler.annotations import BaselineAnnotation, PeakAnnotation


HDI_PROB = 0.95
_N_SAMPLES_MAX = 2000


# ---------------------------------------------------------------------------
# Posterior evaluation (ArviZ-native, model-backed)
# ---------------------------------------------------------------------------


def _extract(
    posterior: InferenceData,
    var_names: list[str],
    n_samples_max: int,
) -> dict[str, NDArray[np.float64]]:
    """Extract posterior arrays with ``sample`` as the leading axis.

    Returns ``{name: ndarray}`` where each array is shaped
    ``[n_sample, *var_dims]``. Deterministic subsampling via ``rng=0``.
    """
    ds = az.extract(posterior, var_names=var_names, num_samples=n_samples_max, rng=0)
    out: dict[str, NDArray[np.float64]] = {}
    for name in var_names:
        da = ds[name].transpose("sample", ...)
        out[name] = np.asarray(da.values, dtype=float)
    return out


def _evaluate_components(
    posterior: InferenceData,
    peaks: list[PeakAnnotation],
    x: NDArray[np.float64],
    *,
    trace_indices: NDArray[np.intp] | None = None,
    n_samples_max: int = _N_SAMPLES_MAX,
) -> dict[str, NDArray[np.float64]]:
    """Evaluate baseline / left / right / total components on grid ``x``.

    Uses :func:`model.log_split_normal_pdf` so viz and MCMC share one PDF.

    Returns a dict with keys:

    - ``baseline``: ``[n_sample, n_sel, n_x]``
    - ``comp_l`` / ``comp_r``: ``[n_sample, n_sel, n_peak, n_x]``
    - ``total``: ``[n_sample, n_sel, n_x]``
    """
    var_names = [
        "apex_l", "apex_r", "sl_l", "sl_r", "sr_l", "sr_r",
        "area_l", "area_r", "baseline_intercept", "baseline_slope",
    ]
    arrs = _extract(posterior, var_names, n_samples_max)

    apex_l = arrs["apex_l"]  # [n_sample, n_trace, n_peak]
    if trace_indices is not None:
        idx = np.asarray(trace_indices, dtype=int)
        for k in ("apex_l", "apex_r", "sl_l", "sl_r", "sr_l", "sr_r", "area_l", "area_r"):
            arrs[k] = arrs[k][:, idx]
        for k in ("baseline_intercept", "baseline_slope"):
            arrs[k] = arrs[k][:, idx]
        apex_l = arrs["apex_l"]

    n_sample, n_sel, n_peak = apex_l.shape
    n_x = x.shape[0]
    n_flat = n_sample * n_sel

    def _flat(key: str) -> NDArray[np.float64]:
        return arrs[key].reshape(n_flat, n_peak)

    x_jax = jnp.broadcast_to(jnp.asarray(x)[None, :], (n_flat, n_x))

    pdf_l = np.asarray(model.split_normal_pdf(
        x_jax, jnp.asarray(_flat("apex_l")), jnp.asarray(_flat("sl_l")), jnp.asarray(_flat("sr_l")),
    ))  # [n_flat, n_peak, n_x]
    pdf_r = np.asarray(model.split_normal_pdf(
        x_jax, jnp.asarray(_flat("apex_r")), jnp.asarray(_flat("sl_r")), jnp.asarray(_flat("sr_r")),
    ))

    comp_l = (_flat("area_l")[:, :, None] * pdf_l).reshape(n_sample, n_sel, n_peak, n_x)
    comp_r = (_flat("area_r")[:, :, None] * pdf_r).reshape(n_sample, n_sel, n_peak, n_x)

    # Model-side baseline centering: x - x_mid
    x_mid = 0.5 * (min(p.rt_min for p in peaks) + max(p.rt_max for p in peaks))
    b_int = arrs["baseline_intercept"][:, :, None]  # [n_sample, n_sel, 1]
    b_slp = arrs["baseline_slope"][:, :, None]
    baseline = b_int + b_slp * (x[None, None, :] - x_mid)

    total = comp_l.sum(axis=2) + comp_r.sum(axis=2) + baseline
    return {"baseline": baseline, "comp_l": comp_l, "comp_r": comp_r, "total": total}


# ---------------------------------------------------------------------------
# HDI plotting helper
# ---------------------------------------------------------------------------


def _hdi_line(
    ax: Axes,
    x: NDArray[np.float64],
    samples: NDArray[np.float64],
    *,
    color: str,
    linestyle: str = "-",
    linewidth: float = 1.5,
    label: str = "",
) -> None:
    """Plot posterior median line + central HDI band from ``samples`` ``[n_draw, n_x]``."""
    lo_pct = 100.0 * (1.0 - HDI_PROB) / 2.0
    hi_pct = 100.0 - lo_pct
    lower, median, upper = np.percentile(samples, [lo_pct, 50, hi_pct], axis=0)
    fin = np.isfinite(x) & np.isfinite(median)
    if not fin.any():
        return
    ax.plot(x[fin], median[fin], color=color, linestyle=linestyle, linewidth=linewidth, label=label)
    ax.fill_between(x[fin], lower[fin], upper[fin], color=color, alpha=0.25, linewidth=0)


def _window_bounds(ax: Axes, peak: PeakAnnotation) -> None:
    """Red dashed vertical lines at peak window edges."""
    ax.axvline(peak.rt_min, color="red", linestyle="--", linewidth=0.8)
    ax.axvline(peak.rt_max, color="red", linestyle="--", linewidth=0.8)


def _trace_label(t: int, chromatogram_ids: list[str] | None) -> str:
    if chromatogram_ids is not None:
        return str(chromatogram_ids[t])
    return f"trace {t}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_fit_peaks(
    time: NDArray[np.float64],
    signal: NDArray[np.float64],
    peaks: list[PeakAnnotation],
    posterior: InferenceData | None,
    *,
    trace_indices: NDArray[np.intp] | None = None,
    chromatogram_ids: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, np.ndarray[Any, Any]]:
    """Per-peak posterior fit grid.

    Grid: rows = selected traces, cols = peak windows. Each cell shows the
    raw scatter within ``[peak.rt_min, peak.rt_max]`` and, if a posterior is
    supplied, median+HDI of baseline (solid gray), left component (dotted
    gray), right component (dashed gray), and total fit (blue).
    """
    time_arr = np.asarray(time, dtype=float)
    signal_arr = np.asarray(signal, dtype=float)
    n_trace = time_arr.shape[0]
    n_peak = len(peaks)

    if trace_indices is None:
        sel: NDArray[np.intp] = np.arange(n_trace, dtype=np.intp)
    else:
        sel = np.asarray(trace_indices, dtype=np.intp)
    n_sel = int(sel.shape[0])

    components = None
    peak_windows: list[tuple[NDArray[np.bool_], NDArray[np.float64]]] = []
    if posterior is not None and n_sel > 0:
        x_full = np.asarray(np.nanmedian(time_arr, axis=0), dtype=float)
        components = _evaluate_components(posterior, peaks, x_full, trace_indices=sel)
        for peak in peaks:
            mask_win: NDArray[np.bool_] = (x_full >= peak.rt_min) & (x_full <= peak.rt_max)
            peak_windows.append((mask_win, x_full[mask_win]))

    if figsize is None:
        figsize = (4 * max(n_peak, 1), 3 * max(n_sel, 1))
    fig, axes = plt.subplots(n_sel, n_peak, figsize=figsize, squeeze=False)

    for i, t in enumerate(sel):
        x_trace = time_arr[t]
        y_trace = signal_arr[t]
        for p, peak in enumerate(peaks):
            ax = axes[i, p]
            mask = (x_trace >= peak.rt_min) & (x_trace <= peak.rt_max)
            xr = x_trace[mask]
            yr = y_trace[mask]
            fin = np.isfinite(xr) & np.isfinite(yr)
            if fin.any():
                ax.scatter(
                    xr[fin], yr[fin], s=25, color="gray", zorder=1,
                    label="Raw signal" if i == 0 and p == 0 else "",
                )

            if components is not None:
                win, x_win = peak_windows[p]
                first = i == 0 and p == 0
                _hdi_line(
                    ax, x_win, components["baseline"][:, i, win],
                    color="gray", linestyle="-", linewidth=1.0,
                    label="Baseline" if first else "",
                )
                _hdi_line(
                    ax, x_win, components["comp_l"][:, i, p, win],
                    color="gray", linestyle=":", linewidth=0.9,
                    label="Left comp." if first else "",
                )
                _hdi_line(
                    ax, x_win, components["comp_r"][:, i, p, win],
                    color="gray", linestyle="--", linewidth=0.9,
                    label="Right comp." if first else "",
                )
                _hdi_line(
                    ax, x_win, components["total"][:, i, win],
                    color="C0", linestyle="-", linewidth=1.5,
                    label="Fitted signal" if first else "",
                )

            _window_bounds(ax, peak)
            ax.set_title(f"{peak.molecule_id} ({_trace_label(int(t), chromatogram_ids)})", fontsize=9)
            if p == 0:
                ax.set_ylabel("Intensity [AU]", fontsize=8)
            if i == n_sel - 1:
                ax.set_xlabel("Retention time [min]", fontsize=8)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.tick_params(labelsize=7)

    if n_sel > 0 and n_peak > 0 and axes[0, 0].has_data():
        axes[0, 0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig, axes


def plot_fit_combined(
    time: NDArray[np.float64],
    signal: NDArray[np.float64],
    peaks: list[PeakAnnotation],
    posterior: InferenceData | None,
    *,
    baselines: list[BaselineAnnotation] | None = None,
    trace_indices: NDArray[np.intp] | None = None,
    chromatogram_ids: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, np.ndarray[Any, Any]]:
    """Combined posterior fit view.

    One row per selected trace, single column spanning the union of peak
    windows and (optional) baseline regions. Shows raw scatter plus
    baseline (gray) and total fit (blue) median+HDI when a posterior is
    available.
    """
    time_arr = np.asarray(time, dtype=float)
    signal_arr = np.asarray(signal, dtype=float)
    n_trace = time_arr.shape[0]

    if trace_indices is None:
        sel: NDArray[np.intp] = np.arange(n_trace, dtype=np.intp)
    else:
        sel = np.asarray(trace_indices, dtype=np.intp)
    n_sel = int(sel.shape[0])

    rt_lo = float(min(p.rt_min for p in peaks))
    rt_hi = float(max(p.rt_max for p in peaks))
    if baselines:
        rt_lo = min(rt_lo, float(min(b.rt_min for b in baselines)))
        rt_hi = max(rt_hi, float(max(b.rt_max for b in baselines)))

    components = None
    win: NDArray[np.bool_] = np.zeros(0, dtype=bool)
    x_win: NDArray[np.float64] = np.zeros(0, dtype=float)
    if posterior is not None and n_sel > 0:
        x_full = np.asarray(np.nanmedian(time_arr, axis=0), dtype=float)
        components = _evaluate_components(posterior, peaks, x_full, trace_indices=sel)
        win = (x_full >= rt_lo) & (x_full <= rt_hi)
        x_win = x_full[win]

    if figsize is None:
        figsize = (8, 3 * max(n_sel, 1))
    fig, axes = plt.subplots(n_sel, 1, figsize=figsize, squeeze=False)

    for i, t in enumerate(sel):
        ax = axes[i, 0]
        x_trace = time_arr[t]
        y_trace = signal_arr[t]
        mask = (x_trace >= rt_lo) & (x_trace <= rt_hi)
        xr = x_trace[mask]
        yr = y_trace[mask]
        fin = np.isfinite(xr) & np.isfinite(yr)
        if fin.any():
            ax.scatter(
                xr[fin], yr[fin], s=25, color="gray", zorder=1,
                label="Raw signal" if i == 0 else "",
            )

        if components is not None:
            first = i == 0
            _hdi_line(
                ax, x_win, components["baseline"][:, i, win],
                color="gray", linestyle="-", linewidth=1.0,
                label="Baseline" if first else "",
            )
            _hdi_line(
                ax, x_win, components["total"][:, i, win],
                color="C0", linestyle="-", linewidth=1.5,
                label="Fitted signal" if first else "",
            )

        for peak in peaks:
            _window_bounds(ax, peak)

        ax.set_title(f"Combined — {_trace_label(int(t), chromatogram_ids)}", fontsize=9)
        ax.set_ylabel("Intensity [AU]", fontsize=8)
        if i == n_sel - 1:
            ax.set_xlabel("Retention time [min]", fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=7)

    if n_sel > 0 and axes[0, 0].has_data():
        axes[0, 0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Pre-fit geometric diagnostic
# ---------------------------------------------------------------------------


def plot_geometric_diagnostic(
    time: NDArray[np.float64],
    signal: NDArray[np.float64],
    peaks: list[PeakAnnotation],
    *,
    k_mad: float = 3.0,
    show_mad_region: bool = False,
    show_outlier_labels: bool = False,
    chromatogram_ids: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, np.ndarray[Any, Any], list[int]]:
    """Pre-fit per-trace ``(sigma_eff, alpha_asym)`` scatter with MAD bounds.

    One subplot per peak window. Traces whose ``(sigma_eff, alpha_asym)``
    falls outside ``k_mad * MAD`` on either axis (computed per peak over
    valid-FWHM traces) are marked as outliers. A trace is flagged overall
    if it is an outlier in at least one peak window.

    Returns ``(fig, axes, outlier_trace_indices)``. Works without a
    posterior — this is a prior diagnostic.
    """
    time_arr = np.asarray(time, dtype=float)
    signal_arr = np.asarray(signal, dtype=float)
    n_trace = time_arr.shape[0]
    n_peak = len(peaks)

    if figsize is None:
        figsize = (4 * max(n_peak, 1), 3.5)
    fig, axes = plt.subplots(1, max(n_peak, 1), figsize=figsize, squeeze=False)

    x_common = np.nanmedian(time_arr, axis=0)
    outlier_set: set[int] = set()

    for p, peak in enumerate(peaks):
        ax = axes[0, p]
        win = (x_common >= peak.rt_min) & (x_common <= peak.rt_max)
        y_win = signal_arr[:, win]
        x_win = x_common[win]

        if x_win.size < 4:
            ax.set_title(f"{peak.molecule_id} (window too narrow)", fontsize=9)
            continue

        geo = _trace_fwhm_geometry(x_win, y_win)
        sigma_eff, alpha_asym = fwhm_geometry_to_sigma_alpha(geo)
        valid = np.asarray(geo.fwhm_valid, dtype=bool)

        if not valid.any():
            ax.set_title(f"{peak.molecule_id} (no valid FWHM)", fontsize=9)
            continue

        sig_v = sigma_eff[valid]
        alp_v = alpha_asym[valid]
        med_s, med_a = float(np.median(sig_v)), float(np.median(alp_v))
        mad_s = float(median_abs_deviation(sig_v, nan_policy="omit")) if sig_v.size >= 2 else 0.0  # pyright: ignore[reportUnknownArgumentType]
        mad_a = float(median_abs_deviation(alp_v, nan_policy="omit")) if alp_v.size >= 2 else 0.0  # pyright: ignore[reportUnknownArgumentType]

        outlier = np.zeros(n_trace, dtype=bool)
        if mad_s > 0.0:
            outlier |= np.abs(sigma_eff - med_s) > k_mad * mad_s
        if mad_a > 0.0:
            outlier |= np.abs(alpha_asym - med_a) > k_mad * mad_a
        outlier &= valid

        inlier = valid & ~outlier

        heights = np.asarray(geo.apex_height, dtype=float)
        h_max = float(np.nanmax(heights[valid])) if valid.any() else 0.0
        sizes = np.full(n_trace, 30.0)
        if h_max > 0.0:
            sizes = 15.0 + 120.0 * np.clip(heights / h_max, 0.0, 1.0)

        if inlier.any():
            ax.scatter(
                alpha_asym[inlier], sigma_eff[inlier],
                s=sizes[inlier], color="C0", edgecolor="none",
                label="Inlier" if p == 0 else "",
            )
        if outlier.any():
            ax.scatter(
                alpha_asym[outlier], sigma_eff[outlier],
                s=sizes[outlier], color="red", edgecolor="none",
                label="Outlier" if p == 0 else "",
            )
            for t in np.where(outlier)[0]:
                if show_outlier_labels:
                    ax.annotate(
                        _trace_label(int(t), chromatogram_ids),
                        (alpha_asym[t], sigma_eff[t]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points",
                    )
                outlier_set.add(int(t))

        ax.scatter(
            [med_a], [med_s],
            marker="x", color="black", s=60, linewidths=1.5,
            label="Median" if p == 0 else "",
        )
        if show_mad_region and mad_s > 0.0 and mad_a > 0.0:
            ax.add_patch(plt.Rectangle(  # type: ignore[attr-defined]
                (med_a - k_mad * mad_a, med_s - k_mad * mad_s),
                2 * k_mad * mad_a, 2 * k_mad * mad_s,
                fill=False, edgecolor="red", linestyle="--", linewidth=0.8,
                label=f"±{k_mad:g}·MAD" if p == 0 else "",
            ))

        ax.set_title(peak.molecule_id, fontsize=9)
        ax.set_xlabel("alpha_asym", fontsize=8)
        if p == 0:
            ax.set_ylabel("sigma_eff [min]", fontsize=8)
        ax.axvline(0.0, color="gray", linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=7)

    if n_peak > 0 and axes[0, 0].has_data():
        axes[0, 0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig, axes, sorted(outlier_set)
