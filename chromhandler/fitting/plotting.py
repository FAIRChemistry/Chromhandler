"""Diagnostic and posterior plotting for chromatographic fitting.

Two layers:

1. **Axes-level primitives** -- ``add_*`` functions that take an existing
   :class:`matplotlib.axes.Axes`, mutate it, and return it. Composable
   building blocks. Idiomatic matplotlib.

2. **Figure-level convenience** -- ``plot_*`` functions that build a
   complete :class:`matplotlib.figure.Figure` for a common case by
   composing the axes primitives.

Matplotlib is the only plot dependency. The foundations modules
(``preprocessing``, ``baseline``, ``noise``, ``prepared_dataset``,
``annotations``) deliberately do not import matplotlib; that import
lives only here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from chromhandler.fitting.prepared_dataset import PreparedDataset


def add_signal(
    ax: Axes,
    dataset: PreparedDataset,
    trace_idx: int,
    *,
    color: str = "tab:gray",
    linewidth: float = 0.8,
    alpha: float = 0.85,
) -> Axes:
    """Plot one trace's raw signal on the given axes.

    NaN-padded samples are masked out before plotting.

    Args:
        ax: Target axes (mutated and returned).
        dataset: The prepared dataset.
        trace_idx: Which trace to plot.
        color: Line colour.
        linewidth: Line width in points.
        alpha: Line opacity.

    Returns:
        The same ``ax`` passed in.
    """
    valid = dataset.valid_mask[trace_idx]
    t = dataset.time[trace_idx][valid]
    s = dataset.signal[trace_idx][valid]
    ax.plot(t, s, color=color, linewidth=linewidth, alpha=alpha)
    return ax


def add_annotation_regions(
    ax: Axes,
    dataset: PreparedDataset,
    *,
    peak_color: str = "tab:orange",
    baseline_color: str = "tab:green",
    alpha: float = 0.15,
) -> Axes:
    """Shade peak windows and baseline regions across the time axis.

    Each :class:`PeakAnnotation` and :class:`BaselineAnnotation` becomes a
    vertical band spanning its ``[rt_min, rt_max]`` range.

    Args:
        ax: Target axes (mutated and returned).
        dataset: Source of the annotations.
        peak_color: Fill colour for peak windows.
        baseline_color: Fill colour for baseline regions.
        alpha: Fill opacity.

    Returns:
        The same ``ax`` passed in.
    """
    for p in dataset.peak_annotations:
        ax.axvspan(p.rt_min, p.rt_max, color=peak_color, alpha=alpha)
    for b in dataset.baseline_annotations:
        ax.axvspan(b.rt_min, b.rt_max, color=baseline_color, alpha=alpha)
    return ax


def add_baseline(
    ax: Axes,
    dataset: PreparedDataset,
    trace_idx: int,
    *,
    show_noise_band: bool = True,
    color: str = "tab:blue",
    linewidth: float = 1.0,
    linestyle: str = "--",
    band_alpha: float = 0.15,
) -> Axes:
    """Overlay one trace's OLS baseline, optionally with a +/-noise ribbon.

    The baseline is drawn across the full valid time range of the trace
    (NaN-padded samples excluded). The optional noise ribbon is a
    translucent fill at ``baseline +/- noise_per_trace[trace_idx]``.

    Args:
        ax: Target axes (mutated and returned).
        dataset: Source of baseline parameters and noise.
        trace_idx: Which trace to plot.
        show_noise_band: If True, fill between ``baseline +/- noise``.
        color: Line + fill colour.
        linewidth: Line width in points.
        linestyle: Line style (default dashed).
        band_alpha: Noise-band fill opacity.

    Returns:
        The same ``ax`` passed in.
    """
    valid = dataset.valid_mask[trace_idx]
    t = dataset.time[trace_idx][valid]
    intercept = dataset.baseline_intercept[trace_idx]
    slope = dataset.baseline_slope[trace_idx]
    noise = dataset.noise_per_trace[trace_idx]
    baseline = intercept + slope * t
    if show_noise_band:
        ax.fill_between(
            t,
            baseline - noise,
            baseline + noise,
            color=color,
            alpha=band_alpha,
            linewidth=0,
        )
    ax.plot(t, baseline, color=color, linewidth=linewidth, linestyle=linestyle)
    return ax


def add_model(
    ax: Axes,
    dataset: PreparedDataset,
    trace_idx: int,
    model_fn: Callable[[NDArray[np.float64], int], NDArray[np.float64]],
    *,
    color: str = "tab:red",
    linewidth: float = 1.0,
    linestyle: str = "-",
) -> Axes:
    """Overlay an arbitrary model evaluation on one trace.

    Decoupled from MCMC: ``model_fn`` is any callable that takes
    ``(time[n_valid], trace_idx)`` and returns predicted signal of the
    same shape. NaN-padded samples are masked out before the call.

    Use cases include prior predictive checks, MAP fits, posterior
    median overlays, and hand-specified parameter sweeps.

    Args:
        ax: Target axes (mutated and returned).
        dataset: Source of time + valid mask.
        trace_idx: Which trace to evaluate.
        model_fn: Callable ``(time, trace_idx) -> predicted_signal``.
        color: Line colour.
        linewidth: Line width in points.
        linestyle: Line style.

    Returns:
        The same ``ax`` passed in.
    """
    valid = dataset.valid_mask[trace_idx]
    t = dataset.time[trace_idx][valid]
    predicted = model_fn(t, trace_idx)
    ax.plot(t, predicted, color=color, linewidth=linewidth, linestyle=linestyle)
    return ax


def _grid_shape(n_trace: int, n_cols: int) -> tuple[int, int]:
    """Compute ``(n_rows, n_cols)`` for a square-ish grid of n_trace panels."""
    n_rows = (n_trace + n_cols - 1) // n_cols
    return n_rows, n_cols


def _hide_unused_axes(fig: Figure, n_trace: int) -> None:
    """Hide any axes beyond the n_trace data panels (trailing grid cells)."""
    for ax in fig.axes[n_trace:]:
        ax.set_visible(False)


def plot_overview(
    dataset: PreparedDataset,
    path: str | Path | None = None,
    *,
    n_cols: int = 3,
    figsize_per_panel: tuple[float, float] = (4.0, 2.5),
) -> Figure:
    """Per-trace grid: signal + annotation regions.

    A quick sanity check that the data and annotations look right before
    any fitting. One panel per trace; trailing grid cells (if any) are
    hidden.

    Args:
        dataset: The prepared dataset.
        path: If provided, save the figure to this path with
            ``fig.savefig``. The figure is always returned regardless.
        n_cols: Number of columns in the grid.
        figsize_per_panel: ``(width, height)`` per panel in inches.

    Returns:
        The constructed :class:`matplotlib.figure.Figure`.
    """
    n_rows, _ = _grid_shape(dataset.n_trace, n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        sharex=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for i in range(dataset.n_trace):
        ax = flat_axes[i]
        add_signal(ax, dataset, trace_idx=i)
        add_annotation_regions(ax, dataset)
        ax.set_title(f"trace {i}")
    _hide_unused_axes(fig, dataset.n_trace)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path)
    return fig


def plot_baseline_prior(
    dataset: PreparedDataset,
    *,
    overlay: str = "single",
    ax_size: tuple[float, float] = (10.0, 2.8),
    save: str | Path | None = None,
) -> Figure:
    """Plot the baseline prior (median + OLS-SE band) per group.

    One row per group, single panel per row. The x-axis spans the union
    of peak-window and baseline-region bounds. Points inside peak windows
    are drawn at full opacity; points inside baseline regions are
    overlaid as red 'x' markers; points in the gaps between annotated
    regions are dimmed (alpha=0.5). The baseline median is drawn as a
    dashed line and the +/- OLS SE band is shaded.

    Args:
        dataset: The prepared dataset.
        overlay: ``"single"`` -> one row per trace, ``"sample"`` -> one
            row per sample (groups traces whose ``trace_id`` shares a
            ``sample_id/`` prefix), ``"all"`` -> single row with every
            trace overlaid.
        ax_size: ``(width, height)`` in inches per row.
        save: If given, write the figure to this path before returning.

    Returns:
        The constructed :class:`matplotlib.figure.Figure`.
    """
    import numpy as np

    from chromhandler.fitting.model import _compute_baseline_se  # type: ignore[attr-defined]

    peak_anns = dataset.peak_annotations
    baseline_anns = dataset.baseline_annotations
    if not peak_anns and not baseline_anns:
        raise ValueError(
            "plot_baseline_prior: dataset has no peak or baseline "
            "annotations to plot against."
        )

    rt_mins = [a.rt_min for a in peak_anns] + [a.rt_min for a in baseline_anns]
    rt_maxs = [a.rt_max for a in peak_anns] + [a.rt_max for a in baseline_anns]
    x_lo = float(min(rt_mins))
    x_hi = float(max(rt_maxs))

    intercept_se, slope_se = _compute_baseline_se(dataset)

    if overlay == "single":
        groups: list[list[int]] = [[i] for i in range(dataset.n_trace)]
        group_labels = [dataset.trace_ids[i] for i in range(dataset.n_trace)]
    elif overlay == "sample":
        by_sample: dict[str, list[int]] = {}
        for i, tid in enumerate(dataset.trace_ids):
            key = tid.split("/", 1)[0] if "/" in tid else tid
            by_sample.setdefault(key, []).append(i)
        group_labels = list(by_sample.keys())
        groups = [by_sample[k] for k in group_labels]
    elif overlay == "all":
        groups = [list(range(dataset.n_trace))]
        group_labels = ["all traces"]
    else:
        raise ValueError(
            f"plot_baseline_prior: overlay must be 'single', 'sample', "
            f"or 'all'; got {overlay!r}."
        )

    n_rows = len(groups)
    width, height = ax_size
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(width, height * n_rows),
        squeeze=False, sharex=True,
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])

    for row, (idxs, label) in enumerate(zip(groups, group_labels, strict=True)):
        ax = axes[row, 0]
        for k, tr in enumerate(idxs):
            color = colors[k % len(colors)]
            t = dataset.time[tr]
            s = dataset.signal[tr]
            in_x = (t >= x_lo) & (t <= x_hi) & np.isfinite(s)

            peak_mask = np.zeros_like(t, dtype=bool)
            for ann in peak_anns:
                peak_mask |= (t >= ann.rt_min) & (t <= ann.rt_max)
            base_mask = np.zeros_like(t, dtype=bool)
            for ann in baseline_anns:
                base_mask |= (t >= ann.rt_min) & (t <= ann.rt_max)

            gap_mask = in_x & ~peak_mask & ~base_mask
            peak_in = in_x & peak_mask
            base_in = in_x & base_mask

            ax.plot(
                t[peak_in], s[peak_in],
                color=color, lw=0.9, alpha=1.0,
                label=dataset.trace_ids[tr] if overlay != "single" else None,
            )
            ax.plot(t[gap_mask], s[gap_mask], color=color, lw=0.8, alpha=0.5)
            ax.plot(
                t[base_in], s[base_in],
                linestyle="none", marker="x", color="red",
                markersize=5, alpha=0.9,
            )

            t_dense = np.linspace(x_lo, x_hi, 400)
            intercept = float(dataset.baseline_intercept[tr])
            slope = float(dataset.baseline_slope[tr])
            baseline = intercept + slope * t_dense
            band = np.sqrt(intercept_se[tr] ** 2 + (slope_se[tr] * t_dense) ** 2)
            ax.fill_between(
                t_dense, baseline - band, baseline + band,
                color=color, alpha=0.2, linewidth=0,
            )
            ax.plot(t_dense, baseline, color=color, lw=1.4, linestyle="--")

        for ann in peak_anns:
            ax.axvspan(ann.rt_min, ann.rt_max, color="tab:orange", alpha=0.08)
        for ann in baseline_anns:
            ax.axvspan(ann.rt_min, ann.rt_max, color="tab:green", alpha=0.08)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylabel(label)
        if overlay != "single" and len(idxs) > 1:
            ax.legend(fontsize=7, loc="best")

    axes[-1, 0].set_xlabel("retention time (min)")
    axes[0, 0].set_title("baseline prior: median (dashed) ± OLS SE band")
    fig.tight_layout()
    if save is not None:
        fig.savefig(save)
    return fig


def plot_baseline_diagnostic(
    dataset: PreparedDataset,
    path: str | Path | None = None,
    *,
    n_cols: int = 3,
    figsize_per_panel: tuple[float, float] = (4.0, 2.5),
) -> Figure:
    """Per-trace grid: signal + annotation regions + OLS baseline + noise.

    The canonical pre-fit diagnostic for the foundations layer. Each
    panel title shows the per-trace ``dt`` and ``noise`` so the user
    sees at a glance whether the OLS baseline and noise estimate look
    sensible.

    Args:
        dataset: The prepared dataset.
        path: If provided, save the figure to this path. The figure is
            always returned regardless.
        n_cols: Number of columns in the grid.
        figsize_per_panel: ``(width, height)`` per panel in inches.

    Returns:
        The constructed :class:`matplotlib.figure.Figure`.
    """
    n_rows, _ = _grid_shape(dataset.n_trace, n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        sharex=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for i in range(dataset.n_trace):
        ax = flat_axes[i]
        add_annotation_regions(ax, dataset)
        add_signal(ax, dataset, trace_idx=i)
        add_baseline(ax, dataset, trace_idx=i, show_noise_band=True)
        dt = dataset.dt_per_trace[i]
        noise = dataset.noise_per_trace[i]
        ax.set_title(f"trace {i}: dt={dt:.4f}, noise={noise:.3f}")
    _hide_unused_axes(fig, dataset.n_trace)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path)
    return fig
