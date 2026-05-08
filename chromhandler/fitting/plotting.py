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

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from matplotlib.axes import Axes
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
