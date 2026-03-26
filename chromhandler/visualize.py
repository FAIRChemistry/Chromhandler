from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from .annotations import PeakAnnotation
    from .handler import Handler
    from .model import Chromatogram, Peak


@dataclass(frozen=True)
class PlotSample:
    """Lightweight plotting view of one sample."""

    id: str
    chromatograms: tuple[Chromatogram, ...]


def _peak_rt_min(peak: Peak) -> float:
    """Retention time (minutes) for plotting — uses :attr:`Peak.location` mean."""
    return float(peak.location.mean)


def _peak_area_mean(peak: Peak) -> float:
    """Peak area scalar for bar height — uses :attr:`Peak.area` mean."""
    return float(peak.area.mean)


def _iter_chromatograms(plot_samples: Sequence[PlotSample]) -> Iterable[Chromatogram]:
    """Yield chromatograms in stable handler/sample order."""
    for sample in plot_samples:
        yield from sample.chromatograms


def _resolve_plot_samples(handler: Handler, chromatogram_ids: Sequence[str]) -> list[PlotSample]:
    """Resolve the effective chromatograms to plot once, up front."""
    requested_ids = list(dict.fromkeys(chromatogram_ids))
    available_ids = [chrom.id for sample in handler.samples for chrom in sample.chromatograms]

    selected_ids: set[str] | None = None
    if requested_ids:
        available_id_set = set(available_ids)
        missing = [chrom_id for chrom_id in requested_ids if chrom_id not in available_id_set]
        if missing:
            raise ValueError(f"Unknown chromatogram IDs for visualize(): {missing}")
        selected_ids = set(requested_ids)

    plot_samples: list[PlotSample] = []
    for sample in handler.samples:
        chromatograms = tuple(
            chrom for chrom in sample.chromatograms if selected_ids is None or chrom.id in selected_ids
        )
        if chromatograms:
            plot_samples.append(PlotSample(id=sample.id, chromatograms=chromatograms))

    if plot_samples:
        return plot_samples

    if requested_ids:
        raise ValueError("visualize(): chromatogram_ids filtering left no chromatograms to plot.")
    raise ValueError("visualize(): handler contains no chromatograms to plot.")


def _global_y_bounds(plot_samples: Sequence[PlotSample]) -> tuple[float, float, bool]:
    """Return padded (y_min, y_max) and whether any chromatogram signal exists."""
    y_min = float("inf")
    y_max = float("-inf")
    has_signal = False

    for chrom in _iter_chromatograms(plot_samples):
        if chrom.signal:
            y_min = min(y_min, min(chrom.signal))
            y_max = max(y_max, max(chrom.signal))
            has_signal = True

    if not has_signal:
        for chrom in _iter_chromatograms(plot_samples):
            for peak in chrom.peaks:
                y_min = min(y_min, 0.0)
                y_max = max(y_max, _peak_area_mean(peak))
                has_signal = True

    if not has_signal:
        return 0.0, 1.0, False

    span = y_max - y_min
    if span > 0:
        return y_min - 0.05 * span, y_max + 0.05 * span, True
    return 0.0, 1.0, True


def _sample_y_bounds(sample: PlotSample) -> tuple[float, float]:
    """Y-limits for one sample (signal or peak-area fallback), with padding."""
    smin, smax = float("inf"), float("-inf")
    has_data = False

    for chrom in sample.chromatograms:
        if chrom.signal:
            smin = min(smin, min(chrom.signal))
            smax = max(smax, max(chrom.signal))
            has_data = True

    if not has_data:
        for chrom in sample.chromatograms:
            for peak in chrom.peaks:
                smin = min(smin, 0.0)
                smax = max(smax, _peak_area_mean(peak))
                has_data = True

    if not has_data or smax <= smin:
        return 0.0, 1.0

    pad = 0.05 * (smax - smin)
    return smin - pad, smax + pad


def _molecule_colors_and_rt_mapper(
    plt: Any,
    plot_samples: Sequence[PlotSample],
    *,
    show_peaks: bool,
) -> tuple[dict[str, object], Any]:
    """tab10 colors per molecule_id; viridis mapper for unassigned peaks."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    all_rts: list[float] = []
    molecule_ids: set[str] = set()

    if show_peaks:
        for chrom in _iter_chromatograms(plot_samples):
            for peak in chrom.peaks:
                all_rts.append(_peak_rt_min(peak))
                if peak.molecule_id:
                    molecule_ids.add(peak.molecule_id)

    tab10 = plt.get_cmap("tab10")
    molecule_colors: dict[str, object] = {}
    if molecule_ids:
        ordered_ids = sorted(molecule_ids)
        molecule_colors = {molecule_id: tab10(index % 10) for index, molecule_id in enumerate(ordered_ids)}

    if all_rts:
        rt_arr = np.asarray(all_rts, dtype=float)
        vmin, vmax = float(rt_arr.min()), float(rt_arr.max())
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        rt_norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        rt_norm = Normalize(vmin=0.0, vmax=1.0)

    scalar_mappable = ScalarMappable(norm=rt_norm, cmap=plt.get_cmap("viridis"))
    scalar_mappable.set_array([])
    return molecule_colors, scalar_mappable


def _peak_color_for_plot(
    peak: Peak,
    *,
    molecule_colors: dict[str, object],
    sm: Any,
) -> object:
    """Return the display color for a peak marker."""
    if peak.molecule_id and peak.molecule_id in molecule_colors:
        return molecule_colors[peak.molecule_id]
    rt = _peak_rt_min(peak)
    rt_discrete = round(rt / 0.05) * 0.05
    return sm.to_rgba(np.array([rt_discrete]))[0]


def _chromatogram_label(chrom: Chromatogram, index: int) -> str:
    """Human-readable legend label for one chromatogram trace."""
    if chrom.reaction_time is not None:
        return f"{chrom.reaction_time:.1f}"
    return chrom.id or f"chrom {index}"


def _draw_peak_annotations(
    ax: Any,
    plt: Any,
    *,
    show_peak_annotations: bool,
    peak_annotations: list[PeakAnnotation] | None,
) -> None:
    """Shade configured peak windows on the provided axis."""
    if not show_peak_annotations or not peak_annotations:
        return

    ann_tab10 = plt.get_cmap("tab10")
    mol_ids: list[str] = []
    for ann in peak_annotations:
        if ann.molecule_id not in mol_ids:
            mol_ids.append(ann.molecule_id)
    mol_to_color = {mid: ann_tab10(i % 10) for i, mid in enumerate(mol_ids)}

    for ann in peak_annotations:
        color = mol_to_color.get(ann.molecule_id, "gray")
        ax.axvspan(
            ann.rt_min,
            ann.rt_max,
            alpha=0.2,
            color=color,
            label=None,
            zorder=0,
        )


def _draw_signal_traces(ax: Any, plt: Any, chromatograms: Sequence[Chromatogram]) -> None:
    """Draw raw chromatogram signal traces for one plotted sample."""
    viridis_cmap = plt.get_cmap("viridis")
    total = len(chromatograms)

    for index, chrom in enumerate(chromatograms):
        if not (chrom.time and chrom.signal):
            continue
        color_value = index / (total - 1) if total > 1 else 0.5
        ax.plot(
            chrom.time,
            chrom.signal,
            label=_chromatogram_label(chrom, index),
            color=viridis_cmap(color_value),
            zorder=2,
        )


def _draw_peak_markers(
    ax: Any,
    chromatograms: Sequence[Chromatogram],
    *,
    molecule_colors: dict[str, object],
    sm: Any,
    assigned_only: bool,
    linestyle_for_index: Callable[[int], object],
    linewidth: float,
) -> None:
    """Draw vertical peak markers for one plotted sample."""
    for chrom_index, chrom in enumerate(chromatograms):
        for peak in chrom.peaks:
            if assigned_only and not peak.molecule_id:
                continue
            peak_rt = _peak_rt_min(peak)
            peak_height = _peak_area_mean(peak)
            ax.plot(
                [peak_rt, peak_rt],
                [0, peak_height],
                color=_peak_color_for_plot(peak, molecule_colors=molecule_colors, sm=sm),
                linestyle=linestyle_for_index(chrom_index),
                alpha=0.7,
                linewidth=linewidth,
                label=None,
                zorder=1,
            )


def _overlay_peak_linestyle(chrom_index: int) -> object:
    """Dash pattern used to distinguish chromatograms in overlay mode."""
    return (0, (1, chrom_index + 1))


def _grid_peak_linestyle(_: int) -> object:
    """Grid mode keeps peak markers visually simple."""
    return "-"


def _axis_ylabel(has_signal_data: bool) -> str:
    """Return the y-axis label matching the rendered data."""
    return "Intensity" if has_signal_data else "Peak Area"


def _configure_axis(
    ax: Any,
    *,
    rt_min: float | None,
    rt_max: float | None,
    ylabel: str | None,
) -> None:
    """Apply common axis labels, grid, and x-limits."""
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("Retention time [min]")
    ax.tick_params(axis="x", labelbottom=True)
    ax.grid(True, alpha=0.3)
    if rt_min is not None and rt_max is not None:
        ax.set_xlim(rt_min, rt_max)


def _draw_signal_legend(ax: Any, *, deduplicate: bool) -> None:
    """Render the signal legend when traces produced legend entries."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if deduplicate:
        by_label: dict[str, Any] = {}
        for label, handle in zip(labels, handles, strict=False):
            if label not in by_label:
                by_label[label] = handle
        handles = list(by_label.values())
        labels = list(by_label.keys())
    ax.legend(
        handles,
        labels,
        loc="upper right",
        fontsize=6,
        title="reaction time [min]",
        title_fontsize=7,
    )


def _create_grid_axes(
    plt: Any,
    *,
    n_samples: int,
    n_cols: int,
    figsize: tuple[float, float],
    share_y: bool,
) -> tuple[Figure, NDArray[Any]]:
    """Create a subplot grid and return only the visible axes."""
    n_rows = int(np.ceil(n_samples / n_cols))
    fig, axes = cast(
        "tuple[Figure, NDArray[Any]]",
        plt.subplots(n_rows, n_cols, figsize=figsize, sharey=share_y, sharex=True),
    )  # pylint: disable=fixme
    fig.patch.set_facecolor("none")

    axes_array = np.atleast_1d(np.asarray(axes)).flatten()
    for axis in axes_array:
        axis.patch.set_facecolor("none")

    for axis in axes_array[n_samples:]:
        axis.set_visible(False)

    return fig, axes_array[:n_samples]


def _plot_overlay(
    plot_samples: Sequence[PlotSample],
    *,
    figsize: tuple[float, float],
    y_min: float,
    y_max: float,
    has_signal_data: bool,
    molecule_colors: dict[str, object],
    sm: Any,
    show_peaks: bool,
    assigned_only: bool,
    rt_min: float | None,
    rt_max: float | None,
    show_peak_annotations: bool,
    peak_annotations: list[PeakAnnotation] | None,
    show_legend: bool,
) -> tuple[Figure, Axes]:
    """Render all selected samples into a single overlay axis."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("none")
    ax.patch.set_facecolor("none")

    for sample in plot_samples:
        _draw_signal_traces(ax, plt, sample.chromatograms)
        if show_peaks:
            _draw_peak_markers(
                ax,
                sample.chromatograms,
                molecule_colors=molecule_colors,
                sm=sm,
                assigned_only=assigned_only,
                linestyle_for_index=_overlay_peak_linestyle,
                linewidth=1.5,
            )

    _configure_axis(
        ax,
        rt_min=rt_min,
        rt_max=rt_max,
        ylabel=_axis_ylabel(has_signal_data),
    )
    if show_legend:
        _draw_signal_legend(ax, deduplicate=True)
    _draw_peak_annotations(
        ax,
        plt,
        show_peak_annotations=show_peak_annotations,
        peak_annotations=peak_annotations,
    )
    ax.set_ylim(y_min, y_max)
    fig.tight_layout()
    return fig, ax


def _plot_grid(
    plot_samples: Sequence[PlotSample],
    *,
    n_cols: int,
    figsize: tuple[float, float],
    y_min: float,
    y_max: float,
    has_signal_data: bool,
    molecule_colors: dict[str, object],
    sm: Any,
    show_peaks: bool,
    assigned_only: bool,
    share_y: bool,
    rt_min: float | None,
    rt_max: float | None,
    show_peak_annotations: bool,
    peak_annotations: list[PeakAnnotation] | None,
    show_legend: bool,
) -> tuple[Figure, NDArray[Any]]:
    """Render one subplot per plotted sample."""
    import matplotlib.pyplot as plt

    fig, axes = _create_grid_axes(
        plt,
        n_samples=len(plot_samples),
        n_cols=n_cols,
        figsize=figsize,
        share_y=share_y,
    )

    for index, (sample, ax) in enumerate(zip(plot_samples, axes, strict=False)):
        if show_peaks:
            _draw_peak_markers(
                ax,
                sample.chromatograms,
                molecule_colors=molecule_colors,
                sm=sm,
                assigned_only=assigned_only,
                linestyle_for_index=_grid_peak_linestyle,
                linewidth=2.0,
            )
        _draw_signal_traces(ax, plt, sample.chromatograms)

        ax.text(
            0.02,
            0.95,
            sample.id,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="left",
        )
        _configure_axis(
            ax,
            rt_min=rt_min,
            rt_max=rt_max,
            ylabel=_axis_ylabel(has_signal_data) if index % n_cols == 0 else None,
        )
        _draw_peak_annotations(
            ax,
            plt,
            show_peak_annotations=show_peak_annotations,
            peak_annotations=peak_annotations,
        )
        if show_legend:
            _draw_signal_legend(ax, deduplicate=False)
        if share_y:
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(*_sample_y_bounds(sample))

    fig.tight_layout()
    return fig, axes


def visualize(
    handler: Handler,
    n_cols: int = 1,
    figsize: tuple[float, float] | None = None,
    width_per_ax: float = 10.0,
    height_per_ax: float = 3.0,
    show_peaks: bool = True,
    rt_min: float | None = None,
    rt_max: float | None = None,
    assigned_only: bool = False,
    overlay: bool = False,
    share_y: bool = False,
    show_peak_annotations: bool = True,
    peak_annotations: list[PeakAnnotation] | None = None,
    chromatogram_ids: list[str] | None = None,
    show_legend: bool = True,
) -> tuple[Figure, Any]:
    """Build a matplotlib figure for chromatograms.

    Returns ``(fig, ax)`` in *overlay* mode (single axis) or ``(fig, axes)``
    in grid mode, where *axes* is a 1-D array containing only plotted subplots.

    The caller owns the figure: save with ``fig.savefig(...)``, show with
    ``plt.show()``, or close with ``plt.close(fig)``.

    Args:
        handler: The Handler instance containing the data.
        n_cols: Number of columns in the subplot grid (grid mode only).
        figsize: Figure size in inches. If None, derived from *width_per_ax*
            and *height_per_ax*.
        width_per_ax: Subplot width in inches when *figsize* is None.
        height_per_ax: Subplot height in inches when *figsize* is None.
        show_peaks: If True, draw detected peaks as vertical segments.
        rt_min: Minimum retention time; both *rt_min* and *rt_max* must be set
            to clip the x-axis.
        rt_max: Maximum retention time.
        assigned_only: If True, only peaks with a ``molecule_id``.
        overlay: If True, all samples share one axis.
        share_y: If True, subplots share y-limits (grid mode; uses global range).
        show_peak_annotations: Shade :class:`~chromhandler.annotations.PeakAnnotation`
            or peak-window regions.
        peak_annotations: List of shaded peak-window annotations.
        chromatogram_ids: Chromatogram IDs to plot. An empty list means all
            chromatograms. Unknown IDs raise ``ValueError``.
        show_legend: If True, show legends for reaction-time labels.
    """
    import matplotlib.pyplot as plt

    if chromatogram_ids is None:
        chromatogram_ids = []
    plot_samples = _resolve_plot_samples(handler, chromatogram_ids)
    n_plot_samples = len(plot_samples)

    if figsize is None:
        if overlay:
            figsize = (width_per_ax * max(n_cols, 2), height_per_ax)
        else:
            n_rows = int(np.ceil(n_plot_samples / n_cols))
            figsize = (width_per_ax * n_cols, height_per_ax * n_rows)

    y_min, y_max, has_signal_data = _global_y_bounds(plot_samples)
    molecule_colors, sm = _molecule_colors_and_rt_mapper(plt, plot_samples, show_peaks=show_peaks)

    if overlay:
        return _plot_overlay(
            plot_samples,
            figsize=figsize,
            y_min=y_min,
            y_max=y_max,
            has_signal_data=has_signal_data,
            molecule_colors=molecule_colors,
            sm=sm,
            show_peaks=show_peaks,
            assigned_only=assigned_only,
            rt_min=rt_min,
            rt_max=rt_max,
            show_peak_annotations=show_peak_annotations,
            peak_annotations=peak_annotations,
            show_legend=show_legend,
        )

    return _plot_grid(
        plot_samples,
        n_cols=n_cols,
        figsize=figsize,
        y_min=y_min,
        y_max=y_max,
        has_signal_data=has_signal_data,
        molecule_colors=molecule_colors,
        sm=sm,
        show_peaks=show_peaks,
        assigned_only=assigned_only,
        share_y=share_y,
        rt_min=rt_min,
        rt_max=rt_max,
        show_peak_annotations=show_peak_annotations,
        peak_annotations=peak_annotations,
        show_legend=show_legend,
    )
