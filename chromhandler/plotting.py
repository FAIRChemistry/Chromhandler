"""Handler-level chromatogram plotting.

Provides two figure builders:

- :func:`plot_traces`: one signal panel per group (overlay = single / sample / all).
- :func:`plot_window_grid`: rows x windows grid with the same overlay semantics.

Plus small internal helpers (:func:`_group_chromatograms`, :func:`_line_colors`)
that drive both builders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from chromhandler.annotations import PeakAnnotation
    from chromhandler.handler import Handler
    from chromhandler.model import Chromatogram

OverlayMode = Literal["all", "sample", "single"]


def _group_chromatograms(handler: Handler, overlay: OverlayMode) -> list[list[Chromatogram]]:
    """Flatten ``handler.samples`` → chromatograms into grouped rows.

    - ``"single"``: one group per chromatogram (flat).
    - ``"sample"``: one group per sample, in handler order.
    - ``"all"``: one group containing every chromatogram (flat).

    Raises:
        ValueError: If the handler has no chromatograms across any sample.
    """
    flat: list[Chromatogram] = []
    per_sample: list[list[Chromatogram]] = []
    for sample in handler.samples:
        if not sample.chromatograms:
            continue
        per_sample.append(list(sample.chromatograms))
        flat.extend(sample.chromatograms)
    if not flat:
        raise ValueError("Handler has no chromatograms across any sample.")
    if overlay == "single":
        return [[c] for c in flat]
    if overlay == "sample":
        return per_sample
    if overlay == "all":
        return [flat]
    raise ValueError(f"Unknown overlay mode: {overlay!r}")


def _line_colors(n: int) -> list[tuple[float, float, float, float]]:
    """Return ``n`` line colors per the project rule.

    1 line → ``tab:blue``; >= 2 lines → viridis evenly spaced over ``[0, 1]``.
    """
    if n <= 0:
        return []
    if n == 1:
        return [mcolors.to_rgba("tab:blue")]
    cmap = plt.get_cmap("viridis")
    return [cmap(i / (n - 1)) for i in range(n)]


def plot_traces(
    handler: Handler,
    *,
    overlay: OverlayMode = "single",
    ax_size: tuple[float, float] = (4.0, 3.0),
    share_y: bool = False,
    save: Path | str | None = None,
) -> tuple[Figure, NDArray[Any]]:
    """Plot raw chromatograms with the project overlay/color rules.

    Args:
        handler: Source of chromatograms.
        overlay: ``"single"`` = one ax per chromatogram (flat);
            ``"sample"`` = one ax per sample, chromatograms overlaid;
            ``"all"`` = one ax containing every chromatogram.
        ax_size: ``(width, height)`` in inches per axis. Total ``figsize`` is
            ``(width, n_rows * height)``.
        share_y: If ``True``, all axes share y-limits.
        save: If given, write the figure to this path before returning.

    Returns:
        ``(fig, axes)`` where ``axes`` is a 2-D ``ndarray`` of shape
        ``(n_groups, 1)``.
    """
    groups = _group_chromatograms(handler, overlay)
    n_rows = len(groups)
    width, height = ax_size
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(width, n_rows * height),
        squeeze=False,
        sharey=share_y,
    )
    for row, group in enumerate(groups):
        ax = axes[row, 0]
        colors = _line_colors(len(group))
        for chrom, color in zip(group, colors, strict=True):
            ax.plot(
                np.asarray(chrom.time),
                np.asarray(chrom.signal),
                color=color,
                lw=1.0,
                label=chrom.id,
            )
        ax.set_xlabel("retention time (min)")
        ax.set_ylabel("signal")
        if overlay == "sample":
            ax.set_title(group[0].sample_id)
        elif overlay == "single":
            ax.set_title(group[0].id)
    fig.tight_layout()
    if save is not None:
        fig.savefig(save)
    return fig, axes


def plot_window_grid(
    handler: Handler,
    annotations: list[PeakAnnotation],
    *,
    overlay: OverlayMode = "single",
    ax_size: tuple[float, float] = (4.0, 3.0),
    share_y: bool = False,
    save: Path | str | None = None,
) -> tuple[Figure, NDArray[Any]]:
    """Plot per-window panels in a ``(group, window)`` grid.

    Each row is a group (defined by ``overlay`` as in :func:`plot_traces`)
    and each column corresponds to one ``PeakAnnotation``. The panel's
    x-axis is clipped to ``[rt_min, rt_max]`` (no bounds are drawn -- the
    clip is implicit).

    Args:
        handler: Source of chromatograms.
        annotations: One :class:`PeakAnnotation` per column. Must be
            non-empty.
        overlay: Same semantics as :func:`plot_traces`.
        ax_size: ``(width, height)`` in inches per panel.
        share_y: If ``True``, all panels share y-limits.
        save: If given, write the figure to this path before returning.

    Returns:
        ``(fig, axes)`` with ``axes`` shape ``(n_groups, len(annotations))``.
    """
    if not annotations:
        raise ValueError("plot_window_grid: need at least one PeakAnnotation.")
    annotations = sorted(annotations, key=lambda a: a.rt_min)
    groups = _group_chromatograms(handler, overlay)
    n_rows = len(groups)
    n_cols = len(annotations)
    width, height = ax_size
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * width, n_rows * height),
        squeeze=False,
        sharey=share_y,
    )
    for row, group in enumerate(groups):
        colors = _line_colors(len(group))
        for col, ann in enumerate(annotations):
            ax = axes[row, col]
            for chrom, color in zip(group, colors, strict=True):
                t = np.asarray(chrom.time)
                s = np.asarray(chrom.signal)
                in_window = (t >= ann.rt_min) & (t <= ann.rt_max)
                ax.plot(t[in_window], s[in_window], color=color, lw=1.0, label=chrom.id)
            ax.set_xlim(ann.rt_min, ann.rt_max)
            if row == n_rows - 1:
                ax.set_xlabel("retention time (min)")
            if col == 0:
                if overlay == "sample":
                    ax.set_ylabel(group[0].sample_id)
                elif overlay == "single":
                    ax.set_ylabel(group[0].id)
                else:
                    ax.set_ylabel("signal")
            if row == 0:
                ax.set_title(ann.molecule_id)
    fig.tight_layout()
    if save is not None:
        fig.savefig(save)
    return fig, axes
