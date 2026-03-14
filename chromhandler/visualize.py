from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .handler import Handler


def _next_figure_path(stem: str, requested_path: str | None = None) -> Path:
    figs_dir = Path("figs")
    figs_dir.mkdir(parents=True, exist_ok=True)

    if requested_path:
        return figs_dir / Path(requested_path).name

    idx = 1
    while True:
        candidate = figs_dir / f"{stem}_{idx:03d}.png"
        if not candidate.exists():
            return candidate
        idx += 1


def visualize(
    handler: Handler,
    n_cols: int = 1,
    figsize: tuple[float, float] | None = None,
    width_per_ax: float = 10.0,
    height_per_ax: float = 3.0,
    show_peaks: bool = True,
    rt_min: float | None = None,
    rt_max: float | None = None,
    save_path: str | None = None,
    assigned_only: bool = False,
    overlay: bool = False,
    share_y: bool = False,
    show_peak_annotations: bool = True,
    peak_annotations: list | None = None,
    show_legend: bool = True,
) -> None:
    """Creates a matplotlib figure with subplots for each sample.

    Figure size is computed from per-axis dimensions when figsize is None,
    so each subplot stays readable regardless of grid size.

    Args:
        handler: The Handler instance containing the data.
        n_cols: Number of columns in the subplot grid.
        figsize: Figure size in inches (width, height). If None, computed from
            width_per_ax and height_per_ax.
        width_per_ax: Inches per subplot width when figsize is None.
        height_per_ax: Inches per subplot height when figsize is None.
        show_peaks: If True, shows detected peaks.
        rt_min: Minimum retention time to display. If None, shows all data.
        rt_max: Maximum retention time to display. If None, shows all data.
        save_path: Optional filename; figure is always saved inside figs/.
        assigned_only: If True, only shows peaks assigned to a molecule.
        overlay: If True, plots all chromatograms on a single axis.
        share_y: If True, subplots share the same y-axis scale (grid mode only).
        show_peak_annotations: If True, draws shaded regions for each
            :class:`~chromhandler.annotations.PeakAnnotation` supplied via
            *peak_annotations*.
        peak_annotations: Optional list of
            :class:`~chromhandler.annotations.PeakAnnotation` objects to overlay
            as shaded windows.  Obtained from a fitted
            :class:`~chromhandler.fitting.better_fitter.BetterFitter` via
            ``fitter.get_subset("__default__").peaks``.
        show_legend: If True, shows the legend with chromatogram reaction times.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    n_measurements = len(handler.samples)
    n_rows = int(np.ceil(n_measurements / n_cols))

    if figsize is None:
        if overlay:
            figsize = (width_per_ax * max(n_cols, 2), height_per_ax)
        else:
            figsize = (width_per_ax * n_cols, height_per_ax * n_rows)

    # First pass: collect all y-values to determine global y-range
    y_min = float("inf")
    y_max = float("-inf")
    has_signal_data = False

    for sample in handler.samples:
        for chrom in sample.chromatograms:
            if chrom.signal:
                y_min = min(y_min, min(chrom.signal))
                y_max = max(y_max, max(chrom.signal))
                has_signal_data = True

    # If no signal data is available, collect peak areas for y-range
    if not has_signal_data:
        for sample in handler.samples:
            for chrom in sample.chromatograms:
                if chrom.peaks:
                    for peak in chrom.peaks:
                        if peak.area is not None:
                            y_min = min(y_min, 0)  # Start from 0 for peak areas
                            y_max = max(y_max, peak.area)
                            has_signal_data = True

    # If still no data, set default range
    if not has_signal_data:
        y_min = 0
        y_max = 1

    # Add some padding to the y-range
    y_range = y_max - y_min
    if y_range > 0:
        y_min = y_min - 0.05 * y_range
        y_max = y_max + 0.05 * y_range
    else:
        y_min = 0
        y_max = 1

    # Collect all retention times for consistent coloring
    all_retention_times = []
    molecule_ids = set()
    for sample in handler.samples:
        for chrom in sample.chromatograms:
            if show_peaks and chrom.peaks:
                for peak in chrom.peaks:
                    if peak.retention_time is not None:
                        all_retention_times.append(peak.retention_time)
                        if peak.molecule_id:
                            molecule_ids.add(peak.molecule_id)

    if all_retention_times:
        # Create colormap for retention times
        retention_times = np.array(all_retention_times)
        norm = Normalize(vmin=min(retention_times), vmax=max(retention_times))
        cmap = plt.cm.get_cmap("viridis")
        sm = ScalarMappable(norm=norm, cmap=cmap)

        # Create a colormap for molecules (use a different colormap to distinguish from retention times)
        molecule_colors = {}
        if molecule_ids:
            molecule_list = list(molecule_ids)
            molecule_colors_list = plt.cm.get_cmap("tab10")(
                np.linspace(0, 1, len(molecule_list))
            )
            molecule_colors = {
                mol_id: color
                for mol_id, color in zip(molecule_list, molecule_colors_list)
            }

    def _draw_peak_annotations(
        ax_target: object,
        annotation_colors: dict[str, object],
    ) -> None:
        """Shade each PeakAnnotation window on *ax_target*.

        Colors cycle through the ``tab10`` colormap by molecule_id so that
        regions are distinguishable but consistent across subplots.
        """
        if not show_peak_annotations or not peak_annotations:
            return

        tab10 = plt.get_cmap("tab10")

        # Stable order of molecule ids as they appear in annotations
        mol_ids: list[str] = []
        for ann in peak_annotations:
            if ann.molecule_id not in mol_ids:
                mol_ids.append(ann.molecule_id)

        mol_to_color: dict[str, object] = {
            mol_id: tab10(i % 10) for i, mol_id in enumerate(mol_ids)
        }

        for ann in peak_annotations:
            color = mol_to_color.get(ann.molecule_id, "gray")
            ax_target.axvspan(  # type: ignore[attr-defined]
                ann.rt_min,
                ann.rt_max,
                alpha=0.2,
                color=color,
                label=None,
                zorder=0,
            )

    if overlay:
        # Create a single figure with one axis
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("none")
        ax.patch.set_facecolor("none")

        # Map chromatogram index within each sample to viridis (0→start, last→end)
        viridis_cmap = plt.get_cmap("viridis")

        for i, sample in enumerate(handler.samples):
            n = len(sample.chromatograms)
            for j, chrom in enumerate(sample.chromatograms):
                if chrom.time and chrom.signal:
                    t = j / (n - 1) if n > 1 else 0.5
                    label = (
                        f"{chrom.reaction_time:.1f}"
                        if chrom.reaction_time is not None
                        else (chrom.id or f"chrom {j}")
                    )
                    ax.plot(
                        chrom.time,
                        chrom.signal,
                        label=label,
                        color=viridis_cmap(t),
                        zorder=2,
                    )

            # Plot peaks if requested
            if show_peaks:
                for pci, chrom in enumerate(sample.chromatograms):
                    if chrom.peaks:
                        for peak in chrom.peaks:
                            # Skip unassigned peaks if assigned_only is True
                            if assigned_only and not peak.molecule_id:
                                continue

                            if peak.retention_time is not None:
                                # Determine color based on whether peak is assigned to a molecule
                                if (
                                    peak.molecule_id
                                    and peak.molecule_id in molecule_colors
                                ):
                                    # Use molecule-specific color for assigned peaks
                                    color = molecule_colors[peak.molecule_id]
                                else:
                                    # Use retention time color for unassigned peaks
                                    # Round to nearest 0.05 interval for discrete colors
                                    rt_discrete = (
                                        round(peak.retention_time / 0.05) * 0.05
                                    )
                                    color = sm.to_rgba(np.array([rt_discrete]))[0]

                                # Create label for legend
                                if peak.molecule_id:
                                    try:
                                        molecule = handler.get_molecule(
                                            peak.molecule_id
                                        )
                                        label = (
                                            f"{molecule.id} {peak.retention_time:.2f}"
                                        )
                                    except ValueError:
                                        label = f"Peak {peak.retention_time:.2f}"
                                else:
                                    label = f"Peak {peak.retention_time:.2f}"

                                # Plot vertical line with height based on peak area
                                peak_height = peak.area if peak.area is not None else 0

                                # Use a dashed line with increasing dash length based on chrom index
                                linestyle = (
                                    0,
                                    (1, pci + 1),
                                )  # (0, (1, 1)) for first chrom, (0, (1, 2)) for second, etc.

                                ax.plot(
                                    [peak.retention_time, peak.retention_time],
                                    [0, peak_height],
                                    color=color,
                                    linestyle=linestyle,
                                    alpha=0.7,
                                    linewidth=1.5,
                                    label=None,
                                    zorder=1,  # Put behind signal
                                )

        # Set plot properties
        ylabel = "Peak Area" if not has_signal_data else "Intensity"
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Retention time [min]")
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(True, alpha=0.3)

        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(
                by_label.values(),
                by_label.keys(),
                loc="upper right",
                fontsize=6,
                title="reaction time [min]",
                title_fontsize=7,
            )

        # Draw peak annotation shaded regions
        _draw_peak_annotations(ax, molecule_colors if all_retention_times else {})

        # Set y-axis limits
        ax.set_ylim(y_min, y_max)

        # Set x-axis limits if specified
        if rt_min is not None and rt_max is not None:
            ax.set_xlim(rt_min, rt_max)

    else:
        # Create figure with multiple subplots for each measurement
        n_rows = int(np.ceil(n_measurements / n_cols))

        # Create figure with independent y-axes per subplot (share_y=False by default)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize, sharey=share_y, sharex=True
        )
        fig.patch.set_facecolor("none")
        if n_measurements == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        for ax in axes:
            ax.patch.set_facecolor("none")

        # Hide unused subplots
        for i in range(n_measurements, len(axes)):
            axes[i].set_visible(False)

        # Map chromatogram index within each sample to viridis (0→start, last→end)
        viridis_cmap = plt.get_cmap("viridis")

        # Compute per-sample y-range for independent axes (no shared y)
        def _sample_y_range(sample: object) -> tuple[float, float]:
            smin, smax = float("inf"), float("-inf")
            has_data = False
            for chrom in sample.chromatograms:
                if chrom.signal:
                    smin = min(smin, min(chrom.signal))
                    smax = max(smax, max(chrom.signal))
                    has_data = True
            if not has_data:
                for chrom in sample.chromatograms:
                    if chrom.peaks:
                        for peak in chrom.peaks:
                            if peak.area is not None:
                                smin = min(smin, 0)
                                smax = max(smax, peak.area)
                                has_data = True
            if not has_data or smax <= smin:
                return 0.0, 1.0
            pad = 0.05 * (smax - smin)
            return smin - pad, smax + pad

        # Second pass: plot all data with per-sample y-range
        for idx, (sample, ax) in enumerate(zip(handler.samples, axes)):
            # Plot peaks first (behind the signal)
            if show_peaks:
                for chrom in sample.chromatograms:
                    if chrom.peaks:
                        for peak in chrom.peaks:
                            # Skip unassigned peaks if assigned_only is True
                            if assigned_only and not peak.molecule_id:
                                continue

                            if peak.retention_time is not None:
                                # Determine color based on whether peak is assigned to a molecule
                                if (
                                    peak.molecule_id
                                    and peak.molecule_id in molecule_colors
                                ):
                                    # Use molecule-specific color for assigned peaks
                                    color = molecule_colors[peak.molecule_id]
                                else:
                                    # Use retention time color for unassigned peaks
                                    # Round to nearest 0.05 interval for discrete colors
                                    rt_discrete = (
                                        round(peak.retention_time / 0.05) * 0.05
                                    )
                                    color = sm.to_rgba(np.array([rt_discrete]))[0]

                                # Create label for legend
                                if peak.molecule_id:
                                    try:
                                        molecule = handler.get_molecule(
                                            peak.molecule_id
                                        )
                                        label = (
                                            f"{molecule.id} {peak.retention_time:.2f}"
                                        )
                                    except ValueError:
                                        label = f"Peak {peak.retention_time:.2f}"
                                else:
                                    label = f"Peak {peak.retention_time:.2f}"

                                # Plot vertical line with height based on peak area
                                peak_height = peak.area if peak.area is not None else 0

                                ax.plot(
                                    [peak.retention_time, peak.retention_time],
                                    [0, peak_height],
                                    color=color,
                                    linestyle="-",
                                    alpha=0.7,
                                    linewidth=2,
                                    label=None,
                                    zorder=1,  # Put behind signal
                                )

            # Plot raw signal: all chromatograms of this sample overlaid
            n_chroms = len(sample.chromatograms)
            for j, chrom in enumerate(sample.chromatograms):
                if chrom.time and chrom.signal:
                    t = j / (n_chroms - 1) if n_chroms > 1 else 0.5
                    color = viridis_cmap(t)
                    rt = chrom.reaction_time
                    label = (
                        f"{rt:.1f}" if rt is not None else (chrom.id or f"chrom {j}")
                    )
                    ax.plot(
                        chrom.time,
                        chrom.signal,
                        label=label,
                        color=color,
                        zorder=2,
                    )

            # Remove title and add text annotation in top left corner
            ax.text(
                0.02,
                0.95,
                sample.id,
                transform=ax.transAxes,
                fontsize=10,
                va="top",
                ha="left",
            )

            ax.set_xlabel("Retention time [min]")
            ax.tick_params(axis="x", labelbottom=True)
            if idx % n_cols == 0:  # Only show y-label for leftmost plots
                ylabel = "Peak Area" if not has_signal_data else "Intensity"
                ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            _draw_peak_annotations(ax, molecule_colors if all_retention_times else {})
            if show_legend:
                ax.legend(
                    loc="upper right",
                    fontsize=6,
                    title="reaction time [min]",
                    title_fontsize=7,
                )
            if share_y:
                ax.set_ylim(y_min, y_max)
            else:
                ax.set_ylim(*_sample_y_range(sample))

            # Set x-axis limits if specified
            if rt_min is not None and rt_max is not None:
                ax.set_xlim(rt_min, rt_max)

    plt.tight_layout()
    out_path = _next_figure_path("visualize", save_path)
    plt.savefig(out_path, bbox_inches="tight", transparent=True, dpi=300)
    plt.close(fig)
