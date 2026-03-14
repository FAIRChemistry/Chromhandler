from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from .handler import Handler
    from .molecule import Molecule


def _safe_emoji(emoji: str, fallback: str) -> str:
    """Return emoji if encoding supports it, otherwise return ASCII fallback."""
    try:
        # Test if we can encode the emoji in the current output encoding
        encoding = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
        emoji.encode(encoding)
        return emoji
    except (UnicodeEncodeError, LookupError):
        return fallback


def create_overview_panel(handler: Handler) -> Panel:
    """Create the Handler overview panel with basic information."""
    overview_content = []
    overview_content.append(f"[bold]ID:[/bold] {handler.id}")
    overview_content.append(f"[bold]Name:[/bold] {handler.name}")

    # Add sample summary
    total_peaks = sum(
        len(chrom.peaks)
        for sample in handler.samples
        for chrom in sample.chromatograms
    )
    assigned_peaks = sum(
        1
        for sample in handler.samples
        for chrom in sample.chromatograms
        for peak in chrom.peaks
        if peak.molecule_id
    )

    overview_content.append(f"[bold]Samples:[/bold] {len(handler.samples)}")
    if total_peaks > 0:
        assignment_rate = (assigned_peaks / total_peaks) * 100
        overview_content.append(
            f"[bold]Peak Assignment:[/bold] {assigned_peaks}/{total_peaks} ({assignment_rate:.1f}%)"
        )

    return Panel(
        "\n".join(overview_content),
        title="📋 Handler Overview",
        title_align="left",
        border_style="blue",
    )


def create_statistics_table(handler: Handler) -> Table:
    """Create a table summarizing component statistics."""
    stats_table = Table(
        title="📊 Component Statistics",
        show_header=True,
        header_style="bold magenta",
    )
    stats_table.add_column("Component", no_wrap=True)
    stats_table.add_column("Count", justify="right")

    # Calculate statistics
    total_chromatograms = sum(len(sample.chromatograms) for sample in handler.samples)
    total_peaks = sum(
        len(chrom.peaks)
        for sample in handler.samples
        for chrom in sample.chromatograms
    )
    assigned_peaks = sum(
        1
        for sample in handler.samples
        for chrom in sample.chromatograms
        for peak in chrom.peaks
        if peak.molecule_id
    )

    # Add component counts
    component_counts = [
        ("Molecules", len(handler.molecules)),
        ("Proteins", len(handler.proteins)),
        ("Samples", len(handler.samples)),
        ("Chromatograms", total_chromatograms),
        ("Total Peaks", total_peaks),
        ("Assigned Peaks", assigned_peaks),
    ]

    for component_name, count in component_counts:
        stats_table.add_row(component_name, str(count))

    return stats_table


def create_species_table(handler: Handler) -> Table:
    """Create a table summarizing molecules and proteins."""
    species_emoji = _safe_emoji("🧬", "SPECIES")
    species_table = Table(
        title=f"{species_emoji} Species Details",
        show_header=True,
        header_style="bold green",
    )
    species_table.add_column("Type", min_width=12)
    species_table.add_column("ID", style="magenta", min_width=15)
    species_table.add_column("Name", min_width=20)
    species_table.add_column("Details")

    # Add molecules
    for molecule in handler.molecules:  # Show all molecules
        details = []
        if hasattr(molecule, "formula") and molecule.formula:
            details.append(f"Formula: {molecule.formula}")
        if hasattr(molecule, "molecular_weight") and molecule.molecular_weight:
            details.append(f"MW: {molecule.molecular_weight}")
        if hasattr(molecule, "standard") and molecule.standard:
            details.append("Has calibration")

        species_table.add_row(
            "Molecule",
            f"[magenta]{molecule.id}[/magenta]",
            molecule.name,
            " | ".join(details) if details else "—",
        )

    # Add proteins
    for protein in handler.proteins:  # Show all proteins
        details = []
        if hasattr(protein, "molecular_weight") and protein.molecular_weight:
            details.append(f"MW: {protein.molecular_weight}")
        if hasattr(protein, "organism") and getattr(protein, "organism", None):
            details.append(f"Organism: {protein.organism}")

        species_table.add_row(
            "Protein",
            f"[magenta]{protein.id}[/magenta]",
            protein.name,
            " | ".join(details) if details else "—",
        )

    return species_table


def create_stats_and_species_content(handler: Handler) -> Table | Columns:
    """Create the statistics and species content."""
    stats_table = create_statistics_table(handler)

    if handler.molecules or handler.proteins:
        species_table = create_species_table(handler)
        return Columns([stats_table, species_table], equal=True)
    return stats_table


def create_measurements_content(handler: Handler) -> Table:
    """Create samples/chromatograms content."""
    measurements_emoji = _safe_emoji("📈", "DATA")
    measurements_table = Table(
        title=f"{measurements_emoji} Samples",
        show_header=True,
        header_style="bold cyan",
    )
    measurements_table.add_column("Sample ID", style="yellow", min_width=12)
    measurements_table.add_column("Chromatograms", justify="center")
    measurements_table.add_column("Peaks", justify="center")
    measurements_table.add_column("Assigned", justify="center")
    measurements_table.add_column("Reaction Times (min)", justify="right")

    for sample in handler.samples:
        sample_peaks = sum(len(chrom.peaks) for chrom in sample.chromatograms)
        sample_assigned = sum(
            1
            for chrom in sample.chromatograms
            for peak in chrom.peaks
            if peak.molecule_id
        )

        reaction_times = sorted(
            {chrom.reaction_time for chrom in sample.chromatograms if chrom.reaction_time is not None}
        )
        rt_str = ", ".join(f"{rt:.1f}" for rt in reaction_times) if reaction_times else "—"

        measurements_table.add_row(
            sample.id,
            str(len(sample.chromatograms)),
            str(sample_peaks),
            f"[green]{sample_assigned}[/green]" if sample_assigned > 0 else "0",
            rt_str,
        )

    return measurements_table


def create_peak_assignment_summary_table(
    handler: Handler, assignment_results: list[dict[str, Any]]
) -> Table:
    """Create the main peak assignment summary table."""
    target_emoji = _safe_emoji("🎯", ">>")
    summary_table = Table(
        title=f"{target_emoji} Peak Assignment Summary of {handler.id}",
        show_header=True,
        header_style="bold cyan",
        border_style="blue",
    )
    summary_table.add_column("Molecule", style="bold green", min_width=20)
    summary_table.add_column("Status", justify="center")
    summary_table.add_column("Details", style="yellow")

    # Populate summary table
    for result in assignment_results:
        molecule = result["molecule"]
        assigned_count = result["assigned_peak_count"]
        multiple_peaks = result["measurements_with_multiple_peaks"]
        no_peaks = result["measurements_with_no_peaks"]

        # Calculate total samples for this molecule
        total_measurements = len(handler.samples)

        # Safe emojis for status
        success_emoji = _safe_emoji("✅", "[OK]")
        partial_emoji = _safe_emoji("🟡", "[PARTIAL]")
        failed_emoji = _safe_emoji("❌", "[FAILED]")
        overlap_emoji = _safe_emoji("⚠️", "[OVERLAP]")

        # Determine status and details (consistent format)
        if assigned_count > 0 and not multiple_peaks and not no_peaks:
            status = f"{success_emoji} Success"
            details = f"({assigned_count}/{total_measurements}) peaks assigned"
        else:
            if assigned_count > 0:
                status = f"{partial_emoji} Partial"
            else:
                status = f"{failed_emoji} Failed"

            if multiple_peaks:
                status = f"{overlap_emoji} Overlaps"
                details = f"({assigned_count}/{total_measurements}) peaks assigned"
            elif no_peaks:
                # Include retention time details for failed molecules
                ret_time = molecule.retention_time
                tolerance = result["retention_tolerance"]
                if assigned_count == 0:
                    details = f"({assigned_count}/{total_measurements}) peaks at {ret_time:.1f} ± {tolerance:.1f} min"
                else:
                    details = f"({assigned_count}/{total_measurements}) peaks assigned"
            else:
                details = f"({assigned_count}/{total_measurements}) peaks assigned"

        summary_table.add_row(f"{molecule.id} ({molecule.name})", status, details)

    return summary_table


def print_peak_assignment_summary(
    handler: Handler,
    molecule: Molecule,
    assigned_peak_count: int,
    measurements_with_multiple_peaks: list[dict[str, Any]],
    measurements_with_no_peaks: list[str],
    ret_tolerance: float,
) -> None:
    """Print a formatted summary of peak assignment results (backward compatibility)."""
    # Use force_terminal=False to avoid encoding issues on Windows CI
    console = Console(force_terminal=False)

    # Only show success message if peaks were assigned, or if there were warnings
    has_warnings = measurements_with_multiple_peaks or measurements_with_no_peaks

    # Use safe emoji that falls back to ASCII on encoding issues
    target_emoji = _safe_emoji("🎯", ">>")

    if assigned_peak_count > 0:
        console.print(
            f"{target_emoji} Assigned [bold green]{molecule.name}[/bold green] to [bold]{assigned_peak_count}[/bold] peaks"
        )
    elif not has_warnings:
        # Only show 0 peaks message if there are no warnings (warnings will cover this)
        console.print(
            f"{target_emoji} Assigned [bold green]{molecule.name}[/bold green] to [bold]0[/bold] peaks"
        )

    # Warning for multiple peaks in tolerance
    warning_emoji = _safe_emoji("⚠️", "!!")
    if measurements_with_multiple_peaks:
        console.print(
            f"{warning_emoji}  [bold yellow]Warning:[/bold yellow] Multiple peaks found within tolerance for [bold]{molecule.name}[/bold]"
        )

        # Create table for multiple peaks details
        table = Table(
            show_header=True, header_style="bold yellow", box=None, padding=(0, 1)
        )
        table.add_column("Measurement", style="magenta")
        table.add_column("Peaks Found", justify="center")
        table.add_column("Retention Times", style="cyan")
        table.add_column("Assigned RT", style="green")

        for warning in measurements_with_multiple_peaks:
            rts_str = (
                ", ".join(f"{rt:.3f}" for rt in warning["all_rts"])
                if isinstance(warning["all_rts"], list)
                else ""
            )
            table.add_row(
                warning["sample_id"],
                str(warning["num_peaks"]),
                f"[{rts_str}]",
                f"{warning['assigned_rt']:.3f}",
            )

        console.print(table)
        tip_emoji = _safe_emoji("💡", "TIP:")
        console.print(
            f"   {tip_emoji} [dim]Tip: Consider setting a higher min_signal value for {molecule.name} to filter out smaller peaks[/dim]"
        )
        console.print(
            f"   {tip_emoji} [dim]Current min_signal: {molecule.min_signal}[/dim]"
        )

    # Warning for no peaks found
    if measurements_with_no_peaks:
        console.print(
            f"{warning_emoji}  [bold red]Warning:[/bold red] No peaks found for [bold]{molecule.name}[/bold] in [bold]{len(measurements_with_no_peaks)}[/bold] measurement(s)"
        )

        # Create table for no peaks details
        table = Table(
            show_header=True, header_style="bold red", box=None, padding=(0, 1)
        )
        table.add_column("Measurements with no peaks", style="red")
        table.add_column("Expected RT", justify="center", style="cyan")
        table.add_column("Tolerance", justify="center", style="yellow")
        table.add_column("Min Signal", justify="center", style="green")

        table.add_row(
            ", ".join(measurements_with_no_peaks[:5])
            + (
                f" ... (+{len(measurements_with_no_peaks) - 5} more)"
                if len(measurements_with_no_peaks) > 5
                else ""
            ),
            f"{molecule.retention_time:.3f} min",
            f"±{ret_tolerance:.3f} min",
            str(molecule.min_signal),
        )

        console.print(table)


def display_rich_handler(
    handler: Handler, console: Console | None = None, debug: bool = False
) -> None:
    """
    Display a comprehensive rich text visualization of the Handler instance.

    This function provides a beautiful, structured overview of the Handler including:
    - Basic information (ID, name, mode)
    - Molecules and their properties
    - Proteins and their properties
    - Measurements summary with peak statistics
    - Chromatogram details

    Args:
        handler: The Handler instance to display
        console (Console | None, optional): Rich console instance. If None, creates a new one. Defaults to None.
        debug (bool, optional): If True, shows debug information about what sections are being displayed. Defaults to False.
    """
    if console is None:
        # Use force_terminal=False to avoid encoding issues on Windows CI
        console = Console(force_terminal=False)

    # Debug information
    if debug:
        console.print(
            f"[dim]Debug: Molecules: {len(handler.molecules)}, Proteins: {len(handler.proteins)}, Samples: {len(handler.samples)}[/dim]"
        )

    # Collect all content panels
    content_panels: list[Panel | Table | Columns] = []

    # Overview panel is always shown
    content_panels.append(create_overview_panel(handler))
    if debug:
        console.print("[dim]Debug: Added overview panel[/dim]")

    # Statistics and species panels
    stats_species = create_stats_and_species_content(handler)
    if stats_species:
        content_panels.append(stats_species)
        if debug:
            console.print("[dim]Debug: Added stats and species content[/dim]")
    elif debug:
        console.print("[dim]Debug: No stats/species content to add[/dim]")

    # Only add sections that have content
    if handler.samples:
        content_panels.append(create_measurements_content(handler))
        if debug:
            console.print(
                f"[dim]Debug: Added samples content ({len(handler.samples)} samples)[/dim]"
            )
    elif debug:
        console.print("[dim]Debug: No measurements to add[/dim]")

    if debug:
        console.print(f"[dim]Debug: Total content panels: {len(content_panels)}[/dim]")

    # Create a group of all content
    spaced_content: list[Panel | Table | Columns | str] = []
    for i, content in enumerate(content_panels):
        spaced_content.append(content)
        if i < len(content_panels) - 1:
            spaced_content.append("")

    # Print title and content
    console.print("🧪 [bold cyan]Handler Summary[/bold cyan]")
    content_group = Group(*spaced_content)
    console.print(content_group)


def create_rich_handler_group(handler: Handler) -> Group:
    """
    Create a rich group representation for automatic display in rich-aware contexts.

    This function is used for automatic display when you:
    - print(handler) in a rich-enabled terminal
    - Display handler in Jupyter notebooks
    - Use handler in any rich-aware context

    Args:
        handler: The Handler instance

    Returns:
        Group: A rich group with the full Handler visualization.
    """
    # Collect all content panels
    content_panels: list[Panel | Table | Columns] = []

    # Overview panel is always shown
    content_panels.append(create_overview_panel(handler))

    # Statistics and species panels
    stats_species = create_stats_and_species_content(handler)
    if stats_species:
        content_panels.append(stats_species)

    # Only add sections that have content
    if handler.samples:
        content_panels.append(create_measurements_content(handler))

    # Create a group of all content with spacing
    spaced_content: list[Panel | Table | Columns | str] = []
    for i, content in enumerate(content_panels):
        spaced_content.append(content)
        if i < len(content_panels) - 1:
            spaced_content.append("")

    return Group(*spaced_content)


def display_consolidated_assignment_report(
    handler: Handler, assignment_results: list[dict[str, Any]]
) -> None:
    """Display a consolidated peak assignment report for all molecules."""
    # Use force_terminal=False to avoid encoding issues on Windows CI
    console = Console(force_terminal=False)

    # Create and display main assignment summary table
    summary_table = create_peak_assignment_summary_table(handler, assignment_results)
    console.print(summary_table)
