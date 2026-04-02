from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from .calibration import LinearCalibration
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


def _format_molecule_label(molecule: Molecule) -> str:
    """Return a short display label: ``"<id> (<name>)"``."""
    return f"{molecule.id} ({molecule.name})"


def create_overview_panel(handler: Handler) -> Panel:
    """Create the Handler overview panel with basic information."""
    overview_content: list[str] = []

    # Add sample summary
    total_peaks = sum(len(chrom.peaks) for sample in handler.samples for chrom in sample.chromatograms)
    assigned_peaks = sum(
        1
        for sample in handler.samples
        for chrom in sample.chromatograms
        for peak in chrom.peaks
        if peak.molecule_id
    )

    total_chromatograms = sum(len(sample.chromatograms) for sample in handler.samples)
    overview_content.append(f"[bold]Samples:[/bold] {len(handler.samples)}")
    overview_content.append(f"[bold]Chromatograms:[/bold] {total_chromatograms}")
    overview_content.append(f"[bold]Peak Windows:[/bold] {len(handler.peak_windows)}")
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
    total_peaks = sum(len(chrom.peaks) for sample in handler.samples for chrom in sample.chromatograms)
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
    for molecule in handler.molecules.values():
        mol_details: list[str] = []
        if molecule.standard:
            mol_details.append("Has calibration")

        species_table.add_row(
            "Molecule",
            f"[magenta]{molecule.id}[/magenta]",
            molecule.name,
            " | ".join(mol_details) if mol_details else "—",
        )

    # Add proteins
    for protein in handler.proteins.values():
        prot_details: list[str] = []
        molecular_weight = getattr(protein, "molecular_weight", None)
        if molecular_weight:
            prot_details.append(f"MW: {molecular_weight}")
        if protein.organism:
            prot_details.append(f"Organism: {protein.organism}")

        species_table.add_row(
            "Protein",
            f"[magenta]{protein.id}[/magenta]",
            protein.name,
            " | ".join(prot_details) if prot_details else "—",
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
        sample_assigned = sum(1 for chrom in sample.chromatograms for peak in chrom.peaks if peak.molecule_id)

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


def create_peak_assignment_summary_table(handler: Handler, assignment_results: list[dict[str, Any]]) -> Table:
    """Create the main molecule-assignment summary table."""
    target_emoji = _safe_emoji("🎯", ">>")
    summary_table = Table(
        title=f"{target_emoji} Molecule Assignment Summary",
        show_header=True,
        header_style="bold cyan",
        border_style="blue",
    )
    summary_table.add_column("Molecule", style="bold green", min_width=20)
    summary_table.add_column("Window", style="cyan")
    summary_table.add_column("Assigned", justify="right")
    summary_table.add_column("Missing", justify="right")
    summary_table.add_column("Ambiguous", justify="right")
    summary_table.add_column("Details", style="yellow", min_width=20)

    for result in assignment_results:
        molecule = result["molecule"]
        window = result["window"]
        assigned_count = result["assigned_peak_count"]
        missing = result["chromatograms_with_no_peaks"]
        ambiguous = result["chromatograms_with_multiple_peaks"]
        details: list[str] = []
        if window.wavelength is not None:
            details.append(f"{window.wavelength:g} nm")
        if result["min_amplitude"] is not None:
            details.append(f"min amp {result['min_amplitude']:g}")
        if result.get("on_multiple") == "skip":
            details.append("on multiple: skip")

        summary_table.add_row(
            _format_molecule_label(molecule),
            f"[{window.rt_min:.3f}, {window.rt_max:.3f}]",
            str(assigned_count),
            str(len(missing)),
            str(len(ambiguous)),
            " | ".join(details) if details else "—",
        )

    return summary_table


def _truncated_chromatogram_list(chromatograms: list[str]) -> str:
    return ", ".join(chromatograms[:5]) + (
        f" ... (+{len(chromatograms) - 5} more)" if len(chromatograms) > 5 else ""
    )


def display_rich_handler(handler: Handler, console: Console | None = None, debug: bool = False) -> None:
    """
    Display a comprehensive rich text visualization of the Handler instance.

    This function provides a beautiful, structured overview of the Handler including:
    - Basic information and assignment status
    - Molecules and their properties
    - Proteins and their properties
    - Sample summary with peak statistics

    Args:
        handler: The Handler instance to display
        console (Console | None, optional): Rich console instance. If None, creates a new one.
        debug (bool, optional): If True, shows debug information about what sections are being displayed.
    """
    if console is None:
        # Use force_terminal=False to avoid encoding issues on Windows CI
        console = Console(force_terminal=False)

    # Debug information
    if debug:
        console.print(
            f"[dim]Debug: Molecules: {len(handler.molecules)}, Proteins: {len(handler.proteins)}, "
            f"Samples: {len(handler.samples)}[/dim]"
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
            console.print(f"[dim]Debug: Added samples content ({len(handler.samples)} samples)[/dim]")
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

    missing_entries = [
        (result["molecule"], result["chromatograms_with_no_peaks"])
        for result in assignment_results
        if result["chromatograms_with_no_peaks"]
    ]
    ambiguous_entries = [
        (result["molecule"], result["chromatograms_with_multiple_peaks"])
        for result in assignment_results
        if result["chromatograms_with_multiple_peaks"]
    ]

    if missing_entries:
        missing_table = Table(
            title="Missing Assignments",
            show_header=True,
            header_style="bold red",
            border_style="red",
        )
        missing_table.add_column("Molecule", style="bold green")
        missing_table.add_column("Chromatograms", style="red")
        for molecule, chromatograms in missing_entries:
            missing_table.add_row(
                _format_molecule_label(molecule),
                _truncated_chromatogram_list(chromatograms),
            )
        console.print(missing_table)

    if ambiguous_entries:
        ambiguous_table = Table(
            title="Ambiguous Assignments",
            show_header=True,
            header_style="bold yellow",
            border_style="yellow",
        )
        ambiguous_table.add_column("Molecule", style="bold green")
        ambiguous_table.add_column("Chromatograms", style="yellow")
        for molecule, chromatograms in ambiguous_entries:
            ambiguous_table.add_row(
                _format_molecule_label(molecule),
                _truncated_chromatogram_list(chromatograms),
            )
        console.print(ambiguous_table)


def display_molecule_assignment_report(handler: Handler, assignment_results: list[dict[str, Any]]) -> None:
    """Display a consolidated molecule-assignment report."""
    display_consolidated_assignment_report(handler, assignment_results)


def _fmt(value: float) -> str:
    """Format a numeric value: no decimals if |value| >= 1, else 4 decimal places."""
    return f"{value:.0f}" if abs(value) >= 1 else f"{value:.4f}"


def _r2_colored(r2: float) -> str:
    """Return an R² value formatted with a colour hint for the rich table."""
    formatted = f"{r2:.4f}"
    if r2 >= 0.999:
        return f"[bold green]{formatted}[/bold green]"
    if r2 >= 0.99:
        return f"[green]{formatted}[/green]"
    if r2 >= 0.95:
        return f"[yellow]{formatted}[/yellow]"
    return f"[bold red]{formatted}[/bold red]"


def display_calibration_summary(
    results: list[tuple[Molecule, LinearCalibration | None]],
) -> None:
    """Print a transposed rich table summarising the output of ``calibrate_molecules()``.

    Rows are metrics; each molecule gets its own data column.

    Args:
        results: Sequence of ``(molecule, calibration)`` pairs where
            *calibration* is ``None`` for molecules that were skipped due to
            insufficient calibration data.
    """
    console = Console(force_terminal=False)

    cal_emoji = _safe_emoji("🔬", "CAL")
    table = Table(
        title=f"{cal_emoji} Calibration Summary",
        show_header=True,
        header_style="bold cyan",
        border_style="blue",
    )

    # Index column (metric names) — dim so it reads as a label, not data
    table.add_column("", style="bold dim", min_width=12, no_wrap=True)

    # One data column per molecule
    for molecule, _ in results:
        table.add_column(
            _format_molecule_label(molecule),
            justify="right",
            min_width=16,
        )

    # Helper: produce one cell value per calibration
    _D = "—"

    def _standards(c: LinearCalibration | None) -> str:
        return str(c.n_standards) if c else _D

    def _r2(c: LinearCalibration | None) -> str:
        return _r2_colored(c.r_squared) if c else _D

    def _slope(c: LinearCalibration | None) -> str:
        return _fmt(c.slope) if c else _D

    def _intercept(c: LinearCalibration | None) -> str:
        return _fmt(c.intercept) if c else _D

    def _range(c: LinearCalibration | None) -> str:
        if c is None:
            return _D
        unit = str(c.conc_unit) if c.conc_unit else "AU"
        return f"{_fmt(c.min_conc)} - {_fmt(c.max_conc)} {unit}"

    def _status(c: LinearCalibration | None) -> str:
        return "[bold green]✓ fitted[/bold green]" if c else "[bold red]⚠ no data[/bold red]"

    cals = [cal for _, cal in results]
    table.add_row("Standards", *[_standards(c) for c in cals])
    table.add_row("R²", *[_r2(c) for c in cals])
    table.add_row("Slope", *[_slope(c) for c in cals])
    table.add_row("Intercept", *[_intercept(c) for c in cals])
    table.add_row("Range", *[_range(c) for c in cals])
    table.add_row("Status", *[_status(c) for c in cals])

    console.print(table)
