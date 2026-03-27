from __future__ import annotations

import re
from pathlib import Path  # noqa: TC003 - used at runtime for path.read_text()

from chromhandler.model import Chromatogram, Estimate, Peak

_SECTION_RE = re.compile(r"^\[(.+)\]$", re.MULTILINE)


class ShimadzuReader:
    """Reader for Shimadzu LabSolutions ASCII export files.

    Implements the :class:`AbstractReader` protocol: parses a single
    Shimadzu ``.txt`` export and returns a
    :class:`~chromhandler.model.Chromatogram`.

    File format:
    - Sections delimited by ``[Section Name]`` lines
    - ``[Peak Table(...)]`` section: tab-separated peaks with comma decimals
    - ``[... Chromatogram(...)]`` section: metadata lines, then
      ``R.Time (min)\\tIntensity`` header, then tab-separated data rows

    Example::

        reader = ShimadzuReader()
        chrom = reader.read_file(
            Path("P0-0.0_min.txt"),
            chromatogram_id="P0-0.0_min",
            sample_id="experiment_1",
            reaction_time=0.0,
        )
    """

    def read_file(
        self,
        path: Path,
        *,
        chromatogram_id: str,
        sample_id: str,
        reaction_time: float | None = None,
    ) -> Chromatogram:
        """Parse a single Shimadzu LabSolutions TXT export file.

        Args:
            path: Path to the ``.txt`` file.
            chromatogram_id: Identifier for this chromatogram.
            sample_id: Identifier of the parent sample.
            reaction_time: Time since reaction start in minutes, or ``None``.

        Returns:
            A :class:`~chromhandler.model.Chromatogram` with signal, time,
            wavelength, and any peaks found in the peak table.

        Raises:
            ValueError: If required sections or headers are not found.
        """
        content = path.read_text(encoding="ISO-8859-1")
        sections = _parse_sections(content, path)

        chrom_body = _find_chromatogram_section(sections, path)
        time, signal, wavelength = _parse_chromatogram_body(chrom_body, path)

        peak_body = _find_peak_table_section(sections)
        peaks = _parse_peak_table(peak_body, chromatogram_id) if peak_body else []

        return Chromatogram(
            id=chromatogram_id,
            sample_id=sample_id,
            signal=signal,
            time=time,
            reaction_time=reaction_time,
            peaks=peaks,
            wavelength=wavelength,
        )


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------


def _parse_sections(content: str, path: Path) -> dict[str, str]:
    """Split file content into a ``{section_name: body}`` dict.

    Raises:
        ValueError: If the file does not start with a section header.
    """
    parts = _SECTION_RE.split(content)
    if parts[0].strip():
        raise ValueError(f"'{path}' does not start with a section header.")
    names = parts[1::2]
    bodies = parts[2::2]
    return dict(zip(names, bodies, strict=False))


def _find_chromatogram_section(sections: dict[str, str], path: Path) -> str:
    """Return the body of the first chromatogram data section.

    Matches section names containing ``"Chromatogram"`` but not
    ``"LC Status"`` (which are instrument status traces, not signal data).

    Raises:
        ValueError: If no matching section is found.
    """
    for name, body in sections.items():
        if "Chromatogram" in name and "LC Status" not in name:
            return body
    raise ValueError(f"No chromatogram section found in '{path}'.")


def _find_peak_table_section(sections: dict[str, str]) -> str | None:
    """Return the body of the peak table section, or ``None`` if absent."""
    for name, body in sections.items():
        if "Peak Table" in name:
            return body
    return None


# ---------------------------------------------------------------------------
# Chromatogram data parsing
# ---------------------------------------------------------------------------


def _parse_float(s: str) -> float:
    """Parse a float that may use a comma as decimal separator."""
    return float(s.strip().replace(",", "."))


def _parse_chromatogram_body(
    body: str, path: Path
) -> tuple[list[float], list[float], float | None]:
    """Parse a chromatogram section body.

    Scans metadata lines for ``Wavelength(nm)``, then reads tab-separated
    ``R.Time (min)\\tIntensity`` rows after the data header.

    Returns:
        ``(time, signal, wavelength)`` — wavelength is ``None`` if not found.

    Raises:
        ValueError: If the ``R.Time (min)`` data header is not found.
    """
    lines = body.strip().splitlines()
    wavelength: float | None = None
    data_start: int | None = None

    for i, line in enumerate(lines):
        if line.startswith("Wavelength(nm)"):
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    wavelength = float(parts[1].strip())
                except ValueError:
                    pass
        if line.startswith("R.Time (min)"):
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(
            f"Chromatogram data header 'R.Time (min)' not found in '{path}'."
        )

    time: list[float] = []
    signal: list[float] = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            time.append(_parse_float(parts[0]))
            signal.append(_parse_float(parts[1]))

    return time, signal, wavelength


# ---------------------------------------------------------------------------
# Peak table parsing
# ---------------------------------------------------------------------------


def _parse_peak_table(body: str, chromatogram_id: str) -> list[Peak]:
    """Parse a peak table section body into a list of :class:`Peak` objects.

    Finds the ``Peak#`` header row, then maps each data row to a ``Peak``.
    Rows that cannot be parsed are silently skipped.
    """
    lines = body.strip().splitlines()

    header_idx: int | None = None
    for i, line in enumerate(lines):
        if line.startswith("Peak#"):
            header_idx = i
            break

    if header_idx is None:
        return []

    headers = lines[header_idx].split("\t")
    peaks: list[Peak] = []

    for line in lines[header_idx + 1 :]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        row = dict(zip(headers, parts, strict=False))
        try:
            peak = Peak(
                chromatogram_id=chromatogram_id,
                location=Estimate(mean=_parse_float(row["R.Time"])),
                area=Estimate(mean=_parse_float(row["Area"])),
                peak_start=_parse_float(row["I.Time"]),
                peak_end=_parse_float(row["F.Time"]),
                amplitude=_parse_float(row["Height"]),
                tailing_factor=_parse_float(row.get("Tailing", "0")) or None,
                separation_factor=_parse_float(row.get("Sep.Factor", "0")) or None,
            )
            peaks.append(peak)
        except (KeyError, ValueError):
            continue

    return peaks
