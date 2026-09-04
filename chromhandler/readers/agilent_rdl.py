import re
from itertools import zip_longest

from chromhandler.model import Chromatogram, Data, Measurement, Peak
from chromhandler.readers.abstractreader import AbstractReader

# Normalised report column name -> Peak field. "RT [min]" normalises to "rt",
# "Width [min]" to "width", so a template that renames a unit suffix still maps.
_PEAK_FIELDS = {
    "rt": "retention_time",
    "type": "type",
    "width": "width",
    "area": "area",
    "height": "amplitude",
    "area%": "percent_area",
}


def _cells(line: str) -> list[str] | None:
    """Interior cells of one box-drawn table line, stripped.

    Returns ``None`` for anything that is not a data line: the ``┌ ├ └`` rules,
    the trailing ``═`` banner, and blank lines. Leading indentation is discarded
    first, which is why two report templates differing only in margin width parse
    identically.
    """
    stripped = line.strip()
    if not stripped.startswith("│"):
        return None
    return [cell.strip() for cell in stripped.strip("│").split("│")]


def _records(lines: list[str]) -> list[list[str]]:
    """Group consecutive data lines into records, joining cells column-wise.

    The box's own rules delimit records, so a value the report wrapped across two
    physical lines (``"13."`` then ``"198"``) is rejoined into ``"13.198"``. Ragged
    groups are padded rather than truncated.
    """
    records: list[list[str]] = []
    group: list[list[str]] = []

    def flush() -> None:
        if group:
            records.append(
                ["".join(parts).strip() for parts in zip_longest(*group, fillvalue="")]
            )
            group.clear()

    for line in lines:
        cells = _cells(line)
        if cells is None:
            flush()
        else:
            group.append(cells)
    flush()
    return records


def _header_columns(record: list[str]) -> dict[str, int] | None:
    """Map normalised column name -> position, or ``None`` if this is not the header."""
    columns = {cell.split("[")[0].strip().lower(): i for i, cell in enumerate(record)}
    return columns if "rt" in columns and "area" in columns else None


def _maybe_float(value: str) -> float | None:
    """Parse a report cell as a float, or ``None`` if it is blank or non-numeric."""
    try:
        return float(value)
    except ValueError:
        return None


class AgilentRDLReader(AbstractReader):
    """Reader for Agilent OpenLab box-drawn summary reports.

    Report templates differ in left margin, column width and title wording
    ("Sequence Summary Report" vs "Cross Sequence Summary Report"), and wrap a
    cell onto a second physical line when its value does not fit. The parser
    therefore reads the box grammar — ``├─┼─┤`` rules delimit records, ``│``
    delimits cells, wrapped lines are rejoined column-wise, and columns are found
    by header name — rather than matching fixed offsets.
    """

    def read(self) -> list[Measurement]:
        measurements = []
        for path_id, path in enumerate(self.file_paths):
            lines = self.read_file(path)

            chromatogram = Chromatogram(
                peaks=self.parse_peaks(lines),
                wavelength=self.extract_wavelength(lines),
            )

            data = Data(
                value=self.values[path_id],
                unit=self.unit.name,
                data_type=self.mode,
            )

            measurements.append(
                Measurement(
                    id=f"m{path_id}",
                    chromatograms=[chromatogram],
                    temperature=self.temperature,
                    temperature_unit=self.temperature_unit.name,
                    ph=self.ph,
                    data=data,
                )
            )

        return measurements

    @staticmethod
    def read_file(file_path: str) -> list[str]:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        return lines

    @staticmethod
    def parse_peaks(lines: list[str]) -> list[Peak]:
        """Every peak in the report's peak table.

        A record is a peak iff its ``RT`` and ``Area`` cells both parse as floats,
        which is also what discards the table's trailing ``Sum`` row and the report
        footer — neither carries a retention time.
        """
        records = _records(lines)
        for i, record in enumerate(records):
            columns = _header_columns(record)
            if columns is None:
                continue

            peaks = []
            for row in records[i + 1 :]:
                fields = {
                    field: _maybe_float(row[position])
                    if field != "type"
                    else row[position]
                    for name, field in _PEAK_FIELDS.items()
                    if (position := columns.get(name)) is not None
                    and position < len(row)
                }
                if fields.get("retention_time") is None or fields.get("area") is None:
                    continue
                peaks.append(Peak(**fields))
            return peaks
        return []

    @staticmethod
    def extract_wavelength(lines: list[str]) -> int | None:
        """Detection wavelength in nm from the report's ``Signal:`` box.

        ``DAD1A,Sig=254,4  Ref=360,100`` -> ``254``. ``None`` when the report has no
        signal box, or one without a wavelength.
        """
        for line in lines:
            if "│Signal:│" in line.replace(" ", ""):
                match = re.search(r"Sig=(\d+)", line)
                return int(match.group(1)) if match else None
        return None
