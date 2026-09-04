import re
from itertools import zip_longest

from chromhandler.model import Chromatogram, Data, Measurement, Peak
from chromhandler.readers.abstractreader import AbstractReader


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


def _boxes(lines: list[str]) -> list[list[list[str]]]:
    """The report's records, grouped by the box that encloses them.

    Two levels of grouping, both taken from the report's own drawing. Within a box,
    ``├─┼─┤`` rules delimit records, so a value the report wrapped across two
    physical lines (``"13."`` then ``"198"``) is rejoined into ``"13.198"``; ragged
    groups are padded rather than truncated. A ``└`` closes the box, which is what
    keeps one peak table's rows from bleeding into the next one's — a report with a
    table per detector signal would otherwise merge them.
    """
    boxes: list[list[list[str]]] = []
    records: list[list[str]] = []
    group: list[list[str]] = []

    def flush_group() -> None:
        if group:
            records.append(
                ["".join(parts).strip() for parts in zip_longest(*group, fillvalue="")]
            )
            group.clear()

    def flush_box() -> None:
        flush_group()
        if records:
            boxes.append(records.copy())
            records.clear()

    for line in lines:
        cells = _cells(line)
        if cells is None:
            flush_group()
            if line.strip().startswith("└"):
                flush_box()
        else:
            group.append(cells)
    flush_box()
    return boxes


def _header_columns(record: list[str]) -> dict[str, int] | None:
    """Map normalised column name -> position, or ``None`` if this is not the header.

    ``"RT [min]"`` normalises to ``"rt"``, ``"Width [min]"`` to ``"width"``. If a
    template ever repeated a column name the last occurrence would win; no known
    template does.
    """
    columns = {cell.split("[")[0].strip().lower(): i for i, cell in enumerate(record)}
    return columns if "rt" in columns and "area" in columns else None


def _cell(row: list[str], columns: dict[str, int], key: str) -> str:
    """The *key* column of *row*, or ``""`` when this template has no such column."""
    position = columns.get(key)
    return row[position] if position is not None and position < len(row) else ""


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
        """Every peak in the report's first peak table.

        A record is a peak iff its ``RT`` and ``Area`` cells both parse as floats,
        which is also what discards the table's trailing ``Sum`` row — it carries no
        retention time. Only the table's own box is read, so a second table further
        down the report cannot contribute rows to this one.
        """
        for box in _boxes(lines):
            for i, record in enumerate(box):
                columns = _header_columns(record)
                if columns is None:
                    continue

                peaks = []
                for row in box[i + 1 :]:
                    retention_time = _maybe_float(_cell(row, columns, "rt"))
                    area = _maybe_float(_cell(row, columns, "area"))
                    if retention_time is None or area is None:
                        continue
                    peaks.append(
                        Peak(
                            retention_time=retention_time,
                            area=area,
                            type=_cell(row, columns, "type") or None,
                            width=_maybe_float(_cell(row, columns, "width")),
                            amplitude=_maybe_float(_cell(row, columns, "height")),
                            percent_area=_maybe_float(_cell(row, columns, "area%")),
                        )
                    )
                return peaks
        return []

    @staticmethod
    def extract_wavelength(lines: list[str]) -> int | None:
        """Detection wavelength in nm from the report's ``Signal:`` box.

        ``DAD1A,Sig=254,4  Ref=360,100`` -> ``254``. Reading the box rather than the
        raw line means a signal cell the report wrapped is rejoined first. ``None``
        when the report has no signal box, or one without a wavelength.
        """
        for box in _boxes(lines):
            for record in box:
                if (
                    len(record) >= 2
                    and record[0].rstrip(":").strip().lower() == "signal"
                ):
                    match = re.search(r"Sig=(\d+)", record[1])
                    return int(match.group(1)) if match else None
        return None
