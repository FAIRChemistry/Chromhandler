import io
from datetime import datetime
from pathlib import Path
from typing import Any

from chromhandler.model import Chromatogram, Data, Measurement, Peak
from chromhandler.readers.abstractreader import AbstractReader


class ThermoTX0Reader(AbstractReader):
    def __init__(self, dirpath: str, **kwargs: Any) -> None:
        # Find all TX0 files in the directory first
        path = Path(dirpath)
        file_paths = sorted([str(f) for f in path.glob("*.TX0")])
        # Then initialize parent class with all paths already set
        super().__init__(dirpath=dirpath, file_paths=file_paths, **kwargs)

    def read(self) -> list[Measurement]:
        measurements: list[Measurement] = []
        for idx, path in enumerate(self.file_paths):
            peaks = _read_peaks_from_csv(path)
            metadata = _read_metadata(path)

            chromatogram = Chromatogram(peaks=peaks)
            data = Data(
                value=self.values[idx],
                unit=self.unit.name,
                data_type=self.mode,
            )

            measurements.append(
                Measurement(
                    id=metadata.get(
                        "sample_name", self._get_measurement_id_from_file(path)
                    ),
                    chromatograms=[chromatogram],
                    data=data,
                    timestamp=metadata.get("acquisition_time"),
                    temperature=self.temperature,
                    temperature_unit=self.temperature_unit.name,
                    ph=self.ph,
                )
            )

        if not self.silent:
            self.print_success(len(measurements))

        return measurements


def _parse_line_with_decimal_comma(line: str) -> list[str]:
    """Parse a line where comma is used as decimal separator.

    Args:
        line: Line to parse, e.g. '1,"0,038","257,10","696,06","0,00","0,00","BB","0,3694"'

    Returns:
        List of values with quotes removed but preserving empty strings.
        Always returns a list of the same length as the number of comma-separated values.
    """
    result: list[str] = []
    current: list[str] = []
    in_quotes = False

    # Remove newline if present
    line = line.rstrip("\n")

    for char in line:
        if char == '"':
            in_quotes = not in_quotes
        elif char == "," and not in_quotes:
            # End of field
            result.append("".join(current).strip().strip('"'))
            current = []
        else:
            current.append(char)

    # Append last field
    if current:
        result.append("".join(current).strip().strip('"'))

    return result


def _convert_value(value: str) -> float:
    """Convert a string value to float, handling special cases.

    Args:
        value: String value to convert, e.g. "0,038", "257,10", "696,06,"

    Returns:
        Converted float value
    """
    # Remove trailing comma if present
    value = value.rstrip(",")
    value = value.lstrip(",")
    # Replace decimal comma with period
    value = value.replace(",", ".")
    # Remove any special characters
    value = value.replace("ｵVｷs", "").replace("ｵV", "")
    return float(value)


def _safe_open(path: str) -> io.TextIOWrapper:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1", "shift_jis"]
    for enc in encodings:
        try:
            # read a few bytes to test decoding
            with open(path, mode="r", encoding=enc) as test:
                test.read(2048)
            return open(path, mode="r", encoding=enc)
        except UnicodeDecodeError:
            continue
    # if nothing worked, raise
    raise UnicodeDecodeError("none", b"", 0, 0, "No valid encoding found")


def _read_peaks_from_csv(path: str) -> list[Peak]:
    peaks: list[Peak] = []

    # use robust open instead of fixed Shift-JIS
    with _safe_open(path) as file:
        lines = file.readlines()

        # Find the start of peak data
        start_idx = None
        for i, line in enumerate(lines):
            if '"Peak"' in line and '"Time"' in line:
                start_idx = i + 2  # Skip header and separator
                break

        if start_idx is None:
            return peaks

        # Parse peaks until we hit metadata or end
        for line in lines[start_idx:]:
            if line.startswith('"Warning') or line.startswith('"Missing'):
                break

            parts = _parse_line_with_decimal_comma(line)

            # Skip separator lines or empty peaks
            if (
                not parts
                or any(p.strip("- ") == "" for p in parts)
                or all("-" in p for p in parts)
            ):
                continue

            if parts[1] == "":
                continue

            time = _convert_value(parts[1])
            area = _convert_value(parts[2])
            height = _convert_value(parts[3])
            percent_area = _convert_value(parts[4])

            peaks.append(
                Peak(
                    retention_time=time,
                    area=area,
                    amplitude=height,
                    percent_area=percent_area,
                )
            )

    return peaks


def _extract_value(parts: list[str], key: str) -> str | None:
    """Extract a value from a list of parts given a key.

    Args:
        parts: List of strings to search in
        key: The key to search for (e.g. "Software Version:")

    Returns:
        The value after the key if found, None otherwise
    """
    try:
        idx = parts.index(key)
        return parts[idx + 1]
    except (ValueError, IndexError):
        return None


def _read_metadata(path: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}

    # use the robust opener with fallback encodings
    with _safe_open(path) as file:
        lines = file.readlines()

        for line in lines:
            # skip empty / separator lines
            if not line.strip() or line.startswith("==="):
                continue

            parts = [p.strip('"\n ') for p in line.split(",")]

            # Extract metadata using helper function
            if value := _extract_value(parts, "Software Version:"):
                metadata["software_version"] = value

            elif value := _extract_value(parts, "Sample Name:"):
                metadata["sample_name"] = value

            elif value := _extract_value(parts, "Sample Amount:"):
                try:
                    metadata["sample_amount"] = float(value.replace(",", "."))
                except ValueError:
                    pass

            elif value := _extract_value(parts, "Data Acquisition Time:"):
                try:
                    metadata["acquisition_time"] = str(
                        datetime.strptime(value, "%d-%m-%Y %H:%M:%S")
                    )
                except ValueError:
                    pass

            # Stop once we reach the start of the peak table
            if '"Peak"' in line and '"Time"' in line:
                break

    return metadata
