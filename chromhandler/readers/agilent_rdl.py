import re

from chromhandler.model import Chromatogram, Data, Measurement, Peak
from chromhandler.readers.abstractreader import AbstractReader


class AgilentRDLReader(AbstractReader):
    def read(self) -> list[Measurement]:
        measurements = []
        for path_id, path in enumerate(self.file_paths):
            lines = self.read_file(path)

            peak_data, sample_name, signal = self.extract_information(lines)

            peak_data = [
                self.align_and_concatenate_columns(*pair) for pair in peak_data
            ]
            peaks = [self.map_peak(peak) for peak in peak_data]

            sample_name = self.align_and_concatenate_columns(*sample_name)[1]

            wavelength = self.extract_wavelength(signal)

            data = Data(
                value=self.values[path_id],
                unit=self.unit.name,
                data_type=self.mode,
            )

            chromatogram = Chromatogram(peaks=peaks, wavelength=wavelength)

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
        with open(file_path, encoding="utf-8") as file:
            lines = file.readlines()

        return lines

    @staticmethod
    def extract_information(lines: list[str]) -> tuple[list[list[str]], list[str], str]:
        data = []
        for i, line in enumerate(lines):
            # Check for first line with float values after space and pipe characters
            if re.search(r"^\s{2}│\s+\d+\.\d+\s+│", line):
                line_pair = [line]

                if lines[i + 1].startswith("  │"):
                    line_pair.append(lines[i + 1])
                else:
                    line_pair.append("")

                data.append(line_pair)

            if "│Sample Name     │" in line:
                sample_name = [line]
                if lines[i + 1].startswith("   │"):
                    sample_name.append(lines[i + 1])
                else:
                    sample_name.append("")

            if "│Signal:│" in line:
                signal = line

        return data, sample_name, signal

    @staticmethod
    def extract_wavelength(line: str) -> int | None:
        pattern = r"Sig=(\d+)"

        match = re.search(pattern, line)
        if match:
            return int(match.group(1))
        else:
            return None

    @staticmethod
    def align_and_concatenate_columns(row1: str, row2: str) -> list[str]:
        # Split each string by the vertical bar '│' and strip whitespace from each column
        row1_columns = [col.strip() for col in re.split(r"│", row1) if col]
        row2_columns = [col.strip() for col in re.split(r"│", row2) if col]

        # Concatenate aligned columns
        if row2_columns:
            aligned_columns = [
                f"{col1}{col2}".strip()
                for col1, col2 in zip(row1_columns, row2_columns, strict=False)
            ]
            return aligned_columns[1:-1]

        return row1_columns[1:-1]

    @staticmethod
    def map_peak(peak_list: list[str]) -> Peak:
        return Peak(
            retention_time=float(peak_list[0]),
            type=peak_list[1],
            width=float(peak_list[2]),
            area=float(peak_list[3]),
            amplitude=float(peak_list[4]),
            percent_area=float(peak_list[5]),
        )
