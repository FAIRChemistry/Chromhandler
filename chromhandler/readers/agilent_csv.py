from pathlib import Path

import pandas as pd

from chromhandler.model import Chromatogram, Data, Measurement, Peak
from chromhandler.readers.abstractreader import AbstractReader


class AgilentCSVReader(AbstractReader):
    def read(self) -> list[Measurement]:
        """Reads chromatographic data from the specified Agilent CSV files.

        Returns:
            list[Measurement]: A list of Measurement objects representing the chromatographic data.
        """

        assert len(self.values) == len(self.file_paths), f"""
        The number of reaction times {len(self.values)} does not match the number of
        'RESULTS.CSV' files {len(self.file_paths)}.
        """

        measurements = []
        for path_idx, csv_path in enumerate(sorted(self.file_paths)):
            peaks = self._read_peaks_from_csv(csv_path)
            chromatogram = Chromatogram(peaks=peaks)

            data = Data(
                value=self.values[path_idx],
                unit=self.unit.name,
                data_type=self.mode,
            )

            measurements.append(
                Measurement(
                    id=f"m{path_idx}",
                    chromatograms=[chromatogram],
                    temperature=self.temperature,
                    temperature_unit=self.temperature_unit.name,
                    ph=self.ph,
                    data=data,
                )
            )

        if not self.silent:
            self.print_success(len(measurements))

        return measurements

    def _read_peaks_from_csv(self, path: str, skiprows: int = 6) -> list[Peak]:
        """Reads peaks from an Agilent CSV file."""
        peaks = []
        df = pd.read_csv(path, skiprows=skiprows)
        records = df.to_dict(orient="records")

        for record in records:
            peaks.append(
                Peak(
                    retention_time=record["R.T."],
                    area=record["Area"],
                    amplitude=record["Height"],
                    percent_area=record["Pct Total"],
                )
            )

        return peaks

    @staticmethod
    def sort_paths_by_last_parent(paths: list[str]) -> list[str]:
        """
        Sorts a list of file paths by the name of the last parent directory.

        Args:
            paths (List[str]): A list of file paths as strings.

        Returns:
            List[str]: A list of sorted file paths by the last parent directory.
        """
        # Sort paths by their last parent directory and return them as strings
        sorted_paths = sorted(paths, key=lambda p: Path(p).parent.name)

        # Return the sorted paths as strings
        return [str(path) for path in sorted_paths]
