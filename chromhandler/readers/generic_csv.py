from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from chromhandler.model import Chromatogram, Data, Measurement, Peak
from chromhandler.readers.abstractreader import AbstractReader


class GenericCSVReader(AbstractReader):
    def model_post_init(self, __context: Any) -> None:
        if not self.file_paths:
            logger.debug(
                "Collecting file paths without reaction time and unit parsing."
            )
            self._get_file_paths()

    def read(self) -> list[Measurement]:
        return []

    def extract_peaks(
        self,
        df: pd.DataFrame,
        retention_time_col_name: str,
        peak_area_col_name: str,
    ) -> list[Peak]:
        """Extracts peaks from a pandas DataFrame.

        Args:
            df (pd.DataFrame): The pandas DataFrame to extract peaks from.
            retention_time_col_name (str): The name of the column containing the retention times.
            peak_area_col_name (str): The name of the column containing the peak areas.
        """
        peaks = []
        for _, row in df.iterrows():
            peak = Peak(
                retention_time=row[retention_time_col_name],
                area=row[peak_area_col_name],
            )

            peaks.append(peak)
        return peaks

    def read_generic_csv(
        self,
        retention_time_col_name: str,
        peak_area_col_name: str,
        header: int | None,
    ) -> list[Measurement]:
        measurements = []
        for i, file_name in enumerate(self.file_paths):
            df = pd.read_csv(file_name, header=header)
            peaks = self.extract_peaks(df, retention_time_col_name, peak_area_col_name)

            chromatogram = Chromatogram(
                peaks=peaks,
            )

            data = Data(
                value=self.values[i],
                unit=self.unit.name,
                data_type=self.mode,
            )

            measurement = Measurement(
                id=self._get_measurement_id_from_file(file_name),
                chromatograms=[chromatogram],
                data=data,
                temperature=self.temperature,
                temperature_unit=self.temperature_unit.name,
                ph=self.ph,
            )
            measurements.append(measurement)
        return measurements

    def _get_file_paths(self) -> None:
        """Collects the file paths from the directory."""

        files = []
        directory = Path(self.dirpath)

        # check if directory exists
        assert directory.exists(), f"Directory '{self.dirpath}' does not exist."
        assert directory.is_dir(), f"'{self.dirpath}' is not a directory."
        assert any(
            directory.rglob("*.csv")
        ), f"No .csv files found in '{self.dirpath}'."

        for file_path in directory.iterdir():
            if file_path.name.startswith(".") or not file_path.name.endswith(".csv"):
                continue

            files.append(str(file_path.absolute()))

        assert (
            len(files) == len(self.values)
        ), f"Number of files ({len(files)}) does not match the number of reaction times ({len(self.values)})."

        self.file_paths = sorted(files)
