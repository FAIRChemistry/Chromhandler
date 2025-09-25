import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from loguru import logger
from mdmodels.units.annotation import UnitDefinition

from chromhandler.model import Chromatogram, Data, Measurement
from chromhandler.readers.abstractreader import AbstractReader


class ChromeleonReader(AbstractReader):
    def model_post_init(self, __context: Any) -> None:
        if not self.file_paths:
            logger.debug(
                "Collecting file paths without reaction time and unit parsing."
            )
            self._get_file_paths()

    def read(self) -> list[Measurement]:
        """Reads the chromatographic data from the specified files.

        Returns:
            list[Measurement]: A list of Measurement objects representing the chromatographic data.
        """

        measurements = []
        for file_id, file in enumerate(self.file_paths):
            content = self._read_chromeleon_file(file)
            measurement = self._map_measurement(
                content, self.values[file_id], self.unit, file
            )
            measurements.append(measurement)

        if not self.silent:
            self.print_success(len(measurements))

        return measurements

    def _read_chromeleon_file(self, file_path: str) -> Dict[str, Any]:
        """Reads and processes the content of a Chromeleon file."""

        with open(file_path, "r", encoding="ISO-8859-1") as file:
            content = file.read()

        sections = content.split("\n\n")
        sections = [section.lstrip() for section in sections if len(section) > 0]

        content_dict = {}
        for section in sections[1:]:
            section_lines = section.split("\n")
            section_name = section_lines[0][:-1]
            section_content = [line.split("\t") for line in section_lines[1:]]
            for line_id, line in enumerate(section_content):
                section_content[line_id] = [value for value in line if value]

            content_dict[section_name] = section_content

        content_dict["Raw Data"] = self._transpose_data(content_dict["Raw Data"])

        return content_dict

    def _map_measurement(
        self,
        content: Dict[str, Any],
        reaction_time: float,
        time_unit: UnitDefinition,
        file_path: str,
    ) -> Measurement:
        """Maps the parsed content to a Measurement object."""

        chromatogram = Chromatogram(
            wavelength=int(content["Signal Parameter Information"][1][1].split(" ")[0]),
            times=content["Raw Data"]["time"],
            signals=content["Raw Data"]["value"],
        )

        # reaction_time, unit = self._extract_reaction_time(file_name)

        data = Data(
            value=reaction_time,
            unit=time_unit.name,
            data_type=self.mode,
        )

        # Try to get ID from sample information, fallback to filename
        try:
            measurement_id = content["Sample Information"][5][1]
        except (KeyError, IndexError):
            measurement_id = self._get_measurement_id_from_file(file_path)

        return Measurement(
            id=measurement_id,
            chromatograms=[chromatogram],
            injection_volume=float(
                content["Sample Information"][13][1].replace(",", ".")
            ),
            injection_volume_unit="ul",
            dilution_factor=float(
                content["Sample Information"][14][1].replace(",", ".")
            ),
            ph=self.ph,
            temperature=self.temperature,
            temperature_unit=self.temperature_unit.name,
            data=data,
        )

    def _extract_reaction_time(
        self, file_name: str
    ) -> tuple[float | None, UnitDefinition | None]:
        """Extracts reaction time and unit from the file name."""

        pattern = r"\b(\d+(?:\.\d+)?)\s*(h|min)\b"
        matches = re.findall(pattern, file_name)

        if len(matches) == 0:
            return None, None

        reaction_time, unit_str = matches[0]

        return float(reaction_time), unit_str

    def _transpose_data(self, data: List[List[str]]) -> pd.DataFrame:
        """Transposes the raw data from the file into a DataFrame."""

        df = pd.DataFrame(data[1:], columns=["time", "step", "value"])

        df["time"] = df["time"].str.replace(",", ".").astype(float)
        df["step"] = df["step"].str.replace(",", ".").astype(float)
        df["value"] = df["value"].str.replace(",", ".").astype(float)

        # drop rows with NaN values
        df.dropna(inplace=True)

        return df

    def _get_file_paths(self) -> None:
        """Collects the file paths from the directory."""

        files = []
        directory = Path(self.dirpath)

        # check if directory exists
        assert directory.exists(), f"Directory '{self.dirpath}' does not exist."
        assert directory.is_dir(), f"'{self.dirpath}' is not a directory."
        assert any(
            directory.rglob("*.txt")
        ), f"No .txt files found in '{self.dirpath}'."

        for file_path in directory.iterdir():
            if file_path.name.startswith(".") or not file_path.name.endswith(".txt"):
                continue

            files.append(str(file_path.absolute()))

        assert (
            len(files) == len(self.values)
        ), f"Number of files ({len(files)}) does not match the number of reaction times ({len(self.values)})."

        self.file_paths = sorted(files)
