from __future__ import annotations

import re
from abc import abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger
from mdmodels.units.annotation import UnitDefinitionAnnot
from pydantic import BaseModel, Field, field_validator, model_validator

from chromhandler.model import DataType, Measurement


class MetadataExtractionError(Exception):
    def __init__(self, message: str, suggestion: str | None = None) -> None:
        if suggestion:
            message += f"\n{str(suggestion)}"
        super().__init__(message)


class UnitConsistencyError(Exception):
    def __init__(self, message: str, suggestion: str | None = None) -> None:
        if suggestion:
            message += f"\n{str(suggestion)}"
        super().__init__(message)


class FileNotFoundInDirectoryError(Exception):
    def __init__(self, message: str, suggestion: str | None = None) -> None:
        if suggestion:
            message += f"\n{str(suggestion)}"
        super().__init__(message)


def natural_key(path: str | Path) -> list[str | float]:
    """Sort key that compares the numbers inside a path as numbers, so `s_2.88min` < `s_10.38min`.

    Readers zip `values` onto their file paths positionally, so the paths must be in the
    order a human reads the file names in, not text order.
    """
    parts = re.split(r"(\d+(?:\.\d+)?)", str(path))
    return [float(part) if i % 2 else part for i, part in enumerate(parts)]


class AbstractReader(BaseModel):
    """
    Abstract class for reading chromatographic data from files.

    Attributes:
        dirpath (str | Path): Path to the directory containing chromatographic data files.
        mode (str): Mode of data processing, either 'calibration' or 'timecourse'.
        values (Optional[List[float]]): List of reaction times or concentrations based on the mode.
        unit (Optional[UnitDefinitionAnnot]): Unit of the values (time unit for timecourse, concentration unit for calibration).
        ph (float): pH value of the measurement.
        temperature (float): Temperature of the measurement.
        temperature_unit (UnitDefinitionAnnot): Unit of the temperature.
        silent (bool): If True, suppresses output messages.
        file_paths (List[str]): List of file paths to process.
    """

    dirpath: str | Path = Field(
        ...,
        description="Path to the directory containing chromatographic data files.",
    )

    mode: str = Field(
        ...,
        description="Mode of data processing: 'calibration' or 'timecourse'.",
    )

    values: list[float] = Field(
        ...,
        description=(
            "List of reaction times for 'timecourse' mode or concentrations for 'calibration' mode."
        ),
    )

    unit: UnitDefinitionAnnot = Field(
        ...,
        description=(
            "Unit of the values: "
            "Use time units (e.g., 'minute', 'second') for 'timecourse' mode and concentration units (e.g., 'mM', 'uM') for 'calibration' mode."
        ),
    )

    ph: float = Field(
        ...,
        description="pH value of the measurement.",
    )

    temperature: float = Field(
        ...,
        description="Temperature of the measurement.",
    )

    temperature_unit: UnitDefinitionAnnot = Field(
        default="Celsius",
        description="Unit of the temperature. Defaults to Celsius.",
    )

    silent: bool = Field(
        False,
        description="If True, suppresses output messages.",
    )

    file_paths: list[str] = Field(
        default_factory=list,
        description="List of file paths to process.",
    )

    @field_validator("mode", mode="before")
    def validate_mode(cls, value: str) -> str:
        value = value.lower()
        if value not in {DataType.CALIBRATION.value, DataType.TIMECOURSE.value}:
            raise ValueError("Invalid mode. Must be 'calibration' or 'timecourse'.")
        return value

    @model_validator(mode="before")
    @classmethod
    def infer_fields_based_on_mode(cls, values: dict[str, Any]) -> dict[str, Any]:
        mode = values.get("mode")
        if mode == DataType.TIMECOURSE.value:
            # Ensure 'values' (reaction times) and 'unit' are provided or parsed
            if not values.get("values"):
                try:
                    values = cls._parse_data_and_unit(values, mode)
                except MetadataExtractionError as e:
                    logger.debug(e)
                    raise MetadataExtractionError(
                        str(e),
                        suggestion="Alternatively, provide the reaction times and unit manually, using the `values` attribute.",
                    )
                except UnitConsistencyError as e:
                    logger.debug(e)
                    raise UnitConsistencyError(str(e))
        elif mode == DataType.CALIBRATION.value:
            # Ensure 'values' (concentrations) and 'unit' are provided or parsed
            if not values.get("values"):
                try:
                    values = cls._parse_data_and_unit(values, mode)
                except MetadataExtractionError as e:
                    logger.debug(e)
                    raise MetadataExtractionError(
                        str(e),
                        suggestion="Alternatively, provide the concentrations and unit manually.",
                    )
                except UnitConsistencyError as e:
                    logger.debug(e)
                    raise UnitConsistencyError(str(e))
        else:
            raise ValueError("Invalid mode. Must be 'timecourse' or 'calibration'.")
        return values

    @model_validator(mode="after")
    def validate_data_consistency(self) -> AbstractReader:
        if not self.file_paths:
            raise FileNotFoundInDirectoryError(
                f"No files found in the directory {self.dirpath}."
            )

        if self.mode == DataType.TIMECOURSE.value:
            if not self.values:
                raise ValueError("No reaction times provided for timecourse mode.")
            if not self.unit:
                raise ValueError("No time unit provided for timecourse mode.")
            if len(self.file_paths) != len(self.values):
                raise ValueError(
                    f"Number of files ({len(self.file_paths)}) does not match the number of reaction times ({len(self.values)})."
                )
        elif self.mode == DataType.CALIBRATION.value:
            if not self.values:
                raise ValueError("No concentrations provided for calibration mode.")
            if not self.unit:
                raise ValueError("No concentration unit provided for calibration mode.")
            if len(self.file_paths) != len(self.values):
                raise ValueError(
                    f"Number of files ({len(self.file_paths)}) does not match the number of concentrations ({len(self.values)})."
                )
        return self

    @classmethod
    def _parse_data_and_unit(cls, data: dict[str, Any], mode: str) -> dict[str, Any]:
        """Parse data and unit from filenames based on the mode."""

        try:
            filenames = sorted(
                [Path(f) for f in data.get("file_paths", [])], key=natural_key
            )
            if not filenames:
                raise KeyError
        except KeyError:
            path = Path(data["dirpath"])
            if not path.exists():
                raise FileNotFoundError(
                    f"Directory '{data['dirpath']}' does not exist."
                )
            if not path.is_dir():
                raise NotADirectoryError(f"'{data['dirpath']}' is not a directory.")

            # Get all filenames of normal files in the directory, exclude hidden files
            filenames = sorted(
                [f for f in path.iterdir() if not f.name.startswith(".")],
                key=natural_key,
            )

        # Define patterns based on the mode
        if mode == DataType.TIMECOURSE.value:
            # Pattern to extract reaction times and units
            pattern = r".*?(\d+(\.\d+)?)\s*[_-]?\s*(min|minutes?|sec|seconds?|hours?).*"
        elif mode == DataType.CALIBRATION.value:
            # Pattern to extract concentrations and units
            pattern = r".*?(\d+(\.\d+)?)\s*[_-]?\s*(mM|µM|uM|nM|mol|mmol|umol|nmol).*"

        else:
            raise ValueError("Invalid mode.")

        # Extract data and units from filenames
        data_dict: dict[str, float] = {}
        units = []

        for file in filenames:
            match = re.search(pattern, file.name)
            if not match:
                match = re.search(pattern, file.parent.name)
            if match:
                value = float(match.group(1))
                unit_str = match.group(3)
                data_dict[str(file.absolute())] = value
                units.append(unit_str)
            else:
                logger.debug(f"Could not parse value from '{file.name}'.")
                raise MetadataExtractionError(
                    f"Could not parse value from '{file.name}'."
                )

        # Check if all units are the same
        if not all(unit == units[0] for unit in units):
            logger.debug(
                f"Units in directory '{data['dirpath']}' are not consistent: {units}"
            )
            raise UnitConsistencyError(
                f"Units in directory '{data['dirpath']}' are not consistent: {units}"
            )

        try:
            unit_definition = cls._map_unit_str_to_UnitDefinitionAnnot(units[0], mode)
        except ValueError:
            logger.debug(
                f"Unit {units[0]} from directory '{data['dirpath']}' not recognized."
            )
            raise MetadataExtractionError(
                f"Unit {units[0]} from directory '{data['dirpath']}' not recognized."
            )

        # Sort the data and file paths
        sorted_items = sorted(data_dict.items(), key=lambda x: x[1])
        file_paths = [item[0] for item in sorted_items]
        data_values = [item[1] for item in sorted_items]

        data["file_paths"] = file_paths
        data["values"] = data_values
        data["unit"] = unit_definition

        return data

    @staticmethod
    def _map_unit_str_to_UnitDefinitionAnnot(
        unit_str: str,
        mode: str,
    ) -> UnitDefinitionAnnot:
        """Maps a string representation of a unit to a `UnitDefinitionAnnot` object based on the mode."""

        unit_str = unit_str.lower()
        if mode == DataType.TIMECOURSE.value:
            # Time units

            match unit_str:
                case "min" | "mins" | "minute" | "minutes":
                    return "minute"
                case "sec" | "secs" | "second" | "seconds":
                    return "second"
                case "hour" | "hours":
                    return "hour"
                case _:
                    raise ValueError(f"Time unit '{unit_str}' not recognized.")
        elif mode == DataType.CALIBRATION.value:
            # Concentration units

            match unit_str:
                case "m" | "M":
                    return "mol / l"
                case "mm" | "mM":
                    return "mmol / l"
                case "um" | "uM" | "µM":
                    return "umol / l"
                case "nm" | "nM":
                    return "nmol / l"
                case _:
                    raise ValueError(f"Concentration unit '{unit_str}' not recognized.")
        else:
            raise ValueError(f"Invalid mode '{mode}'.")

    def _get_measurement_id_from_file(self, file_path: str) -> str:
        """Helper method to get consistent measurement ID from file path.

        Args:
            file_path (str): Path to the file

        Returns:
            str: The filename stem (without extension) to use as measurement ID
        """
        return Path(file_path).stem

    @abstractmethod
    def read(self) -> list[Measurement]:
        """Abstract method that must be implemented by subclasses."""
        pass

    def print_success(self, n_measurement_objects: int) -> None:
        """Prints a success message."""
        print(f"Loaded {n_measurement_objects} chromatograms.")
