import json
from pathlib import Path
from typing import Any

from loguru import logger

from chromhandler.model import Chromatogram, Data, Measurement, Peak
from chromhandler.readers.abstractreader import AbstractReader


class ASMReader(AbstractReader):
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
        for i, file in enumerate(self.file_paths):
            content = self._read_asm_file(file)
            measurement = self._map_measurement(content, self.values[i], file)
            measurements.append(measurement)

        if not self.silent:
            self.print_success(len(measurements))

        return measurements

    def _get_file_paths(self) -> None:
        """Collects the file paths from the directory."""

        files = []
        directory = Path(self.dirpath)

        # check if directory exists
        assert directory.exists(), f"Directory '{self.dirpath}' does not exist."
        assert directory.is_dir(), f"'{self.dirpath}' is not a directory."
        assert any(
            directory.rglob("*.json")
        ), f"No .json files found in '{self.dirpath}'."

        for file_path in directory.iterdir():
            if file_path.name.startswith(".") or not file_path.name.endswith(".json"):
                continue

            files.append(str(file_path.absolute()))

        assert (
            len(files) == len(self.values)
        ), f"Number of files ({len(files)}) does not match the number of reaction times ({len(self.values)})."

        self.file_paths = sorted(files)

    def _read_asm_file(self, file_path: str) -> Any:
        with open(file_path, "r") as file:
            content = json.load(file)

        return content

    def _map_measurement(
        self,
        content: dict[str, Any],
        reaction_time: float,
        path: str,
    ) -> Measurement:
        if "liquid chromatography aggregate document" in content:
            return self._map_lc_measurement(content, reaction_time, path)
        elif "gas chromatography aggregate document" in content:
            return self._map_gc_measurement(content, reaction_time, path)
        else:
            raise ValueError("Chromatogram type not recognized.")

    def _map_lc_measurement(
        self,
        content: dict[str, Any],
        reaction_time: float,
        path: str,
    ) -> Measurement:
        doc = content["liquid chromatography aggregate document"][
            "liquid chromatography document"
        ]

        if len(doc) > 1:
            logger.warning(
                f"More than one chromatogram found in file '{path}'. Using the first chromatogram only."
            )

        try:
            sample_document = doc[0]["sample document"]
            meas_document = doc[0]["measurement document"]
        except KeyError:
            sample_document = doc[0]["measurement aggregate document"][
                "measurement document"
            ]["sample document"]
            meas_document = doc[0]["measurement aggregate document"][
                "measurement document"
            ]

        # sample info
        name = sample_document.get("written name")
        sample_id = sample_document.get("sample identifier")
        if not sample_id and name:
            sample_id = name

        # Fallback to filename if no sample ID available
        if not sample_id:
            sample_id = self._get_measurement_id_from_file(path)

        if isinstance(meas_document, list):
            meas_document = meas_document[0]

        # signal and time
        signal = meas_document["chromatogram data cube"]["data"]["measures"][0]
        time = meas_document["chromatogram data cube"]["data"]["dimensions"][0]
        time_unit = meas_document["chromatogram data cube"]["cube-structure"][
            "dimensions"
        ][0]["unit"]

        if time_unit == "s":
            # to min
            time = [t / 60 for t in time]
        elif time_unit == "min":
            pass
        else:
            raise ValueError(f"Unit '{time_unit}' not recognized")

        try:
            peak_list = meas_document["peak list"]["peak"]
        except KeyError:
            analyte_document = doc[0]["analyte aggregate document"]["analyteDocument"]

            if len(analyte_document) > 1:
                logger.warning(
                    f"More than one analyte document found in '{path}'. Using the first analyte document only."
                )

            peak_list = analyte_document[0]["peak list"]["peak"]

        peaks = [self.map_peaks(peak) for peak in peak_list]

        chrom = Chromatogram(
            peaks=peaks,
            signals=signal,
            times=time,
        )

        data = Data(
            value=reaction_time,
            unit=self.unit.name,
            data_type=self.mode,
        )

        return Measurement(
            id=sample_id,
            sample_name=name,
            temperature=self.temperature,
            temperature_unit=self.temperature_unit.name,
            ph=self.ph,
            chromatograms=[chrom],
            data=data,
        )

    def _map_gc_measurement(
        self,
        content: dict[str, Any],
        reaction_time: float,
        path: str,
    ) -> Measurement:
        doc = content["gas chromatography aggregate document"][
            "gas chromatography document"
        ]

        if len(doc) > 1:
            logger.warning(
                f"More than one chromatogram found in file '{path}'. Using the first chromatogram only."
            )

        meas_document = doc[0]["measurement aggregate document"]["measurement document"]

        if isinstance(meas_document, list):
            meas_document = meas_document[0]

        sample_document = meas_document["sample document"]

        # sample info
        name = sample_document.get("written name")
        sample_id = sample_document.get("sample identifier")
        if not sample_id and name:
            sample_id = name

        # Fallback to filename if no sample ID available
        if not sample_id:
            sample_id = self._get_measurement_id_from_file(path)

        # signal and time
        signal = meas_document["chromatogram data cube"]["data"]["measures"][0]
        time = meas_document["chromatogram data cube"]["data"]["dimensions"][0]
        time_unit = meas_document["chromatogram data cube"]["cube-structure"][
            "dimensions"
        ][0]["unit"]

        if time_unit == "s":
            # to min
            time = [t / 60 for t in time]

        elif time_unit == "min":
            pass
        else:
            raise ValueError(f"Unit '{time_unit}' not recognized")

        peak_list = meas_document["processed data document"]["peak list"]["peak"]
        peaks = [self.map_peaks(peak) for peak in peak_list]

        chrom = Chromatogram(
            peaks=peaks,
            signals=signal,
            times=time,
        )

        data = Data(
            value=reaction_time,
            unit=self.unit.name,
            data_type=self.mode,
        )

        return Measurement(
            id=sample_id,
            sample_name=name,
            temperature=self.temperature,
            temperature_unit=self.temperature_unit.name,
            ph=self.ph,
            chromatograms=[chrom],
            data=data,
        )

    def map_peaks(self, peak_dict: dict[str, Any]) -> Peak:
        area = peak_dict["peak area"]
        peak_area = area["value"]
        if len(list(area.keys())) == 2:
            if area["unit"] == "mAU.s":
                peak_area = area["value"] * 60
            elif area["unit"] == "mAU.min":
                peak_area = area["value"]
            elif area["unit"] == "unitless":
                peak_area = area["value"]
            else:
                raise ValueError(f"Unit '{area['unit']}' not recognized")

        if "peak width at half height" in peak_dict:
            width = peak_dict["peak width at half height"]
            if width["unit"] == "s":
                width["value"] /= 60
            elif width["unit"] == "min":
                pass
            else:
                raise ValueError(f"Unit '{width['unit']}' not recognized")
            width_value = width["value"]
        else:
            width_value = None

        retention_time = peak_dict["retention time"]
        if retention_time["unit"] == "s":
            retention_time["value"] /= 60
        elif retention_time["unit"] == "min":
            pass
        else:
            raise ValueError(f"Unit '{retention_time['unit']}' not recognized")

        peak_start = peak_dict["peak start"]
        if peak_start["unit"] == "s":
            peak_start["value"] /= 60
        elif peak_start["unit"] == "min":
            pass
        else:
            raise ValueError(f"Unit '{peak_start['unit']}' not recognized")

        peak_end = peak_dict["peak end"]
        if peak_end["unit"] == "s":
            peak_end["value"] /= 60
        elif peak_end["unit"] == "min":
            pass
        else:
            raise ValueError(f"Unit '{peak_end['unit']}' not recognized")

        try:
            asym_factor = peak_dict["chromatographic peak asymmetry factor"]["value"]
        except KeyError:
            asym_factor = None
        except TypeError:
            asym_factor = None

        return Peak(
            retention_time=peak_dict["retention time"]["value"],
            area=peak_area,
            amplitude=peak_dict["peak height"]["value"],
            width=width_value,
            skew=asym_factor,
            percent_area=peak_dict["relative peak area"]["value"],
            peak_start=peak_dict["peak start"]["value"],
            peak_end=peak_dict["peak end"]["value"],
        )
