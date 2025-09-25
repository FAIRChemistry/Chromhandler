import re
from io import StringIO
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import pandas as pd

from chromhandler.model import Data, Measurement, Peak, SignalType
from chromhandler.readers.abstractreader import AbstractReader


class ShimadzuReader(AbstractReader):
    SECTION_PATTERN: ClassVar[re.Pattern] = re.compile(r"\[(.*)\]")

    def model_post_init(self, __context: Any) -> None:
        if not self.file_paths:
            self._get_file_paths()
        self._detector_id: Optional[str] = None
        self._channel_ids: List[str] = []

    def read(self) -> list[Measurement]:
        """Reads the chromatographic data from the specified file.

        Returns:
            Measurement: A Measurement object representing the chromatographic data.
        """

        if len(self.file_paths) == 0:
            raise ValueError("No files found. Is the directory empty?")

        measurements = []
        for i, file in enumerate(sorted(self.file_paths)):
            content = self.open_file(file)
            sections = self.create_sections(content)
            self._get_available_detectors(sections)

            info_section = sections.get("File Information")
            assert info_section, "No file information section found."

            acqisition_date = self.get_section_dict(info_section).get("Generated")
            assert acqisition_date, "No acquisition date found."

            # timestamp = datetime.strptime(acqisition_date, "%d.%m.%Y. %H:%M:%S")

            sample_section = sections.get("Sample Information")
            assert sample_section, "No sample information section found."

            sample_dict = self.get_section_dict(sample_section)
            assert sample_dict, "No sample information found."

            dilution_factor = sample_dict.get("Dilution Factor", 1)
            injection_volume = sample_dict.get("Injection Volume")

            data = Data(
                value=self.values[i],
                unit=self.unit.name,
                data_type=self.mode,
            )

            measurement = Measurement(
                id=str(Path(file).stem),
                data=data,
                temperature=self.temperature,
                temperature_unit=self.temperature_unit.name,
                ph=self.ph,
                injection_volume=injection_volume,
                dilution_factor=dilution_factor,
                injection_volume_unit="ul",
            )

            # peak tables
            for channel_id in self._channel_ids:
                for section_key, section in sections.items():
                    if channel_id not in section_key:
                        continue

                    section = section.strip()

                    # Extract peak table
                    if "Peak Table" in section_key:
                        peak_list = self.add_peaks(section)
                        if not peak_list:
                            continue

                        peaks = [Peak(**peak) for peak in peak_list]
                        # add timestamp from measurement to peaks

                    # Extract measurement data
                    if "Chromatogram" in section_key:
                        meta_section, section = re.split(r"(?=R\.Time)", section)

                        chromatogram_meta = self.get_section_dict(meta_section)

                        chromatogram_df = self.parse_chromatogram(section)

                        measurement.add_to_chromatograms(
                            wavelength=int(chromatogram_meta["Wavelength(nm)"]),
                            type=SignalType.UV,
                            signals=chromatogram_df["Intensity"].tolist(),
                            times=chromatogram_df["R.Time (min)"].tolist(),
                            peaks=peaks,
                        )
            measurements.append(measurement)

        if not self.silent:
            self.print_success(len(measurements))

        return measurements

    def parse_chromatogram(self, table_string: str) -> pd.DataFrame:
        """Parse the data in a section as a table."""

        lines = table_string.split("\n")
        header = lines[0]
        data = lines[1:]
        if "\t" in table_string:
            separator = "\t"
        else:
            separator = ","

        if not header.count(",") == data[0].count(","):
            data = "\n".join(data)
            pattern = r"(\b\d+),(\d+\b)"
            data = re.sub(pattern, r"\1.\2", data)
        else:
            data = "\n".join(data)

        table = pd.read_csv(StringIO(header + "\n" + data), sep=separator, header=0)

        return table

    def open_file(self, path: str) -> str:
        """Read the content of a file as a string."""

        try:
            return Path(path).read_text(encoding="ISO-8859-1")
        except UnicodeDecodeError:
            raise IOError(f"Could not read file {path}")

    def create_sections(self, file_content: str) -> dict:
        """Parse a Shimadzu ASCII-export file into sections."""

        # Split file into sections using section header pattern
        section_splits = re.split(self.SECTION_PATTERN, file_content)
        if len(section_splits[0]) != 0:
            raise IOError("The file should start with a section header")

        section_names = section_splits[1::2]
        section_contents = [content for content in section_splits[2::2]]

        return dict(zip(section_names, section_contents))

    def get_section_dict(self, section: str, nrows: int | None = None) -> dict:
        """Parse the metadata in a section as keys-values."""

        if "\t" in section:
            seperator = "\t"
        else:
            seperator = ","

        try:
            meta_table = (
                pd.read_table(
                    StringIO(section),
                    nrows=nrows,
                    header=None,
                    names=range(3),
                    sep=seperator,
                )
                .set_index(0)[1]
                .to_dict()
            )

            return meta_table

        except pd.errors.ParserError:
            try:
                pattern = r"(\b\d+),(\d+\b)"
                section = re.sub(pattern, r"\1.\2", section)

                meta_table = (
                    pd.read_table(
                        StringIO(section),
                        nrows=nrows,
                        header=None,
                        sep=seperator,
                    )
                    .set_index(0)[1]
                    .to_dict()
                )

                return meta_table
            except pd.errors.ParserError:
                return preprocess_to_dict(section)

    def preprocess_decimal_delimiters(self, data_str):
        pattern = r"(\d),(\d{3}\b)"

        return re.sub(pattern, r"\1.\2", data_str)

    def _map_peak_table(self, table: pd.DataFrame) -> dict:
        try:
            return table.apply(
                lambda row: {
                    "retention_time": row["R.Time"],
                    "peak_start": row["I.Time"],
                    "peak_end": row["F.Time"],
                    "amplitude": row["Height"],
                    "area": row["Area"],
                    # "width": row["F.Time"] - row["I.Time"],
                    "tailing_factor": row["Tailing"],
                    "separation_factor": row["Sep.Factor"],
                },
                axis=1,
            ).tolist()
        except KeyError as e:
            raise e

    def add_peaks(self, section: str) -> dict:
        try:
            meta, section = re.split(r"(?=Peak#)", section)
        except ValueError:
            return {}  # No peaks in the section

        data_lines = section.split("\n")
        header = data_lines[0]
        data = data_lines[1:]
        seperators_header = header.count(",")
        seperators_data = data[0].count(",")
        if seperators_header != seperators_data:
            data = "\n".join(data)
            data = self.preprocess_decimal_delimiters(data)
        else:
            data = "\n".join(data)

        section = header + "\n" + data
        if "\t" in section:
            separator = "\t"
        else:
            separator = ","
        table = pd.read_csv(StringIO(section), sep=separator, header=0)
        peaks = self._map_peak_table(table)

        return peaks

    def _get_available_detectors(self, sections: Dict[str, str]) -> None:
        configuration = sections.get("Configuration")
        assert configuration, "No configuration section found."

        detector_info = self.get_section_dict(configuration)
        assert detector_info, "No detector information found."

        detector_id = detector_info.get("Detector ID")[0]
        assert detector_id, "No detector ID found."

        n_channels = detector_info.get("# of Channels")[0]
        assert n_channels, "No number of channels found."

        channel_ids = [f"Ch{i}" for i in range(1, int(n_channels) + 1)]

        self._detector_id = detector_id
        self._channel_ids = channel_ids

    def _get_file_paths(self) -> None:
        files = []
        directory = Path(self.dirpath)

        for file_path in directory.glob("*.txt"):
            if file_path.name.startswith("."):
                continue

            files.append(str(file_path.absolute()))

        assert (
            len(files) == len(self.values)
        ), f"Number of files ({len(files)}) does not match the number of reaction times ({len(self.values)})."

        self.file_paths = files

    def get_detector_info(self, section: str) -> dict:
        """Parse the metadata in a section using regex."""
        config_dict = {}

        # Regex pattern to match lines like 'Key<TAB>Value(s)'
        pattern = re.compile(r"^(.*?)\t(.*)$", re.MULTILINE)

        matches = pattern.findall(section.strip())

        for key, values_str in matches:
            # Split values by tab and strip whitespace
            values = [v.strip() for v in values_str.strip().split("\t") if v.strip()]
            key = key.strip()
            # Store values as list if multiple, or as a single string
            if len(values) == 1:
                config_dict[key] = values[0]
            elif values:
                config_dict[key] = values
            else:
                config_dict[key] = None

        return config_dict


def preprocess_to_dict(input_string: str) -> dict:
    # Initialize the result dictionary
    result = {}

    # Split the input string into lines
    lines = input_string.strip().split("\n")

    for line in lines:
        # Split the line by the first comma
        parts = line.split(",", 1)  # Split only at the first comma

        # Extract key and value, stripping any leading/trailing whitespace
        key = parts[0].strip()
        value = parts[1].strip() if len(parts) > 1 else None

        # Try to convert the value to a number if possible
        try:
            value = float(value) if "." in value else float(value)
        except (ValueError, TypeError):
            pass  # If conversion fails, keep the value as a string

        # Add the key-value pair to the dictionary
        result[key] = value

    return result
