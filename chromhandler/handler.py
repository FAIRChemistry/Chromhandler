from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from typing import Any, Literal, Optional

from calipytion.model import Calibration
from calipytion.tools.utility import pubchem_request_molecule_name
from loguru import logger
from mdmodels.units.annotation import UnitDefinitionAnnot
from pydantic import BaseModel, Field, field_validator
from pyenzyme import EnzymeMLDocument
from rich.console import Console, Group

from . import pretty, visualize
from .model import (
    Chromatogram,
    DataType,
    Measurement,
    Peak,
)
from .molecule import Molecule
from .protein import Protein
from .utility import _resolve_chromatogram


class Handler(BaseModel):
    id: str = Field(
        description="Unique identifier of the given object.",
    )

    name: str = Field(
        description="Name of the Handler object.",
    )

    mode: str = Field(
        ..., description="Mode of data processing: 'calibration' or 'timecourse'."
    )

    molecules: list[Molecule] = Field(
        description="List of species present in the measurements.",
        default_factory=list,
    )

    proteins: list[Protein] = Field(
        description="List of proteins present in the measurements.",
        default_factory=list,
    )

    measurements: list[Measurement] = Field(
        description="List of measurements to be analyzed.",
        default_factory=list,
    )

    internal_standard: Molecule | None = Field(
        description="Internal standard molecule used for concentration calculation.",
        default=None,
    )

    @field_validator("mode", mode="before")
    def validate_mode(cls, value: str) -> str:
        value = value.lower()
        if value not in {DataType.CALIBRATION.value, DataType.TIMECOURSE.value}:
            raise ValueError("Invalid mode. Must be 'calibration' or 'timecourse'.")
        return value

    def add_molecule_from_standard(
        self,
        standard: Calibration,
        init_conc: float,
        conc_unit: str,
    ) -> None:
        """Adds a molecule to the list of species based on a `Calibration` object."""

        molecule = Molecule.from_standard(standard, init_conc, conc_unit)

        self.define_molecule(**molecule.model_dump())

    def add_molecule(
        self,
        molecule: Molecule,
        init_conc: Optional[float] = None,
        conc_unit: Optional[str] = None,
        retention_tolerance: Optional[float] = None,
        min_signal: float = 0.0,
        auto_assign: bool = False,
    ) -> None:
        """
        Adds a molecule to the list of species, allowing to update the initial concentration,
        concentration unit, and retention time tolerance.

        Args:
            molecule (Molecule): The molecule object to be added.
            init_conc (float | None, optional): The initial concentration of the molecule. Defaults to None.
            conc_unit (UnitDefinitionAnnot | None, optional): The unit of the concentration. Defaults to None.
            retention_tolerance (float | None, optional): Retention time tolerance for peak annotation
                in minutes. Defaults to None.
            min_signal (float): Minimum signal threshold for peak assignment. Peaks must have
                an area >= this value to be assigned to this molecule. Defaults to 0.0 (no minimum threshold).
            auto_assign (bool): If True, automatically assigns peaks after adding molecule.
                Set to False to add molecules without assignment for later consolidated reporting. Defaults to False.
        """

        new_mol = copy.deepcopy(molecule)

        if init_conc is not None:
            new_mol.init_conc = init_conc

        if conc_unit is not None:
            new_mol.conc_unit = conc_unit

        if retention_tolerance is not None:
            new_mol.retention_tolerance = retention_tolerance

        new_mol.min_signal = min_signal

        self._update_molecule(new_mol)

        if auto_assign and new_mol.has_retention_time:
            self._register_peaks(
                new_mol, new_mol.retention_tolerance, new_mol.wavelength
            )

    def define_molecule(
        self,
        id: str,
        pubchem_cid: int,
        retention_time: Optional[float],
        retention_tolerance: float = 0.1,
        init_conc: Optional[float] = None,
        conc_unit: Optional[str] = None,
        name: Optional[str] = None,
        wavelength: Optional[float] = None,
        is_internal_standard: bool = False,
        min_signal: float = 0.0,
        auto_assign: bool = False,
    ) -> Molecule:
        """
        Defines and adds a molecule to the list of molecules.

        Args:
            id (str): Internal identifier of the molecule, such as `s0` or `asd45`.
            pubchem_cid (int): PubChem CID of the molecule, which uniquely identifies the compound.
            retention_time (Optional[float]): Retention time for peak annotation in minutes. If the molecule is not
                detected in the chromatograms, this can be set to None.
            retention_tolerance (float, optional): Tolerance in minutes for retention time used in peak annotation.
                Defaults to 0.1.
            init_conc (Optional[float], optional): Initial concentration of the molecule. If not provided, it defaults to None.
            conc_unit (Optional[str], optional): Unit of the initial concentration. If not provided, it defaults to None.
            name (Optional[str], optional): Name of the molecule. If not provided, the name is fetched from the PubChem
                database. Defaults to None.
            wavelength (Optional[float], optional): Wavelength of the detector where the molecule was detected. If not provided,
                it defaults to None.
            is_internal_standard (bool, optional): If True, the molecule is used as internal standard. Defaults to False.
            min_signal (float): Minimum signal threshold for peak assignment. Peaks must have
                an area >= this value to be assigned to this molecule. Defaults to 0.0 (no minimum threshold).
            auto_assign (bool): If True, automatically assigns peaks after defining molecule.
                Set to False to define molecules without assignment for later consolidated reporting. Defaults to False.

        Returns:
            Molecule: The molecule object that was added to the list of species.
        """

        if conc_unit:
            assert (
                init_conc is not None
            ), "Initial concentration must be provided if concentration unit is given."
        if init_conc:
            assert (
                conc_unit
            ), "Concentration unit must be provided if initial concentration is given."

        if name is None:
            name = pubchem_request_molecule_name(pubchem_cid)

        molecule = Molecule(
            id=id,
            pubchem_cid=pubchem_cid,
            name=name,
            init_conc=init_conc,
            conc_unit=conc_unit,
            retention_time=retention_time,
            retention_tolerance=retention_tolerance,
            wavelength=wavelength,
            internal_standard=is_internal_standard,
            min_signal=min_signal,
        )

        self._update_molecule(molecule)

        if auto_assign and molecule.has_retention_time:
            self._register_peaks(molecule, retention_tolerance, wavelength)

        return molecule

    def get_peaks(self, molecule_id: str) -> list[Peak]:
        peaks: list[Peak] = []
        for meas in self.measurements:
            for chrom in meas.chromatograms:
                for peak in chrom.peaks:
                    if peak.molecule_id == molecule_id:
                        peaks.append(peak)

        if not peaks:
            raise ValueError(f"No peaks found for molecule {molecule_id}.")
        return peaks

    def _register_peaks(
        self,
        molecule: Molecule,
        ret_tolerance: float,
        wavelength: float | None,
        silent: bool = False,
    ) -> dict[str, Any]:
        """Registers the peaks of a molecule based on the retention time tolerance and wavelength.

        Args:
            molecule (Molecule): The molecule for which the peaks should be registered.
            ret_tolerance (float): Retention time tolerance for peak annotation in minutes.
            wavelength (float | None): Wavelength of the detector on which the molecule was detected.
            silent (bool): If True, doesn't print output immediately (for consolidated reporting).

        Returns:
            dict: Assignment results containing assigned_peak_count, multiple_peaks_info, and no_peaks_info.
        """
        assigned_peak_count = 0
        measurements_with_multiple_peaks = []
        measurements_with_no_peaks = []

        if not molecule.has_retention_time:
            raise ValueError(f"Molecule {molecule.id} has no retention time.")

        for meas in self.measurements:
            chrom = _resolve_chromatogram(meas.chromatograms, wavelength)

            # Find all peaks within tolerance
            candidate_peaks = []
            for peak in chrom.peaks:
                if (
                    peak.retention_time is not None
                    and molecule.retention_time is not None
                    and abs(peak.retention_time - molecule.retention_time)
                    <= ret_tolerance
                ):
                    # Check if min_signal criterion is met
                    if peak.area >= molecule.min_signal:
                        candidate_peaks.append(peak)

            if len(candidate_peaks) == 0:
                # No peaks found in this measurement
                measurements_with_no_peaks.append(meas.id)

            elif len(candidate_peaks) == 1:
                # Exactly one peak found - assign it
                peak = candidate_peaks[0]
                peak.molecule_id = molecule.id
                assigned_peak_count += 1
                logger.debug(
                    f"{molecule.id} assigned as molecule ID for peak at {peak.retention_time} in measurement {meas.id}."
                )

            else:
                # Multiple peaks found - assign the closest one and warn
                closest_peak = min(
                    candidate_peaks,
                    key=lambda p: abs(p.retention_time - molecule.retention_time)
                    if p.retention_time and molecule.retention_time
                    else float("inf"),
                )
                closest_peak.molecule_id = molecule.id
                assigned_peak_count += 1

                measurements_with_multiple_peaks.append(
                    {
                        "measurement_id": meas.id,
                        "num_peaks": len(candidate_peaks),
                        "assigned_rt": closest_peak.retention_time,
                        "all_rts": [p.retention_time for p in candidate_peaks],
                    }
                )

                logger.debug(
                    f"{molecule.id} assigned to closest peak at {closest_peak.retention_time} in measurement {meas.id}."
                )

        # Prepare return data
        assignment_result = {
            "molecule": molecule,
            "assigned_peak_count": assigned_peak_count,
            "measurements_with_multiple_peaks": measurements_with_multiple_peaks,
            "measurements_with_no_peaks": measurements_with_no_peaks,
            "retention_tolerance": ret_tolerance,
        }

        # Print summary if not silent (for backward compatibility)
        if not silent:
            self._print_peak_assignment_summary(
                molecule,
                assigned_peak_count,
                measurements_with_multiple_peaks,
                measurements_with_no_peaks,
                ret_tolerance,
            )

        return assignment_result

    def assign_all_peaks(self, silent_individual: bool = True) -> None:
        """Assign peaks for all molecules and display a consolidated report.

        Args:
            silent_individual (bool): If True, suppress individual molecule output (default: True).
        """
        if not self.molecules:
            print("📊 No molecules defined for peak assignment.")
            return

        assignment_results = []

        # Process all molecules
        for molecule in self.molecules:
            if molecule.has_retention_time:
                result = self._register_peaks(
                    molecule,
                    molecule.retention_tolerance,
                    molecule.wavelength,
                    silent=silent_individual,
                )
                assignment_results.append(result)

        # Display consolidated report
        if assignment_results:
            self._display_consolidated_assignment_report(assignment_results)

    def _display_consolidated_assignment_report(
        self, assignment_results: list[dict[str, Any]]
    ) -> None:
        """Display a consolidated peak assignment report for all molecules."""
        pretty.display_consolidated_assignment_report(self, assignment_results)

    def _print_peak_assignment_summary(
        self,
        molecule: Molecule,
        assigned_peak_count: int,
        measurements_with_multiple_peaks: list[dict[str, Any]],
        measurements_with_no_peaks: list[str],
        ret_tolerance: float,
    ) -> None:
        """Print a formatted summary of peak assignment results."""
        pretty.print_peak_assignment_summary(
            self,
            molecule,
            assigned_peak_count,
            measurements_with_multiple_peaks,
            measurements_with_no_peaks,
            ret_tolerance,
        )

    def define_protein(
        self,
        id: str,
        name: str,
        init_conc: float,
        conc_unit: str,
        sequence: str | None = None,
        organism: str | None = None,
        organism_tax_id: str | None = None,
        constant: bool = True,
    ) -> None:
        """Adds a protein to the list of proteins or updates an existing protein
        based on the pubchem_cid of the molecule.

        Args:
            id (str): Internal identifier of the protein such as `p0` or `asd45`.
            name (str): Name of the protein.
            init_conc (float): Initial concentration of the protein.
            conc_unit (str): Unit of the concentration.
            sequence (str, optional): Amino acid sequence of the protein. Defaults to None.
            organism (str, optional): Name of the organism. Defaults to None.
            organism_tax_id (int, optional): NCBI taxonomy ID of the organism. Defaults to None.
            constant (bool, optional): Boolean indicating whether the protein concentration is constant. Defaults to True.
        """
        protein = Protein(
            id=id,
            name=name,
            init_conc=init_conc,
            conc_unit=conc_unit,
            sequence=sequence,
            organism=organism,
            organism_tax_id=organism_tax_id,
            constant=constant,
        )

        self._update_protein(protein)

    def add_protein(
        self,
        protein: Protein,
        init_conc: float | None = None,
        conc_unit: UnitDefinitionAnnot | None = None,
    ) -> None:
        """Adds a protein to the list of proteins or updates an existing protein
        based on the pubchem_cid of the molecule.

        Args:
            protein (Protein): The protein object to be added.
        """

        nu_prot = copy.deepcopy(protein)

        if init_conc:
            nu_prot.init_conc = init_conc

        if conc_unit:
            nu_prot.conc_unit = conc_unit

        self._update_protein(nu_prot)

    def set_dilution_factor(self, dilution_factor: float) -> None:
        """Sets the dilution factor for all measurements."""

        if not isinstance(dilution_factor, float | int):
            raise ValueError("Dilution factor must be a float or integer.")

        for meas in self.measurements:
            meas.dilution_factor = dilution_factor

    @classmethod
    def read_asm(
        cls,
        path: str | Path,
        ph: float,
        temperature: float,
        temperature_unit: UnitDefinitionAnnot = "Celsius",
        mode: Optional[Literal["timecourse", "calibration"]] = None,
        values: Optional[list[float]] = None,
        unit: Optional[UnitDefinitionAnnot] = None,
        id: str | None = None,
        name: str = "Chromatographic measurement",
        silent: bool = False,
    ) -> Handler:
        """Reads chromatographic data from a directory containing Allotrope Simple Model (ASM) json files.
        Measurements are assumed to be named alphabetically, allowing sorting by file name.

        Args:
            path (str | Path): Path to the directory containing the ASM files.
            ph (float): pH value of the measurement.
            temperature (float): Temperature of the measurement.
            temperature_unit (UnitDefinitionAnnot, optional): Unit of the temperature. Defaults to Celsius (C).
            mode (Optional[Literal["timecourse", "calibration"]], optional): Mode of the data. If "timecourse",
                `values` should be a list of reaction times. If "calibration", `values` should be a list of concentrations.
                Defaults to None.
            values (list[float], optional): A list of reaction times (for "timecourse" mode) or concentrations
                (for "calibration" mode), corresponding to each measurement in the directory.
            unit (UnitDefinitionAnnot, optional): Unit of the `values` provided. It can be the time unit for reaction times or
                the concentration unit for calibration mode, depending on the mode.
            id (str, optional): Unique identifier of the Handler object. If not provided, the `path` is used as ID.
            name (str, optional): Name of the measurement. Defaults to "Chromatographic measurement".
            silent (bool, optional): If True, no success message is printed. Defaults to False.

        Returns:
            Handler: Handler object containing the measurements.
        """
        from .readers.asm import ASMReader

        data = {
            "dirpath": str(path),
            "values": values,
            "unit": unit,
            "ph": ph,
            "temperature": temperature,
            "temperature_unit": temperature_unit,
            "silent": silent,
            "mode": mode,
        }

        reader = ASMReader(**data)
        measurements = reader.read()

        if id is None:
            id = Path(path).name

        return cls(id=id, name=name, measurements=measurements, mode=reader.mode)

    @classmethod
    def read_shimadzu(
        cls,
        path: str | Path,
        ph: float,
        temperature: float,
        temperature_unit: UnitDefinitionAnnot = "Celsius",
        mode: Optional[Literal["timecourse", "calibration"]] = None,
        values: Optional[list[float]] = None,
        unit: Optional[UnitDefinitionAnnot] = None,
        id: str | None = None,
        name: str = "Chromatographic measurement",
        silent: bool = False,
    ) -> Handler:
        """Reads chromatographic data from a directory containing Shimadzu files.
        Measurements are assumed to be named alphabetically, allowing sorting by file name.

        Args:
            path (str | Path): Path to the directory containing the Shimadzu files.
            ph (float): pH value of the measurement.
            temperature (float): Temperature of the measurement.
            temperature_unit (UnitDefinitionAnnot, optional): Unit of the temperature. Defaults to Celsius (C).
            mode (Optional[Literal["timecourse", "calibration"]], optional): Mode of the data. If "timecourse",
                `values` should be a list of reaction times. If "calibration", `values` should be a list of concentrations.
                Defaults to None.
            values (list[float], optional): A list of reaction times (for "timecourse" mode) or concentrations
                (for "calibration" mode), corresponding to each measurement in the directory.
            unit (UnitDefinitionAnnot, optional): Unit of the `values` provided. It can be the time unit for reaction times or
                the concentration unit for calibration mode, depending on the mode.
            id (str, optional): Unique identifier of the Handler object. If not provided, the `path` is used as ID.
            name (str, optional): Name of the measurement. Defaults to "Chromatographic measurement".
            silent (bool, optional): If True, no success message is printed. Defaults to False.

        Returns:
            Handler: Handler object containing the measurements.
        """
        from .readers.shimadzu import ShimadzuReader

        data = {
            "dirpath": str(path),
            "values": values,
            "unit": unit,
            "ph": ph,
            "temperature": temperature,
            "temperature_unit": temperature_unit,
            "silent": silent,
            "mode": mode,
        }

        reader = ShimadzuReader(**data)
        measurements = reader.read()

        if id is None:
            id = Path(path).name

        return cls(
            id=id,
            name=name,
            measurements=measurements,
            mode=reader.mode,
        )

    @classmethod
    def read_agilent(
        cls,
        path: str | Path,
        ph: float,
        temperature: float,
        temperature_unit: UnitDefinitionAnnot = "Celsius",
        mode: Optional[Literal["timecourse", "calibration"]] = None,
        values: Optional[list[float]] = None,
        unit: Optional[UnitDefinitionAnnot] = None,
        id: str | None = None,
        name: str = "Chromatographic measurement",
        silent: bool = False,
    ) -> Handler:
        """Reads Agilent `Report.txt` or `RESULTS.csv` files within a `*.D` directories within the specified path.

        Args:
            path (str | Path): Path to the directory containing the Agilent files.
            ph (float): pH value of the measurement.
            temperature (float): Temperature of the measurement.
            temperature_unit (UnitDefinitionAnnot, optional): Unit of the temperature. Defaults to Celsius (C).
            mode (Optional[Literal["timecourse", "calibration"]], optional): Mode of the data. If "timecourse",
                `values` should be a list of reaction times. If "calibration", `values` should be a list of concentrations.
                Defaults to None.
            values (list[float], optional): A list of reaction times (for "timecourse" mode) or concentrations
                (for "calibration" mode), corresponding to each measurement in the directory.
            unit (UnitDefinitionAnnot, optional): Unit of the `values` provided. It can be the time unit for reaction times or
                the concentration unit for calibration mode, depending on the mode.
            id (str, optional): Unique identifier of the Handler object. If not provided, the `path` is used as ID.
            name (str, optional): Name of the measurement. Defaults to "Chromatographic measurement".
            silent (bool, optional): If True, no success message is printed. Defaults to False.

        Returns:
            Handler: Handler object containing the measurements.
        """
        from .readers.agilent_csv import AgilentCSVReader
        from .readers.agilent_rdl import AgilentRDLReader
        from .readers.agilent_txt import AgilentTXTReader

        directory = Path(path)

        txt_paths = []
        csv_paths = []
        rdl_paths: list[str] = []

        txt_paths = [
            str(f.absolute())
            for f in directory.rglob("Report.TXT")
            if f.parent.parent == directory
        ]
        csv_paths = [
            str(f.absolute())
            for f in directory.rglob("RESULTS.CSV")
            if f.parent.parent == directory
        ]
        rdl_paths = []

        try:
            txt_path = next(directory.rglob("*.txt"))
            try:
                lines = AgilentRDLReader.read_file(str(txt_path))
                if lines[0].startswith("┌─────"):
                    rdl_paths = [str(f.absolute()) for f in directory.rglob("*.txt")]
                else:
                    txt_paths = txt_paths
            except UnicodeDecodeError:
                txt_paths = txt_paths

        except StopIteration:
            txt_paths = txt_paths

        data = {
            "dirpath": str(path),
            "values": values,
            "unit": unit,
            "ph": ph,
            "temperature": temperature,
            "temperature_unit": temperature_unit,
            "silent": silent,
            "mode": mode,
        }

        if rdl_paths:
            data["file_paths"] = rdl_paths
            reader = AgilentRDLReader(**data)
            measurements = reader.read()
        elif not csv_paths and txt_paths:
            data["file_paths"] = txt_paths
            reader = AgilentTXTReader(**data)  # type: ignore
            measurements = reader.read()
        elif csv_paths and not txt_paths:
            data["file_paths"] = csv_paths
            reader = AgilentCSVReader(**data)  # type: ignore
            measurements = reader.read()
        else:
            raise IOError(f"No 'REPORT.TXT' or 'RESULTS.CSV' files found in '{path}'.")

        if id is None:
            id = Path(path).name

        return cls(id=id, name=name, measurements=measurements, mode=reader.mode)

    @classmethod
    def read_chromeleon(
        cls,
        path: str | Path,
        ph: float,
        temperature: float,
        temperature_unit: UnitDefinitionAnnot = "Celsius",
        mode: Optional[Literal["timecourse", "calibration"]] = None,
        values: Optional[list[float]] = None,
        unit: Optional[UnitDefinitionAnnot] = None,
        id: str | None = None,
        name: str = "Chromatographic measurement",
        silent: bool = False,
    ) -> Handler:
        """Reads Chromeleon txt files from a directory. The files in the directory are assumed to be of
        one calibration or timecourse measurement series.

        Args:
            path (str | Path): Path to the directory containing the Chromeleon files.
            ph (float): pH value of the measurement.
            temperature (float): Temperature of the measurement.
            temperature_unit (UnitDefinitionAnnot, optional): Unit of the temperature. Defaults to Celsius (C).
            mode (Optional[Literal["timecourse", "calibration"]], optional): Mode of the data. If "timecourse",
                `values` should be a list of reaction times. If "calibration", `values` should be a list of concentrations.
                Defaults to None.
            values (list[float], optional): A list of reaction times (for "timecourse" mode) or concentrations
                (for "calibration" mode), corresponding to each measurement in the directory.
            unit (UnitDefinitionAnnot, optional): Unit of the `values` provided. It can be the time unit for reaction times or
                the concentration unit for calibration mode, depending on the mode.
            id (str, optional): Unique identifier of the Handler object. If not provided, the `path` is used as ID.
            name (str, optional): Name of the measurement. Defaults to "Chromatographic measurement".
            silent (bool, optional): If True, no success message is printed. Defaults to False.

        Returns:
            Handler: Handler object containing the measurements.
        """
        from .readers.chromeleon import ChromeleonReader

        data = {
            "dirpath": str(path),
            "values": values,
            "unit": unit,
            "ph": ph,
            "temperature": temperature,
            "temperature_unit": temperature_unit,
            "silent": silent,
            "mode": mode,
        }

        if id is None:
            id = Path(path).name

        reader = ChromeleonReader(**data)
        measurements = reader.read()

        return cls(id=id, name=name, measurements=measurements, mode=reader.mode)

    @classmethod
    def read_thermo(
        cls,
        path: str | Path,
        ph: float,
        temperature: float,
        temperature_unit: UnitDefinitionAnnot = "Celsius",
        mode: Optional[Literal["timecourse", "calibration"]] = None,
        values: Optional[list[float]] = None,
        unit: Optional[UnitDefinitionAnnot] = None,
        id: str | None = None,
        name: str = "Chromatographic measurement",
        silent: bool = False,
    ) -> Handler:
        """Reads chromatographic data from a directory containing Thermo Scientific TX0 files.
        Measurements are assumed to be named alphabetically, allowing sorting by file name.

        Args:
            path (str | Path): Path to the directory containing the TX0 files.
            ph (float): pH value of the measurement.
            temperature (float): Temperature of the measurement.
            temperature_unit (UnitDefinitionAnnot, optional): Unit of the temperature. Defaults to Celsius (C).
            mode (Optional[Literal["timecourse", "calibration"]], optional): Mode of the data. If "timecourse",
                `values` should be a list of reaction times. If "calibration", `values` should be a list of concentrations.
                Defaults to None.
            values (list[float], optional): A list of reaction times (for "timecourse" mode) or concentrations
                (for "calibration" mode), corresponding to each measurement in the directory.
            unit (UnitDefinitionAnnot, optional): Unit of the `values` provided. It can be the time unit for reaction times or
                the concentration unit for calibration mode, depending on the mode.
            id (str, optional): Unique identifier of the Handler object. If not provided, the `path` is used as ID.
            name (str, optional): Name of the measurement. Defaults to "Chromatographic measurement".
            silent (bool, optional): If True, no success message is printed. Defaults to False.

        Returns:
            Handler: Handler object containing the measurements.
        """
        from .readers.thermo_txt import ThermoTX0Reader

        data = {
            "dirpath": str(path),
            "values": values,
            "unit": unit,
            "ph": ph,
            "temperature": temperature,
            "temperature_unit": temperature_unit,
            "silent": silent,
            "mode": mode,
        }

        if id is None:
            id = Path(path).name

        reader = ThermoTX0Reader(**data)
        measurements = reader.read()

        return cls(id=id, name=name, measurements=measurements, mode=reader.mode)

    def to_json(self, path: str | Path) -> None:
        """
        Serialize the instance to a JSON file.
        Parameters:
            path (str | Path): The file path where the JSON data will be saved.
                                If the parent directory does not exist, it will be created.
        Returns:
            None: This method does not return a value. It writes the instance's
            attributes to a JSON file at the specified path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize the instance to JSON, allowing overwriting
        with open(path, "w") as file:
            file.write(self.model_dump_json(indent=2))

    @classmethod
    def read_csv(
        cls,
        path: str | Path,
        mode: Literal["timecourse", "calibration"],
        ph: float,
        temperature: float,
        temperature_unit: UnitDefinitionAnnot,
        retention_time_col_name: str,
        peak_area_col_name: str,
        id: str | None = None,
        values: Optional[list[float]] = None,
        unit: Optional[UnitDefinitionAnnot] = None,
        silent: bool = False,
        header: int | None = 0,
    ) -> Handler:
        """Reads chromatographic data from a CSV file.

        Args:
            path (str | Path): Path to the CSV file.
            mode (Literal["timecourse", "calibration"]): Mode of the data.
            values (Optional[list[float]]): List of values. Defaults to None.
            unit (Optional[UnitDefinitionAnnot]): Unit of the values. Defaults to None.
            ph (float): pH value of the measurement.
            temperature (float): Temperature of the measurement.
            temperature_unit (UnitDefinitionAnnot): Unit of the temperature.
            retention_time_col_name (str): Name of the retention time column.
            peak_area_col_name (str): Name of the peak area column.
            id (str, optional): Unique identifier of the Handler object. If not provided, the `path` is used as ID.
            silent (bool, optional): If True, no success message is printed. Defaults to False.
            header (int | None, optional): Header row of the CSV file. Defaults to 0.

        Returns:
            Handler: Handler object containing the measurements.
        """
        from .readers.generic_csv import GenericCSVReader

        if id is None:
            id = Path(path).name

        reader = GenericCSVReader(
            dirpath=str(path),
            mode=mode,
            values=values,
            unit=unit,
            ph=ph,
            temperature=temperature,
            temperature_unit=temperature_unit,
            silent=silent,
        )
        measurements = reader.read_generic_csv(
            retention_time_col_name=retention_time_col_name,
            peak_area_col_name=peak_area_col_name,
            header=header,
        )
        return cls(id=id, name=str(path), measurements=measurements, mode=reader.mode)

    @classmethod
    def from_json(cls, path: str | Path) -> Handler:
        """
        Load an instance of the class from a JSON file.

        Args:
            path (str | Path): The file path to the JSON file.

        Returns:
            An instance of the class populated with data from the JSON file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        # Load from a JSON file
        with open(path, "r") as file:
            data = json.load(file)

        # Return an instance of the class
        return cls(**data)

    def to_enzymeml(
        self,
        name: str,
        calculate_concentration: bool = True,
        extrapolate: bool = False,
    ) -> EnzymeMLDocument:
        """Creates an EnzymeML document from the data in the Handler.

        Args:
            name (str): Name of the EnzymeML document.
            calculate_concentration (bool, optional): If True, the concentrations of the species
                are calculated. Defaults to True.
            extrapolate (bool, optional): If True, the concentrations are extrapolated to if the
                measured peak areas are outside the calibration range. Defaults to False.

        Returns:
            EnzymeMLDocument: _description_
        """
        from .enzymeml import create_enzymeml

        warnings.warn(
            "The to_enzymeml method is deprecated and will be removed in version 1.0.0. Use chromhandler.to_enzymeml instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return create_enzymeml(
            document_name=name,
            molecules=self.molecules,
            proteins=self.proteins,
            measurement_id=self.id,
            measurements=self.measurements,
            calculate_concentration=calculate_concentration,
            internal_standards={self.internal_standard.id: self.internal_standard}  # type: ignore
            if self.internal_standard
            else None,
            extrapolate=extrapolate,
        )

    def get_molecule(self, molecule_id: str) -> Molecule:
        for molecule in self.molecules:
            if molecule.id == molecule_id:
                return molecule

        raise ValueError(f"Molecule with ID {molecule_id} not found.")

    def add_standard(
        self,
        molecule: Molecule,
        wavelength: float | None = None,
        visualize: bool = True,
    ) -> None:
        """Creates a standard curve for a molecule based on the peak areas and concentrations.

        Args:
            molecule (Molecule): The molecule for which the standard curve should be created.
            wavelength (float | None, optional): The wavelength of the detector. Defaults to None.
            visualize (bool, optional): If True, the standard curve is visualized. Defaults to True.
        """
        assert any(
            [molecule in [mol for mol in self.molecules]]
        ), "Molecule not found in molecules of handler."

        # check if all measurements only contain one chromatogram
        if all([len(meas.chromatograms) == 1 for meas in self.measurements]):
            chroms = [
                chrom for meas in self.measurements for chrom in meas.chromatograms
            ]
        else:
            assert (
                wavelength is not None
            ), "Multiple chromatograms found for each measurment, wavelength needs to be provided."

            chroms = self._get_chromatograms_by_wavelegnth(wavelength)

            assert (
                len(chroms) > 0
            ), "No chromatograms found at the specified wavelength."

        peak_areas = [
            peak.area for chrom in chroms for peak in chrom.peaks if peak.molecule_id
        ]

        concs = [meas.data.value for meas in self.measurements]
        conc_unit = self.measurements[0].data.unit

        assert (
            len(peak_areas) == len(concs)
        ), f"Number of {molecule.name} peak areas {len(peak_areas)} and concentrations {len(concs)} do not match."

        assert all(
            meas.ph == self.measurements[0].ph for meas in self.measurements
        ), "All measurements need to have the same pH value."
        ph = self.measurements[0].ph

        assert all(
            meas.temperature == self.measurements[0].temperature
            for meas in self.measurements
        ), "All measurements need to have the same temperature value."
        temperature = self.measurements[0].temperature

        assert all(
            meas.temperature_unit.name == self.measurements[0].temperature_unit.name
            for meas in self.measurements
        ), "All measurements need to have the same temperature unit."
        temperature_unit = self.measurements[0].temperature_unit

        molecule.create_standard(
            areas=peak_areas,
            concs=concs,
            conc_unit=conc_unit.name,
            ph=ph,
            temperature=temperature,
            temp_unit=temperature_unit.name,
            visualize=visualize,
        )

    def _get_chromatograms_by_wavelegnth(self, wavelength: float) -> list[Chromatogram]:
        """Returns a list of chromatograms at a specified wavelength.

        Args:
            wavelength (float): The wavelength of the detector.

        Returns:
            list[Chromatogram]: A list of chromatograms at the specified wavelength.
        """

        chroms = []
        for meas in self.measurements:
            for chrom in meas.chromatograms:
                if chrom.wavelength == wavelength:
                    chroms.append(chrom)

        return chroms

    def _update_molecule(self, molecule: Molecule) -> None:
        """Updates the molecule if it already exists in the list of species.
        Otherwise, the molecule is added to the list of species."""
        for idx, mol in enumerate(self.molecules):
            if mol.id == molecule.id:
                self.molecules[idx] = molecule
                assert self.molecules[idx] is molecule
                return

        self.molecules.append(molecule)

    def _update_protein(self, protein: Protein) -> None:
        """Updates the protein if it already exists in the list of proteins.
        Otherwise, the protein is added to the list of proteins.
        """
        for idx, prot in enumerate(self.proteins):
            if prot.id == protein.id:
                self.proteins[idx] = protein
                return

        self.proteins.append(protein)

    def visualize(
        self,
        n_cols: int = 2,
        figsize: tuple[float, float] = (15, 10),
        show_peaks: bool = True,
        show_processed: bool = False,
        rt_min: float | None = None,
        rt_max: float | None = None,
        save_path: str | None = None,
        assigned_only: bool = False,
        overlay: bool = False,
    ) -> None:
        """Creates a matplotlib figure with subplots for each measurement.

        Args:
            n_cols (int, optional): Number of columns in the subplot grid. Defaults to 2.
            figsize (tuple[float, float], optional): Figure size in inches (width, height). Defaults to (15, 10).
            show_peaks (bool, optional): If True, shows detected peaks. Defaults to True.
            show_processed (bool, optional): If True, shows processed signal. Defaults to False.
            rt_min (float | None, optional): Minimum retention time to display. If None, shows all data. Defaults to None.
            rt_max (float | None, optional): Maximum retention time to display. If None, shows all data. Defaults to None.
            save_path (str | None, optional): Path to save the figure. If None, the figure is not saved. Defaults to None.
            assigned_only (bool, optional): If True, only shows peaks that are assigned to a molecule. Defaults to False.
            overlay (bool, optional): If True, plots all chromatograms on a single axis. Defaults to False.
        """
        visualize.visualize(
            self,
            n_cols,
            figsize,
            show_peaks,
            show_processed,
            rt_min,
            rt_max,
            save_path,
            assigned_only,
            overlay,
        )

    def rich_display(self, console: Console | None = None, debug: bool = False) -> None:
        """
        Display a comprehensive rich text visualization of the Handler instance.

        This method provides a beautiful, structured overview of the Handler including:
        - Basic information (ID, name, mode)
        - Molecules and their properties
        - Proteins and their properties
        - Measurements summary with peak statistics
        - Chromatogram details

        Args:
            console (Console | None, optional): Rich console instance. If None, creates a new one. Defaults to None.
            debug (bool, optional): If True, shows debug information about what sections are being displayed. Defaults to False.
        """
        pretty.display_rich_handler(self, console, debug)

    def __rich__(self) -> Group:
        """
        Rich representation for automatic display in rich-aware contexts.

        This method is called automatically when you:
        - print(Handler) in a rich-enabled terminal
        - Display Handler in Jupyter notebooks
        - Use Handler in any rich-aware context

        Returns:
            Group: A rich group with the full Handler visualization.
        """
        return pretty.create_rich_handler_group(self)

    def __call__(self) -> None:
        """
        Make the Handler callable to display rich visualization.

        This allows you to use: Handler() to get the full rich display.
        """
        self.rich_display()
