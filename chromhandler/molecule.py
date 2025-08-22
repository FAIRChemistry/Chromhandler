from __future__ import annotations

import json
from typing import Optional

from calipytion.model import Calibration
from calipytion.tools.calibrator import Calibrator
from mdmodels.units.annotation import UnitDefinitionAnnot
from pydantic import BaseModel, ConfigDict, Field


class Molecule(BaseModel):
    model_config: ConfigDict = ConfigDict(  # type: ignore
        validate_assignment=True,
        use_enum_values=True,
    )

    id: str = Field(
        description="ID of the molecule",
    )
    pubchem_cid: int = Field(
        description="PubChem CID of the molecule",
    )
    name: str = Field(
        description="Name of the molecule",
    )
    init_conc: Optional[float] = Field(
        description="Initial concentration of the molecule at t=0",
        default=None,
    )
    conc_unit: Optional[UnitDefinitionAnnot] = Field(
        description="Unit of the concentration",
        default=None,
    )
    retention_time: Optional[float] = Field(
        description="Retention time of the molecule in minutes",
        default=None,
    )
    wavelength: Optional[float] = Field(
        description="Wavelength at which the molecule was detected",
        default=None,
    )
    standard: Optional[Calibration] = Field(
        description="Calibration instance associated with the molecule",
        default=None,
    )
    retention_tolerance: float = Field(
        description="Tolerance for the retention time of the molecule",
        default=0.1,
    )
    constant: bool = Field(
        description="Boolean indicating whether the molecule concentration is constant throughout the experiment",
        default=False,
    )
    internal_standard: bool = Field(
        description="Boolean indicating whether the molecule is an internal standard",
        default=False,
    )
    min_signal: float = Field(
        description="Minimum signal threshold for peak assignment. Peaks must have an area >= this value to be assigned to this molecule.",
        default=0.0,
    )

    @classmethod
    def from_standard(
        cls, standard: Calibration, init_conc: float, conc_unit: UnitDefinitionAnnot
    ) -> Molecule:
        """Creates a Molecule instance from a Calibration instance.

        Args:
            standard (Calibration): The calibration instance to create the molecule from.
            init_conc (float): The initial concentration of the molecule.
            conc_unit (UnitDefinition): The unit of the concentration.

        Returns:
            Molecule: The created Molecule instance.
        """

        assert standard.retention_time, """
        The retention time of the calibration needs to be defined. 
        Specify the `retention_time` attribute of the calibration.
        """

        return cls(
            id=standard.molecule_id,
            pubchem_cid=standard.pubchem_cid,
            name=standard.molecule_name,
            init_conc=init_conc,
            conc_unit=conc_unit,
            retention_time=standard.retention_time,
            standard=standard,
        )

    def create_standard(
        self,
        areas: list[float],
        concs: list[float],
        conc_unit: str,
        ph: float,
        temperature: float,
        temp_unit: UnitDefinitionAnnot = "Celsius",
        visualize: bool = True,
    ) -> Calibration:
        """Creates a linear standard from the molecule's calibration data.

        Args:
            areas (list[float]): The areas of the molecule.
            concs (list[float]): The concentrations of the molecule.
            conc_unit (str): The unit of the concentration.
            ph (float): The pH of the solution.
            temperature (float): The temperature of the solution.
            temp_unit (UnitDefinition): The unit of the temperature.
            visualize (bool): Whether to visualize the standard.

        Returns:
            Calibration: The created Calibration instance.
        """

        calibrator = Calibrator(
            molecule_id=self.id,
            pubchem_cid=self.pubchem_cid,
            molecule_name=self.name,
            wavelength=self.wavelength,
            concentrations=concs,
            conc_unit=conc_unit,
            signals=areas,
        )
        calibrator.models = []
        model = calibrator.add_model(
            name="linear",
            signal_law=f"{self.id} * a",
            lower_bound=0.001,
            upper_bound=1e10,
        )

        calibrator.fit_models()
        model.calibration_range.conc_lower = 0.0
        model.calibration_range.signal_lower = 0.0

        if visualize:
            calibrator.visualize_static()

        standard = calibrator.create_standard(
            model=model,
            ph=ph,
            temperature=temperature,
            temp_unit=temp_unit,
        )

        # check if the `conc` attribute of the molecule is defined and if, it must have the same baseunit names as the calibration unit
        if self.conc_unit:
            assert self.conc_unit.name == conc_unit, """
            The concentration unit of the molecule does not match the calibration unit defined in its standard.
            """
        else:
            self.conc_unit = conc_unit

        self.standard = standard

        return standard

    @classmethod
    def read_json(cls, path: str) -> Molecule:
        """Creates a Molecule instance from a JSON file.

        Args:
            path (str): The path to the JSON file.

        Returns:
            Molecule: The created Molecule instance.
        """

        with open(path, "r") as f:
            data = json.load(f)

        return cls(**data)

    def save_json(self, path: str) -> None:
        """Saves the Molecule instance to a JSON file.

        Args:
            path (str): The path to the JSON file.

        Returns:
            None
        """

        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=4))

    @property
    def has_retention_time(self) -> bool:
        """
        Checks if the molecule has a retention time defined. And if so,
        it is assumed that the molecule is present in the chromatogram.
        """
        return self.retention_time is not None
