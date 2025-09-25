import warnings
from enum import Enum

from calipytion.tools.calibrator import Calibrator
from mdmodels.units.annotation import UnitDefinition
from pyenzyme import (
    DataTypes,
    EnzymeMLDocument,
    MeasurementData,
    Protein,
    SmallMolecule,
)
from pyenzyme import Measurement as EnzymeMLMeasurement

from .handler import Handler
from .internal_standard import InternalStandard
from .model import Chromatogram, Measurement
from .molecule import Molecule
from .protein import Protein as ChromProtein


class CalibratorType(Enum):
    EXTERNAL = "external_standard"
    INTERNAL = "internal_standard"
    NONE = "none"


def to_enzymeml(
    document_name: str,
    handlers: list[Handler],
    calculate_concentration: bool,
    extrapolate: bool = False,
    internal_standard: bool = False,
) -> EnzymeMLDocument:
    """Converts a list of Handler instances to an EnzymeMLDocument instance.

    Args:
        document_name (str): Name of the EnzymeMLDocument instance.
        handlers (list[Handler]): List of Handler objects containing measurement data.
        calculate_concentration (bool): If True, the concentration of the molecules will be calculated.
        extrapolate (bool): If True, the concentration of the molecules will be extrapolated. Default is False.
        internal_standard (bool): If True, the internal standard will be used to calculate the concentration of the molecules. Default is False.

    Returns:
        EnzymeMLDocument: The created EnzymeML document with all measurements added.
    """

    if not isinstance(handlers, list):
        handlers = [handlers]

    for idx, handler in enumerate(handlers):
        if idx == 0:
            if internal_standard:
                internal_std_molecules = [
                    mol for mol in handler.molecules if mol.internal_standard
                ]
                if len(internal_std_molecules) != 1:
                    raise ValueError(
                        f"Exaclty one internal standard molecule needs to be defined. Currently {len(internal_std_molecules)} internal standard molecules are defined."
                    )

                # setup internal calibrator
                internal_standards = setup_internal_calibrators(
                    internal_standard=internal_std_molecules[0],
                    molecules=[
                        mol for mol in handler.molecules if not mol.internal_standard
                    ],
                    first_measurement=handler.measurements[0],
                )
            else:
                internal_standards = None

            doc = create_enzymeml(
                document_name=document_name,
                molecules=handler.molecules,
                proteins=handler.proteins,
                measurements=handler.measurements,
                measurement_id=handler.id,
                internal_standards=internal_standards,
                calculate_concentration=calculate_concentration,
                extrapolate=extrapolate,
            )
        else:
            if internal_standard:
                internal_std_molecules = [
                    mol for mol in handler.molecules if mol.internal_standard
                ]
                if len(internal_std_molecules) != 1:
                    raise ValueError(
                        f"Exaclty one internal standard molecule needs to be defined. Currently {len(internal_std_molecules)} internal standard molecules are defined."
                    )

                # setup internal calibrator
                internal_standards = setup_internal_calibrators(
                    internal_standard=internal_std_molecules[0],
                    molecules=[
                        mol for mol in handler.molecules if not mol.internal_standard
                    ],
                    first_measurement=handler.measurements[0],
                )
            else:
                internal_standards = None

            add_measurements_to_enzymeml(
                doc=doc,
                new_measurements=handler.measurements,
                measurement_id=handler.id,
                molecules=handler.molecules,
                proteins=handler.proteins,
                calculate_concentration=calculate_concentration,
                extrapolate=extrapolate,
                internal_standards=internal_standards,
            )
    return doc


def create_enzymeml(
    document_name: str,
    molecules: list[Molecule],
    proteins: list[ChromProtein],
    measurements: list[Measurement],
    measurement_id: str,
    calculate_concentration: bool,
    extrapolate: bool,
    internal_standards: dict[str, InternalStandard] | None = None,
) -> EnzymeMLDocument:
    """Creates an EnzymeMLDocument instance from a list of Molecule and Measurement instances.

    Args:
        document_name (str): Name of the EnzymeMLDocument instance.
        molecules (list[Molecule]): List of Molecule instances.
        proteins (list[Protein]): List of Protein instances.
        measurements (list[Measurement]): List of Measurement instances.
        measurement_id (str): ID of the measurement.
        calculate_concentration (bool): If True, the concentration of the molecules will be calculated,
            using the internal standard or defined standard.
        extrapolate (bool): If True, the concentration of the molecules will be extrapolated.

    Returns:
        EnzymeMLDocument: The EnzymeMLDocument instance.
    """

    doc = EnzymeMLDocument(name=document_name)
    meas_data: dict[str, MeasurementData] = {}

    for protein in proteins:
        add_protein(doc, protein)
        create_MeasurementData_instances(meas_data, protein)

    for molecule in molecules:
        if not molecule.internal_standard:
            add_molecule(doc, molecule)
            create_MeasurementData_instances(meas_data, molecule)

    # add data to MeasurementData instances
    measurement_data_instances = add_measurement_to_MeasurementData(
        measurements=measurements,
        measurement_data_instances=meas_data,
        calculate_concentration=calculate_concentration,
        molecules=molecules,
        internal_standards=internal_standards,
        extrapolate=extrapolate,
    )

    # create EnzymeML Measurement
    ph, temp, time_unit, temp_unit = extract_measurement_conditions(measurements)

    species_data = list(measurement_data_instances.values())
    for species in species_data:
        species.time_unit = time_unit.name

    enzml_measurement = EnzymeMLMeasurement(
        id=measurement_id,
        name=measurement_id,
        temperature=temp,
        temperature_unit=temp_unit.name,
        ph=ph,
        species_data=list(measurement_data_instances.values()),
    )

    doc.measurements.append(enzml_measurement)

    patch_init_t0(doc)

    return doc


def add_protein(doc: EnzymeMLDocument, protein: ChromProtein) -> None:
    """Adds Protein instance to an existing EnzymeMLDocument instance.

    Args:
        doc (EnzymeMLDocument): The existing EnzymeMLDocument instance.
        protein (Protein): Protein instance to be added.
    """

    protein_data = protein.model_dump()
    protein_data.pop("conc_unit")
    protein_data.pop("init_conc")

    doc.proteins.append(Protein(**protein_data))


def add_molecule(doc: EnzymeMLDocument, molecule: Molecule) -> None:
    """Adds a Molecule instance to an existing EnzymeMLDocument instance.

    Args:
        doc (EnzymeMLDocument): The existing EnzymeMLDocument instance.
        molecule (Molecule): Molecule instance to be added.

    Returns:
        EnzymeMLDocument: The updated EnzymeMLDocument instance.
    """

    pubchem_base_url = "https://pubchem.ncbi.nlm.nih.gov/compound/"

    mol_data = {
        "id": molecule.id,
        "name": molecule.name,
        "constant": molecule.constant,
    }

    if molecule.pubchem_cid > -1:
        mol_data["ld_id"] = f"{pubchem_base_url}{molecule.pubchem_cid}"

    doc.small_molecules.append(SmallMolecule(**mol_data))


def create_MeasurementData_instances(
    meas_data_dict: dict[str, MeasurementData],
    species: Molecule | ChromProtein,
) -> None:
    """Adds a MeasurementData for a Protein or Molecule instance to a dictionary
    of MeasurementData instances.

    Args:
        meas_data_dict (dict[str, MeasurementData]): Dictionary of MeasurementData instances.
        species (Molecule | ChromProtein): Molecule or Protein instance.

    Raises:
        ValueError: If a MeasurementData instance for the molecule or protein already exists.
        TypeError: If the species is neither a Molecule nor a Protein instance.
        AssertionError: If the concentration unit of the molecule / protein is not defined.
    """

    _check_molecule_conc_unit_and_init_conc(species)

    if species.id in meas_data_dict:
        raise ValueError(f"""
        A MeasurementData instance for molecule {species.name} already exists.
        """)

    if not species.conc_unit:
        raise ValueError(f"""
        The concentration unit of the molecule {species.name} needs to be defined.
        Please specify the `conc_unit` attribute of {species.name}.
        """)

    # Determine the data type based on the species type
    if isinstance(species, ChromProtein):
        data_type = DataTypes.CONCENTRATION
        # For proteins, we create an empty data array since they're not measured
        meas_data = MeasurementData(
            species_id=species.id,
            initial=species.init_conc,
            prepared=species.init_conc,
            data_unit=species.conc_unit.name,
            data_type=data_type,
        )
    elif isinstance(species, Molecule):
        data_type = DataTypes.PEAK_AREA
        meas_data = MeasurementData(
            species_id=species.id,
            initial=species.init_conc,
            prepared=species.init_conc,
            data_unit=species.conc_unit.name,
            data_type=data_type,
        )

    meas_data_dict[species.id] = meas_data


def add_measurements_to_enzymeml(
    doc: EnzymeMLDocument,
    new_measurements: list[Measurement],
    molecules: list[Molecule],
    proteins: list[ChromProtein],
    calculate_concentration: bool,
    extrapolate: bool,
    measurement_id: str,
    internal_standards: dict[str, InternalStandard] | None = None,
) -> EnzymeMLDocument:
    """
    Adds new measurements to an existing EnzymeMLDocument instance.

    Args:
        doc (EnzymeMLDocument): The existing EnzymeMLDocument instance.
        new_measurements (list[Measurement]): List of new Measurement instances to be added.
        molecules (list[Molecule]): List of Molecule instances.
        proteins (list[ChromProtein]): List of Protein instances.
        calculate_concentration (bool): If True, the concentration of the molecules will be calculated.
        extrapolate (bool): If True, the concentration of the molecules will be extrapolated.
        measurement_id (str): ID of the measurement.
        internal_standards (dict[str, InternalStandard], optional): Dictionary containing
            the species IDs as keys and InternalStandard instances as values.

    Returns:
        EnzymeMLDocument: The updated EnzymeMLDocument instance.
    """

    # Extract measurement conditions from the new measurements
    ph, temp, time_unit, temp_unit = extract_measurement_conditions(new_measurements)

    # Create MeasurementData instances for existing molecules
    measurement_data_instances: dict[str, MeasurementData] = {}
    for mol in molecules:
        if mol.internal_standard:
            continue
        if not mol.conc_unit:
            raise ValueError(
                f"Concentration unit is not defined for molecule {mol.name}."
            )

        measurement_data_instances[mol.id] = MeasurementData(
            species_id=mol.id,
            initial=mol.init_conc,
            prepared=mol.init_conc,
            data_unit=mol.conc_unit.name,
            data_type=DataTypes.PEAK_AREA,
            time_unit=time_unit.name,
        )

    # Add MeasurementData instances for proteins
    for protein in proteins:
        if not protein.conc_unit:
            raise ValueError(
                f"Concentration unit is not defined for protein {protein.name}."
            )
        if protein.id not in measurement_data_instances:
            measurement_data_instances[protein.id] = MeasurementData(
                species_id=protein.id,
                initial=protein.init_conc,
                prepared=protein.init_conc,
                data_unit=protein.conc_unit.name,
                data_type=DataTypes.CONCENTRATION,
                time_unit=time_unit.name,
            )

    # Convert new measurements to MeasurementData instances
    measurement_data_instances = add_measurement_to_MeasurementData(
        measurements=new_measurements,
        measurement_data_instances=measurement_data_instances,
        calculate_concentration=calculate_concentration,
        molecules=molecules,
        internal_standards=internal_standards,
        extrapolate=extrapolate,
    )

    # Create new EnzymeMLMeasurement and append to the document
    enzml_measurement = EnzymeMLMeasurement(
        id=measurement_id,
        name=measurement_id,
        temperature=temp,
        temperature_unit=temp_unit.name,
        ph=ph,
        species_data=list(measurement_data_instances.values()),
    )

    doc.measurements.append(enzml_measurement)

    patch_init_t0(doc)

    return doc


def add_measurement_to_MeasurementData(
    measurements: list[Measurement],
    measurement_data_instances: dict[str, MeasurementData],
    calculate_concentration: bool,
    molecules: list[Molecule],
    internal_standards: dict[str, InternalStandard] | None = None,
    extrapolate: bool = False,
) -> dict[str, MeasurementData]:
    """Converts a list of chromatographic Measurement instances to
    EnzymeML MeasurementData instances.
    This method handles both molecules (with chromatographic data) and proteins
    (with constant concentration values).

    Args:
        measurements (list[Measurement]): List of Measurement instances.
        measurement_data_instances (dict[str, MeasurementData]): Dictionary containing
            the species IDs as keys and MeasurementData instances as values.
        calculate_concentration (bool): If True, the concentration of the molecules will be calculated,
            using the internal standard or defined standard.
        molecules (list[Molecule]): List of Molecule instances.
        internal_standards (dict[str, InternalStandard], optional): Dictionary containing
            the species IDs as keys and InternalStandard instances as values.
        extrapolate (bool): If True, the concentration of the molecules will be extrapolated.

    Returns:
        dict[str, EnzymeMLMeasurement]: Dictionary containing the species IDs as keys
            and EnzymeMLMeasurement instances as values.
    """
    # Get molecules that have peaks
    molecule_ids = {
        molecule.id for molecule in molecules if not molecule.internal_standard
    }
    measured_once = get_measured_once(list(molecule_ids), measurements)

    # check if any molecule has an external standard
    has_external_standard = any([molecule.standard for molecule in molecules])

    # decide concentration calculation strategy for each molecule
    if calculate_concentration:
        if internal_standards and has_external_standard:
            raise ValueError(
                """
                Both internal and external standards are defined. Please choose one.
                """
            )
        elif has_external_standard:
            strategy = CalibratorType.EXTERNAL
            calibrators = setup_external_calibrators(molecules)
        elif internal_standards:
            strategy = CalibratorType.INTERNAL
            calibrators = internal_standards
        else:
            warnings.warn(
                "`calculate_concentration` is set to True, but no internal or external standards are defined."
            )
            strategy = CalibratorType.NONE
            calibrators = {}
            calculate_concentration = False
    else:
        strategy = CalibratorType.NONE
        calibrators = {}
        calculate_concentration = False

    # Process all species in measurement_data_instances
    for species_id, meas_data in measurement_data_instances.items():
        if species_id in measured_once:
            # Handle molecules with peaks
            for measurement in measurements:
                dilution_factor = (
                    measurement.dilution_factor
                    if measurement.dilution_factor is not None
                    else 1.0
                )

                for chrom in measurement.chromatograms:
                    add_data(
                        measurement_data=meas_data,
                        chromatogram=chrom,
                        reaction_time=measurement.data.value,
                        calibrators=calibrators,
                        calibrator_type=strategy,
                        extrapolate=extrapolate,
                        dilution_factor=dilution_factor,
                    )
        # Remove the else block that adds zeros for molecules without peaks
        # Molecules without peaks should have empty data arrays, not arrays filled with zeros

    # Update data_type for molecules without peaks when calculate_concentration=True
    if calculate_concentration:
        molecule_ids = {
            molecule.id for molecule in molecules if not molecule.internal_standard
        }
        for species_id, meas_data in measurement_data_instances.items():
            if species_id in molecule_ids and species_id not in measured_once:
                meas_data.data_type = DataTypes.CONCENTRATION

    return measurement_data_instances


def add_data(
    measurement_data: MeasurementData,
    chromatogram: Chromatogram,
    reaction_time: float,
    calibrators: dict[str, Calibrator | InternalStandard],
    calibrator_type: CalibratorType,
    extrapolate: bool,
    dilution_factor: float,
) -> None:
    peak = next(
        (
            peak
            for peak in chromatogram.peaks
            if peak.molecule_id == measurement_data.species_id
        ),
        None,
    )

    measurement_data.time.append(reaction_time)

    if calibrator_type == CalibratorType.EXTERNAL:
        calibrator = calibrators[measurement_data.species_id]
        assert isinstance(
            calibrator, Calibrator
        ), "Calibrator must be of type Calibrator."

        if peak is not None:
            conc = calibrator.calculate_concentrations(
                model=calibrator.standard.result,
                signals=[peak.area],
                extrapolate=extrapolate,
            )[0]
            # The dilution_factor will always be available with a default of 1
            conc *= dilution_factor

            measurement_data.data.append(conc)
            measurement_data.data_type = DataTypes.CONCENTRATION

        else:
            measurement_data.data.append(float(0))
            measurement_data.data_type = DataTypes.CONCENTRATION

    elif calibrator_type == CalibratorType.INTERNAL:
        calibrator = calibrators[measurement_data.species_id]
        assert isinstance(
            calibrator, InternalStandard
        ), "Calibrator must be of type InternalStandard."

        if peak is not None:
            internal_std_peak = next(
                (
                    peak
                    for peak in chromatogram.peaks
                    if peak.molecule_id == calibrator.standard_molecule_id
                ),
                None,
            )

            assert internal_std_peak is not None, f"""
                No peak for the internal standard molecule {calibrator.molecule_id}
                was detected in one of the chromatograms.
                """

            conc = calibrator.calculate_conc(peak.area, internal_std_peak.area)
            # The dilution_factor will always be available with a default of 1
            conc *= dilution_factor

            measurement_data.data.append(conc)
            measurement_data.data_type = DataTypes.CONCENTRATION

        else:
            measurement_data.data.append(float(0))
            measurement_data.data_type = DataTypes.CONCENTRATION

    elif calibrator_type == CalibratorType.NONE:
        assert calibrators == {}, "Calibrators must be empty."

        if peak is not None:
            measurement_data.data.append(peak.area)
            measurement_data.data_type = DataTypes.PEAK_AREA

        else:
            measurement_data.data.append(float(0))
            measurement_data.data_type = DataTypes.PEAK_AREA


def setup_external_calibrators(
    molecules: list[Molecule],
) -> dict[str, Calibrator]:
    """Creates calibrators for molecules with defined external standards.

    Args:
        molecules (list[Molecule]): List of Molecule instances.

    Returns:
        dict[str, Calibrator]: Dictionary containing molecule IDs as keys and
        Calibrator instances as values.

    Raises:
        AssertionError: If no calibrators were created.
    """

    calibrators: dict[str, Calibrator] = {}
    for molecule in molecules:
        if molecule.standard:
            calibrators[molecule.id] = Calibrator.from_standard(molecule.standard)

    assert (
        calibrators
    ), "No calibrators were created. Please define standards for the molecules."

    return calibrators


def setup_internal_calibrators(
    internal_standard: Molecule,
    molecules: list[Molecule],
    first_measurement: Measurement,
) -> dict[str, InternalStandard]:
    """Creates an internal calibrator for each measured molecule."""

    calibrators: dict[str, InternalStandard] = {}

    for molecule in molecules:
        if molecule.id == internal_standard.id:
            continue

        if internal_standard.init_conc is None:
            raise ValueError(f"""
            No initial concentration is defined for the internal standard molecule {internal_standard.name}.
            """)
        if internal_standard.conc_unit is None:
            raise ValueError(f"""
            No concentration unit is defined for the internal standard molecule {internal_standard.name}.
            """)
        if molecule.init_conc is None:
            raise ValueError(f"""
            No initial concentration is defined for molecule {molecule.name}.
            """)
        if molecule.conc_unit is None:
            raise ValueError(f"""
            No concentration unit is defined for molecule {molecule.name}.
            """)

        peak_analyte = next(
            (
                peak
                for peak in first_measurement.chromatograms[0].peaks
                if peak.molecule_id == molecule.id
            ),
            None,
        )

        peak_internal_standard = next(
            (
                peak
                for peak in first_measurement.chromatograms[0].peaks
                if peak.molecule_id == internal_standard.id
            ),
            None,
        )

        if peak_analyte and peak_internal_standard:
            calibrators[molecule.id] = InternalStandard(
                molecule_id=molecule.id,
                standard_molecule_id=internal_standard.id,
                molecule_init_conc=molecule.init_conc,
                standard_init_conc=internal_standard.init_conc,
                molecule_conc_unit=molecule.conc_unit.name,
                molecule_t0_signal=peak_analyte.area,
                standard_t0_signal=peak_internal_standard.area,
            )

    return calibrators


def extract_measurement_conditions(
    measurements: list[Measurement],
) -> tuple[float, float, UnitDefinition, UnitDefinition]:
    """Asserts and extracts the measurement conditions from a list of Measurement instances.

    Args:
        measurements (list[Measurement]): List of Measurement instances.

    Returns:
        tuple: A tuple containing the extracted measurement conditions
            (ph, temperature, time_unit, temperature_unit).
    """

    # extract measurement conditions
    phs = [measurement.ph for measurement in measurements]
    temperatures = [measurement.temperature for measurement in measurements]
    time_units = [measurement.data.unit.name for measurement in measurements]
    temperature_units = [
        measurement.temperature_unit.name for measurement in measurements
    ]

    assert len(set(phs)) == 1, "All measurements need to have the same pH."
    assert (
        len(set(temperatures)) == 1
    ), "All measurements need to have the same temperature."
    assert (
        len(set(time_units)) == 1
    ), "All measurements need to have the same time unit."
    assert (
        len(set(temperature_units)) == 1
    ), "All measurements need to have the same temperature unit."

    assert measurements[0].ph is not None, "The pH needs to be defined."
    assert (
        measurements[0].temperature is not None
    ), "The temperature needs to be defined."
    assert measurements[0].data.unit is not None, "The time unit needs to be defined."
    assert (
        measurements[0].temperature_unit is not None
    ), "The temperature unit needs to be defined."

    ph = measurements[0].ph
    temperature = measurements[0].temperature
    time_unit = measurements[0].data.unit
    temperature_unit = measurements[0].temperature_unit

    return ph, temperature, time_unit, temperature_unit


def get_measured_once(
    molecule_ids: list[str], measurements: list[Measurement]
) -> set[str]:
    """Checks if a molecule is assigned to a peak at least once in the measurements.

    Args:
        molecule_ids (list[str]): List of molecule IDs.
        measurements (list[Measurement]): List of Measurement instances.

    Returns:
        set[str]: Set containing the molecule IDs that are assigned to a peak at least once.
    """

    # Initialize the dictionary with False for each molecule ID
    return {
        peak.molecule_id
        for measurement in measurements
        for chrom in measurement.chromatograms
        for peak in chrom.peaks
        if peak.molecule_id is not None
    }


def _check_molecule_conc_unit_and_init_conc(molecule: Molecule | Protein) -> None:
    if molecule.init_conc is None:
        raise ValueError(f"""
        No initial concentration is defined for molecule {molecule.name}.
        Please specify the initial concentration of the molecule.
        """)

    if molecule.conc_unit is None:
        raise ValueError(f"""
            No concentration unit is defined for molecule {molecule.name}.
            Please specify the concentration unit or define a standard for the molecule.
            """)


def patch_init_t0(doc: EnzymeMLDocument) -> None:
    for meas in doc.measurements:
        for species_data in meas.species_data:
            if species_data.data:
                # species_data.prepared = species_data.data[0]
                if not species_data.initial == species_data.data[0]:
                    species_data.initial = species_data.data[0]
