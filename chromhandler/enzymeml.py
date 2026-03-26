"""Map chromhandler models to pyenzyme EnzymeML documents."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from pyenzyme import DataTypes, EnzymeMLDocument, MeasurementData, SmallMolecule
from pyenzyme import Measurement as EnzymeMLMeasurement
from pyenzyme import Protein as EnzymeMLProtein

if TYPE_CHECKING:
    from .model import Chromatogram, Sample
    from .molecule import Molecule
    from .protein import Protein


def _enzml_unit_to_str(unit: Any) -> str:
    """Normalize mdmodels :class:`~mdmodels.units.UnitDefinition` or strings for pyenzyme."""
    if unit is None:
        return ""
    if isinstance(unit, str):
        return unit
    uid = getattr(unit, "id", None)
    if isinstance(uid, str) and uid:
        return uid
    return str(unit)


def _enzml_filter_samples(
    samples: list[Sample],
    sample_ids: list[str] | None,
) -> list[Sample]:
    """Return the subset of *samples* matching *sample_ids* (or all if ``None``)."""
    if sample_ids is None:
        return list(samples)
    id_set = set(sample_ids)
    missing = id_set - {s.id for s in samples}
    if missing:
        raise ValueError(f"to_enzymeml: sample_ids not found in handler: {missing}")
    return [s for s in samples if s.id in id_set]


def _enzml_assert_calibrations(molecules: list[Molecule]) -> None:
    """Raise ``ValueError`` if any non-constant molecule is missing a calibration."""
    missing = [m.id for m in molecules if not m.constant and m.calibration is None]
    if missing:
        raise ValueError(
            f"to_enzymeml: to_concentration=True but the following molecules "
            f"have no calibration — run handler.calibrate_molecules() first: {missing}"
        )


def _enzml_add_species(doc: Any, molecules: list[Molecule], proteins: list[Protein]) -> None:
    """Populate *doc* with SmallMolecule and Protein entries."""

    pubchem_base = "https://pubchem.ncbi.nlm.nih.gov/compound/"
    for mol in molecules:
        kwargs: dict[str, Any] = {
            "id": mol.id,
            "name": mol.name,
            "constant": mol.constant,
        }
        if mol.pubchem_cid and mol.pubchem_cid > 0:
            kwargs["ld_id"] = f"{pubchem_base}{mol.pubchem_cid}"
        doc.small_molecules.append(SmallMolecule(**kwargs))

    for prot in proteins:
        doc.proteins.append(
            EnzymeMLProtein(
                id=prot.id,
                name=prot.name,
                constant=prot.constant,
                sequence=prot.sequence or "",
            )
        )


def _enzml_build_measurement(
    *,
    sample: Sample,
    molecules: list[Molecule],
    proteins: list[Protein],
    temperature: float,
    temperature_unit: str,
    ph: float,
    to_concentration: bool,
    extrapolate: bool,
    draw_idx: int | None,
) -> EnzymeMLMeasurement:
    """Build one pyenzyme ``Measurement`` for a single *sample* / *draw_idx* pair."""

    meas_id = sample.id if draw_idx is None else f"{sample.id}_draw{draw_idx:04d}"

    # Chromatograms sorted by reaction_time (skip those with no time assigned)
    filtered: list[Chromatogram] = [c for c in sample.chromatograms if c.reaction_time is not None]
    chroms: list[Chromatogram] = sorted(
        filtered,
        key=lambda c: c.reaction_time if c.reaction_time is not None else 0.0,
    )

    species_data: list[MeasurementData] = []

    for mol in molecules:
        md = _enzml_build_measurement_data(
            chroms=chroms,
            sample=sample,
            species=mol,
            to_concentration=to_concentration,
            extrapolate=extrapolate,
            draw_idx=draw_idx,
        )
        if md is not None:
            species_data.append(md)

    return EnzymeMLMeasurement(
        id=meas_id,
        name=meas_id,
        temperature=temperature,
        temperature_unit=temperature_unit,
        ph=ph,
        species_data=species_data,
    )


def _enzml_build_measurement_data(
    *,
    chroms: list[Chromatogram],
    sample: Sample,
    species: Molecule,
    to_concentration: bool,
    extrapolate: bool,
    draw_idx: int | None,
) -> MeasurementData | None:
    """Build one ``MeasurementData`` for a single *(sample, molecule)* pair."""

    times: list[float] = []
    values: list[float] = []
    time_unit = "min"

    for chrom in chroms:
        if chrom.reaction_time_unit:
            time_unit = _enzml_unit_to_str(chrom.reaction_time_unit)

        peak = next((p for p in chrom.peaks if p.molecule_id == species.id), None)
        area = _enzml_resolve_area(peak, draw_idx)

        # Convert area → concentration when requested and calibration is available
        if to_concentration and area is not None and species.calibration is not None:
            try:
                value: float | None = species.calibration.area_to_conc(area, extrapolate=extrapolate).mean
            except ValueError:
                value = None  # outside calibration range and extrapolate=False
        else:
            value = area

        if value is not None:
            times.append(float(chrom.reaction_time))  # type: ignore[arg-type]
            values.append(value)

    # prepared = user-set initial concentration from InitialCondition
    ic = next(
        (ic for ic in sample.initial_conditions if ic.molecule_id == species.id),
        None,
    )
    prepared: float | None = float(ic.init_conc) if ic is not None else None

    # initial = actual measured value at t=0 (fallback: prepared)
    t0_idx = next((i for i, t in enumerate(times) if t == 0.0), None)
    if t0_idx is not None and not math.isnan(values[t0_idx]):
        initial: float | None = values[t0_idx]
    else:
        initial = prepared

    data_type = DataTypes.CONCENTRATION if to_concentration else DataTypes.PEAK_AREA
    if to_concentration and species.calibration is not None:
        unit_str = species.calibration.conc_unit
    elif ic is not None:
        unit_str = _enzml_unit_to_str(ic.conc_unit)
    else:
        unit_str = "AU"
    return MeasurementData(
        species_id=species.id,
        initial=initial,
        prepared=prepared,
        data_unit=unit_str,
        data_type=data_type,
        time_unit=time_unit,
        data=values,
        time=times,
    )


def _enzml_resolve_area(peak: Any, draw_idx: int | None) -> float | None:
    """Extract a scalar area value from a ``Peak.area`` ``Estimate``.

    With *draw_idx* given and posterior samples available, returns the
    sample at position ``draw_idx % len(samples)``.  Otherwise returns
    the median (preferred) or mean point estimate.
    """
    if peak is None:
        return None
    if draw_idx is not None and peak.area.samples:
        return float(peak.area.samples[draw_idx % len(peak.area.samples)])
    if peak.area.median is not None:
        return float(peak.area.median)
    return float(peak.area.mean)


def handler_to_enzymeml_document(
    *,
    samples: list[Sample],
    molecules: list[Molecule],
    proteins: list[Protein],
    name: str,
    sample_ids: list[str] | None,
    temperature: float,
    temperature_unit: str,
    ph: float,
    to_concentration: bool,
    n_samples: int | None,
    extrapolate: bool,
) -> EnzymeMLDocument:
    """Build a pyenzyme :class:`~pyenzyme.EnzymeMLDocument` from handler state.

    *molecules* should already exclude internal standards (caller filters).
    """

    targets = _enzml_filter_samples(samples, sample_ids)

    if to_concentration:
        _enzml_assert_calibrations(molecules)

    doc = EnzymeMLDocument(name=name)
    _enzml_add_species(doc, molecules, proteins)

    draw_indices: list[int | None] = list(range(n_samples)) if n_samples is not None else [None]
    for sample in targets:
        for draw_idx in draw_indices:
            meas = _enzml_build_measurement(
                sample=sample,
                molecules=molecules,
                proteins=proteins,
                temperature=temperature,
                temperature_unit=temperature_unit,
                ph=ph,
                to_concentration=to_concentration,
                extrapolate=extrapolate,
                draw_idx=draw_idx,
            )
            doc.measurements.append(meas)

    return doc


__all__ = ["handler_to_enzymeml_document"]
