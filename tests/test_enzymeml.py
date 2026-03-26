"""Tests for EnzymeML export: :meth:`~chromhandler.handler.Handler.to_enzymeml` and
:class:`~chromhandler.enzymeml.handler_to_enzymeml_document`.
"""

from __future__ import annotations

import pytest
from pyenzyme import DataTypes

from chromhandler.enzymeml import handler_to_enzymeml_document
from chromhandler.handler import Handler
from chromhandler.model import Chromatogram, Estimate, InitialCondition, Peak, Sample
from chromhandler.molecule import Molecule
from chromhandler.protein import Protein


def _mol(mol_id: str = "Sub", *, internal_standard: bool = False) -> Molecule:
    return Molecule(
        id=mol_id,
        name="Substrate",
        pubchem_cid=6021,
        internal_standard=internal_standard,
    )


def _peak(mol_id: str, area: float, chrom_id: str) -> Peak:
    return Peak(
        chromatogram_id=chrom_id,
        location=Estimate(mean=5.0),
        area=Estimate(mean=area),
        molecule_id=mol_id,
    )


def _timecourse_sample(
    sample_id: str,
    mol_id: str,
    *,
    points: list[tuple[float, float]],
) -> Sample:
    """Chromatograms at given (reaction_time_min, area) points."""
    chroms: list[Chromatogram] = []
    for t, area in points:
        cid = f"{sample_id}_t{t:g}"
        chroms.append(
            Chromatogram(
                id=cid,
                sample_id=sample_id,
                reaction_time=t,
                reaction_time_unit="min",
                peaks=[_peak(mol_id, area, cid)],
            )
        )
    ic = InitialCondition(
        molecule_id=mol_id,
        init_conc=0.0,
        conc_unit="umol / l",
    )
    return Sample(id=sample_id, chromatograms=chroms, initial_conditions=[ic])


def _handler_basic() -> Handler:
    mol = _mol("Sub")
    samples = [_timecourse_sample("S1", "Sub", points=[(0.0, 100.0), (15.0, 50.0)])]
    return Handler(samples=samples, molecules={mol.id: mol})


def _cal_sample(sample_id: str, mol_id: str, init_conc: float, area: float) -> Sample:
    chrom_id = f"{sample_id}_chrom"
    chrom = Chromatogram(
        id=chrom_id,
        sample_id=sample_id,
        reaction_time=0.0,
        reaction_time_unit="min",
        peaks=[_peak(mol_id, area, chrom_id)],
    )
    ic = InitialCondition(
        molecule_id=mol_id,
        init_conc=init_conc,
        conc_unit="umol / l",
    )
    return Sample(id=sample_id, chromatograms=[chrom], initial_conditions=[ic])


def test_to_enzymeml_returns_document_with_species_and_measurement() -> None:
    handler = _handler_basic()
    doc = handler.to_enzymeml(
        name="exp1",
        temperature=37.0,
        temperature_unit="Celsius",
        ph=7.4,
    )
    assert doc.name == "exp1"
    assert len(doc.small_molecules) == 1
    assert doc.small_molecules[0].id == "Sub"
    assert len(doc.measurements) == 1
    meas = doc.measurements[0]
    assert meas.id == "S1"
    assert meas.temperature == 37.0
    assert meas.ph == 7.4
    mol_data = next(sd for sd in meas.species_data if sd.species_id == "Sub")
    assert mol_data.data_type == DataTypes.PEAK_AREA
    assert mol_data.time == [0.0, 15.0]
    assert mol_data.data == [100.0, 50.0]


def test_handler_to_enzymeml_document_matches_handler_method() -> None:
    handler = _handler_basic()
    direct = handler_to_enzymeml_document(
        samples=handler.samples,
        molecules=[m for m in handler.molecules.values() if not m.internal_standard],
        proteins=list(handler.proteins.values()),
        name="direct",
        sample_ids=None,
        temperature=25.0,
        temperature_unit="Celsius",
        ph=7.0,
        to_concentration=False,
        n_samples=None,
        extrapolate=False,
    )
    via_handler = handler.to_enzymeml(
        name="direct",
        temperature=25.0,
        temperature_unit="Celsius",
        ph=7.0,
    )
    assert direct.name == via_handler.name
    assert len(direct.measurements) == len(via_handler.measurements)


def test_sample_ids_filter_and_missing_raises() -> None:
    mol = _mol("Sub")
    samples = [
        _timecourse_sample("A", "Sub", points=[(0.0, 1.0)]),
        _timecourse_sample("B", "Sub", points=[(0.0, 2.0)]),
    ]
    handler = Handler(samples=samples, molecules={mol.id: mol})
    doc = handler.to_enzymeml(
        name="x",
        sample_ids=["B"],
        temperature=25.0,
        temperature_unit="Celsius",
        ph=7.0,
    )
    assert len(doc.measurements) == 1
    assert doc.measurements[0].id == "B"

    with pytest.raises(ValueError, match="sample_ids not found"):
        handler.to_enzymeml(
            name="x",
            sample_ids=["nosuch"],
            temperature=25.0,
            temperature_unit="Celsius",
            ph=7.0,
        )


def test_to_concentration_requires_calibration() -> None:
    handler = _handler_basic()
    with pytest.raises(ValueError, match="no calibration"):
        handler.to_enzymeml(
            name="x",
            temperature=25.0,
            temperature_unit="Celsius",
            ph=7.0,
            to_concentration=True,
        )


def test_to_concentration_after_calibrate() -> None:
    mol = _mol("Ino")
    samples = [
        _cal_sample("std_low", "Ino", init_conc=100.0, area=400_000.0),
        _cal_sample("std_high", "Ino", init_conc=400.0, area=1_600_000.0),
        _timecourse_sample("run1", "Ino", points=[(0.0, 800_000.0), (30.0, 400_000.0)]),
    ]
    handler = Handler(samples=samples, molecules={mol.id: mol})
    handler.calibrate_molecules(verbose=False)
    doc = handler.to_enzymeml(
        name="cal",
        sample_ids=["run1"],
        temperature=30.0,
        temperature_unit="Celsius",
        ph=6.5,
        to_concentration=True,
    )
    meas = doc.measurements[0]
    md = next(sd for sd in meas.species_data if sd.species_id == "Ino")
    assert md.data_type == DataTypes.CONCENTRATION
    assert len(md.data) == 2
    assert all(isinstance(v, float) for v in md.data)


def test_internal_standard_excluded_from_export() -> None:
    sub = _mol("Sub")
    istd = _mol("IS", internal_standard=True)
    samples = [_timecourse_sample("S1", "Sub", points=[(0.0, 10.0)])]
    handler = Handler(samples=samples, molecules={sub.id: sub, istd.id: istd})
    doc = handler.to_enzymeml(
        name="x",
        temperature=25.0,
        temperature_unit="Celsius",
        ph=7.0,
    )
    ids = {sm.id for sm in doc.small_molecules}
    assert ids == {"Sub"}
    assert "IS" not in ids


def test_protein_passed_to_measurement_data() -> None:
    mol = _mol("Sub")
    prot = Protein(
        id="E1",
        name="Enzyme",
    )
    samples = [_timecourse_sample("S1", "Sub", points=[(0.0, 1.0)])]
    handler = Handler(samples=samples, molecules={mol.id: mol}, proteins={prot.id: prot})
    doc = handler.to_enzymeml(
        name="x",
        temperature=25.0,
        temperature_unit="Celsius",
        ph=7.0,
    )
    assert len(doc.proteins) == 1
    assert doc.proteins[0].id == "E1"
    meas = doc.measurements[0]
    # Protein is in the document but no concentration data since Protein no longer has init_conc/conc_unit
    assert all(sd.species_id != "E1" for sd in meas.species_data)


def test_n_samples_duplicates_measurements() -> None:
    mol = _mol("Sub")
    cid = "S1_c"
    chrom = Chromatogram(
        id=cid,
        sample_id="S1",
        reaction_time=0.0,
        reaction_time_unit="min",
        peaks=[
            Peak(
                chromatogram_id=cid,
                location=Estimate(mean=1.0),
                area=Estimate(mean=100.0, samples=[90.0, 100.0, 110.0]),
                molecule_id="Sub",
            )
        ],
    )
    ic = InitialCondition(molecule_id="Sub", init_conc=0.0, conc_unit="umol / l")
    sample = Sample(id="S1", chromatograms=[chrom], initial_conditions=[ic])
    handler = Handler(samples=[sample], molecules={mol.id: mol})
    doc = handler.to_enzymeml(
        name="x",
        temperature=25.0,
        temperature_unit="Celsius",
        ph=7.0,
        n_samples=3,
    )
    assert len(doc.measurements) == 3
    drawn: list[float] = []
    for m in doc.measurements:
        md = next(sd for sd in m.species_data if sd.species_id == "Sub")
        assert len(md.data) == 1
        drawn.append(md.data[0])
    assert drawn == [90.0, 100.0, 110.0]
