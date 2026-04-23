"""Unit tests for basic Handler functionality.

Extracted from:
  - tests/integration/test_peak_assignment.py
  - tests/integration/test_enzymeml_export.py

Content: Handler initialization, basic accessors, simple properties.
"""

from __future__ import annotations

import pytest

from chromhandler.handler import Handler
from chromhandler.model import Chromatogram, Estimate, Peak, Sample
from chromhandler.molecule import Molecule
from chromhandler.protein import Protein


def _molecule(mol_id: str = "test_mol") -> Molecule:
    """Helper: create a test molecule."""
    return Molecule(id=mol_id, name="Test Molecule", pubchem_cid=12345)


def _peak(
    retention_time: float,
    area: float,
    *,
    chrom_id: str = "chrom_0",
    mol_id: str | None = None,
) -> Peak:
    """Helper: create a test peak."""
    return Peak(
        chromatogram_id=chrom_id,
        location=Estimate(mean=retention_time),
        area=Estimate(mean=area),
        molecule_id=mol_id,
    )


def _chromatogram(
    chrom_id: str,
    sample_id: str,
    peaks: list[Peak],
) -> Chromatogram:
    """Helper: create a test chromatogram."""
    return Chromatogram(id=chrom_id, sample_id=sample_id, peaks=peaks)


def _sample(sample_id: str, *chromatograms: Chromatogram) -> Sample:
    """Helper: create a test sample."""
    return Sample(id=sample_id, chromatograms=list(chromatograms))


@pytest.mark.unit
def test_handler_initialization_empty() -> None:
    """Handler can be initialized with no arguments."""
    handler = Handler()
    assert handler.samples == []
    assert handler.molecules == {}
    assert handler.proteins == {}


@pytest.mark.unit
def test_handler_initialization_with_molecules() -> None:
    """Handler can be initialized with a DottedDict of molecules."""
    from dotted_dict import DottedDict

    mol = _molecule("mol_a")
    molecules = DottedDict({mol.id: mol})
    handler = Handler(molecules=molecules)
    assert len(handler.molecules) == 1
    assert handler.molecules["mol_a"].id == "mol_a"
    assert isinstance(handler.molecules, DottedDict)
    assert handler.molecules.mol_a.id == "mol_a"  # type: ignore[reportGeneralTypeIssues]


@pytest.mark.unit
def test_handler_initialization_with_samples() -> None:
    """Handler can be initialized with samples list."""
    sample = _sample("sample_1", _chromatogram("chrom_1", "sample_1", []))
    handler = Handler(samples=[sample])
    assert len(handler.samples) == 1
    assert handler.samples[0].id == "sample_1"


@pytest.mark.unit
def test_handler_rejects_registry_key_mismatch() -> None:
    """Handler registry keys must match the contained object ID."""
    mol = _molecule("test_mol")
    with pytest.raises(ValueError, match="molecules key"):
        Handler(molecules={"other_name": mol})


@pytest.mark.unit
def test_molecule_id_must_be_valid_python_identifier() -> None:
    """Molecule IDs are validated before they can be used in Handler registries."""
    with pytest.raises(ValueError, match=r"Molecule\.id"):
        Molecule(id="4ATP+", name="ATP", pubchem_cid=12345)


@pytest.mark.unit
def test_protein_id_must_be_valid_python_identifier() -> None:
    """Protein IDs follow the same DottedDict-safe contract as molecule IDs."""
    with pytest.raises(ValueError, match=r"Protein\.id"):
        Protein(id="4ATP+", name="Protein")


@pytest.mark.unit
def test_handler_samples_list_access() -> None:
    """Handler.samples list can be accessed directly."""
    sample = _sample("sample_1")
    handler = Handler(samples=[sample])
    assert handler.samples[0] == sample
    assert handler.samples[0].id == "sample_1"


@pytest.mark.unit
def test_handler_add_peak_annotation_requires_existing_molecule() -> None:
    """Handler.add_peak_annotation() raises for unknown molecule."""
    handler = Handler()
    with pytest.raises(ValueError, match="Molecule unknown not found"):
        handler.add_peak_annotation("unknown", 4.8, 5.2)


@pytest.mark.unit
def test_handler_add_peak_annotation_creates_annotation() -> None:
    """Handler.add_peak_annotation() creates a peak annotation for known molecule."""
    handler = Handler()
    handler.create_molecule(id="test_mol", pubchem_cid=123, name="Test")
    ann = handler.add_peak_annotation("test_mol", 4.8, 5.2)
    assert ann.molecule_id == "test_mol"
    assert ann.rt_min == 4.8
    assert ann.rt_max == 5.2
    assert ann.mode == "single"
    assert ann.wavelength is None


@pytest.mark.unit
def test_handler_peak_annotations_property() -> None:
    """Handler.peak_annotations dict tracks added annotations."""
    handler = Handler()
    handler.create_molecule(id="test_mol", pubchem_cid=123, name="Test")
    ann = handler.add_peak_annotation("test_mol", 4.8, 5.2)
    assert "test_mol" in handler.peak_annotations
    assert handler.peak_annotations["test_mol"] == ann


@pytest.mark.unit
def test_handler_subset_creates_independent_handler() -> None:
    """Handler.subset() creates an independent handler with filtered samples."""
    sample_a = _sample(
        "sample_a",
        _chromatogram("chrom_a", "sample_a", [_peak(5.0, 100.0, chrom_id="chrom_a")]),
    )
    sample_b = _sample(
        "sample_b",
        _chromatogram("chrom_b", "sample_b", [_peak(6.0, 200.0, chrom_id="chrom_b")]),
    )
    parent = Handler(samples=[sample_a, sample_b], molecules={})
    child = parent.subset(["chrom_b"])
    assert len(child.samples) == 1
    assert child.samples[0].id == "sample_b"


@pytest.mark.unit
def test_peak_annotation_defaults_to_single_mode() -> None:
    """PeakAnnotation.mode defaults to 'single' so handler-only users omit it."""
    from chromhandler.annotations import PeakAnnotation

    ann = PeakAnnotation(molecule_id="mol", rt_min=2.8, rt_max=3.2)
    assert ann.mode == "single"
    assert ann.artefact_side is None
    assert ann.wavelength is None


@pytest.mark.unit
def test_peak_annotation_accepts_wavelength() -> None:
    """PeakAnnotation carries an optional wavelength for handler filtering."""
    from chromhandler.annotations import PeakAnnotation

    ann = PeakAnnotation(molecule_id="mol", rt_min=2.8, rt_max=3.2, wavelength=280.0)
    assert ann.wavelength == 280.0


@pytest.mark.unit
def test_chromatogram_roundtrips_trace_stats_through_json() -> None:
    """Chromatogram.trace_stats serialises + deserialises without loss."""
    from chromhandler.trace_statistics import TraceStatistics

    chrom = Chromatogram(
        id="c0",
        sample_id="s0",
        signal=[1.0, 2.0, 3.0, 2.0, 1.0],
        time=[0.0, 0.1, 0.2, 0.3, 0.4],
        trace_stats=TraceStatistics(sigma_noise=0.75),
    )

    dumped = chrom.model_dump_json()
    restored = Chromatogram.model_validate_json(dumped)

    assert restored.trace_stats is not None
    assert restored.trace_stats.sigma_noise == pytest.approx(0.75)


@pytest.mark.unit
def test_chromatogram_defaults_trace_stats_to_none() -> None:
    chrom = Chromatogram(id="c0", sample_id="s0")
    assert chrom.trace_stats is None
