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
    """Handler can be initialized with molecules dict."""
    mol = _molecule("mol_a")
    handler = Handler(molecules={mol.id: mol})
    assert len(handler.molecules) == 1
    assert handler.molecules["mol_a"].id == "mol_a"


@pytest.mark.unit
def test_handler_initialization_with_samples() -> None:
    """Handler can be initialized with samples list."""
    sample = _sample("sample_1", _chromatogram("chrom_1", "sample_1", []))
    handler = Handler(samples=[sample])
    assert len(handler.samples) == 1
    assert handler.samples[0].id == "sample_1"


@pytest.mark.unit
def test_handler_get_molecule_returns_molecule() -> None:
    """Handler.get_molecule() returns the correct molecule."""
    mol = _molecule("test_mol")
    handler = Handler(molecules={mol.id: mol})
    assert handler.get_molecule("test_mol") == mol


@pytest.mark.unit
def test_handler_get_molecule_raises_for_unknown_molecule() -> None:
    """Handler.get_molecule() raises ValueError for unknown molecule ID."""
    handler = Handler()
    with pytest.raises(ValueError, match="Unknown molecule"):
        handler.get_molecule("unknown_mol")


@pytest.mark.unit
def test_handler_samples_list_access() -> None:
    """Handler.samples list can be accessed directly."""
    sample = _sample("sample_1")
    handler = Handler(samples=[sample])
    assert handler.samples[0] == sample
    assert handler.samples[0].id == "sample_1"


@pytest.mark.unit
def test_handler_add_peak_window_requires_existing_molecule() -> None:
    """Handler.add_peak_window() raises for unknown molecule."""
    handler = Handler()
    with pytest.raises(ValueError, match="unknown"):
        handler.add_peak_window("unknown", 4.8, 5.2)


@pytest.mark.unit
def test_handler_add_peak_window_creates_window() -> None:
    """Handler.add_peak_window() creates a peak window for known molecule."""
    mol = _molecule("test_mol")
    handler = Handler(molecules={mol.id: mol})
    window = handler.add_peak_window("test_mol", 4.8, 5.2)
    assert window.rt_min == 4.8
    assert window.rt_max == 5.2
    assert window.molecule_id == "test_mol"


@pytest.mark.unit
def test_handler_peak_windows_property() -> None:
    """Handler.peak_windows dict tracks added windows."""
    mol = _molecule("test_mol")
    handler = Handler(molecules={mol.id: mol})
    window = handler.add_peak_window("test_mol", 4.8, 5.2)
    assert "test_mol" in handler.peak_windows
    assert handler.peak_windows["test_mol"] == window


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
