"""Unit tests for Molecule model.

Extracted from:
  - tests/integration/test_enzymeml_export.py

Content: Molecule initialization, pubchem_cid handling, properties.
"""

from __future__ import annotations

import pytest

from chromhandler.molecule import Molecule


@pytest.mark.unit
def test_molecule_initialization_minimal() -> None:
    """Molecule can be created with required parameters."""
    mol = Molecule(id="mol_a", name="Molecule A", pubchem_cid=0)
    assert mol.id == "mol_a"


@pytest.mark.unit
def test_molecule_initialization_with_name() -> None:
    """Molecule can be created with name."""
    mol = Molecule(id="Substrate", name="Substrate Molecule", pubchem_cid=6021)
    assert mol.id == "Substrate"
    assert mol.name == "Substrate Molecule"


@pytest.mark.unit
def test_molecule_initialization_with_pubchem_cid() -> None:
    """Molecule can be created with pubchem_cid."""
    mol = Molecule(id="Ino", name="Inosine", pubchem_cid=6021)
    assert mol.id == "Ino"
    assert mol.pubchem_cid == 6021


@pytest.mark.unit
def test_molecule_initialization_all_fields() -> None:
    """Molecule can be created with all parameters."""
    mol = Molecule(
        id="Sub",
        name="Substrate",
        pubchem_cid=6021,
    )
    assert mol.id == "Sub"
    assert mol.name == "Substrate"
    assert mol.pubchem_cid == 6021


@pytest.mark.unit
def test_molecule_internal_standard_default_false() -> None:
    """Molecule.internal_standard defaults to False."""
    mol = Molecule(id="Sub", name="Substrate", pubchem_cid=6021)
    assert mol.internal_standard is False


@pytest.mark.unit
def test_molecule_internal_standard_can_be_true() -> None:
    """Molecule.internal_standard can be set to True."""
    mol = Molecule(id="IS", name="Internal Standard", pubchem_cid=0, internal_standard=True)
    assert mol.internal_standard is True


@pytest.mark.unit
def test_molecule_equality() -> None:
    """Two molecules with same fields are equal."""
    mol1 = Molecule(id="Sub", name="Substrate", pubchem_cid=6021)
    mol2 = Molecule(id="Sub", name="Substrate", pubchem_cid=6021)
    assert mol1 == mol2


@pytest.mark.unit
def test_molecule_inequality_different_id() -> None:
    """Molecules with different IDs are not equal."""
    mol1 = Molecule(id="Sub", name="Substrate", pubchem_cid=6021)
    mol2 = Molecule(id="Prod", name="Product", pubchem_cid=6021)
    assert mol1 != mol2


@pytest.mark.unit
def test_molecule_inequality_different_pubchem_cid() -> None:
    """Molecules with different pubchem_cid are not equal."""
    mol1 = Molecule(id="Sub", name="Substrate", pubchem_cid=6021)
    mol2 = Molecule(id="Sub", name="Substrate", pubchem_cid=12345)
    assert mol1 != mol2


@pytest.mark.unit
def test_molecule_pubchem_cid_zero_valid() -> None:
    """Molecule pubchem_cid can be zero."""
    mol = Molecule(id="Unknown", name="Unknown Compound", pubchem_cid=0)
    assert mol.pubchem_cid == 0
