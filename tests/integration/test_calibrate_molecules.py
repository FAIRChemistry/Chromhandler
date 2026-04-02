"""Tests for Handler.calibrate_molecules() — return type, verbose flag, and rich output."""

from __future__ import annotations

import pytest

from chromhandler.handler import Handler
from chromhandler.model import Chromatogram, Estimate, InitialCondition, Peak, Sample
from chromhandler.molecule import Molecule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mol(mol_id: str = "Ino") -> Molecule:
    return Molecule(id=mol_id, name="Inosine", pubchem_cid=6021)


def _peak_for(mol_id: str, area: float, chrom_id: str) -> Peak:
    return Peak(
        chromatogram_id=chrom_id,
        location=Estimate(mean=5.0),
        area=Estimate(mean=area),
        molecule_id=mol_id,
    )


def _cal_sample(
    sample_id: str,
    mol_id: str,
    init_conc: float,
    area: float,
) -> Sample:
    """One calibration standard: t=0 chromatogram with a peak + InitialCondition."""
    chrom_id = f"{sample_id}_chrom"
    chrom = Chromatogram(
        id=chrom_id,
        sample_id=sample_id,
        reaction_time=0.0,
        reaction_time_unit="min",
        peaks=[_peak_for(mol_id, area, chrom_id)],
    )
    ic = InitialCondition(
        molecule_id=mol_id,
        init_conc=init_conc,
        conc_unit="umol / l",
    )
    return Sample(id=sample_id, chromatograms=[chrom], initial_conditions=[ic])


def _handler_with_calibration_data() -> Handler:
    """Handler with one molecule and two calibration standards."""
    mol = _mol("Ino")
    samples = [
        _cal_sample("std_low", "Ino", init_conc=100.0, area=400_000.0),
        _cal_sample("std_high", "Ino", init_conc=400.0, area=1_600_000.0),
    ]
    return Handler(samples=samples, molecules={mol.id: mol})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_calibrate_molecules_returns_none() -> None:
    """calibrate_molecules must be a side-effect-only method — return value is None."""
    handler = _handler_with_calibration_data()
    result = handler.calibrate_molecules(verbose=False)
    assert result is None


@pytest.mark.integration
def test_calibrate_molecules_sets_calibration_on_molecule() -> None:
    """The side effect (molecule.calibration) must still be set after the call."""
    handler = _handler_with_calibration_data()
    handler.calibrate_molecules(verbose=False)
    mol = handler.molecules["Ino"]
    assert mol.calibration is not None
    assert mol.calibration.slope > 0


@pytest.mark.integration
def test_calibrate_molecules_verbose_false_suppresses_output(capsys: pytest.CaptureFixture[str]) -> None:
    """verbose=False must produce no stdout output at all."""
    handler = _handler_with_calibration_data()
    handler.calibrate_molecules(verbose=False)
    captured = capsys.readouterr()
    assert captured.out == ""


@pytest.mark.integration
def test_calibrate_molecules_verbose_true_prints_summary(capsys: pytest.CaptureFixture[str]) -> None:
    """verbose=True (default) must print a calibration summary to stdout."""
    handler = _handler_with_calibration_data()
    handler.calibrate_molecules()  # default verbose=True
    captured = capsys.readouterr()
    # Should mention the molecule id somewhere in the output
    assert "Ino" in captured.out


@pytest.mark.integration
def test_calibrate_molecules_skipped_molecule_not_in_output(capsys: pytest.CaptureFixture[str]) -> None:
    """A molecule with no calibration data should still be reported (as skipped), not silently dropped."""
    mol = _mol("Ghost")
    handler = Handler(samples=[], molecules={mol.id: mol})
    handler.calibrate_molecules()
    captured = capsys.readouterr()
    # Should mention Ghost in some skipped/warning row
    assert "Ghost" in captured.out
