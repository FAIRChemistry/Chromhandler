from typing import Any

import pytest
from pytest import CaptureFixture

from chromhandler.handler import Handler
from chromhandler.model import Chromatogram, Estimate, Peak, Sample
from chromhandler.molecule import Molecule


class TestPeakAssignment:
    """Test class for peak assignment functionality."""

    @pytest.fixture
    def mock_molecule(self) -> Molecule:
        """Create a test molecule with defined retention time."""
        return Molecule(
            id="test_mol",
            pubchem_cid=12345,
            name="Test Molecule",
            retention_time=5.0,
            retention_tolerance=0.2,
            min_signal=100.0,
        )

    def create_peak(self, retention_time: float, area: float, chrom_id: str = "chrom_0") -> Peak:
        return Peak(
            chromatogram_id=chrom_id,
            location=Estimate(mean=retention_time),
            area=Estimate(mean=area),
        )

    def create_chromatogram(self, peaks: list[Peak], sample_id: str = "s0") -> Chromatogram:
        return Chromatogram(id="chrom_0", sample_id=sample_id, peaks=peaks, wavelength=254.0)

    def create_sample(self, sample_id: str, peaks: list[Peak]) -> Sample:
        chrom = self.create_chromatogram(peaks, sample_id=sample_id)
        return Sample(id=sample_id, chromatograms=[chrom])

    def create_analyzer(self, samples: list[Sample]) -> Handler:
        return Handler(id="test_analyzer", name="Test Analyzer", samples=samples)

    def test_single_peak_assignment(
        self, mock_molecule: Molecule, capsys: CaptureFixture[Any]
    ) -> None:
        """Test normal case: single peak within tolerance gets assigned."""
        peaks = [self.create_peak(retention_time=5.0, area=500.0)]
        sample = self.create_sample("meas_001", peaks)
        analyzer = self.create_analyzer([sample])
        analyzer.molecules.append(mock_molecule)

        analyzer._register_peaks(mock_molecule, mock_molecule.retention_tolerance, 254.0)

        assert peaks[0].molecule_id == "test_mol"

        captured = capsys.readouterr()
        assert "Assigned Test Molecule to 1 peaks" in captured.out
        assert "Warning" not in captured.out

    def test_multiple_peaks_closest_assigned(
        self, mock_molecule: Molecule, capsys: CaptureFixture[Any]
    ) -> None:
        """Test multiple peaks: closest one gets assigned with warning."""
        peaks = [
            self.create_peak(retention_time=4.9, area=300.0),
            self.create_peak(retention_time=5.15, area=400.0),
            self.create_peak(retention_time=4.85, area=200.0),
        ]
        sample = self.create_sample("meas_001", peaks)
        analyzer = self.create_analyzer([sample])
        analyzer.molecules.append(mock_molecule)

        analyzer._register_peaks(mock_molecule, mock_molecule.retention_tolerance, 254.0)

        assigned_peaks = [p for p in peaks if p.molecule_id == "test_mol"]
        assert len(assigned_peaks) == 1
        assert assigned_peaks[0].location.mean == 4.9

        captured = capsys.readouterr()
        assert "Assigned Test Molecule to 1 peaks" in captured.out
        assert "Warning: Multiple peaks found within tolerance" in captured.out
        assert "4.900" in captured.out and "5.150" in captured.out and "4.850" in captured.out
        assert "Tip: Consider setting a higher min_signal value" in captured.out

    def test_no_peaks_found_warning(
        self, mock_molecule: Molecule, capsys: CaptureFixture[Any]
    ) -> None:
        """Test no peaks found: warning is displayed."""
        peaks = [
            self.create_peak(retention_time=3.0, area=500.0),
            self.create_peak(retention_time=7.0, area=500.0),
        ]
        sample = self.create_sample("meas_001", peaks)
        analyzer = self.create_analyzer([sample])
        analyzer.molecules.append(mock_molecule)

        analyzer._register_peaks(mock_molecule, mock_molecule.retention_tolerance, 254.0)

        assigned_peaks = [p for p in peaks if p.molecule_id == "test_mol"]
        assert len(assigned_peaks) == 0

        captured = capsys.readouterr()
        assert "Assigned Test Molecule to 0 peaks" not in captured.out
        assert "Warning: No peaks found for Test Molecule in 1 measurement(s)" in captured.out
        assert "meas_001" in captured.out
        assert "5.000 min" in captured.out

    def test_min_signal_filtering(
        self, mock_molecule: Molecule, capsys: CaptureFixture[Any]
    ) -> None:
        """Test that peaks below min_signal are filtered out."""
        peaks = [
            self.create_peak(retention_time=5.0, area=50.0),   # below min_signal
            self.create_peak(retention_time=5.1, area=150.0),  # above min_signal
        ]
        sample = self.create_sample("meas_001", peaks)
        analyzer = self.create_analyzer([sample])
        analyzer.molecules.append(mock_molecule)

        analyzer._register_peaks(mock_molecule, mock_molecule.retention_tolerance, 254.0)

        assigned_peaks = [p for p in peaks if p.molecule_id == "test_mol"]
        assert len(assigned_peaks) == 1
        assert assigned_peaks[0].area.mean == 150.0

        captured = capsys.readouterr()
        assert "Assigned Test Molecule to 1 peaks" in captured.out

    def test_multiple_samples_mixed_scenarios(
        self, mock_molecule: Molecule, capsys: CaptureFixture[Any]
    ) -> None:
        """Test complex scenario with multiple samples."""
        peaks1 = [self.create_peak(retention_time=5.0, area=300.0)]
        peaks2 = [
            self.create_peak(retention_time=4.95, area=200.0),
            self.create_peak(retention_time=5.1, area=400.0),
        ]
        peaks3 = [self.create_peak(retention_time=3.0, area=500.0)]
        peaks4 = [self.create_peak(retention_time=5.0, area=50.0)]  # below min_signal

        analyzer = self.create_analyzer([
            self.create_sample("meas_001", peaks1),
            self.create_sample("meas_002", peaks2),
            self.create_sample("meas_003", peaks3),
            self.create_sample("meas_004", peaks4),
        ])
        analyzer.molecules.append(mock_molecule)

        analyzer._register_peaks(mock_molecule, mock_molecule.retention_tolerance, 254.0)

        assert len([p for p in peaks1 if p.molecule_id == "test_mol"]) == 1
        assigned2 = [p for p in peaks2 if p.molecule_id == "test_mol"]
        assert len(assigned2) == 1
        assert assigned2[0].location.mean == 4.95
        assert len([p for p in peaks3 if p.molecule_id == "test_mol"]) == 0
        assert len([p for p in peaks4 if p.molecule_id == "test_mol"]) == 0

        captured = capsys.readouterr()
        assert "Assigned Test Molecule to 2 peaks" in captured.out
        assert "Warning: Multiple peaks found within tolerance" in captured.out
        assert "Warning: No peaks found for Test Molecule in 2 measurement(s)" in captured.out
        assert "meas_003" in captured.out and "meas_004" in captured.out

    def test_no_retention_time_raises_error(self) -> None:
        """Test that molecule without retention time raises ValueError."""
        molecule_no_rt = Molecule(
            id="no_rt_mol",
            pubchem_cid=67890,
            name="No RT Molecule",
            retention_time=None,
        )
        peaks = [self.create_peak(retention_time=5.0, area=300.0)]
        analyzer = self.create_analyzer([self.create_sample("meas_001", peaks)])
        analyzer.molecules.append(molecule_no_rt)

        with pytest.raises(ValueError, match="no_rt_mol"):
            analyzer._register_peaks(molecule_no_rt, 0.2, 254.0)

    def test_tolerance_boundary_conditions(
        self,
        capsys: CaptureFixture[Any],
    ) -> None:
        """Test peaks exactly at tolerance boundaries."""
        molecule = Molecule(
            id="boundary_mol",
            pubchem_cid=11111,
            name="Boundary Molecule",
            retention_time=5.0,
            retention_tolerance=0.2,
            min_signal=100.0,
        )

        peaks = [
            self.create_peak(retention_time=4.85, area=300.0),  # within tolerance
            self.create_peak(retention_time=5.15, area=400.0),  # within tolerance
            self.create_peak(retention_time=4.75, area=200.0),  # outside tolerance
        ]
        analyzer = self.create_analyzer([self.create_sample("meas_001", peaks)])
        analyzer.molecules.append(molecule)

        analyzer._register_peaks(molecule, molecule.retention_tolerance, 254.0)

        assigned_peaks = [p for p in peaks if p.molecule_id == "boundary_mol"]
        assert len(assigned_peaks) == 1
        assert assigned_peaks[0].location.mean == 4.85

        captured = capsys.readouterr()
        assert "Warning: Multiple peaks found within tolerance" in captured.out
        assert "4.850" in captured.out and "5.150" in captured.out

    def test_empty_samples(self, mock_molecule: Molecule) -> None:
        """Test behavior with empty samples list."""
        analyzer = self.create_analyzer([])
        analyzer.molecules.append(mock_molecule)

        analyzer._register_peaks(mock_molecule, mock_molecule.retention_tolerance, 254.0)

    def test_chromatogram_with_no_peaks(
        self, mock_molecule: Molecule, capsys: CaptureFixture[Any]
    ) -> None:
        """Test behavior with chromatogram containing no peaks."""
        sample = Sample(
            id="empty_meas",
            chromatograms=[Chromatogram(id="c0", sample_id="empty_meas", peaks=[], wavelength=254.0)],
        )
        analyzer = self.create_analyzer([sample])
        analyzer.molecules.append(mock_molecule)

        analyzer._register_peaks(mock_molecule, mock_molecule.retention_tolerance, 254.0)

        captured = capsys.readouterr()
        assert "Assigned Test Molecule to 0 peaks" not in captured.out
        assert "Warning: No peaks found for Test Molecule in 1 measurement(s)" in captured.out
        assert "empty_meas" in captured.out
