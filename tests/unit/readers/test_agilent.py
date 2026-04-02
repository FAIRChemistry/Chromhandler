"""Unit tests for AgilentReader using real fixture data."""
from __future__ import annotations

from pathlib import Path

import pytest

from chromhandler.readers.agilent import AgilentReader

FIXTURE_DIR = Path(__file__).parent.parent.parent / "fixtures" / "agilent"
D_DIR = FIXTURE_DIR / "001F0130_0h.D"


@pytest.mark.readers
class TestAgilentReaderAutoSelect:
    def test_raises_when_multiple_files_and_no_selector(self) -> None:
        reader = AgilentReader()
        with pytest.raises(ValueError, match="FID1A"):
            reader.read_file(D_DIR, chromatogram_id="x", sample_id="s")

    def test_single_file_auto_selected(self, tmp_path: Path) -> None:
        """If a .D dir has only one datafile it is auto-selected."""
        import shutil

        single_d = tmp_path / "single.D"
        shutil.copytree(D_DIR, single_d)
        # Remove one detector file to leave a single one.
        tcd = single_d / "TCD2B.ch"
        if tcd.exists():
            tcd.unlink()

        reader = AgilentReader()
        chrom = reader.read_file(single_d, chromatogram_id="c", sample_id="s", reaction_time=0.0)
        assert chrom.id == "c"
        assert chrom.sample_id == "s"
        assert chrom.reaction_time == 0.0
        assert len(chrom.signal) > 0
        assert len(chrom.time) == len(chrom.signal)
        assert chrom.peaks == []


@pytest.mark.readers
class TestAgilentReaderChannelSelection:
    def test_selects_channel_case_insensitive(self) -> None:
        for name in ("FID1A.CH", "fid1a.ch", "Fid1A.ch"):
            reader = AgilentReader(channel=name)
            chrom = reader.read_file(D_DIR, chromatogram_id="c", sample_id="s")
            assert isinstance(chrom.signal, list)
            assert len(chrom.signal) > 0

    def test_raises_on_unknown_channel(self) -> None:
        reader = AgilentReader(channel="DAD1A.CH")
        with pytest.raises(ValueError, match="DAD1A.CH"):
            reader.read_file(D_DIR, chromatogram_id="c", sample_id="s")

    def test_error_lists_available_channels(self) -> None:
        reader = AgilentReader(channel="NOPE.CH")
        with pytest.raises(ValueError, match="FID1A"):
            reader.read_file(D_DIR, chromatogram_id="c", sample_id="s")

    def test_chromatogram_fields(self) -> None:
        reader = AgilentReader(channel="FID1A.CH")
        chrom = reader.read_file(D_DIR, chromatogram_id="my_id", sample_id="my_sample", reaction_time=5.0)
        assert chrom.id == "my_id"
        assert chrom.sample_id == "my_sample"
        assert chrom.reaction_time == 5.0
        assert chrom.peaks == []
        assert chrom.wavelength is None
        assert all(isinstance(v, float) for v in chrom.signal[:5])
        assert all(isinstance(v, float) for v in chrom.time[:5])


@pytest.mark.readers
class TestAgilentReaderWavelengthSelection:
    def test_raises_when_no_wavelength_data(self) -> None:
        reader = AgilentReader(wavelength=254.0)
        with pytest.raises(ValueError, match="wavelength data"):
            reader.read_file(D_DIR, chromatogram_id="c", sample_id="s")

    def test_error_suggests_channel(self) -> None:
        reader = AgilentReader(wavelength=254.0)
        with pytest.raises(ValueError, match="channel"):
            reader.read_file(D_DIR, chromatogram_id="c", sample_id="s")
