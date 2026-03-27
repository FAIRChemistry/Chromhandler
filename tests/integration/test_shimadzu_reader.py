from __future__ import annotations

from pathlib import Path

import pytest

from chromhandler.handler import Handler

FIXTURES = Path(__file__).parent.parent / "fixtures" / "shimadzu"


def test_read_shimadzu_timecourse_flat() -> None:
    """Flat fixture dir: 3 files → 1 sample, 3 chromatograms sorted by RT."""
    handler = Handler.read_shimadzu(FIXTURES, mode="timecourse")

    assert len(handler.samples) == 1
    sample = handler.samples[0]
    assert sample.id == "shimadzu"
    assert len(sample.chromatograms) == 3

    rts = [c.reaction_time for c in sample.chromatograms]
    assert rts == pytest.approx([0.0, 3.10, 6.28])

    ids = [c.id for c in sample.chromatograms]
    assert ids == ["P0-0.0_min", "P1-3.10_min", "P2-6.28_min"]


def test_read_shimadzu_first_chromatogram() -> None:
    """Validate signal, time, wavelength, and peaks for P0-0.0_min."""
    handler = Handler.read_shimadzu(FIXTURES, mode="timecourse")
    chrom = handler.samples[0].chromatograms[0]

    assert chrom.id == "P0-0.0_min"
    assert chrom.sample_id == "shimadzu"
    assert chrom.reaction_time == pytest.approx(0.0)

    assert len(chrom.time) == len(chrom.signal)
    assert len(chrom.time) > 0

    # first data point: time=0, signal=0
    assert chrom.time[0] == pytest.approx(0.0)
    assert chrom.signal[0] == pytest.approx(0.0)

    # wavelength from [PDA Multi Chromatogram(Ch1)] metadata
    assert chrom.wavelength == pytest.approx(215.0)

    # peaks from [Peak Table(PDA-Ch1)]
    assert len(chrom.peaks) == 2
    first_peak = chrom.peaks[0]
    assert first_peak.location.mean == pytest.approx(10.618)
    assert first_peak.area.mean == pytest.approx(25899.0)
    assert first_peak.peak_start == pytest.approx(10.517)
    assert first_peak.peak_end == pytest.approx(10.912)
    assert first_peak.amplitude == pytest.approx(3994.0)


def test_read_shimadzu_endpoint() -> None:
    """Endpoint mode: each file → own sample, no reaction time."""
    handler = Handler.read_shimadzu(FIXTURES, mode="endpoint")

    assert len(handler.samples) == 3
    sample_ids = sorted(s.id for s in handler.samples)
    assert sample_ids == ["P0-0.0_min", "P1-3.10_min", "P2-6.28_min"]

    for sample in handler.samples:
        assert len(sample.chromatograms) == 1
        chrom = sample.chromatograms[0]
        assert chrom.reaction_time is None
        assert chrom.id == sample.id


def test_read_shimadzu_not_a_directory() -> None:
    with pytest.raises(NotADirectoryError):
        Handler.read_shimadzu(FIXTURES / "P0-0.0_min.txt")


def test_read_shimadzu_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Handler.read_shimadzu(tmp_path, mode="timecourse")
