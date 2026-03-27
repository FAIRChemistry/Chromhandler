from __future__ import annotations

from pathlib import Path

import pytest

from chromhandler.handler import Handler

FIXTURES = Path(__file__).parent.parent / "fixtures" / "knauer_txt"


def test_read_knauer_timecourse_flat() -> None:
    """Flat fixture dir: 4 files → 1 sample, 4 chromatograms sorted by RT."""
    handler = Handler.read_knauer(FIXTURES, mode="timecourse")

    assert len(handler.samples) == 1
    sample = handler.samples[0]
    assert sample.id == "knauer_txt"
    assert len(sample.chromatograms) == 4

    rts = [c.reaction_time for c in sample.chromatograms]
    assert rts == [0.0, 15.0, 30.0, 60.0]

    ids = [c.id for c in sample.chromatograms]
    assert ids == ["knauer_0_min", "knauer_15_min", "knauer_30_min", "knauer_60_min"]

    # first chromatogram = knauer_0_min.txt
    chrom = sample.chromatograms[0]
    assert chrom.sample_id == "knauer_txt"
    assert len(chrom.time) == len(chrom.signal)
    assert len(chrom.time) > 0

    # time is already in minutes — first value is 0
    assert chrom.time[0] == pytest.approx(0.0)
    # first signal value from fixture file
    assert chrom.signal[0] == pytest.approx(0.00073207623790949583, rel=1e-4)
    # no peaks (format has no peak table)
    assert chrom.peaks == []


def test_read_knauer_endpoint() -> None:
    """Endpoint mode: each file → own sample, no reaction time."""
    handler = Handler.read_knauer(FIXTURES, mode="endpoint")

    assert len(handler.samples) == 4
    sample_ids = sorted(s.id for s in handler.samples)
    assert sample_ids == ["knauer_0_min", "knauer_15_min", "knauer_30_min", "knauer_60_min"]

    for sample in handler.samples:
        assert len(sample.chromatograms) == 1
        chrom = sample.chromatograms[0]
        assert chrom.reaction_time is None
        assert chrom.id == sample.id


def test_read_knauer_not_a_directory() -> None:
    with pytest.raises(NotADirectoryError):
        Handler.read_knauer(FIXTURES / "knauer_0_min.txt")


def test_read_knauer_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Handler.read_knauer(tmp_path, mode="timecourse")
