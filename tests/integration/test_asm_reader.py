from __future__ import annotations

from pathlib import Path

import pytest

from chromhandler.handler import Handler

FIXTURES = Path(__file__).parent.parent / "fixtures" / "asm"


def test_read_asm_timecourse_flat() -> None:
    """Flat fixture dir: 3 files → 1 sample with 3 chromatograms sorted by RT."""
    handler = Handler.read_asm(FIXTURES, mode="timecourse")

    assert len(handler.samples) == 1
    sample = handler.samples[0]
    assert sample.id == "asm"
    assert len(sample.chromatograms) == 3

    rts = [c.reaction_time for c in sample.chromatograms]
    assert rts == [20.0, 30.0, 60.0]

    # chromatogram IDs come from file stems
    ids = [c.id for c in sample.chromatograms]
    assert ids == ["CV6_20min", "CV6_30min", "CV6_60min"]

    # first chromatogram = CV6_20min.json
    chrom = sample.chromatograms[0]
    assert chrom.sample_id == "asm"
    assert len(chrom.signal) > 0
    assert len(chrom.time) == len(chrom.signal)
    assert len(chrom.peaks) > 0

    # first peak validated against CV6_20min.json values
    p = chrom.peaks[0]
    assert p.peak_start == pytest.approx(156.815 / 60, rel=1e-4)
    assert p.peak_end == pytest.approx(170.815 / 60, rel=1e-4)
    assert p.location.mean == pytest.approx(162.815 / 60, rel=1e-4)
    assert p.area.mean == pytest.approx(3820321.2235922245 * 60, rel=1e-4)
    assert p.amplitude == pytest.approx(78459.4296875, rel=1e-4)
    assert p.percent_area == pytest.approx(13.519827166781642, rel=1e-4)


def test_read_asm_endpoint() -> None:
    """Endpoint mode: each file → its own sample, no reaction time."""
    handler = Handler.read_asm(FIXTURES, mode="endpoint")

    assert len(handler.samples) == 3
    sample_ids = sorted(s.id for s in handler.samples)
    assert sample_ids == ["CV6_20min", "CV6_30min", "CV6_60min"]

    for sample in handler.samples:
        assert len(sample.chromatograms) == 1
        chrom = sample.chromatograms[0]
        assert chrom.reaction_time is None
        assert chrom.id == sample.id


def test_read_asm_not_a_directory() -> None:
    with pytest.raises(NotADirectoryError):
        Handler.read_asm(FIXTURES / "CV6_20min.json")


def test_read_asm_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Handler.read_asm(tmp_path, mode="timecourse")
