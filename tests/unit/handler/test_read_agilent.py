"""Handler-level tests for Handler.read_agilent()."""

from __future__ import annotations

from pathlib import Path

import pytest

from chromhandler.handler import Handler

FIXTURE_DIR = Path(__file__).parent.parent.parent / "fixtures" / "agilent"


@pytest.mark.readers
class TestHandlerReadAgilent:
    def test_endpoint_mode_loads_all_samples(self) -> None:
        handler = Handler.read_agilent(FIXTURE_DIR, mode="endpoint", channel="FID1A.CH")
        assert len(handler.samples) == 3
        for sample in handler.samples:
            assert len(sample.chromatograms) == 1
            chrom = sample.chromatograms[0]
            assert len(chrom.signal) > 0
            assert len(chrom.time) == len(chrom.signal)
            assert chrom.peaks == []

    def test_timecourse_flat_layout(self) -> None:
        handler = Handler.read_agilent(FIXTURE_DIR, mode="timecourse", channel="FID1A.CH")
        assert len(handler.samples) == 1
        sample = handler.samples[0]
        assert sample.id == FIXTURE_DIR.name
        assert len(sample.chromatograms) == 3
        rts: list[float] = []
        for c in sample.chromatograms:
            assert c.reaction_time is not None
            rts.append(c.reaction_time)
        assert rts == sorted(rts)

    def test_raises_on_not_a_directory(self, tmp_path: Path) -> None:
        fake_file = tmp_path / "not_a_dir"
        fake_file.touch()
        with pytest.raises(NotADirectoryError):
            Handler.read_agilent(fake_file, channel="FID1A.CH")

    def test_raises_when_no_d_dirs(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"\.D"):
            Handler.read_agilent(tmp_path, mode="endpoint", channel="FID1A.CH")

    def test_endpoint_sample_ids_match_dir_stems(self) -> None:
        handler = Handler.read_agilent(FIXTURE_DIR, mode="endpoint", channel="FID1A.CH")
        stems = {p.stem for p in FIXTURE_DIR.iterdir() if p.name.endswith(".D")}
        sample_ids = {s.id for s in handler.samples}
        assert sample_ids == stems
