"""Tests for the generic Handler.read() auto-detecting entry point."""
from __future__ import annotations

from pathlib import Path

import pytest

from chromhandler.handler import Handler

FIXTURE_ROOT = Path(__file__).parent.parent.parent / "fixtures"


class TestHandlerReadDispatch:
    def test_reads_agilent(self) -> None:
        handler = Handler.read(FIXTURE_ROOT / "agilent", channel="FID1A.CH")
        assert len(handler.samples) > 0

    def test_reads_asm(self) -> None:
        handler = Handler.read(FIXTURE_ROOT / "asm")
        assert len(handler.samples) > 0

    def test_reads_knauer(self) -> None:
        handler = Handler.read(FIXTURE_ROOT / "knauer_txt")
        assert len(handler.samples) > 0

    def test_reads_shimadzu(self) -> None:
        handler = Handler.read(FIXTURE_ROOT / "shimadzu")
        assert len(handler.samples) > 0

    def test_channel_ignored_for_non_agilent(self) -> None:
        # channel kwarg must not raise for non-Agilent formats
        handler = Handler.read(FIXTURE_ROOT / "asm", channel="irrelevant")
        assert len(handler.samples) > 0

    def test_raises_not_a_directory(self, tmp_path: Path) -> None:
        fake = tmp_path / "notadir"
        fake.touch()
        with pytest.raises(NotADirectoryError):
            Handler.read(fake)

    def test_raises_on_unknown_format(self, tmp_path: Path) -> None:
        (tmp_path / "data.xyz").touch()
        with pytest.raises(ValueError, match="No reader"):
            Handler.read(tmp_path)

    def test_error_message_lists_found_content(self, tmp_path: Path) -> None:
        (tmp_path / "weird.csv").touch()
        with pytest.raises(ValueError, match=r"\.csv"):
            Handler.read(tmp_path)
