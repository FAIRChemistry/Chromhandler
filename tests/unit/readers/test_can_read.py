"""Tests for Reader.can_read() detection classmethods."""
from __future__ import annotations

from pathlib import Path

FIXTURE_ROOT = Path(__file__).parent.parent.parent / "fixtures"


class TestAgilentCanRead:
    def test_detects_agilent_fixture(self) -> None:
        from chromhandler.readers.agilent import AgilentReader
        assert AgilentReader.can_read(FIXTURE_ROOT / "agilent") is True

    def test_rejects_asm_dir(self) -> None:
        from chromhandler.readers.agilent import AgilentReader
        assert AgilentReader.can_read(FIXTURE_ROOT / "asm") is False

    def test_rejects_empty_dir(self, tmp_path: Path) -> None:
        from chromhandler.readers.agilent import AgilentReader
        assert AgilentReader.can_read(tmp_path) is False


class TestASMCanRead:
    def test_detects_asm_fixture(self) -> None:
        from chromhandler.readers.asm import ASMReader
        assert ASMReader.can_read(FIXTURE_ROOT / "asm") is True

    def test_rejects_agilent_dir(self) -> None:
        from chromhandler.readers.asm import ASMReader
        assert ASMReader.can_read(FIXTURE_ROOT / "agilent") is False

    def test_rejects_empty_dir(self, tmp_path: Path) -> None:
        from chromhandler.readers.asm import ASMReader
        assert ASMReader.can_read(tmp_path) is False


class TestKnauerCanRead:
    def test_detects_knauer_fixture(self) -> None:
        from chromhandler.readers.knauer_txt import KnauerTXTReader
        assert KnauerTXTReader.can_read(FIXTURE_ROOT / "knauer_txt") is True

    def test_rejects_shimadzu_dir(self) -> None:
        from chromhandler.readers.knauer_txt import KnauerTXTReader
        assert KnauerTXTReader.can_read(FIXTURE_ROOT / "shimadzu") is False

    def test_rejects_empty_dir(self, tmp_path: Path) -> None:
        from chromhandler.readers.knauer_txt import KnauerTXTReader
        assert KnauerTXTReader.can_read(tmp_path) is False

    def test_rejects_dir_without_txt(self, tmp_path: Path) -> None:
        from chromhandler.readers.knauer_txt import KnauerTXTReader
        (tmp_path / "data.json").touch()
        assert KnauerTXTReader.can_read(tmp_path) is False

    def test_rejects_blank_header_lines(self, tmp_path: Path) -> None:
        from chromhandler.readers.knauer_txt import KnauerTXTReader
        (tmp_path / "data.txt").write_text("\n\n\n\n\n")
        assert KnauerTXTReader.can_read(tmp_path) is False
