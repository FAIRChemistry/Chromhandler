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
