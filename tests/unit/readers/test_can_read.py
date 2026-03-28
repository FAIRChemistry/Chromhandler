"""Tests for Reader.can_read() detection classmethods."""
from __future__ import annotations

import shutil
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

    def test_detects_nested_layout(self, tmp_path: Path) -> None:
        from chromhandler.readers.agilent import AgilentReader
        sub = tmp_path / "experiment"
        sub.mkdir()
        (sub / "sample.D").mkdir()
        assert AgilentReader.can_read(tmp_path) is True

    def test_hidden_subdir_not_probed(self, tmp_path: Path) -> None:
        from chromhandler.readers.agilent import AgilentReader
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "sample.D").mkdir()
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

    def test_detects_nested_layout(self, tmp_path: Path) -> None:
        from chromhandler.readers.asm import ASMReader
        sub = tmp_path / "sample_A"
        sub.mkdir()
        (sub / "data.json").touch()
        assert ASMReader.can_read(tmp_path) is True

    def test_hidden_subdir_not_probed(self, tmp_path: Path) -> None:
        from chromhandler.readers.asm import ASMReader
        hidden = tmp_path / ".DS_Store_dir"
        hidden.mkdir()
        (hidden / "data.json").touch()
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

    def test_detects_nested_layout(self, tmp_path: Path) -> None:
        from chromhandler.readers.knauer_txt import KnauerTXTReader
        sub = tmp_path / "sample_A"
        sub.mkdir()
        shutil.copy(FIXTURE_ROOT / "knauer_txt" / "knauer_0_min.txt", sub / "run.txt")
        assert KnauerTXTReader.can_read(tmp_path) is True


class TestShimadzuCanRead:
    def test_detects_shimadzu_fixture(self) -> None:
        from chromhandler.readers.shimadzu import ShimadzuReader
        assert ShimadzuReader.can_read(FIXTURE_ROOT / "shimadzu") is True

    def test_rejects_knauer_dir(self) -> None:
        from chromhandler.readers.shimadzu import ShimadzuReader
        assert ShimadzuReader.can_read(FIXTURE_ROOT / "knauer_txt") is False

    def test_rejects_empty_dir(self, tmp_path: Path) -> None:
        from chromhandler.readers.shimadzu import ShimadzuReader
        assert ShimadzuReader.can_read(tmp_path) is False

    def test_rejects_incomplete_sections(self, tmp_path: Path) -> None:
        from chromhandler.readers.shimadzu import ShimadzuReader
        # Only first two sections present — not a valid Shimadzu file
        (tmp_path / "data.txt").write_text("[Header]\n[File Information]\n")
        assert ShimadzuReader.can_read(tmp_path) is False

    def test_detects_nested_layout(self, tmp_path: Path) -> None:
        from chromhandler.readers.shimadzu import ShimadzuReader
        sub = tmp_path / "sample_A"
        sub.mkdir()
        shutil.copy(FIXTURE_ROOT / "shimadzu" / "P0-0.0_min.txt", sub / "run.txt")
        assert ShimadzuReader.can_read(tmp_path) is True
