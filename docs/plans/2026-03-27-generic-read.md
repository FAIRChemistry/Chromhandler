# Generic `Handler.read()` Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `Handler.read(path)` that auto-detects the instrument format and delegates to the correct `read_*` classmethod.

**Architecture:** Each of the four active readers gains a `can_read(path) -> bool` classmethod that owns its own detection logic. A `READERS` registry in `readers/__init__.py` defines probe order. `Handler.read()` iterates the registry, finds the first match, and dispatches to the matching `read_*` method. Agilent-specific kwargs (`channel`, `wavelength`) are passed through and ignored for non-Agilent formats.

**Tech Stack:** Python stdlib only for detection (`pathlib.Path`, file reads). No new dependencies. Tests use `pytest` + `tmp_path` fixtures. Lint/type checks via `uv run ruff check` and `uv run pyright`.

---

## Task 1: Add `can_read` to `AbstractReader` protocol

**Files:**
- Modify: `chromhandler/readers/abstractreader.py`

**Step 1: Add the classmethod to the Protocol**

Open `chromhandler/readers/abstractreader.py`. The `Path` import is already in the `TYPE_CHECKING` block. Add `can_read` right before `read_file`:

```python
class AbstractReader(Protocol):
    """..."""  # docstring unchanged

    @classmethod
    def can_read(cls, path: Path) -> bool:
        """Return True if this reader can handle the contents of *path*."""
        ...

    def read_file(
        self,
        path: Path,
        *,
        chromatogram_id: str,
        sample_id: str,
        reaction_time: float | None = None,
    ) -> Chromatogram: ...
```

**Step 2: Lint and type-check**

```bash
uv run ruff check chromhandler/readers/abstractreader.py
uv run pyright chromhandler/readers/abstractreader.py
```
Expected: 0 errors.

**Step 3: Commit**

```bash
git add chromhandler/readers/abstractreader.py
git commit -m "feat: add can_read classmethod to AbstractReader protocol"
```

---

## Task 2: `AgilentReader.can_read`

**Files:**
- Modify: `chromhandler/readers/agilent.py`
- Test: `tests/unit/readers/test_can_read.py` (create)

**Step 1: Write the failing test**

Create `tests/unit/readers/test_can_read.py`:

```python
"""Tests for Reader.can_read() detection classmethods."""
from __future__ import annotations

from pathlib import Path

import pytest

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
```

**Step 2: Run test — confirm FAIL**

```bash
uv run pytest tests/unit/readers/test_can_read.py -v
```
Expected: `AttributeError: type object 'AgilentReader' has no attribute 'can_read'`

**Step 3: Implement `AgilentReader.can_read`**

Open `chromhandler/readers/agilent.py`. Add after the `__init__` method (before `read_file`):

```python
@classmethod
def can_read(cls, path: Path) -> bool:
    """Return True if *path* contains at least one ``.D`` sub-directory."""
    try:
        return any(p.is_dir() and p.name.endswith(".D") for p in path.iterdir())
    except OSError:
        return False
```

**Step 4: Run test — confirm PASS**

```bash
uv run pytest tests/unit/readers/test_can_read.py::TestAgilentCanRead -v
```
Expected: 3 passed.

**Step 5: Lint and type-check**

```bash
uv run ruff check chromhandler/readers/agilent.py
uv run pyright chromhandler/readers/agilent.py
```
Expected: 0 errors.

**Step 6: Commit**

```bash
git add chromhandler/readers/agilent.py tests/unit/readers/test_can_read.py
git commit -m "feat: add AgilentReader.can_read"
```

---

## Task 3: `ASMReader.can_read`

**Files:**
- Modify: `chromhandler/readers/asm.py`
- Test: `tests/unit/readers/test_can_read.py`

**Step 1: Add tests to `test_can_read.py`**

Append this class to `tests/unit/readers/test_can_read.py`:

```python
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
```

**Step 2: Run — confirm FAIL**

```bash
uv run pytest tests/unit/readers/test_can_read.py::TestASMCanRead -v
```
Expected: `AttributeError`

**Step 3: Implement `ASMReader.can_read`**

Open `chromhandler/readers/asm.py`. Add after `__init__` / before `read_file`:

```python
@classmethod
def can_read(cls, path: Path) -> bool:
    """Return True if *path* contains at least one ``.json`` file."""
    try:
        return any(p.is_file() and p.suffix == ".json" for p in path.iterdir())
    except OSError:
        return False
```

**Step 4: Run — confirm PASS**

```bash
uv run pytest tests/unit/readers/test_can_read.py::TestASMCanRead -v
```
Expected: 3 passed.

**Step 5: Lint and type-check**

```bash
uv run ruff check chromhandler/readers/asm.py
uv run pyright chromhandler/readers/asm.py
```

**Step 6: Commit**

```bash
git add chromhandler/readers/asm.py tests/unit/readers/test_can_read.py
git commit -m "feat: add ASMReader.can_read"
```

---

## Task 4: `KnauerTXTReader.can_read`

**Files:**
- Modify: `chromhandler/readers/knauer_txt.py`
- Test: `tests/unit/readers/test_can_read.py`

**Step 1: Add tests**

Append to `tests/unit/readers/test_can_read.py`:

```python
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
```

**Step 2: Run — confirm FAIL**

```bash
uv run pytest tests/unit/readers/test_can_read.py::TestKnauerCanRead -v
```

**Step 3: Implement `KnauerTXTReader.can_read`**

Detection: find first `.txt`, read its first 5 lines, check the **first word** of each line (split on whitespace) is exactly: `Analyst`, `SampleID`, `Sample`, `Sample`, `Range`.

Open `chromhandler/readers/knauer_txt.py`. Add this classmethod:

```python
_KNAUER_HEADER_WORDS = ("Analyst", "SampleID", "Sample", "Sample", "Range")

@classmethod
def can_read(cls, path: Path) -> bool:
    """Return True if *path* contains a ClarityChrom (Knauer) TXT file."""
    try:
        txt_files = [p for p in path.iterdir() if p.is_file() and p.suffix == ".txt"]
        if not txt_files:
            return False
        lines = txt_files[0].read_text(encoding="utf-8", errors="ignore").splitlines()
        if len(lines) < 5:
            return False
        return all(
            lines[i].split()[0] == word
            for i, word in enumerate(_KNAUER_HEADER_WORDS)
            if lines[i].split()
        )
    except OSError:
        return False
```

**Step 4: Run — confirm PASS**

```bash
uv run pytest tests/unit/readers/test_can_read.py::TestKnauerCanRead -v
```
Expected: 4 passed.

**Step 5: Lint and type-check**

```bash
uv run ruff check chromhandler/readers/knauer_txt.py
uv run pyright chromhandler/readers/knauer_txt.py
```

**Step 6: Commit**

```bash
git add chromhandler/readers/knauer_txt.py tests/unit/readers/test_can_read.py
git commit -m "feat: add KnauerTXTReader.can_read"
```

---

## Task 5: `ShimadzuReader.can_read`

**Files:**
- Modify: `chromhandler/readers/shimadzu.py`
- Test: `tests/unit/readers/test_can_read.py`

**Step 1: Add tests**

Append to `tests/unit/readers/test_can_read.py`:

```python
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
```

**Step 2: Run — confirm FAIL**

```bash
uv run pytest tests/unit/readers/test_can_read.py::TestShimadzuCanRead -v
```

**Step 3: Implement `ShimadzuReader.can_read`**

Detection: find first `.txt`, scan the whole file checking that these four section headers appear **in order** (each must be found after the previous one):

```
[Header]
[File Information]
[Sample Information]
[Original Files]
```

Open `chromhandler/readers/shimadzu.py`. Add:

```python
_SHIMADZU_SECTIONS = ("[Header]", "[File Information]", "[Sample Information]", "[Original Files]")

@classmethod
def can_read(cls, path: Path) -> bool:
    """Return True if *path* contains a Shimadzu LabSolutions TXT file."""
    try:
        txt_files = [p for p in path.iterdir() if p.is_file() and p.suffix == ".txt"]
        if not txt_files:
            return False
        lines = txt_files[0].read_text(encoding="utf-8", errors="ignore").splitlines()
        needle = 0
        for line in lines:
            if needle < len(_SHIMADZU_SECTIONS) and line.strip() == _SHIMADZU_SECTIONS[needle]:
                needle += 1
        return needle == len(_SHIMADZU_SECTIONS)
    except OSError:
        return False
```

**Step 4: Run — confirm PASS**

```bash
uv run pytest tests/unit/readers/test_can_read.py::TestShimadzuCanRead -v
```
Expected: 4 passed.

**Step 5: Lint and type-check**

```bash
uv run ruff check chromhandler/readers/shimadzu.py
uv run pyright chromhandler/readers/shimadzu.py
```

**Step 6: Run full can_read suite**

```bash
uv run pytest tests/unit/readers/test_can_read.py -v
```
Expected: all 14 tests pass.

**Step 7: Commit**

```bash
git add chromhandler/readers/shimadzu.py tests/unit/readers/test_can_read.py
git commit -m "feat: add ShimadzuReader.can_read"
```

---

## Task 6: `READERS` registry in `readers/__init__.py`

**Files:**
- Modify: `chromhandler/readers/__init__.py`

**Step 1: Add the registry**

The current `__init__.py` is nearly empty (one line). Replace its entire contents with:

```python
from __future__ import annotations

from chromhandler.readers.agilent import AgilentReader
from chromhandler.readers.asm import ASMReader
from chromhandler.readers.knauer_txt import KnauerTXTReader
from chromhandler.readers.shimadzu import ShimadzuReader

# Ordered probe list for Handler.read() auto-detection.
# Agilent and ASM are checked first (directory scan, no file I/O).
# Knauer precedes Shimadzu (5-line sniff vs. full-file scan); their
# signatures are mutually exclusive so order does not affect correctness.
READERS: list[type] = [
    AgilentReader,
    ASMReader,
    KnauerTXTReader,
    ShimadzuReader,
]

__all__ = ["AgilentReader", "ASMReader", "KnauerTXTReader", "ShimadzuReader", "READERS"]
```

**Step 2: Lint and type-check**

```bash
uv run ruff check chromhandler/readers/__init__.py
uv run pyright chromhandler/readers/__init__.py
```

**Step 3: Verify existing tests still pass**

```bash
uv run pytest -m "not readers and not fitting" -q
```
Expected: all pass.

**Step 4: Commit**

```bash
git add chromhandler/readers/__init__.py
git commit -m "feat: add READERS registry to readers/__init__.py"
```

---

## Task 7: `Handler.read()` classmethod

**Files:**
- Modify: `chromhandler/handler.py`
- Test: `tests/unit/handler/test_read_generic.py` (create)

**Step 1: Write the failing tests**

Create `tests/unit/handler/test_read_generic.py`:

```python
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
```

**Step 2: Run — confirm FAIL**

```bash
uv run pytest tests/unit/handler/test_read_generic.py -v
```
Expected: `AttributeError: type object 'Handler' has no attribute 'read'`

**Step 3: Implement `Handler.read()`**

Open `chromhandler/handler.py`. Add the following classmethod after `read_agilent` (around line 490). The necessary imports (`Path`, `Literal`) are already present at the top of the file.

```python
@classmethod
def read(
    cls,
    path: Path | str,
    *,
    mode: Literal["timecourse", "endpoint"] = "timecourse",
    channel: str | None = None,
    wavelength: float | None = None,
) -> Handler:
    """Auto-detect instrument format and read chromatography data.

    Tries each registered reader in order (Agilent → ASM → Knauer →
    Shimadzu) and delegates to the matching ``read_*`` classmethod.

    Agilent-specific kwargs (``channel``, ``wavelength``) are forwarded
    only when the Agilent format is detected; they are silently ignored
    for all other formats.

    Args:
        path: Root directory containing chromatography data.
        mode: ``"timecourse"`` (default) or ``"endpoint"``.
        channel: Agilent detector-file name (e.g. ``"FID1A.CH"``).
        wavelength: Agilent DAD wavelength in nm.

    Returns:
        A fully populated :class:`Handler`.

    Raises:
        NotADirectoryError: If *path* is not a directory.
        ValueError: If no registered reader recognises the contents of
            *path*.
    """
    from .readers import READERS, AgilentReader

    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(f"'{root}' is not a directory.")

    for reader_cls in READERS:
        if reader_cls.can_read(root):
            if reader_cls is AgilentReader:
                return cls.read_agilent(
                    root, mode=mode, channel=channel, wavelength=wavelength
                )
            dispatch = {
                "ASMReader": cls.read_asm,
                "KnauerTXTReader": cls.read_knauer,
                "ShimadzuReader": cls.read_shimadzu,
            }
            return dispatch[reader_cls.__name__](root, mode=mode)

    # Build a helpful error listing what was actually found.
    try:
        found = sorted({p.suffix or p.name for p in root.iterdir()})
    except OSError:
        found = []
    found_str = ", ".join(found) if found else "nothing"
    raise ValueError(
        f"No reader recognised the contents of '{root}' (found: {found_str}). "
        "Use a specific read_* method: read_agilent, read_asm, read_knauer, read_shimadzu."
    )
```

**Step 4: Run tests — confirm PASS**

```bash
uv run pytest tests/unit/handler/test_read_generic.py -v
```
Expected: 8 passed.

**Step 5: Lint and type-check**

```bash
uv run ruff check chromhandler/handler.py
uv run pyright chromhandler/handler.py
```
Expected: 0 errors.

**Step 6: Run full fast suite**

```bash
uv run pytest -m "not readers and not fitting" -q
```
Expected: all pass.

**Step 7: Commit**

```bash
git add chromhandler/handler.py tests/unit/handler/test_read_generic.py
git commit -m "feat: add Handler.read() generic auto-detecting entry point"
```

---

## Final Verification

```bash
uv run pytest tests/unit/readers/test_can_read.py tests/unit/handler/test_read_generic.py -v
uv run ruff check chromhandler/readers/ chromhandler/handler.py
uv run pyright chromhandler/readers/ chromhandler/handler.py
```

All checks must pass before the branch is considered complete.
