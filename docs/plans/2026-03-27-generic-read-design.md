# Design: `Handler.read()` — Generic Auto-Detecting Entry Point

**Date:** 2026-03-27
**Status:** Approved

---

## Problem

`Handler` exposes one classmethod per instrument format (`read_asm`, `read_knauer`,
`read_shimadzu`, `read_agilent`).  Users must know which reader to call.  A generic
`Handler.read(path)` that auto-detects the format would lower friction, especially
when writing format-agnostic pipelines.

---

## Scope

Four readers are in scope for auto-detection:

| Reader | Format |
|---|---|
| `AgilentReader` | `.D` sub-directories (rainbow) |
| `ASMReader` | `.json` files (Allotrope Simple Model) |
| `KnauerTXTReader` | `.txt` (ClarityChrom/Knauer HPLC ASCII export) |
| `ShimadzuReader` | `.txt` (LabSolutions ASCII export) |

`ThermoTX0Reader` and `ChromeleonReader` are legacy and excluded.

---

## Detection Strategy

### `can_read(path: Path) -> bool` classmethod on each reader

Detection logic is owned by the reader itself, keeping `Handler.read()` thin.
`AbstractReader` protocol gains the new `can_read` classmethod signature.

#### AgilentReader
Directory-scan only (no I/O): returns `True` if any entry inside `path` is a
`.D` sub-directory.

#### ASMReader
Directory-scan only: returns `True` if any `.json` file is present directly under
`path`.

#### KnauerTXTReader
Finds the first `.txt` file; reads its first 5 lines; checks that the **first word**
of each line (split on whitespace) matches exactly:

```
line 0 → "Analyst"
line 1 → "SampleID"
line 2 → "Sample"
line 3 → "Sample"
line 4 → "Range"
```

#### ShimadzuReader
Finds the first `.txt` file; scans the full file checking that these section
headers appear **in order** (each on its own line, exact string match):

```
[Header]
[File Information]
[Sample Information]
[Original Files]
```

Returns `True` only when all four are found in sequence.

---

## Registry

```python
# chromhandler/readers/__init__.py
READERS: list[type] = [
    AgilentReader,    # .D dirs  — directory scan, cheapest
    ASMReader,        # .json    — directory scan, cheap
    KnauerTXTReader,  # .txt     — 5-line header sniff
    ShimadzuReader,   # .txt     — full-file section-order sniff
]
```

Agilent and ASM are tried first (zero file I/O).  Knauer precedes Shimadzu because
its sniff is cheaper (5 lines vs. full file), and their signatures are mutually
exclusive so order does not affect correctness.

---

## `Handler.read()` Signature

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
```

- `mode` is forwarded to every specific `read_*` method.
- `channel` and `wavelength` are forwarded **only** for `AgilentReader`; silently
  ignored for other formats (allows reuse across instrument types without errors).

### Dispatch

```python
_DISPATCH = {
    AgilentReader:   lambda: cls.read_agilent(path, mode=mode, channel=channel, wavelength=wavelength),
    ASMReader:       lambda: cls.read_asm(path, mode=mode),
    KnauerTXTReader: lambda: cls.read_knauer(path, mode=mode),
    ShimadzuReader:  lambda: cls.read_shimadzu(path, mode=mode),
}
```

### Error handling

- `path` not a directory → `NotADirectoryError`
- No reader matches → `ValueError` listing what was found (extensions present,
  whether `.D` dirs exist) so the user understands why detection failed and which
  specific `read_*` method to call instead.

---

## Files Changed

| File | Change |
|---|---|
| `chromhandler/readers/abstractreader.py` | Add `can_read(path) -> bool` to protocol |
| `chromhandler/readers/agilent.py` | Add `AgilentReader.can_read()` |
| `chromhandler/readers/asm.py` | Add `ASMReader.can_read()` |
| `chromhandler/readers/knauer_txt.py` | Add `KnauerTXTReader.can_read()` |
| `chromhandler/readers/shimadzu.py` | Add `ShimadzuReader.can_read()` |
| `chromhandler/readers/__init__.py` | Add `READERS` registry |
| `chromhandler/handler.py` | Add `Handler.read()` classmethod |
| `tests/unit/readers/test_can_read.py` | Unit tests for all four `can_read()` impls |
| `tests/unit/handler/test_read_generic.py` | Integration tests for `Handler.read()` |
