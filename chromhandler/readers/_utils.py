"""Shared utilities for reader auto-detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def find_representative_file(path: Path, suffix: str) -> Path | None:
    """Return the first file with *suffix* at top-level or one level deep.

    Hidden entries (names starting with ``'.'``) are skipped at every level,
    which avoids probing macOS ``.DS_Store``, ``.git``, and similar artefacts.

    Returns ``None`` if no matching file is found.
    """
    try:
        for p in sorted(path.iterdir()):
            if p.is_file() and p.suffix == suffix and not p.name.startswith("."):
                return p
        for sub in sorted(
            p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")
        ):
            try:
                for p in sorted(sub.iterdir()):
                    if p.is_file() and p.suffix == suffix and not p.name.startswith("."):
                        return p
            except OSError:  # noqa: PERF203
                continue
    except OSError:
        pass
    return None
