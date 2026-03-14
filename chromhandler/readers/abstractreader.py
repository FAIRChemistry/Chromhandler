from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from chromhandler.model import Peak


class MetadataExtractionError(Exception):
    def __init__(self, message: str, suggestion: str | None = None) -> None:
        if suggestion:
            message += f"\n{suggestion}"
        super().__init__(message)


class UnitConsistencyError(Exception):
    def __init__(self, message: str, suggestion: str | None = None) -> None:
        if suggestion:
            message += f"\n{suggestion}"
        super().__init__(message)


class FileNotFoundInDirectoryError(Exception):
    def __init__(self, message: str, suggestion: str | None = None) -> None:
        if suggestion:
            message += f"\n{suggestion}"
        super().__init__(message)


@dataclass(frozen=True)
class ChromatogramData:
    """Raw chromatogram data returned by a reader before domain model construction."""

    signal: list[float]
    time: list[float]
    peaks: list[Peak] = field(default_factory=list)
    wavelength: float | None = None


@runtime_checkable
class AbstractReader(Protocol):
    """Protocol for single-file chromatogram readers.

    Implementors parse one instrument file and return a :class:`ChromatogramData`
    containing the raw signal/time arrays and any peaks extracted from the file.
    All contextual metadata (sample identity, reaction time, chromatogram_id) is
    provided by the caller and attached during :meth:`Handler.read_chromatogram`.

    Example::

        class MyReader:
            def read_file(
                self, path: Path, *, chromatogram_id: str
            ) -> ChromatogramData:
                ...
    """

    def read_file(self, path: Path, *, chromatogram_id: str) -> ChromatogramData: ...
