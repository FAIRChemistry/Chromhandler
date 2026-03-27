from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from chromhandler.model import Chromatogram


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


class AbstractReader(Protocol):
    """Protocol for single-file chromatogram readers.

    Implementors parse one instrument file and return a fully constructed
    :class:`~chromhandler.model.Chromatogram`.  All contextual metadata
    (``sample_id``, ``reaction_time``) is supplied by the caller so that the
    reader stays stateless and focused on file parsing.

    Example::

        class MyReader:
            def read_file(
                self,
                path: Path,
                *,
                chromatogram_id: str,
                sample_id: str,
                reaction_time: float | None = None,
            ) -> Chromatogram:
                ...
    """

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
