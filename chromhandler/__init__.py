from __future__ import annotations

import warnings as _warnings
from typing import Any

from .enzymeml import to_enzymeml
from .handler import Handler
from .molecule import Molecule
from .protein import Protein


# Backward compatibility with deprecation warning
class ChromAnalyzer(Handler):
    """
    Deprecated: ChromAnalyzer has been renamed to Handler.
    Please use Handler instead.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _warnings.warn(
            "ChromAnalyzer is deprecated and will be removed in version 1.0.0. "
            "Use 'Handler' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


# Store reference to ChromAnalyzer before module-level __getattr__
_ChromAnalyzer = ChromAnalyzer

# Remove ChromAnalyzer from global namespace so __getattr__ is called
del ChromAnalyzer

# Track if we've already warned about ChromAnalyzer import
_chromanalyzer_import_warned = False


def __getattr__(name: str) -> Handler:
    """Module-level __getattr__ to issue deprecation warnings on import."""
    global _chromanalyzer_import_warned

    if name == "ChromAnalyzer":
        if not _chromanalyzer_import_warned:
            _warnings.warn(
                "ChromAnalyzer is deprecated and will be removed in version 1.0.0. "
                "Use 'Handler' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            _chromanalyzer_import_warned = True
        return _ChromAnalyzer  # type: ignore
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["Handler", "ChromAnalyzer", "Molecule", "Protein", "to_enzymeml"]

__version__ = "0.10.2"
