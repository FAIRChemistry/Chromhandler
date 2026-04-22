"""Data types and schemas for peak fitting module.

Defines:
- :class:`ModelHyperparams`: Hyperparameter configuration
- :data:`PeakMode`: Peak mode enumeration
- Mode query functions: :func:`peak_component_count`, :func:`peak_is_doublet_mode`, etc.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

PeakMode = Literal["single", "artefact_doublet", "free_doublet"]
PEAK_MODE_TO_CODE: dict[str, int] = {
    "single": 0,
    "artefact_doublet": 1,
    "free_doublet": 2,
}


@dataclasses.dataclass(frozen=True)
class ModelHyperparams:
    """Tunable hyperparameters for ``model.model()``.

    All values have research-validated defaults.  Pass a custom instance to
    :class:`~chromhandler.fitting.Fitter` to override for sensitivity
    analysis or domain-specific tuning.
    """

    # Area prior log-space SD (multiplicative uncertainty on prior centre).
    # e^0.4 ~ 1.5 — the 68 % CI spans roughly /1.5 to *1.5 around the
    # Gaussian-approximation area estimate.  Fixed rather than S/N-adaptive:
    # at high S/N the likelihood dominates regardless; at low S/N a fixed
    # moderate width is no worse than an empirical-Bayes adaptive one.
    area_log_sigma: float = 0.4

    # Artefact area — hierarchical model: shared mean + per-trace offset
    area_art_log_sigma: float = 0.3  # log-space SD on the population mean (hyperprior width)
    area_art_trace_log_scale: float = 0.15  # log-space SD of per-trace deviations from mean
    # Interpretation: at 0.15, each trace can deviate ~+-15% (1 SD) from the shared mean.
    # Larger values = weaker pooling (more per-trace freedom); smaller = stricter sharing.

    # Free-doublet separation prior (LogNormal in log-space)
    free_sep_log_sigma: float = 0.4


def peak_component_count(mode: str) -> int:
    """Return the number of mixture components implied by a peak mode."""
    return 1 if mode == "single" else 2


def peak_is_doublet_mode(mode: str) -> bool:
    """Return True for all two-component peak modes."""
    return peak_component_count(mode) == 2


def peak_is_artefact_mode(mode: str) -> bool:
    """Return True when the peak uses the artefact-doublet branch."""
    return mode == "artefact_doublet"


def peak_is_free_mode(mode: str) -> bool:
    """Return True when the peak uses the free-doublet branch."""
    return mode == "free_doublet"
