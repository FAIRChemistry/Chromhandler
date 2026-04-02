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

    # Half-width prior scale floor (log-space CV)
    w_prior_log_scale: float = 0.4

    # Area prior spread — S/N-dependent linear interpolation
    area_log_sigma_high_snr: float = 0.3   # tight for clear peaks (S/N > threshold_high)
    area_log_sigma_low_snr: float = 0.8    # wide for ambiguous peaks (S/N < threshold_low)
    area_snr_threshold_high: float = 10.0
    area_snr_threshold_low: float = 3.0

    # Artefact area
    area_art_log_sigma: float = 0.3        # shared artefact area CV ~30%
    area_art_trace_log_scale: float = 0.15  # per-trace artefact multiplicative noise

    # Separation priors (LogNormal in log-space)
    free_sep_loc_mult: float = 1.5         # typical separation in sigma units
    free_sep_log_sigma: float = 0.4

    art_sep_min_w_mult: float = 0.5        # min separation in half-width units
    art_sep_max_window_frac: float = 0.5


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
