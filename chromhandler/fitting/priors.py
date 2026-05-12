"""Controls-based prior construction for the skew-normal peak model.

This module turns a ``PreparedDataset`` plus its ``PeakAnnotation`` list
into a list of :class:`SkewNormalPriors`, one per peak.

All magic numbers and fallback heuristics live in :class:`PriorConfig`.
Users can override the config to change behaviour; defaults are tuned for
typical chromatographic data.

For ``artefact_doublet`` peaks, all artefact-related priors are derived
**directly from control traces** (samples with no analyte). For shape
quantities where only one control is available, scale fallbacks borrow
from the analyte's empirical population (same chromatographic system ->
same drift and shape variation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass(frozen=True)
class PriorConfig:
    """Centralised configuration for prior construction.

    All knobs in one place — users can override any field to change
    behaviour without touching the priors module itself.
    """

    # --- Distribution bounds (geometric / mathematical) ---
    gamma1_bound_fraction: float = 0.99
    sigma_low_n_points_per_fwhm: int = 8
    sigma_high_window_fraction: float = 6.0
    delta_low_dt_multiplier: float = 3.0
    delta_high_window_fraction: float = 2.0

    # --- Side check ---
    side_check_epsilon_dt_multiplier: float = 3.0

    # --- n=1 control fallbacks ---
    delta_scale_dt_multiplier_n1: float = 1.5
    log_A_artefact_min_scale: float = 0.2

    # --- Single-trace fallbacks (n_trace=1 in single-peak aggregation) ---
    log_sigma_scale_n1: float = 0.15
    gamma1_scale_n1: float = 0.20
    log_A_scale_n1_min: float = 0.10

    # --- Universal floors ---
    mu_scale_dt_floor_multiplier: float = 1.0


@dataclass(frozen=True)
class SkewNormalPriors:
    """Empirical priors for one peak window.

    Each field parameterizes exactly one NumPyro distribution in
    ``model.py`` per the distribution table at the top of the priors plan.
    ``_left_*`` fields are always populated; ``_right_*`` and ``Delta_*``
    are populated iff ``n_components == 2``.
    """

    n_components: int

    mu_left_loc: float
    mu_left_scale: float
    mu_left_low: float
    mu_left_high: float

    log_sigma_left_loc: float
    log_sigma_left_scale: float
    log_sigma_left_low: float
    log_sigma_left_high: float

    gamma1_left_loc: float
    gamma1_left_scale: float

    log_A_left_loc_per_trace: NDArray[np.float64]
    log_A_left_scale: float

    Delta_loc: float | None
    Delta_scale: float | None
    Delta_low: float | None
    Delta_high: float | None

    log_sigma_right_loc: float | None
    log_sigma_right_scale: float | None
    log_sigma_right_low: float | None
    log_sigma_right_high: float | None

    gamma1_right_loc: float | None
    gamma1_right_scale: float | None

    log_A_right_loc_per_trace: NDArray[np.float64] | None
    log_A_right_scale: float | None

    def __post_init__(self) -> None:
        right_fields = (
            self.Delta_loc, self.Delta_scale, self.Delta_low, self.Delta_high,
            self.log_sigma_right_loc, self.log_sigma_right_scale,
            self.log_sigma_right_low, self.log_sigma_right_high,
            self.gamma1_right_loc, self.gamma1_right_scale,
            self.log_A_right_loc_per_trace, self.log_A_right_scale,
        )
        if self.n_components == 1:
            if any(f is not None for f in right_fields):
                raise ValueError(
                    "Single-component priors require all right-component "
                    "fields (Delta_*, *_right_*) to be None."
                )
        elif self.n_components == 2:
            if any(f is None for f in right_fields):
                raise ValueError(
                    "For doublet peaks, all right-component fields are required "
                    "(Delta_*, *_right_*); got at least one None."
                )
        else:
            raise ValueError(f"n_components must be 1 or 2, got {self.n_components}.")
