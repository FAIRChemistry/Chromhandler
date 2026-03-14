"""Fitting-internal helpers and data types.

:class:`PeakAnnotation` and :class:`BaselineAnnotation` have been consolidated
into :mod:`chromhandler.annotations` and are re-exported from here for
backwards compatibility within the fitting subpackage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation

# ---------------------------------------------------------------------------
# Mode constants and helpers
# ---------------------------------------------------------------------------

PeakMode = Literal["single", "artefact_doublet", "free_doublet"]
PEAK_MODE_TO_CODE: dict[str, int] = {
    "single": 0,
    "artefact_doublet": 1,
    "free_doublet": 2,
}


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


# ---------------------------------------------------------------------------
# Supporting dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeakPriorHints:
    """Prior parameter hints for one logical peak, used to inform MCMC initialisation."""

    mu_loc: float
    mu_scale: float
    sigma_loc: float
    sigma_scale: float
    alpha_loc: float
    alpha_scale: float
    area_loc: float
    area_scale: float
    trace_count: int
    sep_est: float = 0.0  # data-driven component separation estimate for doublets


@dataclass
class ChromatogramRecord:
    """One chromatogram with optional annotations.

    Attributes:
        sample_id: Parent sample identifier.
        chromatogram_id: Unique chromatogram identifier.
        time: 1D retention-time axis.
        signal: 1D signal vector.
        peaks: Per-chromatogram peak annotations.
        baselines: Per-chromatogram baseline annotations.
    """

    sample_id: str
    chromatogram_id: str
    time: list[float]
    signal: list[float]
    peaks: list[PeakAnnotation] = field(default_factory=list)
    baselines: list[BaselineAnnotation] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.time) != len(self.signal):
            raise ValueError(
                "time and signal must have same length, got "
                f"{len(self.time)} and {len(self.signal)}"
            )

    def add_peak(
        self,
        molecule_id: str,
        rt_min: float,
        rt_max: float,
        mode: PeakMode,
        artefact_side: Literal["left", "right"] | None = None,
    ) -> None:
        """Add a peak annotation to the chromatogram (replace by molecule_id if exists)."""
        if rt_min < min(self.time) or rt_max > max(self.time):
            raise ValueError(
                "rt_min and rt_max must be within the time range, got "
                f"rt_min={rt_min}, rt_max={rt_max}, "
                f"time range={min(self.time)} to {max(self.time)}"
            )
        peak = PeakAnnotation(
            molecule_id=molecule_id,
            rt_min=rt_min,
            rt_max=rt_max,
            mode=mode,
            artefact_side=artefact_side,
        )
        for i, p in enumerate(self.peaks):
            if p.molecule_id == molecule_id:
                self.peaks[i] = peak
                return
        self.peaks.append(peak)

    def add_baseline(self, rt_min: float, rt_max: float) -> None:
        """Add a baseline annotation (no duplicates allowed)."""
        if rt_min < min(self.time) or rt_max > max(self.time):
            raise ValueError(
                "rt_min and rt_max must be within the time range, got "
                f"rt_min={rt_min}, rt_max={rt_max}, "
                f"time range={min(self.time)} to {max(self.time)}"
            )
        if rt_min >= rt_max:
            raise ValueError(
                f"rt_min must be less than rt_max, got rt_min={rt_min}, rt_max={rt_max}"
            )
        baseline = BaselineAnnotation(rt_min=rt_min, rt_max=rt_max)
        for b in self.baselines:
            if b.rt_min == rt_min and b.rt_max == rt_max:
                raise ValueError(
                    f"baseline already exists, got rt_min={rt_min}, rt_max={rt_max}"
                )
        self.baselines.append(baseline)


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------


def get_chromatogram_tensor(chromatograms: list[ChromatogramRecord]) -> jnp.ndarray:
    signal_tensor = []
    for chromatogram in chromatograms:
        signal_tensor.append(chromatogram.signal)
    return jnp.array(signal_tensor).reshape(len(chromatograms), -1)


def stack_and_pad_signal(
    x_lists: list[list[float]], y_lists: list[list[float]]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Stack and pad data lists into a single tensor (NaN-padded to equal length)."""
    if len(x_lists) != len(y_lists):
        raise ValueError("x_lists and y_lists must have the same length")

    max_length = max(
        max(len(x) for x in x_lists),
        max(len(y) for y in y_lists),
    )
    padded_x = [x + [float("nan")] * (max_length - len(x)) for x in x_lists]
    padded_y = [y + [float("nan")] * (max_length - len(y)) for y in y_lists]

    return jnp.array(padded_x), jnp.array(padded_y)


def region_to_mask(low: float, high: float, time: jnp.ndarray) -> jnp.ndarray:
    """Mask True for all time points in [low, high]."""
    return (time >= low) & (time <= high)


def baseline_to_mask(
    baselines: list[BaselineAnnotation], time: jnp.ndarray
) -> jnp.ndarray:
    """Boolean mask True for time points in any baseline region."""
    if not baselines:
        return jnp.zeros(time.shape, dtype=bool)
    masks = jnp.stack([region_to_mask(b.rt_min, b.rt_max, time) for b in baselines])
    return jnp.any(masks, axis=0)


def peaks_to_mask(peaks: list[PeakAnnotation], time: jnp.ndarray) -> jnp.ndarray:
    """Boolean mask True for time points in any peak region.

    Returns shape ``(n_peaks, n_chromatograms, n_timepoints)``.
    """
    peak_centers = jnp.array([(p.rt_min + p.rt_max) / 2 for p in peaks])
    sorted_indices = jnp.argsort(peak_centers)
    sorted_peaks = [peaks[i] for i in sorted_indices]
    return jnp.stack([region_to_mask(p.rt_min, p.rt_max, time) for p in sorted_peaks])
