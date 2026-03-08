"""Data classes for chromatography fitting."""

from dataclasses import dataclass, field
from typing import Literal

import jax.numpy as jnp


@dataclass(frozen=True)
class PeakAnnotation:
    """Per-spectrum peak annotation defining a single skew-normal component.

    Attributes:
        name: Logical peak name.
        low: Broad lower bound used for initialization and diagnostics.
        high: Broad upper bound used for initialization and diagnostics.
        shoulder: Optional side where a shoulder peak is expected.
    """

    name: str
    low: float
    high: float
    shoulder: Literal["left", "right"] | None = None
    separation: float | None = None

    def __post_init__(self) -> None:
        if float(self.high) <= float(self.low):
            raise ValueError(
                f"Invalid annotation bounds for `{self.name}`: low={self.low}, high={self.high}"
            )
        if self.shoulder not in (None, "left", "right"):
            raise ValueError(
                "shoulder must be one of None, 'left', or 'right', "
                f"got {self.shoulder!r}."
            )
        if self.shoulder is not None and self.separation is None:
            raise ValueError(
                f"Peak '{self.name}': 'separation' is required when 'shoulder' is set. "
                "Provide the full inter-apex distance from main peak to shoulder peak "
                "(e.g. PeakAnnotation(..., shoulder='right', separation=0.08))."
            )
        if self.separation is not None and float(self.separation) <= 0:
            raise ValueError(
                f"Peak '{self.name}': separation must be positive, got {self.separation}."
            )


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
    sep_est: float = 0.0  # data-driven mode separation estimate for shoulder peaks


@dataclass(frozen=True)
class BaselineAnnotation:
    """Baseline annotation for a chromatogram.

    Attributes:
        low: Lower bound of the baseline region.
        high: Upper bound of the baseline region.
    """

    low: float
    high: float


@dataclass
class ChromatogramRecord:
    """One chromatogram with optional annotations.

    Attributes:
        sample_id: Parent sample identifier.
        chromatogram_id: Unique chromatogram identifier.
        time: 1D retention-time axis.
        signal: 1D signal vector.
        peaks: Per-chromatogram peaks.
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
        name: str,
        low: float,
        high: float,
        shoulder: Literal["left", "right"] | None = None,
    ) -> None:
        """Add a peak annotation to the chromatogram.

        Args:
            name: Name of the peak.
            low: Lower bound of the peak window.
            high: Upper bound of the peak window.
            shoulder: Optional expected shoulder side.

        Raises:
            ValueError: If bounds fall outside the time range or low >= high.
        """
        if low < min(self.time) or high > max(self.time):
            raise ValueError(
                "low and high must be within the time range, got "
                f"low={low}, high={high}, time range={min(self.time)} to {max(self.time)}"
            )
        peak = PeakAnnotation(name=name, low=low, high=high, shoulder=shoulder)
        for i, p in enumerate(self.peaks):
            if p.name == name:
                self.peaks[i] = peak
                return
        self.peaks.append(peak)

    def add_baseline(self, low: float, high: float) -> None:
        """Add a baseline to the chromatogram.

        Args:
            low: Lower bound of the baseline.
            high: Upper bound of the baseline.

        Raises:
            ValueError: If low and high are not within the time range or low is not less than high.
            ValueError: If baseline already exists.
        """

        # check if low and high are within the time range
        if low < min(self.time) or high > max(self.time):
            raise ValueError(
                "low and high must be within the time range, got "
                f"low={low}, high={high}, time range={min(self.time)} to {max(self.time)}"
            )

        # check if low is less than high
        if low >= high:
            raise ValueError(f"low must be less than high, got low={low}, high={high}")

        baseline = BaselineAnnotation(low=low, high=high)

        # check if baseline section exists
        for i, b in enumerate(self.baselines):
            if b.low == low and b.high == high:
                raise ValueError(f"baseline already exists, got low={low}, high={high}")

        self.baselines.append(baseline)


def get_chromatogram_tensor(chromatograms: list[ChromatogramRecord]) -> jnp.ndarray:
    signal_tensor = []
    for chromatogram in chromatograms:
        signal_tensor.append(chromatogram.signal)
    return jnp.array(signal_tensor).reshape(len(chromatograms), -1)


def stack_and_pad_signal(
    x_lists: list[list[float]], y_lists: list[list[float]]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Stack and pad data lists into a single tensor.
    Checks for longest list within each list and pads the shorter lists with zeros to the length of the longest list.
    Returns a tuple of the padded tensors.
    """

    if not len(x_lists) == len(y_lists):
        raise ValueError("x_lists and y_lists must have the same length")

    max_x_length = max(len(x) for x in x_lists)
    max_y_length = max(len(y) for y in y_lists)
    max_length = max(max_x_length, max_y_length)
    padded_x_lists = [x + [float("nan")] * (max_length - len(x)) for x in x_lists]
    padded_y_lists = [y + [float("nan")] * (max_length - len(y)) for y in y_lists]

    x_arrays = jnp.array(padded_x_lists)
    y_arrays = jnp.array(padded_y_lists)
    return x_arrays, y_arrays


def region_to_mask(low: float, high: float, time: jnp.ndarray) -> jnp.ndarray:
    """Mask True for all time points in [low, high].

    Args:
        low: Lower time bound (inclusive).
        high: Upper time bound (inclusive).
        time: 2D array of time values (e.g. from stack_and_pad_signal), shape
            (n_chromatograms, n_timepoints). Padded positions may be NaN.

    Returns:
        Boolean array same shape as time: True where low <= t <= high and t is
        not NaN.
    """
    return (time >= low) & (time <= high)


def baseline_to_mask(
    baselines: list[BaselineAnnotation], time: jnp.ndarray
) -> jnp.ndarray:
    """Boolean mask True for time points in any baseline region [low, high].

    Args:
        baselines: List of baseline annotations.
        time: 2D array of time values (e.g. from stack_and_pad_signal), shape
            (n_chromatograms, n_timepoints). Padded positions may be NaN.

    Returns:
        Boolean array same shape as time: True where time falls in any baseline.
    """
    if not baselines:
        return jnp.zeros(time.shape, dtype=bool)
    masks = jnp.stack([region_to_mask(b.low, b.high, time) for b in baselines])
    return jnp.any(masks, axis=0)


def peaks_to_mask(peaks: list[PeakAnnotation], time: jnp.ndarray) -> jnp.ndarray:
    """Boolean mask True for time points in any peak region [low, high].

    Returns:
        Boolean array shape (n_peaks, n_chromatograms, n_timepoints). mask[i]
        is True only for time points in [peaks[i].low, peaks[i].high].
    """

    peak_centers = jnp.array([(p.low + p.high) / 2 for p in peaks])
    # sort peaks by center
    sorted_indices = jnp.argsort(peak_centers)
    sorted_peaks = [peaks[i] for i in sorted_indices]

    return jnp.stack([region_to_mask(p.low, p.high, time) for p in sorted_peaks])


if __name__ == "__main__":
    from rich import print

    baselines = [BaselineAnnotation(low=1, high=2), BaselineAnnotation(low=4, high=5)]
    time = jnp.array([jnp.arange(1, 10), jnp.arange(1, 10)])

    peaks = [
        PeakAnnotation(name="peak1", low=0, high=4),
        PeakAnnotation(name="peak3", low=3, high=10),
        PeakAnnotation(name="peak2", low=1, high=2),
    ]
    mask = peaks_to_mask(peaks, time)
    print(mask)
