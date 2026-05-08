"""Immutable bundle of all foundations outputs and the top-level orchestrator.

``PreparedDataset`` is the canonical input to the priors/model layer. It
contains everything that data preparation produces: padded time/signal
arrays, a validity mask, per-trace and global dt, the user's annotations,
per-trace baseline parameters, and per-trace noise std.

``prepare_dataset`` runs the full preparation pipeline end-to-end:
overlap validation -> padding -> dt -> baseline OLS -> noise -> bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from chromhandler.annotations import (
    BaselineAnnotation,
    PeakAnnotation,
    check_baseline_peak_disjoint,
)
from chromhandler.fitting.baseline import estimate_baselines
from chromhandler.fitting.noise import estimate_noise_per_trace
from chromhandler.fitting.preprocessing import (
    compute_dt_per_trace,
    compute_global_dt,
    pad_to_common_axis,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class PreparedDataset:
    """Canonical input to the priors/model layer.

    Attributes:
        time: ``[n_trace, n_time]`` time array, NaN where padded.
        signal: ``[n_trace, n_time]`` signal array, NaN where padded.
        valid_mask: ``[n_trace, n_time]`` bool, True where signal is real.
        dt_per_trace: ``[n_trace]`` per-trace median sampling interval.
        dt_global: Global median dt.
        n_trace: Number of traces.
        peak_annotations: User peak windows.
        baseline_annotations: User baseline regions.
        baseline_intercept: ``[n_trace]`` per-trace OLS intercept.
        baseline_slope: ``[n_trace]`` per-trace OLS slope.
        noise_per_trace: ``[n_trace]`` MAD-based noise std.
    """

    time: NDArray[np.float64]
    signal: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    dt_per_trace: NDArray[np.float64]
    dt_global: float
    n_trace: int
    peak_annotations: list[PeakAnnotation]
    baseline_annotations: list[BaselineAnnotation]
    baseline_intercept: NDArray[np.float64]
    baseline_slope: NDArray[np.float64]
    noise_per_trace: NDArray[np.float64]


def prepare_dataset(
    times: list[NDArray[np.float64]],
    signals: list[NDArray[np.float64]],
    peak_annotations: list[PeakAnnotation],
    baseline_annotations: list[BaselineAnnotation],
) -> PreparedDataset:
    """Run the full data-preparation pipeline.

    Args:
        times: List of 1-D time arrays, one per trace.
        signals: List of 1-D signal arrays, matching lengths.
        peak_annotations: User peak windows.
        baseline_annotations: User baseline regions.

    Returns:
        :class:`PreparedDataset` with padded arrays, dt, baselines, noise.

    Raises:
        ValueError: If a baseline window overlaps any peak window, or if
            any preparation step fails (see component functions).
    """
    check_baseline_peak_disjoint(peak_annotations, baseline_annotations)
    time, signal = pad_to_common_axis(times, signals)
    valid_mask = ~np.isnan(signal)
    dt_per_trace = compute_dt_per_trace(time)
    dt_global = compute_global_dt(dt_per_trace)
    intercept, slope = estimate_baselines(time, signal, baseline_annotations)
    noise = estimate_noise_per_trace(
        time, signal, baseline_annotations, intercept, slope
    )
    return PreparedDataset(
        time=time,
        signal=signal,
        valid_mask=valid_mask,
        dt_per_trace=dt_per_trace,
        dt_global=dt_global,
        n_trace=len(times),
        peak_annotations=list(peak_annotations),
        baseline_annotations=list(baseline_annotations),
        baseline_intercept=intercept,
        baseline_slope=slope,
        noise_per_trace=noise,
    )
