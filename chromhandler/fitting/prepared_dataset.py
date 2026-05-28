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
        trace_ids: ``[n_trace]`` tuple of human-readable identifiers for each
            trace, used to name traces in error messages. When fed via
            :meth:`~chromhandler.handler.Handler.prepare_dataset`, the format is
            ``"{sample_id}/{chromatogram_id}"``. Defaults to ``"trace_{i}"``.
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
    trace_ids: tuple[str, ...]


def prepare_dataset(
    times: list[NDArray[np.float64]],
    signals: list[NDArray[np.float64]],
    peak_annotations: list[PeakAnnotation],
    baseline_annotations: list[BaselineAnnotation],
    trace_ids: list[str] | None = None,
) -> PreparedDataset:
    """Run the full data-preparation pipeline.

    Args:
        times: List of 1-D time arrays, one per trace.
        signals: List of 1-D signal arrays, matching lengths.
        peak_annotations: User peak windows.
        baseline_annotations: User baseline regions.
        trace_ids: Optional per-trace string identifiers used to name traces in
            error messages from the priors / model layers. When ``None``,
            defaults to ``["trace_0", "trace_1", ...]``. Length must match
            ``len(times)``.

    Returns:
        :class:`PreparedDataset` with padded arrays, dt, baselines, noise,
        and per-trace ``trace_ids``.

    Raises:
        ValueError: If a baseline window overlaps any peak window, if any
            preparation step fails, or if ``trace_ids`` length does not
            match the number of traces.
    """
    n_trace = len(times)
    if trace_ids is not None and len(trace_ids) != n_trace:
        raise ValueError(
            f"trace_ids length ({len(trace_ids)}) must match number of "
            f"traces ({n_trace})."
        )
    trace_ids_tuple: tuple[str, ...] = (
        tuple(trace_ids)
        if trace_ids is not None
        else tuple(f"trace_{i}" for i in range(n_trace))
    )
    check_baseline_peak_disjoint(peak_annotations, baseline_annotations)
    time, signal = pad_to_common_axis(times, signals)

    # Restrict the likelihood mask to only the annotated windows: baseline
    # regions + peak windows. Points outside these windows are not part of the
    # model and should not contribute to the likelihood. Evaluating the full
    # chromatogram with a narrow noise prior (estimated from quiet baseline
    # regions) causes catastrophic divergences when other large peaks are
    # present in the run.
    # Per-trace window mask (shape == time.shape): True where the trace's
    # OWN time axis lies inside any peak/baseline window. Earlier code used
    # time[0] only, which silently mis-masked traces with different padding
    # lengths. NaN padded cells compare False and are excluded automatically.
    window_mask = np.zeros_like(time, dtype=np.bool_)
    for ann in peak_annotations:
        window_mask |= (time >= ann.rt_min) & (time <= ann.rt_max)
    for ann in baseline_annotations:
        window_mask |= (time >= ann.rt_min) & (time <= ann.rt_max)
    valid_mask = ~np.isnan(signal) & window_mask

    # dt / baselines / noise are computed on the FULL padded arrays so
    # they reflect the original sampling, not pruned gaps.
    dt_per_trace = compute_dt_per_trace(time)
    dt_global = compute_global_dt(dt_per_trace)
    intercept, slope = estimate_baselines(time, signal, baseline_annotations)
    noise = estimate_noise_per_trace(
        time, signal, baseline_annotations, intercept, slope
    )

    # Prune: keep columns where at least one trace contributes a valid
    # likelihood point. Downstream readers index by valid_mask or by
    # retention-time ranges, so column count is invariant for them.
    keep_cols = np.any(valid_mask, axis=0)
    time = time[:, keep_cols]
    signal = signal[:, keep_cols]
    valid_mask = valid_mask[:, keep_cols]

    return PreparedDataset(
        time=time,
        signal=signal,
        valid_mask=valid_mask,
        dt_per_trace=dt_per_trace,
        dt_global=dt_global,
        n_trace=n_trace,
        peak_annotations=list(peak_annotations),
        baseline_annotations=list(baseline_annotations),
        baseline_intercept=intercept,
        baseline_slope=slope,
        noise_per_trace=noise,
        trace_ids=trace_ids_tuple,
    )
