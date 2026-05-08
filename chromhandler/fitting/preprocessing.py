"""Preprocessing utilities: variable-length trace padding and dt computation.

Variable-length signal arrays (one per chromatogram) are padded to a
rectangular ``[n_trace, n_time]`` matrix with trailing ``NaN`` values.
``NaN`` is the canonical missing-data marker downstream — likelihood and
prior code mask it out explicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def pad_to_common_axis(
    times: list[NDArray[np.float64]],
    signals: list[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pad variable-length traces to a rectangular array.

    Args:
        times: List of 1-D arrays of length ``n_time_i``, one per trace.
        signals: List of 1-D arrays of matching length, one per trace.

    Returns:
        Tuple ``(time, signal)`` of shape ``[n_trace, max(n_time_i)]``.
        Padding values are ``NaN``.

    Raises:
        ValueError: If ``times`` and ``signals`` have different outer
            lengths, or if any per-trace ``time[i]`` and ``signal[i]``
            differ in length.
    """
    if len(times) != len(signals):
        raise ValueError(
            f"times and signals must have the same number of traces, "
            f"got {len(times)} and {len(signals)}."
        )
    for i, (t, s) in enumerate(zip(times, signals, strict=True)):
        if t.shape != s.shape:
            raise ValueError(
                f"trace {i}: time length {t.shape} != signal length {s.shape}."
            )
    n_trace = len(times)
    if n_trace == 0:
        empty = np.empty((0, 0), dtype=np.float64)
        return empty, empty
    n_max = max(t.shape[0] for t in times)
    time_out = np.full((n_trace, n_max), np.nan, dtype=np.float64)
    signal_out = np.full((n_trace, n_max), np.nan, dtype=np.float64)
    for i, (t, s) in enumerate(zip(times, signals, strict=True)):
        n = t.shape[0]
        time_out[i, :n] = t
        signal_out[i, :n] = s
    return time_out, signal_out


def compute_dt_per_trace(time: NDArray[np.float64]) -> NDArray[np.float64]:
    """Median sampling interval per trace.

    Args:
        time: ``[n_trace, n_time]`` array; trailing ``NaN`` values
            represent padding.

    Returns:
        ``[n_trace]`` array of median ``dt`` per trace.
    """
    diffs = np.diff(time, axis=1)
    return np.nanmedian(diffs, axis=1)


def compute_global_dt(dt_per_trace: NDArray[np.float64]) -> float:
    """Median of per-trace ``dt`` values.

    Args:
        dt_per_trace: ``[n_trace]`` array of per-trace median dt.

    Returns:
        Global median dt as a Python float.
    """
    return float(np.median(dt_per_trace))
