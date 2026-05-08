"""Per-trace baseline estimation from user-annotated regions only.

Fits ``baseline(t) = intercept + slope * t`` per trace via ordinary least
squares on the points lying inside any user-supplied
:class:`~chromhandler.annotations.BaselineAnnotation` window. Peak-edge
low-point anchors are deliberately not used: they pollute the baseline
estimate with peak-tail contributions. The user's annotations are the
single source of truth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from chromhandler.annotations import BaselineAnnotation

_MIN_POINTS_PER_TRACE: int = 2


def baseline_region_mask(
    time: NDArray[np.float64],
    regions: list[BaselineAnnotation],
) -> NDArray[np.bool_]:
    """Boolean mask of points lying inside any baseline region.

    Public helper used by ``estimate_baselines`` and by
    :func:`chromhandler.fitting.noise.estimate_noise_per_trace`.

    Args:
        time: ``[n_trace, n_time]`` time array (NaN-padded allowed).
        regions: Baseline annotations.

    Returns:
        ``[n_trace, n_time]`` bool array; True iff that ``(trace, time)``
        sample is inside any region (and the time value is not NaN).
    """
    valid = ~np.isnan(time)
    inside = np.zeros_like(time, dtype=bool)
    for r in regions:
        inside |= (time >= r.rt_min) & (time <= r.rt_max)
    return inside & valid


def estimate_baselines(
    time: NDArray[np.float64],
    signal: NDArray[np.float64],
    regions: list[BaselineAnnotation],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-trace OLS baseline through the user-annotated regions.

    Args:
        time: ``[n_trace, n_time]`` time array (NaN-padded allowed).
        signal: ``[n_trace, n_time]`` signal array.
        regions: At least one baseline annotation. Multiple regions are
            unioned.

    Returns:
        Tuple ``(intercept, slope)`` of ``[n_trace]`` arrays.

    Raises:
        ValueError: If ``regions`` is empty, or if any trace has fewer
            than 2 points inside the unioned baseline region (cannot fit
            a line).
    """
    if not regions:
        raise ValueError("estimate_baselines requires at least one BaselineAnnotation.")
    mask = baseline_region_mask(time, regions)
    n_trace = time.shape[0]
    intercept = np.zeros(n_trace, dtype=float)
    slope = np.zeros(n_trace, dtype=float)
    for i in range(n_trace):
        idx = np.flatnonzero(mask[i])
        if idx.size < _MIN_POINTS_PER_TRACE:
            raise ValueError(
                f"Trace {i}: too few baseline points ({idx.size}) inside the "
                f"annotated regions; need at least {_MIN_POINTS_PER_TRACE}."
            )
        t_anchor = time[i, idx]
        s_anchor = signal[i, idx]
        slope_i, intercept_i = np.polyfit(t_anchor, s_anchor, 1)
        slope[i] = slope_i
        intercept[i] = intercept_i
    return intercept, slope
