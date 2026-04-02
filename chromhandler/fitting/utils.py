"""Fitting module utilities for array operations and masking.

Functions:
- :func:`pad_traces`: Pad time/signal lists to equal length
- :func:`region_to_mask`: Create boolean mask for time region
- :func:`baseline_to_mask`: Create mask for baseline annotation regions
- :func:`peaks_to_mask`: Create mask for peak annotation regions
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from chromhandler.annotations import BaselineAnnotation, PeakAnnotation


def pad_traces(
    x_lists: list[list[float]], y_lists: list[list[float]]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Pad time/signal lists to equal length (NaN-padded) and stack into 2-D arrays."""
    if len(x_lists) != len(y_lists):
        raise ValueError("x_lists and y_lists must have the same length")
    max_len = max(max(len(x) for x in x_lists), max(len(y) for y in y_lists))
    padded_x = [x + [float("nan")] * (max_len - len(x)) for x in x_lists]
    padded_y = [y + [float("nan")] * (max_len - len(y)) for y in y_lists]
    return np.array(padded_x, dtype=float), np.array(padded_y, dtype=float)


def region_to_mask(low: float, high: float, time: jnp.ndarray) -> jnp.ndarray:
    """Mask True for all time points in [low, high]."""
    return (time >= low) & (time <= high)


def baseline_to_mask(baselines: list[BaselineAnnotation], time: jnp.ndarray) -> jnp.ndarray:
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
    sorted_indices = [int(i) for i in jnp.argsort(peak_centers).tolist()]
    sorted_peaks = [peaks[i] for i in sorted_indices]
    return jnp.stack([region_to_mask(low=p.rt_min, high=p.rt_max, time=time) for p in sorted_peaks])
