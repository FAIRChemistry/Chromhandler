"""Per-trace noise estimation from baseline-region residuals.

Estimates pure measurement noise (random variation) by computing the
median absolute deviation of the residuals between observed signal and
the OLS baseline ``intercept + slope * t`` within the user-annotated
baseline regions. Robust to outliers and to small baseline-fit errors.

We deliberately do not estimate "noise" from peak regions or from the
whole trace: that would conflate genuine measurement noise with model
misfit, falsely widening the likelihood and masking model-shape issues
downstream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from chromhandler.fitting.baseline import baseline_region_mask

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from chromhandler.annotations import BaselineAnnotation

_MAD_TO_STD: float = 1.4826  # consistent estimator of std under Gaussian noise


def estimate_noise_per_trace(
    time: NDArray[np.float64],
    signal: NDArray[np.float64],
    regions: list[BaselineAnnotation],
    baseline_intercept: NDArray[np.float64],
    baseline_slope: NDArray[np.float64],
) -> NDArray[np.float64]:
    """MAD-based per-trace noise std from baseline-region residuals.

    Args:
        time: ``[n_trace, n_time]`` time array (NaN-padded allowed).
        signal: ``[n_trace, n_time]`` signal array.
        regions: At least one baseline annotation.
        baseline_intercept: ``[n_trace]`` per-trace OLS intercept.
        baseline_slope: ``[n_trace]`` per-trace OLS slope.

    Returns:
        ``[n_trace]`` array of noise std estimates.

    Raises:
        ValueError: If ``regions`` is empty.
    """
    if not regions:
        raise ValueError(
            "estimate_noise_per_trace requires at least one BaselineAnnotation."
        )
    mask = baseline_region_mask(time, regions)
    predicted = baseline_intercept[:, None] + baseline_slope[:, None] * time
    residual = signal - predicted
    masked = np.where(mask, residual, np.nan)
    mad = np.nanmedian(np.abs(masked), axis=1)
    return _MAD_TO_STD * mad
