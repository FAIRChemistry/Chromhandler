"""Per-chromatogram trace statistics (noise, scale, sampling).

This module is pure NumPy and has no chromhandler-internal imports, so it
is safe to import from :mod:`chromhandler.model` without circularity.

Only :attr:`TraceStatistics.sigma_noise` is populated today; further
fields (``dt_median``, quantiles, drift, quantization) land in a follow-up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

_MAD_TO_SIGMA = 1.4826
_DER_SNR_DENOM = np.sqrt(6.0)


class TraceStatistics(BaseModel):
    """Summary statistics computed once on a full, untruncated trace."""

    model_config: ConfigDict = ConfigDict(validate_assignment=True)  # type: ignore

    sigma_noise: float = Field(
        ...,
        description=(
            "DER_SNR noise estimate (1.4826 * median(|d2|) / sqrt(6)) "
            "computed on the full trace. Robust to linear baselines and "
            "isolated peak curvature."
        ),
    )


def compute_trace_statistics(time: ArrayLike, signal: ArrayLike) -> TraceStatistics:
    """Compute trace-level statistics on a *full, untruncated* trace.

    Args:
        time:   1-D retention-time axis (minutes). Unused for sigma_noise
                but accepted now so the signature is stable for follow-up fields.
        signal: 1-D signal values, same length as ``time``.

    Returns:
        A :class:`TraceStatistics` with ``sigma_noise`` populated via DER_SNR.
    """
    y = np.asarray(signal, dtype=float)
    if y.ndim != 1:
        raise ValueError("signal must be 1-D.")
    if y.size < 3:
        raise ValueError("signal must have at least 3 finite samples for DER_SNR.")

    sigma = _der_snr(y)
    return TraceStatistics(sigma_noise=float(sigma))


def _der_snr(y: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
    """DER_SNR estimator: sigma = 1.4826 * median(|d2|) / sqrt(6)."""
    d2 = y[2:] - 2.0 * y[1:-1] + y[:-2]
    d2 = d2[np.isfinite(d2)]
    if d2.size == 0:
        raise ValueError("No finite 2nd-differences; trace is all-NaN or too short.")
    return _MAD_TO_SIGMA * float(np.median(np.abs(d2))) / _DER_SNR_DENOM
