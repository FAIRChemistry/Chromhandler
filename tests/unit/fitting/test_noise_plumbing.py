"""Unit tests for Fitter noise plumbing (trace_sigma_noise attribute)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from chromhandler.fitting.fitter import Fitter
from chromhandler.trace_statistics import compute_trace_statistics


def _noisy_matrix(
    n_trace: int = 3, n_time: int = 4000, sigma: float = 1.5, seed: int = 0
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    time = np.tile(np.linspace(0.0, 10.0, n_time), (n_trace, 1))
    signal = 100.0 + rng.normal(0.0, sigma, size=(n_trace, n_time))
    return time, signal


@pytest.mark.unit
def test_init_auto_computes_trace_sigma_noise_from_rows() -> None:
    """When trace_sigma_noise is not supplied, __init__ auto-computes per row."""
    time, signal = _noisy_matrix(n_trace=3, sigma=1.5)
    fitter = Fitter(time, signal)

    assert fitter.trace_sigma_noise.shape == (3,)
    assert fitter.trace_sigma_noise.dtype == np.float64
    # Values match compute_trace_statistics called directly on each row.
    for t in range(3):
        expected = compute_trace_statistics(time[t], signal[t]).sigma_noise
        assert fitter.trace_sigma_noise[t] == pytest.approx(expected, rel=1e-10)


@pytest.mark.unit
def test_init_accepts_explicit_trace_sigma_noise() -> None:
    """Explicit trace_sigma_noise is stored verbatim."""
    time, signal = _noisy_matrix()
    explicit = np.array([1.1, 2.2, 3.3])
    fitter = Fitter(time, signal, trace_sigma_noise=explicit)

    np.testing.assert_array_equal(fitter.trace_sigma_noise, explicit)


@pytest.mark.unit
def test_init_rejects_wrong_shape_trace_sigma_noise() -> None:
    """trace_sigma_noise with mismatched length is rejected."""
    time, signal = _noisy_matrix(n_trace=3)
    with pytest.raises(ValueError, match="trace_sigma_noise must have length n_traces=3"):
        Fitter(time, signal, trace_sigma_noise=np.array([1.0, 2.0]))


@pytest.mark.unit
def test_init_rejects_non_finite_trace_sigma_noise() -> None:
    """Non-finite entries in trace_sigma_noise are rejected."""
    time, signal = _noisy_matrix(n_trace=2)
    with pytest.raises(ValueError, match="trace_sigma_noise must be finite and positive"):
        Fitter(time, signal, trace_sigma_noise=np.array([1.0, np.nan]))


@pytest.mark.unit
def test_init_rejects_non_positive_trace_sigma_noise() -> None:
    """Zero or negative entries in trace_sigma_noise are rejected."""
    time, signal = _noisy_matrix(n_trace=2)
    with pytest.raises(ValueError, match="trace_sigma_noise must be finite and positive"):
        Fitter(time, signal, trace_sigma_noise=np.array([1.0, 0.0]))


@pytest.mark.unit
def test_init_auto_compute_reraises_row_failure_with_index() -> None:
    """Auto-compute re-raises compute_trace_statistics failures with row index."""
    # Row 1 has <3 finite samples — compute_trace_statistics will raise.
    time = np.tile(np.linspace(0.0, 1.0, 10), (2, 1))
    signal = np.vstack([np.linspace(100.0, 101.0, 10), np.full(10, np.nan)])
    with pytest.raises(ValueError, match="trace row 1"):
        Fitter(time, signal)
