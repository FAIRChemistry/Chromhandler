"""Unit tests for chromhandler.trace_statistics."""

from __future__ import annotations

import numpy as np
import pytest

from chromhandler.trace_statistics import TraceStatistics, compute_trace_statistics


def test_der_snr_recovers_known_gaussian_noise_on_flat_trace() -> None:
    """DER_SNR on a flat trace + iid Gaussian noise recovers sigma to ~5%."""
    rng = np.random.default_rng(0)
    n = 5000
    true_sigma = 2.5
    time = np.linspace(0.0, 10.0, n)
    signal = 100.0 + rng.normal(0.0, true_sigma, size=n)

    stats = compute_trace_statistics(time, signal)

    assert isinstance(stats, TraceStatistics)
    assert stats.sigma_noise == pytest.approx(true_sigma, rel=0.05)


def test_der_snr_unaffected_by_linear_baseline() -> None:
    """2nd-difference operator annihilates linear baselines."""
    rng = np.random.default_rng(1)
    n = 5000
    true_sigma = 1.0
    time = np.linspace(0.0, 10.0, n)
    signal = 50.0 + 3.0 * time + rng.normal(0.0, true_sigma, size=n)

    stats = compute_trace_statistics(time, signal)

    assert stats.sigma_noise == pytest.approx(true_sigma, rel=0.07)
