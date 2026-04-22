"""Fitting unit test-specific fixtures.

Fixtures for testing fitting module components in isolation:
  - Priors (geometric, baseline) without MCMC
  - Baseline fitting without full fitter
  - Data structures (Subset, PeakAnnotation)
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest


def w_min_from_dt(dt: float) -> float:
    """Nyquist-like HWHM floor — mirrors ``chromhandler.fitting.priors``.

    One source of truth for the formula so test fixtures can't drift from
    ``build_peak_priors`` if the bound rule ever changes.
    """
    return 8.0 * dt / math.sqrt(8.0 * math.log(2.0))


@pytest.fixture
def simple_time_axis() -> npt.NDArray[np.float64]:
    """Fixture: Simple 1D time axis for fitting tests."""
    return np.linspace(0.0, 1.0, 100)


@pytest.fixture
def simple_signal() -> npt.NDArray[np.float64]:
    """Fixture: Simple 1D signal for baseline/fitting tests."""
    time = np.linspace(0.0, 1.0, 100)
    # Gentle baseline with a peak
    return 0.1 + 0.05 * time + np.exp(-100 * (time - 0.5) ** 2)
