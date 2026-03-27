"""Fitting unit test-specific fixtures.

Fixtures for testing fitting module components in isolation:
  - Priors (geometric, baseline) without MCMC
  - Baseline fitting without full fitter
  - Data structures (Subset, PeakAnnotation)
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def simple_time_axis() -> np.ndarray:
    """Fixture: Simple 1D time axis for fitting tests."""
    return np.linspace(0.0, 1.0, 100)


@pytest.fixture
def simple_signal() -> np.ndarray:
    """Fixture: Simple 1D signal for baseline/fitting tests."""
    time = np.linspace(0.0, 1.0, 100)
    # Gentle baseline with a peak
    return 0.1 + 0.05 * time + np.exp(-100 * (time - 0.5) ** 2)
