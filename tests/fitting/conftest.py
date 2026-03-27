"""Fitting integration test-specific fixtures.

Fixtures for full-pipeline fitting tests:
  - BetterFitter instances with complete setup
  - MCMC posteriors and samples
  - Posterior curves and visualizations
"""

from __future__ import annotations

import numpy as np
import pytest

from chromhandler.fitting.better_fitter import BetterFitter

# Import from parent conftest (tests/conftest.py)
# Pytest automatically loads parent conftest files, so we can import directly
from tests.conftest import _peak_annotation, _make_posterior_samples


@pytest.fixture
def better_fitter_basic() -> BetterFitter:
    """Fixture: Minimal BetterFitter for integration tests.

    Single trace, single peak, no subsets.
    """
    time = np.asarray([[0.0, 0.5, 1.0]], dtype=float)  # [n_trace=1, n_time=3]
    signal = np.asarray([[0.0, 1.0, 0.0]], dtype=float)  # [n_trace=1, n_time=3]
    peaks = [_peak_annotation("mol_a", rt_min=0.2, rt_max=0.8)]

    return BetterFitter(
        time=time,
        signal=signal,
        peaks=peaks,
        baselines=[],
        trace_sample_ids=["sample_a"],
        trace_chromatogram_ids=["chrom_a"],
    )


@pytest.fixture
def posterior_samples_simple() -> dict[str, np.ndarray]:
    """Fixture: Simple posterior samples for 3 draws."""
    return _make_posterior_samples(
        area_samples=[10.0, 12.0, 14.0],
        apex_samples=[0.48, 0.50, 0.52],
    )
