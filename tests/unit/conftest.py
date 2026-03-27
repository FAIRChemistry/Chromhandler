"""Unit test-specific fixtures.

This conftest.py provides fixtures specialized for unit tests:
  - Lightweight instances with minimal dependencies
  - Isolated model objects
  - Single-module validation tests
"""

from __future__ import annotations

import pytest

from chromhandler.model import Estimate


@pytest.fixture
def estimate_empty() -> Estimate:
    """Fixture: Empty Estimate."""
    return Estimate(mean=0.0)


@pytest.fixture
def estimate_with_samples() -> Estimate:
    """Fixture: Estimate with posterior samples."""
    return Estimate(mean=10.0, sd=1.0, samples=[9.5, 10.0, 10.5, 10.2, 9.8])
