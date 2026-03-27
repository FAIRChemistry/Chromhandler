"""Integration test-specific fixtures.

This conftest.py provides fixtures for integration tests that span multiple modules:
  - Complex Handler setups with multiple components
  - Full workflows (e.g., calibration → export)
  - Multi-module interaction tests
"""

from __future__ import annotations

# Import builders from parent conftest (pytest conftest loader makes parent available)
# Using a dynamic import to avoid circular dependencies
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from chromhandler.handler import Handler

_parent_conftest_path = Path(__file__).parent.parent / "conftest.py"
spec = importlib.util.spec_from_file_location("parent_conftest", _parent_conftest_path)
_parent_conftest_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_parent_conftest_module)

_handler = _parent_conftest_module._handler
_molecule = _parent_conftest_module._molecule
_sample = _parent_conftest_module._sample
_chromatogram = _parent_conftest_module._chromatogram
_peak = _parent_conftest_module._peak
_initial_condition = _parent_conftest_module._initial_condition


@pytest.fixture
def handler_with_chromatograms() -> Handler:
    """Fixture: Handler with multiple samples and chromatograms."""
    mol_a = _molecule("mol_a")

    chrom_a = _chromatogram("chrom_a", "sample_a", peaks=[_peak("mol_a", "chrom_a", area_mean=1000.0)])
    sample_a = _sample("sample_a", chromatograms=[chrom_a])

    chrom_b = _chromatogram("chrom_b", "sample_b", peaks=[_peak("mol_a", "chrom_b", area_mean=2000.0)])
    sample_b = _sample("sample_b", chromatograms=[chrom_b])

    return _handler(
        samples=[sample_a, sample_b],
        molecules={"mol_a": mol_a},
    )


@pytest.fixture
def handler_with_calibration_data() -> Handler:
    """Fixture: Handler with calibration standards (t=0 samples with known concentrations)."""
    mol = _molecule("Ino")

    # Build calibration standards (t=0 samples with peaks + initial conditions)
    std_low_chrom = _chromatogram(
        "std_low_chrom",
        "std_low",
        peaks=[_peak("Ino", "std_low_chrom", area_mean=400_000.0)],
        reaction_time=0.0,
    )
    std_low = _sample(
        "std_low",
        chromatograms=[std_low_chrom],
        initial_conditions=[_initial_condition("Ino", init_conc=100.0)],
    )

    std_high_chrom = _chromatogram(
        "std_high_chrom",
        "std_high",
        peaks=[_peak("Ino", "std_high_chrom", area_mean=1_600_000.0)],
        reaction_time=0.0,
    )
    std_high = _sample(
        "std_high",
        chromatograms=[std_high_chrom],
        initial_conditions=[_initial_condition("Ino", init_conc=400.0)],
    )

    return _handler(
        samples=[std_low, std_high],
        molecules={"Ino": mol},
    )
