"""Global pytest fixtures and shared test utilities.

This conftest.py provides reusable fixtures and builders used across the entire
test suite. Subdirectory conftest.py files inherit these fixtures and can add
their own specialized fixtures.

Fixture Organization:
  - Model builders: Lightweight constructors for Molecule, Sample, Handler
  - Pytest fixtures: Lazy-evaluated builders for common test objects
  - Pytest markers: Custom markers for categorizing tests
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chromhandler.handler import Handler
from chromhandler.model import Chromatogram, Estimate, InitialCondition, Peak, Sample
from chromhandler.molecule import Molecule
from chromhandler.protein import Protein

# ============================================================================
# Constants
# ============================================================================

TEST_DATA_DIR = Path(__file__).parent / "test_readers" / "data"


# ============================================================================
# Model Builders (shared across unit & integration tests)
# ============================================================================


def _molecule(mol_id: str = "mol_a", name: str | None = None, pubchem_cid: int | None = None) -> Molecule:
    """Build a minimal Molecule."""
    return Molecule(
        id=mol_id,
        name=name or f"Molecule_{mol_id}",
        pubchem_cid=pubchem_cid or 0,
    )


def _protein(protein_id: str = "prot_a", name: str | None = None) -> Protein:
    """Build a minimal Protein."""
    return Protein(
        id=protein_id,
        name=name or f"Protein_{protein_id}",
    )


def _estimate(mean: float, std: float | None = None, samples: list[float] | None = None) -> Estimate:  # type: ignore[reportUnusedFunction]
    """Build an Estimate with mean and optional statistics."""
    return Estimate(
        mean=mean,
        std=std,
        samples=samples or [],  # type: ignore[arg-type]
    )


def _peak(  # type: ignore[reportUnusedFunction]
    mol_id: str = "mol_a",
    chrom_id: str = "chrom_a",
    location_mean: float = 5.0,
    area_mean: float = 1000.0,
) -> Peak:
    """Build a minimal Peak."""
    return Peak(
        chromatogram_id=chrom_id,
        molecule_id=mol_id,
        location=_estimate(mean=location_mean),
        area=_estimate(mean=area_mean),
    )


def _chromatogram(  # type: ignore[reportUnusedFunction]
    chrom_id: str = "chrom_a",
    sample_id: str = "sample_a",
    peaks: list[Peak] | None = None,
    reaction_time: float | None = None,
) -> Chromatogram:
    """Build a minimal Chromatogram."""
    return Chromatogram(
        id=chrom_id,
        sample_id=sample_id,
        reaction_time=reaction_time,
        reaction_time_unit="min",
        peaks=peaks or [],
    )


def _initial_condition(  # type: ignore[reportUnusedFunction]
    mol_id: str = "mol_a",
    init_conc: float = 100.0,
    conc_unit: str = "umol / l",
) -> InitialCondition:
    """Build an InitialCondition."""
    return InitialCondition(
        molecule_id=mol_id,
        init_conc=init_conc,
        conc_unit=conc_unit,
    )


def _sample(
    sample_id: str = "sample_a",
    chromatograms: list[Chromatogram] | None = None,
    initial_conditions: list[InitialCondition] | None = None,
) -> Sample:
    """Build a minimal Sample."""
    return Sample(
        id=sample_id,
        chromatograms=chromatograms or [],
        initial_conditions=initial_conditions or [],
    )


def _handler(
    samples: list[Sample] | None = None,
    molecules: dict[str, Molecule] | None = None,
    proteins: dict[str, Protein] | None = None,
) -> Handler:
    """Build a Handler with optional samples, molecules, and proteins."""
    return Handler(
        samples=samples or [],
        molecules=molecules or {},
        proteins=proteins or {},
    )


# ============================================================================
# Pytest Fixtures (lazy-evaluated builders)
# ============================================================================


@pytest.fixture
def molecule_a() -> Molecule:
    """Fixture: Basic molecule for testing."""
    return _molecule("mol_a")


@pytest.fixture
def protein_a() -> Protein:
    """Fixture: Basic protein for testing."""
    return _protein("prot_a")


@pytest.fixture
def sample_a() -> Sample:
    """Fixture: Basic sample with no chromatograms or initial conditions."""
    return _sample("sample_a")


@pytest.fixture
def handler_empty() -> Handler:
    """Fixture: Empty handler (no samples, molecules, or proteins)."""
    return _handler()


@pytest.fixture
def matplotlib_agg():
    """Fixture: Set matplotlib backend to Agg (no display needed)."""
    import matplotlib

    matplotlib.use("Agg")
    yield
    # Teardown: reset to default
    matplotlib.use("Qt5Agg")


# ============================================================================
# pytest Configuration (register markers)
# ============================================================================


def pytest_configure(config: pytest.Config) -> None:  # type: ignore[name-defined]
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "unit: marks test as unit test (fast, isolated)")
    config.addinivalue_line("markers", "integration: marks test as integration test (may be slow)")
    config.addinivalue_line("markers", "fitting: marks test as fitting-specific")
    config.addinivalue_line("markers", "slow: marks tests as slow readers (deselect with '-m \"not slow\"')")
