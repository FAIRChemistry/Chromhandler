"""Tests for chromhandler.annotations."""

from __future__ import annotations

import pytest

from chromhandler.annotations import PeakAnnotation


class TestNComponents:
    """The ``n_components`` computed field on PeakAnnotation."""

    def test_single_returns_one(self) -> None:
        ann = PeakAnnotation(molecule_id="x", rt_min=1.0, rt_max=2.0, mode="single")
        assert ann.n_components == 1

    def test_artefact_doublet_returns_two(self) -> None:
        ann = PeakAnnotation(
            molecule_id="x",
            rt_min=1.0,
            rt_max=2.0,
            mode="artefact_doublet",
            artefact_side="right",
        )
        assert ann.n_components == 2

    def test_free_doublet_returns_two(self) -> None:
        ann = PeakAnnotation(
            molecule_id="x", rt_min=1.0, rt_max=2.0, mode="free_doublet"
        )
        assert ann.n_components == 2

    def test_appears_in_serialization(self) -> None:
        ann = PeakAnnotation(molecule_id="x", rt_min=1.0, rt_max=2.0)
        dumped = ann.model_dump()
        assert dumped["n_components"] == 1

    def test_is_read_only(self) -> None:
        ann = PeakAnnotation(molecule_id="x", rt_min=1.0, rt_max=2.0)
        with pytest.raises(ValueError):
            ann.n_components = 2  # type: ignore[misc]
