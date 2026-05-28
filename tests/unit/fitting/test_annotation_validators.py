"""Tests for chromhandler.annotations."""

from __future__ import annotations

import pytest

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation


class TestNComponents:
    """The ``n_components`` computed field on PeakAnnotation."""

    def test_single_returns_one(self) -> None:
        ann = PeakAnnotation(molecule_id="x", rt_min=1.0, rt_max=2.0, mode="single")
        assert ann.n_components == 1

    def test_appears_in_serialization(self) -> None:
        ann = PeakAnnotation(molecule_id="x", rt_min=1.0, rt_max=2.0)
        dumped = ann.model_dump()
        assert dumped["n_components"] == 1

    def test_is_read_only(self) -> None:
        ann = PeakAnnotation(molecule_id="x", rt_min=1.0, rt_max=2.0)
        with pytest.raises(ValueError):
            ann.n_components = 2  # type: ignore[misc]


class TestBaselinePeakDisjoint:
    """The ``check_baseline_peak_disjoint`` validator."""

    def test_disjoint_passes(self) -> None:
        from chromhandler.annotations import check_baseline_peak_disjoint

        peaks = [PeakAnnotation(molecule_id="x", rt_min=2.0, rt_max=3.0)]
        baselines = [BaselineAnnotation(rt_min=0.5, rt_max=1.5)]
        check_baseline_peak_disjoint(peaks, baselines)  # no error

    def test_touching_boundary_passes(self) -> None:
        from chromhandler.annotations import check_baseline_peak_disjoint

        peaks = [PeakAnnotation(molecule_id="x", rt_min=2.0, rt_max=3.0)]
        baselines = [BaselineAnnotation(rt_min=1.5, rt_max=2.0)]
        check_baseline_peak_disjoint(peaks, baselines)  # no error

    def test_overlap_raises(self) -> None:
        from chromhandler.annotations import check_baseline_peak_disjoint

        peaks = [PeakAnnotation(molecule_id="x", rt_min=2.0, rt_max=3.0)]
        baselines = [BaselineAnnotation(rt_min=2.5, rt_max=3.5)]
        with pytest.raises(ValueError, match="overlaps peak"):
            check_baseline_peak_disjoint(peaks, baselines)

    def test_peak_inside_baseline_raises(self) -> None:
        from chromhandler.annotations import check_baseline_peak_disjoint

        peaks = [PeakAnnotation(molecule_id="x", rt_min=2.0, rt_max=3.0)]
        baselines = [BaselineAnnotation(rt_min=1.0, rt_max=4.0)]
        with pytest.raises(ValueError, match="overlaps peak"):
            check_baseline_peak_disjoint(peaks, baselines)

    def test_peak_peak_overlap_allowed(self) -> None:
        from chromhandler.annotations import check_baseline_peak_disjoint

        peaks = [
            PeakAnnotation(molecule_id="x", rt_min=2.0, rt_max=3.0),
            PeakAnnotation(molecule_id="y", rt_min=2.5, rt_max=3.5),
        ]
        baselines: list[BaselineAnnotation] = []
        check_baseline_peak_disjoint(peaks, baselines)  # explicit policy: allowed

    def test_baseline_baseline_overlap_allowed(self) -> None:
        from chromhandler.annotations import check_baseline_peak_disjoint

        peaks: list[PeakAnnotation] = []
        baselines = [
            BaselineAnnotation(rt_min=0.5, rt_max=1.5),
            BaselineAnnotation(rt_min=1.0, rt_max=2.0),
        ]
        check_baseline_peak_disjoint(peaks, baselines)  # explicit policy: allowed

    def test_empty_inputs_pass(self) -> None:
        from chromhandler.annotations import check_baseline_peak_disjoint

        check_baseline_peak_disjoint([], [])  # no error
