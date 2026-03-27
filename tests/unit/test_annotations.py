"""Unit tests for PeakAnnotation and BaselineAnnotation.

Extracted from:
  - tests/integration/test_peak_assignment.py

Content: Annotation creation, validation, window boundary testing.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation


@pytest.mark.unit
def test_peak_annotation_single_mode() -> None:
    """PeakAnnotation can be created with single mode."""
    ann = PeakAnnotation(molecule_id="mol_a", rt_min=0.2, rt_max=0.8, mode="single")
    assert ann.molecule_id == "mol_a"
    assert ann.rt_min == 0.2
    assert ann.rt_max == 0.8
    assert ann.mode == "single"


@pytest.mark.unit
def test_peak_annotation_artefact_doublet_mode() -> None:
    """PeakAnnotation can be created with artefact_doublet mode."""
    ann = PeakAnnotation(
        molecule_id="mol_b",
        rt_min=2.3,
        rt_max=2.85,
        mode="artefact_doublet",
        artefact_side="right",
    )
    assert ann.molecule_id == "mol_b"
    assert ann.mode == "artefact_doublet"
    assert ann.artefact_side == "right"


@pytest.mark.unit
def test_peak_annotation_free_doublet_mode() -> None:
    """PeakAnnotation can be created with free_doublet mode."""
    ann = PeakAnnotation(
        molecule_id="mol_c",
        rt_min=3.9,
        rt_max=4.5,
        mode="free_doublet",
    )
    assert ann.molecule_id == "mol_c"
    assert ann.mode == "free_doublet"


@pytest.mark.unit
def test_peak_annotation_rt_bounds() -> None:
    """PeakAnnotation stores rt_min and rt_max."""
    ann = PeakAnnotation(molecule_id="mol", rt_min=4.8, rt_max=5.2, mode="single")
    assert ann.rt_min == 4.8
    assert ann.rt_max == 5.2


@pytest.mark.unit
def test_peak_annotation_rt_max_greater_than_min() -> None:
    """PeakAnnotation requires rt_max > rt_min."""
    with pytest.raises(ValidationError):
        PeakAnnotation(molecule_id="mol", rt_min=5.2, rt_max=4.8, mode="single")


@pytest.mark.unit
def test_peak_annotation_artefact_side_left() -> None:
    """PeakAnnotation can have artefact_side='left'."""
    ann = PeakAnnotation(
        molecule_id="mol",
        rt_min=2.3,
        rt_max=2.85,
        mode="artefact_doublet",
        artefact_side="left",
    )
    assert ann.artefact_side == "left"


@pytest.mark.unit
def test_peak_annotation_artefact_side_right() -> None:
    """PeakAnnotation can have artefact_side='right'."""
    ann = PeakAnnotation(
        molecule_id="mol",
        rt_min=2.3,
        rt_max=2.85,
        mode="artefact_doublet",
        artefact_side="right",
    )
    assert ann.artefact_side == "right"


@pytest.mark.unit
def test_baseline_annotation_minimal() -> None:
    """BaselineAnnotation can be created with rt_min and rt_max."""
    ann = BaselineAnnotation(rt_min=0.0, rt_max=0.1)
    assert ann.rt_min == 0.0
    assert ann.rt_max == 0.1


@pytest.mark.unit
def test_baseline_annotation_boundaries() -> None:
    """BaselineAnnotation stores rt_min and rt_max."""
    ann = BaselineAnnotation(rt_min=5.3, rt_max=5.8)
    assert ann.rt_min == 5.3
    assert ann.rt_max == 5.8


@pytest.mark.unit
def test_baseline_annotation_rt_max_greater_than_min() -> None:
    """BaselineAnnotation requires rt_max > rt_min."""
    with pytest.raises(ValidationError):
        BaselineAnnotation(rt_min=5.8, rt_max=5.3)


@pytest.mark.unit
def test_baseline_annotation_floating_point_bounds() -> None:
    """BaselineAnnotation handles floating point bounds."""
    ann = BaselineAnnotation(rt_min=0.10, rt_max=0.45)
    assert ann.rt_min == 0.10
    assert ann.rt_max == 0.45


@pytest.mark.unit
def test_peak_annotation_window_width() -> None:
    """PeakAnnotation window width is correct."""
    ann = PeakAnnotation(molecule_id="mol", rt_min=4.8, rt_max=5.2, mode="single")
    width = ann.rt_max - ann.rt_min
    assert abs(width - 0.4) < 1e-9


@pytest.mark.unit
def test_baseline_annotation_window_width() -> None:
    """BaselineAnnotation window width is correct."""
    ann = BaselineAnnotation(rt_min=0.0, rt_max=0.1)
    width = ann.rt_max - ann.rt_min
    assert abs(width - 0.1) < 1e-9


@pytest.mark.unit
def test_peak_annotation_equality() -> None:
    """Two PeakAnnotations with same fields are equal."""
    ann1 = PeakAnnotation(molecule_id="mol", rt_min=4.8, rt_max=5.2, mode="single")
    ann2 = PeakAnnotation(molecule_id="mol", rt_min=4.8, rt_max=5.2, mode="single")
    assert ann1 == ann2


@pytest.mark.unit
def test_baseline_annotation_equality() -> None:
    """Two BaselineAnnotations with same fields are equal."""
    ann1 = BaselineAnnotation(rt_min=0.0, rt_max=0.1)
    ann2 = BaselineAnnotation(rt_min=0.0, rt_max=0.1)
    assert ann1 == ann2
