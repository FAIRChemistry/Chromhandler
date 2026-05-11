"""Tests for Handler.prepare_dataset convenience wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.handler import Handler
from chromhandler.model import Chromatogram, Sample


def _chrom(chrom_id: str, sample_id: str, t_axis: np.ndarray) -> Chromatogram:  # type: ignore[type-arg]
    sig = 100.0 + 10.0 * np.exp(-((t_axis - 2.8) ** 2) / 0.02)
    return Chromatogram(
        id=chrom_id,
        sample_id=sample_id,
        time=t_axis.tolist(),
        signal=sig.tolist(),
        peaks=[],
    )


def _handler_with_two_samples(*, control_first: bool) -> Handler:
    h = Handler()
    t = np.arange(2.5, 3.6, 0.01)
    sample_a = Sample(
        id="A",
        chromatograms=[_chrom("c_a1", "A", t)],
        is_control=control_first,
    )
    sample_b = Sample(
        id="B",
        chromatograms=[_chrom("c_b1", "B", t)],
        is_control=False,
    )
    h.samples = [sample_a, sample_b]
    return h


def test_handler_prepare_dataset_basic() -> None:
    h = _handler_with_two_samples(control_first=False)
    peak_anns = [PeakAnnotation(molecule_id="A", rt_min=2.7, rt_max=2.9)]
    base_anns = [
        BaselineAnnotation(rt_min=2.55, rt_max=2.58),
        BaselineAnnotation(rt_min=3.50, rt_max=3.55),
    ]
    ds = h.prepare_dataset(peak_anns, base_anns)
    assert ds.n_trace == 2
    assert not ds.is_control.any()


def test_handler_prepare_dataset_collects_controls() -> None:
    h = _handler_with_two_samples(control_first=True)
    peak_anns = [PeakAnnotation(molecule_id="A", rt_min=2.7, rt_max=2.9)]
    base_anns = [
        BaselineAnnotation(rt_min=2.55, rt_max=2.58),
        BaselineAnnotation(rt_min=3.50, rt_max=3.55),
    ]
    ds = h.prepare_dataset(peak_anns, base_anns)
    assert ds.n_trace == 2
    np.testing.assert_array_equal(ds.is_control, np.array([True, False]))


def test_handler_prepare_dataset_raises_on_empty_samples() -> None:
    h = Handler()
    peak_anns = [PeakAnnotation(molecule_id="A", rt_min=2.7, rt_max=2.9)]
    base_anns = [
        BaselineAnnotation(rt_min=2.55, rt_max=2.58),
        BaselineAnnotation(rt_min=3.50, rt_max=3.55),
    ]
    with pytest.raises(ValueError, match="no chromatograms"):
        h.prepare_dataset(peak_anns, base_anns)
