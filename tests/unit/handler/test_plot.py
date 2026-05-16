"""Tests for ``chromhandler.plotting`` helpers and ``Handler.plot``."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from chromhandler.handler import Handler
from chromhandler.model import Chromatogram, Estimate, Peak, Sample
from chromhandler.plotting import _group_chromatograms, _line_colors


def _make_handler(n_samples: int = 2, chroms_per_sample: int = 2) -> Handler:
    handler = Handler()
    t = np.linspace(0.0, 10.0, 201, dtype=float).tolist()
    for s in range(n_samples):
        chroms = []
        for c in range(chroms_per_sample):
            sig = (np.sin(np.linspace(0, 6.28, 201)) + 0.1 * c).tolist()
            peak = Peak(
                chromatogram_id=f"s{s}_c{c}",
                location=Estimate(mean=5.0),
                area=Estimate(mean=1.0),
            )
            chroms.append(
                Chromatogram(
                    id=f"s{s}_c{c}",
                    sample_id=f"s{s}",
                    time=t,
                    signal=sig,
                    peaks=[peak],
                )
            )
        handler.samples.append(Sample(id=f"s{s}", chromatograms=chroms))
    return handler


def test_group_chromatograms_single() -> None:
    handler = _make_handler(n_samples=2, chroms_per_sample=2)
    groups = _group_chromatograms(handler, overlay="single")
    assert len(groups) == 4
    assert all(len(g) == 1 for g in groups)
    assert [g[0].id for g in groups] == ["s0_c0", "s0_c1", "s1_c0", "s1_c1"]


def test_group_chromatograms_sample() -> None:
    handler = _make_handler(n_samples=2, chroms_per_sample=2)
    groups = _group_chromatograms(handler, overlay="sample")
    assert len(groups) == 2
    assert [len(g) for g in groups] == [2, 2]
    assert [g[0].sample_id for g in groups] == ["s0", "s1"]


def test_group_chromatograms_all() -> None:
    handler = _make_handler(n_samples=2, chroms_per_sample=2)
    groups = _group_chromatograms(handler, overlay="all")
    assert len(groups) == 1
    assert len(groups[0]) == 4


def test_line_colors_single() -> None:
    colors = _line_colors(1)
    assert colors == [matplotlib.colors.to_rgba("tab:blue")]


def test_line_colors_multi_uses_viridis() -> None:
    colors = _line_colors(4)
    cmap = matplotlib.colormaps["viridis"]
    assert colors == [cmap(i / 3) for i in range(4)]
    assert colors[0] != colors[-1]


def test_group_chromatograms_empty_raises() -> None:
    handler = Handler()
    with pytest.raises(ValueError, match="no chromatograms"):
        _group_chromatograms(handler, overlay="single")
