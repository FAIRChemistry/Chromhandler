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


import matplotlib.pyplot as plt  # noqa: E402

from chromhandler.plotting import plot_traces  # noqa: E402


def test_plot_traces_single_default() -> None:
    handler = _make_handler(n_samples=2, chroms_per_sample=2)
    fig, axes = plot_traces(handler)
    try:
        assert axes.shape == (4, 1)
        for ax in axes.flatten():
            lines = ax.get_lines()
            assert len(lines) == 1
            assert tuple(lines[0].get_color()[:3]) == matplotlib.colors.to_rgba("tab:blue")[:3]
    finally:
        plt.close(fig)


def test_plot_traces_sample_mode_groups_per_sample() -> None:
    handler = _make_handler(n_samples=2, chroms_per_sample=2)
    fig, axes = plot_traces(handler, overlay="sample")
    try:
        assert axes.shape == (2, 1)
        for ax in axes.flatten():
            assert len(ax.get_lines()) == 2
            colors = [tuple(line.get_color()) for line in ax.get_lines()]
            assert colors[0] != colors[1]
    finally:
        plt.close(fig)


def test_plot_traces_all_mode_one_ax() -> None:
    handler = _make_handler(n_samples=2, chroms_per_sample=2)
    fig, axes = plot_traces(handler, overlay="all")
    try:
        assert axes.shape == (1, 1)
        ax = axes[0, 0]
        assert len(ax.get_lines()) == 4
    finally:
        plt.close(fig)


def test_plot_traces_ax_size_drives_figsize() -> None:
    handler = _make_handler(n_samples=3, chroms_per_sample=1)
    fig, axes = plot_traces(handler, overlay="single", ax_size=(2.5, 1.5))
    try:
        assert fig.get_size_inches()[0] == pytest.approx(2.5)
        assert fig.get_size_inches()[1] == pytest.approx(3 * 1.5)
        assert axes.shape == (3, 1)
    finally:
        plt.close(fig)


def test_plot_traces_share_y() -> None:
    handler = _make_handler(n_samples=2, chroms_per_sample=1)
    # Give the two chromatograms different signal magnitudes so y-limits would
    # differ when not shared.
    handler.samples[1].chromatograms[0].signal = [
        v * 10.0 for v in handler.samples[1].chromatograms[0].signal
    ]
    fig, axes = plot_traces(handler, overlay="single", share_y=True)
    try:
        ylims = [ax.get_ylim() for ax in axes.flatten()]
        assert ylims[0] == ylims[1]
    finally:
        plt.close(fig)


def test_plot_traces_save_writes_file(tmp_path) -> None:
    handler = _make_handler(n_samples=1, chroms_per_sample=1)
    out = tmp_path / "plot.png"
    fig, _ = plot_traces(handler, save=out)
    try:
        assert out.exists() and out.stat().st_size > 0
    finally:
        plt.close(fig)
