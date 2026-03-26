import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from chromhandler.handler import Handler
from chromhandler.model import Chromatogram, Estimate, Peak, Sample


def _peak(
    chromatogram_id: str,
    retention_time: float,
    area: float,
    *,
    molecule_id: str | None = None,
) -> Peak:
    return Peak(
        chromatogram_id=chromatogram_id,
        location=Estimate(mean=retention_time),
        area=Estimate(mean=area),
        molecule_id=molecule_id,
    )


def _chromatogram(
    chrom_id: str,
    sample_id: str,
    reaction_time: float,
    signal_max: float,
    peaks: list[Peak],
) -> Chromatogram:
    return Chromatogram(
        id=chrom_id,
        sample_id=sample_id,
        time=[0.0, 1.0, 2.0],
        signal=[0.0, signal_max, 0.0],
        peaks=peaks,
        reaction_time=reaction_time,
        wavelength=254.0,
    )


def _sample(sample_id: str, *chromatograms: Chromatogram) -> Sample:
    return Sample(id=sample_id, chromatograms=list(chromatograms))


@pytest.fixture
def handler_with_chromatograms() -> Handler:
    sample_a = _sample(
        "sample_a",
        _chromatogram(
            "chrom_a1",
            "sample_a",
            0.0,
            50.0,
            [_peak("chrom_a1", 1.0, 12.0, molecule_id="mol")],
        ),
        _chromatogram(
            "chrom_a2",
            "sample_a",
            1.0,
            5.0,
            [
                _peak("chrom_a2", 1.0, 8.0, molecule_id="mol"),
                _peak("chrom_a2", 1.6, 3.0),
            ],
        ),
    )
    sample_b = _sample(
        "sample_b",
        _chromatogram(
            "chrom_b1",
            "sample_b",
            2.0,
            8.0,
            [_peak("chrom_b1", 1.1, 6.0, molecule_id="mol")],
        ),
    )

    handler = Handler(samples=[sample_a, sample_b])
    handler.create_molecule(id="mol", pubchem_cid=1, name="Mol")
    handler.add_peak_window("mol", 0.8, 1.2)
    return handler


def test_visualize_empty_chromatogram_ids_plots_all(
    handler_with_chromatograms: Handler,
) -> None:
    fig, axes = handler_with_chromatograms.visualize(
        chromatogram_ids=[],
        show_peaks=False,
        show_peak_annotations=False,
    )

    assert len(axes) == 2
    assert [ax.texts[0].get_text() for ax in axes] == ["sample_a", "sample_b"]
    assert len(axes[0].lines) == 2
    assert len(axes[1].lines) == 1

    plt.close(fig)


def test_visualize_filters_to_selected_chromatogram(
    handler_with_chromatograms: Handler,
) -> None:
    fig, axes = handler_with_chromatograms.visualize(
        chromatogram_ids=["chrom_a2"],
        show_peaks=False,
        show_peak_annotations=False,
    )

    assert len(axes) == 1
    assert axes[0].texts[0].get_text() == "sample_a"
    assert len(axes[0].lines) == 1
    assert max(axes[0].lines[0].get_ydata()) == pytest.approx(5.0)

    plt.close(fig)


def test_visualize_filter_drops_empty_samples_and_reduces_axes(
    handler_with_chromatograms: Handler,
) -> None:
    fig, axes = handler_with_chromatograms.visualize(
        chromatogram_ids=["chrom_a2", "chrom_b1"],
        show_peaks=False,
        show_peak_annotations=False,
    )

    assert len(axes) == 2
    assert [ax.texts[0].get_text() for ax in axes] == ["sample_a", "sample_b"]
    assert all(len(ax.lines) == 1 for ax in axes)

    plt.close(fig)


def test_visualize_raises_for_unknown_chromatogram_ids(
    handler_with_chromatograms: Handler,
) -> None:
    with pytest.raises(ValueError, match="missing"):
        handler_with_chromatograms.visualize(chromatogram_ids=["missing"])


def test_visualize_overlay_uses_filtered_data_for_lines_and_limits(
    handler_with_chromatograms: Handler,
) -> None:
    fig, ax = handler_with_chromatograms.visualize(
        chromatogram_ids=["chrom_a2"],
        overlay=True,
        show_peak_annotations=False,
    )

    assert len(ax.lines) == 3
    assert ax.get_ylim()[1] < 10.0

    plt.close(fig)


def test_visualize_assigned_only_respects_chromatogram_filter(
    handler_with_chromatograms: Handler,
) -> None:
    fig, ax = handler_with_chromatograms.visualize(
        chromatogram_ids=["chrom_a2"],
        overlay=True,
        assigned_only=True,
        show_peak_annotations=False,
    )

    peak_lines = [line for line in ax.lines if len(line.get_xdata()) == 2]

    assert len(peak_lines) == 1
    assert list(peak_lines[0].get_xdata()) == [1.0, 1.0]

    plt.close(fig)


def test_visualize_uses_handler_peak_windows_when_filtering(
    handler_with_chromatograms: Handler,
) -> None:
    fig, ax = handler_with_chromatograms.visualize(
        chromatogram_ids=["chrom_a2"],
        overlay=True,
    )

    assert len(ax.patches) >= 1

    plt.close(fig)
