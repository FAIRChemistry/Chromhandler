import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

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


def test_plot_can_annotate_points_with_chromatogram_ids() -> None:
    sample = Sample(
        id="sample_1",
        chromatograms=[
            Chromatogram(
                id="chrom_t0",
                sample_id="sample_1",
                peaks=[_peak("chrom_t0", 1.0, 10.0, molecule_id="Hyp")],
                reaction_time=0.0,
            ),
            Chromatogram(
                id="chrom_t30",
                sample_id="sample_1",
                peaks=[_peak("chrom_t30", 1.0, 8.0, molecule_id="Hyp")],
                reaction_time=30.0,
            ),
        ],
    )
    handler = Handler(samples=[sample])

    fig, axes = handler.plot(show_chromatogram_ids=True)

    labels = {text.get_text() for text in axes[0].texts}
    assert {"chrom_t0", "chrom_t30"} <= labels

    plt.close(fig)
