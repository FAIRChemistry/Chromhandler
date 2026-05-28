"""Integration test for the foundations layer on real ASM kinetic data.

Exercises ``Handler.read_asm`` + ``prepare_dataset`` end-to-end against the
CV10 kinetic-series fixture (7 timepoints, single sample). Asserts shape,
sampling-rate, baseline, and noise outputs are sane on real-world data.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.prepared_dataset import prepare_dataset
from chromhandler.handler import Handler

if TYPE_CHECKING:
    from numpy.typing import NDArray

ASM_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "asm_kinetic_series"


def _times_and_signals_from_handler(
    handler: Handler,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Extract per-trace (time, signal) arrays from a Handler.

    Iterates all chromatograms across all samples in order.
    """
    times: list[NDArray[np.float64]] = []
    signals: list[NDArray[np.float64]] = []
    for sample in handler.samples:
        for chrom in sample.chromatograms:
            times.append(np.asarray(chrom.time, dtype=np.float64))
            signals.append(np.asarray(chrom.signal, dtype=np.float64))
    return times, signals


def test_prepare_dataset_on_asm_kinetic_series() -> None:
    handler = Handler.read_asm(path=ASM_DIR, mode="timecourse")
    times, signals = _times_and_signals_from_handler(handler)

    # CV10 kinetic series: 1 sample with 7 timepoints = 7 traces.
    assert len(times) == 7
    assert len(signals) == 7

    peaks = [
        PeakAnnotation(molecule_id="other", rt_min=2.55, rt_max=2.80, mode="single"),
        PeakAnnotation(molecule_id="SIH", rt_min=2.80, rt_max=3.15, mode="single"),
        PeakAnnotation(molecule_id="third", rt_min=3.15, rt_max=3.45, mode="single"),
    ]
    baselines = [
        BaselineAnnotation(rt_min=2.50, rt_max=2.52),
        BaselineAnnotation(rt_min=3.55, rt_max=3.58),
    ]

    ds = prepare_dataset(times, signals, peaks, baselines)

    assert ds.n_trace == 7
    assert ds.time.shape[0] == 7
    assert ds.dt_global > 0
    assert ds.dt_global < 0.01  # HPLC sampling well below 10 ms
    assert np.all(ds.noise_per_trace > 0)
    assert np.all(np.isfinite(ds.baseline_intercept))
    assert np.all(np.isfinite(ds.baseline_slope))
    assert ds.peak_annotations[0].n_components == 1
    assert ds.peak_annotations[1].n_components == 1
    assert ds.peak_annotations[2].n_components == 1
