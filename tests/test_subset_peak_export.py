import numpy as np
import pytest

from chromhandler.annotations import PeakAnnotation
from chromhandler.fitting import BetterFitter
from chromhandler.fitting.subsets import Subset
from chromhandler.handler import Handler
from chromhandler.model import Chromatogram, Sample


def _peak_annotation(
    molecule_id: str,
    *,
    mode: str = "single",
    artefact_side: str | None = None,
) -> PeakAnnotation:
    return PeakAnnotation(
        molecule_id=molecule_id,
        rt_min=0.2,
        rt_max=0.8,
        mode=mode,  # type: ignore[arg-type]
        artefact_side=artefact_side,  # type: ignore[arg-type]
    )


def _build_fitter(
    *,
    sample_id: str,
    chromatogram_id: str,
    peaks: list[PeakAnnotation],
) -> BetterFitter:
    return BetterFitter(
        np.asarray([[0.0, 0.5, 1.0]], dtype=float),
        np.asarray([[0.0, 1.0, 0.0]], dtype=float),
        peaks=peaks,
        baselines=[],
        trace_sample_ids=[sample_id],
        trace_chromatogram_ids=[chromatogram_id],
    )


def _make_samples(
    *,
    area_samples: list[float],
    apex_samples: list[float],
) -> dict:
    """Build a minimal posterior samples dict [n_sample, n_trace, n_peak]."""
    area_arr = np.asarray(area_samples, dtype=float).reshape(-1, 1, 1)
    apex_arr = np.asarray(apex_samples, dtype=float).reshape(-1, 1, 1)
    zero_arr = np.zeros_like(area_arr)
    return {
        "area_l": area_arr,
        "area_r": zero_arr,
        "apex_l": apex_arr,
        "apex_r": zero_arr,
    }


def _handler_with_chromatograms(
    *chrom_specs: tuple[str, str],
) -> Handler:
    samples: dict[str, Sample] = {}
    for sample_id, chrom_id in chrom_specs:
        sample = samples.setdefault(sample_id, Sample(id=sample_id))
        sample.chromatograms.append(Chromatogram(id=chrom_id, sample_id=sample_id, peaks=[]))
    return Handler(id="handler", name="test", samples=list(samples.values()))


class TestSubsetPeakExport:
    def test_to_peaks_and_write_fitted_peaks_non_subset(self) -> None:
        peak = _peak_annotation("mol_a")
        fitter = _build_fitter(
            sample_id="sample_a",
            chromatogram_id="chrom_a",
            peaks=[peak],
        )

        # Populate internal storage for the auto-created __default__ subset
        from chromhandler.fitting.better_fitter import _DEFAULT_SUBSET_NAME

        fitter._samples_dict[_DEFAULT_SUBSET_NAME] = _make_samples(
            area_samples=[10.0, 14.0, 18.0],
            apex_samples=[0.30, 0.31, 0.32],
        )
        fitter._subset_trace_ids[_DEFAULT_SUBSET_NAME] = np.asarray(
            ["chrom_a"], dtype=object
        )

        peaks = fitter.to_peaks()

        assert len(peaks) == 1
        assert peaks[0].chromatogram_id == "chrom_a"
        assert peaks[0].molecule_id == "mol_a"
        assert peaks[0].area.mean == pytest.approx(14.0)

        handler = _handler_with_chromatograms(("sample_a", "chrom_a"))
        written = handler.write_fitted_peaks(fitter)

        assert len(written) == 1
        stored_peak = handler.samples[0].chromatograms[0].peaks[0]
        assert stored_peak.molecule_id == "mol_a"
        assert stored_peak.area.median == pytest.approx(14.0)

    def test_to_peaks_and_write_fitted_peaks_aggregate_subset_children(self) -> None:
        peak_a = _peak_annotation("mol_a")
        peak_b = _peak_annotation("mol_b")

        # Build a parent fitter with no peaks so _subsets starts empty
        parent = _build_fitter(
            sample_id="parent_sample",
            chromatogram_id="parent_chrom",
            peaks=[],
        )

        # Manually wire subset state (bypasses validation intentionally)
        parent._subsets["subset_a"] = Subset(name="subset_a", peaks=[peak_a])
        parent._subset_trace_ids["subset_a"] = np.asarray(["chrom_a"], dtype=object)
        parent._samples_dict["subset_a"] = _make_samples(
            area_samples=[8.0, 10.0, 12.0],
            apex_samples=[0.25, 0.26, 0.27],
        )

        parent._subsets["subset_b"] = Subset(name="subset_b", peaks=[peak_b])
        parent._subset_trace_ids["subset_b"] = np.asarray(["chrom_b"], dtype=object)
        parent._samples_dict["subset_b"] = _make_samples(
            area_samples=[20.0, 22.0, 24.0],
            apex_samples=[0.65, 0.66, 0.67],
        )

        peaks = parent.to_peaks()

        assert parent.posteriors == {}
        assert [(peak.chromatogram_id, peak.molecule_id) for peak in peaks] == [
            ("chrom_a", "mol_a"),
            ("chrom_b", "mol_b"),
        ]

        handler = _handler_with_chromatograms(
            ("sample_a", "chrom_a"),
            ("sample_b", "chrom_b"),
        )
        written = handler.write_fitted_peaks(parent)

        assert len(written) == 2
        chrom_a = next(
            chrom
            for sample in handler.samples
            for chrom in sample.chromatograms
            if chrom.id == "chrom_a"
        )
        chrom_b = next(
            chrom
            for sample in handler.samples
            for chrom in sample.chromatograms
            if chrom.id == "chrom_b"
        )
        assert chrom_a.peaks[0].molecule_id == "mol_a"
        assert chrom_b.peaks[0].molecule_id == "mol_b"

    def test_to_peaks_raises_for_duplicate_subset_peak_keys(self) -> None:
        peak = _peak_annotation("mol_a")
        parent = _build_fitter(
            sample_id="parent_sample",
            chromatogram_id="parent_chrom",
            peaks=[],
        )

        # Two subsets covering the same chromatogram_id — should raise on to_peaks()
        parent._subsets["subset_one"] = Subset(name="subset_one", peaks=[peak])
        parent._subset_trace_ids["subset_one"] = np.asarray(["chrom_dup"], dtype=object)
        parent._samples_dict["subset_one"] = _make_samples(
            area_samples=[5.0, 6.0, 7.0],
            apex_samples=[0.20, 0.21, 0.22],
        )

        parent._subsets["subset_two"] = Subset(name="subset_two", peaks=[peak])
        parent._subset_trace_ids["subset_two"] = np.asarray(["chrom_dup"], dtype=object)
        parent._samples_dict["subset_two"] = _make_samples(
            area_samples=[15.0, 16.0, 17.0],
            apex_samples=[0.70, 0.71, 0.72],
        )

        with pytest.raises(ValueError, match="duplicate fitted peaks across subsets"):
            parent.to_peaks()

    def test_to_peaks_raises_when_subsets_registered_but_not_fitted(self) -> None:
        peak = _peak_annotation("mol_a")
        parent = _build_fitter(
            sample_id="sample_a",
            chromatogram_id="chrom_a",
            peaks=[],
        )
        s = parent.add_subset("subset_one", sample_ids=["sample_a"])
        s.add_peak_annotation(peak)

        with pytest.raises(RuntimeError, match="fitted subset posteriors"):
            parent.to_peaks()
