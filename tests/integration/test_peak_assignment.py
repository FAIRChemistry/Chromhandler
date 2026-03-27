import math

import pytest

from chromhandler import pretty
from chromhandler.handler import Handler
from chromhandler.model import Chromatogram, Estimate, Peak, Sample
from chromhandler.molecule import Molecule


def _molecule(molecule_id: str = "test_mol") -> Molecule:
    return Molecule(
        id=molecule_id,
        pubchem_cid=12345,
        name="Test Molecule",
    )


def _peak(
    retention_time: float,
    area: float,
    *,
    amplitude: float | None = None,
    chrom_id: str = "chrom_0",
    molecule_id: str | None = None,
) -> Peak:
    return Peak(
        chromatogram_id=chrom_id,
        location=Estimate(mean=retention_time),
        area=Estimate(mean=area),
        amplitude=amplitude,
        molecule_id=molecule_id,
    )


def _chromatogram(
    chrom_id: str,
    sample_id: str,
    peaks: list[Peak],
    *,
    wavelength: float | None = 254.0,
) -> Chromatogram:
    return Chromatogram(
        id=chrom_id,
        sample_id=sample_id,
        peaks=peaks,
        wavelength=wavelength,
    )


def _sample(sample_id: str, *chromatograms: Chromatogram) -> Sample:
    return Sample(id=sample_id, chromatograms=list(chromatograms))


def _handler(*samples: Sample, molecules: list[Molecule] | None = None) -> Handler:
    mol_dict = {mol.id: mol for mol in (molecules or [])}
    return Handler(samples=list(samples), molecules=mol_dict)


@pytest.mark.integration
def test_add_peak_window_requires_existing_molecule() -> None:
    handler = Handler()

    with pytest.raises(ValueError, match="unknown"):
        handler.add_peak_window("unknown", 4.8, 5.2)


@pytest.mark.integration
def test_assign_molecules_assigns_single_matching_peak() -> None:
    peak = _peak(5.0, 500.0, amplitude=250.0, chrom_id="chrom_1")
    handler = _handler(
        _sample("sample_1", _chromatogram("chrom_1", "sample_1", [peak])),
        molecules=[_molecule()],
    )
    handler.add_peak_window("test_mol", 4.8, 5.2)

    handler.assign_molecules(silent=True)

    assert peak.molecule_id == "test_mol"


@pytest.mark.integration
def test_assign_molecules_reports_missing_peaks() -> None:
    peak = _peak(3.0, 500.0, amplitude=250.0, chrom_id="chrom_1")
    handler = _handler(
        _sample("sample_1", _chromatogram("chrom_1", "sample_1", [peak])),
        molecules=[_molecule()],
    )
    handler.add_peak_window("test_mol", 4.8, 5.2)

    handler.assign_molecules(silent=True)

    assert peak.molecule_id is None


@pytest.mark.integration
def test_assign_molecules_raises_for_multiple_peaks_in_window() -> None:
    peaks = [
        _peak(4.95, 300.0, amplitude=180.0, chrom_id="chrom_1"),
        _peak(5.05, 400.0, amplitude=220.0, chrom_id="chrom_1"),
    ]
    handler = _handler(
        _sample("sample_1", _chromatogram("chrom_1", "sample_1", peaks)),
        molecules=[_molecule()],
    )
    handler.add_peak_window("test_mol", 4.8, 5.2)

    with pytest.raises(ValueError, match="matched multiple peaks"):
        handler.assign_molecules(silent=True)

    assert all(peak.molecule_id is None for peak in peaks)


@pytest.mark.integration
def test_assign_molecules_skip_multiple_peaks_in_window() -> None:
    ambiguous_peaks = [
        _peak(4.95, 300.0, amplitude=180.0, chrom_id="chrom_1"),
        _peak(5.05, 400.0, amplitude=220.0, chrom_id="chrom_1"),
    ]
    single_peak = _peak(5.0, 350.0, amplitude=200.0, chrom_id="chrom_2")
    handler = _handler(
        _sample(
            "sample_1",
            _chromatogram("chrom_1", "sample_1", ambiguous_peaks),
            _chromatogram("chrom_2", "sample_1", [single_peak]),
        ),
        molecules=[_molecule()],
    )
    handler.add_peak_window("test_mol", 4.8, 5.2)

    handler.assign_molecules(on_multiple="skip", silent=True)

    assert all(peak.molecule_id is None for peak in ambiguous_peaks)
    assert single_peak.molecule_id == "test_mol"


@pytest.mark.integration
def test_assign_molecules_filters_small_artefact_peaks_by_amplitude() -> None:
    artefact = _peak(4.95, 50.0, amplitude=25.0, chrom_id="chrom_1")
    main_peak = _peak(5.02, 400.0, amplitude=250.0, chrom_id="chrom_1")
    handler = _handler(
        _sample("sample_1", _chromatogram("chrom_1", "sample_1", [artefact, main_peak])),
        molecules=[_molecule()],
    )
    handler.add_peak_window("test_mol", 4.8, 5.2)

    handler.assign_molecules(min_amplitude=100.0, silent=True)

    assert artefact.molecule_id is None
    assert main_peak.molecule_id == "test_mol"


@pytest.mark.integration
def test_assign_molecules_skip_mode_separates_missing_and_ambiguous() -> None:
    ambiguous_peaks = [
        _peak(4.95, 300.0, amplitude=180.0, chrom_id="chrom_1"),
        _peak(5.05, 400.0, amplitude=220.0, chrom_id="chrom_1"),
    ]
    handler = _handler(
        _sample(
            "sample_1",
            _chromatogram("chrom_1", "sample_1", ambiguous_peaks),
            _chromatogram("chrom_2", "sample_1", []),
        ),
        molecules=[_molecule()],
    )
    handler.add_peak_window("test_mol", 4.8, 5.2)

    handler.assign_molecules(on_multiple="skip", silent=True)


@pytest.mark.integration
def test_assign_molecules_skip_mode_still_reports_ambiguity_after_filtering() -> None:
    peaks = [
        _peak(4.95, 50.0, amplitude=25.0, chrom_id="chrom_1"),
        _peak(5.02, 400.0, amplitude=250.0, chrom_id="chrom_1"),
        _peak(5.08, 380.0, amplitude=240.0, chrom_id="chrom_1"),
    ]
    handler = _handler(
        _sample("sample_1", _chromatogram("chrom_1", "sample_1", peaks)),
        molecules=[_molecule()],
    )
    handler.add_peak_window("test_mol", 4.8, 5.2)

    handler.assign_molecules(
        min_amplitude=100.0,
        on_multiple="skip",
        silent=True,
    )

    assert all(peak.molecule_id is None for peak in peaks)


@pytest.mark.integration
def test_assign_molecules_is_idempotent_for_targeted_molecules() -> None:
    first_peak = _peak(5.0, 300.0, amplitude=200.0, chrom_id="chrom_1")
    second_peak = _peak(6.0, 350.0, amplitude=220.0, chrom_id="chrom_1")
    handler = _handler(
        _sample("sample_1", _chromatogram("chrom_1", "sample_1", [first_peak, second_peak])),
        molecules=[_molecule()],
    )
    handler.add_peak_window("test_mol", 4.8, 5.2)
    handler.assign_molecules(silent=True)

    handler.add_peak_window("test_mol", 5.8, 6.2)
    handler.assign_molecules(silent=True)

    assert first_peak.molecule_id is None
    assert second_peak.molecule_id == "test_mol"


@pytest.mark.integration
def test_assign_molecules_skip_mode_clears_previous_assignments() -> None:
    first_peak = _peak(5.0, 300.0, amplitude=200.0, chrom_id="chrom_1")
    handler = _handler(
        _sample("sample_1", _chromatogram("chrom_1", "sample_1", [first_peak])),
        molecules=[_molecule()],
    )
    handler.add_peak_window("test_mol", 4.8, 5.2)
    handler.assign_molecules(silent=True)

    second_peak = _peak(5.05, 280.0, amplitude=190.0, chrom_id="chrom_1")
    handler.samples[0].chromatograms[0].peaks.append(second_peak)

    handler.assign_molecules(on_multiple="skip", silent=True)

    assert first_peak.molecule_id is None
    assert second_peak.molecule_id is None


@pytest.mark.integration
def test_assign_molecules_assigns_across_multiple_chromatograms_per_sample() -> None:
    """Time-course style: one sample, several chromatograms, same wavelength."""
    p0 = _peak(5.0, 300.0, amplitude=200.0, chrom_id="chrom_t0")
    p30 = _peak(5.02, 280.0, amplitude=190.0, chrom_id="chrom_t30")
    sample = _sample(
        "sample_1",
        _chromatogram("chrom_t0", "sample_1", [p0], wavelength=254.0),
        _chromatogram("chrom_t30", "sample_1", [p30], wavelength=254.0),
    )
    handler = _handler(sample, molecules=[_molecule()])
    handler.add_peak_window("test_mol", 4.8, 5.2)

    handler.assign_molecules(silent=True)

    assert p0.molecule_id == "test_mol"
    assert p30.molecule_id == "test_mol"


@pytest.mark.integration
def test_assign_molecules_uses_window_wavelength_to_pick_chromatogram() -> None:
    peak_254 = _peak(5.0, 300.0, amplitude=200.0, chrom_id="chrom_254")
    peak_280 = _peak(5.02, 450.0, amplitude=260.0, chrom_id="chrom_280")
    sample = _sample(
        "sample_1",
        _chromatogram("chrom_254", "sample_1", [peak_254], wavelength=254.0),
        _chromatogram("chrom_280", "sample_1", [peak_280], wavelength=280.0),
    )
    handler = _handler(sample, molecules=[_molecule()])
    handler.add_peak_window("test_mol", 4.8, 5.2, wavelength=280.0)

    handler.assign_molecules(silent=True)

    assert peak_254.molecule_id is None
    assert peak_280.molecule_id == "test_mol"


@pytest.mark.integration
def test_unassign_peaks_clears_specific_molecule_in_selected_chromatogram() -> None:
    peak_target = _peak(5.0, 300.0, chrom_id="chrom_1", molecule_id="test_mol")
    peak_other = _peak(6.0, 200.0, chrom_id="chrom_1", molecule_id="other_mol")
    handler = _handler(
        _sample("sample_1", _chromatogram("chrom_1", "sample_1", [peak_target, peak_other])),
        molecules=[_molecule(), _molecule("other_mol")],
    )

    removed = handler.unassign_peaks(
        chromatogram_ids=["chrom_1"],
        molecule_ids=["test_mol"],
    )

    assert peak_target.molecule_id is None
    assert peak_other.molecule_id == "other_mol"
    assert removed == [
        {
            "sample_id": "sample_1",
            "chromatogram_id": "chrom_1",
            "reaction_time": None,
            "peak_index": 0,
            "peak_rt": 5.0,
            "previous_molecule_id": "test_mol",
        }
    ]


@pytest.mark.integration
def test_unassign_peaks_can_match_sample_and_reaction_time() -> None:
    peak_t0 = _peak(5.0, 300.0, chrom_id="chrom_t0", molecule_id="test_mol")
    peak_t30 = _peak(5.0, 280.0, chrom_id="chrom_t30", molecule_id="test_mol")
    sample = _sample(
        "sample_1",
        Chromatogram(
            id="chrom_t0",
            sample_id="sample_1",
            peaks=[peak_t0],
            wavelength=254.0,
            reaction_time=0.0,
        ),
        Chromatogram(
            id="chrom_t30",
            sample_id="sample_1",
            peaks=[peak_t30],
            wavelength=254.0,
            reaction_time=30.0,
        ),
    )
    handler = _handler(sample, molecules=[_molecule()])

    removed = handler.unassign_peaks(
        sample_ids=["sample_1"],
        molecule_ids=["test_mol"],
        reaction_times=[30.0],
    )

    assert peak_t0.molecule_id == "test_mol"
    assert peak_t30.molecule_id is None
    assert removed[0]["chromatogram_id"] == "chrom_t30"


@pytest.mark.integration
def test_unassign_peaks_raises_for_unknown_chromatogram_id() -> None:
    handler = _handler(
        _sample("sample_1", _chromatogram("chrom_1", "sample_1", [])),
        molecules=[_molecule()],
    )

    with pytest.raises(ValueError, match="Unknown chromatogram IDs"):
        handler.unassign_peaks(chromatogram_ids=["missing"])


@pytest.mark.integration
def test_peak_assignment_summary_table_handles_ambiguous_entries() -> None:
    handler = _handler(molecules=[_molecule()])
    window = handler.add_peak_window("test_mol", 4.8, 5.2)

    table = pretty.create_peak_assignment_summary_table(
        handler,
        [
            {
                "molecule": handler.get_molecule("test_mol"),
                "window": window,
                "assigned_peak_count": 1,
                "chromatograms_with_no_peaks": ["chrom_missing"],
                "chromatograms_with_multiple_peaks": ["chrom_ambiguous"],
                "chromatograms_considered": ["chrom_missing", "chrom_ambiguous"],
                "min_amplitude": 100.0,
                "on_multiple": "skip",
            }
        ],
    )

    assert [column.header for column in table.columns] == [
        "Molecule",
        "Window",
        "Assigned",
        "Missing",
        "Ambiguous",
        "Details",
    ]


@pytest.mark.integration
def test_subset_returns_independent_handler_with_copied_peak_windows() -> None:
    sample_a = _sample(
        "sample_a",
        _chromatogram(
            "chrom_a",
            "sample_a",
            [_peak(5.0, 100.0, amplitude=120.0, chrom_id="chrom_a")],
        ),
    )
    sample_b = _sample(
        "sample_b",
        _chromatogram(
            "chrom_b",
            "sample_b",
            [_peak(6.0, 200.0, amplitude=180.0, chrom_id="chrom_b")],
        ),
    )
    parent = _handler(sample_a, sample_b, molecules=[_molecule()])
    parent.add_peak_window("test_mol", 4.8, 5.2)

    child = parent.subset(["chrom_b"])
    child.add_peak_window("test_mol", 5.8, 6.2)

    assert len(child.samples) == 1
    assert child.samples[0].id == "sample_b"
    assert [chrom.id for chrom in child.samples[0].chromatograms] == ["chrom_b"]
    assert math.isclose(parent.peak_windows["test_mol"].rt_min, 4.8, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(child.peak_windows["test_mol"].rt_min, 5.8, rel_tol=0.0, abs_tol=1e-9)


@pytest.mark.integration
def test_subset_raises_for_unknown_chromatogram_ids() -> None:
    handler = _handler(
        _sample("sample_1", _chromatogram("chrom_1", "sample_1", [])),
        molecules=[_molecule()],
    )

    with pytest.raises(ValueError, match="not found"):
        handler.subset(["missing"])
