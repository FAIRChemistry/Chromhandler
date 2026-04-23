"""Unit tests for Fitter input assembly — no MCMC.

Tests verify compute_model_inputs(), noise_prior(), observation mask,
and window slicing without running any MCMC.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from chromhandler.annotations import ArtefactSide, BaselineAnnotation, PeakAnnotation, PeakMode
from chromhandler.fitting import Fitter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gaussian(
    x: npt.NDArray[np.float64], apex: float, sigma: float, area: float
) -> npt.NDArray[np.float64]:
    return area / (sigma * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((x - apex) / sigma) ** 2)  # type: ignore[return-value]


def _make_fitter(
    n_trace: int = 4,
    n_time: int = 200,
    apex: float = 3.0,
    sigma: float = 0.04,
    area: float = 150.0,
    noise_std: float = 0.3,
    peak_mode: PeakMode = "single",
    artefact_side: ArtefactSide | None = None,
) -> Fitter:
    x = np.linspace(2.5, 3.5, n_time)
    rng = np.random.default_rng(42)
    signal = np.stack([
        _gaussian(x, apex, sigma, area) + rng.normal(0, noise_std, n_time)
        for _ in range(n_trace)
    ])
    time = np.tile(x, (n_trace, 1))

    fitter = Fitter(time, signal)
    fitter.add_baseline_annotation(BaselineAnnotation(rt_min=2.5, rt_max=2.62))
    fitter.add_baseline_annotation(BaselineAnnotation(rt_min=3.38, rt_max=3.5))

    if artefact_side:
        fitter.add_peak_annotation(
            PeakAnnotation(
                molecule_id="A", rt_min=2.7, rt_max=3.3, mode=peak_mode, artefact_side=artefact_side
            )
        )
    else:
        fitter.add_peak_annotation(
            PeakAnnotation(molecule_id="A", rt_min=2.7, rt_max=3.3, mode=peak_mode)
        )
    return fitter


# ---------------------------------------------------------------------------
# compute_model_inputs keys
# ---------------------------------------------------------------------------

_EXPECTED_PRIOR_KEYS = {
    "apex_loc", "apex_offset_scale",
    "w_left_loc", "w_left_scale",
    "w_right_loc", "w_right_scale",
    "w_min", "w_max", "dt", "n_valid",
    "window_lo", "window_hi",
    "area_gaussian_pt", "area_trapz_pt",
    "area_art_shared",
    "trace_shift_scale",
}

_EXPECTED_STRUCTURE_KEYS = {
    "peak_mode_code", "artefact_side",
    "artefact_peak_index", "free_peak_index",
    "nonfree_idx", "nonfree_position",
}

_EXPECTED_BASELINE_KEYS = {
    "baseline_intercept_loc", "baseline_intercept_scale",
    "baseline_slope_loc", "baseline_slope_scale",
}

_EXPECTED_NOISE_KEYS = {"sigma_y_prior_loc"}


@pytest.mark.unit
def test_compute_model_inputs_has_all_expected_keys() -> None:
    """compute_model_inputs() returns all expected key groups."""
    fitter = _make_fitter()
    inputs = fitter.compute_model_inputs()
    all_expected = (
        _EXPECTED_PRIOR_KEYS
        | _EXPECTED_STRUCTURE_KEYS
        | _EXPECTED_BASELINE_KEYS
        | _EXPECTED_NOISE_KEYS
    )
    missing = all_expected - set(inputs.keys())
    assert not missing, f"Missing keys: {missing}"


@pytest.mark.unit
def test_compute_model_inputs_shapes_single_peak() -> None:
    """Array shapes match n_trace and n_peak for a single-peak fitter."""
    n_trace = 4
    fitter = _make_fitter(n_trace=n_trace)
    inputs = fitter.compute_model_inputs()
    n_peak = 1

    assert inputs["apex_loc"].shape == (n_peak,)
    assert inputs["w_left_loc"].shape == (n_peak,)
    assert inputs["w_right_loc"].shape == (n_peak,)
    assert inputs["area_gaussian_pt"].shape == (n_trace, n_peak)
    assert inputs["baseline_intercept_loc"].shape == (n_trace,)
    assert inputs["sigma_y_prior_loc"].shape == (n_trace,)
    # Single peak → no artefact, no free
    assert inputs["artefact_peak_index"].shape == (0,)
    assert inputs["free_peak_index"].shape == (0,)
    assert len(inputs["nonfree_idx"]) == 1


@pytest.mark.unit
def test_compute_model_inputs_shapes_artefact_peak() -> None:
    """Artefact doublet: artefact_peak_index has 1 entry."""
    fitter = _make_fitter(peak_mode="artefact_doublet", artefact_side="right")
    inputs = fitter.compute_model_inputs()
    assert inputs["artefact_peak_index"].shape == (1,)
    assert inputs["free_peak_index"].shape == (0,)


@pytest.mark.unit
def test_compute_model_inputs_width_bounds_ordering() -> None:
    """w_min < w_loc < w_max must hold per peak; dt and n_valid populated."""
    fitter = _make_fitter()
    inputs = fitter.compute_model_inputs()
    w_min = np.asarray(inputs["w_min"])
    w_max = np.asarray(inputs["w_max"])
    w_left = np.asarray(inputs["w_left_loc"])
    w_right = np.asarray(inputs["w_right_loc"])
    dt = np.asarray(inputs["dt"])
    n_valid = np.asarray(inputs["n_valid"])

    assert np.all(w_min > 0.0)
    assert np.all(dt > 0.0)
    assert np.all(n_valid >= 1.0)
    assert np.all(w_min < w_max)
    assert np.all(w_left > w_min) and np.all(w_left < w_max)
    assert np.all(w_right > w_min) and np.all(w_right < w_max)


@pytest.mark.unit
def test_compute_model_inputs_all_positive_scale_params() -> None:
    """All scale/loc parameters are strictly positive."""
    fitter = _make_fitter()
    inputs = fitter.compute_model_inputs()
    for key in ("w_left_loc", "w_right_loc", "w_left_scale", "w_right_scale",
                "area_gaussian_pt", "area_trapz_pt", "trace_shift_scale",
                "baseline_intercept_scale", "baseline_slope_scale", "sigma_y_prior_loc"):
        arr = np.asarray(inputs[key])
        assert np.all(arr > 0), f"{key} contains non-positive values: {arr}"


# ---------------------------------------------------------------------------
# noise_prior
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_noise_prior_data_derived_floor() -> None:
    """Noise floor scales with signal magnitude, not hardcoded 1.0."""
    # High-amplitude signal
    fitter_high = _make_fitter(area=1e5)
    fitter_high.add_peak_annotation(PeakAnnotation(molecule_id="B", rt_min=2.7, rt_max=3.3, mode="single"))
    # Noise floor should be >> 1.0 for large signal
    noise_high = fitter_high.noise_prior()
    # noise_floor = 1e-3 * signal_range; for area=1e5 with sigma=0.04, peak ≈ 1e5/(0.04*sqrt(2π)) ≈ 1e6
    # signal_range >> 1, so noise_floor >> 1.0
    assert np.all(noise_high > 0.0)

    # Low-amplitude signal: noise floor << 1.0
    fitter_low = _make_fitter(area=0.01, noise_std=1e-4)
    noise_low = fitter_low.noise_prior()
    assert np.all(noise_low > 0.0)
    # Verify low-amplitude fitter noise floor is much smaller than high-amplitude
    assert np.median(noise_high) > np.median(noise_low)


@pytest.mark.unit
def test_noise_prior_all_positive() -> None:
    """noise_prior() always returns strictly positive values."""
    fitter = _make_fitter()
    noise = fitter.noise_prior()
    assert np.all(noise > 0.0)
    assert noise.shape == (fitter.n_traces,)


# ---------------------------------------------------------------------------
# Observation mask and window slicing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_observation_mask_covers_peak_and_baseline_regions() -> None:
    """Mask is True inside registered peak and baseline windows."""
    fitter = _make_fitter()
    mask = fitter.create_observation_mask()
    t = fitter.common_time()

    # Points inside peak window [2.7, 3.3]
    inside_peak = (t >= 2.7) & (t <= 3.3)
    assert np.all(mask[inside_peak]), "Some peak-window points are not masked"

    # Points inside baseline regions [2.5, 2.62] and [3.38, 3.5]
    inside_bl1 = (t >= 2.5) & (t <= 2.62)
    inside_bl2 = (t >= 3.38) & (t <= 3.5)
    assert np.all(mask[inside_bl1]), "Some baseline-1 points are not masked"
    assert np.all(mask[inside_bl2]), "Some baseline-2 points are not masked"

    # Points outside all regions should be False
    outside = ~((t >= 2.5) & (t <= 3.5))
    assert np.all(~mask[outside]), "Points outside all regions should not be masked"


@pytest.mark.unit
def test_slice_to_observed_windows_shape() -> None:
    """Sliced arrays have shape [n_trace, n_masked]."""
    fitter = _make_fitter()
    mask = fitter.create_observation_mask()
    x_sliced, y_sliced = fitter.slice_to_observed_windows()

    n_masked = int(np.sum(mask))
    assert x_sliced.shape == (fitter.n_traces, n_masked)
    assert y_sliced.shape == (fitter.n_traces, n_masked)


# ---------------------------------------------------------------------------
# Fitter basic construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fitter_shape_mismatch_raises() -> None:
    """Mismatched time/signal shapes raise ValueError."""
    time = np.linspace(0.0, 1.0, 100).reshape(1, 100)
    signal = np.ones((2, 100))
    with pytest.raises(ValueError, match="same shape"):
        Fitter(time, signal)


@pytest.mark.unit
def test_fitter_1d_input_raises() -> None:
    """1-D time/signal raise ValueError."""
    time = np.linspace(0.0, 1.0, 100)
    signal = np.ones(100)
    with pytest.raises(ValueError, match="2-D"):
        Fitter(time, signal)


@pytest.mark.unit
def test_from_handler_populates_trace_stats_if_missing() -> None:
    """Users who never call cut_chromatograms still get trace_stats."""
    from chromhandler.fitting.fitter import Fitter
    from chromhandler.handler import Handler
    from chromhandler.model import Chromatogram, Sample

    rng = np.random.default_rng(7)
    n = 4000
    handler = Handler()
    for i in range(2):
        time = np.linspace(0.0, 10.0, n)
        signal = 100.0 + rng.normal(0.0, 0.8, size=n)
        handler.samples.append(
            Sample(
                id=f"s{i}",
                chromatograms=[
                    Chromatogram(
                        id=f"c{i}", sample_id=f"s{i}",
                        time=time.tolist(), signal=signal.tolist(),
                    )
                ],
            )
        )

    assert all(
        c.trace_stats is None for s in handler.samples for c in s.chromatograms
    )

    _ = Fitter.from_handler(handler)

    for sample in handler.samples:
        for chrom in sample.chromatograms:
            assert chrom.trace_stats is not None
            assert chrom.trace_stats.sigma_noise == pytest.approx(0.8, rel=0.1)
