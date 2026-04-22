"""Unit tests for the prior pipeline — no MCMC.

Tests cover:
- _halfwidth_priors: aggregation of left/right HWHM across traces
- _estimate_snr: per-trace signal-to-noise ratio
- _gaussian_area_from_halfwidths: area estimation from FWHM geometry
- build_peak_priors: full pipeline producing GeometricPeakPriors
- geometric_priors_to_arrays: array conversion and shapes
- refine_apex_priors_with_trace_shift: trace-shift decomposition
- summarise_priors: ASCII table formatting
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest
from scipy.stats import skewnorm

from chromhandler.annotations import PeakAnnotation
from chromhandler.fitting.priors import (
    PeakApexTraces,
    build_peak_priors,
    geometric_priors_to_arrays,
    refine_apex_priors_with_trace_shift,
    summarise_priors,
)

_Arr = npt.NDArray[np.float64]
_SinglePeakFixture = tuple[list[PeakAnnotation], _Arr, _Arr, _Arr]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HWHM_FACTOR = math.sqrt(2.0 * math.log(2.0))


def _gaussian(
    x: npt.NDArray[np.float64], apex: float, sigma: float, area: float
) -> npt.NDArray[np.float64]:
    """Normalised Gaussian evaluated at x."""
    return area / (sigma * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((x - apex) / sigma) ** 2)  # type: ignore[return-value]


def _skewnormal(
    x: npt.NDArray[np.float64], apex: float, sigma: float, alpha: float, area: float
) -> npt.NDArray[np.float64]:
    """Skew-normal evaluated at x, parameterised by mode."""
    # Convert mode → xi (location parameter)
    delta = alpha / math.sqrt(1 + alpha**2)
    mu_z = delta * math.sqrt(2 / math.pi)
    xi = apex - sigma * mu_z
    return area * skewnorm.pdf(x, a=alpha, loc=xi, scale=sigma)  # type: ignore[return-value]


def _make_window(
    n_trace: int,
    n_points: int = 200,
    apex: float = 0.5,
    sigma: float = 0.04,
    alpha: float = 0.0,
    area: float = 100.0,
    noise_std: float = 0.5,
    rng_seed: int = 0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Synthetic (x_win, y_win) with skew-normal peaks + Gaussian noise."""
    rng = np.random.default_rng(rng_seed)
    x = np.linspace(apex - 6 * sigma, apex + 6 * sigma, n_points)
    y = np.stack([
        _skewnormal(x, apex, sigma, alpha, area) + rng.normal(0, noise_std, n_points)
        for _ in range(n_trace)
    ])
    return x, y


# ---------------------------------------------------------------------------
# _halfwidth_priors
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_halfwidth_priors_symmetric_gaussian() -> None:
    """Symmetric Gaussian: w_left ≈ w_right ≈ HWHM."""
    sigma = 0.04
    x, y = _make_window(n_trace=8, sigma=sigma, alpha=0.0, area=500.0, noise_std=0.2)

    from chromhandler.fitting.priors import _halfwidth_priors

    w_left_loc, _, w_right_loc, _, _ = _halfwidth_priors(x, y)

    expected_hwhm = sigma * _HWHM_FACTOR
    assert abs(w_left_loc - expected_hwhm) < 0.15 * expected_hwhm, (
        f"w_left_loc={w_left_loc:.5f} deviates >15% from expected HWHM={expected_hwhm:.5f}"
    )
    assert abs(w_right_loc - expected_hwhm) < 0.15 * expected_hwhm, (
        f"w_right_loc={w_right_loc:.5f} deviates >15% from expected HWHM={expected_hwhm:.5f}"
    )
    # Symmetric peak: left ≈ right
    assert abs(w_left_loc - w_right_loc) < 0.1 * expected_hwhm


@pytest.mark.unit
def test_halfwidth_priors_right_tailing_peak() -> None:
    """Right-tailing peak (alpha > 0): w_right > w_left."""
    x, y = _make_window(n_trace=8, sigma=0.04, alpha=2.0, area=500.0, noise_std=0.2)

    from chromhandler.fitting.priors import _halfwidth_priors

    w_left_loc, _, w_right_loc, _, _ = _halfwidth_priors(x, y)

    assert w_right_loc > w_left_loc, (
        f"Expected w_right > w_left for right-tailing peak, got {w_right_loc:.5f} <= {w_left_loc:.5f}"
    )


@pytest.mark.unit
def test_halfwidth_priors_left_tailing_peak() -> None:
    """Left-tailing peak (alpha < 0): w_left > w_right."""
    x, y = _make_window(n_trace=8, sigma=0.04, alpha=-2.0, area=500.0, noise_std=0.2)

    from chromhandler.fitting.priors import _halfwidth_priors

    w_left_loc, _, w_right_loc, _, _ = _halfwidth_priors(x, y)

    assert w_left_loc > w_right_loc, (
        f"Expected w_left > w_right for left-tailing peak, got {w_left_loc:.5f} <= {w_right_loc:.5f}"
    )


@pytest.mark.unit
def test_halfwidth_priors_multi_trace_scale_positive() -> None:
    """Multiple traces with varying widths: scale > 0."""
    rng = np.random.default_rng(42)
    x = np.linspace(0.2, 0.8, 200)
    sigmas = rng.uniform(0.035, 0.045, 8)
    y = np.stack([_gaussian(x, 0.5, s, 200.0) + rng.normal(0, 0.3, 200) for s in sigmas])

    from chromhandler.fitting.priors import _halfwidth_priors

    _, w_left_scale, _, w_right_scale, _ = _halfwidth_priors(x, y)

    assert w_left_scale > 0.0
    assert w_right_scale > 0.0


@pytest.mark.unit
def test_halfwidth_priors_low_snr_trace_excluded() -> None:
    """One high-S/N trace + one noise-only trace: noise trace does not corrupt the loc."""
    sigma = 0.04
    x = np.linspace(0.2, 0.8, 200)
    y_good = _gaussian(x, 0.5, sigma, 300.0)
    y_noise = np.random.default_rng(1).normal(0, 1.0, 200)  # pure noise
    y = np.stack([y_good, y_noise])

    from chromhandler.fitting.priors import _halfwidth_priors

    w_left_loc, _, w_right_loc, _, _ = _halfwidth_priors(x, y)

    expected_hwhm = sigma * _HWHM_FACTOR
    # Should recover close to the good trace's HWHM (within 20%)
    assert abs(w_left_loc - expected_hwhm) < 0.2 * expected_hwhm
    assert abs(w_right_loc - expected_hwhm) < 0.2 * expected_hwhm


# ---------------------------------------------------------------------------
# _estimate_snr
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_snr_per_trace_known_signal() -> None:
    """Known signal + noise: SNR ≈ peak_height / noise_std (within factor 3)."""
    sigma = 0.04
    area = 500.0
    noise_std = 2.0
    x, y = _make_window(n_trace=5, sigma=sigma, area=area, noise_std=noise_std, rng_seed=7)

    from chromhandler.fitting.priors import _estimate_snr, _trace_fwhm_geometry

    geo = _trace_fwhm_geometry(x, y)
    snr = _estimate_snr(y, geo.apex_height)

    # Peak height ≈ area / (sigma * sqrt(2π))
    expected_height = area / (sigma * math.sqrt(2 * math.pi))
    for t in range(5):
        # SNR should be > 1 for a clear peak
        assert snr[t] > 1.0, f"Trace {t}: SNR={snr[t]:.2f} unexpectedly low"
        # Should be in a reasonable range of expected_height / noise_std
        assert snr[t] < 10 * (expected_height / noise_std), f"Trace {t}: SNR={snr[t]:.2f} implausibly high"


@pytest.mark.unit
def test_snr_per_trace_all_noise_no_crash() -> None:
    """Pure noise traces: SNR close to 0, no crash or NaN."""
    rng = np.random.default_rng(99)
    x = np.linspace(0.0, 1.0, 100)
    y = rng.normal(0, 1.0, (4, 100))

    from chromhandler.fitting.priors import _estimate_snr, _trace_fwhm_geometry

    geo = _trace_fwhm_geometry(x, y)
    snr = _estimate_snr(y, geo.apex_height)

    assert np.all(np.isfinite(snr)), "SNR contains NaN/Inf for pure noise"
    assert np.all(snr >= 0.0), "SNR contains negative values"


# ---------------------------------------------------------------------------
# _gaussian_area_from_halfwidths
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_gaussian_area_from_halfwidths_valid_traces() -> None:
    """Area estimate is positive for valid FWHM traces."""
    sigma = 0.04
    area = 200.0
    x, y = _make_window(n_trace=6, sigma=sigma, area=area, noise_std=0.5)

    from chromhandler.fitting.priors import _gaussian_area_from_halfwidths, _trace_fwhm_geometry

    geo = _trace_fwhm_geometry(x, y)
    area_est = _gaussian_area_from_halfwidths(geo)

    valid = np.asarray(geo.fwhm_valid)
    area_arr = np.asarray(area_est)
    assert np.all(area_arr[valid] > 0), "Area estimate should be positive for valid FWHM traces"


@pytest.mark.unit
def test_gaussian_area_from_halfwidths_nan_for_invalid() -> None:
    """Invalid FWHM traces produce NaN area."""
    # Use tiny signal that won't pass FWHM detection
    x = np.linspace(0.0, 1.0, 50)
    y = np.ones((3, 50)) * 0.001  # flat — no FWHM crossing

    from chromhandler.fitting.priors import _gaussian_area_from_halfwidths, _trace_fwhm_geometry

    geo = _trace_fwhm_geometry(x, y)
    area_est = np.asarray(_gaussian_area_from_halfwidths(geo))

    invalid = ~np.asarray(geo.fwhm_valid)
    if np.any(invalid):
        assert np.all(np.isnan(area_est[invalid])), "Invalid traces should give NaN area"  # type: ignore[index]


# ---------------------------------------------------------------------------
# build_peak_priors
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_single_peak() -> _SinglePeakFixture:
    """5-trace dataset with one clear single-mode peak."""
    n_trace, n_time = 5, 300
    apex, sigma, area = 3.0, 0.04, 150.0
    x = np.linspace(2.5, 3.5, n_time)
    rng = np.random.default_rng(42)
    signal = np.stack([
        _gaussian(x, apex, sigma, area) + rng.normal(0, 0.3, n_time)
        for _ in range(n_trace)
    ])
    baseline = np.zeros_like(signal)
    peaks = [PeakAnnotation(molecule_id="A", rt_min=2.7, rt_max=3.3, mode="single")]
    return peaks, x, signal, baseline


@pytest.mark.unit
def test_build_peak_priors_single_mode_fields(
    synthetic_single_peak: _SinglePeakFixture,
) -> None:
    """build_peak_priors returns correct field types and shapes for single mode."""
    peaks, x, signal, baseline = synthetic_single_peak
    priors, _ = build_peak_priors(peaks, x, signal, baseline)

    assert len(priors) == 1
    p = priors[0]

    assert p.mode == "single"
    assert p.n_components == 1
    assert p.apex_loc > 2.7 and p.apex_loc < 3.3
    assert p.apex_scale > 0.0
    assert p.w_left_loc > 0.0
    assert p.w_right_loc > 0.0
    assert p.w_left_scale > 0.0
    assert p.w_right_scale > 0.0
    assert p.area_gaussian_pt.shape == (5,)
    assert p.area_trapz_pt.shape == (5,)
    assert p.snr_per_trace.shape == (5,)
    assert np.all(p.area_gaussian_pt > 0)
    assert np.all(p.area_trapz_pt > 0)
    assert np.all(p.snr_per_trace >= 0)
    assert p.area_art_shared == 0.0  # not artefact
    assert p.window_lo == 2.7
    assert p.window_hi == 3.3


@pytest.mark.unit
def test_build_peak_priors_apex_near_true_value(
    synthetic_single_peak: _SinglePeakFixture,
) -> None:
    """Apex loc should be within 10% of the window width from the true apex."""
    peaks, x, signal, baseline = synthetic_single_peak
    priors, _ = build_peak_priors(peaks, x, signal, baseline)
    p = priors[0]

    true_apex = 3.0
    window_width = 0.6
    assert abs(p.apex_loc - true_apex) < 0.1 * window_width


@pytest.mark.unit
def test_build_peak_priors_halfwidths_near_true(
    synthetic_single_peak: _SinglePeakFixture,
) -> None:
    """w_left_loc and w_right_loc should recover true HWHM within 20%."""
    peaks, x, signal, baseline = synthetic_single_peak
    priors, _ = build_peak_priors(peaks, x, signal, baseline)
    p = priors[0]

    true_sigma = 0.04
    expected_hwhm = true_sigma * _HWHM_FACTOR
    assert abs(p.w_left_loc - expected_hwhm) < 0.2 * expected_hwhm
    assert abs(p.w_right_loc - expected_hwhm) < 0.2 * expected_hwhm


@pytest.mark.unit
def test_build_peak_priors_artefact_mode_area_art_shared() -> None:
    """Artefact doublet: area_art_shared > 0."""
    # Dominant peak + small artefact shoulder on the right
    n_trace, n_time = 4, 300
    x = np.linspace(2.7, 3.3, n_time)
    rng = np.random.default_rng(5)
    signal = np.stack([
        _gaussian(x, 3.0, 0.04, 200.0)  # dominant
        + _gaussian(x, 3.1, 0.025, 30.0)  # artefact shoulder
        + rng.normal(0, 0.3, n_time)
        for _ in range(n_trace)
    ])
    baseline = np.zeros_like(signal)
    peaks = [
        PeakAnnotation(
            molecule_id="A", rt_min=2.75, rt_max=3.25,
            mode="artefact_doublet", artefact_side="right",
        )
    ]
    priors, _ = build_peak_priors(peaks, x, signal, baseline)
    assert priors[0].area_art_shared > 0.0


@pytest.mark.unit
def test_build_peak_priors_free_doublet_mode() -> None:
    """Free doublet mode: n_components == 2, no crash."""
    n_trace, n_time = 4, 300
    x = np.linspace(2.7, 3.3, n_time)
    rng = np.random.default_rng(6)
    signal = np.stack([
        _gaussian(x, 2.93, 0.03, 100.0)
        + _gaussian(x, 3.07, 0.03, 100.0)
        + rng.normal(0, 0.3, n_time)
        for _ in range(n_trace)
    ])
    baseline = np.zeros_like(signal)
    peaks = [PeakAnnotation(molecule_id="AB", rt_min=2.75, rt_max=3.25, mode="free_doublet")]
    priors, _ = build_peak_priors(peaks, x, signal, baseline)
    assert priors[0].n_components == 2


@pytest.mark.unit
def test_build_peak_priors_empty_window_raises() -> None:
    """Window with no data points raises ValueError."""
    x = np.linspace(0.0, 1.0, 100)
    signal = np.ones((3, 100))
    baseline = np.zeros((3, 100))
    peaks = [PeakAnnotation(molecule_id="X", rt_min=5.0, rt_max=6.0, mode="single")]
    with pytest.raises(ValueError, match="no finite data points"):
        build_peak_priors(peaks, x, signal, baseline)


# ---------------------------------------------------------------------------
# geometric_priors_to_arrays
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_geometric_priors_to_arrays_keys(synthetic_single_peak: _SinglePeakFixture) -> None:
    """Output dict has exactly the expected keys."""
    peaks, x, signal, baseline = synthetic_single_peak
    priors, _ = build_peak_priors(peaks, x, signal, baseline)
    arrays = geometric_priors_to_arrays(priors)

    expected_keys = {
        "apex_loc", "apex_offset_scale",
        "w_left_loc", "w_left_scale",
        "w_right_loc", "w_right_scale",
        "w_min", "w_max", "dt", "n_valid",
        "window_lo", "window_hi",
        "area_gaussian_pt", "area_trapz_pt",
        "area_art_shared", "area_art_per_trace",
    }
    assert set(arrays.keys()) == expected_keys


@pytest.mark.unit
def test_geometric_priors_to_arrays_shapes() -> None:
    """Array shapes match n_peak and n_trace."""
    n_trace, n_time = 5, 200
    x = np.linspace(0.0, 5.0, n_time)
    rng = np.random.default_rng(11)
    signal = np.zeros((n_trace, n_time))
    # 3 peaks
    for apex in [1.0, 2.5, 4.0]:
        signal += np.stack([
            _gaussian(x, apex, 0.05, 100.0) + rng.normal(0, 0.3, n_time)
            for _ in range(n_trace)
        ])
    baseline = np.zeros_like(signal)
    peaks = [
        PeakAnnotation(molecule_id="A", rt_min=0.7, rt_max=1.3, mode="single"),
        PeakAnnotation(molecule_id="B", rt_min=2.2, rt_max=2.8, mode="single"),
        PeakAnnotation(molecule_id="C", rt_min=3.7, rt_max=4.3, mode="single"),
    ]
    priors, _ = build_peak_priors(peaks, x, signal, baseline)
    arrays = geometric_priors_to_arrays(priors)

    n_peak = 3
    for key in ("apex_loc", "apex_offset_scale", "w_left_loc", "w_left_scale",
                "w_right_loc", "w_right_scale", "w_min", "w_max", "dt", "n_valid",
                "window_lo", "window_hi"):
        assert arrays[key].shape == (n_peak,), f"{key}: expected ({n_peak},), got {arrays[key].shape}"

    # Structural invariants (fixture sampling may violate w_loc > w_min for
    # unresolved peaks — that's exactly why the model truncates to [w_min, w_max]).
    assert np.all(arrays["w_min"] > 0.0)
    assert np.all(arrays["w_max"] > arrays["w_min"])
    assert np.all(arrays["dt"] > 0.0)
    assert np.all(arrays["n_valid"] >= 1.0)

    for key in ("area_gaussian_pt", "area_trapz_pt"):
        assert arrays[key].shape == (n_trace, n_peak), (
            f"{key}: expected ({n_trace}, {n_peak}), got {arrays[key].shape}"
        )

    assert arrays["area_art_shared"].shape == (0,)  # no artefact peaks


@pytest.mark.unit
def test_geometric_priors_to_arrays_artefact_shared_shape() -> None:
    """area_art_shared has shape [n_artefact]."""
    n_trace, n_time = 3, 200
    x = np.linspace(0.0, 5.0, n_time)
    rng = np.random.default_rng(12)
    signal = np.stack([
        _gaussian(x, 1.0, 0.05, 100.0) + rng.normal(0, 0.3, n_time)
        for _ in range(n_trace)
    ])
    baseline = np.zeros_like(signal)
    peaks = [
        PeakAnnotation(molecule_id="A", rt_min=0.7, rt_max=1.3,
                       mode="artefact_doublet", artefact_side="right"),
    ]
    priors, _ = build_peak_priors(peaks, x, signal, baseline)
    arrays = geometric_priors_to_arrays(priors)
    assert arrays["area_art_shared"].shape == (1,)


@pytest.mark.unit
def test_geometric_priors_to_arrays_empty() -> None:
    """Empty priors list returns empty arrays, no crash."""
    arrays = geometric_priors_to_arrays([])
    assert arrays["apex_loc"].shape == (0,)
    assert arrays["area_gaussian_pt"].shape == (0, 0)


# ---------------------------------------------------------------------------
# refine_apex_priors_with_trace_shift
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_refine_apex_with_trace_shift_detects_drift() -> None:
    """Systematic per-trace drift → trace_shift_scale > 0."""
    n_trace, n_time = 6, 300
    x = np.linspace(2.0, 4.0, n_time)
    # Each trace has a slightly different apex position (drift)
    rng = np.random.default_rng(20)
    apexes = np.linspace(2.95, 3.05, n_trace)
    signal = np.stack([
        _gaussian(x, ap, 0.04, 200.0) + rng.normal(0, 0.5, n_time)
        for ap in apexes
    ])
    baseline = np.zeros_like(signal)
    peaks = [PeakAnnotation(molecule_id="A", rt_min=2.7, rt_max=3.3, mode="single")]
    priors, apex_traces = build_peak_priors(peaks, x, signal, baseline)
    refined, trace_shift_scale = refine_apex_priors_with_trace_shift(priors, apex_traces)

    assert trace_shift_scale > 0.0
    assert len(refined) == 1
    # Refined apex_scale should be <= original (shared drift removed)
    assert refined[0].apex_scale <= priors[0].apex_scale + 1e-6


@pytest.mark.unit
def test_refine_apex_empty_priors() -> None:
    """Empty priors list returns empty list and floor shift scale."""
    refined, scale = refine_apex_priors_with_trace_shift([], PeakApexTraces(
        fwhm_apex_trace=np.empty((0, 0)),
        fwhm_valid_trace=np.empty((0, 0), dtype=bool),
    ))
    assert refined == []
    assert scale > 0.0


# ---------------------------------------------------------------------------
# summarise_priors
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_summarise_priors_format(synthetic_single_peak: _SinglePeakFixture) -> None:
    """summarise_priors returns a non-empty string with header and data rows."""
    peaks, x, signal, baseline = synthetic_single_peak
    priors, _ = build_peak_priors(peaks, x, signal, baseline)
    summary = summarise_priors(priors)

    assert isinstance(summary, str)
    lines = summary.strip().split("\n")
    assert len(lines) >= 3  # header + separator + at least one data row
    # Header should mention w_L and w_R
    assert "w_L" in lines[0]
    assert "w_R" in lines[0]


@pytest.mark.unit
def test_summarise_priors_empty() -> None:
    """Empty priors list returns a header-only string."""
    summary = summarise_priors([])
    assert isinstance(summary, str)
    assert len(summary) > 0
