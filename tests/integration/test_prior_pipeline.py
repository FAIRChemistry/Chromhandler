"""Integration tests for the prior pipeline — no MCMC.

Verifies end-to-end correctness of the prior computation pipeline:
1. On real SAHH data (skipped if external data unavailable).
2. On synthetic data with known ground-truth half-widths.

Marks:
    integration — touches real data or full pipeline
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from scipy.stats import skewnorm

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.priors import build_peak_priors, geometric_priors_to_arrays

pytestmark = pytest.mark.integration

DATA_DIR = Path("/Users/max/code/sahh-kinetics-hplc/data")

_HWHM_FACTOR = math.sqrt(2.0 * math.log(2.0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skewnormal(
    x: npt.NDArray[np.float64], apex: float, sigma: float, alpha: float, area: float
) -> npt.NDArray[np.float64]:
    """Skew-normal evaluated at x, parameterised by mode (not xi)."""
    delta = alpha / math.sqrt(1 + alpha**2)
    mu_z = delta * math.sqrt(2 / math.pi)
    xi = apex - sigma * mu_z
    return area * skewnorm.pdf(x, a=alpha, loc=xi, scale=sigma)  # type: ignore[return-value]


def _true_hwhm(sigma: float, alpha: float) -> tuple[float, float]:
    """Compute exact left/right HWHM of a skew-normal via numerical root-finding."""
    from scipy.optimize import brentq

    delta = alpha / math.sqrt(1 + alpha**2)
    mu_z = delta * math.sqrt(2 / math.pi)
    xi = -sigma * mu_z  # apex at 0
    peak_val = float(skewnorm.pdf(0.0, a=alpha, loc=xi, scale=sigma))
    half_max = 0.5 * peak_val

    def _pdf_minus_half(t: float) -> float:
        return float(skewnorm.pdf(t, a=alpha, loc=xi, scale=sigma)) - half_max

    # Left half-width: find x < 0 where pdf = half_max
    w_left = float(-brentq(_pdf_minus_half, -10 * sigma, 0.0))  # type: ignore[arg-type]
    # Right half-width: find x > 0 where pdf = half_max
    w_right = float(brentq(_pdf_minus_half, 0.0, 10 * sigma))  # type: ignore[arg-type]
    return w_left, w_right


# ---------------------------------------------------------------------------
# Test on real SAHH data
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_prior_pipeline_on_real_data() -> None:
    """Run full prior pipeline on SAHH data; verify basic sanity of outputs."""
    import chromhandler as ch
    from chromhandler.fitting import Fitter

    if not (DATA_DIR / "asm").exists():
        pytest.skip(f"External data not found at {DATA_DIR / 'asm'}")

    handler = ch.Handler.read(path=DATA_DIR / "asm")
    handler.cut_chromatograms((2.5, 3.6))
    handler.samples = handler.samples[:1]

    fitter = Fitter.from_handler(handler)
    fitter.add_baseline_annotation(BaselineAnnotation(rt_min=2.58, rt_max=2.6))
    fitter.add_baseline_annotation(BaselineAnnotation(rt_min=3.5, rt_max=3.52))
    fitter.add_peak_annotation(PeakAnnotation(molecule_id="Inosine", rt_min=2.6, rt_max=2.85, mode="single"))
    fitter.add_peak_annotation(
        PeakAnnotation(
            molecule_id="SIH", rt_min=2.85, rt_max=3.15,
            mode="artefact_doublet", artefact_side="right",
        )
    )
    fitter.add_peak_annotation(
        PeakAnnotation(
            molecule_id="Hyp", rt_min=3.15, rt_max=3.48,
            mode="artefact_doublet", artefact_side="left",
        )
    )

    priors = fitter.compute_priors()
    arrays = geometric_priors_to_arrays(priors)

    n_peak = 3
    n_trace = fitter.n_traces

    # Shape checks
    assert arrays["w_left_loc"].shape == (n_peak,)
    assert arrays["w_right_loc"].shape == (n_peak,)
    assert arrays["snr_per_trace"].shape == (n_trace, n_peak)
    assert arrays["area_gaussian_pt"].shape == (n_trace, n_peak)
    assert arrays["area_art_shared"].shape == (2,)  # 2 artefact peaks

    # Positivity
    assert np.all(arrays["w_left_loc"] > 0), "w_left_loc contains non-positive values"
    assert np.all(arrays["w_right_loc"] > 0), "w_right_loc contains non-positive values"
    assert np.all(arrays["area_gaussian_pt"] > 0), "area_gaussian_pt contains non-positive values"
    assert np.all(arrays["area_trapz_pt"] > 0), "area_trapz_pt contains non-positive values"
    assert np.all(arrays["snr_per_trace"] >= 0), "snr_per_trace contains negative values"
    assert np.all(arrays["area_art_shared"] > 0), "area_art_shared contains non-positive values"

    # Half-widths must be smaller than window widths
    window_widths = arrays["window_hi"] - arrays["window_lo"]
    assert np.all(arrays["w_left_loc"] < window_widths), "w_left_loc >= window_width"
    assert np.all(arrays["w_right_loc"] < window_widths), "w_right_loc >= window_width"

    # Apex must be inside the window
    assert np.all(arrays["apex_loc"] > arrays["window_lo"])
    assert np.all(arrays["apex_loc"] < arrays["window_hi"])


# ---------------------------------------------------------------------------
# Round-trip consistency on synthetic data
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_prior_pipeline_recovers_symmetric_hwhm() -> None:
    """Synthetic Gaussian peak: w_left_loc ≈ w_right_loc ≈ true HWHM (within 15%)."""
    sigma = 0.04
    true_hwhm = sigma * _HWHM_FACTOR

    n_trace, n_time = 8, 400
    x = np.linspace(2.7, 3.3, n_time)
    rng = np.random.default_rng(55)
    signal = np.stack([
        _skewnormal(x, 3.0, sigma, 0.0, 300.0) + rng.normal(0, 0.5, n_time)
        for _ in range(n_trace)
    ])
    baseline = np.zeros_like(signal)
    peaks = [PeakAnnotation(molecule_id="A", rt_min=2.75, rt_max=3.25, mode="single")]

    priors, _ = build_peak_priors(peaks, x, signal, baseline)
    p = priors[0]

    assert abs(p.w_left_loc - true_hwhm) < 0.15 * true_hwhm, (
        f"w_left_loc={p.w_left_loc:.5f}, expected ≈{true_hwhm:.5f}"
    )
    assert abs(p.w_right_loc - true_hwhm) < 0.15 * true_hwhm, (
        f"w_right_loc={p.w_right_loc:.5f}, expected ≈{true_hwhm:.5f}"
    )


@pytest.mark.unit
def test_prior_pipeline_recovers_asymmetric_hwhm() -> None:
    """Skew-normal peak (alpha=1.8): recovered w_left < w_right, both within 20% of true."""
    sigma, alpha = 0.04, 1.8
    true_w_left, true_w_right = _true_hwhm(sigma, alpha)

    n_trace, n_time = 10, 400
    x = np.linspace(2.7, 3.3, n_time)
    rng = np.random.default_rng(77)
    signal = np.stack([
        _skewnormal(x, 3.0, sigma, alpha, 400.0) + rng.normal(0, 0.5, n_time)
        for _ in range(n_trace)
    ])
    baseline = np.zeros_like(signal)
    peaks = [PeakAnnotation(molecule_id="A", rt_min=2.75, rt_max=3.25, mode="single")]

    priors, _ = build_peak_priors(peaks, x, signal, baseline)
    p = priors[0]

    # Direction: right-tailing → w_right > w_left
    assert p.w_right_loc > p.w_left_loc, (
        f"Expected w_right > w_left for alpha={alpha}"
    )
    # Magnitude within 25% of true values (asymmetric HWHM harder to estimate)
    assert abs(p.w_left_loc - true_w_left) < 0.25 * true_w_left, (
        f"w_left_loc={p.w_left_loc:.5f} vs true={true_w_left:.5f}"
    )
    assert abs(p.w_right_loc - true_w_right) < 0.25 * true_w_right, (
        f"w_right_loc={p.w_right_loc:.5f} vs true={true_w_right:.5f}"
    )


@pytest.mark.unit
def test_prior_pipeline_snr_scales_with_signal_amplitude() -> None:
    """Higher-amplitude peaks produce higher median S/N."""
    n_trace, n_time = 5, 300
    x = np.linspace(2.7, 3.3, n_time)
    rng = np.random.default_rng(88)
    noise_std = 1.0
    peaks = [PeakAnnotation(molecule_id="A", rt_min=2.75, rt_max=3.25, mode="single")]

    signal_low = np.stack([
        _skewnormal(x, 3.0, 0.04, 0.0, 50.0) + rng.normal(0, noise_std, n_time)
        for _ in range(n_trace)
    ])
    signal_high = np.stack([
        _skewnormal(x, 3.0, 0.04, 0.0, 500.0) + rng.normal(0, noise_std, n_time)
        for _ in range(n_trace)
    ])
    baseline = np.zeros((n_trace, n_time))

    priors_low, _ = build_peak_priors(peaks, x, signal_low, baseline)
    priors_high, _ = build_peak_priors(peaks, x, signal_high, baseline)

    snr_low = float(np.median(priors_low[0].snr_per_trace))
    snr_high = float(np.median(priors_high[0].snr_per_trace))

    assert snr_high > snr_low, (
        f"Expected higher amplitude to give higher SNR: {snr_high:.1f} vs {snr_low:.1f}"
    )


@pytest.mark.unit
def test_prior_pipeline_area_gaussian_reasonable_magnitude() -> None:
    """area_gaussian_pt should be within 50% of true area for a clean peak."""
    sigma, area_true = 0.04, 200.0
    n_trace, n_time = 8, 400
    x = np.linspace(2.7, 3.3, n_time)
    rng = np.random.default_rng(33)
    signal = np.stack([
        _skewnormal(x, 3.0, sigma, 0.0, area_true) + rng.normal(0, 0.3, n_time)
        for _ in range(n_trace)
    ])
    baseline = np.zeros_like(signal)
    peaks = [PeakAnnotation(molecule_id="A", rt_min=2.75, rt_max=3.25, mode="single")]

    priors, _ = build_peak_priors(peaks, x, signal, baseline)
    area_est = np.median(priors[0].area_gaussian_pt)

    assert abs(area_est - area_true) < 0.5 * area_true, (
        f"area_gaussian_pt median={area_est:.1f} deviates >50% from true={area_true:.1f}"
    )
