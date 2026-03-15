"""Tests for better_visualize module."""
import pytest
import matplotlib.pyplot as plt
import numpy as np

from chromhandler.fitting.better_visualize import _validate_hex_colors, plot_fit
from chromhandler.fitting.better_fitter import PosteriorCurves


def test_validate_hex_colors_valid():
    """Test that valid hex codes pass validation."""
    colors = ["#FF5733", "#33FF57", "#3357FF"]
    _validate_hex_colors(colors, n_peak=3)  # Should not raise


def test_validate_hex_colors_wrong_length():
    """Test that wrong number of colors raises ValueError."""
    colors = ["#FF5733", "#33FF57"]
    with pytest.raises(ValueError, match="must have length n_peak=3"):
        _validate_hex_colors(colors, n_peak=3)


def test_validate_hex_colors_invalid_format():
    """Test that invalid hex format raises ValueError."""
    colors = ["FF5733", "#33FF57", "#3357FF"]  # Missing # on first
    with pytest.raises(ValueError, match="not a valid hex code"):
        _validate_hex_colors(colors, n_peak=3)


def test_validate_hex_colors_invalid_length():
    """Test that too-short hex code raises ValueError."""
    colors = ["#FF5", "#33FF57", "#3357FF"]  # Too short
    with pytest.raises(ValueError, match="not a valid hex code"):
        _validate_hex_colors(colors, n_peak=3)


# Fixtures for plot_fit tests
@pytest.fixture
def sample_posterior_curves():
    """Create minimal sample data for plot_fit tests."""
    from chromhandler.annotations import PeakAnnotation

    # Create simple data
    n_trace, n_time, n_peak = 2, 50, 2
    time_1d = np.linspace(0, 10, n_time)
    time = np.tile(time_1d, (n_trace, 1))  # Make 2D: [n_trace, n_time]
    signal = np.random.randn(n_trace, n_time) + 5

    peaks = [
        PeakAnnotation(molecule_id="peak1", rt_min=1.0, rt_max=4.0, mode="single"),
        PeakAnnotation(molecule_id="peak2", rt_min=5.0, rt_max=8.0, mode="single"),
    ]

    # Create dummy posterior curves
    x = np.linspace(0, 10, n_time)
    total_median = np.random.randn(n_trace, n_time) + 5
    total_lower = total_median - 0.5
    total_upper = total_median + 0.5
    baseline_median = np.ones((n_trace, n_time)) * 2
    baseline_lower = baseline_median - 0.1
    baseline_upper = baseline_median + 0.1
    comp_l_median = np.random.randn(n_trace, n_peak, n_time) + 2
    comp_l_lower = comp_l_median - 0.3
    comp_l_upper = comp_l_median + 0.3
    comp_r_median = np.random.randn(n_trace, n_peak, n_time) + 2
    comp_r_lower = comp_r_median - 0.3
    comp_r_upper = comp_r_median + 0.3

    curves = PosteriorCurves(
        x=x,
        total_median=total_median,
        total_lower=total_lower,
        total_upper=total_upper,
        baseline_median=baseline_median,
        baseline_lower=baseline_lower,
        baseline_upper=baseline_upper,
        comp_l_median=comp_l_median,
        comp_l_lower=comp_l_lower,
        comp_l_upper=comp_l_upper,
        comp_r_median=comp_r_median,
        comp_r_lower=comp_r_lower,
        comp_r_upper=comp_r_upper,
        trace_indices=np.arange(n_trace),
        chromatogram_ids=None,
    )

    return time, signal, peaks, curves


def test_plot_fit_with_custom_colors(sample_posterior_curves):
    """Test that plot_fit accepts colors parameter."""
    time, signal, peaks, curves = sample_posterior_curves

    # Should not raise with valid colors
    fig, axes = plot_fit(
        time, signal, peaks, curves,
        colors=["#FF5733", "#33FF57"]
    )
    assert fig is not None
    plt.close(fig)


def test_plot_fit_colors_wrong_length(sample_posterior_curves):
    """Test that plot_fit raises ValueError for mismatched colors length."""
    time, signal, peaks, curves = sample_posterior_curves
    n_peak = len(peaks)

    with pytest.raises(ValueError, match="colors must have length"):
        plot_fit(
            time, signal, peaks, curves,
            colors=["#FF5733"]  # Wrong length
        )
