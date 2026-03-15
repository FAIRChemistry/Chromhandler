"""Tests for better_visualize module."""
import pytest
import matplotlib.pyplot as plt
import numpy as np

from chromhandler.fitting.better_visualize import _validate_hex_colors, plot_fit


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
