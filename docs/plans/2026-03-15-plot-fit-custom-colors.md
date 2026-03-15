# Custom Hex Colors for Total Fitted Signal in `plot_fit` Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `colors` parameter to `plot_fit()` and `BetterFitter.plot_fit()` to colorize total fitted signal lines per peak using hex codes.

**Architecture:**
Add hex color validation as a helper function, then inject the `colors` parameter into both the standalone `plot_fit()` function and the `BetterFitter.plot_fit()` method. Validation occurs early in `plot_fit()` before any plotting. The combined column always uses default blue.

**Tech Stack:** NumPy, Matplotlib, NumPy arrays for curve data, hex color validation with regex.

---

## Task 1: Add Hex Color Validation Helper

**Files:**
- Modify: `chromhandler/fitting/better_visualize.py` (add helper at top of file after imports)

**Step 1: Write the failing test**

```python
# tests/fitting/test_better_visualize.py
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/fitting/test_better_visualize.py::test_validate_hex_colors_valid -v
```

Expected: FAIL with "function not defined"

**Step 3: Write the validation helper**

Add to `chromhandler/fitting/better_visualize.py` after imports:

```python
def _validate_hex_colors(colors: list[str], n_peak: int) -> None:
    """Validate that colors is a list of valid hex codes with correct length.

    Parameters
    ----------
    colors : list[str]
        List of hex color codes.
    n_peak : int
        Expected number of peaks/colors.

    Raises
    ------
    ValueError
        If length doesn't match n_peak or any color is not a valid hex code.
    """
    if len(colors) != n_peak:
        raise ValueError(
            f"colors must have length n_peak={n_peak}, got {len(colors)}. "
            "Provide one hex color code per peak."
        )

    for i, color in enumerate(colors):
        if not isinstance(color, str):
            raise ValueError(
                f"colors[{i}] is not a string, got {type(color).__name__}."
            )
        if not color.startswith("#"):
            raise ValueError(
                f"colors[{i}]='{color}' is not a valid hex code. "
                "Use format '#RRGGBB' (e.g., '#FF0000' for red)."
            )
        if len(color) not in (7, 9):  # #RRGGBB or #RRGGBBAA
            raise ValueError(
                f"colors[{i}]='{color}' is not a valid hex code. "
                "Use format '#RRGGBB' (e.g., '#FF0000' for red) or "
                "'#RRGGBBAA' with alpha."
            )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/fitting/test_better_visualize.py::test_validate_hex_colors_valid \
        tests/fitting/test_better_visualize.py::test_validate_hex_colors_wrong_length \
        tests/fitting/test_better_visualize.py::test_validate_hex_colors_invalid_format \
        tests/fitting/test_better_visualize.py::test_validate_hex_colors_invalid_length -v
```

Expected: All PASS

**Step 5: Commit**

```bash
git add chromhandler/fitting/better_visualize.py tests/fitting/test_better_visualize.py
git commit -m "feat: add hex color validation helper

- Add _validate_hex_colors() to validate hex code format and count
- Support both #RRGGBB and #RRGGBBAA formats
- Includes comprehensive tests for valid/invalid cases

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Add `colors` Parameter to `better_visualize.plot_fit()`

**Files:**
- Modify: `chromhandler/fitting/better_visualize.py:1256-1310` (function signature and docstring)

**Step 1: Update function signature**

In `chromhandler/fitting/better_visualize.py`, update the `plot_fit()` function signature:

```python
def plot_fit(
    time: np.ndarray,
    signal: np.ndarray,
    peaks: list[PeakAnnotation],
    curves: "PosteriorCurves | None",
    *,
    fitted_rows: Optional[np.ndarray] = None,
    baselines: Optional[list[BaselineAnnotation]] = None,
    chromatogram_ids: Optional[list[str]] = None,
    hdi_prob: float = 0.95,
    figsize: Optional[tuple[float, float]] = None,
    colors: Optional[list[str]] = None,
) -> tuple[plt.Figure, np.ndarray]:
```

**Step 2: Update docstring**

In the docstring, add to the Parameters section:

```python
    colors : list[str] or None
        List of hex color codes (e.g., ['#FF5733', '#33FF57']) for the total
        fitted signal line + HDI band, one per peak.  Length must match
        ``n_peak``.  When ``None`` (default), uses blue ('C0') for all peaks.
        The combined column (when n_peak > 1) always uses blue.
```

**Step 3: Add validation after `n_peak` is determined**

In the function body, after line `n_peak = len(peaks)`, add:

```python
    if colors is not None:
        _validate_hex_colors(colors, n_peak)
```

**Step 4: Write a test for the parameter acceptance**

```python
# In tests/fitting/test_better_visualize.py
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
```

**Step 5: Run tests**

```bash
pytest tests/fitting/test_better_visualize.py::test_plot_fit_with_custom_colors \
        tests/fitting/test_better_visualize.py::test_plot_fit_colors_wrong_length -v
```

Expected: Both PASS

**Step 6: Commit**

```bash
git add chromhandler/fitting/better_visualize.py tests/fitting/test_better_visualize.py
git commit -m "feat: add colors parameter to plot_fit() signature and validation

- Add colors: Optional[list[str]] parameter
- Add validation using _validate_hex_colors()
- Update docstring
- Add tests for parameter acceptance and validation

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Apply Custom Colors to Total Fitted Signal

**Files:**
- Modify: `chromhandler/fitting/better_visualize.py:1446-1458` (total signal plotting)

**Step 1: Extract color in the per-peak loop**

In the per-peak-window loop (around line 1372), after `for p, peak in enumerate(peaks):`, add:

```python
            # Determine color for this peak's total fitted signal
            color_for_peak = colors[p] if colors is not None else "C0"
```

**Step 2: Update the total signal plotting call**

Replace the existing call to `_plot_hdi_line()` for the total signal (lines 1447–1458):

From:
```python
                    _plot_hdi_line(
                        ax,
                        x_c,
                        curves.total_median[ci, win],
                        curves.total_lower[ci, win],
                        curves.total_upper[ci, win],
                        color="C0",
                        alpha=0.3,
                        linewidth=1.5,
                        linestyle="-",
                        label=f"Fitted signal ({hdi_label})" if first else "",
                    )
```

To:
```python
                    _plot_hdi_line(
                        ax,
                        x_c,
                        curves.total_median[ci, win],
                        curves.total_lower[ci, win],
                        curves.total_upper[ci, win],
                        color=color_for_peak,
                        alpha=0.3,
                        linewidth=1.5,
                        linestyle="-",
                        label=f"Fitted signal ({hdi_label})" if first else "",
                    )
```

**Step 3: Write a visual integration test**

```python
# In tests/fitting/test_better_visualize.py
def test_plot_fit_custom_colors_rendering(sample_posterior_curves):
    """Test that custom colors are actually rendered (sanity check)."""
    time, signal, peaks, curves = sample_posterior_curves
    n_peak = len(peaks)
    colors = ["#FF0000", "#00FF00", "#0000FF"][:n_peak]

    fig, axes = plot_fit(
        time, signal, peaks, curves,
        colors=colors
    )

    # Check that figure was created
    assert fig is not None
    assert axes.shape[1] >= n_peak

    plt.close(fig)
```

**Step 4: Run the test**

```bash
pytest tests/fitting/test_better_visualize.py::test_plot_fit_custom_colors_rendering -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add chromhandler/fitting/better_visualize.py tests/fitting/test_better_visualize.py
git commit -m "feat: apply custom colors to total fitted signal in plot_fit

- Extract color per peak in loop
- Replace hardcoded 'C0' with color_for_peak
- Add visual integration test
- Combined column always uses blue

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Wire `colors` Through `BetterFitter.plot_fit()`

**Files:**
- Modify: `chromhandler/fitting/better_fitter.py:1538-1640` (method signature, docstring, call)

**Step 1: Update method signature**

In `chromhandler/fitting/better_fitter.py`, update `BetterFitter.plot_fit()`:

```python
    def plot_fit(
        self,
        *,
        subset: str | None = None,
        sample_ids: list[str] | None = None,
        chromatogram_ids: list[str] | None = None,
        hdi_prob: float = 0.95,
        n_samples_max: int = 2000,
        figsize: tuple[float, float] | None = None,
        colors: list[str] | None = None,
    ) -> tuple[object, np.ndarray]:
```

**Step 2: Update docstring**

Add to the Args section:

```python
            colors: List of hex color codes (e.g., ['#FF5733', '#33FF57'])
                for the total fitted signal per peak.  Length must match the
                number of peaks in the subset.  When ``None`` (default), uses
                blue ('C0') for all peaks.
```

**Step 3: Wire colors through to `_bv_plot_fit()`**

Find the call to `_bv_plot_fit()` (around line 1630) and add `colors=colors`:

```python
        fig, axes = _bv_plot_fit(
            time_display,
            signal_display,
            peaks,
            curves,
            fitted_rows=fitted_display_idx - display_idx[0]
            if len(display_idx) > 0
            else np.array([], dtype=int),
            baselines=eff_baselines,
            chromatogram_ids=chromatogram_ids,
            hdi_prob=hdi_prob,
            figsize=figsize,
            colors=colors,
        )
```

**Step 4: Write an integration test**

```python
# In tests/fitting/test_better_fitter.py
def test_better_fitter_plot_fit_with_colors(fitter_with_posterior):
    """Test that BetterFitter.plot_fit accepts colors parameter."""
    fitter = fitter_with_posterior
    n_peak = len(fitter.peaks)
    colors = ["#FF0000", "#00FF00", "#0000FF"][:n_peak]

    fig, axes = fitter.plot_fit(colors=colors)

    assert fig is not None
    assert axes is not None
    plt.close(fig)

def test_better_fitter_plot_fit_colors_validation(fitter_with_posterior):
    """Test that BetterFitter.plot_fit validates colors length."""
    fitter = fitter_with_posterior

    with pytest.raises(ValueError, match="colors must have length"):
        fitter.plot_fit(colors=["#FF0000"])  # Wrong length
```

**Step 5: Run tests**

```bash
pytest tests/fitting/test_better_fitter.py::test_better_fitter_plot_fit_with_colors \
        tests/fitting/test_better_fitter.py::test_better_fitter_plot_fit_colors_validation -v
```

Expected: Both PASS

**Step 6: Commit**

```bash
git add chromhandler/fitting/better_fitter.py tests/fitting/test_better_fitter.py
git commit -m "feat: wire colors parameter through BetterFitter.plot_fit()

- Add colors: list[str] | None parameter
- Wire through to _bv_plot_fit()
- Update docstring
- Add integration tests

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Full Integration Test

**Files:**
- Create: `tests/fitting/test_plot_fit_custom_colors.py` (new test file)

**Step 1: Write comprehensive integration test**

```python
# tests/fitting/test_plot_fit_custom_colors.py
"""Integration tests for custom colors in plot_fit."""
import numpy as np
import pytest
import matplotlib.pyplot as plt

from chromhandler.fitting.better_visualize import plot_fit
from chromhandler.fitting.better_fitter import BetterFitter
from chromhandler.model import PeakAnnotation, BaselineAnnotation


@pytest.fixture
def three_peak_posterior(sample_posterior_curves):
    """Create a posterior with 3 peaks for color testing."""
    time, signal, peaks, curves = sample_posterior_curves
    # Ensure we have at least 3 peaks for testing
    if len(peaks) < 3:
        pytest.skip("Need at least 3 peaks for this test")
    return time, signal, peaks[:3], curves


def test_plot_fit_three_colors_valid(three_peak_posterior):
    """Test plot_fit with 3 custom colors."""
    time, signal, peaks, curves = three_peak_posterior
    colors = ["#FF0000", "#00FF00", "#0000FF"]

    fig, axes = plot_fit(
        time, signal, peaks, curves,
        colors=colors
    )

    assert fig is not None
    assert axes.shape[1] >= 3  # 3 peak columns + possibly combined
    plt.close(fig)


def test_plot_fit_color_none_backward_compat(three_peak_posterior):
    """Test that colors=None uses default blue (backward compatibility)."""
    time, signal, peaks, curves = three_peak_posterior

    # Should work without colors parameter
    fig, axes = plot_fit(time, signal, peaks, curves)

    assert fig is not None
    plt.close(fig)


def test_plot_fit_color_length_mismatch(three_peak_posterior):
    """Test that mismatched color count raises ValueError."""
    time, signal, peaks, curves = three_peak_posterior
    colors = ["#FF0000", "#00FF00"]  # Only 2, need 3

    with pytest.raises(ValueError, match="colors must have length n_peak=3"):
        plot_fit(time, signal, peaks, curves, colors=colors)


def test_plot_fit_invalid_hex_format(three_peak_posterior):
    """Test that invalid hex codes raise ValueError."""
    time, signal, peaks, curves = three_peak_posterior
    colors = ["FF0000", "#00FF00", "#0000FF"]  # Missing # on first

    with pytest.raises(ValueError, match="not a valid hex code"):
        plot_fit(time, signal, peaks, curves, colors=colors)


def test_better_fitter_plot_fit_colors(fitter_with_posterior):
    """Test BetterFitter.plot_fit with custom colors."""
    fitter = fitter_with_posterior
    n_peak = len(fitter.peaks)

    if n_peak < 2:
        pytest.skip("Need at least 2 peaks for this test")

    colors = ["#FF5733", "#33FF57"][:n_peak]

    fig, axes = fitter.plot_fit(colors=colors)

    assert fig is not None
    assert axes is not None
    plt.close(fig)
```

**Step 2: Run the full test suite**

```bash
pytest tests/fitting/test_plot_fit_custom_colors.py -v
```

Expected: All PASS

**Step 3: Run all better_visualize and better_fitter tests**

```bash
pytest tests/fitting/test_better_visualize.py tests/fitting/test_better_fitter.py -v
```

Expected: All PASS (including new tests)

**Step 4: Commit**

```bash
git add tests/fitting/test_plot_fit_custom_colors.py
git commit -m "test: add comprehensive integration tests for plot_fit colors

- Test valid 3-color rendering
- Test backward compatibility (colors=None)
- Test length validation
- Test hex format validation
- Test BetterFitter integration

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Documentation & Example

**Files:**
- Modify: `chromhandler/fitting/better_visualize.py` (docstring)
- Modify: `chromhandler/fitting/better_fitter.py` (docstring)
- Create: `docs/examples/plot_fit_custom_colors.py` (optional example)

**Step 1: Ensure docstrings are complete**

Verify that both function docstrings clearly describe the `colors` parameter with examples. Example for docstring:

```python
    Examples
    --------
    Plot with custom colors per peak:

    >>> colors = ["#FF5733", "#33FF57", "#3357FF"]
    >>> fig, axes = fitter.plot_fit(colors=colors)

    Plot with default blue colors:

    >>> fig, axes = fitter.plot_fit()
```

**Step 2: Create optional example file**

```python
# docs/examples/plot_fit_custom_colors.py
"""Example: Using custom colors in plot_fit visualization.

This example demonstrates how to customize the color of the total fitted
signal line + HDI band in the plot_fit visualization.
"""

from chromhandler.fitting.better_fitter import BetterFitter
import numpy as np

# Assume you have a BetterFitter instance with posterior samples
fitter = BetterFitter(...)

# Define custom hex colors (one per peak)
n_peak = len(fitter.peaks)
colors = ["#FF5733", "#33FF57", "#3357FF"][:n_peak]

# Plot with custom colors
fig, axes = fitter.plot_fit(colors=colors)

# Or use the standalone function
from chromhandler.fitting.better_visualize import plot_fit
curves = fitter.posterior_curves()
fig, axes = plot_fit(
    fitter.time, fitter.signal, fitter.peaks, curves,
    colors=colors
)
```

**Step 3: Run linter on modified files**

```bash
ruff check chromhandler/fitting/better_visualize.py chromhandler/fitting/better_fitter.py
ruff format chromhandler/fitting/better_visualize.py chromhandler/fitting/better_fitter.py
```

Expected: No errors

**Step 4: Commit**

```bash
git add chromhandler/fitting/better_visualize.py chromhandler/fitting/better_fitter.py docs/examples/plot_fit_custom_colors.py
git commit -m "docs: add example and complete docstrings for plot_fit colors

- Add docstring examples showing colors parameter usage
- Create docs/examples/plot_fit_custom_colors.py with usage patterns
- Ensure both functions have clear parameter documentation

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Final Verification

**Files:**
- Run: All tests + linting

**Step 1: Run full test suite**

```bash
pytest tests/fitting/ -v --tb=short
```

Expected: All PASS, no skipped tests related to colors

**Step 2: Run linting and type checking**

```bash
ruff check chromhandler/fitting/
pyright chromhandler/fitting/better_visualize.py chromhandler/fitting/better_fitter.py
```

Expected: No errors

**Step 3: Manual smoke test (optional)**

Create a simple script to visually verify colors are rendered:

```python
# Quick manual test (not committed)
import numpy as np
from chromhandler.fitting.better_fitter import BetterFitter

fitter = BetterFitter(...)  # Load your data
colors = ["#FF0000", "#00FF00", "#0000FF"][:len(fitter.peaks)]
fig, axes = fitter.plot_fit(colors=colors)
fig.savefig("/tmp/plot_fit_colors_test.png")
```

**Step 4: Final commit with summary**

```bash
git log --oneline docs/plans/2026-03-15-plot-fit-custom-colors.md~0..HEAD
git commit --allow-empty -m "feat: custom hex colors for plot_fit - implementation complete

Summary:
- Added _validate_hex_colors() helper for hex code validation
- Added colors: Optional[list[str]] parameter to plot_fit()
- Added colors: list[str] | None parameter to BetterFitter.plot_fit()
- Validation: len(colors) must equal n_peak, else ValueError
- Validation: each color must be valid hex (#RRGGBB or #RRGGBBAA)
- Combined column always uses default blue
- Backward compatible: colors=None uses blue for all peaks
- Comprehensive test coverage (validation, rendering, integration)
- Documentation with examples

All tests pass. Linting clean.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

---

## Validation Checklist

Before claiming completion:

- [ ] All tests pass (`pytest tests/fitting/ -v`)
- [ ] Linting passes (`ruff check chromhandler/fitting/`)
- [ ] Type checking passes (`pyright`)
- [ ] Docstrings updated with examples
- [ ] `colors=None` backward compatible (blue for all peaks)
- [ ] Combined column always blue (when `n_peak > 1`)
- [ ] Invalid hex codes raise `ValueError` with clear message
- [ ] Wrong color count raises `ValueError` with clear message
- [ ] Example file created/documented
- [ ] All commits present with clear messages
