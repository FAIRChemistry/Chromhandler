# Design: Custom Hex Colors for Total Fitted Signal in `plot_fit`

**Date:** 2026-03-15
**Status:** Approved
**Scope:** Visualization feature for `plot_fit` and `BetterFitter.plot_fit()`

## Overview

Add a `colors` parameter to inject custom hex color codes for the **total fitted signal line + HDI band** on a per-peak basis. The combined column (when `n_peak > 1`) always remains blue.

## Requirements

1. Accept `colors: Optional[list[str]]` parameter (list of hex color codes)
2. Validate: `len(colors) == n_peak`, else raise `ValueError`
3. Validate: Each color string must be a valid hex code (`#RRGGBB` format), else raise `ValueError`
4. Apply colors only to total fitted signal (line 1447–1458 in `better_visualize.py`)
5. Keep combined column at default blue (`"C0"`)
6. Backward compatible: `colors=None` → use blue for all peaks

## Parameters

### `better_visualize.plot_fit()`

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
    colors: Optional[list[str]] = None,  # NEW
) -> tuple[plt.Figure, np.ndarray]:
```

### `BetterFitter.plot_fit()`

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
    colors: Optional[list[str]] = None,  # NEW
) -> tuple[object, np.ndarray]:
```

## Validation Logic

```python
if colors is not None:
    if len(colors) != n_peak:
        raise ValueError(
            f"colors must have length n_peak={n_peak}, got {len(colors)}. "
            "Provide one hex color code per peak."
        )
    # Validate each color is a valid hex code
    for i, color in enumerate(colors):
        if not isinstance(color, str) or not color.startswith("#"):
            raise ValueError(
                f"colors[{i}]='{color}' is not a valid hex code. "
                "Use format '#RRGGBB' (e.g., '#FF0000' for red)."
            )
```

## Implementation Points

1. Add validation check in `plot_fit()` function (after `n_peak` is determined)
2. Extract color for each peak in the per-peak-window loop:
   ```python
   color_for_peak = colors[p] if colors is not None else "C0"
   ```
3. Pass `color=color_for_peak` to `_plot_hdi_line()` call for total signal (lines 1447–1458)
4. Wire `colors` parameter through `BetterFitter.plot_fit()` → `_bv_plot_fit()`
5. Update docstrings in both functions

## Example Usage

```python
# Using BetterFitter
fig, axes = fitter.plot_fit(
    colors=["#FF5733", "#33FF57", "#3357FF"]  # One hex per peak
)

# Using standalone function
curves = fitter.posterior_curves(...)
fig, axes = plot_fit(
    time, signal, peaks, curves,
    colors=["#FF5733", "#33FF57", "#3357FF"]
)
```

## Backward Compatibility

- `colors=None` (default) → use `"C0"` (blue) for all peaks
- Existing code without `colors` parameter works unchanged

## Testing

- Test validation: wrong length → `ValueError`
- Test validation: invalid hex → `ValueError`
- Test valid hex codes render correctly
- Test `colors=None` uses default blue
- Test combined column always blue (when `n_peak > 1`)
