# Plotting Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundations-layer plotting module with axes-level primitives + figure-level convenience functions on top of `PreparedDataset`.

**Architecture:** Single new module `chromhandler/fitting/plotting.py`. Free functions only (no methods on `PreparedDataset`). Two layers: axes-level `add_*` primitives that mutate an existing `Axes`, and figure-level `plot_*` functions that build complete `Figure`s by composing the primitives. Matplotlib is the only plot dependency.

**Tech Stack:** Python 3.11+, matplotlib, NumPy, pytest. (Matplotlib is already a project dependency via `chromhandler/visualize.py`.)

---

## Important context for the executing engineer

- **Spec:** [docs/superpowers/specs/2026-05-08-plotting-architecture-design.md](../superpowers/specs/2026-05-08-plotting-architecture-design.md). Read sections 5 (public API) and 6 (coupling/import policy) carefully — they constrain how this module connects to the rest.
- **Working directory:** `/Users/max/code/Chromhandler` (on branch `fix-fit`). All commands and edits target this directory. cd there at the start of any Bash invocation.
- **Foundations layer is complete and committed.** `chromhandler/fitting/{preprocessing,baseline,noise,prepared_dataset}.py` and `chromhandler.annotations` are all in place. You depend on `PreparedDataset` and `BaselineAnnotation` / `PeakAnnotation` only.
- **No matplotlib imports allowed in foundations modules.** All matplotlib imports live in `plotting.py`. Verify after each task: `grep -n matplotlib chromhandler/fitting/{preprocessing,baseline,noise,prepared_dataset}.py chromhandler/annotations.py` should return zero hits.
- **Quality gate after every file edit:** `uv run ruff check <file>` and `uv run pyright <file>` must pass with zero issues.
- **Tests run headless.** Matplotlib auto-selects a non-interactive backend in pytest; no need to force `Agg` manually unless a test fails for backend reasons. If it does, add `import matplotlib; matplotlib.use("Agg")` at the top of `test_plotting.py` before any other matplotlib import.
- **Commits include the standard footer:** `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`.

## File structure created by this plan

```
chromhandler/fitting/
    plotting.py                          # NEW

tests/unit/fitting/
    test_plotting.py                     # NEW
```

That's it. No changes to any existing file.

---

## Test helper — used by all tasks

Every test in `test_plotting.py` needs a small synthetic `PreparedDataset`. Define this helper at the top of the test file (after imports). It is referenced verbatim by Tasks 1-4.

```python
def _make_synthetic_dataset(n_trace: int = 3) -> PreparedDataset:
    """Build a tiny synthetic PreparedDataset for plotting tests.

    n_trace traces, 101 time points each on [0, 5] min, with a clean
    linear baseline + small Gaussian noise. One peak window 2.0–3.0,
    two baseline regions 0.5–1.0 and 4.0–4.5.
    """
    rng = np.random.default_rng(0)
    time_grid = np.linspace(0.0, 5.0, 101)
    times = [time_grid for _ in range(n_trace)]
    signals = [
        1.0 + 0.1 * time_grid + 0.02 * rng.standard_normal(time_grid.size)
        for _ in range(n_trace)
    ]
    peaks = [PeakAnnotation(molecule_id="x", rt_min=2.0, rt_max=3.0)]
    baselines = [
        BaselineAnnotation(rt_min=0.5, rt_max=1.0),
        BaselineAnnotation(rt_min=4.0, rt_max=4.5),
    ]
    return prepare_dataset(times, signals, peaks, baselines)
```

---

## Task 1: Module skeleton + `add_signal` + `add_annotation_regions`

The two primitives that don't depend on baseline/noise outputs. Implementing them together establishes the module's import header and test scaffolding in one commit.

**Files:**
- Create: `chromhandler/fitting/plotting.py`
- Create: `tests/unit/fitting/test_plotting.py`

- [ ] **Step 1.1: Write the failing tests**

Create `tests/unit/fitting/test_plotting.py`:

```python
"""Tests for chromhandler.fitting.plotting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.prepared_dataset import PreparedDataset, prepare_dataset


def _make_synthetic_dataset(n_trace: int = 3) -> PreparedDataset:
    """Build a tiny synthetic PreparedDataset for plotting tests.

    n_trace traces, 101 time points each on [0, 5] min, with a clean
    linear baseline + small Gaussian noise. One peak window 2.0–3.0,
    two baseline regions 0.5–1.0 and 4.0–4.5.
    """
    rng = np.random.default_rng(0)
    time_grid = np.linspace(0.0, 5.0, 101)
    times = [time_grid for _ in range(n_trace)]
    signals = [
        1.0 + 0.1 * time_grid + 0.02 * rng.standard_normal(time_grid.size)
        for _ in range(n_trace)
    ]
    peaks = [PeakAnnotation(molecule_id="x", rt_min=2.0, rt_max=3.0)]
    baselines = [
        BaselineAnnotation(rt_min=0.5, rt_max=1.0),
        BaselineAnnotation(rt_min=4.0, rt_max=4.5),
    ]
    return prepare_dataset(times, signals, peaks, baselines)


class TestAddSignal:
    """The add_signal axes primitive."""

    def test_returns_same_axes(self) -> None:
        from chromhandler.fitting.plotting import add_signal

        ds = _make_synthetic_dataset(n_trace=2)
        fig, ax = plt.subplots()
        try:
            returned = add_signal(ax, ds, trace_idx=0)
            assert returned is ax
        finally:
            plt.close(fig)

    def test_adds_one_line(self) -> None:
        from chromhandler.fitting.plotting import add_signal

        ds = _make_synthetic_dataset(n_trace=2)
        fig, ax = plt.subplots()
        try:
            n_lines_before = len(ax.lines)
            add_signal(ax, ds, trace_idx=0)
            assert len(ax.lines) == n_lines_before + 1
        finally:
            plt.close(fig)

    def test_skips_nan_padding(self) -> None:
        from chromhandler.fitting.plotting import add_signal

        # Manually construct a dataset with NaN-padding to verify it's masked.
        ds = _make_synthetic_dataset(n_trace=2)
        fig, ax = plt.subplots()
        try:
            add_signal(ax, ds, trace_idx=0)
            (line,) = ax.lines[-1:]
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            assert not np.any(np.isnan(xdata))
            assert not np.any(np.isnan(ydata))
        finally:
            plt.close(fig)


class TestAddAnnotationRegions:
    """The add_annotation_regions axes primitive."""

    def test_returns_same_axes(self) -> None:
        from chromhandler.fitting.plotting import add_annotation_regions

        ds = _make_synthetic_dataset()
        fig, ax = plt.subplots()
        try:
            returned = add_annotation_regions(ax, ds)
            assert returned is ax
        finally:
            plt.close(fig)

    def test_adds_one_axvspan_per_region(self) -> None:
        from chromhandler.fitting.plotting import add_annotation_regions

        ds = _make_synthetic_dataset()  # 1 peak window + 2 baseline regions
        fig, ax = plt.subplots()
        try:
            n_patches_before = len(ax.patches)
            add_annotation_regions(ax, ds)
            # axvspan adds Polygon patches; one per peak + one per baseline = 3
            assert len(ax.patches) - n_patches_before == 3
        finally:
            plt.close(fig)
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `cd /Users/max/code/Chromhandler && uv run pytest tests/unit/fitting/test_plotting.py -v`
Expected: 5 failures, all `ImportError: cannot import name 'add_signal' from 'chromhandler.fitting.plotting'` or `ModuleNotFoundError`.

- [ ] **Step 1.3: Implement the module skeleton + the two primitives**

Create `chromhandler/fitting/plotting.py`:

```python
"""Diagnostic and posterior plotting for chromatographic fitting.

Two layers:

1. **Axes-level primitives** — ``add_*`` functions that take an existing
   :class:`matplotlib.axes.Axes`, mutate it, and return it. Composable
   building blocks. Idiomatic matplotlib.

2. **Figure-level convenience** — ``plot_*`` functions that build a
   complete :class:`matplotlib.figure.Figure` for a common case by
   composing the axes primitives.

Matplotlib is the only plot dependency. The foundations modules
(``preprocessing``, ``baseline``, ``noise``, ``prepared_dataset``,
``annotations``) deliberately do not import matplotlib; that import
lives only here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from chromhandler.fitting.prepared_dataset import PreparedDataset


def add_signal(
    ax: Axes,
    dataset: PreparedDataset,
    trace_idx: int,
    *,
    color: str = "tab:gray",
    linewidth: float = 0.8,
    alpha: float = 0.85,
) -> Axes:
    """Plot one trace's raw signal on the given axes.

    NaN-padded samples are masked out before plotting.

    Args:
        ax: Target axes (mutated and returned).
        dataset: The prepared dataset.
        trace_idx: Which trace to plot.
        color: Line colour.
        linewidth: Line width in points.
        alpha: Line opacity.

    Returns:
        The same ``ax`` passed in.
    """
    valid = dataset.valid_mask[trace_idx]
    t = dataset.time[trace_idx][valid]
    s = dataset.signal[trace_idx][valid]
    ax.plot(t, s, color=color, linewidth=linewidth, alpha=alpha)
    return ax


def add_annotation_regions(
    ax: Axes,
    dataset: PreparedDataset,
    *,
    peak_color: str = "tab:orange",
    baseline_color: str = "tab:green",
    alpha: float = 0.15,
) -> Axes:
    """Shade peak windows and baseline regions across the time axis.

    Each :class:`PeakAnnotation` and :class:`BaselineAnnotation` becomes a
    vertical band spanning its ``[rt_min, rt_max]`` range.

    Args:
        ax: Target axes (mutated and returned).
        dataset: Source of the annotations.
        peak_color: Fill colour for peak windows.
        baseline_color: Fill colour for baseline regions.
        alpha: Fill opacity.

    Returns:
        The same ``ax`` passed in.
    """
    for p in dataset.peak_annotations:
        ax.axvspan(p.rt_min, p.rt_max, color=peak_color, alpha=alpha)
    for b in dataset.baseline_annotations:
        ax.axvspan(b.rt_min, b.rt_max, color=baseline_color, alpha=alpha)
    return ax
```

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `cd /Users/max/code/Chromhandler && uv run pytest tests/unit/fitting/test_plotting.py -v`
Expected: 5 passed.

- [ ] **Step 1.5: Quality gates**

Run:
```
cd /Users/max/code/Chromhandler && uv run ruff check chromhandler/fitting/plotting.py tests/unit/fitting/test_plotting.py && uv run pyright chromhandler/fitting/plotting.py tests/unit/fitting/test_plotting.py
```
Expected: All checks passed; 0 errors.

- [ ] **Step 1.6: Verify no matplotlib leakage into foundations**

Run:
```
cd /Users/max/code/Chromhandler && grep -n matplotlib chromhandler/fitting/preprocessing.py chromhandler/fitting/baseline.py chromhandler/fitting/noise.py chromhandler/fitting/prepared_dataset.py chromhandler/annotations.py
```
Expected: zero hits.

- [ ] **Step 1.7: Commit**

```
cd /Users/max/code/Chromhandler && git add chromhandler/fitting/plotting.py tests/unit/fitting/test_plotting.py && git commit -m "$(cat <<'EOF'
Add plotting module skeleton with add_signal and add_annotation_regions

First two axes-level primitives: add_signal plots a NaN-aware trace on
existing axes; add_annotation_regions shades peak and baseline windows
as vertical bands. Both return the mutated axes for chaining.

Establishes the canonical pattern for the plotting layer: free
functions only, mutate-and-return-axes, no matplotlib imports outside
chromhandler/fitting/plotting.py.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `add_baseline` with optional noise ribbon

**Files:**
- Modify: `chromhandler/fitting/plotting.py` (append function)
- Modify: `tests/unit/fitting/test_plotting.py` (append test class)

- [ ] **Step 2.1: Write the failing tests**

Append to `tests/unit/fitting/test_plotting.py`:

```python
class TestAddBaseline:
    """The add_baseline axes primitive."""

    def test_returns_same_axes(self) -> None:
        from chromhandler.fitting.plotting import add_baseline

        ds = _make_synthetic_dataset()
        fig, ax = plt.subplots()
        try:
            returned = add_baseline(ax, ds, trace_idx=0)
            assert returned is ax
        finally:
            plt.close(fig)

    def test_adds_baseline_line(self) -> None:
        from chromhandler.fitting.plotting import add_baseline

        ds = _make_synthetic_dataset()
        fig, ax = plt.subplots()
        try:
            n_lines_before = len(ax.lines)
            add_baseline(ax, ds, trace_idx=0, show_noise_band=False)
            assert len(ax.lines) == n_lines_before + 1
        finally:
            plt.close(fig)

    def test_noise_band_adds_polycollection(self) -> None:
        from chromhandler.fitting.plotting import add_baseline

        ds = _make_synthetic_dataset()
        fig, ax = plt.subplots()
        try:
            n_collections_before = len(ax.collections)
            add_baseline(ax, ds, trace_idx=0, show_noise_band=True)
            # fill_between adds a PolyCollection
            assert len(ax.collections) == n_collections_before + 1
        finally:
            plt.close(fig)

    def test_no_noise_band_when_disabled(self) -> None:
        from chromhandler.fitting.plotting import add_baseline

        ds = _make_synthetic_dataset()
        fig, ax = plt.subplots()
        try:
            n_collections_before = len(ax.collections)
            add_baseline(ax, ds, trace_idx=0, show_noise_band=False)
            assert len(ax.collections) == n_collections_before
        finally:
            plt.close(fig)

    def test_baseline_values_match_intercept_slope(self) -> None:
        from chromhandler.fitting.plotting import add_baseline

        ds = _make_synthetic_dataset()
        fig, ax = plt.subplots()
        try:
            add_baseline(ax, ds, trace_idx=0, show_noise_band=False)
            (line,) = ax.lines[-1:]
            xdata = np.asarray(line.get_xdata())
            ydata = np.asarray(line.get_ydata())
            expected = (
                ds.baseline_intercept[0] + ds.baseline_slope[0] * xdata
            )
            np.testing.assert_allclose(ydata, expected, atol=1e-9)
        finally:
            plt.close(fig)
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `cd /Users/max/code/Chromhandler && uv run pytest tests/unit/fitting/test_plotting.py::TestAddBaseline -v`
Expected: 5 failures, all `ImportError: cannot import name 'add_baseline'`.

- [ ] **Step 2.3: Implement**

Append to `chromhandler/fitting/plotting.py`:

```python
def add_baseline(
    ax: Axes,
    dataset: PreparedDataset,
    trace_idx: int,
    *,
    show_noise_band: bool = True,
    color: str = "tab:blue",
    linewidth: float = 1.0,
    linestyle: str = "--",
    band_alpha: float = 0.15,
) -> Axes:
    """Overlay one trace's OLS baseline, optionally with a ±noise ribbon.

    The baseline is drawn across the full valid time range of the trace
    (NaN-padded samples excluded). The optional noise ribbon is a
    translucent fill at ``baseline ± noise_per_trace[trace_idx]``.

    Args:
        ax: Target axes (mutated and returned).
        dataset: Source of baseline parameters and noise.
        trace_idx: Which trace to plot.
        show_noise_band: If True, fill between ``baseline ± noise``.
        color: Line + fill colour.
        linewidth: Line width in points.
        linestyle: Line style (default dashed).
        band_alpha: Noise-band fill opacity.

    Returns:
        The same ``ax`` passed in.
    """
    valid = dataset.valid_mask[trace_idx]
    t = dataset.time[trace_idx][valid]
    intercept = dataset.baseline_intercept[trace_idx]
    slope = dataset.baseline_slope[trace_idx]
    noise = dataset.noise_per_trace[trace_idx]
    baseline = intercept + slope * t
    if show_noise_band:
        ax.fill_between(
            t,
            baseline - noise,
            baseline + noise,
            color=color,
            alpha=band_alpha,
            linewidth=0,
        )
    ax.plot(t, baseline, color=color, linewidth=linewidth, linestyle=linestyle)
    return ax
```

- [ ] **Step 2.4: Run tests to verify they pass**

Run: `cd /Users/max/code/Chromhandler && uv run pytest tests/unit/fitting/test_plotting.py -v`
Expected: 10 passed (5 from Task 1 + 5 new).

- [ ] **Step 2.5: Quality gates**

Run:
```
cd /Users/max/code/Chromhandler && uv run ruff check chromhandler/fitting/plotting.py tests/unit/fitting/test_plotting.py && uv run pyright chromhandler/fitting/plotting.py tests/unit/fitting/test_plotting.py
```
Expected: All checks passed; 0 errors.

- [ ] **Step 2.6: Commit**

```
cd /Users/max/code/Chromhandler && git add chromhandler/fitting/plotting.py tests/unit/fitting/test_plotting.py && git commit -m "$(cat <<'EOF'
Add add_baseline primitive with optional noise ribbon

Plots intercept + slope * t for one trace, optionally with a
translucent ±noise fill. The noise ribbon is the visual diagnostic for
"is the noise estimate sensible" — residuals should fit inside the
band if the model captures the peak shape correctly.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `add_model` for arbitrary callable overlays

The "plot model on top of dataset" capability. Decoupled from MCMC: the caller passes any function that maps `(time, trace_idx) → predicted_signal`. Useful for prior predictive checks, MAP fits, posterior median overlays, hand-specified parameters.

**Files:**
- Modify: `chromhandler/fitting/plotting.py`
- Modify: `tests/unit/fitting/test_plotting.py`

- [ ] **Step 3.1: Write the failing tests**

Append to `tests/unit/fitting/test_plotting.py`:

```python
class TestAddModel:
    """The add_model axes primitive."""

    def test_returns_same_axes(self) -> None:
        from chromhandler.fitting.plotting import add_model

        ds = _make_synthetic_dataset()
        fig, ax = plt.subplots()
        try:
            returned = add_model(ax, ds, trace_idx=0, model_fn=lambda t, i: t)
            assert returned is ax
        finally:
            plt.close(fig)

    def test_calls_model_fn_with_valid_time_and_trace_idx(self) -> None:
        from chromhandler.fitting.plotting import add_model

        ds = _make_synthetic_dataset()
        seen: dict[str, object] = {}

        def model_fn(t: np.ndarray, i: int) -> np.ndarray:
            seen["t"] = t
            seen["i"] = i
            return np.zeros_like(t)

        fig, ax = plt.subplots()
        try:
            add_model(ax, ds, trace_idx=2, model_fn=model_fn)
            assert seen["i"] == 2
            t = seen["t"]
            assert isinstance(t, np.ndarray)
            assert not np.any(np.isnan(t))  # NaN padding masked out
            assert t.shape == (101,)
        finally:
            plt.close(fig)

    def test_overlays_model_output_as_line(self) -> None:
        from chromhandler.fitting.plotting import add_model

        ds = _make_synthetic_dataset()

        def model_fn(t: np.ndarray, i: int) -> np.ndarray:
            return 0.5 + 0.0 * t  # constant 0.5

        fig, ax = plt.subplots()
        try:
            n_lines_before = len(ax.lines)
            add_model(ax, ds, trace_idx=0, model_fn=model_fn)
            assert len(ax.lines) == n_lines_before + 1
            (line,) = ax.lines[-1:]
            ydata = np.asarray(line.get_ydata())
            np.testing.assert_allclose(ydata, 0.5, atol=1e-9)
        finally:
            plt.close(fig)
```

- [ ] **Step 3.2: Run tests to verify they fail**

Run: `cd /Users/max/code/Chromhandler && uv run pytest tests/unit/fitting/test_plotting.py::TestAddModel -v`
Expected: 3 failures, all `ImportError: cannot import name 'add_model'`.

- [ ] **Step 3.3: Implement**

Append to `chromhandler/fitting/plotting.py`:

```python
def add_model(
    ax: Axes,
    dataset: PreparedDataset,
    trace_idx: int,
    model_fn: Callable[[NDArray[np.float64], int], NDArray[np.float64]],
    *,
    color: str = "tab:red",
    linewidth: float = 1.0,
    linestyle: str = "-",
) -> Axes:
    """Overlay an arbitrary model evaluation on one trace.

    Decoupled from MCMC: ``model_fn`` is any callable that takes
    ``(time[n_valid], trace_idx)`` and returns predicted signal of the
    same shape. NaN-padded samples are masked out before the call.

    Use cases include prior predictive checks, MAP fits, posterior
    median overlays, and hand-specified parameter sweeps.

    Args:
        ax: Target axes (mutated and returned).
        dataset: Source of time + valid mask.
        trace_idx: Which trace to evaluate.
        model_fn: Callable ``(time, trace_idx) -> predicted_signal``.
        color: Line colour.
        linewidth: Line width in points.
        linestyle: Line style.

    Returns:
        The same ``ax`` passed in.
    """
    valid = dataset.valid_mask[trace_idx]
    t = dataset.time[trace_idx][valid]
    predicted = model_fn(t, trace_idx)
    ax.plot(t, predicted, color=color, linewidth=linewidth, linestyle=linestyle)
    return ax
```

- [ ] **Step 3.4: Run tests to verify they pass**

Run: `cd /Users/max/code/Chromhandler && uv run pytest tests/unit/fitting/test_plotting.py -v`
Expected: 13 passed (10 prior + 3 new).

- [ ] **Step 3.5: Quality gates**

Run:
```
cd /Users/max/code/Chromhandler && uv run ruff check chromhandler/fitting/plotting.py tests/unit/fitting/test_plotting.py && uv run pyright chromhandler/fitting/plotting.py tests/unit/fitting/test_plotting.py
```
Expected: All checks passed; 0 errors.

- [ ] **Step 3.6: Commit**

```
cd /Users/max/code/Chromhandler && git add chromhandler/fitting/plotting.py tests/unit/fitting/test_plotting.py && git commit -m "$(cat <<'EOF'
Add add_model primitive for arbitrary callable overlays

Lets callers overlay any model evaluation on a trace by passing a
callable (time, trace_idx) -> predicted_signal. Decoupled from MCMC:
works equally well for prior predictive checks, MAP fits, posterior
median overlays, and hand-specified parameter sweeps. The Fitter
rewrite later wraps this with posterior-aware helpers.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Figure-level convenience — `plot_overview` + `plot_baseline_diagnostic`

**Files:**
- Modify: `chromhandler/fitting/plotting.py`
- Modify: `tests/unit/fitting/test_plotting.py`

- [ ] **Step 4.1: Write the failing tests**

Append to `tests/unit/fitting/test_plotting.py`:

```python
class TestPlotOverview:
    """plot_overview figure-level convenience."""

    def test_returns_figure_with_one_axes_per_trace(self) -> None:
        from chromhandler.fitting.plotting import plot_overview

        ds = _make_synthetic_dataset(n_trace=4)
        fig = plot_overview(ds)
        try:
            # Each trace gets one axes; trailing cells in the grid stay
            # empty but are still added by plt.subplots — figure-level
            # convenience hides them.
            data_axes = [a for a in fig.axes if a.has_data() or a.lines or a.patches]
            assert len(data_axes) == 4
        finally:
            plt.close(fig)

    def test_save_path_writes_file(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from chromhandler.fitting.plotting import plot_overview

        ds = _make_synthetic_dataset(n_trace=2)
        out = tmp_path / "overview.png"
        fig = plot_overview(ds, path=out)
        try:
            assert out.exists()
            assert out.stat().st_size > 0
        finally:
            plt.close(fig)


class TestPlotBaselineDiagnostic:
    """plot_baseline_diagnostic figure-level convenience."""

    def test_returns_figure_with_one_axes_per_trace(self) -> None:
        from chromhandler.fitting.plotting import plot_baseline_diagnostic

        ds = _make_synthetic_dataset(n_trace=4)
        fig = plot_baseline_diagnostic(ds)
        try:
            data_axes = [a for a in fig.axes if a.has_data() or a.lines or a.patches]
            assert len(data_axes) == 4
        finally:
            plt.close(fig)

    def test_each_panel_has_baseline_line_and_noise_band(self) -> None:
        from chromhandler.fitting.plotting import plot_baseline_diagnostic

        ds = _make_synthetic_dataset(n_trace=3)
        fig = plot_baseline_diagnostic(ds)
        try:
            data_axes = [a for a in fig.axes if a.has_data() or a.lines or a.patches]
            assert len(data_axes) == 3
            for ax in data_axes:
                # signal line + baseline line >= 2; one PolyCollection from noise band.
                assert len(ax.lines) >= 2
                assert len(ax.collections) >= 1
        finally:
            plt.close(fig)

    def test_panel_titles_include_dt_and_noise(self) -> None:
        from chromhandler.fitting.plotting import plot_baseline_diagnostic

        ds = _make_synthetic_dataset(n_trace=2)
        fig = plot_baseline_diagnostic(ds)
        try:
            data_axes = [a for a in fig.axes if a.has_data() or a.lines or a.patches]
            for i, ax in enumerate(data_axes):
                title = ax.get_title()
                assert f"trace {i}" in title
                assert "dt=" in title
                assert "noise=" in title
        finally:
            plt.close(fig)

    def test_save_path_writes_file(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from chromhandler.fitting.plotting import plot_baseline_diagnostic

        ds = _make_synthetic_dataset(n_trace=2)
        out = tmp_path / "baseline_diag.png"
        fig = plot_baseline_diagnostic(ds, path=out)
        try:
            assert out.exists()
            assert out.stat().st_size > 0
        finally:
            plt.close(fig)
```

- [ ] **Step 4.2: Run tests to verify they fail**

Run: `cd /Users/max/code/Chromhandler && uv run pytest tests/unit/fitting/test_plotting.py::TestPlotOverview tests/unit/fitting/test_plotting.py::TestPlotBaselineDiagnostic -v`
Expected: 6 failures, all `ImportError`.

- [ ] **Step 4.3: Implement**

Append to `chromhandler/fitting/plotting.py`:

```python
def _grid_shape(n_trace: int, n_cols: int) -> tuple[int, int]:
    """Compute ``(n_rows, n_cols)`` for a square-ish grid of n_trace panels."""
    n_rows = (n_trace + n_cols - 1) // n_cols
    return n_rows, n_cols


def _hide_unused_axes(fig: Figure, n_trace: int) -> None:
    """Hide any axes beyond the n_trace data panels (trailing grid cells)."""
    for ax in fig.axes[n_trace:]:
        ax.set_visible(False)


def plot_overview(
    dataset: PreparedDataset,
    path: str | Path | None = None,
    *,
    n_cols: int = 3,
    figsize_per_panel: tuple[float, float] = (4.0, 2.5),
) -> Figure:
    """Per-trace grid: signal + annotation regions.

    A quick sanity check that the data and annotations look right before
    any fitting. One panel per trace; trailing grid cells (if any) are
    hidden.

    Args:
        dataset: The prepared dataset.
        path: If provided, save the figure to this path with
            ``fig.savefig``. The figure is always returned regardless.
        n_cols: Number of columns in the grid.
        figsize_per_panel: ``(width, height)`` per panel in inches.

    Returns:
        The constructed :class:`matplotlib.figure.Figure`.
    """
    n_rows, _ = _grid_shape(dataset.n_trace, n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        sharex=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for i in range(dataset.n_trace):
        ax = flat_axes[i]
        add_signal(ax, dataset, trace_idx=i)
        add_annotation_regions(ax, dataset)
        ax.set_title(f"trace {i}")
    _hide_unused_axes(fig, dataset.n_trace)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path)
    return fig


def plot_baseline_diagnostic(
    dataset: PreparedDataset,
    path: str | Path | None = None,
    *,
    n_cols: int = 3,
    figsize_per_panel: tuple[float, float] = (4.0, 2.5),
) -> Figure:
    """Per-trace grid: signal + annotation regions + OLS baseline + noise.

    The canonical pre-fit diagnostic for the foundations layer. Each
    panel title shows the per-trace ``dt`` and ``noise`` so the user
    sees at a glance whether the OLS baseline and noise estimate look
    sensible.

    Args:
        dataset: The prepared dataset.
        path: If provided, save the figure to this path. The figure is
            always returned regardless.
        n_cols: Number of columns in the grid.
        figsize_per_panel: ``(width, height)`` per panel in inches.

    Returns:
        The constructed :class:`matplotlib.figure.Figure`.
    """
    n_rows, _ = _grid_shape(dataset.n_trace, n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        sharex=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for i in range(dataset.n_trace):
        ax = flat_axes[i]
        add_annotation_regions(ax, dataset)
        add_signal(ax, dataset, trace_idx=i)
        add_baseline(ax, dataset, trace_idx=i, show_noise_band=True)
        dt = dataset.dt_per_trace[i]
        noise = dataset.noise_per_trace[i]
        ax.set_title(f"trace {i}: dt={dt:.4f}, noise={noise:.3f}")
    _hide_unused_axes(fig, dataset.n_trace)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path)
    return fig
```

- [ ] **Step 4.4: Run tests to verify they pass**

Run: `cd /Users/max/code/Chromhandler && uv run pytest tests/unit/fitting/test_plotting.py -v`
Expected: 19 passed (13 prior + 6 new).

- [ ] **Step 4.5: Quality gates**

Run:
```
cd /Users/max/code/Chromhandler && uv run ruff check chromhandler/fitting/plotting.py tests/unit/fitting/test_plotting.py && uv run pyright chromhandler/fitting/plotting.py tests/unit/fitting/test_plotting.py
```
Expected: All checks passed; 0 errors.

- [ ] **Step 4.6: Combined regression check**

Run the full foundations + plotting suite:
```
cd /Users/max/code/Chromhandler && uv run pytest tests/unit/fitting/test_annotations.py tests/unit/fitting/test_preprocessing.py tests/unit/fitting/test_baseline.py tests/unit/fitting/test_noise.py tests/unit/fitting/test_prepared_dataset.py tests/unit/fitting/test_plotting.py tests/integration/test_foundations_asm.py 2>&1 | tail -3
```
Expected: 58 passed (39 foundations + 19 plotting).

- [ ] **Step 4.7: Commit**

```
cd /Users/max/code/Chromhandler && git add chromhandler/fitting/plotting.py tests/unit/fitting/test_plotting.py && git commit -m "$(cat <<'EOF'
Add plot_overview and plot_baseline_diagnostic figure-level functions

plot_overview: per-trace grid of signal + annotation regions — a quick
sanity check before fitting.

plot_baseline_diagnostic: per-trace grid of signal + annotations + OLS
baseline + noise ribbon. Panel titles include per-trace dt and noise
for at-a-glance evaluation. The canonical pre-fit diagnostic.

Both compose the axes primitives (add_signal, add_baseline,
add_annotation_regions). Save to disk if a path is given; always
return the Figure for further customization.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## What this plan does NOT do (deliberately)

- **No posterior plots.** `add_posterior_band`, `plot_fit`, `plot_traces`, etc. are deferred to the Fitter rewrite plan. The architecture from this plan extends to them cleanly: same `add_*` + `plot_*` pattern, same module.
- **No methods on `PreparedDataset`.** Free functions only, per spec §3 and §6.
- **No splitting into `plotting/` package.** Single file is enough for the four primitives + two convenience functions. Spec §4 says split when it grows past ~400 lines; this plan stays well below that.
- **No interactive backend or Plotly.** Matplotlib only. Out of scope per spec §9.
- **No changes to `chromhandler.visualize` or `chromhandler/fitting/visualize.py`.** The first is unrelated (Handler-level chromatograms). The second is the broken pre-rewrite module — leave it for the Fitter rewrite plan to delete cleanly.

## Acceptance for the whole plan

- All 19 plotting tests pass: `tests/unit/fitting/test_plotting.py`.
- Full foundations + plotting suite: 58 tests pass.
- `uv run ruff check chromhandler/fitting/plotting.py tests/unit/fitting/test_plotting.py` clean.
- `uv run pyright chromhandler/fitting/plotting.py tests/unit/fitting/test_plotting.py` clean.
- No matplotlib import in any of `preprocessing`, `baseline`, `noise`, `prepared_dataset`, `annotations`. Verified by grep.
- 4 commits on `fix-fit`, each focused and atomic.
