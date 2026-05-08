# Plotting Architecture — Design

**Date:** 2026-05-08
**Branch:** `fix-fit`
**Status:** Design — pending user review

---

## 1. Goal

A coherent plotting layer for the chromatographic fitting workflow that supports both **pre-fit diagnostics** (now, on top of the foundations layer) and **post-fit visualizations** (later, on top of the Fitter rewrite). The architecture is fixed now so the post-fit code drops in cleanly later.

## 2. Core architectural choice

**Layered free functions are canonical; figure-level convenience functions wrap them.** No methods on `PreparedDataset`. Methods only appear later on `Fitter`, which already carries heavy dependencies anyway.

Two layers:

1. **Axes-level building blocks** — free functions that take an existing `matplotlib.axes.Axes`, mutate it, return it. Composable, layerable, idiomatic matplotlib.
2. **Figure-level convenience functions** — free functions that build a complete `Figure` for a common case by calling the axes primitives.

User-facing consumption:

```python
from chromhandler.fitting.plotting import plot_baseline_diagnostic
fig = plot_baseline_diagnostic(ds, path="diagnostic.png")
```

…or for custom layouts:

```python
from chromhandler.fitting.plotting import add_signal, add_baseline, add_annotation_regions
fig, ax = plt.subplots()
add_signal(ax, ds, trace_idx=0)
add_baseline(ax, ds, trace_idx=0)
add_annotation_regions(ax, ds)
```

## 3. Why this and not the alternatives

**Pure methods on entities** (`ds.plot_baseline()`):
- Easy to discover but rigid. Adding "show baseline AND model overlay" forces another method or kwargs sprawl.
- Couples the dataclass to matplotlib (lazy imports work but add complexity).

**Pure layered free functions only**:
- Maximum flexibility but discoverability suffers — users hunting for "how do I plot this?" don't autocomplete on a free function.

**Hybrid (this design)**:
- Free functions are the canonical implementation, no duplication.
- Figure-level convenience is one composition layer above; ~10–20 lines per function.
- Methods only on `Fitter` (later), and they will themselves call the same free functions. No code duplication.

## 4. File layout

Single module for now:

```
chromhandler/fitting/plotting.py     # NEW
```

Sized for a single file. If it grows past ~400 lines, split by layer (`plotting/data.py`, `plotting/diagnostics.py`, `plotting/posterior.py`).

Tests:

```
tests/unit/fitting/test_plotting.py  # NEW
```

## 5. Public API for the foundations-layer scope

Built on top of `PreparedDataset` from the foundations rewrite. No dependency on Fitter or MCMC.

### 5.1 Axes-level primitives

```python
def add_signal(
    ax: Axes,
    dataset: PreparedDataset,
    trace_idx: int,
    *,
    color: str = "tab:gray",
    linewidth: float = 0.8,
    alpha: float = 0.85,
) -> Axes:
    """Plot the raw baseline-subtracted-or-not signal of one trace."""

def add_baseline(
    ax: Axes,
    dataset: PreparedDataset,
    trace_idx: int,
    *,
    show_noise_band: bool = True,
    color: str = "tab:blue",
    linewidth: float = 1.0,
    linestyle: str = "--",
) -> Axes:
    """Plot the OLS baseline line, optionally with a ±noise ribbon."""

def add_annotation_regions(
    ax: Axes,
    dataset: PreparedDataset,
    *,
    peak_color: str = "tab:orange",
    baseline_color: str = "tab:green",
    alpha: float = 0.15,
) -> Axes:
    """Shade peak windows and baseline regions across the time axis."""

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
    """Overlay an arbitrary model function on a trace.

    `model_fn` takes ``(time[n_time], trace_idx)`` and returns predicted
    signal. Decoupled from MCMC: pass any callable that evaluates the
    model at parameters of choice (prior median, MAP, posterior median,
    hand-picked).
    """
```

### 5.2 Figure-level convenience functions

```python
def plot_overview(
    dataset: PreparedDataset,
    path: str | Path | None = None,
    *,
    n_cols: int = 3,
    figsize_per_panel: tuple[float, float] = (4.0, 2.5),
) -> Figure:
    """Per-trace grid: signal + annotation regions only. Quick sanity check
    that data + annotations look right before any fitting."""

def plot_baseline_diagnostic(
    dataset: PreparedDataset,
    path: str | Path | None = None,
    *,
    n_cols: int = 3,
    figsize_per_panel: tuple[float, float] = (4.0, 2.5),
) -> Figure:
    """Per-trace grid: signal + annotation regions + OLS baseline + noise
    ribbon. The canonical pre-fit diagnostic for the foundations layer."""
```

Both convenience functions:
- Build a square-ish subplot grid (`n_cols` chosen so trailing panels are at most one short row).
- Share x-axis across panels (synchronized zoom).
- Save to `path` if provided; always return the `Figure` for further customization.
- Title each panel with `f"trace {i}: dt={dt:.4f}, noise={noise:.3f}"` for `plot_baseline_diagnostic`.

### 5.3 What's deferred to the Fitter-rewrite plan

Not part of this scope, but the architecture admits them cleanly later:

```python
def add_posterior_band(ax, samples, trace_idx, peak_idx) -> Axes
def add_posterior_residuals(ax, samples, dataset, trace_idx) -> Axes
def plot_fit(fitter, path=None) -> Figure          # signal + baseline + posterior
def plot_traces(fitter, path=None) -> Figure       # MCMC trace plots
def plot_prior_predictive(ds, prior_model, path=None) -> Figure
```

Each follows the same axes-primitive + figure-convenience pattern. `Fitter` will gain thin methods (`fitter.plot_fit(...)`) that call the figure-level free functions.

## 6. Coupling and import policy

- **No matplotlib imports inside `chromhandler.fitting.preprocessing`, `baseline`, `noise`, `prepared_dataset`, or `annotations`.** The foundations stay pure-data.
- `chromhandler/fitting/plotting.py` is the only module that imports matplotlib in the foundations + plotting scope.
- Matplotlib is imported eagerly at the top of `plotting.py` (no lazy-import gymnastics — if the user is calling plotting code, they want matplotlib loaded).
- `chromhandler/fitting/__init__.py` does **not** re-export plotting symbols; users import from `chromhandler.fitting.plotting` explicitly. Keeps package import light.

## 7. Testing strategy

Plotting tests focus on **smoke + structure**, not pixel-level rendering:

- Build a synthetic `PreparedDataset` (small, ~3 traces, 100 samples each, 1 peak window, 2 baseline regions).
- Call each axes-primitive, assert the returned `ax` is the same object passed in (mutation, not creation).
- Call each figure-level function, assert:
  - Returned object is a `Figure`.
  - `len(fig.axes) == n_trace` (or `n_trace + 1` for any colorbar).
  - For `plot_baseline_diagnostic`: noise band is present (axes contain a `PolyCollection`); baseline line is present.
- Exercise `path` saving: tmp_path fixture, assert file exists, non-zero size.

No image diffing. We trust matplotlib to render correctly; we test our composition logic.

## 8. Naming policy

- Axes primitives prefixed `add_*` (matplotlib idiom: they ADD content to existing axes).
- Figure-level prefixed `plot_*` (they CREATE a figure end-to-end).

This makes the role of each function obvious from the name.

## 9. Out of scope

- Posterior visualizations — deferred to Fitter rewrite plan.
- Interactive plots (Plotly / Bokeh) — sticking with matplotlib for parity with existing code.
- Multi-dataset comparison plots — possible later via the axes primitives, no work needed in this scope.
- The existing `chromhandler.visualize` (Handler-level chromatogram plots) — separate concern, untouched.
- The existing broken `chromhandler/fitting/visualize.py` — will be deleted as part of the Fitter rewrite, not in this plan.

## 10. Acceptance

- `chromhandler/fitting/plotting.py` exists with the four axes primitives and two figure-level functions.
- `tests/unit/fitting/test_plotting.py` exercises every public function; all pass.
- ruff + pyright clean.
- The user's existing test script can call `plot_baseline_diagnostic(prepare_dataset(...))` and produce a sensible PNG with no warnings.
- Foundations modules unchanged — no new matplotlib dependency anywhere outside `plotting.py`.
