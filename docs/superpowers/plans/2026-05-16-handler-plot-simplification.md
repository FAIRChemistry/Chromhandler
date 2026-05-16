# Handler Plot Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing `Handler.plot` (peak-area kinetics) and `Handler.visualize` (feature-heavy chromatogram plotter) with a single, minimal `Handler.plot` for raw chromatograms plus a sibling `Handler.plot_windows` for retention-time windows, both supporting three overlay modes (`"single"` / `"sample"` / `"all"`).

**Architecture:** A new module `chromhandler/plotting.py` holds two free figure-builder functions (`plot_traces`, `plot_window_grid`) plus small grouping/color helpers; `Handler.plot` and `Handler.plot_windows` are thin wrappers that pass through. All legacy plotting code (`chromhandler/visualize.py`, the old `Handler.plot`, `Handler.visualize`, and their tests) is deleted outright — no compatibility shims.

**Tech Stack:** matplotlib (`pyplot`, `colormaps.viridis`), numpy, pydantic models from `chromhandler.model` / `chromhandler.annotations`.

---

## File Structure

| Path | Role | Action |
| --- | --- | --- |
| `chromhandler/plotting.py` | New module: free functions `plot_traces`, `plot_window_grid`, helpers `_group_chromatograms`, `_line_colors`. | Create |
| `chromhandler/handler.py` | Drop old `plot` (lines ~1709–1880) and `visualize` (lines ~1885–1925). Add new `plot` and `plot_windows` methods that delegate to `chromhandler.plotting`. | Modify |
| `chromhandler/visualize.py` | Entire 574-line module is obsolete. | Delete |
| `tests/unit/handler/test_plot.py` | New unit tests for the new methods. | Create |
| `tests/integration/test_handler_plot.py` | Tests the deleted kinetics plotter. | Delete |
| `tests/integration/test_visualize.py` | Tests the deleted `Handler.visualize`. | Delete |
| `notebooks/alignment_demo.ipynb` | Currently rolls its own `plot_traces` helper. Switch to `handler.plot(overlay="all")`. | Modify (optional last task) |

## API Summary

```python
# chromhandler/plotting.py

OverlayMode = Literal["all", "sample", "single"]

def plot_traces(
    handler: Handler,
    *,
    overlay: OverlayMode = "single",
    ax_size: tuple[float, float] = (4.0, 3.0),
    share_y: bool = False,
    save: Path | str | None = None,
) -> tuple[Figure, NDArray[np.object_]]: ...

def plot_window_grid(
    handler: Handler,
    annotations: list[PeakAnnotation],
    *,
    overlay: OverlayMode = "single",
    ax_size: tuple[float, float] = (4.0, 3.0),
    share_y: bool = False,
    save: Path | str | None = None,
) -> tuple[Figure, NDArray[np.object_]]: ...
```

```python
# chromhandler/handler.py

def plot(self, **kwargs) -> tuple[Figure, NDArray[np.object_]]:
    return plot_traces(self, **kwargs)

def plot_windows(self, annotations: list[PeakAnnotation], **kwargs):
    return plot_window_grid(self, annotations, **kwargs)
```

## Grouping Semantics (drives both methods)

| overlay     | groups (rows)                                    | colors within a group           |
| ----------- | ------------------------------------------------ | ------------------------------- |
| `"single"`  | one group per chromatogram (flatten samples)     | 1 line → `tab:blue`             |
| `"sample"`  | one group per sample, holding its chromatograms  | viridis across chromatograms    |
| `"all"`     | one group containing every chromatogram (flat)   | viridis across flat list        |

`plot_traces` → grid shape `(n_groups, 1)`. `plot_window_grid` → grid shape `(n_groups, len(annotations))`.

Color rule: **one line on an axis → `"tab:blue"`; ≥2 lines → viridis evenly spaced over `[0, 1]` in handler order.**

---

## Task 1: Delete legacy plotting code

**Files:**
- Delete: `chromhandler/visualize.py`
- Delete: `tests/integration/test_handler_plot.py`
- Delete: `tests/integration/test_visualize.py`
- Modify: `chromhandler/handler.py` (remove old `plot` and `visualize` methods; remove the `from . import visualize` import)

- [ ] **Step 1: Remove the old `Handler.plot` method (lines ~1709–1880)**

In `chromhandler/handler.py`, locate the method beginning with:

```python
    def plot(
        self,
        samples: list[str] | None = None,
        figsize: tuple[float, float] | None = None,
        show_balance: bool = False,
```

and delete the entire method (everything up to and including the matching final `return` and trailing blank line before the next method). Do not yet add a replacement — the new `plot` is added in Task 4.

- [ ] **Step 2: Remove the old `Handler.visualize` method (lines ~1885–1925)**

Delete the entire method starting at:

```python
    def visualize(
        self,
        n_cols: int = 2,
```

and ending at its closing `)`.

- [ ] **Step 3: Drop the `visualize` module import**

Find the top of `chromhandler/handler.py`:

```python
from . import pretty, visualize
```

Change it to:

```python
from . import pretty
```

- [ ] **Step 4: Delete the obsolete files**

```bash
rm chromhandler/visualize.py
rm tests/integration/test_handler_plot.py
rm tests/integration/test_visualize.py
```

- [ ] **Step 5: Verify nothing else references `chromhandler.visualize`**

```bash
grep -rn "chromhandler\.visualize\|from chromhandler import visualize\|from \.visualize\|from \. import visualize" chromhandler tests notebooks
```
Expected: no output. If a hit appears, remove the import.

- [ ] **Step 6: Verify ruff and pyright still pass on the trimmed handler**

```bash
uv run ruff check chromhandler/handler.py
uv run pyright chromhandler/handler.py
```
Expected: both clean.

- [ ] **Step 7: Commit**

```bash
git add chromhandler/handler.py
git add -u chromhandler/visualize.py tests/integration/test_handler_plot.py tests/integration/test_visualize.py
git commit -m "refactor(handler): drop kinetics plot and visualize() ahead of plot rewrite"
```

---

## Task 2: Create `chromhandler/plotting.py` skeleton with helpers

**Files:**
- Create: `chromhandler/plotting.py`
- Create: `tests/unit/handler/test_plot.py` (helper tests only at this stage)

- [ ] **Step 1: Write the failing helper tests**

Create `tests/unit/handler/test_plot.py`:

```python
"""Tests for ``chromhandler.plotting`` helpers and ``Handler.plot``."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from chromhandler.annotations import PeakAnnotation
from chromhandler.handler import Handler
from chromhandler.model import Chromatogram, Estimate, Peak, Sample
from chromhandler.plotting import _group_chromatograms, _line_colors


def _make_handler(n_samples: int = 2, chroms_per_sample: int = 2) -> Handler:
    handler = Handler()
    t = np.linspace(0.0, 10.0, 201, dtype=float).tolist()
    for s in range(n_samples):
        chroms = []
        for c in range(chroms_per_sample):
            sig = (np.sin(np.linspace(0, 6.28, 201)) + 0.1 * c).tolist()
            peak = Peak(
                chromatogram_id=f"s{s}_c{c}",
                location=Estimate(mean=5.0),
                area=Estimate(mean=1.0),
            )
            chroms.append(
                Chromatogram(
                    id=f"s{s}_c{c}",
                    sample_id=f"s{s}",
                    time=t,
                    signal=sig,
                    peaks=[peak],
                )
            )
        handler.samples.append(Sample(id=f"s{s}", chromatograms=chroms))
    return handler


def test_group_chromatograms_single() -> None:
    handler = _make_handler(n_samples=2, chroms_per_sample=2)
    groups = _group_chromatograms(handler, overlay="single")
    assert len(groups) == 4
    assert all(len(g) == 1 for g in groups)
    assert [g[0].id for g in groups] == ["s0_s0_c0", "s0_s0_c1", "s1_s1_c0", "s1_s1_c1"]


def test_group_chromatograms_sample() -> None:
    handler = _make_handler(n_samples=2, chroms_per_sample=2)
    groups = _group_chromatograms(handler, overlay="sample")
    assert len(groups) == 2
    assert [len(g) for g in groups] == [2, 2]
    assert [g[0].sample_id for g in groups] == ["s0", "s1"]


def test_group_chromatograms_all() -> None:
    handler = _make_handler(n_samples=2, chroms_per_sample=2)
    groups = _group_chromatograms(handler, overlay="all")
    assert len(groups) == 1
    assert len(groups[0]) == 4


def test_line_colors_single() -> None:
    colors = _line_colors(1)
    assert colors == [matplotlib.colors.to_rgba("tab:blue")]


def test_line_colors_multi_uses_viridis() -> None:
    colors = _line_colors(4)
    cmap = matplotlib.colormaps["viridis"]
    assert colors == [cmap(i / 3) for i in range(4)]
    assert colors[0] != colors[-1]


def test_group_chromatograms_empty_raises() -> None:
    handler = Handler()
    with pytest.raises(ValueError, match="no chromatograms"):
        _group_chromatograms(handler, overlay="single")
```

- [ ] **Step 2: Run the new tests to confirm they fail**

```bash
uv run pytest tests/unit/handler/test_plot.py -v
```
Expected: collection error / `ImportError: cannot import name '_group_chromatograms' from 'chromhandler.plotting'`.

- [ ] **Step 3: Create the helpers module**

Create `chromhandler/plotting.py`:

```python
"""Handler-level chromatogram plotting.

Provides two figure builders:

- :func:`plot_traces`: one signal panel per group (overlay = single / sample / all).
- :func:`plot_window_grid`: rows × windows grid with the same overlay semantics.

Plus small internal helpers (:func:`_group_chromatograms`, :func:`_line_colors`)
that drive both builders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from chromhandler.annotations import PeakAnnotation
    from chromhandler.handler import Handler
    from chromhandler.model import Chromatogram

OverlayMode = Literal["all", "sample", "single"]


def _group_chromatograms(handler: Handler, overlay: OverlayMode) -> list[list[Chromatogram]]:
    """Flatten handler.samples → chromatograms into grouped rows.

    - ``"single"``: one group per chromatogram (flat).
    - ``"sample"``: one group per sample, in handler order.
    - ``"all"``: one group containing every chromatogram (flat).

    Raises ``ValueError`` if the handler has no chromatograms.
    """
    flat: list[Chromatogram] = []
    per_sample: list[list[Chromatogram]] = []
    for sample in handler.samples:
        if not sample.chromatograms:
            continue
        per_sample.append(list(sample.chromatograms))
        flat.extend(sample.chromatograms)
    if not flat:
        raise ValueError("Handler has no chromatograms across any sample.")
    if overlay == "single":
        return [[c] for c in flat]
    if overlay == "sample":
        return per_sample
    if overlay == "all":
        return [flat]
    raise ValueError(f"Unknown overlay mode: {overlay!r}")


def _line_colors(n: int) -> list[tuple[float, float, float, float]]:
    """Return ``n`` line colors per the project rule.

    1 line → ``tab:blue``; ≥ 2 lines → viridis evenly spaced over ``[0, 1]``.
    """
    if n <= 0:
        return []
    if n == 1:
        return [mcolors.to_rgba("tab:blue")]
    cmap = plt.get_cmap("viridis")
    return [cmap(i / (n - 1)) for i in range(n)]
```

- [ ] **Step 4: Run helper tests to verify they pass**

```bash
uv run pytest tests/unit/handler/test_plot.py -v
```
Expected: 6 PASSED.

- [ ] **Step 5: Lint and type-check**

```bash
uv run ruff check chromhandler/plotting.py tests/unit/handler/test_plot.py
uv run pyright chromhandler/plotting.py tests/unit/handler/test_plot.py
```
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add chromhandler/plotting.py tests/unit/handler/test_plot.py
git commit -m "feat(plotting): add chromhandler.plotting with grouping and color helpers"
```

---

## Task 3: Implement `plot_traces`

**Files:**
- Modify: `chromhandler/plotting.py`
- Modify: `tests/unit/handler/test_plot.py`

- [ ] **Step 1: Append `plot_traces` tests**

Append to `tests/unit/handler/test_plot.py`:

```python
from chromhandler.plotting import plot_traces


def _close(fig: Any) -> None:
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_traces_single_default() -> None:
    handler = _make_handler(n_samples=2, chroms_per_sample=2)
    fig, axes = plot_traces(handler)
    try:
        assert axes.shape == (4, 1)
        for ax in axes.flatten():
            lines = ax.get_lines()
            assert len(lines) == 1
            assert tuple(lines[0].get_color()[:3]) == matplotlib.colors.to_rgba("tab:blue")[:3]
    finally:
        _close(fig)


def test_plot_traces_sample_mode_groups_per_sample() -> None:
    handler = _make_handler(n_samples=2, chroms_per_sample=2)
    fig, axes = plot_traces(handler, overlay="sample")
    try:
        assert axes.shape == (2, 1)
        for ax in axes.flatten():
            assert len(ax.get_lines()) == 2
            colors = [tuple(l.get_color()) for l in ax.get_lines()]
            assert colors[0] != colors[1]
    finally:
        _close(fig)


def test_plot_traces_all_mode_one_ax() -> None:
    handler = _make_handler(n_samples=2, chroms_per_sample=2)
    fig, axes = plot_traces(handler, overlay="all")
    try:
        assert axes.shape == (1, 1)
        ax = axes[0, 0]
        assert len(ax.get_lines()) == 4
    finally:
        _close(fig)


def test_plot_traces_ax_size_drives_figsize() -> None:
    handler = _make_handler(n_samples=3, chroms_per_sample=1)
    fig, axes = plot_traces(handler, overlay="single", ax_size=(2.5, 1.5))
    try:
        # figsize = (cols * w, rows * h)
        assert fig.get_size_inches()[0] == pytest.approx(2.5)
        assert fig.get_size_inches()[1] == pytest.approx(3 * 1.5)
        assert axes.shape == (3, 1)
    finally:
        _close(fig)


def test_plot_traces_share_y() -> None:
    handler = _make_handler(n_samples=2, chroms_per_sample=1)
    fig, axes = plot_traces(handler, overlay="single", share_y=True)
    try:
        ylims = [ax.get_ylim() for ax in axes.flatten()]
        assert ylims[0] == ylims[1]
    finally:
        _close(fig)


def test_plot_traces_save_writes_file(tmp_path: Any) -> None:
    handler = _make_handler(n_samples=1, chroms_per_sample=1)
    out = tmp_path / "plot.png"
    fig, _ = plot_traces(handler, save=out)
    try:
        assert out.exists() and out.stat().st_size > 0
    finally:
        _close(fig)
```

- [ ] **Step 2: Run the new tests to confirm they fail**

```bash
uv run pytest tests/unit/handler/test_plot.py -v
```
Expected: 6 NEW tests fail with `ImportError: cannot import name 'plot_traces'`.

- [ ] **Step 3: Implement `plot_traces` in `chromhandler/plotting.py`**

Append to `chromhandler/plotting.py`:

```python
def plot_traces(
    handler: Handler,
    *,
    overlay: OverlayMode = "single",
    ax_size: tuple[float, float] = (4.0, 3.0),
    share_y: bool = False,
    save: "Path | str | None" = None,
) -> tuple[Figure, NDArray[np.object_]]:
    """Plot raw chromatograms with the project overlay/color rules.

    Args:
        handler: Source of chromatograms.
        overlay: ``"single"`` = one ax per chromatogram (flat);
            ``"sample"`` = one ax per sample, chromatograms overlaid;
            ``"all"`` = one ax containing every chromatogram.
        ax_size: ``(width, height)`` in inches per axis. Total ``figsize`` is
            ``(width, n_rows * height)``.
        share_y: If ``True``, all axes share y-limits.
        save: If given, write the figure to this path before returning.

    Returns:
        ``(fig, axes)`` where ``axes`` is a 2-D ``ndarray`` of shape
        ``(n_groups, 1)``.
    """
    groups = _group_chromatograms(handler, overlay)
    n_rows = len(groups)
    width, height = ax_size
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(width, n_rows * height),
        squeeze=False,
        sharey=share_y,
    )
    for row, group in enumerate(groups):
        ax = axes[row, 0]
        colors = _line_colors(len(group))
        for chrom, color in zip(group, colors, strict=True):
            ax.plot(
                np.asarray(chrom.time),
                np.asarray(chrom.signal),
                color=color,
                lw=1.0,
                label=chrom.id,
            )
        ax.set_xlabel("retention time (min)")
        ax.set_ylabel("signal")
        if overlay == "sample":
            ax.set_title(group[0].sample_id)
        elif overlay == "single":
            ax.set_title(group[0].id)
    fig.tight_layout()
    if save is not None:
        fig.savefig(save)
    return fig, axes
```

- [ ] **Step 4: Run all `test_plot.py` tests to verify they pass**

```bash
uv run pytest tests/unit/handler/test_plot.py -v
```
Expected: 12 PASSED.

- [ ] **Step 5: Lint and type-check**

```bash
uv run ruff check chromhandler/plotting.py tests/unit/handler/test_plot.py
uv run pyright chromhandler/plotting.py tests/unit/handler/test_plot.py
```
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add chromhandler/plotting.py tests/unit/handler/test_plot.py
git commit -m "feat(plotting): add plot_traces with overlay modes and color rule"
```

---

## Task 4: Wire up `Handler.plot`

**Files:**
- Modify: `chromhandler/handler.py`
- Modify: `tests/unit/handler/test_plot.py`

- [ ] **Step 1: Append a handler-method test**

Add to `tests/unit/handler/test_plot.py`:

```python
def test_handler_plot_delegates_to_plot_traces() -> None:
    handler = _make_handler(n_samples=1, chroms_per_sample=2)
    fig, axes = handler.plot(overlay="sample")
    try:
        assert axes.shape == (1, 1)
        assert len(axes[0, 0].get_lines()) == 2
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
uv run pytest tests/unit/handler/test_plot.py::test_handler_plot_delegates_to_plot_traces -v
```
Expected: `AttributeError: 'Handler' object has no attribute 'plot'`.

- [ ] **Step 3: Add `Handler.plot` and update imports**

In `chromhandler/handler.py`, at the top of the file (after existing imports), add a forward import. Inside the `Handler` class, locate the empty space where the old `plot` method used to live (between the alignment section and the next section). Insert:

```python
    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        *,
        overlay: Literal["all", "sample", "single"] = "single",
        ax_size: tuple[float, float] = (4.0, 3.0),
        share_y: bool = False,
        save: Path | str | None = None,
    ) -> tuple[Figure, npt.NDArray[Any]]:
        """Plot raw chromatograms.

        Thin wrapper over :func:`chromhandler.plotting.plot_traces`.

        Args:
            overlay: Grouping mode. ``"single"`` puts each chromatogram on its
                own axis (tab:blue). ``"sample"`` groups chromatograms per
                sample (viridis within each axis). ``"all"`` overlays all
                chromatograms on one axis (viridis).
            ax_size: ``(width, height)`` in inches per axis.
            share_y: If ``True``, all axes share y-limits.
            save: If set, write the figure to this path before returning.

        Returns:
            ``(fig, axes)`` with ``axes`` shape ``(n_groups, 1)``.
        """
        from chromhandler.plotting import plot_traces

        return plot_traces(
            self,
            overlay=overlay,
            ax_size=ax_size,
            share_y=share_y,
            save=save,
        )
```

`Literal` is already imported from `typing`. `npt` and `Any` are already imported (under TYPE_CHECKING; the method body uses no runtime references to them, so the deferred-annotations import is fine).

- [ ] **Step 4: Run the handler test to verify it passes**

```bash
uv run pytest tests/unit/handler/test_plot.py -v
```
Expected: 13 PASSED.

- [ ] **Step 5: Lint and type-check**

```bash
uv run ruff check chromhandler/handler.py
uv run pyright chromhandler/handler.py
```
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add chromhandler/handler.py tests/unit/handler/test_plot.py
git commit -m "feat(handler): add Handler.plot wrapper for plot_traces"
```

---

## Task 5: Implement `plot_window_grid`

**Files:**
- Modify: `chromhandler/plotting.py`
- Modify: `tests/unit/handler/test_plot.py`

- [ ] **Step 1: Append `plot_window_grid` tests**

Add to `tests/unit/handler/test_plot.py`:

```python
from chromhandler.plotting import plot_window_grid


def _annotations() -> list[PeakAnnotation]:
    return [
        PeakAnnotation(molecule_id="A", rt_min=1.0, rt_max=2.0),
        PeakAnnotation(molecule_id="B", rt_min=4.0, rt_max=6.0),
    ]


def test_plot_window_grid_single_shape_and_xlim() -> None:
    handler = _make_handler(n_samples=2, chroms_per_sample=2)
    fig, axes = plot_window_grid(handler, _annotations(), overlay="single")
    try:
        assert axes.shape == (4, 2)
        # Column 0 clipped to [1.0, 2.0], column 1 to [4.0, 6.0]
        for row in range(4):
            assert axes[row, 0].get_xlim() == pytest.approx((1.0, 2.0))
            assert axes[row, 1].get_xlim() == pytest.approx((4.0, 6.0))
            for col in range(2):
                assert len(axes[row, col].get_lines()) == 1
    finally:
        plt.close(fig)


def test_plot_window_grid_sample_shape() -> None:
    handler = _make_handler(n_samples=2, chroms_per_sample=3)
    fig, axes = plot_window_grid(handler, _annotations(), overlay="sample")
    try:
        assert axes.shape == (2, 2)
        for row in range(2):
            for col in range(2):
                assert len(axes[row, col].get_lines()) == 3
    finally:
        plt.close(fig)


def test_plot_window_grid_all_shape() -> None:
    handler = _make_handler(n_samples=2, chroms_per_sample=2)
    fig, axes = plot_window_grid(handler, _annotations(), overlay="all")
    try:
        assert axes.shape == (1, 2)
        for col in range(2):
            assert len(axes[0, col].get_lines()) == 4
    finally:
        plt.close(fig)


def test_plot_window_grid_save_writes_file(tmp_path: Any) -> None:
    handler = _make_handler(n_samples=1, chroms_per_sample=1)
    out = tmp_path / "windows.png"
    fig, _ = plot_window_grid(handler, _annotations(), save=out)
    try:
        assert out.exists() and out.stat().st_size > 0
    finally:
        plt.close(fig)


def test_plot_window_grid_empty_annotations_raises() -> None:
    handler = _make_handler(n_samples=1, chroms_per_sample=1)
    with pytest.raises(ValueError, match="at least one"):
        plot_window_grid(handler, [], overlay="single")
```

Add `import matplotlib.pyplot as plt` to the imports near the top of the test file if not already present.

- [ ] **Step 2: Run the new tests to confirm they fail**

```bash
uv run pytest tests/unit/handler/test_plot.py -v
```
Expected: 5 new failures with `ImportError: cannot import name 'plot_window_grid'`.

- [ ] **Step 3: Implement `plot_window_grid`**

Append to `chromhandler/plotting.py`:

```python
def plot_window_grid(
    handler: Handler,
    annotations: "list[PeakAnnotation]",
    *,
    overlay: OverlayMode = "single",
    ax_size: tuple[float, float] = (4.0, 3.0),
    share_y: bool = False,
    save: "Path | str | None" = None,
) -> tuple[Figure, NDArray[np.object_]]:
    """Plot per-window panels in a ``(group, window)`` grid.

    Each row is a group (defined by ``overlay`` as in :func:`plot_traces`)
    and each column corresponds to one ``PeakAnnotation``. The panel's
    x-axis is clipped to ``[rt_min, rt_max]`` (no bounds are drawn —
    the clip is implicit).

    Args:
        handler: Source of chromatograms.
        annotations: One :class:`PeakAnnotation` per column. Must be
            non-empty.
        overlay: Same semantics as :func:`plot_traces`.
        ax_size: ``(width, height)`` in inches per panel.
        share_y: If ``True``, all panels share y-limits.
        save: If given, write the figure to this path before returning.

    Returns:
        ``(fig, axes)`` with ``axes`` shape ``(n_groups, len(annotations))``.
    """
    if not annotations:
        raise ValueError("plot_window_grid: need at least one PeakAnnotation.")
    groups = _group_chromatograms(handler, overlay)
    n_rows = len(groups)
    n_cols = len(annotations)
    width, height = ax_size
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * width, n_rows * height),
        squeeze=False,
        sharey=share_y,
    )
    for row, group in enumerate(groups):
        colors = _line_colors(len(group))
        for col, ann in enumerate(annotations):
            ax = axes[row, col]
            for chrom, color in zip(group, colors, strict=True):
                t = np.asarray(chrom.time)
                s = np.asarray(chrom.signal)
                in_window = (t >= ann.rt_min) & (t <= ann.rt_max)
                ax.plot(t[in_window], s[in_window], color=color, lw=1.0, label=chrom.id)
            ax.set_xlim(ann.rt_min, ann.rt_max)
            if row == n_rows - 1:
                ax.set_xlabel("retention time (min)")
            if col == 0:
                if overlay == "sample":
                    ax.set_ylabel(group[0].sample_id)
                elif overlay == "single":
                    ax.set_ylabel(group[0].id)
                else:
                    ax.set_ylabel("signal")
            if row == 0:
                ax.set_title(ann.molecule_id)
    fig.tight_layout()
    if save is not None:
        fig.savefig(save)
    return fig, axes
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/unit/handler/test_plot.py -v
```
Expected: 18 PASSED.

- [ ] **Step 5: Lint and type-check**

```bash
uv run ruff check chromhandler/plotting.py tests/unit/handler/test_plot.py
uv run pyright chromhandler/plotting.py tests/unit/handler/test_plot.py
```
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add chromhandler/plotting.py tests/unit/handler/test_plot.py
git commit -m "feat(plotting): add plot_window_grid for per-window panels"
```

---

## Task 6: Wire up `Handler.plot_windows`

**Files:**
- Modify: `chromhandler/handler.py`
- Modify: `tests/unit/handler/test_plot.py`

- [ ] **Step 1: Append the handler-method test**

Add to `tests/unit/handler/test_plot.py`:

```python
def test_handler_plot_windows_delegates() -> None:
    handler = _make_handler(n_samples=2, chroms_per_sample=2)
    fig, axes = handler.plot_windows(_annotations(), overlay="sample")
    try:
        assert axes.shape == (2, 2)
    finally:
        plt.close(fig)
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
uv run pytest tests/unit/handler/test_plot.py::test_handler_plot_windows_delegates -v
```
Expected: `AttributeError: 'Handler' object has no attribute 'plot_windows'`.

- [ ] **Step 3: Add `Handler.plot_windows`**

Insert immediately after the `Handler.plot` method in `chromhandler/handler.py`:

```python
    def plot_windows(
        self,
        annotations: list[PeakAnnotation],
        *,
        overlay: Literal["all", "sample", "single"] = "single",
        ax_size: tuple[float, float] = (4.0, 3.0),
        share_y: bool = False,
        save: Path | str | None = None,
    ) -> tuple[Figure, npt.NDArray[Any]]:
        """Plot a ``(group, window)`` grid of chromatograms.

        Thin wrapper over :func:`chromhandler.plotting.plot_window_grid`.

        Args:
            annotations: One :class:`PeakAnnotation` per column.
            overlay: Same semantics as :meth:`plot`.
            ax_size: ``(width, height)`` in inches per panel.
            share_y: If ``True``, all panels share y-limits.
            save: If set, write the figure to this path before returning.

        Returns:
            ``(fig, axes)`` with ``axes`` shape
            ``(n_groups, len(annotations))``.
        """
        from chromhandler.plotting import plot_window_grid

        return plot_window_grid(
            self,
            annotations,
            overlay=overlay,
            ax_size=ax_size,
            share_y=share_y,
            save=save,
        )
```

- [ ] **Step 4: Run the handler test to verify it passes**

```bash
uv run pytest tests/unit/handler/test_plot.py -v
```
Expected: 19 PASSED.

- [ ] **Step 5: Lint and type-check**

```bash
uv run ruff check chromhandler/handler.py
uv run pyright chromhandler/handler.py
```
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add chromhandler/handler.py tests/unit/handler/test_plot.py
git commit -m "feat(handler): add Handler.plot_windows wrapper for plot_window_grid"
```

---

## Task 7: Update the alignment demo notebook

**Files:**
- Modify: `notebooks/alignment_demo.ipynb`

- [ ] **Step 1: Replace the local `plot_traces` helper with `handler.plot`**

Open `notebooks/alignment_demo.ipynb`. In the cell with id `plotfn-code`, replace the body (currently defines a `plot_traces` helper) with:

```python
LOWER_RT, UPPER_RT = 6.5, 8.1


def show(handler, title):
    fig, _ = handler.plot(overlay="all", ax_size=(9.0, 4.0))
    ax = fig.axes[0]
    ax.axvspan(LOWER_RT, UPPER_RT, alpha=0.1, color="orange", label="alignment window")
    ax.set_xlim(LOWER_RT - 0.2, UPPER_RT + 0.2)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig
```

- [ ] **Step 2: Update the three plot cells to use `show(...)`**

In cells `before-code`, `after-code`, and `wide-code`, replace each body:

`before-code`:

```python
fig_before = show(handler, title="Before alignment")
fig_before
```

`after-code`:

```python
fig_after = show(handler, title="After alignment")
fig_after
```

`wide-code` — keep the full-range version using `handler.plot` directly:

```python
fig_wide, _ = handler.plot(overlay="all", ax_size=(9.0, 4.0))
fig_wide.axes[0].set_title("After alignment (full range)")
fig_wide
```

- [ ] **Step 3: Execute the notebook end-to-end**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/alignment_demo.ipynb
```
Expected: no execution errors; three figures produced.

- [ ] **Step 4: Commit**

```bash
git add notebooks/alignment_demo.ipynb
git commit -m "docs(notebook): use Handler.plot in alignment_demo"
```

---

## Self-Review

- ✅ Spec coverage: every bullet from the conversation (delete kinetics + visualize, new `overlay` Literal, viridis vs tab:blue rule, `ax_size`, `share_y`, `save`, `plot_windows` taking `PeakAnnotation` list, no bounds drawn, clip via `xlim`) maps to a task above.
- ✅ Placeholder scan: no `TODO` / `TBD` / "add error handling" — all code is shown.
- ✅ Type consistency: `OverlayMode` is used in `plotting.py`; `Handler.plot` and `Handler.plot_windows` re-spell the literal directly to keep the handler module independent of `chromhandler.plotting` at import time. `_line_colors` and `_group_chromatograms` signatures match call sites. `save` is `Path | str | None` everywhere.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-16-handler-plot-simplification.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
