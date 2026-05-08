# Foundations Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the basic data-preparation pipeline (annotations, padding, dt, baseline, noise, prepared dataset) as small focused modules with fresh tests, replacing convoluted equivalents on `fix-fit`.

**Architecture:** Five single-responsibility modules under `chromhandler/`. Pure-Python, NumPy-based. No NumPyro, no JAX in this layer. The skew-normal model rewrite (priors / model / posterior) lives in a separate later plan; this layer is its prerequisite. The current `chromhandler/fitting/fitter.py`, `model.py`, `priors.py` and their tests **will break** during this rewrite — that is acceptable per the user's directive and will be fixed in the follow-up plan.

**Tech Stack:** Python 3.11+, NumPy, pydantic v2, pytest, scipy (stats helpers only), uv, ruff, pyright.

---

## Important context for the executing engineer

- **Working directory:** `/Users/max/code/Chromhandler` (on branch `fix-fit`).
- **Quality gate after every file edit:** `uv run ruff check <file>` and `uv run pyright <file>` must pass with zero issues. Tests added in a step must pass before moving on.
- **Real data fixtures live at:** `tests/fixtures/asm_kinetic_series/` — 7 ASM JSON files representing a kinetic series. Used by smoke tests for end-to-end validation.
- **Spec:** see `docs/superpowers/specs/2026-05-07-skew-normal-fitter-rewrite-design.md` on branch `claude/hardcore-vaughan-c92ade` (worktree at `.claude/worktrees/hardcore-vaughan-c92ade/`). The spec is the canonical design record; this plan implements its prerequisites.
- **Branches that exist:**
  - `fix-fit` — current branch, has older fitting scaffolding that we are replacing piece by piece.
  - `claude/hardcore-vaughan-c92ade` — has the spec + brainstorm artefacts; do not commit code here.
- **Stash:** `stash@{0}: On fix-fit: WIP fix-fit before foundations rewrite (2026-05-08)` contains uncommitted edits to `fitter.py`, `model.py`, `utils.py` and several tests. Do **not** apply this stash during foundations work — it predates the rewrite and would re-introduce the old patterns.
- **Commits should be small and frequent.** Each task ends with a commit. Use `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>` in commit messages.

## File structure created by this plan

```
chromhandler/
    annotations.py                          # MODIFY: add overlap-validation function
    fitting/
        preprocessing.py                    # NEW
        baseline.py                         # REWRITE (replace existing)
        noise.py                            # NEW
        prepared_dataset.py                 # NEW

tests/
    unit/fitting/
        conftest.py                         # MODIFY: add foundations fixtures
        test_annotations.py                 # NEW
        test_preprocessing.py               # NEW
        test_baseline.py                    # REWRITE (replace existing)
        test_noise.py                       # NEW (replaces test_noise_plumbing.py)
        test_prepared_dataset.py            # NEW
    integration/
        test_foundations_asm.py             # NEW: real-data smoke test
```

Modules each have one job:
- `annotations.py` — user-facing window dataclasses + cross-annotation validation.
- `preprocessing.py` — turn variable-length raw signals into rectangular `[n_trace, n_time]` arrays + dt.
- `baseline.py` — per-trace OLS line through baseline regions only.
- `noise.py` — per-trace residual std from baseline regions.
- `prepared_dataset.py` — immutable bundle that downstream priors/model layer will consume.

---

## Task 1: Commit the existing `n_components` computed field

The `PeakAnnotation.n_components` computed field has already been added to `chromhandler/annotations.py` in the working tree (uncommitted). Add a test, then commit both.

**Files:**
- Create: `tests/unit/fitting/test_annotations.py`
- Modify: already-modified `chromhandler/annotations.py` (uncommitted)

- [ ] **Step 1.1: Verify the working-tree state**

Run: `cd /Users/max/code/Chromhandler && git status --short chromhandler/annotations.py`
Expected: ` M chromhandler/annotations.py`

If clean (nothing modified), the change was lost — re-apply it as shown in this step. The expected diff adds `computed_field` to the `pydantic` import and adds this property block to `PeakAnnotation`:

```python
    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_components(self) -> int:
        """Number of skew-normal components implied by ``mode``.

        Returns:
            1 for ``"single"``, 2 for ``"artefact_doublet"`` or ``"free_doublet"``.
        """
        return 1 if self.mode == "single" else 2
```

- [ ] **Step 1.2: Write the test**

Create `tests/unit/fitting/test_annotations.py`:

```python
"""Tests for chromhandler.annotations."""

from __future__ import annotations

import pytest

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation


class TestNComponents:
    """The ``n_components`` computed field on PeakAnnotation."""

    def test_single_returns_one(self) -> None:
        ann = PeakAnnotation(molecule_id="x", rt_min=1.0, rt_max=2.0, mode="single")
        assert ann.n_components == 1

    def test_artefact_doublet_returns_two(self) -> None:
        ann = PeakAnnotation(
            molecule_id="x",
            rt_min=1.0,
            rt_max=2.0,
            mode="artefact_doublet",
            artefact_side="right",
        )
        assert ann.n_components == 2

    def test_free_doublet_returns_two(self) -> None:
        ann = PeakAnnotation(
            molecule_id="x", rt_min=1.0, rt_max=2.0, mode="free_doublet"
        )
        assert ann.n_components == 2

    def test_appears_in_serialization(self) -> None:
        ann = PeakAnnotation(molecule_id="x", rt_min=1.0, rt_max=2.0)
        dumped = ann.model_dump()
        assert dumped["n_components"] == 1

    def test_is_read_only(self) -> None:
        ann = PeakAnnotation(molecule_id="x", rt_min=1.0, rt_max=2.0)
        with pytest.raises(ValueError):
            ann.n_components = 2  # type: ignore[misc]
```

- [ ] **Step 1.3: Run the test (must pass — implementation is already in place)**

Run: `cd /Users/max/code/Chromhandler && uv run pytest tests/unit/fitting/test_annotations.py -v`
Expected: 5 passed.

- [ ] **Step 1.4: Quality gates**

Run:
```
cd /Users/max/code/Chromhandler && uv run ruff check chromhandler/annotations.py tests/unit/fitting/test_annotations.py && uv run pyright chromhandler/annotations.py tests/unit/fitting/test_annotations.py
```
Expected: All checks passed; 0 errors.

- [ ] **Step 1.5: Commit**

```
cd /Users/max/code/Chromhandler && git add chromhandler/annotations.py tests/unit/fitting/test_annotations.py && git commit -m "$(cat <<'EOF'
Add n_components computed field to PeakAnnotation

Derives the number of skew-normal components from the mode:
1 for "single", 2 for "artefact_doublet" / "free_doublet". Read-only,
included in model_dump output. Lets downstream priors/model code branch
on n_components without inspecting the mode string.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add baseline-peak overlap validation

Free function in `chromhandler/annotations.py`. Errors only on baseline-peak overlap; peak-peak and baseline-baseline overlaps are explicitly allowed and not checked. Touching boundaries (`a.rt_max == b.rt_min`) are not overlap.

**Files:**
- Modify: `chromhandler/annotations.py` (add function at end)
- Modify: `tests/unit/fitting/test_annotations.py` (add test class)

- [ ] **Step 2.1: Write the failing tests**

Append to `tests/unit/fitting/test_annotations.py`:

```python
class TestBaselinePeakDisjoint:
    """The ``check_baseline_peak_disjoint`` validator."""

    def test_disjoint_passes(self) -> None:
        from chromhandler.annotations import check_baseline_peak_disjoint

        peaks = [PeakAnnotation(molecule_id="x", rt_min=2.0, rt_max=3.0)]
        baselines = [BaselineAnnotation(rt_min=0.5, rt_max=1.5)]
        check_baseline_peak_disjoint(peaks, baselines)  # no error

    def test_touching_boundary_passes(self) -> None:
        from chromhandler.annotations import check_baseline_peak_disjoint

        peaks = [PeakAnnotation(molecule_id="x", rt_min=2.0, rt_max=3.0)]
        baselines = [BaselineAnnotation(rt_min=1.5, rt_max=2.0)]
        check_baseline_peak_disjoint(peaks, baselines)  # no error

    def test_overlap_raises(self) -> None:
        from chromhandler.annotations import check_baseline_peak_disjoint

        peaks = [PeakAnnotation(molecule_id="x", rt_min=2.0, rt_max=3.0)]
        baselines = [BaselineAnnotation(rt_min=2.5, rt_max=3.5)]
        with pytest.raises(ValueError, match="overlaps peak"):
            check_baseline_peak_disjoint(peaks, baselines)

    def test_peak_inside_baseline_raises(self) -> None:
        from chromhandler.annotations import check_baseline_peak_disjoint

        peaks = [PeakAnnotation(molecule_id="x", rt_min=2.0, rt_max=3.0)]
        baselines = [BaselineAnnotation(rt_min=1.0, rt_max=4.0)]
        with pytest.raises(ValueError, match="overlaps peak"):
            check_baseline_peak_disjoint(peaks, baselines)

    def test_peak_peak_overlap_allowed(self) -> None:
        from chromhandler.annotations import check_baseline_peak_disjoint

        peaks = [
            PeakAnnotation(molecule_id="x", rt_min=2.0, rt_max=3.0),
            PeakAnnotation(molecule_id="y", rt_min=2.5, rt_max=3.5),
        ]
        baselines: list[BaselineAnnotation] = []
        check_baseline_peak_disjoint(peaks, baselines)  # explicit policy: allowed

    def test_baseline_baseline_overlap_allowed(self) -> None:
        from chromhandler.annotations import check_baseline_peak_disjoint

        peaks: list[PeakAnnotation] = []
        baselines = [
            BaselineAnnotation(rt_min=0.5, rt_max=1.5),
            BaselineAnnotation(rt_min=1.0, rt_max=2.0),
        ]
        check_baseline_peak_disjoint(peaks, baselines)  # explicit policy: allowed

    def test_empty_inputs_pass(self) -> None:
        from chromhandler.annotations import check_baseline_peak_disjoint

        check_baseline_peak_disjoint([], [])  # no error
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `cd /Users/max/code/Chromhandler && uv run pytest tests/unit/fitting/test_annotations.py::TestBaselinePeakDisjoint -v`
Expected: 7 failures, all `ImportError: cannot import name 'check_baseline_peak_disjoint'`.

- [ ] **Step 2.3: Implement**

Append to `chromhandler/annotations.py` (after the `BaselineAnnotation` class):

```python
def check_baseline_peak_disjoint(
    peaks: list[PeakAnnotation],
    baselines: list[BaselineAnnotation],
) -> None:
    """Raise ``ValueError`` if any baseline window overlaps any peak window.

    Peak-peak and baseline-baseline overlaps are explicitly allowed and not
    checked. Touching boundaries (``a.rt_max == b.rt_min``) are not overlap.

    Args:
        peaks: Peak annotations to validate.
        baselines: Baseline annotations to validate.

    Raises:
        ValueError: If a baseline window overlaps any peak window.
    """
    for b in baselines:
        for p in peaks:
            if b.rt_min < p.rt_max and p.rt_min < b.rt_max:
                raise ValueError(
                    f"Baseline window [{b.rt_min}, {b.rt_max}] overlaps peak "
                    f"window [{p.rt_min}, {p.rt_max}] for molecule "
                    f"{p.molecule_id!r}. Baseline regions must be peak-free."
                )
```

- [ ] **Step 2.4: Run tests to verify they pass**

Run: `cd /Users/max/code/Chromhandler && uv run pytest tests/unit/fitting/test_annotations.py -v`
Expected: 12 passed (5 from Task 1 + 7 new).

- [ ] **Step 2.5: Quality gates**

Run:
```
cd /Users/max/code/Chromhandler && uv run ruff check chromhandler/annotations.py tests/unit/fitting/test_annotations.py && uv run pyright chromhandler/annotations.py tests/unit/fitting/test_annotations.py
```
Expected: All checks passed; 0 errors.

- [ ] **Step 2.6: Commit**

```
cd /Users/max/code/Chromhandler && git add chromhandler/annotations.py tests/unit/fitting/test_annotations.py && git commit -m "$(cat <<'EOF'
Add check_baseline_peak_disjoint validator

Errors only when a baseline window overlaps any peak window. Peak-peak
and baseline-baseline overlaps are explicitly allowed. Touching
boundaries are not overlap. Lives next to the dataclasses for discovery.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `preprocessing.py` — pad to common axis + dt computation

Two pure functions: pad variable-length signal/time arrays to `[n_trace, n_time]` with NaN, and compute median per-trace and global `dt`. NaN-aware throughout.

**Files:**
- Create: `chromhandler/fitting/preprocessing.py`
- Create: `tests/unit/fitting/test_preprocessing.py`

- [ ] **Step 3.1: Write the failing tests**

Create `tests/unit/fitting/test_preprocessing.py`:

```python
"""Tests for chromhandler.fitting.preprocessing."""

from __future__ import annotations

import numpy as np
import pytest


class TestPadToCommonAxis:
    """Padding variable-length traces to a rectangular array."""

    def test_equal_lengths_no_padding(self) -> None:
        from chromhandler.fitting.preprocessing import pad_to_common_axis

        times = [np.array([0.0, 0.1, 0.2]), np.array([0.0, 0.1, 0.2])]
        signals = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
        t, s = pad_to_common_axis(times, signals)
        assert t.shape == (2, 3)
        assert s.shape == (2, 3)
        np.testing.assert_array_equal(s[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(s[1], [4.0, 5.0, 6.0])

    def test_short_trace_padded_with_nan(self) -> None:
        from chromhandler.fitting.preprocessing import pad_to_common_axis

        times = [np.array([0.0, 0.1, 0.2, 0.3]), np.array([0.0, 0.1])]
        signals = [np.array([1.0, 2.0, 3.0, 4.0]), np.array([5.0, 6.0])]
        t, s = pad_to_common_axis(times, signals)
        assert t.shape == (2, 4)
        assert s.shape == (2, 4)
        assert np.isnan(t[1, 2:]).all()
        assert np.isnan(s[1, 2:]).all()
        np.testing.assert_array_equal(s[1, :2], [5.0, 6.0])

    def test_mismatched_time_signal_lengths_raises(self) -> None:
        from chromhandler.fitting.preprocessing import pad_to_common_axis

        times = [np.array([0.0, 0.1])]
        signals = [np.array([1.0, 2.0, 3.0])]
        with pytest.raises(ValueError, match="length"):
            pad_to_common_axis(times, signals)

    def test_unequal_outer_lengths_raises(self) -> None:
        from chromhandler.fitting.preprocessing import pad_to_common_axis

        times = [np.array([0.0, 0.1])]
        signals: list[np.ndarray] = []
        with pytest.raises(ValueError, match="same number"):
            pad_to_common_axis(times, signals)


class TestComputeDtPerTrace:
    """Median sampling interval per trace."""

    def test_uniform_grid(self) -> None:
        from chromhandler.fitting.preprocessing import compute_dt_per_trace

        time = np.array([[0.0, 0.1, 0.2, 0.3], [0.0, 0.1, 0.2, 0.3]])
        dt = compute_dt_per_trace(time)
        np.testing.assert_allclose(dt, [0.1, 0.1])

    def test_nan_padding_ignored(self) -> None:
        from chromhandler.fitting.preprocessing import compute_dt_per_trace

        time = np.array([[0.0, 0.1, 0.2, 0.3], [0.0, 0.1, np.nan, np.nan]])
        dt = compute_dt_per_trace(time)
        np.testing.assert_allclose(dt, [0.1, 0.1])

    def test_irregular_grid_uses_median(self) -> None:
        from chromhandler.fitting.preprocessing import compute_dt_per_trace

        time = np.array([[0.0, 0.1, 0.2, 0.3, 0.5]])
        dt = compute_dt_per_trace(time)
        np.testing.assert_allclose(dt, [0.1])  # median of [0.1, 0.1, 0.1, 0.2]


class TestComputeGlobalDt:
    """Global dt = median of per-trace dt values."""

    def test_simple(self) -> None:
        from chromhandler.fitting.preprocessing import compute_global_dt

        assert compute_global_dt(np.array([0.1, 0.1, 0.1])) == 0.1

    def test_uses_median(self) -> None:
        from chromhandler.fitting.preprocessing import compute_global_dt

        assert compute_global_dt(np.array([0.1, 0.1, 0.5])) == 0.1
```

- [ ] **Step 3.2: Run tests to verify failure**

Run: `cd /Users/max/code/Chromhandler && uv run pytest tests/unit/fitting/test_preprocessing.py -v`
Expected: 9 failures with `ModuleNotFoundError: chromhandler.fitting.preprocessing`.

- [ ] **Step 3.3: Implement**

Create `chromhandler/fitting/preprocessing.py`:

```python
"""Preprocessing utilities: variable-length trace padding and dt computation.

Variable-length signal arrays (one per chromatogram) are padded to a
rectangular ``[n_trace, n_time]`` matrix with trailing ``NaN`` values.
``NaN`` is the canonical missing-data marker downstream — likelihood and
prior code mask it out explicitly.
"""

from __future__ import annotations

import numpy as np


def pad_to_common_axis(
    times: list[np.ndarray],
    signals: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Pad variable-length traces to a rectangular array.

    Args:
        times: List of 1-D arrays of length ``n_time_i``, one per trace.
        signals: List of 1-D arrays of matching length, one per trace.

    Returns:
        Tuple ``(time, signal)`` of shape ``[n_trace, max(n_time_i)]``.
        Padding values are ``NaN``.

    Raises:
        ValueError: If ``times`` and ``signals`` have different outer
            lengths, or if any per-trace ``time[i]`` and ``signal[i]``
            differ in length.
    """
    if len(times) != len(signals):
        raise ValueError(
            f"times and signals must have the same number of traces, "
            f"got {len(times)} and {len(signals)}."
        )
    for i, (t, s) in enumerate(zip(times, signals, strict=True)):
        if t.shape != s.shape:
            raise ValueError(
                f"trace {i}: time length {t.shape} != signal length {s.shape}."
            )
    n_trace = len(times)
    if n_trace == 0:
        empty = np.empty((0, 0), dtype=float)
        return empty, empty
    n_max = max(t.shape[0] for t in times)
    time_out = np.full((n_trace, n_max), np.nan, dtype=float)
    signal_out = np.full((n_trace, n_max), np.nan, dtype=float)
    for i, (t, s) in enumerate(zip(times, signals, strict=True)):
        n = t.shape[0]
        time_out[i, :n] = t
        signal_out[i, :n] = s
    return time_out, signal_out


def compute_dt_per_trace(time: np.ndarray) -> np.ndarray:
    """Median sampling interval per trace.

    Args:
        time: ``[n_trace, n_time]`` array; trailing ``NaN`` values
            represent padding.

    Returns:
        ``[n_trace]`` array of median ``dt`` per trace.
    """
    diffs = np.diff(time, axis=1)
    return np.nanmedian(diffs, axis=1)


def compute_global_dt(dt_per_trace: np.ndarray) -> float:
    """Median of per-trace ``dt`` values.

    Args:
        dt_per_trace: ``[n_trace]`` array of per-trace median dt.

    Returns:
        Global median dt as a Python float.
    """
    return float(np.median(dt_per_trace))
```

- [ ] **Step 3.4: Run tests to verify they pass**

Run: `cd /Users/max/code/Chromhandler && uv run pytest tests/unit/fitting/test_preprocessing.py -v`
Expected: 9 passed.

- [ ] **Step 3.5: Quality gates**

Run:
```
cd /Users/max/code/Chromhandler && uv run ruff check chromhandler/fitting/preprocessing.py tests/unit/fitting/test_preprocessing.py && uv run pyright chromhandler/fitting/preprocessing.py tests/unit/fitting/test_preprocessing.py
```
Expected: All checks passed; 0 errors.

- [ ] **Step 3.6: Commit**

```
cd /Users/max/code/Chromhandler && git add chromhandler/fitting/preprocessing.py tests/unit/fitting/test_preprocessing.py && git commit -m "$(cat <<'EOF'
Add preprocessing module: pad-to-common-axis + dt computation

pad_to_common_axis pads variable-length time/signal arrays to a
rectangular [n_trace, n_time] matrix with NaN. compute_dt_per_trace
and compute_global_dt return median sampling intervals (NaN-aware).
Replaces convoluted bits of the old fitting/utils.py::pad_traces with
a single-purpose, fully tested module.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `baseline.py` rewrite — OLS from baseline regions only

Replace the current `chromhandler/fitting/baseline.py` (which mixes user regions with peak-edge low-point anchors) with a simple, single-purpose OLS fitter that uses **only** the user-annotated baseline regions.

**Files:**
- Modify (full replacement): `chromhandler/fitting/baseline.py`
- Replace: `tests/unit/fitting/test_baseline.py`

- [ ] **Step 4.1: Write the failing tests**

Replace the entire contents of `tests/unit/fitting/test_baseline.py` with:

```python
"""Tests for chromhandler.fitting.baseline."""

from __future__ import annotations

import numpy as np
import pytest

from chromhandler.annotations import BaselineAnnotation


class TestEstimateBaselines:
    """Per-trace OLS baseline estimation from user-annotated regions."""

    def test_constant_baseline(self) -> None:
        from chromhandler.fitting.baseline import estimate_baselines

        time = np.linspace(0.0, 5.0, 501).reshape(1, -1)
        signal = np.full_like(time, 7.5)
        regions = [BaselineAnnotation(rt_min=0.5, rt_max=1.0)]
        intercept, slope = estimate_baselines(time, signal, regions)
        np.testing.assert_allclose(intercept, [7.5], atol=1e-9)
        np.testing.assert_allclose(slope, [0.0], atol=1e-9)

    def test_linear_baseline_recovered(self) -> None:
        from chromhandler.fitting.baseline import estimate_baselines

        time = np.linspace(0.0, 5.0, 501).reshape(1, -1)
        true_intercept, true_slope = 2.0, 0.3
        signal = true_intercept + true_slope * time
        regions = [
            BaselineAnnotation(rt_min=0.5, rt_max=1.0),
            BaselineAnnotation(rt_min=4.0, rt_max=4.5),
        ]
        intercept, slope = estimate_baselines(time, signal, regions)
        np.testing.assert_allclose(intercept, [true_intercept], atol=1e-6)
        np.testing.assert_allclose(slope, [true_slope], atol=1e-6)

    def test_per_trace_independence(self) -> None:
        from chromhandler.fitting.baseline import estimate_baselines

        rng = np.random.default_rng(0)
        time = np.tile(np.linspace(0.0, 5.0, 501), (3, 1))
        true_intercepts = np.array([1.0, 2.0, 3.0])
        true_slopes = np.array([0.0, 0.1, -0.2])
        signal = (
            true_intercepts[:, None]
            + true_slopes[:, None] * time
            + 0.01 * rng.standard_normal(time.shape)
        )
        regions = [
            BaselineAnnotation(rt_min=0.5, rt_max=1.0),
            BaselineAnnotation(rt_min=4.0, rt_max=4.5),
        ]
        intercept, slope = estimate_baselines(time, signal, regions)
        np.testing.assert_allclose(intercept, true_intercepts, atol=0.05)
        np.testing.assert_allclose(slope, true_slopes, atol=0.02)

    def test_nan_padded_trace_handled(self) -> None:
        from chromhandler.fitting.baseline import estimate_baselines

        time = np.full((1, 600), np.nan)
        signal = np.full((1, 600), np.nan)
        time[0, :501] = np.linspace(0.0, 5.0, 501)
        signal[0, :501] = 1.5  # constant baseline
        regions = [BaselineAnnotation(rt_min=0.5, rt_max=1.0)]
        intercept, slope = estimate_baselines(time, signal, regions)
        np.testing.assert_allclose(intercept, [1.5], atol=1e-9)
        np.testing.assert_allclose(slope, [0.0], atol=1e-9)

    def test_too_few_baseline_points_raises(self) -> None:
        from chromhandler.fitting.baseline import estimate_baselines

        time = np.linspace(0.0, 5.0, 11).reshape(1, -1)
        signal = np.zeros_like(time)
        regions = [BaselineAnnotation(rt_min=0.0, rt_max=0.05)]
        with pytest.raises(ValueError, match="too few"):
            estimate_baselines(time, signal, regions)

    def test_no_regions_raises(self) -> None:
        from chromhandler.fitting.baseline import estimate_baselines

        time = np.linspace(0.0, 5.0, 501).reshape(1, -1)
        signal = np.zeros_like(time)
        with pytest.raises(ValueError, match="at least one"):
            estimate_baselines(time, signal, [])
```

- [ ] **Step 4.2: Run tests to verify failure**

Run: `cd /Users/max/code/Chromhandler && uv run pytest tests/unit/fitting/test_baseline.py -v`
Expected: 6 failures (the new tests fail; old `baseline.py` API mismatches).

- [ ] **Step 4.3: Replace baseline.py with the new implementation**

Replace the entire contents of `chromhandler/fitting/baseline.py` with:

```python
"""Per-trace baseline estimation from user-annotated regions only.

Fits ``baseline(t) = intercept + slope * t`` per trace via ordinary least
squares on the points lying inside any user-supplied
:class:`~chromhandler.annotations.BaselineAnnotation` window. Peak-edge
low-point anchors are deliberately not used: they pollute the baseline
estimate with peak-tail contributions. The user's annotations are the
single source of truth.
"""

from __future__ import annotations

import numpy as np

from chromhandler.annotations import BaselineAnnotation

_MIN_POINTS_PER_TRACE: int = 2


def baseline_region_mask(
    time: np.ndarray,
    regions: list[BaselineAnnotation],
) -> np.ndarray:
    """Boolean mask of points lying inside any baseline region.

    Public helper used by ``estimate_baselines`` and by
    :func:`chromhandler.fitting.noise.estimate_noise_per_trace`.

    Args:
        time: ``[n_trace, n_time]`` time array (NaN-padded allowed).
        regions: Baseline annotations.

    Returns:
        ``[n_trace, n_time]`` bool array; True iff that ``(trace, time)``
        sample is inside any region (and the time value is not NaN).
    """
    valid = ~np.isnan(time)
    inside = np.zeros_like(time, dtype=bool)
    for r in regions:
        inside |= (time >= r.rt_min) & (time <= r.rt_max)
    return inside & valid


def estimate_baselines(
    time: np.ndarray,
    signal: np.ndarray,
    regions: list[BaselineAnnotation],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-trace OLS baseline through the user-annotated regions.

    Args:
        time: ``[n_trace, n_time]`` time array (NaN-padded allowed).
        signal: ``[n_trace, n_time]`` signal array.
        regions: At least one baseline annotation. Multiple regions are
            unioned.

    Returns:
        Tuple ``(intercept, slope)`` of ``[n_trace]`` arrays.

    Raises:
        ValueError: If ``regions`` is empty, or if any trace has fewer
            than 2 points inside the unioned baseline region (cannot fit
            a line).
    """
    if not regions:
        raise ValueError("estimate_baselines requires at least one BaselineAnnotation.")
    mask = baseline_region_mask(time, regions)
    n_trace = time.shape[0]
    intercept = np.zeros(n_trace, dtype=float)
    slope = np.zeros(n_trace, dtype=float)
    for i in range(n_trace):
        idx = np.flatnonzero(mask[i])
        if idx.size < _MIN_POINTS_PER_TRACE:
            raise ValueError(
                f"Trace {i}: too few baseline points ({idx.size}) inside the "
                f"annotated regions; need at least {_MIN_POINTS_PER_TRACE}."
            )
        t_anchor = time[i, idx]
        s_anchor = signal[i, idx]
        slope_i, intercept_i = np.polyfit(t_anchor, s_anchor, 1)
        slope[i] = slope_i
        intercept[i] = intercept_i
    return intercept, slope
```

- [ ] **Step 4.4: Run tests to verify they pass**

Run: `cd /Users/max/code/Chromhandler && uv run pytest tests/unit/fitting/test_baseline.py -v`
Expected: 6 passed.

- [ ] **Step 4.5: Quality gates**

Run:
```
cd /Users/max/code/Chromhandler && uv run ruff check chromhandler/fitting/baseline.py tests/unit/fitting/test_baseline.py && uv run pyright chromhandler/fitting/baseline.py tests/unit/fitting/test_baseline.py
```
Expected: All checks passed; 0 errors.

- [ ] **Step 4.6: Commit**

```
cd /Users/max/code/Chromhandler && git add chromhandler/fitting/baseline.py tests/unit/fitting/test_baseline.py && git commit -m "$(cat <<'EOF'
Replace baseline.py with regions-only OLS estimator

The previous implementation mixed user-annotated baseline regions with
bottom-percentile anchors taken from peak-window edges. Peak edges are
contaminated by peak tails — using them as baseline anchors biases the
fit. The user's baseline annotations are the single source of truth.

New estimate_baselines fits intercept + slope * t per trace via OLS on
points inside the unioned baseline regions only. NaN-padded traces
handled. Errors on empty regions or fewer than 2 points per trace.

Note: downstream callers in fitter.py / priors.py will break — those
will be repaired in the follow-up rewrite plan.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `noise.py` — per-trace noise from baseline-region residuals

Estimate **pure noise** (random variation) per trace from the residuals of the OLS baseline within the annotated baseline regions. We deliberately do **not** absorb model misfit into the noise scale; that would mask model-shape problems.

**Files:**
- Create: `chromhandler/fitting/noise.py`
- Create: `tests/unit/fitting/test_noise.py`

- [ ] **Step 5.1: Write the failing tests**

Create `tests/unit/fitting/test_noise.py`:

```python
"""Tests for chromhandler.fitting.noise."""

from __future__ import annotations

import numpy as np
import pytest

from chromhandler.annotations import BaselineAnnotation


class TestEstimateNoisePerTrace:
    """MAD-based per-trace noise std from baseline residuals."""

    def test_recovers_known_noise(self) -> None:
        from chromhandler.fitting.noise import estimate_noise_per_trace

        rng = np.random.default_rng(0)
        time = np.linspace(0.0, 5.0, 5001).reshape(1, -1)
        true_sigma = 0.05
        signal = 1.0 + 0.2 * time + true_sigma * rng.standard_normal(time.shape)
        regions = [
            BaselineAnnotation(rt_min=0.5, rt_max=1.0),
            BaselineAnnotation(rt_min=4.0, rt_max=4.5),
        ]
        intercept = np.array([1.0])
        slope = np.array([0.2])
        noise = estimate_noise_per_trace(time, signal, regions, intercept, slope)
        assert noise.shape == (1,)
        np.testing.assert_allclose(noise, [true_sigma], rtol=0.10)

    def test_per_trace_independence(self) -> None:
        from chromhandler.fitting.noise import estimate_noise_per_trace

        rng = np.random.default_rng(1)
        time = np.tile(np.linspace(0.0, 5.0, 5001), (3, 1))
        true_sigmas = np.array([0.01, 0.05, 0.20])
        signal = true_sigmas[:, None] * rng.standard_normal(time.shape)
        regions = [BaselineAnnotation(rt_min=0.5, rt_max=4.5)]
        intercept = np.zeros(3)
        slope = np.zeros(3)
        noise = estimate_noise_per_trace(time, signal, regions, intercept, slope)
        np.testing.assert_allclose(noise, true_sigmas, rtol=0.10)

    def test_nan_padding_ignored(self) -> None:
        from chromhandler.fitting.noise import estimate_noise_per_trace

        rng = np.random.default_rng(2)
        time = np.full((1, 6000), np.nan)
        signal = np.full((1, 6000), np.nan)
        time[0, :5001] = np.linspace(0.0, 5.0, 5001)
        signal[0, :5001] = 0.05 * rng.standard_normal(5001)
        regions = [BaselineAnnotation(rt_min=0.5, rt_max=4.5)]
        intercept = np.zeros(1)
        slope = np.zeros(1)
        noise = estimate_noise_per_trace(time, signal, regions, intercept, slope)
        np.testing.assert_allclose(noise, [0.05], rtol=0.10)

    def test_robust_to_outliers(self) -> None:
        from chromhandler.fitting.noise import estimate_noise_per_trace

        rng = np.random.default_rng(3)
        time = np.linspace(0.0, 5.0, 5001).reshape(1, -1)
        signal = 0.05 * rng.standard_normal(time.shape)
        # Inject 1% extreme outliers
        signal[0, ::100] += 5.0
        regions = [BaselineAnnotation(rt_min=0.5, rt_max=4.5)]
        intercept = np.zeros(1)
        slope = np.zeros(1)
        noise = estimate_noise_per_trace(time, signal, regions, intercept, slope)
        # MAD-based estimate should still be near 0.05 despite outliers
        np.testing.assert_allclose(noise, [0.05], rtol=0.20)

    def test_no_regions_raises(self) -> None:
        from chromhandler.fitting.noise import estimate_noise_per_trace

        time = np.linspace(0.0, 5.0, 501).reshape(1, -1)
        signal = np.zeros_like(time)
        with pytest.raises(ValueError, match="at least one"):
            estimate_noise_per_trace(
                time, signal, [], np.zeros(1), np.zeros(1)
            )
```

- [ ] **Step 5.2: Run tests to verify failure**

Run: `cd /Users/max/code/Chromhandler && uv run pytest tests/unit/fitting/test_noise.py -v`
Expected: 5 failures with `ModuleNotFoundError: chromhandler.fitting.noise`.

- [ ] **Step 5.3: Implement**

Create `chromhandler/fitting/noise.py`:

```python
"""Per-trace noise estimation from baseline-region residuals.

Estimates pure measurement noise (random variation) by computing the
median absolute deviation of the residuals between observed signal and
the OLS baseline ``intercept + slope * t`` within the user-annotated
baseline regions. Robust to outliers and to small baseline-fit errors.

We deliberately do not estimate "noise" from peak regions or from the
whole trace: that would conflate genuine measurement noise with model
misfit, falsely widening the likelihood and masking model-shape issues
downstream.
"""

from __future__ import annotations

import numpy as np

from chromhandler.annotations import BaselineAnnotation
from chromhandler.fitting.baseline import baseline_region_mask

_MAD_TO_STD: float = 1.4826  # consistent estimator of std under Gaussian noise


def estimate_noise_per_trace(
    time: np.ndarray,
    signal: np.ndarray,
    regions: list[BaselineAnnotation],
    baseline_intercept: np.ndarray,
    baseline_slope: np.ndarray,
) -> np.ndarray:
    """MAD-based per-trace noise std from baseline-region residuals.

    Args:
        time: ``[n_trace, n_time]`` time array (NaN-padded allowed).
        signal: ``[n_trace, n_time]`` signal array.
        regions: At least one baseline annotation.
        baseline_intercept: ``[n_trace]`` per-trace OLS intercept.
        baseline_slope: ``[n_trace]`` per-trace OLS slope.

    Returns:
        ``[n_trace]`` array of noise std estimates.

    Raises:
        ValueError: If ``regions`` is empty.
    """
    if not regions:
        raise ValueError(
            "estimate_noise_per_trace requires at least one BaselineAnnotation."
        )
    mask = baseline_region_mask(time, regions)
    predicted = baseline_intercept[:, None] + baseline_slope[:, None] * time
    residual = signal - predicted
    masked = np.where(mask, residual, np.nan)
    mad = np.nanmedian(np.abs(masked), axis=1)
    return _MAD_TO_STD * mad
```

- [ ] **Step 5.4: Run tests to verify they pass**

Run: `cd /Users/max/code/Chromhandler && uv run pytest tests/unit/fitting/test_noise.py -v`
Expected: 5 passed.

- [ ] **Step 5.5: Quality gates**

Run:
```
cd /Users/max/code/Chromhandler && uv run ruff check chromhandler/fitting/noise.py tests/unit/fitting/test_noise.py && uv run pyright chromhandler/fitting/noise.py tests/unit/fitting/test_noise.py
```
Expected: All checks passed; 0 errors.

- [ ] **Step 5.6: Commit**

```
cd /Users/max/code/Chromhandler && git add chromhandler/fitting/noise.py tests/unit/fitting/test_noise.py && git commit -m "$(cat <<'EOF'
Add noise.py: per-trace MAD-based noise from baseline residuals

Estimates pure measurement noise from residuals of the OLS baseline
inside the user-annotated baseline regions. MAD * 1.4826 (consistent
Gaussian std) — robust to outliers. Per-trace independent.

Deliberately scoped to baseline regions only: noise estimated from peak
regions or the whole trace would conflate measurement noise with model
misfit, widening the likelihood falsely.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `prepared_dataset.py` — immutable bundle + orchestrator

A frozen dataclass that bundles all foundations outputs into one object that downstream priors/model code will consume. Plus a top-level orchestrator function `prepare_dataset` that runs the full pipeline.

**Files:**
- Create: `chromhandler/fitting/prepared_dataset.py`
- Create: `tests/unit/fitting/test_prepared_dataset.py`

- [ ] **Step 6.1: Write the failing tests**

Create `tests/unit/fitting/test_prepared_dataset.py`:

```python
"""Tests for chromhandler.fitting.prepared_dataset."""

from __future__ import annotations

import numpy as np
import pytest

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation


class TestPreparedDatasetConstruction:
    """The PreparedDataset frozen dataclass."""

    def test_fields_present(self) -> None:
        from chromhandler.fitting.prepared_dataset import PreparedDataset

        time = np.zeros((2, 10))
        signal = np.zeros((2, 10))
        ds = PreparedDataset(
            time=time,
            signal=signal,
            valid_mask=np.ones((2, 10), dtype=bool),
            dt_per_trace=np.full(2, 0.1),
            dt_global=0.1,
            n_trace=2,
            peak_annotations=[],
            baseline_annotations=[],
            baseline_intercept=np.zeros(2),
            baseline_slope=np.zeros(2),
            noise_per_trace=np.full(2, 0.01),
        )
        assert ds.n_trace == 2
        assert ds.dt_global == 0.1
        np.testing.assert_array_equal(ds.time, time)

    def test_is_frozen(self) -> None:
        from chromhandler.fitting.prepared_dataset import PreparedDataset

        ds = PreparedDataset(
            time=np.zeros((1, 5)),
            signal=np.zeros((1, 5)),
            valid_mask=np.ones((1, 5), dtype=bool),
            dt_per_trace=np.full(1, 0.1),
            dt_global=0.1,
            n_trace=1,
            peak_annotations=[],
            baseline_annotations=[],
            baseline_intercept=np.zeros(1),
            baseline_slope=np.zeros(1),
            noise_per_trace=np.full(1, 0.01),
        )
        with pytest.raises(Exception):
            ds.n_trace = 99  # type: ignore[misc]


class TestPrepareDataset:
    """End-to-end orchestrator."""

    def test_simple_pipeline_runs(self) -> None:
        from chromhandler.fitting.prepared_dataset import prepare_dataset

        rng = np.random.default_rng(0)
        time_grid = np.linspace(0.0, 5.0, 501)
        true_sigma = 0.02
        baseline = 1.0 + 0.1 * time_grid
        signals = [
            baseline + true_sigma * rng.standard_normal(time_grid.size)
            for _ in range(3)
        ]
        times = [time_grid for _ in range(3)]
        peaks = [PeakAnnotation(molecule_id="x", rt_min=2.0, rt_max=3.0)]
        baselines = [
            BaselineAnnotation(rt_min=0.5, rt_max=1.5),
            BaselineAnnotation(rt_min=3.5, rt_max=4.5),
        ]

        ds = prepare_dataset(times, signals, peaks, baselines)

        assert ds.n_trace == 3
        assert ds.time.shape == (3, 501)
        assert ds.signal.shape == (3, 501)
        np.testing.assert_allclose(ds.dt_global, 0.01, rtol=1e-3)
        np.testing.assert_allclose(ds.baseline_intercept, [1.0] * 3, atol=0.05)
        np.testing.assert_allclose(ds.baseline_slope, [0.1] * 3, atol=0.05)
        np.testing.assert_allclose(ds.noise_per_trace, [true_sigma] * 3, rtol=0.20)
        assert ds.peak_annotations == peaks
        assert ds.baseline_annotations == baselines

    def test_baseline_in_peak_window_raises(self) -> None:
        from chromhandler.fitting.prepared_dataset import prepare_dataset

        time_grid = np.linspace(0.0, 5.0, 501)
        signals = [np.ones_like(time_grid)]
        times = [time_grid]
        peaks = [PeakAnnotation(molecule_id="x", rt_min=2.0, rt_max=3.0)]
        baselines = [BaselineAnnotation(rt_min=2.5, rt_max=3.5)]

        with pytest.raises(ValueError, match="overlaps peak"):
            prepare_dataset(times, signals, peaks, baselines)

    def test_variable_length_traces_padded(self) -> None:
        from chromhandler.fitting.prepared_dataset import prepare_dataset

        rng = np.random.default_rng(0)
        long_t = np.linspace(0.0, 5.0, 501)
        short_t = np.linspace(0.0, 4.0, 401)
        long_s = 1.0 + 0.01 * rng.standard_normal(long_t.size)
        short_s = 1.0 + 0.01 * rng.standard_normal(short_t.size)
        times = [long_t, short_t]
        signals = [long_s, short_s]
        peaks = [PeakAnnotation(molecule_id="x", rt_min=2.0, rt_max=3.0)]
        baselines = [BaselineAnnotation(rt_min=0.5, rt_max=1.5)]

        ds = prepare_dataset(times, signals, peaks, baselines)

        assert ds.time.shape == (2, 501)
        assert np.isnan(ds.signal[1, 401:]).all()
        np.testing.assert_array_equal(ds.valid_mask[1, 401:], False)
```

- [ ] **Step 6.2: Run tests to verify failure**

Run: `cd /Users/max/code/Chromhandler && uv run pytest tests/unit/fitting/test_prepared_dataset.py -v`
Expected: 5 failures with `ModuleNotFoundError: chromhandler.fitting.prepared_dataset`.

- [ ] **Step 6.3: Implement**

Create `chromhandler/fitting/prepared_dataset.py`:

```python
"""Immutable bundle of all foundations outputs and the top-level orchestrator.

``PreparedDataset`` is the canonical input to the priors/model layer. It
contains everything that data preparation produces: padded time/signal
arrays, a validity mask, per-trace and global dt, the user's annotations,
per-trace baseline parameters, and per-trace noise std.

``prepare_dataset`` runs the full preparation pipeline end-to-end:
overlap validation → padding → dt → baseline OLS → noise → bundle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chromhandler.annotations import (
    BaselineAnnotation,
    PeakAnnotation,
    check_baseline_peak_disjoint,
)
from chromhandler.fitting.baseline import estimate_baselines
from chromhandler.fitting.noise import estimate_noise_per_trace
from chromhandler.fitting.preprocessing import (
    compute_dt_per_trace,
    compute_global_dt,
    pad_to_common_axis,
)


@dataclass(frozen=True)
class PreparedDataset:
    """Canonical input to the priors/model layer.

    Attributes:
        time: ``[n_trace, n_time]`` time array, NaN where padded.
        signal: ``[n_trace, n_time]`` signal array, NaN where padded.
        valid_mask: ``[n_trace, n_time]`` bool, True where signal is real.
        dt_per_trace: ``[n_trace]`` per-trace median sampling interval.
        dt_global: Global median dt.
        n_trace: Number of traces.
        peak_annotations: User peak windows.
        baseline_annotations: User baseline regions.
        baseline_intercept: ``[n_trace]`` per-trace OLS intercept.
        baseline_slope: ``[n_trace]`` per-trace OLS slope.
        noise_per_trace: ``[n_trace]`` MAD-based noise std.
    """

    time: np.ndarray
    signal: np.ndarray
    valid_mask: np.ndarray
    dt_per_trace: np.ndarray
    dt_global: float
    n_trace: int
    peak_annotations: list[PeakAnnotation]
    baseline_annotations: list[BaselineAnnotation]
    baseline_intercept: np.ndarray
    baseline_slope: np.ndarray
    noise_per_trace: np.ndarray


def prepare_dataset(
    times: list[np.ndarray],
    signals: list[np.ndarray],
    peak_annotations: list[PeakAnnotation],
    baseline_annotations: list[BaselineAnnotation],
) -> PreparedDataset:
    """Run the full data-preparation pipeline.

    Args:
        times: List of 1-D time arrays, one per trace.
        signals: List of 1-D signal arrays, matching lengths.
        peak_annotations: User peak windows.
        baseline_annotations: User baseline regions.

    Returns:
        :class:`PreparedDataset` with padded arrays, dt, baselines, noise.

    Raises:
        ValueError: If a baseline window overlaps any peak window, or if
            any preparation step fails (see component functions).
    """
    check_baseline_peak_disjoint(peak_annotations, baseline_annotations)
    time, signal = pad_to_common_axis(times, signals)
    valid_mask = ~np.isnan(signal)
    dt_per_trace = compute_dt_per_trace(time)
    dt_global = compute_global_dt(dt_per_trace)
    intercept, slope = estimate_baselines(time, signal, baseline_annotations)
    noise = estimate_noise_per_trace(
        time, signal, baseline_annotations, intercept, slope
    )
    return PreparedDataset(
        time=time,
        signal=signal,
        valid_mask=valid_mask,
        dt_per_trace=dt_per_trace,
        dt_global=dt_global,
        n_trace=len(times),
        peak_annotations=list(peak_annotations),
        baseline_annotations=list(baseline_annotations),
        baseline_intercept=intercept,
        baseline_slope=slope,
        noise_per_trace=noise,
    )
```

- [ ] **Step 6.4: Run tests to verify they pass**

Run: `cd /Users/max/code/Chromhandler && uv run pytest tests/unit/fitting/test_prepared_dataset.py -v`
Expected: 5 passed.

- [ ] **Step 6.5: Quality gates**

Run:
```
cd /Users/max/code/Chromhandler && uv run ruff check chromhandler/fitting/prepared_dataset.py tests/unit/fitting/test_prepared_dataset.py && uv run pyright chromhandler/fitting/prepared_dataset.py tests/unit/fitting/test_prepared_dataset.py
```
Expected: All checks passed; 0 errors.

- [ ] **Step 6.6: Commit**

```
cd /Users/max/code/Chromhandler && git add chromhandler/fitting/prepared_dataset.py tests/unit/fitting/test_prepared_dataset.py && git commit -m "$(cat <<'EOF'
Add PreparedDataset and prepare_dataset orchestrator

PreparedDataset bundles all foundations outputs (padded time/signal,
valid mask, dt, baselines, noise, annotations) into one immutable
object — the canonical input the priors/model layer will consume.

prepare_dataset runs the full pipeline: overlap validation → padding →
dt → baseline OLS → noise → bundle. Single entry point, single source
of truth for preparation order.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Real-data smoke test on the ASM kinetic series

End-to-end integration test: load the 7-trace ASM kinetic series, run `prepare_dataset` with realistic annotations, assert sane outputs. No fitting yet — just verifying the foundations layer survives real data.

**Files:**
- Create: `tests/integration/test_foundations_asm.py`

- [ ] **Step 7.1: Write the test**

Create `tests/integration/test_foundations_asm.py`:

```python
"""Integration test for the foundations layer on real ASM kinetic data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.prepared_dataset import prepare_dataset
from chromhandler.handler import Handler

ASM_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "asm_kinetic_series"


def _times_and_signals_from_handler(handler: Handler) -> tuple[
    list[np.ndarray], list[np.ndarray]
]:
    """Extract per-trace (time, signal) arrays from a Handler.

    Assumes one chromatogram per measurement and uses the first one.
    """
    times: list[np.ndarray] = []
    signals: list[np.ndarray] = []
    for meas in handler.measurements:
        chrom = meas.chromatograms[0]
        times.append(np.asarray(chrom.times, dtype=float))
        signals.append(np.asarray(chrom.signals, dtype=float))
    return times, signals


def test_prepare_dataset_on_asm_kinetic_series() -> None:
    handler = Handler.read_asm(
        path=ASM_DIR,
        ph=7.4,
        temperature=25.0,
        mode="timecourse",
        values=[0.0, 10.0, 20.0, 30.0, 60.0, 120.0, 240.0],
        unit="min",
        silent=True,
    )
    times, signals = _times_and_signals_from_handler(handler)

    assert len(times) == 7
    assert len(signals) == 7

    peaks = [
        PeakAnnotation(molecule_id="Ino", rt_min=2.55, rt_max=2.85),
        PeakAnnotation(
            molecule_id="SIH",
            rt_min=2.85,
            rt_max=3.15,
            mode="artefact_doublet",
            artefact_side="right",
        ),
        PeakAnnotation(
            molecule_id="Hyp",
            rt_min=3.15,
            rt_max=3.48,
            mode="artefact_doublet",
            artefact_side="left",
        ),
    ]
    baselines = [
        BaselineAnnotation(rt_min=2.49, rt_max=2.51),
        BaselineAnnotation(rt_min=3.50, rt_max=3.52),
    ]

    ds = prepare_dataset(times, signals, peaks, baselines)

    assert ds.n_trace == 7
    assert ds.time.shape[0] == 7
    assert ds.dt_global > 0
    assert ds.dt_global < 0.01  # HPLC sampling well below 10 ms
    assert np.all(ds.noise_per_trace > 0)
    assert np.all(np.isfinite(ds.baseline_intercept))
    assert np.all(np.isfinite(ds.baseline_slope))
    assert ds.peak_annotations[0].n_components == 1
    assert ds.peak_annotations[1].n_components == 2
    assert ds.peak_annotations[2].n_components == 2
```

- [ ] **Step 7.2: Run the test**

Run: `cd /Users/max/code/Chromhandler && uv run pytest tests/integration/test_foundations_asm.py -v`
Expected: 1 passed.

If the test fails because the baseline regions (2.49–2.51 and 3.50–3.52) yield fewer than 2 sample points per trace at the actual data resolution, widen them slightly (e.g. to 2.45–2.51 and 3.50–3.55) and rerun. The acceptance criterion is "passes on the real fixture", not "passes with these exact numbers".

- [ ] **Step 7.3: Quality gates**

Run:
```
cd /Users/max/code/Chromhandler && uv run ruff check tests/integration/test_foundations_asm.py && uv run pyright tests/integration/test_foundations_asm.py
```
Expected: All checks passed; 0 errors.

- [ ] **Step 7.4: Commit**

```
cd /Users/max/code/Chromhandler && git add tests/integration/test_foundations_asm.py && git commit -m "$(cat <<'EOF'
Add integration smoke test for foundations on real ASM kinetic series

Loads the 7-trace ASM kinetic fixture, runs prepare_dataset with the
realistic Ino/SIH/Hyp annotations from the user's test script, and
asserts shapes and sanity bounds on dt, noise, baseline parameters.
End-to-end check that the foundations layer survives real data.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Clean up obsolete tests

The previous fitting tests reference modules and APIs that no longer exist or have changed shape. They block CI. Remove the ones whose subjects have been replaced by the foundations rewrite; leave the rest for the follow-up plan.

**Files:**
- Delete: `tests/unit/fitting/test_noise_plumbing.py` (replaced by `test_noise.py`)
- Delete: `tests/unit/fitting/test_data.py` if it tests the old data bundle (review first)

- [ ] **Step 8.1: Inspect and delete obsolete tests**

Run:
```
cd /Users/max/code/Chromhandler && head -30 tests/unit/fitting/test_noise_plumbing.py
```
If the file targets the old whole-trace DER_SNR or fix-fit's `_resolve_trace_sigma_noise`, delete it (those code paths are being replaced):
```
cd /Users/max/code/Chromhandler && git rm tests/unit/fitting/test_noise_plumbing.py
```

Run:
```
cd /Users/max/code/Chromhandler && head -30 tests/unit/fitting/test_data.py
```
If it tests the old `Fitter` data bundle that has no equivalent yet, delete:
```
cd /Users/max/code/Chromhandler && git rm tests/unit/fitting/test_data.py
```
Otherwise leave it.

- [ ] **Step 8.2: Run only foundations tests to confirm green**

Run:
```
cd /Users/max/code/Chromhandler && uv run pytest tests/unit/fitting/test_annotations.py tests/unit/fitting/test_preprocessing.py tests/unit/fitting/test_baseline.py tests/unit/fitting/test_noise.py tests/unit/fitting/test_prepared_dataset.py tests/integration/test_foundations_asm.py -v
```
Expected: all passed (sum of green tests from Tasks 1–7).

- [ ] **Step 8.3: Commit the deletions**

```
cd /Users/max/code/Chromhandler && git commit -m "$(cat <<'EOF'
Remove tests for replaced foundations modules

test_noise_plumbing.py targeted the old whole-trace noise estimator;
test_data.py (if removed) targeted the old Fitter data bundle. Their
replacements live in the new foundations test suite. Other fitting
tests stay broken and will be repaired in the follow-up rewrite plan.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## What this plan does NOT do (deliberately)

- **No skew-normal math.** No `skew_normal.py`, no CP↔DP bijection, no density evaluation. That's the next plan.
- **No NumPyro model rewrite.** The existing `chromhandler/fitting/model.py` and `priors.py` will be left in their broken-by-our-changes state; they are repaired by the follow-up plan.
- **No `Fitter` orchestration changes.** The user-facing `Fitter` class on fix-fit will be broken until the follow-up plan rewires it onto the new foundations layer.
- **No visualization, no posterior, no plot_fit_combined.** Out of scope.
- **No fix to the `chromhandler.handler.Handler` peak-assignment system.** That's a separate concern (instrument-detected peaks, not fitting).

The single deliverable of this plan is: **a clean, tested `prepare_dataset(...)` that returns a `PreparedDataset` ready for the priors/model layer to consume.**

---

## Acceptance for the whole plan

- All foundations tests green: `tests/unit/fitting/test_{annotations,preprocessing,baseline,noise,prepared_dataset}.py` and `tests/integration/test_foundations_asm.py`.
- `uv run ruff check chromhandler/annotations.py chromhandler/fitting/preprocessing.py chromhandler/fitting/baseline.py chromhandler/fitting/noise.py chromhandler/fitting/prepared_dataset.py` clean.
- `uv run pyright` clean for the same set.
- 8 commits on `fix-fit`, each focused and atomic.
- Other (non-foundations) fitting tests may still fail — that is expected and explicitly out of scope.
