# Handler Controls Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-class support for control samples (samples with zero analyte concentration by experimental design) so the fitter can use them to extract direct, principled priors for artefact peaks in `artefact_doublet` annotations.

**Architecture:** A control is a per-sample property — same chromatographic run conditions as experimental samples but with the analyte known absent. We add a boolean `is_control` flag to `Sample`, auto-detect it during `load_initial_conditions` (when *all* listed initial concentrations are zero, the sample is a control), thread the bit through to `PreparedDataset` as `control_trace_indices`, and add a thin `Handler.prepare_dataset()` convenience wrapper that produces a fitter-ready `PreparedDataset` from the handler state in one call.

**Tech Stack:** Python 3.11+, Pydantic v2, NumPy, pandas (used by existing `load_initial_conditions`), pytest, ruff, pyright. All execution via `uv run`.

**Why this plan before the priors plan:** The priors module (`chromhandler/fitting/priors.py`) consumes `dataset.control_trace_indices`. Until that field exists and is populated by the data-prep pipeline, the priors plan can't be executed cleanly. This plan is the small, isolated prerequisite.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `chromhandler/model.py` | Modify (~5 lines) | Add `is_control: bool = False` to `Sample` |
| `chromhandler/handler.py` | Modify (~40 lines) | Auto-detect controls in `load_initial_conditions`; add `prepare_dataset()` convenience method |
| `chromhandler/fitting/prepared_dataset.py` | Modify (~10 lines) | Add `is_control: NDArray[np.bool_]` field; accept optional `is_control` arg in `prepare_dataset()` |
| `tests/unit/handler/test_handler_basics.py` | Extend | Add auto-detection tests for `load_initial_conditions` (the meaningful test) |
| `tests/unit/fitting/test_prepared_dataset.py` | Extend | Add `is_control` propagation tests |
| `tests/unit/handler/test_handler_prepare_dataset.py` | Create | Test the new `Handler.prepare_dataset()` convenience wrapper |

---

## Conventions

- After every file edit, run:
  ```bash
  uv run ruff check <file>
  uv run pyright <file>
  ```
  Both must report zero issues before committing.
- Tests run with `uv run pytest <file> -v`.
- All new public methods get Google-style docstrings (`Args`, `Returns`, `Raises`).
- One commit per task. Commit message format: `feat(handler): <task summary>` for additive features, `feat(prepared_dataset): <summary>` for fitting-layer changes.

---

## Task 1: Add `is_control` field to `Sample`

**Why first:** Every later task assumes this field exists. The behavior here (a defaulted Pydantic field) is Pydantic's own behavior; no test needed at this layer. The meaningful test is the auto-detection logic in Task 2, which exercises the field end-to-end through `load_initial_conditions`. We do run the existing test suite as a smoke check to confirm nothing regresses.

**Files:**
- Modify: `chromhandler/model.py` (the `Sample` class, line 76)

- [ ] **Step 1: Add the field to `Sample`**

In `chromhandler/model.py`, locate the `Sample` class (line 76). After the existing `injection_volume_unit` field (around line 108) and **before** the JSON-LD fields, add:

```python
    is_control: bool = Field(
        default=False,
        description="""Whether this sample is an experimental control
        (e.g., no substrate, no enzyme). Controls are used to extract
        direct priors for artefact peaks in the fitter.""",
    )
```

- [ ] **Step 2: Run the existing suite as a smoke test**

```bash
uv run ruff check chromhandler/model.py
uv run pyright chromhandler/model.py
uv run pytest tests/ -x -q
```
Expected: all existing tests still pass (the new field has a safe default).

- [ ] **Step 3: Commit**

```bash
git add chromhandler/model.py
git commit -m "feat(model): add is_control flag to Sample with default False"
```

---

## Task 2: Auto-detect controls in `load_initial_conditions`

**Why:** Users shouldn't have to remember to mark controls manually. If every listed initial concentration for a sample is zero, that sample is — by experimental design — a control. Detect and set `is_control=True` automatically while loading the conditions CSV.

**Files:**
- Modify: `chromhandler/handler.py` (the `load_initial_conditions` method, line 599)
- Extend: `tests/unit/handler/test_handler_basics.py` (add the control-detection tests alongside any existing `load_initial_conditions` tests)

- [ ] **Step 1: Append the failing tests to `tests/unit/handler/test_handler_basics.py`**

Add these tests at the end of `tests/unit/handler/test_handler_basics.py`. If the file does not already import `pandas` or `Sample`, add those imports at the top:

```python
import pandas as pd

from chromhandler.handler import Handler
from chromhandler.model import Sample


def _handler_with_samples(*sample_ids: str) -> Handler:
    h = Handler()
    h.samples = [Sample(id=sid) for sid in sample_ids]
    return h


def test_sample_with_all_zero_concs_marked_as_control() -> None:
    h = _handler_with_samples("control_1", "treatment_1")
    df = pd.DataFrame(
        {
            "sample_id": ["control_1", "treatment_1"],
            "Substrate": [0.0, 100.0],
            "Enzyme": [0.0, 1.0],
        }
    )
    h.load_initial_conditions(df, conc_unit="umol / l")
    assert h._get_sample("control_1").is_control is True
    assert h._get_sample("treatment_1").is_control is False


def test_partial_zero_concs_not_a_control() -> None:
    """Mixed-zero is not a control — at least one component is present."""
    h = _handler_with_samples("partial_zero")
    df = pd.DataFrame(
        {
            "sample_id": ["partial_zero"],
            "Substrate": [0.0],
            "Enzyme": [1.0],
        }
    )
    h.load_initial_conditions(df, conc_unit="umol / l")
    assert h._get_sample("partial_zero").is_control is False


def test_nan_treated_as_missing_not_zero() -> None:
    """NaN entries don't count toward 'all zero'; missing data ≠ zero conc."""
    h = _handler_with_samples("ambiguous")
    df = pd.DataFrame(
        {
            "sample_id": ["ambiguous"],
            "Substrate": [0.0],
            "Enzyme": [float("nan")],
        }
    )
    h.load_initial_conditions(df, conc_unit="umol / l")
    # Only one declared concentration, and it's zero → control.
    # (NaN means "not specified", not "zero".)
    assert h._get_sample("ambiguous").is_control is True


def test_explicit_is_control_preserved_if_already_true() -> None:
    """If user already set is_control=True, auto-detection doesn't override it."""
    h = _handler_with_samples("manual_control")
    h._get_sample("manual_control").is_control = True
    df = pd.DataFrame(
        {
            "sample_id": ["manual_control"],
            "Substrate": [100.0],  # not a control by concentration, but user said so
        }
    )
    h.load_initial_conditions(df, conc_unit="umol / l")
    assert h._get_sample("manual_control").is_control is True
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/handler/test_handler_basics.py -v -k "control"
```
Expected: all four new tests fail — none of the assertions hold because auto-detection isn't implemented yet.

- [ ] **Step 3: Implement auto-detection in `load_initial_conditions`**

In `chromhandler/handler.py`, modify the `load_initial_conditions` method (line 599). At the end of the per-sample loop (after the existing `if not added_any: raise ValueError(...)` block at line 656–657), add auto-detection. Replace the existing loop body (lines 647–657):

```python
        for i, sample_id in enumerate(sample_ids):
            if sample_id not in existing_ids:
                continue
            added_any = False
            declared_concs: list[float] = []
            for mol_id in df_mol.columns:
                val = df_mol.iloc[i, df_mol.columns.get_loc(mol_id)]
                if not pd.isna(val):  # type: ignore[arg-type]
                    self.add_initial_condition(sample_id, str(mol_id), float(val), conc_unit)  # type: ignore[arg-type]
                    added_any = True
                    declared_concs.append(float(val))  # type: ignore[arg-type]
            if not added_any:
                raise ValueError(f"Sample '{sample_id}' has no initial conditions in the file.")
            # Auto-detect controls: if every declared (non-NaN) concentration is
            # zero, this is an experimental control. Don't override an explicit
            # user-set True (only flip False -> True, never True -> False).
            sample = self._get_sample(sample_id)
            if not sample.is_control and all(c == 0.0 for c in declared_concs):
                sample.is_control = True
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/unit/handler/test_handler_basics.py -v -k "control"
uv run ruff check chromhandler/handler.py tests/unit/handler/test_handler_basics.py
uv run pyright chromhandler/handler.py tests/unit/handler/test_handler_basics.py
```
Expected: 4 new tests pass, ruff clean, pyright clean.

- [ ] **Step 5: Smoke-test the broader suite**

```bash
uv run pytest tests/ -x -q
```
Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add chromhandler/handler.py tests/unit/handler/test_handler_basics.py
git commit -m "feat(handler): auto-detect controls in load_initial_conditions"
```

---

## Task 3: Add `is_control` field to `PreparedDataset`

**Why:** The fitter (`priors.py`) needs to know which traces are controls. Threading this through `PreparedDataset` keeps the fitter independent of `Handler` internals.

**Storage choice — per-trace bool array, not indices.** Every other per-trace property on `PreparedDataset` is stored as an `NDArray` of length `n_trace` (`dt_per_trace`, `baseline_intercept`, `baseline_slope`, `noise_per_trace`). For consistency, controls are stored the same way: `is_control: NDArray[np.bool_]` with shape `[n_trace]`. Code that wants indices can get them with `np.where(ds.is_control)[0]` — one trivial call at the use site, no schema asymmetry.

**Files:**
- Modify: `chromhandler/fitting/prepared_dataset.py`
- Extend: `tests/unit/fitting/test_prepared_dataset.py` (add control-related tests alongside the existing ones)

- [ ] **Step 1: Append the failing tests to `tests/unit/fitting/test_prepared_dataset.py`**

Add at the end of `tests/unit/fitting/test_prepared_dataset.py`. If the file does not already import `numpy as np`, `BaselineAnnotation`, `PeakAnnotation`, or `prepare_dataset`, those imports may already be present — reuse them. The helper `_make_inputs` is local to these tests:

```python
def _make_inputs_with_n_traces(n_trace: int = 3):
    t = np.arange(2.5, 3.6, 0.01)
    times = [t.copy() for _ in range(n_trace)]
    signals = [
        np.full_like(t, 100.0) + 10.0 * np.exp(-((t - 2.8) ** 2) / 0.02)
        for _ in range(n_trace)
    ]
    peak_anns = [PeakAnnotation(molecule_id="A", rt_min=2.6, rt_max=3.0)]
    base_anns = [
        BaselineAnnotation(rt_min=2.55, rt_max=2.58),
        BaselineAnnotation(rt_min=3.50, rt_max=3.55),
    ]
    return times, signals, peak_anns, base_anns


def test_prepared_dataset_is_control_default_all_false() -> None:
    times, signals, peak_anns, base_anns = _make_inputs_with_n_traces(3)
    ds = prepare_dataset(times, signals, peak_anns, base_anns)
    assert ds.is_control.shape == (3,)
    assert ds.is_control.dtype == np.bool_
    assert not ds.is_control.any()


def test_prepared_dataset_is_control_propagates() -> None:
    times, signals, peak_anns, base_anns = _make_inputs_with_n_traces(4)
    ds = prepare_dataset(
        times, signals, peak_anns, base_anns,
        is_control=[False, True, False, True],
    )
    np.testing.assert_array_equal(
        ds.is_control, np.array([False, True, False, True])
    )
    # And the derived indices work:
    np.testing.assert_array_equal(np.where(ds.is_control)[0], np.array([1, 3]))


def test_prepared_dataset_is_control_length_mismatch_raises() -> None:
    times, signals, peak_anns, base_anns = _make_inputs_with_n_traces(3)
    with pytest.raises(ValueError, match="is_control"):
        prepare_dataset(
            times, signals, peak_anns, base_anns,
            is_control=[True, False],  # 2 entries, but 3 traces
        )
```

(If `pytest` isn't already imported in the file, add `import pytest` at the top.)

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_prepared_dataset.py -v -k "is_control"
```
Expected: `TypeError` on the `is_control` kwarg or `AttributeError` on `ds.is_control` — neither exists yet.

- [ ] **Step 3: Implement the field and the argument**

In `chromhandler/fitting/prepared_dataset.py`, add the field to `PreparedDataset` (around line 64, after `noise_per_trace`):

```python
    is_control: NDArray[np.bool_]
```

(Per-trace bool array of shape `[n_trace]`. Place it after `noise_per_trace` to match the existing ordering of per-trace fields. No default value — `prepare_dataset()` always constructs it explicitly.)

Update the docstring's Attributes section to mention:

```
is_control: ``[n_trace]`` bool array, True where the trace comes from a
    control sample (analyte known absent by experimental design). Used by
    the priors layer to extract direct artefact priors.
```

Then update `prepare_dataset()` (line 67):

```python
def prepare_dataset(
    times: list[NDArray[np.float64]],
    signals: list[NDArray[np.float64]],
    peak_annotations: list[PeakAnnotation],
    baseline_annotations: list[BaselineAnnotation],
    is_control: list[bool] | None = None,
) -> PreparedDataset:
    """Run the full data-preparation pipeline.

    Args:
        times: List of 1-D time arrays, one per trace.
        signals: List of 1-D signal arrays, matching lengths.
        peak_annotations: User peak windows.
        baseline_annotations: User baseline regions.
        is_control: Optional per-trace boolean flags marking control traces
            (analyte known absent). When ``None``, all traces are treated as
            non-controls (the ``PreparedDataset.is_control`` field is all
            ``False``). Length must match ``len(times)``.

    Returns:
        :class:`PreparedDataset` with padded arrays, dt, baselines, noise,
        and a per-trace ``is_control`` mask.

    Raises:
        ValueError: If a baseline window overlaps any peak window, if any
            preparation step fails, or if ``is_control`` length does not
            match the number of traces.
    """
    n_trace = len(times)
    if is_control is not None and len(is_control) != n_trace:
        raise ValueError(
            f"is_control length ({len(is_control)}) must match number of "
            f"traces ({n_trace})."
        )
    is_control_arr: NDArray[np.bool_] = (
        np.asarray(is_control, dtype=np.bool_)
        if is_control is not None
        else np.zeros(n_trace, dtype=np.bool_)
    )
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
        n_trace=n_trace,
        peak_annotations=list(peak_annotations),
        baseline_annotations=list(baseline_annotations),
        baseline_intercept=intercept,
        baseline_slope=slope,
        noise_per_trace=noise,
        is_control=is_control_arr,
    )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/unit/fitting/test_prepared_dataset.py -v
uv run ruff check chromhandler/fitting/prepared_dataset.py tests/unit/fitting/test_prepared_dataset.py
uv run pyright chromhandler/fitting/prepared_dataset.py tests/unit/fitting/test_prepared_dataset.py
```
Expected: all tests pass (new ones plus existing ones — existing ones do not pass `is_control`, so they exercise the default path), clean.

- [ ] **Step 5: Smoke-test the broader suite**

```bash
uv run pytest tests/ -x -q
```
Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add chromhandler/fitting/prepared_dataset.py tests/unit/fitting/test_prepared_dataset.py
git commit -m "feat(prepared_dataset): add per-trace is_control mask"
```

---

## Task 4: `Handler.prepare_dataset()` convenience method

**Why:** With Tasks 1–3 in place, the user can manually flatten samples → traces and pass `is_control` per trace. That works but it's the same boilerplate the existing `foundations_demo.ipynb` shows. A `Handler.prepare_dataset()` method collapses it to one call and reads `is_control` automatically from each sample.

**Files:**
- Modify: `chromhandler/handler.py`
- Create: `tests/unit/handler/test_handler_prepare_dataset.py` (new file — this method is large enough to warrant its own test module under the handler test directory, consistent with `test_handler_basics.py`, `test_molecule.py`, etc.)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/handler/test_handler_prepare_dataset.py`:

```python
"""Tests for Handler.prepare_dataset convenience wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.handler import Handler
from chromhandler.model import Chromatogram, Sample


def _chrom(chrom_id: str, sample_id: str, t_axis: np.ndarray) -> Chromatogram:
    sig = 100.0 + 10.0 * np.exp(-((t_axis - 2.8) ** 2) / 0.02)
    return Chromatogram(
        id=chrom_id,
        sample_id=sample_id,
        time=t_axis.tolist(),
        signal=sig.tolist(),
        time_unit="min",
        wavelength=254.0,
        peaks=[],
    )


def _handler_with_two_samples(*, control_first: bool) -> Handler:
    h = Handler()
    t = np.arange(2.5, 3.6, 0.01)
    sample_a = Sample(
        id="A",
        chromatograms=[_chrom("c_a1", "A", t)],
        is_control=control_first,
    )
    sample_b = Sample(
        id="B",
        chromatograms=[_chrom("c_b1", "B", t)],
        is_control=False,
    )
    h.samples = [sample_a, sample_b]
    return h


def test_handler_prepare_dataset_basic() -> None:
    h = _handler_with_two_samples(control_first=False)
    peak_anns = [PeakAnnotation(molecule_id="A", rt_min=2.7, rt_max=2.9)]
    base_anns = [BaselineAnnotation(rt_min=2.55, rt_max=2.58),
                 BaselineAnnotation(rt_min=3.50, rt_max=3.55)]
    ds = h.prepare_dataset(peak_anns, base_anns)
    assert ds.n_trace == 2
    assert not ds.is_control.any()


def test_handler_prepare_dataset_collects_controls() -> None:
    h = _handler_with_two_samples(control_first=True)
    peak_anns = [PeakAnnotation(molecule_id="A", rt_min=2.7, rt_max=2.9)]
    base_anns = [BaselineAnnotation(rt_min=2.55, rt_max=2.58),
                 BaselineAnnotation(rt_min=3.50, rt_max=3.55)]
    ds = h.prepare_dataset(peak_anns, base_anns)
    assert ds.n_trace == 2
    import numpy as np
    np.testing.assert_array_equal(ds.is_control, np.array([True, False]))


def test_handler_prepare_dataset_raises_on_empty_samples() -> None:
    h = Handler()
    peak_anns = [PeakAnnotation(molecule_id="A", rt_min=2.7, rt_max=2.9)]
    base_anns = [BaselineAnnotation(rt_min=2.55, rt_max=2.58),
                 BaselineAnnotation(rt_min=3.50, rt_max=3.55)]
    with pytest.raises(ValueError, match="no chromatograms"):
        h.prepare_dataset(peak_anns, base_anns)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/handler/test_handler_prepare_dataset.py -v
```
Expected: `AttributeError: 'Handler' object has no attribute 'prepare_dataset'`.

- [ ] **Step 3: Implement `Handler.prepare_dataset()`**

In `chromhandler/handler.py`, add the import at the top with the other imports:

```python
from chromhandler.fitting.prepared_dataset import (
    PreparedDataset,
    prepare_dataset as _prepare_dataset,
)
```

Then add the method to the `Handler` class. Place it near `load_initial_conditions` (around line 660, after the closing `# ----...` divider following `load_initial_conditions`):

```python
    def prepare_dataset(
        self,
        peak_annotations: list[PeakAnnotation],
        baseline_annotations: list[BaselineAnnotation],
    ) -> PreparedDataset:
        """Build a :class:`PreparedDataset` from this handler's chromatograms.

        Flattens ``handler.samples → sample.chromatograms`` into the per-trace
        arrays the fitter consumes. Each sample's ``is_control`` flag is
        propagated to ``PreparedDataset.control_trace_indices`` for every
        chromatogram that sample contributes.

        Args:
            peak_annotations: User peak windows.
            baseline_annotations: User baseline regions.

        Returns:
            :class:`PreparedDataset` with controls already marked.

        Raises:
            ValueError: If the handler has no chromatograms across all samples.
        """
        times: list[NDArray[np.float64]] = []
        signals: list[NDArray[np.float64]] = []
        is_control: list[bool] = []
        for sample in self.samples:
            for chrom in sample.chromatograms:
                times.append(np.asarray(chrom.time, dtype=np.float64))
                signals.append(np.asarray(chrom.signal, dtype=np.float64))
                is_control.append(bool(sample.is_control))
        if not times:
            raise ValueError("Handler has no chromatograms across any sample.")
        return _prepare_dataset(
            times=times,
            signals=signals,
            peak_annotations=peak_annotations,
            baseline_annotations=baseline_annotations,
            is_control=is_control,
        )
```

If `NDArray` / `np` / `PeakAnnotation` / `BaselineAnnotation` imports are not already at the top of `handler.py`, add them. Check with `grep -n "^from numpy\|^import numpy\|PeakAnnotation\|BaselineAnnotation" chromhandler/handler.py` first.

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/unit/handler/test_handler_prepare_dataset.py -v
uv run ruff check chromhandler/handler.py tests/unit/handler/test_handler_prepare_dataset.py
uv run pyright chromhandler/handler.py tests/unit/handler/test_handler_prepare_dataset.py
```
Expected: 3 tests pass, clean.

- [ ] **Step 5: Smoke-test the broader suite**

```bash
uv run pytest tests/ -x -q
```
Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add chromhandler/handler.py tests/unit/handler/test_handler_prepare_dataset.py
git commit -m "feat(handler): add prepare_dataset convenience method propagating is_control"
```

---

## Self-Review

**Spec coverage check:** This plan covers the four refactor points from the design discussion in `docs/superpowers/plans/2026-05-11-priors-module.md`'s parent conversation:

1. ✅ `Sample.is_control` field — Task 1.
2. ✅ Auto-detection during conditions CSV import — Task 2.
3. ✅ `PreparedDataset.is_control` per-trace mask — Task 3.
4. ✅ `Handler.prepare_dataset()` convenience method — Task 4.

No `mark_as_control` method (user explicitly rejected manual marking — auto-detection from concentrations is the only path).
No standalone trivial test files for the bare Pydantic field default (rejected as testing-the-framework). The meaningful behavior is exercised end-to-end through `load_initial_conditions` in Task 2.

**Placeholder scan:** No "TBD" / "TODO" / "fill in later". Every step shows the actual code to add.

**Type consistency:**
- `Sample.is_control: bool` (Task 1) → consumed unchanged in Tasks 2 and 4.
- `PreparedDataset.is_control: NDArray[np.bool_]` (Task 3, shape `[n_trace]`) → matches the storage convention of every other per-trace field on `PreparedDataset` (`dt_per_trace`, `baseline_intercept`, `baseline_slope`, `noise_per_trace`). Indices are a one-liner derivative (`np.where(ds.is_control)[0]`) at use sites that want them.
- `is_control: list[bool] | None = None` kwarg name is identical between `prepare_dataset()` (Task 3) and the internal call from `Handler.prepare_dataset()` (Task 4).

**Cross-cutting:**

- **Backwards compatibility:** every change is additive with a safe default (`is_control=False`, `is_control: list[bool] | None = None`, `control_trace_indices=()`). All existing tests must continue to pass after each task — Steps 5 in Tasks 1, 2, and 4 verify this explicitly.
- **Pydantic `validate_assignment`:** `Sample.model_config` already has `validate_assignment=True`, so mutating `is_control` post-construction is type-checked.
- **`Handler._get_sample`:** already exists (used in `load_initial_conditions`). Tasks 2 and the test fixtures reuse it.
- **What's *not* in this plan:**
  - Reader-level control detection from instrument metadata — out of scope; controls are detected from the conditions CSV.
  - EnzymeML export awareness of controls — separate concern, deferred.
  - The actual prior-construction work that uses `control_trace_indices` — that's the priors plan (`2026-05-11-priors-module.md`), which will be revised after this plan ships.

**Edge cases covered by tests:**
- All-zero concentrations → control (Task 2).
- Partial-zero (some non-zero) → not a control (Task 2).
- NaN ≠ zero (Task 2).
- User-set `is_control=True` not overridden by non-zero concs (Task 2 — "auto-detection only flips False → True").
- Empty handler raises clear error (Task 4).
- Length-mismatched `is_control` raises clear error (Task 3).

**Open questions intentionally left for the priors-plan revision (not this plan):**
- How to handle peaks whose annotation window is `single` but some traces in the dataset are controls. Probably ignore controls for single-peak feature extraction; surface during priors-plan revision.
- Whether `Handler.prepare_dataset` should also detect orphan control samples (controls without matching experimental samples for the same peak).
