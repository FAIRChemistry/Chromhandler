# Fitting Module Rename Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rename temporary development module names (`better_*`) to production names (`fitter`, `model`, `visualize`) and split `data.py` into `types.py` and `utils.py`.

**Architecture:** Direct one-shot rename using `git mv` for files, inline class renames, split `data.py` by concern (types vs utilities), update all imports in `__init__.py`, run full test suite, delete old files.

**Tech Stack:** Git, pytest, Python standard library (dataclasses, typing)

---

## Task 1: Rename `better_fitter.py` → `fitter.py` and `BetterFitter` → `Fitter`

**Files:**
- Rename: `chromhandler/fitting/better_fitter.py` → `chromhandler/fitting/fitter.py`
- Modify: class name and docstrings

**Step 1: Rename file using git**

Run: `cd /Users/max/code/Chromhandler && git mv chromhandler/fitting/better_fitter.py chromhandler/fitting/fitter.py`

Expected: File renamed in working tree, staged for commit.

**Step 2: Update class name and docstrings**

Read the file to find all occurrences:
```bash
grep -n "BetterFitter\|better_fitter" chromhandler/fitting/fitter.py
```

Then edit `chromhandler/fitting/fitter.py`:
- Replace class definition: `class BetterFitter:` → `class Fitter:`
- Replace docstring references to `BetterFitter` → `Fitter`
- Replace references to module name in docstrings: `better_fitter` → `fitter`

**Step 3: Verify no syntax errors**

Run: `python3 -m py_compile chromhandler/fitting/fitter.py`

Expected: No output (success).

**Step 4: Commit**

```bash
git add chromhandler/fitting/fitter.py
git commit -m "refactor: rename better_fitter.py to fitter.py and BetterFitter to Fitter"
```

---

## Task 2: Rename `better_model.py` → `model.py`

**Files:**
- Rename: `chromhandler/fitting/better_model.py` → `chromhandler/fitting/model.py`

**Step 1: Rename file using git**

Run: `git mv chromhandler/fitting/better_model.py chromhandler/fitting/model.py`

Expected: File renamed.

**Step 2: Update class/function names and docstrings**

Read the file to identify all public names:
```bash
grep -n "^class Better\|^def better\|better_model" chromhandler/fitting/model.py | head -20
```

Edit `chromhandler/fitting/model.py`:
- If there's a `BetterModel` class, rename to `Model`
- Update docstrings referencing `better_model` → `model`
- Update docstrings referencing `BetterModel` → `Model`

**Step 3: Verify syntax**

Run: `python3 -m py_compile chromhandler/fitting/model.py`

Expected: No output.

**Step 4: Commit**

```bash
git add chromhandler/fitting/model.py
git commit -m "refactor: rename better_model.py to model.py"
```

---

## Task 3: Rename `better_visualize.py` → `visualize.py`

**Files:**
- Rename: `chromhandler/fitting/better_visualize.py` → `chromhandler/fitting/visualize.py`

**Step 1: Rename file using git**

Run: `git mv chromhandler/fitting/better_visualize.py chromhandler/fitting/visualize.py`

Expected: File renamed.

**Step 2: Update docstrings if needed**

Read and check for references to `better_visualize`:
```bash
grep -n "better_visualize" chromhandler/fitting/visualize.py
```

If any exist, edit and replace with `visualize`.

**Step 3: Verify syntax**

Run: `python3 -m py_compile chromhandler/fitting/visualize.py`

Expected: No output.

**Step 4: Commit**

```bash
git add chromhandler/fitting/visualize.py
git commit -m "refactor: rename better_visualize.py to visualize.py"
```

---

## Task 4: Create `types.py` from `data.py` (types and mode queries)

**Files:**
- Create: `chromhandler/fitting/types.py`
- Source: Extract from `chromhandler/fitting/data.py`

**Step 1: Create new `types.py`**

Create `/Users/max/code/Chromhandler/chromhandler/fitting/types.py` with:

```python
"""Data types and schemas for peak fitting module.

Defines:
- :class:`ModelHyperparams`: Hyperparameter configuration
- :data:`PeakMode`: Peak mode enumeration
- Mode query functions: :func:`peak_component_count`, :func:`peak_is_doublet_mode`, etc.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

PeakMode = Literal["single", "artefact_doublet", "free_doublet"]
PEAK_MODE_TO_CODE: dict[str, int] = {
    "single": 0,
    "artefact_doublet": 1,
    "free_doublet": 2,
}


@dataclasses.dataclass(frozen=True)
class ModelHyperparams:
    """Tunable hyperparameters for ``model.model()``.

    All values have research-validated defaults.  Pass a custom instance to
    :class:`~chromhandler.fitting.Fitter` to override for sensitivity
    analysis or domain-specific tuning.
    """

    # Half-width prior scale floor (log-space CV)
    w_prior_log_scale: float = 0.4

    # Area prior spread — S/N-dependent linear interpolation
    area_log_sigma_high_snr: float = 0.3   # tight for clear peaks (S/N > threshold_high)
    area_log_sigma_low_snr: float = 0.8    # wide for ambiguous peaks (S/N < threshold_low)
    area_snr_threshold_high: float = 10.0
    area_snr_threshold_low: float = 3.0

    # Artefact area
    area_art_log_sigma: float = 0.3        # shared artefact area CV ~30%
    area_art_trace_log_scale: float = 0.15  # per-trace artefact multiplicative noise

    # Separation priors (LogNormal in log-space)
    free_sep_loc_mult: float = 1.5         # typical separation in sigma units
    free_sep_log_sigma: float = 0.4

    art_sep_min_w_mult: float = 0.5        # min separation in half-width units
    art_sep_max_window_frac: float = 0.5


def peak_component_count(mode: str) -> int:
    """Return the number of mixture components implied by a peak mode."""
    return 1 if mode == "single" else 2


def peak_is_doublet_mode(mode: str) -> bool:
    """Return True for all two-component peak modes."""
    return peak_component_count(mode) == 2


def peak_is_artefact_mode(mode: str) -> bool:
    """Return True when the peak uses the artefact-doublet branch."""
    return mode == "artefact_doublet"


def peak_is_free_mode(mode: str) -> bool:
    """Return True when the peak uses the free-doublet branch."""
    return mode == "free_doublet"
```

**Step 2: Verify syntax**

Run: `python3 -m py_compile chromhandler/fitting/types.py`

Expected: No output.

**Step 3: Add to git**

Run: `git add chromhandler/fitting/types.py`

(Don't commit yet; we'll batch commits later.)

---

## Task 5: Create `utils.py` from `data.py` (array utilities)

**Files:**
- Create: `chromhandler/fitting/utils.py`
- Source: Extract from `chromhandler/fitting/data.py`

**Step 1: Create new `utils.py`**

Create `/Users/max/code/Chromhandler/chromhandler/fitting/utils.py` with:

```python
"""Fitting module utilities for array operations and masking.

Functions:
- :func:`pad_traces`: Pad time/signal lists to equal length
- :func:`region_to_mask`: Create boolean mask for time region
- :func:`baseline_to_mask`: Create mask for baseline annotation regions
- :func:`peaks_to_mask`: Create mask for peak annotation regions
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from chromhandler.annotations import BaselineAnnotation, PeakAnnotation


def pad_traces(
    x_lists: list[list[float]], y_lists: list[list[float]]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Pad time/signal lists to equal length (NaN-padded) and stack into 2-D arrays."""
    if len(x_lists) != len(y_lists):
        raise ValueError("x_lists and y_lists must have the same length")
    max_len = max(max(len(x) for x in x_lists), max(len(y) for y in y_lists))
    padded_x = [x + [float("nan")] * (max_len - len(x)) for x in x_lists]
    padded_y = [y + [float("nan")] * (max_len - len(y)) for y in y_lists]
    return np.array(padded_x, dtype=float), np.array(padded_y, dtype=float)


def region_to_mask(low: float, high: float, time: jnp.ndarray) -> jnp.ndarray:
    """Mask True for all time points in [low, high]."""
    return (time >= low) & (time <= high)


def baseline_to_mask(baselines: list[BaselineAnnotation], time: jnp.ndarray) -> jnp.ndarray:
    """Boolean mask True for time points in any baseline region."""
    if not baselines:
        return jnp.zeros(time.shape, dtype=bool)
    masks = jnp.stack([region_to_mask(b.rt_min, b.rt_max, time) for b in baselines])
    return jnp.any(masks, axis=0)


def peaks_to_mask(peaks: list[PeakAnnotation], time: jnp.ndarray) -> jnp.ndarray:
    """Boolean mask True for time points in any peak region.

    Returns shape ``(n_peaks, n_chromatograms, n_timepoints)``.
    """
    peak_centers = jnp.array([(p.rt_min + p.rt_max) / 2 for p in peaks])
    sorted_indices = [int(i) for i in jnp.argsort(peak_centers).tolist()]
    sorted_peaks = [peaks[i] for i in sorted_indices]
    return jnp.stack([region_to_mask(low=p.rt_min, high=p.rt_max, time=time) for p in sorted_peaks])
```

**Step 2: Verify syntax**

Run: `python3 -m py_compile chromhandler/fitting/utils.py`

Expected: No output.

**Step 3: Add to git**

Run: `git add chromhandler/fitting/utils.py`

---

## Task 6: Update `chromhandler/fitting/__init__.py` imports

**Files:**
- Modify: `chromhandler/fitting/__init__.py`

**Step 1: Read current file**

Read `/Users/max/code/Chromhandler/chromhandler/fitting/__init__.py` to see all current imports and exports.

**Step 2: Update imports**

Edit the file. Replace:

```python
"""Chromatographic peak fitting module.

Sub-modules
-----------
- ``better_fitter``: Main :class:`BetterFitter` class (MCMC, area extraction).
- ``priors``: Window-geometry prior construction and FWHM diagnostics.
- ``baseline``: Linear baseline estimation.
- ``better_model``: NumPyro probabilistic model.
- ``better_visualize``: Posterior and diagnostic plots.
- ``shift``: Retention-time alignment via per-trace shift optimization.
"""

from .better_fitter import AreaRecord, BetterFitter, PosteriorCurves
from .data import ModelHyperparams

__all__ = ["AreaRecord", "BetterFitter", "ModelHyperparams", "PosteriorCurves"]
```

With:

```python
"""Chromatographic peak fitting module.

Sub-modules
-----------
- ``fitter``: Main :class:`Fitter` class (MCMC, area extraction).
- ``priors``: Window-geometry prior construction and FWHM diagnostics.
- ``baseline``: Linear baseline estimation.
- ``model``: NumPyro probabilistic model.
- ``visualize``: Posterior and diagnostic plots.
- ``shift``: Retention-time alignment via per-trace shift optimization.
- ``types``: Data types and hyperparameter configuration.
- ``utils``: Array utility functions.
"""

from .fitter import AreaRecord, Fitter, PosteriorCurves
from .types import ModelHyperparams

__all__ = ["AreaRecord", "Fitter", "ModelHyperparams", "PosteriorCurves"]
```

**Step 3: Verify syntax**

Run: `python3 -m py_compile chromhandler/fitting/__init__.py`

Expected: No output.

**Step 4: Add to git**

Run: `git add chromhandler/fitting/__init__.py`

---

## Task 7: Update docstring references in `fitter.py`

**Files:**
- Modify: `chromhandler/fitting/fitter.py`

**Step 1: Search for old references**

Run: `grep -n "better_model\|better_visualize\|BetterFitter" chromhandler/fitting/fitter.py`

**Step 2: Fix references**

Edit `chromhandler/fitting/fitter.py`:
- Replace `:class:`~chromhandler.fitting.BetterFitter`` → `:class:`~chromhandler.fitting.Fitter``
- Replace references to `better_model` module → `model`
- Replace references to `better_visualize` module → `visualize`

(Check if there are any; if not, skip this step.)

**Step 3: Verify syntax**

Run: `python3 -m py_compile chromhandler/fitting/fitter.py`

Expected: No output.

**Step 4: Add to git**

Run: `git add chromhandler/fitting/fitter.py`

---

## Task 8: Update imports in other fitting modules

**Files:**
- Modify: `chromhandler/fitting/model.py`, `chromhandler/fitting/visualize.py`, `chromhandler/fitting/priors.py`, `chromhandler/fitting/baseline.py`, `chromhandler/fitting/shift.py` (any that import from old locations)

**Step 1: Search for imports from old modules**

Run: `grep -r "from .better_fitter\|from .better_model\|from .better_visualize\|from .data import" chromhandler/fitting/ --include="*.py"`

**Step 2: Update imports**

For each match found:
- `from .better_fitter import X` → `from .fitter import X`
- `from .better_model import X` → `from .model import X`
- `from .better_visualize import X` → `from .visualize import X`
- `from .data import ModelHyperparams` → `from .types import ModelHyperparams`
- `from .data import pad_traces, ...` → `from .utils import pad_traces, ...`

Edit the relevant files.

**Step 3: Verify all syntax**

Run: `python3 -m py_compile chromhandler/fitting/*.py`

Expected: No output.

**Step 4: Add to git**

Run: `git add chromhandler/fitting/*.py`

---

## Task 9: Update imports in main `chromhandler` package

**Files:**
- Modify: any files in `chromhandler/` root that import from fitting subpackage

**Step 1: Search for imports**

Run: `grep -r "from chromhandler.fitting import\|from chromhandler.fitting.better" chromhandler/ --include="*.py" | grep -v "^chromhandler/fitting/"`

**Step 2: Update imports**

For each match:
- `from chromhandler.fitting import BetterFitter` → `from chromhandler.fitting import Fitter`
- Any docstring references to `BetterFitter` → `Fitter`

**Step 3: Verify syntax of modified files**

Run: `python3 -m py_compile <modified_file>` for each file.

**Step 4: Add to git**

Run: `git add <modified_files>`

---

## Task 10: Delete old files

**Files:**
- Delete: `chromhandler/fitting/better_fitter.py`, `chromhandler/fitting/better_model.py`, `chromhandler/fitting/better_visualize.py`, `chromhandler/fitting/data.py`

**Step 1: Remove files from git**

Run:
```bash
git rm chromhandler/fitting/better_fitter.py
git rm chromhandler/fitting/better_model.py
git rm chromhandler/fitting/better_visualize.py
git rm chromhandler/fitting/data.py
```

Expected: Files staged for deletion.

**Step 2: Verify no lingering references**

Run: `grep -r "better_fitter\|better_model\|better_visualize" chromhandler/ tests/ --include="*.py"`

Expected: No matches (or only in comments/doc strings that are okay).

**Step 3: Stage deletions**

Files should already be staged. Verify with:

Run: `git status`

Expected: Deleted files listed.

---

## Task 11: Run full test suite

**Files:**
- Test all: `pytest`

**Step 1: Run pytest**

Run: `cd /Users/max/code/Chromhandler && uv run pytest -v`

Expected: All tests pass (0 failures).

**Step 2: Verify import errors are caught**

If any test fails with `ImportError` or `ModuleNotFoundError`, fix the relevant import in the source or test files.

**Step 3: Check coverage**

Run: `uv run coverage report --skip-covered`

Expected: Coverage stable (no significant drop).

---

## Task 12: Final commit

**Files:**
- All staged changes

**Step 1: Review changes**

Run: `git diff --staged`

Expected: Renames, new files, import updates, deleted files all present.

**Step 2: Create atomic commit**

Run:
```bash
git commit -m "refactor: rename fitting modules (better_* → final names) and split data.py

- Rename better_fitter.py → fitter.py, BetterFitter → Fitter
- Rename better_model.py → model.py
- Rename better_visualize.py → visualize.py
- Split data.py into types.py (schemas) and utils.py (helpers)
- Update __init__.py and all internal imports
- All tests passing

This completes the production naming for the fitting subpackage."
```

Expected: Commit succeeds.

**Step 3: Verify commit**

Run: `git log -1 --stat`

Expected: Shows all files renamed/created/deleted as expected.

---

## Summary

**Total tasks:** 12
**Total commits:** 5 (Task 1-3 individual, Tasks 4-5 batched, Tasks 6-9 batched, Task 10 deletion, Task 12 final)
**Expected duration:** ~20-30 minutes
**Risk level:** Low (modular changes, well-tested by pytest)

