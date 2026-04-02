# Fitting Module Restructuring: Rename & Split Design

**Date:** 2026-04-02
**Scope:** Rename temporary development names to final Pythonic names; split mixed utility/type file.

## Overview

The fitting subpackage contains remnants of development naming (`better_fitter`, `better_model`, `better_visualize`) and a catch-all `data.py` file. This design finalizes the module structure for production.

## Changes

### File Renames
- `chromhandler/fitting/better_fitter.py` → `fitter.py`
- `chromhandler/fitting/better_model.py` → `model.py`
- `chromhandler/fitting/better_visualize.py` → `visualize.py`

### File Split
- `chromhandler/fitting/data.py` → split into:
  - **`types.py`**: `ModelHyperparams` dataclass, `PeakMode` type alias, mode query functions
  - **`utils.py`**: Array helpers (`pad_traces`, `region_to_mask`, `baseline_to_mask`, `peaks_to_mask`)

### Class Renames
- `BetterFitter` → `Fitter`
- `BetterModel` (if exists) → `Model`
- `BetterVisualize` (if exists) → functions in `visualize.py`

### Import Updates
1. **`chromhandler/fitting/__init__.py`**: Update re-exports
   - `from .fitter import ...` (was `from .better_fitter import ...`)
   - `from .types import ...` (was `from .data import ...`)
   - Remove any `from .data import ...` that's now in `utils.py`

2. **Internal references**: Update docstrings and comments referencing old module names

### Old Files
- Delete: `better_fitter.py`, `better_model.py`, `better_visualize.py`, `data.py`
- Deletion happens after imports are updated and tests pass

## Testing Strategy

1. Run full test suite after rename (`pytest`)
2. Verify no import errors or missing references
3. Ensure public API remains stable (users still import from `chromhandler.fitting`)

## Success Criteria

- ✅ All files renamed as specified
- ✅ All classes renamed
- ✅ All imports updated (internal + `__init__.py`)
- ✅ `pytest` passes with 0 failures
- ✅ Old files deleted
- ✅ Public API unchanged (backward compatible for users)

## Execution Method

**Approach 1: Direct One-Shot Rename** — Rename all files and classes in git using `git mv`, update imports in one coherent commit, then run tests. Clean git history, low risk given modular scope.
