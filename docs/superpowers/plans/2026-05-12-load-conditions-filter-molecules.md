# Filter `load_initial_conditions` by Registered Molecules + Real-Data Demo

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Handler.load_initial_conditions()` ignore CSV columns whose name isn't a registered molecule on the handler (so derivatization reagents, internal standards, and unused species don't trip the all-zero → `is_control` auto-detection). Demonstrate the resulting flow on real ASM kinetic-series data in `priors_demo.ipynb`.

**Architecture:** Two small touches:

1. **`chromhandler/handler.py`** — `load_initial_conditions()` filters CSV columns by `self.molecules` when any molecules are registered. When no molecules are registered, current behavior is preserved (all columns parsed) — backwards-compatible.
2. **`notebooks/priors_demo.ipynb`** — appends a "real data" section that loads the ASM kinetic series + `conditions.csv` fixture, registers only the relevant analytes (SIH, Hyp, Ino), and walks the user through the `is_control` mask, `prepare_dataset` output, and the `build_priors`-without-controls error path (which is the genuinely informative real-data outcome given that the fixture has no control chromatograms yet).

**Tech Stack:** Python 3.11+, pandas (already a dependency, used by `load_initial_conditions`), pytest, ruff, pyright. Notebook execution via `uv run jupyter nbconvert --execute`.

**Why the filter matters:** The new `tests/fixtures/asm_kinetic_series/conditions.csv` lists DTNB (Ellman's derivatization reagent) at 800 µM in every single row. Without filtering, no row ever has all-zero declared concentrations, so the auto-detection of controls is permanently broken on any real biochemistry data with a derivatization step or a constant internal standard. Filtering by registered molecules is the principled fix: the user already declares what they care about via `handler.create_molecule(...)`, so the conditions loader should respect that declaration.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `chromhandler/handler.py` | Modify (~15 line change in `load_initial_conditions`) | Filter CSV columns by `self.molecules`; current behavior preserved when handler has no molecules. |
| `tests/unit/handler/test_handler_basics.py` | Extend (4 new tests) | Cover the filter behavior, the auto-detection interaction, and the backwards-compat path. |
| `notebooks/priors_demo.ipynb` | Extend (new "Section 9: Real data" at the end) | Real-data demonstration on the ASM kinetic series fixture. |

---

## Conventions

- Quality gates after every file edit:
  ```bash
  uv run ruff check <file>
  uv run pyright <file>
  ```
  Both must report zero issues before commit.
- Per-file pytest runs are fine (the suite has pre-existing duplicate-name collection issues at the suite root):
  ```bash
  uv run pytest tests/unit/handler/test_handler_basics.py -v
  ```
- One commit per task. `feat(handler): ...` or `docs(notebooks): ...` style.
- The notebook is executed end-to-end via `jupyter nbconvert --execute` before commit — outputs must match what the markdown promises.

---

## Task 1: Filter `load_initial_conditions` by registered molecules

**Why first:** Task 2 (the notebook update) depends on this filter actually working so that real-data loading produces the expected `is_control` behavior.

**Behavior:**
- If `self.molecules` is non-empty: for each CSV row, only columns whose name matches a registered molecule ID are loaded as `InitialCondition`s; all other columns are silently skipped. Only registered-molecule values count toward the all-zero → `is_control` check.
- If `self.molecules` is empty: every CSV column is parsed (current behavior; backwards-compatible with existing tests that don't register molecules first).

**Files:**
- Modify: `chromhandler/handler.py` (the `load_initial_conditions` method, around lines 647–666)
- Extend: `tests/unit/handler/test_handler_basics.py` (append 4 new tests at the end)

- [ ] **Step 1: Read the current state of `load_initial_conditions`**

Confirm the current loop structure before editing. Run:

```bash
sed -n '599,670p' chromhandler/handler.py
```

Expected: the loop iterates `df_mol.columns` (all CSV columns), calls `self.add_initial_condition(...)` for every non-NaN value, and runs the auto-detection of `is_control` at the end.

- [ ] **Step 2: Append the four new failing tests**

Append the following at the end of `tests/unit/handler/test_handler_basics.py`. If `pandas`, `Handler`, and `Sample` are already imported in the file, do not re-import them. If `Molecule` is not yet imported in the file, add `from chromhandler.model import Molecule` near the existing imports.

```python
def _handler_with_samples_and_molecules(
    sample_ids: list[str],
    molecule_ids: list[str],
) -> Handler:
    h = Handler()
    h.samples = [Sample(id=sid) for sid in sample_ids]
    for mol_id in molecule_ids:
        h.create_molecule(id=mol_id, pubchem_cid=1)
    return h


def test_load_initial_conditions_filters_by_registered_molecules() -> None:
    """Columns for unregistered molecules (e.g. derivatization reagent) are
    silently skipped, even if their value is non-NaN."""
    h = _handler_with_samples_and_molecules(["CV10"], ["SIH", "Hyp", "Ino"])
    df = pd.DataFrame({
        "sample_id": ["CV10"],
        "SAH": [0.0],      # not registered → ignored
        "SIH": [50.0],     # registered → loaded
        "DTNB": [800.0],   # not registered → ignored (derivatization reagent)
        "Hyp": [0.0],      # registered → loaded
        "Ino": [0.0],      # registered → loaded
        "SIHH": [5.0],     # not registered → ignored
    })
    h.load_initial_conditions(df, conc_unit="umol / l")
    sample = h._get_sample("CV10")
    mol_ids = {ic.molecule_id for ic in sample.initial_conditions}
    assert mol_ids == {"SIH", "Hyp", "Ino"}
    assert sample.is_control is False  # SIH=50 is non-zero


def test_load_initial_conditions_auto_detects_control_with_reagent_present() -> None:
    """A sample with all registered analytes = 0 is auto-detected as a
    control, even when an unregistered reagent column is non-zero. This is
    the canonical biochemistry case (derivatization reagent always present)."""
    h = _handler_with_samples_and_molecules(["CV4"], ["SIH", "Hyp", "Ino"])
    df = pd.DataFrame({
        "sample_id": ["CV4"],
        "SAH": [0.0],
        "SIH": [0.0],
        "DTNB": [800.0],   # reagent, ignored
        "Hyp": [0.0],
        "Ino": [0.0],
        "SIHH": [0.0],
    })
    h.load_initial_conditions(df, conc_unit="umol / l")
    assert h._get_sample("CV4").is_control is True


def test_load_initial_conditions_no_molecules_registered_parses_all() -> None:
    """Backwards-compat: when no molecules are registered, every column is
    parsed as an initial condition (current behavior preserved)."""
    h = Handler()
    h.samples = [Sample(id="CV4")]
    df = pd.DataFrame({
        "sample_id": ["CV4"],
        "SIH": [0.0],
        "DTNB": [800.0],
    })
    h.load_initial_conditions(df, conc_unit="umol / l")
    sample = h._get_sample("CV4")
    mol_ids = {ic.molecule_id for ic in sample.initial_conditions}
    assert mol_ids == {"SIH", "DTNB"}
    # All-column check: DTNB=800 is non-zero → not detected as control.
    assert sample.is_control is False


def test_load_initial_conditions_filter_raises_when_no_registered_in_row() -> None:
    """If filtering leaves zero usable columns for a sample (all NaN under
    the filter), the existing 'no initial conditions' ValueError still
    fires — the failure mode is unchanged, just observed through the filter."""
    h = _handler_with_samples_and_molecules(["CV99"], ["SIH"])
    df = pd.DataFrame({
        "sample_id": ["CV99"],
        "DTNB": [800.0],       # unregistered, filtered out
        "SIH": [float("nan")],  # registered but NaN → contributes nothing
    })
    with pytest.raises(ValueError, match="no initial conditions"):
        h.load_initial_conditions(df, conc_unit="umol / l")
```

- [ ] **Step 3: Run the new tests to confirm they fail**

```bash
uv run pytest tests/unit/handler/test_handler_basics.py -v -k "load_initial_conditions and (filter or auto_detect or no_molecules)"
```

Expected: tests fail. Specifically:
- `test_load_initial_conditions_filters_by_registered_molecules`: fails because `mol_ids` will include `SAH`, `DTNB`, `SIHH` (current code parses all columns).
- `test_load_initial_conditions_auto_detects_control_with_reagent_present`: fails because `declared_concs` includes `DTNB=800.0`, so `all(c == 0.0)` is False.
- `test_load_initial_conditions_no_molecules_registered_parses_all`: passes already (this is backwards-compat).
- `test_load_initial_conditions_filter_raises_when_no_registered_in_row`: fails — the filter doesn't exist yet, so DTNB gets through.

- [ ] **Step 4: Implement the filter**

In `chromhandler/handler.py`, locate the per-sample loop inside `load_initial_conditions` (currently at lines 647–666). Replace the inner column-iteration block with the filtered version. The full updated loop body should be:

```python
        registered_mols: set[str] = set(self.molecules.keys())
        filter_active: bool = bool(registered_mols)

        for i, sample_id in enumerate(sample_ids):
            if sample_id not in existing_ids:
                continue
            added_any = False
            declared_concs: list[float] = []
            for mol_id in df_mol.columns:
                if filter_active and str(mol_id) not in registered_mols:
                    continue
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

The two new lines at the top (`registered_mols = ...` and `filter_active = ...`) and the new `if filter_active and str(mol_id) not in registered_mols: continue` line inside the inner loop are the entire change.

Also update the method's docstring (`def load_initial_conditions`, around line 599) to add a paragraph after the existing description:

```
        Note: when the handler has any molecules registered via
        :meth:`create_molecule` or :meth:`register_molecule`, CSV columns
        whose names do not match a registered molecule ID are silently
        skipped. They are not loaded as :class:`~chromhandler.model.InitialCondition`
        objects, and they do not count toward the all-zero auto-detection
        of ``is_control``. This lets the user ignore derivatization
        reagents, internal standards, or unused species without flagging
        them in the CSV. When no molecules are registered, every column
        is parsed (backwards-compatible default).
```

- [ ] **Step 5: Run the new tests to confirm they pass**

```bash
uv run pytest tests/unit/handler/test_handler_basics.py -v -k "load_initial_conditions"
```

Expected: all `load_initial_conditions` tests pass (the four new ones plus any pre-existing tests for the original behavior).

- [ ] **Step 6: Run quality gates and broader smoke test**

```bash
uv run ruff check chromhandler/handler.py tests/unit/handler/test_handler_basics.py
uv run pyright chromhandler/handler.py
uv run pytest tests/unit/handler/ -q
uv run pytest tests/unit/fitting/ -q
```

Expected: ruff and pyright clean on the files we touched, all unit tests pass.

- [ ] **Step 7: Commit**

```bash
git add chromhandler/handler.py tests/unit/handler/test_handler_basics.py
git commit -m "$(cat <<'EOF'
feat(handler): load_initial_conditions filters by registered molecules

When the handler has any molecules registered, CSV columns whose names
do not match a registered molecule ID are silently skipped — neither
loaded as InitialConditions nor counted toward the all-zero is_control
auto-detection. This unblocks real biochemistry datasets where
derivatization reagents (e.g. DTNB) or internal standards are present
at constant non-zero concentration in every sample.

When no molecules are registered, every column is parsed
(backwards-compatible default).
EOF
)"
```

---

## Task 2: Real-data section in `priors_demo.ipynb`

**Why:** The synthetic demo at the start of `priors_demo.ipynb` already exercises every code path of `build_priors`. The user has now added a real conditions CSV alongside the ASM kinetic series fixture. A short closing section on real data is the most useful demo of the *new* filter behavior: it shows the actual sample CV10 being correctly classified as a non-control, the `prepare_dataset` outputs on real chromatograms, and the helpful error message that `build_priors` produces when the fixture lacks control chromatograms.

**Important real-data caveats baked into the section:**

- The fixture `tests/fixtures/asm_kinetic_series/` only contains the 7 CV10 timepoint ASM JSONs. No chromatogram files exist for any of the rows that *would* be controls (CW2, CV4, CV5). Loading the conditions CSV produces a handler with one non-control sample (CV10) and zero control samples.
- This means `build_priors` will raise `ValueError: no control traces in dataset; ...` when given an `artefact_doublet` annotation — and that's the right answer. The notebook documents this as the expected outcome and shows that the error message lists every available trace by ID, so users have an obvious next step ("add a control ASM JSON to the fixture / mark a sample manually / switch to single-peak annotation").

**Files:**
- Modify: `notebooks/priors_demo.ipynb` (append a new section "Section 9" before any closing markdown cell; reorder the existing "What's next" cell if it's the last cell so it remains last).

- [ ] **Step 1: Open the notebook and locate the last cell**

```bash
uv run python -c "
import json, pathlib
p = pathlib.Path('notebooks/priors_demo.ipynb')
nb = json.loads(p.read_text())
for i, c in enumerate(nb['cells']):
    print(i, c['cell_type'], c.get('id', '-'), repr(''.join(c['source'])[:60]))
"
```

Note the index and `id` of the final markdown cell (the "What's next" / closing section). New cells will be inserted immediately *before* it so that the closing cell remains last.

- [ ] **Step 2: Add the real-data section heading (markdown cell)**

Use the `NotebookEdit` tool with `edit_mode=insert` and `cell_id` set to the id of the cell that should *precede* the new one (the section-8 viz cell `viz-code`). The new cell goes after that one, before the closing markdown.

```
Markdown source:
## 9. Real data — ASM kinetic series + conditions CSV

The synthetic demo above exercises every code path. Now we point the same
pipeline at the real ASM kinetic-series fixture and demonstrate the new
**registered-molecule filter** in `load_initial_conditions`.

What this section verifies:

- Only molecules registered with the handler get loaded from
  `conditions.csv` as `InitialCondition`s. The derivatization reagent
  (`DTNB`, present at 800 µM in every row) is silently skipped.
- The auto-detection of `is_control` uses only the registered analytes.
  CV10 has `SIH=50` (non-zero substrate) → correctly classified as a
  sample, not a control.
- `Handler.prepare_dataset` produces a real `PreparedDataset` with
  `trace_ids` of the form `"CV10/<chromatogram_id>"`.
- `build_priors` with an `artefact_doublet` annotation raises a helpful
  error that names the specific traces — the fixture has no control
  chromatograms yet, so this is the *expected* outcome and shows the
  error path on real IDs.
```

Cell id suggestion: `sec9-md`. Cell type: `markdown`.

- [ ] **Step 3: Add the load-and-register code cell**

Insert a code cell after `sec9-md`. Cell id suggestion: `realdata-load`.

```python
from pathlib import Path

from chromhandler.handler import Handler

ASM_DIR = Path.cwd().parent / "tests" / "fixtures" / "asm_kinetic_series"
CONDITIONS_CSV = ASM_DIR / "conditions.csv"

handler = Handler.read_asm(path=ASM_DIR, mode="timecourse")

# Register only the analytes we care about. DTNB (derivatization reagent),
# SAH, SIHH are also in the conditions CSV but we deliberately do NOT
# register them — the filter in load_initial_conditions will skip them.
for mol_id in ("SIH", "Hyp", "Ino"):
    handler.create_molecule(id=mol_id, pubchem_cid=1)

print(f"handler.samples           = {[s.id for s in handler.samples]}")
print(f"handler.molecules.keys()  = {list(handler.molecules.keys())}")
print(f"n_chromatograms per sample = "
      f"{[len(s.chromatograms) for s in handler.samples]}")
```

- [ ] **Step 4: Add the conditions-loading cell**

Insert another code cell. Cell id suggestion: `realdata-conditions`.

```python
handler.load_initial_conditions(CONDITIONS_CSV, conc_unit="umol / l")

cv10 = handler._get_sample("CV10")
print(f"CV10 is_control       = {cv10.is_control}")
print(f"CV10 initial conditions:")
for ic in cv10.initial_conditions:
    print(f"  {ic.molecule_id:>5} = {ic.init_conc}")
print()
print("Note: SAH, DTNB, SIHH columns were in the CSV but skipped because "
      "they are not registered as molecules on the handler. The "
      "all-zero check used to set is_control therefore considers only "
      "SIH, Hyp, Ino — and since SIH=50, CV10 is correctly classified "
      "as a sample (not a control).")
```

- [ ] **Step 5: Add the prepare-dataset cell**

Insert another code cell. Cell id suggestion: `realdata-prepare`.

```python
from chromhandler.annotations import BaselineAnnotation, PeakAnnotation

peak_anns = [
    PeakAnnotation(
        molecule_id="SIH", rt_min=2.85, rt_max=3.15,
        mode="artefact_doublet", artefact_side="right",
    ),
]
base_anns = [
    BaselineAnnotation(rt_min=2.50, rt_max=2.52),
    BaselineAnnotation(rt_min=3.50, rt_max=3.52),
]

real_dataset = handler.prepare_dataset(peak_anns, base_anns)
print(f"n_trace          = {real_dataset.n_trace}")
print(f"is_control mask  = {real_dataset.is_control}")
print(f"trace_ids        = {real_dataset.trace_ids}")
print(f"dt_global        = {real_dataset.dt_global:.5f} min")
print(f"noise_per_trace  = {[round(float(x), 3) for x in real_dataset.noise_per_trace]}")
```

- [ ] **Step 6: Add the build-priors-with-error cell**

Insert another code cell. Cell id suggestion: `realdata-build-priors-error`.

```python
from chromhandler.fitting.priors import build_priors

try:
    build_priors(real_dataset)
except ValueError as e:
    msg = str(e)
    print("ValueError raised (truncated):")
    print(msg[:500] + ("..." if len(msg) > 500 else ""))
```

- [ ] **Step 7: Add a closing-context markdown cell**

Insert another markdown cell with id `sec9-closing-md`:

```
**What this tells you about the real-data path:**

- The filter in `load_initial_conditions` works as designed — `DTNB`,
  `SAH`, and `SIHH` were in the CSV but did not become
  `InitialCondition` entries on CV10, and they did not contribute to
  the auto-detection of `is_control`.
- `Handler.prepare_dataset` produces a real `PreparedDataset` with
  `trace_ids` of the form `"CV10/<chromatogram_id>"`. Those IDs flow
  through to error messages from the priors layer.
- `build_priors` raises cleanly when the dataset has no controls and
  the annotation is `artefact_doublet`. The error message lists every
  available trace by ID, so the user can immediately see what's
  loaded and decide how to proceed (add a control chromatogram to the
  fixture, mark a sample manually for testing, or switch to a
  `single`-mode annotation).

**To actually run `build_priors` on this fixture**, drop a no-substrate
control ASM JSON (e.g. a CV4 or CV5 injection from the same experiment)
into `tests/fixtures/asm_kinetic_series/`, then re-run the notebook. The
conditions CSV already has CV4 and CV5 rows with all analyte
concentrations set to zero, so the auto-detection will flag them as
controls without any additional configuration.
```

- [ ] **Step 8: Execute the notebook end-to-end**

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/priors_demo.ipynb --output priors_demo.ipynb
```

Expected: notebook executes without exception. The new section's output cells should contain the printed text shown in Step 4/5/6 above. The size of the resulting `.ipynb` should be slightly larger than before (a few extra cells with text-only outputs).

- [ ] **Step 9: Sanity-check the new outputs**

```bash
uv run python -c "
import json, pathlib
nb = json.loads(pathlib.Path('notebooks/priors_demo.ipynb').read_text())
for cell in nb['cells']:
    cid = cell.get('id', '')
    if cid.startswith('realdata-'):
        print(f'=== {cid} ===')
        for out in cell.get('outputs', []):
            if 'text' in out:
                print(''.join(out['text']).rstrip())
        print()
"
```

Expected output (approximately):

```
=== realdata-load ===
handler.samples           = ['CV10']
handler.molecules.keys()  = ['SIH', 'Hyp', 'Ino']
n_chromatograms per sample = [7]

=== realdata-conditions ===
CV10 is_control       = False
CV10 initial conditions:
    SIH = 50.0
    Hyp = 0.0
    Ino = 0.0
Note: SAH, DTNB, SIHH columns were in the CSV but skipped ...

=== realdata-prepare ===
n_trace          = 7
is_control mask  = [False False False False False False False]
trace_ids        = ('CV10/...', 'CV10/...', ...)  # 7 entries
...

=== realdata-build-priors-error ===
ValueError raised (truncated):
Peak SIH: no control traces in dataset; cannot extract artefact priors. Available traces: ['CV10/...', 'CV10/...', ...]. Mark controls in the conditions CSV or switch annotation mode.
```

If the build_priors error message doesn't reference the trace IDs, the wiring of `dataset.trace_ids` into the "no controls" branch of `extract_artefact_from_controls` is missing — it was added for the mismatch and too-close branches but should also fire here. Verify by inspecting `chromhandler/fitting/priors.py:380–392`.

- [ ] **Step 10: Commit**

```bash
git add notebooks/priors_demo.ipynb
git commit -m "$(cat <<'EOF'
docs(notebooks): add real-data section to priors_demo using ASM fixture

Walks through loading the ASM kinetic series + conditions.csv with only
SIH/Hyp/Ino registered as molecules. Demonstrates:

- DTNB, SAH, SIHH columns silently filtered out of InitialCondition
  parsing and auto-detection
- CV10 correctly classified as a sample (not a control) because
  registered analyte SIH=50
- Handler.prepare_dataset producing real trace_ids of the form
  "CV10/<chrom_id>"
- build_priors raising a helpful error that names the available traces
  when the fixture has no control chromatograms

The section ends with explicit instructions for what's missing to make
the full doublet path work end-to-end (drop a CV4/CV5 control ASM JSON
into the fixtures directory).
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- ✅ Default behavior (no molecules registered) preserved — Task 1 Step 4 keeps `filter_active = False` when `self.molecules` is empty, falling through to the original column-iteration loop.
- ✅ Filter activates when molecules are registered — Task 1 Step 4 adds the `if filter_active and str(mol_id) not in registered_mols: continue` guard.
- ✅ Auto-detection uses only registered molecules — Task 1 Step 4: `declared_concs` is only appended inside the same loop, so the filter applies to both `add_initial_condition` calls and to the `is_control` check.
- ✅ Tests cover (a) filter behavior, (b) auto-detect with reagent present, (c) backwards-compat no-registration path, (d) edge case where filter empties the row entirely — Task 1 Step 2.
- ✅ Real-data demo on the new conditions.csv fixture — Task 2.

**Placeholder scan:** No "TBD" / "TODO" / "fill in later". Every step shows code. Cell IDs are explicit.

**Type consistency:**
- `set[str]` and `bool` for the two new locals in `load_initial_conditions`.
- `Handler.create_molecule(id, pubchem_cid)` is the registration API used in tests and notebook — verified against `chromhandler/handler.py:962`.
- `Handler.load_initial_conditions(path | DataFrame, conc_unit=str)` signature is unchanged.

**Out of scope (intentionally):**
- Adding a control chromatogram fixture (CV4/CV5 ASM JSON). The notebook's closing markdown explicitly tells the user how to add one and what's needed; the plan does not synthesize a fake one.
- A per-row override mechanism for `is_control` (e.g. an explicit `is_control` column in the CSV). Not requested.
- Changes to the priors module itself. Already handles `trace_ids` propagation from Task 3 of the earlier `2026-05-11-priors-module.md` plan.

**Dependencies cleared:**
- `Handler.create_molecule` exists at `chromhandler/handler.py:962` (verified).
- `Sample.is_control` field exists (added in `2026-05-11-handler-controls.md`).
- `PreparedDataset.trace_ids` field exists (added in the inline session work captured in commits `f6796c3` through `3c61c82`).
- `build_priors` "no controls" error path includes `Available traces: [...]` listing (verified in `chromhandler/fitting/priors.py`).
