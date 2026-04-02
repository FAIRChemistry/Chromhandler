# Data-Flow Analysis: Handler → BetterFitter → priors.py → better_model.py → Handler

Generated: 2026-03-28.

---

## 1. Pipeline Diagram (ASCII)

```
Handler
  .samples[]                 list[Sample] → list[Chromatogram]
  .chromatograms[].time      list[float]
  .chromatograms[].signal    list[float]
         │
         │  BetterFitter.from_handler(handler)   ← caller pre-selects traces
         │  stack_and_pad_signal()   — NaN-pads to rectangular shape
         ▼
BetterFitter
  .time   [n_trace, n_time]  numpy float64, NaN-padded
  .signal [n_trace, n_time]  numpy float64, NaN-padded
  .peaks  list[PeakAnnotation]        (via add_peak_annotation)
  .baselines list[BaselineAnnotation] (via add_baseline_annotation)
  .trace_sample_ids [n_trace] object array of str
  .trace_chromatogram_ids [n_trace] object array of str
         │
         │  fit()  → _run_mcmc()
         │
         │  compute_model_inputs()
         │    ├── _compute_position_priors()  → (list[GeometricPeakPriors], trace_shift_scale)
         │    │     ├── baseline_signal()      → [n_trace, n_time]  OLS baseline
         │    │     ├── build_geometric_priors()    [priors.py]
         │    │     ├── compute_fwhm_shape_diagnostics()  [priors.py]
         │    │     └── refine_apex_priors_with_trace_shift()
         │    │
         │    ├── geometric_priors_to_arrays()   → dict, axes [n_peak, n_trace]
         │    │     (dominant_area_loc_per_trace, area_total_loc_per_trace)
         │    │
         │    ├── .T transpose on area matrices  → [n_trace, n_peak]
         │    ├── _stabilize_area_prior_matrix()
         │    ├── peak_structure()    → index arrays for artefact/free/nonfree
         │    ├── baseline_priors()   → BaselinePriors  [baseline.py]
         │    └── noise_prior()       → [n_trace]
         │
         │  slice_to_observed_windows()  → [n_trace, n_masked_time] (x, y)
         │
         │  numpy → JAX conversion  (bulk loop in _run_mcmc)
         ▼
better_model.model(x, y, ...) — NumPyro MCMC
  posterior latents: area_l, area_r  [n_trace, n_peak]
                     apex_l, apex_r  [n_trace, n_peak]
                     sigma_l/r, alpha_l/r, xi_l/r, baseline_curve, ...
         │
         │  arviz.from_numpyro()   → InferenceData stored in fitter._posterior
         │
         │  to_peaks() / area_records()
         │    → _peaks_from_samples() / _records_from_samples()
         │    → list[Peak(Estimate)] / list[AreaRecord]
         ▼
handler.write_fitted_peaks(fitter)
  → upserts Peak objects into Chromatogram.peaks[]
  → Peak.area   is Estimate{mean, median, std, q05, q95, samples[]}
  → Peak.location is Estimate{...} (posterior apex)

handler.collect_areas(fitter)
  → area_records() → joins chromatogram_id → reaction_time
  → dict[molecule_id, list[tuple[reaction_time, area_median]]]
```

---

## 2. Per-Stage Findings

### Stage 1: Handler → BetterFitter.from_handler()

**Produces:** `BetterFitter` with `time [n_trace, n_time]`, `signal [n_trace, n_time]`, numpy float64, NaN-padded.

**Consumes:** `handler.samples[].chromatograms[].time` (Python `list[float]`), same for `.signal`.

**Findings:**

1. **Redundant JAX roundtrip in `stack_and_pad_signal`.** The function constructs arrays via `jnp.array(list_of_lists)` then `from_handler` immediately calls `np.asarray(...)` on the result. This roundtrip is unnecessary: a pure numpy implementation would be simpler and avoid loading JAX for a padding operation.
2. **`stack_and_pad_signal` return type mismatch.** It returns `(jnp.ndarray, jnp.ndarray)` but every caller immediately converts to numpy. The function should return numpy arrays to match the fitter's numpy-first design.

**Simplification proposals:**

- Replace the JAX-based implementation in `stack_and_pad_signal` with pure numpy.

---

### Stage 2: BetterFitter → priors.py (build_ge ometric_priors + compute_fwhm_shape_diagnostics)

**Produces:** `list[GeometricPeakPriors]` and `FwhmShapeDiagnostics`.

**Consumes:** `peaks: list[PeakAnnotation]`, `x [n_time]`, `signal [n_trace, n_time]`, `baseline [n_trace, n_time]`.

**Findings:**

1. **Double computation of FWHM geometry.** `_compute_position_priors` calls both `build_geometric_priors` AND `compute_fwhm_shape_diagnostics` on the same window data. Both internally call `_trace_fwhm_geometry` and `_fwhm_to_sigma_alpha` on each peak window. `build_geometric_priors` also calls `_shape_priors_from_fwhm` → `_trace_fwhm_geometry` and `_main_peak_approximation` → `_fwhm_to_sigma_alpha`. Total: `_trace_fwhm_geometry` is called 3 times per peak window per `_compute_position_priors` invocation.
2. **`FwhmShapeDiagnostics` is mostly unused.** It is a large dataclass (15 fields, `[n_trace, n_peak]` arrays each) but only `fwhm_apex_trace` and `fwhm_valid_trace` are accessed by `refine_apex_priors_with_trace_shift`. The other 13 fields are computed and immediately abandoned.
3. **`apex_scale` divided by 4 is semantically dead.** `build_geometric_priors` hardcodes `apex_scale = apex_scale_legacy / 4` (marked "legacy" in a comment). `refine_apex_priors_with_trace_shift` then unconditionally overwrites this value for all traces with ≥2 valid residuals, making the `/4` division a no-op in normal usage.
4. **`baseline_signal()` → `baseline_priors()` call chain makes the cache implicit.** `_compute_position_priors` calls `baseline_signal()`, which calls `baseline_priors()`, which caches the OLS result under `_bp_direct`. `compute_model_inputs` then calls `baseline_priors()` again (cache hit). The cache dependency is correct but the two-layer call makes it non-obvious.

**Simplification proposals:**

- Merge `build_geometric_priors` and `compute_fwhm_shape_diagnostics` into one function returning both outputs, computing FWHM geometry once per window.
- Move the `/4` into `refine_apex_priors_with_trace_shift` or remove it; let that function own the final scale value unconditionally.
- Reduce `FwhmShapeDiagnostics` to the two fields actually consumed, or make the full diagnostics opt-in.

---

### Stage 3: priors.py → geometric_priors_to_arrays → compute_model_inputs (axis transposition)

**Produces:** dict with `dominant_area_loc_per_trace [n_peak, n_trace]` and `area_total_loc_per_trace [n_peak, n_trace]`.

**Consumes (by model):** `dominant_area_loc_per_trace [n_trace, n_peak]`.

**This is the most concrete data-structure bug in the pipeline.**

`geometric_priors_to_arrays` stacks `p.main_area_per_trace` (shape `[n_trace]`) across peaks, producing `[n_peak, n_trace]`. But `model()` expects `[n_trace, n_peak]`. `compute_model_inputs` applies `.T` to both matrices to correct the axis order. The fix is technically correct but applied silently in the middle of `compute_model_inputs`, not at the boundary where the mismatch originates.

**Concrete evidence:**

- `geometric_priors_to_arrays` docstring: `dominant_area_loc_per_trace [n_peak, n_trace]`
- `model()` signature: `dominant_area_loc_per_trace: jax.Array,  # [n_trace, n_peak]`
- `compute_model_inputs` lines 799–804: `.T` applied to both area arrays

**Consequence of not fixing:** If `geometric_priors_to_arrays` output is ever fed directly to `model()` without the `.T`, per-trace area priors are silently scrambled across peaks.

**Simplification proposal:**

- Change `geometric_priors_to_arrays` to return `[n_trace, n_peak]` directly and remove the `.T` calls from `compute_model_inputs`.

---

### Stage 4: peak_structure() → model index arrays

**Produces:** `artefact_peak_index [n_artefact]`, `free_peak_index [n_free]`, `nonfree_peak_index [n_nonfree]`.

**Consumes (by model):** All three index arrays separately.

**Findings:**

1. **`nonfree_peak_index` is derivable from `free_peak_index`.** It is simply the complement within `[0, n_peak)`. Passing it separately requires the caller and model to independently agree on the same definition, creating a consistency risk if a new peak mode is added.
2. **`artefact_peak_index` is always a subset of `nonfree_peak_index`.** The model uses both simultaneously creating a two-step index chain (`artefact_peak_index` → absolute positions, `artefact_nonfree_idx` → positions within `area_dominant`). Correct but requires understanding the two coordinate systems.
3. **`free_vary_local` and `free_fixed_local` are dead computation.** Built inside `peak_structure()` but never returned and never used.
4. **`vary_separation` attribute may be undefined.** `peak_structure` reads `peak.vary_separation` on `free_doublet` peaks, but this field is not visible in the `PeakAnnotation` definition. If absent, any `free_doublet` fit will raise `AttributeError`.

**Simplification proposals:**

- Remove `nonfree_peak_index` from the `model()` signature; derive it inside the model from `free_peak_index` and `n_peak`.
- Remove the dead `free_vary_local` / `free_fixed_local` computation.
- Verify `vary_separation` is defined on `PeakAnnotation` or remove the read.

---

### Stage 5: model() → posterior → to_peaks() / area_records()

**Produces:** `list[Peak]` with `Estimate` area/location; `list[AreaRecord]`.

**Consumes:** `self.samples` — a dict of `{param: numpy_array}` from MCMC samples.

**Findings:**

1. **`area_l`/`area_r` naming is left/right, not dominant/secondary.** For a `single` peak, `area_r` is always zero by construction. The left/right naming reflects the model's internal geometry but is confusing for single-component peaks and for artefact peaks where "dominant" vs "artefact" is the meaningful distinction.
2. **`_molecule_area_slice` logic is repeated across four methods.** `to_peaks`, `area_records`, `molecule_areas`, and `posterior_area_matrix` each dispatch on peak mode slightly differently. The `@staticmethod` exists to centralize this, but `molecule_areas` still duplicates the loop logic from `_peaks_from_samples`.
3. **`PosteriorCurves` is visualization-only.** `posterior_curves()` produces `PosteriorCurves` which is not consumed by any handler method (`write_fitted_peaks` uses `to_peaks()`, `collect_areas` uses `area_records()`). Fine, but worth marking explicitly so it is not confused for a pipeline intermediate.
4. **Full posterior is discarded by default.** `to_peaks` embeds raw samples in `Estimate.samples` only when `n_samples` is passed. The default `n_samples=None` means downstream calibration only sees point estimates, losing posterior uncertainty for propagation.
5. **`_peaks_from_samples` uses unseeded `np.random.choice`.** When `n_samples` is not None, subsampling is non-reproducible. No seed or RNG parameter is accepted.

**Simplification proposals:**

- Rename `area_l`/`area_r` in model outputs to `area_dominant`/`area_secondary`, remapping to left/right only in the assembly deterministics.
- Add a `seed` parameter to `to_peaks` → `_peaks_from_samples` for reproducible sample embedding.

---

### Stage 6: to_peaks() → Handler.write_fitted_peaks → Chromatogram.peaks

**Produces:** `Chromatogram.peaks[]` updated with `Peak(area=Estimate, location=Estimate)`.

**Consumes:** `list[Peak]` from `to_peaks()`, joined to chromatograms by `chromatogram_id`.

**Findings:**

1. **`Peak.chromatogram_id` is an untyped string join.** If a chromatogram is renamed after fitting, the link silently breaks with no type-system protection.
2. **`write_fitted_peaks` upserts on `molecule_id` with no provenance flag.** Peaks from Bayesian fitting and from the legacy peak-detection path both live in `Chromatogram.peaks[]` with no field distinguishing them.
3. **`collect_areas` discards uncertainty.** It uses `area_records()` (median only). The `Estimate.std`/`q05`/`q95` fields are available but the return type `dict[str, list[tuple[float, float]]]` carries only `(reaction_time, area_median)` pairs.

---

## 3. Cross-Cutting Issues

### 3a. numpy / JAX conversion multiplicity

The flow performs these dtype/framework conversions on the signal data:

1. `Chromatogram.signal: list[float]` → `jnp.array(...)` in `stack_and_pad_signal`
2. `jnp.ndarray → np.asarray(...)` immediately after in `from_handler`
3. `np.ndarray → jnp.asarray(...)` in `baseline_priors` (for `estimate_baseline`)
4. `np.ndarray → jnp.asarray(...)` in `compute_fwhm_shape_diagnostics`
5. `np.ndarray → jnp.asarray(...)` in `build_geometric_priors`
6. `np.ndarray → jnp.float32` in the bulk loop in `_run_mcmc`
7. JAX posterior samples → `np.asarray(...)` in `_peaks_from_samples`

Steps 1–2 are a no-op roundtrip. Steps 3–5 repeatedly convert the same numpy arrays to JAX for different prior functions. All prior computation in `priors.py` and `baseline.py` is pure math with no JAX `jit`/`vmap` — it could all run in numpy, eliminating conversions 1–5.

### 3b. `apex_scale` is silently dropped

`GeometricPeakPriors.apex_scale` and the `apex_scale [n_peak]` array from `geometric_priors_to_arrays` are present in the dict passed through `compute_model_inputs`, but `_run_mcmc` filters inputs to `model_param_names` which does not include `apex_scale`. The value is computed through the full prior pipeline and then discarded without any indication at the call site.

### 3c. `sigma_loc` / `sigma_scale` naming overload

`sigma_loc` / `sigma_scale` are the prior center/spread for the skew-normal shape parameter. `sigma_y_prior_loc` is the prior center for observation noise. The `sigma_y` variable inside the model is the sampled noise. The `sigma` prefix covers three distinct concepts — peak shape, noise prior, and noise sample — with no consistent convention.

### 3d. Baseline intercept re-centering is invisible to callers

`estimate_baseline` returns `intercept` at `x=0`. The model re-centers at `x_mid = 0.5 * (min(window_lo) + max(window_hi))` internally (transforming the OLS intercept silently). The sampled `baseline_intercept` therefore represents the baseline level at `x_mid`, not at `x=0`. `BaselinePriors` and `estimate_baseline` carry no annotation about this transformation.

---

## 4. Prioritised Refactoring Actions

### P1 (Correctness / Clarity — fix first)

1. **Fix the axis-order contract in `geometric_priors_to_arrays`.** Change the return to `[n_trace, n_peak]` directly and remove the `.T` in `compute_model_inputs`. This removes the silent axis flip that would cause wrong results if the transpose step were ever skipped.
2. **Verify `PeakAnnotation.vary_separation` exists.** The attribute is read in `peak_structure()` for `free_doublet` peaks but is not visible in any dataclass definition. If absent, every `free_doublet` fit will raise `AttributeError`.
3. **Remove dead computation in `peak_structure`.** `free_vary_local` and `free_fixed_local` are built but never returned or used.

### P2 (Performance / Redundancy — medium priority)

1. **Merge `build_geometric_priors` + `compute_fwhm_shape_diagnostics` into one function.** Both share the same inner loop and call identical sub-functions. Merging eliminates 2 out of 3 redundant FWHM geometry computations per peak window per call.
2. **Eliminate the JAX roundtrip in `stack_and_pad_signal`.** Return `np.ndarray` instead of `jnp.ndarray`.
3. **Drop `apex_scale` from `geometric_priors_to_arrays` output**, or add it to `model_param_names`. Currently computed through the full prior pipeline and silently discarded.

### P3 (Design / Simplification — lower priority)

1. **Rename `dominant_area_loc_per_trace` / `artefact_shoulder_area_loc` in `GeometricPeakPriors`.** `artefact_shoulder_area_loc` is `0.0` for non-artefact peaks. Consider `Optional[float]` with `None` to make the sentinel explicit.
2. **Make baseline re-centering explicit.** Document in `BaselinePriors` that `intercept` is the OLS value at `x=0` and that `model()` will transform it to `x_mid`. Or recentre in `estimate_baseline` and pass the midpoint as a field.
3. **Add `reaction_time` to the fitter.** Store `trace_reaction_times: NDArray | None` in `from_handler` so `collect_areas`-style queries do not require a round-trip through the handler's chromatogram index.
4. **Seed `np.random.choice` in `_peaks_from_samples`.** Pass a `seed` or `rng` parameter through `to_peaks` → `_peaks_from_samples` for reproducible sample embedding.
5. **Clean up `subsets.py`.** `AreaRecord.subset_name` is always `""` — consider removing the field. `SubsetSpec` is a dead deprecated class that can be deleted.
