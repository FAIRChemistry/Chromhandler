# Plan A: Prior Pipeline & Fitter Refactoring

**Goal**: Restructure `priors.py`, `data.py`, and `better_fitter.py` so the prior
pipeline produces `(w_left, w_right)` half-width priors instead of `(sigma, alpha)`,
cleans up area estimation, and simplifies the fitter input assembly — all targeted
at what the new model (Plan B) will consume.

**No backwards compatibility required** — this is a v1 rewrite.

**Prerequisite for**: Plan B (model reparameterization).

---

## Context: Current data flow

```
priors.py::_trace_fwhm_geometry()      → w_left, w_right per trace (MEASURED)
priors.py::_fwhm_to_sigma_alpha()      → sigma, alpha per trace (DERIVED, lossy)
priors.py::_shape_priors_from_fwhm()   → sigma_loc/scale, alpha_loc/scale (AGGREGATED)
priors.py::build_peak_priors()         → GeometricPeakPriors (sigma/alpha only, w_left/w_right discarded)
priors.py::geometric_priors_to_arrays()→ dict of numpy arrays for model
better_fitter.py::compute_model_inputs()  → adds baseline, noise, peak structure
better_fitter.py::_run_mcmc()          → converts to JAX, filters, runs NUTS
```

**Problem**: We measure `(w_left, w_right)` directly from FWHM, then immediately
convert to `(sigma, alpha)` and throw away the half-widths. The model (Plan B) needs
the half-widths. We also discard per-trace half-width vectors that carry S/N information.

---

## Step 1: Pass half-widths through `GeometricPeakPriors`

### 1.1 Add half-width fields to `GeometricPeakPriors`

**File**: `priors.py`

Replace `sigma_loc`, `sigma_scale`, `alpha_loc`, `alpha_scale` with:

```python
@dataclasses.dataclass(frozen=True)
class GeometricPeakPriors:
    mode: PeakMode
    apex_loc: float
    apex_scale: float

    # Half-width priors (from FWHM geometry)
    w_left_loc: float       # height-weighted mean left HWHM [time units]
    w_left_scale: float     # height-weighted std of left HWHM
    w_right_loc: float      # height-weighted mean right HWHM [time units]
    w_right_scale: float    # height-weighted std of right HWHM

    # Area priors
    area_gaussian_pt: NDArray[np.float64]   # [n_trace] Gaussian approx area
    area_trapz_pt: NDArray[np.float64]      # [n_trace] trapezoid integration
    area_art_shared: float                  # artefact shared area (0.0 if not artefact)

    # S/N per trace (for adaptive area prior width)
    snr_per_trace: NDArray[np.float64]      # [n_trace] apex_height / noise_estimate

    # Window metadata
    window_lo: float
    window_hi: float
    n_valid_traces: int
```

**Key changes**:
- `sigma_loc/scale` and `alpha_loc/scale` → `w_left_loc/scale` and `w_right_loc/scale`
- `main_area_per_trace` → `area_gaussian_pt` (name clarifies method)
- `total_area_per_trace` → `area_trapz_pt` (name clarifies method)
- `artefact_shoulder_area_loc` → `area_art_shared` (shorter)
- New: `snr_per_trace` — needed for S/N-dependent area prior width in the model

### 1.2 Modify `_shape_priors_from_fwhm()` → `_halfwidth_priors()`

**File**: `priors.py`

Instead of converting `(w_left, w_right)` → `(sigma, alpha)` and aggregating those,
aggregate `(w_left, w_right)` directly:

```python
def _halfwidth_priors(
    x_win: NDArray[np.float64],
    y_win: NDArray[np.float64],
    *,
    level: float = 0.5,
    min_height_frac: float = _MIN_APEX_HEIGHT_FRAC,
) -> tuple[float, float, float, float, _TraceFwhmGeometry]:
    """Height-weighted population priors for left/right half-widths.

    Returns (w_left_loc, w_left_scale, w_right_loc, w_right_scale, geometry).
    """
    geometry = _trace_fwhm_geometry(x_win, y_win, level=level, min_height_frac=min_height_frac)

    w_left_loc = _weighted_loc(geometry.w_left, geometry.apex_height, geometry.fwhm_valid)
    w_left_scale = _weighted_scale(geometry.w_left, geometry.apex_height, geometry.fwhm_valid, w_left_loc)
    w_right_loc = _weighted_loc(geometry.w_right, geometry.apex_height, geometry.fwhm_valid)
    w_right_scale = _weighted_scale(geometry.w_right, geometry.apex_height, geometry.fwhm_valid, w_right_loc)

    return w_left_loc, w_left_scale, w_right_loc, w_right_scale, geometry
```

**Delete**: `_fwhm_to_sigma_alpha()` — no longer needed in the prior pipeline.
(Keep it as a private utility if `better_visualize.py` still needs it for display.)

### 1.3 Add S/N estimation per trace

**File**: `priors.py`, inside `build_peak_priors()` loop

```python
# After computing apex_height per trace:
noise_est = float(np.median(np.abs(np.diff(y_win, axis=1)))) * 0.7071  # MAD of first differences
snr_per_trace = np.maximum(np.asarray(geometry.apex_height) / max(noise_est, 1e-12), 0.0)
```

This gives a quick per-trace signal-to-noise ratio without requiring baseline regions.

### 1.4 Update `build_peak_priors()` to use new fields

**File**: `priors.py`

Replace the `_shape_priors_from_fwhm` call with `_halfwidth_priors`. Wire new fields
into `GeometricPeakPriors` constructor. Drop `_main_peak_approximation()` — use the
simpler approach:

```python
# Area from FWHM geometry: A = height * sigma * sqrt(2*pi)
# where sigma = sqrt(0.5 * (w_left² + w_right²)) / HWHM_factor
# For traces without valid FWHM, fall back to cross-trace median.
sigma_trace = jnp.sqrt(0.5 * (geometry.w_left**2 + geometry.w_right**2)) / _GAUSSIAN_HWHM_FACTOR
area_gaussian = _GAUSSIAN_AREA_FROM_HEIGHT_SIGMA * geometry.apex_height * sigma_trace
area_gaussian = jnp.where(geometry.fwhm_valid, area_gaussian, jnp.nan)
# Fill NaN with cross-trace median
median_area = float(jnp.nanmedian(area_gaussian))
area_gaussian_pt = np.where(np.isfinite(area_gaussian), area_gaussian, median_area * 0.01)
```

**Delete**: `_main_peak_approximation()` (60 lines replaced by ~6 lines above).

### 1.5 Update `geometric_priors_to_arrays()`

**File**: `priors.py`

Output dict changes:

```python
{
    "apex_loc":          [n_peak],
    "apex_scale":        [n_peak],
    "w_left_loc":        [n_peak],
    "w_left_scale":      [n_peak],
    "w_right_loc":       [n_peak],
    "w_right_scale":     [n_peak],
    "window_lo":         [n_peak],
    "window_hi":         [n_peak],
    "area_gaussian_pt":  [n_trace, n_peak],
    "area_trapz_pt":     [n_trace, n_peak],
    "area_art_shared":   [n_artefact],
    "snr_per_trace":     [n_trace, n_peak],
}
```

Remove: `sigma_loc`, `sigma_scale`, `alpha_loc`, `alpha_scale`,
`dominant_area_loc_per_trace`, `area_total_loc_per_trace`, `artefact_area_loc_shared`.

---

## Step 2: Simplify area estimation

### 2.1 Remove `_stabilize_area_prior_matrix()`

**File**: `better_fitter.py`

This method patches non-positive areas after the fact. Instead, ensure `build_peak_priors()`
always returns positive areas (clamp at source in Step 1.4). Delete the static method.

### 2.2 S/N-dependent area prior width

The model (Plan B) will use `snr_per_trace` to set per-peak `area_log_sigma`:
- High S/N (> 10): tight prior, `area_log_sigma ≈ 0.3`
- Low S/N (< 3): wide prior, `area_log_sigma ≈ 0.8`
- Formula: `area_log_sigma = clip(0.8 - 0.05 * snr, 0.25, 0.8)`

This computation happens in the model (Plan B), but `priors.py` must provide `snr_per_trace`.

---

## Step 3: Clean up `data.py`

### 3.1 Rename `ModelHyperparams` fields

**File**: `data.py`

Remove obsolete fields related to `(sigma, alpha)`:

```python
@dataclasses.dataclass(frozen=True)
class ModelHyperparams:
    # Half-width prior scale multiplier (log-space)
    w_prior_log_scale: float = 0.4       # CV for log(w_left), log(w_right)

    # Area prior spread
    area_log_sigma_high_snr: float = 0.3  # tight for clear peaks
    area_log_sigma_low_snr: float = 0.8   # wide for ambiguous peaks
    area_snr_threshold_high: float = 10.0
    area_snr_threshold_low: float = 3.0

    # Artefact area
    area_art_log_sigma: float = 0.3
    area_art_trace_log_scale: float = 0.15

    # Free-doublet separation
    free_sep_loc_mult: float = 1.5    # typical separation in sigma units
    free_sep_log_sigma: float = 0.4

    # Artefact separation
    art_sep_min_w_mult: float = 0.5   # min separation in half-width units
    art_sep_max_window_frac: float = 0.5
```

Remove: `alpha_max`, `alpha_bound_eps`, `raw_alpha_scale_floor`, `area_log_sigma`,
`sh_area_log_sigma`, all the old `free_sep_*` fields.

---

## Step 4: Clean up `better_fitter.py`

### 4.1 Split `_run_mcmc()` into 3 methods

**File**: `better_fitter.py`

```python
def _prepare_model_inputs(self) -> dict[str, jax.Array]:
    """Compute priors + structure + baseline + noise → JAX arrays."""
    ...

def _run_nuts(self, model_inputs: dict, **mcmc_kwargs) -> MCMC:
    """Execute NUTS sampler. Returns MCMC object."""
    ...

def _process_posterior(self, mcmc: MCMC, model_inputs: dict) -> None:
    """Reconstruct derived quantities, build ArviZ InferenceData."""
    ...
```

### 4.2 Remove the `model_param_names` whitelist

**File**: `better_fitter.py`

Instead of maintaining a hardcoded set of allowed keys, use `inspect.signature`:

```python
import inspect
sig = inspect.signature(better_model.model)
model_param_names = set(sig.parameters.keys()) - {"hyperparams"}
```

### 4.3 Derive noise floor from data

**File**: `better_fitter.py`, `noise_prior()`

Replace `np.maximum(sigma_y, 1.0)` with:

```python
signal_range = np.ptp(self.signal, axis=1)
noise_floor = 1e-3 * np.maximum(signal_range, 1e-6)
return np.maximum(sigma_y, noise_floor)
```

### 4.4 Update `compute_model_inputs()` for new prior keys

Wire through the new key names from `geometric_priors_to_arrays()`. Drop
`_stabilize_area_prior_matrix` calls.

### 4.5 Clean up `fitting/__init__.py`

Remove the legacy EMG docstring. Keep only the current public API exports.

---

## Step 5: Update `summarise_priors()`

**File**: `priors.py`

Update the ASCII table to show `w_left`, `w_right` instead of `sigma`, `alpha`.
Add S/N column.

---

## Tests

### Unit tests: `tests/unit/fitting/test_priors.py` (rewrite)

```
test_halfwidth_priors_single_symmetric_peak
    → synthetic Gaussian, verify w_left ≈ w_right ≈ HWHM

test_halfwidth_priors_skewed_peak
    → synthetic skew-normal, verify w_left < w_right (or vice versa)

test_halfwidth_priors_multi_trace_aggregation
    → 5 traces with slightly different widths, verify loc ≈ mean, scale > 0

test_halfwidth_priors_low_snr_trace_excluded
    → 1 high-S/N trace + 1 noise-only trace, verify noise trace doesn't corrupt loc

test_snr_per_trace_computation
    → known signal + noise, verify snr_per_trace ≈ expected

test_snr_per_trace_all_noise
    → pure noise traces, verify snr_per_trace ≈ 0 (no crash)

test_area_gaussian_fallback_for_invalid_fwhm
    → trace with no valid FWHM crossing, verify area falls back to median

test_area_trapz_positive
    → verify trapezoid areas are always > 0 after clamp

test_build_peak_priors_single_mode
    → verify output dataclass has correct fields and shapes

test_build_peak_priors_artefact_mode
    → verify area_art_shared > 0 when residual exists

test_build_peak_priors_free_doublet_mode
    → verify shapes for free doublet (n_components=2)

test_geometric_priors_to_arrays_keys
    → verify output dict has exactly the expected keys

test_geometric_priors_to_arrays_shapes
    → n_peak=3, n_trace=5, verify all array shapes

test_refine_apex_with_trace_shift
    → 3 peaks, systematic per-trace drift, verify trace_shift_scale > 0
      and per-peak apex_scale reduced after refinement

test_summarise_priors_format
    → verify output is a non-empty string with header + data lines
```

### Unit tests: `tests/unit/fitting/test_data.py` (new)

```
test_model_hyperparams_defaults
    → verify all defaults are finite and positive

test_peak_mode_helpers
    → test peak_component_count, peak_is_doublet_mode, etc.

test_pad_traces_equal_length
    → verify no NaN padding when lengths match

test_pad_traces_unequal_length
    → verify NaN padding and shape

test_region_to_mask
    → verify mask for a known interval
```

### Unit tests: `tests/unit/fitting/test_fitter_inputs.py` (new)

```
test_compute_model_inputs_keys
    → create BetterFitter with synthetic data + 1 peak annotation,
      verify compute_model_inputs() returns all expected keys

test_compute_model_inputs_shapes
    → verify array shapes match n_trace, n_peak, n_artefact, n_free

test_noise_prior_data_derived_floor
    → verify noise floor scales with signal range, not hardcoded

test_observation_mask_covers_peaks_and_baselines
    → verify mask is True inside annotated regions, False outside

test_slice_to_observed_windows_shape
    → verify output shape matches mask sum
```

### Integration test: `tests/integration/test_prior_pipeline.py` (new)

```
test_prior_pipeline_on_real_data
    → load SAHH data (skip if not available), run full prior pipeline,
      verify:
      - w_left_loc, w_right_loc > 0 for all peaks
      - w_left_loc < window_width for all peaks
      - snr_per_trace has correct shape
      - area_gaussian_pt > 0 for all (trace, peak) pairs
      - area_art_shared > 0 for artefact peaks

test_prior_pipeline_round_trip_consistency
    → synthetic data with known skew-normal peaks,
      verify w_left_loc, w_right_loc recover the true half-widths (within 15%)
```

---

## Files modified

| File | Action |
|------|--------|
| `chromhandler/fitting/priors.py` | Major rewrite: half-width priors, S/N, drop sigma/alpha |
| `chromhandler/fitting/data.py` | Update `ModelHyperparams`, rename fields |
| `chromhandler/fitting/better_fitter.py` | Split _run_mcmc, remove whitelist, update keys |
| `chromhandler/fitting/__init__.py` | Clean up legacy docstring |
| `tests/unit/fitting/test_priors.py` | Full rewrite |
| `tests/unit/fitting/test_data.py` | New |
| `tests/unit/fitting/test_fitter_inputs.py` | New |
| `tests/integration/test_prior_pipeline.py` | New |

## Files NOT modified

| File | Reason |
|------|--------|
| `better_model.py` | Plan B |
| `better_visualize.py` | Deferred — separate effort after Plans A+B |
| `shift.py` | Independent, no changes needed |
| `baseline.py` | Solid as-is |

## Cleanup: delete legacy code

- `fitting/__init__.py`: Remove entire legacy EMG docstring and dead imports.
  Keep only current public API: `BetterFitter`, `AreaRecord`, `PosteriorCurves`, `ModelHyperparams`.
- `priors.py`: Delete `_fwhm_to_sigma_alpha()` entirely (no longer used anywhere).
