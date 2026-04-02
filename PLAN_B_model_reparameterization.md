# Plan B: Model Reparameterization

**Goal**: Rewrite `better_model.py` to sample `(log_w_left, log_w_right)` instead of
`(log_sigma, alpha_raw)`, simplify separation priors, flatten component assembly, and
unify `model()` / `compute_derived_quantities()`.

**Depends on**: Plan A (prior pipeline provides `w_left_loc/scale`, `w_right_loc/scale`,
`snr_per_trace`).

---

## Context: Current model issues

1. **Sigma/alpha banana**: `(sigma, alpha)` are jointly non-identifiable — many
   `(sigma, alpha)` pairs produce the same peak shape. NUTS wastes time traversing
   curved ridges.
2. **LogUniform ignores measured spread**: Sigma uses `Uniform(log(0.5*sigma_loc), log(2*sigma_loc))`,
   discarding the FWHM-derived scale information entirely.
3. **Area prior too rigid**: Fixed `area_log_sigma = 0.4` regardless of S/N.
4. **Separation over-engineered**: Three different parameterizations (Uniform, sigmoid-bounded).
5. **200+ lines of left/right assembly** duplicated between `model()` and
   `compute_derived_quantities()`.

---

## Step 1: Core reparameterization — `(log_w_left, log_w_right)`

### 1.1 New sampled parameters

Replace `log_sigma_base` and `alpha_raw_base` with:

```python
# One per peak, shared across traces (column chemistry is constant)
log_w_left = numpyro.sample(
    "log_w_left",
    dist.Normal(jnp.log(w_left_loc), w_left_log_scale),
)  # [n_peak]

log_w_right = numpyro.sample(
    "log_w_right",
    dist.Normal(jnp.log(w_right_loc), w_right_log_scale),
)  # [n_peak]
```

Where `w_left_log_scale` and `w_right_log_scale` are derived from the prior pipeline:

```python
# In _prepare_model_inputs or at top of model():
w_left_log_scale = jnp.maximum(w_left_scale / w_left_loc, hyperparams.w_prior_log_scale)
w_right_log_scale = jnp.maximum(w_right_scale / w_right_loc, hyperparams.w_prior_log_scale)
```

The CV from FWHM measurement → log-space scale, with a floor from hyperparams.

### 1.2 Deterministic conversion to `(sigma, alpha)`

Inside `model()`, after sampling, convert via the Gaussian-HWHM approximation:

```python
w_left = jnp.exp(log_w_left)    # [n_peak]
w_right = jnp.exp(log_w_right)  # [n_peak]

# Gaussian HWHM factor: HWHM = sigma * sqrt(2*ln(2))
HWHM_FACTOR = jnp.sqrt(2.0 * jnp.log(2.0))

s_left = w_left / HWHM_FACTOR     # left-side sigma
s_right = w_right / HWHM_FACTOR   # right-side sigma

sigma = jnp.sqrt(0.5 * (s_left**2 + s_right**2))
delta = (s_right - s_left) / jnp.maximum(s_right + s_left, 1e-12)
delta = jnp.clip(delta, -0.95, 0.95)
alpha = delta / jnp.sqrt(jnp.maximum(1.0 - delta**2, 1e-8))
```

These are local variables (not `numpyro.deterministic`), same as current design.

### 1.3 Doublet second-component half-widths

For doublets (artefact or free), the second component gets its own sampled half-widths:

```python
if n_doublet > 0:  # artefact + free combined
    log_w_left_2 = numpyro.sample(
        "log_w_left_2",
        dist.Normal(jnp.log(w_left_loc_2), w_left_log_scale_2),
    )  # [n_doublet]
    log_w_right_2 = numpyro.sample(
        "log_w_right_2",
        dist.Normal(jnp.log(w_right_loc_2), w_right_log_scale_2),
    )  # [n_doublet]
```

**Prior centers for doublet second component** (computed in Plan A's prior pipeline):
- **Artefact**: `w_loc_2 = 0.5 * w_obs` with wide scale (0.5 in log-space).
  The dominant component keeps the observed half-widths; the artefact is
  "probably narrower, but uncertain."
- **Free**: Both components get `w_loc = 0.6 * w_obs` with moderate scale (0.4).
  Neither is privileged; the data resolves which is wider.

### 1.4 Delete removed infrastructure

Remove entirely:
- `_bounded_alpha_prior_to_raw()` — no longer needed (no alpha transform)
- `_broadcast_peak_to_traces()` — folded into flattened assembly
- `alpha_max`, `alpha_bound_eps`, `raw_alpha_scale_floor` from `ModelHyperparams`

---

## Step 2: Flatten component assembly

### 2.1 Replace left/right canonical form with flat component list

The current model builds 8 parallel `[n_trace, n_peak]` matrices (apex_l, apex_r,
sigma_l, sigma_r, ...) then stacks them. This requires 3 assembly functions
(`_assemble_nonfree`, `_assemble_artefact`, `_assemble_free`) totaling 120 lines.

**New approach**: Build a flat component array `[n_component]` where
`n_component = n_single + 2*n_doublet`. Map back to peaks via an index array.

```python
# Pre-computed by fitter (or at top of model):
# comp_to_peak: [n_component] → which peak this component belongs to
# comp_is_primary: [n_component] → bool, True for primary (or single) component
# comp_is_left: [n_component] → bool, True if this component is the earlier-eluting one

# For each component, sigma and alpha come from its own (w_left, w_right):
sigma_comp = ...  # [n_component]
alpha_comp = ...  # [n_component]
apex_comp = ...   # [n_trace, n_component]  (apex ± separation/2)
area_comp = ...   # [n_trace, n_component]
```

Then the signal is simply:
```python
xi_comp = apex_comp - sigma_comp[None, :] * delta_comp[None, :] * SQRT_2_OVER_PI
mu_y = mixture_signal(x, xi_comp, sigma_comp, alpha_comp, area_comp) + baseline
```

No left/right stacking, no `_stack_left_right()`.

### 2.2 Component index arrays (computed by fitter)

`better_fitter.py::peak_structure()` should produce:

```python
{
    "comp_to_peak": np.array([0, 1, 1, 2, 2, 3], dtype=np.int32),  # example: peak 0 single, 1-2 doublet, 3 single... wait no. peak 1 = doublet → 2 comps, etc.
    "comp_is_primary": np.array([True, True, False, True, False, True]),
    "n_comp_per_peak": np.array([1, 2, 2, 1], dtype=np.int32),
    "doublet_comp_index": np.array([2, 4], dtype=np.int32),  # indices of secondary components
}
```

This replaces the current `nonfree_idx`, `nonfree_position`, `artefact_peak_index`,
`free_peak_index` — all of which exist only to support the left/right assembly.

---

## Step 3: Separation priors — simplify

### 3.1 Unified separation sampling

Replace the three current approaches (Uniform for artefact, sigmoid-bounded Normal for free)
with a single `LogNormal` prior:

```python
if n_doublet > 0:
    log_separation = numpyro.sample(
        "log_separation",
        dist.Normal(jnp.log(sep_loc), sep_log_scale),
    )  # [n_doublet]
    separation = jnp.exp(log_separation)
```

**Prior centers** (from Plan A):
- **Artefact**: `sep_loc = 1.0 * w_left_loc` (typical artefact is ~1 half-width away),
  `sep_log_scale = 0.5` (wide).
- **Free**: `sep_loc = 1.5 * sigma_loc` (derived from `w_left_loc, w_right_loc`),
  `sep_log_scale = 0.4`.

LogNormal is positive by construction (no bounds needed), peaked at the expected value,
and has a natural log-space geometry that NUTS handles well.

### 3.2 Delete `_bounded_separation_prior_to_raw()`

No longer needed — LogNormal replaces the sigmoid-bounded parameterization.

---

## Step 4: S/N-dependent area prior

### 4.1 Per-peak, per-trace area_log_sigma

Instead of a single `area_log_sigma = 0.4` for all peaks and traces:

```python
# snr_pt: [n_trace, n_peak] from prior pipeline
area_log_sigma = jnp.clip(
    hyperparams.area_log_sigma_low_snr
    - (hyperparams.area_log_sigma_low_snr - hyperparams.area_log_sigma_high_snr)
    * (snr_pt - hyperparams.area_snr_threshold_low)
    / (hyperparams.area_snr_threshold_high - hyperparams.area_snr_threshold_low),
    hyperparams.area_log_sigma_high_snr,
    hyperparams.area_log_sigma_low_snr,
)  # [n_trace, n_peak]
```

This linearly interpolates between 0.3 (high S/N) and 0.8 (low S/N).

### 4.2 Area prior remains LogNormal (no shrinkage to zero)

Keep `dist.LogNormal(log(area_loc), area_log_sigma)`. As discussed: shrinkage-to-zero
creates a geometric degeneracy where the sampler slides peaks to flat baseline regions.
The wider prior for low-S/N peaks gives enough room for the data to correct inflated
FWHM-based area estimates.

---

## Step 5: Unify `model()` and `compute_derived_quantities()`

### 5.1 Factor shared math into pure functions

```python
def _halfwidths_to_shape(
    log_w_left: jax.Array,
    log_w_right: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Convert log half-widths to (sigma, alpha, delta). Works on any batch shape."""
    ...

def _build_component_params(
    sigma_primary: jax.Array,
    alpha_primary: jax.Array,
    sigma_secondary: jax.Array | None,
    alpha_secondary: jax.Array | None,
    apex: jax.Array,
    separation: jax.Array | None,
    area_primary: jax.Array,
    area_secondary: jax.Array | None,
    comp_to_peak: jax.Array,
    doublet_comp_index: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Assemble flat (xi, sigma, alpha, area) component arrays."""
    ...
```

Both `model()` and `compute_derived_quantities()` call these same functions.
This eliminates the 240-line duplication.

### 5.2 Simplify `compute_derived_quantities()`

With flat component arrays, the reconstruction is straightforward:

```python
def compute_derived_quantities(samples, model_inputs, hyperparams):
    log_w_left = samples["log_w_left"]       # [n_total, n_peak]
    log_w_right = samples["log_w_right"]
    sigma, alpha, delta = _halfwidths_to_shape(log_w_left, log_w_right)
    # ... component assembly via _build_component_params ...
    return {"sigma": sigma, "alpha": alpha, "w_left": w_left, "w_right": w_right, ...}
```

---

## Step 6: Update `SUMMARY_PARAMETER_NAMES`

```python
SUMMARY_PARAMETER_NAMES = (
    "trace_shift",
    "apex",
    "log_w_left",
    "log_w_right",
    "log_w_left_2",
    "log_w_right_2",
    "log_separation",
    "area_left",
    "area_right",
    "area_total",
    "baseline_intercept",
    "baseline_slope",
    "sigma_y",
)
```

---

## Step 7: Update `better_fitter.py` post-sampling code

### 7.1 `_process_posterior()` (from Plan A split)

Update to work with new sample keys (`log_w_left`, `log_w_right` instead of
`log_sigma_base`, `alpha_raw_base`).

### 7.2 `posterior_curves()`

The skew-normal PDF evaluation stays the same — it still takes `(xi, sigma, alpha, area)`.
The only change is how those are extracted from samples: via `compute_derived_quantities()`
which now produces `sigma`, `alpha` from the half-widths.

Update the sample key references: `sigma_l/r` → component-based indexing.

### 7.3 `_molecule_area_slice()` and area extraction

Update to use flat component indexing instead of `area_l` / `area_r`.
The new keys are `area_left` and `area_right`.

---

## Tests

### Unit tests: `tests/unit/fitting/test_model_math.py` (new)

```
test_halfwidths_to_shape_symmetric
    → w_left == w_right → alpha ≈ 0, sigma ≈ w / HWHM_factor

test_halfwidths_to_shape_right_tailing
    → w_right > w_left → alpha > 0

test_halfwidths_to_shape_left_tailing
    → w_left > w_right → alpha < 0

test_halfwidths_to_shape_extreme_asymmetry
    → w_right = 3 * w_left → alpha clipped, no NaN

test_halfwidths_to_shape_batch_dims
    → input [5, 3] → output shapes [5, 3] for sigma, alpha, delta

test_skew_normal_pdf_integrates_to_one
    → numerical integration of skew_normal_pdf ≈ 1.0

test_mixture_signal_single_component
    → single component mixture matches skew_normal_pdf * area

test_mixture_signal_two_components_additive
    → sum of two components = mixture output

test_build_component_params_single_peak
    → 1 peak, 1 component, verify shapes

test_build_component_params_mixed_modes
    → 1 single + 1 doublet → 3 components, verify indexing
```

### Unit tests: `tests/unit/fitting/test_model_sampling.py` (new)

These use `numpyro.handlers.seed` + `numpyro.handlers.trace` to verify the
model runs without errors and produces correct sample shapes (no MCMC needed).

```
test_model_prior_predictive_single_peak
    → 1 single peak, verify model() runs and produces y with correct shape

test_model_prior_predictive_artefact_doublet
    → 1 artefact doublet, verify separation and area samples exist

test_model_prior_predictive_free_doublet
    → 1 free doublet, verify both component areas sampled

test_model_prior_predictive_mixed
    → 1 single + 1 artefact + 1 free, verify all sample site shapes

test_model_no_nan_in_prior_predictive
    → sample 100 draws, verify no NaN in predicted y

test_snr_dependent_area_prior_width
    → high-snr peak: verify area samples have lower variance than low-snr peak
      (using prior predictive, not MCMC — just checking the prior spread)
```

### Unit tests: `tests/unit/fitting/test_derived_quantities.py` (new)

```
test_compute_derived_matches_model
    → run model with seed, extract local variables via numpyro.handlers.trace,
      then run compute_derived_quantities on the same samples,
      verify sigma, alpha, apex, area arrays match exactly

test_compute_derived_batch_shape
    → n_total=10, n_trace=3, n_peak=2 (1 single + 1 doublet),
      verify all output shapes are [n_total, n_trace, n_peak] or [n_total, n_peak]
```

### Integration test: `tests/integration/test_fitting_speedup.py` (update)

Update existing test to work with new model. Key changes:

```
- sample keys: log_sigma_base → log_w_left, log_w_right
- derived keys: area_l/area_r → area_left/area_right
- Rhat check: update key names
- Shape check: update expected dimensions

test_fit_completes                          → unchanged logic
test_area_rhat_below_threshold              → update key names
test_samples_contain_derived_keys           → update expected key set
test_area_shape                             → update for flat component indexing
```

### Integration test: `tests/integration/test_parameter_recovery.py` (new)

Synthetic data with known ground truth — the gold standard for validating
the reparameterized model.

```
test_single_peak_parameter_recovery
    → Generate synthetic data: 5 traces, 1 skew-normal peak with known
      (w_left=0.04, w_right=0.06, area=[100, 90, 80, 70, 60], apex=3.0).
    → Fit with BetterFitter.
    → Verify posterior medians recover true parameters within 10%.
    → Verify 90% credible intervals contain true values.

test_doublet_deconvolution_recovery
    → Generate synthetic data: 5 traces, 2 overlapping skew-normals.
      Trace 1: component A=100, B=0.
      Trace 3: A=50, B=50.
      Trace 5: A=0, B=100.
    → Fit as free_doublet.
    → Verify recovered areas track the true mixing ratios.
    → Verify component shapes (w_left, w_right) recovered within 20%.

test_artefact_separation_recovery
    → Generate synthetic data: dominant peak + small artefact shoulder.
    → Fit as artefact_doublet.
    → Verify separation recovered within 20%.
    → Verify artefact area << dominant area.
```

These tests use small data (50 timepoints, 5 traces) and few MCMC samples
(200 warmup, 200 draws, 1 chain) to run fast (~5-10s each).

---

## Files modified

| File | Action |
|------|--------|
| `chromhandler/fitting/better_model.py` | Full rewrite: (w_left, w_right), flat components, unified math |
| `chromhandler/fitting/data.py` | Update `ModelHyperparams` (Plan A starts this, Plan B finishes) |
| `chromhandler/fitting/better_fitter.py` | Update post-sampling code, area extraction, posterior_curves |
| `tests/unit/fitting/test_model_math.py` | New |
| `tests/unit/fitting/test_model_sampling.py` | New |
| `tests/unit/fitting/test_derived_quantities.py` | New |
| `tests/integration/test_fitting_speedup.py` | Update key names and shapes |
| `tests/integration/test_parameter_recovery.py` | New |

## Files NOT modified

| File | Reason |
|------|--------|
| `priors.py` | Done in Plan A |
| `better_visualize.py` | Deferred |
| `shift.py` | Independent |
| `baseline.py` | No changes needed |

---

## Execution order

1. Plan A first — produces the new prior arrays the model needs.
2. Plan B second — consumes those arrays in the reparameterized model.
3. After both: update `better_visualize.py` (separate effort).

Within Plan B, the step order matters:
1. Step 1 (reparameterization) + Step 5 (shared math) first — core change.
2. Step 2 (flatten) — simplification, can be done alongside or after.
3. Steps 3-4 (separation, area) — can be done independently.
4. Steps 6-7 (naming, fitter integration) — last, depends on all above.
