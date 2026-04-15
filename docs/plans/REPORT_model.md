# Model Sampling Efficiency Report: `better_model.py`

Date: 2026-03-28
Model: `chromhandler/fitting/better_model.py`
Analyser: static read + NumPyro docs research

---

## 1. Sampling Inventory

All `numpyro.sample()` calls in the model, with their shapes, distributions, and necessity assessment.

### Always-sampled (all peak modes)

| Site | Shape | Distribution | Necessary? |
|---|---|---|---|
| `log_sigma_base` | `[n_peak]` | `Uniform(log(0.5·σ_ref), log(2·σ_ref))` | **Yes.** Primary shape parameter for all components. |
| `alpha_raw_base` | `[n_peak]` | `Normal(raw_loc, raw_scale)` | **Yes.** Primary skew parameter for all left/dominant components. |
| `trace_shift_raw` | `[n_trace]` | `Normal(0, 1)` (expanded) | **Yes.** Non-centered parameterisation of per-trace retention-time drift. Well-designed. |
| `baseline_intercept` | `[n_trace]` | `Normal(mid_loc, mid_scale)` | **Yes.** Direct observable; centred at x_mid so identifiable. |
| `baseline_slope_pop_mean` | scalar | `Normal(slope_pop_loc, slope_pop_scale_prior)` | **Questionable.** See Section 3. |
| `baseline_slope_pop_scale` | scalar | `HalfNormal(slope_variation_prior)` | **Questionable.** See Section 3. |
| `baseline_slope_raw` | `[n_trace]` | `Normal(0, 1)` (expanded) | **Yes** — given the hierarchical slope design. Non-centered form is correct. |
| `sigma_y` | `[n_trace]` | `LogNormal(log(σ_y_prior_loc), 0.5)` | **Yes.** Noise scale; per-trace is correct. |

### Conditional on `n_nonfree > 0`

| Site | Shape | Distribution | Necessary? |
|---|---|---|---|
| `area_dominant` | `[n_trace, n_nonfree]` | `LogNormal(log(area_safe), 0.4)` | **Yes.** Per-trace area for non-free peaks; captures real run-to-run variation. |

### Conditional on `n_artefact > 0`

| Site | Shape | Distribution | Necessary? |
|---|---|---|---|
| `log_sigma_r_artefact` | `[n_artefact]` | `Uniform(log(0.5·art_ref), log(2·art_ref))` | **Yes** — the artefact component can have a different width than the dominant. |
| `separation_artefact` | `[n_artefact]` | `LogNormal(log(2σ_loc), 0.05)` | **Questionable.** Scale 0.05 is extremely tight (~5% CV). Functionally close to a constant. See Section 3D. |
| `area_artefact_typical` | `[n_artefact]` | `LogNormal(log(area_safe), 0.3)` | **Borderline.** Shared across traces; reasonable as a shared anchor. |
| `area_artefact_trace_offset` | `[n_trace, n_artefact]` | `Normal(0, 1)` (expanded) | **Questionable.** Adds `n_trace × n_artefact` dimensions. See Section 3C. |

### Conditional on `n_free > 0`

| Site | Shape | Distribution | Necessary? |
|---|---|---|---|
| `log_sigma_r_free` | `[n_free]` | `Uniform(log(0.5·free_ref), log(2·free_ref))` | **Yes.** Right component can differ from left. |
| `alpha_raw_r_free` | `[n_free]` | `Normal(raw_loc[free_idx], raw_scale[free_idx])` | **Yes.** Right component skew is independently identifiable from data. |
| `sep_typical_raw` | `[n_free]` | `Normal(raw_loc, raw_scale)` | **Yes.** Separation is fundamental for doublet fitting. |
| `area_total_free` | `[n_trace, n_free]` | `LogNormal(log(area_total_safe), 0.4)` | **Yes.** Per-trace total area variation. |
| `area_frac_left_free` | `[n_trace, n_free]` | `Beta(2, 2)` (expanded) | **Yes.** Area split between components. |

### Total latent dimension count

For a typical run with `n_trace=10`, `n_peak=3` (2 nonfree, 1 artefact, 0 free):

- Always-sampled: 3 + 3 + 10 + 10 + 1 + 1 + 10 + 10 = **48**
- n_nonfree=2: 10 × 2 = **20**
- n_artefact=1: 1 + 1 + 1 + 10 × 1 = **13**
- Total: **81 dimensions**

For n_free=1 doublets instead of artefact: replaces ~13 with 1 + 1 + 1 + (10×1) + (10×1) = **23**, so free doublets are significantly more expensive.

---

## 2. Deterministics Audit

All `numpyro.deterministic()` calls, whether they must be inside the model (i.e., downstream computations reference them via the trace), or are purely for posterior inspection.

| Site | Depends on | Used in likelihood? | Must be inside model? |
|---|---|---|---|
| `sigma_base` | `log_sigma_base` | Yes, via assembly into `sigma_l/r` | **No.** Could be `jnp.exp(log_sigma_base)` inline. The `log_sigma_base` site already records the sampled value. |
| `sigma_r_artefact` | `log_sigma_r_artefact` | Yes, via `sigma_l/r` assembly | **No.** Same as above — pure `jnp.exp(...)`. |
| `sigma_r_free` | `log_sigma_r_free` | Yes, via `sigma_l/r` assembly | **No.** Same. |
| `alpha_base` | `alpha_raw_base` | Yes, via `alpha_l/r` assembly | **No.** Could be computed inline as `_ALPHA_MAX * jnp.tanh(alpha_raw_base)`. |
| `alpha_r_free` | `alpha_raw_r_free` | Yes, via `alpha_r` assembly | **No.** Same. |
| `trace_shift` | `trace_shift_raw` | Yes, via `apex` | **No.** Could be computed inline; `trace_shift_raw` already in trace. |
| `apex` | `trace_shift`, `apex_loc` | Yes, via `apex_l/r` | **No.** Used only to construct `apex_l/r`, which themselves are deterministic. Pure function of sampled sites. |
| `separation_free_min` | constants, `sigma_loc_safe` | Yes, bounds for `sep_typical_raw` | **No.** This is a constant derived from inputs, not from sampled parameters. Compute before calling `model()`. |
| `separation_free_max` | constants, `window_hi/lo` | Yes, bounds for `sep_typical_raw` | **No.** Same — pure function of model inputs. |
| `separation_free` | `sep_typical_raw`, `sep_min/max` | Yes, via `separation.at[...]` | **No.** Derivable post-sampling. |
| `area_artefact` | `area_artefact_typical`, `area_artefact_trace_offset` | Yes, feeds likelihood | **No.** Pure non-centered transform. |
| `area_total` | `area_l`, `area_r` | No | **No.** Post-processing. |
| `apex_l` | `apex`, `separation_*` | Yes, feeds `xi_l` | **No.** Used only to compute `xi_l` and for posterior inspection. |
| `apex_r` | `apex`, `separation_*` | Yes, feeds `xi_r` | **No.** Same. |
| `separation` | various `separation_*` | No, only inspection | **No.** Post-processing. |
| `sigma_l` | `sigma_base`, assembly | Yes, feeds `xi_l`, likelihood | **No.** Could be local variable. |
| `sigma_r` | `sigma_base`, `sigma_r_*`, assembly | Yes | **No.** Same. |
| `alpha_l` | `alpha_base`, assembly | Yes | **No.** |
| `alpha_r` | `alpha_base`, `alpha_r_free`, assembly | Yes | **No.** |
| `area_l` | `area_dominant`, `area_total_free`, `area_frac_left_free` | Yes | **No.** |
| `area_r` | same | Yes | **No.** |
| `xi_l` | `apex_l`, `sigma_l`, `alpha_l` | Yes — passed to `mixture_signal` | **No.** Only `xi_flat` (local) is needed; `xi_l/r` are named for diagnostics only. |
| `xi_r` | `apex_r`, `sigma_r`, `alpha_r` | Yes | **No.** Same. |
| `baseline_slope` | `baseline_slope_pop_mean`, `pop_scale`, `raw` | Yes — baseline computation | **No.** Pure non-centered transform. |
| `baseline_curve` | `baseline_intercept`, `baseline_slope`, `x` | No — `baseline` (local) already computed | **No.** Pure inspection. `baseline` local variable is already used in `mu_y`. |
| `mu_y` | everything | It IS the likelihood mean | **Borderline.** The value is needed for the `y` sample but is also recorded. Removing the deterministic site and using a local variable would avoid storing the full `[n_trace, n_time]` array in the trace at every HMC step. This is the single largest trace payload. |

### Key findings

- **14 of 25 deterministic sites** record intermediate variables whose sole justification is posterior inspection, not likelihood computation. Every deterministic site adds an array to the MCMC trace at every HMC leapfrog step.
- **`mu_y` is the most expensive**: it stores a `[n_trace, n_time]` array (e.g., 10×500 = 5000 floats) at every HMC evaluation. This is orders of magnitude larger than any other deterministic.
- The `sigma_base / alpha_base` pattern (sample `log_*` / `raw_*`, then record `exp(log_*)` / `tanh(*)`) doubles the trace size for those parameters with no inference benefit — the sampled raw site is already in the trace.
- `separation_free_min` and `separation_free_max` are deterministic functions of **model inputs** (not sampled parameters), so they should be computed before `model()` is called and passed as pre-computed arrays.

---

## 3. Hierarchy Assessment

### 3A. Hierarchies that are warranted

**`trace_shift` (non-centered Normal)**
Correct design. The global mean is removed (`trace_shift_raw - mean(trace_shift_raw)`), which prevents the mean from drifting and pins the reference frame to the population. With n_trace=10–30 this hierarchy is well-identified.

**`baseline_slope` (hierarchical Normal, non-centered)**
Reasonable for runs where some traces have weak signal (slope unidentifiable from data alone). The population mean borrows strength across traces. The non-centered parameterisation is correct. However, see 3B below for caveats.

**`area_dominant` (independent LogNormal per trace/peak)**
Correct — run-to-run peak area variation is genuine (concentration changes, injection volume, etc.). No pooling is needed here because the variation is the signal of interest.

### 3B. Hierarchies that are over-engineered

**`baseline_slope` hierarchy**
The model samples 4 sites for baseline slope: `baseline_slope_pop_mean` (scalar), `baseline_slope_pop_scale` (scalar), `baseline_slope_raw` (n_trace). That is `n_trace + 2` dimensions for a parameter that could be handled with `n_trace` independent Normals, each with the per-trace OLS prior already in hand.

The hierarchy adds value only when the OLS slope priors are too wide (low-signal traces) AND when traces genuinely share a common slope. In chromatography, individual column batches do share drift characteristics, so some pooling is reasonable — but the full two-hyperparameter hierarchy is heavy. The `HalfNormal` prior on `baseline_slope_pop_scale` introduces a classic funnel risk: when `pop_scale` is near zero, `baseline_slope_raw` posteriors become very narrow and the geometry degrades.

This hierarchy adds 2 scalar dimensions and a non-trivial funnel geometry risk for modest gain. Alternative: use the per-trace OLS estimates as independent Normals and drop the hierarchy entirely, accepting that weakly-identified traces will have wide posteriors. Or use a single soft pooling via a fixed population scale (not sampled), removing 2 dimensions.

**`area_artefact_typical` + `area_artefact_trace_offset` (LogNormal hierarchy)**
This is a two-level hierarchy for a column artefact whose physical interpretation is "appears at the same area every run." The per-trace offset adds `n_trace × n_artefact` dimensions and a `_ARTEFACT_AREA_TRACE_LOG_SCALE = 0.15` scale that is a magic constant, not inferred from data.

Since the artefact is defined as a spurious column peak (not analyte-dependent), the strong prior should be on constant area, not per-trace variation. The entire `area_artefact_trace_offset` level could be removed, leaving only `area_artefact_typical`. If trace-to-trace artefact variation genuinely matters, use `_AREA_LOG_SIGMA = 0.3` directly on a per-trace `area_artefact` sample — which is structurally simpler and semantically equivalent but without the extra non-centered level.

### 3C. Missing hierarchies (identifiability opportunities)

**`sigma_base` and `alpha_base` — shared across traces but not pooled across peaks**
Currently `sigma_base` is a vector of `n_peak` independent LogUniform samples — one per peak. For peaks on the same column, sigma and alpha are governed by common column chemistry and should be similar. A mild hierarchical prior on `log_sigma_base` (population log-mean + log-scale) would improve identifiability for peaks with overlapping windows or low area, and would reduce the effective dimensionality by replacing `n_peak` free parameters with `n_peak + 2` but with much tighter geometry.

**`sigma_y` — independent per trace**
Noise variance is typically stable within a batch. A pooled LogNormal prior on `sigma_y` (shared log-mean, small log-scale) would reduce n_trace free parameters to `n_trace + 1` but provide useful regularization. Low priority.

---

## 4. NumPyro Patterns from Documentation

### 4A. `numpyro.deterministic()` — official guidance

The docs state deterministic sites are for "recording any values in the model execution trace" and that "most effect handlers will not operate on deterministic sites (except `trace()`), so deterministic sites should be side-effect free." This confirms that deterministics serve a diagnostic/inspection function, not an inference function.

**Implication for this model:** Every deterministic site is carried through HMC's leapfrog integration at each step as part of the trace. In NumPyro's MCMC implementation, the full trace (including all deterministic sites) is stored at each sample. Removing a `[n_trace, n_time]` deterministic like `mu_y` reduces per-sample memory and may speed up the post-processing that happens at each NUTS sample collection.

The correct pattern for inspection-only deterministics: remove them from the `model()` function entirely and compute them post-sampling from the raw sampled parameters using the same math. ArviZ `InferenceData` can hold these as computed variables added after the fact.

### 4B. `numpyro.plate` and the trace dimension

The model currently uses `dist.Normal(0.0, 1.0).expand([n_trace])` for non-centered per-trace parameters. The NumPyro-idiomatic version uses `with numpyro.plate("traces", n_trace):` which:

1. Marks the trace dimension as conditionally independent, giving NUTS better curvature information.
2. Enables future subsampling via `subsample_size` for scalability.
3. Produces cleaner ArviZ trace shapes (plate dimension is named and visible).

Currently no plates are used anywhere in the model. Adding plates for the trace dimension (`trace_shift_raw`, `baseline_intercept`, `baseline_slope_raw`, `area_dominant`, `area_artefact_trace_offset`, `area_total_free`, `area_frac_left_free`, `sigma_y`) would align with NumPyro best practices and make the conditional independence structure explicit.

**Note:** Plates do not change the mathematical posterior but they do affect how NUTS mass matrix adaptation and gradient computations are structured. For large `n_trace`, explicit plate annotation can improve NUTS mass matrix estimation because the plate dimension is treated as a structured block.

### 4C. `LocScaleReparam` for the baseline slope hierarchy

The `baseline_slope` hierarchy uses a manual non-centered form. NumPyro's `LocScaleReparam` automates this for any `loc/scale`-parameterised distribution. Using `reparam(config={"baseline_slope": LocScaleReparam()})` would let NumPyro handle the non-centered form transparently, with the `centered` parameter tuneable between 0 (fully decentered) and 1 (centered). This is a minor refactor but makes the intent explicit and allows runtime tuning.

### 4D. `TransformReparam` for LogNormal sites

`area_dominant`, `area_total_free`, `area_artefact_typical`, and `sigma_y` are all LogNormal. The `TransformReparam` handler or equivalently sampling in log-space and using `jnp.exp()` (which this model already does for sigma) gives the same geometry as the non-centered Normal reparameterisation of a LogNormal. The model already correctly does this for sigma sites (`log_sigma_base ~ Uniform`, then `sigma_base = exp(log_sigma_base)`).

**Gap:** `area_dominant` is sampled directly as `dist.LogNormal(...)` without non-centering. For a LogNormal with per-trace, per-peak variation (`[n_trace, n_nonfree]` shape), a non-centered form is:

```python
area_dominant_raw = numpyro.sample("area_dominant_raw", dist.Normal(0, 1).expand([n_trace, n_nonfree]))
area_dominant = jnp.exp(jnp.log(dominant_area_safe) + _AREA_LOG_SIGMA * area_dominant_raw)
```

With `_AREA_LOG_SIGMA = 0.4`, a prior tight enough to identify the LogNormal mean, the centered form should be fine in practice. Non-centering is most valuable when the prior is very wide or the likelihood is weak. Low priority.

### 4E. Mass matrix and dense-block guidance

NUTS documentation notes that "dense vs. diagonal mass matrices" and "structured block mass matrices are supported for complex models." For this model, `sigma_base` (shared across traces) and `baseline_slope_pop_mean/pop_scale` (hyperparameters) likely have strong posterior correlations with per-trace parameters that depend on them. Using `dense_mass=True` (or a block-dense mass matrix via `dense_mass=[("baseline_slope_pop_mean", "baseline_slope_pop_scale", "baseline_slope_raw")]`) would let NUTS adapt to this correlation structure. Currently the model uses the default diagonal mass matrix, which cannot adapt to off-diagonal curvature.

### 4F. Discrete variable marginalisation

NUTS documentation notes automatic marginalisation of discrete variables with finite support via enumeration. Not directly applicable here (all latents are continuous), but relevant if the model is ever extended with a discrete peak-detection component.

---

## 5. Concrete Proposals (Prioritised)

### Priority 1: Remove `mu_y` from the trace (high impact, trivial change)

**Problem:** `mu_y` is recorded as a `numpyro.deterministic` site of shape `[n_trace, n_time]`. For 10 traces × 500 time points this is 5000 float32 values stored at every HMC sample. At 2000 samples (500 warmup + 1500 draw), that is 10M floats = ~40 MB just for `mu_y`. More importantly, NumPyro's MCMC kernel must allocate and pass this array through the trace at every leapfrog step, not just at sample collection time.

**Fix:** Replace `mu_y = numpyro.deterministic("mu_y", ...)` with a plain local variable `mu_y = ...` and keep only the `y` observation site. After sampling, compute posterior `mu_y` by replaying the model or using the sampled parameters directly.

**Expected gain:** Reduced per-step memory allocation and trace manipulation overhead. Proportional to `n_time`. Could meaningfully speed up NUTS for large time arrays.

---

### Priority 2: Remove inspection-only intermediate deterministics (medium impact, clean-up)

**Problem:** 12–14 sites (see Section 2 audit) record intermediate computations that are pure functions of sampled sites. These include `sigma_base`, `alpha_base`, `trace_shift`, `apex`, `apex_l`, `apex_r`, `sigma_l`, `sigma_r`, `alpha_l`, `alpha_r`, `area_l`, `area_r`, `xi_l`, `xi_r`, `baseline_slope`, `baseline_curve`, `separation`, `area_total`, `area_artefact`.

**Fix:** Remove from `model()`. After sampling, compute them outside the model from the posterior samples of `log_sigma_base`, `alpha_raw_base`, `trace_shift_raw`, etc. ArviZ's `posterior` group accepts additional computed variables via `az.from_numpyro(mcmc, posterior_predictive=..., coords=..., dims=...)`.

**Expected gain:** Reduces the number of arrays stored in the trace at each NUTS step. Each `[n_trace, n_peak]` deterministic (18 of these) is 10×3=30 floats — modest individually, but collectively adds overhead in the trace manipulation Python callbacks.

**Recommended to keep inside the model:** `xi_l` and `xi_r` (only if needed inside the model for `xi_flat` — but they are not; `xi_flat` is computed locally from `xi_l`/`xi_r` anyway). Actually `xi_flat` itself is a local variable — `xi_l`/`xi_r` are only registered as deterministics for inspection. So they too can be removed.

---

### Priority 3: Drop `area_artefact_trace_offset` hierarchy (medium impact, simplification)

**Problem:** Adds `n_trace × n_artefact` latent dimensions (e.g., 10×1 = 10 extra parameters) for a column artefact whose defining characteristic is constant area. The scale `_ARTEFACT_AREA_TRACE_LOG_SCALE = 0.15` is a magic constant that is never calibrated from data.

**Fix:** Remove `area_artefact_trace_offset`. Sample `area_artefact` directly as:

```python
area_artefact = numpyro.sample(
    "area_artefact",
    dist.LogNormal(jnp.log(artefact_area_safe), _SH_AREA_LOG_SIGMA),
)  # [n_artefact] — shared across traces
```

Then broadcast over traces: `area_artefact_bc = jnp.broadcast_to(area_artefact[None, :], (n_trace, n_artefact))`.

**Expected gain:** Removes `n_trace × n_artefact` dimensions from the latent space, reducing geometry complexity. Also removes the artefact-specific funnel risk from the non-centered LogNormal construction.

---

### Priority 4: Consider removing the `baseline_slope` hierarchy (medium impact, debatable)

**Problem:** The two hyperparameters `baseline_slope_pop_mean` (scalar) and `baseline_slope_pop_scale` (HalfNormal) introduce funnel geometry risk (when `pop_scale → 0`, the geometry of `baseline_slope_raw` becomes degenerate). They add 2 dimensions to the latent space. Their benefit — pooling slope estimates across traces — is real but modest when per-trace OLS slope estimates are already available as priors.

**Option A (simplest):** Remove both hyperparameters. Use `dist.Normal(baseline_slope_loc, baseline_slope_scale)` per trace directly (centred on OLS estimates). This loses pooling but gains 2 fewer dimensions and no funnel risk.

**Option B (keep pooling, remove funnel):** Fix `baseline_slope_pop_scale` to a constant (e.g., the mean of `baseline_slope_scale`). This preserves pooling via the population mean, loses the adaptive scale, but eliminates the HalfNormal/funnel interaction.

**Option C (keep as-is):** Use `LocScaleReparam` on `baseline_slope_raw` to make NumPyro handle the non-centered form automatically, and add a weakly informative lower bound on `baseline_slope_pop_scale` to prevent funnel collapse.

Recommendation: **Option A** for speed, **Option C** for inference quality.

---

### Priority 5: Add `numpyro.plate` for trace dimension (low-medium impact, best practice)

**Problem:** Per-trace sample sites use `.expand([n_trace])`, which creates unnamed independent dimensions. NumPyro's NUTS mass matrix adaptation works better with explicit plate annotations because it can treat plate dimensions as structured blocks.

**Fix:** Wrap per-trace samples in `with numpyro.plate("traces", n_trace):`. Affected sites: `trace_shift_raw`, `baseline_intercept`, `baseline_slope_raw`, `sigma_y`. Per-trace-per-peak sites (`area_dominant`, `area_total_free`, `area_frac_left_free`) would use a nested plate or a combined plate.

**Expected gain:** Cleaner ArviZ shapes (named dimensions), potential NUTS mass matrix improvement, enables future mini-batching. No mathematical change to the posterior.

---

### Priority 6: Tighten or fix `separation_artefact` prior (low impact, correctness)

**Problem:** `separation_artefact ~ LogNormal(log(2σ_loc), 0.05)` has a standard deviation of approximately `0.05 × 2σ_loc ≈ 5%` of the prior mean. This is so tight that NUTS is spending leapfrog steps to explore a 5% credible interval around a nearly-fixed value. The parameter is effectively a constant in posterior space.

**Fix:** Either:
- Fix `separation_artefact` as a constant (`2 × sigma_loc[artefact_idx]`) and remove the `numpyro.sample()` call entirely, or
- Widen the prior scale to 0.15–0.3 to allow genuine uncertainty.

If the artefact separation is well-constrained by the data (it typically is, given the peak shape), the likelihood will override the prior regardless. If it is not constrained, the 0.05 prior is appropriate but should be acknowledged as informative. Removing it as a sampled site saves `n_artefact` dimensions.

---

### Priority 7: Non-centered LogNormal for `area_dominant` (low impact, theoretical)

**Problem:** `area_dominant ~ LogNormal(log(area_safe), 0.4)` is in centred parameterisation. With `σ = 0.4`, the LogNormal is moderately wide (CV ≈ 42%). When the likelihood is weak (small peak area), the posterior will be dominated by the prior and the centred form may create mild curvature issues.

**Fix:** Replace with non-centred form:

```python
area_dominant_raw = numpyro.sample(
    "area_dominant_raw",
    dist.Normal(0, 1).expand([n_trace, n_nonfree]),
)
area_dominant = jnp.exp(jnp.log(dominant_area_safe) + _AREA_LOG_SIGMA * area_dominant_raw)
```

**Expected gain:** Low — with `σ = 0.4` the centred form is generally stable. Relevant only for very small peaks where the likelihood does not constrain area.

---

## Summary Table

| # | Proposal | Dimensions removed | Geometry improvement | Effort |
|---|---|---|---|---|
| 1 | Remove `mu_y` deterministic | 0 (trace payload only) | High (per-step memory) | Trivial |
| 2 | Remove 14 inspection deterministics | 0 (trace payload only) | Medium (per-step overhead) | Low |
| 3 | Drop `area_artefact_trace_offset` | `n_trace × n_artefact` | Medium (fewer dims, no funnel) | Low |
| 4A | Drop `baseline_slope` hierarchy | 2 scalars | Medium (no funnel risk) | Low |
| 5 | Add `plate("traces", n_trace)` | 0 | Low-medium (mass matrix) | Medium |
| 6 | Fix `separation_artefact` as constant | `n_artefact` | Low (fewer dims) | Trivial |
| 7 | Non-center `area_dominant` | 0 | Low (theoretical) | Low |

Proposals 1 and 2 together remove the largest source of per-step overhead with no change to the posterior. Proposals 3 and 6 reduce latent dimensionality. Proposal 4 removes funnel risk. These five changes together could reduce NUTS step time by 20–40% and reduce warmup instability.
