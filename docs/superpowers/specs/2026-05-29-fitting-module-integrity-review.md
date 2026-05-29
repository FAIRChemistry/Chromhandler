# Fitting module integrity review — 2026-05-29

Audit of `chromhandler/fitting/` for correctness, statistical integrity,
and numerical safety. Findings are ranked by danger to fits/users, not
by where they live in the code. Each item names the file + line, the
failure mode, and a principled fix.

Branch at time of review: `fix-fit`.

---

## Critical — these silently bias inference

### 1. Baseline prior is fit to the same data as the likelihood

**Where:** [`model.py:47-80`](../../../chromhandler/fitting/model.py) (`_compute_baseline_se`) and
[`model.py:200-218`](../../../chromhandler/fitting/model.py) (prior on `baseline_intercept`, `baseline_slope`).

**What happens:** OLS is run on the baseline-region points to produce
`(intercept_se, slope_se)`. The prior is then

```python
baseline_intercept ~ Normal(loc=ols_estimate, scale=ols_se)
```

Both the location *and* the spread of the "prior" come from the same
data that the likelihood subsequently evaluates over the same baseline
windows.

**Why it's wrong:** This is a Laplace approximation of the baseline
marginal posterior being recycled as a prior — classic data
double-counting. Posterior credible intervals on the baseline (and on
`area` and `noise`, which are correlated with it) are systematically
too narrow.

**Principled fix:** Drop the OLS pre-fit and let the joint model
identify the baseline from the baseline-window points directly. If a
weakly-informative prior is required, fix the `scale` from side
information (e.g., a fraction of the signal std) instead of from the
OLS standard error.

**✅ RESOLVED (2026-05-29, branch `marginalised-baseline`).** Implemented
the strongest form: **analytic flat-prior marginalisation** of the linear
baseline (Tier 1 of the design discussion). The baseline is no longer
sampled — conditional on the peak/noise/warp parameters it is a
linear-Gaussian sub-model and is integrated out in closed form
(`marginal_baseline_loglik` in `model.py`, injected via `numpyro.factor`).
With a flat prior this is a projection: only the residual a straight line
cannot explain is charged against the likelihood. No prior scale to defend
(the whole IQR / p95 / OLS-SE debate dissolves); `_compute_baseline_se` is
deleted. `baseline_intercept`/`baseline_slope` remain available as
Rao–Blackwellised deterministic reconstructions. The implementation uses
the numerically stable direct-residual form (avoids float32 catastrophic
cancellation; unit-tested against `np.linalg.lstsq` to 1e-5, plus
float32-large-baseline and degenerate-trace regression tests).

**A/B outcome on the ASM kinetic-series fixture (real data, no synthetic):**
new fit healthy (0 divergences, r̂ 1.01, ess_min 589). Peak posteriors shift
in a **systematic, baseline-driven** way — in every trace with real peaks,
the left edge peak's area rises (~+2-3%) and the right edge peak's area
falls (~−2-4%) while the central SIH peak is stable; widths follow. This
antisymmetric, geometry-ordered pattern is the mechanical signature of the
linear baseline being freed from the old over-tight prior — i.e. the bias
this item describes, now corrected — not a regression. `mu`/`skew` change
negligibly in absolute terms. Golden reference + harness:
`scripts/capture_golden.py`, `tests/fixtures/asm_kinetic_series/golden_baseline_model.json`,
A/B block in `test.py`.

**Follow-ups surfaced:** `plot_baseline_prior` now visualises an OLS-SE
baseline prior the model no longer uses (relabel or remove); `plot_traces`
keeps `compact`/`combined`/`figsize` as ignored stubs after the ArviZ 1.x
migration (update docstring or drop the params).

---

### 2. `nan_to_num(predicted)` silently hides skew-normal boundary blowups

**Where:** [`model.py:290`](../../../chromhandler/fitting/model.py)
and the CP→DP bijection in [`skew_normal.py:46-52`](../../../chromhandler/fitting/skew_normal.py).

**What happens:** The skew tanh bound is

```python
skew = GAMMA1_MAX * tanh(skew_unconstrained / GAMMA1_MAX)
```

`tanh` approaches but never reaches 1. Combined with float roundoff, HMC
can sample `|skew|` within machine ε of `GAMMA1_MAX`. Inside `cp_to_dp`:

```python
omega = sigma / sqrt(1 - b_delta**2)
alpha = delta / sqrt(1 - delta**2)
```

Both denominators go to zero → `omega`, `alpha` → ∞ → `density_cp` returns
NaN/Inf → `nan_to_num` zeroes them. The Gaussian likelihood then sees
`Normal(loc=0, scale=noise)` against the real signal — which does
eventually penalise the sample, but only *after* the model contribution
has been silently destroyed.

**Why it's wrong:** Divergence counts no longer reflect skew
pathologies. Chains can drift toward the boundary undetected. Posterior
density on `skew` near the bound is meaningless.

**Principled fix:** Either
(a) tighten the bijector: `skew = (GAMMA1_MAX - eps) * tanh(...)` with
`eps ~ 1e-3`; or
(b) keep `delta` (or `alpha`) as the primary sample site — its
parameterisation has no closed boundary at all and inverse mapping to
`gamma1` is for reporting only.
Also remove the silent `nan_to_num` and let divergences fire.

---

### 3. Area prior is empirical-Bayes with tight CV

**Where:** [`priors.py:404-416`](../../../chromhandler/fitting/priors.py).

**What happens:** For supported traces,

```python
area ~ Normal(loc=area_measured, scale=0.3 * area_measured)
```

then `softplus`. Both `loc` and `scale` are functions of the data
(`area_measured = trapezoid integral`).

**Why it's wrong:** Same data-double-counting story as (1), but per
trace. After the likelihood update, the posterior on `area` is shrunk
toward the trapezoidal integral with a "prior" width pretending it's
non-data-derived. For high-SNR traces the 0.3 CV dominates the
likelihood, producing over-confident credible intervals.

**Principled fix:** Pick `scale` from side information (instrument
linearity, calibration repeatability). Or drop the empirical prior
entirely and use a uniform `HalfNormal(scale = noise * window_width *
k)` — the data has plenty of signal to identify area without help.

---

### 4. `softplus(0) ≠ 0` — unsupported areas are anchored at ln 2, not zero

**Where:** [`model.py:193-198`](../../../chromhandler/fitting/model.py)
combined with [`priors.py:410`](../../../chromhandler/fitting/priors.py)
(`area_loc_per_trace = np.where(has_support, areas_measured, 0.0)`).

**What happens:** The docstring claims `softplus` "collapses to ~0 for
unsupported traces (`area_loc = 0`)". But
`softplus(0) = ln 2 ≈ 0.693`. Unsupported traces therefore have an a
priori median area of ~0.7 (signal units × time), not zero.

**Why it's wrong:** Bug. Documented behaviour does not match math. If
`area_scale` is large the offset is negligible; if it is small (low
noise × narrow window), the bias is real and the unsupported-area
prior is no longer the half-normal-at-zero it claims to be.

**Principled fix:** Either subtract `ln(2)` from the deterministic
(i.e., `softplus(x) - ln 2 ≈ x` for `x ≫ 0`, but is signed near 0 —
not a great fix), or — cleaner — switch unsupported traces to a true
`HalfNormal(scale = area_scale)` via `dist.HalfNormal` directly with no
softplus.

---

### 5. Sum-to-zero centring on the transformed value, not the sample

**Where:** [`model.py:240-251`](../../../chromhandler/fitting/model.py)
(both `time_shift` and `time_stretch`).

**What happens:**

```python
time_shift_raw ~ Normal(0, 1)            # n_trace samples
_shift = shift_scale * time_shift_raw
time_shift = _shift - mean(_shift)       # centring after sampling
```

The raw sample `time_shift_raw` still has a global-mean degree of
freedom that is invisible to the likelihood — the deterministic strips
it before it enters `predicted`.

**Why it's wrong:** HMC random-walks along that gauge direction during
warmup, hurting `ess_bulk` for `time_shift_raw` and (less, but
nonzero) for anything correlated with it.

**Principled fix:** Sample `n_trace - 1` raw values and reconstruct the
last as `-sum(others)` with the corresponding Jacobian, or use
`numpyro.handlers.reparam` with `SoftSumZeroReparam`. Same for
`time_stretch` / `log_stretch`.

---

## Numerical hazards

### 6. `_compute_baseline_se` runs inside the model body

[`model.py:200`](../../../chromhandler/fitting/model.py). Pure-NumPy
work invoked during model construction. JAX traces through it (inputs
are untraced) so the SE values become compile-time constants — but the
work executes every time `Predictive` rebuilds the model (prior
predictive, posterior predictive). Precompute once and store on
`PreparedDataset` like the rest.

**✅ RESOLVED (2026-05-29).** `_compute_baseline_se` is deleted entirely
(item 1's marginalisation removed the need for a baseline SE). The only
remaining consumer was a plot helper, now inlined locally in
`plot_baseline_prior` (see item 1 follow-ups). The marginal likelihood's
projection constants are cheap `jnp` ops traced as compile-time constants.

### 7. `np.maximum(s, 0.0)` before `np.trapezoid` biases area high

[`priors.py:234`](../../../chromhandler/fitting/priors.py) and
[`priors.py:266`](../../../chromhandler/fitting/priors.py). Clipping
negative residuals to zero inflates the integral whenever the OLS
baseline slightly over-estimates the true baseline. Compounds (3): the
empirical-Bayes area prior is biased high. Use signed integration; if
a non-negativity guard is wanted, apply `max(area, 0)` once at the end.

### 8. `dt_global` can be NaN and silently propagates

[`preprocessing.py:89`](../../../chromhandler/fitting/preprocessing.py).
If every trace has fewer than 2 valid samples, `nanmedian` returns NaN.
The model then evaluates `shift_scale = warp_shift_scale_dt_multiplier
* dt_global` → NaN → NaN-poisoned `time_shift` → NaN posterior. No
error is raised. Add `assert np.isfinite(dt_global)` (or raise
`ValueError`) in `prepare_dataset`.

### 9. `compute_window_features` does not validate `smoothing_window`

[`priors.py:199`](../../../chromhandler/fitting/priors.py). `poly_min =
min(3, smoothing_window - 1)`. If a user passes `smoothing_window = 1`
(no smoothing), `poly_min = 0` and `savgol_filter` raises. Default is
5 so safe in practice, but add a `>= 3` check.

### 10. `density_cp` is not gradient-stable near `gamma1 = 0`

[`skew_normal.py:46`](../../../chromhandler/fitting/skew_normal.py).
`jnp.cbrt(2 * gamma1 / (4 - π))` is smooth at 0, but
`d cbrt(x)/dx = 1/(3 x^(2/3))` is infinite at 0. NUTS gradients near
`skew = 0` get a 1/0 spike. Rare in practice (the centred bijector
rarely lands exactly on 0) but fragile. Keeping `delta` (or `alpha`) as
the primary parameter avoids this entirely.

---

## API / quality

### 11. `FitResult` is mutable and `save()` is non-reproducible

[`fitter.py:39`](../../../chromhandler/fitting/fitter.py). The
`@dataclass(frozen=False)` is intentional — predictive groups are
lazily added to `idata` on the first plot call. But `save()` then
writes "whatever happens to be in `idata`", so the on-disk content
depends on whether the user called `plot_fit()` first. Reproducibility
footgun. Either (a) eagerly compute predictives in `fit()`, or (b)
make `save()` explicit: `save(path, include=['posterior_predictive',
...])`.

### 12. `shift.py` is dead code in the fitter chain

`align_chromatograms` is defined but nothing in
`fit → build_priors → run_mcmc` calls it. Either wire it in as a
pre-`prepare_dataset` step (in which case document that fitting
auto-aligns) or delete the module — its presence implies a guarantee
the fitter does not provide.

### 13. `_default_user_facing_var_names` includes the warped variables

[`fitter.py:67`](../../../chromhandler/fitting/fitter.py).
`mu_warped` and `width_warped` are shape `[n_trace, n_peak]`.
`plot_trace` emits `2 * n_trace * n_peak` subplots even in compact
mode; the `max_subplots: 200` rc bump hides the symptom but a
20-trace, 3-peak fit has 120 warped sub-rows. Warped variables are
derivable from `mu`, `width`, `time_shift`, `time_stretch` —
default-exclude them.

### 14. Predictive seeds use arithmetic offsets instead of `jax.random.split`

[`posterior.py:54, 93`](../../../chromhandler/fitting/posterior.py).
`PRNGKey(seed + 1)`, `PRNGKey(seed + 2)`. Works, but it's exactly the
pattern JAX docs explicitly warn against. Use
`jax.random.split(PRNGKey(seed), 3)` once at the top.

### 15. "No peak at all" is undetectable

With `signal_threshold=None` (the default), the relative-height gate
inside the supported set always reports support — even when every
trace is pure noise. Fitter runs, area posteriors become `softplus` of
integrated noise, and nothing flags the window as empty. Warn or
raise when `max(apex_height) < k * noise`.

### 16. `ddof=0` on small samples

[`priors.py:319, 327, 336`](../../../chromhandler/fitting/priors.py).
For `n = 2` supported traces, `std(ddof=0)` underestimates the true
spread by `sqrt((n-1)/n) ≈ 0.707`. The `n = 1` fallback paths handle
the single-trace case, but `n = 2, 3` still get tight priors. Switch
to `ddof=1`.

---

## Recommended fix order

1. **(1) baseline data-double-counting** — biggest statistical lie.
2. **(2) skew-boundary + `nan_to_num`** — biggest silent failure mode.
3. **(4) softplus offset** — documented behaviour does not match math; easy fix.
4. **(8) NaN `dt_global` assertion** — one-line guard against silent garbage.
5. **(11) `FitResult.save()` reproducibility** — cheap, removes a real footgun.

Items 3, 5, 6, 7, 12, 13 are all worth doing but can come after.
