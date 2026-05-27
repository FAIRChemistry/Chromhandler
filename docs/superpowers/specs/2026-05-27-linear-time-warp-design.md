# Linear Time-Axis Warp — Design

**Status:** Approved 2026-05-27. Implementation plan to follow.

## Goal

Replace the per-(trace, peak) shape-deviation architecture (current `mu_shift` + just-added `log_sigma_dev` / `gamma1_dev`) with a **per-trace linear time-axis warp** of the form `t' = a[trace] + b[trace] · t`. The warp captures the two dominant physical mechanisms of HPLC retention drift — proportional column compression / flow change (via `b`) and constant offsets like injection-timing or dead-volume changes (via `a`) — in one principled parameterization.

This is the literature-standard PTW (Parametric Time Warping, Eilers 2004) degree-1 polynomial, lifted into the Bayesian model rather than applied as a separate preprocessing step.

## Motivation

The CV-only fit currently converges at `r_hat = 1.08`, `ess_min = 46`. The worst-mixing parameters are `mu_shift_raw` cells, and `log_noise` per trace is inflated 72× over baseline RMS. Diagnostic showed shifts ARE roughly proportional to retention time (the signature of column drift) — meaning the additive-shift model is geometrically the wrong shape.

The just-added hierarchical shape pooling (commit `ff8fef9`) confirmed via small `tau_*` posteriors that peak shape itself is genuinely shared across traces — so the leftover slow mixing and noise inflation are NOT from shape misspecification but from the additive shift trying (and failing) to capture proportional drift.

The warp model is the physically correct replacement: peak width AND position both scale with the per-trace stretch (as column physics requires), and a per-trace shift absorbs constant offsets. This drops ~400 parameters (511 per-trace shape params → 112 warp params) and tests the strict project hypothesis (shape is shared, only drift varies).

## Architecture

### Per-trace warp parameters

Both follow the established non-centred + sum-to-zero pattern.

```python
# Per-trace shift component (additive offset in time units)
a_raw[trace] ~ Normal(0, 1)
a = config.warp_shift_scale_dt_multiplier * dt_global * a_raw
a = a - jnp.mean(a)                 # sum-to-zero per trace
                                    # (breaks anchor↔shift global degeneracy)

# Per-trace stretch component (multiplicative; log-space for positivity)
log_b_raw[trace] ~ Normal(0, 1)
log_b = config.warp_stretch_scale * log_b_raw
log_b = log_b - jnp.mean(log_b)     # sum-to-zero in log space
                                    # (equivalent to geomean(b) == 1)
b = jnp.exp(log_b)
```

Sum-to-zero on both is essential: without it, the chain has two global translation/scaling degeneracies (`mu_anchor + ε ↔ a + ε for all traces`, and `mu_anchor × k ↔ b ÷ k for all traces`). Same pattern that worked for `mu_shift`.

### Per-(trace, peak) effective shape

The warp transforms time `t' = a + b·t`. Inverting for "where does peak `p` appear in observed time for trace `t`":

```python
mu_eff[t, p]     = (mu_anchor[p] - a[t]) / b[t]
sigma_eff[t, p]  = sigma[p] / b[t]      # width scales WITH stretch (physical)
gamma1_eff[t, p] = gamma1[p]            # dimensionless skewness, unchanged
```

`gamma1` is a dimensionless coefficient on the skew-normal density and does not transform under a linear time-axis warp. `mu` and `sigma` both scale by `1/b` because they're in units of time.

### Likelihood (unchanged structure)

```python
sigma_eff_jnp = sigma_eff  # already [n_trace, n_peak]
for peak in range(n_peak):
    dens = density_cp(
        time_arr,
        mu_eff[:, peak:peak+1],
        sigma_eff[:, peak:peak+1],
        gamma1_eff[:, peak:peak+1],
    )
    peak_contrib += A[:, peak:peak+1] * dens
predicted = baseline + peak_contrib
# Then: dist.Normal(predicted, noise[:, None]) inside the mask handler
```

## Configuration

Two new `ModelConfig` fields:

```python
# Per-trace warp parameters: t' = a + b * t
# a anchored at ~5*dt (typical injection-timing scale)
warp_shift_scale_dt_multiplier: float = 5.0
# b anchored near 1 with ~1% deviation (typical HPLC column drift)
warp_stretch_scale: float = 0.01
```

Both have physical units and defendable defaults.

## What gets removed (Cleanup A)

All per-(trace, peak) shape deviation machinery from the previous two iterations:

| Removed | Reason |
|---|---|
| `mu_shift_raw`, `mu_shift` deterministics | Replaced by `a`/`b` warp |
| `log_sigma_shift_raw`, `sigma_shift` hyperprior | Same |
| `log_sigma_dev_raw`, `log_sigma_dev`, `log_sigma_eff` | Replaced by sigma_eff = sigma / b |
| `tau_log_sigma_raw`, `tau_log_sigma` hyperprior | Same |
| `gamma1_dev_raw`, `gamma1_dev`, `gamma1_eff` | gamma1 is dimensionless and stays at gamma1[peak] |
| `tau_gamma1_raw`, `tau_gamma1` hyperprior | Same |
| `ModelConfig.shape_dev_hyperprior_fraction` | No longer used |

## What stays unchanged

- `mu_anchor[peak]`, `log_sigma[peak]`, `gamma1[peak]` — global per peak (population)
- `A[trace, peak]`, `log_noise[trace]`, `baseline_intercept[trace]`, `baseline_slope[trace]` — per-trace
- All non-centred parameterizations, soft priors, tanh bijector for gamma1, softplus for A
- Scalar additive Gaussian likelihood per trace

## `SAMPLED_PARAMETER_NAMES` update

After cleanup + additions:

```python
SAMPLED_PARAMETER_NAMES = (
    "mu_anchor", "log_sigma", "gamma1", "A",
    "a", "b",                                          # NEW
    "mu_eff", "sigma_eff",                             # NEW (effective per-trace shape)
    "baseline_intercept", "baseline_slope", "log_noise",
)
```

(`gamma1_eff` is not added because it equals `gamma1` per-peak; we just use `gamma1` directly in the likelihood call.)

## Files touched

- `chromhandler/fitting/model.py` — main work (architectural change)
- `tests/unit/fitting/test_model_config.py` — config-default assertions
- `tests/integration/test_fitter_asm.py` — currently asserts on `mu_anchor + mu_shift`; switch to `mu_anchor + mu_eff` or equivalent

## Parameter budget (CV-only, 56 traces, 3 peaks)

| Architecture | Per-trace shape params | Hyperparameters |
|---|---|---|
| Before (current `ff8fef9`) | 504 (`mu_shift` 168 + `log_sigma_dev` 168 + `gamma1_dev` 168) | 7 (`sigma_shift` 1 + `tau_log_sigma` 3 + `tau_gamma1` 3) |
| After (warp) | 112 (`a_raw` 56 + `log_b_raw` 56) | 0 |

Net: ~400-parameter reduction. Total model: ~290 → ~170 free parameters.

## Success criteria

| Metric | Current baseline (`ff8fef9`) | Target |
|---|---|---|
| `r_hat_max` (CV-only) | 1.08 | < 1.02 |
| `ess_min_bulk` | 46 | > 500 |
| `n_divergent` | 0 | 0 |
| `log_noise` inflation median | 72× | < 10× |
| Worst-mixing parameter | `mu_shift_raw[*, *]` | anything but `a_raw` / `log_b_raw` |

Scientific deliverables:
- `b[trace]` posterior per trace — measured column drift per injection
- `a[trace]` posterior per trace — measured constant offset per injection
- If `log_noise` inflation drops dramatically: previous "noise" was retention drift in disguise. Hypothesis vindicated.
- If `log_noise` inflation persists: deeper model-form issue (likely baseline curvature). Identified for next iteration.

## Anti-goals (explicit non-features)

- No backup shape-pooling residual (Cleanup A is strict — measure failure, decide later)
- No higher-order warp (degree-2+ is for later if needed; literature warns against it for HPLC)
- No changes to noise, baseline-shape, or area architectures
- No sample-level grouping (still per-trace; mirrors current per-trace shift architecture)

## Risk factors

- **The warp model may not capture true non-linear drift** (e.g., temperature-gradient effects). If `log_noise` inflation persists after this change, that's diagnostic — propose quadratic warp or per-trace baseline polynomial degree.
- **Identifiability between `a` and `b`** — they're correlated within each trace (small `a` + 1.01 `b` ≈ small `a'` + 1.0 `b`). The sum-to-zero constraints on both should help; if not, may need a joint reparameterization.
- **The strict-coupling `sigma_eff = sigma / b`** assumes peak width comes ENTIRELY from column geometry. Injection-related band broadening (sample matrix effects) would not be captured. Diagnostic: if `log_noise` inflation persists in a peak-width-correlated way, this is the cause.

## Out-of-scope follow-ups

- Quadratic warp `t' = a + b·t + c·t²` if linear is insufficient
- Per-trace shape residual (`sigma_eff = sigma / b + dev[t, p]`) if width variation persists beyond what stretch explains
- Baseline polynomial extension (quadratic baseline) — clearly orthogonal but plausibly needed
