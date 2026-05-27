# Per-Trace Heteroscedastic Noise — Design

**Status:** Approved 2026-05-27. Implementation plan to follow.

## Goal

Add a per-trace **fractional** noise component `sigma_rel[t]` to complement the existing per-trace **additive** noise `log_noise[t]` (renamed conceptually to `sigma_abs[t]`). The combined likelihood becomes:

```
noise²[t, x] = sigma_abs[t]² + (sigma_rel[t] · predicted[t, x])²
```

This addresses the persistent 70× `log_noise` inflation diagnosed in the warp-model fit: the inflation is driven by skew-normal peak-shape misfit at the apex of high-amplitude peaks (~1-3% of peak height absolute), which the current homoscedastic per-trace noise model can only absorb by inflating the SINGLE scalar `log_noise[t]` to cover the worst residual. Per-trace heteroscedastic decomposes that one scalar into two physically-interpretable components.

## Motivation

The residual diagnostic on commit `74e0a46` showed:

| Region | RMS residual | Max residual |
|---|---|---|
| Baseline window | 822 counts | 1,981 |
| Peak flank | 1,885 | 30,036 |
| Peak apex | **10,893** | **168,775** |

Apex residuals are 13× the baseline residuals; max apex residual is 85× the max baseline residual. The Gaussian likelihood requires ONE `log_noise[t]` per trace to cover BOTH regions, so it inflates to ~1500 counts even though true measurement noise is ~20 counts. This:

1. Wastes information at baseline (likelihood is too permissive)
2. Makes the posterior too soft (slow sampling: ESS 154/2000 = 7-8% efficiency)
3. Doesn't actually fit the apex well (residuals still up to 168k)

A heteroscedastic noise model with `sigma_rel[t]` per trace lets:
- `sigma_abs[t]` settle near actual baseline RMS (~20 counts)
- `sigma_rel[t]` absorb the ~1-3% fractional apex misfit

Both are identifiable per-trace from data alone: baseline-window points pin `sigma_abs`, peak-window points pin `sigma_rel`. The previous global-`sigma_rel` attempt (commit `2d1e7e2`, reverted in `75b32db`) failed because **identifiability was across traces** — different traces fought over what single `sigma_rel` should fit. Per-trace eliminates that fight.

## Architecture

### Sample site (new)

```python
# Per-trace fractional noise — non-centred LogNormal in log-space
log_sigma_rel_raw[trace] ~ Normal(0, 1)
log_sigma_rel = log(sigma_rel_prior_loc) + log_sigma_rel_scale * log_sigma_rel_raw
sigma_rel = numpyro.deterministic("sigma_rel", jnp.exp(log_sigma_rel))
```

Defaults:
- `sigma_rel_prior_loc = 0.02` (2% — typical HPLC peak-area RSD)
- `log_sigma_rel_scale = 1.0` (95% prior CI: ~ [0.3%, 15%]; weakly informative)

### Existing sample site (unchanged structure)

```python
# Per-trace additive noise (currently called log_noise)
log_noise_raw[trace] ~ Normal(0, 1)
log_noise = log_noise_loc + log_noise_scale * log_noise_raw
sigma_abs = jnp.exp(log_noise)
```

We KEEP the name `log_noise` for the additive component (no rename) to minimize churn in downstream code (`posterior.py`, summary plots, etc.).

### Likelihood (replaces current homoscedastic line)

```python
# Per-point heteroscedastic noise
noise = jnp.sqrt(sigma_abs[:, None]**2 + (sigma_rel[:, None] * predicted)**2)
with numpyro.handlers.mask(mask=jnp.asarray(dataset.valid_mask)):
    numpyro.sample("obs", dist.Normal(predicted, noise))
```

Per-(trace, time) noise: `sigma_abs` broadcasts across time (constant per trace), `sigma_rel * predicted` varies across time (amplitude-scaling per trace).

## Config additions

Two new fields in `ModelConfig`:

```python
# Per-trace fractional noise prior (heteroscedastic component)
sigma_rel_prior_loc: float = 0.02
log_sigma_rel_scale: float = 1.0
```

Both physically motivated:
- `sigma_rel_prior_loc = 0.02`: 2% is the typical HPLC peak-area RSD (literature). Data can pull `sigma_rel[t]` anywhere from 0.3% to 15% within the prior.
- `log_sigma_rel_scale = 1.0`: weakly informative; matches the existing `log_noise_scale = 2.0` style of "use a scale broad enough that the data dominates."

## What stays unchanged

- All shape parameters (`mu_anchor`, `log_sigma`, `gamma1`) and the warp (`a`, `b`)
- `log_noise[trace]` per-trace (renamed conceptually to `sigma_abs` but kept as `log_noise` in code/posterior)
- `A[trace, peak]`, baseline parameters
- Non-centred parameterization throughout
- Scalar additive Gaussian likelihood structure — only the `scale` argument changes from per-trace to per-(trace, time)

## `SAMPLED_PARAMETER_NAMES` update

Add `"sigma_rel"`:

```python
SAMPLED_PARAMETER_NAMES = (
    "mu_anchor", "log_sigma", "gamma1", "A",
    "a", "b",
    "mu_eff", "sigma_eff",
    "baseline_intercept", "baseline_slope", "log_noise", "sigma_rel",
)
```

## Files touched

- `chromhandler/fitting/model.py` — main work (add sample site, update likelihood, update `SAMPLED_PARAMETER_NAMES` and docstring)
- `tests/unit/fitting/test_model_config.py` — assert new config defaults

Two files.

## Parameter budget

Adds n_trace new parameters (`log_sigma_rel_raw[trace]`).

For CV-only (56 traces): +56 params. Net model goes from ~170 → ~225 params. Modest.

## Success criteria

| Metric | Current baseline (`74e0a46`) | Target |
|---|---|---|
| `log_noise` posterior median (effective per-trace) | ~1165 counts | ~20-50 counts (close to baseline RMS) |
| `sigma_rel` posterior median across traces | n/a | 1-5% (matches apex misfit fraction) |
| `r_hat_max` | 1.02 | < 1.02 (no regression) |
| `ess_min_bulk` | 154 | > 500 (likelihood tighter → faster mixing) |
| `n_divergent` | 0 | 0 (no funnel) |
| Sampling speed | ~4 iter/s | > 20 iter/s (much tighter likelihood) |
| Residual z-score (apex) | ~1.1 RMS | ~1.0 (Gaussian-correct) |

## Scientific deliverables

- `sigma_abs[trace]` posterior per trace — measured baseline noise per injection
- `sigma_rel[trace]` posterior per trace — measured fractional peak-shape misfit per injection
- If `sigma_rel[t]` posterior is small (<1%) for all traces: skew-normal fit is excellent
- If `sigma_rel[t]` is larger for some traces (e.g., 5%+): those traces have peak shapes the skew-normal can't capture (likely high-amplitude traces with strong tailing)
- The per-trace `sigma_rel` distribution itself becomes a diagnostic for peak-model adequacy

## Anti-goals (explicit non-features)

- **No hierarchical hyperprior on `sigma_rel`** — independent per trace (we chose Option A over Option B). Add hierarchy later if cross-trace shrinkage proves useful.
- **No change to peak shape** (skew-normal stays; same as PeakPerformance)
- **No change to baseline model** (linear stays)
- **No global `sigma_rel`** — that was the previous failed attempt; per-trace is the fix

## Risk factors

- **Funnel near `sigma_rel[t] → 0`**: if a trace genuinely has zero fractional misfit, the LogNormal hyperprior anchors at 2% which pulls back. Should not collapse to zero. If it does, propose tightening `log_sigma_rel_scale` to 0.5.
- **Identifiability between `sigma_abs[t]` and `sigma_rel[t]`** within a trace: in principle these are well-separated (baseline anchors abs, apex anchors rel), but for traces with only flank residuals (no clear apex or baseline), the split is undetermined. Mitigation: the LogNormal anchors on both keep posterior tight.
- **Possible interaction with the time-axis warp**: if `sigma_rel` absorbs apex misfit, the warp may have less to do and `b[trace]` posterior could shift toward 1.0. Worth checking but not a defect.

## Out-of-scope follow-ups

- Hierarchical hyperprior on `sigma_rel` (Option B from brainstorm — add later if useful)
- Student-t likelihood as a heavy-tail alternative (orthogonal direction)
- Different parametrization of heteroscedastic structure (e.g., Poisson-like sqrt scaling)
- Peak model upgrade (EMG, bi-Gaussian) — explicitly off the table per user
