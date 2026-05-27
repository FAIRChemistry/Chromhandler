# Hierarchical Shape Pooling — Design

**Status:** Approved 2026-05-27. Implementation plan to follow.

## Goal

Generalize the project's shared-shape hypothesis into a hierarchical Bayesian model. Per-peak hyperparameters (`tau_log_sigma[peak]`, `tau_gamma1[peak]`) quantify how much each shape dimension varies across traces. If the hypothesis holds, the hyperparameters shrink toward zero; if not, they measure the actual variation. This is a **research contribution** beyond PeakPerformance (no pooling) and beyond the current strict-shared-shape model.

## Motivation

The current model forces `(mu_anchor, log_sigma, gamma1)` to be shared across all traces. On heterogeneous datasets (e.g., 56 traces from 8 different injections), this is empirically refuted by the data — the priors module shows apex spreads of 6σ for peak `third`, and post-MCMC `log_noise` inflates to 72× the baseline RMS to absorb the shape mismatch. The shared-shape constraint leaks into the noise model and into the `mu_shift` cells, causing the worst-mixing parameters (`ess ≈ 50`, `r_hat ≈ 1.06`) after the sum-to-zero centering fix.

Letting shape parameters flex per-trace under a hierarchical hyperprior provides three benefits:
1. The hyperprior shrinks individual deviations toward zero when shape is truly shared (the hypothesis-friendly outcome).
2. The data can pull deviations away from zero when there's genuine shape variation.
3. The posterior on each `tau_*` directly quantifies how much shape variation exists — a publishable measurement.

## Architectural pattern

Mirror the existing `mu_shift` hierarchy (sum-to-zero per peak, non-centred reparameterization, learned hyperprior) for the other two shape dimensions.

### What's added

```python
# Per-peak hyperpriors on shape-deviation magnitude
tau_log_sigma[peak] ~ HalfNormal(0.5 * priors[peak].log_sigma_scale)
tau_gamma1[peak]    ~ HalfNormal(0.5 * priors[peak].gamma1_scale)

# Per-(trace, peak) non-centred deviations
log_sigma_dev_raw ~ Normal((n_trace, n_peak), 1)
gamma1_dev_raw    ~ Normal((n_trace, n_peak), 1)

# Apply tau, then per-peak sum-to-zero centering (break anchor-dev degeneracy)
log_sigma_dev = tau_log_sigma * log_sigma_dev_raw
log_sigma_dev = log_sigma_dev - mean(log_sigma_dev, axis=0)

gamma1_dev = tau_gamma1 * gamma1_dev_raw
gamma1_dev = gamma1_dev - mean(gamma1_dev, axis=0)

# Effective per-(trace, peak) shape used in the likelihood
log_sigma_eff[t, p] = log_sigma[p] + log_sigma_dev[t, p]
gamma1_eff[t, p]    = GAMMA1_MAX * tanh((gamma1[p] + gamma1_dev[t, p]) / GAMMA1_MAX)
```

The likelihood, currently using shared `log_sigma[peak]` and `gamma1[peak]`, switches to using `log_sigma_eff[trace, peak]` and `gamma1_eff[trace, peak]`.

### What stays unchanged

- `mu_anchor[peak]` global + `mu_shift[trace, peak]` per-trace with `sigma_shift` hyperprior and sum-to-zero centering. Already established and converging.
- `A[trace, peak]`, `log_noise[trace]`, `baseline_intercept[trace]`, `baseline_slope[trace]` — all per-trace, no hierarchy added. These are physically per-trace quantities.
- All non-centred parameterizations, all soft priors, no hard truncations, scalar additive Gaussian likelihood.
- No sample-level grouping. We do **not** track which traces come from which sample; all traces are drawn from one population per peak.

### New config field

A single new field in `ModelConfig`:

```python
# in ModelConfig
shape_dev_hyperprior_fraction: float = 0.5
# tau_log_sigma[peak] ~ HalfNormal(fraction * priors[peak].log_sigma_scale)
# tau_gamma1[peak]    ~ HalfNormal(fraction * priors[peak].gamma1_scale)
# Translation: "across-trace variation is at most ~half of within-trace
# prior uncertainty by default; data can pull tau larger if needed".
```

One field, one magic number (0.5), defended physically.

## Parameter budget

Reference: CV-only dataset, 56 traces, 3 peaks.

- 6 new hyperparameters (`tau_log_sigma[3]`, `tau_gamma1[3]`)
- 336 new per-(trace, peak) deviation raw cells (`log_sigma_dev_raw[56, 3]`, `gamma1_dev_raw[56, 3]`)
- Total model: ~290 → ~630 parameters

Hierarchical shrinkage controls effective DOF — when shape is shared, individual deviations are pulled tight by the small `tau_*`.

## Sample sites and deterministics

New sample sites: `tau_log_sigma_raw`, `tau_gamma1_raw`, `log_sigma_dev_raw`, `gamma1_dev_raw`.

New deterministic sites (exposed for downstream consumers):
- `tau_log_sigma[peak]` = `priors[peak].log_sigma_scale * shape_dev_hyperprior_fraction * softplus(tau_log_sigma_raw)`
- `tau_gamma1[peak]` = `priors[peak].gamma1_scale * shape_dev_hyperprior_fraction * softplus(tau_gamma1_raw)`
- `log_sigma_dev[trace, peak]` = `tau_log_sigma * log_sigma_dev_raw - mean(...)` (centred)
- `gamma1_dev[trace, peak]` = same pattern
- `log_sigma_eff[trace, peak]`, `gamma1_eff[trace, peak]` = the per-(trace, peak) values used in the likelihood

(Actual non-centred parameterization for the HalfNormal: sample `tau_*_raw ~ Normal(0, 1)`, then `tau_* = anchor * softplus(tau_*_raw)` so positivity is enforced smoothly via softplus.)

## Likelihood changes

The peak density evaluation, currently:

```python
for peak in range(n_peak):
    dens = density_cp(time_arr, mu[:, peak:peak+1], sigma[peak], gamma1[peak])
    peak_contrib += A[:, peak:peak+1] * dens
```

becomes per-(trace, peak):

```python
sigma_eff = jnp.exp(log_sigma_eff)   # [n_trace, n_peak]
for peak in range(n_peak):
    dens = density_cp(
        time_arr,
        mu[:, peak:peak+1],
        sigma_eff[:, peak:peak+1],
        gamma1_eff[:, peak:peak+1],
    )
    peak_contrib += A[:, peak:peak+1] * dens
```

`density_cp` already supports broadcasting; verify it accepts per-trace sigma/gamma1.

## Files touched

- `chromhandler/fitting/model.py` — only production file. Adds 4 sample sites, 6 deterministics (or however many of the named ones we expose), updates `SAMPLED_PARAMETER_NAMES`, updates likelihood loop.
- `tests/unit/fitting/test_model_config.py` — add assertion for the new default `shape_dev_hyperprior_fraction == 0.5`.

No changes to `PreparedDataset`, `priors.py`, `posterior.py`, `plotting.py`, or `handler.py`.

## Success criteria

After implementation, on the CV-only dataset (`test.py` filtered to `s.id.startswith("CV")`):

| Metric | Current baseline (d68d6ee) | Target |
|---|---|---|
| `r_hat_max` | 1.06 | < 1.02 |
| `ess_min_bulk` | 50 | > 500 |
| `log_noise` inflation factor median | 72× | < 10× |
| Worst-mixing parameter family | `mu_shift_raw` | anything but `log_noise` or `*_dev_raw` |

Scientific deliverables:
- `tau_log_sigma[peak]` posterior mean for each peak — quantifies how much peak width varies across traces
- `tau_gamma1[peak]` posterior mean for each peak — quantifies how much peak skew varies across traces
- If `tau_*` values are small (e.g., < 0.1 × prior scale): the strict shared-shape hypothesis is supported by the data
- If `tau_*` values are large: the data demands per-trace shape variation, and we've measured it

## Anti-goals (explicit non-features)

- **No sample-level grouping.** All traces draw from one population per peak. PreparedDataset is unchanged.
- **No per-trace baseline curvature.** Linear baseline stays. Orthogonal concern.
- **No per-trace `A` hierarchy.** `A` is genuinely per-trace; no shrinkage useful.
- **No noise model changes.** Scalar additive Gaussian per trace stays.

## Risk factors

- **Adding 336 parameters may slow MCMC.** Hierarchical shrinkage helps but per-step cost grows. Mitigation: still much smaller than the 70-trace heteroscedastic experiment that failed; should be tractable.
- **`tau_*` may run into funnel pathology when shape is genuinely shared** (tau → 0 + non-centred deviations create funnel). Mitigation: non-centred parameterization for both tau and deviations (which we already do everywhere else); HalfNormal hyperprior on tau anchored well above zero.
- **Sum-to-zero on dev creates the same anchor-dev degeneracy we saw with mu.** Mitigation: identical centring trick that worked for mu_shift. Apply uniformly.
