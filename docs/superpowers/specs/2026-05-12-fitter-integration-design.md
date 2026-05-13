# Fitter Integration — Design Document

**Status:** approved during brainstorm 2026-05-12. Implementation plan to follow.

## 1. Goal

Add three new modules — `model.py`, `posterior.py`, `fitter.py` — that together consume the existing `PreparedDataset` + `SkewNormalPriors` and run NumPyro-based MCMC inference. The user-facing surface is a single function `fit()` returning a `FitResult` with chromatography-specific debug plots. Single-mode peaks only; `artefact_doublet` raises `NotImplementedError` from a documented, structurally additive extension point.

## 2. Module layout

```
chromhandler/fitting/
    skew_normal.py        # shipped — pure math
    prepared_dataset.py   # shipped — data prep
    priors.py             # shipped — SkewNormalPriors + PriorConfig
    model.py              # NEW — NumPyro model() + run_mcmc()
    posterior.py          # NEW — derived quantities, prior/posterior predictive
    fitter.py             # NEW — fit() entry point + FitResult class
```

Each new module has one clear responsibility:

- **`model.py`** — defines the NumPyro probabilistic model and runs MCMC. Returns an `arviz.InferenceData`. No plotting, no orchestration.
- **`posterior.py`** — pure-function layer that adds derived quantities (mode, FWHM, area), computes prior/posterior predictive samples, and produces diagnostics dicts. No NumPyro, no plotting.
- **`fitter.py`** — user-facing orchestrator. Defines `fit()` and the `FitResult` dataclass with all the plotting methods.

## 3. User-facing API

```python
from chromhandler.fitting import fit, PriorConfig, ModelConfig

# defaults
result = fit(dataset)

# overrides
result = fit(
    dataset,
    prior_config=PriorConfig(gamma1_scale_n1=0.3),
    model_config=ModelConfig(num_warmup=1000, num_chains=4),
)

# inspect
result.plot_traces()             # MCMC convergence
result.plot_prior_overlay()      # prior loc curves on data
result.plot_prior_predictive()   # prior predictive 95% HDI band
result.plot_fit()                # posterior predictive 95% HDI band
result.summary()                 # ArviZ DataFrame (mean/sd/HDI/R-hat/ESS)
result.diagnostics()             # one-liner dict: r_hat_max, ess_min, n_divergent
result.save("fit.nc")            # netCDF (ArviZ native)

# raw access
import arviz as az
az.plot_pair(result.idata)
```

### `fit()` signature

```python
def fit(
    dataset: PreparedDataset,
    *,
    prior_config: PriorConfig | None = None,
    model_config: ModelConfig | None = None,
) -> FitResult:
    """Build priors from dataset, run MCMC, return wrapped result."""
```

### `FitResult` class

```python
@dataclass(frozen=False)   # mutable so lazy idata groups can be added
class FitResult:
    idata: arviz.InferenceData
    dataset: PreparedDataset
    priors: list[SkewNormalPriors]
    model_config: ModelConfig

    # Plotting (all return matplotlib Figure)
    def plot_traces(self, var_names: list[str] | None = None) -> Figure: ...
    def plot_prior_overlay(self) -> Figure: ...
    def plot_prior_predictive(self) -> Figure: ...
    def plot_fit(self) -> Figure: ...

    # Tabular / programmatic
    def summary(self, var_names: list[str] | None = None) -> pd.DataFrame: ...
    def diagnostics(self) -> dict[str, float | int | bool | str]: ...

    # I/O
    def save(self, path: Path | str) -> None: ...
```

## 4. Data flow

```
PreparedDataset
    │
    │  fit() internally calls build_priors()
    ▼
list[SkewNormalPriors]
    │
    │  fit() internally calls run_mcmc()
    ▼
arviz.InferenceData (posterior + observed_data)
    │
    │  bundled into FitResult alongside dataset + priors
    ▼
FitResult
    │
    │  .plot_fit() computes posterior_predictive lazily; caches in idata.posterior_predictive
    │  .plot_prior_predictive() computes prior + prior_predictive lazily; caches likewise
    │
    ▼
matplotlib Figures, pandas DataFrames, etc.
```

`fit()` itself does *not* eagerly compute posterior predictive (only on first `plot_fit()` call). Same for prior predictive. This keeps `fit()` fast.

## 5. Model architecture (`model.py`)

### 5.1 NumPyro `model()` function

```python
def model(
    dataset: PreparedDataset,
    priors_list: list[SkewNormalPriors],
    config: ModelConfig,
) -> None:
    # Validation is hoisted to run_mcmc() — see §5.3 — so this function is hot.

    # === Left components: sample for ALL peaks ===
    mu_anchor_left  = numpyro.sample("mu_anchor_left",  ...)  # shape [n_peak]
    log_sigma_left  = numpyro.sample("log_sigma_left",  ...)  # shape [n_peak]
    gamma1_left     = numpyro.sample("gamma1_left",     ...)  # shape [n_peak]
    log_A_left      = numpyro.sample("log_A_left",      ...)  # shape [n_trace, n_peak]

    # === Per-trace nuisance ===
    trace_shift        = numpyro.sample("trace_shift",        ...)  # shape [n_trace], non-centred
    baseline_intercept = numpyro.sample("baseline_intercept", ...)  # shape [n_trace]
    baseline_slope     = numpyro.sample("baseline_slope",     ...)  # shape [n_trace]

    # === DOUBLET EXTENSION HOOK — empty today ===
    # right components for doublet peaks go here (see §10)

    # === Predicted signal ===
    predicted = _baseline_contribution(dataset.time, baseline_intercept, baseline_slope)
    predicted = predicted + _left_component_contribution(
        dataset.time, mu_anchor_left, trace_shift, log_sigma_left, gamma1_left, log_A_left,
    )
    # + right contribution from doublet hook (future)

    # === Likelihood (NaN-masked) ===
    with numpyro.handlers.mask(mask=dataset.valid_mask):
        numpyro.sample(
            "obs",
            dist.Normal(predicted, dataset.noise_per_trace[:, None]),
            obs=dataset.signal,
        )
```

### 5.2 Distribution wiring (from priors plan §1 distribution table)

| Sample site | Distribution | Source |
|---|---|---|
| `mu_anchor_left` | `TruncatedNormal(loc, scale, low=window_low, high=window_high)` | `SkewNormalPriors.mu_left_*` |
| `log_sigma_left` | `TruncatedNormal(loc, scale, low, high)` | `SkewNormalPriors.log_sigma_left_*` |
| `gamma1_left` | `TruncatedNormal(loc, scale, low=-c·GAMMA1_MAX, high=+c·GAMMA1_MAX)` | `SkewNormalPriors.gamma1_left_*` + `PriorConfig.gamma1_bound_fraction` |
| `log_A_left` | `Normal(loc_per_trace, scale)` per `(trace, peak)` | `SkewNormalPriors.log_A_left_*` |
| `trace_shift` | `Normal(0, drift_scale)` non-centred per trace | `drift_scale = ModelConfig.trace_shift_scale_dt_multiplier * dataset.dt_global` |
| `baseline_intercept` | `Normal(intercept_ols[trace], max(intercept_se[trace], floor))` | OLS + `ModelConfig.baseline_intercept_se_floor` |
| `baseline_slope` | `Normal(slope_ols[trace], max(slope_se[trace], floor))` | OLS + `ModelConfig.baseline_slope_se_floor` |

### 5.3 `run_mcmc()` orchestrator

```python
def run_mcmc(
    dataset: PreparedDataset,
    priors_list: list[SkewNormalPriors],
    config: ModelConfig,
) -> arviz.InferenceData:
    """Validate, run NUTS, wrap result as InferenceData."""
    _validate_single_mode_only(priors_list)

    kernel = numpyro.infer.NUTS(
        model,
        target_accept_prob=config.target_accept_prob,
        max_tree_depth=config.max_tree_depth,
    )
    mcmc = numpyro.infer.MCMC(
        kernel,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
        progress_bar=True,
    )
    mcmc.run(jax.random.PRNGKey(config.seed), dataset, priors_list, config)
    return arviz.from_numpyro(mcmc)
```

### 5.4 Sample-site name constants

```python
SAMPLED_LEFT_SHARED = ("mu_anchor_left", "log_sigma_left", "gamma1_left")
SAMPLED_LEFT_PER_TRACE = ("log_A_left",)
SAMPLED_TRACE_NUISANCE = ("trace_shift", "baseline_intercept", "baseline_slope")

# Doublet extension (defined but empty until doublet ships):
SAMPLED_RIGHT_SHARED: tuple[str, ...] = ()
SAMPLED_RIGHT_PER_TRACE: tuple[str, ...] = ()
```

`summary()` uses these to filter parameters and to label rows.

## 6. `ModelConfig`

```python
@dataclass(frozen=True)
class ModelConfig:
    # HMC / NUTS
    num_warmup: int = 500
    num_samples: int = 500
    num_chains: int = 4
    target_accept_prob: float = 0.9
    max_tree_depth: int = 10
    seed: int = 0

    # Model-layer priors (per-trace, not per-peak)
    trace_shift_scale_dt_multiplier: float = 5.0
    baseline_intercept_se_floor: float = 1.0
    baseline_slope_se_floor: float = 0.01

    # Prior predictive sampling
    prior_predictive_n_samples: int = 200
```

Defaults tuned for fast development iteration on the priors_demo dataset (17 traces, 1 peak, ~30–60 s on a laptop). Users override for publication-quality runs.

## 7. `posterior.py` — derived quantities and predictive samples

Pure-function module operating on `arviz.InferenceData`. No NumPyro at the API level (uses NumPyro's `Predictive` internally for prior/posterior predictive).

```python
def compute_posterior_predictive(idata, dataset, priors_list, config) -> arviz.InferenceData:
    """Run NumPyro Predictive with posterior samples. Returns idata with
    posterior_predictive group added."""

def compute_prior_predictive(idata, dataset, priors_list, config) -> arviz.InferenceData:
    """Sample from priors, run model forward. Returns idata with
    prior + prior_predictive groups added."""

def derived_areas(idata) -> NDArray[np.float64]:
    """Per-trace per-peak posterior areas, computed from log_A samples.
    Shape [chain, draw, trace, peak]."""

def diagnostics(idata) -> dict[str, Any]:
    """{'r_hat_max': ..., 'r_hat_max_param': ..., 'ess_min_bulk': ...,
        'ess_min_param': ..., 'n_divergent': ..., 'fit_healthy': bool}"""
```

`FitResult` methods delegate to these helpers.

## 8. Plotting (within `fitter.py` or as `chromhandler/fitting/plotting_posterior.py`)

All four plot methods on `FitResult`. 95% HDI for all band displays.

### 8.1 `plot_traces()`

ArviZ-native: `arviz.plot_trace(idata, var_names=...)`. KDE on the left, chain iterations on the right, one row per parameter (or per `(parameter, peak)` for vector-valued params).

### 8.2 `plot_prior_overlay()`

Direct port of the `priors_demo` Section 9b plot. For each non-control trace × each peak:

- Plot baseline-subtracted data, sliced to the peak window.
- Overlay the prior loc curve (skew-normal density with shared shape, per-trace amplitude from `log_A_left_loc_per_trace`).
- Layout: one figure per peak; subplots = non-control traces in a grid.

Does not require posterior — usable before any MCMC run.

### 8.3 `plot_prior_predictive()`

Computes prior predictive lazily on first call. For each trace:

- 95% HDI band from `prior_predictive_n_samples` simulated signals.
- Prior predictive median line.
- Observed data line on top.

Layout: one panel per trace, contiguous along the chromatogram region.

### 8.4 `plot_fit()` (posterior predictive overlay)

Same layout as prior predictive but using posterior samples. The band should hug the data tightly — wide band = high posterior uncertainty, indicates a problem.

## 9. Controls handling

### 9.1 Priors layer

Controls are **excluded** from shape-prior aggregation (current behavior in `aggregate_single_peak_priors`). Their `log_A_left_loc_per_trace` entries are set to `log(A_floor)` (a wide, uninformative amplitude prior). The `is_control` mask comes from `PreparedDataset.is_control`.

### 9.2 Model layer

Controls **are fit through the likelihood normally**. No special-casing in `model()`. Their per-trace parameters (`trace_shift`, `baseline_intercept`, `baseline_slope`, `log_A_left`) are sampled like any other trace; their data goes into the `obs` likelihood with the same noise model.

The posterior naturally pins `log_A_left[control_trace]` near the noise floor (since their data shows no peak), which is the correct result.

### 9.3 Required priors.py fix (prerequisite)

Today: `aggregate_single_peak_priors` returns `log_A_left_loc_per_trace` of length `n_non_control` (number of trace features passed in).

Required: it must return length `n_trace`, with control entries set to `log(A_floor)`. Wire this through `build_priors` so the array indexes into the full dataset, not a subset.

This is a small change (~10 lines + test) and is the *only* priors.py modification required by this plan.

## 10. Doublet extension hooks (single-mode-only today)

Six hooks confirmed during brainstorm:

1. **Sample-site naming with `_left` suffix from day one.** No renaming when doublet ships.
2. **`model()` body is strictly additive.** Doublet adds:
   - One line to sample right-component parameters (vectorised over doublet peaks).
   - One line to add the right contribution to `predicted`.
   - Removal of `_validate_single_mode_only` call.
3. **`SAMPLED_RIGHT_*` constants defined as empty tuples today.** Populated when doublet ships.
4. **Mixed-mode handling: single `model()` function with index arrays.** No separate `model_doublet()`. When implemented:
   ```python
   doublet_idx = jnp.array([i for i, p in enumerate(priors_list) if p.n_components == 2])
   if doublet_idx.size > 0:
       Delta = numpyro.sample("Delta", ..., shape=(doublet_idx.size,))
       # etc.
       predicted = predicted + _right_component_contribution(...)
   ```
5. **`_validate_single_mode_only` placed at the top of `run_mcmc()`,** not inside `model()` (which is JIT-compiled). Raises `NotImplementedError` with actionable message + pointer to extension docs.
6. **Plotting forward-compatible.** Each plot method iterates `priors_list` and checks `n_components` per peak. Adding doublet support is a one-line `if p.n_components == 2: also_plot_right(...)` per plot method.

**Documentation pattern:** both inline `# TODO(doublet):` markers at each hook (greppable) AND a "Doublet extension hooks" section in the `model.py` module docstring listing each hook with `file:line` references (high-level overview).

## 11. Testing strategy

Three test layers, increasing cost:

### 11.1 Unit / smoke tests (`tests/unit/fitting/test_model.py`)

- `ModelConfig` dataclass construction and overrides.
- `_validate_single_mode_only` raises on doublet input.
- `_compute_baseline_se` helper correctness on a toy signal.
- `_left_component_contribution` shape + finite output on small inputs.
- `model()` prior-only sampling produces shapes matching `dataset.signal`.
- A 1-iteration MCMC run completes without error.

Run time: <5 s each.

### 11.2 Synthetic recovery tests (`tests/unit/fitting/test_model_recovery.py`)

- Synthesize a 5-trace single-peak kinetic series with known `(mu, sigma, gamma1, A)`.
- Run small MCMC (warmup=300, samples=300, chains=2).
- Assert posterior median is within tolerance:
  - `mu`: within `2·dt`
  - `sigma`: within 15%
  - `gamma1`: within 0.1
  - `log_A`: within 5%
- R-hat < 1.1, no divergences.

Run time: 30–60 s each.

### 11.3 Real-data smoke test (`tests/integration/test_fitter_asm.py`)

- Load the ASM kinetic-series fixture + conditions CSV.
- Register SIH/Hyp/Ino molecules.
- Run `fit()` on a per-analyte subset.
- Assert `diagnostics()["fit_healthy"]` is True.
- Assert posterior median `mu` is within `5·dt` of the priors_demo recovered value (3.008 min for SIH).

Run time: 1–2 min.

### 11.4 Plotting tests

Smoke tests only — each plot method runs and returns a `Figure` without error. No image comparison.

## 12. Out of scope (intentional)

- **Doublet mode implementation** (hooks only; future plan).
- **Free-doublet mode** (raise `NotImplementedError` like in current priors.py).
- **Posterior predictive checks (PPC) statistics** beyond what ArviZ provides natively.
- **Variational inference** as a fast alternative to MCMC (could be a future option, but YAGNI now).
- **GPU MCMC** — NumPyro supports it, but the JAX backend in this repo currently uses CPU; not changing.
- **Result.predict(new_dataset)** for held-out prediction — explicitly rejected during brainstorm.
- **Result merging** (combining multiple fits into one InferenceData) — future, not blocking.
- **`Fitter` class wrapping** — explicitly rejected in favor of `fit()` function + `FitResult`.

## 13. Open questions for plan-writing

These don't need to be answered in this design but inform the plan:

- Should `posterior.py` and the plotting code live in their own modules or as private helpers within `fitter.py`? Lean: separate modules for testability; expose only public surface from `fitter.py`.
- Test-time MCMC seeds: fix to a known seed (deterministic) or sweep (robust to flake)? Lean: fix for unit tests, but document that the threshold tolerances assume seed=0.

## 14. Self-review checklist

- ✅ Modules each have one responsibility (model = math + MCMC; posterior = derived; fitter = orchestration + plots).
- ✅ User-facing surface is one function plus one class.
- ✅ Single-mode now, doublet hooks documented at every site.
- ✅ Controls handled uniformly in model; priors do the special-casing.
- ✅ All sampling sites' distributions traced to a config field or a dataclass attribute.
- ✅ No hidden magic numbers; every threshold either in `PriorConfig` or `ModelConfig`.
- ✅ Tests are layered (unit / synthetic recovery / real-data smoke).
- ✅ Plot methods consistent in band width (95% HDI) and layout philosophy (per-trace / per-peak slicing).
- ✅ ArviZ-native output container; standard tooling works directly.
