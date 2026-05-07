# Skew-Normal Fitter Rewrite — Design

**Date:** 2026-05-07
**Branch:** new branch off `fix-fit`
**Scope:** Full rewrite of `chromhandler/fitting/`
**Status:** Design — pending user review

---

## 1. Motivation

The current fitting module (split-normal model, FWHM-based priors) suffers from
HMC divergences on real datasets. Investigation surfaced three structural issues:

1. **Wrong peak shape.** The split-Gaussian has a kink in the derivative at the
   mode and Gaussian tails on both sides. HPLC peaks have smooth shapes with
   exponential-style tailing on the lagging side. The kink alone is a
   divergence factory for HMC.
2. **Tangled parameterization.** Mixed terminology (`xi`, `mode`, `apex`,
   `retention_time`, `mu`) without a single clear convention.
3. **Heuristic priors.** Magic widening factors (e.g. `sd=0.5`) and FWHM
   bracket-finding instead of principled, data-driven priors.

This rewrite addresses all three by switching to a **skew-normal model in
centred-parameter (CP) form**, with priors derived from method-of-moments on
the windowed signal across traces, and strict spatial naming throughout.

## 2. Mathematical foundation

### 2.1 Why skew-normal in CP

The skew-normal distribution has two standard parameterizations:

- **DP (direct):** `(ξ, ω, α)`. Density evaluation is clean:
  `f(x) = (2/ω) φ((x−ξ)/ω) Φ(α(x−ξ)/ω)`. But the Fisher information matrix
  is **singular at α = 0**, producing a degenerate ridge in the posterior
  geometry whenever the data is near-symmetric. This is the documented reason
  Azzalini introduced CP.

- **CP (centred):** `(μ, σ, γ₁)` where μ is the mean, σ is the SD, and γ₁ is
  the skewness coefficient. Fisher info is non-singular at γ₁ = 0, parameters
  are orthogonal at the symmetric point, and γ₁ is **bounded** by the
  saturation of the SN family at the half-normal limit.

CP is the mathematically defensible choice for inference. Internally we
convert CP → DP for density evaluation; the bijection is a fixed nonlinear
function and JAX-differentiable.

### 2.2 The γ₁ bound

The maximum |skewness| achievable by any skew-normal distribution equals the
skewness of the limiting half-normal:

```
GAMMA1_MAX = ((4 − π) / 2) · (√(2/π))³ / (1 − 2/π)^(3/2)
           ≈ 0.9952717464311558
```

Values of γ₁ outside this range correspond to no SN. We enforce this bound
via reparameterization:

```
gamma1_raw ~ Normal(gamma1_raw_loc, gamma1_raw_scale)
gamma1     = GAMMA1_MAX · tanh(gamma1_raw)
```

where `(gamma1_raw_loc, gamma1_raw_scale)` are derived from the empirical
priors `(gamma1_loc, gamma1_scale)` via:

```
gamma1_raw_loc   = atanh(clip(gamma1_loc / GAMMA1_MAX, ±0.999))
gamma1_raw_scale = gamma1_scale / (GAMMA1_MAX · (1 − (gamma1_loc/GAMMA1_MAX)²))
```

(local linearization of `tanh` at the empirical center). This sidesteps the
unidentifiable α → ∞ ridge of DP automatically: extreme `gamma1_raw` values
map to nearly the same γ₁ near saturation, and the prior on `gamma1_raw`
regularizes against wandering.

### 2.3 CP ↔ DP bijection

Standard Azzalini formulas:

```
b      = √(2/π)
δ      = α / √(1 + α²)
μ      = ξ + ω·b·δ
σ²     = ω² · (1 − b²·δ²)
γ₁     = ((4 − π)/2) · (b·δ)³ / (1 − b²·δ²)^(3/2)
```

Inversion (CP → DP, used inside the model on every leapfrog step):

```
1. Solve γ₁ → δ via the monotone relation above (closed form).
2. ω = σ / √(1 − b²·δ²)
3. α = δ / √(1 − δ²) · sign(γ₁)
4. ξ = μ − ω·b·δ
```

Step 1 has a closed form:
```
c   = (2·γ₁ / (4 − π))^(1/3)
b·δ = c / √(1 + c²)
```

All operations are pure JAX, fully differentiable, no iterative solvers.

### 2.4 Derived quantities (post-hoc)

The mode of a skew-normal has no closed form. Used only for *reporting*, not
inside the model:

```
mode = ξ + ω · m₀(α)
```

where `m₀(α)` is Azzalini's standard approximation, accurate to ~10⁻⁴
everywhere. FWHM is similarly computed numerically from posterior samples.

## 3. Naming convention

### 3.1 Spatial-only in the model layer

Every peak window has a **left component**, always. Doublet windows
additionally have a **right component**. No "main", "shoulder", "artefact",
or "analyte" terminology appears in the model.

| Concept | Name | Layer |
|---|---|---|
| Mean of peak shape (sampled, CP) | `mu_left`, `mu_right` | model |
| SD of peak shape (sampled, CP) | `sigma_left`, `sigma_right` | model |
| Skewness coefficient (sampled, CP) | `gamma1_left`, `gamma1_right` | model |
| DP location (derived from CP) | `xi_left`, `xi_right` | internal density |
| DP scale (derived from CP) | `omega_left`, `omega_right` | internal density |
| DP slant (derived from CP) | `alpha_left`, `alpha_right` | internal density |
| Mode of fitted peak (derived) | `mode_left`, `mode_right` | reporting |
| FWHM of fitted peak (derived) | `fwhm_left`, `fwhm_right` | reporting |
| Empirical apex from raw data | `apex_observed` | data / diagnostics |

`xi`, `omega`, `alpha` exist only inside the density evaluation path and the
CP↔DP helpers. They never appear in sample sites, posterior summaries, or
plot labels.

### 3.2 Annotation-side metadata

Each peak window carries:

- `n_components: 1 | 2` — whether right-side parameters are allocated.
- `artefact_side: "left" | "right" | None` — for reporting only.
- `include_artefact_in_area: bool` — for reporting only.

The model layer is unaware of `artefact_side`. It fits left and right
symmetrically.

## 4. Pooling structure

| Parameter | Scope | Rationale |
|---|---|---|
| `log_sigma_left[peak]` | shared per peak | column efficiency constant across traces |
| `gamma1_left_raw[peak]` | shared per peak | shape constant across traces |
| `log_sigma_right[peak]` (doublet only) | shared per peak | same |
| `gamma1_right_raw[peak]` (doublet only) | shared per peak | same |
| `Delta[peak]` (doublet only) | shared per peak | chemical separation is fixed |
| `mu_anchor_left[peak]` | shared per peak | true retention time per analyte |
| `trace_shift[trace]` | shared per trace | flow/temperature drift, applied to all peaks |
| `log_A_left[trace, peak]` | per (trace, peak) | concentration varies per injection |
| `log_A_right[trace, peak]` (doublet only) | per (trace, peak) | impurity ratio varies |
| `baseline_intercept[trace]` | per trace | baseline differs per injection |
| `baseline_slope[trace]` | per trace | same |

Composition rules:

```
mu_left[trace, peak]  = mu_anchor_left[peak] + trace_shift[trace]
mu_right[trace, peak] = mu_left[trace, peak] + Delta[peak]
```

`trace_shift` is non-centered with prior loc 0 — the per-peak anchor absorbs
the absolute position, the trace shift absorbs the drift.

## 5. Doublet treatment

- Two skew-normal components per doublet window, summed in the likelihood.
- Both components have **independent shape parameters** (`σ`, `γ₁`).
- Locations coupled by signed-positive `Delta`: `mu_right = mu_left + Delta`.
- Identifiability comes from three asymmetric priors working together:
  1. `Delta > 0` by construction (LogNormal prior; repels from zero).
  2. Per-trace empirical area priors `A_left[trace] ≠ A_right[trace]`,
     extracted from window decomposition.
  3. Empirical σ priors per side may differ — natural data-driven asymmetry.
- No explicit ordering constraint on amplitudes. Empirical priors do the job.

## 6. Priors — method of moments, no magic numbers

### 6.1 Per-trace, per-window extraction

For each trace, on the baseline-subtracted signal `s(t)` within `[window_min, window_max]`:

**Single peak window:**
```
weights = clip(s, 0) / sum(clip(s, 0))
μ̂      = Σ t · weights
σ̂²     = Σ (t − μ̂)² · weights
γ̂₁     = Σ ((t − μ̂)/σ̂)³ · weights
Â       = trapezoid(s, t)
```

**Doublet window:** decompose into left and right halves at the
between-peak minimum (or `window_midpoint` as fallback), then apply the same
moments separately on each half. Δ̂ = (μ̂_right − μ̂_left).

### 6.2 Aggregation across traces

For each parameter and each peak, compute population statistics:

```
prior_loc   = mean(per_trace_estimates)
prior_scale = std(per_trace_estimates)
```

For log-space parameters (`log_sigma`, `log_A`, `log_Delta`), aggregate after
the log transform.

### 6.3 Principled scale floors

When empirical std collapses (single trace, identical estimates), fall back to
resolution- or pooling-limited floors. Each floor is derived, not heuristic:

| Parameter | Floor formula | Justification |
|---|---|---|
| `mu`, `Delta` | `dt` | sub-sample resolution unidentifiable |
| `log_sigma` | `1 / √n_trace` | precision of mean over n_trace measurements |
| `gamma1` | `√(6 / n_eff)` | large-sample SE of sample skewness |
| `log_A` | `1 / √n_trace` | pooling precision |

Final scale: `prior_scale = max(empirical_std, floor)`.

### 6.4 Output structure

```python
@dataclass(frozen=True)
class SkewNormalPriors:
    """Empirically-derived priors for one peak window."""

    n_components: int                    # 1 or 2

    # Left component (always present)
    mu_left_loc: float
    mu_left_scale: float
    log_sigma_left_loc: float
    log_sigma_left_scale: float
    gamma1_left_loc: float
    gamma1_left_scale: float
    log_A_left_loc_per_trace: jax.Array  # [n_trace]
    log_A_left_scale: float

    # Right component (only if n_components == 2)
    log_Delta_loc: float | None
    log_Delta_scale: float | None
    log_sigma_right_loc: float | None
    log_sigma_right_scale: float | None
    gamma1_right_loc: float | None
    gamma1_right_scale: float | None
    log_A_right_loc_per_trace: jax.Array | None
    log_A_right_scale: float | None
```

## 7. Module layout

```
chromhandler/fitting/
    __init__.py
    skew_normal.py     # Pure math, no NumPyro. Fully unit-testable.
    priors.py          # Method-of-moments → SkewNormalPriors per peak.
    model.py           # NumPyro model. Samples in CP. Density via DP internally.
    posterior.py       # Derived quantities from posterior samples.
    fitter.py          # Orchestration. No math.
    visualize.py       # Plotting only. Imports skew_normal for density.
    annotations.py     # Peak/baseline annotation dataclasses.
    types.py           # Shared dataclasses (ModelInputs, etc.).
```

### 7.1 `skew_normal.py` — pure math layer

Public API:

```python
GAMMA1_MAX: float = ((4.0 - math.pi) / 2.0) * (math.sqrt(2.0 / math.pi) ** 3) / (1.0 - 2.0 / math.pi) ** 1.5

def cp_to_dp(mu: jax.Array, sigma: jax.Array, gamma1: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Convert (μ, σ, γ₁) → (ξ, ω, α). Fully vectorized."""

def dp_to_cp(xi: jax.Array, omega: jax.Array, alpha: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Inverse of cp_to_dp. Used in tests and post-processing."""

def density_dp(x: jax.Array, xi: jax.Array, omega: jax.Array, alpha: jax.Array) -> jax.Array:
    """Skew-normal density in DP form. Stable for large |α|."""

def density_cp(x: jax.Array, mu: jax.Array, sigma: jax.Array, gamma1: jax.Array) -> jax.Array:
    """Skew-normal density in CP form. Internally calls cp_to_dp."""

def mode_dp(xi: jax.Array, omega: jax.Array, alpha: jax.Array) -> jax.Array:
    """Mode of SN(ξ, ω, α). Uses Azzalini's m₀ approximation."""

def fwhm_dp(xi: jax.Array, omega: jax.Array, alpha: jax.Array) -> jax.Array:
    """Full width at half maximum, computed numerically from the density."""
```

No NumPyro imports. No state. Fully unit-testable with property-based tests
(γ₁ → δ → γ₁ round-trip, CP→DP→CP round-trip, density integrates to 1, etc.).

### 7.2 `priors.py`

Public API:

```python
@dataclass(frozen=True)
class WindowMoments:
    """Per-trace moments within one peak window."""
    mu: float
    sigma: float
    gamma1: float
    area: float

def compute_window_moments(
    time: jax.Array,        # [n_time]
    signal: jax.Array,      # [n_time]
    baseline: jax.Array,    # [n_time]
    window_low: float,
    window_high: float,
    split_at: float | None = None,  # for doublets
) -> WindowMoments | tuple[WindowMoments, WindowMoments]:
    """Method-of-moments extraction. Returns one or two depending on split_at."""

def aggregate_priors(
    per_trace_moments: list[WindowMoments],
    n_trace: int,
    dt: float,
    n_components: int,
) -> SkewNormalPriors:
    """Aggregate across traces with principled floors."""
```

### 7.3 `model.py` — NumPyro layer

Single `model()` function. Samples all CP parameters with the pooling
structure from §4. Internally calls `cp_to_dp` and `density_dp` for
likelihood evaluation. No control flow on sampled values.

Samples are grouped into static index arrays:
- `single_idx`: peaks with `n_components == 1`.
- `doublet_idx`: peaks with `n_components == 2`.

Density evaluation is two batched ops: one for all left components (always
all peaks), one for all right components (only doublets). Summed into the
predicted signal.

```python
SAMPLED_PARAMETER_NAMES_SINGLE = (
    "mu_anchor_left", "log_sigma_left", "gamma1_left_raw",
    "log_A_left", "trace_shift", "baseline_intercept", "baseline_slope",
)
SAMPLED_PARAMETER_NAMES_DOUBLET = (
    *SAMPLED_PARAMETER_NAMES_SINGLE,
    "log_Delta", "log_sigma_right", "gamma1_right_raw", "log_A_right",
)
```

### 7.4 `posterior.py`

Pure functions that take posterior samples (CP) and return derived
quantities:

```python
def compute_dp_samples(samples: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Add (ξ, ω, α) per side to the samples dict."""

def compute_mode_samples(samples: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Add mode_left, mode_right per peak."""

def compute_fwhm_samples(samples: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Add fwhm_left, fwhm_right per peak."""

def compute_reported_areas(
    samples: dict[str, jax.Array],
    annotations: list[PeakAnnotation],
) -> jax.Array:
    """Apply artefact_side / include_artefact_in_area logic."""
```

### 7.5 `fitter.py`

Orchestration only. No math, no NumPyro internals exposed. Public API mirrors
the existing `Fitter` (`from_handler`, `add_peak_annotation`,
`add_baseline_annotation`, `fit`, `save_summary`, `plot_traces`,
`plot_fit_combined`) but built on the new layers underneath.

## 8. Code style

- Google-style docstrings on every public function/class (`Args`, `Returns`,
  `Raises`).
- One responsibility per module (§7). When a file grows past ~400 lines,
  split.
- Type hints on every signature. `from __future__ import annotations` at top.
- Pure functions wherever possible. Side-effecting code lives in `fitter.py`.
- Constants UPPER_SNAKE_CASE (`GAMMA1_MAX`, `B_CONST = √(2/π)`).
- Quality gates after every edit: `uv run ruff check <file>` and
  `uv run pyright <file>` must pass.

## 9. Testing strategy

- **`skew_normal.py`**: round-trip property tests (CP→DP→CP, DP→CP→DP),
  density-integrates-to-1, density-matches-`scipy.stats.skewnorm`, mode is a
  local maximum of the density, FWHM is consistent with density.
- **`priors.py`**: synthetic peaks with known `(μ, σ, γ₁, A)`, verify
  recovered moments within tolerance; aggregation tests with controlled
  variance; floor tests when n_trace=1.
- **`model.py`**: prior predictive sanity checks, MCMC on synthetic data
  with known ground truth.
- **`posterior.py`**: deterministic checks that derived quantities match
  closed-form values for known CP inputs.
- **End-to-end**: existing test dataset in `data/raw` with the user's test
  script. Convergence diagnostic (no divergences, R̂ < 1.01, ESS > 400)
  required before merge.

## 10. Migration

- New branch off `fix-fit`. No backward-compatibility shims. The split-normal
  path is removed entirely.
- Tests in `tests/unit/fitting/` are rewritten to match the new module
  structure. Tests targeting the old split-normal model are deleted.
- The `Fitter` public API surface (method names and signatures) is preserved
  so handler-side code does not change. Internal implementation is fully
  rewritten.

## 11. Out of scope for this rewrite

- Changing the `Handler` API or annotation API beyond what's strictly needed
  for the new model.
- Adding EMG or other peak shapes. Skew-normal only. EMG is a candidate for
  a future phase.
- Changing how baseline regions are annotated or how the baseline prior is
  computed. The rewrite reuses the existing baseline annotation mechanism.
- Free doublets without an `artefact_side`. If needed later, the model
  already handles this (just `artefact_side=None` in metadata, both areas
  reported).
