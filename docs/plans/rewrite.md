# Implementation Plan — Skew-Normal Fitter Rewrite

**Spec:** [docs/superpowers/specs/2026-05-07-skew-normal-fitter-rewrite-design.md](../superpowers/specs/2026-05-07-skew-normal-fitter-rewrite-design.md)
**Branch:** new branch off `fix-fit` (e.g. `fit-rewrite-skew-normal`)
**Style:** Google docstrings, full type hints, `from __future__ import annotations`, ruff + pyright clean.

## Conventions used in this plan

- **Steps are ordered by dependency.** Each phase builds on the previous; each step within a phase can be done in isolation.
- **Each step lists its files, tests, and acceptance criteria.** Acceptance is the gate: a step is not "done" until criteria pass.
- **Quality gate after every file edit:** `uv run ruff check <file>` and `uv run pyright <file>` must pass with zero issues. Tests added in a step must pass before moving on.
- **Worktree-based execution recommended** for the larger phases (math layer, model layer) to keep concerns isolated.

## Phase 0 — Branch and scaffolding

### Step 0.1 — Create the branch
- Create branch `fit-rewrite-skew-normal` from `fix-fit`.
- No code changes.
- Acceptance: `git branch --show-current` matches; `git log fix-fit..HEAD` is empty.

### Step 0.2 — Wipe existing fitting module
- Delete the entire `chromhandler/fitting/` directory contents (we are rewriting it).
- Delete tests under `tests/unit/fitting/` — they target the split-normal model and are not salvageable.
- Keep `tests/integration/test_fitting_speedup.py` for now; flag for revision in Phase 6.
- Add a placeholder `chromhandler/fitting/__init__.py` (empty) so imports don't break the rest of the package while we rebuild.
- Acceptance: `uv run python -c "import chromhandler"` succeeds.

## Phase 1 — Pure math layer (`skew_normal.py`)

This phase has zero NumPyro dependencies. It is the most-tested and most-isolated module. Every public function is pure, JAX-compatible, and unit-tested.

### Step 1.1 — Constants and CP↔DP bijections
- Create `chromhandler/fitting/skew_normal.py`.
- Implement:
  ```python
  GAMMA1_MAX: float                                  # ≈ 0.9952717464311558
  B_CONST: float                                     # √(2/π)
  def cp_to_dp(mu, sigma, gamma1) -> (xi, omega, alpha)
  def dp_to_cp(xi, omega, alpha) -> (mu, sigma, gamma1)
  ```
- Use the closed-form δ inversion from §2.3 of the spec.
- Acceptance:
  - Unit tests `tests/unit/fitting/test_skew_normal_bijection.py`:
    - Round-trip `cp_to_dp(dp_to_cp(x)) == x` for randomized DP triples (γ₁ inside bounds).
    - Round-trip `dp_to_cp(cp_to_dp(x)) == x` for randomized CP triples.
    - Edge case: γ₁ = 0 gives α = 0, ω = σ, ξ = μ.
    - Edge case: |γ₁| → GAMMA1_MAX gives |α| → ∞ smoothly (assert |α| > 1000 for γ₁ = 0.99·MAX).
  - Tolerances: `1e-10` for round-trips on float64, `1e-6` on float32.

### Step 1.2 — Density evaluation (DP and CP)
- Implement:
  ```python
  def density_dp(x, xi, omega, alpha) -> array
  def density_cp(x, mu, sigma, gamma1) -> array          # internally uses cp_to_dp
  ```
- Use `jax.scipy.stats.norm.pdf` and `jax.scipy.stats.norm.cdf` for `2/ω · φ(z) · Φ(αz)` — they're stable for moderate |α|. For the doublet shoulder case we don't expect |α| > 50 in practice, but document numerical bounds.
- Acceptance:
  - `tests/unit/fitting/test_skew_normal_density.py`:
    - Density matches `scipy.stats.skewnorm.pdf` on a grid for assorted `(ξ, ω, α)`.
    - Density integrates to 1 (trapezoid on a fine grid).
    - α=0 reduces to standard Normal density.
    - `density_cp` agrees with `density_dp(x, *cp_to_dp(...))`.

### Step 1.3 — Derived quantities (mode, FWHM)
- Implement:
  ```python
  def mode_dp(xi, omega, alpha) -> array         # Azzalini's m₀ approximation
  def fwhm_dp(xi, omega, alpha) -> array         # numerical: solve density = max/2
  ```
- For `mode_dp`, use the standard approximation:
  ```
  m₀(α) ≈ μ_z(α) − γ₁(α) · σ_z(α)/2 − sign(α)·exp(−2π/|α|)/2
  ```
  where `μ_z, σ_z, γ₁(α)` are the moments of standardized SN(α). Document that it's accurate to ~10⁻⁴.
- For `fwhm_dp`, use a small bracketed bisection on each side of the mode (JAX-friendly).
- Acceptance:
  - `tests/unit/fitting/test_skew_normal_derived.py`:
    - Mode is a local maximum: `density(mode) > density(mode ± ε)` for small ε.
    - At α=0, mode equals ξ (within tolerance).
    - FWHM at α=0 equals `2·√(2 ln 2)·ω` within 1e-3 relative.
    - For α=10, mode is shifted positively from ξ.

### Step 1.4 — Module documentation and re-exports
- Add module-level Google-style docstring summarizing the math.
- Re-export from `chromhandler/fitting/__init__.py`:
  `GAMMA1_MAX, cp_to_dp, dp_to_cp, density_dp, density_cp, mode_dp, fwhm_dp`.
- Acceptance: `uv run ruff check chromhandler/fitting/skew_normal.py` and `uv run pyright chromhandler/fitting/skew_normal.py` pass clean.

## Phase 2 — Annotation and types layer

### Step 2.1 — Peak and baseline annotations
- Create `chromhandler/fitting/annotations.py`.
- Define:
  ```python
  @dataclass(frozen=True)
  class PeakAnnotation:
      molecule_id: str
      rt_min: float
      rt_max: float
      n_components: int                              # 1 or 2
      artefact_side: Literal["left", "right"] | None
      include_artefact_in_area: bool

  @dataclass(frozen=True)
  class BaselineAnnotation:
      rt_min: float
      rt_max: float
  ```
- Validation in `__post_init__`: rt_min < rt_max, n_components ∈ {1, 2}, artefact_side only allowed when n_components == 2.
- Acceptance: `tests/unit/fitting/test_annotations.py` covering valid construction, invalid construction (raises), and field accessors.

### Step 2.2 — Shared types
- Create `chromhandler/fitting/types.py`.
- Define:
  ```python
  @dataclass(frozen=True)
  class SkewNormalPriors:
      n_components: int
      mu_left_loc: float
      mu_left_scale: float
      log_sigma_left_loc: float
      log_sigma_left_scale: float
      gamma1_left_loc: float
      gamma1_left_scale: float
      log_A_left_loc_per_trace: jnp.ndarray            # [n_trace]
      log_A_left_scale_per_trace: jnp.ndarray          # [n_trace]
      Delta_low: float | None                          # Uniform lower bound = 5·dt
      Delta_high: float | None                         # Uniform upper bound = window_width/2
      log_sigma_right_loc: float | None
      log_sigma_right_scale: float | None
      gamma1_right_loc: float | None
      gamma1_right_scale: float | None
      log_A_right_loc_per_trace: jnp.ndarray | None
      log_A_right_scale_per_trace: jnp.ndarray | None

  @dataclass(frozen=True)
  class ModelInputs:
      time: jnp.ndarray                                # [n_trace, n_time]
      signal: jnp.ndarray                              # [n_trace, n_time]
      baseline_mask: jnp.ndarray                       # [n_time] bool
      peak_priors: list[SkewNormalPriors]              # one per peak
      single_idx: jnp.ndarray                          # peak indices with n_components=1
      doublet_idx: jnp.ndarray                         # peak indices with n_components=2
      noise_per_trace: jnp.ndarray                     # [n_trace]
      dt: float
      n_trace: int
      n_peak: int
  ```
- Acceptance: types import cleanly; `tests/unit/fitting/test_types.py` smoke test.

## Phase 3 — Priors layer (`priors.py`)

### Step 3.1 — Single-peak window moments
- Create `chromhandler/fitting/priors.py`.
- Implement:
  ```python
  @dataclass(frozen=True)
  class WindowMoments:
      mu: float
      sigma: float
      gamma1: float
      area: float

  def compute_single_window_moments(
      time: np.ndarray,
      signal_baseline_subtracted: np.ndarray,
      window_low: float,
      window_high: float,
  ) -> WindowMoments
  ```
- Method of moments per §6.1 single-peak block.
- Acceptance: `tests/unit/fitting/test_priors_moments.py`:
  - Synthetic SN with known `(μ, σ, γ₁, A)` recovered within 5% on dense grid, low noise.
  - Tested across γ₁ ∈ {-0.9, -0.3, 0, 0.3, 0.9}.

### Step 3.2 — Dominant apex detection (smoothed)
- Implement:
  ```python
  def detect_dominant_apex(
      time: np.ndarray,
      signal_baseline_subtracted: np.ndarray,
      window_low: float,
      window_high: float,
      smoothing_window: int = 5,
  ) -> tuple[float, float]                         # (apex_loc, height)
  ```
- Use Savitzky-Golay smoothing (`scipy.signal.savgol_filter`) before peak
  finding (`scipy.signal.find_peaks`). We only need the dominant apex —
  per-trace secondary apex detection is not used (Δ is uniform, not
  empirical).
- Acceptance: `tests/unit/fitting/test_priors_apex.py`:
  - Single peak: dominant detected at the synthesized μ.
  - Two well-separated peaks: dominant detected at the larger one.
  - Noise-only window: dominant detection still returns argmax (don't crash).

### Step 3.3 — Outer-side HWHM and Gaussian-residual amplitude split
- Implement:
  ```python
  def estimate_outer_hwhm(
      time: np.ndarray,
      signal_baseline_subtracted: np.ndarray,
      apex_loc: float,
      apex_height: float,
      outer_side: Literal["left", "right"],
      window_low: float,
      window_high: float,
  ) -> float                                       # HWHM in time units

  def split_doublet_areas(
      time: np.ndarray,
      signal_baseline_subtracted: np.ndarray,
      window_low: float,
      window_high: float,
      window_midpoint: float,
      noise_per_trace: float,
      dt: float,
  ) -> tuple[float, float]                         # (A_left, A_right)
  ```
- `split_doublet_areas` implements the spatial-position rule from §6.1
  doublet block: detect dominant apex, classify spatial side, fit
  outer-HWHM Gaussian, residual to other side, both floored at noise level.
- Acceptance: `tests/unit/fitting/test_priors_split.py`:
  - Synthetic doublet, analyte left dominant: A_left ≈ true A_main, A_right ≈ true A_artefact (within 20%, this is a rough prior).
  - Synthetic where analyte left is *zero* and only artefact right exists: A_right ≈ A_artefact, A_left ≈ A_floor (no ghost peak).
  - Symmetric: same test with sides reversed.

### Step 3.4 — Population aggregation
- Implement:
  ```python
  def aggregate_single_peak_priors(
      per_trace_moments: list[WindowMoments],
      n_trace_pooled: int,
      dt: float,
  ) -> SkewNormalPriors

  def aggregate_doublet_priors(
      per_trace_dominant_apex: list[tuple[float, float]],
      per_trace_areas: list[tuple[float, float]],   # (A_left, A_right)
      shared_shape_priors: tuple[float, float, float, float],  # (sigma_loc, sigma_scale, gamma1_loc, gamma1_scale)
      window_low: float,
      window_high: float,
      noise_per_trace: np.ndarray,
      dt: float,
      n_trace_pooled: int,
  ) -> SkewNormalPriors
  ```
- Apply principled scale floors per §6.3:
  - `mu`, `Delta`: floor `dt`.
  - `log_sigma`: floor `1/√n_trace`.
  - `gamma1`: floor `√(6/n_eff)`.
  - `log_A`: floor `1/√n_trace`.
- For doublets, Δ always gets a `Uniform(Δ_low, Δ_high)` prior with
  `Δ_low = 5·dt` and `Δ_high = window_width/2`. Per-trace separation
  measurements are not used to build a Normal prior — separation cannot
  be reliably measured from data even when two maxima are visible
  per-trace; uniform on derived bounds is the principled max-entropy
  choice (see spec §6.1 doublet block).
- Acceptance: `tests/unit/fitting/test_priors_aggregate.py`:
  - n_trace=10 with controlled spread → recovered (loc, scale) match.
  - n_trace=1 → scale collapses to floor exactly.
  - Doublet: `Delta_low == 5·dt` and `Delta_high == window_width/2`
    regardless of how many traces show secondary maxima.

### Step 3.5 — Top-level prior orchestrator
- Implement:
  ```python
  def build_priors(
      time: jnp.ndarray,                           # [n_trace, n_time]
      signal: jnp.ndarray,                         # [n_trace, n_time]
      baseline: jnp.ndarray,                       # [n_trace, n_time]
      noise_per_trace: jnp.ndarray,                # [n_trace]
      dt: float,
      annotations: list[PeakAnnotation],
  ) -> list[SkewNormalPriors]
  ```
- Pipeline:
  1. Per single-peak annotation: per-trace moments → aggregate.
  2. Compute *shared shape priors* from the population of single-peak windows
     (median μ, σ, γ₁ across all single-peak windows in the dataset).
  3. Per doublet annotation: per-trace dominant apex + outer-HWHM split,
     then aggregate using shared shape priors. Δ uniform-bound prior is
     computed from window geometry alone.
- Acceptance: `tests/unit/fitting/test_priors_orchestrator.py` — end-to-end on a synthetic dataset with mixed single/doublet annotations.

## Phase 4 — NumPyro model (`model.py`)

### Step 4.1 — Component density helpers
- Create `chromhandler/fitting/model.py`.
- Implement:
  ```python
  def _gamma1_from_raw(gamma1_raw: jax.Array) -> jax.Array:
      return GAMMA1_MAX * jnp.tanh(gamma1_raw)

  def _component_signal(
      time: jax.Array,                             # [n_trace, n_time]
      mu: jax.Array,                               # [n_trace, n_peak]
      sigma: jax.Array,                            # [n_peak]
      gamma1: jax.Array,                           # [n_peak]
      log_A: jax.Array,                            # [n_trace, n_peak]
  ) -> jax.Array:                                  # [n_trace, n_time]
      """Sum of skew-normal components evaluated at every time point."""
  ```
- Vectorize across `(trace, peak)` cleanly. No Python loops over peaks.
- Acceptance: smoke test that synthetic inputs produce finite, non-negative output of correct shape.

### Step 4.2 — `model()` function
- Implement the full NumPyro model per §4 pooling structure.
- Sample sites:
  - `mu_anchor_left[peak]`
  - `log_sigma_left[peak]`, `gamma1_left_raw[peak]`
  - For doublet peaks only: `Delta[peak]` (Uniform), `log_sigma_right[peak]`, `gamma1_right_raw[peak]`, `log_A_right[trace, peak]`
  - `log_A_left[trace, peak]`, `trace_shift[trace]`, `baseline_intercept[trace]`, `baseline_slope[trace]`
- Compose:
  - `mu_left[t,p] = mu_anchor_left[p] + trace_shift[t]`
  - For doublet `p`: `mu_right[t,p] = mu_left[t,p] + Delta[p]`
- Sum component contributions + baseline → predicted signal.
- Likelihood: `Normal(predicted, noise_per_trace)` per `(trace, time)`.
- Acceptance:
  - Prior predictive sampling produces shapes matching observations.
  - One MCMC step (warmup=10, samples=10) completes without errors on a synthetic dataset.

### Step 4.3 — `SAMPLED_PARAMETER_NAMES` helpers
- Implement helpers that return the list of sampled site names for a given mix of peaks (used by ArviZ summaries):
  ```python
  def sampled_parameter_names(annotations: list[PeakAnnotation]) -> list[str]
  ```
- Acceptance: covers all sites correctly when (singles only, doublets only, mix).

## Phase 5 — Posterior derived quantities (`posterior.py`)

### Step 5.1 — DP samples and derived quantities
- Create `chromhandler/fitting/posterior.py`.
- Implement:
  ```python
  def add_dp_samples(samples: dict, annotations: list[PeakAnnotation]) -> dict
  def add_mode_samples(samples: dict, annotations: list[PeakAnnotation]) -> dict
  def add_fwhm_samples(samples: dict, annotations: list[PeakAnnotation]) -> dict
  def compute_reported_areas(samples: dict, annotations: list[PeakAnnotation]) -> jax.Array
  ```
- `compute_reported_areas` applies the `artefact_side` + `include_artefact_in_area` reporting logic from §3.2.
- Acceptance: `tests/unit/fitting/test_posterior.py`:
  - Synthetic CP samples → derived DP / mode / FWHM match `skew_normal.py` direct calls.
  - Reported areas correctly include/exclude artefacts per metadata.

## Phase 6 — Orchestration layer (`fitter.py`)

### Step 6.1 — `Fitter` class skeleton
- Create `chromhandler/fitting/fitter.py`.
- Public API mirrors the existing surface used by the user's test script:
  ```python
  class Fitter:
      @classmethod
      def from_handler(cls, handler) -> Fitter
      def add_peak_annotation(self, *, molecule_id, rt_min, rt_max, mode, artefact_side=None, include_artefact_in_area=True)
      def add_baseline_annotation(self, *, rt_min, rt_max)
      def fit(self, *, num_warmup, num_samples, num_chains, seed)
      def save_summary(self, path: str | Path)
      def plot_traces(self, path: str | Path)
      def plot_fit_combined(self, path: str | Path)
  ```
- `mode` parameter: `"single" | "artefact_doublet"` (no `free_doublet` initially per §11).
- Internally translate API into `PeakAnnotation` instances.
- Acceptance: smoke test constructs a Fitter, adds annotations without error.

### Step 6.2 — `fit()` end-to-end orchestration
- Implementation pipeline:
  1. Compute baseline (existing baseline-region approach, ported from old code).
  2. Compute `noise_per_trace` from baseline regions.
  3. Call `priors.build_priors(...)` → `list[SkewNormalPriors]`.
  4. Construct `ModelInputs`.
  5. Run `numpyro.infer.MCMC(numpyro.infer.NUTS(model), ...)`.
  6. Store posterior samples.
- Acceptance:
  - Runs the user's test script (`fitter.fit(num_warmup=500, num_samples=500, num_chains=8, seed=42)`) on `data/raw` to completion.
  - **Diagnostic gate:** zero divergences, R̂ < 1.01 for all sampled sites, ESS > 400.

### Step 6.3 — `save_summary` and integration with ArviZ
- Implement `save_summary(path)`:
  - Uses `arviz.summary` filtered by `sampled_parameter_names(annotations)`.
  - Adds a Derived Quantities section with mode, FWHM, reported area.
- Acceptance: produced file is non-empty, includes all expected sections.

## Phase 7 — Visualization (`visualize.py`)

### Step 7.1 — Trace plot
- Create `chromhandler/fitting/visualize.py`.
- Implement `plot_traces(fitter, path)`:
  - Multi-panel grid: one row per peak window, one column per trace (or grouped).
  - Raw signal as scatter, baseline as dashed line, peak windows as vertical bands.
- Acceptance: produces a non-empty PNG; visually inspect on the test dataset.

### Step 7.2 — Posterior fit plot
- Implement `plot_fit_combined(fitter, path)`:
  - Per (trace, peak window): raw data + posterior median fit + 90% CI band.
  - Doublet windows show left and right components in distinct line styles, sum in solid.
- Acceptance: produces a non-empty PNG; manual visual check.

## Phase 8 — End-to-end validation

### Step 8.1 — Run the user's test script
- Execute the user's test from the conversation:
  ```python
  from chromhandler.fitting import Fitter
  from chromhandler.handler import Handler
  # ... full script ...
  fitter.fit(num_warmup=500, num_samples=500, num_chains=8, seed=42)
  fitter.save_summary("posterior_summary.txt")
  fitter.plot_traces(path="traces.png")
  fitter.plot_fit_combined(path="fit.png")
  ```
- Acceptance:
  - Runs to completion.
  - **Zero divergences.**
  - Visual check: peaks fit visibly well, doublet shoulders correctly attributed.
  - R̂ < 1.01 on all sites; ESS > 400.

### Step 8.2 — Synthetic-dataset regression test
- Add `tests/integration/test_skew_normal_fitter_synthetic.py`:
  - Generate a synthetic dataset with known ground truth (mix of singles and doublets, varying concentrations including some traces with vanished analyte).
  - Run the full Fitter pipeline.
  - Assert: per-peak `(mu, sigma, gamma1, A)` recovered within tolerance; reported areas match ground truth; zero divergences.
- Acceptance: test passes consistently on three random seeds.

### Step 8.3 — Update CI test paths
- Update `.github/workflows/tests.yml` if needed so the new test paths are picked up.
- Run `uv run pytest tests/` locally with coverage; confirm > 75% line coverage on `chromhandler/fitting/`.
- Acceptance: full local test suite green; coverage target met.

## Phase 9 — Documentation and cleanup

### Step 9.1 — Module-level docstrings
- Each module gets a Google-style module docstring summarizing scope, public API, and links to the spec.

### Step 9.2 — Update or remove old fitting docs
- Remove docs referencing the split-normal model.
- Add a brief migration note in `docs/usage/` explaining the new `mode` values and `artefact_side` semantics.

### Step 9.3 — Final lint/type pass
- `uv run ruff check chromhandler/fitting/ tests/unit/fitting/ tests/integration/`
- `uv run pyright chromhandler/fitting/ tests/unit/fitting/ tests/integration/`
- All green.

## Phase 10 — Merge prep

### Step 10.1 — PR description
- Draft a PR description summarising:
  - Switched from split-normal to skew-normal in CP form (mathematical justification: Fisher info, bounded skewness).
  - New module layout and naming conventions.
  - Method-of-moments priors with no magic numbers.
  - Doublet handling via signed Δ and outer-HWHM area decomposition with spatial assignment.
  - Convergence diagnostics: zero divergences on the test dataset.
- Link to the spec.

### Step 10.2 — Squash review pass
- Self-review the diff for stale comments, leftover prints, magic numbers, missed type hints.

---

## Estimated phase ordering and dependencies

```
Phase 0 ──▶ Phase 1 (math) ──┐
                              ├─▶ Phase 4 (model) ──▶ Phase 6 (fitter) ──▶ Phase 8 (validation) ──▶ Phase 9 ──▶ Phase 10
Phase 0 ──▶ Phase 2 (types) ─┘                           ▲
                              └─▶ Phase 3 (priors) ──────┘
                              └─▶ Phase 5 (posterior) ───┘
                              └─▶ Phase 7 (visualize) ──┐
                                                         ▼
                                                    Phase 8
```

Phases 1, 2, 3, 5, 7 are largely independent and could be parallelised across worktrees once Phase 0 is done. Phase 4 depends on 1 + 2; Phase 6 depends on everything except 7; Phase 7 depends on 1, 2, 6.
