# Fitter Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `model.py` (NumPyro MCMC), `posterior.py` (derived quantities + predictive sampling), and `fitter.py` (user-facing `fit()` + `FitResult` with debug plots) on top of the existing priors layer. Single-mode peaks only; doublet hooks documented in place.

**Architecture:** `fit(dataset, prior_config=None, model_config=None) -> FitResult`. Internally calls `build_priors` → `run_mcmc` → wraps `arviz.InferenceData` in `FitResult`. `FitResult` exposes `.plot_traces()`, `.plot_prior_overlay()`, `.plot_prior_predictive()`, `.plot_fit()`, `.summary()`, `.diagnostics()`, `.save()`. Lazy posterior_predictive / prior_predictive computed on first plot call and cached in `idata`.

**Tech Stack:** Python 3.11+, NumPyro, JAX (float64), NumPy, ArviZ, matplotlib, pandas, pytest. All execution via `uv run`.

**Spec:** [`docs/superpowers/specs/2026-05-12-fitter-integration-design.md`](../specs/2026-05-12-fitter-integration-design.md)

---

## Conventions

- Quality gates after every file edit: `uv run ruff check <file>` AND `uv run pyright <file>` must both pass.
- Per-file pytest runs are fine; the suite has pre-existing duplicate-name collection issues at the suite root.
- One commit per task. Format: `feat(model): ...`, `feat(posterior): ...`, `feat(fitter): ...`, `feat(priors): ...`.
- `from __future__ import annotations` at the top of every new module.
- All sample-site names use the `_left` suffix from day one (per spec §10, hook 1).
- Comments containing `TODO(doublet)` mark every extension point (per spec §10, hook documentation pattern).

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `chromhandler/fitting/priors.py` | Modify | Pad `log_A_left_loc_per_trace` to length `n_trace` for control compatibility |
| `chromhandler/fitting/model.py` | Create | `ModelConfig`, NumPyro `model()`, `run_mcmc()`, helpers |
| `chromhandler/fitting/posterior.py` | Create | predictive sampling + derived quantities + diagnostics |
| `chromhandler/fitting/fitter.py` | Create | `fit()` entry point, `FitResult` class with plot methods |
| `chromhandler/fitting/__init__.py` | Modify | Re-export `fit`, `FitResult`, `ModelConfig`, `PriorConfig` |
| `tests/unit/fitting/test_priors_control_padding.py` | Create | Task 0 |
| `tests/unit/fitting/test_model_config.py` | Create | Task 1 |
| `tests/unit/fitting/test_model_helpers.py` | Create | Task 2 |
| `tests/unit/fitting/test_model.py` | Create | Tasks 3 + 4 |
| `tests/unit/fitting/test_posterior.py` | Create | Task 5 |
| `tests/unit/fitting/test_fitter_class.py` | Create | Tasks 6 + 7 + 8 |
| `tests/unit/fitting/test_fitter_entry.py` | Create | Task 9 |
| `tests/unit/fitting/test_model_recovery.py` | Create | Task 10 |
| `tests/integration/test_fitter_asm.py` | Create | Task 11 |

---

## Task 0: Pad `log_A_left_loc_per_trace` to `n_trace` in `build_priors`

**Prerequisite for the model layer.** Today `build_priors` produces a `SkewNormalPriors.log_A_left_loc_per_trace` of length `n_non_control` because `aggregate_single_peak_priors` only sees non-control features. The model layer needs length `n_trace` with control entries pinned to `log(A_floor)` so the per-trace amplitude prior is well-defined for every trace.

**Files:**
- Modify: `chromhandler/fitting/priors.py` (the `build_priors` orchestrator, specifically the `mode == "single"` branch)
- Create: `tests/unit/fitting/test_priors_control_padding.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/fitting/test_priors_control_padding.py`:

```python
"""log_A_left_loc_per_trace must be length n_trace, with control entries at floor."""

from __future__ import annotations

import numpy as np
from scipy.stats import skewnorm

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.prepared_dataset import prepare_dataset
from chromhandler.fitting.priors import PriorConfig, build_priors


def _synth(n_sample: int = 3, n_control: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    times, signals, is_ctrl = [], [], []
    for amp in np.linspace(100.0, 30.0, n_sample):
        t = np.arange(2.5, 3.6, 0.001)
        s = amp * skewnorm.pdf(t, 0.0, loc=2.95, scale=0.025)
        s = s + 5.0 + rng.normal(0.0, 0.5, size=t.shape)
        times.append(t); signals.append(s); is_ctrl.append(False)
    for _ in range(n_control):
        t = np.arange(2.5, 3.6, 0.001)
        s = 5.0 + rng.normal(0.0, 0.5, size=t.shape)  # baseline + noise only
        times.append(t); signals.append(s); is_ctrl.append(True)
    peaks = [PeakAnnotation(molecule_id="A", rt_min=2.85, rt_max=3.10, mode="single")]
    bases = [
        BaselineAnnotation(rt_min=2.50, rt_max=2.52),
        BaselineAnnotation(rt_min=3.55, rt_max=3.58),
    ]
    return prepare_dataset(times, signals, peaks, bases, is_control=is_ctrl)


def test_log_A_array_has_full_n_trace_length() -> None:
    ds = _synth(n_sample=3, n_control=2)
    priors = build_priors(ds, config=PriorConfig())
    assert priors[0].log_A_left_loc_per_trace.shape == (ds.n_trace,)


def test_control_entries_are_at_floor() -> None:
    ds = _synth(n_sample=3, n_control=2)
    priors = build_priors(ds, config=PriorConfig())
    p = priors[0]
    control_idx = np.where(ds.is_control)[0]
    non_control_idx = np.where(~ds.is_control)[0]
    # Control entries should be much smaller than non-control entries.
    assert float(p.log_A_left_loc_per_trace[control_idx].max()) < float(
        p.log_A_left_loc_per_trace[non_control_idx].min()
    )
    # And finite.
    assert np.all(np.isfinite(p.log_A_left_loc_per_trace))
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_priors_control_padding.py -v
```

Expected: `test_log_A_array_has_full_n_trace_length` fails with shape `(3,)` not `(5,)`. The second test depends on the first.

- [ ] **Step 3: Patch `build_priors` single-mode branch**

In `chromhandler/fitting/priors.py`, find the `if ann.mode == "single":` branch inside `build_priors`. Replace with the padded version. Import `dataclasses` at the top of the file if not present.

```python
import dataclasses
```

Then replace the `mode == "single"` branch body (currently appends a `SkewNormalPriors` directly) with:

```python
        if ann.mode == "single":
            feats = [
                compute_single_window_features(
                    dataset.time[tr], baseline_sub[tr], ann.rt_min, ann.rt_max
                )
                for tr in non_control_idx
            ]
            subset_priors = aggregate_single_peak_priors(
                per_trace_features=feats,
                window_low=ann.rt_min, window_high=ann.rt_max,
                dt=dataset.dt_global,
                noise_per_trace=dataset.noise_per_trace[non_control_idx],
                n_window_points=n_pts, config=cfg,
            )
            # Pad log_A_left_loc_per_trace to length n_trace.
            # Control entries get log(A_floor) — the prior says "no analyte
            # expected here" (model layer fits them uniformly with a wide
            # amplitude prior; their posterior naturally pins at the floor).
            A_floor = (
                float(np.median(dataset.noise_per_trace))
                * float(np.sqrt(n_pts))
                * dataset.dt_global
            )
            log_A_full = np.full(dataset.n_trace, float(np.log(A_floor)), dtype=np.float64)
            log_A_full[non_control_idx] = subset_priors.log_A_left_loc_per_trace
            out.append(dataclasses.replace(
                subset_priors,
                log_A_left_loc_per_trace=log_A_full,
            ))
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/unit/fitting/test_priors_control_padding.py -v
uv run pytest tests/unit/fitting/ -q
uv run ruff check chromhandler/fitting/priors.py tests/unit/fitting/test_priors_control_padding.py
uv run pyright chromhandler/fitting/priors.py
```

Expected: 2 new tests pass, all priors tests still pass, ruff/pyright clean.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/priors.py tests/unit/fitting/test_priors_control_padding.py
git commit -m "feat(priors): pad log_A_left_loc_per_trace to n_trace for controls"
```

---

## Task 1: `ModelConfig` dataclass

**Files:**
- Create: `chromhandler/fitting/model.py`
- Create: `tests/unit/fitting/test_model_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/fitting/test_model_config.py`:

```python
"""ModelConfig defaults and overridability."""

from __future__ import annotations

from chromhandler.fitting.model import ModelConfig


def test_model_config_defaults() -> None:
    c = ModelConfig()
    assert c.num_warmup == 500
    assert c.num_samples == 500
    assert c.num_chains == 4
    assert c.target_accept_prob == 0.9
    assert c.max_tree_depth == 10
    assert c.seed == 0
    assert c.trace_shift_scale_dt_multiplier == 5.0
    assert c.baseline_intercept_se_floor == 1.0
    assert c.baseline_slope_se_floor == 0.01
    assert c.prior_predictive_n_samples == 200


def test_model_config_overridable() -> None:
    c = ModelConfig(num_warmup=1000, num_chains=2, seed=42)
    assert c.num_warmup == 1000
    assert c.num_chains == 2
    assert c.seed == 42
    # Untouched fields keep defaults
    assert c.target_accept_prob == 0.9
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_model_config.py -v
```

Expected: `ImportError` from `chromhandler.fitting.model`.

- [ ] **Step 3: Implement `ModelConfig`**

Create `chromhandler/fitting/model.py`:

```python
"""NumPyro Bayesian model for the skew-normal peak fitter.

Single-mode peaks only at present. Doublet support is a documented
extension — see TODO(doublet) markers throughout this module and the
"Doublet extension hooks" section of the design spec.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """User-facing configuration for the NumPyro fit.

    Tuned defaults for fast development iteration on chromatographic data.
    Override fields directly when constructing for publication-quality runs.
    """

    # --- HMC / NUTS settings ---
    num_warmup: int = 500
    num_samples: int = 500
    num_chains: int = 4
    target_accept_prob: float = 0.9
    max_tree_depth: int = 10
    seed: int = 0

    # --- Model-layer priors (per-trace, not per-peak) ---
    trace_shift_scale_dt_multiplier: float = 5.0
    """drift_scale = N * dt_global. trace_shift ~ Normal(0, drift_scale)."""

    baseline_intercept_se_floor: float = 1.0
    """Minimum SE for the baseline intercept prior (signal units)."""

    baseline_slope_se_floor: float = 0.01
    """Minimum SE for the baseline slope prior (signal units per minute)."""

    # --- Prior predictive ---
    prior_predictive_n_samples: int = 200
    """Number of prior samples used to compute prior predictive band."""
```

- [ ] **Step 4: Run tests + quality gates**

```bash
uv run pytest tests/unit/fitting/test_model_config.py -v
uv run ruff check chromhandler/fitting/model.py tests/unit/fitting/test_model_config.py
uv run pyright chromhandler/fitting/model.py tests/unit/fitting/test_model_config.py
```

Expected: 2 tests pass, clean.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/model.py tests/unit/fitting/test_model_config.py
git commit -m "feat(model): ModelConfig dataclass with HMC + per-trace prior defaults"
```

---

## Task 2: Model helpers — validation, baseline SE, density evaluator

**Files:**
- Modify: `chromhandler/fitting/model.py`
- Create: `tests/unit/fitting/test_model_helpers.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/fitting/test_model_helpers.py`:

```python
"""Tests for model.py private helpers."""

from __future__ import annotations

import numpy as np
import pytest

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.model import (
    SAMPLED_LEFT_PER_TRACE,
    SAMPLED_LEFT_SHARED,
    SAMPLED_TRACE_NUISANCE,
    _baseline_contribution,
    _compute_baseline_se,
    _left_component_contribution,
    _validate_single_mode_only,
)
from chromhandler.fitting.prepared_dataset import prepare_dataset
from chromhandler.fitting.priors import PriorConfig, build_priors


def _toy_dataset(n_trace: int = 3):
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 5.0, 0.01)
    times = [t.copy() for _ in range(n_trace)]
    signals = [10.0 + 0.5 * t + rng.normal(0.0, 0.1, size=t.shape) for _ in range(n_trace)]
    peaks = [PeakAnnotation(molecule_id="A", rt_min=2.0, rt_max=3.0, mode="single")]
    bases = [BaselineAnnotation(rt_min=0.5, rt_max=1.5),
             BaselineAnnotation(rt_min=3.5, rt_max=4.5)]
    return prepare_dataset(times, signals, peaks, bases)


def test_sample_name_constants_present() -> None:
    assert "mu_anchor_left" in SAMPLED_LEFT_SHARED
    assert "log_sigma_left" in SAMPLED_LEFT_SHARED
    assert "gamma1_left" in SAMPLED_LEFT_SHARED
    assert "log_A_left" in SAMPLED_LEFT_PER_TRACE
    assert "trace_shift" in SAMPLED_TRACE_NUISANCE
    assert "baseline_intercept" in SAMPLED_TRACE_NUISANCE
    assert "baseline_slope" in SAMPLED_TRACE_NUISANCE


def test_validate_single_mode_passes_on_single() -> None:
    ds = _toy_dataset()
    priors = build_priors(ds, config=PriorConfig())
    _validate_single_mode_only(priors)  # no raise


def test_validate_single_mode_raises_on_doublet() -> None:
    # Hand-construct a doublet prior by replace
    import dataclasses

    from chromhandler.fitting.priors import SkewNormalPriors
    ds = _toy_dataset()
    p = build_priors(ds, config=PriorConfig())[0]
    p_doublet = dataclasses.replace(
        p,
        n_components=2,
        Delta_loc=0.05, Delta_scale=0.005, Delta_low=0.003, Delta_high=0.125,
        log_sigma_right_loc=p.log_sigma_left_loc, log_sigma_right_scale=p.log_sigma_left_scale,
        log_sigma_right_low=p.log_sigma_left_low, log_sigma_right_high=p.log_sigma_left_high,
        gamma1_right_loc=p.gamma1_left_loc, gamma1_right_scale=p.gamma1_left_scale,
        log_A_right_loc_per_trace=p.log_A_left_loc_per_trace, log_A_right_scale=p.log_A_left_scale,
    )
    with pytest.raises(NotImplementedError, match="single"):
        _validate_single_mode_only([p_doublet])


def test_compute_baseline_se_returns_per_trace() -> None:
    ds = _toy_dataset(n_trace=3)
    intercept_se, slope_se = _compute_baseline_se(ds)
    assert intercept_se.shape == (3,)
    assert slope_se.shape == (3,)
    assert np.all(intercept_se >= 0)
    assert np.all(slope_se >= 0)


def test_baseline_contribution_shape() -> None:
    ds = _toy_dataset(n_trace=3)
    intercept = np.zeros(3)
    slope = np.zeros(3)
    bc = _baseline_contribution(ds.time, intercept, slope)
    assert bc.shape == ds.time.shape
    np.testing.assert_array_equal(bc, np.zeros_like(ds.time))


def test_left_component_contribution_shape_and_finite() -> None:
    ds = _toy_dataset(n_trace=3)
    n_peak = 1
    mu_anchor = np.array([2.5])
    trace_shift = np.zeros(3)
    log_sigma = np.array([np.log(0.05)])
    gamma1 = np.array([0.0])
    log_A = np.full((3, n_peak), np.log(10.0))
    out = _left_component_contribution(ds.time, mu_anchor, trace_shift, log_sigma, gamma1, log_A)
    assert out.shape == ds.time.shape
    assert np.all(np.isfinite(out))
    # Signal should be positive everywhere a peak can exist
    assert float(out.max()) > 0
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_model_helpers.py -v
```

Expected: `ImportError` on the helper names.

- [ ] **Step 3: Implement the helpers**

Append to `chromhandler/fitting/model.py`:

```python
import numpy as np
from numpy.typing import NDArray

from chromhandler.fitting.prepared_dataset import PreparedDataset
from chromhandler.fitting.priors import SkewNormalPriors
from chromhandler.fitting.skew_normal import density_cp

# --- Sample-site name constants (TODO(doublet): populate SAMPLED_RIGHT_* below) ---
SAMPLED_LEFT_SHARED: tuple[str, ...] = ("mu_anchor_left", "log_sigma_left", "gamma1_left")
SAMPLED_LEFT_PER_TRACE: tuple[str, ...] = ("log_A_left",)
SAMPLED_TRACE_NUISANCE: tuple[str, ...] = (
    "trace_shift", "baseline_intercept", "baseline_slope",
)
SAMPLED_RIGHT_SHARED: tuple[str, ...] = ()        # TODO(doublet)
SAMPLED_RIGHT_PER_TRACE: tuple[str, ...] = ()     # TODO(doublet)


def _validate_single_mode_only(priors_list: list[SkewNormalPriors]) -> None:
    """Raise if any peak in priors_list has n_components > 1.

    Hoisted out of model() so the JIT-compiled hot path is clean.
    """
    doublet = [i for i, p in enumerate(priors_list) if p.n_components == 2]
    if doublet:
        raise NotImplementedError(
            f"model.py supports n_components=1 (single) peaks only. "
            f"Doublet peaks at indices {doublet}. Doublet support is a "
            f"documented future extension — see model.py module docstring "
            f"and `# TODO(doublet)` markers."
        )


def _compute_baseline_se(
    dataset: PreparedDataset,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-trace OLS standard errors for the baseline intercept and slope.

    Computed from the residuals of the baseline OLS fit on each trace's
    annotated baseline regions. Returns ``(intercept_se, slope_se)``,
    both shape ``[n_trace]``.

    Used by ``model()`` to set the Normal priors on baseline parameters.
    """
    n_trace = dataset.n_trace
    intercept_se = np.zeros(n_trace, dtype=np.float64)
    slope_se = np.zeros(n_trace, dtype=np.float64)

    for tr in range(n_trace):
        t = dataset.time[tr]
        s = dataset.signal[tr]
        baseline_mask = np.zeros_like(t, dtype=bool)
        for ba in dataset.baseline_annotations:
            baseline_mask |= ((t >= ba.rt_min) & (t <= ba.rt_max) & np.isfinite(s))
        if baseline_mask.sum() < 3:
            # Fall back to noise std as a wide-but-finite SE.
            intercept_se[tr] = float(dataset.noise_per_trace[tr])
            slope_se[tr] = float(dataset.noise_per_trace[tr])
            continue
        t_b = t[baseline_mask]
        s_b = s[baseline_mask]
        # OLS via lstsq with design matrix [1, t]
        X = np.column_stack([np.ones_like(t_b), t_b])
        beta, *_ = np.linalg.lstsq(X, s_b, rcond=None)
        residuals = s_b - X @ beta
        # Standard OLS covariance
        sigma2 = float(np.sum(residuals**2) / max(t_b.size - 2, 1))
        try:
            cov = sigma2 * np.linalg.inv(X.T @ X)
            intercept_se[tr] = float(np.sqrt(max(cov[0, 0], 0.0)))
            slope_se[tr] = float(np.sqrt(max(cov[1, 1], 0.0)))
        except np.linalg.LinAlgError:
            intercept_se[tr] = float(dataset.noise_per_trace[tr])
            slope_se[tr] = float(dataset.noise_per_trace[tr])
    return intercept_se, slope_se


def _baseline_contribution(
    time: NDArray[np.float64],
    intercept: NDArray[np.float64],
    slope: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-trace baseline = intercept + slope * t. Shape [n_trace, n_time]."""
    return intercept[:, None] + slope[:, None] * time


def _left_component_contribution(
    time: NDArray[np.float64],
    mu_anchor: NDArray[np.float64],
    trace_shift: NDArray[np.float64],
    log_sigma: NDArray[np.float64],
    gamma1: NDArray[np.float64],
    log_A: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sum of left-component skew-normal densities per (trace, time).

    Args:
        time: [n_trace, n_time]
        mu_anchor: [n_peak]
        trace_shift: [n_trace]
        log_sigma: [n_peak]
        gamma1: [n_peak]
        log_A: [n_trace, n_peak]

    Returns:
        Predicted signal [n_trace, n_time].
    """
    n_trace, n_time = time.shape
    n_peak = mu_anchor.shape[0]
    sigma = np.exp(log_sigma)
    # mu[trace, peak] = mu_anchor[peak] + trace_shift[trace]
    mu = mu_anchor[None, :] + trace_shift[:, None]    # [n_trace, n_peak]
    A = np.exp(log_A)                                  # [n_trace, n_peak]

    out = np.zeros((n_trace, n_time), dtype=np.float64)
    for peak in range(n_peak):
        # density_cp accepts vectorised inputs; here we evaluate per-peak
        # over all (trace, time) at once.
        density = np.asarray(density_cp(
            time,                                      # [n_trace, n_time]
            mu[:, peak:peak + 1],                      # [n_trace, 1]
            sigma[peak],
            gamma1[peak],
        ))
        out = out + A[:, peak:peak + 1] * density
    return out
```

- [ ] **Step 4: Run tests + quality gates**

```bash
uv run pytest tests/unit/fitting/test_model_helpers.py -v
uv run ruff check chromhandler/fitting/model.py tests/unit/fitting/test_model_helpers.py
uv run pyright chromhandler/fitting/model.py tests/unit/fitting/test_model_helpers.py
```

Expected: 6 tests pass, clean.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/model.py tests/unit/fitting/test_model_helpers.py
git commit -m "feat(model): helpers — validate, baseline SE, density evaluator"
```

---

## Task 3: NumPyro `model()` function

**Files:**
- Modify: `chromhandler/fitting/model.py`
- Create: `tests/unit/fitting/test_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/fitting/test_model.py`:

```python
"""Tests for the NumPyro model() function."""

from __future__ import annotations

import jax
import numpy as np
import numpyro
from scipy.stats import skewnorm

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.model import ModelConfig, model
from chromhandler.fitting.prepared_dataset import prepare_dataset
from chromhandler.fitting.priors import PriorConfig, build_priors


def _toy_setup(n_trace: int = 3):
    rng = np.random.default_rng(0)
    t = np.arange(2.5, 3.6, 0.001)
    times = [t.copy() for _ in range(n_trace)]
    signals = []
    for amp in np.linspace(100.0, 40.0, n_trace):
        s = amp * skewnorm.pdf(t, 0.0, loc=3.0, scale=0.025)
        s = s + 5.0 + rng.normal(0.0, 0.5, size=t.shape)
        signals.append(s)
    peaks = [PeakAnnotation(molecule_id="A", rt_min=2.85, rt_max=3.15, mode="single")]
    bases = [BaselineAnnotation(rt_min=2.50, rt_max=2.52),
             BaselineAnnotation(rt_min=3.55, rt_max=3.58)]
    ds = prepare_dataset(times, signals, peaks, bases)
    priors = build_priors(ds, config=PriorConfig())
    return ds, priors


def test_model_prior_predictive_runs_and_has_right_shape() -> None:
    ds, priors = _toy_setup(n_trace=3)
    config = ModelConfig(num_warmup=1, num_samples=1, num_chains=1)
    predictive = numpyro.infer.Predictive(model, num_samples=2)
    rng_key = jax.random.PRNGKey(0)
    samples = predictive(rng_key, ds, priors, config)
    # All expected sites present
    assert "mu_anchor_left" in samples
    assert "log_sigma_left" in samples
    assert "gamma1_left" in samples
    assert "log_A_left" in samples
    assert "trace_shift" in samples
    assert "baseline_intercept" in samples
    assert "baseline_slope" in samples
    assert "obs" in samples
    # Shapes
    assert samples["mu_anchor_left"].shape == (2, 1)            # (n_samples, n_peak)
    assert samples["log_A_left"].shape == (2, ds.n_trace, 1)    # (n_samples, n_trace, n_peak)
    assert samples["trace_shift"].shape == (2, ds.n_trace)
    assert samples["obs"].shape == (2, ds.n_trace, ds.time.shape[1])


def test_model_obs_values_are_finite_under_prior() -> None:
    ds, priors = _toy_setup(n_trace=3)
    config = ModelConfig(num_warmup=1, num_samples=1, num_chains=1)
    predictive = numpyro.infer.Predictive(model, num_samples=5)
    rng_key = jax.random.PRNGKey(0)
    samples = predictive(rng_key, ds, priors, config)
    # `obs` from prior predictive can be large but should be finite
    assert np.all(np.isfinite(np.asarray(samples["obs"])))
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_model.py -v
```

Expected: ImportError on `model` function name.

- [ ] **Step 3: Implement `model()`**

Append to `chromhandler/fitting/model.py`. First add the imports near the top with the existing ones:

```python
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
```

Then append after the existing helpers:

```python
from chromhandler.fitting.skew_normal import GAMMA1_MAX, density_cp as _density_cp  # noqa: F401


def model(
    dataset: PreparedDataset,
    priors_list: list[SkewNormalPriors],
    config: ModelConfig,
) -> None:
    """NumPyro Bayesian model for the skew-normal peak fitter.

    Single-mode peaks only. ``run_mcmc`` calls
    ``_validate_single_mode_only`` before invoking this function.

    Sample sites (single mode):
        - mu_anchor_left[peak]
        - log_sigma_left[peak]
        - gamma1_left[peak]
        - log_A_left[trace, peak]
        - trace_shift[trace]
        - baseline_intercept[trace]
        - baseline_slope[trace]
        - obs (likelihood, NaN-masked)

    TODO(doublet): when adding doublet support,
        - sample Delta[doublet_peak], log_sigma_right[doublet_peak],
          gamma1_right[doublet_peak], log_A_right[trace, doublet_peak]
        - add right-component contribution to predicted
        - remove the _validate_single_mode_only call from run_mcmc
    """
    n_trace = dataset.n_trace
    n_peak = len(priors_list)
    dt_global = float(dataset.dt_global)

    # === Left-component shared shape priors ===
    mu_loc = jnp.asarray([p.mu_left_loc for p in priors_list])
    mu_scale = jnp.asarray([p.mu_left_scale for p in priors_list])
    mu_low = jnp.asarray([p.mu_left_low for p in priors_list])
    mu_high = jnp.asarray([p.mu_left_high for p in priors_list])
    mu_anchor_left = numpyro.sample(
        "mu_anchor_left",
        dist.TruncatedNormal(loc=mu_loc, scale=mu_scale, low=mu_low, high=mu_high),
    )  # [n_peak]

    log_sigma_loc = jnp.asarray([p.log_sigma_left_loc for p in priors_list])
    log_sigma_scale = jnp.asarray([p.log_sigma_left_scale for p in priors_list])
    log_sigma_low = jnp.asarray([p.log_sigma_left_low for p in priors_list])
    log_sigma_high = jnp.asarray([p.log_sigma_left_high for p in priors_list])
    log_sigma_left = numpyro.sample(
        "log_sigma_left",
        dist.TruncatedNormal(
            loc=log_sigma_loc, scale=log_sigma_scale,
            low=log_sigma_low, high=log_sigma_high,
        ),
    )

    gamma1_loc = jnp.asarray([p.gamma1_left_loc for p in priors_list])
    gamma1_scale = jnp.asarray([p.gamma1_left_scale for p in priors_list])
    gamma1_bound = 0.99 * float(GAMMA1_MAX)
    gamma1_left = numpyro.sample(
        "gamma1_left",
        dist.TruncatedNormal(
            loc=gamma1_loc, scale=gamma1_scale,
            low=-gamma1_bound, high=gamma1_bound,
        ),
    )

    # === Per-trace amplitude: Normal(loc_per_trace, scale) ===
    log_A_loc = jnp.asarray(
        np.stack([p.log_A_left_loc_per_trace for p in priors_list], axis=1)
    )  # [n_trace, n_peak]
    log_A_scale = jnp.asarray([p.log_A_left_scale for p in priors_list])  # [n_peak]
    log_A_left = numpyro.sample(
        "log_A_left",
        dist.Normal(loc=log_A_loc, scale=log_A_scale[None, :]),
    )

    # === Per-trace nuisance ===
    drift_scale = config.trace_shift_scale_dt_multiplier * dt_global
    trace_shift = numpyro.sample(
        "trace_shift",
        dist.Normal(loc=jnp.zeros(n_trace), scale=drift_scale),
    )

    intercept_se, slope_se = _compute_baseline_se(dataset)
    intercept_se_eff = np.maximum(intercept_se, config.baseline_intercept_se_floor)
    slope_se_eff = np.maximum(slope_se, config.baseline_slope_se_floor)
    baseline_intercept = numpyro.sample(
        "baseline_intercept",
        dist.Normal(
            loc=jnp.asarray(dataset.baseline_intercept),
            scale=jnp.asarray(intercept_se_eff),
        ),
    )
    baseline_slope = numpyro.sample(
        "baseline_slope",
        dist.Normal(
            loc=jnp.asarray(dataset.baseline_slope),
            scale=jnp.asarray(slope_se_eff),
        ),
    )

    # === DOUBLET EXTENSION HOOK ===
    # TODO(doublet): sample right-component params here:
    #   Delta, log_sigma_right, gamma1_right, log_A_right
    # and add right_contrib = ... to `predicted` below.

    # === Predicted signal ===
    sigma_left = jnp.exp(log_sigma_left)
    A_left = jnp.exp(log_A_left)
    mu = mu_anchor_left[None, :] + trace_shift[:, None]  # [n_trace, n_peak]

    baseline = baseline_intercept[:, None] + baseline_slope[:, None] * jnp.asarray(dataset.time)

    left_contrib = jnp.zeros_like(jnp.asarray(dataset.time))
    for peak in range(n_peak):
        dens = density_cp(
            jnp.asarray(dataset.time),
            mu[:, peak:peak + 1],
            sigma_left[peak],
            gamma1_left[peak],
        )
        left_contrib = left_contrib + A_left[:, peak:peak + 1] * dens
    # TODO(doublet): + right_contrib
    predicted = baseline + left_contrib

    # === Likelihood (NaN-masked) ===
    noise = jnp.asarray(dataset.noise_per_trace)
    with numpyro.handlers.mask(mask=jnp.asarray(dataset.valid_mask)):
        numpyro.sample(
            "obs",
            dist.Normal(predicted, noise[:, None]),
            obs=jnp.asarray(dataset.signal),
        )
```

- [ ] **Step 4: Run tests + quality gates**

```bash
uv run pytest tests/unit/fitting/test_model.py -v
uv run ruff check chromhandler/fitting/model.py tests/unit/fitting/test_model.py
uv run pyright chromhandler/fitting/model.py tests/unit/fitting/test_model.py
```

Expected: 2 tests pass. The prior predictive run takes a few seconds.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/model.py tests/unit/fitting/test_model.py
git commit -m "feat(model): NumPyro model() with TruncatedNormal/Normal priors + NaN-masked likelihood"
```

---

## Task 4: `run_mcmc()` orchestrator

**Files:**
- Modify: `chromhandler/fitting/model.py`
- Modify: `tests/unit/fitting/test_model.py` (extend)

- [ ] **Step 1: Append the failing test**

Append to `tests/unit/fitting/test_model.py`:

```python
import arviz as az

from chromhandler.fitting.model import run_mcmc


def test_run_mcmc_returns_inferencedata() -> None:
    ds, priors = _toy_setup(n_trace=3)
    config = ModelConfig(num_warmup=20, num_samples=20, num_chains=2, seed=0)
    idata = run_mcmc(ds, priors, config)
    assert isinstance(idata, az.InferenceData)
    # posterior group exists
    assert hasattr(idata, "posterior")
    # expected variables
    posterior_vars = set(idata.posterior.data_vars)
    assert "mu_anchor_left" in posterior_vars
    assert "log_sigma_left" in posterior_vars
    assert "gamma1_left" in posterior_vars
    assert "log_A_left" in posterior_vars


def test_run_mcmc_validates_single_mode() -> None:
    import dataclasses

    from chromhandler.fitting.priors import SkewNormalPriors
    ds, priors = _toy_setup(n_trace=3)
    p = priors[0]
    p_doublet = dataclasses.replace(
        p,
        n_components=2,
        Delta_loc=0.05, Delta_scale=0.005, Delta_low=0.003, Delta_high=0.125,
        log_sigma_right_loc=p.log_sigma_left_loc, log_sigma_right_scale=p.log_sigma_left_scale,
        log_sigma_right_low=p.log_sigma_left_low, log_sigma_right_high=p.log_sigma_left_high,
        gamma1_right_loc=p.gamma1_left_loc, gamma1_right_scale=p.gamma1_left_scale,
        log_A_right_loc_per_trace=p.log_A_left_loc_per_trace, log_A_right_scale=p.log_A_left_scale,
    )
    config = ModelConfig(num_warmup=10, num_samples=10, num_chains=1)
    import pytest
    with pytest.raises(NotImplementedError, match="single"):
        run_mcmc(ds, [p_doublet], config)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_model.py::test_run_mcmc_returns_inferencedata -v
```

Expected: ImportError on `run_mcmc`.

- [ ] **Step 3: Implement `run_mcmc`**

Append to `chromhandler/fitting/model.py`:

```python
import arviz
import jax


def run_mcmc(
    dataset: PreparedDataset,
    priors_list: list[SkewNormalPriors],
    config: ModelConfig,
) -> arviz.InferenceData:
    """Run NUTS sampling and return an ArviZ InferenceData.

    Args:
        dataset: PreparedDataset to fit.
        priors_list: One SkewNormalPriors per peak annotation.
        config: ModelConfig with HMC settings.

    Returns:
        arviz.InferenceData with `posterior` and `observed_data` groups.

    Raises:
        NotImplementedError: If any prior has n_components > 1.
    """
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
    mcmc.run(
        jax.random.PRNGKey(config.seed),
        dataset, priors_list, config,
    )
    return arviz.from_numpyro(mcmc)
```

- [ ] **Step 4: Run tests + quality gates**

```bash
uv run pytest tests/unit/fitting/test_model.py -v
uv run ruff check chromhandler/fitting/model.py tests/unit/fitting/test_model.py
uv run pyright chromhandler/fitting/model.py tests/unit/fitting/test_model.py
```

Expected: 2 new tests pass. `test_run_mcmc_returns_inferencedata` takes ~10-20 seconds.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/model.py tests/unit/fitting/test_model.py
git commit -m "feat(model): run_mcmc() orchestrator returning arviz.InferenceData"
```

---

## Task 5: `posterior.py` — predictive sampling + derived quantities + diagnostics

**Files:**
- Create: `chromhandler/fitting/posterior.py`
- Create: `tests/unit/fitting/test_posterior.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/fitting/test_posterior.py`:

```python
"""Tests for posterior.py helpers."""

from __future__ import annotations

import arviz as az
import numpy as np

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.model import ModelConfig, run_mcmc
from chromhandler.fitting.posterior import (
    compute_posterior_predictive,
    compute_prior_predictive,
    derived_areas,
    diagnostics,
)
from chromhandler.fitting.prepared_dataset import prepare_dataset
from chromhandler.fitting.priors import PriorConfig, build_priors


def _idata_fixture():
    rng = np.random.default_rng(0)
    from scipy.stats import skewnorm
    t = np.arange(2.5, 3.6, 0.001)
    times = [t.copy() for _ in range(3)]
    signals = [
        amp * skewnorm.pdf(t, 0.0, loc=3.0, scale=0.025) + 5.0
        + rng.normal(0.0, 0.5, size=t.shape)
        for amp in (100.0, 60.0, 30.0)
    ]
    peaks = [PeakAnnotation(molecule_id="A", rt_min=2.85, rt_max=3.15, mode="single")]
    bases = [BaselineAnnotation(rt_min=2.50, rt_max=2.52),
             BaselineAnnotation(rt_min=3.55, rt_max=3.58)]
    ds = prepare_dataset(times, signals, peaks, bases)
    priors = build_priors(ds, config=PriorConfig())
    config = ModelConfig(num_warmup=30, num_samples=30, num_chains=2, seed=0)
    idata = run_mcmc(ds, priors, config)
    return idata, ds, priors, config


def test_compute_posterior_predictive_adds_group() -> None:
    idata, ds, priors, config = _idata_fixture()
    out = compute_posterior_predictive(idata, ds, priors, config)
    assert isinstance(out, az.InferenceData)
    assert hasattr(out, "posterior_predictive")


def test_compute_prior_predictive_adds_group() -> None:
    idata, ds, priors, config = _idata_fixture()
    out = compute_prior_predictive(idata, ds, priors, config)
    assert isinstance(out, az.InferenceData)
    assert hasattr(out, "prior")
    assert hasattr(out, "prior_predictive")


def test_derived_areas_shape() -> None:
    idata, ds, priors, config = _idata_fixture()
    areas = derived_areas(idata)
    # [chain, draw, trace, peak]
    assert areas.ndim == 4
    assert areas.shape[2] == ds.n_trace
    assert areas.shape[3] == len(priors)
    assert np.all(areas > 0)  # exp(log_A) > 0


def test_diagnostics_keys_and_types() -> None:
    idata, *_ = _idata_fixture()
    d = diagnostics(idata)
    assert set(d.keys()) >= {
        "r_hat_max", "r_hat_max_param",
        "ess_min_bulk", "ess_min_param",
        "n_divergent", "fit_healthy",
    }
    assert isinstance(d["r_hat_max"], float)
    assert isinstance(d["fit_healthy"], bool)
    assert isinstance(d["n_divergent"], int)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_posterior.py -v
```

Expected: ImportError on `chromhandler.fitting.posterior`.

- [ ] **Step 3: Implement `posterior.py`**

Create `chromhandler/fitting/posterior.py`:

```python
"""Derived quantities and predictive sampling on top of an ArviZ InferenceData.

Pure functions — no plotting. All inputs/outputs are arviz.InferenceData
or numpy arrays. Called from FitResult methods to compute predictive
samples lazily and to extract diagnostics dicts.
"""

from __future__ import annotations

from typing import Any

import arviz
import jax
import numpy as np
import numpyro

from chromhandler.fitting.model import ModelConfig, model
from chromhandler.fitting.prepared_dataset import PreparedDataset
from chromhandler.fitting.priors import SkewNormalPriors


def compute_posterior_predictive(
    idata: arviz.InferenceData,
    dataset: PreparedDataset,
    priors_list: list[SkewNormalPriors],
    config: ModelConfig,
) -> arviz.InferenceData:
    """Sample posterior predictive `obs` and add a `posterior_predictive` group.

    Mutates the passed InferenceData and returns it.
    """
    # Extract posterior samples flattened across (chain, draw)
    posterior = idata.posterior
    n_chain = int(posterior.sizes["chain"])
    n_draw = int(posterior.sizes["draw"])
    # Build a flat posterior dict for Predictive
    flat_posterior = {
        name: np.asarray(posterior[name]).reshape((n_chain * n_draw, *posterior[name].shape[2:]))
        for name in posterior.data_vars
    }
    predictive = numpyro.infer.Predictive(
        model, posterior_samples=flat_posterior, return_sites=["obs"],
    )
    rng_key = jax.random.PRNGKey(config.seed + 1)
    samples = predictive(rng_key, dataset, priors_list, config)
    obs = np.asarray(samples["obs"]).reshape(
        (n_chain, n_draw, dataset.n_trace, dataset.time.shape[1])
    )
    # Build posterior_predictive group manually
    coords = {
        "chain": np.arange(n_chain),
        "draw": np.arange(n_draw),
        "trace": np.arange(dataset.n_trace),
        "time_idx": np.arange(dataset.time.shape[1]),
    }
    pp = arviz.from_dict(
        posterior_predictive={"obs": obs},
        coords=coords,
        dims={"obs": ["trace", "time_idx"]},
    )
    idata.extend(pp)
    return idata


def compute_prior_predictive(
    idata: arviz.InferenceData,
    dataset: PreparedDataset,
    priors_list: list[SkewNormalPriors],
    config: ModelConfig,
) -> arviz.InferenceData:
    """Sample from the prior, run model forward; adds `prior` + `prior_predictive`."""
    predictive = numpyro.infer.Predictive(model, num_samples=config.prior_predictive_n_samples)
    rng_key = jax.random.PRNGKey(config.seed + 2)
    samples = predictive(rng_key, dataset, priors_list, config)

    n_samples = config.prior_predictive_n_samples
    n_trace = dataset.n_trace
    n_time = dataset.time.shape[1]

    obs = np.asarray(samples["obs"]).reshape((1, n_samples, n_trace, n_time))
    coords = {
        "chain": [0],
        "draw": np.arange(n_samples),
        "trace": np.arange(n_trace),
        "time_idx": np.arange(n_time),
    }
    pp = arviz.from_dict(
        prior={
            name: np.asarray(samples[name]).reshape((1, n_samples, *np.asarray(samples[name]).shape[1:]))
            for name in samples if name != "obs"
        },
        prior_predictive={"obs": obs},
        coords=coords,
        dims={"obs": ["trace", "time_idx"]},
    )
    idata.extend(pp)
    return idata


def derived_areas(idata: arviz.InferenceData) -> np.ndarray:
    """Per-(chain, draw, trace, peak) posterior areas = exp(log_A_left).

    Returns:
        Array shape ``[n_chain, n_draw, n_trace, n_peak]``.
    """
    log_A = np.asarray(idata.posterior["log_A_left"])
    return np.exp(log_A)


def diagnostics(idata: arviz.InferenceData) -> dict[str, Any]:
    """Quick "did MCMC converge?" summary.

    Returns:
        {
            "r_hat_max": float, "r_hat_max_param": str,
            "ess_min_bulk": float, "ess_min_param": str,
            "n_divergent": int,
            "n_samples_total": int,
            "fit_healthy": bool,
        }
    """
    summary = arviz.summary(idata, kind="diagnostics")
    r_hat = summary["r_hat"]
    ess_bulk = summary["ess_bulk"]

    r_hat_max = float(r_hat.max())
    r_hat_max_param = str(r_hat.idxmax())
    ess_min_bulk = float(ess_bulk.min())
    ess_min_param = str(ess_bulk.idxmin())

    n_divergent = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        n_divergent = int(np.asarray(idata.sample_stats["diverging"]).sum())

    n_chain = int(idata.posterior.sizes["chain"])
    n_draw = int(idata.posterior.sizes["draw"])
    n_samples_total = n_chain * n_draw

    fit_healthy = bool(
        r_hat_max < 1.01 and ess_min_bulk > 400 and n_divergent == 0
    )

    return {
        "r_hat_max": r_hat_max,
        "r_hat_max_param": r_hat_max_param,
        "ess_min_bulk": ess_min_bulk,
        "ess_min_param": ess_min_param,
        "n_divergent": n_divergent,
        "n_samples_total": n_samples_total,
        "fit_healthy": fit_healthy,
    }
```

- [ ] **Step 4: Run tests + quality gates**

```bash
uv run pytest tests/unit/fitting/test_posterior.py -v
uv run ruff check chromhandler/fitting/posterior.py tests/unit/fitting/test_posterior.py
uv run pyright chromhandler/fitting/posterior.py tests/unit/fitting/test_posterior.py
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/posterior.py tests/unit/fitting/test_posterior.py
git commit -m "feat(posterior): predictive sampling + derived_areas + diagnostics"
```

---

## Task 6: `FitResult` class — construction + save

**Files:**
- Create: `chromhandler/fitting/fitter.py`
- Create: `tests/unit/fitting/test_fitter_class.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/fitting/test_fitter_class.py`:

```python
"""Tests for FitResult class construction + save."""

from __future__ import annotations

from pathlib import Path

import arviz as az
import numpy as np
from scipy.stats import skewnorm

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.fitter import FitResult
from chromhandler.fitting.model import ModelConfig, run_mcmc
from chromhandler.fitting.prepared_dataset import prepare_dataset
from chromhandler.fitting.priors import PriorConfig, build_priors


def _result_fixture():
    rng = np.random.default_rng(0)
    t = np.arange(2.5, 3.6, 0.001)
    times = [t.copy() for _ in range(3)]
    signals = [
        amp * skewnorm.pdf(t, 0.0, loc=3.0, scale=0.025) + 5.0
        + rng.normal(0.0, 0.5, size=t.shape)
        for amp in (100.0, 60.0, 30.0)
    ]
    peaks = [PeakAnnotation(molecule_id="A", rt_min=2.85, rt_max=3.15, mode="single")]
    bases = [BaselineAnnotation(rt_min=2.50, rt_max=2.52),
             BaselineAnnotation(rt_min=3.55, rt_max=3.58)]
    ds = prepare_dataset(times, signals, peaks, bases)
    priors = build_priors(ds, config=PriorConfig())
    config = ModelConfig(num_warmup=30, num_samples=30, num_chains=2, seed=0)
    idata = run_mcmc(ds, priors, config)
    return FitResult(idata=idata, dataset=ds, priors=priors, model_config=config)


def test_fitresult_construction() -> None:
    result = _result_fixture()
    assert isinstance(result.idata, az.InferenceData)
    assert result.dataset.n_trace == 3
    assert len(result.priors) == 1
    assert result.model_config.seed == 0


def test_fitresult_save_and_load(tmp_path: Path) -> None:
    result = _result_fixture()
    out_path = tmp_path / "result.nc"
    result.save(out_path)
    assert out_path.exists()
    # Roundtrip through ArviZ
    reloaded = az.from_netcdf(out_path)
    assert hasattr(reloaded, "posterior")
    assert "mu_anchor_left" in reloaded.posterior.data_vars
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_fitter_class.py -v
```

Expected: ImportError on `FitResult`.

- [ ] **Step 3: Implement skeleton `FitResult`**

Create `chromhandler/fitting/fitter.py`:

```python
"""User-facing fitter entry point.

Exposes ``fit()`` which orchestrates build_priors -> run_mcmc -> FitResult,
and the ``FitResult`` class with debug-plot methods.

Single-mode peaks only at present; doublet hooks documented inline.
TODO(doublet): extend plot methods to overlay right components when ready.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import arviz

from chromhandler.fitting.prepared_dataset import PreparedDataset
from chromhandler.fitting.priors import SkewNormalPriors
from chromhandler.fitting.model import ModelConfig

if TYPE_CHECKING:
    import matplotlib.figure
    import pandas as pd


@dataclass(frozen=False)  # mutable: lazy groups added to idata over time
class FitResult:
    """Bundle of MCMC output, original inputs, and debug-plot methods.

    Attributes:
        idata: ArviZ InferenceData. `posterior` and `observed_data` are
            present from the moment `fit()` returns. `posterior_predictive`
            is added on the first `plot_fit()` call (lazy); `prior` and
            `prior_predictive` on the first `plot_prior_predictive()` call.
        dataset: The PreparedDataset that was fit.
        priors: The SkewNormalPriors that were used.
        model_config: The ModelConfig that was used.
    """

    idata: arviz.InferenceData
    dataset: PreparedDataset
    priors: list[SkewNormalPriors]
    model_config: ModelConfig

    def save(self, path: Path | str) -> None:
        """Write the full InferenceData to netCDF.

        Whatever groups are currently in `idata` get saved — call
        `plot_fit()` / `plot_prior_predictive()` first if you want the
        predictive samples persisted.
        """
        self.idata.to_netcdf(str(path))
```

- [ ] **Step 4: Run tests + quality gates**

```bash
uv run pytest tests/unit/fitting/test_fitter_class.py -v
uv run ruff check chromhandler/fitting/fitter.py tests/unit/fitting/test_fitter_class.py
uv run pyright chromhandler/fitting/fitter.py tests/unit/fitting/test_fitter_class.py
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/fitter.py tests/unit/fitting/test_fitter_class.py
git commit -m "feat(fitter): FitResult class skeleton + save()"
```

---

## Task 7: `FitResult` cheap methods — `summary`, `diagnostics`, `plot_traces`, `plot_prior_overlay`

**Files:**
- Modify: `chromhandler/fitting/fitter.py`
- Modify: `tests/unit/fitting/test_fitter_class.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/unit/fitting/test_fitter_class.py`:

```python
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for tests


def test_summary_returns_dataframe() -> None:
    result = _result_fixture()
    df = result.summary()
    import pandas as pd
    assert isinstance(df, pd.DataFrame)
    assert "mean" in df.columns
    assert "r_hat" in df.columns
    # Sanity: mu_anchor_left should be in the table
    assert any("mu_anchor_left" in str(idx) for idx in df.index)


def test_diagnostics_returns_dict() -> None:
    result = _result_fixture()
    d = result.diagnostics()
    assert isinstance(d, dict)
    assert "fit_healthy" in d
    assert "r_hat_max" in d


def test_plot_traces_returns_figure() -> None:
    result = _result_fixture()
    fig = result.plot_traces()
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_prior_overlay_returns_figure() -> None:
    result = _result_fixture()
    fig = result.plot_prior_overlay()
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_fitter_class.py -v
```

Expected: 4 new failures (`AttributeError` for the new methods).

- [ ] **Step 3: Implement the cheap methods**

Append to `chromhandler/fitting/fitter.py`. First update imports at the top to include:

```python
import numpy as np
from chromhandler.fitting.posterior import diagnostics as _diagnostics_fn
from chromhandler.fitting.skew_normal import density_cp
```

Then add methods to the `FitResult` class:

```python
    def summary(self, var_names: list[str] | None = None) -> "pd.DataFrame":
        """ArviZ summary (mean / sd / hdi / r_hat / ess) as a DataFrame."""
        return arviz.summary(self.idata, var_names=var_names)

    def diagnostics(self) -> dict[str, Any]:
        """Quick convergence summary dict (see posterior.diagnostics)."""
        return _diagnostics_fn(self.idata)

    def plot_traces(self, var_names: list[str] | None = None) -> "matplotlib.figure.Figure":
        """ArviZ trace plot for the listed variables (or all if None)."""
        import matplotlib.pyplot as plt
        axes = arviz.plot_trace(self.idata, var_names=var_names)
        # arviz returns a 2D ndarray of Axes; grab the parent Figure
        if hasattr(axes, "flat"):
            return axes.flat[0].figure
        return axes[0].figure if hasattr(axes, "__iter__") else axes.figure

    def plot_prior_overlay(self) -> "matplotlib.figure.Figure":
        """For each non-control trace, plot data + prior loc curve at the
        per-trace amplitude. Single-mode peaks only.

        TODO(doublet): when doublet ships, add a right-component dashed
        curve in panels for doublet peaks.
        """
        import matplotlib.pyplot as plt
        from chromhandler.fitting.skew_normal import density_cp

        dataset = self.dataset
        priors_list = self.priors
        n_peak = len(priors_list)
        non_control_idx = np.where(~dataset.is_control)[0]

        fig, axes = plt.subplots(
            n_peak, len(non_control_idx),
            figsize=(3.5 * len(non_control_idx), 2.8 * n_peak),
            squeeze=False,
        )

        for peak_idx, p in enumerate(priors_list):
            sigma_loc = float(np.exp(p.log_sigma_left_loc))
            t_dense = np.linspace(p.mu_left_low, p.mu_left_high, 500)
            sn_unit = np.asarray(density_cp(
                t_dense, p.mu_left_loc, sigma_loc, p.gamma1_left_loc,
            ))
            for col, tr in enumerate(non_control_idx):
                ax = axes[peak_idx, col]
                t = dataset.time[tr]
                s = dataset.signal[tr]
                bs = s - (dataset.baseline_intercept[tr] + dataset.baseline_slope[tr] * t)
                mask = ((t >= p.mu_left_low) & (t <= p.mu_left_high) & np.isfinite(bs))
                ax.plot(t[mask], bs[mask], color="C0", lw=1.0, label="data")
                A = float(np.exp(p.log_A_left_loc_per_trace[tr]))
                ax.plot(t_dense, A * sn_unit, "k--", lw=1.2, label="prior loc")
                ax.set_title(f"trace {dataset.trace_ids[tr]} (peak {peak_idx})", fontsize=8)
                ax.axhline(0, color="k", lw=0.3, alpha=0.3)
                if col == 0 and peak_idx == 0:
                    ax.legend(fontsize=7)
        fig.tight_layout()
        return fig
```

- [ ] **Step 4: Run tests + quality gates**

```bash
uv run pytest tests/unit/fitting/test_fitter_class.py -v
uv run ruff check chromhandler/fitting/fitter.py
uv run pyright chromhandler/fitting/fitter.py
```

Expected: 6 tests pass total.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/fitter.py tests/unit/fitting/test_fitter_class.py
git commit -m "feat(fitter): summary, diagnostics, plot_traces, plot_prior_overlay"
```

---

## Task 8: `FitResult` lazy methods — `plot_prior_predictive`, `plot_fit`

**Files:**
- Modify: `chromhandler/fitting/fitter.py`
- Modify: `tests/unit/fitting/test_fitter_class.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/unit/fitting/test_fitter_class.py`:

```python
def test_plot_fit_returns_figure_and_caches() -> None:
    result = _result_fixture()
    assert not hasattr(result.idata, "posterior_predictive")
    fig = result.plot_fit()
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)
    # Lazy cache
    assert hasattr(result.idata, "posterior_predictive")


def test_plot_prior_predictive_returns_figure_and_caches() -> None:
    result = _result_fixture()
    assert not hasattr(result.idata, "prior_predictive")
    fig = result.plot_prior_predictive()
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)
    assert hasattr(result.idata, "prior_predictive")
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_fitter_class.py -v -k "lazy or predictive"
```

Expected: `AttributeError` on the new methods.

- [ ] **Step 3: Implement the lazy plot methods**

In `chromhandler/fitting/fitter.py`, add imports if not already there:

```python
from chromhandler.fitting.posterior import (
    compute_posterior_predictive as _compute_pp,
    compute_prior_predictive as _compute_prior_pp,
)
```

Add to `FitResult`:

```python
    def plot_fit(self) -> "matplotlib.figure.Figure":
        """Posterior predictive 95% HDI band + median + observed data per trace.

        Lazily computes posterior predictive on first call; caches in `idata`.

        TODO(doublet): for doublet peaks, overlay separate dashed lines for
        left and right components.
        """
        import matplotlib.pyplot as plt

        if not hasattr(self.idata, "posterior_predictive"):
            _compute_pp(self.idata, self.dataset, self.priors, self.model_config)

        return self._plot_band(
            samples_group="posterior_predictive",
            label="posterior",
            band_color="tab:blue",
        )

    def plot_prior_predictive(self) -> "matplotlib.figure.Figure":
        """Prior predictive 95% HDI band + median + observed data per trace.

        Lazily computes prior + prior_predictive on first call; caches both.
        """
        if not hasattr(self.idata, "prior_predictive"):
            _compute_prior_pp(self.idata, self.dataset, self.priors, self.model_config)

        return self._plot_band(
            samples_group="prior_predictive",
            label="prior",
            band_color="tab:purple",
        )

    def _plot_band(
        self,
        samples_group: str,
        label: str,
        band_color: str,
    ) -> "matplotlib.figure.Figure":
        """Shared implementation for plot_fit + plot_prior_predictive."""
        import matplotlib.pyplot as plt

        group = getattr(self.idata, samples_group)
        # obs shape: [chain, draw, trace, time_idx]
        obs = np.asarray(group["obs"])
        flat = obs.reshape(-1, obs.shape[-2], obs.shape[-1])  # [draws, trace, time]
        n_trace = self.dataset.n_trace
        ncols = min(4, n_trace)
        nrows = (n_trace + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.6 * ncols, 2.6 * nrows),
            squeeze=False, sharex=False,
        )
        ax_flat = axes.flatten()
        for tr in range(n_trace):
            ax = ax_flat[tr]
            t = self.dataset.time[tr]
            s = self.dataset.signal[tr]
            valid = np.isfinite(s)
            # 95% HDI per time-point
            samples_tr = flat[:, tr, :]            # [draws, time]
            hdi = arviz.hdi(samples_tr[None, :, :], hdi_prob=0.95)["x"]   # [time, 2]
            hdi = np.asarray(hdi)
            median = np.median(samples_tr, axis=0)
            ax.fill_between(
                t[valid], hdi[valid, 0], hdi[valid, 1],
                color=band_color, alpha=0.35, label=f"{label} 95% HDI",
            )
            ax.plot(t[valid], median[valid], color=band_color, lw=1.4, label=f"{label} median")
            ax.plot(t[valid], s[valid], color="k", lw=0.8, label="data")
            ax.set_title(self.dataset.trace_ids[tr], fontsize=8)
            if tr == 0:
                ax.legend(fontsize=7)
        for ax in ax_flat[n_trace:]:
            ax.axis("off")
        fig.tight_layout()
        return fig
```

- [ ] **Step 4: Run tests + quality gates**

```bash
uv run pytest tests/unit/fitting/test_fitter_class.py -v
uv run ruff check chromhandler/fitting/fitter.py tests/unit/fitting/test_fitter_class.py
uv run pyright chromhandler/fitting/fitter.py tests/unit/fitting/test_fitter_class.py
```

Expected: 8 tests pass total.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/fitter.py tests/unit/fitting/test_fitter_class.py
git commit -m "feat(fitter): plot_fit + plot_prior_predictive with lazy HDI band caching"
```

---

## Task 9: `fit()` entry point + `__init__.py` re-exports

**Files:**
- Modify: `chromhandler/fitting/fitter.py`
- Modify: `chromhandler/fitting/__init__.py`
- Create: `tests/unit/fitting/test_fitter_entry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/fitting/test_fitter_entry.py`:

```python
"""Tests for the fit() entry point."""

from __future__ import annotations

import numpy as np
from scipy.stats import skewnorm

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting import FitResult, ModelConfig, PriorConfig, fit
from chromhandler.fitting.prepared_dataset import prepare_dataset


def _toy_dataset():
    rng = np.random.default_rng(0)
    t = np.arange(2.5, 3.6, 0.001)
    times = [t.copy() for _ in range(3)]
    signals = [
        amp * skewnorm.pdf(t, 0.0, loc=3.0, scale=0.025) + 5.0
        + rng.normal(0.0, 0.5, size=t.shape)
        for amp in (100.0, 60.0, 30.0)
    ]
    peaks = [PeakAnnotation(molecule_id="A", rt_min=2.85, rt_max=3.15, mode="single")]
    bases = [BaselineAnnotation(rt_min=2.50, rt_max=2.52),
             BaselineAnnotation(rt_min=3.55, rt_max=3.58)]
    return prepare_dataset(times, signals, peaks, bases)


def test_fit_returns_fitresult() -> None:
    ds = _toy_dataset()
    result = fit(ds, model_config=ModelConfig(num_warmup=30, num_samples=30, num_chains=2))
    assert isinstance(result, FitResult)
    assert result.dataset is ds


def test_fit_with_default_configs() -> None:
    ds = _toy_dataset()
    # Use small config for speed
    result = fit(ds, model_config=ModelConfig(num_warmup=20, num_samples=20, num_chains=2))
    assert "mu_anchor_left" in result.idata.posterior.data_vars
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_fitter_entry.py -v
```

Expected: ImportError on `fit` from `chromhandler.fitting`.

- [ ] **Step 3: Implement `fit()` and update `__init__.py`**

Append to `chromhandler/fitting/fitter.py`:

```python
def fit(
    dataset: PreparedDataset,
    *,
    prior_config: "PriorConfig | None" = None,
    model_config: ModelConfig | None = None,
) -> FitResult:
    """Build priors, run MCMC, return a FitResult.

    Args:
        dataset: PreparedDataset to fit.
        prior_config: Optional PriorConfig override. Defaults to PriorConfig().
        model_config: Optional ModelConfig override. Defaults to ModelConfig().

    Returns:
        FitResult with .plot_* methods and .idata for raw access.
    """
    from chromhandler.fitting.model import run_mcmc
    from chromhandler.fitting.priors import PriorConfig, build_priors

    pc = prior_config if prior_config is not None else PriorConfig()
    mc = model_config if model_config is not None else ModelConfig()
    priors = build_priors(dataset, config=pc)
    idata = run_mcmc(dataset, priors, mc)
    return FitResult(idata=idata, dataset=dataset, priors=priors, model_config=mc)
```

Replace `chromhandler/fitting/__init__.py` content with:

```python
"""Bayesian skew-normal peak fitting layer."""

from __future__ import annotations

from chromhandler.fitting.fitter import FitResult, fit
from chromhandler.fitting.model import ModelConfig
from chromhandler.fitting.priors import PriorConfig

__all__ = ["FitResult", "ModelConfig", "PriorConfig", "fit"]
```

- [ ] **Step 4: Run tests + quality gates**

```bash
uv run pytest tests/unit/fitting/test_fitter_entry.py -v
uv run ruff check chromhandler/fitting/fitter.py chromhandler/fitting/__init__.py
uv run pyright chromhandler/fitting/fitter.py chromhandler/fitting/__init__.py
uv run pytest tests/unit/fitting/ -q   # smoke
```

Expected: 2 new tests pass, all unit fitting tests pass.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/fitter.py chromhandler/fitting/__init__.py tests/unit/fitting/test_fitter_entry.py
git commit -m "feat(fitter): fit() entry point + package re-exports"
```

---

## Task 10: Synthetic recovery test

**Files:**
- Create: `tests/unit/fitting/test_model_recovery.py`

- [ ] **Step 1: Write the recovery test**

Create `tests/unit/fitting/test_model_recovery.py`:

```python
"""Synthetic-data recovery: known parameters -> small MCMC -> posterior near truth."""

from __future__ import annotations

import numpy as np
from scipy.stats import skewnorm

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting import ModelConfig, PriorConfig, fit
from chromhandler.fitting.prepared_dataset import prepare_dataset
from chromhandler.fitting.skew_normal import cp_to_dp


TRUE_MU = 3.00
TRUE_SIGMA = 0.025
TRUE_GAMMA1 = 0.2
TRUE_AREAS = [100.0, 60.0, 30.0, 10.0]
NOISE_STD = 0.5


def _synthetic_dataset():
    rng = np.random.default_rng(42)
    t = np.arange(2.5, 3.6, 0.001)
    xi, omega, alpha = (float(x) for x in cp_to_dp(TRUE_MU, TRUE_SIGMA, TRUE_GAMMA1))
    times, signals = [], []
    for A in TRUE_AREAS:
        s = A * skewnorm.pdf(t, alpha, loc=xi, scale=omega)
        s = s + 5.0 + rng.normal(0.0, NOISE_STD, size=t.shape)
        times.append(t.copy()); signals.append(s)
    peaks = [PeakAnnotation(molecule_id="A", rt_min=2.85, rt_max=3.15, mode="single")]
    bases = [BaselineAnnotation(rt_min=2.50, rt_max=2.52),
             BaselineAnnotation(rt_min=3.55, rt_max=3.58)]
    return prepare_dataset(times, signals, peaks, bases)


def test_posterior_recovers_known_parameters() -> None:
    ds = _synthetic_dataset()
    config = ModelConfig(num_warmup=300, num_samples=300, num_chains=2, seed=0)
    result = fit(ds, prior_config=PriorConfig(), model_config=config)

    diag = result.diagnostics()
    # Allow a slightly loose r_hat threshold given small MCMC
    assert diag["r_hat_max"] < 1.10, f"r_hat too high: {diag}"
    assert diag["n_divergent"] == 0, f"divergences: {diag}"

    posterior = result.idata.posterior
    mu_median = float(np.median(np.asarray(posterior["mu_anchor_left"])))
    sigma_median = float(np.median(np.exp(np.asarray(posterior["log_sigma_left"]))))
    gamma1_median = float(np.median(np.asarray(posterior["gamma1_left"])))
    log_A = np.asarray(posterior["log_A_left"])
    A_median_per_trace = np.median(np.exp(log_A), axis=(0, 1))[:, 0]  # [n_trace]

    assert abs(mu_median - TRUE_MU) < 2 * ds.dt_global, (
        f"mu off: {mu_median} vs {TRUE_MU} (tol = 2 dt = {2 * ds.dt_global})"
    )
    assert abs(sigma_median - TRUE_SIGMA) / TRUE_SIGMA < 0.15, (
        f"sigma off: {sigma_median} vs {TRUE_SIGMA}"
    )
    assert abs(gamma1_median - TRUE_GAMMA1) < 0.15, (
        f"gamma1 off: {gamma1_median} vs {TRUE_GAMMA1}"
    )
    for tr, true_A in enumerate(TRUE_AREAS):
        recovered = float(A_median_per_trace[tr])
        assert abs(recovered - true_A) / true_A < 0.10, (
            f"A[{tr}] off: {recovered} vs {true_A}"
        )
```

- [ ] **Step 2: Run the recovery test**

```bash
uv run pytest tests/unit/fitting/test_model_recovery.py -v
```

Expected: passes within 30-90 s. If it fails on tolerances, **do not loosen blindly** — investigate first (check `summary()` output for the seed, confirm the priors look reasonable, etc.). If after investigation the failure is due to small-MCMC noise, document the looser tolerance and commit.

- [ ] **Step 3: Quality gates + commit**

```bash
uv run ruff check tests/unit/fitting/test_model_recovery.py
uv run pyright tests/unit/fitting/test_model_recovery.py
git add tests/unit/fitting/test_model_recovery.py
git commit -m "test(model): synthetic recovery — known params recovered within tolerance"
```

---

## Task 11: Real-data smoke test on ASM fixture

**Files:**
- Create: `tests/integration/test_fitter_asm.py`

- [ ] **Step 1: Write the smoke test**

Create `tests/integration/test_fitter_asm.py`:

```python
"""End-to-end smoke test on the real ASM kinetic-series fixture."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting import ModelConfig, fit
from chromhandler.handler import Handler

ASM_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "asm_kinetic_series"
CONDITIONS_CSV = ASM_DIR / "conditions.csv"


def test_fit_on_sih_kinetic_subset() -> None:
    """Fit the SIH analyte on the CV10 kinetic series + CV1 (no enzyme) +
    CV4 + CV5 controls. Single-mode, contiguous window 2.80-3.15 min."""
    handler = Handler.read_asm(path=ASM_DIR, mode="timecourse")
    for mol_id in ("SIH", "Hyp", "Ino"):
        handler.create_molecule(id=mol_id, pubchem_cid=1)
    handler.load_initial_conditions(CONDITIONS_CSV, conc_unit="umol / l")

    # Subset to samples relevant for SIH + controls
    h_sub = Handler()
    h_sub.molecules = deepcopy(handler.molecules)
    h_sub.samples = [deepcopy(s) for s in handler.samples
                     if s.id in {"CV10", "CV1", "CV4", "CV5"}]

    peak_anns = [PeakAnnotation(molecule_id="SIH", rt_min=2.80, rt_max=3.15, mode="single")]
    base_anns = [BaselineAnnotation(rt_min=2.50, rt_max=2.52),
                 BaselineAnnotation(rt_min=3.55, rt_max=3.58)]
    dataset = h_sub.prepare_dataset(peak_anns, base_anns)

    result = fit(dataset, model_config=ModelConfig(
        num_warmup=200, num_samples=200, num_chains=2, seed=0,
    ))

    diag = result.diagnostics()
    # Real data is messier; allow slightly looser thresholds than synthetic
    assert diag["r_hat_max"] < 1.15, f"r_hat too high: {diag}"
    assert diag["n_divergent"] < 10, f"too many divergences: {diag}"

    # mu posterior should land near 3.008 (the priors_demo recovered value)
    import numpy as np
    posterior_mu = float(np.median(np.asarray(result.idata.posterior["mu_anchor_left"])))
    assert abs(posterior_mu - 3.008) < 0.02, (
        f"mu posterior {posterior_mu} too far from priors_demo value 3.008"
    )

    # Plot smoke tests
    fig_traces = result.plot_traces()
    assert fig_traces is not None
    fig_fit = result.plot_fit()
    assert fig_fit is not None
```

- [ ] **Step 2: Run the smoke test**

```bash
uv run pytest tests/integration/test_fitter_asm.py -v
```

Expected: passes within 1-3 minutes. Investigate, don't loosen, on failure.

- [ ] **Step 3: Quality gates + commit**

```bash
uv run ruff check tests/integration/test_fitter_asm.py
uv run pyright tests/integration/test_fitter_asm.py
git add tests/integration/test_fitter_asm.py
git commit -m "test(integration): end-to-end fit() on ASM kinetic series fixture"
```

---

## Self-Review

**Spec coverage check (against `docs/superpowers/specs/2026-05-12-fitter-integration-design.md`):**

- §2 Module layout: model.py (Tasks 1-4), posterior.py (Task 5), fitter.py (Tasks 6-9) ✅
- §3 User-facing API — `fit()` + `FitResult`: Tasks 6-9 ✅
- §4 Data flow — build_priors → run_mcmc → FitResult: Task 9 ✅
- §5 Model architecture — sample sites, distribution wiring, sample-name constants: Tasks 2, 3, 4 ✅
- §6 ModelConfig: Task 1 ✅
- §7 posterior.py: Task 5 ✅
- §8 Plotting (4 plot types): Tasks 7, 8 ✅
- §9 Controls handling — priors.py fix + uniform model: Task 0 ✅
- §10 Doublet extension hooks (6 hooks): inline TODO(doublet) markers in Tasks 2-3, _left suffix in Task 3, _validate_single_mode_only at run_mcmc level (Task 4), plot forward-compat (Tasks 7-8) ✅
- §11 Testing strategy — unit + synthetic recovery + real-data smoke: Tasks 1-9 (unit), 10 (synthetic), 11 (real-data) ✅
- §12 Out of scope honoured (no doublet, no predict, no Fitter class wrapper, etc.) ✅

**Placeholder scan:** No "TBD", "implement later", or vague language. Every step shows complete code. `TODO(doublet)` markers are intentional (greppable hooks for future work, not unfinished work).

**Type consistency:**
- `FitResult` field names — `idata`, `dataset`, `priors`, `model_config` — used identically in Tasks 6, 7, 8, 9.
- `ModelConfig` field names match between dataclass def (Task 1) and consumers (Tasks 2-4).
- Sample-site names (`mu_anchor_left`, etc.) match between `model()` (Task 3), constants (Task 2), and tests (Tasks 3, 4, 5, 10, 11).
- `SkewNormalPriors.log_A_left_loc_per_trace` shape contract: `[n_trace]` after Task 0; consumed by Task 3.

**Cross-task dependencies:**
- Task 0 unblocks everything (priors.py shape contract).
- Tasks 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 strictly sequential.
- Tasks 10, 11 depend on Task 9.

**Estimated execution time** (subagent-driven, including reviews):
- Tasks 0-2: 30 min total (small)
- Tasks 3-4: 45 min (MCMC verification adds latency)
- Task 5: 30 min
- Tasks 6-8: 60 min total (plotting smoke tests + lazy logic)
- Task 9: 15 min (mostly wiring)
- Task 10: 30 min (synthetic recovery — MCMC run + assertions)
- Task 11: 30 min (real-data smoke)
- **Total: ~4 hours**, dominated by MCMC test runs.
