# Priors Module Implementation Plan (v3 — controls + central config)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `chromhandler/fitting/priors.py` — a controls-based prior-construction layer that turns a `PreparedDataset` plus its `PeakAnnotation`s into a `list[SkewNormalPriors]`. All magic numbers and fallback heuristics live in a single user-overridable `PriorConfig` dataclass.

**Architecture:** Pure-function module on top of `skew_normal.py` and `PreparedDataset` (both shipped). Two schema types: `PriorConfig` (knobs) and `SkewNormalPriors` (output contract). Per-trace FWHM extraction for analyte peaks; direct extraction of raw artefact measurements from control traces; assembly into typed priors with config-driven scale fallbacks. Doublet artefact priors borrow scale magnitudes from the analyte's empirical population (same chromatographic system → same drift/shape variation) instead of relying on incoherent large-sample formulas.

**Tech Stack:** Python 3.11+, NumPy, SciPy (`signal.savgol_filter`), JAX/NumPyro (downstream only), pytest, ruff, pyright. All execution via `uv run`.

**Spec:** [`docs/superpowers/specs/2026-05-07-skew-normal-fitter-rewrite-design.md`](../specs/2026-05-07-skew-normal-fitter-rewrite-design.md), §3–§7.2. Deviations from §6.1: controls-based artefact extraction replaces per-trace deconvolution; `PriorConfig` centralizes all heuristics.

**Prerequisite:** [`docs/superpowers/plans/2026-05-11-handler-controls.md`](2026-05-11-handler-controls.md) (complete — commits 015c7c1, 1e8ec70, e58af57, a1d049d). `PreparedDataset.is_control: NDArray[np.bool_]` is the canonical control marker.

---

## Distribution Table

| Sampled site (model.py) | Distribution | Parameters | Where computed |
|---|---|---|---|
| `mu_anchor_left[peak]` | `TruncatedNormal` | `loc, scale, low=window_low, high=window_high` | Task 5: mean/std of non-control apex locations |
| `log_sigma_left[peak]` | `TruncatedNormal` | `loc, scale, low=log(n_pts_per_fwhm·dt·FWHM_TO_SIGMA), high=log(window_width/N)` | Task 5: mean/std of `log(σ)` across non-control traces; bounds from config |
| `gamma1_left[peak]` | `TruncatedNormal` | `loc, scale, low=-c·GAMMA1_MAX, high=+c·GAMMA1_MAX` | Task 5; c from config |
| `log_A_left[trace, peak]` | `Normal` | `loc_per_trace, scale` | Singles (Task 5): per-trace `log(trapezoid)`. Doublets (Task 6): `log(max(A_total − A_artefact, A_floor))` |
| `Delta[peak]` (doublet) | `TruncatedNormal` | `loc=`abs(`μ_artefact − μ_analyte_ref)`, `scale`, `low=N_dt·dt`, `high=window_width/N` | Task 6: empirical from controls; scale uses analyte's `mu_left_scale` |
| `log_sigma_right[peak]` (doublet) | `TruncatedNormal` | same shape as `log_sigma_left` | Task 6: `loc` from controls, `scale` borrowed from analyte's `log_sigma_left_scale` |
| `gamma1_right[peak]` (doublet) | `TruncatedNormal` | same shape as `gamma1_left` | Task 6: `loc` from controls, `scale` borrowed from analyte's `gamma1_left_scale` |
| `log_A_right[trace, peak]` (doublet) | `Normal` | `loc_per_trace` (constant), `scale` | Task 6: `loc = log(A_artefact_from_controls)`, `scale` from baseline-OLS + noise propagation, floored by config |

**Hard requirements:**
1. `artefact_doublet` annotations require ≥1 control trace in the dataset. Otherwise `build_priors` raises with a clear message.
2. `free_doublet` annotations are not implemented (raises `NotImplementedError`).
3. A side check verifies that the control's dominant apex sits on `ann.artefact_side`; mismatch raises a clear error.

---

## `PriorConfig` — the central knob box

Every heuristic, threshold, and fallback in this module lives in `PriorConfig`. Users can override any field; defaults are tuned for typical chromatographic data.

```python
@dataclass(frozen=True)
class PriorConfig:
    """Centralised configuration for prior construction."""

    # --- Distribution bounds (geometric / mathematical) ---

    gamma1_bound_fraction: float = 0.99
    """Fraction of GAMMA1_MAX for the γ₁ TruncatedNormal bounds.
    Bounds: ``±gamma1_bound_fraction · GAMMA1_MAX``."""

    sigma_low_n_points_per_fwhm: int = 8
    """Nyquist-derived ``log_sigma`` lower bound:
    ``log_sigma_low = log(n · dt · FWHM_TO_SIGMA)``. Default 8 samples per FWHM
    is the chromatography rule of thumb for usable shape information."""

    sigma_high_window_fraction: float = 6.0
    """``log_sigma`` upper bound: ``log(window_width / N)``. Default 6 means
    ``±3σ`` of the peak must fit inside the annotation window."""

    delta_low_dt_multiplier: float = 3.0
    """Minimum Δ in units of dt. Below this, components are unresolvable
    at the sampling resolution."""

    delta_high_window_fraction: float = 2.0
    """Maximum Δ = ``window_width / N``. Default 2: artefact should sit
    within the same window the user annotated."""

    # --- Side check ---

    side_check_epsilon_dt_multiplier: float = 3.0
    """The minimum ``|μ_artefact − μ_analyte_ref|`` (in dt units) for the
    side check to succeed. Below this, the two peaks are indistinguishable
    at sampling resolution and we raise rather than commit to a side."""

    # --- n=1 fallbacks (single control) ---

    delta_scale_dt_multiplier_n1: float = 1.5
    """``Delta_scale`` when ``n_controls == 1``: ``1.5·dt`` reflects apex
    sampling resolution plus a small allowance for non-perfect drift
    cancellation between artefact and analyte."""

    log_A_artefact_min_scale: float = 0.2
    """Floor on ``log_A_right_scale`` regardless of computed value
    (``log(1.2) ≈ ±20%``). Prevents underconfidence on small artefacts."""

    # --- Single-trace fallbacks for single-peak aggregation ---

    log_sigma_scale_n1: float = 0.15
    """``log_sigma`` scale when only one trace is available
    (n_trace = 1 single-peak case). 15% in σ ≈ typical chromatographic
    HWHM-extraction precision at moderate S/N."""

    gamma1_scale_n1: float = 0.20
    """``γ₁`` scale when only one trace is available. Typical
    chromatographic ``γ₁`` extraction precision at moderate S/N."""

    log_A_scale_n1_min: float = 0.10
    """``log_A_left_scale`` floor when n_trace=1. log(1.1) ≈ ±10%."""

    # --- Universal floors ---

    mu_scale_dt_floor_multiplier: float = 1.0
    """``mu_left_scale`` floor in units of dt — always at least one
    sample's worth of uncertainty."""
```

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `chromhandler/fitting/_legacy_priors.py` | New (renamed) | Quarantined legacy code |
| `chromhandler/fitting/priors.py` | New (overwrites legacy) | Deliverable |
| `tests/unit/fitting/test_priors_legacy.py` | New (renamed) | Legacy tests redirected |
| `tests/unit/fitting/test_priors_schema.py` | New | `PriorConfig` and `SkewNormalPriors` dataclasses |
| `tests/unit/fitting/test_priors_features.py` | New | Single-window FWHM extraction |
| `tests/unit/fitting/test_priors_apex.py` | New | Dominant apex detection |
| `tests/unit/fitting/test_priors_controls.py` | New | Artefact measurements from controls |
| `tests/unit/fitting/test_priors_aggregate.py` | New | Single + doublet aggregation |
| `tests/unit/fitting/test_priors_orchestrator.py` | New | `build_priors` end-to-end |
| `tests/unit/fitting/test_priors_summary.py` | New | `summarise_priors` |

---

## Conventions

- Quality gate after every file edit: `uv run ruff check <file>` and `uv run pyright <file>` must report zero issues.
- Tests via `uv run pytest <file> -v`.
- Google-style docstrings on every public function.
- `from __future__ import annotations` at top of every new module.
- One commit per task. `feat(priors): <summary>` for code, `chore(priors): <summary>` for renames.

---

## Task 0: Quarantine the legacy priors module

**Why first:** The existing `chromhandler/fitting/priors.py` has a different API and is imported by legacy `fitter.py`, `visualize.py`, and `tests/unit/fitting/test_priors.py`. We need the canonical name free.

**Files:**
- Rename: `chromhandler/fitting/priors.py` → `chromhandler/fitting/_legacy_priors.py`
- Modify: `chromhandler/fitting/fitter.py` (line 44 import)
- Modify: `chromhandler/fitting/visualize.py` (line 26 import)
- Rename: `tests/unit/fitting/test_priors.py` → `tests/unit/fitting/test_priors_legacy.py`
- Modify: `from chromhandler.fitting.priors import …` → `from chromhandler.fitting._legacy_priors import …` in `test_priors_legacy.py`

- [ ] **Step 1: Rename source**

```bash
git mv chromhandler/fitting/priors.py chromhandler/fitting/_legacy_priors.py
```

- [ ] **Step 2: Patch importers**

In `chromhandler/fitting/fitter.py` line 44 and `chromhandler/fitting/visualize.py` line 26, change `.priors` → `._legacy_priors`.

- [ ] **Step 3: Rename legacy test + patch imports**

```bash
git mv tests/unit/fitting/test_priors.py tests/unit/fitting/test_priors_legacy.py
uv run python -c "
import pathlib
p = pathlib.Path('tests/unit/fitting/test_priors_legacy.py')
p.write_text(p.read_text().replace('chromhandler.fitting.priors', 'chromhandler.fitting._legacy_priors'))
"
```

- [ ] **Step 4: Verify**

```bash
uv run ruff check chromhandler/fitting/_legacy_priors.py chromhandler/fitting/fitter.py chromhandler/fitting/visualize.py tests/unit/fitting/test_priors_legacy.py
uv run pyright chromhandler/fitting/_legacy_priors.py chromhandler/fitting/fitter.py chromhandler/fitting/visualize.py
uv run pytest tests/unit/fitting/test_priors_legacy.py -v
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore(priors): quarantine legacy priors as _legacy_priors to free the name"
```

---

## Task 1: `PriorConfig` + `SkewNormalPriors` dataclasses

**Why:** Defines the input (PriorConfig) and output (SkewNormalPriors) contracts. Every subsequent task consumes one or both.

**Files:**
- Create: `chromhandler/fitting/priors.py`
- Test: `tests/unit/fitting/test_priors_schema.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/fitting/test_priors_schema.py`:

```python
"""Tests for PriorConfig and SkewNormalPriors dataclasses."""

from __future__ import annotations

import numpy as np
import pytest

from chromhandler.fitting.priors import PriorConfig, SkewNormalPriors


def test_prior_config_defaults() -> None:
    c = PriorConfig()
    assert c.gamma1_bound_fraction == 0.99
    assert c.sigma_low_n_points_per_fwhm == 8
    assert c.sigma_high_window_fraction == 6.0
    assert c.delta_low_dt_multiplier == 3.0
    assert c.delta_high_window_fraction == 2.0


def test_prior_config_overridable() -> None:
    c = PriorConfig(gamma1_bound_fraction=0.95, log_sigma_scale_n1=0.10)
    assert c.gamma1_bound_fraction == 0.95
    assert c.log_sigma_scale_n1 == 0.10
    # Other fields keep defaults
    assert c.delta_low_dt_multiplier == 3.0


def _single() -> SkewNormalPriors:
    return SkewNormalPriors(
        n_components=1,
        mu_left_loc=2.7, mu_left_scale=0.005, mu_left_low=2.55, mu_left_high=2.85,
        log_sigma_left_loc=np.log(0.03), log_sigma_left_scale=0.1,
        log_sigma_left_low=np.log(0.005), log_sigma_left_high=np.log(0.05),
        gamma1_left_loc=0.2, gamma1_left_scale=0.05,
        log_A_left_loc_per_trace=np.array([np.log(100.0), np.log(80.0)]),
        log_A_left_scale=0.1,
        Delta_loc=None, Delta_scale=None, Delta_low=None, Delta_high=None,
        log_sigma_right_loc=None, log_sigma_right_scale=None,
        log_sigma_right_low=None, log_sigma_right_high=None,
        gamma1_right_loc=None, gamma1_right_scale=None,
        log_A_right_loc_per_trace=None, log_A_right_scale=None,
    )


def test_single_priors_constructs() -> None:
    p = _single()
    assert p.n_components == 1 and p.Delta_loc is None


def test_single_priors_rejects_doublet_fields() -> None:
    with pytest.raises(ValueError, match="right.*None"):
        SkewNormalPriors(
            n_components=1,
            mu_left_loc=2.7, mu_left_scale=0.005, mu_left_low=2.55, mu_left_high=2.85,
            log_sigma_left_loc=np.log(0.03), log_sigma_left_scale=0.1,
            log_sigma_left_low=np.log(0.005), log_sigma_left_high=np.log(0.05),
            gamma1_left_loc=0.2, gamma1_left_scale=0.05,
            log_A_left_loc_per_trace=np.array([np.log(100.0)]), log_A_left_scale=0.1,
            Delta_loc=0.05, Delta_scale=0.005, Delta_low=0.003, Delta_high=0.15,  # invalid
            log_sigma_right_loc=None, log_sigma_right_scale=None,
            log_sigma_right_low=None, log_sigma_right_high=None,
            gamma1_right_loc=None, gamma1_right_scale=None,
            log_A_right_loc_per_trace=None, log_A_right_scale=None,
        )


def test_doublet_priors_requires_all_right_fields() -> None:
    with pytest.raises(ValueError, match="doublet.*required"):
        SkewNormalPriors(
            n_components=2,
            mu_left_loc=2.7, mu_left_scale=0.005, mu_left_low=2.55, mu_left_high=2.85,
            log_sigma_left_loc=np.log(0.03), log_sigma_left_scale=0.1,
            log_sigma_left_low=np.log(0.005), log_sigma_left_high=np.log(0.05),
            gamma1_left_loc=0.2, gamma1_left_scale=0.05,
            log_A_left_loc_per_trace=np.array([np.log(100.0)]), log_A_left_scale=0.1,
            Delta_loc=None, Delta_scale=None, Delta_low=None, Delta_high=None,  # missing
            log_sigma_right_loc=None, log_sigma_right_scale=None,
            log_sigma_right_low=None, log_sigma_right_high=None,
            gamma1_right_loc=None, gamma1_right_scale=None,
            log_A_right_loc_per_trace=None, log_A_right_scale=None,
        )
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_priors_schema.py -v
```
Expected: `ImportError`.

- [ ] **Step 3: Implement the schema**

Create `chromhandler/fitting/priors.py`:

```python
"""Controls-based prior construction for the skew-normal peak model.

This module turns a ``PreparedDataset`` plus its ``PeakAnnotation`` list
into a list of :class:`SkewNormalPriors`, one per peak.

All magic numbers and fallback heuristics live in :class:`PriorConfig`.
Users can override the config to change behaviour; defaults are tuned for
typical chromatographic data.

For ``artefact_doublet`` peaks, all artefact-related priors are derived
**directly from control traces** (samples with no analyte). For shape
quantities where only one control is available, scale fallbacks borrow
from the analyte's empirical population (same chromatographic system →
same drift and shape variation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PriorConfig:
    """Centralised configuration for prior construction.

    All knobs in one place — users can override any field to change
    behaviour without touching the priors module itself.
    """

    # --- Distribution bounds (geometric / mathematical) ---
    gamma1_bound_fraction: float = 0.99
    sigma_low_n_points_per_fwhm: int = 8
    sigma_high_window_fraction: float = 6.0
    delta_low_dt_multiplier: float = 3.0
    delta_high_window_fraction: float = 2.0

    # --- Side check ---
    side_check_epsilon_dt_multiplier: float = 3.0

    # --- n=1 control fallbacks ---
    delta_scale_dt_multiplier_n1: float = 1.5
    log_A_artefact_min_scale: float = 0.2

    # --- Single-trace fallbacks (n_trace=1 in single-peak aggregation) ---
    log_sigma_scale_n1: float = 0.15
    gamma1_scale_n1: float = 0.20
    log_A_scale_n1_min: float = 0.10

    # --- Universal floors ---
    mu_scale_dt_floor_multiplier: float = 1.0


@dataclass(frozen=True)
class SkewNormalPriors:
    """Empirical priors for one peak window.

    Each field parameterizes exactly one NumPyro distribution in
    ``model.py`` per the distribution table at the top of the priors plan.
    ``_left_*`` fields are always populated; ``_right_*`` and ``Delta_*``
    are populated iff ``n_components == 2``.
    """

    n_components: int

    mu_left_loc: float
    mu_left_scale: float
    mu_left_low: float
    mu_left_high: float

    log_sigma_left_loc: float
    log_sigma_left_scale: float
    log_sigma_left_low: float
    log_sigma_left_high: float

    gamma1_left_loc: float
    gamma1_left_scale: float

    log_A_left_loc_per_trace: NDArray[np.float64]
    log_A_left_scale: float

    Delta_loc: float | None
    Delta_scale: float | None
    Delta_low: float | None
    Delta_high: float | None

    log_sigma_right_loc: float | None
    log_sigma_right_scale: float | None
    log_sigma_right_low: float | None
    log_sigma_right_high: float | None

    gamma1_right_loc: float | None
    gamma1_right_scale: float | None

    log_A_right_loc_per_trace: NDArray[np.float64] | None
    log_A_right_scale: float | None

    def __post_init__(self) -> None:
        right_fields = (
            self.Delta_loc, self.Delta_scale, self.Delta_low, self.Delta_high,
            self.log_sigma_right_loc, self.log_sigma_right_scale,
            self.log_sigma_right_low, self.log_sigma_right_high,
            self.gamma1_right_loc, self.gamma1_right_scale,
            self.log_A_right_loc_per_trace, self.log_A_right_scale,
        )
        if self.n_components == 1:
            if any(f is not None for f in right_fields):
                raise ValueError(
                    "Single-component priors require all right-component "
                    "fields (Delta_*, *_right_*) to be None."
                )
        elif self.n_components == 2:
            if any(f is None for f in right_fields):
                raise ValueError(
                    "Doublet priors require all right-component fields "
                    "(Delta_*, *_right_*) to be set; got at least one None."
                )
        else:
            raise ValueError(f"n_components must be 1 or 2, got {self.n_components}.")
```

- [ ] **Step 4: Run tests + quality gates**

```bash
uv run pytest tests/unit/fitting/test_priors_schema.py -v
uv run ruff check chromhandler/fitting/priors.py tests/unit/fitting/test_priors_schema.py
uv run pyright chromhandler/fitting/priors.py tests/unit/fitting/test_priors_schema.py
```
Expected: 5 pass, clean.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/priors.py tests/unit/fitting/test_priors_schema.py
git commit -m "feat(priors): PriorConfig and SkewNormalPriors dataclasses"
```

---

## Task 2: `compute_single_window_features` — per-trace FWHM extraction

**What:** For one trace and one window, returns `WindowFeatures(mu, sigma, gamma1, area)`. Used by Task 5 (single peaks) and Task 4 (artefact extraction from controls). No config dependency.

**Files:**
- Modify: `chromhandler/fitting/priors.py`
- Test: `tests/unit/fitting/test_priors_features.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/fitting/test_priors_features.py`:

```python
"""Tests for single-window FWHM-based feature extraction."""

from __future__ import annotations

import numpy as np
from scipy.stats import skewnorm

from chromhandler.fitting.priors import WindowFeatures, compute_single_window_features
from chromhandler.fitting.skew_normal import cp_to_dp


def _synth(mu, sigma, gamma1, area, dt=0.001):
    t = np.arange(mu - 1.0, mu + 1.0, dt)
    xi, omega, alpha = (float(x) for x in cp_to_dp(mu, sigma, gamma1))
    return t, area * skewnorm.pdf(t, alpha, loc=xi, scale=omega)


def test_features_dataclass_fields() -> None:
    f = WindowFeatures(mu=2.7, sigma=0.03, gamma1=0.2, area=5.0)
    assert (f.mu, f.sigma, f.gamma1, f.area) == (2.7, 0.03, 0.2, 5.0)


def test_recovers_symmetric_peak() -> None:
    t, s = _synth(2.7, 0.03, 0.0, 5.0)
    f = compute_single_window_features(t, s, 2.55, 2.85)
    assert abs(f.mu - 2.7) < 0.005
    assert abs(f.sigma - 0.03) / 0.03 < 0.05
    assert abs(f.gamma1) < 0.05
    assert abs(f.area - 5.0) / 5.0 < 0.02


def test_recovers_positively_skewed_peak() -> None:
    t, s = _synth(2.7, 0.03, 0.5, 5.0)
    f = compute_single_window_features(t, s, 2.55, 2.85)
    assert abs(f.gamma1 - 0.5) < 0.10


def test_low_snr_average_is_unbiased() -> None:
    rng = np.random.default_rng(0)
    estimates = []
    for _ in range(100):
        t, s = _synth(2.7, 0.03, 0.3, 5.0)
        noise = rng.normal(0.0, np.max(s) / 5.0, size=s.shape)
        f = compute_single_window_features(t, s + noise, 2.55, 2.85)
        estimates.append(f.gamma1)
    assert abs(float(np.mean(estimates)) - 0.3) < 0.05
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_priors_features.py -v
```

- [ ] **Step 3: Implement**

Append to `chromhandler/fitting/priors.py`:

```python
from scipy.signal import savgol_filter

from chromhandler.fitting.skew_normal import sn_asymmetry_to_gamma1

_FWHM_TO_SIGMA: float = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


@dataclass(frozen=True)
class WindowFeatures:
    """Per-trace, per-window FWHM-based features.

    Attributes:
        mu: Apex location (minutes), smoothed argmax inside the window.
        sigma: ``(HWHM_L + HWHM_R) * FWHM_TO_SIGMA``.
        gamma1: ``sn_asymmetry_to_gamma1(HWHM_R / HWHM_L)``.
        area: ``trapezoid(signal, time)`` over the window.
    """

    mu: float
    sigma: float
    gamma1: float
    area: float


def _interp_threshold_crossing(t, s, apex_idx, threshold, direction):
    i = apex_idx
    n = s.size
    while 0 <= i + direction < n and s[i + direction] >= threshold:
        i += direction
    j = i + direction
    if not (0 <= j < n):
        return None
    if s[i] == s[j]:
        return float(t[i])
    f = (s[i] - threshold) / (s[i] - s[j])
    return float(t[i] + f * (t[j] - t[i]))


def compute_single_window_features(
    time: NDArray[np.float64],
    signal_baseline_subtracted: NDArray[np.float64],
    window_low: float,
    window_high: float,
    smoothing_window: int = 5,
) -> WindowFeatures:
    """Extract FWHM-based features from a single-peak window.

    Args:
        time: 1-D time array.
        signal_baseline_subtracted: 1-D baseline-subtracted signal.
        window_low: Inclusive lower bound.
        window_high: Inclusive upper bound.
        smoothing_window: Savitzky-Golay length, odd >= 5.

    Returns:
        :class:`WindowFeatures`.

    Raises:
        ValueError: If too few valid points in window, or half-max never
            resolved on either side.
    """
    mask = (time >= window_low) & (time <= window_high) & np.isfinite(
        signal_baseline_subtracted
    )
    t = np.asarray(time[mask], dtype=np.float64)
    s = np.asarray(signal_baseline_subtracted[mask], dtype=np.float64)
    if s.size < smoothing_window:
        raise ValueError(
            f"Window [{window_low}, {window_high}] has only {s.size} valid "
            f"points; need at least {smoothing_window}."
        )
    polyorder = min(3, smoothing_window - 1)
    s_smooth = savgol_filter(s, smoothing_window, polyorder)
    apex_idx = int(np.argmax(s_smooth))
    apex_height = float(s_smooth[apex_idx])
    mu = float(t[apex_idx])

    half = apex_height / 2.0
    t_left = _interp_threshold_crossing(t, s, apex_idx, half, -1)
    t_right = _interp_threshold_crossing(t, s, apex_idx, half, +1)
    if t_left is None and t_right is None:
        raise ValueError(
            f"Could not bracket half-max in window [{window_low}, {window_high}]."
        )
    if t_left is None:
        assert t_right is not None
        hwhm_r = t_right - mu
        hwhm_l = hwhm_r
    elif t_right is None:
        hwhm_l = mu - t_left
        hwhm_r = hwhm_l
    else:
        hwhm_l = mu - t_left
        hwhm_r = t_right - mu

    sigma = (hwhm_l + hwhm_r) * _FWHM_TO_SIGMA
    ratio = hwhm_r / hwhm_l if hwhm_l > 0 else 1.0
    gamma1 = float(sn_asymmetry_to_gamma1(ratio))
    area = float(np.trapezoid(s, t))
    return WindowFeatures(mu=mu, sigma=sigma, gamma1=gamma1, area=area)
```

- [ ] **Step 4: Run tests + quality gates**

```bash
uv run pytest tests/unit/fitting/test_priors_features.py -v
uv run ruff check chromhandler/fitting/priors.py tests/unit/fitting/test_priors_features.py
uv run pyright chromhandler/fitting/priors.py tests/unit/fitting/test_priors_features.py
```

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/priors.py tests/unit/fitting/test_priors_features.py
git commit -m "feat(priors): compute_single_window_features"
```

---

## Task 3: `detect_dominant_apex`

**What:** `(apex_loc, apex_height)` via Savitzky-Golay smoothed argmax. Used by Task 4 to locate the analyte reference apex. No config dependency.

**Files:**
- Modify: `chromhandler/fitting/priors.py`
- Test: `tests/unit/fitting/test_priors_apex.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/fitting/test_priors_apex.py`:

```python
"""Tests for dominant apex detection."""

from __future__ import annotations

import numpy as np
from scipy.stats import skewnorm

from chromhandler.fitting.priors import detect_dominant_apex
from chromhandler.fitting.skew_normal import cp_to_dp


def test_dominant_apex_on_single_peak() -> None:
    t = np.arange(2.5, 2.9, 0.001)
    xi, omega, alpha = (float(x) for x in cp_to_dp(2.7, 0.03, 0.0))
    s = skewnorm.pdf(t, alpha, loc=xi, scale=omega)
    apex_loc, apex_height = detect_dominant_apex(t, s, 2.5, 2.9)
    assert abs(apex_loc - 2.7) < 0.005 and apex_height > 0.0


def test_dominant_apex_picks_taller() -> None:
    t = np.arange(2.5, 2.9, 0.001)
    xi1, om1, a1 = (float(x) for x in cp_to_dp(2.65, 0.02, 0.0))
    xi2, om2, a2 = (float(x) for x in cp_to_dp(2.75, 0.02, 0.0))
    s = 1.0 * skewnorm.pdf(t, a1, loc=xi1, scale=om1) + 0.3 * skewnorm.pdf(
        t, a2, loc=xi2, scale=om2
    )
    apex_loc, _ = detect_dominant_apex(t, s, 2.5, 2.9)
    assert abs(apex_loc - 2.65) < 0.01


def test_dominant_apex_on_noise_returns_argmax() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(2.5, 2.9, 0.001)
    s = rng.normal(0.0, 1.0, size=t.shape)
    apex_loc, apex_height = detect_dominant_apex(t, s, 2.5, 2.9)
    assert 2.5 <= apex_loc <= 2.9 and np.isfinite(apex_height)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_priors_apex.py -v
```

- [ ] **Step 3: Implement**

Append to `chromhandler/fitting/priors.py`:

```python
def detect_dominant_apex(
    time: NDArray[np.float64],
    signal_baseline_subtracted: NDArray[np.float64],
    window_low: float,
    window_high: float,
    smoothing_window: int = 5,
) -> tuple[float, float]:
    """Locate the dominant apex inside a window via smoothed argmax."""
    mask = (time >= window_low) & (time <= window_high) & np.isfinite(
        signal_baseline_subtracted
    )
    t = np.asarray(time[mask], dtype=np.float64)
    s = np.asarray(signal_baseline_subtracted[mask], dtype=np.float64)
    if s.size < smoothing_window:
        raise ValueError(
            f"Window [{window_low}, {window_high}] has only {s.size} valid "
            f"points; need at least {smoothing_window}."
        )
    polyorder = min(3, smoothing_window - 1)
    s_smooth = savgol_filter(s, smoothing_window, polyorder)
    idx = int(np.argmax(s_smooth))
    return float(t[idx]), float(s_smooth[idx])
```

- [ ] **Step 4: Run tests + quality gates**

```bash
uv run pytest tests/unit/fitting/test_priors_apex.py -v
uv run ruff check chromhandler/fitting/priors.py tests/unit/fitting/test_priors_apex.py
uv run pyright chromhandler/fitting/priors.py tests/unit/fitting/test_priors_apex.py
```

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/priors.py tests/unit/fitting/test_priors_apex.py
git commit -m "feat(priors): detect_dominant_apex"
```

---

## Task 4: `extract_artefact_from_controls` — raw measurements + side check

**What:** Returns **raw measurements** (not yet assembled into priors) from control traces, plus per-trace `A_total` for downstream analyte residual computation. The side check happens here. Scale assembly happens in Task 6 (which has access to analyte priors for borrowing).

```python
@dataclass(frozen=True)
class ArtefactMeasurements:
    mu_per_control: NDArray[np.float64]
    log_sigma_per_control: NDArray[np.float64]
    gamma1_per_control: NDArray[np.float64]
    log_area_per_control: NDArray[np.float64]
    A_artefact_est: float                # mean(area_per_control)
    A_total_per_trace: NDArray[np.float64]   # [n_trace], for analyte residual
    mu_artefact: float                       # mean(mu_per_control)
    mu_analyte_ref: float                    # apex of max-total non-control trace
    delta_signed: float                      # mu_artefact - mu_analyte_ref
```

**Files:**
- Modify: `chromhandler/fitting/priors.py`
- Test: `tests/unit/fitting/test_priors_controls.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/fitting/test_priors_controls.py`:

```python
"""Tests for controls-based artefact measurement extraction."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import skewnorm

from chromhandler.annotations import PeakAnnotation
from chromhandler.fitting.priors import (
    ArtefactMeasurements, PriorConfig, extract_artefact_from_controls,
)


def _trace(mu_analyte, A_analyte, mu_artefact, A_artefact):
    t = np.arange(2.5, 3.2, 0.001)
    s = A_artefact * skewnorm.pdf(t, 0.0, loc=mu_artefact, scale=0.025)
    if mu_analyte is not None and A_analyte > 0:
        s += A_analyte * skewnorm.pdf(t, 0.0, loc=mu_analyte, scale=0.025)
    return t, s


def _make_dataset(mu_artefact=2.95, A_artefact=5.0, n_control=2):
    rows_t, rows_s, ic = [], [], []
    for A_an in [100.0, 60.0, 20.0]:
        t, s = _trace(2.85, A_an, mu_artefact, A_artefact)
        rows_t.append(t); rows_s.append(s); ic.append(False)
    for _ in range(n_control):
        t, s = _trace(None, 0.0, mu_artefact, A_artefact)
        rows_t.append(t); rows_s.append(s); ic.append(True)
    return np.array(rows_t), np.array(rows_s), np.array(ic, dtype=np.bool_)


def _ann(side="right"):
    return PeakAnnotation(
        molecule_id="ana", rt_min=2.78, rt_max=3.05,
        mode="artefact_doublet", artefact_side=side,
    )


def test_returns_measurements_dataclass() -> None:
    time, signal, is_control = _make_dataset()
    out = extract_artefact_from_controls(
        time=time, signal=signal, is_control=is_control,
        annotation=_ann(), dt=0.001, config=PriorConfig(),
    )
    assert isinstance(out, ArtefactMeasurements)
    assert out.A_total_per_trace.shape == (5,)
    assert out.mu_per_control.shape == (2,)


def test_recovers_artefact_area() -> None:
    time, signal, is_control = _make_dataset(A_artefact=5.0)
    out = extract_artefact_from_controls(
        time=time, signal=signal, is_control=is_control,
        annotation=_ann(), dt=0.001, config=PriorConfig(),
    )
    assert abs(out.A_artefact_est - 5.0) / 5.0 < 0.05


def test_recovers_delta_signed() -> None:
    time, signal, is_control = _make_dataset(mu_artefact=2.95)
    out = extract_artefact_from_controls(
        time=time, signal=signal, is_control=is_control,
        annotation=_ann(), dt=0.001, config=PriorConfig(),
    )
    assert abs(out.delta_signed - 0.10) < 0.01


def test_side_check_raises_on_mismatch() -> None:
    time, signal, is_control = _make_dataset(mu_artefact=2.95)
    with pytest.raises(ValueError, match="artefact_side"):
        extract_artefact_from_controls(
            time=time, signal=signal, is_control=is_control,
            annotation=_ann(side="left"),  # artefact actually on right
            dt=0.001, config=PriorConfig(),
        )


def test_side_check_raises_when_peaks_too_close() -> None:
    time, signal, is_control = _make_dataset(mu_artefact=2.852)
    with pytest.raises(ValueError, match="too close"):
        extract_artefact_from_controls(
            time=time, signal=signal, is_control=is_control,
            annotation=_ann(), dt=0.001, config=PriorConfig(),
        )


def test_raises_when_no_controls() -> None:
    time, signal, _ = _make_dataset()
    is_control = np.zeros(5, dtype=np.bool_)
    with pytest.raises(ValueError, match="no control"):
        extract_artefact_from_controls(
            time=time, signal=signal, is_control=is_control,
            annotation=_ann(), dt=0.001, config=PriorConfig(),
        )


def test_config_override_changes_epsilon() -> None:
    """A larger epsilon makes the side check stricter."""
    time, signal, is_control = _make_dataset(mu_artefact=2.86)  # delta ≈ 0.01 ≈ 10·dt
    # Default epsilon = 3·dt = 0.003; 0.01 passes.
    extract_artefact_from_controls(
        time=time, signal=signal, is_control=is_control,
        annotation=_ann(), dt=0.001, config=PriorConfig(),
    )
    # Bump to 30·dt → too tight; raises.
    with pytest.raises(ValueError, match="too close"):
        extract_artefact_from_controls(
            time=time, signal=signal, is_control=is_control,
            annotation=_ann(), dt=0.001,
            config=PriorConfig(side_check_epsilon_dt_multiplier=30.0),
        )
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_priors_controls.py -v
```

- [ ] **Step 3: Implement**

Append to `chromhandler/fitting/priors.py`:

```python
from chromhandler.annotations import PeakAnnotation


@dataclass(frozen=True)
class ArtefactMeasurements:
    """Raw artefact measurements from control traces + analyte-residual inputs.

    Outputs of :func:`extract_artefact_from_controls`. Scale assembly is
    deferred to :func:`aggregate_doublet_priors`, which has access to
    analyte-side scales for principled borrowing.

    Attributes:
        mu_per_control: ``[n_controls]`` per-control apex locations.
        log_sigma_per_control: ``[n_controls]`` log of per-control sigmas.
        gamma1_per_control: ``[n_controls]`` per-control gamma1 estimates.
        log_area_per_control: ``[n_controls]`` log of per-control areas.
        A_artefact_est: ``mean(area_per_control)`` in linear units.
        A_total_per_trace: ``[n_trace]`` per-trace total area in the window
            (trapezoid over baseline-subtracted signal). Used for the analyte
            residual ``A_analyte[trace] = A_total[trace] - A_artefact_est``.
        mu_artefact: ``mean(mu_per_control)``.
        mu_analyte_ref: Apex location in the non-control trace with the
            largest ``A_total``.
        delta_signed: ``mu_artefact - mu_analyte_ref`` (positive when
            artefact is later than analyte, i.e. on the right).
    """

    mu_per_control: NDArray[np.float64]
    log_sigma_per_control: NDArray[np.float64]
    gamma1_per_control: NDArray[np.float64]
    log_area_per_control: NDArray[np.float64]
    A_artefact_est: float
    A_total_per_trace: NDArray[np.float64]
    mu_artefact: float
    mu_analyte_ref: float
    delta_signed: float


def _trapezoid_per_trace_in_window(
    time: NDArray[np.float64],
    signal_baseline_subtracted: NDArray[np.float64],
    window_low: float,
    window_high: float,
) -> NDArray[np.float64]:
    n_trace = time.shape[0]
    out = np.zeros(n_trace, dtype=np.float64)
    for tr in range(n_trace):
        mask = (
            (time[tr] >= window_low)
            & (time[tr] <= window_high)
            & np.isfinite(signal_baseline_subtracted[tr])
        )
        if mask.sum() >= 2:
            out[tr] = float(np.trapezoid(
                signal_baseline_subtracted[tr][mask], time[tr][mask]
            ))
    return out


def extract_artefact_from_controls(
    time: NDArray[np.float64],
    signal: NDArray[np.float64],
    is_control: NDArray[np.bool_],
    annotation: PeakAnnotation,
    dt: float,
    config: PriorConfig,
) -> ArtefactMeasurements:
    """Extract raw artefact measurements from control traces; check side.

    Args:
        time: ``[n_trace, n_time]`` NaN-padded time array.
        signal: ``[n_trace, n_time]`` baseline-subtracted signal.
        is_control: ``[n_trace]`` bool mask.
        annotation: doublet :class:`PeakAnnotation` with ``artefact_side``.
        dt: Sampling interval.
        config: :class:`PriorConfig` controlling thresholds.

    Returns:
        :class:`ArtefactMeasurements`.

    Raises:
        ValueError: if no controls, if peaks are too close to distinguish
            at sampling resolution, or if observed side mismatches
            ``annotation.artefact_side``.
    """
    if annotation.artefact_side is None:
        raise ValueError(
            f"annotation.artefact_side must be set for artefact_doublet "
            f"mode (peak {annotation.molecule_id})."
        )

    control_idx = np.where(is_control)[0]
    if control_idx.size == 0:
        raise ValueError(
            f"Peak {annotation.molecule_id}: no control traces in dataset; "
            f"cannot extract artefact priors. Mark controls in the conditions "
            f"CSV or switch annotation mode."
        )

    # Per-control FWHM features.
    control_features = [
        compute_single_window_features(
            time[i], signal[i], annotation.rt_min, annotation.rt_max
        )
        for i in control_idx
    ]
    mu_per_control = np.array([f.mu for f in control_features])
    sigma_per_control = np.clip(
        np.array([f.sigma for f in control_features]), 1e-9, None
    )
    log_sigma_per_control = np.log(sigma_per_control)
    gamma1_per_control = np.array([f.gamma1 for f in control_features])
    area_per_control = np.array([f.area for f in control_features])
    log_area_per_control = np.log(np.clip(np.abs(area_per_control), 1e-9, None))
    mu_artefact = float(np.mean(mu_per_control))
    A_artefact_est = float(np.mean(area_per_control))

    # Per-trace A_total over window.
    A_total = _trapezoid_per_trace_in_window(
        time, signal, annotation.rt_min, annotation.rt_max
    )

    # Reference analyte trace = non-control with max A_total.
    non_control_mask = ~is_control
    if not non_control_mask.any():
        raise ValueError(
            f"Peak {annotation.molecule_id}: dataset has no non-control traces."
        )
    non_control_idx = np.where(non_control_mask)[0]
    ref_trace_idx = int(non_control_idx[int(np.argmax(A_total[non_control_idx]))])
    mu_analyte_ref, _ = detect_dominant_apex(
        time[ref_trace_idx], signal[ref_trace_idx],
        annotation.rt_min, annotation.rt_max,
    )

    # Side check.
    delta_signed = mu_artefact - mu_analyte_ref
    epsilon = config.side_check_epsilon_dt_multiplier * dt
    if abs(delta_signed) < epsilon:
        raise ValueError(
            f"Peak {annotation.molecule_id}: artefact apex from controls "
            f"({mu_artefact:.4f}) and analyte apex from max-total trace "
            f"({mu_analyte_ref:.4f}) differ by {delta_signed:+.4f} min, which "
            f"is too close to distinguish at sampling resolution "
            f"({config.side_check_epsilon_dt_multiplier}*dt = {epsilon:.4f}). "
            f"Peaks unresolved; widen the annotation window or pick different "
            f"control traces."
        )
    observed_side = "right" if delta_signed > 0 else "left"
    if observed_side != annotation.artefact_side:
        raise ValueError(
            f"Peak {annotation.molecule_id}: artefact_side="
            f"'{annotation.artefact_side}' but controls indicate artefact is "
            f"on the {observed_side} side (mu_artefact={mu_artefact:.4f}, "
            f"mu_analyte_ref={mu_analyte_ref:.4f}, delta={delta_signed:+.4f}). "
            f"Fix artefact_side or check control trace identity."
        )

    return ArtefactMeasurements(
        mu_per_control=mu_per_control,
        log_sigma_per_control=log_sigma_per_control,
        gamma1_per_control=gamma1_per_control,
        log_area_per_control=log_area_per_control,
        A_artefact_est=A_artefact_est,
        A_total_per_trace=A_total,
        mu_artefact=mu_artefact,
        mu_analyte_ref=mu_analyte_ref,
        delta_signed=delta_signed,
    )
```

- [ ] **Step 4: Run tests + quality gates**

```bash
uv run pytest tests/unit/fitting/test_priors_controls.py -v
uv run ruff check chromhandler/fitting/priors.py tests/unit/fitting/test_priors_controls.py
uv run pyright chromhandler/fitting/priors.py tests/unit/fitting/test_priors_controls.py
```
Expected: 7 pass, clean.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/priors.py tests/unit/fitting/test_priors_controls.py
git commit -m "feat(priors): extract_artefact_from_controls — raw measurements + side check"
```

---

## Task 5: `aggregate_single_peak_priors` — single peaks with config

**What:** Aggregates per-trace `WindowFeatures` (from non-control traces) into a `SkewNormalPriors` with `n_components=1`. Uses `PriorConfig` for all bounds and fallback scales. Replaces `sqrt(6/n)` etc. with proper measurement-uncertainty-based fallbacks.

**Files:**
- Modify: `chromhandler/fitting/priors.py`
- Test: `tests/unit/fitting/test_priors_aggregate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/fitting/test_priors_aggregate.py`:

```python
"""Tests for single-peak prior aggregation."""

from __future__ import annotations

import numpy as np

from chromhandler.fitting.priors import (
    PriorConfig, WindowFeatures, aggregate_single_peak_priors,
)
from chromhandler.fitting.skew_normal import GAMMA1_MAX


def _features(mus, sigmas, gamma1s, areas):
    return [
        WindowFeatures(mu=mu, sigma=sigma, gamma1=gamma1, area=area)
        for mu, sigma, gamma1, area in zip(mus, sigmas, gamma1s, areas, strict=True)
    ]


def test_recovers_population_stats() -> None:
    rng = np.random.default_rng(0)
    n = 50
    mus = rng.normal(2.70, 0.002, size=n).tolist()
    sigmas = np.exp(rng.normal(np.log(0.03), 0.05, size=n)).tolist()
    gamma1s = rng.normal(0.2, 0.05, size=n).tolist()
    areas = np.exp(rng.normal(np.log(100.0), 0.1, size=n)).tolist()
    p = aggregate_single_peak_priors(
        per_trace_features=_features(mus, sigmas, gamma1s, areas),
        window_low=2.55, window_high=2.85, dt=0.001,
        noise_per_trace=np.full(n, 1.0), n_window_points=300,
        config=PriorConfig(),
    )
    assert abs(p.mu_left_loc - 2.70) < 0.001
    assert 0.0005 < p.mu_left_scale < 0.005
    assert p.mu_left_low == 2.55 and p.mu_left_high == 2.85
    assert abs(p.log_sigma_left_loc - np.log(0.03)) < 0.02
    assert abs(p.gamma1_left_loc - 0.2) < 0.02
    assert p.log_A_left_loc_per_trace.shape == (n,)
    assert p.n_components == 1


def test_single_trace_uses_config_fallbacks() -> None:
    cfg = PriorConfig(log_sigma_scale_n1=0.15, gamma1_scale_n1=0.20,
                      mu_scale_dt_floor_multiplier=1.0, log_A_scale_n1_min=0.10)
    p = aggregate_single_peak_priors(
        per_trace_features=_features([2.70], [0.03], [0.2], [100.0]),
        window_low=2.55, window_high=2.85, dt=0.001,
        noise_per_trace=np.array([1.0]), n_window_points=300, config=cfg,
    )
    assert p.mu_left_scale == 0.001                      # 1 * dt
    assert abs(p.log_sigma_left_scale - 0.15) < 1e-9     # config.log_sigma_scale_n1
    assert abs(p.gamma1_left_scale - 0.20) < 1e-9        # config.gamma1_scale_n1
    assert p.log_A_left_scale >= 0.10                    # at least log_A_scale_n1_min


def test_geometric_bounds_from_config() -> None:
    cfg = PriorConfig(sigma_low_n_points_per_fwhm=8, sigma_high_window_fraction=6.0)
    p = aggregate_single_peak_priors(
        per_trace_features=_features([2.70] * 5, [0.03] * 5, [0.0] * 5, [100.0] * 5),
        window_low=2.55, window_high=2.85, dt=0.001,
        noise_per_trace=np.full(5, 1.0), n_window_points=300, config=cfg,
    )
    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    assert abs(p.log_sigma_left_low - np.log(8 * 0.001 * fwhm_to_sigma)) < 1e-9
    assert abs(p.log_sigma_left_high - np.log((2.85 - 2.55) / 6.0)) < 1e-9


def test_gamma1_scale_capped_by_max() -> None:
    cfg = PriorConfig(gamma1_scale_n1=2.5)  # > GAMMA1_MAX
    p = aggregate_single_peak_priors(
        per_trace_features=_features([2.70], [0.03], [0.0], [100.0]),
        window_low=2.55, window_high=2.85, dt=0.001,
        noise_per_trace=np.array([1.0]), n_window_points=300, config=cfg,
    )
    assert p.gamma1_left_scale <= GAMMA1_MAX + 1e-9
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_priors_aggregate.py -v
```

- [ ] **Step 3: Implement**

Append to `chromhandler/fitting/priors.py`:

```python
from chromhandler.fitting.skew_normal import GAMMA1_MAX


def _log_sigma_bounds(window_low, window_high, dt, config):
    sigma_low = config.sigma_low_n_points_per_fwhm * dt * _FWHM_TO_SIGMA
    sigma_high = (window_high - window_low) / config.sigma_high_window_fraction
    return float(np.log(sigma_low)), float(np.log(sigma_high))


def _gamma1_bounds(config):
    bound = config.gamma1_bound_fraction * GAMMA1_MAX
    return float(-bound), float(bound)


def _log_A_scale_from_noise_propagation(
    areas, noise_per_trace, n_window_points, dt, n_trace, config,
):
    """log_A scale for a single-peak set: noise propagation, floored."""
    median_noise = float(np.median(noise_per_trace))
    sigma_area = median_noise * float(np.sqrt(n_window_points)) * float(dt)
    median_area = float(np.median(np.abs(areas))) if areas.size > 0 else 0.0
    cv = 1.0 if median_area <= 0.0 else sigma_area / median_area
    propagated = float(np.log1p(cv))
    if n_trace == 1:
        return max(propagated, config.log_A_scale_n1_min)
    # n>=2: take max of empirical residual and propagation / sqrt(n)
    return max(propagated, config.log_A_scale_n1_min / float(np.sqrt(n_trace)))


def aggregate_single_peak_priors(
    per_trace_features: list[WindowFeatures],
    window_low: float,
    window_high: float,
    dt: float,
    noise_per_trace: NDArray[np.float64],
    n_window_points: int,
    config: PriorConfig,
) -> SkewNormalPriors:
    """Aggregate per-trace single-peak features into a :class:`SkewNormalPriors`.

    All scale fallbacks for the n=1 case come from ``config``.
    """
    n = len(per_trace_features)
    if n == 0:
        raise ValueError("per_trace_features must be non-empty.")

    mus = np.asarray([f.mu for f in per_trace_features])
    sigmas = np.asarray([f.sigma for f in per_trace_features])
    gamma1s = np.asarray([f.gamma1 for f in per_trace_features])
    areas = np.asarray([f.area for f in per_trace_features])
    log_sigmas = np.log(np.clip(sigmas, 1e-9, None))
    log_areas = np.log(np.clip(np.abs(areas), 1e-9, None))

    mu_floor = config.mu_scale_dt_floor_multiplier * dt
    mu_loc = float(np.mean(mus))
    mu_scale = float(max(np.std(mus, ddof=0), mu_floor))

    log_sigma_loc = float(np.mean(log_sigmas))
    if n == 1:
        log_sigma_scale = config.log_sigma_scale_n1
    else:
        log_sigma_scale = float(max(
            np.std(log_sigmas, ddof=0),
            config.log_sigma_scale_n1 / float(np.sqrt(n)),
        ))

    gamma1_loc = float(np.mean(gamma1s))
    if n == 1:
        gamma1_scale = config.gamma1_scale_n1
    else:
        gamma1_scale = float(max(
            np.std(gamma1s, ddof=0),
            config.gamma1_scale_n1 / float(np.sqrt(n)),
        ))
    gamma1_bound_low, gamma1_bound_high = _gamma1_bounds(config)
    gamma1_scale = min(gamma1_scale, gamma1_bound_high)  # never exceed support

    log_sigma_low, log_sigma_high = _log_sigma_bounds(window_low, window_high, dt, config)
    log_A_scale = _log_A_scale_from_noise_propagation(
        areas, noise_per_trace, n_window_points, dt, n, config,
    )

    return SkewNormalPriors(
        n_components=1,
        mu_left_loc=mu_loc, mu_left_scale=mu_scale,
        mu_left_low=window_low, mu_left_high=window_high,
        log_sigma_left_loc=log_sigma_loc, log_sigma_left_scale=log_sigma_scale,
        log_sigma_left_low=log_sigma_low, log_sigma_left_high=log_sigma_high,
        gamma1_left_loc=gamma1_loc, gamma1_left_scale=gamma1_scale,
        log_A_left_loc_per_trace=log_areas, log_A_left_scale=log_A_scale,
        Delta_loc=None, Delta_scale=None, Delta_low=None, Delta_high=None,
        log_sigma_right_loc=None, log_sigma_right_scale=None,
        log_sigma_right_low=None, log_sigma_right_high=None,
        gamma1_right_loc=None, gamma1_right_scale=None,
        log_A_right_loc_per_trace=None, log_A_right_scale=None,
    )
```

- [ ] **Step 4: Run tests + quality gates**

```bash
uv run pytest tests/unit/fitting/test_priors_aggregate.py -v
uv run ruff check chromhandler/fitting/priors.py tests/unit/fitting/test_priors_aggregate.py
uv run pyright chromhandler/fitting/priors.py tests/unit/fitting/test_priors_aggregate.py
```

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/priors.py tests/unit/fitting/test_priors_aggregate.py
git commit -m "feat(priors): aggregate_single_peak_priors with PriorConfig fallbacks"
```

---

## Task 6: `aggregate_doublet_priors` — assembly with analyte-borrowed scales

**What:** Takes the analyte's already-aggregated `SkewNormalPriors` (n_components=1) + `ArtefactMeasurements` from Task 4, plus config, and produces the full doublet `SkewNormalPriors` (n_components=2).

Key scale logic — **for n_controls=1, borrow from the analyte's empirical scales**:
- `mu_artefact_scale = analyte_priors.mu_left_scale` (same chromatographic drift)
- `log_sigma_right_scale = analyte_priors.log_sigma_left_scale` (same column chemistry)
- `gamma1_right_scale = analyte_priors.gamma1_left_scale` (same shape physics)

For `Delta_scale` (n=1): `config.delta_scale_dt_multiplier_n1 * dt`.

For `A_artefact_scale` (n=1): noise propagation including baseline OLS uncertainty, floored at `config.log_A_artefact_min_scale`.

For n≥2 controls: `max(empirical_std, n1_fallback / sqrt(n))`.

**Files:**
- Modify: `chromhandler/fitting/priors.py`
- Test: extend `tests/unit/fitting/test_priors_aggregate.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/unit/fitting/test_priors_aggregate.py`:

```python
from chromhandler.fitting.priors import (
    ArtefactMeasurements, aggregate_doublet_priors,
)


def _analyte_single():
    return aggregate_single_peak_priors(
        per_trace_features=[
            WindowFeatures(mu=2.85, sigma=0.025, gamma1=0.1, area=100.0),
            WindowFeatures(mu=2.851, sigma=0.025, gamma1=0.1, area=60.0),
            WindowFeatures(mu=2.849, sigma=0.025, gamma1=0.1, area=20.0),
        ],
        window_low=2.78, window_high=3.05, dt=0.001,
        noise_per_trace=np.full(3, 0.1), n_window_points=270,
        config=PriorConfig(),
    )


def _measurements_single_control():
    return ArtefactMeasurements(
        mu_per_control=np.array([2.95]),
        log_sigma_per_control=np.array([np.log(0.025)]),
        gamma1_per_control=np.array([0.05]),
        log_area_per_control=np.array([np.log(5.0)]),
        A_artefact_est=5.0,
        A_total_per_trace=np.array([105.0, 65.0, 25.0, 5.0, 5.0]),
        mu_artefact=2.95, mu_analyte_ref=2.85, delta_signed=0.10,
    )


def test_single_control_borrows_analyte_scales() -> None:
    analyte = _analyte_single()
    artefact = _measurements_single_control()
    p = aggregate_doublet_priors(
        analyte_priors=analyte, artefact=artefact,
        window_low=2.78, window_high=3.05, dt=0.001,
        n_window_points=270, noise_per_trace=np.full(5, 0.1),
        baseline_se_per_trace=np.full(5, 0.05),
        config=PriorConfig(),
    )
    # n_controls=1: shape scales borrow from analyte
    assert p.log_sigma_right_scale == analyte.log_sigma_left_scale
    assert p.gamma1_right_scale == analyte.gamma1_left_scale
    # Delta_scale = 1.5 * dt
    assert abs(p.Delta_scale - 0.0015) < 1e-9
    # Delta bounds from config
    assert p.Delta_low == 3.0 * 0.001
    assert p.Delta_high == (3.05 - 2.78) / 2.0


def test_doublet_assembly_correctness() -> None:
    analyte = _analyte_single()
    artefact = _measurements_single_control()
    p = aggregate_doublet_priors(
        analyte_priors=analyte, artefact=artefact,
        window_low=2.78, window_high=3.05, dt=0.001,
        n_window_points=270, noise_per_trace=np.full(5, 0.1),
        baseline_se_per_trace=np.full(5, 0.05),
        config=PriorConfig(),
    )
    assert p.n_components == 2
    # Left fields come from analyte_priors
    assert p.mu_left_loc == analyte.mu_left_loc
    assert p.log_sigma_left_loc == analyte.log_sigma_left_loc
    # Right shape locs from controls
    assert p.log_sigma_right_loc == np.log(0.025)
    assert p.gamma1_right_loc == 0.05
    # Delta from artefact.delta_signed (abs)
    assert p.Delta_loc == 0.10
    # log_A_left from A_total residual
    assert p.log_A_left_loc_per_trace is not None
    np.testing.assert_allclose(
        p.log_A_left_loc_per_trace,
        np.log(np.maximum(np.array([105.0, 65.0, 25.0, 5.0, 5.0]) - 5.0,
                          # A_floor from noise propagation
                          0.1 * np.sqrt(270) * 0.001)),
        atol=1e-6,
    )
    # log_A_right constant across all traces
    assert p.log_A_right_loc_per_trace is not None
    assert all(p.log_A_right_loc_per_trace == np.log(5.0))


def test_multi_control_uses_empirical_scale() -> None:
    analyte = _analyte_single()
    # Two controls with some spread
    artefact = ArtefactMeasurements(
        mu_per_control=np.array([2.948, 2.952]),
        log_sigma_per_control=np.array([np.log(0.024), np.log(0.026)]),
        gamma1_per_control=np.array([0.04, 0.06]),
        log_area_per_control=np.array([np.log(4.9), np.log(5.1)]),
        A_artefact_est=5.0,
        A_total_per_trace=np.array([105.0, 65.0, 25.0, 5.0, 5.0]),
        mu_artefact=2.95, mu_analyte_ref=2.85, delta_signed=0.10,
    )
    p = aggregate_doublet_priors(
        analyte_priors=analyte, artefact=artefact,
        window_low=2.78, window_high=3.05, dt=0.001,
        n_window_points=270, noise_per_trace=np.full(5, 0.1),
        baseline_se_per_trace=np.full(5, 0.05),
        config=PriorConfig(),
    )
    # Empirical Delta_scale ~ std(|0.098, 0.102|) ~ 0.002; max with 1.5*dt/sqrt(2) ~ 0.001
    assert p.Delta_scale is not None
    assert p.Delta_scale >= 1e-3  # at least the per-sqrt-n floor
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_priors_aggregate.py -v
```

- [ ] **Step 3: Implement**

Append to `chromhandler/fitting/priors.py`:

```python
def aggregate_doublet_priors(
    analyte_priors: SkewNormalPriors,
    artefact: ArtefactMeasurements,
    window_low: float,
    window_high: float,
    dt: float,
    n_window_points: int,
    noise_per_trace: NDArray[np.float64],
    baseline_se_per_trace: NDArray[np.float64],
    config: PriorConfig,
) -> SkewNormalPriors:
    """Assemble doublet priors from analyte single-peak priors + artefact measurements.

    For n_controls=1, shape and position scales borrow from analyte_priors.
    For n_controls>=2, scales are ``max(empirical_std, borrowed/sqrt(n))``.

    Args:
        analyte_priors: Output of :func:`aggregate_single_peak_priors` on
            non-control traces (must have ``n_components == 1``).
        artefact: :class:`ArtefactMeasurements` from
            :func:`extract_artefact_from_controls`.
        window_low: Annotation lower bound.
        window_high: Annotation upper bound.
        dt: Sampling interval.
        n_window_points: Median in-window sample count.
        noise_per_trace: ``[n_trace]`` per-trace noise std (full dataset).
        baseline_se_per_trace: ``[n_trace]`` per-trace OLS baseline standard
            error (signal units). Used to widen ``log_A_right_scale``.
        config: :class:`PriorConfig`.

    Returns:
        :class:`SkewNormalPriors` with ``n_components=2``.

    Raises:
        ValueError: If ``analyte_priors.n_components != 1``.
    """
    if analyte_priors.n_components != 1:
        raise ValueError(
            "analyte_priors must be a single-peak prior (n_components=1)."
        )

    n_c = artefact.mu_per_control.size

    # --- Δ ---
    delta_loc = abs(artefact.delta_signed)
    delta_scale_n1 = config.delta_scale_dt_multiplier_n1 * dt
    if n_c == 1:
        delta_scale = delta_scale_n1
    else:
        per_control_seps = np.abs(artefact.mu_per_control - artefact.mu_analyte_ref)
        empirical = float(np.std(per_control_seps, ddof=0))
        delta_scale = max(empirical, delta_scale_n1 / float(np.sqrt(n_c)))
    delta_low = config.delta_low_dt_multiplier * dt
    delta_high = (window_high - window_low) / config.delta_high_window_fraction

    # --- Right component shape: borrow from analyte for n=1 ---
    log_sigma_right_loc = float(np.mean(artefact.log_sigma_per_control))
    if n_c == 1:
        log_sigma_right_scale = analyte_priors.log_sigma_left_scale
    else:
        empirical = float(np.std(artefact.log_sigma_per_control, ddof=0))
        log_sigma_right_scale = max(
            empirical, analyte_priors.log_sigma_left_scale / float(np.sqrt(n_c)),
        )

    gamma1_right_loc = float(np.mean(artefact.gamma1_per_control))
    if n_c == 1:
        gamma1_right_scale = analyte_priors.gamma1_left_scale
    else:
        empirical = float(np.std(artefact.gamma1_per_control, ddof=0))
        gamma1_right_scale = max(
            empirical, analyte_priors.gamma1_left_scale / float(np.sqrt(n_c)),
        )
    _, gamma1_bound_high = _gamma1_bounds(config)
    gamma1_right_scale = min(gamma1_right_scale, gamma1_bound_high)

    log_sigma_low, log_sigma_high = _log_sigma_bounds(window_low, window_high, dt, config)

    # --- A_artefact scale: noise + baseline propagation, floored ---
    median_noise = float(np.median(noise_per_trace))
    sigma_A_noise = median_noise * float(np.sqrt(n_window_points)) * dt
    median_baseline_se = float(np.median(baseline_se_per_trace))
    sigma_A_baseline = median_baseline_se * (window_high - window_low)
    sigma_A_total = float(np.sqrt(sigma_A_noise**2 + sigma_A_baseline**2))
    A_artefact_est = max(artefact.A_artefact_est, 1e-9)
    propagated = float(np.log1p(sigma_A_total / A_artefact_est))
    if n_c >= 2:
        empirical = float(np.std(artefact.log_area_per_control, ddof=0))
        log_A_right_scale = max(empirical, propagated, config.log_A_artefact_min_scale)
    else:
        log_A_right_scale = max(propagated, config.log_A_artefact_min_scale)

    # --- log_A_left from A_total residual; A_floor from noise propagation ---
    A_floor = sigma_A_noise  # noise·sqrt(n_pts)·dt
    A_analyte = np.maximum(
        artefact.A_total_per_trace - artefact.A_artefact_est, A_floor,
    )
    log_A_left_loc_per_trace = np.log(A_analyte)
    log_A_left_scale = _log_A_scale_from_noise_propagation(
        A_analyte, noise_per_trace, n_window_points, dt,
        A_analyte.size, config,
    )

    # --- log_A_right per trace (constant) ---
    n_total = artefact.A_total_per_trace.size
    log_A_right_loc_per_trace = np.full(
        n_total, float(np.log(A_artefact_est)), dtype=np.float64,
    )

    return SkewNormalPriors(
        n_components=2,
        mu_left_loc=analyte_priors.mu_left_loc,
        mu_left_scale=analyte_priors.mu_left_scale,
        mu_left_low=window_low, mu_left_high=window_high,
        log_sigma_left_loc=analyte_priors.log_sigma_left_loc,
        log_sigma_left_scale=analyte_priors.log_sigma_left_scale,
        log_sigma_left_low=log_sigma_low, log_sigma_left_high=log_sigma_high,
        gamma1_left_loc=analyte_priors.gamma1_left_loc,
        gamma1_left_scale=analyte_priors.gamma1_left_scale,
        log_A_left_loc_per_trace=log_A_left_loc_per_trace,
        log_A_left_scale=log_A_left_scale,
        Delta_loc=delta_loc, Delta_scale=delta_scale,
        Delta_low=delta_low, Delta_high=delta_high,
        log_sigma_right_loc=log_sigma_right_loc,
        log_sigma_right_scale=log_sigma_right_scale,
        log_sigma_right_low=log_sigma_low, log_sigma_right_high=log_sigma_high,
        gamma1_right_loc=gamma1_right_loc,
        gamma1_right_scale=gamma1_right_scale,
        log_A_right_loc_per_trace=log_A_right_loc_per_trace,
        log_A_right_scale=log_A_right_scale,
    )
```

- [ ] **Step 4: Run tests + quality gates**

```bash
uv run pytest tests/unit/fitting/test_priors_aggregate.py -v
uv run ruff check chromhandler/fitting/priors.py tests/unit/fitting/test_priors_aggregate.py
uv run pyright chromhandler/fitting/priors.py tests/unit/fitting/test_priors_aggregate.py
```

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/priors.py tests/unit/fitting/test_priors_aggregate.py
git commit -m "feat(priors): aggregate_doublet_priors with analyte-borrowed scale fallbacks"
```

---

## Task 7: `build_priors` orchestrator

**What:** Threads `PriorConfig` through and orders the doublet pipeline: analyte aggregation first, then artefact extraction, then doublet assembly. Computes per-trace baseline SE for use in artefact area scale.

**Files:**
- Modify: `chromhandler/fitting/priors.py`
- Test: `tests/unit/fitting/test_priors_orchestrator.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/fitting/test_priors_orchestrator.py`:

```python
"""End-to-end orchestrator tests."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import skewnorm

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.prepared_dataset import prepare_dataset
from chromhandler.fitting.priors import PriorConfig, build_priors


def _synth_dataset(n_sample: int = 3, n_control: int = 1, seed: int = 0):
    rng = np.random.default_rng(seed)
    times, signals, is_control = [], [], []
    for A_an in np.linspace(100.0, 10.0, n_sample):
        t = np.arange(2.5, 3.6, 0.001)
        s_ino = A_an * skewnorm.pdf(t, 2.0, loc=2.69, scale=0.025)
        s_main = 80.0 * skewnorm.pdf(t, 0.0, loc=3.00, scale=0.025)
        s_art = 5.0 * skewnorm.pdf(t, 0.0, loc=3.05, scale=0.025)
        baseline = 10.0 + 0.5 * t
        noise = rng.normal(0.0, 1.0, size=t.shape)
        times.append(t); signals.append(s_ino + s_main + s_art + baseline + noise)
        is_control.append(False)
    for _ in range(n_control):
        t = np.arange(2.5, 3.6, 0.001)
        s = 5.0 * skewnorm.pdf(t, 0.0, loc=3.05, scale=0.025)
        baseline = 10.0 + 0.5 * t
        noise = rng.normal(0.0, 1.0, size=t.shape)
        times.append(t); signals.append(s + baseline + noise)
        is_control.append(True)
    peak_anns = [
        PeakAnnotation(molecule_id="Ino", rt_min=2.55, rt_max=2.85),
        PeakAnnotation(
            molecule_id="SIH", rt_min=2.90, rt_max=3.15,
            mode="artefact_doublet", artefact_side="right",
        ),
    ]
    base_anns = [
        BaselineAnnotation(rt_min=2.50, rt_max=2.52),
        BaselineAnnotation(rt_min=3.55, rt_max=3.57),
    ]
    return prepare_dataset(times, signals, peak_anns, base_anns, is_control=is_control)


def test_returns_one_per_annotation() -> None:
    priors = build_priors(_synth_dataset())
    assert len(priors) == 2
    assert priors[0].n_components == 1
    assert priors[1].n_components == 2


def test_single_recovers_mu() -> None:
    priors = build_priors(_synth_dataset())
    assert abs(priors[0].mu_left_loc - 2.70) < 0.02


def test_doublet_delta_from_controls() -> None:
    priors = build_priors(_synth_dataset())
    p = priors[1]
    assert p.Delta_loc is not None
    assert abs(p.Delta_loc - 0.05) < 0.01


def test_doublet_borrows_scales_with_single_control() -> None:
    ds = _synth_dataset(n_control=1)
    priors = build_priors(ds)
    p = priors[1]
    # With single control, shape scales should equal whatever the analyte
    # population had. The "analyte" here for SIH is the SIH main peak,
    # aggregated from non-control traces.
    assert p.log_sigma_right_scale is not None
    assert p.gamma1_right_scale is not None
    assert p.log_sigma_right_scale > 0
    assert p.gamma1_right_scale > 0


def test_config_override_propagates() -> None:
    ds = _synth_dataset()
    cfg = PriorConfig(delta_low_dt_multiplier=5.0)
    priors = build_priors(ds, config=cfg)
    p = priors[1]
    assert p.Delta_low == 5.0 * ds.dt_global


def test_raises_on_doublet_without_controls() -> None:
    ds = _synth_dataset(n_control=0)
    with pytest.raises(ValueError, match="no control"):
        build_priors(ds)


def test_raises_on_free_doublet() -> None:
    ds = _synth_dataset()
    new_anns = list(ds.peak_annotations)
    new_anns[1] = PeakAnnotation(
        molecule_id="X", rt_min=2.90, rt_max=3.15, mode="free_doublet",
    )
    object.__setattr__(ds, "peak_annotations", new_anns)
    with pytest.raises(NotImplementedError, match="free_doublet"):
        build_priors(ds)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_priors_orchestrator.py -v
```

- [ ] **Step 3: Implement**

Append to `chromhandler/fitting/priors.py`:

```python
from chromhandler.fitting.prepared_dataset import PreparedDataset


def _baseline_subtracted(dataset: PreparedDataset) -> NDArray[np.float64]:
    intercept = dataset.baseline_intercept[:, None]
    slope = dataset.baseline_slope[:, None]
    return dataset.signal - (intercept + slope * dataset.time)


def _baseline_se_per_trace(dataset: PreparedDataset) -> NDArray[np.float64]:
    """OLS baseline residual std per trace, evaluated on the baseline regions
    the user annotated. Quantifies how uncertain the baseline subtraction is.
    """
    n_trace = dataset.n_trace
    out = np.zeros(n_trace, dtype=np.float64)
    baseline_sub = _baseline_subtracted(dataset)
    for tr in range(n_trace):
        residuals = []
        for ba in dataset.baseline_annotations:
            mask = (
                (dataset.time[tr] >= ba.rt_min)
                & (dataset.time[tr] <= ba.rt_max)
                & np.isfinite(baseline_sub[tr])
            )
            residuals.extend(baseline_sub[tr][mask].tolist())
        if residuals:
            out[tr] = float(np.std(np.asarray(residuals, dtype=np.float64), ddof=0))
        else:
            out[tr] = float(dataset.noise_per_trace[tr])
    return out


def _count_window_points(time, low, high):
    masks = (time >= low) & (time <= high) & np.isfinite(time)
    return int(np.median(masks.sum(axis=1)))


def build_priors(
    dataset: PreparedDataset,
    config: PriorConfig | None = None,
) -> list[SkewNormalPriors]:
    """Build per-annotation :class:`SkewNormalPriors` from a prepared dataset.

    Args:
        dataset: Output of :func:`prepare_dataset`. Must have ``is_control``
            populated; if any annotation is ``artefact_doublet``, at least
            one trace must be a control.
        config: Optional :class:`PriorConfig`. Defaults to ``PriorConfig()``.

    Returns:
        One :class:`SkewNormalPriors` per ``dataset.peak_annotations``.

    Raises:
        ValueError: For ``artefact_doublet`` with no controls.
        NotImplementedError: For ``free_doublet``.
    """
    cfg = config if config is not None else PriorConfig()
    baseline_sub = _baseline_subtracted(dataset)
    baseline_se = _baseline_se_per_trace(dataset)
    non_control_idx = np.where(~dataset.is_control)[0]
    if non_control_idx.size == 0:
        raise ValueError("Dataset contains only control traces; cannot build priors.")

    out: list[SkewNormalPriors] = []
    for ann in dataset.peak_annotations:
        n_pts = _count_window_points(dataset.time, ann.rt_min, ann.rt_max)
        if ann.mode == "single":
            feats = [
                compute_single_window_features(
                    dataset.time[tr], baseline_sub[tr], ann.rt_min, ann.rt_max
                )
                for tr in non_control_idx
            ]
            out.append(aggregate_single_peak_priors(
                per_trace_features=feats,
                window_low=ann.rt_min, window_high=ann.rt_max,
                dt=dataset.dt_global,
                noise_per_trace=dataset.noise_per_trace[non_control_idx],
                n_window_points=n_pts, config=cfg,
            ))
        elif ann.mode == "artefact_doublet":
            # 1. Analyte single-peak priors from non-control traces.
            analyte_feats = [
                compute_single_window_features(
                    dataset.time[tr], baseline_sub[tr], ann.rt_min, ann.rt_max
                )
                for tr in non_control_idx
            ]
            analyte_priors = aggregate_single_peak_priors(
                per_trace_features=analyte_feats,
                window_low=ann.rt_min, window_high=ann.rt_max,
                dt=dataset.dt_global,
                noise_per_trace=dataset.noise_per_trace[non_control_idx],
                n_window_points=n_pts, config=cfg,
            )
            # 2. Artefact measurements from controls.
            artefact = extract_artefact_from_controls(
                time=dataset.time, signal=baseline_sub,
                is_control=dataset.is_control, annotation=ann,
                dt=dataset.dt_global, config=cfg,
            )
            # 3. Assemble doublet.
            out.append(aggregate_doublet_priors(
                analyte_priors=analyte_priors, artefact=artefact,
                window_low=ann.rt_min, window_high=ann.rt_max,
                dt=dataset.dt_global, n_window_points=n_pts,
                noise_per_trace=dataset.noise_per_trace,
                baseline_se_per_trace=baseline_se, config=cfg,
            ))
        elif ann.mode == "free_doublet":
            raise NotImplementedError(
                f"Peak {ann.molecule_id}: mode='free_doublet' is not yet "
                f"supported. Use 'artefact_doublet' with controls or wait "
                f"for the free_doublet implementation."
            )
        else:
            raise ValueError(f"Unknown peak mode '{ann.mode}'.")
    return out
```

- [ ] **Step 4: Run tests + quality gates**

```bash
uv run pytest tests/unit/fitting/test_priors_orchestrator.py -v
uv run ruff check chromhandler/fitting/priors.py tests/unit/fitting/test_priors_orchestrator.py
uv run pyright chromhandler/fitting/priors.py tests/unit/fitting/test_priors_orchestrator.py
```

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/priors.py tests/unit/fitting/test_priors_orchestrator.py
git commit -m "feat(priors): build_priors orchestrator with PriorConfig threading"
```

---

## Task 8: `summarise_priors` — human-readable inspection

**Files:**
- Modify: `chromhandler/fitting/priors.py`
- Test: `tests/unit/fitting/test_priors_summary.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/fitting/test_priors_summary.py`:

```python
"""Tests for summarise_priors."""

from __future__ import annotations

import numpy as np

from chromhandler.fitting.priors import (
    PriorConfig, SkewNormalPriors, summarise_priors,
)


def _single():
    return SkewNormalPriors(
        n_components=1,
        mu_left_loc=2.70, mu_left_scale=0.005,
        mu_left_low=2.55, mu_left_high=2.85,
        log_sigma_left_loc=np.log(0.03), log_sigma_left_scale=0.1,
        log_sigma_left_low=np.log(0.005), log_sigma_left_high=np.log(0.05),
        gamma1_left_loc=0.2, gamma1_left_scale=0.05,
        log_A_left_loc_per_trace=np.array([np.log(100.0), np.log(80.0)]),
        log_A_left_scale=0.1,
        Delta_loc=None, Delta_scale=None, Delta_low=None, Delta_high=None,
        log_sigma_right_loc=None, log_sigma_right_scale=None,
        log_sigma_right_low=None, log_sigma_right_high=None,
        gamma1_right_loc=None, gamma1_right_scale=None,
        log_A_right_loc_per_trace=None, log_A_right_scale=None,
    )


def test_summary_mentions_distributions() -> None:
    out = summarise_priors([_single()], config=PriorConfig())
    assert "TruncatedNormal" in out and "Normal" in out
    for site in ("mu_anchor_left", "log_sigma_left", "gamma1_left", "log_A_left"):
        assert site in out


def test_doublet_summary_uses_truncated_delta() -> None:
    s = _single()
    d = SkewNormalPriors(
        n_components=2,
        mu_left_loc=3.00, mu_left_scale=0.005,
        mu_left_low=2.90, mu_left_high=3.15,
        log_sigma_left_loc=s.log_sigma_left_loc, log_sigma_left_scale=s.log_sigma_left_scale,
        log_sigma_left_low=s.log_sigma_left_low, log_sigma_left_high=s.log_sigma_left_high,
        gamma1_left_loc=s.gamma1_left_loc, gamma1_left_scale=s.gamma1_left_scale,
        log_A_left_loc_per_trace=np.array([np.log(80.0)]), log_A_left_scale=0.1,
        Delta_loc=0.05, Delta_scale=0.002, Delta_low=0.003, Delta_high=0.125,
        log_sigma_right_loc=s.log_sigma_left_loc, log_sigma_right_scale=s.log_sigma_left_scale,
        log_sigma_right_low=s.log_sigma_left_low, log_sigma_right_high=s.log_sigma_left_high,
        gamma1_right_loc=s.gamma1_left_loc, gamma1_right_scale=s.gamma1_left_scale,
        log_A_right_loc_per_trace=np.array([np.log(5.0)]), log_A_right_scale=np.log(1.5),
    )
    out = summarise_priors([s, d], config=PriorConfig())
    delta_row = next(line for line in out.split("\n") if "Delta" in line)
    assert "TruncatedNormal" in delta_row
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_priors_summary.py -v
```

- [ ] **Step 3: Implement**

Append to `chromhandler/fitting/priors.py`:

```python
def summarise_priors(
    priors: list[SkewNormalPriors],
    config: PriorConfig,
) -> str:
    """Format a list of :class:`SkewNormalPriors` as an inspection table.

    Args:
        priors: Per-peak priors.
        config: Used to display γ₁ bounds in the table.

    Returns:
        Multi-line string.
    """
    gamma1_low, gamma1_high = _gamma1_bounds(config)
    lines: list[str] = []
    header = (
        f"{'peak':>4} {'site':<22} {'distribution':<16} "
        f"{'loc':>10} {'scale':>10} {'low':>10} {'high':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    def fmt(v):
        return f"{v:>10.4g}" if v is not None else f"{'-':>10}"

    for i, p in enumerate(priors):
        lines.append(
            f"{i:>4} {'mu_anchor_left':<22} {'TruncatedNormal':<16} "
            f"{fmt(p.mu_left_loc)} {fmt(p.mu_left_scale)} "
            f"{fmt(p.mu_left_low)} {fmt(p.mu_left_high)}"
        )
        lines.append(
            f"{i:>4} {'log_sigma_left':<22} {'TruncatedNormal':<16} "
            f"{fmt(p.log_sigma_left_loc)} {fmt(p.log_sigma_left_scale)} "
            f"{fmt(p.log_sigma_left_low)} {fmt(p.log_sigma_left_high)}"
        )
        lines.append(
            f"{i:>4} {'gamma1_left':<22} {'TruncatedNormal':<16} "
            f"{fmt(p.gamma1_left_loc)} {fmt(p.gamma1_left_scale)} "
            f"{fmt(gamma1_low)} {fmt(gamma1_high)}"
        )
        log_A_left_mean = float(np.mean(p.log_A_left_loc_per_trace))
        lines.append(
            f"{i:>4} {'log_A_left (mean)':<22} {'Normal':<16} "
            f"{fmt(log_A_left_mean)} {fmt(p.log_A_left_scale)} "
            f"{fmt(None)} {fmt(None)}"
        )
        if p.n_components == 2:
            lines.append(
                f"{i:>4} {'Delta':<22} {'TruncatedNormal':<16} "
                f"{fmt(p.Delta_loc)} {fmt(p.Delta_scale)} "
                f"{fmt(p.Delta_low)} {fmt(p.Delta_high)}"
            )
            lines.append(
                f"{i:>4} {'log_sigma_right':<22} {'TruncatedNormal':<16} "
                f"{fmt(p.log_sigma_right_loc)} {fmt(p.log_sigma_right_scale)} "
                f"{fmt(p.log_sigma_right_low)} {fmt(p.log_sigma_right_high)}"
            )
            lines.append(
                f"{i:>4} {'gamma1_right':<22} {'TruncatedNormal':<16} "
                f"{fmt(p.gamma1_right_loc)} {fmt(p.gamma1_right_scale)} "
                f"{fmt(gamma1_low)} {fmt(gamma1_high)}"
            )
            assert p.log_A_right_loc_per_trace is not None
            log_A_right_mean = float(np.mean(p.log_A_right_loc_per_trace))
            lines.append(
                f"{i:>4} {'log_A_right (mean)':<22} {'Normal':<16} "
                f"{fmt(log_A_right_mean)} {fmt(p.log_A_right_scale)} "
                f"{fmt(None)} {fmt(None)}"
            )
    return "\n".join(lines)
```

- [ ] **Step 4: Full module sweep**

```bash
uv run pytest tests/unit/fitting/ -v -k "test_priors"
uv run ruff check chromhandler/fitting/priors.py
uv run pyright chromhandler/fitting/priors.py
```

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/priors.py tests/unit/fitting/test_priors_summary.py
git commit -m "feat(priors): summarise_priors uses PriorConfig for bound display"
```

---

## Self-Review

**Coverage of decisions from the parent conversation:**

- ✅ `PriorConfig` centralizes every magic number and fallback (n=1 scales, bounds, side-check epsilon, area floors).
- ✅ `extract_artefact_from_controls` returns raw measurements; scale assembly deferred to `aggregate_doublet_priors` where analyte scales are accessible.
- ✅ For n_controls=1, shape scales borrow from analyte's empirical scales (`mu_left_scale`, `log_sigma_left_scale`, `gamma1_left_scale`) — chromatographic drift and shape physics are shared between artefact and analyte.
- ✅ `Delta_scale` for n_controls=1 = `delta_scale_dt_multiplier_n1 * dt` (default 1.5·dt).
- ✅ `A_artefact_scale` includes baseline OLS uncertainty (`baseline_se_per_trace · window_width`), not just trapezoid noise. Computed in Task 7, passed to Task 6.
- ✅ `sqrt(6/n)` and `1/sqrt(n)` fallbacks replaced with measurement-uncertainty-based scales throughout.
- ✅ Δ TruncatedNormal with empirical `(loc, scale)` + geometric bounds.
- ✅ Side check via relative apex shift, with epsilon controlled by config.
- ✅ `artefact_doublet` without controls raises; `free_doublet` raises `NotImplementedError`.

**Type consistency:** `PriorConfig` is threaded through Tasks 4–8. `WindowFeatures` from Task 2 feeds Tasks 4 and 5. `ArtefactMeasurements` from Task 4 feeds Task 6. `SkewNormalPriors` from Task 5 (analyte single-peak) feeds Task 6 as `analyte_priors`. All field names consistent.

**Placeholder scan:** No "TBD" / "TODO". All code blocks complete. n=1 fallbacks named consistently as `*_n1` in `PriorConfig`.

**Future model config:** This plan establishes the `PriorConfig` pattern. A future plan for `model.py` should follow the same pattern: a `ModelConfig` dataclass with HMC settings (warmup, samples, chains, target_accept), distribution choices for model-layer-only parameters (`trace_shift`, `baseline_intercept`, `baseline_slope`), and any other knobs. Both configs can be combined into a top-level `FitterConfig(priors=PriorConfig(), model=ModelConfig())` once both modules exist.

**Out of scope (intentional):**
- `free_doublet` mode (separate future plan).
- Noise-derivation of `log_sigma_scale_n1` and `gamma1_scale_n1` (currently fixed defaults — easy future refinement if needed).
- `ModelConfig` (separate plan when `model.py` is implemented).

**Cross-link:** This plan supersedes the priors content of `docs/plans/rewrite.md` Phase 3.
