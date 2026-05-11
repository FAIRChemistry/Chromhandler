# Priors Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `chromhandler/fitting/priors.py` — an empirical, FWHM-based prior-construction layer that turns a `PreparedDataset` plus its `PeakAnnotation`s into a `list[SkewNormalPriors]`, one per peak, with every field directly parameterizing a named NumPyro distribution in the upcoming `model.py`.

**Architecture:** Pure-function module on top of `skew_normal.py` (already shipped) and `PreparedDataset` (already shipped in foundations). Per-trace FWHM-based feature extraction → population aggregation with principled scale floors → `SkewNormalPriors` dataclass that explicitly carries `(loc, scale, low, high)` quadruples wired to a fixed distribution table. No `_raw` parameters, no tanh squashes — bounded parameters use `TruncatedNormal` directly. Identifiability for doublets is provided by Uniform Δ + spatial area split, not by ordering constraints.

**Tech Stack:** Python 3.11+, NumPy, SciPy (`signal.savgol_filter`, `signal.find_peaks`), JAX/NumPyro (for downstream use only — `priors.py` itself stays NumPy-side), pytest, ruff, pyright. All execution via `uv run`.

**Spec:** [`docs/superpowers/specs/2026-05-07-skew-normal-fitter-rewrite-design.md`](../specs/2026-05-07-skew-normal-fitter-rewrite-design.md), §3–§7.2.

---

## Distribution Table (the contract this module implements)

Every field of `SkewNormalPriors` parameterizes exactly one NumPyro distribution in the future `model.py`. The mapping is fixed below; no other distribution choices are permitted without amending this table.

| Sampled site (model.py) | Distribution | Parameters (loc, scale, low, high) | Prior calculation (where in this plan) |
|---|---|---|---|
| `mu_anchor_left[peak]` | `TruncatedNormal` | `loc = mu_left_loc`, `scale = mu_left_scale`, `low = mu_left_low`, `high = mu_left_high` | Task 5: mean/std of per-trace apex locations; `(low, high) = (window_low, window_high)` from annotation |
| `log_sigma_left[peak]` | `TruncatedNormal` | `loc = log_sigma_left_loc`, `scale = log_sigma_left_scale`, `low = log_sigma_left_low`, `high = log_sigma_left_high` | Task 5: mean/std of per-trace `log((HWHM_L+HWHM_R)/(2·√(2 ln 2)))`; bounds derived from geometry (Task 5) |
| `gamma1_left[peak]` | `TruncatedNormal` | `loc = gamma1_left_loc`, `scale = gamma1_left_scale`, `low = -0.99·GAMMA1_MAX`, `high = +0.99·GAMMA1_MAX` | Task 5: mean/std of per-trace `sn_asymmetry_to_gamma1(HWHM_R/HWHM_L)`; bounds are math constants |
| `log_A_left[trace, peak]` | `Normal` | `loc = log_A_left_loc_per_trace[trace]`, `scale = log_A_left_scale` | Task 5: per-trace `log(trapezoid_area)`; scale from noise propagation |
| `Delta[peak]` (doublet only) | `Uniform` | `low = Delta_low`, `high = Delta_high` | Task 6: `Delta_low = 5·dt`, `Delta_high = window_width/2` — geometry only, no empirical fit |
| `log_sigma_right[peak]` (doublet only) | `TruncatedNormal` | same shape as `log_sigma_left` | Task 6: borrowed from the **population of single-peak windows** in the same dataset (or outer-HWHM fallback) |
| `gamma1_right[peak]` (doublet only) | `TruncatedNormal` | same shape as `gamma1_left` | Task 6: same source as `log_sigma_right` |
| `log_A_right[trace, peak]` (doublet only) | `Normal` | `loc = log_A_right_loc_per_trace[trace]`, `scale = log_A_right_scale` | Task 6: outer-HWHM Gaussian-residual split with spatial assignment |

**Out-of-scope for `priors.py`** (these are model-layer concerns, parameterized directly from `PreparedDataset`, not from `SkewNormalPriors`):

| Sampled site | Distribution | Source |
|---|---|---|
| `trace_shift[trace]` | `Normal(0, drift_scale)` non-centered | `drift_scale = min_peak_window_half_width / 3` — computed in `model.py` from annotations |
| `baseline_intercept[trace]` | `Normal(dataset.baseline_intercept[trace], intercept_se[trace])` | OLS standard errors from `PreparedDataset` (`model.py` reads them) |
| `baseline_slope[trace]` | `Normal(dataset.baseline_slope[trace], slope_se[trace])` | same |

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `chromhandler/fitting/_legacy_priors.py` | New (renamed from existing `priors.py`) | Quarantined legacy code so legacy `fitter.py`/`visualize.py` keep importing while we rewrite |
| `chromhandler/fitting/priors.py` | New (overwrites legacy) | The new module, this plan's deliverable |
| `tests/unit/fitting/test_priors_legacy.py` | New (renamed from existing `test_priors.py`) | Keeps legacy test pointed at quarantined module |
| `tests/unit/fitting/test_priors_features.py` | New | Single-window FWHM feature extraction |
| `tests/unit/fitting/test_priors_apex.py` | New | Dominant apex detection |
| `tests/unit/fitting/test_priors_split.py` | New | Doublet outer-HWHM area split |
| `tests/unit/fitting/test_priors_aggregate.py` | New | Single + doublet aggregation, floors |
| `tests/unit/fitting/test_priors_orchestrator.py` | New | `build_priors` end-to-end |

---

## Conventions

- Quality gate after every file edit:
  ```bash
  uv run ruff check <file>
  uv run pyright <file>
  ```
  Both must report zero issues before committing.
- Tests run with `uv run pytest <file> -v`.
- All new public functions get Google-style docstrings (`Args`, `Returns`, `Raises`).
- `from __future__ import annotations` at top of every new module.
- NumPy is the working numerics layer for this module. JAX appears only in type hints where downstream callers will pass `jnp` arrays (NumPy and JAX arrays are duck-compatible for the operations we use here).
- One commit per task. Commit message format: `feat(priors): <task summary>` for code tasks, `chore(priors): <summary>` for quarantine/rename tasks.

---

## Task 0: Quarantine the legacy priors module

**Why first:** The existing `chromhandler/fitting/priors.py` has a totally different API and is imported by legacy `fitter.py`, `visualize.py`, and `tests/unit/fitting/test_priors.py`. We need the canonical name `priors.py` free for the rewrite, but the legacy code must keep working until later phases replace `fitter.py`. Solution: rename to `_legacy_priors.py` and patch the importers.

**Files:**
- Rename: `chromhandler/fitting/priors.py` → `chromhandler/fitting/_legacy_priors.py`
- Modify: `chromhandler/fitting/fitter.py` (line 44 import)
- Modify: `chromhandler/fitting/visualize.py` (line 26 import)
- Rename: `tests/unit/fitting/test_priors.py` → `tests/unit/fitting/test_priors_legacy.py`
- Modify: all `from chromhandler.fitting.priors import …` lines in `test_priors_legacy.py` → `from chromhandler.fitting._legacy_priors import …`

- [ ] **Step 1: Rename source file**

```bash
git mv chromhandler/fitting/priors.py chromhandler/fitting/_legacy_priors.py
```

- [ ] **Step 2: Patch importers in `fitter.py` and `visualize.py`**

In `chromhandler/fitting/fitter.py` line 44, change:
```python
from .priors import (
```
to:
```python
from ._legacy_priors import (
```

In `chromhandler/fitting/visualize.py` line 26, change:
```python
from .priors import _trace_fwhm_geometry, fwhm_geometry_to_sigma_alpha
```
to:
```python
from ._legacy_priors import _trace_fwhm_geometry, fwhm_geometry_to_sigma_alpha
```

- [ ] **Step 3: Rename legacy test file and patch its imports**

```bash
git mv tests/unit/fitting/test_priors.py tests/unit/fitting/test_priors_legacy.py
```

Then replace every occurrence of `from chromhandler.fitting.priors import` with `from chromhandler.fitting._legacy_priors import` in the renamed file. Use:
```bash
uv run python -c "
import pathlib
p = pathlib.Path('tests/unit/fitting/test_priors_legacy.py')
p.write_text(p.read_text().replace('chromhandler.fitting.priors', 'chromhandler.fitting._legacy_priors'))
"
```

- [ ] **Step 4: Verify legacy still imports and legacy tests still pass**

```bash
uv run ruff check chromhandler/fitting/_legacy_priors.py chromhandler/fitting/fitter.py chromhandler/fitting/visualize.py tests/unit/fitting/test_priors_legacy.py
uv run pyright chromhandler/fitting/_legacy_priors.py chromhandler/fitting/fitter.py chromhandler/fitting/visualize.py
uv run pytest tests/unit/fitting/test_priors_legacy.py -v
```
Expected: ruff clean, pyright clean, all legacy tests pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore(priors): quarantine legacy priors as _legacy_priors to free the name"
```

---

## Task 1: `SkewNormalPriors` dataclass

**Why:** Defines the contract from the distribution table. Every aggregator in later tasks returns this type.

**Files:**
- Create: `chromhandler/fitting/priors.py`
- Test: `tests/unit/fitting/test_priors_dataclass.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/fitting/test_priors_dataclass.py`:

```python
"""Tests for the SkewNormalPriors dataclass."""

from __future__ import annotations

import numpy as np
import pytest

from chromhandler.fitting.priors import SkewNormalPriors


def _single_priors() -> SkewNormalPriors:
    return SkewNormalPriors(
        n_components=1,
        mu_left_loc=2.7,
        mu_left_scale=0.005,
        mu_left_low=2.55,
        mu_left_high=2.85,
        log_sigma_left_loc=np.log(0.03),
        log_sigma_left_scale=0.1,
        log_sigma_left_low=np.log(0.005),
        log_sigma_left_high=np.log(0.05),
        gamma1_left_loc=0.2,
        gamma1_left_scale=0.05,
        log_A_left_loc_per_trace=np.array([np.log(100.0), np.log(80.0)]),
        log_A_left_scale=0.1,
        Delta_low=None,
        Delta_high=None,
        log_sigma_right_loc=None,
        log_sigma_right_scale=None,
        log_sigma_right_low=None,
        log_sigma_right_high=None,
        gamma1_right_loc=None,
        gamma1_right_scale=None,
        log_A_right_loc_per_trace=None,
        log_A_right_scale=None,
    )


def test_single_priors_constructs() -> None:
    p = _single_priors()
    assert p.n_components == 1
    assert p.Delta_low is None


def test_single_priors_rejects_right_fields() -> None:
    with pytest.raises(ValueError, match="right.*None"):
        SkewNormalPriors(
            n_components=1,
            mu_left_loc=2.7,
            mu_left_scale=0.005,
            mu_left_low=2.55,
            mu_left_high=2.85,
            log_sigma_left_loc=np.log(0.03),
            log_sigma_left_scale=0.1,
            log_sigma_left_low=np.log(0.005),
            log_sigma_left_high=np.log(0.05),
            gamma1_left_loc=0.2,
            gamma1_left_scale=0.05,
            log_A_left_loc_per_trace=np.array([np.log(100.0)]),
            log_A_left_scale=0.1,
            Delta_low=0.01,  # invalid: single peaks have no Delta
            Delta_high=0.1,
            log_sigma_right_loc=None,
            log_sigma_right_scale=None,
            log_sigma_right_low=None,
            log_sigma_right_high=None,
            gamma1_right_loc=None,
            gamma1_right_scale=None,
            log_A_right_loc_per_trace=None,
            log_A_right_scale=None,
        )


def test_doublet_priors_requires_right_fields() -> None:
    with pytest.raises(ValueError, match="doublet.*required"):
        SkewNormalPriors(
            n_components=2,
            mu_left_loc=2.7,
            mu_left_scale=0.005,
            mu_left_low=2.55,
            mu_left_high=2.85,
            log_sigma_left_loc=np.log(0.03),
            log_sigma_left_scale=0.1,
            log_sigma_left_low=np.log(0.005),
            log_sigma_left_high=np.log(0.05),
            gamma1_left_loc=0.2,
            gamma1_left_scale=0.05,
            log_A_left_loc_per_trace=np.array([np.log(100.0)]),
            log_A_left_scale=0.1,
            Delta_low=None,  # invalid: doublet needs all right fields
            Delta_high=None,
            log_sigma_right_loc=None,
            log_sigma_right_scale=None,
            log_sigma_right_low=None,
            log_sigma_right_high=None,
            gamma1_right_loc=None,
            gamma1_right_scale=None,
            log_A_right_loc_per_trace=None,
            log_A_right_scale=None,
        )
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_priors_dataclass.py -v
```
Expected: `ImportError: cannot import name 'SkewNormalPriors' from 'chromhandler.fitting.priors'`.

- [ ] **Step 3: Implement the dataclass**

Create `chromhandler/fitting/priors.py`:

```python
"""Empirical prior construction for the skew-normal peak model.

This module is the bridge between the data-preparation layer (``PreparedDataset``,
shipped) and the NumPyro model (``model.py``, future). It converts per-trace
FWHM-based peak measurements into a :class:`SkewNormalPriors` per peak window.

Every field of :class:`SkewNormalPriors` parameterizes exactly one NumPyro
distribution in ``model.py``. See ``docs/superpowers/plans/2026-05-11-priors-module.md``
for the full distribution table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SkewNormalPriors:
    """Empirical priors for one peak window.

    Each field maps to a sampled site in the future ``model.py`` per the
    distribution table in the plan document. ``_left_*`` fields are always
    populated; ``_right_*`` and ``Delta_*`` are populated iff
    ``n_components == 2``.

    Attributes:
        n_components: 1 for single peaks, 2 for doublets.
        mu_left_loc: Mean of per-trace apex locations (minutes).
        mu_left_scale: Std of per-trace apex locations, floored at ``dt``.
        mu_left_low: Annotation window lower bound (minutes).
        mu_left_high: Annotation window upper bound (minutes).
        log_sigma_left_loc: Mean of per-trace ``log(sigma)`` where
            ``sigma = (HWHM_L + HWHM_R) / (2 * sqrt(2 * ln 2))``.
        log_sigma_left_scale: Std of per-trace ``log(sigma)``, floored at
            ``1 / sqrt(n_trace)``.
        log_sigma_left_low: ``log(8 * dt * FWHM_TO_SIGMA)`` where
            ``FWHM_TO_SIGMA = 1 / (2 * sqrt(2 * ln 2))`` — Nyquist-derived
            8-points-per-FWHM lower bound on sigma.
        log_sigma_left_high: ``log(window_width / 6)`` — upper bound so that
            ``+/-3 sigma`` of the peak fits within the annotation window.
        gamma1_left_loc: Mean of per-trace
            ``sn_asymmetry_to_gamma1(HWHM_R / HWHM_L)``.
        gamma1_left_scale: Std of per-trace ``gamma1`` estimates, floored at
            ``sqrt(6 / n_eff)`` (large-sample SE of sample skewness).
        log_A_left_loc_per_trace: ``[n_trace]`` per-trace ``log(area)`` where
            ``area = trapezoid(signal_baseline_subtracted, time)`` over the
            window.
        log_A_left_scale: ``log(1 + noise * sqrt(n_points_in_window) * dt
            / area_median)``, floored at ``1 / sqrt(n_trace)``.
        Delta_low: ``5 * dt`` for doublets, ``None`` for single peaks.
        Delta_high: ``window_width / 2`` for doublets, ``None`` for single.
        log_sigma_right_*: Same structure as ``log_sigma_left_*``, populated
            from the population of single-peak windows (or outer-HWHM fallback)
            for doublets. ``None`` for single peaks.
        gamma1_right_*: Same as ``log_sigma_right_*``.
        log_A_right_loc_per_trace: Per-trace areas of the non-dominant
            component, computed via outer-HWHM Gaussian residual with spatial
            assignment. ``None`` for single peaks.
        log_A_right_scale: Same scale logic as ``log_A_left_scale``.
    """

    n_components: int

    # mu — TruncatedNormal(loc, scale, low, high)
    mu_left_loc: float
    mu_left_scale: float
    mu_left_low: float
    mu_left_high: float

    # log_sigma — TruncatedNormal(loc, scale, low, high)
    log_sigma_left_loc: float
    log_sigma_left_scale: float
    log_sigma_left_low: float
    log_sigma_left_high: float

    # gamma1 — TruncatedNormal(loc, scale, low=-0.99*GAMMA1_MAX, high=+0.99*GAMMA1_MAX)
    gamma1_left_loc: float
    gamma1_left_scale: float

    # log_A — Normal(loc_per_trace[trace], scale)
    log_A_left_loc_per_trace: NDArray[np.float64]
    log_A_left_scale: float

    # Doublet-only fields
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
            self.Delta_low,
            self.Delta_high,
            self.log_sigma_right_loc,
            self.log_sigma_right_scale,
            self.log_sigma_right_low,
            self.log_sigma_right_high,
            self.gamma1_right_loc,
            self.gamma1_right_scale,
            self.log_A_right_loc_per_trace,
            self.log_A_right_scale,
        )
        if self.n_components == 1:
            if any(f is not None for f in right_fields):
                raise ValueError(
                    "Single-component priors require all right-component fields "
                    "(Delta_*, *_right_*) to be None."
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

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/unit/fitting/test_priors_dataclass.py -v
uv run ruff check chromhandler/fitting/priors.py tests/unit/fitting/test_priors_dataclass.py
uv run pyright chromhandler/fitting/priors.py tests/unit/fitting/test_priors_dataclass.py
```
Expected: 3 tests pass, ruff clean, pyright clean.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/priors.py tests/unit/fitting/test_priors_dataclass.py
git commit -m "feat(priors): SkewNormalPriors dataclass with explicit distribution-bound fields"
```

---

## Task 2: `compute_single_window_features` — per-trace FWHM-based extraction

**What it computes:** For one trace and one peak window, returns `(mu, sigma, gamma1, area)` via:
- `mu` = smoothed-signal argmax inside the window
- `HWHM_L`, `HWHM_R` = linearly-interpolated half-max crossings walking outward from apex
- `sigma = (HWHM_L + HWHM_R) / (2 · √(2 ln 2))`
- `gamma1 = sn_asymmetry_to_gamma1(HWHM_R / HWHM_L)` (table lookup from `skew_normal.py`)
- `area = trapezoid(signal_baseline_subtracted, time)` over the window

**Why this feeds which prior:** Aggregating `mu` across traces → `mu_left_loc, mu_left_scale`. Aggregating `log(sigma)` → `log_sigma_left_loc, log_sigma_left_scale`. Aggregating `gamma1` → `gamma1_left_loc, gamma1_left_scale`. `log(area)` per trace → `log_A_left_loc_per_trace`.

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


def _synth_sn_trace(
    mu: float, sigma: float, gamma1: float, area: float, dt: float = 0.001
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize a skew-normal peak on a dense grid (no baseline, no noise)."""
    t = np.arange(mu - 1.0, mu + 1.0, dt)
    xi, omega, alpha = (float(x) for x in cp_to_dp(mu, sigma, gamma1))
    pdf = skewnorm.pdf(t, alpha, loc=xi, scale=omega)
    return t, area * pdf


def test_features_dataclass_fields() -> None:
    f = WindowFeatures(mu=2.7, sigma=0.03, gamma1=0.2, area=5.0)
    assert f.mu == 2.7
    assert f.sigma == 0.03
    assert f.gamma1 == 0.2
    assert f.area == 5.0


def test_recovers_symmetric_peak() -> None:
    t, s = _synth_sn_trace(mu=2.7, sigma=0.03, gamma1=0.0, area=5.0)
    f = compute_single_window_features(t, s, window_low=2.55, window_high=2.85)
    assert abs(f.mu - 2.7) < 0.005
    assert abs(f.sigma - 0.03) / 0.03 < 0.05
    assert abs(f.gamma1) < 0.05
    assert abs(f.area - 5.0) / 5.0 < 0.02


def test_recovers_positively_skewed_peak() -> None:
    t, s = _synth_sn_trace(mu=2.7, sigma=0.03, gamma1=0.5, area=5.0)
    f = compute_single_window_features(t, s, window_low=2.55, window_high=2.85)
    assert abs(f.mu - 2.7) < 0.01
    assert abs(f.sigma - 0.03) / 0.03 < 0.10
    assert abs(f.gamma1 - 0.5) < 0.10
    assert abs(f.area - 5.0) / 5.0 < 0.05


def test_recovers_negatively_skewed_peak() -> None:
    t, s = _synth_sn_trace(mu=2.7, sigma=0.03, gamma1=-0.5, area=5.0)
    f = compute_single_window_features(t, s, window_low=2.55, window_high=2.85)
    assert abs(f.gamma1 + 0.5) < 0.10


def test_low_snr_average_is_unbiased() -> None:
    rng = np.random.default_rng(0)
    estimates = []
    for _ in range(100):
        t, s = _synth_sn_trace(mu=2.7, sigma=0.03, gamma1=0.3, area=5.0)
        noise = rng.normal(0.0, np.max(s) / 5.0, size=s.shape)  # S/N = 5
        f = compute_single_window_features(t, s + noise, 2.55, 2.85)
        estimates.append(f.gamma1)
    mean_gamma1 = float(np.mean(estimates))
    assert abs(mean_gamma1 - 0.3) < 0.05
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_priors_features.py -v
```
Expected: ImportError for `WindowFeatures` and `compute_single_window_features`.

- [ ] **Step 3: Implement `WindowFeatures` and `compute_single_window_features`**

Append to `chromhandler/fitting/priors.py`:

```python
from scipy.signal import savgol_filter

from chromhandler.fitting.skew_normal import sn_asymmetry_to_gamma1

_FWHM_TO_SIGMA: float = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


@dataclass(frozen=True)
class WindowFeatures:
    """Per-trace, per-window FWHM-based features for a single peak.

    Attributes:
        mu: Apex location (minutes), from smoothed argmax inside the window.
        sigma: ``(HWHM_L + HWHM_R) * FWHM_TO_SIGMA``, the symmetric-equivalent
            scale.
        gamma1: Skewness, from ``sn_asymmetry_to_gamma1(HWHM_R / HWHM_L)``.
        area: ``trapezoid(signal, time)`` over the window.
    """

    mu: float
    sigma: float
    gamma1: float
    area: float


def _interp_threshold_crossing(
    t: NDArray[np.float64],
    s: NDArray[np.float64],
    apex_idx: int,
    threshold: float,
    direction: int,
) -> float | None:
    """Walk from ``apex_idx`` in ``direction`` (``-1`` left, ``+1`` right)
    until ``s`` falls below ``threshold``; linearly interpolate the crossing
    time.

    Returns ``None`` if the signal never falls below the threshold within
    the array bounds.
    """
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
        time: 1-D time array (minutes), monotone increasing.
        signal_baseline_subtracted: 1-D baseline-subtracted signal, same shape
            as ``time``. NaNs are dropped.
        window_low: Lower bound of the peak window (minutes, inclusive).
        window_high: Upper bound of the peak window (minutes, inclusive).
        smoothing_window: Savitzky-Golay window length used for apex finding.
            Must be odd and at least 5. The smoothed signal is used only to
            locate the apex; HWHM brackets walk on the raw signal.

    Returns:
        :class:`WindowFeatures` with ``(mu, sigma, gamma1, area)``.

    Raises:
        ValueError: If fewer than ``smoothing_window`` points fall inside
            the window after NaN removal.
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
    t_left = _interp_threshold_crossing(t, s, apex_idx, half, direction=-1)
    t_right = _interp_threshold_crossing(t, s, apex_idx, half, direction=+1)
    if t_left is None or t_right is None:
        # Window too tight for one side — fall back to symmetric HWHM
        # from whichever side resolved.
        if t_left is None and t_right is not None:
            hwhm_r = t_right - mu
            hwhm_l = hwhm_r
        elif t_right is None and t_left is not None:
            hwhm_l = mu - t_left
            hwhm_r = hwhm_l
        else:
            raise ValueError(
                f"Could not bracket half-max on either side in window "
                f"[{window_low}, {window_high}]; window may be too narrow."
            )
    else:
        hwhm_l = mu - t_left
        hwhm_r = t_right - mu

    sigma = (hwhm_l + hwhm_r) * _FWHM_TO_SIGMA
    ratio = hwhm_r / hwhm_l if hwhm_l > 0 else 1.0
    gamma1 = float(sn_asymmetry_to_gamma1(ratio))
    area = float(np.trapezoid(s, t))

    return WindowFeatures(mu=mu, sigma=sigma, gamma1=gamma1, area=area)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/unit/fitting/test_priors_features.py -v
uv run ruff check chromhandler/fitting/priors.py tests/unit/fitting/test_priors_features.py
uv run pyright chromhandler/fitting/priors.py tests/unit/fitting/test_priors_features.py
```
Expected: 5 tests pass, ruff clean, pyright clean.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/priors.py tests/unit/fitting/test_priors_features.py
git commit -m "feat(priors): compute_single_window_features — FWHM hybrid extraction"
```

---

## Task 3: `detect_dominant_apex` — for doublets

**What it computes:** Smoothed-argmax apex location and height. Used by the doublet path (Task 4) to find the dominant component before splitting.

**Why this feeds which prior:** Per-trace `(apex_loc, apex_height)` → aggregated to `mu_left_loc, mu_left_scale` for doublets (the "left" anchor refers to the dominant analyte position when `artefact_side="right"`, and to the artefact-aware analyte when `artefact_side="left"` — orientation is handled in Task 6).

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


def _single_peak(mu: float = 2.7) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(2.5, 2.9, 0.001)
    xi, omega, alpha = (float(x) for x in cp_to_dp(mu, 0.03, 0.0))
    s = skewnorm.pdf(t, alpha, loc=xi, scale=omega)
    return t, s


def test_dominant_apex_on_single_peak() -> None:
    t, s = _single_peak(mu=2.7)
    apex_loc, apex_height = detect_dominant_apex(t, s, 2.5, 2.9)
    assert abs(apex_loc - 2.7) < 0.005
    assert apex_height > 0.0


def test_dominant_apex_picks_taller_of_two() -> None:
    t = np.arange(2.5, 2.9, 0.001)
    xi1, om1, a1 = (float(x) for x in cp_to_dp(2.65, 0.02, 0.0))
    xi2, om2, a2 = (float(x) for x in cp_to_dp(2.75, 0.02, 0.0))
    s = 1.0 * skewnorm.pdf(t, a1, loc=xi1, scale=om1) + 0.3 * skewnorm.pdf(
        t, a2, loc=xi2, scale=om2
    )
    apex_loc, _ = detect_dominant_apex(t, s, 2.5, 2.9)
    assert abs(apex_loc - 2.65) < 0.01


def test_dominant_apex_on_noise_only_returns_argmax() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(2.5, 2.9, 0.001)
    s = rng.normal(0.0, 1.0, size=t.shape)
    apex_loc, apex_height = detect_dominant_apex(t, s, 2.5, 2.9)
    assert 2.5 <= apex_loc <= 2.9
    assert np.isfinite(apex_height)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_priors_apex.py -v
```
Expected: ImportError for `detect_dominant_apex`.

- [ ] **Step 3: Implement `detect_dominant_apex`**

Append to `chromhandler/fitting/priors.py`:

```python
def detect_dominant_apex(
    time: NDArray[np.float64],
    signal_baseline_subtracted: NDArray[np.float64],
    window_low: float,
    window_high: float,
    smoothing_window: int = 5,
) -> tuple[float, float]:
    """Find the dominant apex (location and height) inside a window.

    Args:
        time: 1-D time array (minutes).
        signal_baseline_subtracted: Baseline-subtracted signal.
        window_low: Lower window bound (minutes, inclusive).
        window_high: Upper window bound (minutes, inclusive).
        smoothing_window: Savitzky-Golay window length for apex finding.
            Must be odd and at least 5.

    Returns:
        ``(apex_loc, apex_height)`` from the smoothed signal's argmax.

    Raises:
        ValueError: If fewer than ``smoothing_window`` points fall inside
            the window after NaN removal.
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
    idx = int(np.argmax(s_smooth))
    return float(t[idx]), float(s_smooth[idx])
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/unit/fitting/test_priors_apex.py -v
uv run ruff check chromhandler/fitting/priors.py tests/unit/fitting/test_priors_apex.py
uv run pyright chromhandler/fitting/priors.py tests/unit/fitting/test_priors_apex.py
```
Expected: 3 tests pass, clean.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/priors.py tests/unit/fitting/test_priors_apex.py
git commit -m "feat(priors): detect_dominant_apex via Savitzky-Golay smoothed argmax"
```

---

## Task 4: `estimate_outer_hwhm` + `split_doublet_areas`

**What it computes:** For a doublet window, splits the total trapezoid area into `(A_left, A_right)` using the spatial-assignment rule from spec §6.1:

1. Detect dominant apex.
2. Classify dominant spatial side: `"left" if apex_loc < (window_low+window_high)/2 else "right"`.
3. Walk the outer side (away from window centre) until half-max → outer HWHM.
4. Reconstruct dominant component as Gaussian: `A_dominant = apex_height · sigma_dominant · √(2π)` with `sigma_dominant = outer_HWHM / √(2 ln 2)`.
5. Residual to other side: `A_other = max(A_total - A_dominant, A_floor)` with `A_floor = noise · √(n_points) · dt`.
6. Floor both at `A_floor`.

**Why this feeds which prior:** Per-trace `(A_left, A_right)` → `log_A_left_loc_per_trace`, `log_A_right_loc_per_trace`. Floor prevents log(0) when one component is absent.

**Files:**
- Modify: `chromhandler/fitting/priors.py`
- Test: `tests/unit/fitting/test_priors_split.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/fitting/test_priors_split.py`:

```python
"""Tests for outer-HWHM doublet area splitting."""

from __future__ import annotations

import numpy as np
from scipy.stats import skewnorm

from chromhandler.fitting.priors import split_doublet_areas


def _two_gaussian_trace(
    A_main: float, A_artefact: float, dt: float = 0.001
) -> tuple[np.ndarray, np.ndarray]:
    """Analyte at 2.65 (left), artefact at 2.75 (right). Window is [2.55, 2.85]."""
    t = np.arange(2.5, 2.9, dt)
    main = A_main * skewnorm.pdf(t, 0.0, loc=2.65, scale=0.02)
    art = A_artefact * skewnorm.pdf(t, 0.0, loc=2.75, scale=0.02)
    return t, main + art


def test_left_dominant_split() -> None:
    t, s = _two_gaussian_trace(A_main=5.0, A_artefact=1.0)
    A_left, A_right = split_doublet_areas(
        t, s, window_low=2.55, window_high=2.85,
        window_midpoint=2.70, noise_per_trace=0.01, dt=0.001,
    )
    assert abs(A_left - 5.0) / 5.0 < 0.20
    assert abs(A_right - 1.0) / 1.0 < 0.40


def test_right_dominant_split() -> None:
    t, s = _two_gaussian_trace(A_main=1.0, A_artefact=5.0)
    A_left, A_right = split_doublet_areas(
        t, s, 2.55, 2.85, 2.70, 0.01, 0.001
    )
    assert abs(A_left - 1.0) / 1.0 < 0.40
    assert abs(A_right - 5.0) / 5.0 < 0.20


def test_absent_left_collapses_to_floor() -> None:
    t, s = _two_gaussian_trace(A_main=0.0, A_artefact=5.0)
    A_left, A_right = split_doublet_areas(
        t, s, 2.55, 2.85, 2.70, 0.01, 0.001
    )
    A_floor = 0.01 * np.sqrt((t >= 2.55).sum() & (t <= 2.85).sum()) * 0.001
    assert A_left <= max(A_floor * 5, 0.1)  # near floor, not a ghost peak
    assert abs(A_right - 5.0) / 5.0 < 0.20


def test_floor_never_returns_zero() -> None:
    t = np.arange(2.5, 2.9, 0.001)
    s = np.zeros_like(t)
    A_left, A_right = split_doublet_areas(
        t, s, 2.55, 2.85, 2.70, 0.01, 0.001
    )
    assert A_left > 0.0
    assert A_right > 0.0
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_priors_split.py -v
```
Expected: ImportError for `split_doublet_areas`.

- [ ] **Step 3: Implement helpers**

Append to `chromhandler/fitting/priors.py`:

```python
def _outer_hwhm(
    t: NDArray[np.float64],
    s: NDArray[np.float64],
    apex_idx: int,
    apex_height: float,
    outer_direction: int,
) -> float:
    """Half-width on the outer side (away from window centre) of the dominant
    apex. Falls back to ``2 * dt`` if half-max never resolved within the array.
    """
    half = apex_height / 2.0
    crossing = _interp_threshold_crossing(t, s, apex_idx, half, outer_direction)
    if crossing is None:
        dt = float(np.median(np.diff(t))) if t.size > 1 else 1e-6
        return 2.0 * dt
    apex_t = float(t[apex_idx])
    return abs(crossing - apex_t)


def split_doublet_areas(
    time: NDArray[np.float64],
    signal_baseline_subtracted: NDArray[np.float64],
    window_low: float,
    window_high: float,
    window_midpoint: float,
    noise_per_trace: float,
    dt: float,
    smoothing_window: int = 5,
) -> tuple[float, float]:
    """Split the total window area into ``(A_left, A_right)`` via outer-HWHM
    Gaussian residual with spatial assignment.

    Args:
        time: 1-D time array.
        signal_baseline_subtracted: Baseline-subtracted signal.
        window_low: Window lower bound (inclusive).
        window_high: Window upper bound (inclusive).
        window_midpoint: Spatial midpoint used to classify the dominant side.
            Typically ``(window_low + window_high) / 2``.
        noise_per_trace: Per-trace noise standard deviation (signal units),
            from :class:`PreparedDataset`.
        dt: Sampling interval (minutes).
        smoothing_window: Savitzky-Golay window for apex detection.

    Returns:
        ``(A_left, A_right)``. Both are floored at
        ``A_floor = noise * sqrt(n_points_in_window) * dt`` to prevent
        log(0) downstream.
    """
    mask = (time >= window_low) & (time <= window_high) & np.isfinite(
        signal_baseline_subtracted
    )
    t = np.asarray(time[mask], dtype=np.float64)
    s = np.asarray(signal_baseline_subtracted[mask], dtype=np.float64)
    n_points = s.size
    A_floor = float(noise_per_trace) * float(np.sqrt(n_points)) * float(dt)

    if n_points < smoothing_window:
        # Degenerate: split A_total naively, fall back to floor.
        A_total = max(float(np.trapezoid(s, t)) if n_points >= 2 else 0.0, 2.0 * A_floor)
        return max(A_total / 2.0, A_floor), max(A_total / 2.0, A_floor)

    A_total = float(np.trapezoid(s, t))
    polyorder = min(3, smoothing_window - 1)
    s_smooth = savgol_filter(s, smoothing_window, polyorder)
    apex_idx = int(np.argmax(s_smooth))
    apex_loc = float(t[apex_idx])
    apex_height = float(s_smooth[apex_idx])

    dominant_side = "left" if apex_loc < window_midpoint else "right"
    outer_direction = -1 if dominant_side == "left" else +1
    hwhm_outer = _outer_hwhm(t, s, apex_idx, apex_height, outer_direction)
    sigma_dom = hwhm_outer / np.sqrt(2.0 * np.log(2.0))
    A_dominant = max(apex_height * sigma_dom * np.sqrt(2.0 * np.pi), A_floor)
    A_other = max(A_total - A_dominant, A_floor)

    if dominant_side == "left":
        return max(A_dominant, A_floor), max(A_other, A_floor)
    return max(A_other, A_floor), max(A_dominant, A_floor)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/unit/fitting/test_priors_split.py -v
uv run ruff check chromhandler/fitting/priors.py tests/unit/fitting/test_priors_split.py
uv run pyright chromhandler/fitting/priors.py tests/unit/fitting/test_priors_split.py
```
Expected: 4 tests pass, clean.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/priors.py tests/unit/fitting/test_priors_split.py
git commit -m "feat(priors): split_doublet_areas via outer-HWHM Gaussian residual"
```

---

## Task 5: `aggregate_single_peak_priors` — population aggregation with floors

**What it computes:** Given a list of per-trace `WindowFeatures` for one single-peak window, produces a `SkewNormalPriors` with:
- `mu_left_loc = mean(per_trace.mu)`, `mu_left_scale = max(std(per_trace.mu), dt)`
- `mu_left_low, mu_left_high = window_low, window_high` (annotation bounds)
- `log_sigma_left_loc = mean(log(per_trace.sigma))`, `log_sigma_left_scale = max(std(log(per_trace.sigma)), 1/√n_trace)`
- `log_sigma_left_low = log(8 · dt · FWHM_TO_SIGMA)` (8-points-per-FWHM Nyquist floor)
- `log_sigma_left_high = log(window_width / 6)` (±3σ fits in window)
- `gamma1_left_loc = mean(per_trace.gamma1)`, `gamma1_left_scale = max(std(per_trace.gamma1), √(6/n_eff))`
- `log_A_left_loc_per_trace[trace] = log(per_trace[trace].area)`
- `log_A_left_scale = max(noise-propagated CV, 1/√n_trace)`

**Why this feeds which prior:** Direct one-to-one with the distribution table at the top of the document.

**Files:**
- Modify: `chromhandler/fitting/priors.py`
- Test: `tests/unit/fitting/test_priors_aggregate.py` (created in this task, extended in Task 6)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/fitting/test_priors_aggregate.py`:

```python
"""Tests for population-level prior aggregation."""

from __future__ import annotations

import numpy as np

from chromhandler.fitting.priors import (
    WindowFeatures,
    aggregate_single_peak_priors,
)
from chromhandler.fitting.skew_normal import GAMMA1_MAX


def _features(mus, sigmas, gamma1s, areas):
    return [
        WindowFeatures(mu=mu, sigma=sigma, gamma1=gamma1, area=area)
        for mu, sigma, gamma1, area in zip(mus, sigmas, gamma1s, areas, strict=True)
    ]


def test_aggregate_single_peak_matches_population_stats() -> None:
    rng = np.random.default_rng(0)
    n = 50
    mus = rng.normal(2.70, 0.002, size=n).tolist()
    sigmas = np.exp(rng.normal(np.log(0.03), 0.05, size=n)).tolist()
    gamma1s = rng.normal(0.2, 0.05, size=n).tolist()
    areas = np.exp(rng.normal(np.log(100.0), 0.1, size=n)).tolist()
    feats = _features(mus, sigmas, gamma1s, areas)

    p = aggregate_single_peak_priors(
        per_trace_features=feats,
        window_low=2.55,
        window_high=2.85,
        dt=0.001,
        noise_per_trace=np.full(n, 1.0),
        n_window_points=300,
    )

    assert abs(p.mu_left_loc - 2.70) < 0.001
    assert 0.0005 < p.mu_left_scale < 0.005
    assert p.mu_left_low == 2.55
    assert p.mu_left_high == 2.85
    assert abs(p.log_sigma_left_loc - np.log(0.03)) < 0.02
    assert abs(p.gamma1_left_loc - 0.2) < 0.02
    assert p.log_A_left_loc_per_trace.shape == (n,)
    assert p.n_components == 1
    assert p.Delta_low is None


def test_single_trace_collapses_to_floor() -> None:
    feats = _features([2.70], [0.03], [0.2], [100.0])
    p = aggregate_single_peak_priors(
        per_trace_features=feats,
        window_low=2.55,
        window_high=2.85,
        dt=0.001,
        noise_per_trace=np.array([1.0]),
        n_window_points=300,
    )
    assert p.mu_left_scale == 0.001  # mu floor = dt
    # log_sigma floor = 1/sqrt(1) = 1.0
    assert p.log_sigma_left_scale >= 1.0 - 1e-12
    # gamma1 floor = sqrt(6/1) ~= 2.45, but clipped at GAMMA1_MAX-derived bound
    assert p.gamma1_left_scale > 1.0
    assert p.log_A_left_scale >= 1.0 - 1e-12


def test_log_sigma_bounds_from_geometry() -> None:
    feats = _features([2.70] * 5, [0.03] * 5, [0.0] * 5, [100.0] * 5)
    p = aggregate_single_peak_priors(
        per_trace_features=feats,
        window_low=2.55,
        window_high=2.85,
        dt=0.001,
        noise_per_trace=np.full(5, 1.0),
        n_window_points=300,
    )
    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    expected_low = np.log(8.0 * 0.001 * fwhm_to_sigma)
    expected_high = np.log((2.85 - 2.55) / 6.0)
    assert abs(p.log_sigma_left_low - expected_low) < 1e-9
    assert abs(p.log_sigma_left_high - expected_high) < 1e-9


def test_gamma1_scale_capped_by_max() -> None:
    feats = _features([2.70], [0.03], [0.0], [100.0])
    p = aggregate_single_peak_priors(
        per_trace_features=feats,
        window_low=2.55,
        window_high=2.85,
        dt=0.001,
        noise_per_trace=np.array([1.0]),
        n_window_points=300,
    )
    # Scale floor sqrt(6/1) = 2.45, but the truncation interval is only
    # ~2*GAMMA1_MAX = ~1.99 wide; cap the scale at GAMMA1_MAX so the prior
    # is not pathologically wider than its support.
    assert p.gamma1_left_scale <= GAMMA1_MAX + 1e-9
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_priors_aggregate.py -v
```
Expected: ImportError for `aggregate_single_peak_priors`.

- [ ] **Step 3: Implement `aggregate_single_peak_priors`**

Append to `chromhandler/fitting/priors.py`:

```python
from chromhandler.fitting.skew_normal import GAMMA1_MAX

_GAMMA1_BOUND: float = 0.99 * GAMMA1_MAX


def _log_sigma_bounds(window_low: float, window_high: float, dt: float) -> tuple[float, float]:
    """Geometric bounds for ``log_sigma``:
    - low: 8-points-per-FWHM Nyquist floor.
    - high: ``+/-3 sigma`` of the peak must fit inside the window.
    """
    sigma_low = 8.0 * dt * _FWHM_TO_SIGMA
    sigma_high = (window_high - window_low) / 6.0
    return float(np.log(sigma_low)), float(np.log(sigma_high))


def _log_A_scale_from_noise(
    areas: NDArray[np.float64],
    noise_per_trace: NDArray[np.float64],
    n_window_points: int,
    dt: float,
    n_trace: int,
) -> float:
    """Per-area noise CV propagated to log-space, floored at pooling precision.

    The trapezoid-area uncertainty for a window of ``n`` samples with per-sample
    noise std ``s`` is approximately ``s * sqrt(n) * dt`` (uncorrelated sum).
    Divided by area gives a CV; converted to log-space via ``log(1+CV)``.
    """
    median_noise = float(np.median(noise_per_trace))
    sigma_area = median_noise * float(np.sqrt(n_window_points)) * float(dt)
    median_area = float(np.median(np.abs(areas))) if areas.size > 0 else 0.0
    if median_area <= 0.0:
        cv = 1.0
    else:
        cv = sigma_area / median_area
    log_scale = float(np.log1p(cv))
    return max(log_scale, 1.0 / float(np.sqrt(max(n_trace, 1))))


def aggregate_single_peak_priors(
    per_trace_features: list[WindowFeatures],
    window_low: float,
    window_high: float,
    dt: float,
    noise_per_trace: NDArray[np.float64],
    n_window_points: int,
) -> SkewNormalPriors:
    """Aggregate per-trace single-peak features into a :class:`SkewNormalPriors`.

    Args:
        per_trace_features: One :class:`WindowFeatures` per trace.
        window_low: Annotation window lower bound (minutes).
        window_high: Annotation window upper bound (minutes).
        dt: Sampling interval (minutes).
        noise_per_trace: Per-trace noise std, shape ``[n_trace]``.
        n_window_points: Number of valid (non-NaN, in-window) samples per trace.
            Used for log_A scale propagation. Use the median count if it varies
            across traces.

    Returns:
        :class:`SkewNormalPriors` with ``n_components=1`` and all
        ``_right_*`` / ``Delta_*`` fields set to ``None``.
    """
    n = len(per_trace_features)
    if n == 0:
        raise ValueError("per_trace_features must be non-empty.")

    mus = np.asarray([f.mu for f in per_trace_features], dtype=np.float64)
    sigmas = np.asarray([f.sigma for f in per_trace_features], dtype=np.float64)
    gamma1s = np.asarray([f.gamma1 for f in per_trace_features], dtype=np.float64)
    areas = np.asarray([f.area for f in per_trace_features], dtype=np.float64)

    log_sigmas = np.log(np.clip(sigmas, 1e-9, None))
    log_areas = np.log(np.clip(np.abs(areas), 1e-9, None))

    sqrt_n = float(np.sqrt(max(n, 1)))
    mu_loc = float(np.mean(mus))
    mu_scale = float(max(np.std(mus, ddof=0), dt))
    log_sigma_loc = float(np.mean(log_sigmas))
    log_sigma_scale = float(max(np.std(log_sigmas, ddof=0), 1.0 / sqrt_n))
    gamma1_loc = float(np.mean(gamma1s))
    gamma1_scale = float(
        min(max(np.std(gamma1s, ddof=0), np.sqrt(6.0 / max(n, 1))), GAMMA1_MAX)
    )

    log_sigma_low, log_sigma_high = _log_sigma_bounds(window_low, window_high, dt)
    log_A_scale = _log_A_scale_from_noise(areas, noise_per_trace, n_window_points, dt, n)

    return SkewNormalPriors(
        n_components=1,
        mu_left_loc=mu_loc,
        mu_left_scale=mu_scale,
        mu_left_low=window_low,
        mu_left_high=window_high,
        log_sigma_left_loc=log_sigma_loc,
        log_sigma_left_scale=log_sigma_scale,
        log_sigma_left_low=log_sigma_low,
        log_sigma_left_high=log_sigma_high,
        gamma1_left_loc=gamma1_loc,
        gamma1_left_scale=gamma1_scale,
        log_A_left_loc_per_trace=log_areas,
        log_A_left_scale=log_A_scale,
        Delta_low=None,
        Delta_high=None,
        log_sigma_right_loc=None,
        log_sigma_right_scale=None,
        log_sigma_right_low=None,
        log_sigma_right_high=None,
        gamma1_right_loc=None,
        gamma1_right_scale=None,
        log_A_right_loc_per_trace=None,
        log_A_right_scale=None,
    )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/unit/fitting/test_priors_aggregate.py -v
uv run ruff check chromhandler/fitting/priors.py tests/unit/fitting/test_priors_aggregate.py
uv run pyright chromhandler/fitting/priors.py tests/unit/fitting/test_priors_aggregate.py
```
Expected: 4 tests pass, clean.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/priors.py tests/unit/fitting/test_priors_aggregate.py
git commit -m "feat(priors): aggregate_single_peak_priors with geometric bounds and principled floors"
```

---

## Task 6: `aggregate_doublet_priors` — borrows shape from population

**What it computes:** Given per-trace `(apex_loc, apex_height)` and per-trace `(A_left, A_right)` for a doublet window, plus *shared shape priors* `(log_sigma_loc, log_sigma_scale, gamma1_loc, gamma1_scale)` drawn from the population of single-peak windows, returns a `SkewNormalPriors` with `n_components=2`.

Key choices:
- `mu_left_loc, mu_left_scale` = mean/std of dominant apex locations.
- `mu_left_low, mu_left_high` = annotation bounds.
- `log_sigma_left/right_*` and `gamma1_left/right_*` are **identical** — both come from the shared population shape priors. Identifiability is enforced by Δ + area asymmetry, not by differing shape priors (spec §5).
- `log_sigma_left/right_low/high` = geometric bounds from this window.
- `Delta_low = 5·dt`, `Delta_high = (window_high - window_low) / 2`.
- `log_A_left/right_loc_per_trace` = per-trace `log(A_left)` / `log(A_right)`.

**Why this feeds which prior:** Distribution table row by row.

**Files:**
- Modify: `chromhandler/fitting/priors.py`
- Test: `tests/unit/fitting/test_priors_aggregate.py` (extend)

- [ ] **Step 1: Add failing tests**

Append to `tests/unit/fitting/test_priors_aggregate.py`:

```python
from chromhandler.fitting.priors import aggregate_doublet_priors


def test_aggregate_doublet_basic() -> None:
    n = 5
    rng = np.random.default_rng(0)
    apex = [(2.70 + rng.normal(0.0, 0.002), 1.0) for _ in range(n)]
    areas = [(5.0, 1.0) for _ in range(n)]
    shared_shape = (np.log(0.03), 0.05, 0.2, 0.04)
    p = aggregate_doublet_priors(
        per_trace_dominant_apex=apex,
        per_trace_areas=areas,
        shared_shape_priors=shared_shape,
        window_low=2.55,
        window_high=2.85,
        noise_per_trace=np.full(n, 1.0),
        dt=0.001,
        n_window_points=300,
    )
    assert p.n_components == 2
    assert p.Delta_low == 5 * 0.001
    assert p.Delta_high == (2.85 - 2.55) / 2.0
    # Shape priors identical between left and right
    assert p.log_sigma_left_loc == p.log_sigma_right_loc
    assert p.gamma1_left_loc == p.gamma1_right_loc
    assert p.log_sigma_left_scale == p.log_sigma_right_scale
    assert p.gamma1_left_scale == p.gamma1_right_scale
    # Areas distinct per side
    assert p.log_A_left_loc_per_trace.shape == (n,)
    assert p.log_A_right_loc_per_trace is not None
    assert p.log_A_right_loc_per_trace.shape == (n,)
    assert float(p.log_A_left_loc_per_trace.mean()) > float(
        p.log_A_right_loc_per_trace.mean()
    )


def test_doublet_delta_bounds_independent_of_traces() -> None:
    n = 1
    p = aggregate_doublet_priors(
        per_trace_dominant_apex=[(2.70, 1.0)],
        per_trace_areas=[(5.0, 1.0)],
        shared_shape_priors=(np.log(0.03), 0.05, 0.2, 0.04),
        window_low=2.55,
        window_high=2.85,
        noise_per_trace=np.array([1.0]),
        dt=0.001,
        n_window_points=300,
    )
    assert p.Delta_low == 5 * 0.001
    assert p.Delta_high == 0.15
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_priors_aggregate.py -v
```
Expected: ImportError for `aggregate_doublet_priors`.

- [ ] **Step 3: Implement `aggregate_doublet_priors`**

Append to `chromhandler/fitting/priors.py`:

```python
def aggregate_doublet_priors(
    per_trace_dominant_apex: list[tuple[float, float]],
    per_trace_areas: list[tuple[float, float]],
    shared_shape_priors: tuple[float, float, float, float],
    window_low: float,
    window_high: float,
    noise_per_trace: NDArray[np.float64],
    dt: float,
    n_window_points: int,
) -> SkewNormalPriors:
    """Aggregate per-trace doublet measurements into a :class:`SkewNormalPriors`.

    Args:
        per_trace_dominant_apex: Per-trace ``(apex_loc, apex_height)`` from
            :func:`detect_dominant_apex`.
        per_trace_areas: Per-trace ``(A_left, A_right)`` from
            :func:`split_doublet_areas`.
        shared_shape_priors: Tuple
            ``(log_sigma_loc, log_sigma_scale, gamma1_loc, gamma1_scale)``
            drawn from the population of single-peak windows. Used for **both**
            left and right components — identifiability comes from Δ and area
            asymmetry per spec §5.
        window_low: Annotation lower bound (minutes).
        window_high: Annotation upper bound (minutes).
        noise_per_trace: Per-trace noise std, shape ``[n_trace]``.
        dt: Sampling interval (minutes).
        n_window_points: Median in-window sample count per trace.

    Returns:
        :class:`SkewNormalPriors` with ``n_components=2`` and all
        ``_right_*`` / ``Delta_*`` fields populated.
    """
    n = len(per_trace_dominant_apex)
    if n == 0 or n != len(per_trace_areas):
        raise ValueError(
            f"per_trace_dominant_apex (len={n}) and per_trace_areas "
            f"(len={len(per_trace_areas)}) must be non-empty and same length."
        )

    apex_locs = np.asarray([loc for loc, _ in per_trace_dominant_apex], dtype=np.float64)
    A_left = np.asarray([a[0] for a in per_trace_areas], dtype=np.float64)
    A_right = np.asarray([a[1] for a in per_trace_areas], dtype=np.float64)

    sqrt_n = float(np.sqrt(max(n, 1)))
    mu_loc = float(np.mean(apex_locs))
    mu_scale = float(max(np.std(apex_locs, ddof=0), dt))

    log_sigma_loc, log_sigma_scale, gamma1_loc, gamma1_scale = shared_shape_priors
    log_sigma_low, log_sigma_high = _log_sigma_bounds(window_low, window_high, dt)

    Delta_low = 5.0 * dt
    Delta_high = (window_high - window_low) / 2.0

    log_A_left = np.log(np.clip(A_left, 1e-9, None))
    log_A_right = np.log(np.clip(A_right, 1e-9, None))
    log_A_left_scale = _log_A_scale_from_noise(A_left, noise_per_trace, n_window_points, dt, n)
    log_A_right_scale = _log_A_scale_from_noise(A_right, noise_per_trace, n_window_points, dt, n)

    return SkewNormalPriors(
        n_components=2,
        mu_left_loc=mu_loc,
        mu_left_scale=mu_scale,
        mu_left_low=window_low,
        mu_left_high=window_high,
        log_sigma_left_loc=float(log_sigma_loc),
        log_sigma_left_scale=float(log_sigma_scale),
        log_sigma_left_low=log_sigma_low,
        log_sigma_left_high=log_sigma_high,
        gamma1_left_loc=float(gamma1_loc),
        gamma1_left_scale=float(gamma1_scale),
        log_A_left_loc_per_trace=log_A_left,
        log_A_left_scale=log_A_left_scale,
        Delta_low=Delta_low,
        Delta_high=Delta_high,
        log_sigma_right_loc=float(log_sigma_loc),
        log_sigma_right_scale=float(log_sigma_scale),
        log_sigma_right_low=log_sigma_low,
        log_sigma_right_high=log_sigma_high,
        gamma1_right_loc=float(gamma1_loc),
        gamma1_right_scale=float(gamma1_scale),
        log_A_right_loc_per_trace=log_A_right,
        log_A_right_scale=log_A_right_scale,
    )
```

Note: `sqrt_n` is intentionally unused here — `_log_A_scale_from_noise` carries the same floor internally. Keep the local for readability symmetry with `aggregate_single_peak_priors`; if ruff flags it, remove and rely on the helper's floor.

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/unit/fitting/test_priors_aggregate.py -v
uv run ruff check chromhandler/fitting/priors.py tests/unit/fitting/test_priors_aggregate.py
uv run pyright chromhandler/fitting/priors.py tests/unit/fitting/test_priors_aggregate.py
```
Expected: 6 tests pass, clean. If ruff flags unused `sqrt_n`, delete the line and re-run.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/priors.py tests/unit/fitting/test_priors_aggregate.py
git commit -m "feat(priors): aggregate_doublet_priors with shared shape and Uniform Delta"
```

---

## Task 7: `build_priors` — top-level orchestrator

**What it does:**
1. For each `single`-mode annotation, run `compute_single_window_features` per trace, then `aggregate_single_peak_priors`.
2. Compute *shared shape priors* from the population of single-peak features: `log_sigma_loc, log_sigma_scale, gamma1_loc, gamma1_scale` are the medians of the per-peak `aggregate_single_peak_priors` outputs across all single-peak annotations. Fallback if zero single-peaks exist: use outer-HWHM from doublet dominant apex (Task 4) to estimate per-trace shape.
3. For each doublet annotation (`artefact_doublet` or `free_doublet`), run `detect_dominant_apex` + `split_doublet_areas` per trace, then `aggregate_doublet_priors` with the shared shape.
4. Return `list[SkewNormalPriors]` in the same order as `dataset.peak_annotations`.

**Files:**
- Modify: `chromhandler/fitting/priors.py`
- Test: `tests/unit/fitting/test_priors_orchestrator.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/fitting/test_priors_orchestrator.py`:

```python
"""End-to-end orchestrator tests for build_priors."""

from __future__ import annotations

import numpy as np
from scipy.stats import skewnorm

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.prepared_dataset import prepare_dataset
from chromhandler.fitting.priors import build_priors
from chromhandler.fitting.skew_normal import cp_to_dp


def _synth_dataset(n_trace: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    times, signals = [], []
    for _ in range(n_trace):
        t = np.arange(2.5, 3.6, 0.001)
        # Ino single peak at 2.70
        mu_ino, sig_ino, g_ino = 2.70, 0.03, 0.2
        xi, om, al = (float(x) for x in cp_to_dp(mu_ino, sig_ino, g_ino))
        s_ino = 100.0 * skewnorm.pdf(t, al, loc=xi, scale=om)
        # SIH doublet at 3.00 (main) + 3.05 (artefact right)
        s_main = 80.0 * skewnorm.pdf(t, 0.0, loc=3.00, scale=0.025)
        s_art = 15.0 * skewnorm.pdf(t, 0.0, loc=3.05, scale=0.025)
        baseline = 10.0 + 0.5 * t
        noise = rng.normal(0.0, 1.0, size=t.shape)
        signals.append(s_ino + s_main + s_art + baseline + noise)
        times.append(t)

    peak_anns = [
        PeakAnnotation(molecule_id="Ino", rt_min=2.55, rt_max=2.85),
        PeakAnnotation(
            molecule_id="SIH",
            rt_min=2.90,
            rt_max=3.15,
            mode="artefact_doublet",
            artefact_side="right",
        ),
    ]
    base_anns = [
        BaselineAnnotation(rt_min=2.50, rt_max=2.52),
        BaselineAnnotation(rt_min=3.55, rt_max=3.57),
    ]
    return prepare_dataset(times, signals, peak_anns, base_anns)


def test_build_priors_returns_one_per_annotation() -> None:
    ds = _synth_dataset()
    priors = build_priors(ds)
    assert len(priors) == 2
    assert priors[0].n_components == 1
    assert priors[1].n_components == 2


def test_build_priors_single_recovers_mu() -> None:
    ds = _synth_dataset()
    priors = build_priors(ds)
    assert abs(priors[0].mu_left_loc - 2.70) < 0.01


def test_build_priors_doublet_uses_geometric_delta() -> None:
    ds = _synth_dataset()
    priors = build_priors(ds)
    p = priors[1]
    assert p.Delta_low == 5 * ds.dt_global
    assert p.Delta_high == (3.15 - 2.90) / 2.0


def test_build_priors_doublet_shape_borrowed_from_singles() -> None:
    ds = _synth_dataset()
    priors = build_priors(ds)
    # Doublet shape priors must equal the single-peak aggregated shape
    # (only one single-peak in this dataset, so median == that single's value).
    assert priors[1].log_sigma_left_loc == priors[0].log_sigma_left_loc
    assert priors[1].gamma1_left_loc == priors[0].gamma1_left_loc
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_priors_orchestrator.py -v
```
Expected: ImportError for `build_priors`.

- [ ] **Step 3: Implement `build_priors`**

Append to `chromhandler/fitting/priors.py`:

```python
from chromhandler.annotations import PeakAnnotation
from chromhandler.fitting.prepared_dataset import PreparedDataset


def _shared_shape_from_singles(
    single_priors: list[SkewNormalPriors],
) -> tuple[float, float, float, float]:
    """Median across single-peak windows of ``(log_sigma_loc, log_sigma_scale,
    gamma1_loc, gamma1_scale)`` — used as the shape prior for doublet
    components.

    Raises:
        ValueError: If ``single_priors`` is empty (caller is responsible for
            the no-single-peak fallback).
    """
    if not single_priors:
        raise ValueError("Need at least one single-peak prior for shape borrowing.")
    return (
        float(np.median([p.log_sigma_left_loc for p in single_priors])),
        float(np.median([p.log_sigma_left_scale for p in single_priors])),
        float(np.median([p.gamma1_left_loc for p in single_priors])),
        float(np.median([p.gamma1_left_scale for p in single_priors])),
    )


def _baseline_subtracted(dataset: PreparedDataset) -> NDArray[np.float64]:
    """``signal - (intercept + slope * time)`` per trace, NaNs preserved."""
    t = dataset.time
    intercept = dataset.baseline_intercept[:, None]
    slope = dataset.baseline_slope[:, None]
    return dataset.signal - (intercept + slope * t)


def _count_window_points(
    time: NDArray[np.float64], low: float, high: float
) -> int:
    """Median number of valid time samples inside ``[low, high]`` across traces."""
    masks = (time >= low) & (time <= high) & np.isfinite(time)
    counts = masks.sum(axis=1)
    return int(np.median(counts))


def build_priors(dataset: PreparedDataset) -> list[SkewNormalPriors]:
    """Build per-annotation :class:`SkewNormalPriors` from a prepared dataset.

    Pipeline:

    1. For each ``single``-mode annotation, extract per-trace
       :class:`WindowFeatures` and aggregate.
    2. Compute shared shape priors from the population of single-peak windows.
    3. For each doublet annotation, detect dominant apexes, split areas, and
       aggregate using the shared shape.

    Args:
        dataset: Output of :func:`prepare_dataset`.

    Returns:
        List of :class:`SkewNormalPriors`, one per ``dataset.peak_annotations``,
        in the same order.

    Raises:
        ValueError: If the dataset contains doublet annotations but no single
            annotations to borrow shape from (no outer-HWHM fallback in this
            iteration).
    """
    baseline_sub = _baseline_subtracted(dataset)
    annotations = dataset.peak_annotations
    n_trace = dataset.n_trace
    dt = dataset.dt_global

    # Pass 1: aggregate single-peak priors
    single_priors_by_idx: dict[int, SkewNormalPriors] = {}
    for idx, ann in enumerate(annotations):
        if ann.n_components != 1:
            continue
        feats: list[WindowFeatures] = []
        for tr in range(n_trace):
            feats.append(
                compute_single_window_features(
                    dataset.time[tr],
                    baseline_sub[tr],
                    ann.rt_min,
                    ann.rt_max,
                )
            )
        n_pts = _count_window_points(dataset.time, ann.rt_min, ann.rt_max)
        single_priors_by_idx[idx] = aggregate_single_peak_priors(
            per_trace_features=feats,
            window_low=ann.rt_min,
            window_high=ann.rt_max,
            dt=dt,
            noise_per_trace=dataset.noise_per_trace,
            n_window_points=n_pts,
        )

    # Pass 2: shared shape priors for doublets
    has_doublet = any(a.n_components == 2 for a in annotations)
    shared_shape: tuple[float, float, float, float] | None = None
    if has_doublet:
        shared_shape = _shared_shape_from_singles(list(single_priors_by_idx.values()))

    # Pass 3: aggregate doublet priors
    doublet_priors_by_idx: dict[int, SkewNormalPriors] = {}
    for idx, ann in enumerate(annotations):
        if ann.n_components != 2:
            continue
        assert shared_shape is not None  # has_doublet ⇒ shared_shape set
        midpoint = 0.5 * (ann.rt_min + ann.rt_max)
        apex_list: list[tuple[float, float]] = []
        area_list: list[tuple[float, float]] = []
        for tr in range(n_trace):
            apex_list.append(
                detect_dominant_apex(
                    dataset.time[tr], baseline_sub[tr], ann.rt_min, ann.rt_max
                )
            )
            area_list.append(
                split_doublet_areas(
                    dataset.time[tr],
                    baseline_sub[tr],
                    ann.rt_min,
                    ann.rt_max,
                    midpoint,
                    float(dataset.noise_per_trace[tr]),
                    dt,
                )
            )
        n_pts = _count_window_points(dataset.time, ann.rt_min, ann.rt_max)
        doublet_priors_by_idx[idx] = aggregate_doublet_priors(
            per_trace_dominant_apex=apex_list,
            per_trace_areas=area_list,
            shared_shape_priors=shared_shape,
            window_low=ann.rt_min,
            window_high=ann.rt_max,
            noise_per_trace=dataset.noise_per_trace,
            dt=dt,
            n_window_points=n_pts,
        )

    # Combine, preserving annotation order
    out: list[SkewNormalPriors] = []
    for idx in range(len(annotations)):
        if idx in single_priors_by_idx:
            out.append(single_priors_by_idx[idx])
        else:
            out.append(doublet_priors_by_idx[idx])
    return out
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/unit/fitting/test_priors_orchestrator.py -v
uv run ruff check chromhandler/fitting/priors.py tests/unit/fitting/test_priors_orchestrator.py
uv run pyright chromhandler/fitting/priors.py tests/unit/fitting/test_priors_orchestrator.py
```
Expected: 4 tests pass, clean.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/priors.py tests/unit/fitting/test_priors_orchestrator.py
git commit -m "feat(priors): build_priors orchestrator with shared shape across single-peak population"
```

---

## Task 8: `summarise_priors` — human-readable inspection helper

**Why:** Lets the user see what was inferred before launching MCMC. Prints a table of `(annotation, parameter, distribution, loc, scale, bounds)` mirroring the distribution table at the top of this document.

**Files:**
- Modify: `chromhandler/fitting/priors.py`
- Test: `tests/unit/fitting/test_priors_summary.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/fitting/test_priors_summary.py`:

```python
"""Tests for the prior summary printer."""

from __future__ import annotations

import numpy as np

from chromhandler.fitting.priors import (
    SkewNormalPriors,
    summarise_priors,
)


def _single() -> SkewNormalPriors:
    return SkewNormalPriors(
        n_components=1,
        mu_left_loc=2.70,
        mu_left_scale=0.005,
        mu_left_low=2.55,
        mu_left_high=2.85,
        log_sigma_left_loc=np.log(0.03),
        log_sigma_left_scale=0.1,
        log_sigma_left_low=np.log(0.005),
        log_sigma_left_high=np.log(0.05),
        gamma1_left_loc=0.2,
        gamma1_left_scale=0.05,
        log_A_left_loc_per_trace=np.array([np.log(100.0), np.log(80.0)]),
        log_A_left_scale=0.1,
        Delta_low=None,
        Delta_high=None,
        log_sigma_right_loc=None,
        log_sigma_right_scale=None,
        log_sigma_right_low=None,
        log_sigma_right_high=None,
        gamma1_right_loc=None,
        gamma1_right_scale=None,
        log_A_right_loc_per_trace=None,
        log_A_right_scale=None,
    )


def test_summary_mentions_each_distribution() -> None:
    out = summarise_priors([_single()])
    assert "TruncatedNormal" in out  # mu, log_sigma, gamma1
    assert "Normal" in out           # log_A
    assert "mu_anchor_left" in out
    assert "log_sigma_left" in out
    assert "gamma1_left" in out
    assert "log_A_left" in out


def test_summary_handles_doublet() -> None:
    s = _single()
    d = SkewNormalPriors(
        n_components=2,
        mu_left_loc=3.00,
        mu_left_scale=0.005,
        mu_left_low=2.90,
        mu_left_high=3.15,
        log_sigma_left_loc=s.log_sigma_left_loc,
        log_sigma_left_scale=s.log_sigma_left_scale,
        log_sigma_left_low=s.log_sigma_left_low,
        log_sigma_left_high=s.log_sigma_left_high,
        gamma1_left_loc=s.gamma1_left_loc,
        gamma1_left_scale=s.gamma1_left_scale,
        log_A_left_loc_per_trace=np.array([np.log(80.0)]),
        log_A_left_scale=0.1,
        Delta_low=0.005,
        Delta_high=0.125,
        log_sigma_right_loc=s.log_sigma_left_loc,
        log_sigma_right_scale=s.log_sigma_left_scale,
        log_sigma_right_low=s.log_sigma_left_low,
        log_sigma_right_high=s.log_sigma_left_high,
        gamma1_right_loc=s.gamma1_left_loc,
        gamma1_right_scale=s.gamma1_left_scale,
        log_A_right_loc_per_trace=np.array([np.log(15.0)]),
        log_A_right_scale=0.1,
    )
    out = summarise_priors([s, d])
    assert "Uniform" in out  # Delta
    assert "Delta" in out
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/fitting/test_priors_summary.py -v
```
Expected: ImportError for `summarise_priors`.

- [ ] **Step 3: Implement `summarise_priors`**

Append to `chromhandler/fitting/priors.py`:

```python
def summarise_priors(priors: list[SkewNormalPriors]) -> str:
    """Format a list of :class:`SkewNormalPriors` as an inspection table.

    Each row shows ``(peak_idx, sampled_site, distribution, loc, scale,
    low, high)``. Matches the distribution table at the top of the priors
    plan document.

    Args:
        priors: Per-peak priors, in annotation order.

    Returns:
        Multi-line string. Print or log directly.
    """
    lines: list[str] = []
    header = (
        f"{'peak':>4} {'site':<22} {'distribution':<16} "
        f"{'loc':>10} {'scale':>10} {'low':>10} {'high':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    def fmt(v: float | None) -> str:
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
            f"{fmt(-_GAMMA1_BOUND)} {fmt(_GAMMA1_BOUND)}"
        )
        log_A_left_mean = float(np.mean(p.log_A_left_loc_per_trace))
        lines.append(
            f"{i:>4} {'log_A_left (mean)':<22} {'Normal':<16} "
            f"{fmt(log_A_left_mean)} {fmt(p.log_A_left_scale)} "
            f"{fmt(None)} {fmt(None)}"
        )
        if p.n_components == 2:
            lines.append(
                f"{i:>4} {'Delta':<22} {'Uniform':<16} "
                f"{fmt(None)} {fmt(None)} {fmt(p.Delta_low)} {fmt(p.Delta_high)}"
            )
            lines.append(
                f"{i:>4} {'log_sigma_right':<22} {'TruncatedNormal':<16} "
                f"{fmt(p.log_sigma_right_loc)} {fmt(p.log_sigma_right_scale)} "
                f"{fmt(p.log_sigma_right_low)} {fmt(p.log_sigma_right_high)}"
            )
            lines.append(
                f"{i:>4} {'gamma1_right':<22} {'TruncatedNormal':<16} "
                f"{fmt(p.gamma1_right_loc)} {fmt(p.gamma1_right_scale)} "
                f"{fmt(-_GAMMA1_BOUND)} {fmt(_GAMMA1_BOUND)}"
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

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/unit/fitting/test_priors_summary.py -v
uv run ruff check chromhandler/fitting/priors.py tests/unit/fitting/test_priors_summary.py
uv run pyright chromhandler/fitting/priors.py tests/unit/fitting/test_priors_summary.py
```
Expected: 2 tests pass, clean.

- [ ] **Step 5: Full module sweep**

```bash
uv run pytest tests/unit/fitting/test_priors_dataclass.py tests/unit/fitting/test_priors_features.py tests/unit/fitting/test_priors_apex.py tests/unit/fitting/test_priors_split.py tests/unit/fitting/test_priors_aggregate.py tests/unit/fitting/test_priors_orchestrator.py tests/unit/fitting/test_priors_summary.py -v
uv run ruff check chromhandler/fitting/priors.py
uv run pyright chromhandler/fitting/priors.py
```
Expected: all tests pass, ruff clean, pyright clean.

- [ ] **Step 6: Commit**

```bash
git add chromhandler/fitting/priors.py tests/unit/fitting/test_priors_summary.py
git commit -m "feat(priors): summarise_priors — human-readable distribution table"
```

---

## Self-Review

**Spec coverage check (spec §§3, 6, 7.2):**
- §3.1 spatial-only naming: ✅ left/right used throughout, no analyte/artefact in `SkewNormalPriors`.
- §6.1 single-peak hybrid extraction: ✅ Task 2.
- §6.1 doublet outer-HWHM split with spatial assignment: ✅ Task 4.
- §6.1 Δ Uniform on geometric bounds: ✅ Task 6.
- §6.1 shape borrowing across windows: ✅ Task 7 `_shared_shape_from_singles`.
- §6.2 aggregation: ✅ Tasks 5 and 6.
- §6.3 scale floors: ✅ Task 5 (mu→dt, log_sigma→1/√n, gamma1→√(6/n), log_A→1/√n); doublet Δ has no floor (uniform on bounds), per §6.3 footnote.
- §6.4 output structure: ✅ `SkewNormalPriors` in Task 1, with the addition of explicit `_low/_high` bounds for the distribution table.
- §7.2 public API: ✅ `WindowFeatures`, `compute_single_window_features`, `detect_dominant_apex`, `split_doublet_areas`, `aggregate_single_peak_priors`, `aggregate_doublet_priors`, `build_priors` all present. `estimate_outer_hwhm` is folded into `split_doublet_areas` as a private helper (`_outer_hwhm`) — same functionality, smaller surface.

**Type consistency check:** All `SkewNormalPriors` field names in Task 1 are used identically in Tasks 5–8. `WindowFeatures(mu, sigma, gamma1, area)` from Task 2 is consumed unchanged in Task 5. `(loc, height)` apex tuple from Task 3 flows into Task 6 as `per_trace_dominant_apex`. `(A_left, A_right)` tuple from Task 4 flows into Task 6 as `per_trace_areas`.

**Placeholder scan:** No "TBD" / "TODO" / "implement later". Every code step shows full code. No "similar to Task N" references. Note in Task 6 about possibly removing `sqrt_n` is conditional on ruff output, not a placeholder.

**Distribution-mapping coverage:** Every field in `SkewNormalPriors` is named in the distribution table at the top, traced to its computation in a specific task, and consumed by `summarise_priors` in Task 8 — closing the loop between data → priors → distribution.

**Out of scope for this plan (intentional):**
- Model-layer priors (`trace_shift`, `baseline_intercept`, `baseline_slope`) are stated in the distribution table as model-layer concerns, sourced from `PreparedDataset` directly. They appear here only as a reference table for the future `model.py` plan.
- The no-single-peaks fallback (outer-HWHM-only doublets) raises `ValueError` for now. Adding the fallback is a clean extension once we have a dataset to drive it.

**Cross-link:** This plan supersedes the priors content of `docs/plans/rewrite.md` Phase 3. After merging this plan's work, `rewrite.md` Phase 3 can be marked complete.
