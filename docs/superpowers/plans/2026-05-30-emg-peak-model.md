# EMG Peak Model + Per-Peak Selection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an exponentially-modified Gaussian (EMG) peak model alongside the skew-normal, selectable per peak via `PeakAnnotation.peak_model`, so a fit can mix shapes — motivated by genuinely tailing peaks (the ATP fixture) whose exponential tail a skew-normal can't capture.

**Architecture:** A pure `emg.py` math layer (regime-switched stable density on `jax.scipy.special.erfc`, mirroring `skew_normal.py`), an `EMGPriors` dataclass + `build_priors` dispatch, and a `_latent_block` refactor that partitions peaks into an SN group and an EMG group (each vectorised, summed). The all-skew-normal path stays byte-identical (sample-site order preserved; EMG sites appended only when present).

**Tech Stack:** NumPyro/JAX, `jax.scipy.special.erfc`, `scipy.stats.exponnorm` (test reference), pytest.

**Spec:** `docs/superpowers/specs/2026-05-30-emg-peak-model-design.md`.

---

## Background facts (verified)

- `jax.scipy.special.erfc` exists; **`erfcx` does not**, and TFP's JAX substrate is broken with jax 0.9.
- `EMG(μ,σ,τ)` density equals `scipy.stats.exponnorm.pdf(x, K=τ/σ, loc=μ, scale=σ)` — the test reference.
- The regime-switched `density_emg` below was validated vs `exponnorm` to ~1e-6 (incl. far tail, float32, finite gradients).
- Model posterior sites today (order matters for RNG): `mu_raw, width_raw, skew_raw, area_raw, noise_raw, baseline_*` (no obs site — factor likelihood), then deterministics `mu, width, skew, area, noise, time_shift, time_stretch, mu_warped, width_warped`.

## File structure

- **Create `chromhandler/fitting/emg.py`** — `density_emg`, `_erfcx_pos`, `mode_emg`, `fwhm_emg`, `emg_from_peak_features`, `_emg_asymmetry_table`. Pure math, mirrors `skew_normal.py`.
- **Modify `chromhandler/annotations.py`** — add `PeakAnnotation.peak_model`.
- **Modify `chromhandler/fitting/priors.py`** — add `EMGPriors` + `_build_one_emg_peak`; `build_priors` dispatches on `peak_model`.
- **Modify `chromhandler/fitting/model.py`** — `_latent_block` partitions SN/EMG.
- **Create `tests/fixtures/atp_tailing/ATP_sig.csv`** + a small loader in the test/dev script.
- **Create** `tests/unit/fitting/test_emg.py`, `tests/unit/fitting/test_emg_priors.py`, `tests/unit/fitting/test_emg_model.py`.
- **Create `dev/emg_vs_skewnormal.py`** — ATP SN-vs-EMG comparison.

---

## Task 1: `emg.py` — stable density + math tests

**Files:** Create `chromhandler/fitting/emg.py`, `tests/unit/fitting/test_emg.py`

- [ ] **Step 1: Write failing tests** (`tests/unit/fitting/test_emg.py`):

```python
"""EMG math: density correctness vs scipy, float32 stability, Gaussian limit."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import exponnorm

from chromhandler.fitting.emg import density_emg


@pytest.mark.parametrize("sigma,tau", [(1.0, 0.5), (1.0, 2.0), (1.0, 5.0), (0.03, 0.05)])
def test_density_matches_exponnorm(sigma, tau):
    mu = 5.0
    x = np.linspace(mu - 4 * sigma, mu + 12 * tau, 400)
    ref = exponnorm.pdf(x, tau / sigma, loc=mu, scale=sigma)
    mine = np.asarray(density_emg(jnp.asarray(x), jnp.asarray(mu),
                                  jnp.asarray(sigma), jnp.asarray(tau)))
    assert np.max(np.abs(mine - ref)) / np.max(ref) < 1e-5


def test_density_integrates_to_one():
    mu, sigma, tau = 5.0, 0.05, 0.1
    x = np.linspace(mu - 1.0, mu + 3.0, 20001)
    d = np.asarray(density_emg(jnp.asarray(x), jnp.asarray(mu),
                               jnp.asarray(sigma), jnp.asarray(tau)))
    assert abs(np.trapezoid(d, x) - 1.0) < 1e-4


def test_gaussian_limit_small_tau():
    # tau -> 0 : EMG -> Gaussian(mu, sigma)
    mu, sigma, tau = 0.0, 1.0, 1e-4
    x = np.linspace(-5, 5, 401)
    emg = np.asarray(density_emg(jnp.asarray(x), jnp.asarray(mu),
                                 jnp.asarray(sigma), jnp.asarray(tau)))
    gauss = np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    assert np.max(np.abs(emg - gauss)) < 1e-2


def test_float32_far_tail_and_gradient_finite():
    x = jnp.linspace(2.0, 30.0, 3000, dtype=jnp.float32)
    d = density_emg(x, jnp.float32(5.0), jnp.float32(0.05), jnp.float32(0.1))
    assert bool(np.all(np.isfinite(np.asarray(d))))
    g = jax.grad(lambda tau: jnp.sum(
        density_emg(jnp.linspace(4.5, 8.0, 200), 5.0, 0.1, tau)))(jnp.asarray(0.3))
    assert np.isfinite(float(g))
```

- [ ] **Step 2: Run, verify fail** — `uv run pytest tests/unit/fitting/test_emg.py -q` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Create `chromhandler/fitting/emg.py`** (this exact, validated code):

```python
"""Pure-math exponentially-modified Gaussian (EMG) layer.

EMG = Gaussian(mu, sigma) convolved with a right Exp(mean=tau), tau > 0.
Equivalent to scipy.stats.exponnorm with K = tau/sigma, loc = mu, scale = sigma.

The density is evaluated with a regime switch on w = (sigma/tau - (x-mu)/sigma)/sqrt(2)
because the two analytically-equal forms each overflow on one side. jax has no
erfcx, so the w>=0 branch builds it from erfc with an asymptotic tail; both
branches use the safe-`where` pattern (inputs clamped) so gradients stay finite.
No NumPyro imports, no state.
"""
from __future__ import annotations

import math

import jax.numpy as jnp
from jax.scipy.special import erfc

_SQRT2 = math.sqrt(2.0)
_INV_SQRTPI = 1.0 / math.sqrt(math.pi)


def _erfcx_pos(w: jnp.ndarray) -> jnp.ndarray:
    """Scaled complementary error function erfcx(w)=exp(w^2)erfc(w), for w>=0.

    Direct exp(w^2)*erfc(w) for small w; asymptotic series for large w (where,
    in float32, exp(w^2) overflows and erfc(w) underflows). Both branch inputs
    are clamped so the inactive branch can't overflow (finite gradients).
    """
    w_small = jnp.minimum(w, 6.0)
    small = jnp.exp(w_small ** 2) * erfc(w_small)
    w_large = jnp.maximum(w, 1.0)
    inv = 1.0 / (w_large ** 2)
    asymp = (_INV_SQRTPI / w_large) * (1.0 - 0.5 * inv + 0.75 * inv ** 2 - 1.875 * inv ** 3)
    return jnp.where(w < 6.0, small, asymp)


def density_emg(
    x: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, tau: jnp.ndarray
) -> jnp.ndarray:
    """EMG density (unit area). mu = Gaussian centre, sigma > 0, tau > 0 (tail)."""
    u = (x - mu) / sigma
    lam = sigma / tau
    w = (lam - u) / _SQRT2
    core = (1.0 / (2.0 * tau)) * jnp.exp(-0.5 * u ** 2) * _erfcx_pos(jnp.maximum(w, 0.0))
    u_tail = jnp.maximum(u, lam)  # clamp inactive branch so exp can't overflow
    tail = (1.0 / (2.0 * tau)) * jnp.exp(0.5 * lam ** 2 - lam * u_tail) * erfc(jnp.minimum(w, 0.0))
    return jnp.where(w >= 0.0, core, tail)
```

- [ ] **Step 4: Run, verify pass** — `uv run pytest tests/unit/fitting/test_emg.py -q` → PASS (4 tests).

- [ ] **Step 5: Lint + commit**

Run: `uv run ruff check chromhandler/fitting/emg.py tests/unit/fitting/test_emg.py`
```bash
git add chromhandler/fitting/emg.py tests/unit/fitting/test_emg.py
git commit -m "feat(fitting): stable regime-switched EMG density"
```

---

## Task 2: `emg.py` — mode + FWHM (reporting helpers)

**Files:** Modify `chromhandler/fitting/emg.py`, `tests/unit/fitting/test_emg.py`

- [ ] **Step 1: Write failing tests** (append):

```python
def test_mode_emg_matches_grid_argmax():
    from chromhandler.fitting.emg import mode_emg
    mu, sigma, tau = 5.0, 0.05, 0.1
    xs = np.linspace(mu - 0.5, mu + 1.0, 400001)
    d = np.asarray(density_emg(jnp.asarray(xs), jnp.asarray(mu),
                               jnp.asarray(sigma), jnp.asarray(tau)))
    grid_mode = xs[int(np.argmax(d))]
    assert abs(float(mode_emg(mu, sigma, tau)) - grid_mode) < 1e-3


def test_fwhm_emg_matches_grid():
    from chromhandler.fitting.emg import fwhm_emg, mode_emg
    mu, sigma, tau = 5.0, 0.05, 0.1
    xs = np.linspace(mu - 1.0, mu + 3.0, 2000001)
    d = np.asarray(density_emg(jnp.asarray(xs), jnp.asarray(mu),
                               jnp.asarray(sigma), jnp.asarray(tau)))
    peak = d.max()
    above = xs[d >= peak / 2]
    assert abs(float(fwhm_emg(mu, sigma, tau)) - (above.max() - above.min())) < 5e-3
```

- [ ] **Step 2: Run, verify fail** → FAIL (`cannot import name 'mode_emg'`).

- [ ] **Step 3: Implement** (append to `emg.py`; mirrors `skew_normal._fwhm_scalar` — `scipy.optimize` for reporting only, not on the HMC path):

```python
import numpy as np
from scipy.optimize import brentq, minimize_scalar


def mode_emg(mu: float, sigma: float, tau: float) -> float:
    """Mode (apex) of EMG(mu, sigma, tau), numerically. Reporting only."""
    def neg(x: float) -> float:
        return -float(density_emg(jnp.asarray(x), jnp.asarray(mu),
                                  jnp.asarray(sigma), jnp.asarray(tau)))
    res = minimize_scalar(neg, bounds=(mu - 5 * sigma, mu + 20 * tau + 5 * sigma),
                          method="bounded")
    return float(res.x)


def fwhm_emg(mu: float, sigma: float, tau: float) -> float:
    """Full width at half maximum of EMG(mu, sigma, tau), numerically."""
    m = mode_emg(mu, sigma, tau)
    peak = float(density_emg(jnp.asarray(m), jnp.asarray(mu),
                             jnp.asarray(sigma), jnp.asarray(tau)))
    half = peak / 2.0

    def shifted(x: float) -> float:
        return float(density_emg(jnp.asarray(x), jnp.asarray(mu),
                                 jnp.asarray(sigma), jnp.asarray(tau))) - half

    lo, hi = m - sigma, m + tau + sigma
    while shifted(lo) > 0.0:
        lo -= sigma
    while shifted(hi) > 0.0:
        hi += tau + sigma
    x_left = float(brentq(shifted, lo, m))
    x_right = float(brentq(shifted, m, hi))
    return x_right - x_left
```

- [ ] **Step 4: Run, verify pass** → PASS.

- [ ] **Step 5: Lint + commit**
```bash
git add chromhandler/fitting/emg.py tests/unit/fitting/test_emg.py
git commit -m "feat(fitting): EMG mode + FWHM reporting helpers"
```

---

## Task 3: `emg.py` — feature inversion `(apex, FWHM, HWHM-ratio) -> (mu, sigma, tau)`

**Files:** Modify `chromhandler/fitting/emg.py`, `tests/unit/fitting/test_emg.py`

- [ ] **Step 1: Write failing round-trip test** (append):

```python
@pytest.mark.parametrize("sigma,tau", [(0.04, 0.02), (0.04, 0.08), (0.04, 0.2)])
def test_emg_from_peak_features_roundtrip(sigma, tau):
    from chromhandler.fitting.emg import (
        emg_from_peak_features, fwhm_emg, mode_emg, hwhm_ratio_emg,
    )
    mu_true = 5.0
    apex = mode_emg(mu_true, sigma, tau)
    fwhm = fwhm_emg(mu_true, sigma, tau)
    ratio = hwhm_ratio_emg(sigma, tau)
    mu, s, t = emg_from_peak_features(apex, fwhm, ratio)
    assert abs(mu - mu_true) < 5e-3
    assert abs(s - sigma) / sigma < 0.1
    assert abs(t - tau) / tau < 0.1
```

- [ ] **Step 2: Run, verify fail** → FAIL.

- [ ] **Step 3: Implement** (append to `emg.py`). `hwhm_ratio_emg` measures the right/left half-width ratio (a function of `K=tau/sigma` only); a cached table inverts ratio → K (mirrors `skew_normal._asymmetry_table`). Then FWHM sets `sigma` at fixed `K`, `tau = K*sigma`, and the apex sets `mu` (offset-corrected via `mode_emg`).

```python
import functools


def hwhm_ratio_emg(sigma: float, tau: float) -> float:
    """Right/left HWHM ratio of EMG. Depends only on K = tau/sigma."""
    m = mode_emg(0.0, sigma, tau)
    peak = float(density_emg(jnp.asarray(m), jnp.asarray(0.0),
                             jnp.asarray(sigma), jnp.asarray(tau)))
    half = peak / 2.0

    def shifted(x: float) -> float:
        return float(density_emg(jnp.asarray(x), jnp.asarray(0.0),
                                 jnp.asarray(sigma), jnp.asarray(tau))) - half

    lo, hi = m - sigma, m + tau + sigma
    while shifted(lo) > 0.0:
        lo -= sigma
    while shifted(hi) > 0.0:
        hi += tau + sigma
    xl = float(brentq(shifted, lo, m))
    xr = float(brentq(shifted, m, hi))
    return (xr - m) / (m - xl)


@functools.lru_cache(maxsize=1)
def _emg_ratio_table() -> tuple[np.ndarray, np.ndarray]:
    """(HWHM-ratio -> K) inversion table, monotone in ratio. Cached once."""
    ks = np.geomspace(1e-3, 50.0, 400)
    ratios = np.array([hwhm_ratio_emg(1.0, float(k)) for k in ks])  # sigma=1 -> tau=k
    order = np.argsort(ratios)
    return ratios[order], ks[order]


def emg_from_peak_features(apex: float, fwhm: float, hwhm_ratio: float) -> tuple[float, float, float]:
    """Invert measured (apex, FWHM, HWHM-ratio) to EMG (mu, sigma, tau)."""
    ratios, ks = _emg_ratio_table()
    K = float(np.interp(hwhm_ratio, ratios, ks))
    fwhm_unit = fwhm_emg(0.0, 1.0, K)        # FWHM of EMG(0,1,K); FWHM ∝ sigma at fixed K
    sigma = fwhm / fwhm_unit
    tau = K * sigma
    mode_unit = mode_emg(0.0, 1.0, K)        # apex = mu + sigma * mode_unit(K)
    mu = apex - sigma * mode_unit
    return float(mu), float(sigma), float(tau)
```

- [ ] **Step 4: Run, verify pass** → PASS. (If the 10% tolerance is tight for the largest-tail case due to table resolution, widen the geomspace upper bound / point count — do NOT loosen the test beyond 10%.)

- [ ] **Step 5: Lint + commit**
```bash
git add chromhandler/fitting/emg.py tests/unit/fitting/test_emg.py
git commit -m "feat(fitting): EMG feature inversion (apex/FWHM/HWHM-ratio -> mu,sigma,tau)"
```

---

## Task 4: `PeakAnnotation.peak_model`

**Files:** Modify `chromhandler/annotations.py`, Test: `tests/unit/fitting/test_annotation_validators.py`

- [ ] **Step 1: Write failing test** (append to `tests/unit/fitting/test_annotation_validators.py`):

```python
def test_peak_annotation_peak_model_field():
    from chromhandler.annotations import PeakAnnotation
    a = PeakAnnotation(molecule_id="x", rt_min=1.0, rt_max=2.0, mode="single")
    assert a.peak_model == "skew_normal"  # default preserves behaviour
    b = PeakAnnotation(molecule_id="x", rt_min=1.0, rt_max=2.0, mode="single",
                       peak_model="emg")
    assert b.peak_model == "emg"
```

- [ ] **Step 2: Run, verify fail** → FAIL (`unexpected keyword 'peak_model'` or attribute missing).

- [ ] **Step 3: Implement.** Read `chromhandler/annotations.py`, find `PeakAnnotation`, and add a field `peak_model: Literal["skew_normal", "emg"] = "skew_normal"` (import `Literal` from `typing` if absent). Match the existing field style (dataclass/pydantic — mirror how `mode` is declared).

- [ ] **Step 4: Run, verify pass** → PASS.

- [ ] **Step 5: Lint + commit**
```bash
git add chromhandler/annotations.py tests/unit/fitting/test_annotation_validators.py
git commit -m "feat(fitting): PeakAnnotation.peak_model selector (skew_normal|emg)"
```

---

## Task 5: `EMGPriors` + `build_priors` dispatch

**Files:** Modify `chromhandler/fitting/priors.py`, Test: `tests/unit/fitting/test_emg_priors.py`

- [ ] **Step 1: Write failing tests** (`tests/unit/fitting/test_emg_priors.py`):

```python
import numpy as np

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.prepared_dataset import prepare_dataset
from chromhandler.fitting.priors import EMGPriors, PriorConfig, build_priors


def _toy_emg_dataset():
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 10.0, 0.02)
    # a right-tailing peak near 5.0 (sum of gaussian + exp-ish tail)
    peak = 1000.0 * np.exp(-0.5 * ((t - 5.0) / 0.05) ** 2)
    tail = np.where(t > 5.0, 400.0 * np.exp(-(t - 5.0) / 0.15), 0.0)
    s = peak + tail + 1.0 + rng.normal(0, 0.5, t.shape)
    pk = [PeakAnnotation(molecule_id="x", rt_min=4.7, rt_max=5.8,
                         mode="single", peak_model="emg")]
    bs = [BaselineAnnotation(rt_min=0.0, rt_max=1.0),
          BaselineAnnotation(rt_min=9.0, rt_max=10.0)]
    return prepare_dataset([t], [s], pk, bs)


def test_build_priors_emits_emg_priors():
    ds = _toy_emg_dataset()
    priors = build_priors(ds, PriorConfig(signal_threshold=50.0))
    p = priors[0]
    assert isinstance(p, EMGPriors)
    assert p.emg_sigma_loc > 0 and p.emg_tau_loc > 0
    assert np.all(p.area_loc_per_trace > 0)
    assert p.area_log_scale == 1.0
    # mu is offset-corrected: emg_mu (Gaussian centre) sits LEFT of the apex (~5.0)
    assert p.emg_mu_loc < 5.05


def test_build_priors_mixed_types():
    rng = np.random.default_rng(1)
    t = np.arange(0.0, 10.0, 0.02)
    s = 1000.0 * np.exp(-0.5 * ((t - 3.0) / 0.05) ** 2) + \
        1000.0 * np.exp(-0.5 * ((t - 6.0) / 0.05) ** 2) + 1.0 + rng.normal(0, 0.5, t.shape)
    from chromhandler.fitting.priors import SkewNormalPriors
    pk = [PeakAnnotation(molecule_id="a", rt_min=2.7, rt_max=3.3, mode="single"),
          PeakAnnotation(molecule_id="b", rt_min=5.7, rt_max=6.3, mode="single", peak_model="emg")]
    bs = [BaselineAnnotation(rt_min=0.0, rt_max=1.0), BaselineAnnotation(rt_min=9.0, rt_max=10.0)]
    ds = prepare_dataset([t], [s], pk, bs)
    priors = build_priors(ds, PriorConfig(signal_threshold=50.0))
    assert isinstance(priors[0], SkewNormalPriors)
    assert isinstance(priors[1], EMGPriors)
```

- [ ] **Step 2: Run, verify fail** → FAIL (`cannot import name 'EMGPriors'`).

- [ ] **Step 3: Implement in `priors.py`.** Add the dataclass (scalar log-space locs/scales for sigma & tau, mirroring `width_loc`/`width_log_scale`; area fields identical to `SkewNormalPriors`):

```python
@dataclass(frozen=True)
class EMGPriors:
    """Single-peak EMG priors for one window (native params)."""
    emg_mu_loc: float
    emg_mu_scale: float
    emg_sigma_loc: float          # natural-space LogNormal median
    emg_sigma_log_scale: float
    emg_tau_loc: float            # natural-space LogNormal median
    emg_tau_log_scale: float
    area_loc_per_trace: NDArray[np.float64]
    area_log_scale: float
    has_support_per_trace: NDArray[np.bool_]
```

Add `_build_one_emg_peak(dataset, baseline_sub, ann, config)` mirroring `_build_one_peak` but: per supported trace, invert features via `emg_from_peak_features(apex, fwhm, ratio)` (using the same `compute_window_features` apex/FWHM/HWHM-ratio already computed) to `(mu, sigma, tau)`; aggregate `emg_mu_loc`=mean(mu), `emg_mu_scale`=max(std(mu), dt-floor); `emg_sigma_loc`=geomean(sigma), `emg_tau_loc`=geomean(tau), with fixed log-scales + `n=1` fallbacks (reuse `config.log_width_scale_n1` for both, and a new `config.emg_tau_log_scale_n1: float = 0.3`); `area_*` identical to the SN path. Unsupported/un-invertible traces fall back to window-geometry σ and a default `τ = window_width/10`.

Then make `build_priors` dispatch:
```python
    out = []
    for ann in dataset.peak_annotations:
        if getattr(ann, "peak_model", "skew_normal") == "emg":
            out.append(_build_one_emg_peak(dataset, baseline_sub, ann, cfg))
        else:
            out.append(_build_one_peak(dataset, baseline_sub, ann, cfg))
    return out
```
Also add `compute_window_features` must expose the HWHM-ratio it already computes (it currently returns CP via `cp_from_peak_features`); refactor so the raw `(apex, mean_fwhm, mean_ratio)` are available to both SN and EMG builders (e.g. return them on `WindowFeatures` or have a shared `_measure_features` returning the triple). Keep SN behaviour unchanged.

- [ ] **Step 4: Run, verify pass** — `uv run pytest tests/unit/fitting/test_emg_priors.py -q` → PASS. Also `uv run pytest tests/unit/fitting/test_priors.py -q` (SN priors unchanged).

- [ ] **Step 5: Lint + typecheck + commit**
```bash
uv run ruff check chromhandler/fitting/priors.py && uv run pyright chromhandler/fitting/priors.py
git add chromhandler/fitting/priors.py tests/unit/fitting/test_emg_priors.py
git commit -m "feat(fitting): EMGPriors + build_priors dispatch on peak_model"
```

---

## Task 6: Mixed-type `_latent_block` + recovery/regression tests

**Files:** Modify `chromhandler/fitting/model.py`, Test: `tests/unit/fitting/test_emg_model.py`

- [ ] **Step 1: Write failing/verification tests** (`tests/unit/fitting/test_emg_model.py`):

```python
import numpyro
numpyro.set_host_device_count(4)
import numpy as np

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting import ModelConfig, fit
from chromhandler.fitting.prepared_dataset import prepare_dataset
from chromhandler.fitting.priors import PriorConfig


def _emg_trace(mu_g=5.0, sigma=0.05, tau=0.12, area=3000.0, seed=0):
    from scipy.stats import exponnorm
    rng = np.random.default_rng(seed)
    t = np.arange(2.0, 8.0, 0.02)
    s = area * exponnorm.pdf(t, tau / sigma, loc=mu_g, scale=sigma) + 2.0 + rng.normal(0, 1.0, t.shape)
    return t, s


def test_emg_recovers_known_tau_no_divergence():
    t, s = _emg_trace(tau=0.12)
    pk = [PeakAnnotation(molecule_id="x", rt_min=4.6, rt_max=6.0, mode="single", peak_model="emg")]
    bs = [BaselineAnnotation(rt_min=2.0, rt_max=2.5), BaselineAnnotation(rt_min=7.5, rt_max=8.0)]
    ds = prepare_dataset([t, t], [s, _emg_trace(tau=0.12, seed=1)[1]], pk, bs)
    r = fit(ds, prior_config=PriorConfig(signal_threshold=50.0),
            model_config=ModelConfig(num_warmup=400, num_samples=400, num_chains=2, seed=0))
    tau_post = float(np.asarray(r.idata.posterior["emg_tau"]).mean())
    assert 0.08 < tau_post < 0.18                     # recovers true tau ~0.12
    assert r.diagnostics()["n_divergent"] == 0        # non-centred geometry OK


def test_all_skew_normal_unchanged():
    # an all-SN fit must still run and expose mu/width/skew (no emg_* sites)
    rng = np.random.default_rng(2)
    t = np.arange(2.0, 8.0, 0.02)
    s = 3000.0 * np.exp(-0.5 * ((t - 5.0) / 0.05) ** 2) + 2.0 + rng.normal(0, 1.0, t.shape)
    pk = [PeakAnnotation(molecule_id="x", rt_min=4.6, rt_max=5.6, mode="single")]
    bs = [BaselineAnnotation(rt_min=2.0, rt_max=2.5), BaselineAnnotation(rt_min=7.5, rt_max=8.0)]
    ds = prepare_dataset([t, t], [s, s], pk, bs)
    r = fit(ds, prior_config=PriorConfig(signal_threshold=50.0),
            model_config=ModelConfig(num_warmup=200, num_samples=200, num_chains=2, seed=0))
    assert "mu" in r.idata.posterior and "emg_tau" not in r.idata.posterior
```

- [ ] **Step 2: Run, verify fail** → FAIL (`emg_tau` not a posterior var).

- [ ] **Step 3: Refactor `_latent_block`** in `model.py`. Partition peaks by type; **preserve the existing site order** (SN shape → area → noise → warp) so all-SN fits are byte-identical, and **append EMG shape sites at the end** (new sites don't perturb earlier RNG). Compute both groups' contributions and sum. Key structure (adapt to the current code; `density_emg` imported from `chromhandler.fitting.emg`):

```python
    from chromhandler.fitting.emg import density_emg
    from chromhandler.fitting.priors import EMGPriors

    sn_idx = [i for i, p in enumerate(priors_list) if not isinstance(p, EMGPriors)]
    emg_idx = [i for i, p in enumerate(priors_list) if isinstance(p, EMGPriors)]
    time_arr = jnp.asarray(dataset.time)
    peak_contrib = jnp.zeros((n_trace, time_arr.shape[1]))

    # --- SN group (same sites/order as today; for an all-SN fit this is the whole model) ---
    if sn_idx:
        sn = [priors_list[i] for i in sn_idx]
        # ... existing mu_raw/width_raw/skew_raw sampling + mu/width/skew deterministics,
        #     but over `sn` instead of all peaks ...
        mu_warped = (mu[None, :] - time_shift[:, None]) / time_stretch[:, None]
        width_warped = width[None, :] / time_stretch[:, None]
        dens = density_cp(time_arr[:, None, :], mu_warped[:, :, None],
                          width_warped[:, :, None], skew[None, :, None])
        peak_contrib += jnp.sum(area[:, jnp.asarray(sn_idx)][:, :, None] * dens, axis=1)

    # --- area / noise / warp: sampled in the SAME positions as today (unchanged) ---

    # --- EMG group: NEW sites, appended after warp so SN RNG is unperturbed ---
    if emg_idx:
        em = [priors_list[i] for i in emg_idx]
        n_e = len(em)
        emg_mu_loc = jnp.asarray([p.emg_mu_loc for p in em])
        emg_mu_scale = jnp.asarray([p.emg_mu_scale for p in em])
        emg_mu_raw = numpyro.sample("emg_mu_raw", dist.Normal(jnp.zeros(n_e), 1.0))
        emg_mu = numpyro.deterministic("emg_mu", emg_mu_loc + emg_mu_scale * emg_mu_raw)
        log_sig_loc = jnp.log(jnp.asarray([p.emg_sigma_loc for p in em]))
        sig_ls = jnp.asarray([p.emg_sigma_log_scale for p in em])
        emg_sigma_raw = numpyro.sample("emg_sigma_raw", dist.Normal(jnp.zeros(n_e), 1.0))
        emg_sigma = numpyro.deterministic("emg_sigma", jnp.exp(log_sig_loc + sig_ls * emg_sigma_raw))
        log_tau_loc = jnp.log(jnp.asarray([p.emg_tau_loc for p in em]))
        tau_ls = jnp.asarray([p.emg_tau_log_scale for p in em])
        emg_tau_raw = numpyro.sample("emg_tau_raw", dist.Normal(jnp.zeros(n_e), 1.0))
        emg_tau = numpyro.deterministic("emg_tau", jnp.exp(log_tau_loc + tau_ls * emg_tau_raw))
        # warp: shift subtracts, stretch divides (sigma and tau scale like a width)
        emg_mu_w = (emg_mu[None, :] - time_shift[:, None]) / time_stretch[:, None]
        emg_sigma_w = emg_sigma[None, :] / time_stretch[:, None]
        emg_tau_w = emg_tau[None, :] / time_stretch[:, None]
        dens_e = density_emg(time_arr[:, None, :], emg_mu_w[:, :, None],
                             emg_sigma_w[:, :, None], emg_tau_w[:, :, None])
        peak_contrib += jnp.sum(area[:, jnp.asarray(emg_idx)][:, :, None] * dens_e, axis=1)

    return {"peak_contrib": peak_contrib, "noise": noise}
```

Notes: `area` (all peaks) and `noise`/warp keep their current sampling positions; only the SN shape block is now scoped to `sn_idx`. For an all-SN dataset `emg_idx == []` → the EMG block is skipped entirely → identical sites/RNG to today. Update the `model()` docstring's deterministic-site list to mention the conditional `emg_*` sites. `predictive_model` already consumes `_latent_block`'s `peak_contrib`/`noise`, so it needs no change.

- [ ] **Step 4: Run, verify pass** — `uv run pytest tests/unit/fitting/test_emg_model.py -q` (recovery + all-SN). Then the **full fitting suite**: `uv run pytest tests/unit/fitting/ -q` → 0 failed (the regression gate — all-SN behaviour preserved).

- [ ] **Step 5: Lint + typecheck + commit**
```bash
uv run ruff check chromhandler/fitting/model.py && uv run pyright chromhandler/fitting/model.py
git add chromhandler/fitting/model.py tests/unit/fitting/test_emg_model.py
git commit -m "feat(fitting): mixed SN/EMG peaks in the model (partition + sum)"
```

---

## Task 7: ATP fixture + SN-vs-EMG comparison + cross-model area agreement

**Files:** Create `tests/fixtures/atp_tailing/ATP_sig.csv`, `dev/emg_vs_skewnormal.py`; Test: append to `tests/unit/fitting/test_emg_model.py`

- [ ] **Step 1: Add the fixture file.** Copy the provided CSV into the repo:
```bash
mkdir -p tests/fixtures/atp_tailing
cp /tmp/ATP_sig.csv tests/fixtures/atp_tailing/ATP_sig.csv
```
(If `/tmp/ATP_sig.csv` is absent, ask the controller to re-stage it from `/Users/max/Downloads/IL01/ATP_sig.csv`.)

- [ ] **Step 2: Write the loader + cross-model area-agreement test** (append to `tests/unit/fitting/test_emg_model.py`):

```python
def _load_atp():
    from pathlib import Path
    p = Path("tests/fixtures/atp_tailing/ATP_sig.csv")
    data = np.genfromtxt(p, delimiter=",", names=True)
    t = data["RTminutes__NOT_USED_BY_IMPORT"]
    s = data["260"]
    m = (t >= 4.6) & (t <= 6.4)   # focus region around the ATP peak
    return t[m], s[m]


def test_atp_emg_area_matches_trapezoid_and_beats_sn_tail():
    t, s = _load_atp()
    base_lo, base_hi = (4.62, 4.72), (6.2, 6.38)
    bs = [BaselineAnnotation(rt_min=base_lo[0], rt_max=base_lo[1]),
          BaselineAnnotation(rt_min=base_hi[0], rt_max=base_hi[1])]
    cfg = ModelConfig(num_warmup=400, num_samples=400, num_chains=2, seed=0)
    out = {}
    for model_name in ("skew_normal", "emg"):
        pk = [PeakAnnotation(molecule_id="ATP", rt_min=4.9, rt_max=5.7,
                             mode="single", peak_model=model_name)]
        ds = prepare_dataset([t], [s], pk, bs)
        r = fit(ds, prior_config=PriorConfig(signal_threshold=1e6), model_config=cfg)
        out[model_name] = (r, ds)
    # EMG area matches a direct trapezoid integral of the baseline-subtracted peak
    r, ds = out["emg"]
    area_emg = float(np.asarray(r.idata.posterior["area"]).mean())
    b_int = float(np.asarray(r.idata.posterior["baseline_intercept"]).mean())
    b_slp = float(np.asarray(r.idata.posterior["baseline_slope"]).mean())
    tt = np.asarray(ds.time)[0]; ss = np.asarray(ds.signal)[0]; vmask = np.asarray(ds.valid_mask)[0]
    win = vmask & (tt >= 4.9) & (tt <= 5.7)
    trap = float(np.trapezoid((ss - (b_int + b_slp * tt))[win], tt[win]))
    assert abs(area_emg - trap) / trap < 0.05      # EMG area is accurate
    # EMG noise (absorbed misfit) is lower than SN's, since EMG fits the tail
    noise_emg = float(np.asarray(out["emg"][0].idata.posterior["noise"]).mean())
    noise_sn = float(np.asarray(out["skew_normal"][0].idata.posterior["noise"]).mean())
    assert noise_emg < noise_sn
```

- [ ] **Step 3: Run, verify** — `uv run pytest "tests/unit/fitting/test_emg_model.py::test_atp_emg_area_matches_trapezoid_and_beats_sn_tail" -v`. Expected: PASS (EMG area within 5% of trapezoid; EMG noise < SN noise). If baseline windows give too few points, adjust `base_lo/base_hi` to flat regions read off the data; if the `noise_emg < noise_sn` margin is borderline, report the two values (do not weaken the area assertion).

- [ ] **Step 4: Write the dev comparison script** `dev/emg_vs_skewnormal.py`: load ATP, fit with both `peak_model`s, print areas + noise + diagnostics, and save an overlay of data + both fits + residuals to `dev/emg_vs_skewnormal.png`. (Mirror `test.py`'s structure; use `result.plot_fit()` for each.)

- [ ] **Step 5: Run it** — `uv run python dev/emg_vs_skewnormal.py 2>&1 | tail -15`. Confirm it writes the PNG and that EMG's tail residual is visibly smaller. Report areas + noise for both.

- [ ] **Step 6: Lint + commit**
```bash
uv run ruff check dev/emg_vs_skewnormal.py tests/unit/fitting/test_emg_model.py
git add tests/fixtures/atp_tailing/ATP_sig.csv dev/emg_vs_skewnormal.py tests/unit/fitting/test_emg_model.py
git commit -m "test(fitting): ATP tailing fixture — EMG area-accurate + beats SN tail"
```

---

## Self-Review notes

- **Spec coverage:** density (T1), reporting (T2), inversion (T3), selector (T4), priors+dispatch (T5), mixed model+recovery+regression (T6), fixture+area-agreement+SN/EMG compare (T7). τ identifiability test = T6's recovery (recovers true τ) + T7 (prior-sensitivity is optional follow-up, noted). All-SN-unchanged = T6 regression + full-suite gate.
- **Type consistency:** `EMGPriors` fields (`emg_mu_loc/emg_mu_scale/emg_sigma_loc/emg_sigma_log_scale/emg_tau_loc/emg_tau_log_scale/area_loc_per_trace/area_log_scale/has_support_per_trace`) defined T5, consumed T6. Deterministic sites `emg_mu/emg_sigma/emg_tau` named consistently T6↔tests. `density_emg(x,mu,sigma,tau)` signature stable T1→T6.
- **RNG-order preservation** is the regression-critical detail (T6): SN/area/noise/warp sites keep positions; EMG sites appended. The full-suite gate + `test_all_skew_normal_unchanged` enforce it.
- **Highest risk = T1 density** (now verified vs scipy) and **T6 refactor** (regression-gated). T5's feature refactor must not change SN priors — `test_priors.py` is the guard.
- **Deferred (not blocking):** τ prior-sensitivity refit and reparameterise-on-mode fallback (only if T6 recovery shows divergences); LOO/model comparison (out of scope per spec).
