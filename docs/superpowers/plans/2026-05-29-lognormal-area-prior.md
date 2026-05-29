# LogNormal Area Prior Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the supported-trace area prior `softplus(Normal(area_measured, 0.3·area_measured))` with a `LogNormal(log(area_measured), sigma_log)` whose scale is a fixed `PriorConfig.area_sigma_log` (default 1.0), resolving the empirical-Bayes precision double-counting (review item 3) and the `softplus(0) ≠ 0` offset (review item 4) in one stroke.

**Architecture:** Decouple the two jobs the old `0.3·CV` was conflating: **positivity / away-from-zero** (which keeps the per-trace `area ↔ warp` geometry out of a funnel) becomes *structural* via log-space, while the **scale** becomes a *fixed, data-independent* constant. `loc` stays at the measured trapezoid area (an order-of-magnitude anchor that sits on the likelihood peak, so it adds no bias under a wide scale). Every trace's `area` is sampled as one LogNormal site; the support gate only selects the per-trace `loc` (measured area for supported traces, a noise-floor area for unsupported), never the scale or the family.

**Tech Stack:** NumPyro/JAX model, NumPy prior construction, pytest.

**Why not marginalise area (like the baseline in item 1):** `area` is the estimand (→ concentration) and must stay positive, so its posterior is wanted — unlike the baseline nuisance.

---

## Background / current code

- `PriorConfig` ([priors.py:71-78](../../../chromhandler/fitting/priors.py)) has `area_cv = 0.3` and `area_zero_noise_multiplier = 3.0`.
- `SkewNormalPriors` ([priors.py:121-123](../../../chromhandler/fitting/priors.py)) stores `area_loc_per_trace` + `area_scale_per_trace` (both NDArray, linear space) + `has_support_per_trace`.
- `_build_one_peak` Stage 4 ([priors.py:404-416](../../../chromhandler/fitting/priors.py)) builds those arrays: `loc = where(support, measured, 0)`, `scale = where(support, max(0.3·measured, noise·ww·3), noise·ww·3)`.
- `_latent_block` ([model.py:173-180](../../../chromhandler/fitting/model.py)) does `area = softplus(area_loc + area_scale·area_raw)`.
- Only `priors.py` and `model.py` reference these (verified via grep); no test references `area_scale_per_trace`/`area_cv`; `test_priors.py` has no `area` tests.

---

## Task 1: LogNormal area prior in the priors layer

**Files:**
- Modify: `chromhandler/fitting/priors.py` (`PriorConfig`, `SkewNormalPriors`, `_build_one_peak`)
- Test: `tests/unit/fitting/test_priors.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/unit/fitting/test_priors.py`:

```python
import numpy as np

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.prepared_dataset import prepare_dataset
from chromhandler.fitting.priors import PriorConfig, build_priors


def _toy_area_dataset():
    """2 traces: trace 0 has a clear Gaussian peak (supported); trace 1 is
    flat baseline (unsupported). Small noise so the noise floor is realistic."""
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 10.0, 0.05)
    peak = 100.0 * np.exp(-0.5 * ((t - 5.0) / 0.2) ** 2)
    s0 = peak + 1.0 + rng.normal(0.0, 0.5, t.shape)
    s1 = 1.0 + rng.normal(0.0, 0.5, t.shape)
    peak_anns = [PeakAnnotation(molecule_id="x", rt_min=4.0, rt_max=6.0, mode="single")]
    base_anns = [
        BaselineAnnotation(rt_min=0.0, rt_max=1.0),
        BaselineAnnotation(rt_min=9.0, rt_max=10.0),
    ]
    return prepare_dataset([t, t], [s0, s1], peak_anns, base_anns)


def test_area_prior_is_lognormal_positive_with_fixed_scale():
    ds = _toy_area_dataset()
    priors = build_priors(ds, PriorConfig(signal_threshold=10.0))
    p = priors[0]
    # Every per-trace loc is strictly positive so log() is finite and the
    # LogNormal can never reach area=0 (no funnel).
    assert np.all(p.area_loc_per_trace > 0.0)
    # Scale is the fixed config default, NOT data-derived.
    assert p.area_log_scale == 1.0
    # Gate: strong peak supported, flat trace not; supported loc >> unsupported.
    assert bool(p.has_support_per_trace[0])
    assert not bool(p.has_support_per_trace[1])
    assert p.area_loc_per_trace[0] > p.area_loc_per_trace[1]
    # Supported loc is anchored near the measured trapezoid area (~50).
    assert 30.0 < p.area_loc_per_trace[0] < 70.0


def test_area_sigma_log_is_configurable():
    ds = _toy_area_dataset()
    priors = build_priors(ds, PriorConfig(signal_threshold=10.0, area_sigma_log=0.5))
    assert priors[0].area_log_scale == 0.5
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/fitting/test_priors.py -k "area" -v`
Expected: FAIL — `TypeError`/`AttributeError` (no `area_sigma_log` config field, no `area_log_scale` attribute).

- [ ] **Step 3: Update `PriorConfig`** — replace the `# --- Area prior ---` block ([priors.py:71-78](../../../chromhandler/fitting/priors.py)):

```python
    # --- Area prior (LogNormal) ---
    area_sigma_log: float = 1.0
    """Fixed sigma of the underlying Normal on ``log(area)`` for the
    per-trace LogNormal area prior. Data-INDEPENDENT (this is what removes
    the old empirical-Bayes precision double-counting): the data sets the
    prior's median via ``area_measured``, this constant sets its spread.
    ~1.0 means area is weakly held within a factor of ~e per sigma, so the
    likelihood dominates the value while log-space keeps area > 0."""

    area_zero_noise_multiplier: float = 3.0
    """Sets the LogNormal median for UNSUPPORTED traces (and the positive
    floor for all traces): ``noise * window_width * multiplier`` — the area
    a noise-level signal would integrate to over the window ("if anything
    is here it's at most noise-level")."""
```

(Removes `area_cv`.)

- [ ] **Step 4: Update `SkewNormalPriors`** — replace the area fields ([priors.py:121-123](../../../chromhandler/fitting/priors.py)) and the area bullet in its class docstring ([priors.py:100](../../../chromhandler/fitting/priors.py)).

Class-docstring bullet — replace the `area[trace] ~ Normal(...)` line with:
```python
    - ``area[trace] ~ LogNormal(log(area_loc_per_trace), area_log_scale)``
      — positive by construction (exp), so it never reaches 0; ``loc``
      anchors at the measured trapezoid area (supported) or a noise-floor
      area (unsupported).
```

Fields — replace `area_loc_per_trace` / `area_scale_per_trace` / `has_support_per_trace`:
```python
    area_loc_per_trace: NDArray[np.float64]
    """Per-trace linear-space LogNormal median (strictly positive, >= the
    noise-floor area). Positivity is what keeps the per-trace area<->warp
    geometry out of a funnel."""
    area_log_scale: float
    """Fixed sigma on ``log(area)`` (from ``PriorConfig.area_sigma_log``),
    shared across traces. NOT data-derived — removes the precision
    double-counting of the old ``0.3 * area_measured`` scale."""
    has_support_per_trace: NDArray[np.bool_]
```

- [ ] **Step 5: Update `_build_one_peak` Stage 4** — replace [priors.py:404-416](../../../chromhandler/fitting/priors.py):

```python
    # --- Stage 4: per-trace LogNormal area prior -------------------------
    # Linear-space median per trace: the measured trapezoid area for
    # supported traces, a noise-floor area for unsupported ones. Floored at
    # the noise floor (and a tiny absolute floor) so every loc is strictly
    # positive -> log() is finite and the LogNormal can never reach area=0,
    # which keeps the per-trace area<->warp geometry funnel-free.
    noise_floor = (
        config.area_zero_noise_multiplier
        * dataset.noise_per_trace
        * window_width
    )
    area_loc_per_trace = np.where(has_support, areas_measured, 0.0)
    area_loc_per_trace = np.maximum(area_loc_per_trace, noise_floor)
    area_loc_per_trace = np.maximum(area_loc_per_trace, 1e-12)
```

And update the `return SkewNormalPriors(...)` ([priors.py:418-425](../../../chromhandler/fitting/priors.py)) to pass the new fields:
```python
    return SkewNormalPriors(
        mu_loc=mu_loc, mu_scale=mu_scale,
        width_loc=width_loc, width_log_scale=width_log_scale,
        skew_loc=skew_loc, skew_scale=skew_scale,
        area_loc_per_trace=area_loc_per_trace,
        area_log_scale=float(config.area_sigma_log),
        has_support_per_trace=has_support,
    )
```

- [ ] **Step 6: Run to verify pass**

Run: `uv run pytest tests/unit/fitting/test_priors.py -k "area" -v`
Expected: PASS (both new tests).

- [ ] **Step 7: Lint + typecheck**

Run: `uv run ruff check chromhandler/fitting/priors.py tests/unit/fitting/test_priors.py && uv run pyright chromhandler/fitting/priors.py`
Expected: no new errors. (If `summarise_priors` references `area_scale_per_trace`, leave it for Task 3 — but if it breaks `pyright`, note it and proceed.)

- [ ] **Step 8: Commit**

```bash
git add chromhandler/fitting/priors.py tests/unit/fitting/test_priors.py
git commit -m "feat(fitting): LogNormal area prior with fixed area_sigma_log (review item 3)"
```

---

## Task 2: Use the LogNormal area site in the model

**Files:**
- Modify: `chromhandler/fitting/model.py` (`_latent_block` area block + docstring)
- Test: `tests/unit/fitting/test_fitter_entry.py` (existing real-fit smoke test)

- [ ] **Step 1: Replace the area block in `_latent_block`** — replace [model.py:173-180](../../../chromhandler/fitting/model.py) (the `area_loc`/`area_scale`/`softplus` lines):

```python
    # area: LogNormal in natural space. loc is the per-trace linear-space
    # median (strictly positive); area_log_scale is the fixed sigma on
    # log(area). exp() guarantees positivity and bounds area away from 0
    # (no per-trace area<->warp funnel); the fixed scale avoids the old
    # empirical-Bayes precision double-counting.
    area_log_loc = jnp.log(
        jnp.asarray(np.stack([p.area_loc_per_trace for p in priors_list], axis=1))
    )  # [n_trace, n_peak]
    area_log_scale = jnp.asarray([p.area_log_scale for p in priors_list])  # [n_peak]
    area_raw = numpyro.sample(
        "area_raw", dist.Normal(jnp.zeros((n_trace, n_peak)), 1.0)
    )
    area = numpyro.deterministic(
        "area", jnp.exp(area_log_loc + area_log_scale[None, :] * area_raw)
    )
```

- [ ] **Step 2: Update the `model()` docstring deterministic-site line** for `area` ([model.py:238](../../../chromhandler/fitting/model.py)):

```python
        - ``area[trace, peak]``     = exp(log(area_loc) + area_log_scale * area_raw)
```

- [ ] **Step 3: Lint + typecheck** (catches a now-unused `jax.nn` import if `softplus` was its only use)

Run: `uv run ruff check chromhandler/fitting/model.py && uv run pyright chromhandler/fitting/model.py`
Expected: clean except pre-existing JAX-stub warnings. If ruff flags an unused `import jax` / `jax.nn`, confirm `jax` is unused elsewhere in `model.py` before removing — `jax.random`/`jax.nn` may still be used; only remove if genuinely unused.

- [ ] **Step 4: Smoke-test a real fit** (the LogNormal area must sample and stay positive)

Run: `uv run pytest tests/unit/fitting/test_fitter_entry.py -v`
Expected: PASS. Then confirm positivity with a one-liner:

Run:
```bash
uv run python -c "
import numpyro; numpyro.set_host_device_count(4)
import numpy as np
from tests.unit.fitting.test_priors import _toy_area_dataset
from chromhandler.fitting import fit, ModelConfig
from chromhandler.fitting.priors import PriorConfig
ds = _toy_area_dataset()
r = fit(ds, prior_config=PriorConfig(signal_threshold=10.0),
        model_config=ModelConfig(num_warmup=150, num_samples=150, num_chains=2, seed=0))
a = np.asarray(r.idata.posterior['area'])
print('area min =', float(a.min()), '| all positive:', bool((a > 0).all()))
print('divergences:', r.diagnostics()['n_divergent'])
"
```
Expected: `all positive: True` and `divergences: 0` (the funnel check — area bounded away from 0 should keep divergences at 0). Report both numbers.

- [ ] **Step 5: Run the full fitting suite**

Run: `uv run pytest tests/unit/fitting/ -q`
Expected: 0 failed.

- [ ] **Step 6: Commit**

```bash
git add chromhandler/fitting/model.py
git commit -m "feat(fitting): sample area as LogNormal in the model (review items 3+4)"
```

---

## Task 3: Update `summarise_priors`, verify end-to-end, record in review doc

**Files:**
- Modify: `chromhandler/fitting/priors.py` (`summarise_priors`)
- Modify: `docs/superpowers/specs/2026-05-29-fitting-module-integrity-review.md`

- [ ] **Step 1: Update the area row in `summarise_priors`** — replace the `area (mean)` block ([priors.py:482-489](../../../chromhandler/fitting/priors.py)):

```python
        n_supp = int(np.sum(p.has_support_per_trace))
        n_total = p.has_support_per_trace.size
        # LogNormal: geometric-mean median across traces +/- 1 log-sigma.
        med_area = float(np.exp(np.mean(np.log(p.area_loc_per_trace))))
        a_p16 = med_area * float(np.exp(-p.area_log_scale))
        a_p84 = med_area * float(np.exp(+p.area_log_scale))
        lines.append(
            f"{i:>4} {'area (median)':<14} {'LogNormal':<18} "
            f"{med_area:>10.4g} {p.area_log_scale:>10.4g} "
            f"{a_p16:>10.4g} {a_p84:>10.4g}"
            f"  [supported {n_supp}/{n_total}]"
        )
```

- [ ] **Step 2: Verify `summarise_priors` runs**

Run:
```bash
uv run python -c "
from tests.unit.fitting.test_priors import _toy_area_dataset
from chromhandler.fitting.priors import PriorConfig, build_priors, summarise_priors
ds = _toy_area_dataset(); pc = PriorConfig(signal_threshold=10.0)
print(summarise_priors(build_priors(ds, pc), pc))
"
```
Expected: prints a table with a `LogNormal` area row, no exception.

- [ ] **Step 3: Lint + full fitting suite (final gate)**

Run: `uv run ruff check chromhandler/fitting/priors.py && uv run pytest tests/unit/fitting/ -q`
Expected: ruff clean; 0 failed.

- [ ] **Step 4: End-to-end run on the real fixture**

Run: `uv run python test.py 2>&1 | tail -20`
Expected: completes; prints the LogNormal area row in the prior table, diagnostics with `n_divergent 0`, writes the three plots. Note in the report whether the `area` posterior credible intervals are now visibly *wider* than before (expected — the honest uncertainty the tight `0.3·CV` was suppressing) and whether divergences stayed at 0.

- [ ] **Step 5: Record resolution in the review doc** — under item 3, add a `✅ RESOLVED` note (LogNormal with fixed `area_sigma_log=1.0`, `loc` = measured/noise-floor, positivity kills the funnel, scale is data-independent). Under item 4 (`softplus(0) ≠ 0`), add `✅ RESOLVED — softplus removed; area is now exp(LogNormal), so unsupported traces sit at a positive noise-floor median rather than `softplus(0)=ln2`.`

- [ ] **Step 6: Commit**

```bash
git add chromhandler/fitting/priors.py docs/superpowers/specs/2026-05-29-fitting-module-integrity-review.md
git commit -m "docs(fitting): LogNormal area summary + record review items 3 & 4 resolved"
```

---

## Self-Review notes

- **Spec coverage:** item 3 (data-derived scale) → fixed `area_sigma_log` (Tasks 1-2); item 4 (`softplus(0)≠0`) → softplus removed (Task 2). Both recorded (Task 3).
- **Type consistency:** `area_log_scale: float` is set in `_build_one_peak` (Task 1 Step 5), consumed in `_latent_block` (Task 2 Step 1) and `summarise_priors` (Task 3 Step 1); `area_scale_per_trace` is fully removed from all three sites. `area_loc_per_trace` stays an `NDArray` and is now guaranteed `> 0`.
- **Funnel guard:** positivity is structural (`exp`), so the fixed wide scale (default 1.0) cannot reintroduce `area=0`. The Task 2 Step 4 divergence check is the empirical confirmation.
- **Unsupported semantics:** absent peaks now get a small positive noise-floor median rather than exactly 0 — acceptable for quantification ("below detection"); confirmed acceptable with the user.
- **Out of scope:** `plot_prior_overlay` reads `p.area_loc_per_trace[tr]` (still valid, the LogNormal median) and `has_support_per_trace` (unchanged) — no change needed.
