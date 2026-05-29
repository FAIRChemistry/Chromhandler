# Marginalised Linear Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the sampled per-trace linear baseline in the NumPyro model with an analytic (flat-prior) marginalisation, eliminating the data-double-counting prior (review items 1 + 6) and the baseline↔area sampling correlation.

**Architecture:** The baseline `a + b·t` is linear and the noise is Gaussian, so conditional on the peak/noise/warp parameters `(a, b)` is a linear-Gaussian sub-model and can be integrated out in closed form. With a flat prior this reduces to a *projection*: at each HMC step, the part of the peak-subtracted residual that a straight line can explain is removed for free, and only the orthogonal residual is charged against the likelihood. The two baseline sample sites disappear; the baseline is reconstructed post-hoc from its conditional (Rao–Blackwellised) for reporting and predictive sampling.

**Tech Stack:** NumPyro / JAX (model + factor), ArviZ (idata), pytest (math unit tests), `test.py` script (end-to-end A/B on the real ASM kinetic-series fixture).

**Correctness bar (per user decision):** A/B comparison in `test.py` on the real fixture — *no synthetic data*. The replacement is accepted iff the new model's `area`, `mu`, `width`, `skew` posteriors agree with the golden (current-model) posteriors within tolerance, with divergences no worse and ESS/sec no worse. Two fast closed-form math unit tests are included as TDD safety for the projection formula (these verify the *math*, not statistical recovery; drop them only if you accept replacing a likelihood with zero unit coverage).

**Pre-existing ArviZ 1.x breakage (folded into this plan):** the environment has ArviZ **1.1.0**, which removed `arviz.InferenceData` (now `xarray.DataTree`), the `arviz.from_dict(posterior_predictive=...)` kwarg form, and `idata.extend`. This already breaks `compute_posterior_predictive`, `compute_prior_predictive`, `plot_fit`, `plot_prior_predictive`, `plot_traces`, and `save` on `main`. The migration is folded in here (Phase 4 rewrites the predictive functions against the 1.x API anyway; Task 4b migrates `save`/`plot_traces`). ArviZ 1.x cheat-sheet for implementers:

- `az.from_dict({"posterior_predictive": {"obs": obs}}, coords=..., dims=...)` (positional dict, arbitrary group names) → returns an `xarray.DataTree`.
- `idata.extend(other)` → `idata.update(other)` (DataTree left-merge).
- `hasattr(idata, "posterior_predictive")` → `"posterior_predictive" in idata.children`.
- `idata.to_netcdf(path, engine="h5netcdf")` to save; `az.from_netcdf(path)` to load.
- `az.summary`, `az.hdi`, `az.plot_trace`, `az.rc_context` still exist (the `summary`/`diagnostics` paths already pass).
- **Verify empirically:** the worktree has a live ArviZ 1.1.0; run the failing tests after each change rather than trusting this cheat-sheet blindly.

---

## Math reference (flat-prior marginalisation)

Per trace, with `N` valid points, design `X = [1, t_centred]` (centre `t` per trace so `XᵀX` is diagonal), peak-subtracted residual `r = y − peak_contrib`, noise `σ`:

```
t_mean   = mean(t over valid)
tc       = (t − t_mean)   [0 where invalid]
n        = N
Stt      = Σ tc²
Sr       = Σ r            (valid only)
Str      = Σ tc·r         (valid only)
rss      = Σ r²           (valid only)
rss_perp = rss − Sr²/n − Str²/Stt          # residual a line cannot absorb
loglik   = −0.5·(n − 2)·log(2π σ²) − rss_perp / (2σ²)
```

Dropped additive constants (`−0.5·log det XᵀX`, the `2π` from the β integral) are data-only / parameter-free and do not affect the posterior. The `(n − 2)` exponent is the automatic degrees-of-freedom correction for the two baseline DOF.

Reconstructed baseline (conditional mean, original coords):

```
slope_hat     = Str / Stt
intercept_hat = Sr / n − slope_hat · t_mean
```

Conditional covariance (diagonal, centred coords): `Var(a_c) = σ²/n`, `Var(b_c) = σ²/Stt` — used only for predictive sampling.

---

## File Structure

- `chromhandler/fitting/model.py` — **primary changes**: add `marginal_baseline_loglik()` (pure, testable), add `_latent_block()` (shared sampling + peak_contrib), rewrite `model()` to use the factor, add `predictive_model()`, delete `_compute_baseline_se`, update `run_mcmc`.
- `chromhandler/fitting/posterior.py` — `compute_posterior_predictive` / `compute_prior_predictive` switch to `predictive_model`.
- `tests/unit/fitting/test_marginal_baseline.py` — **new**: closed-form unit tests for `marginal_baseline_loglik`.
- `test.py` — add golden-reference capture + A/B assessment on the real fixture.
- `tests/fixtures/asm_kinetic_series/golden_baseline_model.json` — **new**: committed golden posteriors from the *current* model (captured in Phase 0, before any edit).

---

## Phase 0: Capture the golden reference (BEFORE editing any source)

This must run against the **current, unmodified** model on `main`/`fix-fit` HEAD, since we are replacing it.

### Task 0: Snapshot current-model posteriors + diagnostics

**Files:**
- Create: `scripts/capture_golden.py` (throwaway capture script)
- Create: `tests/fixtures/asm_kinetic_series/golden_baseline_model.json`

- [ ] **Step 1: Write the capture script**

```python
# scripts/capture_golden.py
"""One-time capture of CURRENT-model posteriors as the A/B golden reference.
Run on unmodified HEAD before the marginalisation change."""
import json
import time as _time
from pathlib import Path

import numpyro

numpyro.set_host_device_count(8)

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting import ModelConfig, fit
from chromhandler.fitting.priors import PriorConfig
from chromhandler.handler import Handler


def build_dataset():
    data = Path("tests/fixtures/asm_kinetic_series")
    handler = Handler.read_asm(path=data, mode="timecourse")
    handler.load_initial_conditions(
        "tests/fixtures/asm_kinetic_series/conditions.csv", conc_unit="umol / l"
    )
    handler.create_molecule(id="SIH", pubchem_cid=135398693, name="S-inosyl-L-homocysteine")
    handler.create_molecule(id="other", pubchem_cid=0, name="other")
    handler.create_molecule(id="third", pubchem_cid=0, name="third")
    peak_anns = [
        PeakAnnotation(molecule_id="other", rt_min=2.55, rt_max=2.80, mode="single"),
        PeakAnnotation(molecule_id="SIH", rt_min=2.80, rt_max=3.15, mode="single"),
        PeakAnnotation(molecule_id="third", rt_min=3.15, rt_max=3.45, mode="single"),
    ]
    base_anns = [
        BaselineAnnotation(rt_min=2.50, rt_max=2.52),
        BaselineAnnotation(rt_min=3.55, rt_max=3.58),
    ]
    return handler.prepare_dataset(peak_anns, base_anns)


def main():
    dataset = build_dataset()
    pc = PriorConfig(min_height_frac=0.05)
    t0 = _time.perf_counter()
    result = fit(dataset, prior_config=pc,
                 model_config=ModelConfig(num_warmup=500, num_samples=500, num_chains=4, seed=0))
    wall = _time.perf_counter() - t0

    summ = result.summary(var_names=["area", "mu", "width", "skew"])
    diag = result.diagnostics()
    payload = {
        "wall_seconds": wall,
        "diagnostics": {k: (float(v) if isinstance(v, (int, float)) else v)
                        for k, v in diag.items()},
        "summary": json.loads(summ[["mean", "sd", "ess_bulk"]].to_json(orient="index")),
    }
    out = Path("tests/fixtures/asm_kinetic_series/golden_baseline_model.json")
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out} (wall={wall:.1f}s, ess_min={diag['ess_min_bulk']:.0f}, "
          f"div={diag['n_divergent']})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it on the unmodified model**

Run: `uv run python scripts/capture_golden.py`
Expected: prints `wrote tests/fixtures/.../golden_baseline_model.json (wall=…s, ess_min=…, div=…)`; the JSON exists and contains `summary` keys like `area[0, 0]`, `mu[0]`, etc.

- [ ] **Step 3: Commit the golden reference**

```bash
git add scripts/capture_golden.py tests/fixtures/asm_kinetic_series/golden_baseline_model.json
git commit -m "test(fitting): capture current-model posteriors as A/B golden reference"
```

---

## Phase 1: Extract the shared latent/peak-contribution block (pure refactor)

No behaviour change — existing tests must stay green. This isolates the sampling so `model()` and the new `predictive_model()` share one code path (DRY).

### Task 1: Factor out `_latent_block`

**Files:**
- Modify: `chromhandler/fitting/model.py:121-285` (the `model` body)
- Test: existing `tests/unit/fitting/test_fitter_class.py`, `tests/unit/fitting/test_model_config.py`

- [ ] **Step 1: Add the helper above `model()`**

```python
def _latent_block(
    dataset: "PreparedDataset",
    priors_list: list["SkewNormalPriors"],
    config: ModelConfig,
) -> dict[str, jnp.ndarray]:
    """Sample all latent peak/noise/warp sites and return peak_contrib + noise.

    Shared by ``model`` (which marginalises the baseline) and
    ``predictive_model`` (which draws the baseline from its conditional).
    Registers the same ``numpyro.deterministic`` sites as before so
    downstream summary/plot code is unchanged.
    """
    n_trace = dataset.n_trace
    n_peak = len(priors_list)
    dt_global = float(dataset.dt_global)

    mu_loc = jnp.asarray([p.mu_loc for p in priors_list])
    mu_scale = jnp.asarray([p.mu_scale for p in priors_list])
    mu_raw = numpyro.sample("mu_raw", dist.Normal(jnp.zeros(n_peak), 1.0))
    mu = numpyro.deterministic("mu", mu_loc + mu_scale * mu_raw)

    log_width_loc = jnp.log(jnp.asarray([p.width_loc for p in priors_list]))
    width_log_scale = jnp.asarray([p.width_log_scale for p in priors_list])
    width_raw = numpyro.sample("width_raw", dist.Normal(jnp.zeros(n_peak), 1.0))
    width = numpyro.deterministic(
        "width", jnp.exp(log_width_loc + width_log_scale * width_raw)
    )

    skew_loc = jnp.asarray([p.skew_loc for p in priors_list])
    skew_scale = jnp.asarray([p.skew_scale for p in priors_list])
    skew_max = float(GAMMA1_MAX)
    skew_raw = numpyro.sample("skew_raw", dist.Normal(jnp.zeros(n_peak), 1.0))
    skew = numpyro.deterministic(
        "skew", skew_max * jnp.tanh((skew_loc + skew_scale * skew_raw) / skew_max)
    )

    area_loc = jnp.asarray(np.stack([p.area_loc_per_trace for p in priors_list], axis=1))
    area_scale = jnp.asarray(np.stack([p.area_scale_per_trace for p in priors_list], axis=1))
    area_raw = numpyro.sample("area_raw", dist.Normal(jnp.zeros((n_trace, n_peak)), 1.0))
    area = numpyro.deterministic("area", jax.nn.softplus(area_loc + area_scale * area_raw))

    log_noise_loc = jnp.log(jnp.asarray(dataset.noise_per_trace))
    noise_raw = numpyro.sample("noise_raw", dist.Normal(jnp.zeros(n_trace), 1.0))
    noise = numpyro.deterministic(
        "noise", jnp.exp(log_noise_loc + config.log_noise_scale * noise_raw)
    )

    shift_scale = config.warp_shift_scale_dt_multiplier * dt_global
    time_shift_raw = numpyro.sample("time_shift_raw", dist.Normal(jnp.zeros(n_trace), 1.0))
    _shift = shift_scale * time_shift_raw
    time_shift = numpyro.deterministic("time_shift", _shift - jnp.mean(_shift))

    time_stretch_raw = numpyro.sample("time_stretch_raw", dist.Normal(jnp.zeros(n_trace), 1.0))
    _log_stretch = config.warp_stretch_scale * time_stretch_raw
    time_stretch = numpyro.deterministic(
        "time_stretch", jnp.exp(_log_stretch - jnp.mean(_log_stretch))
    )

    mu_warped = numpyro.deterministic(
        "mu_warped", (mu[None, :] - time_shift[:, None]) / time_stretch[:, None]
    )
    width_warped = numpyro.deterministic(
        "width_warped", width[None, :] / time_stretch[:, None]
    )

    time_arr = jnp.asarray(dataset.time)
    dens_all = density_cp(
        time_arr[:, None, :],
        mu_warped[:, :, None],
        width_warped[:, :, None],
        skew[None, :, None],
    )
    peak_contrib = jnp.sum(area[:, :, None] * dens_all, axis=1)  # [n_trace, n_time]
    return {"peak_contrib": peak_contrib, "noise": noise}
```

- [ ] **Step 2: Run the existing fitting unit tests to confirm the helper imports cleanly**

Run: `uv run pytest tests/unit/fitting/test_model_config.py -v`
Expected: PASS (no behaviour referenced yet; this just checks the module still imports).

- [ ] **Step 3: Commit**

```bash
git add chromhandler/fitting/model.py
git commit -m "refactor(fitting): extract _latent_block from model body"
```

---

## Phase 2: Implement + unit-test the marginal likelihood (TDD)

### Task 2: `marginal_baseline_loglik` with closed-form tests

**Files:**
- Create: `tests/unit/fitting/test_marginal_baseline.py`
- Modify: `chromhandler/fitting/model.py` (add `marginal_baseline_loglik` near the top, after imports)

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/fitting/test_marginal_baseline.py
"""Closed-form checks for the flat-prior baseline marginalisation."""
import numpy as np

from chromhandler.fitting.model import marginal_baseline_loglik


def _lstsq_reference(y, peak, t, mask, sigma):
    """Independent reference: OLS of (y - peak) on [1, t] over valid points."""
    loglik, intercept, slope = [], [], []
    for tr in range(y.shape[0]):
        m = mask[tr]
        r = (y[tr] - peak[tr])[m]
        tt = t[tr][m]
        X = np.column_stack([np.ones_like(tt), tt])
        beta, *_ = np.linalg.lstsq(X, r, rcond=None)
        resid = r - X @ beta
        n = r.size
        rss_perp = float(resid @ resid)
        s2 = float(sigma[tr] ** 2)
        ll = -0.5 * (n - 2) * np.log(2 * np.pi * s2) - rss_perp / (2 * s2)
        loglik.append(ll)
        intercept.append(float(beta[0]))
        slope.append(float(beta[1]))
    return np.array(loglik), np.array(intercept), np.array(slope)


def test_loglik_matches_lstsq_reference():
    rng = np.random.default_rng(0)
    n_trace, n_time = 3, 40
    t = np.tile(np.linspace(2.0, 4.0, n_time), (n_trace, 1))
    peak = rng.normal(size=(n_trace, n_time))
    # Construct y = line + peak + noise, with per-trace different baselines
    y = np.empty((n_trace, n_time))
    for tr, (a, b) in enumerate([(5.0, 1.0), (-2.0, 0.3), (0.0, -0.7)]):
        y[tr] = a + b * t[tr] + peak[tr] + rng.normal(scale=0.1, size=n_time)
    mask = np.ones((n_trace, n_time), dtype=bool)
    sigma = np.array([0.1, 0.2, 0.15])

    ll, intercept, slope = marginal_baseline_loglik(y, peak, t, mask, sigma)
    ll_ref, int_ref, slope_ref = _lstsq_reference(y, peak, t, mask, sigma)

    np.testing.assert_allclose(np.asarray(ll), ll_ref, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(intercept), int_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(slope), slope_ref, rtol=1e-5, atol=1e-6)


def test_mask_excludes_invalid_points():
    # Padded/invalid points must not affect loglik or the recovered line.
    n_time = 30
    t = np.linspace(2.0, 4.0, n_time)[None, :]
    peak = np.zeros((1, n_time))
    y = (3.0 + 0.5 * t + np.zeros_like(t))  # exact line, zero residual
    mask = np.ones((1, n_time), dtype=bool)
    mask[0, -5:] = False
    y_poison = y.copy()
    y_poison[0, -5:] = 1e6  # garbage in masked-out region
    sigma = np.array([0.2])

    ll_clean, _, _ = marginal_baseline_loglik(y, peak, t, mask, sigma)
    ll_poison, int_p, slope_p = marginal_baseline_loglik(y_poison, peak, t, mask, sigma)

    np.testing.assert_allclose(np.asarray(ll_clean), np.asarray(ll_poison), rtol=1e-5)
    np.testing.assert_allclose(float(int_p), 3.0, atol=1e-4)
    np.testing.assert_allclose(float(slope_p), 0.5, atol=1e-4)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/fitting/test_marginal_baseline.py -v`
Expected: FAIL with `ImportError: cannot import name 'marginal_baseline_loglik'`.

- [ ] **Step 3: Implement `marginal_baseline_loglik`**

Add to `chromhandler/fitting/model.py` (after the imports, before `_compute_baseline_se`):

```python
def marginal_baseline_loglik(
    signal: jnp.ndarray,     # [n_trace, n_time]
    peak_contrib: jnp.ndarray,  # [n_trace, n_time]
    time: jnp.ndarray,       # [n_trace, n_time]
    valid_mask: jnp.ndarray,  # [n_trace, n_time] bool
    noise: jnp.ndarray,      # [n_trace]
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Flat-prior analytic marginalisation of the per-trace linear baseline.

    Integrates out ``baseline = a + b·t`` (independent per trace) under an
    improper-flat prior. Returns ``(loglik_per_trace, intercept_hat,
    slope_hat)`` where the loglik is the marginal Gaussian log-density of
    the peak-subtracted residual (up to a parameter-free additive constant),
    and the hats are the Rao–Blackwellised conditional-mean baseline in
    ORIGINAL (uncentred) coordinates. See the design spec for the math.
    """
    w = valid_mask.astype(jnp.float64)
    n = jnp.sum(w, axis=1)                                  # [n_trace]
    n_safe = jnp.maximum(n, 1.0)
    t_clean = jnp.where(valid_mask, time, 0.0)
    t_mean = jnp.sum(w * t_clean, axis=1) / n_safe          # [n_trace]
    tc = jnp.where(valid_mask, time - t_mean[:, None], 0.0)
    Stt = jnp.sum(tc * tc, axis=1)                          # [n_trace]
    Stt_safe = jnp.maximum(Stt, 1e-30)

    r = jnp.where(valid_mask, jnp.nan_to_num(signal) - peak_contrib, 0.0)
    Sr = jnp.sum(r, axis=1)                                 # [n_trace]
    Str = jnp.sum(tc * r, axis=1)                           # [n_trace]
    rss = jnp.sum(r * r, axis=1)                            # [n_trace]
    rss_perp = rss - Sr**2 / n_safe - Str**2 / Stt_safe

    sigma2 = noise**2
    loglik = -0.5 * (n - 2.0) * jnp.log(2.0 * jnp.pi * sigma2) - rss_perp / (2.0 * sigma2)

    slope_hat = Str / Stt_safe
    intercept_hat = Sr / n_safe - slope_hat * t_mean
    return loglik, intercept_hat, slope_hat
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/fitting/test_marginal_baseline.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Lint + typecheck the new function**

Run: `uv run ruff check chromhandler/fitting/model.py && uv run pyright chromhandler/fitting/model.py`
Expected: no errors on `marginal_baseline_loglik`.

- [ ] **Step 6: Commit**

```bash
git add chromhandler/fitting/model.py tests/unit/fitting/test_marginal_baseline.py
git commit -m "feat(fitting): flat-prior analytic baseline marginalisation"
```

---

## Phase 3: Rewrite `model()` to use the marginal likelihood

### Task 3: Replace baseline sampling with the factor

**Files:**
- Modify: `chromhandler/fitting/model.py` (`model` body, `run_mcmc`; delete `_compute_baseline_se`)

- [ ] **Step 1: Replace the `model()` body**

Replace the entire body of `model()` (everything after the docstring) with:

```python
    block = _latent_block(dataset, priors_list, config)
    peak_contrib = block["peak_contrib"]
    noise = block["noise"]

    loglik, intercept_hat, slope_hat = marginal_baseline_loglik(
        jnp.asarray(dataset.signal),
        peak_contrib,
        jnp.asarray(dataset.time),
        jnp.asarray(dataset.valid_mask),
        noise,
    )
    # Rao–Blackwellised baseline (conditional mean) exposed for reporting.
    numpyro.deterministic("baseline_intercept", intercept_hat)
    numpyro.deterministic("baseline_slope", slope_hat)
    numpyro.factor("obs_marginal", jnp.sum(loglik))
```

- [ ] **Step 2: Delete `_compute_baseline_se`**

Remove the entire `_compute_baseline_se` function ([model.py:47-80](../../../chromhandler/fitting/model.py)) and the two `ModelConfig` fields that only it used: `baseline_intercept_se_floor`, `baseline_slope_se_floor` ([model.py:96-97](../../../chromhandler/fitting/model.py)).

- [ ] **Step 3: Simplify `run_mcmc` (no more conditioning)**

The inference model now reads `dataset.signal` directly inside the factor, so the `condition` handler is obsolete. Replace the `conditioned_model = ...` line and the `NUTS(conditioned_model, ...)` argument so NUTS wraps `model` directly:

```python
    kernel = numpyro.infer.NUTS(
        model,
        target_accept_prob=config.target_accept_prob,
        max_tree_depth=config.max_tree_depth,
    )
```

Remove the now-unused `numpyro.handlers.condition(...)` block.

- [ ] **Step 4: Lint + typecheck**

Run: `uv run ruff check chromhandler/fitting/model.py && uv run pyright chromhandler/fitting/model.py`
Expected: no errors. (If `test_model_config.py` referenced the deleted SE-floor fields, it will surface in the next step.)

- [ ] **Step 5: Update model-config test if it referenced deleted fields**

Run: `uv run pytest tests/unit/fitting/test_model_config.py -v`
If it fails on `baseline_intercept_se_floor` / `baseline_slope_se_floor`, delete those assertions from the test (the fields no longer exist). Re-run; expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add chromhandler/fitting/model.py tests/unit/fitting/test_model_config.py
git commit -m "refactor(fitting): marginalise baseline in model, drop sampled baseline sites"
```

---

## Phase 4: Predictive sampling against the marginalised model

The inference `model` has no `"obs"` sample site (it's a factor), so `Predictive` can no longer generate `obs`. Add a generative `predictive_model` that draws the baseline from its conditional and samples `obs`.

### Task 4: Add `predictive_model` and rewire `posterior.py`

**Files:**
- Modify: `chromhandler/fitting/model.py` (add `predictive_model`)
- Modify: `chromhandler/fitting/posterior.py:24-121`
- Test: `tests/unit/fitting/test_fitter_class.py` (exercises `plot_fit` / predictive)

- [ ] **Step 1: Add `predictive_model` after `model()`**

```python
def predictive_model(
    dataset: "PreparedDataset",
    priors_list: list["SkewNormalPriors"],
    config: ModelConfig,
) -> None:
    """Generative twin of ``model`` for prior/posterior predictive sampling.

    Samples the same latent sites (so posterior samples substitute cleanly),
    draws the baseline from its conditional given the observed data, then
    samples ``obs``. For prior predictive the conditional is taken against
    the real data (a data-anchored prior predictive) so the band sits at a
    sensible level despite the improper flat baseline prior — documented
    caveat, used for visualisation only.
    """
    block = _latent_block(dataset, priors_list, config)
    peak_contrib = block["peak_contrib"]
    noise = block["noise"]

    time = jnp.asarray(dataset.time)
    valid_mask = jnp.asarray(dataset.valid_mask)
    w = valid_mask.astype(jnp.float64)
    n = jnp.maximum(jnp.sum(w, axis=1), 1.0)
    t_clean = jnp.where(valid_mask, time, 0.0)
    t_mean = jnp.sum(w * t_clean, axis=1) / n
    tc = jnp.where(valid_mask, time - t_mean[:, None], 0.0)
    Stt = jnp.maximum(jnp.sum(tc * tc, axis=1), 1e-30)

    r = jnp.where(valid_mask, jnp.nan_to_num(jnp.asarray(dataset.signal)) - peak_contrib, 0.0)
    a_hat = jnp.sum(r, axis=1) / n              # centred intercept
    b_hat = jnp.sum(tc * r, axis=1) / Stt
    n_trace = dataset.n_trace
    eps = numpyro.sample("baseline_raw", dist.Normal(jnp.zeros((n_trace, 2)), 1.0))
    a_c = a_hat + jnp.sqrt(noise**2 / n) * eps[:, 0]
    b_c = b_hat + jnp.sqrt(noise**2 / Stt) * eps[:, 1]
    baseline = a_c[:, None] + b_c[:, None] * tc

    predicted = baseline + peak_contrib
    predicted = jnp.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0)
    with numpyro.handlers.mask(mask=valid_mask):
        numpyro.sample("obs", dist.Normal(predicted, noise[:, None]))
```

- [ ] **Step 2: Point `posterior.py` predictive functions at `predictive_model` AND migrate to ArviZ 1.x**

Two changes per function: (a) switch the `Predictive` target from `model` to `predictive_model`; (b) migrate the group construction from the removed `arviz.from_dict(posterior_predictive=...)` + `idata.extend` to the ArviZ 1.x `from_dict` positional-dict form + `idata.update`.

In `compute_posterior_predictive`:

```python
    predictive = numpyro.infer.Predictive(
        predictive_model, posterior_samples=flat_posterior, return_sites=["obs"],
    )
    # ... obs reshape unchanged ...
    pp = arviz.from_dict(
        {"posterior_predictive": {"obs": obs}},
        coords=coords,
        dims={"obs": ["chain", "draw", "trace", "time_idx"]},
    )
    idata.update(pp)   # ArviZ 1.x: DataTree merge (was idata.extend)
    return idata
```

In `compute_prior_predictive`, build both groups with the positional-dict form and merge:

```python
    predictive = numpyro.infer.Predictive(
        predictive_model, num_samples=config.prior_predictive_n_samples,
    )
    # ... obs + prior_dict reshape unchanged ...
    pp = arviz.from_dict(
        {"prior": prior_dict, "prior_predictive": {"obs": obs}},
        coords=coords,
        dims={"obs": ["chain", "draw", "trace", "time_idx"]},
    )
    idata.update(pp)
    return idata
```

Update the import at the top of `posterior.py`:

```python
from chromhandler.fitting.model import ModelConfig, model, predictive_model
```

In `fitter.py`, the lazy-compute guards must use the DataTree membership test (ArviZ 1.x): replace `if not hasattr(self.idata, "posterior_predictive"):` ([fitter.py:227](../../../chromhandler/fitting/fitter.py)) with `if "posterior_predictive" not in self.idata.children:` and likewise for `prior_predictive` ([fitter.py:241](../../../chromhandler/fitting/fitter.py)). Reading a group stays `getattr(self.idata, samples_group)` only if that still works on a DataTree — if not, use `self.idata[samples_group]`; verify against the running test.

- [ ] **Step 3: Run the fitter-class test (small fit + predictive + plot)**

Run: `uv run pytest tests/unit/fitting/test_fitter_class.py -v`
Expected: PASS — `plot_fit()` and `plot_prior_predictive()` produce figures (the `"obs"` group now comes from `predictive_model`). If a test asserts the presence of `baseline_intercept`/`baseline_slope` posterior sites, they still exist (now deterministic) so it passes.

- [ ] **Step 4: Lint + typecheck both files**

Run: `uv run ruff check chromhandler/fitting/model.py chromhandler/fitting/posterior.py && uv run pyright chromhandler/fitting/model.py chromhandler/fitting/posterior.py`
Expected: no errors.

- [ ] **Step 5: Run the full fitting unit suite**

Run: `uv run pytest tests/unit/fitting/ -v`
Expected: PASS. Fix any test that referenced the deleted `_compute_baseline_se` or the SE-floor config fields by removing those specific assertions (the concepts no longer exist — clean removal, no shim).

- [ ] **Step 6: Commit**

```bash
git add chromhandler/fitting/model.py chromhandler/fitting/posterior.py tests/unit/fitting/
git commit -m "feat(fitting): predictive_model with conditional baseline draw"
```

### Task 4b: Migrate `save` and `plot_traces` to ArviZ 1.x

These two paths are broken by the ArviZ 1.1.0 upgrade independently of the marginalisation. Bring them green.

**Files:**
- Modify: `chromhandler/fitting/fitter.py` (`save`, `plot_traces`)
- Test: `tests/unit/fitting/test_fitter_class.py::test_fitresult_save_and_load`, `::test_plot_traces_returns_figure`

- [ ] **Step 1: Run the two failing tests to confirm the pre-existing ArviZ errors**

Run: `uv run pytest "tests/unit/fitting/test_fitter_class.py::test_fitresult_save_and_load" "tests/unit/fitting/test_fitter_class.py::test_plot_traces_returns_figure" -v`
Expected: both FAIL with ArviZ 1.x errors (e.g. `InferenceData` attribute / `to_netcdf` / `plot_trace` signature). Record the exact error each raises before changing code.

- [ ] **Step 2: Fix `save` for DataTree**

`FitResult.save` ([fitter.py:58-65](../../../chromhandler/fitting/fitter.py)) currently calls `self.idata.to_netcdf(str(path))`. On an `xarray.DataTree` (ArviZ 1.x) specify the engine:

```python
    def save(self, path: Path | str) -> None:
        """Write the full InferenceData (DataTree) to netCDF."""
        self.idata.to_netcdf(str(path), engine="h5netcdf")
```

If `test_fitresult_save_and_load` reloads via `arviz.InferenceData.from_netcdf` (removed in 1.x), update the test's load call to `arviz.from_netcdf(path)`. Verify the round-trip preserves the `posterior` group.

- [ ] **Step 3: Fix `plot_traces` for the ArviZ 1.x `plot_trace` API**

`plot_traces` ([fitter.py:112-165](../../../chromhandler/fitting/fitter.py)) uses `arviz.rc_context`, `arviz.plot_trace`, and reads `first_ax.figure`. Run the test (Step 1) to see which call breaks under 1.x, then adjust minimally — likely `arviz.plot_trace` returns a different axes container or `rc_context`/`max_subplots` key moved. Do **not** restructure the method; make the smallest change that returns a `matplotlib.figure.Figure`. Verify empirically.

- [ ] **Step 4: Run both tests to verify pass**

Run: `uv run pytest "tests/unit/fitting/test_fitter_class.py::test_fitresult_save_and_load" "tests/unit/fitting/test_fitter_class.py::test_plot_traces_returns_figure" -v`
Expected: PASS.

- [ ] **Step 5: Run the FULL fitting suite to confirm everything is green**

Run: `uv run pytest tests/unit/fitting/ -q`
Expected: 0 failed (was 4 failed at baseline). This is the gate that the ArviZ migration + marginalisation together leave the module fully green.

- [ ] **Step 6: Lint + typecheck + commit**

Run: `uv run ruff check chromhandler/fitting/fitter.py && uv run pyright chromhandler/fitting/fitter.py`

```bash
git add chromhandler/fitting/fitter.py tests/unit/fitting/test_fitter_class.py
git commit -m "fix(fitting): migrate save + plot_traces to ArviZ 1.x DataTree API"
```

---

## Phase 5: A/B assessment on the real fixture (the correctness bar)

### Task 5: Extend `test.py` with golden A/B comparison

**Files:**
- Modify: `test.py`

- [ ] **Step 1: Add the A/B assessment block to `test.py`**

Append after the diagnostics/summary prints in `main()` (replacing the bare `print(result.summary())` tail), an assessment that loads the Phase-0 golden and compares:

```python
    import json
    from pathlib import Path as _P

    golden_path = _P("tests/fixtures/asm_kinetic_series/golden_baseline_model.json")
    golden = json.loads(golden_path.read_text())
    new_summ = result.summary(var_names=["area", "mu", "width", "skew"])
    new = json.loads(new_summ[["mean", "sd", "ess_bulk"]].to_json(orient="index"))
    g = golden["summary"]

    print("\n=== A/B vs golden (current) model ===")
    print(f"{'param':<16}{'old_mean':>12}{'new_mean':>12}{'Δ/σ':>8}{'old_ess':>9}{'new_ess':>9}")
    failures: list[str] = []
    for key in g:
        if key not in new:
            failures.append(f"missing param {key} in new model")
            continue
        om, nm = g[key]["mean"], new[key]["mean"]
        osd = max(g[key]["sd"], new[key]["sd"], 1e-12)
        z = abs(nm - om) / osd                       # disagreement in posterior-sd units
        print(f"{key:<16}{om:>12.4g}{nm:>12.4g}{z:>8.2f}"
              f"{g[key]['ess_bulk']:>9.0f}{new[key]['ess_bulk']:>9.0f}")
        # Tolerance: posteriors of a well-identified parameter must agree to
        # within 0.5 posterior-sd. (Flat baseline prior differs from the old
        # data-derived-SE prior, so exact equality is not expected.)
        if z > 0.5:
            failures.append(f"{key}: |Δ|/σ = {z:.2f} > 0.5  (old={om:.4g} new={nm:.4g})")

    new_diag = result.diagnostics()
    old_diag = golden["diagnostics"]
    print(f"\ndivergences: old={old_diag['n_divergent']} new={new_diag['n_divergent']}")
    print(f"ess_min:     old={old_diag['ess_min_bulk']:.0f} new={new_diag['ess_min_bulk']:.0f}")
    print(f"wall:        old={golden['wall_seconds']:.1f}s new≈(see run)")
    if new_diag["n_divergent"] > old_diag["n_divergent"]:
        failures.append(f"divergences regressed: {old_diag['n_divergent']} -> {new_diag['n_divergent']}")

    if failures:
        print("\n❌ A/B FAILURES:")
        for f in failures:
            print(f"   - {f}")
        raise SystemExit(1)
    print("\n✅ A/B PASS: marginalised model agrees with golden within tolerance.")
```

- [ ] **Step 2: Run the end-to-end A/B**

Run: `uv run python test.py`
Expected: prints the prior table, diagnostics, the A/B comparison table, and either `✅ A/B PASS` (exit 0) or `❌ A/B FAILURES` (exit 1) with the specific parameters that diverged.

- [ ] **Step 3: Interpret the result (decision gate)**

- **PASS** → marginalisation reproduces the peak posteriors; proceed to Step 4.
- **FAIL on `area`/`mu`/`width`/`skew`** → investigate before committing. Likely causes, in order: (a) sign/term error in `marginal_baseline_loglik` (Phase 2 tests would normally catch this — re-check), (b) the old data-derived baseline prior was genuinely biasing the peak areas (this is the *expected* finding if the review's item-1 critique is correct — the disagreement is the bug being fixed, not a regression). Distinguish the two by inspecting `plot_fit()` output and whether the new baseline reconstruction tracks the annotated baseline regions. Record the finding in the review doc.
- **PASS on peaks but divergences/ESS worse** → unexpected (geometry should improve); capture `result.diagnostics()` and the trace plot for follow-up.

- [ ] **Step 4: Commit the assessment harness**

```bash
git add test.py
git commit -m "test(fitting): A/B-assess marginalised baseline vs golden on ASM fixture"
```

---

## Phase 6: Update the review doc with the outcome

### Task 6: Record resolution of review items 1 and 6

**Files:**
- Modify: `docs/superpowers/specs/2026-05-29-fitting-module-integrity-review.md`

- [ ] **Step 1: Mark items 1 and 6 resolved, items 11/13 noted**

Under item 1, add a short resolution note: replaced the data-derived-SE baseline prior with flat-prior analytic marginalisation (this plan); record the A/B finding from Phase 5 Step 3 (agreement within X·σ, or the measured area shift if the old prior was biasing). Under item 6, note `_compute_baseline_se` was deleted. Leave items 2, 3, 4, 5, 7, 8, 9, 10, 12, 14, 15, 16 open.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-05-29-fitting-module-integrity-review.md
git commit -m "docs(fitting): record baseline-marginalisation outcome in integrity review"
```

---

## Self-Review notes (for the executor)

- **Tolerance is a judgment call.** The `0.5·σ` gate in Phase 5 is deliberately strict for *area/mu/width* (well-identified). If the run shows a consistent, explainable area shift, that is the item-1 bias being corrected — not a failure of the implementation. The plan flags this explicitly so the executor does not "fix" a correct result back to the biased one.
- **Prior predictive caveat:** with a flat baseline prior, true prior predictive of `obs` is improper; `predictive_model` uses a data-anchored conditional baseline for the band. This is for visualisation only and is documented in the function docstring.
- **`noise` unchanged:** σ is still the sampled LogNormal site; marginalisation removes only `(a, b)`.
- **No backwards-compat shim:** baseline sample sites and `_compute_baseline_se` are deleted outright per the project's no-dual-paths convention. `baseline_intercept`/`baseline_slope` remain available as deterministic (Rao–Blackwellised) sites, so downstream summary/plot code is unaffected.
- **`PreparedDataset` is untouched** — the projection constants are cheap JAX ops computed inside the model from `dataset.time`/`valid_mask`, traced as compile-time constants.
