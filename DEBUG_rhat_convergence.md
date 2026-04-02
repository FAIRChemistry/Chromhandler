# Rhat Convergence Failure Investigation

**Test:** `tests/integration/test_fitting_speedup.py::test_area_rhat_below_threshold`
**Assertion:** 90th-percentile Rhat of `area_l` / `area_r` ≤ 1.05
**Status:** Still failing after 3 fix attempts. Rhat history:

| Attempt | Change | Rhat (P90 area_l) |
|---------|--------|-------------------|
| 1 | baseline (Plan B model, 500 warmup) | ~2.1 |
| 2 | increased warmup 500 → 1000 | 2.097 |
| 3 | fixed plate naming + simplified artefact area | **1.797** |

---

## What the Plate Fix Changed

**Root cause confirmed (partially):** The model reused the plate name `"traces"` with two different `dim` values in the same model execution:

```python
# dim=-1  (line ~403) — per-trace scalar params
with numpyro.plate("traces", n_trace):
    trace_shift_raw = ...

# dim=-2  (lines ~425, ~439, ~450) — per-trace × per-peak 2D params
with numpyro.plate("traces", n_trace, dim=-2):
    area_dominant = ...
```

NumPyro identifies plates by name. Reusing the same name with different dims while JAX's `pmap` vectorises over chains causes NumPyro to misidentify batch dimensions → incorrect gradient computation → chains explore different regions → Rhat ≈ 2.

**Fix applied:** Replaced all 2D plate usages with uniquely named nested plates:

```python
with numpyro.plate("traces_nonfree", n_trace, dim=-2):
    with numpyro.plate("nonfree_peaks", n_nonfree, dim=-1):
        area_dominant = ...
```

Also removed `area_artefact_trace_offset` entirely — artefact peak areas are now constant across traces (constant column chemistry). This eliminated one more incorrectly-shaped sample site.

**Effect:** Rhat dropped 2.097 → 1.797. Confirms the plate issue was real, but there is a second independent mixing problem.

---

## Remaining Rhat ≈ 1.8 — Hypotheses

### H1: True multimodality in artefact separation/shape (most likely)

The `log_separation_artefact` prior is `Normal(log(w_left_loc), 0.6)`. With only one sample and short chromatogram windows (2.5–3.6 min), the two artefact shoulders (SIH right, Hyp left) may have near-degenerate solutions where:
- Small separation + large artefact area
- Larger separation + small artefact area

NUTS cannot cross the low-probability valley between these modes → chains stay in different modes.

**Evidence to gather:** Run `numpyro.util.format_shapes(model, ...)` to confirm shapes, then plot `az.plot_pair` for `(log_separation_artefact, area_artefact_typical)` to see if there's a bimodal banana.

### H2: `n_trace` is large (≥ 10) and 1000 warmup is insufficient

`BetterFitter.from_handler` collects ALL chromatograms from the selected sample. If the ASM data stores multiple wavelengths per injection (e.g., 260 nm, 280 nm, 310 nm, 360 nm), `handler.samples[:1]` gives `n_trace = n_wavelengths`. Each added trace multiplies the per-trace parameter count.

**Quick diagnostic:**
```python
fitter = BetterFitter.from_handler(handler)
print("n_trace =", fitter.n_traces)
```

If `n_trace >= 10` → warmup needs to scale proportionally (2000–4000 steps).

### H3: `area_log_sigma_low_snr = 0.8` causes funnel geometry

For any trace where SNR < 3, `area_log_sigma = 0.8` → the area prior allows a ×2.2 factor. If the actual posterior width is narrow (data is informative), the ratio prior_width/posterior_width ≈ 10 → NUTS needs O(100×) more steps to explore the prior and settle on the posterior.

**Quick test:** Set `hyperparams=ModelHyperparams(area_log_sigma_low_snr=0.4)` (half the log-sigma) in the test fixture and re-run. If Rhat drops below 1.05, this is the cause.

### H4: `trace_shift_raw` is unidentified when n_trace = 1

If `n_trace = 1`:
```python
trace_shift = trace_shift_scale * (trace_shift_raw - jnp.mean(trace_shift_raw))
# With n_trace=1: trace_shift = scale * (x - x) = 0 always
```

`trace_shift_raw` has a FLAT likelihood (any value gives zero shift). Its posterior = prior = N(0,1). This is fine on its own, but it couples to the apex parameters through the mean-centering. If NUTS correlates `trace_shift_raw` with `apex_loc` exploration, it can slow mixing everywhere.

**Quick test:** Check `fitter.n_traces`. If 1, add `if n_trace > 1:` guard around trace_shift sampling.

---

## Suggested Debugging Order

1. **Print `n_trace`** — determines scale of the problem.

2. **Run ArviZ diagnostics** — the model already prints an ArviZ summary during `fit()`. Capture it:
   ```python
   fitter.fit(num_warmup=1000, num_samples=500, num_chains=8, seed=42,
              save_summary="posterior_summary.txt")
   ```
   Look at the `r_hat` column for ALL parameters (not just area_l). Which raw parameters (`log_w_left`, `log_separation_artefact`, `area_artefact_typical`) have bad Rhat? This pinpoints the problematic site.

3. **Plot pair posteriors:**
   ```python
   import arviz as az
   az.plot_pair(fitter.posterior,
                var_names=["log_separation_artefact", "area_artefact_typical",
                           "log_w_left", "log_w_right"],
                divergences=True)
   ```
   Bimodal clouds → multimodality. Banana shapes → reparameterization needed.

4. **Test with tighter area prior:**
   ```python
   from chromhandler.fitting.data import ModelHyperparams
   fitter = BetterFitter.from_handler(handler,
       hyperparams=ModelHyperparams(area_log_sigma_low_snr=0.4))
   ```

5. **Check divergences:**
   ```python
   print(fitter.mcmc.get_extra_fields()["diverging"].sum())
   ```
   >5% divergences → model geometry issue (funnel / multimodality).

---

## Code Pointers

| File | Line | Issue |
|------|------|-------|
| `chromhandler/fitting/better_model.py:380` | `log_separation_artefact` prior | `Normal(log(w_left_loc), 0.6)` — 0.6 is wide; try 0.3 |
| `chromhandler/fitting/data.py:52` | `area_log_sigma_low_snr = 0.8` | May be too wide for single-sample fit |
| `chromhandler/fitting/better_model.py:403` | `trace_shift_raw` plate | Unidentified when n_trace=1 |
| `chromhandler/fitting/better_model.py:333` | `log_w_left/right` prior scale | `w_prior_log_scale=0.4` floor — may cause wide initial exploration |

---

## What Was Already Fixed (do not revert)

- **Plate naming conflict** (`"traces"` dim=-1 vs dim=-2): fixed in commit on `fix-fit` branch.
- **`area_artefact_trace_offset`**: removed; artefact areas now constant across traces (correct physics, fewer params).
