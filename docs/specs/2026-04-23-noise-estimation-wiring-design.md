# Noise-Estimation Wiring Design

**Date:** 2026-04-23
**Branch context:** follow-up to `fix-fit` (trace-statistics feature already landed)
**Predecessor plan:** [`docs/plans/2026-04-23-trace-statistics.md`](../plans/2026-04-23-trace-statistics.md)

## Goal

Replace four signal-scale-dependent magic numbers in the Bayesian fitting
pipeline with per-trace `sigma_noise` from `Chromatogram.trace_stats`, making
every noise-aware prior physically grounded in the DER_SNR estimate computed
on the full untruncated trace.

## Scope

All four call sites identified in the earlier code review:

- **A.** `chromhandler/fitting/priors.py::_estimate_snr` — first-difference MAD
  on each peak window (biased upward inside peaks).
- **B.** `chromhandler/fitting/fitter.py::Fitter.noise_prior` — `1e-3 *
  signal_range` numerical floor.
- **C.** `chromhandler/fitting/fitter.py::Fitter.noise_prior` — the
  MAD-in-baseline-regions estimator itself (less reliable than DER_SNR on a
  full trace; fails when baseline annotations are sparse).
- **D.** `chromhandler/fitting/baseline.py` — `_MIN_INTERCEPT_SCALE = 1.0` and
  `_MIN_SLOPE_SCALE = 1e-3` absolute-scale constants used as OLS-SE floors.

## Architecture

Single source of truth: `Chromatogram.trace_stats.sigma_noise`, computed by
`chromhandler/trace_statistics.py::compute_trace_statistics` (DER_SNR,
already landed on `fix-fit`).

Flow into fitting:

```
Handler.compute_trace_statistics()
    → Chromatogram.trace_stats.sigma_noise        [per chromatogram]
Fitter.from_handler(handler)
    → self.trace_sigma_noise: NDArray[n_trace]    [per fitter trace]
Fitter(time, signal)  (direct construction)
    → auto-compute via compute_trace_statistics per row
```

Three fitting-side consumers receive `sigma_noise` as an explicit per-trace
argument (no global fallback, no hidden state):

- `estimate_baseline(time, signal, *, peaks, baselines, sigma_noise, ...)`
- `build_peak_priors(peaks, x, signal, baseline, *, sigma_noise)`
- `Fitter.noise_prior()` — thin passthrough returning `self.trace_sigma_noise`

Strict at the handler boundary: if any chromatogram in the handler still has
`trace_stats is None` after `handler.compute_trace_statistics(overwrite=False)`
(all-NaN signals are silently skipped by the orchestrator today),
`Fitter.from_handler` raises `ValueError` listing the offending chromatogram
IDs. Direct `Fitter(time, signal)` construction auto-computes `sigma_noise`
from its signal rows, so test and dev paths remain ergonomic.

## Components

### `chromhandler/fitting/fitter.py`

- New attribute `trace_sigma_noise: NDArray[np.float64]` of shape `[n_trace]`,
  strictly positive and finite. Sits alongside `trace_chromatogram_ids`.
- `Fitter.__init__`: accept optional `trace_sigma_noise` kwarg.
  - If `None`: auto-compute per row via
    `compute_trace_statistics(time[t], signal[t])` for each `t in range(n_trace)`.
  - If supplied: validate shape `== (n_trace,)`, all finite, all `> 0`.
- `Fitter.from_handler`:
  - Keep the existing defensive `handler.compute_trace_statistics(overwrite=False)` call.
  - Collect `chrom.trace_stats.sigma_noise` per flattened trace into an
    `[n_trace]` numpy array.
  - Raise `ValueError` with format
    `"Fitter.from_handler: chromatograms missing trace_stats after compute_trace_statistics: [<ids>]. These traces are likely all-NaN or too short — drop them via handler.cut_chromatograms or filter upstream."`
    when any surviving chromatogram has `trace_stats is None`.
  - Pass the array through to `Fitter.__init__`.
- `Fitter.noise_prior()`:
  - Delete the MAD-in-baseline-regions block and `1e-3 * signal_range` floor.
  - Return `self.trace_sigma_noise` directly.
- `Fitter.subset(...)` (and any other slicing helpers): slice
  `trace_sigma_noise` consistently with `trace_chromatogram_ids`.

### `chromhandler/fitting/baseline.py`

- `estimate_baseline(...)`: add required kwarg `sigma_noise: jax.Array  # [n_trace]`.
- `_scale_from_se(se, *, floor)`: change `floor` from `float` to an
  `[n_trace]` array. Per-trace floor; delete the cross-trace "fallback"
  median since every trace now has a per-trace floor.
- Intercept floor: `sigma_noise` (per trace).
- Slope floor: `sigma_noise / time_span`, where
  `time_span = time[:, -1] - time[:, 0]` per trace. If `time_span <= 0` for
  some trace (degenerate single-point chromatogram), fall back to using
  `sigma_noise` as the slope floor for that trace.
- Delete module constants `_MIN_INTERCEPT_SCALE` and `_MIN_SLOPE_SCALE`.

### `chromhandler/fitting/priors.py`

- `_estimate_snr`: signature becomes `_estimate_snr(apex_height, sigma_noise)`;
  body becomes `np.maximum(apex_height / sigma_noise, 0.0)`. Delete the
  first-difference MAD logic and the `_FLOAT_MIN` guard (if unused elsewhere).
- `build_peak_priors(...)`: add required kwarg
  `sigma_noise: NDArray[np.float64]  # [n_trace]`; forward to `_estimate_snr`.
- All other internal logic (FWHM, geometry, area priors) unchanged.

## Error handling

- **Strict handler boundary.** `Fitter.from_handler` lists offending IDs and
  points the user at `handler.cut_chromatograms` as the remediation.
- **Direct construction tolerates short/bad rows.**
  `compute_trace_statistics` raises on `<3` finite samples. In the auto-compute
  path, any row failure is re-raised as `ValueError(f"trace row {t}: ...")`,
  letting the caller see which row is bad.
- **Input validation.** `Fitter.__init__` rejects `trace_sigma_noise` with
  wrong shape, non-finite entries, or non-positive entries.
- **Degenerate time span.** Baseline slope floor falls back to
  `sigma_noise` (not `sigma_noise / time_span`) when `time_span <= 0`.
- **Behavior change.** All-NaN chromatograms that used to silently limp
  through fitting now raise at `Fitter.from_handler`. Intentional — matches
  project convention of clean cuts without shims. The error message is
  actionable.

## Testing

All tests use real arrays (no mocks — project convention).

### `tests/unit/fitting/test_noise_plumbing.py` (new)

- `Fitter.__init__` auto-computes `trace_sigma_noise` from rows when not
  supplied; values match `compute_trace_statistics` called directly.
- `Fitter.__init__` rejects `trace_sigma_noise` with wrong shape, non-finite
  values, or non-positive values.
- `Fitter.from_handler` raises `ValueError` naming the offending chromatogram
  when any surviving chromatogram has `trace_stats is None` (all-NaN case).
- `Fitter.from_handler` happy path: `trace_sigma_noise` populated from
  `chrom.trace_stats.sigma_noise` in trace order.
- `Fitter.noise_prior()` returns `self.trace_sigma_noise` unchanged
  (passthrough).
- `Fitter.subset(chromatogram_ids=[...])` slices `trace_sigma_noise`
  consistently with `trace_chromatogram_ids`.

### `tests/unit/fitting/test_baseline.py` (new)

- Clean linear baseline + known `sigma_noise`: intercept scale ≈ OLS SE
  (above floor), slope scale ≈ OLS SE (above floor).
- Single-anchor OLS-degenerate case: intercept scale `== sigma_noise`,
  slope scale `== sigma_noise / time_span`.
- `estimate_baseline` rejects missing `sigma_noise` kwarg.

### `tests/unit/fitting/test_priors.py` (extend — file already exists)

- `build_peak_priors` SNR output equals `apex_height / sigma_noise`
  elementwise on synthetic fixtures; no dependence on in-window first-diffs.
- `build_peak_priors` rejects missing `sigma_noise` kwarg.

### Existing test fixup

- `tests/unit/fitting/test_fitter_diagnostics.py` and
  `tests/unit/fitting/test_fitter_inputs.py` — two direct `Fitter(time, signal)`
  sites. No code change needed (auto-compute path); add one assertion per
  file that `fitter.trace_sigma_noise.shape == (n_trace,)` to lock the
  contract.

## Out of scope

- Backwards-compat shims or deprecation warnings (project convention: clean cuts).
- Lenient fallback modes for missing `trace_stats`.
- Additional `TraceStatistics` fields (`dt_median`, `signal_q01`, etc.).
- `sigma_noise` wiring into the NumPyro model's noise likelihood prior (that
  uses `noise_prior()` already, which becomes a passthrough automatically).
- Revisiting prior widths on `mu`, `alpha`, `area` given the new noise
  estimate. Future plans.

## Acceptance

- All four magic numbers (A–D) replaced.
- `Fitter.from_handler` strict mode enforced with actionable error.
- Direct construction still works without behavior change.
- Full test suite green (unit + integration).
- `uv run ruff check` and `uv run pyright` clean on all four touched files.
