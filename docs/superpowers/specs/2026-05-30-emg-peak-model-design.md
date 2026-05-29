# EMG peak model + per-peak model selection — design

Add an exponentially-modified Gaussian (EMG) peak model alongside the
existing skew-normal (SN), selectable **per peak**, so a fit can mix peak
shapes. Motivated by genuinely tailing peaks (e.g. the ATP fixture below)
whose **exponential tail** a skew-normal cannot capture — its tail decays
Gaussian-like, so SN under-fits the far tail and the per-trace noise
inflates to compensate.

Branch: `emg-peak-model` (off `fix-fit`).

---

## 1. Scope & non-goals

**In scope:** an `emg.py` math layer; an `EMGPriors` dataclass; per-peak
model selection via `PeakAnnotation.peak_model`; a mixed-type NumPyro
model; an ATP tailing-peak test fixture + loader; tests; a dev script
comparing SN vs EMG on the ATP peak.

**Non-goals (per user):**
- **No formal model comparison (LOO/WAIC).** The marginalised-baseline
  likelihood is a single `numpyro.factor`, not pointwise log-lik, so LOO
  would need extra plumbing. Comparison is by residual/area/eyeball only.
- **No forced common reparameterisation** between SN and EMG. Each model
  reports **area + its native params**.
- **Right-tailing EMG only** (chromatographic tailing). Fronting (left
  tail) is out of scope (τ > 0; fronting would need a mirrored variant).

## 2. EMG parameterisation (native, non-centred)

Standard EMG = Gaussian(`μ`, `σ`) convolved with Exp(`τ`), `τ > 0`,
right-tailing. Per-peak shape params (shared across traces, like SN):
- `emg_mu` — Gaussian centre. Non-centred `Normal(mu_loc, mu_scale)`.
- `emg_sigma` — Gaussian sd. Non-centred **log-space** LogNormal (> 0).
- `emg_tau` — exponential tail constant. Non-centred **log-space**
  LogNormal (> 0).
- `area[trace, peak]` — per-trace, the **same LogNormal-with-fixed-scale
  treatment** already used for SN areas (positivity structural, scale
  data-independent).

**Why log-space for σ and τ:** both must stay positive, and log-space
keeps `τ` away from 0 — the `τ → 0` limit is exactly where EMG degenerates
to a Gaussian and `τ` becomes weakly identified (the EMG analog of the SN
near-symmetry funnel). Non-centred log sampling keeps the HMC geometry
well-conditioned without a hard bound.

**Density — numerically stable, regime-switched.** The naive EMG density
multiplies a growing `exp` by a vanishing `erfc` → `0·∞`. Define `u =
(x−μ)/σ`, `λ = σ/τ`, `w = (λ − u)/√2`. The two equivalent forms each blow
up on *one* side, so switch on the sign of `w`:

- **`w ≥ 0` (core / leading edge):** `f = (1/(2τ)) · exp(−u²/2) ·
  erfcx(w)`, where `erfcx(w) = exp(w²)·erfc(w)`. `exp(−u²/2) ∈ (0,1]`.
- **`w < 0` (right tail):** `f = (1/(2τ)) · exp(λ²/2 − λu) · erfc(w)`
  (`erfc(w<0) ∈ [1,2]`, `exp(λ²/2 − λu) → 0` as `u → ∞`). Stable.

**`jax` has no `erfcx`** (and TFP's JAX substrate is incompatible with the
installed jax 0.9), so implement `erfcx(w)` for `w ≥ 0` ourselves on
`jax.scipy.special.erfc`: `exp(w²)·erfc(w)` for `w < ~5`, switching to the
asymptotic `erfcx(w) ≈ (1/(w√π))(1 − 1/(2w²) + 3/(4w⁴) − …)` for larger
`w` (in **float32**, `exp(w²)` overflows by `w ≈ 9` and `erfc(w)`
underflows there too, so the asymptotic is mandatory). Use the **safe-`where`
pattern** (clamp each branch's input so the *inactive* branch can't
overflow) to keep gradients finite — the EMG counterpart of the SN
boundary guard. Validate against `scipy.stats.exponnorm` across `τ/σ` and
the full `x` range in float32 (see Testing).

## 3. Per-peak model selection

- Add to `PeakAnnotation`: `peak_model: Literal["skew_normal", "emg"] =
  "skew_normal"` (default preserves current behaviour).
- `build_priors` dispatches per annotation: emits `SkewNormalPriors` for
  `"skew_normal"` peaks, a new `EMGPriors` for `"emg"` peaks. The returned
  `priors_list` becomes a heterogeneous list.

## 4. Mixed-type model architecture

**Partition approach** (chosen over a per-peak Python loop):
- Split peak indices into an SN group and an EMG group.
- `_latent_block` samples each group's shape params vectorised within the
  group (SN: `mu/width/skew`; EMG: `emg_mu/emg_sigma/emg_tau`), and `area`
  for all (trace, peak) as today.
- Compute each group's `peak_contrib` (`density_cp` for SN, `density_emg`
  for EMG), scatter back into the full `[n_trace, n_peak]` peak axis, and
  sum — `predicted = baseline + Σ_peak area · density`.
- The per-trace **warp** (`time_shift`, `time_stretch`) and the
  **marginalised baseline** are unchanged and apply to both groups
  (`mu_warped`/`width_warped` computed per group; for EMG, `emg_mu` and
  `emg_sigma`/`emg_tau` warp the same way — shift subtracts, stretch
  divides; `τ` scales with stretch like a width).
- Degenerate cases: an all-SN fit must reproduce today's behaviour
  exactly (empty EMG group → no EMG sites/contrib); an all-EMG fit skips
  the SN group.

This keeps within-group vectorisation while supporting arbitrary mixes.

## 5. Math layer — new `chromhandler/fitting/emg.py`

Parallel to `skew_normal.py`, pure (no NumPyro state):
- `density_emg(x, mu, sigma, tau)` — erfcx-stable density (unit area).
- `mode_emg` / `fwhm_emg` — reporting helpers (numerical, like the SN
  ones; used for prior summaries / reporting, not the HMC path).
- Feature inversion `emg_from_peak_features(apex, fwhm, hwhm_ratio)` →
  `(mu, sigma, tau)`: two measured shape quantities (FWHM + HWHM-ratio)
  plus the apex determine the three EMG params. Reuses the existing
  `compute_window_features` (which already yields apex, FWHM, HWHM-ratio).
  The asymmetry → (σ, τ) split is solved numerically (a small cached
  table mapping HWHM-ratio → τ/σ, analogous to `_asymmetry_table`).

## 6. Priors — `EMGPriors`

Mirror the SN prior pipeline:
- `emg_mu`: loc = mean apex across supported traces; scale = cross-trace
  std (floored to `dt`).
- `emg_sigma`, `emg_tau`: log-space loc = geometric mean of per-trace
  inverted values; fixed weakly-informative log-scales (with `n=1`
  fallbacks), same pattern as `width_log_scale`.
- `area_loc_per_trace`, `area_log_scale`: identical to the SN area prior
  (trapezoid loc, fixed `area_sigma_log`, noise-floor for unsupported).
- `has_support_per_trace`: same gating as SN.

Unsupported / un-invertible traces fall back to geometric defaults
(window-based σ, a modest default τ).

## 7. Exposed / reported quantities

- SN peaks: `mu`, `width`, `skew`, `area` (unchanged).
- EMG peaks: `emg_mu`, `emg_sigma`, `emg_tau`, `area` (native).
No common mode/FWHM is forced; area is directly comparable across models
(both are the unit-area peak integral).

## 8. Test fixture — ATP tailing peak

- Add `tests/fixtures/atp_tailing/ATP_sig.csv` (the provided file).
  Format: column `RT(minutes) - NOT USED BY IMPORT` = retention time
  (min), column `260` = signal (260 nm); `RT(milliseconds)` and `RI`
  ignored. dt ≈ 0.00667 min, ~4950 pts, run 0–33 min.
- Loader helper (test-local or a small `read_signal_csv`) → `(time,
  signal)` arrays → `prepare_dataset`.
- ATP peak: apex ≈ 5.146 min, strong right tail (HWHM R/L ≈ 2.3, 10%-height
  R/L ≈ 3.9). Annotations: peak window `[4.9, 5.7]` (generous on the tail),
  baseline windows in flat regions either side (exact bounds chosen during
  implementation from the data).

## 9. Validation / dev script

A dev script fits the ATP peak **twice** — once `peak_model="skew_normal"`,
once `"emg"` — and overlays the fit + residuals. Expected: EMG captures the
exponential tail with a markedly smaller tail residual (and lower inferred
`noise`) than SN, which under-fits the far tail. This is the evidence that
EMG is warranted and correctly implemented.

## 10. Testing

- **`emg.py` math (no MCMC):** density integrates to `area`; `erfcx` form
  finite and matches a reference EMG (e.g. against a direct
  Gaussian⊛exp numerical convolution or scipy) across `τ/σ` ratios incl.
  large tails; EMG → Gaussian as `τ → 0`; `mode_emg`/`fwhm_emg` vs
  brute-force grid; `emg_from_peak_features` round-trip.
- **Model:** fits a synthetic known-`(μ,σ,τ)` EMG peak and recovers it with
  **0 divergences** (non-centred geometry check); a mixed SN+EMG fit runs;
  an all-SN fit is unchanged vs today.
- **Fixture:** EMG tail-region residual < SN tail-region residual on ATP.

## 11. Risks / mitigations

- **τ↔σ degeneracy** near the Gaussian limit → log-space non-centred +
  weakly-informative τ prior; validated by the 0-divergence synthetic test.
- **No `erfcx` in jax; TFP-JAX broken with jax 0.9** → hand-rolled
  regime-switched density on `jax.scipy.special.erfc` (verified present),
  with the asymptotic `erfcx` branch for large `w` and the safe-`where`
  gradient pattern; validated against `scipy.stats.exponnorm` in float32
  across `τ/σ` ratios and the far tail. This is the single highest-risk
  piece — it gets the most test coverage.
- **Architecture refactor risk** (mixed groups must not regress the
  all-SN path) → explicit all-SN-unchanged test + the full existing
  fitting suite must stay green.
