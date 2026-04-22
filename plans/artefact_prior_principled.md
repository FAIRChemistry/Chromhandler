# Plan: Principled Artefact Prior

Replace the three magic-number constants that govern the `artefact_doublet` width and separation priors with derived, instrument/geometry-based quantities. Pattern mirrors the free-doublet fix already applied.

---

## What changes and why

### Current state in `types.py` / `model.py`

```python
# ModelHyperparams
art_w_prior_center_mult: float = 0.4   # artefact width = 40% of primary HWHM
art_w_log_scale:         float = 0.1   # very tight 10% log-CV
art_sep_min_w_mult:      float = 1.5   # min separation = 1.5 × primary HWHM
```

All three are ad hoc. Together they:

- Bias the artefact width toward 40% of the primary with no physical derivation
- Use a tighter log-scale (0.1) than the primary floor (0.15) — internally inconsistent
  and potentially a NUTS bottleneck if the artefact is unexpectedly broad
- Anchor the minimum separation to the primary width rather than to the artefact itself

---

## Principled replacements

### 1. Artefact width centre — geometric mean of `w_min` and `w_mean_primary`

The only instrument-derived lower bound on any HWHM is the Nyquist-like rule:

```
w_min = 8 · dt / sqrt(8 · ln2)
```

where `dt = _median_dt(x_win)`.  `_median_dt` already exists in `priors.py`
(line 186) but is not called in `build_peak_priors`.

The artefact is, by definition, bounded between `[w_min, w_mean_primary]`.
The maximum-entropy centre for a log-scale parameter on this interval is the
**geometric mean**:

```
w_art_centre = sqrt(w_min · w_mean_primary)
             = exp(0.5 · (log(w_min) + log(w_mean_primary)))
```

No free parameter.  Replaces `art_w_prior_center_mult`.

### 2. Artefact width log-scale — use `hp.w_prior_log_scale` (floor 0.15)

The artefact width is *less* certain than the primary (no direct FWHM
measurement), so the prior must be at least as loose.  Using the same
`hp.w_prior_log_scale` floor (0.15) restores consistency with the primary and
removes `art_w_log_scale` entirely.

### 3. Minimum separation — artefact HWHM itself

The artefact is unidentifiable when its apex is closer to the main peak than
its own half-width — it is completely buried under the main-peak flank.  The
identifiability lower bound is therefore:

```
sep_min = w_art_HWHM = exp(log_w_art)
```

Because `log_w_art` is sampled in step 4a *before* the separation prior in
step 5, the sampled value can be used directly as `sep_min`.  The gradient
flows correctly through NumPyro's `Uniform` log-normalizer.  The existing
`sep_min = min(sep_min, room · 0.5)` safety clamp is retained.

Replaces `art_sep_min_w_mult`.

---

## Net effect

| Removed constant | Replaced by |
|---|---|
| `art_w_prior_center_mult = 0.4` | `sqrt(w_min · w_mean_primary)` — instrument-derived |
| `art_w_log_scale = 0.1` | `hp.w_prior_log_scale` — same floor as primary |
| `art_sep_min_w_mult = 1.5` | `exp(log_w_art)` — sampled artefact HWHM |

One new model input is added: `w_min: jax.Array  # [n_peak]`, computed from
`_median_dt` per peak window.

---

## Files changed

| File | Change |
|---|---|
| [`chromhandler/fitting/priors.py`](../chromhandler/fitting/priors.py) | Call `_median_dt(x_win)` → compute `w_min`; add `w_min: float` field to `GeometricPeakPriors`; include in `geometric_priors_to_arrays` output as `"w_min"` |
| [`chromhandler/fitting/types.py`](../chromhandler/fitting/types.py) | Remove `art_w_prior_center_mult`, `art_w_log_scale`, `art_sep_min_w_mult` from `ModelHyperparams` |
| [`chromhandler/fitting/model.py`](../chromhandler/fitting/model.py) | Add `w_min: jax.Array  # [n_peak]` to `model()` signature; update artefact prior block (step 4a and step 5) |
| [`chromhandler/fitting/fitter.py`](../chromhandler/fitting/fitter.py) | Pass `w_min` from prior arrays to model inputs dict in `_prepare_model_inputs` |

---

## Detailed code changes

### `priors.py` — `build_peak_priors` inner loop

```python
# After computing x_win:
x_win = x[mask]
dt = _median_dt(x_win)
w_min = 8.0 * dt / math.sqrt(8.0 * math.log(2.0))   # Nyquist-like HWHM lower bound
# Store in GeometricPeakPriors(w_min=w_min, ...)
```

### `priors.py` — `geometric_priors_to_arrays`

```python
"w_min": np.array([p.w_min for p in priors], dtype=np.float64),
```

### `model.py` — step 4a (artefact width)

```python
# Before
w_mean_art = 0.5 * (w_left_loc[artefact_idx] + w_right_loc[artefact_idx])
log_w_art = numpyro.sample(
    "log_w_art",
    dist.Normal(jnp.log(hp.art_w_prior_center_mult * w_mean_art), hp.art_w_log_scale),
)

# After
w_mean_art = 0.5 * (w_left_loc[artefact_idx] + w_right_loc[artefact_idx])
w_art_centre = jnp.sqrt(w_min[artefact_idx] * w_mean_art)  # geometric mean
log_w_art = numpyro.sample(
    "log_w_art",
    dist.Normal(jnp.log(w_art_centre), hp.w_prior_log_scale),
)
```

### `model.py` — step 5 (artefact separation)

```python
# Before
sep_min = hp.art_sep_min_w_mult * jnp.minimum(
    w_left_loc[artefact_idx], w_right_loc[artefact_idx]
)

# After — log_w_art already sampled above
sep_min = jnp.exp(log_w_art)   # artefact HWHM: identifiability lower bound
```

---

## Verification

```bash
uv run ruff check chromhandler/fitting/priors.py chromhandler/fitting/types.py chromhandler/fitting/model.py chromhandler/fitting/fitter.py
uv run pyright chromhandler/fitting/priors.py chromhandler/fitting/types.py chromhandler/fitting/model.py chromhandler/fitting/fitter.py
uv run pytest tests/unit/fitting/ -v
```
