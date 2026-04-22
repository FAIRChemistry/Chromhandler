# Chromhandler Bayesian Peak Fitting — Model Description

*Methods section for the peak fitting pipeline implemented in
`chromhandler/fitting/model.py`, `priors.py`, `baseline.py`, and `fitter.py`.*

---

## 1. Overview

The fitting module estimates peak areas, retention times, and shapes from a
batch of chromatographic traces simultaneously.  Rather than deterministic
integration, a full generative Bayesian model is written in NumPyro/JAX and
sampled with the No-U-Turn Sampler (NUTS).  The model output is a posterior
distribution over all latent variables, yielding credible intervals on areas
and propagating uncertainty through every step.

The key design choice is the **split-normal (bi-Gaussian) peak shape** — a
two-sided Gaussian that assigns an independent half-width to each side of the
apex.  This is both the mathematically simplest model that captures
asymmetric peaks *and* the most interpretable: the two half-widths map
directly onto FWHM and a tailing ratio, both standard quantities in
chromatographic practice.

---

## 2. Peak Shape: the Split-Normal Distribution

### 2.1 Definition

A split-normal peak centred at apex position $\mu$ with left half-sigma
$\sigma_L$ and right half-sigma $\sigma_R$ is defined as:

$$
f(x;\mu,\sigma_L,\sigma_R)
= \frac{2}{\sigma_L + \sigma_R}
  \frac{1}{\sqrt{2\pi}}
  \exp\Bigl(-\frac{(x-\mu)^2}{2\sigma(x)^2}\Bigr),
\quad
\sigma(x) =
\begin{cases}
\sigma_L & x \le \mu 
\sigma_R & x > \mu
\end{cases}
$$

The normalisation constant $2/(\sigma_L+\sigma_R)$ ensures continuity
at $x=\mu$ and that the density integrates to unity.  The peak area is
then:

$$
A = \text{(total signal area)} = a \cdot f \text{ integrated}
  = a \quad \text{where } a \text{ is the sampled area parameter.}
$$

This is evaluated as `log_split_normal_pdf` in `model.py` for numerical
stability, and the final signal is $\mu_y = \sum_k a_k f_k(x) + B(x)$
where $k$ indexes mixture components and $B(x)$ is the baseline.

### 2.2 Parameterisation: log HWHM

The model does **not** sample $(\sigma_L, \sigma_R)$ directly.  Instead it
samples the **log of the left and right half-widths at half-maximum (HWHM)**:

$$
\log w_L = \log\bigl(w_{\mathrm{left}}\bigr), \qquad
\log w_R = \log\bigl(w_{\mathrm{right}}\bigr)
$$

and recovers the half-sigmas deterministically:

$$
\sigma_L = \frac{w_L}{\sqrt{2\ln 2}}, \qquad
\sigma_R = \frac{w_R}{\sqrt{2\ln 2}}
$$

Working in log-space is critical for NUTS geometry: it enforces positivity
without a hard boundary and gives the sampler scale-invariant curvature
regardless of whether a peak is narrow (1 s) or broad (60 s).
$w_L$ and $w_R$ are *orthogonal* descriptors — changing one does not
rescale the other — which further conditions the Hessian.

### 2.3 Interpretation as FWHM and tailing

The two HWHM parameters map onto the standard chromatographic descriptors:


| Quantity                   | Formula                                                       |
| -------------------------- | ------------------------------------------------------------- |
| **FWHM**                   | $w_L + w_R$                                                   |
| **Tailing factor** (USP)   | $\approx w_R / w_L$ (> 1 = tailing, < 1 = fronting)           |
| **Asymmetry factor** $A_s$ | $w_R / w_L$ at 10 % height (same ratio, different height cut) |
| **Symmetric peak**         | $w_L = w_R$ → reduces to a standard Gaussian                  |


A *tailing* peak (solute retained by stationary-phase adsorption sites) has
$w_R > w_L$: the right flank decays slowly while the left rise is sharp.
A *fronting* peak (column overload) has $w_L > w_R$.

The posterior samples of `log_w_left` and `log_w_right` therefore give a
full credible distribution over FWHM and tailing simultaneously, without
any post-hoc moment calculation.

---

## 3. Priors from Window Geometry

All priors are derived from the observable geometry of user-annotated peak
windows — no iterative peak detection or FWHM measurement step is required.

### 3.1 Apex prior

For each annotated window $[t_\mathrm{lo}, t_\mathrm{hi}]$ and each trace,
the apex is located at the maximum-height timepoint within the window.  The
prior location is the **height-weighted centroid** across traces:

$$
\mu_\mathrm{apex} = \frac{\sum_i h_i \cdot t_i^*}{\sum_i h_i}
$$

where $t_i^*$ is the argmax time for trace $i$ and $h_i = \max$ signal in
that window.  Low-intensity traces (below 0.25 % of the maximum apex height)
are excluded as outliers.  The prior scale $\sigma_\mathrm{apex}$ is the
corresponding height-weighted standard deviation, reflecting true
run-to-run retention-time drift rather than peak width.

### 3.2 Half-width priors

For each trace, the left and right HWHM are measured from the apex to the
half-maximum crossing on each side (linear interpolation between adjacent
samples).  The prior location and scale are again height-weighted statistics
across valid traces.

Bounds are enforced:

- **Lower bound** $w_\mathrm{min} = 8 \cdot \Delta t / \sqrt{8\ln 2}$:
a peak must span at least 8 sample intervals (Nyquist-like rule for
meaningful shape inference).
- **Upper bound** $w_\mathrm{max} = W/6$ where $W$ is the window width:
a $\pm 3\sigma$ Gaussian must fit within the half-window, preventing
the model from absorbing baseline into the peak shape.

### 3.3 Area prior

The per-trace area centre is estimated from the Gaussian approximation:

$$
a_\mathrm{gauss} = h_i \cdot \hat\sigma \cdot \sqrt{2\pi},
\qquad \hat\sigma = \frac{w_L + w_R}{2\sqrt{2\ln 2}}
$$

This is used as the lognormal prior centre.  The prior width scales
adaptively with the signal-to-noise ratio (see §5.3).

### 3.4 Baseline priors

Anchor points for baseline estimation are drawn from:

1. Explicit baseline-region annotations (all finite points therein).
2. The bottom 15th-percentile of the inner 20 % edges of each peak window.

An OLS line is fitted per trace through the anchor points.  The resulting
OLS standard errors (multiplied by 4.5 as a conservative prior scale, capped
at 3× the robust across-trace spread) form the Gaussian priors on
`baseline_intercept` and `baseline_slope`.

---

## 4. Location Hierarchy

Peak apices follow a three-level hierarchy that separates population-level
retention time from run-level shifts and local jitter:

$$
\mu_{t,p} = \underbrace{\mu_p^0}*{\text{population apex}}
          + \underbrace{\delta_t}*{\text{run shift}}
          + \underbrace{\epsilon_{t,p}}_{\text{residual jitter}}
$$


| Term             | Description                                                 | Prior                                                                                                         |
| ---------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| $\mu_p^0$        | Global apex for peak $p$                                    | Fixed (point estimate from priors, §3.1)                                                                      |
| $\delta_t$       | Global time-axis shift for run $t$, shared across all peaks | $\delta_t = s \cdot (\tilde\delta_t - \bar{\tilde\delta})$, $\tilde\delta_t \sim \mathcal{N}(0,1)$            |
| $\epsilon_{t,p}$ | Residual per-(run, peak) apex jitter                        | $\epsilon_{t,p} = \sigma_p^\epsilon \cdot \tilde\epsilon_{t,p}$, $\tilde\epsilon_{t,p} \sim \mathcal{N}(0,1)$ |


The run shift $\delta_t$ is mean-centred across traces (non-centred
parameterisation: the raw samples are mean-subtracted before scaling) so
that $\mu_p^0$ always refers to the cross-run average.  The shift scale $s$
is determined from the cross-trace apex variability estimated in the prior
pipeline.  Both $\delta_t$ and $\epsilon_{t,p}$ use non-centred
parameterisations to give NUTS well-conditioned geometry even when shifts are
small and weakly identified.

The half-widths $w_L$, $w_R$ are **shared across traces** (one value per
peak): column chemistry and flow rate are assumed constant within a fitting
subset, so peak shape is a property of the molecule and column, not of the
individual injection.

---

## 5. Priors — Full Specification

### 5.1 Half-width priors

$$
\log w_L^{(p)} \sim \mathcal{N}\bigl(\log \hat w_L^{(p)}, \max(s_L^{(p)}/\hat w_L^{(p)}, 0.15)\bigr)
$$

and identically for $\log w_R^{(p)}$.  The denominator converts the
absolute scale $s_L$ to a log-space coefficient of variation; 0.15 is the
floor (≈ 15 % CV) that prevents an overly tight prior when few traces
contributed to the prior estimate.

### 5.2 Baseline priors

$$
b_t \sim \mathcal{N}(b_t^0, \sigma_{b,t}), \qquad
m_t \sim \mathcal{N}(m_t^0, \sigma_{m,t})
$$

where $b_t^0, m_t^0$ are OLS estimates and $\sigma$ values are OLS standard
errors × 4.5.  The baseline is evaluated at $(x - x_\mathrm{mid})$ (centred
on the midpoint of all windows) for numerical stability:

$$
B_t(x) = b_t + m_t \cdot (x - x_\mathrm{mid})
$$

### 5.3 Area prior — fixed log-sigma

$$
\log a_{t,p} \sim \mathcal{N}\bigl(\log a_{t,p}^0, \sigma_a\bigr), \qquad \sigma_a = 0.4
$$

A single fixed log-space SD is used for all traces and peaks.
$e^{0.4} \approx 1.5$, so the 68 % credible interval spans roughly
$[a^0 / 1.5, a^0 \times 1.5]$ — generous enough to accommodate
moderate baseline uncertainty and peak asymmetry, tight enough to
prevent the sampler exploring implausible areas.

The choice of a *fixed* rather than S/N-adaptive sigma is deliberate.
An adaptive scheme based on S/N computed from the raw data constitutes
empirical Bayes: the data inform the prior width, then also inform the
likelihood, causing subtle underestimation of posterior uncertainty.
Moreover, at high S/N the likelihood dominates the prior regardless of
its width; at low S/N a moderate fixed width is no worse than an
empirical-Bayes loose width.  The simpler fixed parameterisation is
therefore both more honest and equally effective in practice.

### 5.4 Noise prior

$$
\sigma_y^{(t)} \sim \mathrm{LogNormal}\bigl(\log \hat\sigma_y^{(t)}, 0.5\bigr)
$$

$\hat\sigma_y^{(t)}$ is estimated per trace as the normalised MAD of the
baseline-corrected signal within annotated baseline regions (MAD × 1.4826 to
match the Gaussian standard deviation scale).  When no baseline regions are
defined it falls back to the standard deviation of the baseline-corrected
signal.

---

## 6. Peak Modes and Multi-Component Peaks

Three peak modes are supported, selected per annotation:

### 6.1 `single`

One split-normal component.  Sampled parameters: `log_w_left`, `log_w_right`,
`area_dominant`, `apex`.

### 6.2 `artefact_doublet`

A dominant main component plus a small, narrow artefact component (e.g. a
ghost peak, solvent front, or co-eluting impurity at known position).  The
artefact is assumed to be substantially narrower than the main peak and is
positioned at a sampled separation from the main apex.

- **Artefact width**: $\log w_\mathrm{art} \sim \mathcal{N}(\log(0.4\bar w), 0.10)$
— centred at 40 % of the mean primary HWHM with a tight 10 % log-scale.
- **Separation**: $\log \Delta \sim \mathrm{Uniform}(\log \Delta_\mathrm{min}, \log r)$
where $\Delta_\mathrm{min} = 1.5\min(w_L, w_R)$ and $r$ is the window
room on the artefact side minus a safety margin.
- **Artefact area**: hierarchical — a population log-mean is sampled, then
per-trace deviations with log-scale 0.15.  This partial pooling is
appropriate when the artefact is expected to be consistent across runs
but not identical.
- The user specifies whether the artefact falls to the left or right of the
main apex via `PeakAnnotation.artefact_side`.

### 6.3 `free_doublet`

Two fully independent split-normal components with a free area split.

- **Second-component width**: $\log w_{L,2}, \log w_{R,2} \sim \mathcal{N}(\log(0.6w), 0.5)$
— weakly anchored near 60 % of the primary HWHM.
- **Separation**: $\log \Delta \sim \mathcal{N}(\log(2w_L), 0.4)$ — prior
centred at twice the left HWHM, corresponding to baseline resolution of
~1.5.
- **Area split**: $f_L \sim \mathrm{Beta}(2,2)$ — symmetric prior on the
fraction assigned to the left component, weakly pulling toward equal split
without ruling out strongly asymmetric cases.

---

## 7. Likelihood

The observed signal is modelled as:

$$
y_{t,i} \sim \mathcal{N}\bigl(\mu_{t,i}, \sigma_y^{(t)}\bigr)
$$

where the deterministic mean is the sum of all mixture components plus
baseline:

$$
\mu_{t,i} = B_t(x_i) + \sum_k a_{t,k} f\bigl(x_i; \mu_{t,k}, \sigma_{L,k}, \sigma_{R,k}\bigr)
$$

NaN-padded timepoints (from rectangular padding of unequal-length traces)
are masked out of the likelihood via NumPyro's `.mask(finite_mask)`.

The likelihood is evaluated only over the **union of baseline regions and
peak windows** — timepoints outside annotated regions are excluded, which
reduces the effective data dimensionality and prevents the model fitting
irrelevant baseline structure between peaks.

---

## 8. Inference

Posterior inference uses the **No-U-Turn Sampler** (NUTS) as implemented in
NumPyro, compiled to XLA by JAX.  Key implementation details:

- `numpyro.set_host_device_count(8)` — up to 8 parallel chains on CPU.
- All intermediate arrays are plain local JAX variables (no `numpyro.deterministic`
sites) to minimise leapfrog overhead.
- Non-centred parameterisations are used throughout: trace shifts, apex
offsets, and artefact log-areas are sampled as unit-Normal variates and
rescaled, which gives NUTS well-conditioned geometry even when the
likelihood gradient is weak (low S/N, weakly identified shifts).
- `compute_derived_quantities` in `model.py` reconstructs all geometric
quantities (apex positions, half-sigmas, component areas) from the raw
posterior samples for use in visualisation and area reporting.

---

## 9. Summary of Sampled Parameters


| Parameter                       | Shape                   | Prior                                               | Interpretation                                   |
| ------------------------------- | ----------------------- | --------------------------------------------------- | ------------------------------------------------ |
| `log_w_left`                    | `[n_peak]`              | $\mathcal{N}(\log \hat w_L, \mathrm{CV})$           | Log left HWHM, shared across traces              |
| `log_w_right`                   | `[n_peak]`              | $\mathcal{N}(\log \hat w_R, \mathrm{CV})$           | Log right HWHM, shared across traces             |
| `trace_shift_raw`               | `[n_trace]`             | $\mathcal{N}(0,1)$ (non-centred)                    | Global run-level retention-time shift            |
| `apex_offset_raw`               | `[n_trace, n_peak]`     | $\mathcal{N}(0,1)$ (non-centred)                    | Residual per-run, per-peak jitter                |
| `area_dominant`                 | `[n_trace, n_nonfree]`  | $\mathrm{LogN}(\log a^0, \sigma_a)$                 | Per-trace area of single/artefact primary peak   |
| `area_total_free`               | `[n_trace, n_free]`     | $\mathrm{LogN}(\log a^0, \sigma_a)$                 | Per-trace total area of free doublets            |
| `area_frac_left_free`           | `[n_trace, n_free]`     | $\mathrm{Beta}(2,2)$                                | Fraction of doublet area in left component       |
| `log_area_art_mean`             | `[n_artefact]`          | $\mathcal{N}(\log a_\mathrm{art}^0, 0.3)$           | Population log-area of artefact component        |
| `log_area_art_raw`              | `[n_trace, n_artefact]` | $\mathcal{N}(0,1)$ (non-centred)                    | Per-trace artefact area deviation                |
| `log_w_art`                     | `[n_artefact]`          | $\mathcal{N}(\log(0.4\bar w), 0.1)$                 | Artefact component width                         |
| `log_separation_artefact`       | `[n_artefact]`          | $\mathrm{Uniform}(\log\Delta_\mathrm{min}, \log r)$ | Artefact apex offset from main apex              |
| `log_w_left_2`, `log_w_right_2` | `[n_free]`              | $\mathcal{N}(\log(0.6 w), 0.5)$                     | Free doublet second-component widths             |
| `log_separation_free`           | `[n_free]`              | $\mathcal{N}(\log(2 w_L), 0.4)$                     | Free doublet apex separation                     |
| `baseline_intercept`            | `[n_trace]`             | $\mathcal{N}(b^0, \sigma_b)$                        | Per-trace baseline intercept at $x_\mathrm{mid}$ |
| `baseline_slope`                | `[n_trace]`             | $\mathcal{N}(m^0, \sigma_m)$                        | Per-trace baseline slope                         |
| `sigma_y`                       | `[n_trace]`             | $\mathrm{LogN}(\log\hat\sigma_y, 0.5)$              | Per-trace observation noise SD                   |


---

## 10. Scientific Assessment: Principled Choices vs. Magic Numbers

This section gives an honest evaluation of which model decisions rest on
statistical or physical grounds, and which are heuristic constants that a
reviewer or collaborator should treat with scepticism.

### 10.1 Well-justified design decisions


| Decision                                  | Justification                                                                                                                                                             |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Split-normal peak shape                   | Minimum-parameter model that captures asymmetry; mode is exact at apex; continuous and differentiable everywhere; directly encodes FWHM + tailing ratio                   |
| Log-parameterisation of HWHM              | Positive-definite by construction; scale-invariant curvature for NUTS; $w_L$ and $w_R$ are orthogonal descriptors (changing one does not rescale the other)               |
| Non-centred parameterisations throughout  | Standard HMC best practice (Betancourt & Girolami 2015); prevents funnel geometry when shifts or jitter are weakly identified                                             |
| LogNormal area prior                      | Positive-definite; log-space SD has direct multiplicative interpretation                                                                                                  |
| Mean-centring of trace shifts             | Hard identifiability constraint; $\mu_p^0$ is uniquely defined as the cross-run mean                                                                                      |
| Baseline centred at $x_\mathrm{mid}$      | Reduces intercept–slope posterior correlation; standard OLS conditioning trick                                                                                            |
| Upper width bound $w_\mathrm{max} = W/6$  | Derived from requiring $\pm 3\sigma$ to fit inside the half-window; 99.7 % Gaussian coverage. Principled.                                                                 |
| Masking likelihood to annotated windows   | Correct: the model makes no claim about timepoints outside annotations, preventing baseline contamination of peak parameters                                              |
| Hierarchical artefact area                | Partial pooling is the statistically correct response when an artefact is expected to be consistent across runs but not identical                                         |
| Beta(2, 2) area fraction for free doublet | Symmetric, unimodal; weakly regularises toward equal split without ruling out asymmetric cases; maximum-entropy choice for a bounded quantity with a symmetry expectation |


### 10.2 Magic numbers — unjustified constants

These constants have no derivation from first principles.  Each is a
heuristic that worked in development but whose sensitivity has not been
formally characterised.

#### Half-width prior floor

```python
w_prior_log_scale = 0.15   # model/types.py
```

A 15 % log-CV floor on width priors prevents pathologically tight priors
when few traces contributed to the prior estimate.  The value 0.15 is
arbitrary.  A principled alternative: tie the floor to the instrument's
measured retention-time reproducibility (e.g. $\sigma_\mathrm{RT} / \hat w$
from system suitability runs).

#### S/N-adaptive area prior — four constants, no derivation

```python
area_log_sigma_high_snr  = 0.30   # tight for S/N ≥ 10
area_log_sigma_low_snr   = 0.80   # loose for S/N ≤ 3
area_snr_threshold_high  = 10.0
area_snr_threshold_low   = 3.0
```

The S/N thresholds echo analytical chemistry LOQ/LOD rules of thumb.  The
mapping from S/N to log-sigma is a linear interpolation with no statistical
derivation.  Furthermore, S/N is computed from the data *before* fitting,
making this an empirical Bayes approximation: the data inform the prior,
then also the likelihood.  This can lead to subtle underestimation of
posterior uncertainty.  **Recommend replacing with a single fixed
log-sigma ≈ 0.4** (see §5.3 in the description section and the
[companion discussion](model_summa.md)).

#### ~~Baseline prior inflation multiplier~~ — **FIXED**

~~`_PRIOR_SE_MULTIPLIER = 4.5` and `_GLOBAL_SCALE_CAP = 3.0` removed from
`baseline.py`.~~  OLS standard errors are now used directly as prior scales
(multiplier = 1.0).  `_robust_scale` was deleted as it only existed to serve
the cap.  See commit history.

#### Baseline anchor selection

```python
_DEFAULT_PERCENTILE    = 15.0   # baseline.py
_DEFAULT_EDGE_FRACTION = 0.20
_MIN_EDGE_POINTS       = 6
```

Taking the bottom 15th-percentile of the inner 20 % of each window edge is a
reasonable heuristic to find baseline-level points, but both thresholds are
uncalibrated.  For narrow peaks (few timepoints) these interact poorly.

#### Artefact doublet shape and separation

```python
art_w_prior_center_mult = 0.4   # artefact width = 40 % of main HWHM
art_w_log_scale         = 0.1   # very tight — 10 % log-CV
art_sep_min_w_mult      = 1.5   # min separation = 1.5 × HWHM
```

These three constants jointly define the artefact sub-model and have no
physical derivation.  The 40 % width and 1.5× separation reflect the
authors' experience, not a chromatographic mass-transfer model.  The 10 %
log-CV for artefact width is inconsistent with the 15 % floor used for main
peaks, and is narrow enough to create a potential NUTS bottleneck if an
artefact is unexpectedly broad.

#### ~~Free doublet second-component width~~ — **FIXED**

~~`log_w_left_2 ~ Normal(log(0.6 * w_left_loc), 0.5)`~~  Both the 0.6
centre multiplier and fixed 0.5 scale have been replaced with the same
data-derived formula used for the primary peak:

```python
w_left_log_scale_2 = max(w_left_scale / w_left_loc, hp.w_prior_log_scale)
log_w_left_2 ~ Normal(log(w_left_loc), w_left_log_scale_2)
# identically for log_w_right_2
```

No assumed size asymmetry; prior geometry is now consistent between both
components.

#### Observation noise prior width

```python
sigma_y ~ LogNormal(log(sigma_y_hat), 0.5)
```

The 0.5 log-scale allows the fitted noise to deviate by a factor of ≈ 1.65
from the MAD estimate.  This is reasonable but not derived from the detector's
noise characteristics.  A more principled approach: use the known detector
dark-current noise as a hard lower bound and let the prior be weakly
informative above that.

#### Lower width bound

```python
sigma_low = 8 * dt / sqrt(8 * ln2)   # priors.py
```

The "8-point rule" requires at least 8 samples to span the FWHM.  This
echoes the signal-processing convention that you need roughly 5–10 samples
per peak for reliable integration, but 8 is a rule of thumb.  Some
instruments routinely acquire at 4 points/peak; others at 40.

#### ~~Dead code: `free_sep_loc_mult`~~ — **FIXED**

`ModelHyperparams.free_sep_loc_mult` has been removed from `types.py`.

### 10.3 Structural limitations

**Label switching in free doublets.**  The model assigns components as "left"
and "right" by construction but does not enforce $\mu_L < \mu_R$ as a hard
constraint.  When two components are nearly equal in size and position, the
posterior can be bimodal (the sampler finds both $(L, R)$ and $(R, L)$
configurations).  The Beta(2, 2) area prior does not prevent this.  A hard
ordering constraint on separation (force $\Delta > 0$) or post-hoc
label-sorting is needed for reliable marginal posteriors on individual
component areas.

**Gaussian likelihood, homoscedastic noise.**  A UV absorbance detector
operating near saturation follows Poisson-like statistics (variance ∝ mean
signal).  At high absorbance the current constant-variance Gaussian
likelihood underestimates uncertainty on the peak apex and overestimates it
on the flanks.  A Student-t likelihood (heavier tails) would also be more
robust to isolated spikes without modelling them explicitly.

**Linear baseline per window.**  This is adequate for isocratic
chromatography over narrow windows, but gradient elution produces curved,
instrument-dependent baselines.  The OLS prior provides no protection
against a systematically curved baseline being absorbed into peak shape
parameters.

**Peak shape vs. physical mechanism.**  The split-normal is an empirical
shape model.  The standard mechanistic model for a tailing chromatographic
peak is the **Exponentially Modified Gaussian (EMG)**:

$$
h(t) = \frac{A}{2\tau}\exp\Bigl(\frac{\sigma^2}{2\tau^2} - \frac{t-\mu}{\tau}\Bigr)\mathrm{erfc}\Bigl(\frac{\sigma/\tau - (t-\mu)/\sigma}{\sqrt{2}}\Bigr)
$$

where $\tau$ is the time constant of an exponential decay representing
first-order mass-transfer kinetics or adsorption.  The EMG reduces exactly
to a Gaussian when $\tau \to 0$ and its tailing parameter maps directly onto
a physical rate constant.  The split-normal approximates EMG shapes well
empirically but $w_R/w_L$ has no direct physical interpretation.  For
peer-reviewed chromatographic method development the EMG (or bi-EMG) would
be a stronger justification.

### 10.4 Summary scorecard


| Component                                | Status                                                                      |
| ---------------------------------------- | --------------------------------------------------------------------------- |
| Peak shape (split-normal)                | ✅ Principled, interpretable — but EMG is more mechanistic                   |
| HWHM log-parameterisation                | ✅ Principled                                                                |
| Location hierarchy                       | ✅ Principled                                                                |
| Non-centred parameterisations            | ✅ Best practice                                                             |
| Upper width bound                        | ✅ Derived                                                                   |
| Hierarchical artefact area               | ✅ Principled                                                                |
| Area prior (fixed σ = 0.4)               | ✅ Fixed, principled; $e^{0.4} \approx 1.5\times$ multiplicative uncertainty |
| Baseline prior multiplier (4.5×)         | ✅ Fixed — raw OLS SE used directly; cap removed                             |
| Baseline anchor selection                | ⚠️ Heuristic (15th percentile, 20 % edge)                                   |
| Lower width bound (8-point rule)         | ⚠️ Rule of thumb                                                            |
| Artefact shape constants (0.4, 0.1, 1.5) | ❌ Uncalibrated                                                              |
| Free doublet second component (0.6, 0.5) | ✅ Fixed — data-derived prior, same formula as primary peak                  |
| Noise prior width (0.5)                  | ⚠️ Arbitrary                                                                |
| Label switching (free doublet)           | ❌ Not handled                                                               |
| Gaussian homoscedastic noise             | ⚠️ Approximate for high-signal traces                                       |
| Linear baseline                          | ⚠️ Adequate for isocratic only                                              |
| `free_sep_loc_mult` dead code            | ✅ Fixed — field removed from `ModelHyperparams`                             |


---

## 11. Derived Quantities from Posterior

After sampling, `compute_derived_quantities` reconstructs:

- **Apex positions** per trace and peak: $\mu_{t,p} = \mu_p^0 + \delta_t + \epsilon_{t,p}$
- **Half-sigmas**: $\sigma_{L/R} = w_{L/R}/\sqrt{2\ln 2}$
- **Component areas**: assembled per the peak-mode logic into `area_l`, `area_r` per component
- **Posterior predictive curves**: `PosteriorCurves` dataclass stores per-trace median and HDI bands for the total fit, baseline, and each left/right component

Peak areas reported to the user are the posterior medians of the dominant
component areas (for `single` and `artefact_doublet`) or the sum of both
component areas (for `free_doublet`), with 5th/95th percentile credible
intervals.

`save_summary()` and `plot_traces()` now use a **denylist** (`INTERNAL_POSTERIOR_VARS`
in `model.py`) rather than an allowlist to select variables for display.  Internal
non-centred parameterisation raw samples (`trace_shift_raw`, `apex_offset_raw`,
`log_area_art_raw`) and intermediate geometric arrays (`sl_base`, `sr_base`,
`apex_l/r`, `sl/sr_l/r`) are excluded; everything else — including any new
sampled site added in future — appears automatically.