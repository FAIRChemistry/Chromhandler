"""Bi-skew-normal chromatographic peak model with window-geometry priors.

Public surface
--------------
``log_skew_normal_pdf``  — numerically stable log-density
``skew_normal_pdf``      — density
``mixture_signal``       — area-scaled sum over all components
``model``                — NumPyro model (NUTS-compatible)

Parametrization (main-apex convention)
---------------------------------------
- ``mu_per_trace[t, p]`` = **mode (apex)** of the main peak for trace *t*, peak *p*.
  This is NOT the skew-normal location parameter ξ.  The conversion
  ``ξ = mode − σ·δ·√(2/π)``  (δ = α/√(1+α²)) is applied internally.
- ``separation[sh]`` = full inter-apex distance (main apex → shoulder apex), signed by
  ``shoulder_side``.  **Sampled** with LogNormal prior centred at user-provided value
  (5% CV in log space).  User input provides the prior centre; flexibility allows
  the MCMC to adjust separation if the data suggests it.

Area parametrization (shared shoulder area)
--------------------------------------------
- ``A_main[t, p]``  — per-trace main component area; prior centre = main-only
  window area (shoulder contribution removed in ``priors.py``).
- ``A_sh_shared[sh]`` — **shared scalar** shoulder area, constant across all
  traces.  Physical rationale: the shoulder is a chromatographic artefact whose
  absolute area depends on the instrument/column state, not the analyte
  concentration.  A single shared scalar (rather than n_trace per-trace values)
  encodes this constraint and drastically reduces the parameter count.

Design differences from ``peak_models.py``
------------------------------------------
- **Sigma**: Shoulder sigma is coupled to main sigma via a sampled scaling factor
  (range [0.5, 2.0]).  This allows flexibility while maintaining the geometric
  constraint that both peaks come from the same column.
- **Alpha**: Shoulder uses ``alpha_main[sh_idx]`` directly (same physical shape).
- **Separation**: Fixed from user annotation — not sampled.  Eliminates the
  separation ↔ area_ratio ridge that caused multi-modal posteriors.
- **Shoulder area**: Single shared ``A_sh_shared`` scalar replaces the old
  per-trace ``area_ratio_shoulder`` parameter.
- Sigma and alpha are **shared across traces** (column chemistry is constant);
  mu and A_main vary per trace.
- Python ``if n_shoulder > 0`` guards are compile-time shape checks, not
  dynamic control flow — safe for JAX tracing.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import log_ndtr

numpyro.set_host_device_count(8)

# Mode → location shift: delta × sqrt(2/π), delta = alpha / sqrt(1 + alpha²)
_SQRT_2_OVER_PI: float = float(jnp.sqrt(2.0 / jnp.pi))
_ALPHA_MAX: float = 1.5  # hard bound via tanh transform

# Area prior: LogNormal sigma for A_main → 95% CI ≈ [0.45×, 2.2×] observed area
_AREA_LOG_SIGMA: float = 0.4
# Shared shoulder area prior: LogNormal sigma (30% CV) — allows slow column drift
# while strongly enforcing the constant-artefact constraint across traces.
_SH_AREA_LOG_SIGMA: float = 0.3


# ---------------------------------------------------------------------------
# Skew-normal math
# ---------------------------------------------------------------------------


def log_skew_normal_pdf(
    x: jnp.ndarray,  # [n_trace, n_time]
    mu: jnp.ndarray,  # [n_trace, n_comp]  — skew-normal location param ξ (NOT mode)
    sigma: jnp.ndarray,  # [n_comp]   — shared across traces
    alpha: jnp.ndarray,  # [n_comp]   — shared across traces
) -> jnp.ndarray:
    """Numerically stable log skew-normal density.

    Parameters ``mu`` must be the skew-normal **location parameter ξ**, not the
    mode.  The model passes ``mu_main_loc`` / ``mu_sh_loc`` (converted from
    sampled modes via ``ξ = mode − σ·δ·√(2/π)``).

    Returns
    -------
    jnp.ndarray
        Shape ``[n_trace, n_comp, n_time]``.
    """
    sigma_s = jnp.maximum(sigma, 1e-6)  # [n_comp]
    z = (x[:, None, :] - mu[:, :, None]) / sigma_s[
        None, :, None
    ]  # [n_trace, n_comp, n_time]
    return (
        jnp.log(2.0)
        - jnp.log(sigma_s)[None, :, None]
        - 0.5 * z**2
        - 0.5 * jnp.log(2.0 * jnp.pi)
        + log_ndtr(alpha[None, :, None] * z)
    )


def skew_normal_pdf(
    x: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    alpha: jnp.ndarray,
) -> jnp.ndarray:
    """Skew-normal density. Same shape convention as ``log_skew_normal_pdf``."""
    return jnp.exp(log_skew_normal_pdf(x, mu, sigma, alpha))


def mixture_signal(
    x: jnp.ndarray,  # [n_trace, n_time]
    mu: jnp.ndarray,  # [n_trace, n_comp]
    sigma: jnp.ndarray,  # [n_comp]
    alpha: jnp.ndarray,  # [n_comp]
    A: jnp.ndarray,  # [n_trace, n_comp]
) -> jnp.ndarray:
    """Area-scaled skew-normal mixture, summed over all components.

    Returns
    -------
    jnp.ndarray
        Shape ``[n_trace, n_time]``.
    """
    pdf = skew_normal_pdf(x, mu, sigma, alpha)  # [n_trace, n_comp, n_time]
    return jnp.sum(A[:, :, None] * pdf, axis=1)  # [n_trace, n_time]


# ---------------------------------------------------------------------------
# NumPyro model
# ---------------------------------------------------------------------------


def model(
    x: jnp.ndarray,  # [n_trace, n_time]
    y: jnp.ndarray | None,  # [n_trace, n_time]  or  None (prior predictive)
    # --- peak structure ---
    shoulder_side: jnp.ndarray,  # [n_peak]  int: -1=left, 0=none, +1=right
    shoulder_peak_index: jnp.ndarray,  # [n_shoulder]  indices into peaks
    # --- peak priors (from geometric_priors_to_arrays) ---
    mu_center_loc: jnp.ndarray,  # [n_peak]
    mu_center_scale: jnp.ndarray,  # [n_peak]
    log_sigma_low: jnp.ndarray,  # [n_peak]
    log_sigma_high: jnp.ndarray,  # [n_peak]
    area_per_trace: jnp.ndarray,  # [n_trace, n_peak]  per-trace MAIN component area
    shoulder_area_prior: jnp.ndarray,  # [n_shoulder]  shared shoulder area prior centres
    separation_loc: jnp.ndarray,  # [n_shoulder]  user-provided separation prior centres
    # --- baseline priors (from estimate_baseline) ---
    baseline_intercept_loc: jnp.ndarray,  # [n_trace]
    baseline_intercept_scale: jnp.ndarray,  # [n_trace]
    baseline_slope_loc: jnp.ndarray,  # [n_trace]
    baseline_slope_scale: jnp.ndarray,  # [n_trace]
    # --- noise prior ---
    sigma_y_prior_loc: jnp.ndarray,  # [n_trace]
) -> None:
    """Bayesian bi-skew-normal chromatographic peak model.

    Sampled variables
    -----------------
    ``log_sigma_main``         [n_peak]       — LogUniform sigma (main peaks only)
    ``sigma_scale_raw_shoulder`` [n_shoulder]  — Normal(0, 0.5); mapped to [0.5, 2.0] via tanh;
                                              multiplier for shoulder sigma
    ``alpha_raw_main``  [n_peak]       — Normal(0, 0.5); alpha = ALPHA_MAX*tanh(raw), shared
    ``mu_per_trace``    [n_trace, n_peak] — main peak apex (mode) per trace
    ``A_main``          [n_trace, n_peak] — per-trace main component area
    ``A_sh_shared``     [n_shoulder]      — shared shoulder area (constant across traces)
    ``separation``      [n_shoulder]      — inter-apex distance per shoulder
                                          LogNormal(log(user_input), 0.05); only when shoulders exist
    ``baseline_intercept`` [n_trace]
    ``baseline_slope``     [n_trace]
    ``sigma_y``            [n_trace]      — observation noise

    Deterministic quantities stored in posterior
    --------------------------------------------
    ``sigma_main``      [n_peak]            — exp(log_sigma_main)
    ``sigma_scale_shoulder`` [n_shoulder]   — scaling factor ∈ [0.5, 2.0] for shoulder sigma
    ``alpha_main``      [n_peak]            — ALPHA_MAX * tanh(alpha_raw_main)
    ``alpha_shoulder``  [n_shoulder]        — alpha_main[sh_idx] (same shape, shared)
    ``mu``              [n_trace, n_comp]   — skew-normal ξ params for all components
    ``A``               [n_trace, n_comp]   — areas for all components
    ``A_total``         [n_trace, n_peak]   — A_main + A_sh (diagnostic)
    ``baseline_curve``  [n_trace, n_time]   — reconstructed baseline
    ``mu_y``            [n_trace, n_time]   — posterior predictive mean
    """
    n_trace, _ = x.shape
    n_peak = int(mu_center_loc.shape[0])
    n_shoulder = int(shoulder_peak_index.shape[0])
    n_comp = 2 * n_peak  # main + shoulder slot for every peak

    side = jnp.asarray(shoulder_side, dtype=jnp.float32)  # [n_peak]
    shoulder_enabled = side != 0.0  # [n_peak] bool
    sh_idx = jnp.asarray(shoulder_peak_index, dtype=jnp.int32)  # [n_shoulder]

    # Guard: ensure Uniform bounds are strictly valid
    log_sigma_lo = jnp.asarray(log_sigma_low, dtype=jnp.float32)
    log_sigma_hi = jnp.maximum(
        jnp.asarray(log_sigma_high, dtype=jnp.float32), log_sigma_lo + 1e-4
    )

    # ------------------------------------------------------------------ sigma
    # Shared across traces: peak shape is determined by column chemistry.
    log_sigma_main = numpyro.sample(
        "log_sigma_main", dist.Uniform(log_sigma_lo, log_sigma_hi)
    )  # [n_peak]
    sigma_main = numpyro.deterministic("sigma_main", jnp.exp(log_sigma_main))

    # Shoulder sigma: coupled to main sigma via a scaling factor.
    # Sampled scaling factor ∈ [0.5, 2.0] allows flexibility while keeping shoulder
    # width geometrically related to main peak width (same column → similar shapes,
    # but shoulder may be slightly broader/narrower depending on separation/artifact).
    if n_shoulder > 0:
        # Scale factor sampled in unconstrained space, mapped to [0.5, 2.0] via tanh.
        # Normal(0, 0.5) is centered at 0, which maps to scale=1.0 (no scaling).
        # Range is log-symmetric: [0.5, 2.0] = [1/2, 2], centered at 1.0 in log space.
        sigma_scale_raw_shoulder = numpyro.sample(
            "sigma_scale_raw_shoulder",
            dist.Normal(0.0, 0.5).expand([n_shoulder]),
        )  # [n_shoulder]
        # Map via tanh in log space: log_scale = ln(2) * tanh(x) ∈ [-ln2, ln2]
        # scale = exp(log_scale) ∈ [0.5, 2.0], centered at 1.0
        sigma_scale_shoulder = jnp.exp(
            jnp.log(2.0) * jnp.tanh(sigma_scale_raw_shoulder)
        )  # [n_shoulder]  ∈ [0.5, 2.0]
        numpyro.deterministic("sigma_scale_shoulder", sigma_scale_shoulder)

        # Construct sigma_pair: [n_peak, 2] where col 0=main, col 1=shoulder.
        sigma_pair = (
            jnp.zeros((n_peak, 2))
            .at[:, 0]
            .set(sigma_main)
            .at[sh_idx, 1]
            .set(sigma_scale_shoulder * sigma_main[sh_idx])  # coupled scaling
        )
    else:
        sigma_pair = jnp.stack([sigma_main, jnp.zeros_like(sigma_main)], axis=-1)
    sigma_flat = sigma_pair.reshape(n_comp)  # [n_comp]

    # ------------------------------------------------------------------ alpha
    # Shared skewness: same column chemistry → same elution profile shape.
    alpha_raw_main = numpyro.sample(
        "alpha_raw_main", dist.Normal(0.0, 0.5).expand([n_peak])
    )  # [n_peak]
    alpha_main = numpyro.deterministic(
        "alpha_main", _ALPHA_MAX * jnp.tanh(alpha_raw_main)
    )  # [n_peak]

    if n_shoulder > 0:
        alpha_shoulder = numpyro.deterministic(
            "alpha_shoulder", alpha_main[sh_idx]
        )  # [n_shoulder]
        alpha_pair = (
            jnp.zeros((n_peak, 2))
            .at[:, 0]
            .set(alpha_main)
            .at[sh_idx, 1]
            .set(alpha_shoulder)
        )
    else:
        alpha_pair = jnp.stack([alpha_main, jnp.zeros_like(alpha_main)], axis=-1)
    alpha_flat = alpha_pair.reshape(n_comp)  # [n_comp]

    # ------------------------------------------------------------------ mu
    # Fully per-trace with shared prior.  mu_per_trace[t, p] = apex (MODE) of
    # peak p in trace t — NOT the skew-normal location parameter ξ.
    mu_scale_safe = jnp.maximum(jnp.asarray(mu_center_scale), 1e-6)
    mu_per_trace = numpyro.sample(
        "mu_per_trace",
        dist.Normal(mu_center_loc, mu_scale_safe).expand([n_trace, n_peak]),
    )  # [n_trace, n_peak]

    # ------------------------------------------------------------------ separation
    # Sampled per-shoulder with LogNormal prior centred at user-provided value.
    # 5% CV allows flexibility while keeping separation near the user-provided estimate.
    if n_shoulder > 0:
        sep_loc_safe = jnp.maximum(jnp.asarray(separation_loc, dtype=jnp.float32), 1e-6)
        separation = numpyro.sample(
            "separation",
            dist.LogNormal(jnp.log(sep_loc_safe), 0.05),
        )  # [n_shoulder]
        sep_all = jnp.zeros(n_peak).at[sh_idx].set(separation)
    else:
        sep_all = jnp.zeros(n_peak)

    # Mode positions: main apex at mu_per_trace, shoulder offset by separation.
    mode_main = mu_per_trace  # [n_trace, n_peak]
    sep_offset = sep_all * side  # [n_peak]  signed by shoulder side
    mode_shoulder = mu_per_trace + sep_offset[None, :]  # [n_trace, n_peak]

    # Convert mode → skew-normal location parameter ξ:
    #   ξ = mode − σ·δ·√(2/π),   δ = α / √(1 + α²)
    delta_pair = alpha_pair / jnp.sqrt(1.0 + alpha_pair**2)  # [n_peak, 2]
    mode_shift = sigma_pair * delta_pair * _SQRT_2_OVER_PI  # [n_peak, 2]
    mu_main_loc = mode_main - mode_shift[None, :, 0]  # [n_trace, n_peak]  — ξ_main
    mu_sh_loc = (
        mode_shoulder - mode_shift[None, :, 1]
    )  # [n_trace, n_peak]  — ξ_shoulder

    # Flatten [n_trace, n_peak, 2] → [n_trace, n_comp]
    mu_flat = jnp.stack([mu_main_loc, mu_sh_loc], axis=-1).reshape(n_trace, n_comp)
    numpyro.deterministic("mu", mu_flat)

    # ------------------------------------------------------------------ area
    # A_main: per-trace, free to track analyte concentration.
    # area_per_trace already represents main-only area (shoulder removed in priors.py).
    main_area_safe = jnp.maximum(jnp.asarray(area_per_trace), 1e-8)  # [n_trace, n_peak]
    A_main = numpyro.sample(
        "A_main",
        dist.LogNormal(jnp.log(main_area_safe), _AREA_LOG_SIGMA),
    )  # [n_trace, n_peak]

    if n_shoulder > 0:
        # A_sh_shared: single scalar per shoulder, shared across all traces.
        # Physical rationale: shoulder = chromatographic artefact with constant
        # absolute area — independent of analyte concentration.
        # 30% CV (log sigma = 0.3) allows slow column/instrument drift.
        sh_prior_safe = jnp.maximum(
            jnp.asarray(shoulder_area_prior, dtype=jnp.float32), 1e-8
        )  # [n_shoulder]
        A_sh_shared = numpyro.sample(
            "A_sh_shared",
            dist.LogNormal(jnp.log(sh_prior_safe * 0.5), _SH_AREA_LOG_SIGMA),
        )  # [n_shoulder]

        # Broadcast shared scalar to all traces: A_sh[t, p] = A_sh_shared[sh] ∀ t
        A_sh_per_trace = jnp.broadcast_to(
            A_sh_shared[None, :], (n_trace, n_shoulder)
        )  # [n_trace, n_shoulder]
        A_sh_all = (
            jnp.zeros((n_trace, n_peak)).at[:, sh_idx].set(A_sh_per_trace)
        )  # [n_trace, n_peak]
    else:
        A_sh_all = jnp.zeros((n_trace, n_peak))

    A_sh = A_sh_all * shoulder_enabled[None, :]  # zero out non-shoulder slots
    numpyro.deterministic("A_total", A_main + A_sh)
    A_flat = jnp.stack([A_main, A_sh], axis=-1).reshape(n_trace, n_comp)
    numpyro.deterministic("A", A_flat)

    # ------------------------------------------------------------------ baseline
    baseline_intercept = numpyro.sample(
        "baseline_intercept",
        dist.Normal(
            baseline_intercept_loc, jnp.maximum(baseline_intercept_scale, 1e-6)
        ),
    )  # [n_trace]
    baseline_slope = numpyro.sample(
        "baseline_slope",
        dist.Normal(baseline_slope_loc, jnp.maximum(baseline_slope_scale, 1e-8)),
    )  # [n_trace]
    baseline = baseline_intercept[:, None] + baseline_slope[:, None] * x
    numpyro.deterministic("baseline_curve", baseline)

    # ------------------------------------------------------------------ likelihood
    mu_y = numpyro.deterministic(
        "mu_y", mixture_signal(x, mu_flat, sigma_flat, alpha_flat, A_flat) + baseline
    )
    sigma_y = numpyro.sample(
        "sigma_y",
        dist.LogNormal(jnp.log(jnp.maximum(sigma_y_prior_loc, 1e-6)), 0.5),
    )  # [n_trace]

    if y is not None:
        finite_mask = jnp.isfinite(y)
        numpyro.sample(
            "y",
            dist.Normal(mu_y, sigma_y[:, None]).mask(finite_mask),
            obs=jnp.where(finite_mask, y, 0.0),
        )


# ---------------------------------------------------------------------------
# Parameter names (for ArviZ / posterior extraction)
# ---------------------------------------------------------------------------

SAMPLED_PARAMETER_NAMES = (
    "log_sigma_main",
    "alpha_raw_main",  # shared with shoulder (alpha_shoulder = alpha_main[sh_idx])
    "mu_per_trace",  # [n_trace, n_peak] per-trace main peak apex (mode)
    "A_main",  # [n_trace, n_peak] per-trace main component area
    "A_sh_shared",  # [n_shoulder] shared artefact shoulder area; only when shoulders exist
    "separation",  # [n_shoulder] inter-apex distance per shoulder; only when shoulders exist
    "sigma_scale_raw_shoulder",  # [n_shoulder] raw scale (mapped to [0.5, 2.0]); only when shoulders exist
    "baseline_intercept",
    "baseline_slope",
    "sigma_y",
)
