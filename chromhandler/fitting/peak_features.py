"""Peak feature extraction and skew-normal prior estimation for chromatographic fitting.

Pipeline
--------
1. ``compute_peak_fwhm_features``  — vectorised FWHM + KDE apex gating.
2. ``compute_skew_normal_guess``   — convert gated FWHM to skew-normal initialisers
   and population priors.
3. ``build_two_stage_component_initializers`` — main-pass + residual shoulder-pass
   initializers in canonical component shape ``[n_trace, n_peak, 2]``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Final

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import log_ndtr

from .peak_models import skew_normal_pdf

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_FLOAT_MIN: Final = 1e-12
_GAUSSIAN_HWHM_FACTOR: Final = math.sqrt(2.0 * math.log(2.0))
_SQRT_TWO_PI: Final = math.sqrt(2.0 * math.pi)
_MAD_TO_SCALE: Final = 1.4826
_SILVERMAN_FACTOR: Final = 1.06
_MIN_KDE_GRID: Final = 64
_N_FEATURE_COMPONENTS: Final = 2


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ApexGate:
    """KDE-derived gate for the dominant species in a peak window.

    When embedded in ``FwhmFeatures``, scalar-like gate fields are carried as
    ``[n_peak, n_component]`` with ``n_component == 2`` and ``keep_mask`` has
    shape ``[n_trace, n_peak, n_component]``.
    Component ``0`` holds the active single-peak values for now; component ``1``
    is reserved and remains padding (NaN/False) until double-peak inference is
    enabled.

    When returned directly by ``kde_apex_gate``, all fields are scalars / 1-D
    arrays for a single peak.
    """

    center: jnp.ndarray
    scale: jnp.ndarray
    low: jnp.ndarray
    high: jnp.ndarray
    bandwidth: jnp.ndarray
    keep_mask: jnp.ndarray  # bool


@dataclasses.dataclass(frozen=True)
class FwhmFeatures:
    """Per-trace/per-peak FWHM features after KDE apex gating.

    All matrix-valued fields carry an explicit component axis with shape
    ``[n_trace, n_peak, n_component]`` where ``n_component == 2``.
    Gated fields (apex_time, left_time, right_time, fwhm) are ``NaN`` for
    traces that did not pass the gate. ``*_all`` fields hold pre-gate values
    for all valid traces (for plotting and diagnostics). For the current
    single-peak path, component ``0`` is populated and component ``1`` is NaN.
    """

    apex_time: jnp.ndarray  # gated
    left_time: jnp.ndarray  # gated
    right_time: jnp.ndarray  # gated
    fwhm: jnp.ndarray  # gated
    valid: jnp.ndarray  # bool — detected pre-gate
    apex_time_all: jnp.ndarray  # pre-gate
    left_time_all: jnp.ndarray  # pre-gate
    right_time_all: jnp.ndarray  # pre-gate
    fwhm_all: jnp.ndarray  # pre-gate
    gate: ApexGate  # [n_peak, n_component] scalars; keep_mask [n_trace, n_peak, n_component]

    def to_dict(self) -> dict[str, jnp.ndarray]:
        """Flatten to a dict for plotting/diagnostics export."""
        return {
            "apex_time": self.apex_time,
            "left_time": self.left_time,
            "right_time": self.right_time,
            "fwhm": self.fwhm,
            "valid_trace": self.valid,
            "apex_time_all": self.apex_time_all,
            "left_time_all": self.left_time_all,
            "right_time_all": self.right_time_all,
            "fwhm_all": self.fwhm_all,
            "gate_keep": self.gate.keep_mask,
            "gate_center": self.gate.center,
            "gate_scale": self.gate.scale,
            "gate_low": self.gate.low,
            "gate_high": self.gate.high,
            "gate_bandwidth": self.gate.bandwidth,
        }


@dataclasses.dataclass(frozen=True)
class PeakPriors:
    """Population-level Bayesian prior parameters, shape ``[n_peak]``.

    ``*_loc`` is the MAD-robust median across gated traces; ``*_scale`` is the
    MAD-based spread.  These parameterise the prior distributions in the model.
    """

    mu_loc: jnp.ndarray
    mu_scale: jnp.ndarray
    sigma_loc: jnp.ndarray
    sigma_scale: jnp.ndarray
    alpha_loc: jnp.ndarray
    alpha_scale: jnp.ndarray
    area_loc: jnp.ndarray
    area_scale: jnp.ndarray


@dataclasses.dataclass(frozen=True)
class BiSkewPriors:
    """Population priors for bi-skew-normal inference, shape ``[n_peak]``/``[n_peak, 2]``."""

    mu_center_loc: jnp.ndarray  # [n_peak]
    mu_center_scale: jnp.ndarray  # [n_peak]
    separation_low: jnp.ndarray  # [n_peak]
    separation_high: jnp.ndarray  # [n_peak]
    sigma_loc: jnp.ndarray  # [n_peak, 2]
    sigma_scale: jnp.ndarray  # [n_peak, 2]
    alpha_loc: jnp.ndarray  # [n_peak, 2]
    alpha_scale: jnp.ndarray  # [n_peak, 2]
    area_total_loc: jnp.ndarray  # [n_peak]
    area_total_scale: jnp.ndarray  # [n_peak]
    area_split_alpha: jnp.ndarray  # [n_peak]
    area_split_beta: jnp.ndarray  # [n_peak]


@dataclasses.dataclass(frozen=True)
class SkewNormalGuess:
    """Per-(trace, peak) skew-normal chain initialisers with population priors.

    The ``mu / sigma / alpha / area`` matrices are ``[n_trace, n_peak]``.
    Traces in ``keep`` have individually estimated values; the rest are
    fallback-filled from ``priors.{param}_loc``.
    """

    mu: jnp.ndarray  # [n_trace, n_peak] xi parameter
    sigma: jnp.ndarray  # [n_trace, n_peak]
    alpha: jnp.ndarray  # [n_trace, n_peak]
    area: jnp.ndarray  # [n_trace, n_peak]
    keep: jnp.ndarray  # [n_trace, n_peak] bool — had valid individual FWHM
    priors: PeakPriors  # [n_peak] — for prior distribution definition


@dataclasses.dataclass(frozen=True)
class ComponentInitializers:
    """Component-space initialiser matrices, shape ``[n_trace, n_peak, n_component]``."""

    mu_init: jnp.ndarray
    sigma_init: jnp.ndarray
    alpha_init: jnp.ndarray
    A_init: jnp.ndarray
    shoulder_keep: jnp.ndarray | None = None  # [n_trace, n_peak] bool
    shoulder_split_time: jnp.ndarray | None = None  # [n_trace, n_peak]
    shoulder_side_points: jnp.ndarray | None = None  # [n_trace, n_peak] int

    def to_dict(self) -> dict[str, jnp.ndarray]:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


# ---------------------------------------------------------------------------
# Public utilities
# ---------------------------------------------------------------------------


def median_and_scale(
    values: jnp.ndarray,
    *,
    scale_floor: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(median, MAD-based scale)`` for finite values in a 1-D array.

    Args:
        values: 1-D array (may contain NaN / Inf).
        scale_floor: Lower bound on the returned scale.

    Returns:
        ``(location, scale)`` as ``float32`` scalars; both NaN when no finite
        values exist.

    Example::

        loc, scale = median_and_scale(jnp.array([1.0, 2.0, 3.0, jnp.nan]))
        # loc ≈ 2.0, scale ≈ 1.48
    """
    arr = jnp.asarray(values, dtype=jnp.float32)
    if arr.ndim != 1:
        raise ValueError("values must be 1-D")
    finite = arr[jnp.isfinite(arr)]
    if int(finite.size) == 0:
        nan = jnp.asarray(jnp.nan, dtype=jnp.float32)
        return nan, nan
    location = jnp.median(finite)
    mad = jnp.median(jnp.abs(finite - location))
    scale = jnp.maximum(_MAD_TO_SCALE * mad, scale_floor)
    return location.astype(jnp.float32), scale.astype(jnp.float32)


def kde_apex_gate(
    apex_times: jnp.ndarray,
    *,
    weights: jnp.ndarray | None = None,
    n_sigma: float = 3.0,
    bandwidth_scale: float = 1.0,
    scale_floor: float = 1e-4,
    n_grid: int = 256,
) -> ApexGate:
    """Gate apex times around the dominant KDE mode.

    Uses a Gaussian KDE to identify the main mode of the apex-time distribution,
    then keeps values within ``n_sigma`` local standard deviations of that mode.

    Args:
        apex_times: Per-trace apex times (may contain NaN).
        weights: Optional non-negative weights (e.g. apex heights) with the same
            shape as ``apex_times``. If provided, KDE mode selection and local
            scale estimation are weighted by these values.
        n_sigma: Gate half-width in local standard deviations.
        bandwidth_scale: Scale factor on Silverman's bandwidth.
        scale_floor: Minimum bandwidth and local scale.
        n_grid: KDE grid resolution.

    Returns:
        :class:`ApexGate` with scalar statistics and a boolean ``keep_mask``.

    Example::

        times = jnp.array([3.1, 3.0, 3.2, 7.5, 3.05])
        gate = kde_apex_gate(times, n_sigma=3.0)
        # gate.center ≈ 3.1, gate.keep_mask → [T, T, T, F, T]
    """
    apex_array = jnp.asarray(apex_times, dtype=jnp.float32).reshape(-1)
    if weights is None:
        weight_array = jnp.ones_like(apex_array, dtype=jnp.float32)
    else:
        weight_array = jnp.asarray(weights, dtype=jnp.float32).reshape(-1)
        if weight_array.shape != apex_array.shape:
            raise ValueError("weights must have the same shape as apex_times.")
        weight_array = jnp.where(
            jnp.isfinite(weight_array) & (weight_array > 0.0), weight_array, 0.0
        )

    finite = jnp.isfinite(apex_array) & (weight_array > 0.0)
    keep = jnp.zeros_like(apex_array, dtype=bool)

    _nan = jnp.asarray(jnp.nan, dtype=jnp.float32)
    if int(jnp.sum(finite)) == 0:
        return ApexGate(
            center=_nan,
            scale=_nan,
            low=_nan,
            high=_nan,
            bandwidth=_nan,
            keep_mask=keep,
        )

    values = apex_array[finite]
    weights_valid = weight_array[finite]
    w_sum = jnp.maximum(jnp.sum(weights_valid), _FLOAT_MIN)
    w_norm = weights_valid / w_sum

    # Weighted effective sample size and weighted spread for bandwidth scaling.
    n_eff = float((w_sum**2) / jnp.maximum(jnp.sum(weights_valid**2), _FLOAT_MIN))
    n_eff = max(n_eff, 2.0)
    mean_raw = jnp.sum(w_norm * values)
    std_raw = jnp.sqrt(jnp.maximum(jnp.sum(w_norm * (values - mean_raw) ** 2), 0.0))
    if not math.isfinite(float(std_raw)) or float(std_raw) <= _FLOAT_MIN:
        _, std_raw = median_and_scale(values, scale_floor=scale_floor)
    std_raw = jnp.maximum(std_raw, scale_floor)

    bandwidth = jnp.maximum(
        _SILVERMAN_FACTOR * std_raw * (float(n_eff) ** (-0.2)) * bandwidth_scale,
        scale_floor,
    )

    x_min, x_max = jnp.nanmin(values), jnp.nanmax(values)
    x_span = jnp.maximum(x_max - x_min, scale_floor)
    x_pad = jnp.maximum(2.0 * bandwidth, 0.1 * x_span)
    grid_size = max(n_grid, _MIN_KDE_GRID)
    x_grid = jnp.linspace(x_min - x_pad, x_max + x_pad, grid_size)

    diffs = (x_grid[:, None] - values[None, :]) / bandwidth
    density = jnp.sum(jnp.exp(-0.5 * diffs**2) * w_norm[None, :], axis=1) / (
        bandwidth * math.sqrt(2.0 * math.pi)
    )

    if int(density.size) == 0 or not bool(jnp.any(jnp.isfinite(density))):
        center = mean_raw
    else:
        center = x_grid[int(jnp.nanargmax(density))]

    local_weights = jnp.exp(-0.5 * ((values - center) / bandwidth) ** 2) * w_norm
    local_weights = jnp.where(jnp.isfinite(local_weights), local_weights, 0.0)
    if not bool(jnp.any(local_weights > 0.0)):
        local_weights = w_norm
    local_norm = local_weights / jnp.maximum(jnp.sum(local_weights), _FLOAT_MIN)
    local_scale = jnp.sqrt(
        jnp.maximum(jnp.sum(local_norm * (values - center) ** 2), 0.0)
    )
    local_scale = jnp.maximum(local_scale, scale_floor)

    threshold = max(n_sigma, _FLOAT_MIN) * local_scale
    keep_finite = jnp.abs(values - center) <= threshold
    if int(jnp.sum(keep_finite)) == 0:
        keep_finite = jnp.ones_like(values, dtype=bool)

    keep = keep.at[jnp.where(finite)[0]].set(keep_finite)

    return ApexGate(
        center=jnp.asarray(center, dtype=jnp.float32),
        scale=jnp.asarray(local_scale, dtype=jnp.float32),
        low=jnp.asarray(center - threshold, dtype=jnp.float32),
        high=jnp.asarray(center + threshold, dtype=jnp.float32),
        bandwidth=jnp.asarray(bandwidth, dtype=jnp.float32),
        keep_mask=keep,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _skew_mode_offsets(alpha_values: jnp.ndarray) -> jnp.ndarray:
    """Standardised mode offset ``(mode - xi) / sigma`` for each alpha value.

    Computed by grid search on the standard skew-normal PDF.  Supports
    arbitrary input shapes by flattening internally.
    """
    alpha_array = jnp.asarray(alpha_values, dtype=jnp.float32)
    flat = alpha_array.reshape(-1)
    offsets = jnp.full_like(flat, jnp.nan)

    valid_idx = jnp.where(jnp.isfinite(flat))[0]
    if int(valid_idx.size) == 0:
        return offsets.reshape(alpha_array.shape)

    grid = jnp.linspace(-8.0, 8.0, 2049, dtype=jnp.float32)
    alpha_valid = flat[valid_idx]
    pdf = skew_normal_pdf(
        grid,
        jnp.zeros(alpha_valid.shape[0], dtype=jnp.float32),
        jnp.ones(alpha_valid.shape[0], dtype=jnp.float32),
        alpha_valid,
    )
    if pdf.ndim == 1:
        pdf = pdf[None, :]
    offsets = offsets.at[valid_idx].set(grid[jnp.argmax(pdf, axis=1)])
    return offsets.reshape(alpha_array.shape)


def _fwhm_to_skew_params(
    w_left: jnp.ndarray,
    w_right: jnp.ndarray,
    *,
    keep: jnp.ndarray,
    alpha_soft_cap: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert HWHM half-widths to skew-normal ``(sigma, alpha)``.

    Args:
        w_left:  Left HWHM values; arbitrary shape, NaN where invalid.
        w_right: Right HWHM values; same shape.
        keep:    Boolean mask — compute only where True; NaN elsewhere.
        alpha_soft_cap: Soft ceiling on ``|alpha|`` via tanh compression.

    Returns:
        ``(sigma, alpha)`` with the same shape; NaN where ``~keep``.
    """
    sl = jnp.where(keep, w_left / _GAUSSIAN_HWHM_FACTOR, jnp.nan)
    sr = jnp.where(keep, w_right / _GAUSSIAN_HWHM_FACTOR, jnp.nan)
    sigma = jnp.where(keep, jnp.sqrt(0.5 * (sl**2 + sr**2)), jnp.nan)
    delta = (sr - sl) / jnp.maximum(sr + sl, _FLOAT_MIN)
    delta = jnp.where(keep, jnp.clip(delta, -0.95, 0.95), jnp.nan)
    alpha_raw = delta / jnp.sqrt(jnp.maximum(1.0 - delta**2, 1e-8))
    cap = max(alpha_soft_cap, 1e-6)
    alpha = jnp.where(keep, cap * jnp.tanh(alpha_raw / cap), jnp.nan)
    return sigma, alpha


def _xi_from_mode(
    mode: jnp.ndarray,
    sigma: jnp.ndarray,
    alpha: jnp.ndarray,
    *,
    low: jnp.ndarray,
    high: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the skew-normal location parameter ``xi`` from the observed mode.

    ``xi = mode - sigma * mode_offset(alpha)``, clipped to ``[low, high]``.
    NaN inputs propagate to NaN outputs.
    """
    offsets = _skew_mode_offsets(alpha)
    return jnp.clip(mode - sigma * offsets, low, high)


def skew_mode_time(
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    alpha: jnp.ndarray,
    *,
    low: jnp.ndarray | None = None,
    high: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Return skew-normal mode time from ``(mu, sigma, alpha)`` parameters.

    Args:
        mu: Location parameter.
        sigma: Positive width parameter.
        alpha: Skew parameter.
        low: Optional lower clip bound (broadcastable to ``mu``).
        high: Optional upper clip bound (broadcastable to ``mu``).

    Returns:
        Mode times with the same shape as ``mu``.
    """
    mode = mu + jnp.maximum(sigma, 1e-8) * _skew_mode_offsets(alpha)
    if low is not None and high is not None:
        return jnp.clip(mode, low, high)
    return mode


def _estimate_area(
    signal: jnp.ndarray,
    time: jnp.ndarray,
    masks: jnp.ndarray,
) -> jnp.ndarray:
    """Estimate peak area by trapezoid integration of the baseline-corrected signal.

    The signal is zeroed outside each peak's window mask before integration.
    Because the signal is already baseline-corrected, zeroing outside the mask
    is exact (no baseline contribution leaks in).

    Args:
        signal: ``[n_trace, n_time]`` baseline-corrected signal.
        time:   ``[n_trace, n_time]`` time axis.
        masks:  ``[n_peak, n_trace, n_time]`` boolean peak window masks.

    Returns:
        ``[n_trace, n_peak]`` area estimates, clamped to ``≥ 1e-8``.
    """
    masks_tp = masks.transpose(1, 0, 2)  # [n_trace, n_peak, n_time]
    # Zero signal outside the window; safe because signal is baseline-corrected.
    sig_windowed = jnp.where(masks_tp, signal[:, None, :], 0.0)  # [n_trace, n_peak, n_time]
    # Trapezoid rule: Σ 0.5 * (f[i] + f[i+1]) * dt[i]
    dt = jnp.diff(time, axis=-1)  # [n_trace, n_time-1]
    area = 0.5 * jnp.sum(
        (sig_windowed[:, :, :-1] + sig_windowed[:, :, 1:]) * dt[:, None, :],
        axis=-1,
    )  # [n_trace, n_peak]
    return jnp.maximum(area, 1e-8)


def _population_priors(
    xi: jnp.ndarray,
    sigma: jnp.ndarray,
    alpha: jnp.ndarray,
    area: jnp.ndarray,
    keep: jnp.ndarray,
    *,
    peak_lows: jnp.ndarray,
    peak_highs: jnp.ndarray,
) -> PeakPriors:
    """Compute per-peak population priors from gated trace data.

    Uses median and MAD-scale over ``keep``-masked traces per peak.
    Physical fallbacks are applied when fewer than one finite value exists.

    Args:
        xi:         ``[n_trace, n_peak]`` location parameter (NaN where ~keep).
        sigma:      ``[n_trace, n_peak]`` width parameter.
        alpha:      ``[n_trace, n_peak]`` skew parameter.
        area:       ``[n_trace, n_peak]`` area estimate.
        keep:       ``[n_trace, n_peak]`` boolean mask of valid individual estimates.
        peak_lows:  ``[n_peak]`` lower window bounds.
        peak_highs: ``[n_peak]`` upper window bounds.

    Returns:
        :class:`PeakPriors` with all fields of shape ``[n_peak]``.
    """
    n_peak = int(xi.shape[1])

    mu_loc = jnp.full((n_peak,), jnp.nan, dtype=jnp.float32)
    mu_scale = jnp.full((n_peak,), jnp.nan, dtype=jnp.float32)
    sl_vec = jnp.full((n_peak,), jnp.nan, dtype=jnp.float32)
    ss_vec = jnp.full((n_peak,), jnp.nan, dtype=jnp.float32)
    al_vec = jnp.full((n_peak,), jnp.nan, dtype=jnp.float32)
    as_vec = jnp.full((n_peak,), jnp.nan, dtype=jnp.float32)
    arl_vec = jnp.full((n_peak,), jnp.nan, dtype=jnp.float32)
    ars_vec = jnp.full((n_peak,), jnp.nan, dtype=jnp.float32)

    for p in range(n_peak):
        low = float(peak_lows[p])
        high = float(peak_highs[p])
        span = max(high - low, 1e-4)
        kp = keep[:, p]

        def _masked(arr: jnp.ndarray) -> jnp.ndarray:
            return jnp.where(kp, arr[:, p], jnp.nan)

        ml, ms = median_and_scale(_masked(xi), scale_floor=1e-4)
        sl, ss = median_and_scale(_masked(sigma), scale_floor=1e-4)
        al, as_ = median_and_scale(_masked(alpha), scale_floor=1e-3)
        arl, ars = median_and_scale(_masked(area), scale_floor=1e-6)

        # Fallbacks for degenerate cases
        ml = jnp.where(jnp.isfinite(ml), ml, 0.5 * (low + high))
        ms = jnp.where(jnp.isfinite(ms) & (ms > 0), ms, max(0.02 * span, 1e-4))
        sl = jnp.where(jnp.isfinite(sl) & (sl > 0), sl, max(span / 6.0, 1e-4))
        ss = jnp.where(jnp.isfinite(ss) & (ss > 0), ss, 0.2 * float(sl))
        al = jnp.where(jnp.isfinite(al), al, 0.0)
        as_ = jnp.where(jnp.isfinite(as_) & (as_ > 0), as_, 1e-3)
        arl = jnp.where(jnp.isfinite(arl) & (arl > 0), arl, 1e-8)
        ars = jnp.where(jnp.isfinite(ars) & (ars > 0), ars, 0.25 * float(arl))

        # Clamp only to physically meaningful values.
        ml = float(jnp.clip(ml, low, high))
        sl = max(float(sl), 1e-8)
        ss = max(float(ss), 1e-4)
        as_ = max(float(as_), 1e-3)
        arl = max(float(arl), 1e-8)
        ars = max(float(ars), 1e-6)

        mu_loc = mu_loc.at[p].set(ml)
        mu_scale = mu_scale.at[p].set(ms)
        sl_vec = sl_vec.at[p].set(sl)
        ss_vec = ss_vec.at[p].set(ss)
        al_vec = al_vec.at[p].set(al)
        as_vec = as_vec.at[p].set(as_)
        arl_vec = arl_vec.at[p].set(arl)
        ars_vec = ars_vec.at[p].set(ars)

    return PeakPriors(
        mu_loc=mu_loc,
        mu_scale=mu_scale,
        sigma_loc=sl_vec,
        sigma_scale=ss_vec,
        alpha_loc=al_vec,
        alpha_scale=as_vec,
        area_loc=arl_vec,
        area_scale=ars_vec,
    )


def _expand_feature_component_axis(
    values: jnp.ndarray,
    *,
    fill_value: float | bool,
) -> jnp.ndarray:
    """Expand ``values`` by a fixed component axis and write into component 0.

    This keeps the feature payload shape-stable for upcoming double-peak support.
    """
    array = jnp.asarray(values)
    if array.ndim not in {1, 2}:
        raise ValueError(
            f"Feature component expansion expects 1-D or 2-D input, got {array.shape}."
        )
    target_shape = array.shape + (_N_FEATURE_COMPONENTS,)
    padded = jnp.full(target_shape, fill_value, dtype=array.dtype)
    return padded.at[..., 0].set(array)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_peak_fwhm_features(
    time: jnp.ndarray,
    signal: jnp.ndarray,
    peak_masks: jnp.ndarray,
    *,
    level: float = 0.5,
    apply_apex_gate: bool = True,
    kde_n_sigma: float = 3.0,
    kde_bandwidth_scale: float = 1.0,
    kde_scale_floor: float = 1e-4,
) -> FwhmFeatures:
    """Compute per-trace/per-peak FWHM features with KDE apex gating.

    The FWHM computation is fully vectorised over the ``[n_peak, n_trace, n_time]``
    batch — no Python loops over traces.

    Args:
        time:       ``[n_trace, n_time]`` time axis.
        signal:     ``[n_trace, n_time]`` baseline-corrected signal.
        peak_masks: ``[n_peak, n_trace, n_time]`` boolean window masks.
        level:      Fractional level for FWHM (default 0.5 → half-maximum).
        apply_apex_gate: Whether to apply KDE apex gating.
        kde_n_sigma:       Gate half-width in local sigma units.
        kde_bandwidth_scale: Bandwidth scale factor for Silverman's rule.
        kde_scale_floor:   Minimum KDE bandwidth and scale.

    Returns:
        :class:`FwhmFeatures` with arrays shaped
        ``[n_trace, n_peak, n_component]`` (or ``[n_peak, n_component]`` for
        gate vectors), where ``n_component == 2``.

    Example::

        time = jnp.linspace(0, 10, 500)[None, :]  # (1, 500)
        signal = jnp.exp(-0.5 * ((time - 5.0) / 0.4) ** 2)
        mask = ((time >= 4.0) & (time <= 6.0))[:, None, :]  # (n_peak=1, 1, 500)
        feats = compute_peak_fwhm_features(time, signal, mask)
    """
    if not (0.0 < level < 1.0):
        raise ValueError("level must satisfy 0 < level < 1.")
    if kde_n_sigma <= 0.0:
        raise ValueError("kde_n_sigma must be > 0.")
    if kde_bandwidth_scale <= 0.0:
        raise ValueError("kde_bandwidth_scale must be > 0.")
    if kde_scale_floor <= 0.0:
        raise ValueError("kde_scale_floor must be > 0.")

    time_matrix = jnp.asarray(time, dtype=jnp.float32)
    signal_matrix = jnp.asarray(signal, dtype=jnp.float32)
    mask_tensor = jnp.asarray(peak_masks, dtype=bool)

    if time_matrix.ndim != 2 or signal_matrix.ndim != 2:
        raise ValueError("time and signal must be 2-D [n_trace, n_time].")
    if time_matrix.shape != signal_matrix.shape:
        raise ValueError("time and signal shape mismatch.")
    if mask_tensor.ndim != 3:
        raise ValueError("peak_masks must be 3-D [n_peak, n_trace, n_time].")
    if mask_tensor.shape[1:] != time_matrix.shape:
        raise ValueError("peak_masks trace/time dimensions must match time shape.")

    n_peak, n_trace, n_time = (
        int(mask_tensor.shape[0]),
        int(time_matrix.shape[0]),
        int(time_matrix.shape[1]),
    )
    if n_peak == 0:
        raise ValueError("No peak masks provided.")

    # ------------------------------------------------------------------
    # Pass 1: vectorised apex + FWHM over [n_peak, n_trace, n_time]
    # ------------------------------------------------------------------
    # Mask signal to peak windows; NaN outside
    sig_masked = jnp.where(mask_tensor, signal_matrix[None], jnp.nan)

    # Apex: argmax over time; treat NaN as -inf for argmax
    sig_for_max = jnp.where(jnp.isfinite(sig_masked), sig_masked, -jnp.inf)
    apex_idx = jnp.argmax(sig_for_max, axis=2)  # [n_peak, n_trace]
    apex_height = jnp.take_along_axis(sig_masked, apex_idx[:, :, None], axis=2).squeeze(
        2
    )  # [n_peak, n_trace]
    apex_time_pt = jnp.take_along_axis(
        jnp.broadcast_to(time_matrix[None], (n_peak, n_trace, n_time)),
        apex_idx[:, :, None],
        axis=2,
    ).squeeze(2)  # [n_peak, n_trace]

    n_points = jnp.sum(mask_tensor, axis=2)  # [n_peak, n_trace]
    valid_pt = (n_points >= 3) & (apex_height > _FLOAT_MIN) & jnp.isfinite(apex_height)

    # Normalise to apex; use safe divisor to avoid 0/NaN
    height_safe = jnp.where(valid_pt, apex_height, 1.0)
    y_norm = sig_masked / height_safe[:, :, None]

    # Crossing detection
    t_idx = jnp.arange(n_time, dtype=jnp.int32)
    left_of_apex = (
        t_idx[None, None, :-1] < apex_idx[:, :, None]
    )  # [n_peak, n_trace, n_time-1]
    left_rising = (
        (y_norm[:, :, :-1] <= level) & (y_norm[:, :, 1:] > level) & left_of_apex
    )
    right_falling = (
        (y_norm[:, :, :-1] >= level) & (y_norm[:, :, 1:] < level) & ~left_of_apex
    )

    has_left = jnp.any(left_rising, axis=2)  # [n_peak, n_trace]
    has_right = jnp.any(right_falling, axis=2)

    # Last left crossing: reverse-argmax trick
    left_i = (n_time - 2) - jnp.argmax(jnp.flip(left_rising, axis=2), axis=2)
    right_i = jnp.argmax(right_falling, axis=2)
    left_i = jnp.clip(left_i, 0, n_time - 2)
    right_i = jnp.clip(right_i, 0, n_time - 2)

    # Gather crossing pairs via take_along_axis
    time_exp = jnp.broadcast_to(time_matrix[None], (n_peak, n_trace, n_time))

    def _take2(arr: jnp.ndarray, idx: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Gather arr[..., idx] and arr[..., idx+1]."""
        idx3 = idx[:, :, None]
        v0 = jnp.take_along_axis(arr, idx3, axis=2).squeeze(2)
        v1 = jnp.take_along_axis(arr, idx3 + 1, axis=2).squeeze(2)
        return v0, v1

    x_l0, x_l1 = _take2(time_exp, left_i)
    y_l0, y_l1 = _take2(y_norm, left_i)
    x_r0, x_r1 = _take2(time_exp, right_i)
    y_r0, y_r1 = _take2(y_norm, right_i)

    # Linear interpolation at crossing level
    def _interp(
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        y0: jnp.ndarray,
        y1: jnp.ndarray,
        valid_cross: jnp.ndarray,
    ) -> jnp.ndarray:
        dy = y1 - y0
        dy_safe = jnp.where(jnp.abs(dy) > _FLOAT_MIN, dy, 1.0)
        crossing = x0 + (level - y0) / dy_safe * (x1 - x0)
        return jnp.where(valid_cross & (jnp.abs(dy) > _FLOAT_MIN), crossing, jnp.nan)

    valid_flag = valid_pt  # [n_peak, n_trace]
    left_time_pt = _interp(x_l0, x_l1, y_l0, y_l1, has_left & valid_flag)
    right_time_pt = _interp(x_r0, x_r1, y_r0, y_r1, has_right & valid_flag)
    fwhm_pt = jnp.where(
        has_left & has_right & valid_flag,
        right_time_pt - left_time_pt,
        jnp.nan,
    )

    # Transpose to [n_trace, n_peak]
    valid_all = valid_pt.T
    apex_height_all = apex_height.T
    apex_time_all = apex_time_pt.T
    left_raw = left_time_pt.T
    right_raw = right_time_pt.T
    fwhm_raw = fwhm_pt.T

    # ------------------------------------------------------------------
    # Pass 2: KDE apex gating per peak (loop over n_peak — small)
    # ------------------------------------------------------------------
    gate_keep = jnp.zeros((n_trace, n_peak), dtype=bool)
    gate_center = jnp.full((n_peak,), jnp.nan, dtype=jnp.float32)
    gate_scale = jnp.full((n_peak,), jnp.nan, dtype=jnp.float32)
    gate_low_v = jnp.full((n_peak,), jnp.nan, dtype=jnp.float32)
    gate_high_v = jnp.full((n_peak,), jnp.nan, dtype=jnp.float32)
    gate_bandwidth = jnp.full((n_peak,), jnp.nan, dtype=jnp.float32)

    for p in range(n_peak):
        candidates = apex_time_all[:, p]
        finite_cand = jnp.isfinite(candidates) & valid_all[:, p]
        if int(jnp.sum(finite_cand)) == 0:
            continue
        candidate_weights = jnp.where(
            finite_cand
            & jnp.isfinite(apex_height_all[:, p])
            & (apex_height_all[:, p] > _FLOAT_MIN),
            apex_height_all[:, p],
            0.0,
        )

        if apply_apex_gate:
            g = kde_apex_gate(
                candidates,
                weights=candidate_weights,
                n_sigma=float(kde_n_sigma),
                bandwidth_scale=float(kde_bandwidth_scale),
                scale_floor=float(kde_scale_floor),
            )
            keep_p = g.keep_mask & finite_cand
            if int(jnp.sum(keep_p)) == 0:
                keep_p = finite_cand
            gate_center = gate_center.at[p].set(g.center)
            gate_scale = gate_scale.at[p].set(g.scale)
            gate_low_v = gate_low_v.at[p].set(g.low)
            gate_high_v = gate_high_v.at[p].set(g.high)
            gate_bandwidth = gate_bandwidth.at[p].set(g.bandwidth)
        else:
            keep_p = finite_cand
            center, scale = median_and_scale(
                jnp.where(finite_cand, candidates, jnp.nan), scale_floor=1e-6
            )
            gate_center = gate_center.at[p].set(center)
            gate_scale = gate_scale.at[p].set(scale)

        gate_keep = gate_keep.at[:, p].set(keep_p)

    # Expand to [n_trace, n_peak, n_component=2]; component 1 is reserved.
    valid_all_c = _expand_feature_component_axis(valid_all, fill_value=False)
    apex_time_all_c = _expand_feature_component_axis(apex_time_all, fill_value=jnp.nan)
    left_raw_c = _expand_feature_component_axis(left_raw, fill_value=jnp.nan)
    right_raw_c = _expand_feature_component_axis(right_raw, fill_value=jnp.nan)
    fwhm_raw_c = _expand_feature_component_axis(fwhm_raw, fill_value=jnp.nan)
    gate_keep_c = _expand_feature_component_axis(gate_keep, fill_value=False)
    gate_center_c = _expand_feature_component_axis(gate_center, fill_value=jnp.nan)
    gate_scale_c = _expand_feature_component_axis(gate_scale, fill_value=jnp.nan)
    gate_low_c = _expand_feature_component_axis(gate_low_v, fill_value=jnp.nan)
    gate_high_c = _expand_feature_component_axis(gate_high_v, fill_value=jnp.nan)
    gate_bandwidth_c = _expand_feature_component_axis(
        gate_bandwidth, fill_value=jnp.nan
    )

    use_mask = gate_keep_c & valid_all_c
    apex_time = jnp.where(use_mask, apex_time_all_c, jnp.nan)
    left_time = jnp.where(use_mask, left_raw_c, jnp.nan)
    right_time = jnp.where(use_mask, right_raw_c, jnp.nan)
    fwhm = jnp.where(use_mask, fwhm_raw_c, jnp.nan)

    gate = ApexGate(
        center=gate_center_c,
        scale=gate_scale_c,
        low=gate_low_c,
        high=gate_high_c,
        bandwidth=gate_bandwidth_c,
        keep_mask=gate_keep_c,
    )
    return FwhmFeatures(
        apex_time=apex_time,
        left_time=left_time,
        right_time=right_time,
        fwhm=fwhm,
        valid=valid_all_c,
        apex_time_all=apex_time_all_c,
        left_time_all=left_raw_c,
        right_time_all=right_raw_c,
        fwhm_all=fwhm_raw_c,
        gate=gate,
    )


def compute_skew_normal_guess(
    fwhm_features: FwhmFeatures,
    time: jnp.ndarray,
    signal: jnp.ndarray,
    peak_masks: jnp.ndarray,
    peak_lows: jnp.ndarray,
    peak_highs: jnp.ndarray,
    *,
    alpha_soft_cap: float = 2.5,
) -> SkewNormalGuess:
    """Convert FWHM features to skew-normal initialisers and population priors.

    For gated traces (``fwhm_features.gate.keep_mask``), parameters are derived
    from the individual FWHM measurements.  Non-gated traces are fallback-filled
    from the population median (``priors.*_loc``).

    Args:
        fwhm_features: Output of :func:`compute_peak_fwhm_features`.
        time:          ``[n_trace, n_time]`` time axis.
        signal:        ``[n_trace, n_time]`` baseline-corrected signal.
        peak_masks:    ``[n_peak, n_trace, n_time]`` boolean window masks.
        peak_lows:     ``[n_peak]`` window lower bounds.
        peak_highs:    ``[n_peak]`` window upper bounds.
        alpha_soft_cap: Soft ceiling on ``|alpha|``.

    Returns:
        :class:`SkewNormalGuess` with ``[n_trace, n_peak]`` arrays and
        embedded :class:`PeakPriors`.

    Example::

        feats = compute_peak_fwhm_features(time, signal, masks)
        guess = compute_skew_normal_guess(feats, time, signal, masks, lows, highs)
    """
    time_matrix = jnp.asarray(time, dtype=jnp.float32)
    signal_matrix = jnp.asarray(signal, dtype=jnp.float32)
    mask_tensor = jnp.asarray(peak_masks, dtype=bool)
    peak_lows_v = jnp.asarray(peak_lows, dtype=jnp.float32).reshape(-1)
    peak_highs_v = jnp.asarray(peak_highs, dtype=jnp.float32).reshape(-1)

    n_peak = int(peak_lows_v.shape[0])

    def _primary_component(values: jnp.ndarray, *, name: str) -> jnp.ndarray:
        """Return component-0 view from canonical feature tensors."""
        array = jnp.asarray(values)
        if array.ndim != 3 or int(array.shape[2]) != _N_FEATURE_COMPONENTS:
            raise ValueError(
                f"{name} must be [n_trace, n_peak, {_N_FEATURE_COMPONENTS}], "
                f"got {array.shape}."
            )
        return array[:, :, 0]

    def _primary_gate(values: jnp.ndarray, *, name: str) -> jnp.ndarray:
        """Return component-0 view from canonical gate tensors."""
        array = jnp.asarray(values)
        if array.ndim != 2 or int(array.shape[1]) != _N_FEATURE_COMPONENTS:
            raise ValueError(
                f"{name} must be [n_peak, {_N_FEATURE_COMPONENTS}], got {array.shape}."
            )
        return array[:, 0]

    # --- Step 1: FWHM half-widths and keep mask (gated + valid FWHM) ----
    mode_gated = _primary_component(
        fwhm_features.apex_time, name="fwhm_features.apex_time"
    )  # [n_trace, n_peak]; NaN for non-gated
    left_time = _primary_component(
        fwhm_features.left_time, name="fwhm_features.left_time"
    )
    right_time = _primary_component(
        fwhm_features.right_time, name="fwhm_features.right_time"
    )
    w_left = mode_gated - left_time
    w_right = right_time - mode_gated
    keep = (
        jnp.isfinite(w_left)
        & jnp.isfinite(w_right)
        & (w_left > 1e-8)
        & (w_right > 1e-8)
        & jnp.isfinite(mode_gated)
    )

    # --- Step 2: sigma, alpha for gated traces ---------------------------
    sigma_raw, alpha_raw = _fwhm_to_skew_params(
        w_left,
        w_right,
        keep=keep,
        alpha_soft_cap=alpha_soft_cap,
    )

    # --- Step 3: xi = location parameter for gated traces ----------------
    xi_raw = _xi_from_mode(
        mode_gated,
        sigma_raw,
        alpha_raw,
        low=peak_lows_v[None, :],
        high=peak_highs_v[None, :],
    )

    # --- Step 4: population medians for fallback fill --------------------
    # Small loop over n_peak (O(n_peak), trace dimension vectorised)
    xi_loc = jnp.full((n_peak,), jnp.nan, dtype=jnp.float32)
    sigma_loc = jnp.full((n_peak,), jnp.nan, dtype=jnp.float32)
    alpha_loc = jnp.full((n_peak,), jnp.nan, dtype=jnp.float32)

    for p in range(n_peak):
        low_f = float(peak_lows_v[p])
        high_f = float(peak_highs_v[p])
        span_f = max(high_f - low_f, 1e-4)
        kp = keep[:, p]

        xl, _ = median_and_scale(jnp.where(kp, xi_raw[:, p], jnp.nan), scale_floor=1e-4)
        sl, _ = median_and_scale(
            jnp.where(kp, sigma_raw[:, p], jnp.nan), scale_floor=1e-4
        )
        al, _ = median_and_scale(
            jnp.where(kp, alpha_raw[:, p], jnp.nan), scale_floor=1e-3
        )

        xl = jnp.where(jnp.isfinite(xl), xl, 0.5 * (low_f + high_f))
        xl = float(jnp.clip(xl, low_f, high_f))
        sl = jnp.where(jnp.isfinite(sl) & (sl > 0), sl, span_f / 6.0)
        sl = max(float(sl), 1e-8)
        al = float(jnp.where(jnp.isfinite(al), al, 0.0))

        xi_loc = xi_loc.at[p].set(xl)
        sigma_loc = sigma_loc.at[p].set(sl)
        alpha_loc = alpha_loc.at[p].set(al)

    # --- Step 5: fallback-fill non-gated traces --------------------------
    xi_filled = jnp.where(keep, xi_raw, xi_loc[None, :])
    sigma_filled = jnp.where(keep, sigma_raw, sigma_loc[None, :])
    alpha_filled = jnp.where(keep, alpha_raw, alpha_loc[None, :])

    # --- Step 6: mode time for ALL traces --------------------------------
    # Non-gated: use median accepted apex time; fall back to gate.center,
    # then to prior-implied mode if needed.
    mode_from_prior = xi_loc + sigma_loc * _skew_mode_offsets(alpha_loc)  # [n_peak]
    gate_center = _primary_gate(
        fwhm_features.gate.center, name="fwhm_features.gate.center"
    )  # [n_peak]
    median_kept_mode = jnp.full((n_peak,), jnp.nan, dtype=jnp.float32)
    for p in range(n_peak):
        median_p, _ = median_and_scale(
            jnp.where(keep[:, p], mode_gated[:, p], jnp.nan),
            scale_floor=1e-6,
        )
        median_kept_mode = median_kept_mode.at[p].set(median_p)
    fallback_mode = jnp.where(
        jnp.isfinite(median_kept_mode), median_kept_mode, gate_center
    )
    fallback_mode = jnp.where(
        jnp.isfinite(fallback_mode), fallback_mode, mode_from_prior
    )
    fallback_mode = jnp.clip(fallback_mode, peak_lows_v, peak_highs_v)

    mode_filled = jnp.where(keep, mode_gated, fallback_mode[None, :])
    mode_filled = jnp.clip(mode_filled, peak_lows_v[None, :], peak_highs_v[None, :])

    # --- Step 7: area estimation for all traces --------------------------
    area = _estimate_area(
        signal_matrix,
        time_matrix,
        mask_tensor,
    )

    # --- Step 8: population priors (including area) ----------------------
    priors = _population_priors(
        xi_raw,
        sigma_raw,
        alpha_raw,
        area,
        keep,
        peak_lows=peak_lows_v,
        peak_highs=peak_highs_v,
    )

    # Keep per-trace area estimates for all traces (including non-gated).
    # Non-gated traces use the per-peak median apex time reference above.
    area_filled = area
    area_filled = jnp.maximum(area_filled, 1e-8)

    return SkewNormalGuess(
        mu=xi_filled,
        sigma=sigma_filled,
        alpha=alpha_filled,
        area=area_filled,
        keep=keep,
        priors=priors,
    )


def build_two_stage_component_initializers(
    *,
    time: jnp.ndarray,
    signal: jnp.ndarray,
    peak_masks: jnp.ndarray,
    peak_lows: jnp.ndarray,
    peak_highs: jnp.ndarray,
    shoulder_side: jnp.ndarray,
    main_guess: SkewNormalGuess,
    side_sigma_multiplier: float = 1.0,
    fwhm_level: float = 0.5,
    apply_apex_gate: bool = True,
    kde_n_sigma: float = 8.0,
    kde_bandwidth_scale: float = 1.0,
    kde_scale_floor: float = 1e-4,
    alpha_soft_cap: float = 2.5,
    min_side_points: int = 3,
) -> ComponentInitializers:
    """Build two-component initializers via main-pass + residual shoulder-pass.

    Component ``0`` is always the first-pass (main) initializer.
    Component ``1`` is estimated only for peaks with ``shoulder_side != 0``:
    a residual signal is built by subtracting the main peak prediction, then
    FWHM + skew-normal initialization is re-run inside a side-restricted mask.

    Args:
        time: ``[n_trace, n_time]`` time axis.
        signal: ``[n_trace, n_time]`` baseline-corrected signal.
        peak_masks: ``[n_peak, n_trace, n_time]`` peak window masks.
        peak_lows: ``[n_peak]`` lower bounds.
        peak_highs: ``[n_peak]`` upper bounds.
        shoulder_side: ``[n_peak]`` with values ``-1`` (left), ``+1`` (right),
            ``0`` (no shoulder).
        main_guess: first-pass initializer output.
        side_sigma_multiplier: side split offset in units of main sigma.
        fwhm_level: crossing level for FWHM extraction.
        apply_apex_gate: whether to apply KDE gate in both passes.
        kde_n_sigma: KDE gate width in local sigma units.
        kde_bandwidth_scale: KDE bandwidth scale.
        kde_scale_floor: minimum KDE scale.
        alpha_soft_cap: skew soft-cap passed to `compute_skew_normal_guess`.
        min_side_points: minimum masked points required for shoulder estimation.

    Returns:
        :class:`ComponentInitializers` with shape ``[n_trace, n_peak, 2]``.
    """
    if float(side_sigma_multiplier) <= 0.0:
        raise ValueError("side_sigma_multiplier must be > 0.")
    if not (0.0 < float(fwhm_level) < 1.0):
        raise ValueError("fwhm_level must satisfy 0 < fwhm_level < 1.")
    min_side_points = max(int(min_side_points), 1)

    time_matrix = jnp.asarray(time, dtype=jnp.float32)
    signal_matrix = jnp.asarray(signal, dtype=jnp.float32)
    mask_tensor = jnp.asarray(peak_masks, dtype=bool)
    peak_lows_v = jnp.asarray(peak_lows, dtype=jnp.float32).reshape(-1)
    peak_highs_v = jnp.asarray(peak_highs, dtype=jnp.float32).reshape(-1)
    shoulder_side_v = jnp.asarray(shoulder_side, dtype=jnp.int32).reshape(-1)

    if time_matrix.ndim != 2 or signal_matrix.ndim != 2:
        raise ValueError("time and signal must be 2-D [n_trace, n_time].")
    if time_matrix.shape != signal_matrix.shape:
        raise ValueError("time and signal shape mismatch.")
    if mask_tensor.ndim != 3:
        raise ValueError("peak_masks must be 3-D [n_peak, n_trace, n_time].")
    if mask_tensor.shape[1:] != time_matrix.shape:
        raise ValueError("peak_masks trace/time dimensions must match time shape.")

    n_peak = int(mask_tensor.shape[0])
    n_trace = int(time_matrix.shape[0])
    if peak_lows_v.shape[0] != n_peak or peak_highs_v.shape[0] != n_peak:
        raise ValueError("peak_lows/peak_highs length must match peak mask count.")
    if shoulder_side_v.shape[0] != n_peak:
        raise ValueError("shoulder_side length must match peak mask count.")
    if int(main_guess.mu.shape[1]) != n_peak or int(main_guess.mu.shape[0]) != n_trace:
        raise ValueError("main_guess must match [n_trace, n_peak] shape.")
    if not bool(
        jnp.all((shoulder_side_v == -1) | (shoulder_side_v == 0) | (shoulder_side_v == 1))
    ):
        raise ValueError("shoulder_side entries must be -1, 0, or +1.")

    span = jnp.maximum(peak_highs_v - peak_lows_v, 1e-4)
    center = 0.5 * (peak_lows_v + peak_highs_v)

    mu_main = jnp.clip(
        jnp.where(jnp.isfinite(main_guess.mu), main_guess.mu, center[None, :]),
        peak_lows_v[None, :],
        peak_highs_v[None, :],
    )
    sigma_main = jnp.where(
        jnp.isfinite(main_guess.sigma) & (main_guess.sigma > 1e-8),
        main_guess.sigma,
        (span / 6.0)[None, :],
    )
    sigma_main = jnp.maximum(sigma_main, 1e-8)
    alpha_main = jnp.where(jnp.isfinite(main_guess.alpha), main_guess.alpha, 0.0)
    area_main = jnp.maximum(
        jnp.where(jnp.isfinite(main_guess.area) & (main_guess.area > 0.0), main_guess.area, 1e-8),
        1e-8,
    )

    mu_init = jnp.full((n_trace, n_peak, _N_FEATURE_COMPONENTS), jnp.nan, dtype=jnp.float32)
    sigma_init = jnp.full((n_trace, n_peak, _N_FEATURE_COMPONENTS), jnp.nan, dtype=jnp.float32)
    alpha_init = jnp.full((n_trace, n_peak, _N_FEATURE_COMPONENTS), jnp.nan, dtype=jnp.float32)
    area_init = jnp.zeros((n_trace, n_peak, _N_FEATURE_COMPONENTS), dtype=jnp.float32)
    shoulder_keep = jnp.zeros((n_trace, n_peak), dtype=bool)
    shoulder_split_time = jnp.full((n_trace, n_peak), jnp.nan, dtype=jnp.float32)
    shoulder_side_points = jnp.zeros((n_trace, n_peak), dtype=jnp.int32)

    mu_init = mu_init.at[:, :, 0].set(mu_main)
    sigma_init = sigma_init.at[:, :, 0].set(sigma_main)
    alpha_init = alpha_init.at[:, :, 0].set(alpha_main)
    area_init = area_init.at[:, :, 0].set(area_main)

    # Deterministic-off shoulder defaults.
    mu_init = mu_init.at[:, :, 1].set(mu_main)
    sigma_init = sigma_init.at[:, :, 1].set(sigma_main)
    alpha_init = alpha_init.at[:, :, 1].set(jnp.zeros_like(alpha_main))

    mode_main = jnp.clip(
        mu_main + sigma_main * _skew_mode_offsets(alpha_main),
        peak_lows_v[None, :],
        peak_highs_v[None, :],
    )

    for p in range(n_peak):
        side = int(shoulder_side_v[p])
        if side == 0:
            continue

        low = float(peak_lows_v[p])
        high = float(peak_highs_v[p])
        span_p = max(high - low, 1e-4)

        pdf_main_p = skew_normal_pdf(
            time_matrix,
            mu_main[:, p : p + 1],
            sigma_main[:, p : p + 1],
            alpha_main[:, p : p + 1],
        )[:, 0, :]
        main_pred_p = area_main[:, p : p + 1] * pdf_main_p
        residual_p = jnp.clip(signal_matrix - main_pred_p, a_min=0.0, a_max=jnp.inf)

        split = mode_main[:, p] + float(side_sigma_multiplier) * side * sigma_main[:, p]
        shoulder_split_time = shoulder_split_time.at[:, p].set(split)
        if side > 0:
            side_mask = time_matrix >= split[:, None]
        else:
            side_mask = time_matrix <= split[:, None]

        candidate_mask = mask_tensor[p] & side_mask
        side_count = jnp.sum(candidate_mask, axis=1)
        shoulder_side_points = shoulder_side_points.at[:, p].set(side_count.astype(jnp.int32))
        enough_points = side_count >= min_side_points
        candidate_mask = candidate_mask & enough_points[:, None]
        if int(jnp.sum(candidate_mask)) == 0:
            continue

        shoulder_mask = candidate_mask[None, :, :]
        fwhm_2 = compute_peak_fwhm_features(
            time_matrix,
            residual_p,
            shoulder_mask,
            level=float(fwhm_level),
            apply_apex_gate=bool(apply_apex_gate),
            kde_n_sigma=float(kde_n_sigma),
            kde_bandwidth_scale=float(kde_bandwidth_scale),
            kde_scale_floor=float(kde_scale_floor),
        )
        guess_2 = compute_skew_normal_guess(
            fwhm_features=fwhm_2,
            time=time_matrix,
            signal=residual_p,
            peak_masks=shoulder_mask,
            peak_lows=peak_lows_v[p : p + 1],
            peak_highs=peak_highs_v[p : p + 1],
            alpha_soft_cap=float(alpha_soft_cap),
        )

        mu_pop = float(guess_2.priors.mu_loc[0])
        sigma_pop = float(guess_2.priors.sigma_loc[0])
        alpha_pop = float(guess_2.priors.alpha_loc[0])

        if not np.isfinite(mu_pop):
            mu_pop = float(main_guess.priors.mu_loc[p])
        if not np.isfinite(sigma_pop) or sigma_pop <= 1e-8:
            sigma_pop = float(main_guess.priors.sigma_loc[p])
        if not np.isfinite(alpha_pop):
            alpha_pop = float(main_guess.priors.alpha_loc[p])

        mu_pop = float(jnp.clip(mu_pop, low, high))
        sigma_pop = max(float(sigma_pop), 1e-8)
        alpha_pop = float(alpha_pop)

        mu2 = jnp.clip(guess_2.mu[:, 0], low, high)
        sigma2 = jnp.maximum(guess_2.sigma[:, 0], 1e-8)
        alpha2 = guess_2.alpha[:, 0]
        area2 = jnp.maximum(guess_2.area[:, 0], 1e-8)
        keep2 = jnp.asarray(guess_2.keep[:, 0], dtype=bool) & enough_points

        mode2 = mu2 + sigma2 * _skew_mode_offsets(alpha2)
        if side > 0:
            side_ok = mode2 >= split
        else:
            side_ok = mode2 <= split

        valid2 = (
            keep2
            & side_ok
            & jnp.isfinite(mu2)
            & jnp.isfinite(sigma2)
            & jnp.isfinite(alpha2)
            & jnp.isfinite(area2)
            & (area2 > 1e-8)
        )
        shoulder_keep = shoulder_keep.at[:, p].set(valid2)

        # Keep location/shape population fallbacks. Area fallback is estimated
        # from side-window residual projection and clipped to non-negative.
        mu_fill = jnp.where(valid2, mu2, mu_pop)
        sigma_fill = jnp.where(valid2, sigma2, sigma_pop)
        alpha_fill = jnp.where(valid2, alpha2, alpha_pop)

        # Keep shoulder initialisation physically plausible and side-consistent.
        split_med = float(jnp.nanmedian(split))
        if not np.isfinite(split_med):
            split_med = float(0.5 * (low + high))
        side_eps = 1e-4 * span_p
        if side > 0:
            mu_fill = jnp.maximum(mu_fill, split_med + side_eps)
        else:
            mu_fill = jnp.minimum(mu_fill, split_med - side_eps)
        mu_fill = jnp.clip(mu_fill, low, high)

        # For traces that fail local FWHM gating, estimate a deterministic
        # non-negative shoulder area by projecting residual onto the fallback
        # shoulder shape inside the side-window mask.
        shoulder_pdf_fill = skew_normal_pdf(
            time_matrix,
            mu_fill[:, None],
            sigma_fill[:, None],
            alpha_fill[:, None],
        )[:, 0, :]
        candidate_weight = candidate_mask.astype(jnp.float32)
        residual_side = jnp.where(candidate_mask, residual_p, 0.0)
        numer = jnp.sum(residual_side * shoulder_pdf_fill, axis=1)
        denom = jnp.sum(
            candidate_weight * shoulder_pdf_fill * shoulder_pdf_fill, axis=1
        )
        area_proj = jnp.where(denom > 1e-12, numer / denom, 0.0)
        area_proj = jnp.where(jnp.isfinite(area_proj), area_proj, 0.0)
        area_proj = jnp.maximum(area_proj, 0.0)
        area_fill = jnp.where(valid2, area2, area_proj)

        mu_init = mu_init.at[:, p, 1].set(mu_fill)
        sigma_init = sigma_init.at[:, p, 1].set(sigma_fill)
        alpha_init = alpha_init.at[:, p, 1].set(alpha_fill)
        area_init = area_init.at[:, p, 1].set(area_fill)

    return ComponentInitializers(
        mu_init=mu_init.astype(jnp.float32),
        sigma_init=sigma_init.astype(jnp.float32),
        alpha_init=alpha_init.astype(jnp.float32),
        A_init=area_init.astype(jnp.float32),
        shoulder_keep=shoulder_keep,
        shoulder_split_time=shoulder_split_time,
        shoulder_side_points=shoulder_side_points,
    )


def compute_bi_skew_priors(
    *,
    component_inits: ComponentInitializers,
    shoulder_side: jnp.ndarray,
    peak_lows: jnp.ndarray,
    peak_highs: jnp.ndarray,
    shoulder_keep: jnp.ndarray | None = None,
) -> BiSkewPriors:
    """Derive bi-skew population priors from component initializers.

    Args:
        component_inits: Component initializers with shape ``[n_trace, n_peak, 2]``.
        shoulder_side:   Shoulder side indicator per peak in ``{-1, 0, +1}``.
        peak_lows:       Peak window lower bounds, shape ``[n_peak]``.
        peak_highs:      Peak window upper bounds, shape ``[n_peak]``.
        shoulder_keep:   Optional per-trace shoulder validity mask.

    Returns:
        :class:`BiSkewPriors` with robust per-peak priors for the bi-skew model.
    """
    mu_init = np.asarray(component_inits.mu_init, dtype=float)
    sigma_init = np.asarray(component_inits.sigma_init, dtype=float)
    alpha_init = np.asarray(component_inits.alpha_init, dtype=float)
    area_init = np.asarray(component_inits.A_init, dtype=float)
    shoulder_side_v = np.asarray(shoulder_side, dtype=int).reshape(-1)
    lows = np.asarray(peak_lows, dtype=float).reshape(-1)
    highs = np.asarray(peak_highs, dtype=float).reshape(-1)

    if (
        mu_init.ndim != 3
        or sigma_init.shape != mu_init.shape
        or alpha_init.shape != mu_init.shape
        or area_init.shape != mu_init.shape
        or mu_init.shape[2] != 2
    ):
        raise ValueError(
            "component_inits must provide [n_trace, n_peak, 2] arrays with matching shapes."
        )
    n_trace, n_peak, _ = mu_init.shape
    if shoulder_side_v.shape[0] != n_peak:
        raise ValueError("shoulder_side length must match peak count.")
    if lows.shape[0] != n_peak or highs.shape[0] != n_peak:
        raise ValueError("peak_lows/peak_highs length must match peak count.")

    keep = (
        np.ones((n_trace, n_peak), dtype=bool)
        if shoulder_keep is None
        else np.asarray(shoulder_keep, dtype=bool)
    )
    if keep.shape != (n_trace, n_peak):
        raise ValueError(
            "shoulder_keep must have shape [n_trace, n_peak] when provided."
        )

    mu_center_loc = np.full((n_peak,), np.nan, dtype=float)
    mu_center_scale = np.full((n_peak,), np.nan, dtype=float)
    separation_low = np.zeros((n_peak,), dtype=float)
    separation_high = np.zeros((n_peak,), dtype=float)

    sigma_loc = np.full((n_peak, 2), np.nan, dtype=float)
    sigma_scale = np.full((n_peak, 2), np.nan, dtype=float)
    alpha_loc = np.full((n_peak, 2), np.nan, dtype=float)
    alpha_scale = np.full((n_peak, 2), np.nan, dtype=float)

    area_total_loc = np.full((n_peak,), np.nan, dtype=float)
    area_total_scale = np.full((n_peak,), np.nan, dtype=float)
    area_split_alpha = np.full((n_peak,), np.nan, dtype=float)
    area_split_beta = np.full((n_peak,), np.nan, dtype=float)

    def _robust_loc_scale(values: np.ndarray, *, floor: float) -> tuple[float, float]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return float("nan"), float("nan")
        loc = float(np.nanmedian(finite))
        mad = float(np.nanmedian(np.abs(finite - loc)))
        scale = max(_MAD_TO_SCALE * mad, float(floor))
        return loc, scale

    for p in range(n_peak):
        low = float(lows[p])
        high = float(highs[p])
        span = max(high - low, 1e-4)
        side = int(shoulder_side_v[p])
        has_shoulder = side != 0

        mu_main = mu_init[:, p, 0]
        mu_sh = mu_init[:, p, 1]
        sigma_main = sigma_init[:, p, 0]
        sigma_sh = sigma_init[:, p, 1]
        alpha_main = alpha_init[:, p, 0]
        alpha_sh = alpha_init[:, p, 1]
        area_main = np.maximum(area_init[:, p, 0], 0.0)
        area_sh = np.maximum(area_init[:, p, 1], 0.0)
        area_total = np.maximum(area_main + area_sh, 1e-12)

        if has_shoulder:
            center_vals = 0.5 * (mu_main + mu_sh)
        else:
            center_vals = mu_main
        center_loc, center_scale = _robust_loc_scale(
            center_vals, floor=max(0.02 * span, 1e-4)
        )
        if not np.isfinite(center_loc):
            center_loc = 0.5 * (low + high)
        if not np.isfinite(center_scale):
            center_scale = max(0.04 * span, 1e-4)
        center_loc = float(np.clip(center_loc, low, high))
        center_scale = float(max(center_scale, 1e-4))
        mu_center_loc[p] = center_loc
        mu_center_scale[p] = center_scale

        if has_shoulder:
            sep_floor = max(1e-4, 0.01 * span)
            sep_vals = side * (mu_sh - mu_main)
            sep_mask = np.isfinite(sep_vals) & (sep_vals > 0.0)
            sep_mask = sep_mask & keep[:, p]
            valid_sep = sep_vals[sep_mask]
            if valid_sep.size >= 3:
                q_low, q_high = np.nanquantile(valid_sep, [0.05, 0.95])
            elif valid_sep.size > 0:
                med = float(np.nanmedian(valid_sep))
                q_low = max(0.5 * med, sep_floor)
                q_high = 1.5 * med
            else:
                q_low = max(0.05 * span, sep_floor)
                q_high = max(0.20 * span, q_low + sep_floor)
            low_sep = max(float(q_low), sep_floor)
            high_sep = min(float(q_high), span - 1e-4)
            if high_sep <= low_sep + 1e-6:
                high_sep = min(max(low_sep + sep_floor, 0.25 * span), span - 1e-4)
            if high_sep <= low_sep + 1e-6:
                low_sep = max(min(low_sep, 0.45 * span), sep_floor)
                high_sep = min(max(low_sep + sep_floor, 0.60 * span), span - 1e-4)
            separation_low[p] = float(low_sep)
            separation_high[p] = float(max(high_sep, low_sep + 1e-6))
        else:
            separation_low[p] = 0.0
            separation_high[p] = 0.0

        sigma_main_loc, sigma_main_scale = _robust_loc_scale(
            sigma_main[sigma_main > 0.0], floor=1e-4
        )
        if not np.isfinite(sigma_main_loc):
            sigma_main_loc = max(span / 6.0, 1e-4)
        if not np.isfinite(sigma_main_scale):
            sigma_main_scale = max(0.20 * sigma_main_loc, 1e-4)

        sigma_sh_loc, sigma_sh_scale = _robust_loc_scale(
            sigma_sh[sigma_sh > 0.0], floor=1e-4
        )
        if not np.isfinite(sigma_sh_loc):
            sigma_sh_loc = max(0.75 * sigma_main_loc, 1e-4)
        if not np.isfinite(sigma_sh_scale):
            sigma_sh_scale = max(0.25 * sigma_sh_loc, 1e-4)

        alpha_main_loc, alpha_main_scale = _robust_loc_scale(alpha_main, floor=1e-3)
        if not np.isfinite(alpha_main_loc):
            alpha_main_loc = 0.0
        if not np.isfinite(alpha_main_scale):
            alpha_main_scale = 1.0

        alpha_sh_loc, alpha_sh_scale = _robust_loc_scale(alpha_sh, floor=1e-3)
        if not np.isfinite(alpha_sh_loc):
            alpha_sh_loc = 0.5 * alpha_main_loc
        if not np.isfinite(alpha_sh_scale):
            alpha_sh_scale = max(0.8 * alpha_main_scale, 1e-3)

        sigma_loc[p, 0] = float(max(sigma_main_loc, 1e-8))
        sigma_scale[p, 0] = float(max(sigma_main_scale, 1e-4))
        sigma_loc[p, 1] = float(max(sigma_sh_loc, 1e-8))
        sigma_scale[p, 1] = float(max(sigma_sh_scale, 1e-4))

        alpha_loc[p, 0] = float(alpha_main_loc)
        alpha_scale[p, 0] = float(max(alpha_main_scale, 1e-3))
        alpha_loc[p, 1] = float(alpha_sh_loc)
        alpha_scale[p, 1] = float(max(alpha_sh_scale, 1e-3))

        total_loc, total_scale = _robust_loc_scale(area_total, floor=1e-6)
        if not np.isfinite(total_loc):
            total_loc = max(float(np.nanmedian(area_main)), 1e-8)
        if not np.isfinite(total_scale):
            total_scale = max(0.25 * total_loc, 1e-6)
        area_total_loc[p] = float(max(total_loc, 1e-8))
        area_total_scale[p] = float(max(total_scale, 1e-6))

        if has_shoulder:
            ratio = area_main / np.maximum(area_total, 1e-12)
            ratio_mask = np.isfinite(ratio)
            ratio_mask = ratio_mask & keep[:, p]
            ratio_values = np.clip(ratio[ratio_mask], 1e-3, 1.0 - 1e-3)
            if ratio_values.size >= 3:
                mean_ratio = float(np.clip(np.mean(ratio_values), 0.05, 0.95))
                var_ratio = float(np.var(ratio_values, ddof=1))
                max_var = 0.95 * mean_ratio * (1.0 - mean_ratio)
                if (not np.isfinite(var_ratio)) or (var_ratio <= 1e-6) or (var_ratio >= max_var):
                    concentration = 40.0
                else:
                    concentration = mean_ratio * (1.0 - mean_ratio) / var_ratio - 1.0
                    concentration = float(np.clip(concentration, 2.0, 200.0))
            elif ratio_values.size > 0:
                mean_ratio = float(np.clip(np.median(ratio_values), 0.05, 0.95))
                concentration = 30.0
            else:
                mean_ratio = 0.85
                concentration = 25.0
            area_split_alpha[p] = max(mean_ratio * concentration, 1.1)
            area_split_beta[p] = max((1.0 - mean_ratio) * concentration, 1.1)
        else:
            area_split_alpha[p] = 120.0
            area_split_beta[p] = 1.2

    return BiSkewPriors(
        mu_center_loc=jnp.asarray(mu_center_loc, dtype=jnp.float32),
        mu_center_scale=jnp.asarray(mu_center_scale, dtype=jnp.float32),
        separation_low=jnp.asarray(separation_low, dtype=jnp.float32),
        separation_high=jnp.asarray(separation_high, dtype=jnp.float32),
        sigma_loc=jnp.asarray(sigma_loc, dtype=jnp.float32),
        sigma_scale=jnp.asarray(sigma_scale, dtype=jnp.float32),
        alpha_loc=jnp.asarray(alpha_loc, dtype=jnp.float32),
        alpha_scale=jnp.asarray(alpha_scale, dtype=jnp.float32),
        area_total_loc=jnp.asarray(area_total_loc, dtype=jnp.float32),
        area_total_scale=jnp.asarray(area_total_scale, dtype=jnp.float32),
        area_split_alpha=jnp.asarray(area_split_alpha, dtype=jnp.float32),
        area_split_beta=jnp.asarray(area_split_beta, dtype=jnp.float32),
    )


__all__ = [
    "ApexGate",
    "BiSkewPriors",
    "ComponentInitializers",
    "FwhmFeatures",
    "PeakPriors",
    "SkewNormalGuess",
    "compute_bi_skew_priors",
    "build_two_stage_component_initializers",
    "compute_peak_fwhm_features",
    "compute_skew_normal_guess",
    "kde_apex_gate",
    "median_and_scale",
    "skew_mode_time",
]
