"""Single-peak skew-normal priors.

Builds one :class:`SkewNormalPriors` per :class:`PeakAnnotation` from a
:class:`PreparedDataset`. The flow per window is:

1. Per-trace FWHM measurement (via Savitzky-Golay + half-max crossings)
   yields ``(mu, sigma, gamma1, area)`` for traces that pass the gate.
2. Per-trace gating: a trace is *supported* iff its max raw signal in the
   window is at least :attr:`PriorConfig.signal_threshold` (absolute, no
   baseline subtraction) AND the relative-height gate passes.
3. Aggregation across supported traces yields shared shape priors
   ``(mu_loc, mu_scale, log_sigma_*, gamma1_*)``.
4. Per-trace **linear-space** area priors:
   - Supported: ``TruncatedNormal(area_measured, cv * area_measured, low=0)``.
   - Unsupported: ``TruncatedNormal(0, noise * window_width * multiplier, low=0)``
     i.e. half-normal-at-zero — the area can collapse to zero when the
     data doesn't support a peak.

Doublet / control / artefact logic has been removed: this module supports
``mode="single"`` peaks only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import savgol_filter

from chromhandler.fitting.skew_normal import GAMMA1_MAX, sn_asymmetry_to_gamma1

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from chromhandler.annotations import PeakAnnotation
    from chromhandler.fitting.prepared_dataset import PreparedDataset


_FWHM_TO_SIGMA: float = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


@dataclass(frozen=True)
class PriorConfig:
    """Configuration for prior construction. Override fields to tune."""

    # --- FWHM feature extraction ---
    smoothing_window: int = 5

    # --- Gating ---
    signal_threshold: float | None = None
    """Absolute raw-signal threshold (no baseline subtraction). Traces
    whose maximum signal in the window is below this value are treated
    as having no peak: they're excluded from shape aggregation and get
    a zero-anchored area prior. ``None`` disables the absolute gate."""

    min_height_frac: float = 0.05
    """Relative-height gate: a trace contributes to FWHM aggregation
    only if its in-window apex height is at least this fraction of the
    dataset's max in-window height for the same window."""

    # --- Shape-prior bounds ---
    gamma1_bound_fraction: float = 0.99
    sigma_low_n_points_per_fwhm: int = 8
    sigma_high_window_fraction: float = 6.0

    # --- n_supported = 1 fallbacks ---
    log_sigma_scale_n1: float = 0.15
    gamma1_scale_n1: float = 0.20

    # --- Area prior ---
    area_cv: float = 0.3
    """Coefficient of variation for supported-trace TruncatedNormal area
    prior: ``area ~ TruncatedNormal(loc=measured, scale=cv*measured, low=0)``."""

    area_zero_noise_multiplier: float = 3.0
    """Scale multiplier for unsupported traces:
    ``area ~ TruncatedNormal(loc=0, scale=multiplier * noise * window_width, low=0)``."""

    # --- Universal floors ---
    mu_scale_dt_floor_multiplier: float = 1.0


@dataclass(frozen=True)
class SkewNormalPriors:
    """Single-peak skew-normal priors for one window.

    Each field maps to exactly one NumPyro sample site in :mod:`model`:

    - ``mu ~ TruncatedNormal(mu_loc, mu_scale, mu_low, mu_high)``
    - ``log_sigma ~ TruncatedNormal(log_sigma_loc, log_sigma_scale,
      log_sigma_low, log_sigma_high)``
    - ``gamma1 ~ TruncatedNormal(gamma1_loc, gamma1_scale,
      -GAMMA1_MAX*frac, +GAMMA1_MAX*frac)``
    - ``A[trace] ~ TruncatedNormal(area_loc_per_trace[trace],
      area_scale_per_trace[trace], low=0)`` — half-normal-at-zero for
      unsupported traces.

    ``mu`` is interpreted by ``density_cp`` as the **mean** (CP form); the
    cp→dp bijection inside ``density_cp`` performs the canonical
    ξ-transformation, so users do not need to reason about ξ directly.
    """

    mu_loc: float
    mu_scale: float
    mu_low: float
    mu_high: float

    log_sigma_loc: float
    log_sigma_scale: float
    log_sigma_low: float
    log_sigma_high: float

    gamma1_loc: float
    gamma1_scale: float

    area_loc_per_trace: NDArray[np.float64]
    area_scale_per_trace: NDArray[np.float64]
    has_support_per_trace: NDArray[np.bool_]


@dataclass(frozen=True)
class WindowFeatures:
    """Per-trace, per-window FWHM-based features. ``None`` when unmeasurable."""

    mu: float
    sigma: float
    gamma1: float
    area: float
    apex_height: float


def _interp_threshold_crossing(
    t: NDArray[np.float64],
    s: NDArray[np.float64],
    apex_idx: int,
    threshold: float,
    direction: int,
) -> float | None:
    i = apex_idx
    n = s.size
    while 0 <= i + direction < n and s[i + direction] >= threshold:
        i += direction
    j = i + direction
    if not (0 <= j < n):
        return None
    if s[i] == s[j]:
        return float(t[i])
    f = (s[i] - threshold) / (s[i] - s[j])
    return float(t[i] + f * (t[j] - t[i]))


def compute_window_features(
    time: NDArray[np.float64],
    signal_baseline_subtracted: NDArray[np.float64],
    window_low: float,
    window_high: float,
    smoothing_window: int = 5,
) -> WindowFeatures | None:
    """Per-trace FWHM features for one peak window.

    Returns ``None`` if too few valid points or if half-max cannot be
    bracketed at any smoothing scale. Ratios across multiple smoothing
    widths are averaged before the ``sn_asymmetry_to_gamma1`` transform
    to avoid Jensen-bias from averaging ``gamma1`` directly.
    """
    mask = (
        (time >= window_low)
        & (time <= window_high)
        & np.isfinite(signal_baseline_subtracted)
    )
    t = np.asarray(time[mask], dtype=np.float64)
    s = np.asarray(signal_baseline_subtracted[mask], dtype=np.float64)
    if s.size < smoothing_window:
        return None

    n = s.size
    widths_raw = [
        smoothing_window,
        max(smoothing_window, int(0.10 * n)),
        max(smoothing_window, int(0.15 * n)),
        max(smoothing_window, int(0.20 * n)),
    ]
    widths: list[int] = sorted({w + (1 - w % 2) for w in widths_raw})

    poly_min = min(3, smoothing_window - 1)
    s_ref: NDArray[np.float64] = np.asarray(
        savgol_filter(s, smoothing_window, poly_min), dtype=np.float64
    )
    apex_idx = int(np.argmax(s_ref))
    mu = float(t[apex_idx])
    apex_height = float(s_ref[apex_idx])

    ratios: list[float] = []
    hwhm_sums: list[float] = []
    for w in widths:
        poly = min(3, w - 1)
        s_w: NDArray[np.float64] = np.asarray(
            savgol_filter(s, w, poly), dtype=np.float64
        )
        a_idx = int(np.argmax(s_w))
        mu_w = float(t[a_idx])
        half = float(s_w[a_idx]) / 2.0
        tl = _interp_threshold_crossing(t, s_w, a_idx, half, -1)
        tr = _interp_threshold_crossing(t, s_w, a_idx, half, +1)
        if tl is None or tr is None:
            continue
        hl = mu_w - tl
        hr = tr - mu_w
        if hl <= 0 or hr <= 0:
            continue
        ratios.append(hr / hl)
        hwhm_sums.append(hl + hr)

    if not ratios:
        return None

    mean_ratio = float(np.mean(ratios))
    sigma = float(np.mean(hwhm_sums)) * _FWHM_TO_SIGMA
    gamma1 = float(sn_asymmetry_to_gamma1(mean_ratio))  # type: ignore[arg-type]
    area = float(np.trapezoid(np.maximum(s, 0.0), t))
    return WindowFeatures(
        mu=mu, sigma=sigma, gamma1=gamma1, area=area, apex_height=apex_height,
    )


def _trace_passes_gate(
    raw_signal_in_window: NDArray[np.float64],
    threshold: float | None,
) -> bool:
    """Absolute raw-signal gate: max(raw signal in window) >= threshold."""
    if threshold is None:
        return True
    finite = raw_signal_in_window[np.isfinite(raw_signal_in_window)]
    if finite.size == 0:
        return False
    return bool(np.max(finite) >= threshold)


def _trapezoid_in_window(
    time: NDArray[np.float64],
    signal_baseline_subtracted: NDArray[np.float64],
    window_low: float,
    window_high: float,
) -> float:
    mask = (
        (time >= window_low)
        & (time <= window_high)
        & np.isfinite(signal_baseline_subtracted)
    )
    if mask.sum() < 2:
        return 0.0
    return float(np.trapezoid(
        np.maximum(signal_baseline_subtracted[mask], 0.0),
        time[mask],
    ))


def _log_sigma_bounds(
    window_low: float,
    window_high: float,
    dt: float,
    config: PriorConfig,
) -> tuple[float, float]:
    sigma_low = config.sigma_low_n_points_per_fwhm * dt * _FWHM_TO_SIGMA
    sigma_high = (window_high - window_low) / config.sigma_high_window_fraction
    return float(np.log(sigma_low)), float(np.log(sigma_high))


def _gamma1_bounds(config: PriorConfig) -> tuple[float, float]:
    bound = config.gamma1_bound_fraction * GAMMA1_MAX
    return float(-bound), float(bound)


def _aggregate_shape_priors(
    features: list[WindowFeatures],
    window_low: float,
    window_high: float,
    dt: float,
    config: PriorConfig,
) -> tuple[float, float, float, float, float, float]:
    """Aggregate per-trace features into shape priors.

    Returns
    -------
    (mu_loc, mu_scale, log_sigma_loc, log_sigma_scale, gamma1_loc, gamma1_scale)
    """
    n = len(features)
    if n == 0:
        # No supported traces — fall back to window-geometry priors.
        mu_loc = 0.5 * (window_low + window_high)
        mu_scale = (window_high - window_low) / 4.0
        # Geometric sigma: span / 6 (~ +/- 3 std-dev across the window).
        sigma_fallback = max(
            (window_high - window_low) / 6.0,
            config.sigma_low_n_points_per_fwhm * dt * _FWHM_TO_SIGMA,
        )
        log_sigma_loc = float(np.log(sigma_fallback))
        log_sigma_scale = config.log_sigma_scale_n1
        gamma1_loc = 0.0
        gamma1_scale = config.gamma1_scale_n1
        return (
            mu_loc, mu_scale, log_sigma_loc, log_sigma_scale,
            gamma1_loc, gamma1_scale,
        )

    mus = np.asarray([f.mu for f in features], dtype=np.float64)
    sigmas = np.asarray([f.sigma for f in features], dtype=np.float64)
    gamma1s = np.asarray([f.gamma1 for f in features], dtype=np.float64)
    log_sigmas = np.log(np.clip(sigmas, 1e-9, None))

    mu_floor = config.mu_scale_dt_floor_multiplier * dt
    mu_loc = float(np.mean(mus))
    mu_scale = float(max(np.std(mus, ddof=0), mu_floor))

    log_sigma_loc = float(np.mean(log_sigmas))
    if n == 1:
        log_sigma_scale = config.log_sigma_scale_n1
    else:
        log_sigma_scale = float(max(
            np.std(log_sigmas, ddof=0),
            config.log_sigma_scale_n1 / float(np.sqrt(n)),
        ))

    gamma1_loc = float(np.mean(gamma1s))
    if n == 1:
        gamma1_scale = config.gamma1_scale_n1
    else:
        gamma1_scale = float(max(
            np.std(gamma1s, ddof=0),
            config.gamma1_scale_n1 / float(np.sqrt(n)),
        ))
    _, gamma1_bound_high = _gamma1_bounds(config)
    gamma1_scale = min(gamma1_scale, gamma1_bound_high)

    return (
        mu_loc, mu_scale, log_sigma_loc, log_sigma_scale,
        gamma1_loc, gamma1_scale,
    )


def _build_one_peak(
    dataset: PreparedDataset,
    baseline_sub: NDArray[np.float64],
    ann: PeakAnnotation,
    config: PriorConfig,
) -> SkewNormalPriors:
    n_trace = dataset.n_trace
    window_width = float(ann.rt_max - ann.rt_min)

    # --- Stage 1: per-trace raw-signal gate + FWHM measurement -----------
    has_support = np.zeros(n_trace, dtype=np.bool_)
    apex_heights = np.zeros(n_trace, dtype=np.float64)
    features_per_trace: list[WindowFeatures | None] = [None] * n_trace
    areas_measured = np.zeros(n_trace, dtype=np.float64)

    for tr in range(n_trace):
        t = dataset.time[tr]
        s_raw = dataset.signal[tr]
        in_win = (t >= ann.rt_min) & (t <= ann.rt_max) & np.isfinite(s_raw)
        if not _trace_passes_gate(s_raw[in_win], config.signal_threshold):
            continue

        feats = compute_window_features(
            t, baseline_sub[tr], ann.rt_min, ann.rt_max,
            smoothing_window=config.smoothing_window,
        )
        if feats is None:
            continue
        features_per_trace[tr] = feats
        apex_heights[tr] = max(feats.apex_height, 0.0)
        areas_measured[tr] = _trapezoid_in_window(
            t, baseline_sub[tr], ann.rt_min, ann.rt_max,
        )
        has_support[tr] = True

    # --- Stage 2: relative-height gate within the supported subset -------
    if has_support.any():
        max_height = float(np.max(apex_heights[has_support]))
        if max_height > 0:
            keep = apex_heights >= config.min_height_frac * max_height
            has_support = has_support & keep

    supported_features: list[WindowFeatures] = [
        features_per_trace[tr]  # type: ignore[misc]
        for tr in range(n_trace)
        if has_support[tr] and features_per_trace[tr] is not None
    ]

    # --- Stage 3: aggregate shape priors ---------------------------------
    (
        mu_loc, mu_scale, log_sigma_loc, log_sigma_scale,
        gamma1_loc, gamma1_scale,
    ) = _aggregate_shape_priors(
        supported_features, ann.rt_min, ann.rt_max, dataset.dt_global, config,
    )
    log_sigma_low, log_sigma_high = _log_sigma_bounds(
        ann.rt_min, ann.rt_max, dataset.dt_global, config,
    )

    # --- Stage 4: per-trace area priors ----------------------------------
    area_zero_scale = (
        config.area_zero_noise_multiplier
        * dataset.noise_per_trace
        * window_width
    )
    area_loc_per_trace = np.where(has_support, areas_measured, 0.0)
    area_scale_per_trace = np.where(
        has_support,
        np.maximum(config.area_cv * areas_measured, area_zero_scale),
        area_zero_scale,
    )
    area_scale_per_trace = np.maximum(area_scale_per_trace, 1e-9)

    return SkewNormalPriors(
        mu_loc=mu_loc, mu_scale=mu_scale,
        mu_low=ann.rt_min, mu_high=ann.rt_max,
        log_sigma_loc=log_sigma_loc, log_sigma_scale=log_sigma_scale,
        log_sigma_low=log_sigma_low, log_sigma_high=log_sigma_high,
        gamma1_loc=gamma1_loc, gamma1_scale=gamma1_scale,
        area_loc_per_trace=area_loc_per_trace,
        area_scale_per_trace=area_scale_per_trace,
        has_support_per_trace=has_support,
    )


def build_priors(
    dataset: PreparedDataset,
    config: PriorConfig | None = None,
) -> list[SkewNormalPriors]:
    """Build per-annotation single-peak skew-normal priors."""
    cfg = config if config is not None else PriorConfig()
    baseline_sub = dataset.signal - (
        dataset.baseline_intercept[:, None]
        + dataset.baseline_slope[:, None] * dataset.time
    )
    return [
        _build_one_peak(dataset, baseline_sub, ann, cfg)
        for ann in dataset.peak_annotations
    ]


def summarise_priors(
    priors: list[SkewNormalPriors],
    config: PriorConfig,
) -> str:
    """Pretty-printed multi-line table for inspection."""
    gamma1_low, gamma1_high = _gamma1_bounds(config)
    header = (
        f"{'peak':>4} {'site':<14} {'distribution':<16} "
        f"{'loc':>10} {'scale':>10} {'low':>10} {'high':>10}"
    )
    lines = [header, "-" * len(header)]
    for i, p in enumerate(priors):
        lines.append(
            f"{i:>4} {'mu':<14} {'TruncatedNormal':<16} "
            f"{p.mu_loc:>10.4g} {p.mu_scale:>10.4g} "
            f"{p.mu_low:>10.4g} {p.mu_high:>10.4g}"
        )
        lines.append(
            f"{i:>4} {'log_sigma':<14} {'TruncatedNormal':<16} "
            f"{p.log_sigma_loc:>10.4g} {p.log_sigma_scale:>10.4g} "
            f"{p.log_sigma_low:>10.4g} {p.log_sigma_high:>10.4g}"
        )
        lines.append(
            f"{i:>4} {'gamma1':<14} {'TruncatedNormal':<16} "
            f"{p.gamma1_loc:>10.4g} {p.gamma1_scale:>10.4g} "
            f"{gamma1_low:>10.4g} {gamma1_high:>10.4g}"
        )
        n_supp = int(np.sum(p.has_support_per_trace))
        n_total = p.has_support_per_trace.size
        mean_area = float(np.mean(p.area_loc_per_trace))
        lines.append(
            f"{i:>4} {'area (mean)':<14} {'TruncNormal':<16} "
            f"{mean_area:>10.4g} {'-':>10} {0.0:>10.4g} {'-':>10}"
            f"  [supported {n_supp}/{n_total}]"
        )
    return "\n".join(lines)
