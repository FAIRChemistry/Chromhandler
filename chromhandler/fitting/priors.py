"""Single-peak skew-normal priors.

Builds one :class:`SkewNormalPriors` per :class:`PeakAnnotation` from a
:class:`PreparedDataset`. The flow per window is:

1. Per-trace FWHM measurement (via Savitzky-Golay + half-max crossings)
   yields ``(mu, width, skew, area)`` for traces that pass the gate.
2. Per-trace gating: a trace is *supported* iff its max raw signal in the
   window is at least :attr:`PriorConfig.signal_threshold` (absolute, no
   baseline subtraction) AND the relative-height gate passes.
3. Aggregation across supported traces yields shared shape priors
   ``(mu_loc, mu_scale, width_loc, width_log_scale, skew_*)``.
4. Per-trace **LogNormal** area priors:
   - Supported: ``LogNormal(log(area_measured), area_sigma_log)`` — median
     anchored at the measured trapezoid area; scale is a FIXED config
     constant (data-independent, removing empirical-Bayes double-counting).
   - Unsupported: ``LogNormal(log(noise_floor_area), area_sigma_log)`` —
     median anchored at the noise floor so the area is weakly shrunk
     toward zero but can never reach it (positivity by construction).

Doublet / control / artefact logic has been removed: this module supports
``mode="single"`` peaks only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import savgol_filter

from chromhandler.fitting.skew_normal import GAMMA1_MAX, cp_from_peak_features

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
    skew_bound_fraction: float = 0.99
    width_low_n_points_per_fwhm: int = 8
    width_high_window_fraction: float = 6.0

    # --- n_supported = 1 fallbacks ---
    log_width_scale_n1: float = 0.15
    skew_scale_n1: float = 0.20

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

    # --- Universal floors ---
    mu_scale_dt_floor_multiplier: float = 1.0


@dataclass(frozen=True)
class SkewNormalPriors:
    """Single-peak skew-normal priors for one window.

    Each field maps to exactly one NumPyro sample site in :mod:`model`:

    - ``mu ~ Normal(mu_loc, mu_scale)`` — unbounded; the likelihood + a
      positive area prior identify it.
    - ``width ~ LogNormal``: parameterised by ``width_loc`` (natural-space
      median, in time units) and ``width_log_scale`` (sigma of the
      underlying Normal on the log axis). The model converts back via
      ``log(width_loc)`` at fit time. ``width_log_scale`` stays on the
      log axis because the LogNormal has no 1-number natural-space scale.
    - ``skew ~ Normal(skew_loc, skew_scale)`` passed through a ``tanh``
      bijector bounded by ``GAMMA1_MAX`` (the skew-normal family's max
      attainable skewness coefficient). Bound is soft, not truncated.
    - ``area[trace] ~ LogNormal(log(area_loc_per_trace), area_log_scale)``
      — positive by construction (exp), so it never reaches 0; ``loc``
      anchors at the measured trapezoid area (supported) or a noise-floor
      area (unsupported).

    ``mu`` is interpreted by ``density_cp`` as the **mean** (CP form); the
    cp -> dp bijection inside ``density_cp`` performs the canonical
    xi-transformation, so users do not need to reason about xi directly.
    """

    mu_loc: float
    mu_scale: float

    width_loc: float
    """Natural-space median of the LogNormal prior on ``width`` (time units)."""
    width_log_scale: float
    """Sigma of the underlying Normal on the log axis. Stays log-axis
    because a LogNormal has no clean 1-number natural-space scale."""

    skew_loc: float
    skew_scale: float

    area_loc_per_trace: NDArray[np.float64]
    """Per-trace linear-space LogNormal median (strictly positive, >= the
    noise-floor area). Positivity is what keeps the per-trace area<->warp
    geometry out of a funnel."""
    area_log_scale: float
    """Fixed sigma on ``log(area)`` (from ``PriorConfig.area_sigma_log``),
    shared across traces. NOT data-derived — removes the precision
    double-counting of the old ``0.3 * area_measured`` scale."""
    has_support_per_trace: NDArray[np.bool_]


@dataclass(frozen=True)
class WindowFeatures:
    """Per-trace, per-window FWHM-based features. ``None`` when unmeasurable."""

    mu: float
    width: float
    skew: float
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
    """Per-trace skew-normal CP features for one peak window.

    Measures three quantities from the smoothed signal — apex location,
    full width at half maximum, and right/left HWHM ratio — then inverts
    them to centred-parameter ``(mu, width, skew)`` via
    :func:`cp_from_peak_features`. Returning CP directly keeps the priors
    aligned with the model's parameterisation (apex != mean, and the
    Gaussian FWHM-to-sigma rule does not hold for skewed peaks).

    Ratios and FWHM are averaged across multiple smoothing scales before
    inversion, which suppresses noise without Jensen-biasing the final
    CP parameters.

    Returns ``None`` if too few valid points or if half-max cannot be
    bracketed at any smoothing scale.
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
    apex = float(t[apex_idx])
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
    mean_fwhm = float(np.mean(hwhm_sums))
    mu, width, skew = cp_from_peak_features(apex, mean_fwhm, mean_ratio)
    return WindowFeatures(
        mu=mu, width=width, skew=skew, apex_height=apex_height,
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
    # Signed integration. Clipping with max(.,0) before integrating rectifies
    # the noise and biases the area up by ~sigma/sqrt(2pi)*window_width
    # (spurious for weak/absent windows, ~few % for real peaks). Positivity
    # of the prior median is handled downstream by the noise-floor, so the
    # integral itself must stay signed.
    return float(np.trapezoid(signal_baseline_subtracted[mask], time[mask]))


def _skew_bounds(config: PriorConfig) -> tuple[float, float]:
    bound = config.skew_bound_fraction * GAMMA1_MAX
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
    (mu_loc, mu_scale, width_loc, width_log_scale, skew_loc, skew_scale)

    ``width_loc`` is the natural-space median of the LogNormal prior on
    width (geometric mean of measured widths). ``width_log_scale`` is the
    sigma of the underlying Normal on the log axis.
    """
    n = len(features)
    if n == 0:
        # No supported traces — fall back to window-geometry priors.
        mu_loc = 0.5 * (window_low + window_high)
        mu_scale = (window_high - window_low) / 4.0
        # Geometric width: span / 6 (~ +/- 3 std-dev across the window).
        width_loc = max(
            (window_high - window_low) / 6.0,
            config.width_low_n_points_per_fwhm * dt * _FWHM_TO_SIGMA,
        )
        width_log_scale = config.log_width_scale_n1
        skew_loc = 0.0
        skew_scale = config.skew_scale_n1
        return (
            mu_loc, mu_scale, width_loc, width_log_scale,
            skew_loc, skew_scale,
        )

    mus = np.asarray([f.mu for f in features], dtype=np.float64)
    widths = np.asarray([f.width for f in features], dtype=np.float64)
    skews = np.asarray([f.skew for f in features], dtype=np.float64)
    log_widths = np.log(np.clip(widths, 1e-9, None))

    mu_floor = config.mu_scale_dt_floor_multiplier * dt
    mu_loc = float(np.mean(mus))
    mu_scale = float(max(np.std(mus, ddof=0), mu_floor))

    # Geometric mean of measured widths = exp(mean(log_widths)) = LogNormal median.
    width_loc = float(np.exp(np.mean(log_widths)))
    if n == 1:
        width_log_scale = config.log_width_scale_n1
    else:
        width_log_scale = float(max(
            np.std(log_widths, ddof=0),
            config.log_width_scale_n1 / float(np.sqrt(n)),
        ))

    skew_loc = float(np.mean(skews))
    if n == 1:
        skew_scale = config.skew_scale_n1
    else:
        skew_scale = float(max(
            np.std(skews, ddof=0),
            config.skew_scale_n1 / float(np.sqrt(n)),
        ))
    _, skew_bound_high = _skew_bounds(config)
    skew_scale = min(skew_scale, skew_bound_high)

    return (
        mu_loc, mu_scale, width_loc, width_log_scale,
        skew_loc, skew_scale,
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
        mu_loc, mu_scale, width_loc, width_log_scale,
        skew_loc, skew_scale,
    ) = _aggregate_shape_priors(
        supported_features, ann.rt_min, ann.rt_max, dataset.dt_global, config,
    )

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

    return SkewNormalPriors(
        mu_loc=mu_loc, mu_scale=mu_scale,
        width_loc=width_loc, width_log_scale=width_log_scale,
        skew_loc=skew_loc, skew_scale=skew_scale,
        area_loc_per_trace=area_loc_per_trace,
        area_log_scale=float(config.area_sigma_log),
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
    skew_low, skew_high = _skew_bounds(config)
    # All values shown in natural space. The "p16 / p84" columns are
    # the ±1-sigma quantiles of each prior, derived per distribution:
    #   Normal:           loc -/+ scale
    #   LogNormal:        loc * exp(-/+ log_scale)   (loc is the median)
    #   Normal + tanh:    loc -/+ scale  (the tanh distortion is mild
    #                     unless |loc|+|scale| approaches GAMMA1_MAX)
    header = (
        f"{'peak':>4} {'site':<14} {'distribution':<18} "
        f"{'loc':>10} {'scale':>10} {'p16':>10} {'p84':>10}"
    )
    lines = [header, "-" * len(header)]
    for i, p in enumerate(priors):
        # mu: Normal
        lines.append(
            f"{i:>4} {'mu':<14} {'Normal':<18} "
            f"{p.mu_loc:>10.4g} {p.mu_scale:>10.4g} "
            f"{p.mu_loc - p.mu_scale:>10.4g} {p.mu_loc + p.mu_scale:>10.4g}"
        )
        # width: LogNormal (loc = natural-space median, scale = log-axis sigma)
        w_p16 = p.width_loc * float(np.exp(-p.width_log_scale))
        w_p84 = p.width_loc * float(np.exp(+p.width_log_scale))
        lines.append(
            f"{i:>4} {'width':<14} {'LogNormal':<18} "
            f"{p.width_loc:>10.4g} {p.width_log_scale:>10.4g} "
            f"{w_p16:>10.4g} {w_p84:>10.4g}"
        )
        # skew: Normal + tanh bound (soft)
        lines.append(
            f"{i:>4} {'skew':<14} {'Normal+tanh':<18} "
            f"{p.skew_loc:>10.4g} {p.skew_scale:>10.4g} "
            f"{p.skew_loc - p.skew_scale:>10.4g} {p.skew_loc + p.skew_scale:>10.4g}"
        )
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
    skew_note = f"  (skew tanh bound: [{skew_low:.4g}, {skew_high:.4g}])"
    lines.append(skew_note)
    return "\n".join(lines)
