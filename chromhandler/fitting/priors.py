"""Controls-based prior construction for the skew-normal peak model.

This module turns a ``PreparedDataset`` plus its ``PeakAnnotation`` list
into a list of :class:`SkewNormalPriors`, one per peak.

All magic numbers and fallback heuristics live in :class:`PriorConfig`.
Users can override the config to change behaviour; defaults are tuned for
typical chromatographic data.

For ``artefact_doublet`` peaks, all artefact-related priors are derived
**directly from control traces** (samples with no analyte). For shape
quantities where only one control is available, scale fallbacks borrow
from the analyte's empirical population (same chromatographic system ->
same drift and shape variation).
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


@dataclass(frozen=True)
class PriorConfig:
    """Centralised configuration for prior construction.

    All knobs in one place — users can override any field to change
    behaviour without touching the priors module itself.
    """

    # --- Distribution bounds (geometric / mathematical) ---
    gamma1_bound_fraction: float = 0.99
    sigma_low_n_points_per_fwhm: int = 8
    sigma_high_window_fraction: float = 6.0
    delta_low_dt_multiplier: float = 3.0
    delta_high_window_fraction: float = 2.0

    # --- Side check ---
    side_check_epsilon_dt_multiplier: float = 3.0

    # --- n=1 control fallbacks ---
    delta_scale_dt_multiplier_n1: float = 1.5
    log_A_artefact_min_scale: float = 0.2

    # --- Single-trace fallbacks (n_trace=1 in single-peak aggregation) ---
    log_sigma_scale_n1: float = 0.15
    gamma1_scale_n1: float = 0.20
    log_A_scale_n1_min: float = 0.10

    # --- Universal floors ---
    mu_scale_dt_floor_multiplier: float = 1.0


@dataclass(frozen=True)
class SkewNormalPriors:
    """Empirical priors for one peak window.

    Each field parameterizes exactly one NumPyro distribution in
    ``model.py`` per the distribution table at the top of the priors plan.
    ``_left_*`` fields are always populated; ``_right_*`` and ``Delta_*``
    are populated iff ``n_components == 2``.
    """

    n_components: int

    mu_left_loc: float
    mu_left_scale: float
    mu_left_low: float
    mu_left_high: float

    log_sigma_left_loc: float
    log_sigma_left_scale: float
    log_sigma_left_low: float
    log_sigma_left_high: float

    gamma1_left_loc: float
    gamma1_left_scale: float

    log_A_left_loc_per_trace: NDArray[np.float64]
    log_A_left_scale: float

    Delta_loc: float | None
    Delta_scale: float | None
    Delta_low: float | None
    Delta_high: float | None

    log_sigma_right_loc: float | None
    log_sigma_right_scale: float | None
    log_sigma_right_low: float | None
    log_sigma_right_high: float | None

    gamma1_right_loc: float | None
    gamma1_right_scale: float | None

    log_A_right_loc_per_trace: NDArray[np.float64] | None
    log_A_right_scale: float | None

    def __post_init__(self) -> None:
        right_fields = (
            self.Delta_loc, self.Delta_scale, self.Delta_low, self.Delta_high,
            self.log_sigma_right_loc, self.log_sigma_right_scale,
            self.log_sigma_right_low, self.log_sigma_right_high,
            self.gamma1_right_loc, self.gamma1_right_scale,
            self.log_A_right_loc_per_trace, self.log_A_right_scale,
        )
        if self.n_components == 1:
            if any(f is not None for f in right_fields):
                raise ValueError(
                    "Single-component priors require all right-component "
                    "fields (Delta_*, *_right_*) to be None."
                )
        elif self.n_components == 2:
            if any(f is None for f in right_fields):
                raise ValueError(
                    "For doublet peaks, all right-component fields are required "
                    "(Delta_*, *_right_*); got at least one None."
                )
        else:
            raise ValueError(f"n_components must be 1 or 2, got {self.n_components}.")


_FWHM_TO_SIGMA: float = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


@dataclass(frozen=True)
class WindowFeatures:
    """Per-trace, per-window FWHM-based features.

    Attributes:
        mu: Apex location (minutes), smoothed argmax inside the window.
        sigma: ``(HWHM_L + HWHM_R) * FWHM_TO_SIGMA``.
        gamma1: ``sn_asymmetry_to_gamma1(HWHM_R / HWHM_L)``.
        area: ``trapezoid(signal, time)`` over the window.
    """

    mu: float
    sigma: float
    gamma1: float
    area: float


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


def compute_single_window_features(
    time: NDArray[np.float64],
    signal_baseline_subtracted: NDArray[np.float64],
    window_low: float,
    window_high: float,
    smoothing_window: int = 5,
) -> WindowFeatures:
    """Extract FWHM-based features from a single-peak window.

    ``mu`` is the smoothed argmax. ``sigma`` and ``gamma1`` are derived from
    HWHM crossings estimated across several Savitzky-Golay window widths; the
    HWHM ratios (``HWHM_R / HWHM_L``) are averaged before the nonlinear
    ``sn_asymmetry_to_gamma1`` transform, which avoids the Jensen's-inequality
    downward bias that would arise from averaging ``gamma1`` estimates directly.

    Args:
        time: 1-D time array.
        signal_baseline_subtracted: 1-D baseline-subtracted signal.
        window_low: Inclusive lower bound.
        window_high: Inclusive upper bound.
        smoothing_window: Minimum Savitzky-Golay window length (odd, >= 5).
            Additional wider windows are also tried automatically.

    Returns:
        :class:`WindowFeatures`.

    Raises:
        ValueError: If too few valid points in window, or half-max never
            resolved on either side for any smoothing scale.
    """
    mask = (time >= window_low) & (time <= window_high) & np.isfinite(
        signal_baseline_subtracted
    )
    t = np.asarray(time[mask], dtype=np.float64)
    s = np.asarray(signal_baseline_subtracted[mask], dtype=np.float64)
    if s.size < smoothing_window:
        raise ValueError(
            f"Window [{window_low}, {window_high}] has only {s.size} valid "
            f"points; need at least {smoothing_window}."
        )

    # Build a set of odd smoothing-window widths ranging from smoothing_window
    # up to ~20 % of the data length.  Averaging ratios across multiple scales
    # reduces variance and keeps the mean approximately unbiased.
    n = s.size
    widths_raw = [
        smoothing_window,
        max(smoothing_window, int(0.10 * n)),
        max(smoothing_window, int(0.15 * n)),
        max(smoothing_window, int(0.20 * n)),
    ]
    widths: list[int] = sorted({w + (1 - w % 2) for w in widths_raw})  # ensure odd

    # Use the minimum width for mu (best spatial resolution).
    poly_min = min(3, smoothing_window - 1)
    s_ref: NDArray[np.float64] = np.asarray(
        savgol_filter(s, smoothing_window, poly_min), dtype=np.float64
    )
    apex_idx = int(np.argmax(s_ref))
    mu = float(t[apex_idx])

    # Collect (ratio, hwhm_sum) from each width; skip widths where either
    # crossing cannot be found.
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
        raise ValueError(
            f"Could not bracket half-max in window [{window_low}, {window_high}]."
        )

    mean_ratio = float(np.mean(ratios))
    sigma = float(np.mean(hwhm_sums)) * _FWHM_TO_SIGMA
    gamma1 = float(sn_asymmetry_to_gamma1(mean_ratio))  # type: ignore[arg-type]
    area = float(np.trapezoid(s, t))
    return WindowFeatures(mu=mu, sigma=sigma, gamma1=gamma1, area=area)


def detect_dominant_apex(
    time: NDArray[np.float64],
    signal_baseline_subtracted: NDArray[np.float64],
    window_low: float,
    window_high: float,
    smoothing_window: int = 5,
) -> tuple[float, float]:
    """Locate the dominant apex inside a window via smoothed argmax."""
    mask = (time >= window_low) & (time <= window_high) & np.isfinite(
        signal_baseline_subtracted
    )
    t = np.asarray(time[mask], dtype=np.float64)
    s = np.asarray(signal_baseline_subtracted[mask], dtype=np.float64)
    if s.size < smoothing_window:
        raise ValueError(
            f"Window [{window_low}, {window_high}] has only {s.size} valid "
            f"points; need at least {smoothing_window}."
        )
    polyorder = min(3, smoothing_window - 1)
    s_smooth: NDArray[np.float64] = np.asarray(
        savgol_filter(s, smoothing_window, polyorder), dtype=np.float64
    )
    idx = int(np.argmax(s_smooth))
    return float(t[idx]), float(s_smooth[idx])


@dataclass(frozen=True)
class ArtefactMeasurements:
    """Raw artefact measurements from control traces + analyte-residual inputs.

    Outputs of :func:`extract_artefact_from_controls`. Scale assembly is
    deferred to :func:`aggregate_doublet_priors`, which has access to
    analyte-side scales for principled borrowing.

    Attributes:
        mu_per_control: ``[n_controls]`` per-control apex locations.
        log_sigma_per_control: ``[n_controls]`` log of per-control sigmas.
        gamma1_per_control: ``[n_controls]`` per-control gamma1 estimates.
        log_area_per_control: ``[n_controls]`` log of per-control areas.
        A_artefact_est: ``mean(area_per_control)`` in linear units.
        A_total_per_trace: ``[n_trace]`` per-trace total area in the window
            (trapezoid over baseline-subtracted signal). Used for the analyte
            residual ``A_analyte[trace] = A_total[trace] - A_artefact_est``.
        mu_artefact: ``mean(mu_per_control)``.
        mu_analyte_ref: Apex location in the non-control trace with the
            largest ``A_total``.
        delta_signed: ``mu_artefact - mu_analyte_ref`` (positive when
            artefact is later than analyte, i.e. on the right).
    """

    mu_per_control: NDArray[np.float64]
    log_sigma_per_control: NDArray[np.float64]
    gamma1_per_control: NDArray[np.float64]
    log_area_per_control: NDArray[np.float64]
    A_artefact_est: float
    A_total_per_trace: NDArray[np.float64]
    mu_artefact: float
    mu_analyte_ref: float
    delta_signed: float


def _trapezoid_per_trace_in_window(
    time: NDArray[np.float64],
    signal_baseline_subtracted: NDArray[np.float64],
    window_low: float,
    window_high: float,
) -> NDArray[np.float64]:
    n_trace = time.shape[0]
    out = np.zeros(n_trace, dtype=np.float64)
    for tr in range(n_trace):
        mask = (
            (time[tr] >= window_low)
            & (time[tr] <= window_high)
            & np.isfinite(signal_baseline_subtracted[tr])
        )
        if mask.sum() >= 2:
            out[tr] = float(np.trapezoid(
                signal_baseline_subtracted[tr][mask], time[tr][mask]
            ))
    return out


def extract_artefact_from_controls(
    time: NDArray[np.float64],
    signal: NDArray[np.float64],
    is_control: NDArray[np.bool_],
    annotation: PeakAnnotation,
    dt: float,
    config: PriorConfig,
) -> ArtefactMeasurements:
    """Extract raw artefact measurements from control traces; check side.

    Args:
        time: ``[n_trace, n_time]`` NaN-padded time array.
        signal: ``[n_trace, n_time]`` baseline-subtracted signal.
        is_control: ``[n_trace]`` bool mask.
        annotation: doublet :class:`PeakAnnotation` with ``artefact_side``.
        dt: Sampling interval.
        config: :class:`PriorConfig` controlling thresholds.

    Returns:
        :class:`ArtefactMeasurements`.

    Raises:
        ValueError: if no controls, if peaks are too close to distinguish
            at sampling resolution, or if observed side mismatches
            ``annotation.artefact_side``.
    """
    if annotation.artefact_side is None:
        raise ValueError(
            f"annotation.artefact_side must be set for artefact_doublet "
            f"mode (peak {annotation.molecule_id})."
        )

    control_idx = np.where(is_control)[0]
    if control_idx.size == 0:
        raise ValueError(
            f"Peak {annotation.molecule_id}: no control traces in dataset; "
            f"cannot extract artefact priors. Mark controls in the conditions "
            f"CSV or switch annotation mode."
        )

    control_features = [
        compute_single_window_features(
            time[i], signal[i], annotation.rt_min, annotation.rt_max
        )
        for i in control_idx
    ]
    mu_per_control = np.array([f.mu for f in control_features])
    sigma_per_control = np.clip(
        np.array([f.sigma for f in control_features]), 1e-9, None
    )
    log_sigma_per_control = np.log(sigma_per_control)
    gamma1_per_control = np.array([f.gamma1 for f in control_features])
    area_per_control = np.array([f.area for f in control_features])
    log_area_per_control = np.log(np.clip(np.abs(area_per_control), 1e-9, None))
    mu_artefact = float(np.mean(mu_per_control))
    A_artefact_est = float(np.mean(area_per_control))

    A_total = _trapezoid_per_trace_in_window(
        time, signal, annotation.rt_min, annotation.rt_max
    )

    non_control_mask = ~is_control
    if not non_control_mask.any():
        raise ValueError(
            f"Peak {annotation.molecule_id}: dataset has no non-control traces."
        )
    non_control_idx = np.where(non_control_mask)[0]
    ref_trace_idx = int(non_control_idx[int(np.argmax(A_total[non_control_idx]))])
    mu_analyte_ref, _ = detect_dominant_apex(
        time[ref_trace_idx], signal[ref_trace_idx],
        annotation.rt_min, annotation.rt_max,
    )

    delta_signed = mu_artefact - mu_analyte_ref
    epsilon = config.side_check_epsilon_dt_multiplier * dt
    if abs(delta_signed) < epsilon:
        raise ValueError(
            f"Peak {annotation.molecule_id}: artefact apex from controls "
            f"({mu_artefact:.4f}) and analyte apex from max-total trace "
            f"({mu_analyte_ref:.4f}) differ by {delta_signed:+.4f} min, which "
            f"is too close to distinguish at sampling resolution "
            f"({config.side_check_epsilon_dt_multiplier}*dt = {epsilon:.4f}). "
            f"Peaks unresolved; widen the annotation window or pick different "
            f"control traces."
        )
    observed_side = "right" if delta_signed > 0 else "left"
    if observed_side != annotation.artefact_side:
        raise ValueError(
            f"Peak {annotation.molecule_id}: artefact_side="
            f"'{annotation.artefact_side}' but controls indicate artefact is "
            f"on the {observed_side} side (mu_artefact={mu_artefact:.4f}, "
            f"mu_analyte_ref={mu_analyte_ref:.4f}, delta={delta_signed:+.4f}). "
            f"Fix artefact_side or check control trace identity."
        )

    return ArtefactMeasurements(
        mu_per_control=mu_per_control,
        log_sigma_per_control=log_sigma_per_control,
        gamma1_per_control=gamma1_per_control,
        log_area_per_control=log_area_per_control,
        A_artefact_est=A_artefact_est,
        A_total_per_trace=A_total,
        mu_artefact=mu_artefact,
        mu_analyte_ref=mu_analyte_ref,
        delta_signed=delta_signed,
    )


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


def _log_A_scale_from_noise_propagation(
    areas: NDArray[np.float64],
    noise_per_trace: NDArray[np.float64],
    n_window_points: int,
    dt: float,
    n_trace: int,
    config: PriorConfig,
) -> float:
    """log_A scale: noise propagation, floored by config."""
    median_noise = float(np.median(noise_per_trace))
    sigma_area = median_noise * float(np.sqrt(n_window_points)) * float(dt)
    median_area = float(np.median(np.abs(areas))) if areas.size > 0 else 0.0
    cv = 1.0 if median_area <= 0.0 else sigma_area / median_area
    propagated = float(np.log1p(cv))
    if n_trace == 1:
        return max(propagated, config.log_A_scale_n1_min)
    return max(propagated, config.log_A_scale_n1_min / float(np.sqrt(n_trace)))


def aggregate_single_peak_priors(
    per_trace_features: list[WindowFeatures],
    window_low: float,
    window_high: float,
    dt: float,
    noise_per_trace: NDArray[np.float64],
    n_window_points: int,
    config: PriorConfig,
) -> SkewNormalPriors:
    """Aggregate per-trace single-peak features into a :class:`SkewNormalPriors`.

    All scale fallbacks for the n=1 case come from ``config``.
    """
    n = len(per_trace_features)
    if n == 0:
        raise ValueError("per_trace_features must be non-empty.")

    mus = np.asarray([f.mu for f in per_trace_features])
    sigmas = np.asarray([f.sigma for f in per_trace_features])
    gamma1s = np.asarray([f.gamma1 for f in per_trace_features])
    areas = np.asarray([f.area for f in per_trace_features])
    log_sigmas = np.log(np.clip(sigmas, 1e-9, None))
    log_areas = np.log(np.clip(np.abs(areas), 1e-9, None))

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

    log_sigma_low, log_sigma_high = _log_sigma_bounds(window_low, window_high, dt, config)
    log_A_scale = _log_A_scale_from_noise_propagation(
        areas, noise_per_trace, n_window_points, dt, n, config,
    )

    return SkewNormalPriors(
        n_components=1,
        mu_left_loc=mu_loc, mu_left_scale=mu_scale,
        mu_left_low=window_low, mu_left_high=window_high,
        log_sigma_left_loc=log_sigma_loc, log_sigma_left_scale=log_sigma_scale,
        log_sigma_left_low=log_sigma_low, log_sigma_left_high=log_sigma_high,
        gamma1_left_loc=gamma1_loc, gamma1_left_scale=gamma1_scale,
        log_A_left_loc_per_trace=log_areas, log_A_left_scale=log_A_scale,
        Delta_loc=None, Delta_scale=None, Delta_low=None, Delta_high=None,
        log_sigma_right_loc=None, log_sigma_right_scale=None,
        log_sigma_right_low=None, log_sigma_right_high=None,
        gamma1_right_loc=None, gamma1_right_scale=None,
        log_A_right_loc_per_trace=None, log_A_right_scale=None,
    )


def aggregate_doublet_priors(
    analyte_priors: SkewNormalPriors,
    artefact: ArtefactMeasurements,
    window_low: float,
    window_high: float,
    dt: float,
    n_window_points: int,
    noise_per_trace: NDArray[np.float64],
    baseline_se_per_trace: NDArray[np.float64],
    config: PriorConfig,
) -> SkewNormalPriors:
    """Assemble doublet priors from analyte single-peak priors + artefact measurements.

    For n_controls=1, shape and position scales borrow from analyte_priors.
    For n_controls>=2, scales are ``max(empirical_std, borrowed/sqrt(n))``.

    Args:
        analyte_priors: Output of :func:`aggregate_single_peak_priors` on
            non-control traces (must have ``n_components == 1``).
        artefact: :class:`ArtefactMeasurements` from
            :func:`extract_artefact_from_controls`.
        window_low: Annotation lower bound.
        window_high: Annotation upper bound.
        dt: Sampling interval.
        n_window_points: Median in-window sample count.
        noise_per_trace: ``[n_trace]`` per-trace noise std (full dataset).
        baseline_se_per_trace: ``[n_trace]`` per-trace OLS baseline standard
            error (signal units). Used to widen ``log_A_right_scale``.
        config: :class:`PriorConfig`.

    Returns:
        :class:`SkewNormalPriors` with ``n_components=2``.

    Raises:
        ValueError: If ``analyte_priors.n_components != 1``.
    """
    if analyte_priors.n_components != 1:
        raise ValueError(
            "analyte_priors must be a single-peak prior (n_components=1)."
        )

    n_c = artefact.mu_per_control.size

    # --- Δ ---
    delta_loc = abs(artefact.delta_signed)
    delta_scale_n1 = config.delta_scale_dt_multiplier_n1 * dt
    if n_c == 1:
        delta_scale = delta_scale_n1
    else:
        per_control_seps = np.abs(artefact.mu_per_control - artefact.mu_analyte_ref)
        empirical = float(np.std(per_control_seps, ddof=0))
        delta_scale = max(empirical, delta_scale_n1 / float(np.sqrt(n_c)))
    delta_low = config.delta_low_dt_multiplier * dt
    delta_high = (window_high - window_low) / config.delta_high_window_fraction

    # --- Right component shape: borrow from analyte for n=1 ---
    log_sigma_right_loc = float(np.mean(artefact.log_sigma_per_control))
    if n_c == 1:
        log_sigma_right_scale = analyte_priors.log_sigma_left_scale
    else:
        empirical_ls = float(np.std(artefact.log_sigma_per_control, ddof=0))
        log_sigma_right_scale = max(
            empirical_ls, analyte_priors.log_sigma_left_scale / float(np.sqrt(n_c)),
        )

    gamma1_right_loc = float(np.mean(artefact.gamma1_per_control))
    if n_c == 1:
        gamma1_right_scale = analyte_priors.gamma1_left_scale
    else:
        empirical_g1 = float(np.std(artefact.gamma1_per_control, ddof=0))
        gamma1_right_scale = max(
            empirical_g1, analyte_priors.gamma1_left_scale / float(np.sqrt(n_c)),
        )
    _, gamma1_bound_high = _gamma1_bounds(config)
    gamma1_right_scale = min(gamma1_right_scale, gamma1_bound_high)

    log_sigma_low, log_sigma_high = _log_sigma_bounds(window_low, window_high, dt, config)

    # --- A_artefact scale: noise + baseline propagation, floored ---
    median_noise = float(np.median(noise_per_trace))
    sigma_A_noise = median_noise * float(np.sqrt(n_window_points)) * dt
    median_baseline_se = float(np.median(baseline_se_per_trace))
    sigma_A_baseline = median_baseline_se * (window_high - window_low)
    sigma_A_total = float(np.sqrt(sigma_A_noise**2 + sigma_A_baseline**2))
    A_artefact_est = max(artefact.A_artefact_est, 1e-9)
    propagated = float(np.log1p(sigma_A_total / A_artefact_est))
    if n_c >= 2:
        empirical_la = float(np.std(artefact.log_area_per_control, ddof=0))
        log_A_right_scale = max(empirical_la, propagated, config.log_A_artefact_min_scale)
    else:
        log_A_right_scale = max(propagated, config.log_A_artefact_min_scale)

    # --- log_A_left from A_total residual; A_floor from noise propagation ---
    A_floor = sigma_A_noise
    A_analyte = np.maximum(
        artefact.A_total_per_trace - artefact.A_artefact_est, A_floor,
    )
    log_A_left_loc_per_trace = np.log(A_analyte)
    log_A_left_scale = _log_A_scale_from_noise_propagation(
        A_analyte, noise_per_trace, n_window_points, dt,
        A_analyte.size, config,
    )

    # --- log_A_right per trace (constant) ---
    n_total = artefact.A_total_per_trace.size
    log_A_right_loc_per_trace = np.full(
        n_total, float(np.log(A_artefact_est)), dtype=np.float64,
    )

    return SkewNormalPriors(
        n_components=2,
        mu_left_loc=analyte_priors.mu_left_loc,
        mu_left_scale=analyte_priors.mu_left_scale,
        mu_left_low=window_low, mu_left_high=window_high,
        log_sigma_left_loc=analyte_priors.log_sigma_left_loc,
        log_sigma_left_scale=analyte_priors.log_sigma_left_scale,
        log_sigma_left_low=log_sigma_low, log_sigma_left_high=log_sigma_high,
        gamma1_left_loc=analyte_priors.gamma1_left_loc,
        gamma1_left_scale=analyte_priors.gamma1_left_scale,
        log_A_left_loc_per_trace=log_A_left_loc_per_trace,
        log_A_left_scale=log_A_left_scale,
        Delta_loc=delta_loc, Delta_scale=delta_scale,
        Delta_low=delta_low, Delta_high=delta_high,
        log_sigma_right_loc=log_sigma_right_loc,
        log_sigma_right_scale=log_sigma_right_scale,
        log_sigma_right_low=log_sigma_low, log_sigma_right_high=log_sigma_high,
        gamma1_right_loc=gamma1_right_loc,
        gamma1_right_scale=gamma1_right_scale,
        log_A_right_loc_per_trace=log_A_right_loc_per_trace,
        log_A_right_scale=log_A_right_scale,
    )


def _baseline_subtracted(dataset: PreparedDataset) -> NDArray[np.float64]:
    intercept = dataset.baseline_intercept[:, None]
    slope = dataset.baseline_slope[:, None]
    return dataset.signal - (intercept + slope * dataset.time)


def _baseline_se_per_trace(dataset: PreparedDataset) -> NDArray[np.float64]:
    """OLS baseline residual std per trace, evaluated on the baseline regions
    the user annotated. Quantifies how uncertain the baseline subtraction is.
    """
    n_trace = dataset.n_trace
    out = np.zeros(n_trace, dtype=np.float64)
    baseline_sub = _baseline_subtracted(dataset)
    for tr in range(n_trace):
        residuals: list[float] = []
        for ba in dataset.baseline_annotations:
            mask = (
                (dataset.time[tr] >= ba.rt_min)
                & (dataset.time[tr] <= ba.rt_max)
                & np.isfinite(baseline_sub[tr])
            )
            residuals.extend(baseline_sub[tr][mask].tolist())
        if residuals:
            out[tr] = float(np.std(np.asarray(residuals, dtype=np.float64), ddof=0))
        else:
            out[tr] = float(dataset.noise_per_trace[tr])
    return out


def _count_window_points(time: NDArray[np.float64], low: float, high: float) -> int:
    masks: NDArray[np.bool_] = (
        (time >= low) & (time <= high) & np.isfinite(time)
    )
    counts: NDArray[np.intp] = masks.sum(axis=1)
    return int(np.median(counts))


def build_priors(
    dataset: PreparedDataset,
    config: PriorConfig | None = None,
) -> list[SkewNormalPriors]:
    """Build per-annotation :class:`SkewNormalPriors` from a prepared dataset.

    Args:
        dataset: Output of :func:`prepare_dataset`. Must have ``is_control``
            populated; if any annotation is ``artefact_doublet``, at least
            one trace must be a control.
        config: Optional :class:`PriorConfig`. Defaults to ``PriorConfig()``.

    Returns:
        One :class:`SkewNormalPriors` per ``dataset.peak_annotations``.

    Raises:
        ValueError: For ``artefact_doublet`` with no controls in the dataset.
        NotImplementedError: For ``free_doublet`` annotations.
    """
    cfg = config if config is not None else PriorConfig()
    baseline_sub = _baseline_subtracted(dataset)
    baseline_se = _baseline_se_per_trace(dataset)
    non_control_idx = np.where(~dataset.is_control)[0]
    if non_control_idx.size == 0:
        raise ValueError("Dataset contains only control traces; cannot build priors.")

    out: list[SkewNormalPriors] = []
    for ann in dataset.peak_annotations:
        n_pts = _count_window_points(dataset.time, ann.rt_min, ann.rt_max)
        if ann.mode == "single":
            feats = [
                compute_single_window_features(
                    dataset.time[tr], baseline_sub[tr], ann.rt_min, ann.rt_max
                )
                for tr in non_control_idx
            ]
            out.append(aggregate_single_peak_priors(
                per_trace_features=feats,
                window_low=ann.rt_min, window_high=ann.rt_max,
                dt=dataset.dt_global,
                noise_per_trace=dataset.noise_per_trace[non_control_idx],
                n_window_points=n_pts, config=cfg,
            ))
        elif ann.mode == "artefact_doublet":
            analyte_feats = [
                compute_single_window_features(
                    dataset.time[tr], baseline_sub[tr], ann.rt_min, ann.rt_max
                )
                for tr in non_control_idx
            ]
            analyte_priors = aggregate_single_peak_priors(
                per_trace_features=analyte_feats,
                window_low=ann.rt_min, window_high=ann.rt_max,
                dt=dataset.dt_global,
                noise_per_trace=dataset.noise_per_trace[non_control_idx],
                n_window_points=n_pts, config=cfg,
            )
            artefact = extract_artefact_from_controls(
                time=dataset.time, signal=baseline_sub,
                is_control=dataset.is_control, annotation=ann,
                dt=dataset.dt_global, config=cfg,
            )
            out.append(aggregate_doublet_priors(
                analyte_priors=analyte_priors, artefact=artefact,
                window_low=ann.rt_min, window_high=ann.rt_max,
                dt=dataset.dt_global, n_window_points=n_pts,
                noise_per_trace=dataset.noise_per_trace,
                baseline_se_per_trace=baseline_se, config=cfg,
            ))
        elif ann.mode == "free_doublet":
            raise NotImplementedError(
                f"Peak {ann.molecule_id}: mode='free_doublet' is not yet "
                f"supported. Use 'artefact_doublet' with controls or wait "
                f"for the free_doublet implementation."
            )
        else:
            raise ValueError(f"Unknown peak mode '{ann.mode}'.")
    return out
