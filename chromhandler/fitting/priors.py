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

import jax.numpy as jnp
import numpy as np
from scipy.signal import savgol_filter

from chromhandler.fitting.skew_normal import sn_asymmetry_to_gamma1

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
    gamma1 = float(sn_asymmetry_to_gamma1(jnp.asarray([mean_ratio]))[0])
    area = float(np.trapezoid(s, t))
    return WindowFeatures(mu=mu, sigma=sigma, gamma1=gamma1, area=area)
