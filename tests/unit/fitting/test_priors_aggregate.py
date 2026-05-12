"""Tests for single-peak prior aggregation."""

from __future__ import annotations

import numpy as np

from chromhandler.fitting.priors import (
    PriorConfig,
    WindowFeatures,
    aggregate_single_peak_priors,
)
from chromhandler.fitting.skew_normal import GAMMA1_MAX


def _features(
    mus: list[float],
    sigmas: list[float],
    gamma1s: list[float],
    areas: list[float],
) -> list[WindowFeatures]:
    return [
        WindowFeatures(mu=mu, sigma=sigma, gamma1=gamma1, area=area)
        for mu, sigma, gamma1, area in zip(mus, sigmas, gamma1s, areas, strict=True)
    ]


def test_recovers_population_stats() -> None:
    rng = np.random.default_rng(0)
    n = 50
    mus = rng.normal(2.70, 0.002, size=n).tolist()
    sigmas = np.exp(rng.normal(np.log(0.03), 0.05, size=n)).tolist()
    gamma1s = rng.normal(0.2, 0.05, size=n).tolist()
    areas = np.exp(rng.normal(np.log(100.0), 0.1, size=n)).tolist()
    p = aggregate_single_peak_priors(
        per_trace_features=_features(mus, sigmas, gamma1s, areas),
        window_low=2.55, window_high=2.85, dt=0.001,
        noise_per_trace=np.full(n, 1.0), n_window_points=300,
        config=PriorConfig(),
    )
    assert abs(p.mu_left_loc - 2.70) < 0.001
    assert 0.0005 < p.mu_left_scale < 0.005
    assert p.mu_left_low == 2.55 and p.mu_left_high == 2.85
    assert abs(p.log_sigma_left_loc - np.log(0.03)) < 0.02
    assert abs(p.gamma1_left_loc - 0.2) < 0.02
    assert p.log_A_left_loc_per_trace.shape == (n,)
    assert p.n_components == 1


def test_single_trace_uses_config_fallbacks() -> None:
    cfg = PriorConfig(log_sigma_scale_n1=0.15, gamma1_scale_n1=0.20,
                      mu_scale_dt_floor_multiplier=1.0, log_A_scale_n1_min=0.10)
    p = aggregate_single_peak_priors(
        per_trace_features=_features([2.70], [0.03], [0.2], [100.0]),
        window_low=2.55, window_high=2.85, dt=0.001,
        noise_per_trace=np.array([1.0]), n_window_points=300, config=cfg,
    )
    assert p.mu_left_scale == 0.001
    assert abs(p.log_sigma_left_scale - 0.15) < 1e-9
    assert abs(p.gamma1_left_scale - 0.20) < 1e-9
    assert p.log_A_left_scale >= 0.10


def test_geometric_bounds_from_config() -> None:
    cfg = PriorConfig(sigma_low_n_points_per_fwhm=8, sigma_high_window_fraction=6.0)
    p = aggregate_single_peak_priors(
        per_trace_features=_features([2.70] * 5, [0.03] * 5, [0.0] * 5, [100.0] * 5),
        window_low=2.55, window_high=2.85, dt=0.001,
        noise_per_trace=np.full(5, 1.0), n_window_points=300, config=cfg,
    )
    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    assert abs(p.log_sigma_left_low - np.log(8 * 0.001 * fwhm_to_sigma)) < 1e-9
    assert abs(p.log_sigma_left_high - np.log((2.85 - 2.55) / 6.0)) < 1e-9


def test_gamma1_scale_capped_by_max() -> None:
    cfg = PriorConfig(gamma1_scale_n1=2.5)  # > GAMMA1_MAX
    p = aggregate_single_peak_priors(
        per_trace_features=_features([2.70], [0.03], [0.0], [100.0]),
        window_low=2.55, window_high=2.85, dt=0.001,
        noise_per_trace=np.array([1.0]), n_window_points=300, config=cfg,
    )
    assert p.gamma1_left_scale <= GAMMA1_MAX + 1e-9
