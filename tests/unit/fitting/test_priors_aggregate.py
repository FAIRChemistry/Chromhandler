"""Tests for single-peak prior aggregation."""

from __future__ import annotations

import numpy as np

from chromhandler.fitting.priors import (
    ArtefactMeasurements,
    PriorConfig,
    SkewNormalPriors,
    WindowFeatures,
    aggregate_doublet_priors,
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


def _analyte_single() -> SkewNormalPriors:
    return aggregate_single_peak_priors(
        per_trace_features=[
            WindowFeatures(mu=2.85, sigma=0.025, gamma1=0.1, area=100.0),
            WindowFeatures(mu=2.851, sigma=0.025, gamma1=0.1, area=60.0),
            WindowFeatures(mu=2.849, sigma=0.025, gamma1=0.1, area=20.0),
        ],
        window_low=2.78, window_high=3.05, dt=0.001,
        noise_per_trace=np.full(3, 0.1), n_window_points=270,
        config=PriorConfig(),
    )


def _measurements_single_control() -> ArtefactMeasurements:
    return ArtefactMeasurements(
        mu_per_control=np.array([2.95]),
        log_sigma_per_control=np.array([np.log(0.025)]),
        gamma1_per_control=np.array([0.05]),
        log_area_per_control=np.array([np.log(5.0)]),
        A_artefact_est=5.0,
        A_total_per_trace=np.array([105.0, 65.0, 25.0, 5.0, 5.0]),
        mu_artefact=2.95, mu_analyte_ref=2.85, delta_signed=0.10,
    )


def test_single_control_borrows_analyte_scales() -> None:
    analyte = _analyte_single()
    artefact = _measurements_single_control()
    p = aggregate_doublet_priors(
        analyte_priors=analyte, artefact=artefact,
        window_low=2.78, window_high=3.05, dt=0.001,
        n_window_points=270, noise_per_trace=np.full(5, 0.1),
        baseline_se_per_trace=np.full(5, 0.05),
        config=PriorConfig(),
    )
    assert p.log_sigma_right_scale == analyte.log_sigma_left_scale
    assert p.gamma1_right_scale == analyte.gamma1_left_scale
    assert p.Delta_scale is not None
    assert abs(p.Delta_scale - 0.0015) < 1e-9
    assert p.Delta_low == 3.0 * 0.001
    assert p.Delta_high == (3.05 - 2.78) / 2.0


def test_doublet_assembly_correctness() -> None:
    analyte = _analyte_single()
    artefact = _measurements_single_control()
    p = aggregate_doublet_priors(
        analyte_priors=analyte, artefact=artefact,
        window_low=2.78, window_high=3.05, dt=0.001,
        n_window_points=270, noise_per_trace=np.full(5, 0.1),
        baseline_se_per_trace=np.full(5, 0.05),
        config=PriorConfig(),
    )
    assert p.n_components == 2
    assert p.mu_left_loc == analyte.mu_left_loc
    assert p.log_sigma_left_loc == analyte.log_sigma_left_loc
    assert p.log_sigma_right_loc == np.log(0.025)
    assert p.gamma1_right_loc == 0.05
    assert p.Delta_loc == 0.10
    assert p.log_A_left_loc_per_trace is not None
    np.testing.assert_allclose(
        p.log_A_left_loc_per_trace,
        np.log(np.maximum(np.array([105.0, 65.0, 25.0, 5.0, 5.0]) - 5.0,
                          0.1 * np.sqrt(270) * 0.001)),
        atol=1e-6,
    )
    assert p.log_A_right_loc_per_trace is not None
    assert all(p.log_A_right_loc_per_trace == np.log(5.0))


def test_multi_control_uses_empirical_scale() -> None:
    analyte = _analyte_single()
    artefact = ArtefactMeasurements(
        mu_per_control=np.array([2.948, 2.952]),
        log_sigma_per_control=np.array([np.log(0.024), np.log(0.026)]),
        gamma1_per_control=np.array([0.04, 0.06]),
        log_area_per_control=np.array([np.log(4.9), np.log(5.1)]),
        A_artefact_est=5.0,
        A_total_per_trace=np.array([105.0, 65.0, 25.0, 5.0, 5.0]),
        mu_artefact=2.95, mu_analyte_ref=2.85, delta_signed=0.10,
    )
    p = aggregate_doublet_priors(
        analyte_priors=analyte, artefact=artefact,
        window_low=2.78, window_high=3.05, dt=0.001,
        n_window_points=270, noise_per_trace=np.full(5, 0.1),
        baseline_se_per_trace=np.full(5, 0.05),
        config=PriorConfig(),
    )
    assert p.Delta_scale is not None
    assert p.Delta_scale >= 1e-3
