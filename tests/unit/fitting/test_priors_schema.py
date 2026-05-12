"""Tests for PriorConfig and SkewNormalPriors dataclasses."""

from __future__ import annotations

import numpy as np
import pytest

from chromhandler.fitting.priors import PriorConfig, SkewNormalPriors


def test_prior_config_defaults() -> None:
    c = PriorConfig()
    assert c.gamma1_bound_fraction == 0.99
    assert c.sigma_low_n_points_per_fwhm == 8
    assert c.sigma_high_window_fraction == 6.0
    assert c.delta_low_dt_multiplier == 3.0
    assert c.delta_high_window_fraction == 2.0


def test_prior_config_overridable() -> None:
    c = PriorConfig(gamma1_bound_fraction=0.95, log_sigma_scale_n1=0.10)
    assert c.gamma1_bound_fraction == 0.95
    assert c.log_sigma_scale_n1 == 0.10
    assert c.delta_low_dt_multiplier == 3.0


def _single() -> SkewNormalPriors:
    return SkewNormalPriors(
        n_components=1,
        mu_left_loc=2.7, mu_left_scale=0.005, mu_left_low=2.55, mu_left_high=2.85,
        log_sigma_left_loc=np.log(0.03), log_sigma_left_scale=0.1,
        log_sigma_left_low=np.log(0.005), log_sigma_left_high=np.log(0.05),
        gamma1_left_loc=0.2, gamma1_left_scale=0.05,
        log_A_left_loc_per_trace=np.array([np.log(100.0), np.log(80.0)]),
        log_A_left_scale=0.1,
        Delta_loc=None, Delta_scale=None, Delta_low=None, Delta_high=None,
        log_sigma_right_loc=None, log_sigma_right_scale=None,
        log_sigma_right_low=None, log_sigma_right_high=None,
        gamma1_right_loc=None, gamma1_right_scale=None,
        log_A_right_loc_per_trace=None, log_A_right_scale=None,
    )


def test_single_priors_constructs() -> None:
    p = _single()
    assert p.n_components == 1 and p.Delta_loc is None


def test_single_priors_rejects_doublet_fields() -> None:
    with pytest.raises(ValueError, match=r"right.*None"):
        SkewNormalPriors(
            n_components=1,
            mu_left_loc=2.7, mu_left_scale=0.005, mu_left_low=2.55, mu_left_high=2.85,
            log_sigma_left_loc=np.log(0.03), log_sigma_left_scale=0.1,
            log_sigma_left_low=np.log(0.005), log_sigma_left_high=np.log(0.05),
            gamma1_left_loc=0.2, gamma1_left_scale=0.05,
            log_A_left_loc_per_trace=np.array([np.log(100.0)]), log_A_left_scale=0.1,
            Delta_loc=0.05, Delta_scale=0.005, Delta_low=0.003, Delta_high=0.15,
            log_sigma_right_loc=None, log_sigma_right_scale=None,
            log_sigma_right_low=None, log_sigma_right_high=None,
            gamma1_right_loc=None, gamma1_right_scale=None,
            log_A_right_loc_per_trace=None, log_A_right_scale=None,
        )


def test_doublet_priors_requires_all_right_fields() -> None:
    with pytest.raises(ValueError, match=r"doublet.*required"):
        SkewNormalPriors(
            n_components=2,
            mu_left_loc=2.7, mu_left_scale=0.005, mu_left_low=2.55, mu_left_high=2.85,
            log_sigma_left_loc=np.log(0.03), log_sigma_left_scale=0.1,
            log_sigma_left_low=np.log(0.005), log_sigma_left_high=np.log(0.05),
            gamma1_left_loc=0.2, gamma1_left_scale=0.05,
            log_A_left_loc_per_trace=np.array([np.log(100.0)]), log_A_left_scale=0.1,
            Delta_loc=None, Delta_scale=None, Delta_low=None, Delta_high=None,
            log_sigma_right_loc=None, log_sigma_right_scale=None,
            log_sigma_right_low=None, log_sigma_right_high=None,
            gamma1_right_loc=None, gamma1_right_scale=None,
            log_A_right_loc_per_trace=None, log_A_right_scale=None,
        )
