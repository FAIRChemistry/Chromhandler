"""Tests for summarise_priors."""

from __future__ import annotations

import numpy as np

from chromhandler.fitting.priors import (
    PriorConfig, SkewNormalPriors, summarise_priors,
)


def _single():
    return SkewNormalPriors(
        n_components=1,
        mu_left_loc=2.70, mu_left_scale=0.005,
        mu_left_low=2.55, mu_left_high=2.85,
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


def test_summary_mentions_distributions() -> None:
    out = summarise_priors([_single()], config=PriorConfig())
    assert "TruncatedNormal" in out and "Normal" in out
    for site in ("mu_anchor_left", "log_sigma_left", "gamma1_left", "log_A_left"):
        assert site in out


def test_doublet_summary_uses_truncated_delta() -> None:
    s = _single()
    d = SkewNormalPriors(
        n_components=2,
        mu_left_loc=3.00, mu_left_scale=0.005,
        mu_left_low=2.90, mu_left_high=3.15,
        log_sigma_left_loc=s.log_sigma_left_loc, log_sigma_left_scale=s.log_sigma_left_scale,
        log_sigma_left_low=s.log_sigma_left_low, log_sigma_left_high=s.log_sigma_left_high,
        gamma1_left_loc=s.gamma1_left_loc, gamma1_left_scale=s.gamma1_left_scale,
        log_A_left_loc_per_trace=np.array([np.log(80.0)]), log_A_left_scale=0.1,
        Delta_loc=0.05, Delta_scale=0.002, Delta_low=0.003, Delta_high=0.125,
        log_sigma_right_loc=s.log_sigma_left_loc, log_sigma_right_scale=s.log_sigma_left_scale,
        log_sigma_right_low=s.log_sigma_left_low, log_sigma_right_high=s.log_sigma_left_high,
        gamma1_right_loc=s.gamma1_left_loc, gamma1_right_scale=s.gamma1_left_scale,
        log_A_right_loc_per_trace=np.array([np.log(5.0)]), log_A_right_scale=np.log(1.5),
    )
    out = summarise_priors([s, d], config=PriorConfig())
    delta_row = next(line for line in out.split("\n") if "Delta" in line)
    assert "TruncatedNormal" in delta_row
