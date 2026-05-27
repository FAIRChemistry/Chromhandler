"""ModelConfig defaults and overridability."""

from __future__ import annotations

from chromhandler.fitting.model import ModelConfig


def test_model_config_defaults() -> None:
    c = ModelConfig()
    assert c.num_warmup == 500
    assert c.num_samples == 500
    assert c.num_chains == 4
    assert c.target_accept_prob == 0.9
    assert c.max_tree_depth == 10
    assert c.seed == 0
    assert c.baseline_intercept_se_floor == 1.0
    assert c.baseline_slope_se_floor == 0.01
    assert c.log_noise_scale == 2.0
    assert c.warp_shift_scale_dt_multiplier == 5.0
    assert c.warp_stretch_scale == 0.01
    assert c.prior_predictive_n_samples == 200


def test_model_config_overridable() -> None:
    c = ModelConfig(num_warmup=1000, num_chains=2, seed=42)
    assert c.num_warmup == 1000
    assert c.num_chains == 2
    assert c.seed == 42
    # Untouched fields keep defaults
    assert c.target_accept_prob == 0.9
