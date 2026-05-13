"""NumPyro Bayesian model for the skew-normal peak fitter.

Single-mode peaks only at present. Doublet support is a documented
extension — see TODO(doublet) markers throughout this module and the
"Doublet extension hooks" section of the design spec
(``docs/superpowers/specs/2026-05-12-fitter-integration-design.md``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """User-facing configuration for the NumPyro fit.

    Tuned defaults for fast development iteration on chromatographic data.
    Override fields directly when constructing for publication-quality runs.
    """

    # --- HMC / NUTS settings ---
    num_warmup: int = 500
    num_samples: int = 500
    num_chains: int = 4
    target_accept_prob: float = 0.9
    max_tree_depth: int = 10
    seed: int = 0

    # --- Model-layer priors (per-trace, not per-peak) ---
    trace_shift_scale_dt_multiplier: float = 5.0
    """drift_scale = N * dt_global. trace_shift ~ Normal(0, drift_scale)."""

    baseline_intercept_se_floor: float = 1.0
    """Minimum SE for the baseline intercept prior (signal units)."""

    baseline_slope_se_floor: float = 0.01
    """Minimum SE for the baseline slope prior (signal units per minute)."""

    # --- Prior predictive ---
    prior_predictive_n_samples: int = 200
    """Number of prior samples used to compute prior predictive band."""
