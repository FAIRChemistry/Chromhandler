"""NumPyro Bayesian model for the skew-normal peak fitter.

Single-mode peaks only at present. Doublet support is a documented
extension — see TODO(doublet) markers throughout this module and the
"Doublet extension hooks" section of the design spec
(``docs/superpowers/specs/2026-05-12-fitter-integration-design.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from chromhandler.fitting.skew_normal import density_cp

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from chromhandler.fitting.prepared_dataset import PreparedDataset
    from chromhandler.fitting.priors import SkewNormalPriors

# --- Sample-site name constants (TODO(doublet): populate SAMPLED_RIGHT_* below) ---
SAMPLED_LEFT_SHARED: tuple[str, ...] = ("mu_anchor_left", "log_sigma_left", "gamma1_left")
SAMPLED_LEFT_PER_TRACE: tuple[str, ...] = ("log_A_left",)
SAMPLED_TRACE_NUISANCE: tuple[str, ...] = (
    "trace_shift", "baseline_intercept", "baseline_slope",
)
SAMPLED_RIGHT_SHARED: tuple[str, ...] = ()        # TODO(doublet)
SAMPLED_RIGHT_PER_TRACE: tuple[str, ...] = ()     # TODO(doublet)


def _validate_single_mode_only(priors_list: list[SkewNormalPriors]) -> None:  # type: ignore[reportUnusedFunction]
    """Raise if any peak in priors_list has n_components > 1.

    Hoisted out of model() so the JIT-compiled hot path is clean.
    """
    doublet = [i for i, p in enumerate(priors_list) if p.n_components == 2]
    if doublet:
        raise NotImplementedError(
            f"model.py supports n_components=1 (single) peaks only. "
            f"Doublet peaks at indices {doublet}. Doublet support is a "
            f"documented future extension — see model.py module docstring "
            f"and `# TODO(doublet)` markers."
        )


def _compute_baseline_se(  # type: ignore[reportUnusedFunction]
    dataset: PreparedDataset,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-trace OLS standard errors for the baseline intercept and slope.

    Computed from the residuals of the baseline OLS fit on each trace's
    annotated baseline regions. Returns ``(intercept_se, slope_se)``,
    both shape ``[n_trace]``.

    Used by ``model()`` to set the Normal priors on baseline parameters.
    """
    n_trace = dataset.n_trace
    intercept_se = np.zeros(n_trace, dtype=np.float64)
    slope_se = np.zeros(n_trace, dtype=np.float64)

    for tr in range(n_trace):
        t = dataset.time[tr]
        s = dataset.signal[tr]
        baseline_mask = np.zeros_like(t, dtype=bool)
        for ba in dataset.baseline_annotations:
            baseline_mask |= ((t >= ba.rt_min) & (t <= ba.rt_max) & np.isfinite(s))
        if baseline_mask.sum() < 3:
            # Fall back to noise std as a wide-but-finite SE.
            intercept_se[tr] = float(dataset.noise_per_trace[tr])
            slope_se[tr] = float(dataset.noise_per_trace[tr])
            continue
        t_b = t[baseline_mask]
        s_b = s[baseline_mask]
        # OLS via lstsq with design matrix [1, t]
        X = np.column_stack([np.ones_like(t_b), t_b])
        beta, *_ = np.linalg.lstsq(X, s_b, rcond=None)
        residuals = s_b - X @ beta
        # Standard OLS covariance
        sigma2 = float(np.sum(residuals**2) / max(t_b.size - 2, 1))  # type: ignore[arg-type]
        try:
            cov = sigma2 * np.linalg.inv(X.T @ X)
            intercept_se[tr] = float(np.sqrt(max(cov[0, 0], 0.0)))
            slope_se[tr] = float(np.sqrt(max(cov[1, 1], 0.0)))
        except np.linalg.LinAlgError:
            intercept_se[tr] = float(dataset.noise_per_trace[tr])
            slope_se[tr] = float(dataset.noise_per_trace[tr])
    return intercept_se, slope_se


def _baseline_contribution(  # type: ignore[reportUnusedFunction]
    time: NDArray[np.float64],
    intercept: NDArray[np.float64],
    slope: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-trace baseline = intercept + slope * t. Shape [n_trace, n_time]."""
    return intercept[:, None] + slope[:, None] * time


def _left_component_contribution(  # type: ignore[reportUnusedFunction]
    time: NDArray[np.float64],
    mu_anchor: NDArray[np.float64],
    trace_shift: NDArray[np.float64],
    log_sigma: NDArray[np.float64],
    gamma1: NDArray[np.float64],
    log_A: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sum of left-component skew-normal densities per (trace, time).

    Args:
        time: [n_trace, n_time]
        mu_anchor: [n_peak]
        trace_shift: [n_trace]
        log_sigma: [n_peak]
        gamma1: [n_peak]
        log_A: [n_trace, n_peak]

    Returns:
        Predicted signal [n_trace, n_time].
    """
    n_trace, n_time = time.shape
    n_peak = mu_anchor.shape[0]
    sigma = np.exp(log_sigma)
    # mu[trace, peak] = mu_anchor[peak] + trace_shift[trace]
    mu = mu_anchor[None, :] + trace_shift[:, None]    # [n_trace, n_peak]
    A = np.exp(log_A)                                  # [n_trace, n_peak]

    out = np.zeros((n_trace, n_time), dtype=np.float64)
    for peak in range(n_peak):
        # density_cp accepts vectorised inputs; here we evaluate per-peak
        # over all (trace, time) at once.
        density = np.asarray(density_cp(
            time,                                      # type: ignore[arg-type]  # [n_trace, n_time]
            mu[:, peak:peak + 1],                      # type: ignore[arg-type]  # [n_trace, 1]
            sigma[peak],
            gamma1[peak],
        ))
        out = out + A[:, peak:peak + 1] * density
    return out


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
