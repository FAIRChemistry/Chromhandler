"""Tests for single-window FWHM-based feature extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from scipy.stats import skewnorm

if TYPE_CHECKING:
    from numpy.typing import NDArray

from chromhandler.fitting.priors import WindowFeatures, compute_single_window_features
from chromhandler.fitting.skew_normal import cp_to_dp


def _synth(
    mu: float,
    sigma: float,
    gamma1: float,
    area: float,
    dt: float = 0.001,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    t: NDArray[np.float64] = np.arange(mu - 1.0, mu + 1.0, dt, dtype=np.float64)
    xi, omega, alpha = (
        float(x) for x in cp_to_dp(jnp.asarray(mu), jnp.asarray(sigma), jnp.asarray(gamma1))
    )
    s: NDArray[np.float64] = np.asarray(
        area * skewnorm.pdf(t, alpha, loc=xi, scale=omega), dtype=np.float64
    )
    return t, s


def test_features_dataclass_fields() -> None:
    f = WindowFeatures(mu=2.7, sigma=0.03, gamma1=0.2, area=5.0)
    assert (f.mu, f.sigma, f.gamma1, f.area) == (2.7, 0.03, 0.2, 5.0)


def test_recovers_symmetric_peak() -> None:
    t, s = _synth(2.7, 0.03, 0.0, 5.0)
    f = compute_single_window_features(t, s, 2.55, 2.85)
    assert abs(f.mu - 2.7) < 0.005
    assert abs(f.sigma - 0.03) / 0.03 < 0.05
    assert abs(f.gamma1) < 0.05
    assert abs(f.area - 5.0) / 5.0 < 0.02


def test_recovers_positively_skewed_peak() -> None:
    t, s = _synth(2.7, 0.03, 0.5, 5.0)
    f = compute_single_window_features(t, s, 2.55, 2.85)
    assert abs(f.gamma1 - 0.5) < 0.10


def test_low_snr_average_is_unbiased() -> None:
    rng = np.random.default_rng(0)
    estimates: list[float] = []
    for _ in range(100):
        t, s = _synth(2.7, 0.03, 0.3, 5.0)
        noise: NDArray[np.float64] = np.asarray(
            rng.normal(0.0, np.max(s) / 5.0, size=s.shape), dtype=np.float64
        )
        f = compute_single_window_features(t, s + noise, 2.55, 2.85)
        estimates.append(f.gamma1)
    assert abs(float(np.mean(np.asarray(estimates))) - 0.3) < 0.05
