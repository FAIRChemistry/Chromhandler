"""Property tests for the pure-math skew-normal layer."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import skewnorm

from chromhandler.fitting import skew_normal as sn

jax.config.update("jax_enable_x64", True)

_B_CONST_NP = math.sqrt(2.0 / math.pi)


def test_gamma1_max_matches_half_normal_limit():
    """GAMMA1_MAX equals the skewness of the half-normal (a -> inf limit)."""
    expected: float = float(skewnorm.stats(a=1e6, moments="s"))  # type: ignore[arg-type]
    assert math.isclose(sn.GAMMA1_MAX, expected, rel_tol=1e-6)


def test_gamma1_max_closed_form():
    """GAMMA1_MAX matches the closed-form expression in spec §2.2."""
    b = math.sqrt(2.0 / math.pi)
    expected = ((4.0 - math.pi) / 2.0) * b**3 / (1.0 - 2.0 / math.pi) ** 1.5
    assert math.isclose(sn.GAMMA1_MAX, expected, rel_tol=1e-12)


def test_cp_to_dp_zero_skew_is_identity_on_mu_sigma():
    """At gamma1 = 0 the SN reduces to N(mu, sigma^2): xi=mu, omega=sigma, alpha=0."""
    xi, omega, alpha = sn.cp_to_dp(jnp.asarray(1.5), jnp.asarray(0.4), jnp.asarray(0.0))
    np.testing.assert_allclose(float(xi), 1.5, atol=1e-12)
    np.testing.assert_allclose(float(omega), 0.4, atol=1e-12)
    np.testing.assert_allclose(float(alpha), 0.0, atol=1e-12)


def test_cp_to_dp_against_scipy_grid():
    """For (alpha, omega, xi) sampled randomly, the forward map cp_to_dp recovers them."""
    rng = np.random.default_rng(0)
    alpha_true = rng.uniform(-15.0, 15.0, size=20)
    omega_true = rng.uniform(0.1, 2.0, size=20)
    xi_true = rng.uniform(-3.0, 3.0, size=20)
    # Build CP from DP using the forward Azzalini formulas, then run cp_to_dp.
    delta = alpha_true / np.sqrt(1.0 + alpha_true**2)
    mu = xi_true + omega_true * _B_CONST_NP * delta
    sigma = omega_true * np.sqrt(1.0 - _B_CONST_NP**2 * delta**2)
    gamma1 = ((4.0 - np.pi) / 2.0) * (_B_CONST_NP * delta) ** 3 / (1.0 - _B_CONST_NP**2 * delta**2) ** 1.5
    xi_pred, omega_pred, alpha_pred = sn.cp_to_dp(
        jnp.asarray(mu), jnp.asarray(sigma), jnp.asarray(gamma1)
    )
    np.testing.assert_allclose(np.asarray(xi_pred), xi_true, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(omega_pred), omega_true, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(alpha_pred), alpha_true, rtol=1e-5, atol=1e-7)
