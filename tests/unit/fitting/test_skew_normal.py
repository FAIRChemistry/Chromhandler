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


def test_dp_to_cp_zero_alpha_is_identity_on_xi_omega():
    """At alpha=0 the SN is N(xi, omega^2): mu=xi, sigma=omega, gamma1=0."""
    mu, sigma, gamma1 = sn.dp_to_cp(jnp.asarray(2.0), jnp.asarray(0.5), jnp.asarray(0.0))
    np.testing.assert_allclose(float(mu), 2.0, atol=1e-12)
    np.testing.assert_allclose(float(sigma), 0.5, atol=1e-12)
    np.testing.assert_allclose(float(gamma1), 0.0, atol=1e-12)


def test_dp_to_cp_matches_scipy_moments():
    """dp_to_cp matches scipy.stats.skewnorm.stats for a grid of alpha."""
    alphas = np.linspace(-12.0, 12.0, 25)
    omegas = np.full_like(alphas, 0.7)
    xis = np.full_like(alphas, 1.3)
    mu_pred, sigma_pred, gamma1_pred = sn.dp_to_cp(
        jnp.asarray(xis), jnp.asarray(omegas), jnp.asarray(alphas)
    )
    mean_sp, var_sp, skew_sp = skewnorm.stats(a=alphas, loc=xis, scale=omegas, moments="mvs")
    np.testing.assert_allclose(np.asarray(mu_pred), mean_sp, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(np.asarray(sigma_pred), np.sqrt(var_sp), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(np.asarray(gamma1_pred), skew_sp, rtol=1e-6, atol=1e-8)


def test_cp_dp_round_trip():
    """cp_to_dp ∘ dp_to_cp = identity on a grid that stays inside the SN family."""
    rng = np.random.default_rng(1)
    mu = rng.uniform(-2.0, 2.0, size=50)
    sigma = rng.uniform(0.2, 1.5, size=50)
    gamma1 = rng.uniform(-0.95 * sn.GAMMA1_MAX, 0.95 * sn.GAMMA1_MAX, size=50)
    xi, omega, alpha = sn.cp_to_dp(jnp.asarray(mu), jnp.asarray(sigma), jnp.asarray(gamma1))
    mu_back, sigma_back, gamma1_back = sn.dp_to_cp(xi, omega, alpha)
    np.testing.assert_allclose(np.asarray(mu_back), mu, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(np.asarray(sigma_back), sigma, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(np.asarray(gamma1_back), gamma1, rtol=1e-6, atol=1e-8)


def test_dp_cp_round_trip():
    """dp_to_cp o cp_to_dp = identity on a grid of (xi, omega, alpha)."""
    rng = np.random.default_rng(2)
    xi = rng.uniform(-2.0, 2.0, size=50)
    omega = rng.uniform(0.2, 1.5, size=50)
    alpha = rng.uniform(-15.0, 15.0, size=50)
    mu, sigma, gamma1 = sn.dp_to_cp(jnp.asarray(xi), jnp.asarray(omega), jnp.asarray(alpha))
    xi_back, omega_back, alpha_back = sn.cp_to_dp(mu, sigma, gamma1)
    np.testing.assert_allclose(np.asarray(xi_back), xi, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(omega_back), omega, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(alpha_back), alpha, rtol=1e-5, atol=1e-7)


def test_density_dp_matches_scipy():
    """density_dp matches scipy.stats.skewnorm.pdf on a grid."""
    alpha = 3.5
    xi = 1.2
    omega = 0.8
    x = np.linspace(-2.0, 5.0, 401)
    pred = sn.density_dp(jnp.asarray(x), jnp.asarray(xi), jnp.asarray(omega), jnp.asarray(alpha))
    expected = skewnorm.pdf(x, a=alpha, loc=xi, scale=omega)
    np.testing.assert_allclose(np.asarray(pred), expected, rtol=1e-6, atol=1e-9)


def test_density_dp_integrates_to_one():
    """density_dp integrates to 1 on a wide grid for several alpha."""
    for alpha in [-8.0, -1.0, 0.0, 1.0, 5.0]:
        x = np.linspace(-15.0, 15.0, 200_000)
        dx = x[1] - x[0]
        pdf = np.asarray(
            sn.density_dp(jnp.asarray(x), jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(alpha))
        )
        assert abs(pdf.sum() * dx - 1.0) < 1e-3, f"alpha={alpha}: integral={pdf.sum() * dx}"


def test_density_cp_equals_density_dp_after_bijection():
    """density_cp(x | mu, sigma, gamma1) == density_dp(x | cp_to_dp(mu, sigma, gamma1))."""
    mu, sigma, gamma1 = 1.0, 0.5, 0.4
    x = np.linspace(-1.0, 3.5, 301)
    cp_pred = sn.density_cp(jnp.asarray(x), jnp.asarray(mu), jnp.asarray(sigma), jnp.asarray(gamma1))
    xi, omega, alpha = sn.cp_to_dp(jnp.asarray(mu), jnp.asarray(sigma), jnp.asarray(gamma1))
    dp_pred = sn.density_dp(jnp.asarray(x), xi, omega, alpha)
    np.testing.assert_allclose(np.asarray(cp_pred), np.asarray(dp_pred), rtol=1e-7, atol=1e-10)


def test_density_dp_is_differentiable():
    """jax.grad of density_dp w.r.t. each DP parameter runs without error."""
    import jax

    def f_xi(xi: jnp.ndarray) -> jnp.ndarray:
        return sn.density_dp(jnp.asarray(0.5), xi, jnp.asarray(1.0), jnp.asarray(2.0))

    def f_omega(om: jnp.ndarray) -> jnp.ndarray:
        return sn.density_dp(jnp.asarray(0.5), jnp.asarray(0.0), om, jnp.asarray(2.0))

    def f_alpha(a: jnp.ndarray) -> jnp.ndarray:
        return sn.density_dp(jnp.asarray(0.5), jnp.asarray(0.0), jnp.asarray(1.0), a)

    g_xi = jax.grad(f_xi)(jnp.asarray(0.0))
    g_omega = jax.grad(f_omega)(jnp.asarray(1.0))
    g_alpha = jax.grad(f_alpha)(jnp.asarray(2.0))
    assert jnp.isfinite(g_xi)
    assert jnp.isfinite(g_omega)
    assert jnp.isfinite(g_alpha)


def test_mode_dp_zero_alpha_is_xi():
    """At alpha = 0 the mode of N(xi, omega^2) is xi."""
    m = sn.mode_dp(jnp.asarray(1.5), jnp.asarray(0.7), jnp.asarray(0.0))
    np.testing.assert_allclose(float(m), 1.5, atol=1e-12)


def test_mode_dp_is_local_maximum():
    """For each alpha in a grid, density at mode >= density at mode +/- eps."""
    alphas = np.array([-8.0, -2.0, 0.5, 2.0, 8.0])
    eps = 1e-3
    for a in alphas:
        m = float(sn.mode_dp(jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(a)))
        f_at = float(sn.density_dp(jnp.asarray(m), jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(a)))
        f_lo = float(sn.density_dp(jnp.asarray(m - eps), jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(a)))
        f_hi = float(sn.density_dp(jnp.asarray(m + eps), jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(a)))
        assert f_at >= f_lo - 1e-6, f"alpha={a}: f(m)={f_at} < f(m-eps)={f_lo}"
        assert f_at >= f_hi - 1e-6, f"alpha={a}: f(m)={f_at} < f(m+eps)={f_hi}"


def test_mode_dp_close_to_numerical_mode():
    """Azzalini's m_0 approximation is within 1e-3 of the true mode."""
    from scipy.optimize import minimize_scalar

    for a in [-5.0, -1.0, 1.0, 5.0]:
        def neg_pdf(x: float, a: float = a) -> float:
            return -float(skewnorm.pdf(x, a=a))
        result = minimize_scalar(neg_pdf, bracket=(-5.0, 5.0))
        m_true: float = float(result.x)  # type: ignore[union-attr]
        m_pred = float(sn.mode_dp(jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(a)))
        assert abs(m_pred - m_true) < 1e-3, f"alpha={a}: pred={m_pred}, true={m_true}"


def test_fwhm_dp_normal_case():
    """For alpha=0, FWHM of N(0, omega^2) equals 2 * omega * sqrt(2 ln 2)."""
    omega = 1.3
    expected = 2.0 * omega * math.sqrt(2.0 * math.log(2.0))
    pred = float(sn.fwhm_dp(0.0, omega, 0.0))
    np.testing.assert_allclose(pred, expected, rtol=1e-5)


def test_fwhm_dp_consistent_with_density():
    """For each alpha, the spread between half-max points (numerically bracketed
    against scipy) matches fwhm_dp."""
    from scipy.optimize import brentq

    for a in [-4.0, -1.0, 1.0, 4.0]:
        xi, omega = 0.5, 0.9
        m = float(sn.mode_dp(jnp.asarray(xi), jnp.asarray(omega), jnp.asarray(a)))
        peak = float(sn.density_dp(jnp.asarray(m), jnp.asarray(xi), jnp.asarray(omega), jnp.asarray(a)))
        w = float(sn.fwhm_dp(xi, omega, a))

        def f(x: float, a: float = a, xi: float = xi, omega: float = omega, peak: float = peak) -> float:
            return float(skewnorm.pdf(x, a=a, loc=xi, scale=omega)) - peak / 2.0

        x_left = brentq(f, m - 5.0 * omega, m)
        x_right = brentq(f, m, m + 5.0 * omega)
        np.testing.assert_allclose(w, x_right - x_left, rtol=1e-4)


def test_fwhm_dp_array_input():
    """fwhm_dp vectorizes over array alpha."""
    alphas = np.array([-3.0, 0.0, 3.0])
    out = sn.fwhm_dp(np.zeros_like(alphas), np.ones_like(alphas), alphas)
    assert out.shape == alphas.shape
    assert np.all(out > 0.0)


def test_hwhm_ratio_at_zero_alpha_is_one():
    """For alpha=0 the SN is symmetric: HWHM_R / HWHM_L = 1."""
    r = float(sn.hwhm_ratio_dp(0.0, 1.0, 0.0))
    np.testing.assert_allclose(r, 1.0, rtol=1e-5)


def test_hwhm_ratio_independent_of_xi_omega():
    """HWHM ratio depends only on alpha."""
    a = 3.0
    r1 = float(sn.hwhm_ratio_dp(0.0, 1.0, a))
    r2 = float(sn.hwhm_ratio_dp(2.5, 0.4, a))
    r3 = float(sn.hwhm_ratio_dp(-1.0, 2.7, a))
    np.testing.assert_allclose(r1, r2, rtol=1e-5)
    np.testing.assert_allclose(r1, r3, rtol=1e-5)


def test_hwhm_ratio_monotone_in_alpha():
    """HWHM_R / HWHM_L is monotone increasing in alpha."""
    alphas = np.linspace(-10.0, 10.0, 41)
    ratios = sn.hwhm_ratio_dp(np.zeros_like(alphas), np.ones_like(alphas), alphas)
    diffs = np.diff(ratios)
    assert np.all(diffs > 0.0), f"non-monotone: smallest diff = {diffs.min()}"


def test_hwhm_ratio_mirror_symmetry():
    """hwhm_ratio_dp(-alpha) = 1 / hwhm_ratio_dp(alpha)."""
    for a in [0.5, 1.0, 3.0, 8.0]:
        r_pos = float(sn.hwhm_ratio_dp(0.0, 1.0, a))
        r_neg = float(sn.hwhm_ratio_dp(0.0, 1.0, -a))
        np.testing.assert_allclose(r_pos * r_neg, 1.0, rtol=1e-4)
