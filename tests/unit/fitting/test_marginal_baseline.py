"""Closed-form checks for the flat-prior baseline marginalisation."""
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from chromhandler.fitting.model import marginal_baseline_loglik


def _lstsq_reference(y, peak, t, mask, sigma):
    """Independent reference: OLS of (y - peak) on [1, t] over valid points."""
    loglik, intercept, slope = [], [], []
    for tr in range(y.shape[0]):
        m = mask[tr]
        r = (y[tr] - peak[tr])[m]
        tt = t[tr][m]
        X = np.column_stack([np.ones_like(tt), tt])
        beta, *_ = np.linalg.lstsq(X, r, rcond=None)
        resid = r - X @ beta
        n = r.size
        rss_perp = float(resid @ resid)
        s2 = float(sigma[tr] ** 2)
        ll = -0.5 * (n - 2) * np.log(2 * np.pi * s2) - rss_perp / (2 * s2)
        loglik.append(ll)
        intercept.append(float(beta[0]))
        slope.append(float(beta[1]))
    return np.array(loglik), np.array(intercept), np.array(slope)


def test_loglik_matches_lstsq_reference():
    rng = np.random.default_rng(0)
    n_trace, n_time = 3, 40
    t = np.tile(np.linspace(2.0, 4.0, n_time), (n_trace, 1))
    peak = rng.normal(size=(n_trace, n_time))
    y = np.empty((n_trace, n_time))
    for tr, (a, b) in enumerate([(5.0, 1.0), (-2.0, 0.3), (0.0, -0.7)]):
        y[tr] = a + b * t[tr] + peak[tr] + rng.normal(scale=0.1, size=n_time)
    mask = np.ones((n_trace, n_time), dtype=bool)
    sigma = np.array([0.1, 0.2, 0.15])

    ll, intercept, slope = marginal_baseline_loglik(y, peak, t, mask, sigma)
    ll_ref, int_ref, slope_ref = _lstsq_reference(y, peak, t, mask, sigma)

    np.testing.assert_allclose(np.asarray(ll), ll_ref, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(intercept), int_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(slope), slope_ref, rtol=1e-5, atol=1e-6)


def test_mask_excludes_invalid_points():
    n_time = 30
    t = np.linspace(2.0, 4.0, n_time)[None, :]
    peak = np.zeros((1, n_time))
    y = (3.0 + 0.5 * t + np.zeros_like(t))  # exact line, zero residual
    mask = np.ones((1, n_time), dtype=bool)
    mask[0, -5:] = False
    y_poison = y.copy()
    y_poison[0, -5:] = 1e6  # garbage in masked-out region
    sigma = np.array([0.2])

    ll_clean, _, _ = marginal_baseline_loglik(y, peak, t, mask, sigma)
    ll_poison, int_p, slope_p = marginal_baseline_loglik(y_poison, peak, t, mask, sigma)

    np.testing.assert_allclose(np.asarray(ll_clean), np.asarray(ll_poison), rtol=1e-5)
    np.testing.assert_allclose(float(np.asarray(int_p)[0]), 3.0, atol=1e-4)
    np.testing.assert_allclose(float(np.asarray(slope_p)[0]), 0.5, atol=1e-4)
