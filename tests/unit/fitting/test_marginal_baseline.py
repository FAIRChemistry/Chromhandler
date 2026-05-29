"""Closed-form checks for the flat-prior baseline marginalisation."""
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from chromhandler.fitting.model import marginal_baseline_loglik  # noqa: E402


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


def test_float32_large_baseline_stable():
    """Direct-residual form is accurate for large baselines in float32.

    With a baseline of magnitude ~1e4 and noise sigma=0.5, the old
    large-minus-large form loses all precision in float32. The direct-
    residual form avoids this: the rss_perp operands stay at noise²
    magnitude, so float32 is sufficient.

    Note: jax_enable_x64 is set at module level, which allows float64 but
    does NOT force upcast of float32 inputs. We verify this by checking the
    returned array dtype. If JAX were upcasting, the test would still pass
    but would not be exercising the float32 code path.
    """
    rng = np.random.default_rng(42)
    n_time = 50
    t_f64 = np.linspace(2.0, 4.0, n_time)[None, :]  # (1, n_time)
    a, b, sigma_val = 1e4, 2e3, 0.5
    peak_f64 = rng.normal(scale=2.0, size=(1, n_time))
    noise_f64 = rng.normal(scale=sigma_val, size=(1, n_time))
    signal_f64 = a + b * t_f64 + peak_f64 + noise_f64
    mask = np.ones((1, n_time), dtype=bool)
    sigma_arr = np.array([sigma_val])

    # Float64 reference computed via lstsq (independent, high-precision)
    ll_ref, _int_ref, _slope_ref = _lstsq_reference(signal_f64, peak_f64, t_f64, mask, sigma_arr)

    # Cast inputs to float32 for the JAX call
    signal_f32 = signal_f64.astype(np.float32)
    peak_f32 = peak_f64.astype(np.float32)
    t_f32 = t_f64.astype(np.float32)
    sigma_f32 = sigma_arr.astype(np.float32)

    ll_f32, _int_f32, _slope_f32 = marginal_baseline_loglik(
        signal_f32, peak_f32, t_f32, mask, sigma_f32
    )

    ll_f32_np = np.asarray(ll_f32)
    # Verify the computation genuinely ran in float32 (x64 enable does not
    # force upcast of float32 inputs; if this assertion fails then JAX is
    # upcasting and the cancellation guard cannot be exercised in float32).
    assert ll_f32_np.dtype == np.float32, (
        f"Expected float32 output (float32 inputs, x64 not forced); got {ll_f32_np.dtype}. "
        "x64 enable allows float64 but must not silently upcast float32 inputs."
    )

    # The direct-residual form must agree with the float64 reference at 1%
    # relative tolerance even in float32.
    np.testing.assert_allclose(ll_f32_np, ll_ref, rtol=1e-2)


def test_degenerate_traces_return_zero_loglik():
    """Traces with 0 or 1 valid points yield loglik == 0 exactly; normal trace is finite+negative."""
    n_time = 20
    t = np.tile(np.linspace(2.0, 4.0, n_time), (3, 1))
    rng = np.random.default_rng(7)
    signal = rng.normal(scale=0.5, size=(3, n_time))
    peak = np.zeros((3, n_time))
    sigma = np.array([0.5, 0.5, 0.5])

    mask = np.ones((3, n_time), dtype=bool)
    mask[0, :] = False                     # trace 0: 0 valid points
    mask[1, :] = False
    mask[1, 5] = True                      # trace 1: exactly 1 valid point
    # trace 2: all 20 valid points (default)

    ll, intercept, slope = marginal_baseline_loglik(signal, peak, t, mask, sigma)
    ll_np = np.asarray(ll)
    int_np = np.asarray(intercept)
    slope_np = np.asarray(slope)

    assert ll_np[0] == 0.0, f"trace 0 (0 valid pts): expected loglik=0, got {ll_np[0]}"
    assert ll_np[1] == 0.0, f"trace 1 (1 valid pt): expected loglik=0, got {ll_np[1]}"
    assert np.isfinite(ll_np[2]) and ll_np[2] < 0.0, (
        f"trace 2 (normal): expected finite negative loglik, got {ll_np[2]}"
    )
    assert np.all(np.isfinite(ll_np)), f"NaN/Inf in loglik: {ll_np}"
    assert np.all(np.isfinite(int_np)), f"NaN/Inf in intercept: {int_np}"
    assert np.all(np.isfinite(slope_np)), f"NaN/Inf in slope: {slope_np}"


def test_identical_times_no_nan():
    """When all valid time points are identical, Stt=0 → slope unidentified; require finiteness."""
    n_time = 10
    t_val = 3.0
    t = np.full((1, n_time), t_val)
    rng = np.random.default_rng(13)
    signal = rng.normal(scale=0.5, size=(1, n_time))
    peak = np.zeros((1, n_time))
    mask = np.ones((1, n_time), dtype=bool)
    sigma = np.array([0.5])

    ll, intercept, slope = marginal_baseline_loglik(signal, peak, t, mask, sigma)

    assert np.isfinite(np.asarray(ll)).all(), f"loglik not finite: {ll}"
    assert np.isfinite(np.asarray(intercept)).all(), f"intercept not finite: {intercept}"
    assert np.isfinite(np.asarray(slope)).all(), f"slope not finite: {slope}"
