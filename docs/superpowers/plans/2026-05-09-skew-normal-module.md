# Skew-Normal Pure-Math Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `chromhandler/fitting/skew_normal.py` per §7.1 of the design — a pure-math skew-normal layer (CP↔DP bijection, density, mode, FWHM, HWHM-ratio, asymmetry inversion) with comprehensive unit tests. No NumPyro, no priors, no fitter wiring.

**Architecture:** Single new file `skew_normal.py` exposing 8 public functions plus the `GAMMA1_MAX` constant. Pure JAX inside the bijection and density (differentiable, vmappable). Mode uses Azzalini's closed-form approximation. FWHM and HWHM-ratio use scipy's `brentq` (post-hoc / fit-time only, not on the HMC path). The asymmetry inversion table is built lazily via `functools.lru_cache` on the first call.

**Tech Stack:** JAX (`jax.numpy`, `jax.scipy.stats.norm`), scipy (`scipy.optimize.brentq`, `scipy.stats.skewnorm` for tests), pytest. Python 3.11+ with `from __future__ import annotations`.

**Reference spec:** `docs/superpowers/specs/2026-05-07-skew-normal-fitter-rewrite-design.md` §2 (math) and §7.1 (API).

---

## File Structure

- **Create:** `chromhandler/fitting/skew_normal.py` — all eight public functions + `GAMMA1_MAX`
- **Create:** `tests/unit/fitting/test_skew_normal.py` — property tests for every function

The file size budget is ~250 lines. If it grows past ~400, the spec says split — but for this scope it should fit comfortably.

## Conventions used in this plan

- Every code step shows complete code. Append to `skew_normal.py` (or to the test file) unless otherwise stated.
- Quality gates after **every** task that adds code: `uv run ruff check <file>` and `uv run pyright <file>` must pass before the commit step.
- Tests run with: `uv run pytest tests/unit/fitting/test_skew_normal.py -v`.
- All `jnp` math operates on `jax.Array`. Tests pass `jnp.asarray(...)` inputs, compare against numpy/scipy with `np.testing.assert_allclose`.
- Tolerance defaults: `rtol=1e-5, atol=1e-7` for closed-form math; `rtol=1e-3` for numerical (FWHM, integrals).

---

## Task 1: Module skeleton and `GAMMA1_MAX`

**Files:**
- Create: `chromhandler/fitting/skew_normal.py`
- Create: `tests/unit/fitting/test_skew_normal.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/fitting/test_skew_normal.py`:

```python
"""Property tests for the pure-math skew-normal layer."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import skewnorm

from chromhandler.fitting import skew_normal as sn


def test_gamma1_max_matches_half_normal_limit():
    """GAMMA1_MAX equals the skewness of the half-normal (α → ∞ limit)."""
    expected = skewnorm.stats(a=1e6, moments="s")
    assert math.isclose(sn.GAMMA1_MAX, float(expected), rel_tol=1e-6)


def test_gamma1_max_closed_form():
    """GAMMA1_MAX matches the closed-form expression in spec §2.2."""
    b = math.sqrt(2.0 / math.pi)
    expected = ((4.0 - math.pi) / 2.0) * b**3 / (1.0 - 2.0 / math.pi) ** 1.5
    assert math.isclose(sn.GAMMA1_MAX, expected, rel_tol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/fitting/test_skew_normal.py -v`
Expected: FAIL with `ImportError` / `ModuleNotFoundError` for `chromhandler.fitting.skew_normal`.

- [ ] **Step 3: Write the module skeleton**

Create `chromhandler/fitting/skew_normal.py`:

```python
"""Pure-math skew-normal layer.

Implements the centred-parameter (CP) ↔ direct-parameter (DP) bijection,
density evaluation in both forms, and the derived quantities (mode, FWHM,
HWHM-ratio, asymmetry-to-γ₁ inversion) needed by the priors and posterior
layers. No NumPyro imports, no state, no side effects.

See ``docs/superpowers/specs/2026-05-07-skew-normal-fitter-rewrite-design.md``
§2 (math) and §7.1 (API).
"""

from __future__ import annotations

import math

import jax.numpy as jnp

# Skewness of the half-normal distribution = max |γ₁| achievable by any
# skew-normal. See spec §2.2.
GAMMA1_MAX: float = (
    ((4.0 - math.pi) / 2.0)
    * (math.sqrt(2.0 / math.pi) ** 3)
    / (1.0 - 2.0 / math.pi) ** 1.5
)

_B_CONST: float = math.sqrt(2.0 / math.pi)
```

- [ ] **Step 4: Run tests**

Run: `uv run ruff check chromhandler/fitting/skew_normal.py tests/unit/fitting/test_skew_normal.py`
Run: `uv run pyright chromhandler/fitting/skew_normal.py`
Run: `uv run pytest tests/unit/fitting/test_skew_normal.py -v`
Expected: ruff/pyright clean, both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/skew_normal.py tests/unit/fitting/test_skew_normal.py
git commit -m "Add skew_normal module skeleton with GAMMA1_MAX"
```

---

## Task 2: Forward bijection `cp_to_dp`

**Files:**
- Modify: `chromhandler/fitting/skew_normal.py` (append)
- Modify: `tests/unit/fitting/test_skew_normal.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `test_skew_normal.py`:

```python
def test_cp_to_dp_zero_skew_is_identity_on_mu_sigma():
    """At γ₁ = 0 the SN reduces to N(μ, σ²): ξ=μ, ω=σ, α=0."""
    xi, omega, alpha = sn.cp_to_dp(jnp.asarray(1.5), jnp.asarray(0.4), jnp.asarray(0.0))
    np.testing.assert_allclose(float(xi), 1.5, atol=1e-12)
    np.testing.assert_allclose(float(omega), 0.4, atol=1e-12)
    np.testing.assert_allclose(float(alpha), 0.0, atol=1e-12)


def test_cp_to_dp_against_scipy_grid():
    """For (α, ω, ξ) sampled from scipy, the inverse via cp_to_dp recovers them."""
    rng = np.random.default_rng(0)
    alpha_true = rng.uniform(-15.0, 15.0, size=20)
    omega_true = rng.uniform(0.1, 2.0, size=20)
    xi_true = rng.uniform(-3.0, 3.0, size=20)
    # Build CP from DP via scipy's mean/var/skew, then run our forward map.
    mu = xi_true + omega_true * _B_CONST_NP * alpha_true / np.sqrt(1.0 + alpha_true**2)
    delta = alpha_true / np.sqrt(1.0 + alpha_true**2)
    sigma = omega_true * np.sqrt(1.0 - _B_CONST_NP**2 * delta**2)
    gamma1 = ((4.0 - np.pi) / 2.0) * (_B_CONST_NP * delta) ** 3 / (1.0 - _B_CONST_NP**2 * delta**2) ** 1.5
    xi_pred, omega_pred, alpha_pred = sn.cp_to_dp(
        jnp.asarray(mu), jnp.asarray(sigma), jnp.asarray(gamma1)
    )
    np.testing.assert_allclose(np.asarray(xi_pred), xi_true, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(omega_pred), omega_true, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(alpha_pred), alpha_true, rtol=1e-5, atol=1e-7)
```

Add this constant at the top of the test file (right after the imports) so the test works:

```python
_B_CONST_NP = math.sqrt(2.0 / math.pi)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/fitting/test_skew_normal.py::test_cp_to_dp_zero_skew_is_identity_on_mu_sigma -v`
Expected: FAIL with `AttributeError: module 'chromhandler.fitting.skew_normal' has no attribute 'cp_to_dp'`.

- [ ] **Step 3: Implement `cp_to_dp`**

Append to `skew_normal.py`:

```python
def cp_to_dp(
    mu: jnp.ndarray, sigma: jnp.ndarray, gamma1: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert centred parameters (μ, σ, γ₁) to direct parameters (ξ, ω, α).

    Closed form via the Azzalini relations (spec §2.3).

    Args:
        mu: Mean of the skew-normal. Any broadcastable shape.
        sigma: Standard deviation. Strictly positive.
        gamma1: Skewness coefficient. Must satisfy ``|gamma1| < GAMMA1_MAX``;
            values outside the open interval are not in the SN family.

    Returns:
        Tuple ``(xi, omega, alpha)`` of DP parameters, broadcast to the
        common shape of the inputs.
    """
    c = jnp.cbrt(2.0 * gamma1 / (4.0 - jnp.pi))
    b_delta = c / jnp.sqrt(1.0 + c**2)
    delta = b_delta / _B_CONST
    omega = sigma / jnp.sqrt(1.0 - b_delta**2)
    alpha = delta / jnp.sqrt(1.0 - delta**2)
    xi = mu - omega * b_delta
    return xi, omega, alpha
```

- [ ] **Step 4: Run quality gates and tests**

Run: `uv run ruff check chromhandler/fitting/skew_normal.py tests/unit/fitting/test_skew_normal.py`
Run: `uv run pyright chromhandler/fitting/skew_normal.py`
Run: `uv run pytest tests/unit/fitting/test_skew_normal.py -v`
Expected: all green, 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/skew_normal.py tests/unit/fitting/test_skew_normal.py
git commit -m "Add cp_to_dp bijection (CP → DP)"
```

---

## Task 3: Inverse bijection `dp_to_cp` and round-trip property

**Files:**
- Modify: `chromhandler/fitting/skew_normal.py` (append)
- Modify: `tests/unit/fitting/test_skew_normal.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `test_skew_normal.py`:

```python
def test_dp_to_cp_zero_alpha_is_identity_on_xi_omega():
    """At α = 0 the SN is N(ξ, ω²): μ=ξ, σ=ω, γ₁=0."""
    mu, sigma, gamma1 = sn.dp_to_cp(jnp.asarray(2.0), jnp.asarray(0.5), jnp.asarray(0.0))
    np.testing.assert_allclose(float(mu), 2.0, atol=1e-12)
    np.testing.assert_allclose(float(sigma), 0.5, atol=1e-12)
    np.testing.assert_allclose(float(gamma1), 0.0, atol=1e-12)


def test_dp_to_cp_matches_scipy_moments():
    """dp_to_cp matches scipy.stats.skewnorm.stats for a grid of α."""
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
    """dp_to_cp ∘ cp_to_dp = identity on a grid of (ξ, ω, α)."""
    rng = np.random.default_rng(2)
    xi = rng.uniform(-2.0, 2.0, size=50)
    omega = rng.uniform(0.2, 1.5, size=50)
    alpha = rng.uniform(-15.0, 15.0, size=50)
    mu, sigma, gamma1 = sn.dp_to_cp(jnp.asarray(xi), jnp.asarray(omega), jnp.asarray(alpha))
    xi_back, omega_back, alpha_back = sn.cp_to_dp(mu, sigma, gamma1)
    np.testing.assert_allclose(np.asarray(xi_back), xi, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(omega_back), omega, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(alpha_back), alpha, rtol=1e-5, atol=1e-7)
```

- [ ] **Step 2: Run test to verify they fail**

Run: `uv run pytest tests/unit/fitting/test_skew_normal.py -v`
Expected: 4 new tests FAIL with `AttributeError: ... 'dp_to_cp'`.

- [ ] **Step 3: Implement `dp_to_cp`**

Append to `skew_normal.py`:

```python
def dp_to_cp(
    xi: jnp.ndarray, omega: jnp.ndarray, alpha: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert direct parameters (ξ, ω, α) to centred parameters (μ, σ, γ₁).

    Forward Azzalini formulas (spec §2.3). Inverse of :func:`cp_to_dp`.

    Args:
        xi: DP location.
        omega: DP scale, strictly positive.
        alpha: DP slant. Any real value.

    Returns:
        Tuple ``(mu, sigma, gamma1)`` of CP parameters, broadcast to the
        common shape of the inputs.
    """
    delta = alpha / jnp.sqrt(1.0 + alpha**2)
    b_delta = _B_CONST * delta
    mu = xi + omega * b_delta
    sigma = omega * jnp.sqrt(1.0 - b_delta**2)
    gamma1 = ((4.0 - jnp.pi) / 2.0) * b_delta**3 / (1.0 - b_delta**2) ** 1.5
    return mu, sigma, gamma1
```

- [ ] **Step 4: Run quality gates and tests**

Run: `uv run ruff check chromhandler/fitting/skew_normal.py tests/unit/fitting/test_skew_normal.py`
Run: `uv run pyright chromhandler/fitting/skew_normal.py`
Run: `uv run pytest tests/unit/fitting/test_skew_normal.py -v`
Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/skew_normal.py tests/unit/fitting/test_skew_normal.py
git commit -m "Add dp_to_cp bijection and round-trip property tests"
```

---

## Task 4: `density_dp` and `density_cp`

**Files:**
- Modify: `chromhandler/fitting/skew_normal.py` (append + add import)
- Modify: `tests/unit/fitting/test_skew_normal.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `test_skew_normal.py`:

```python
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
    """density_dp integrates to 1 on a wide grid for several α."""
    for alpha in [-8.0, -1.0, 0.0, 1.0, 5.0]:
        x = np.linspace(-15.0, 15.0, 200_000)
        dx = x[1] - x[0]
        pdf = np.asarray(
            sn.density_dp(jnp.asarray(x), jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(alpha))
        )
        assert abs(pdf.sum() * dx - 1.0) < 1e-3, f"α={alpha}: integral={pdf.sum() * dx}"


def test_density_cp_equals_density_dp_after_bijection():
    """density_cp(x | μ, σ, γ₁) == density_dp(x | cp_to_dp(μ, σ, γ₁))."""
    mu, sigma, gamma1 = 1.0, 0.5, 0.4
    x = np.linspace(-1.0, 3.5, 301)
    cp_pred = sn.density_cp(jnp.asarray(x), jnp.asarray(mu), jnp.asarray(sigma), jnp.asarray(gamma1))
    xi, omega, alpha = sn.cp_to_dp(jnp.asarray(mu), jnp.asarray(sigma), jnp.asarray(gamma1))
    dp_pred = sn.density_dp(jnp.asarray(x), xi, omega, alpha)
    np.testing.assert_allclose(np.asarray(cp_pred), np.asarray(dp_pred), rtol=1e-7, atol=1e-10)


def test_density_dp_is_differentiable():
    """jax.grad of density_dp w.r.t. each DP parameter runs without error."""
    import jax

    f_xi = lambda xi: sn.density_dp(jnp.asarray(0.5), xi, jnp.asarray(1.0), jnp.asarray(2.0))
    f_omega = lambda om: sn.density_dp(jnp.asarray(0.5), jnp.asarray(0.0), om, jnp.asarray(2.0))
    f_alpha = lambda a: sn.density_dp(jnp.asarray(0.5), jnp.asarray(0.0), jnp.asarray(1.0), a)
    g_xi = jax.grad(f_xi)(jnp.asarray(0.0))
    g_omega = jax.grad(f_omega)(jnp.asarray(1.0))
    g_alpha = jax.grad(f_alpha)(jnp.asarray(2.0))
    assert jnp.isfinite(g_xi)
    assert jnp.isfinite(g_omega)
    assert jnp.isfinite(g_alpha)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/fitting/test_skew_normal.py -v`
Expected: 4 new tests FAIL with `AttributeError: ... 'density_dp'`.

- [ ] **Step 3: Implement densities**

At the top of `skew_normal.py`, add to the imports:

```python
import jax.scipy.stats as jss
```

Append to `skew_normal.py`:

```python
def density_dp(
    x: jnp.ndarray,
    xi: jnp.ndarray,
    omega: jnp.ndarray,
    alpha: jnp.ndarray,
) -> jnp.ndarray:
    """Skew-normal density in DP form.

    ``f(x) = (2/ω) φ((x − ξ)/ω) Φ(α (x − ξ)/ω)``.

    Args:
        x: Evaluation points. Any broadcastable shape.
        xi: DP location, broadcastable with ``x``.
        omega: DP scale, strictly positive, broadcastable with ``x``.
        alpha: DP slant, broadcastable with ``x``.

    Returns:
        Density values with shape broadcast from inputs.
    """
    z = (x - xi) / omega
    phi = jss.norm.pdf(z)
    cdf = jss.norm.cdf(alpha * z)
    return 2.0 * phi * cdf / omega


def density_cp(
    x: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    gamma1: jnp.ndarray,
) -> jnp.ndarray:
    """Skew-normal density in CP form.

    Internally converts (μ, σ, γ₁) → (ξ, ω, α) via :func:`cp_to_dp` then
    delegates to :func:`density_dp`.

    Args:
        x: Evaluation points.
        mu: Mean.
        sigma: Standard deviation, strictly positive.
        gamma1: Skewness coefficient, ``|gamma1| < GAMMA1_MAX``.

    Returns:
        Density values.
    """
    xi, omega, alpha = cp_to_dp(mu, sigma, gamma1)
    return density_dp(x, xi, omega, alpha)
```

- [ ] **Step 4: Run quality gates and tests**

Run: `uv run ruff check chromhandler/fitting/skew_normal.py tests/unit/fitting/test_skew_normal.py`
Run: `uv run pyright chromhandler/fitting/skew_normal.py`
Run: `uv run pytest tests/unit/fitting/test_skew_normal.py -v`
Expected: all 12 tests pass.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/skew_normal.py tests/unit/fitting/test_skew_normal.py
git commit -m "Add density_dp and density_cp with scipy parity tests"
```

---

## Task 5: `mode_dp` via Azzalini's m₀ approximation

**Files:**
- Modify: `chromhandler/fitting/skew_normal.py` (append)
- Modify: `tests/unit/fitting/test_skew_normal.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `test_skew_normal.py`:

```python
def test_mode_dp_zero_alpha_is_xi():
    """At α = 0 the mode of N(ξ, ω²) is ξ."""
    m = sn.mode_dp(jnp.asarray(1.5), jnp.asarray(0.7), jnp.asarray(0.0))
    np.testing.assert_allclose(float(m), 1.5, atol=1e-12)


def test_mode_dp_is_local_maximum():
    """For each α in a grid, density at mode ≥ density at mode ± ε."""
    alphas = np.array([-8.0, -2.0, 0.5, 2.0, 8.0])
    eps = 1e-3
    for a in alphas:
        m = float(sn.mode_dp(jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(a)))
        f_at = float(sn.density_dp(jnp.asarray(m), jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(a)))
        f_lo = float(sn.density_dp(jnp.asarray(m - eps), jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(a)))
        f_hi = float(sn.density_dp(jnp.asarray(m + eps), jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(a)))
        assert f_at >= f_lo - 1e-6, f"α={a}: f(m)={f_at} < f(m-ε)={f_lo}"
        assert f_at >= f_hi - 1e-6, f"α={a}: f(m)={f_at} < f(m+ε)={f_hi}"


def test_mode_dp_close_to_numerical_mode():
    """Azzalini's m₀ approximation is within 1e-3 of the true mode."""
    from scipy.optimize import minimize_scalar

    for a in [-5.0, -1.0, 1.0, 5.0]:
        # True mode found by scipy.
        result = minimize_scalar(lambda x: -skewnorm.pdf(x, a=a), bracket=(-5.0, 5.0))
        m_true = result.x
        m_pred = float(sn.mode_dp(jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(a)))
        assert abs(m_pred - m_true) < 1e-3, f"α={a}: pred={m_pred}, true={m_true}"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/fitting/test_skew_normal.py -v`
Expected: 3 new tests FAIL with `AttributeError: ... 'mode_dp'`.

- [ ] **Step 3: Implement `mode_dp`**

Append to `skew_normal.py`:

```python
def mode_dp(
    xi: jnp.ndarray, omega: jnp.ndarray, alpha: jnp.ndarray
) -> jnp.ndarray:
    """Mode of SN(ξ, ω, α) via Azzalini's m₀ approximation.

    The skew-normal mode has no closed form. The approximation
    ``m₀(α) = μ_z − γ₁_z σ_z / 2 − sign(α)/2 · exp(−2π/|α|)`` is accurate
    to ~10⁻⁴ everywhere; mode = ξ + ω · m₀(α). Used for reporting only.

    Args:
        xi: DP location.
        omega: DP scale.
        alpha: DP slant.

    Returns:
        Mode of the density, shape broadcast from inputs.
    """
    delta = alpha / jnp.sqrt(1.0 + alpha**2)
    mu_z = _B_CONST * delta
    sigma_z = jnp.sqrt(1.0 - mu_z**2)
    gamma1_z = ((4.0 - jnp.pi) / 2.0) * mu_z**3 / (1.0 - mu_z**2) ** 1.5
    abs_alpha = jnp.abs(alpha)
    safe_alpha = jnp.where(abs_alpha > 1e-12, abs_alpha, 1.0)
    exp_term = jnp.where(abs_alpha > 1e-12, jnp.exp(-2.0 * jnp.pi / safe_alpha), 0.0)
    m_0 = mu_z - gamma1_z * sigma_z / 2.0 - jnp.sign(alpha) * exp_term / 2.0
    return xi + omega * m_0
```

- [ ] **Step 4: Run quality gates and tests**

Run: `uv run ruff check chromhandler/fitting/skew_normal.py tests/unit/fitting/test_skew_normal.py`
Run: `uv run pyright chromhandler/fitting/skew_normal.py`
Run: `uv run pytest tests/unit/fitting/test_skew_normal.py -v`
Expected: all 15 tests pass.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/skew_normal.py tests/unit/fitting/test_skew_normal.py
git commit -m "Add mode_dp via Azzalini's m₀ approximation"
```

---

## Task 6: `fwhm_dp` numerical implementation

**Files:**
- Modify: `chromhandler/fitting/skew_normal.py` (append + import)
- Modify: `tests/unit/fitting/test_skew_normal.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `test_skew_normal.py`:

```python
def test_fwhm_dp_normal_case():
    """For α=0, FWHM of N(0, ω²) equals 2 ω √(2 ln 2)."""
    omega = 1.3
    expected = 2.0 * omega * math.sqrt(2.0 * math.log(2.0))
    pred = float(sn.fwhm_dp(0.0, omega, 0.0))
    np.testing.assert_allclose(pred, expected, rtol=1e-5)


def test_fwhm_dp_consistent_with_density():
    """At x = mode ± ... the density at the FWHM endpoints equals peak/2."""
    for a in [-4.0, -1.0, 1.0, 4.0]:
        xi, omega = 0.5, 0.9
        m = float(sn.mode_dp(jnp.asarray(xi), jnp.asarray(omega), jnp.asarray(a)))
        peak = float(sn.density_dp(jnp.asarray(m), jnp.asarray(xi), jnp.asarray(omega), jnp.asarray(a)))
        w = float(sn.fwhm_dp(xi, omega, a))
        # The midpoint of [x_left, x_right] need NOT be the mode for skew
        # densities, so probe both half-max points by bisecting from the
        # mode against density(x) − peak/2 and check the spread = w.
        from scipy.optimize import brentq

        f = lambda x: skewnorm.pdf(x, a=a, loc=xi, scale=omega) - peak / 2.0
        x_left = brentq(f, m - 5.0 * omega, m)
        x_right = brentq(f, m, m + 5.0 * omega)
        np.testing.assert_allclose(w, x_right - x_left, rtol=1e-4)


def test_fwhm_dp_array_input():
    """fwhm_dp vectorizes over array α."""
    alphas = np.array([-3.0, 0.0, 3.0])
    out = sn.fwhm_dp(np.zeros_like(alphas), np.ones_like(alphas), alphas)
    assert out.shape == alphas.shape
    assert np.all(out > 0.0)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/fitting/test_skew_normal.py -v`
Expected: 3 new tests FAIL.

- [ ] **Step 3: Implement `fwhm_dp`**

At the top of `skew_normal.py`, add:

```python
import numpy as np
from scipy.optimize import brentq
```

Append to `skew_normal.py`:

```python
def _fwhm_scalar(xi: float, omega: float, alpha: float) -> float:
    """Scalar FWHM of SN(ξ, ω, α) via two ``brentq`` solves."""
    mode = float(mode_dp(jnp.asarray(xi), jnp.asarray(omega), jnp.asarray(alpha)))
    peak = float(
        density_dp(jnp.asarray(mode), jnp.asarray(xi), jnp.asarray(omega), jnp.asarray(alpha))
    )
    half = peak / 2.0

    def shifted(x: float) -> float:
        return float(
            density_dp(jnp.asarray(x), jnp.asarray(xi), jnp.asarray(omega), jnp.asarray(alpha))
        ) - half

    # Walk outward from mode by `omega` until the density drops below half-max.
    x_lo = mode - omega
    while shifted(x_lo) > 0.0:
        x_lo -= omega
    x_hi = mode + omega
    while shifted(x_hi) > 0.0:
        x_hi += omega
    x_left = brentq(shifted, x_lo, mode)
    x_right = brentq(shifted, mode, x_hi)
    return float(x_right - x_left)


def fwhm_dp(
    xi: float | np.ndarray,
    omega: float | np.ndarray,
    alpha: float | np.ndarray,
) -> np.ndarray:
    """Full width at half maximum of SN(ξ, ω, α), computed numerically.

    Uses :func:`mode_dp` to locate the apex, then ``scipy.optimize.brentq``
    on each side to find where the density drops to half-maximum.
    Vectorized via :func:`numpy.vectorize`. Used for reporting only — not
    on the HMC path.

    Args:
        xi: DP location, scalar or array.
        omega: DP scale, scalar or array.
        alpha: DP slant, scalar or array.

    Returns:
        FWHM as a numpy array with the broadcast shape of the inputs.
    """
    return np.vectorize(_fwhm_scalar, otypes=[float])(
        np.asarray(xi, dtype=float),
        np.asarray(omega, dtype=float),
        np.asarray(alpha, dtype=float),
    )
```

- [ ] **Step 4: Run quality gates and tests**

Run: `uv run ruff check chromhandler/fitting/skew_normal.py tests/unit/fitting/test_skew_normal.py`
Run: `uv run pyright chromhandler/fitting/skew_normal.py`
Run: `uv run pytest tests/unit/fitting/test_skew_normal.py -v`
Expected: all 18 tests pass.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/skew_normal.py tests/unit/fitting/test_skew_normal.py
git commit -m "Add fwhm_dp via brentq half-max bracketing"
```

---

## Task 7: `hwhm_ratio_dp`

**Files:**
- Modify: `chromhandler/fitting/skew_normal.py` (append)
- Modify: `tests/unit/fitting/test_skew_normal.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `test_skew_normal.py`:

```python
def test_hwhm_ratio_at_zero_alpha_is_one():
    """For α=0 the SN is symmetric: HWHM_R / HWHM_L = 1."""
    r = float(sn.hwhm_ratio_dp(0.0, 1.0, 0.0))
    np.testing.assert_allclose(r, 1.0, rtol=1e-5)


def test_hwhm_ratio_independent_of_xi_omega():
    """HWHM ratio depends only on α."""
    a = 3.0
    r1 = float(sn.hwhm_ratio_dp(0.0, 1.0, a))
    r2 = float(sn.hwhm_ratio_dp(2.5, 0.4, a))
    r3 = float(sn.hwhm_ratio_dp(-1.0, 2.7, a))
    np.testing.assert_allclose(r1, r2, rtol=1e-5)
    np.testing.assert_allclose(r1, r3, rtol=1e-5)


def test_hwhm_ratio_monotone_in_alpha():
    """HWHM_R / HWHM_L is monotone increasing in α."""
    alphas = np.linspace(-10.0, 10.0, 41)
    ratios = sn.hwhm_ratio_dp(np.zeros_like(alphas), np.ones_like(alphas), alphas)
    diffs = np.diff(ratios)
    assert np.all(diffs > 0.0), f"non-monotone: smallest diff = {diffs.min()}"


def test_hwhm_ratio_mirror_symmetry():
    """hwhm_ratio_dp(−α) = 1 / hwhm_ratio_dp(α)."""
    for a in [0.5, 1.0, 3.0, 8.0]:
        r_pos = float(sn.hwhm_ratio_dp(0.0, 1.0, a))
        r_neg = float(sn.hwhm_ratio_dp(0.0, 1.0, -a))
        np.testing.assert_allclose(r_pos * r_neg, 1.0, rtol=1e-4)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/fitting/test_skew_normal.py -v`
Expected: 4 new tests FAIL.

- [ ] **Step 3: Implement `hwhm_ratio_dp`**

Append to `skew_normal.py`:

```python
def _hwhm_ratio_scalar(alpha: float) -> float:
    """Scalar HWHM_R / HWHM_L of SN(0, 1, α). Independent of ξ and ω."""
    mode = float(mode_dp(jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(alpha)))
    peak = float(
        density_dp(jnp.asarray(mode), jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(alpha))
    )
    half = peak / 2.0

    def shifted(x: float) -> float:
        return float(
            density_dp(jnp.asarray(x), jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(alpha))
        ) - half

    x_lo = mode - 1.0
    while shifted(x_lo) > 0.0:
        x_lo -= 1.0
    x_hi = mode + 1.0
    while shifted(x_hi) > 0.0:
        x_hi += 1.0
    x_left = brentq(shifted, x_lo, mode)
    x_right = brentq(shifted, mode, x_hi)
    return (x_right - mode) / (mode - x_left)


def hwhm_ratio_dp(
    xi: float | np.ndarray,
    omega: float | np.ndarray,
    alpha: float | np.ndarray,
) -> np.ndarray:
    """Right-to-left HWHM ratio of SN(ξ, ω, α). Independent of ξ and ω.

    Used at fit time to invert measured peak asymmetry to γ₁ via
    :func:`sn_asymmetry_to_gamma1`. The (xi, omega) arguments are accepted
    for signature symmetry with the rest of the DP API but are ignored.

    Args:
        xi: Ignored (kept for API symmetry).
        omega: Ignored (kept for API symmetry).
        alpha: DP slant, scalar or array.

    Returns:
        HWHM_R / HWHM_L as a numpy array, broadcast shape of ``alpha``.
    """
    del xi, omega  # ratio is invariant under (ξ, ω); kept in signature for symmetry.
    return np.vectorize(_hwhm_ratio_scalar, otypes=[float])(
        np.asarray(alpha, dtype=float)
    )
```

- [ ] **Step 4: Run quality gates and tests**

Run: `uv run ruff check chromhandler/fitting/skew_normal.py tests/unit/fitting/test_skew_normal.py`
Run: `uv run pyright chromhandler/fitting/skew_normal.py`
Run: `uv run pytest tests/unit/fitting/test_skew_normal.py -v`
Expected: all 22 tests pass.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/skew_normal.py tests/unit/fitting/test_skew_normal.py
git commit -m "Add hwhm_ratio_dp for SN asymmetry computation"
```

---

## Task 8: `sn_asymmetry_to_gamma1` table inversion

**Files:**
- Modify: `chromhandler/fitting/skew_normal.py` (append + import)
- Modify: `tests/unit/fitting/test_skew_normal.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `test_skew_normal.py`:

```python
def test_sn_asymmetry_at_ratio_one_is_zero_skew():
    """A symmetric peak (ratio = 1) inverts to γ₁ = 0."""
    g = float(sn.sn_asymmetry_to_gamma1(jnp.asarray(1.0)))
    assert abs(g) < 1e-3


def test_sn_asymmetry_round_trip_against_dp_to_cp():
    """For α in a grid, ratio→γ₁ via the table matches dp_to_cp(α)."""
    alphas = np.linspace(-10.0, 10.0, 21)
    ratios = sn.hwhm_ratio_dp(np.zeros_like(alphas), np.ones_like(alphas), alphas)
    _, _, gamma1_true = sn.dp_to_cp(jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(alphas))
    gamma1_table = sn.sn_asymmetry_to_gamma1(jnp.asarray(ratios))
    np.testing.assert_allclose(
        np.asarray(gamma1_table), np.asarray(gamma1_true), rtol=1e-3, atol=1e-3
    )


def test_sn_asymmetry_handles_array_input():
    """Table inversion works on array input."""
    g = sn.sn_asymmetry_to_gamma1(jnp.asarray([0.7, 1.0, 1.4]))
    assert g.shape == (3,)
    assert float(g[0]) < 0.0  # ratio < 1 → left-skewed (γ₁ < 0)
    assert abs(float(g[1])) < 1e-3
    assert float(g[2]) > 0.0  # ratio > 1 → right-skewed (γ₁ > 0)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/fitting/test_skew_normal.py -v`
Expected: 3 new tests FAIL.

- [ ] **Step 3: Implement `sn_asymmetry_to_gamma1`**

At the top of `skew_normal.py`, add:

```python
import functools
```

Append to `skew_normal.py`:

```python
@functools.lru_cache(maxsize=1)
def _asymmetry_table() -> tuple[np.ndarray, np.ndarray]:
    """Build the (ratio → γ₁) inversion table once.

    Sweeps α over a wide grid, computes HWHM ratio via :func:`hwhm_ratio_dp`
    and γ₁ via :func:`dp_to_cp`, then sorts by ratio so the result is
    monotone-increasing in ratio. Cached on the first call.
    """
    alphas = np.linspace(-50.0, 50.0, 4001)
    ratios = np.vectorize(_hwhm_ratio_scalar, otypes=[float])(alphas)
    _, _, gamma1s = dp_to_cp(jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(alphas))
    gamma1s_np = np.asarray(gamma1s)
    order = np.argsort(ratios)
    return ratios[order], gamma1s_np[order]


def sn_asymmetry_to_gamma1(ratio: jnp.ndarray) -> jnp.ndarray:
    """Invert measured HWHM_R/HWHM_L ratio to γ₁ via a precomputed table.

    Used at prior-build time only — once per fit. The table is built on
    first call and cached for the process lifetime.

    Args:
        ratio: Measured HWHM_R / HWHM_L. Scalar or array. For symmetric
            peaks ratio≈1 → γ₁≈0; ratio>1 → γ₁>0; ratio<1 → γ₁<0.

    Returns:
        Interpolated γ₁ values with the shape of ``ratio``.
    """
    ratios_grid, gamma1_grid = _asymmetry_table()
    return jnp.interp(jnp.asarray(ratio), jnp.asarray(ratios_grid), jnp.asarray(gamma1_grid))
```

- [ ] **Step 4: Run quality gates and tests**

Run: `uv run ruff check chromhandler/fitting/skew_normal.py tests/unit/fitting/test_skew_normal.py`
Run: `uv run pyright chromhandler/fitting/skew_normal.py`
Run: `uv run pytest tests/unit/fitting/test_skew_normal.py -v`
Expected: all 25 tests pass.

- [ ] **Step 5: Commit**

```bash
git add chromhandler/fitting/skew_normal.py tests/unit/fitting/test_skew_normal.py
git commit -m "Add sn_asymmetry_to_gamma1 table inversion"
```

---

## Task 9: Final review and full-suite check

- [ ] **Step 1: Run full module quality gates**

Run: `uv run ruff check chromhandler/fitting/skew_normal.py tests/unit/fitting/test_skew_normal.py`
Run: `uv run ruff format chromhandler/fitting/skew_normal.py tests/unit/fitting/test_skew_normal.py`
Run: `uv run pyright chromhandler/fitting/skew_normal.py tests/unit/fitting/test_skew_normal.py`
Run: `uv run pytest tests/unit/fitting/test_skew_normal.py -v`
Expected: 25 tests pass, no lint or type errors.

- [ ] **Step 2: Verify project-wide tests still pass**

Run: `uv run pytest tests/unit/fitting/ -v`
Expected: all existing tests still pass alongside the 25 new ones.

- [ ] **Step 3: If formatting touched any files, commit**

```bash
git add -u
git diff --cached --quiet || git commit -m "Apply ruff format to skew_normal module"
```

---

## Self-review notes (for plan author / reviewer)

**Spec coverage (§7.1):**
- `GAMMA1_MAX` — Task 1 ✓
- `cp_to_dp` — Task 2 ✓
- `dp_to_cp` — Task 3 ✓
- `density_dp` — Task 4 ✓
- `density_cp` — Task 4 ✓
- `mode_dp` — Task 5 ✓
- `fwhm_dp` — Task 6 ✓
- `hwhm_ratio_dp` — Task 7 ✓
- `sn_asymmetry_to_gamma1` — Task 8 ✓

**Spec testing strategy (§9 first bullet):**
- CP→DP→CP and DP→CP→DP round trips — Task 3 ✓
- Density integrates to 1 — Task 4 ✓
- Density matches `scipy.stats.skewnorm` — Task 4 ✓
- Mode is a local maximum — Task 5 ✓
- FWHM consistent with density — Task 6 ✓

**Out of scope (deferred to follow-up plans):**
- `priors.py` (FWHM-based hybrid extraction, aggregation, `SkewNormalPriors` dataclass) — separate plan.
- `model.py` (NumPyro layer, sample sites, pooling) — separate plan.
- `posterior.py`, `fitter.py`, `visualize.py` — separate plans.
- Wiring the new module into `chromhandler/fitting/__init__.py` — happens once `model.py` lands; for this plan the module is reachable via the explicit import path used in tests.
