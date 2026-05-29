"""Pure-math exponentially-modified Gaussian (EMG) layer.

EMG = Gaussian(mu, sigma) convolved with a right Exp(mean=tau), tau > 0.
Equivalent to scipy.stats.exponnorm with K = tau/sigma, loc = mu, scale = sigma.

The density uses a regime switch on w = (sigma/tau - (x-mu)/sigma)/sqrt(2)
because the two analytically-equal forms each overflow on one side. jax has no
erfcx, so the w>=0 branch builds it from erfc with an asymptotic tail; both
branches use the safe-`where` pattern (inputs clamped) so gradients stay finite.
No NumPyro imports, no state.
"""
from __future__ import annotations

import math

import jax.numpy as jnp
from jax.scipy.special import erfc
from scipy.optimize import brentq, minimize_scalar

_SQRT2 = math.sqrt(2.0)
_INV_SQRTPI = 1.0 / math.sqrt(math.pi)


def _erfcx_pos(w: jnp.ndarray) -> jnp.ndarray:
    """Scaled complementary error function erfcx(w)=exp(w^2)erfc(w), for w>=0.

    Direct exp(w^2)*erfc(w) for small w; asymptotic series for large w (where,
    in float32, exp(w^2) overflows and erfc(w) underflows). Both branch inputs
    are clamped so the inactive branch can't overflow (finite gradients).
    """
    w_small = jnp.minimum(w, 6.0)
    small = jnp.exp(w_small ** 2) * erfc(w_small)
    w_large = jnp.maximum(w, 1.0)
    inv = 1.0 / (w_large ** 2)
    asymp = (_INV_SQRTPI / w_large) * (1.0 - 0.5 * inv + 0.75 * inv ** 2 - 1.875 * inv ** 3)
    return jnp.where(w < 6.0, small, asymp)


def density_emg(
    x: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, tau: jnp.ndarray
) -> jnp.ndarray:
    """EMG density (unit area). mu = Gaussian centre, sigma > 0, tau > 0 (tail)."""
    u = (x - mu) / sigma
    lam = sigma / tau
    w = (lam - u) / _SQRT2
    core = (1.0 / (2.0 * tau)) * jnp.exp(-0.5 * u ** 2) * _erfcx_pos(jnp.maximum(w, 0.0))
    u_tail = jnp.maximum(u, lam)  # clamp inactive branch so exp can't overflow
    tail = (1.0 / (2.0 * tau)) * jnp.exp(0.5 * lam ** 2 - lam * u_tail) * erfc(jnp.minimum(w, 0.0))
    return jnp.where(w >= 0.0, core, tail)


def mode_emg(mu: float, sigma: float, tau: float) -> float:
    """Mode (apex) of EMG(mu, sigma, tau), numerically. Reporting only."""
    def neg(x: float) -> float:
        return -float(density_emg(jnp.asarray(x), jnp.asarray(mu),
                                  jnp.asarray(sigma), jnp.asarray(tau)))
    res = minimize_scalar(neg, bounds=(mu - 5 * sigma, mu + 20 * tau + 5 * sigma),
                          method="bounded")
    return float(res.x)


def fwhm_emg(mu: float, sigma: float, tau: float) -> float:
    """Full width at half maximum of EMG(mu, sigma, tau), numerically."""
    m = mode_emg(mu, sigma, tau)
    peak = float(density_emg(jnp.asarray(m), jnp.asarray(mu),
                             jnp.asarray(sigma), jnp.asarray(tau)))
    half = peak / 2.0

    def shifted(x: float) -> float:
        return float(density_emg(jnp.asarray(x), jnp.asarray(mu),
                                 jnp.asarray(sigma), jnp.asarray(tau))) - half

    lo, hi = m - sigma, m + tau + sigma
    while shifted(lo) > 0.0:
        lo -= sigma
    while shifted(hi) > 0.0:
        hi += tau + sigma
    x_left = float(brentq(shifted, lo, m))
    x_right = float(brentq(shifted, m, hi))
    return x_right - x_left
