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
import jax.scipy.stats as jss

# Skewness of the half-normal distribution = max |γ₁| achievable by any
# skew-normal. See spec §2.2.
GAMMA1_MAX: float = (
    ((4.0 - math.pi) / 2.0)
    * (math.sqrt(2.0 / math.pi) ** 3)
    / (1.0 - 2.0 / math.pi) ** 1.5
)

_B_CONST: float = math.sqrt(2.0 / math.pi)


def cp_to_dp(
    mu: jnp.ndarray, sigma: jnp.ndarray, gamma1: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert centred parameters (mu, sigma, gamma1) to direct parameters (xi, omega, alpha).

    Closed form via the Azzalini relations (spec §2.3).

    Args:
        mu: Mean of the skew-normal. Any broadcastable shape.
        sigma: Standard deviation. Strictly positive.
        gamma1: Skewness coefficient. Must satisfy |gamma1| < GAMMA1_MAX;
            values outside the open interval are not in the SN family.

    Returns:
        Tuple (xi, omega, alpha) of DP parameters, broadcast to the
        common shape of the inputs.
    """
    c = jnp.cbrt(2.0 * gamma1 / (4.0 - jnp.pi))
    b_delta = c / jnp.sqrt(1.0 + c**2)
    delta = b_delta / _B_CONST
    omega = sigma / jnp.sqrt(1.0 - b_delta**2)
    alpha = delta / jnp.sqrt(1.0 - delta**2)
    xi = mu - omega * b_delta
    return xi, omega, alpha


def dp_to_cp(
    xi: jnp.ndarray, omega: jnp.ndarray, alpha: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert direct parameters (xi, omega, alpha) to centred parameters (mu, sigma, gamma1).

    Forward Azzalini formulas (spec §2.3). Inverse of :func:`cp_to_dp`.

    Args:
        xi: DP location.
        omega: DP scale, strictly positive.
        alpha: DP slant. Any real value.

    Returns:
        Tuple (mu, sigma, gamma1) of CP parameters, broadcast to the
        common shape of the inputs.
    """
    delta = alpha / jnp.sqrt(1.0 + alpha**2)
    b_delta = _B_CONST * delta
    mu = xi + omega * b_delta
    sigma = omega * jnp.sqrt(1.0 - b_delta**2)
    gamma1 = ((4.0 - jnp.pi) / 2.0) * b_delta**3 / (1.0 - b_delta**2) ** 1.5
    return mu, sigma, gamma1


def density_dp(
    x: jnp.ndarray,
    xi: jnp.ndarray,
    omega: jnp.ndarray,
    alpha: jnp.ndarray,
) -> jnp.ndarray:
    """Skew-normal density in DP form.

    f(x) = (2/omega) * phi((x - xi)/omega) * Phi(alpha * (x - xi)/omega).

    Args:
        x: Evaluation points. Any broadcastable shape.
        xi: DP location, broadcastable with x.
        omega: DP scale, strictly positive, broadcastable with x.
        alpha: DP slant, broadcastable with x.

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

    Internally converts (mu, sigma, gamma1) -> (xi, omega, alpha) via
    :func:`cp_to_dp` then delegates to :func:`density_dp`.

    Args:
        x: Evaluation points.
        mu: Mean.
        sigma: Standard deviation, strictly positive.
        gamma1: Skewness coefficient, |gamma1| < GAMMA1_MAX.

    Returns:
        Density values.
    """
    xi, omega, alpha = cp_to_dp(mu, sigma, gamma1)
    return density_dp(x, xi, omega, alpha)
