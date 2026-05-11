"""Pure-math skew-normal layer.

Implements the centred-parameter (CP) ↔ direct-parameter (DP) bijection,
density evaluation in both forms, and the derived quantities (mode, FWHM,
HWHM-ratio, asymmetry-to-γ₁ inversion) needed by the priors and posterior
layers. No NumPyro imports, no state, no side effects.

See ``docs/superpowers/specs/2026-05-07-skew-normal-fitter-rewrite-design.md``
§2 (math) and §7.1 (API).
"""

from __future__ import annotations

import functools
import math

import jax.numpy as jnp
import jax.scipy.stats as jss
import numpy as np
from scipy.optimize import brentq

# Skewness of the half-normal distribution = max |γ₁| achievable by any
# skew-normal. See spec §2.2.
GAMMA1_MAX: float = ((4.0 - math.pi) / 2.0) * (math.sqrt(2.0 / math.pi) ** 3) / (1.0 - 2.0 / math.pi) ** 1.5

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


def mode_dp(xi: jnp.ndarray, omega: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    """Mode of SN(xi, omega, alpha) via Azzalini's m_0 approximation with Newton refinement.

    Initialises from Azzalini's m_0 approximation:
    m_0(alpha) = mu_z - gamma1_z * sigma_z / 2 - sign(alpha)/2 * exp(-2 * pi / |alpha|)
    then refines with 3 Newton steps on the mode equation
    -z + alpha * phi(alpha * z) / Phi(alpha * z) = 0  (z = (x - xi) / omega).
    The result is accurate to ~1e-8 relative to the true mode.
    Used for reporting only.

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

    # Newton refinement on the mode equation in z-space (z = (x - xi) / omega):
    # f(z) = -z + alpha * phi(alpha * z) / Phi(alpha * z) = 0
    # f'(z) = -1 + alpha^2 * (-alpha * z * r - r^2)  where r = phi(alpha*z)/Phi(alpha*z)
    z = m_0
    for _ in range(3):
        r = jss.norm.pdf(alpha * z) / jss.norm.cdf(alpha * z)
        dr_dz = alpha * (-alpha * z * r - r**2)
        f_val = -z + alpha * r
        f_prime = -1.0 + alpha * dr_dz
        z = z - f_val / f_prime

    return xi + omega * z


def _fwhm_scalar(xi: float, omega: float, alpha: float) -> float:
    """Scalar FWHM of SN(xi, omega, alpha) via two brentq solves."""
    mode = float(mode_dp(jnp.asarray(xi), jnp.asarray(omega), jnp.asarray(alpha)))
    peak = float(density_dp(jnp.asarray(mode), jnp.asarray(xi), jnp.asarray(omega), jnp.asarray(alpha)))
    half = peak / 2.0

    def shifted(x: float) -> float:
        return (
            float(density_dp(jnp.asarray(x), jnp.asarray(xi), jnp.asarray(omega), jnp.asarray(alpha))) - half
        )

    # Walk outward from mode by `omega` until the density drops below half-max.
    x_lo = mode - omega
    while shifted(x_lo) > 0.0:
        x_lo -= omega
    x_hi = mode + omega
    while shifted(x_hi) > 0.0:
        x_hi += omega
    x_left: float = float(brentq(shifted, x_lo, mode))  # type: ignore[arg-type]
    x_right: float = float(brentq(shifted, mode, x_hi))  # type: ignore[arg-type]
    return x_right - x_left


def _hwhm_ratio_scalar(alpha: float) -> float:
    """Scalar HWHM_R / HWHM_L of SN(0, 1, alpha). Independent of xi and omega."""
    mode = float(mode_dp(jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(alpha)))
    peak = float(density_dp(jnp.asarray(mode), jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(alpha)))
    half = peak / 2.0

    def shifted(x: float) -> float:
        return (
            float(density_dp(jnp.asarray(x), jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(alpha))) - half
        )

    x_lo = mode - 1.0
    while shifted(x_lo) > 0.0:
        x_lo -= 1.0
    x_hi = mode + 1.0
    while shifted(x_hi) > 0.0:
        x_hi += 1.0
    x_left = float(brentq(shifted, x_lo, mode))  # type: ignore[arg-type]
    x_right = float(brentq(shifted, mode, x_hi))  # type: ignore[arg-type]
    return (x_right - mode) / (mode - x_left)


def hwhm_ratio_dp(
    xi: float | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    omega: float | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    alpha: float | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Right-to-left HWHM ratio of SN(xi, omega, alpha). Independent of xi and omega.

    Used at fit time to invert measured peak asymmetry to gamma1 via
    :func:`sn_asymmetry_to_gamma1`. The (xi, omega) arguments are accepted
    for signature symmetry with the rest of the DP API but are ignored.

    Args:
        xi: Ignored (kept for API symmetry).
        omega: Ignored (kept for API symmetry).
        alpha: DP slant, scalar or array.

    Returns:
        HWHM_R / HWHM_L as a numpy array, broadcast shape of ``alpha``.
    """
    del xi, omega  # ratio is invariant under (xi, omega); kept in signature for symmetry.
    return np.vectorize(_hwhm_ratio_scalar, otypes=[float])(  # type: ignore[return-value]
        np.asarray(alpha, dtype=float)
    )


def fwhm_dp(
    xi: float | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    omega: float | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    alpha: float | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Full width at half maximum of SN(xi, omega, alpha), computed numerically.

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
    return np.vectorize(_fwhm_scalar, otypes=[float])(  # type: ignore[return-value]
        np.asarray(xi, dtype=float),
        np.asarray(omega, dtype=float),
        np.asarray(alpha, dtype=float),
    )


@functools.lru_cache(maxsize=1)
def _asymmetry_table() -> (  # type: ignore[return]
    tuple[
        np.ndarray[tuple[int, ...], np.dtype[np.float64]],
        np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    ]
):
    """Build the (ratio -> gamma1) inversion table once.

    Sweeps alpha over a wide grid, computes HWHM ratio via :func:`_hwhm_ratio_scalar`
    and gamma1 via :func:`dp_to_cp`, then sorts by ratio so the result is
    monotone-increasing in ratio. Cached on the first call.
    """
    alphas = np.linspace(-50.0, 50.0, 401)
    ratios = np.vectorize(_hwhm_ratio_scalar, otypes=[float])(alphas)
    _, _, gamma1s = dp_to_cp(jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(alphas))
    gamma1s_np = np.asarray(gamma1s)
    order = np.argsort(ratios)
    return ratios[order], gamma1s_np[order]


def sn_asymmetry_to_gamma1(ratio: jnp.ndarray) -> jnp.ndarray:
    """Invert measured HWHM_R/HWHM_L ratio to gamma1 via a precomputed table.

    Used at prior-build time only — once per fit. The table is built on
    first call and cached for the process lifetime.

    Args:
        ratio: Measured HWHM_R / HWHM_L. Scalar or array. For symmetric
            peaks ratio approx 1 -> gamma1 approx 0; ratio>1 -> gamma1>0;
            ratio<1 -> gamma1<0.

    Returns:
        Interpolated gamma1 values with the shape of ``ratio``.
    """
    ratios_grid, gamma1_grid = _asymmetry_table()
    return jnp.interp(jnp.asarray(ratio), jnp.asarray(ratios_grid), jnp.asarray(gamma1_grid))
