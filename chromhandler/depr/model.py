import jax.numpy as jnp
from jax import vmap
from jax.scipy import special as jsp  # gives erf, erfc, erfcx, ndtr
from pathlib import Path
from rich.console import Console

console = Console()


def emg(
    x: jnp.ndarray,
    h: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    tau: jnp.ndarray,
    eps: float = 1e-12,
) -> jnp.ndarray:
    x = jnp.asarray(x)
    sigma = jnp.clip(jnp.abs(sigma), eps, None)
    tau = jnp.where(jnp.abs(tau) < eps, eps, tau)

    z = (x - mu) / sigma
    a = (sigma / (jnp.sqrt(2.0) * tau)) - z / jnp.sqrt(2.0)
    # Textbook EMG with erfcx stabilization:
    # y = (h/(2*tau)) * exp(sigma^2/(2 tau^2) - (x-mu)/tau - a^2) * erfcx(a)
    return (
        (h / (2.0 * tau))
        * jnp.exp((sigma**2) / (2.0 * tau**2) - (x - mu) / tau - a * a)
        * jsp.erfc(a)
    )


def emg_mixture(
    x: jnp.ndarray,
    h: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    tau: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Mixture of K EMG peaks (chromatography form).

    Uses direct physical parameters - no transformations.
    A is the amplitude (height) parameter h in the chromatography formulation.

    Returns
    -------
    y_hat : jnp.ndarray
        Sum of all components
    components : jnp.ndarray, shape (K, N)
        Individual peak contributions
    """
    components = vmap(lambda h, m, s, t: emg(x, h, m, s, t))(h, mu, sigma, tau)
    y_hat = components.sum(axis=0)
    return y_hat, components
