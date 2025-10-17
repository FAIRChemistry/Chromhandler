import jax.numpy as jnp
from jax import vmap
from jax.scipy import special as jsp  # gives erf, erfc, erfcx, ndtr
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


if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt

    x = jnp.linspace(4.7, 5.5, 1000)
    h = jnp.array([60.0, 110.0, 80.0])  # heights h_k
    mu = jnp.array([5.0, 5.05, 5.15])
    sigma = jnp.array([0.05, 0.03, 0.03])
    tau = jnp.array([0.01, 0.01, 0.01])

    y_hat, comps = emg_mixture(x, h, mu, sigma, tau)  # comps: (K, N)

    # Convert to NumPy for plotting (optional but avoids dtype surprises)
    x_np = np.asarray(x)
    y_np = np.asarray(y_hat)
    comps_np = np.asarray(comps)  # (K, N)

    plt.scatter(x_np, y_np, label="observed", s=1, color="tab:gray")
    plt.plot(x_np, comps_np.T, alpha=0.7, label="component")
    plt.legend()
    plt.savefig("emg_mixture.png")
