import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.scipy.special import erfc


def emg(
    x: jnp.ndarray,
    A: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    tau: jnp.ndarray,
    eps: float = 1e-12,
) -> jnp.ndarray:
    s = jnp.maximum(sigma, eps)
    t = jnp.maximum(tau, eps)  # right-tailed only for simplicity
    z = x - mu
    u = (s**2 - t * z) / (jnp.sqrt(2.0) * s * t)
    # stable EMG form
    return A * 0.5 * jnp.exp(-z / t) * erfc(u)


x = jnp.linspace(0, 20, 20000)
A, mu, sigma, tau = 500, 10, 1, 0.1  # safe, visible peak
y = emg(x, A, mu, sigma, tau)

print("max|y| =", float(jnp.max(jnp.abs(y))))
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("EMG")
plt.tight_layout()
plt.show()
