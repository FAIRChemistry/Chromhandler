"""Regression tests for the skew-normal boundary guard (float32 regime).

The CP->DP map ``cp_to_dp`` becomes singular as ``|gamma1|`` approaches and
exceeds ``GAMMA1_MAX``: ``delta -> 1`` so ``alpha = delta / sqrt(1 - delta**2)``
diverges, and for ``|gamma1| > GAMMA1_MAX`` (``delta > 1``) the ``sqrt`` of a
negative number yields NaN, so ``density_cp`` is non-finite. The model bounds
skew with a tanh bijector; in float32 a plain ``GAMMA1_MAX * tanh(...)``
saturates to exactly GAMMA1_MAX for large draws and can overshoot via
roundoff. ``SKEW_EFF_MAX`` (= GAMMA1_MAX - 1e-3) keeps the squashed skew
strictly inside the singular point. Tests run in JAX's default float32 —
the precision the fit actually samples in.
"""

import jax
import jax.numpy as jnp
import numpy as np

from chromhandler.fitting.model import SKEW_EFF_MAX
from chromhandler.fitting.skew_normal import GAMMA1_MAX, density_cp

_X = jnp.linspace(-3.0, 3.0, 51, dtype=jnp.float32)
_F32 = jnp.float32


def _density_at(skew_value: float) -> np.ndarray:
    return np.asarray(
        density_cp(
            _X,
            jnp.asarray(0.0, _F32),
            jnp.asarray(1.0, _F32),
            jnp.asarray(skew_value, _F32),
        )
    )


def test_density_blows_up_above_bound():
    """Sanity: the hazard is real — skew just ABOVE GAMMA1_MAX (delta > 1)
    drives density_cp non-finite. This is the overshoot the guard prevents."""
    dens = _density_at(float(GAMMA1_MAX) + 1e-3)
    assert not np.all(np.isfinite(dens))


def test_density_finite_at_effective_bound():
    """The guarded bound keeps density_cp finite in float32."""
    dens = _density_at(SKEW_EFF_MAX)
    assert np.all(np.isfinite(dens))
    assert dens.max() > 0.0


def test_tanh_bijector_never_reaches_gamma1_max_in_float32():
    """An extreme draw saturates tanh to 1.0 in float32, but the squashed
    skew stays <= SKEW_EFF_MAX < GAMMA1_MAX, so density_cp stays finite."""
    eff = jnp.asarray(SKEW_EFF_MAX, _F32)
    skew = eff * jnp.tanh(jnp.asarray(1e3, _F32) / eff)  # skew_unconstrained huge
    assert float(skew) < GAMMA1_MAX
    assert np.all(np.isfinite(_density_at(float(skew))))


def test_density_gradient_finite_at_effective_bound():
    """HMC differentiates the density; the gradient w.r.t. skew must be
    finite at the bound (the singularity would otherwise give NaN grads)."""
    def peak_sum(skew):
        return jnp.sum(density_cp(_X, jnp.asarray(0.0, _F32),
                                  jnp.asarray(1.0, _F32), skew))

    g = jax.grad(peak_sum)(jnp.asarray(SKEW_EFF_MAX, _F32))
    assert np.isfinite(float(g))
