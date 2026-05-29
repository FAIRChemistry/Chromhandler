"""EMG math: density correctness vs scipy, float32 stability, Gaussian limit."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import exponnorm

from chromhandler.fitting.emg import density_emg

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("sigma,tau", [(1.0, 0.5), (1.0, 2.0), (1.0, 5.0), (0.03, 0.05)])
def test_density_matches_exponnorm(sigma, tau):
    mu = 5.0
    x = np.linspace(mu - 4 * sigma, mu + 12 * tau, 400)
    ref = exponnorm.pdf(x, tau / sigma, loc=mu, scale=sigma)
    mine = np.asarray(density_emg(jnp.asarray(x), jnp.asarray(mu),
                                  jnp.asarray(sigma), jnp.asarray(tau)))
    assert np.max(np.abs(mine - ref)) / np.max(ref) < 1e-5


def test_density_integrates_to_one():
    mu, sigma, tau = 5.0, 0.05, 0.1
    x = np.linspace(mu - 1.0, mu + 3.0, 20001)
    d = np.asarray(density_emg(jnp.asarray(x), jnp.asarray(mu),
                               jnp.asarray(sigma), jnp.asarray(tau)))
    assert abs(np.trapezoid(d, x) - 1.0) < 1e-4


def test_gaussian_limit_small_tau():
    mu, sigma, tau = 0.0, 1.0, 1e-4
    x = np.linspace(-5, 5, 401)
    emg = np.asarray(density_emg(jnp.asarray(x), jnp.asarray(mu),
                                 jnp.asarray(sigma), jnp.asarray(tau)))
    gauss = np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    assert np.max(np.abs(emg - gauss)) < 1e-2


def test_float32_far_tail_and_gradient_finite():
    x = jnp.linspace(2.0, 30.0, 3000, dtype=jnp.float32)
    d = density_emg(x, jnp.float32(5.0), jnp.float32(0.05), jnp.float32(0.1))
    assert bool(np.all(np.isfinite(np.asarray(d))))
    g = jax.grad(lambda tau: jnp.sum(
        density_emg(jnp.linspace(4.5, 8.0, 200), 5.0, 0.1, tau)))(jnp.asarray(0.3))
    assert np.isfinite(float(g))


def test_mode_emg_matches_grid_argmax():
    from chromhandler.fitting.emg import mode_emg
    mu, sigma, tau = 5.0, 0.05, 0.1
    xs = np.linspace(mu - 0.5, mu + 1.0, 400001)
    d = np.asarray(density_emg(jnp.asarray(xs), jnp.asarray(mu),
                               jnp.asarray(sigma), jnp.asarray(tau)))
    grid_mode = xs[int(np.argmax(d))]
    assert abs(float(mode_emg(mu, sigma, tau)) - grid_mode) < 1e-3


def test_fwhm_emg_matches_grid():
    from chromhandler.fitting.emg import fwhm_emg
    mu, sigma, tau = 5.0, 0.05, 0.1
    xs = np.linspace(mu - 1.0, mu + 3.0, 2000001)
    d = np.asarray(density_emg(jnp.asarray(xs), jnp.asarray(mu),
                               jnp.asarray(sigma), jnp.asarray(tau)))
    peak = d.max()
    above = xs[d >= peak / 2]
    assert abs(float(fwhm_emg(mu, sigma, tau)) - (above.max() - above.min())) < 5e-3


@pytest.mark.parametrize("sigma,tau", [(0.04, 0.02), (0.04, 0.08), (0.04, 0.2)])
def test_emg_from_peak_features_roundtrip(sigma, tau):
    from chromhandler.fitting.emg import (
        emg_from_peak_features,
        fwhm_emg,
        hwhm_ratio_emg,
        mode_emg,
    )
    mu_true = 5.0
    apex = mode_emg(mu_true, sigma, tau)
    fwhm = fwhm_emg(mu_true, sigma, tau)
    ratio = hwhm_ratio_emg(sigma, tau)
    mu, s, t = emg_from_peak_features(apex, fwhm, ratio)
    assert abs(mu - mu_true) < 5e-3
    assert abs(s - sigma) / sigma < 0.1
    assert abs(t - tau) / tau < 0.1
