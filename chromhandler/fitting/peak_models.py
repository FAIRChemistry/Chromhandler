"""Skew-normal peak models for chromatographic fitting.

This module exposes a single mixed NumPyro model named ``model`` that supports
both single and double skew-normal peaks in the same fit.

Model behavior is configured with component/logical-peak index metadata that is
assembled by the fitter from user-facing peak definitions.
"""

from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import log_ndtr


def _ensure_two_dimensional(values: jnp.ndarray, name: str) -> jnp.ndarray:
    """Return an array with shape ``[num_spectra, num_points]``.

    Args:
        values: Input array with shape ``[num_points]`` or
            ``[num_spectra, num_points]``.
        name: Human-readable variable name for error messages.

    Returns:
        Two-dimensional float32 array.

    Raises:
        ValueError: If the input dimensionality is not 1D or 2D.
    """
    matrix = jnp.asarray(values, dtype=jnp.float32)
    if matrix.ndim == 1:
        return matrix[None, :]
    if matrix.ndim == 2:
        return matrix
    raise ValueError(f"`{name}` must be 1D or 2D, got shape {matrix.shape}.")


def _to_component_matrix(
    values: jnp.ndarray,
    num_spectra: int,
    num_components: int,
    name: str,
) -> jnp.ndarray:
    """Broadcast or validate a component array to ``[num_spectra, num_components]``.

    Args:
        values: Array with shape ``[num_components]`` or
            ``[num_spectra, num_components]``.
        num_spectra: Number of spectra.
        num_components: Number of model components.
        name: Human-readable variable name for error messages.

    Returns:
        Two-dimensional float32 array.

    Raises:
        ValueError: If the array shape is incompatible.
    """
    array = jnp.asarray(values, dtype=jnp.float32)
    if array.ndim == 1:
        if array.shape[0] != num_components:
            raise ValueError(
                f"`{name}` has shape {array.shape}; expected [{num_components}]."
            )
        return jnp.broadcast_to(array[None, :], (num_spectra, num_components))

    if array.ndim == 2:
        if array.shape == (num_spectra, num_components):
            return array
        if array.shape == (1, num_components):
            return jnp.broadcast_to(array, (num_spectra, num_components))
        raise ValueError(
            f"`{name}` has shape {array.shape}; expected "
            f"[{num_spectra}, {num_components}] or [1, {num_components}]."
        )

    raise ValueError(f"`{name}` must be 1D or 2D, got shape {array.shape}.")


def _to_mask_matrix(
    mask_values: jnp.ndarray,
    num_spectra: int,
    num_points: int,
    name: str,
) -> jnp.ndarray:
    """Broadcast or validate a boolean mask to ``[num_spectra, num_points]``."""
    mask = jnp.asarray(mask_values)
    if mask.ndim == 1:
        if mask.shape[0] != num_points:
            raise ValueError(
                f"`{name}` has shape {mask.shape}; expected [{num_points}]."
            )
        return jnp.broadcast_to(mask[None, :], (num_spectra, num_points)).astype(bool)

    if mask.ndim == 2:
        if mask.shape == (num_spectra, num_points):
            return mask.astype(bool)
        if mask.shape == (1, num_points):
            return jnp.broadcast_to(mask, (num_spectra, num_points)).astype(bool)
        raise ValueError(
            f"`{name}` has shape {mask.shape}; expected "
            f"[{num_spectra}, {num_points}] or [1, {num_points}]."
        )

    raise ValueError(f"`{name}` must be 1D or 2D, got shape {mask.shape}.")


def _build_peak_window_mask(
    x: jnp.ndarray,
    mu_lo: jnp.ndarray,
    mu_hi: jnp.ndarray,
) -> jnp.ndarray:
    """Return a mask for points that fall into at least one component window."""
    return jnp.any(
        (x[..., None, :] >= mu_lo[..., :, None])
        & (x[..., None, :] <= mu_hi[..., :, None]),
        axis=-2,
    )


def _masked_standard_deviation(
    values: jnp.ndarray,
    mask: jnp.ndarray,
    fallback: jnp.ndarray,
) -> jnp.ndarray:
    """Compute per-spectrum standard deviation on a masked matrix.

    Args:
        values: Residual values with shape ``[num_spectra, num_points]``.
        mask: Boolean mask with the same shape.
        fallback: Per-spectrum fallback scale, shape ``[num_spectra]``.

    Returns:
        Per-spectrum standard deviation estimate.
    """
    valid_count = jnp.sum(mask, axis=-1)
    safe_count = jnp.maximum(valid_count, jnp.array(1.0, dtype=jnp.float32))

    masked_values = jnp.where(mask, values, 0.0)
    mean_value = jnp.sum(masked_values, axis=-1) / safe_count
    centered = values - mean_value[:, None]
    variance = jnp.sum(jnp.where(mask, centered**2, 0.0), axis=-1) / safe_count
    standard_deviation = jnp.sqrt(
        jnp.maximum(variance, jnp.array(1e-12, dtype=jnp.float32))
    )

    return jnp.where(valid_count > 2, standard_deviation, fallback)


def log_skew_normal_pdf(
    x: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    alpha: jnp.ndarray,
) -> jnp.ndarray:
    """Compute numerically stable skew-normal log-density values.

    The skew-normal density is ``f(x) = 2/sigma * phi(z) * Phi(alpha * z)`` with
    ``z = (x - mu) / sigma``.

    Args:
        x: Time values with shape ``[..., num_points]``.
        mu: Peak centers with shape ``[..., num_components]``.
        sigma: Peak widths with shape ``[..., num_components]``.
        alpha: Skew parameters with shape ``[..., num_components]``.

    Returns:
        Log-density matrix with shape ``[..., num_components, num_points]``.
    """
    sigma_safe = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float32), 1e-6)
    x_array = jnp.asarray(x, dtype=jnp.float32)
    mu_array = jnp.asarray(mu, dtype=jnp.float32)
    alpha_array = jnp.asarray(alpha, dtype=jnp.float32)

    z_value = (x_array[..., None, :] - mu_array[..., :, None]) / sigma_safe[
        ..., :, None
    ]
    log_standard_normal = -0.5 * z_value**2 - 0.5 * jnp.log(2.0 * jnp.pi)
    return (
        jnp.log(2.0)
        - jnp.log(sigma_safe)[..., :, None]
        + log_standard_normal
        + log_ndtr(alpha_array[..., :, None] * z_value)
    )


def skew_normal_pdf(
    x: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    alpha: jnp.ndarray,
) -> jnp.ndarray:
    """Compute skew-normal density values."""
    return jnp.exp(log_skew_normal_pdf(x, mu, sigma, alpha))


def skew_mixture_area(
    x: jnp.ndarray,
    A: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    alpha: jnp.ndarray,
) -> jnp.ndarray:
    """Compute area-scaled skew-normal mixture signal."""
    probability_density = skew_normal_pdf(x, mu, sigma, alpha)
    return jnp.sum(probability_density * A[..., :, None], axis=-2)


def skew_components_area(
    x: jnp.ndarray,
    A: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    alpha: jnp.ndarray,
) -> jnp.ndarray:
    """Return area-scaled component curves for each skew-normal peak."""
    return skew_normal_pdf(x, mu, sigma, alpha) * A[..., :, None]


def model(
    x: jnp.ndarray,
    y: Optional[jnp.ndarray],
    mu_lo: jnp.ndarray,
    mu_hi: jnp.ndarray,
    sigma_min: float,
    sigma_max: float,
    logical_mu_lo: Sequence[float],
    logical_mu_hi: Sequence[float],
    logical_main_component_index: Sequence[int],
    logical_shoulder_component_index: Sequence[int],
    logical_shoulder_side: Sequence[int],
    component_to_logical_index: Sequence[int],
    component_include_in_total_area: Sequence[bool],
    mu_init: Optional[jnp.ndarray] = None,
    sigma_init: Optional[jnp.ndarray] = None,
    A_init: Optional[jnp.ndarray] = None,
    peak_mask: Optional[jnp.ndarray] = None,
    alpha_prior_sd: float = 1.0,
) -> None:
    """Mixed skew-normal model with single and double peaks in one graph.

    This function is the only model entry point used by the fitter.

    Args:
        x: Time values with shape ``[num_points]`` or ``[num_spectra, num_points]``.
        y: Observed values with shape matching ``x``. Use ``None`` for prior
            predictive sampling.
        mu_lo: Lower component bounds with shape ``[num_components]`` or
            ``[num_spectra, num_components]``.
        mu_hi: Upper component bounds with shape ``[num_components]`` or
            ``[num_spectra, num_components]``.
        sigma_min: Global lower bound for component width.
        sigma_max: Global upper bound for component width.
        logical_mu_lo: Lower bounds for logical peaks.
        logical_mu_hi: Upper bounds for logical peaks.
        logical_main_component_index: Main component index per logical peak.
        logical_shoulder_component_index: Shoulder component index per logical peak,
            ``-1`` for single-peak logical entries.
        logical_shoulder_side: Shoulder side code per logical peak:
            ``-1`` for left, ``+1`` for right, ``0`` for no shoulder.
        component_to_logical_index: Logical-peak index per model component.
        component_include_in_total_area: Whether each component contributes to
            chemical total area reporting.
        mu_init: Optional center initialization matrix.
        sigma_init: Optional width initialization matrix.
        A_init: Optional area initialization matrix.
        peak_mask: Optional explicit likelihood mask for peak points.
        alpha_prior_sd: Standard deviation of ``alpha ~ Normal(0, alpha_prior_sd)``.

    Raises:
        ValueError: If metadata dimensions are inconsistent.
    """
    x_values = _ensure_two_dimensional(x, "x")
    y_observed = None if y is None else _ensure_two_dimensional(y, "y")
    num_spectra, num_points = x_values.shape

    mu_lo_array = jnp.asarray(mu_lo, dtype=jnp.float32)
    mu_hi_array = jnp.asarray(mu_hi, dtype=jnp.float32)
    if mu_lo_array.ndim == 1 and mu_hi_array.ndim == 1:
        num_components = int(mu_lo_array.shape[0])
        mu_lo_matrix = jnp.broadcast_to(
            mu_lo_array[None, :], (num_spectra, num_components)
        )
        mu_hi_matrix = jnp.broadcast_to(
            mu_hi_array[None, :], (num_spectra, num_components)
        )
    elif mu_lo_array.ndim == 2 and mu_hi_array.ndim == 2:
        num_components = int(mu_lo_array.shape[1])
        mu_lo_matrix = mu_lo_array
        mu_hi_matrix = mu_hi_array
    else:
        raise ValueError("`mu_lo` and `mu_hi` must both be 1D or both be 2D.")

    main_index_array = jnp.asarray(
        logical_main_component_index, dtype=jnp.int32
    ).reshape(-1)
    shoulder_index_array = jnp.asarray(
        logical_shoulder_component_index, dtype=jnp.int32
    ).reshape(-1)
    shoulder_side_array = jnp.asarray(logical_shoulder_side, dtype=jnp.float32).reshape(
        -1
    )
    logical_low_array = jnp.asarray(logical_mu_lo, dtype=jnp.float32).reshape(-1)
    logical_high_array = jnp.asarray(logical_mu_hi, dtype=jnp.float32).reshape(-1)

    logical_count = int(main_index_array.shape[0])
    if (
        shoulder_index_array.shape[0] != logical_count
        or shoulder_side_array.shape[0] != logical_count
        or logical_low_array.shape[0] != logical_count
        or logical_high_array.shape[0] != logical_count
    ):
        raise ValueError("Logical-peak metadata arrays must have identical lengths.")

    component_to_logical_array = jnp.asarray(
        component_to_logical_index, dtype=jnp.int32
    ).reshape(-1)
    component_include_array = jnp.asarray(
        component_include_in_total_area, dtype=jnp.float32
    ).reshape(-1)
    if component_to_logical_array.shape[0] != num_components:
        raise ValueError(
            "`component_to_logical_index` length does not match the number of components."
        )
    if component_include_array.shape[0] != num_components:
        raise ValueError(
            "`component_include_in_total_area` length does not match the number of components."
        )

    default_peak_mask = _build_peak_window_mask(x_values, mu_lo_matrix, mu_hi_matrix)
    if y_observed is None:
        finite_mask = jnp.ones((num_spectra, num_points), dtype=bool)
        y_finite = None
    else:
        finite_mask = jnp.isfinite(y_observed)
        y_finite = jnp.where(finite_mask, y_observed, 0.0)

    if peak_mask is None:
        peak_observation_mask = default_peak_mask & finite_mask
    else:
        peak_observation_mask = (
            _to_mask_matrix(peak_mask, num_spectra, num_points, "peak_mask")
            & finite_mask
        )

    mu_initial_matrix = (
        0.5 * (mu_lo_matrix + mu_hi_matrix)
        if mu_init is None
        else _to_component_matrix(mu_init, num_spectra, num_components, "mu_init")
    )

    sigma_initial_matrix = (
        jnp.broadcast_to(
            jnp.maximum(
                jnp.mean(mu_hi_matrix - mu_lo_matrix) / 6.0,
                jnp.array(1e-4, dtype=jnp.float32),
            ),
            (num_spectra, num_components),
        )
        if sigma_init is None
        else _to_component_matrix(sigma_init, num_spectra, num_components, "sigma_init")
    )

    if A_init is not None:
        area_initial_matrix = _to_component_matrix(
            A_init, num_spectra, num_components, "A_init"
        )
    elif y_observed is not None:
        y_peak_only = jnp.where(peak_observation_mask, y_observed, -jnp.inf)
        y_peak_maximum = jnp.max(y_peak_only, axis=-1)
        y_full_maximum = jnp.max(y_finite, axis=-1)
        y_scale = jnp.maximum(
            jnp.where(jnp.isfinite(y_peak_maximum), y_peak_maximum, y_full_maximum),
            jnp.array(1.0, dtype=jnp.float32),
        )
        area_initial_matrix = jnp.broadcast_to(
            jnp.maximum(0.95 * y_scale[:, None], jnp.array(1e-6, dtype=jnp.float32)),
            (num_spectra, num_components),
        )
    else:
        area_initial_matrix = jnp.ones((num_spectra, num_components), dtype=jnp.float32)

    sigma_minimum = jnp.full(
        (num_components,), jnp.asarray(sigma_min, dtype=jnp.float32)
    )
    sigma_maximum = jnp.full(
        (num_components,), jnp.asarray(sigma_max, dtype=jnp.float32)
    )
    sigma_range = jnp.maximum(
        sigma_maximum - sigma_minimum,
        jnp.array(1e-4, dtype=jnp.float32),
    )

    sigma_location = jnp.clip(
        jnp.median(sigma_initial_matrix, axis=0),
        sigma_minimum + 1e-4,
        sigma_maximum - 1e-4,
    )
    sigma_mad = jnp.median(
        jnp.abs(sigma_initial_matrix - sigma_location[None, :]),
        axis=0,
    )
    sigma_robust_standard_deviation = 1.4826 * sigma_mad
    sigma_scale = jnp.clip(
        sigma_robust_standard_deviation + 0.02 * sigma_range,
        jnp.array(1e-4, dtype=jnp.float32),
        0.25 * sigma_range,
    )

    sigma = numpyro.sample(
        "sigma",
        dist.TruncatedNormal(
            loc=jnp.broadcast_to(
                sigma_location[None, :], (num_spectra, num_components)
            ),
            scale=jnp.broadcast_to(sigma_scale[None, :], (num_spectra, num_components)),
            low=jnp.broadcast_to(sigma_minimum[None, :], (num_spectra, num_components)),
            high=jnp.broadcast_to(
                sigma_maximum[None, :], (num_spectra, num_components)
            ),
        ),
    )

    alpha_scale = jnp.full(
        (num_spectra, num_components),
        jnp.maximum(float(alpha_prior_sd), 1e-3),
        dtype=jnp.float32,
    )
    alpha = numpyro.sample(
        "alpha",
        dist.Normal(
            loc=jnp.zeros((num_spectra, num_components), dtype=jnp.float32),
            scale=alpha_scale,
        ),
    )

    has_shoulder = shoulder_index_array >= 0
    has_shoulder_float = has_shoulder.astype(jnp.float32)
    shoulder_index_safe = jnp.where(has_shoulder, shoulder_index_array, 0)
    shoulder_side_effective = shoulder_side_array * has_shoulder_float

    logical_span = jnp.maximum(
        logical_high_array - logical_low_array,
        jnp.array(1e-4, dtype=jnp.float32),
    )
    logical_low_matrix = jnp.broadcast_to(
        logical_low_array[None, :], (num_spectra, logical_count)
    )
    logical_high_matrix = jnp.broadcast_to(
        logical_high_array[None, :], (num_spectra, logical_count)
    )

    mu_main_initial = mu_initial_matrix[:, main_index_array]
    mu_shoulder_initial = mu_initial_matrix[:, shoulder_index_safe]
    mu_center_initial = mu_main_initial + 0.5 * has_shoulder_float[None, :] * (
        mu_shoulder_initial - mu_main_initial
    )

    mu_center_median = jnp.median(mu_center_initial, axis=0)
    mu_center_mad = jnp.median(
        jnp.abs(mu_center_initial - mu_center_median[None, :]),
        axis=0,
    )
    mu_center_robust_standard_deviation = 1.4826 * mu_center_mad
    mu_center_prior_scale = jnp.clip(
        mu_center_robust_standard_deviation + 0.01 * logical_span,
        jnp.array(1e-4, dtype=jnp.float32),
        0.05 * logical_span,
    )
    mu_center = numpyro.sample(
        "mu_center",
        dist.TruncatedNormal(
            loc=mu_center_initial,
            scale=jnp.broadcast_to(
                mu_center_prior_scale[None, :], (num_spectra, logical_count)
            ),
            low=logical_low_matrix,
            high=logical_high_matrix,
        ),
    )

    separation_initial = (
        jnp.abs(mu_main_initial - mu_shoulder_initial) * has_shoulder_float[None, :]
    )
    separation_location = jnp.maximum(
        jnp.mean(separation_initial, axis=0),
        0.05 * logical_span * has_shoulder_float + 1e-4 * (1.0 - has_shoulder_float),
    )
    separation_scale = jnp.maximum(
        0.35 * separation_location,
        0.005 * logical_span * has_shoulder_float + 1e-4 * (1.0 - has_shoulder_float),
    )
    separation_minimum = 0.005 * logical_span * has_shoulder_float + 1e-5 * (
        1.0 - has_shoulder_float
    )
    separation = numpyro.sample(
        "separation",
        dist.TruncatedNormal(
            loc=separation_location,
            scale=separation_scale,
            low=separation_minimum,
        ),
    )

    separation_matrix = jnp.broadcast_to(
        separation[None, :], (num_spectra, logical_count)
    )
    maximum_separation = 2.0 * jnp.minimum(
        mu_center - logical_low_matrix,
        logical_high_matrix - mu_center,
    )
    separation_matrix = jnp.minimum(
        separation_matrix,
        jnp.maximum(maximum_separation, jnp.array(1e-6, dtype=jnp.float32)),
    )

    mu_shoulder_component = (
        mu_center + 0.5 * shoulder_side_effective[None, :] * separation_matrix
    )
    mu_main_component = (
        mu_center - 0.5 * shoulder_side_effective[None, :] * separation_matrix
    )

    mu_values = jnp.zeros((num_spectra, num_components), dtype=jnp.float32)
    mu_values = mu_values.at[:, main_index_array].set(mu_main_component)
    mu_values = mu_values.at[:, shoulder_index_safe].add(
        mu_shoulder_component * has_shoulder_float[None, :]
    )

    mu = numpyro.deterministic("mu", mu_values)

    area_location = jnp.log(
        jnp.maximum(area_initial_matrix, jnp.array(1e-8, dtype=jnp.float32))
    )
    area_scale = jnp.full((num_spectra, num_components), 0.6, dtype=jnp.float32)
    A = numpyro.sample("A", dist.LogNormal(area_location, area_scale))

    numpyro.deterministic("A_total", jnp.sum(A, axis=-1))

    A_total_fit_logical = jnp.zeros((num_spectra, logical_count), dtype=jnp.float32)
    A_total_fit_logical = A_total_fit_logical.at[:, component_to_logical_array].add(A)
    numpyro.deterministic("A_total_fit_logical", A_total_fit_logical)

    A_total_chemical_logical = jnp.zeros(
        (num_spectra, logical_count), dtype=jnp.float32
    )
    A_total_chemical_logical = A_total_chemical_logical.at[
        :, component_to_logical_array
    ].add(A * component_include_array[None, :])
    numpyro.deterministic("A_total_chemical_logical", A_total_chemical_logical)

    peak_signal = skew_mixture_area(x_values, A, mu, sigma, alpha)
    numpyro.deterministic("mu_y", peak_signal)

    if y_observed is None:
        sigma_reference = jnp.maximum(
            jnp.mean(sigma_initial_matrix, axis=-1),
            jnp.array(1e-4, dtype=jnp.float32),
        )
        area_reference = jnp.maximum(
            jnp.max(area_initial_matrix, axis=-1),
            jnp.array(1.0, dtype=jnp.float32),
        )
        signal_scale = jnp.maximum(
            area_reference / (2.5 * sigma_reference), jnp.array(1.0, dtype=jnp.float32)
        )
        noise_guess = jnp.maximum(
            0.05 * signal_scale, jnp.array(1.0, dtype=jnp.float32)
        )
    else:
        y_used_only = jnp.where(peak_observation_mask, jnp.abs(y_observed), 0.0)
        signal_scale = jnp.maximum(
            jnp.max(y_used_only, axis=-1),
            jnp.array(1.0, dtype=jnp.float32),
        )
        residual = y_observed - peak_signal
        noise_guess = _masked_standard_deviation(
            residual,
            peak_observation_mask,
            fallback=jnp.maximum(
                0.02 * signal_scale, jnp.array(1.0, dtype=jnp.float32)
            ),
        )

    sigma_y = numpyro.sample(
        "sigma_y",
        dist.LogNormal(
            jnp.log(jnp.maximum(noise_guess, jnp.array(1e-6, dtype=jnp.float32))), 0.5
        ),
    )

    y_distribution = dist.Normal(peak_signal, sigma_y[:, None]).mask(
        peak_observation_mask
    )
    y_observed_masked = (
        None
        if y_observed is None
        else jnp.where(peak_observation_mask, y_observed, peak_signal)
    )
    numpyro.sample("y", y_distribution, obs=y_observed_masked)


__all__ = [
    "log_skew_normal_pdf",
    "model",
    "skew_components_area",
    "skew_mixture_area",
    "skew_normal_pdf",
]
