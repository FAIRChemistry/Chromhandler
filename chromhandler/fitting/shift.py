from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ShiftAlignmentResult:
    """Single-stage retention-shift alignment result for signals [C, N]."""

    shifts_samples: jnp.ndarray  # [C]
    signal_aligned: jnp.ndarray  # [C, N]
    template: jnp.ndarray  # [N]
    loss_initial: float
    loss_final: float
    loss_history: Optional[jnp.ndarray] = None


def shift_trace_linear(
    signal_trace: jnp.ndarray,
    delta_samples: jnp.ndarray,
) -> jnp.ndarray:
    """Shift one 1D trace by a continuous amount using linear interpolation."""
    signal_trace = jnp.asarray(signal_trace, dtype=jnp.float32)
    delta_samples = jnp.asarray(delta_samples, dtype=jnp.float32)

    num_points = signal_trace.shape[0]
    index = jnp.arange(num_points, dtype=jnp.float32)
    source_index = index - delta_samples

    left_index = jnp.floor(source_index).astype(jnp.int32)
    right_index = left_index + 1

    left_index = jnp.clip(left_index, 0, num_points - 1)
    right_index = jnp.clip(right_index, 0, num_points - 1)

    right_weight = source_index - jnp.floor(source_index)
    left_weight = 1.0 - right_weight
    return (
        left_weight * signal_trace[left_index]
        + right_weight * signal_trace[right_index]
    )


shift_signal_by_trace_shifts = jax.vmap(
    shift_trace_linear,
    in_axes=(0, 0),
    out_axes=0,
)  # [C,N], [C] -> [C,N]


def _masked_template(
    shifted_signal: jnp.ndarray,
    mask: Optional[jnp.ndarray],
) -> jnp.ndarray:
    """Compute per-time template from aligned traces, optionally masked."""
    if mask is None:
        return jnp.mean(shifted_signal, axis=0)

    weights = jnp.asarray(mask, dtype=jnp.float32)
    weight_sum = jnp.sum(weights, axis=0)
    safe_weight_sum = jnp.maximum(weight_sum, jnp.array(1.0, dtype=jnp.float32))
    return jnp.sum(shifted_signal * weights, axis=0) / safe_weight_sum


def _coarse_shift_initialization(
    signal: jnp.ndarray,
    mask: Optional[jnp.ndarray],
    max_shift_samples: Optional[float],
) -> jnp.ndarray:
    """Estimate integer shift seeds by per-trace masked correlation to a template."""
    signal_array = np.asarray(signal, dtype=np.float32)
    num_chromatograms, num_points = signal_array.shape
    if num_points < 3 or num_chromatograms == 0:
        return jnp.zeros((num_chromatograms,), dtype=jnp.float32)

    if mask is None:
        mask_array = np.ones_like(signal_array, dtype=bool)
    else:
        mask_array = np.asarray(mask, dtype=bool)

    template = np.asarray(_masked_template(signal, mask), dtype=np.float32)

    if max_shift_samples is not None:
        search_radius = int(max(1, np.ceil(float(max_shift_samples))))
    else:
        # Conservative default radius; users can enlarge via `max_shift_samples`.
        search_radius = int(max(8, min(num_points // 50, 20)))
    search_radius = min(search_radius, max(num_points - 2, 1))
    lag_grid = np.arange(-search_radius, search_radius + 1, dtype=np.int32)

    initial = np.zeros((num_chromatograms,), dtype=np.float32)
    for chrom_idx in range(num_chromatograms):
        trace = signal_array[chrom_idx]
        trace_mask = mask_array[chrom_idx]
        if int(np.sum(trace_mask)) < 3:
            continue

        best_score = -np.inf
        best_lag = 0
        for lag in lag_grid:
            if lag >= 0:
                n_valid = num_points - int(lag)
                trace_segment = trace[:n_valid]
                template_segment = template[int(lag) :]
                mask_segment = trace_mask[int(lag) :]
            else:
                offset = int(-lag)
                n_valid = num_points - offset
                trace_segment = trace[offset:]
                template_segment = template[:n_valid]
                mask_segment = trace_mask[:n_valid]

            if n_valid < 3 or int(np.sum(mask_segment)) < 3:
                continue

            y_values = np.asarray(trace_segment[mask_segment], dtype=np.float64)
            t_values = np.asarray(template_segment[mask_segment], dtype=np.float64)
            y_centered = y_values - float(np.mean(y_values))
            t_centered = t_values - float(np.mean(t_values))
            denominator = np.sqrt(
                float(np.sum(y_centered * y_centered))
                * float(np.sum(t_centered * t_centered))
            )
            if denominator <= 1e-12:
                continue
            score = float(np.sum(y_centered * t_centered) / denominator)
            if score > best_score:
                best_score = score
                best_lag = int(lag)

        initial[chrom_idx] = float(best_lag)

    return jnp.asarray(initial, dtype=jnp.float32)


def single_stage_alignment_loss(
    shifts_samples: jnp.ndarray,
    signal: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    center_weight: float = 1e3,
) -> jnp.ndarray:
    """Loss for global one-stage alignment of traces [C, N]."""
    signal = jnp.asarray(signal, dtype=jnp.float32)
    shifts_samples = jnp.asarray(shifts_samples, dtype=jnp.float32)
    shifted_signal = shift_signal_by_trace_shifts(signal, shifts_samples)

    template = _masked_template(shifted_signal, mask)
    residual = shifted_signal - template[None, :]

    if mask is None:
        data_term = jnp.sum(residual**2)
    else:
        mask_bool = jnp.asarray(mask, dtype=bool)
        data_term = jnp.sum(jnp.where(mask_bool, residual**2, 0.0))

    center_term = jnp.asarray(center_weight, dtype=jnp.float32) * (
        jnp.mean(shifts_samples) ** 2
    )
    return data_term + center_term


def _adam_optimize(
    initial_params: jnp.ndarray,
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
    lr: float,
    n_steps: int,
    max_shift_samples: Optional[float],
    recenter_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]],
    return_history: bool,
) -> tuple[jnp.ndarray, float, float, Optional[jnp.ndarray]]:
    """Adam optimizer for shift parameters."""
    if n_steps <= 0:
        raise ValueError("`n_steps` must be > 0.")

    params = jnp.asarray(initial_params, dtype=jnp.float32)
    beta1 = jnp.asarray(0.9, dtype=jnp.float32)
    beta2 = jnp.asarray(0.999, dtype=jnp.float32)
    epsilon = jnp.asarray(1e-8, dtype=jnp.float32)
    learning_rate = jnp.asarray(lr, dtype=jnp.float32)

    first_moment = jnp.zeros_like(params)
    second_moment = jnp.zeros_like(params)
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    loss_initial = float(loss_fn(params))
    history = [] if return_history else None

    for step_index in range(1, int(n_steps) + 1):
        loss_value, gradient = loss_and_grad(params)
        if return_history and history is not None:
            history.append(loss_value)

        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * (gradient**2)

        first_unbiased = first_moment / (1.0 - beta1**step_index)
        second_unbiased = second_moment / (1.0 - beta2**step_index)
        params = params - learning_rate * first_unbiased / (
            jnp.sqrt(second_unbiased) + epsilon
        )

        if max_shift_samples is not None:
            clip_value = jnp.asarray(max_shift_samples, dtype=jnp.float32)
            params = jnp.clip(params, -clip_value, clip_value)

        if recenter_fn is not None:
            params = recenter_fn(params)

    loss_final = float(loss_fn(params))
    history_out = None
    if return_history and history is not None:
        history_out = jnp.asarray(history, dtype=jnp.float32)

    return params, loss_initial, loss_final, history_out


def align_chromatogram_shifts(
    signal: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    lr: float = 1e-2,
    n_steps: int = 500,
    center_weight: float = 1e3,
    max_shift_samples: Optional[float] = None,
    enforce_zero_mean: bool = True,
    return_history: bool = False,
) -> ShiftAlignmentResult:
    """Align traces with one shift per chromatogram for signal shape [C, N]."""
    signal = jnp.asarray(signal, dtype=jnp.float32)
    if signal.ndim != 2:
        raise ValueError(f"`signal` must be [C, N], got {signal.shape}")

    finite_signal = jnp.isfinite(signal)
    signal_clean = jnp.where(finite_signal, signal, 0.0)

    mask_clean: Optional[jnp.ndarray]
    if mask is None:
        mask_clean = finite_signal
    else:
        mask_arr = jnp.asarray(mask, dtype=bool)
        if mask_arr.shape != signal.shape:
            raise ValueError(
                f"`mask` must match signal shape {signal.shape}, got {mask_arr.shape}"
            )
        mask_clean = mask_arr & finite_signal

    initial_shifts = _coarse_shift_initialization(
        signal_clean,
        mask_clean,
        max_shift_samples=max_shift_samples,
    )
    if enforce_zero_mean:
        initial_shifts = initial_shifts - jnp.mean(initial_shifts)
    if max_shift_samples is not None:
        clip_value = jnp.asarray(max_shift_samples, dtype=jnp.float32)
        initial_shifts = jnp.clip(initial_shifts, -clip_value, clip_value)
    recenter_fn = (lambda z: z - jnp.mean(z)) if enforce_zero_mean else None

    def loss_fn(shifts: jnp.ndarray) -> jnp.ndarray:
        return single_stage_alignment_loss(
            shifts,
            signal_clean,
            mask=mask_clean,
            center_weight=center_weight,
        )

    shifts_samples, loss_initial, loss_final, history = _adam_optimize(
        initial_params=initial_shifts,
        loss_fn=loss_fn,
        lr=lr,
        n_steps=n_steps,
        max_shift_samples=max_shift_samples,
        recenter_fn=recenter_fn,
        return_history=return_history,
    )

    # X-axis-only contract: keep measured y-values unchanged.
    signal_aligned = signal_clean
    shifted_for_template = shift_signal_by_trace_shifts(signal_clean, shifts_samples)
    template = _masked_template(shifted_for_template, mask_clean)

    return ShiftAlignmentResult(
        shifts_samples=shifts_samples,
        signal_aligned=signal_aligned,
        template=template,
        loss_initial=loss_initial,
        loss_final=loss_final,
        loss_history=history,
    )


def align_groupwise_sample_shifts(
    signal: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    lr: float = 1e-2,
    n_steps: int = 500,
    center_weight: float = 1e3,
    max_shift_samples: Optional[float] = None,
    enforce_zero_mean: bool = True,
    return_history: bool = False,
) -> ShiftAlignmentResult:
    """Backward-compat wrapper for the former two-stage API.

    This project now uses a single-stage aligner for 2D signals with shape
    ``[n_chromatograms, n_timepoints]``. The wrapper is kept to avoid import
    breakage in older modules and forwards to `align_chromatogram_shifts`.
    """
    signal = jnp.asarray(signal)
    if signal.ndim != 2:
        raise ValueError(
            "Single-stage alignment requires 2D signal [n_chromatograms, n_timepoints], "
            f"got {signal.shape}."
        )
    return align_chromatogram_shifts(
        signal=signal,
        mask=mask,
        lr=lr,
        n_steps=n_steps,
        center_weight=center_weight,
        max_shift_samples=max_shift_samples,
        enforce_zero_mean=enforce_zero_mean,
        return_history=return_history,
    )


__all__ = [
    "ShiftAlignmentResult",
    "align_groupwise_sample_shifts",
    "align_chromatogram_shifts",
    "shift_signal_by_trace_shifts",
    "shift_trace_linear",
    "single_stage_alignment_loss",
]
