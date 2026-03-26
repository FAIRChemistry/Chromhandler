"""Chromatogram retention-time alignment via per-trace shift optimization.

Public surface
--------------
``align_chromatograms``   — main entry point: coarse init + multi-start Adam
``alignment_loss``        — MSE alignment loss (JAX, JIT-compatible)
``shift_trace_linear``    — interpolation-based 1-D trace shift
``shift_signal_vmap``     — vectorized shift over a batch of traces
``ShiftAlignmentResult``  — result dataclass

Algorithm
---------
1. Coarse integer-lag initialization via masked cross-correlation to template.
2. Multi-start Adam refinement:  N perturbed copies of the coarse init are
   optimized in parallel via ``jax.jit(jax.vmap(...))``.  The best result
   (minimum final loss) is returned.
3. Loss = MSE between shifted signal and mean template, restricted to the
   caller-supplied mask (typically peak windows + baseline regions), plus
   a small L2 penalty on the mean shift that keeps traces zero-centred.

Intensity weighting
-------------------
Because the loss is sum-of-squared-residuals, high-intensity regions
automatically receive larger gradient contributions and therefore dominate
the alignment.  Restricting the mask to peak windows further concentrates
the alignment on signal-bearing regions.  No explicit normalization is
applied — high-intensity signals are trusted more by construction.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShiftAlignmentResult:
    """Result of chromatogram alignment.

    Attributes
    ----------
    shifts_samples : jnp.ndarray  [C]
        Per-trace shift in sample-index units (positive = shift right).
    signal_aligned : jnp.ndarray  [C, N]
        The original (unmodified) signal.  Alignment is X-axis-only: the
        *time* axis is updated by the caller, not the signal values.
    template : jnp.ndarray  [N]
        Mean template computed from all traces after applying the best shifts.
    loss_initial : float
        Alignment loss before optimization.
    loss_final : float
        Alignment loss after optimization.
    """

    shifts_samples: jnp.ndarray
    signal_aligned: jnp.ndarray
    template: jnp.ndarray
    loss_initial: float
    loss_final: float


# ---------------------------------------------------------------------------
# Trace shifting (interpolation)
# ---------------------------------------------------------------------------


def shift_trace_linear(
    signal_trace: jnp.ndarray,
    delta_samples: jnp.ndarray,
) -> jnp.ndarray:
    """Shift one 1-D trace by a continuous amount using linear interpolation."""
    signal_trace = jnp.asarray(signal_trace, dtype=jnp.float32)
    delta_samples = jnp.asarray(delta_samples, dtype=jnp.float32)

    n = signal_trace.shape[0]
    idx = jnp.arange(n, dtype=jnp.float32)
    src = idx - delta_samples

    left = jnp.clip(jnp.floor(src).astype(jnp.int32), 0, n - 1)
    right = jnp.clip(left + 1, 0, n - 1)
    w_right = src - jnp.floor(src)
    w_left = 1.0 - w_right

    return w_left * signal_trace[left] + w_right * signal_trace[right]


shift_signal_vmap: jnp.ndarray = jax.vmap(
    shift_trace_linear,
    in_axes=(0, 0),
    out_axes=0,
)  # [C, N], [C] -> [C, N]


# ---------------------------------------------------------------------------
# Template computation
# ---------------------------------------------------------------------------


def _compute_template(
    shifted_signal: jnp.ndarray,   # [C, N]
    mask: jnp.ndarray | None,   # [C, N] bool or None
) -> jnp.ndarray:                   # [N]
    """Per-timepoint template from aligned traces, optionally masked."""
    if mask is None:
        return jnp.mean(shifted_signal, axis=0)

    weights = jnp.asarray(mask, dtype=jnp.float32)
    weight_sum = jnp.maximum(jnp.sum(weights, axis=0), 1.0)
    return jnp.sum(shifted_signal * weights, axis=0) / weight_sum


# ---------------------------------------------------------------------------
# Coarse integer-lag initialization (NumPy, one-time call)
# ---------------------------------------------------------------------------


def _coarse_shift_init(
    signal: np.ndarray,            # [C, N]
    mask: np.ndarray | None,       # [C, N] bool or None
    max_shift_samples: float | None,
) -> np.ndarray:                   # [C] float32 integer lag estimates
    """Estimate per-trace integer shifts via masked cross-correlation."""
    C, N = signal.shape
    if N < 3 or C == 0:
        return np.zeros(C, dtype=np.float32)

    mask_arr = (
        np.ones((C, N), dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )

    # Template = unweighted mean over valid (masked) values.
    # Columns with no valid points (all-False mask) naturally produce NaN;
    # suppress the spurious RuntimeWarning and replace with 0.
    with np.errstate(all="ignore"):
        template = np.nanmean(np.where(mask_arr, signal, np.nan), axis=0)
    template = np.where(np.isfinite(template), template, 0.0)

    radius = (
        int(np.ceil(max_shift_samples))
        if max_shift_samples is not None
        else min(max(8, N // 50), 20)
    )
    radius = min(radius, N - 2)

    initial = np.zeros(C, dtype=np.float32)
    for c in range(C):
        trace, m = signal[c], mask_arr[c]
        best_score, best_lag = -np.inf, 0
        for lag in np.arange(-radius, radius + 1, dtype=np.int32):
            lag = int(lag)
            if lag >= 0:
                sl = slice(lag, None)
                tl = slice(None, N - lag) if lag > 0 else slice(None, None)
            else:
                sl = slice(None, lag)
                tl = slice(-lag, None)
            y = trace[sl][m[sl]]
            t = template[tl][m[sl]]
            if y.size < 3:
                continue
            yc = y - y.mean()
            tc = t - t.mean()
            denom = np.sqrt((yc ** 2).sum() * (tc ** 2).sum())
            if denom < 1e-12:
                continue
            score = float((yc * tc).sum() / denom)
            if score > best_score:
                best_score, best_lag = score, lag
        initial[c] = float(best_lag)

    return initial


# ---------------------------------------------------------------------------
# Alignment loss (JAX, JIT-compatible)
# ---------------------------------------------------------------------------


def alignment_loss(
    shifts_samples: jnp.ndarray,   # [C]
    signal: jnp.ndarray,           # [C, N]
    mask: jnp.ndarray | None = None,
    center_weight: float = 1e3,
) -> jnp.ndarray:
    """MSE alignment loss for shift parameters.

    Parameters
    ----------
    shifts_samples : [C]
        Per-trace shifts to evaluate.
    signal : [C, N]
        Signal matrix (should have NaN replaced with 0 before calling).
    mask : [C, N] bool or None
        Restrict loss to masked timepoints.  None = use all points.
    center_weight : float
        Penalty coefficient on ``mean(shifts)^2``, keeps shifts zero-centred.
    """
    signal = jnp.asarray(signal, dtype=jnp.float32)
    shifts_samples = jnp.asarray(shifts_samples, dtype=jnp.float32)

    shifted = shift_signal_vmap(signal, shifts_samples)
    template = _compute_template(shifted, mask)
    residual = shifted - template[None, :]

    if mask is None:
        data_term = jnp.sum(residual ** 2)
    else:
        mask_bool = jnp.asarray(mask, dtype=bool)
        data_term = jnp.sum(jnp.where(mask_bool, residual ** 2, 0.0))

    center_term = jnp.float32(center_weight) * jnp.mean(shifts_samples) ** 2
    return data_term + center_term


# ---------------------------------------------------------------------------
# Adam optimizer via lax.scan (vmappable)
# ---------------------------------------------------------------------------


def _adam_scan(
    initial_params: jnp.ndarray,   # [C]
    signal: jnp.ndarray,            # [C, N]
    mask: jnp.ndarray | None,   # [C, N] bool or None
    *,
    lr: float,
    n_steps: int,
    center_weight: float,
    max_shift_samples: float | None,
    enforce_zero_mean: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:   # (final_params [C], final_loss scalar)
    """Single Adam run using jax.lax.scan — safe to vmap.

    No jax.jit calls inside: the outer jit(vmap(...)) handles compilation.
    Python constants (max_shift_samples, enforce_zero_mean) become
    compile-time branches during JAX tracing.
    """
    val_and_grad = jax.value_and_grad(
        lambda s: alignment_loss(s, signal, mask=mask, center_weight=center_weight)
    )

    def step(carry: tuple, _: None) -> tuple[tuple, None]:
        p, m, v, t = carry
        _, g = val_and_grad(p)
        t = t + jnp.int32(1)
        m = jnp.float32(0.9) * m + jnp.float32(0.1) * g
        v = jnp.float32(0.999) * v + jnp.float32(0.001) * g ** 2
        m_hat = m / (jnp.float32(1.0) - jnp.float32(0.9) ** t)
        v_hat = v / (jnp.float32(1.0) - jnp.float32(0.999) ** t)
        p = p - jnp.float32(lr) * m_hat / (jnp.sqrt(v_hat) + jnp.float32(1e-8))
        if max_shift_samples is not None:
            p = jnp.clip(p, -max_shift_samples, max_shift_samples)
        if enforce_zero_mean:
            p = p - jnp.mean(p)
        return (p, m, v, t), None

    init_carry = (
        jnp.asarray(initial_params, dtype=jnp.float32),
        jnp.zeros_like(initial_params, dtype=jnp.float32),
        jnp.zeros_like(initial_params, dtype=jnp.float32),
        jnp.zeros((), dtype=jnp.int32),
    )
    (p_final, _, _, _), _ = jax.lax.scan(step, init_carry, None, length=int(n_steps))
    final_loss = alignment_loss(p_final, signal, mask=mask, center_weight=center_weight)
    return p_final, final_loss


# ---------------------------------------------------------------------------
# Multi-start optimization via jit(vmap(...))
# ---------------------------------------------------------------------------


def _multistart_optimize(
    coarse_init: jnp.ndarray,       # [C]
    signal: jnp.ndarray,             # [C, N]
    mask: jnp.ndarray | None,    # [C, N] bool or None
    *,
    n_starts: int,
    sigma_perturb: float,
    seed: int,
    lr: float,
    n_steps: int,
    center_weight: float,
    max_shift_samples: float | None,
    enforce_zero_mean: bool,
) -> jnp.ndarray:   # [C] best shifts
    """Run N perturbed Adam starts in parallel; return the best result.

    Uses ``jax.jit(jax.vmap(run_one))`` to compile the whole batch once.
    """
    key = jax.random.PRNGKey(seed)
    noise = jax.random.normal(key, shape=(n_starts - 1, coarse_init.shape[0]))
    perturbed = coarse_init[None, :] + jnp.float32(sigma_perturb) * noise  # [n_starts-1, C]
    all_inits = jnp.concatenate(
        [coarse_init[None, :], perturbed], axis=0
    )  # [n_starts, C]

    if enforce_zero_mean:
        all_inits = all_inits - jnp.mean(all_inits, axis=1, keepdims=True)
    if max_shift_samples is not None:
        all_inits = jnp.clip(all_inits, -max_shift_samples, max_shift_samples)

    adam_kwargs = {
        "lr": lr,
        "n_steps": n_steps,
        "center_weight": center_weight,
        "max_shift_samples": max_shift_samples,
        "enforce_zero_mean": enforce_zero_mean,
    }

    run_one = functools.partial(_adam_scan, signal=signal, mask=mask, **adam_kwargs)

    # jit(vmap(f)): compile the entire batch once
    all_params, all_losses = jax.jit(jax.vmap(run_one))(all_inits)  # [n_starts, C], [n_starts]
    return all_params[jnp.argmin(all_losses)]  # [C]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def align_chromatograms(
    signal: jnp.ndarray,
    mask: jnp.ndarray | None = None,
    *,
    lr: float = 1e-2,
    n_steps: int = 500,
    center_weight: float = 1e3,
    max_shift_samples: float | None = None,
    enforce_zero_mean: bool = True,
    n_starts: int = 16,
    sigma_perturb: float = 3.0,
    seed: int = 0,
) -> ShiftAlignmentResult:
    """Align traces with one shift per chromatogram.

    Parameters
    ----------
    signal : [C, N]
        Signal matrix (chromatograms x timepoints).
    mask : [C, N] bool or None
        Restrict alignment to these timepoints (e.g. peak windows + baseline
        regions).  None = use all finite points.
    lr : float
        Adam learning rate.
    n_steps : int
        Adam iterations per start.
    center_weight : float
        Penalty on ``mean(shifts)^2`` to keep shifts zero-centred.
    max_shift_samples : float or None
        Hard bound on shift magnitudes (samples).
    enforce_zero_mean : bool
        Re-centre shifts after each step.
    n_starts : int
        Number of independent Adam starts.  Start 0 uses the coarse
        correlation estimate; starts 1..n_starts-1 are perturbed copies.
        Set to 1 to skip multi-start (faster, less robust).
    sigma_perturb : float
        Standard deviation of perturbation noise (samples) for starts 1+.
    seed : int
        PRNG seed for perturbation noise.

    Returns
    -------
    ShiftAlignmentResult
        Best shifts, original signal, aligned template, and loss values.
        ``signal_aligned`` is the *original* signal — the caller is expected
        to update the time axis: ``time += shifts_samples * dt``.
    """
    signal = jnp.asarray(signal, dtype=jnp.float32)
    if signal.ndim != 2:
        raise ValueError(f"`signal` must be [C, N], got shape {signal.shape}")

    finite = jnp.isfinite(signal)
    signal_clean = jnp.where(finite, signal, jnp.float32(0.0))

    if mask is None:
        mask_clean: jnp.ndarray | None = finite
    else:
        mask_arr = jnp.asarray(mask, dtype=bool)
        if mask_arr.shape != signal.shape:
            raise ValueError(
                f"`mask` shape {mask_arr.shape} does not match signal shape {signal.shape}"
            )
        mask_clean = mask_arr & finite

    # Coarse integer-lag initialization (NumPy, one-time)
    coarse_init = _coarse_shift_init(
        np.asarray(signal_clean),
        np.asarray(mask_clean) if mask_clean is not None else None,
        max_shift_samples,
    )
    coarse_init_jax = jnp.asarray(coarse_init, dtype=jnp.float32)
    if enforce_zero_mean:
        coarse_init_jax = coarse_init_jax - jnp.mean(coarse_init_jax)
    if max_shift_samples is not None:
        coarse_init_jax = jnp.clip(coarse_init_jax, -max_shift_samples, max_shift_samples)

    loss_initial = float(
        alignment_loss(coarse_init_jax, signal_clean, mask=mask_clean, center_weight=center_weight)
    )

    adam_kwargs = {
        "lr": lr,
        "n_steps": n_steps,
        "center_weight": center_weight,
        "max_shift_samples": max_shift_samples,
        "enforce_zero_mean": enforce_zero_mean,
    }

    if n_starts == 1:
        best_shifts, _ = _adam_scan(coarse_init_jax, signal_clean, mask_clean, **adam_kwargs)
    else:
        best_shifts = _multistart_optimize(
            coarse_init_jax, signal_clean, mask_clean,
            n_starts=n_starts,
            sigma_perturb=sigma_perturb,
            seed=seed,
            **adam_kwargs,
        )

    loss_final = float(
        alignment_loss(best_shifts, signal_clean, mask=mask_clean, center_weight=center_weight)
    )

    shifted_for_template = shift_signal_vmap(signal_clean, best_shifts)
    template = _compute_template(shifted_for_template, mask_clean)

    return ShiftAlignmentResult(
        shifts_samples=best_shifts,
        signal_aligned=signal_clean,   # X-axis-only: caller updates time, not signal
        template=template,
        loss_initial=loss_initial,
        loss_final=loss_final,
    )


__all__ = [
    "ShiftAlignmentResult",
    "align_chromatograms",
    "alignment_loss",
    "shift_signal_vmap",
    "shift_trace_linear",
]
