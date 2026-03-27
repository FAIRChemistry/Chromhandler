import numpy as np
import pytest

from chromhandler.fitting.shift import align_chromatograms, shift_trace_linear


@pytest.mark.unit
def test_shift_trace_linear_uses_continuous_interpolation() -> None:
    trace = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float32)

    shifted_pos = np.asarray(shift_trace_linear(trace, np.array(0.6, dtype=np.float32)))
    shifted_neg = np.asarray(shift_trace_linear(trace, np.array(-1.4, dtype=np.float32)))

    np.testing.assert_allclose(shifted_pos, np.array([0.0, 4.0, 14.0, 24.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(shifted_neg, np.array([14.0, 24.0, 30.0, 30.0], dtype=np.float32), atol=1e-6)


@pytest.mark.unit
def test_alignment_keeps_y_values_unchanged() -> None:
    signal = np.array(
        [
            [0.0, 0.0, 7.0, 19.0, 7.0, 0.0, 0.0],
            [0.0, 7.0, 19.0, 7.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 7.0, 19.0, 7.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = align_chromatograms(
        signal=signal,
        n_steps=20,
        center_weight=0.0,
        max_shift_samples=3,
    )
    shifts = np.asarray(result.shifts_samples)
    assert np.all(np.isfinite(shifts))
    assert np.all(np.abs(shifts) <= 3.0 + 1e-6)

    aligned = np.asarray(result.signal_aligned)
    np.testing.assert_allclose(aligned, signal)


@pytest.mark.unit
def test_alignment_can_return_fractional_shifts() -> None:
    x = np.arange(121, dtype=np.float32)
    base = np.exp(-0.5 * ((x - 60.0) / 5.0) ** 2).astype(np.float32)
    shifted = np.interp(x - 0.35, x, base, left=base[0], right=base[-1]).astype(np.float32)
    signal = np.stack([base, shifted], axis=0)

    result = align_chromatograms(
        signal=signal,
        n_steps=20,
        center_weight=0.0,
        max_shift_samples=3,
    )
    shifts = np.asarray(result.shifts_samples)
    fractional = np.abs(shifts - np.rint(shifts))
    assert np.any(fractional > 1e-3)


@pytest.mark.unit
def test_alignment_initializer_detects_large_coarse_shift() -> None:
    x = np.arange(241, dtype=np.float32)
    base = np.exp(-0.5 * ((x - 120.0) / 6.0) ** 2).astype(np.float32)
    shifted = np.interp(x - 8.0, x, base, left=base[0], right=base[-1]).astype(np.float32)
    signal = np.stack([base, shifted], axis=0)

    # lr=0 and n_steps=1 isolates the initialization path.
    result = align_chromatograms(
        signal=signal,
        lr=0.0,
        n_steps=1,
        center_weight=0.0,
        max_shift_samples=20,
        enforce_zero_mean=False,
    )
    shifts = np.asarray(result.shifts_samples)
    assert np.any(np.abs(shifts) >= 4.0)
