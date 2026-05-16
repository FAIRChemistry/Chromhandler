"""Tests for ``Handler.align_chromatograms`` and the underlying shift module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray

from chromhandler.fitting.shift import align_chromatograms as align_arrays
from chromhandler.handler import Handler
from chromhandler.model import Chromatogram, Estimate, Peak, Sample


def _gaussian(
    t: NDArray[Any], center: float, sigma: float = 0.05, amp: float = 1.0
) -> NDArray[Any]:
    return amp * np.exp(-0.5 * ((t - center) / sigma) ** 2)


def test_align_arrays_recovers_known_shifts() -> None:
    """Synthetic shifts in [C, N] array space are recovered to within dt/4."""
    dt = 0.005
    t = np.arange(0.0, 4.0, dt, dtype=np.float32)
    true_shifts_samples = np.array([-6.0, -2.0, 0.0, 3.0, 5.0], dtype=np.float32)
    rt0 = 2.0

    signals = np.stack(
        [_gaussian(t, center=rt0 + s * dt) for s in true_shifts_samples], axis=0
    ).astype(np.float32)

    mask = (t >= rt0 - 0.3) & (t <= rt0 + 0.3)
    mask = np.broadcast_to(mask[None, :], signals.shape).copy()

    result = align_arrays(
        jnp.asarray(signals),
        mask=jnp.asarray(mask),
        max_shift_samples=20.0,
        enforce_zero_mean=True,
        n_starts=8,
        lr=1e-1,
        n_steps=1500,
        seed=0,
    )

    # A peak placed at index ``i0 + s`` requires shift ``-s`` to come back
    # to ``i0`` (signal moves opposite to where it was originally placed).
    recovered = np.asarray(result.shifts_samples)
    expected = -(true_shifts_samples - float(true_shifts_samples.mean()))
    err = np.max(np.abs(recovered - expected))
    assert err < 0.25, f"recovered shifts off by {err:.3f} samples > 0.25"


def _make_handler_with_known_shifts(
    dt: float = 0.005,
    n_time: int = 800,
    rt0: float = 2.0,
    sigma: float = 0.04,
    rt_offsets: tuple[float, ...] = (-0.03, 0.0, 0.04),
) -> tuple[Handler, list[float]]:
    handler = Handler()
    t = np.arange(n_time, dtype=np.float64) * dt
    chrom_ids: list[str] = []
    for i, offset in enumerate(rt_offsets):
        sig = _gaussian(t.astype(np.float32), center=rt0 + offset, sigma=sigma)
        peak = Peak(
            chromatogram_id=f"c{i}",
            location=Estimate(mean=rt0 + offset),
            area=Estimate(mean=1.0),
        )
        chrom = Chromatogram(
            id=f"c{i}",
            sample_id=f"s{i}",
            time=t.tolist(),
            signal=sig.astype(float).tolist(),
            peaks=[peak],
        )
        handler.samples.append(Sample(id=f"s{i}", chromatograms=[chrom]))
        chrom_ids.append(chrom.id)
    return handler, list(rt_offsets)


def test_handler_align_shifts_times_and_peaks() -> None:
    """Handler.align_chromatograms updates Chromatogram.time and Peak.location."""
    handler, offsets = _make_handler_with_known_shifts()
    orig_times = [list(c.time) for s in handler.samples for c in s.chromatograms]
    orig_peak_means = [
        c.peaks[0].location.mean for s in handler.samples for c in s.chromatograms
    ]

    result = handler.align_chromatograms(lower_rt=1.7, upper_rt=2.3)

    assert result.shifts_samples.shape == (len(offsets),)
    assert result.delta_rt.shape == (len(offsets),)
    assert result.trace_ids == [f"s{i}/c{i}" for i in range(len(offsets))]
    assert result.loss_final <= result.loss_initial

    # delta_rt should approximately cancel the injected offsets (up to a
    # global zero-mean shift). After alignment, all peaks should land at a
    # common retention time.
    aligned_peak_means = np.array(
        [c.peaks[0].location.mean for s in handler.samples for c in s.chromatograms]
    )
    assert float(np.std(aligned_peak_means)) < 0.005

    # Time arrays were shifted by exactly delta_rt.
    for orig, sample, d in zip(
        orig_times,
        [c for s in handler.samples for c in s.chromatograms],
        result.delta_rt,
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(sample.time), np.asarray(orig) + d, atol=1e-9)

    # Peak.location.mean shifted by the same delta_rt.
    for orig_mean, sample, d in zip(
        orig_peak_means,
        [c for s in handler.samples for c in s.chromatograms],
        result.delta_rt,
        strict=True,
    ):
        assert abs(sample.peaks[0].location.mean - (orig_mean + d)) < 1e-9


def test_handler_align_raises_on_empty_window() -> None:
    """A window with no samples for some trace raises with the trace id."""
    handler, _ = _make_handler_with_known_shifts()
    with pytest.raises(ValueError, match="fewer than 3 finite samples"):
        handler.align_chromatograms(lower_rt=10.0, upper_rt=10.5)


def test_handler_align_empty_raises() -> None:
    """No chromatograms at all is a hard error."""
    handler = Handler()
    with pytest.raises(ValueError, match="no chromatograms"):
        handler.align_chromatograms(lower_rt=0.0, upper_rt=1.0)
