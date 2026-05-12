"""Tests for controls-based artefact measurement extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
from scipy.stats import skewnorm

from chromhandler.annotations import PeakAnnotation
from chromhandler.fitting.priors import (
    ArtefactMeasurements,
    PriorConfig,
    extract_artefact_from_controls,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _trace(
    mu_analyte: float | None,
    A_analyte: float,
    mu_artefact: float,
    A_artefact: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    t: NDArray[np.float64] = np.arange(2.5, 3.2, 0.001, dtype=np.float64)
    s: NDArray[np.float64] = np.asarray(
        A_artefact * skewnorm.pdf(t, 0.0, loc=mu_artefact, scale=0.025),
        dtype=np.float64,
    )
    if mu_analyte is not None and A_analyte > 0:
        s = s + np.asarray(
            A_analyte * skewnorm.pdf(t, 0.0, loc=mu_analyte, scale=0.025),
            dtype=np.float64,
        )
    return t, s


def _make_dataset(
    mu_artefact: float = 2.95,
    A_artefact: float = 5.0,
    n_control: int = 2,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    rows_t: list[NDArray[np.float64]] = []
    rows_s: list[NDArray[np.float64]] = []
    ic: list[bool] = []
    for A_an in [100.0, 60.0, 20.0]:
        t, s = _trace(2.85, A_an, mu_artefact, A_artefact)
        rows_t.append(t)
        rows_s.append(s)
        ic.append(False)
    for _ in range(n_control):
        t, s = _trace(None, 0.0, mu_artefact, A_artefact)
        rows_t.append(t)
        rows_s.append(s)
        ic.append(True)
    return (
        np.array(rows_t, dtype=np.float64),
        np.array(rows_s, dtype=np.float64),
        np.array(ic, dtype=np.bool_),
    )


def _ann(side: Literal["left", "right"] = "right") -> PeakAnnotation:
    return PeakAnnotation(
        molecule_id="ana", rt_min=2.78, rt_max=3.05,
        mode="artefact_doublet", artefact_side=side,
    )


def test_returns_measurements_dataclass() -> None:
    time, signal, is_control = _make_dataset()
    out = extract_artefact_from_controls(
        time=time, signal=signal, is_control=is_control,
        annotation=_ann(), dt=0.001, config=PriorConfig(),
    )
    assert isinstance(out, ArtefactMeasurements)
    assert out.A_total_per_trace.shape == (5,)
    assert out.mu_per_control.shape == (2,)


def test_recovers_artefact_area() -> None:
    time, signal, is_control = _make_dataset(A_artefact=5.0)
    out = extract_artefact_from_controls(
        time=time, signal=signal, is_control=is_control,
        annotation=_ann(), dt=0.001, config=PriorConfig(),
    )
    assert abs(out.A_artefact_est - 5.0) / 5.0 < 0.05


def test_recovers_delta_signed() -> None:
    time, signal, is_control = _make_dataset(mu_artefact=2.95)
    out = extract_artefact_from_controls(
        time=time, signal=signal, is_control=is_control,
        annotation=_ann(), dt=0.001, config=PriorConfig(),
    )
    assert abs(out.delta_signed - 0.10) < 0.01


def test_side_check_raises_on_mismatch() -> None:
    time, signal, is_control = _make_dataset(mu_artefact=2.95)
    with pytest.raises(ValueError, match="artefact_side"):
        extract_artefact_from_controls(
            time=time, signal=signal, is_control=is_control,
            annotation=_ann(side="left"),
            dt=0.001, config=PriorConfig(),
        )


def test_side_check_raises_when_peaks_too_close() -> None:
    time, signal, is_control = _make_dataset(mu_artefact=2.852)
    with pytest.raises(ValueError, match="too close"):
        extract_artefact_from_controls(
            time=time, signal=signal, is_control=is_control,
            annotation=_ann(), dt=0.001, config=PriorConfig(),
        )


def test_raises_when_no_controls() -> None:
    time, signal, _ = _make_dataset()
    is_control = np.zeros(5, dtype=np.bool_)
    with pytest.raises(ValueError, match="no control"):
        extract_artefact_from_controls(
            time=time, signal=signal, is_control=is_control,
            annotation=_ann(), dt=0.001, config=PriorConfig(),
        )


def test_config_override_changes_epsilon() -> None:
    """A larger epsilon makes the side check stricter."""
    time, signal, is_control = _make_dataset(mu_artefact=2.86)
    # Default epsilon = 3·dt = 0.003; 0.01 passes.
    extract_artefact_from_controls(
        time=time, signal=signal, is_control=is_control,
        annotation=_ann(), dt=0.001, config=PriorConfig(),
    )
    # Bump to 30·dt → too tight; raises.
    with pytest.raises(ValueError, match="too close"):
        extract_artefact_from_controls(
            time=time, signal=signal, is_control=is_control,
            annotation=_ann(), dt=0.001,
            config=PriorConfig(side_check_epsilon_dt_multiplier=30.0),
        )
