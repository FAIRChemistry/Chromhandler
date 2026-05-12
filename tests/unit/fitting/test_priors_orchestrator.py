"""End-to-end orchestrator tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.stats import skewnorm

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.prepared_dataset import PreparedDataset, prepare_dataset
from chromhandler.fitting.priors import PriorConfig, build_priors

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _synth_dataset(n_sample: int = 3, n_control: int = 1, seed: int = 0) -> PreparedDataset:
    rng = np.random.default_rng(seed)
    times: list[NDArray[np.float64]] = []
    signals: list[NDArray[np.float64]] = []
    is_control: list[bool] = []
    for A_an in np.linspace(100.0, 10.0, n_sample):
        t: NDArray[np.float64] = np.arange(2.5, 3.6, 0.001)
        s_ino: NDArray[np.float64] = A_an * skewnorm.pdf(t, 2.0, loc=2.69, scale=0.025)
        s_main: NDArray[np.float64] = 80.0 * skewnorm.pdf(t, 0.0, loc=3.00, scale=0.025)
        s_art: NDArray[np.float64] = 5.0 * skewnorm.pdf(t, 0.0, loc=3.05, scale=0.025)
        baseline: NDArray[np.float64] = 10.0 + 0.5 * t
        noise: NDArray[np.float64] = rng.standard_normal(size=t.shape)
        times.append(t)
        signals.append(np.asarray(s_ino + s_main + s_art + baseline + noise, dtype=np.float64))
        is_control.append(False)
    for _ in range(n_control):
        t = np.arange(2.5, 3.6, 0.001)
        s: NDArray[np.float64] = 5.0 * skewnorm.pdf(t, 0.0, loc=3.05, scale=0.025)
        baseline = 10.0 + 0.5 * t
        noise = rng.standard_normal(size=t.shape)
        times.append(t)
        signals.append(np.asarray(s + baseline + noise, dtype=np.float64))
        is_control.append(True)
    peak_anns = [
        PeakAnnotation(molecule_id="Ino", rt_min=2.55, rt_max=2.85),
        PeakAnnotation(
            molecule_id="SIH", rt_min=2.90, rt_max=3.15,
            mode="artefact_doublet", artefact_side="right",
        ),
    ]
    base_anns = [
        BaselineAnnotation(rt_min=2.50, rt_max=2.52),
        BaselineAnnotation(rt_min=3.55, rt_max=3.57),
    ]
    return prepare_dataset(times, signals, peak_anns, base_anns, is_control=is_control)


def test_returns_one_per_annotation() -> None:
    priors = build_priors(_synth_dataset())
    assert len(priors) == 2
    assert priors[0].n_components == 1
    assert priors[1].n_components == 2


def test_single_recovers_mu() -> None:
    priors = build_priors(_synth_dataset())
    assert abs(priors[0].mu_left_loc - 2.70) < 0.02


def test_doublet_delta_from_controls() -> None:
    priors = build_priors(_synth_dataset())
    p = priors[1]
    assert p.Delta_loc is not None
    assert abs(p.Delta_loc - 0.05) < 0.01


def test_doublet_borrows_scales_with_single_control() -> None:
    ds = _synth_dataset(n_control=1)
    priors = build_priors(ds)
    p = priors[1]
    assert p.log_sigma_right_scale is not None
    assert p.gamma1_right_scale is not None
    assert p.log_sigma_right_scale > 0
    assert p.gamma1_right_scale > 0


def test_config_override_propagates() -> None:
    ds = _synth_dataset()
    cfg = PriorConfig(delta_low_dt_multiplier=5.0)
    priors = build_priors(ds, config=cfg)
    p = priors[1]
    assert p.Delta_low == 5.0 * ds.dt_global


def test_raises_on_doublet_without_controls() -> None:
    ds = _synth_dataset(n_control=0)
    with pytest.raises(ValueError, match="no control"):
        build_priors(ds)


def test_raises_on_free_doublet() -> None:
    ds = _synth_dataset()
    new_anns = list(ds.peak_annotations)
    new_anns[1] = PeakAnnotation(
        molecule_id="X", rt_min=2.90, rt_max=3.15, mode="free_doublet",
    )
    object.__setattr__(ds, "peak_annotations", new_anns)  # type: ignore[misc]
    with pytest.raises(NotImplementedError, match="free_doublet"):
        build_priors(ds)
