"""log_A_left_loc_per_trace must be length n_trace, with control entries at floor."""

from __future__ import annotations

import numpy as np
from scipy.stats import skewnorm

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.prepared_dataset import prepare_dataset
from chromhandler.fitting.priors import PriorConfig, build_priors


def _synth(n_sample: int = 3, n_control: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    times, signals, is_ctrl = [], [], []
    for amp in np.linspace(100.0, 30.0, n_sample):
        t = np.arange(2.5, 3.6, 0.001)
        s = amp * skewnorm.pdf(t, 0.0, loc=2.95, scale=0.025)
        s = s + 5.0 + rng.normal(0.0, 0.5, size=t.shape)
        times.append(t)
        signals.append(s)
        is_ctrl.append(False)
    for _ in range(n_control):
        t = np.arange(2.5, 3.6, 0.001)
        s = 5.0 + rng.normal(0.0, 0.5, size=t.shape)  # baseline + noise only
        times.append(t)
        signals.append(s)
        is_ctrl.append(True)
    peaks = [PeakAnnotation(molecule_id="A", rt_min=2.85, rt_max=3.10, mode="single")]
    bases = [
        BaselineAnnotation(rt_min=2.50, rt_max=2.52),
        BaselineAnnotation(rt_min=3.55, rt_max=3.58),
    ]
    return prepare_dataset(times, signals, peaks, bases, is_control=is_ctrl)


def test_log_A_array_has_full_n_trace_length() -> None:
    ds = _synth(n_sample=3, n_control=2)
    priors = build_priors(ds, config=PriorConfig())
    assert priors[0].log_A_left_loc_per_trace.shape == (ds.n_trace,)


def test_control_entries_are_at_floor() -> None:
    ds = _synth(n_sample=3, n_control=2)
    priors = build_priors(ds, config=PriorConfig())
    p = priors[0]
    control_idx = np.where(ds.is_control)[0]
    non_control_idx = np.where(~ds.is_control)[0]
    # Control entries should be much smaller than non-control entries.
    assert float(p.log_A_left_loc_per_trace[control_idx].max()) < float(
        p.log_A_left_loc_per_trace[non_control_idx].min()
    )
    # And finite.
    assert np.all(np.isfinite(p.log_A_left_loc_per_trace))
