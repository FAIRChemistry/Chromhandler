"""Unit tests for data.py — ModelHyperparams, mode helpers, pad_traces, masks."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from chromhandler.fitting.types import (
    ModelHyperparams,
    peak_component_count,
    peak_is_artefact_mode,
    peak_is_doublet_mode,
    peak_is_free_mode,
)
from chromhandler.fitting.utils import pad_traces, region_to_mask

# ---------------------------------------------------------------------------
# ModelHyperparams
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_model_hyperparams_defaults_all_finite_positive() -> None:
    """All default hyperparameter values are finite and positive."""
    hp = ModelHyperparams()
    for field in hp.__dataclass_fields__:
        val = getattr(hp, field)
        assert math.isfinite(val), f"{field}={val} is not finite"
        assert val > 0, f"{field}={val} is not positive"


@pytest.mark.unit
def test_model_hyperparams_snr_thresholds_ordered() -> None:
    """S/N thresholds: low < high."""
    hp = ModelHyperparams()
    assert hp.area_snr_threshold_low < hp.area_snr_threshold_high


@pytest.mark.unit
def test_model_hyperparams_area_log_sigma_ordered() -> None:
    """Area prior spread: high-SNR < low-SNR (tight < wide)."""
    hp = ModelHyperparams()
    assert hp.area_log_sigma_high_snr < hp.area_log_sigma_low_snr


@pytest.mark.unit
def test_model_hyperparams_custom_override() -> None:
    """Custom values are stored correctly."""
    hp = ModelHyperparams(w_prior_log_scale=0.6, area_log_sigma_high_snr=0.2)
    assert hp.w_prior_log_scale == 0.6
    assert hp.area_log_sigma_high_snr == 0.2


# ---------------------------------------------------------------------------
# peak_mode helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_peak_component_count() -> None:
    assert peak_component_count("single") == 1
    assert peak_component_count("artefact_doublet") == 2
    assert peak_component_count("free_doublet") == 2


@pytest.mark.unit
def test_peak_is_doublet_mode() -> None:
    assert not peak_is_doublet_mode("single")
    assert peak_is_doublet_mode("artefact_doublet")
    assert peak_is_doublet_mode("free_doublet")


@pytest.mark.unit
def test_peak_is_artefact_mode() -> None:
    assert not peak_is_artefact_mode("single")
    assert peak_is_artefact_mode("artefact_doublet")
    assert not peak_is_artefact_mode("free_doublet")


@pytest.mark.unit
def test_peak_is_free_mode() -> None:
    assert not peak_is_free_mode("single")
    assert not peak_is_free_mode("artefact_doublet")
    assert peak_is_free_mode("free_doublet")


# ---------------------------------------------------------------------------
# pad_traces
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pad_traces_equal_length_no_nan() -> None:
    """Equal-length traces: no NaN padding."""
    x = [[0.0, 1.0, 2.0], [0.1, 1.1, 2.1]]
    y = [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]]
    px, py = pad_traces(x, y)
    assert px.shape == (2, 3)
    assert py.shape == (2, 3)
    assert not np.any(np.isnan(px))
    assert not np.any(np.isnan(py))


@pytest.mark.unit
def test_pad_traces_unequal_length_nan_padding() -> None:
    """Unequal traces: shorter rows padded with NaN."""
    x = [[0.0, 1.0], [0.0, 1.0, 2.0]]
    y = [[10.0, 20.0], [10.0, 20.0, 30.0]]
    px, py = pad_traces(x, y)
    assert px.shape == (2, 3)
    assert np.isnan(px[0, 2])
    assert np.isnan(py[0, 2])
    assert not np.isnan(px[1, 2])


@pytest.mark.unit
def test_pad_traces_mismatched_lengths_raises() -> None:
    """Mismatched x/y list lengths raise ValueError."""
    with pytest.raises(ValueError):
        pad_traces([[0.0, 1.0]], [[10.0, 20.0], [11.0, 21.0]])


# ---------------------------------------------------------------------------
# region_to_mask
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_region_to_mask_basic() -> None:
    """Mask is True inside [low, high], False outside."""
    time = jnp.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0
    mask = region_to_mask(0.25, 0.75, time)
    mask_np = np.asarray(mask)
    # Points inside [0.25, 0.75]: indices 3..7 (0.3, 0.4, 0.5, 0.6, 0.7) plus 0.25→0.3, 0.75→0.7
    assert not mask_np[0]   # 0.0
    assert mask_np[5]       # 0.5
    assert not mask_np[10]  # 1.0


@pytest.mark.unit
def test_region_to_mask_inclusive_boundaries() -> None:
    """Exact boundary points are included."""
    time = jnp.array([0.0, 0.5, 1.0])
    mask = region_to_mask(0.0, 1.0, time)
    assert np.all(np.asarray(mask))
