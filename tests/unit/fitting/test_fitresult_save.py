"""Review item 11: FitResult.save() must be deterministic w.r.t. call history.

Plot methods lazily cache predictive groups into `idata`. `save()` must not
let that incidental state leak into the file: a given FitResult + args must
always produce the same group set, regardless of which plots were drawn.
"""
from __future__ import annotations

import arviz as az
import numpy as np

from chromhandler.fitting.fitter import FitResult


def _fitresult_with_cached_predictives() -> FitResult:
    """A FitResult whose idata has accumulated lazy predictive groups, as if
    plot_fit() and plot_prior_predictive() had been called."""
    idata = az.from_dict(
        {
            "posterior": {"mu": np.zeros((2, 5))},
            "observed_data": {"obs": np.zeros(3)},
            "posterior_predictive": {"obs": np.zeros((2, 5, 3))},
            "prior_predictive": {"obs": np.zeros((1, 5, 3))},
        }
    )
    # dataset/priors/model_config are unused by save() on these paths
    # (posterior_predictive already present -> no recompute).
    return FitResult(idata=idata, dataset=None, priors=[], model_config=None)  # type: ignore[arg-type]


def test_save_writes_canonical_groups_regardless_of_cached_predictives(tmp_path):
    r = _fitresult_with_cached_predictives()
    out = tmp_path / "out.nc"
    r.save(out)
    groups = set(az.from_netcdf(str(out)).children)
    # Predictive groups (incidental, recomputable) are excluded -> deterministic.
    assert groups == {"posterior", "observed_data"}, groups


def test_save_include_predictive_adds_only_posterior_predictive(tmp_path):
    r = _fitresult_with_cached_predictives()
    out = tmp_path / "out_pp.nc"
    r.save(out, include_predictive=True)
    groups = set(az.from_netcdf(str(out)).children)
    assert "posterior_predictive" in groups
    assert "prior_predictive" not in groups  # diagnostic-only, never persisted
    assert {"posterior", "observed_data"} <= groups
