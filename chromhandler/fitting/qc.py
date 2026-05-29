"""Quality-control views over a fit's InferenceData.

Pure functions: each takes an arviz DataTree (`idata`) and returns a
matplotlib Figure or a metric dict. No FitResult/dataset dependency, so
they are unit-testable on a synthetic idata. Plots are chosen by parameter
ROLE so the view scales to the model's many parameters:
  - shape (mu/width/skew): few, interpretable      -> rank plots
  - area[trace, peak]:      many, the estimands      -> forest
  - warp + noise (per trace): nuisance               -> forest
  - everything:             R-hat / ESS overview     -> scatter + scalar gate
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import arviz

if TYPE_CHECKING:
    import matplotlib.figure


def _figure(pc: Any) -> matplotlib.figure.Figure:
    """Extract the matplotlib Figure from an arviz 1.x PlotCollection."""
    return pc.viz["figure"].item()


def _present(idata: arviz.InferenceData, names: tuple[str, ...]) -> list[str]:
    """Subset of ``names`` actually present in the posterior."""
    data_vars = idata.posterior.data_vars  # type: ignore[attr-defined]
    return [n for n in names if n in data_vars]


def plot_warp(idata: arviz.InferenceData) -> matplotlib.figure.Figure:
    """Forest of per-trace warp + noise.

    Answers: are the time shifts/stretches small and zero-centred, and is any
    single trace an outlier? A large ``time_shift[trace]`` flags a problematic
    injection; an inflated ``noise[trace]`` flags model misfit on that trace.
    """
    names = _present(idata, ("time_shift", "time_stretch", "noise"))
    return _figure(arviz.plot_forest(idata, var_names=names))
