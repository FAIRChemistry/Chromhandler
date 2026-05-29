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

if TYPE_CHECKING:
    import arviz
    import matplotlib.figure


def _figure(pc: Any) -> matplotlib.figure.Figure:
    """Extract the matplotlib Figure from an arviz 1.x PlotCollection."""
    return pc.viz["figure"].item()


def _present(idata: arviz.InferenceData, names: tuple[str, ...]) -> list[str]:
    """Subset of ``names`` actually present in the posterior."""
    data_vars = idata.posterior.data_vars  # type: ignore[attr-defined]
    return [n for n in names if n in data_vars]
