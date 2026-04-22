"""Chromatographic peak fitting module.

Sub-modules
-----------
- ``fitter``: Main :class:`Fitter` class (MCMC, area extraction).
- ``priors``: Window-geometry prior construction and FWHM diagnostics.
- ``baseline``: Linear baseline estimation.
- ``model``: NumPyro probabilistic model.
- ``visualize``: Posterior and diagnostic plots.
- ``shift``: Retention-time alignment via per-trace shift optimization.
- ``types``: Data types and hyperparameter configuration.
- ``utils``: Array utility functions.
"""

from .fitter import AreaRecord, Fitter
from .types import ModelHyperparams

__all__ = ["AreaRecord", "Fitter", "ModelHyperparams"]
