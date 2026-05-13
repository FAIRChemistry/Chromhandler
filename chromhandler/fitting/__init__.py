"""Bayesian skew-normal peak fitting layer."""

from __future__ import annotations

from chromhandler.fitting.fitter import FitResult, fit
from chromhandler.fitting.model import ModelConfig
from chromhandler.fitting.priors import PriorConfig

__all__ = ["FitResult", "ModelConfig", "PriorConfig", "fit"]
