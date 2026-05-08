"""Chromatographic peak fitting module.

Foundations layer (in this rewrite):

- :mod:`chromhandler.fitting.preprocessing` — pad-to-common-axis, dt.
- :mod:`chromhandler.fitting.baseline` — per-trace OLS from baseline regions.
- :mod:`chromhandler.fitting.noise` — per-trace MAD noise estimation.
- :mod:`chromhandler.fitting.prepared_dataset` — immutable input bundle.

The previous higher-level surface (``Fitter``, ``ModelHyperparams``,
``priors``, ``model``) is being rewritten on the new foundations and is
intentionally not re-exported here. Modules that survive the rewrite
will be re-exported from this file once the follow-up plan lands.
"""
