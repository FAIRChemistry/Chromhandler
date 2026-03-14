"""Deprecated — use :meth:`Handler.to_enzymeml() <chromhandler.handler.Handler.to_enzymeml>` instead.

This module previously contained standalone functions for exporting
chromatographic data to EnzymeML format.  That functionality has been
re-implemented as ``Handler.to_enzymeml()`` which uses the new
``Sample``/``Chromatogram``/``Peak(area=Estimate(...))`` data model and
``LinearCalibration`` calibration pipeline.
"""

import warnings

warnings.warn(
    "chromhandler.enzymeml is deprecated and will be removed in version 1.0.0. "
    "Use Handler.to_enzymeml() instead.",
    DeprecationWarning,
    stacklevel=2,
)
