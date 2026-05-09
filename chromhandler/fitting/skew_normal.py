"""Pure-math skew-normal layer.

Implements the centred-parameter (CP) ↔ direct-parameter (DP) bijection,
density evaluation in both forms, and the derived quantities (mode, FWHM,
HWHM-ratio, asymmetry-to-γ₁ inversion) needed by the priors and posterior
layers. No NumPyro imports, no state, no side effects.

See ``docs/superpowers/specs/2026-05-07-skew-normal-fitter-rewrite-design.md``
§2 (math) and §7.1 (API).
"""

from __future__ import annotations

import math

# Skewness of the half-normal distribution = max |γ₁| achievable by any
# skew-normal. See spec §2.2.
GAMMA1_MAX: float = (
    ((4.0 - math.pi) / 2.0)
    * (math.sqrt(2.0 / math.pi) ** 3)
    / (1.0 - 2.0 / math.pi) ** 1.5
)

_B_CONST: float = math.sqrt(2.0 / math.pi)
