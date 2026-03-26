"""Chromatographic peak fitting module.

Sub-modules
-----------
- ``better_fitter``: Main :class:`BetterFitter` class (MCMC, subsets, area extraction).
- ``subsets``: :class:`SubsetSpec` and :class:`AreaRecord` data types.
- ``priors``: Window-geometry prior construction and FWHM diagnostics.
- ``baseline``: Linear baseline estimation.
- ``better_model``: NumPyro probabilistic model.
- ``better_visualize``: Posterior and diagnostic plots.

Legacy
------
EMG Peak Fitting Module for Chromatography Data.

This module provides efficient JAX-based fitting of Exponentially Modified Gaussian (EMG)
peaks to chromatography data, with support for:

1. **Parallel batch processing** - fit multiple spectra simultaneously
2. **Variable-length regions** - handle ROIs of different sizes
3. **Multi-component fitting** - automatic forward selection for overlapping peaks
4. **Statistical model selection** - AICc, BIC, and residual diagnostics

Main Functions
--------------
- fit_single_peak: Fit single EMG to one spectrum
- fit_batched_single_peaks_fixed_length: Parallel fitting for same-length regions
- fit_multiple_regions: Smart fitting for variable-length regions
- fit_emg_forward_selection: Automatic multi-component fitting

Example Usage
-------------
```python
from chromhandler.fitting import fit_single_peak, fit_multiple_regions

# Single peak fitting
result = fit_single_peak(x, y, maxiter=100)
print(f"AICc: {result['aicc']:.2f}, Converged: {result['converged']}")

# Multiple regions with variable lengths
regions = [
    {'x': x1, 'y': y1, 'id': 'peak1', 'n_components': 1},
    {'x': x2, 'y': y2, 'id': 'peak2', 'n_components': 2},
]
results = fit_multiple_regions(regions, maxiter=100)
```

Architecture
------------
- model.py: EMG mathematical functions and parameter packing
- residuals.py: Residual functions and information criteria
- initialization.py: Heuristic and dictionary-based initialization
- fit.py: Main fitting routines with parallelization
- stats.py: Forward selection and model comparison

Performance
-----------
- Single peak: ~0.1s for 200 data points
- Batched (100 spectra): ~1-2s with JIT compilation
- Multi-component: ~0.5-2s per region depending on K
"""

from .better_fitter import BetterFitter, PosteriorCurves
from .subsets import AreaRecord, Subset, SubsetSpec

__all__ = ["AreaRecord", "BetterFitter", "PosteriorCurves", "Subset", "SubsetSpec"]
