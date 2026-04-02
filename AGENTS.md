# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Commands

```bash
# Install with dev dependencies (uses uv)
uv sync --dev

# Run all tests
pytest

# Run a single test file
pytest tests/test_peak_assignment.py

# Lint and format
ruff check .
ruff format .

# Type checking
pyright
```

## Architecture

`chromhandler` (formerly `chromatopy`) processes time-resolved chromatographic reaction data and converts it to EnzymeML format.

### Core Data Flow

1. **Readers** (`chromhandler/readers/`) parse raw chromatography files from various instruments (Agilent, Shimadzu, Thermo, Chromeleon, etc.) into the internal data model.

2. **Data Model** (`chromhandler/model.py`) — auto-generated Pydantic models with JSON-LD support:
   - `Measurement`: one injection, contains `Data` (reaction time/concentration) and a list of `Chromatogram`s
   - `Chromatogram`: raw signal arrays + a list of `Peak`s.
   - `Peak`: retention time, area, and optional metadata
   - `DataType`: `CALIBRATION` or `TIMECOURSE`

3. **Handler** (`chromhandler/handler.py`) is the user-facing entry point — a Pydantic `BaseModel` that holds measurements, molecules, proteins, and orchestrates calibration/quantification/EnzymeML export.

4. **Fitting** (`chromhandler/fitting/`) — Bayesian peak fitting using NumPyro/JAX:
   - `nu_bayes.py`: Main `Fitter` class — MCMC-based fitting using NUTS sampler
   - `peak_models.py`: Bi-skew-normal peak model (main + optional shoulder component)
   - `baseline.py`: Linear baseline priors and estimation
   - `data.py`: `PeakAnnotation` and `BaselineAnnotation` dataclasses for annotating chromatogram regions
   - `peak_features.py`: Feature extraction for prior construction (FWHM, KDE apex gate, skew priors)
   - `moments.py`: Statistical moment-based peak metrics
   - `shift.py`: Chromatogram alignment via retention-time shift correction

### Key Conventions

- The fitting module uses JAX/NumPyro and sets `numpyro.set_host_device_count(8)` at import time.
- Readers live in `chromhandler/readers/` and inherit from `AbstractReader` (`abstractreader.py`).
- Units are handled via `mdmodels.units.annotation.UnitDefinitionAnnot` from the `md-models` ecosystem.
- EnzymeML export uses `pyenzyme`.
