# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All Python execution uses the project venv via `uv`:

```bash
# Install with dev dependencies
uv sync --dev

# Run all tests (within project venv)
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .

# Type checking
uv run pyright
```

## Architecture

`chromhandler` processes time-resolved chromatographic reaction data and converts it to EnzymeML format. It also has a Bayesian peak fitting module that can be used to fit peaks to chromatograms.

### Core Data Flow

1. **Readers** (`chromhandler/readers/`) parse raw chromatography files from various instruments (Agilent, Shimadzu, Thermo, Chromeleon, etc.) into the internal data model.

2. **Data Model** (`chromhandler/model.py`) — Pydantic models with JSON-LD support:
   - `Measurement`: one injection, contains `Data` (reaction time/concentration) and a list of `Chromatogram`s
   - `Chromatogram`: raw signal arrays + a list of `Peak`s.
   - `Peak`: retention time, area, and optional metadata

3. **Handler** (`chromhandler/handler.py`) is the user-facing entry point — a Pydantic `BaseModel` that holds measurements, molecules, proteins, and orchestrates calibration/quantification/EnzymeML export.

4. **Fitting** (`chromhandler/fitting/`) — Bayesian peak fitting using NumPyro/JAX:
   - `priors.py`: `GeometricPeakPriors` dataclass and prior construction from window geometry (sampling frequency, peak bounds, baseline regions) — no FWHM measurement
   - `better_fitter.py`: `BetterFitter` class — data pipeline for organizing traces, computing priors, assembling model inputs
   - `better_model.py`: NumPyro Bayesian model — skew-normal peak model with optional shoulder component, noise priors, posterior inference
   - `nu_bayes.py`: Legacy `Fitter` class — older MCMC-based fitting (kept for backwards compatibility)
   - `baseline.py`: Linear baseline priors and estimation utilities
   - `moments.py`: Statistical moment-based peak metrics
   - `shift.py`: Chromatogram alignment via retention-time shift correction
   - `types.py`: Shared dataclasses (`PeakAnnotation`, `BaselineAnnotation`, `PeakStructure`)
   - `utils.py`: Helper utilities for peak/window manipulation

### Key Conventions

- **No mocks in tests** — tests exercise real behavior against actual databases or parsing logic. Integration tests hit live Neo4j/Milvus instances when available.
- **No backwards compatibility shims** — when concepts change, implement clean replacements. Clean up old code instead of maintaining dual paths.
- **Always run after edits**: `uv run ruff check <file>`, `uv run pyright <file>`, and relevant test files with `uv run pytest <test_file> -v`.
- **Always use Context7 MCP** when using any third-party Python library to get current API documentation.
- The fitting module uses JAX/NumPyro and sets `numpyro.set_host_device_count(8)` at import time.
- Readers live in `chromhandler/readers/` and inherit from `AbstractReader` (`abstractreader.py`).
- Units are handled via `mdmodels.units.annotation.UnitDefinitionAnnot` from the `md-models` ecosystem.
- EnzymeML export uses `pyenzyme`.
