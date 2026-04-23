# Trace Statistics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-`Chromatogram` statistics record (starting with noise `sigma_noise` via DER_SNR) computed *once* on the full untruncated trace, so downstream fitting code can consume a signal-scale-aware noise estimate instead of magic-number fallbacks.

**Architecture:** A new pure-function module `chromhandler/trace_statistics.py` exposes a `TraceStatistics` Pydantic model and a `compute_trace_statistics(time, signal) -> TraceStatistics` function (no handler/fitter dependencies). `Chromatogram` gains an optional `trace_stats` field. `Handler` grows a thin `compute_trace_statistics()` orchestrator that iterates samples/chromatograms and fills the field. `Handler.cut_chromatograms()` calls the orchestrator defensively *before* truncating, so stats always reflect the full trace. `Fitter.from_handler()` also calls it defensively as a safety net for callers who never truncated. This plan delivers only `sigma_noise`; adding more fields (dt, quantiles, drift, quantization) and wiring into `priors.py`/`baseline.py` is follow-up work.

**Tech Stack:** Python 3.11+, Pydantic v2 (`BaseModel` + `ConfigDict`), NumPy, pytest, ruff, pyright, `uv` venv.

---

## File Structure

**Create:**
- `chromhandler/trace_statistics.py` — `TraceStatistics` model + `compute_trace_statistics()` function. Pure NumPy, no chromhandler internal imports (avoids circular deps with `model.py`).
- `tests/unit/handler/test_trace_statistics.py` — unit tests for the pure function + Pydantic model.

**Modify:**
- `chromhandler/model.py` — add `trace_stats: TraceStatistics | None = None` field to `Chromatogram` (line ~293, after `reaction_time_unit`).
- `chromhandler/handler.py` — add `Handler.compute_trace_statistics(overwrite: bool = False)` method; invoke it at top of `cut_chromatograms()` (line 1373).
- `chromhandler/fitting/fitter.py` — defensive call `handler.compute_trace_statistics(overwrite=False)` at top of `from_handler()` (line 342).

**Test:**
- `tests/unit/handler/test_trace_statistics.py` (new) — DER_SNR correctness, pure-function contract, Pydantic validation.
- `tests/unit/handler/test_handler_basics.py` (extend) — Handler orchestrator, `cut_chromatograms` integration, JSON round-trip.
- `tests/unit/fitting/test_fitter_inputs.py` (extend) — `from_handler` defensive call.

---

## Task 1: Module skeleton + failing DER_SNR test

**Files:**
- Create: `chromhandler/trace_statistics.py`
- Create: `tests/unit/handler/test_trace_statistics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/handler/test_trace_statistics.py`:

```python
"""Unit tests for chromhandler.trace_statistics."""

from __future__ import annotations

import numpy as np
import pytest

from chromhandler.trace_statistics import TraceStatistics, compute_trace_statistics


def test_der_snr_recovers_known_gaussian_noise_on_flat_trace() -> None:
    """DER_SNR on a flat trace + iid Gaussian noise recovers sigma to ~5%."""
    rng = np.random.default_rng(0)
    n = 5000
    true_sigma = 2.5
    time = np.linspace(0.0, 10.0, n)
    signal = 100.0 + rng.normal(0.0, true_sigma, size=n)

    stats = compute_trace_statistics(time, signal)

    assert isinstance(stats, TraceStatistics)
    assert stats.sigma_noise == pytest.approx(true_sigma, rel=0.05)


def test_der_snr_unaffected_by_linear_baseline() -> None:
    """2nd-difference operator annihilates linear baselines."""
    rng = np.random.default_rng(1)
    n = 5000
    true_sigma = 1.0
    time = np.linspace(0.0, 10.0, n)
    signal = 50.0 + 3.0 * time + rng.normal(0.0, true_sigma, size=n)

    stats = compute_trace_statistics(time, signal)

    assert stats.sigma_noise == pytest.approx(true_sigma, rel=0.07)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/handler/test_trace_statistics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chromhandler.trace_statistics'`.

- [ ] **Step 3: Create the module with minimal DER_SNR implementation**

Create `chromhandler/trace_statistics.py`:

```python
"""Per-chromatogram trace statistics (noise, scale, sampling).

This module is pure NumPy and has no chromhandler-internal imports, so it
is safe to import from :mod:`chromhandler.model` without circularity.

Only :attr:`TraceStatistics.sigma_noise` is populated today; further
fields (``dt_median``, quantiles, drift, quantization) land in a follow-up.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field

_MAD_TO_SIGMA = 1.4826
_DER_SNR_DENOM = np.sqrt(6.0)


class TraceStatistics(BaseModel):
    """Summary statistics computed once on a full, untruncated trace."""

    model_config: ConfigDict = ConfigDict(validate_assignment=True)  # type: ignore

    sigma_noise: float = Field(
        ...,
        description=(
            "DER_SNR noise estimate (1.4826 * median(|d2|) / sqrt(6)) "
            "computed on the full trace. Robust to linear baselines and "
            "isolated peak curvature."
        ),
    )


def compute_trace_statistics(time: ArrayLike, signal: ArrayLike) -> TraceStatistics:
    """Compute trace-level statistics on a *full, untruncated* trace.

    Args:
        time:   1-D retention-time axis (minutes). Unused for sigma_noise
                but accepted now so the signature is stable for follow-up fields.
        signal: 1-D signal values, same length as ``time``.

    Returns:
        A :class:`TraceStatistics` with ``sigma_noise`` populated via DER_SNR.
    """
    y = np.asarray(signal, dtype=float)
    if y.ndim != 1:
        raise ValueError("signal must be 1-D.")
    if y.size < 3:
        raise ValueError("signal must have at least 3 finite samples for DER_SNR.")

    sigma = _der_snr(y)
    return TraceStatistics(sigma_noise=float(sigma))


def _der_snr(y: np.ndarray) -> float:
    """DER_SNR estimator: sigma = 1.4826 * median(|d2|) / sqrt(6)."""
    d2 = y[2:] - 2.0 * y[1:-1] + y[:-2]
    d2 = d2[np.isfinite(d2)]
    if d2.size == 0:
        raise ValueError("No finite 2nd-differences; trace is all-NaN or too short.")
    return _MAD_TO_SIGMA * float(np.median(np.abs(d2))) / _DER_SNR_DENOM
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/handler/test_trace_statistics.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Lint + type-check**

```bash
uv run ruff check chromhandler/trace_statistics.py tests/unit/handler/test_trace_statistics.py
uv run pyright chromhandler/trace_statistics.py tests/unit/handler/test_trace_statistics.py
```
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add chromhandler/trace_statistics.py tests/unit/handler/test_trace_statistics.py
git commit -m "feat(trace-stats): add TraceStatistics model with DER_SNR sigma_noise"
```

---

## Task 2: Input validation — NaN handling + short traces

**Files:**
- Modify: `tests/unit/handler/test_trace_statistics.py`
- Modify: `chromhandler/trace_statistics.py` (only if tests fail after step 2)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/handler/test_trace_statistics.py`:

```python
def test_rejects_non_1d_signal() -> None:
    with pytest.raises(ValueError, match="1-D"):
        compute_trace_statistics(np.zeros((4, 4)), np.zeros((4, 4)))


def test_rejects_too_short_signal() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        compute_trace_statistics(np.array([0.0, 1.0]), np.array([0.0, 1.0]))


def test_tolerates_interior_nans() -> None:
    """NaNs in the signal drop out of the 2nd-difference median, not crash."""
    rng = np.random.default_rng(42)
    n = 5000
    y = 10.0 + rng.normal(0.0, 1.5, size=n)
    y[1000:1010] = np.nan
    stats = compute_trace_statistics(np.linspace(0, 1, n), y)
    assert stats.sigma_noise == pytest.approx(1.5, rel=0.07)


def test_rejects_all_nan_signal() -> None:
    with pytest.raises(ValueError, match="No finite"):
        compute_trace_statistics(np.linspace(0, 1, 100), np.full(100, np.nan))
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/unit/handler/test_trace_statistics.py -v`
Expected: all 6 tests PASS (the existing guards already cover these cases).

- [ ] **Step 3: Lint + type-check**

```bash
uv run ruff check tests/unit/handler/test_trace_statistics.py
uv run pyright tests/unit/handler/test_trace_statistics.py
```
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/handler/test_trace_statistics.py
git commit -m "test(trace-stats): lock in NaN / short-trace / non-1d guards"
```

---

## Task 3: Wire `trace_stats` field into `Chromatogram`

**Files:**
- Modify: `chromhandler/model.py:293` (add field after `reaction_time_unit`)
- Modify: `tests/unit/handler/test_handler_basics.py` (extend with round-trip test)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/handler/test_handler_basics.py`:

```python
def test_chromatogram_roundtrips_trace_stats_through_json() -> None:
    """Chromatogram.trace_stats serialises + deserialises without loss."""
    from chromhandler.trace_statistics import TraceStatistics

    chrom = Chromatogram(
        id="c0",
        sample_id="s0",
        signal=[1.0, 2.0, 3.0, 2.0, 1.0],
        time=[0.0, 0.1, 0.2, 0.3, 0.4],
        trace_stats=TraceStatistics(sigma_noise=0.75),
    )

    dumped = chrom.model_dump_json()
    restored = Chromatogram.model_validate_json(dumped)

    assert restored.trace_stats is not None
    assert restored.trace_stats.sigma_noise == pytest.approx(0.75)


def test_chromatogram_defaults_trace_stats_to_none() -> None:
    chrom = Chromatogram(id="c0", sample_id="s0")
    assert chrom.trace_stats is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/handler/test_handler_basics.py::test_chromatogram_roundtrips_trace_stats_through_json tests/unit/handler/test_handler_basics.py::test_chromatogram_defaults_trace_stats_to_none -v`
Expected: FAIL (`Chromatogram` has no `trace_stats` field).

- [ ] **Step 3: Add the field to `Chromatogram`**

In `chromhandler/model.py`, at the top add a local forward-ref-safe import:

```python
from chromhandler.trace_statistics import TraceStatistics
```

Then insert this field in the `Chromatogram` model, immediately after `reaction_time_unit` (around line 304, before the `# JSON-LD fields` comment):

```python
    trace_stats: TraceStatistics | None = Field(
        default=None,
        description="""Per-trace statistics (noise, scale, sampling) computed
        on the full untruncated signal. Populated lazily by
        ``Handler.compute_trace_statistics`` and before
        ``Handler.cut_chromatograms``.""",
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/handler/test_handler_basics.py -v`
Expected: new tests PASS, prior tests still PASS.

- [ ] **Step 5: Lint + type-check**

```bash
uv run ruff check chromhandler/model.py tests/unit/handler/test_handler_basics.py
uv run pyright chromhandler/model.py tests/unit/handler/test_handler_basics.py
```
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add chromhandler/model.py tests/unit/handler/test_handler_basics.py
git commit -m "feat(model): add Chromatogram.trace_stats field (optional)"
```

---

## Task 4: `Handler.compute_trace_statistics()` orchestrator

**Files:**
- Modify: `chromhandler/handler.py` (add method, near other chromatogram-iterating methods — before `cut_chromatograms` at line 1356)
- Modify: `tests/unit/handler/test_handler_basics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/handler/test_handler_basics.py`:

```python
def _handler_with_noisy_chromatograms(
    *, n_samples: int = 2, n_points: int = 4000, true_sigma: float = 1.2, seed: int = 0
) -> Handler:
    rng = np.random.default_rng(seed)
    handler = Handler(id="h", name="test")
    for i in range(n_samples):
        time = np.linspace(0.0, 10.0, n_points)
        signal = 100.0 + rng.normal(0.0, true_sigma, size=n_points)
        chrom = Chromatogram(
            id=f"c{i}", sample_id=f"s{i}", time=time.tolist(), signal=signal.tolist()
        )
        handler.samples.append(Sample(id=f"s{i}", chromatograms=[chrom]))
    return handler


def test_handler_compute_trace_statistics_fills_every_chromatogram() -> None:
    import numpy as np  # noqa: F401 used inside helper

    handler = _handler_with_noisy_chromatograms(true_sigma=1.2)
    handler.compute_trace_statistics()

    for sample in handler.samples:
        for chrom in sample.chromatograms:
            assert chrom.trace_stats is not None
            assert chrom.trace_stats.sigma_noise == pytest.approx(1.2, rel=0.07)


def test_handler_compute_trace_statistics_skips_existing_by_default() -> None:
    from chromhandler.trace_statistics import TraceStatistics

    handler = _handler_with_noisy_chromatograms()
    sentinel = TraceStatistics(sigma_noise=999.0)
    handler.samples[0].chromatograms[0].trace_stats = sentinel

    handler.compute_trace_statistics()

    assert handler.samples[0].chromatograms[0].trace_stats is sentinel


def test_handler_compute_trace_statistics_overwrite_recomputes() -> None:
    from chromhandler.trace_statistics import TraceStatistics

    handler = _handler_with_noisy_chromatograms(true_sigma=2.0)
    handler.samples[0].chromatograms[0].trace_stats = TraceStatistics(sigma_noise=999.0)

    handler.compute_trace_statistics(overwrite=True)

    stats = handler.samples[0].chromatograms[0].trace_stats
    assert stats is not None
    assert stats.sigma_noise == pytest.approx(2.0, rel=0.07)
```

Also add the `numpy` import at the top of `test_handler_basics.py` if not already present:

```python
import numpy as np
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/handler/test_handler_basics.py -v -k compute_trace_statistics`
Expected: FAIL with `AttributeError: 'Handler' object has no attribute 'compute_trace_statistics'`.

- [ ] **Step 3: Add the orchestrator method to `Handler`**

In `chromhandler/handler.py`, insert this method directly **before** `cut_chromatograms` (line 1356):

```python
    def compute_trace_statistics(self, *, overwrite: bool = False) -> None:
        """Populate ``trace_stats`` on every chromatogram in the handler.

        Stats are computed on the *full, untruncated* signal, so call this
        before :meth:`cut_chromatograms` (which itself invokes this method
        defensively).

        Args:
            overwrite: When ``False`` (default), chromatograms that already
                have ``trace_stats`` are skipped. When ``True``, every
                chromatogram is recomputed.
        """
        import numpy as np

        from .trace_statistics import compute_trace_statistics

        for sample in self.samples:
            for chrom in sample.chromatograms:
                if chrom.trace_stats is not None and not overwrite:
                    continue
                if not chrom.time or not chrom.signal:
                    continue
                chrom.trace_stats = compute_trace_statistics(
                    np.asarray(chrom.time, dtype=float),
                    np.asarray(chrom.signal, dtype=float),
                )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/handler/test_handler_basics.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Lint + type-check**

```bash
uv run ruff check chromhandler/handler.py tests/unit/handler/test_handler_basics.py
uv run pyright chromhandler/handler.py tests/unit/handler/test_handler_basics.py
```
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add chromhandler/handler.py tests/unit/handler/test_handler_basics.py
git commit -m "feat(handler): add compute_trace_statistics orchestrator"
```

---

## Task 5: Trigger stats computation inside `cut_chromatograms`

**Files:**
- Modify: `chromhandler/handler.py:1374` (first line inside `cut_chromatograms`)
- Modify: `tests/unit/handler/test_handler_basics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/handler/test_handler_basics.py`:

```python
def test_cut_chromatograms_captures_stats_on_full_trace() -> None:
    """Stats must be taken on the *full* trace, not the truncated one."""
    handler = _handler_with_noisy_chromatograms(n_points=4000, true_sigma=1.0)

    # Cut away 99% of the trace — a DER_SNR computed *after* truncation
    # would have way too few samples to be reliable.
    handler.cut_chromatograms([(0.0, 0.1)])

    for sample in handler.samples:
        for chrom in sample.chromatograms:
            assert chrom.trace_stats is not None
            # Full-trace DER_SNR should have recovered the true sigma.
            assert chrom.trace_stats.sigma_noise == pytest.approx(1.0, rel=0.1)
            # Truncation still happened.
            assert len(chrom.time) < 4000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/handler/test_handler_basics.py::test_cut_chromatograms_captures_stats_on_full_trace -v`
Expected: FAIL with `AssertionError: assert None is not None` (`trace_stats` never populated).

- [ ] **Step 3: Call the orchestrator at the top of `cut_chromatograms`**

In `chromhandler/handler.py`, modify the body of `cut_chromatograms`. Replace:

```python
        norm = self._normalize_cut_ranges(ranges)
        for sample in self.samples:
            for chrom in sample.chromatograms:
                self._cut_chromatogram(chrom, norm)
```

with:

```python
        # Freeze full-trace stats before we drop samples. No-op if already
        # populated by an earlier call.
        self.compute_trace_statistics(overwrite=False)

        norm = self._normalize_cut_ranges(ranges)
        for sample in self.samples:
            for chrom in sample.chromatograms:
                self._cut_chromatogram(chrom, norm)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/handler/ -v`
Expected: all tests PASS (new test + existing suite).

- [ ] **Step 5: Lint + type-check**

```bash
uv run ruff check chromhandler/handler.py tests/unit/handler/test_handler_basics.py
uv run pyright chromhandler/handler.py tests/unit/handler/test_handler_basics.py
```
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add chromhandler/handler.py tests/unit/handler/test_handler_basics.py
git commit -m "feat(handler): capture trace stats before cut_chromatograms truncates"
```

---

## Task 6: Defensive call in `Fitter.from_handler`

**Files:**
- Modify: `chromhandler/fitting/fitter.py:342` (top of `from_handler` body)
- Modify: `tests/unit/fitting/test_fitter_inputs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/fitting/test_fitter_inputs.py` (import `Handler`, `Sample`, `Chromatogram`, and `numpy` if not already present):

```python
def test_from_handler_populates_trace_stats_if_missing() -> None:
    """Users who never call cut_chromatograms still get trace_stats."""
    import numpy as np

    from chromhandler.fitting.fitter import Fitter
    from chromhandler.handler import Handler
    from chromhandler.model import Chromatogram, Sample

    rng = np.random.default_rng(7)
    n = 4000
    handler = Handler(id="h", name="test")
    for i in range(2):
        time = np.linspace(0.0, 10.0, n)
        signal = 100.0 + rng.normal(0.0, 0.8, size=n)
        handler.samples.append(
            Sample(
                id=f"s{i}",
                chromatograms=[
                    Chromatogram(
                        id=f"c{i}", sample_id=f"s{i}",
                        time=time.tolist(), signal=signal.tolist(),
                    )
                ],
            )
        )

    assert all(
        c.trace_stats is None for s in handler.samples for c in s.chromatograms
    )

    _ = Fitter.from_handler(handler)

    for sample in handler.samples:
        for chrom in sample.chromatograms:
            assert chrom.trace_stats is not None
            assert chrom.trace_stats.sigma_noise == pytest.approx(0.8, rel=0.1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/fitting/test_fitter_inputs.py::test_from_handler_populates_trace_stats_if_missing -v`
Expected: FAIL (`trace_stats` stays `None`).

- [ ] **Step 3: Add the defensive call**

In `chromhandler/fitting/fitter.py`, inside `from_handler` (after the docstring, before the `samples = [...]` filter, around line 342), insert:

```python
        # Ensure every chromatogram has full-trace stats before we read
        # signal arrays. No-op if already populated.
        handler.compute_trace_statistics(overwrite=False)
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/unit/fitting/test_fitter_inputs.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Lint + type-check**

```bash
uv run ruff check chromhandler/fitting/fitter.py tests/unit/fitting/test_fitter_inputs.py
uv run pyright chromhandler/fitting/fitter.py tests/unit/fitting/test_fitter_inputs.py
```
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add chromhandler/fitting/fitter.py tests/unit/fitting/test_fitter_inputs.py
git commit -m "feat(fitter): ensure trace stats populated in Fitter.from_handler"
```

---

## Task 7: Full-suite smoke + final commit

**Files:** none (verification only).

- [ ] **Step 1: Run the full unit suite**

Run: `uv run pytest tests/unit -v`
Expected: all tests PASS.

- [ ] **Step 2: Run integration smoke**

Run: `uv run pytest tests/integration -x -q`
Expected: all tests PASS. (No integration test should have been broken — no public API was renamed or removed.)

- [ ] **Step 3: Lint + type-check the full touched set**

```bash
uv run ruff check chromhandler tests
uv run pyright chromhandler/trace_statistics.py chromhandler/model.py chromhandler/handler.py chromhandler/fitting/fitter.py
```
Expected: no errors.

- [ ] **Step 4: If any follow-up commits are needed, commit them**

```bash
git status
# If clean → nothing to do.
# Otherwise: git add -p && git commit -m "chore(trace-stats): <what>"
```

---

## Out of scope (follow-up plans)

- Additional `TraceStatistics` fields: `dt_median`, `dt_mad`, `signal_q01`, `signal_q99`, `baseline_drift_rate`, `quantization_step`, `n_samples`, `time_span`, `finite_fraction`.
- Consuming `trace_stats.sigma_noise` inside `chromhandler/fitting/priors.py` (replacing first-diff MAD in-peak estimator) and `chromhandler/fitting/baseline.py` (replacing absolute-scale `_MIN_INTERCEPT_SCALE` / `_MIN_SLOPE_SCALE`).
- EnzymeML export of `trace_stats` (JSON-LD context entry).
