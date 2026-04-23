# Noise-Estimation Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace four signal-scale-dependent magic numbers in the fitting pipeline with per-trace `sigma_noise` from `Chromatogram.trace_stats`, making every noise-aware prior physically grounded in the DER_SNR estimate.

**Architecture:** A new `Fitter.trace_sigma_noise: NDArray[n_trace]` attribute becomes the single fitting-side handle for per-trace noise. `Fitter.__init__` auto-computes it from signal rows (direct construction path). `Fitter.from_handler` strictly requires `chrom.trace_stats.sigma_noise` on every surviving chromatogram, raising with a listed remediation otherwise. Three downstream consumers (`estimate_baseline`, `build_peak_priors`, `Fitter.noise_prior`) receive the array as an explicit argument.

**Tech Stack:** NumPy, JAX, NumPyro, Pydantic v2, pytest. Project-standard tooling: `uv run pytest`, `uv run ruff check`, `uv run pyright`.

**Spec:** [`docs/specs/2026-04-23-noise-estimation-wiring-design.md`](../specs/2026-04-23-noise-estimation-wiring-design.md)

---

## File Structure

**Modified:**
- `chromhandler/fitting/fitter.py` — new `trace_sigma_noise` attribute, strict `from_handler`, thin `noise_prior()`, updated call sites to `estimate_baseline` and `build_peak_priors`.
- `chromhandler/fitting/baseline.py` — `estimate_baseline` takes per-trace `sigma_noise`; per-trace floors replace module constants `_MIN_INTERCEPT_SCALE` / `_MIN_SLOPE_SCALE`.
- `chromhandler/fitting/priors.py` — `_estimate_snr` rewrite; `build_peak_priors` adds required `sigma_noise` kwarg.
- `tests/unit/fitting/test_priors.py` — update all `build_peak_priors` and `_estimate_snr` call sites.
- `tests/integration/test_prior_pipeline.py` — update all `build_peak_priors` call sites.
- `tests/unit/fitting/test_fitter_inputs.py` — add `trace_sigma_noise.shape` assertion.
- `tests/unit/fitting/test_fitter_diagnostics.py` — add `trace_sigma_noise.shape` assertion.

**Created:**
- `tests/unit/fitting/test_noise_plumbing.py` — Fitter noise plumbing (init validation, `from_handler` strict mode, `noise_prior` passthrough).
- `tests/unit/fitting/test_baseline.py` — `estimate_baseline` per-trace floor behaviour.

---

## Task 1: Add `trace_sigma_noise` attribute with auto-compute default

Give `Fitter` a strictly-positive per-trace noise vector, auto-computed when not supplied. This is the foundational attribute every later task depends on.

**Files:**
- Modify: `chromhandler/fitting/fitter.py:98-140` (`Fitter.__init__`)
- Create: `tests/unit/fitting/test_noise_plumbing.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/fitting/test_noise_plumbing.py`:

```python
"""Unit tests for Fitter noise plumbing (trace_sigma_noise attribute)."""

from __future__ import annotations

import numpy as np
import pytest

from chromhandler.fitting.fitter import Fitter
from chromhandler.trace_statistics import compute_trace_statistics


def _noisy_matrix(
    n_trace: int = 3, n_time: int = 4000, sigma: float = 1.5, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    time = np.tile(np.linspace(0.0, 10.0, n_time), (n_trace, 1))
    signal = 100.0 + rng.normal(0.0, sigma, size=(n_trace, n_time))
    return time, signal


@pytest.mark.unit
def test_init_auto_computes_trace_sigma_noise_from_rows() -> None:
    """When trace_sigma_noise is not supplied, __init__ auto-computes per row."""
    time, signal = _noisy_matrix(n_trace=3, sigma=1.5)
    fitter = Fitter(time, signal)

    assert fitter.trace_sigma_noise.shape == (3,)
    assert fitter.trace_sigma_noise.dtype == np.float64
    # Values match compute_trace_statistics called directly on each row.
    for t in range(3):
        expected = compute_trace_statistics(time[t], signal[t]).sigma_noise
        assert fitter.trace_sigma_noise[t] == pytest.approx(expected, rel=1e-10)


@pytest.mark.unit
def test_init_accepts_explicit_trace_sigma_noise() -> None:
    """Explicit trace_sigma_noise is stored verbatim."""
    time, signal = _noisy_matrix()
    explicit = np.array([1.1, 2.2, 3.3])
    fitter = Fitter(time, signal, trace_sigma_noise=explicit)

    np.testing.assert_array_equal(fitter.trace_sigma_noise, explicit)


@pytest.mark.unit
def test_init_rejects_wrong_shape_trace_sigma_noise() -> None:
    """trace_sigma_noise with mismatched length is rejected."""
    time, signal = _noisy_matrix(n_trace=3)
    with pytest.raises(ValueError, match="trace_sigma_noise must have length n_traces=3"):
        Fitter(time, signal, trace_sigma_noise=np.array([1.0, 2.0]))


@pytest.mark.unit
def test_init_rejects_non_finite_trace_sigma_noise() -> None:
    """Non-finite entries in trace_sigma_noise are rejected."""
    time, signal = _noisy_matrix(n_trace=2)
    with pytest.raises(ValueError, match="trace_sigma_noise must be finite and positive"):
        Fitter(time, signal, trace_sigma_noise=np.array([1.0, np.nan]))


@pytest.mark.unit
def test_init_rejects_non_positive_trace_sigma_noise() -> None:
    """Zero or negative entries in trace_sigma_noise are rejected."""
    time, signal = _noisy_matrix(n_trace=2)
    with pytest.raises(ValueError, match="trace_sigma_noise must be finite and positive"):
        Fitter(time, signal, trace_sigma_noise=np.array([1.0, 0.0]))


@pytest.mark.unit
def test_init_auto_compute_reraises_row_failure_with_index() -> None:
    """Auto-compute re-raises compute_trace_statistics failures with row index."""
    # Row 1 has <3 finite samples — compute_trace_statistics will raise.
    time = np.tile(np.linspace(0.0, 1.0, 10), (2, 1))
    signal = np.vstack([np.linspace(100.0, 101.0, 10), np.full(10, np.nan)])
    with pytest.raises(ValueError, match="trace row 1"):
        Fitter(time, signal)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/fitting/test_noise_plumbing.py -v`
Expected: FAIL — `AttributeError: 'Fitter' object has no attribute 'trace_sigma_noise'` and unknown kwarg `trace_sigma_noise`.

- [ ] **Step 3: Modify `Fitter.__init__` to accept and validate `trace_sigma_noise`**

Modify `chromhandler/fitting/fitter.py` — add `trace_sigma_noise` kwarg to `__init__` signature and populate the new attribute after `_validate()`:

```python
    def __init__(
        self,
        time: NDArray[np.float64],
        signal: NDArray[np.float64],
        *,
        peaks: list[PeakAnnotation] | None = None,
        baselines: list[BaselineAnnotation] | None = None,
        trace_sample_ids: list[str] | None = None,
        trace_chromatogram_ids: list[str] | None = None,
        trace_sigma_noise: NDArray[np.float64] | None = None,
        hyperparams: ModelHyperparams | None = None,
    ) -> None:
        self.time = np.asarray(time, dtype=float)
        self.signal = np.asarray(signal, dtype=float)
        self.peaks: list[PeakAnnotation] = list(peaks) if peaks else []
        self.baselines: list[BaselineAnnotation] = list(baselines) if baselines else []
        self._validate()

        # Optional per-trace metadata (set by from_handler()).
        if trace_sample_ids is not None and len(trace_sample_ids) != self.n_traces:
            raise ValueError(
                f"trace_sample_ids must have length n_traces={self.n_traces}, got {len(trace_sample_ids)}."
            )
        if trace_chromatogram_ids is not None and len(trace_chromatogram_ids) != self.n_traces:
            raise ValueError(
                f"trace_chromatogram_ids must have length n_traces={self.n_traces}, "
                f"got {len(trace_chromatogram_ids)}."
            )
        self.trace_sample_ids: NDArray[Any] | None = (
            np.asarray(trace_sample_ids, dtype=object) if trace_sample_ids is not None else None
        )
        self.trace_chromatogram_ids: NDArray[Any] | None = (
            np.asarray(trace_chromatogram_ids, dtype=object) if trace_chromatogram_ids is not None else None
        )

        self.trace_sigma_noise: NDArray[np.float64] = self._resolve_trace_sigma_noise(trace_sigma_noise)

        self.hyperparams: ModelHyperparams = hyperparams if hyperparams is not None else ModelHyperparams()

        self.shift_samples: NDArray[np.float64] | None = None  # [n_trace] shifts in samples
        self.shift_time: NDArray[np.float64] | None = None  # [n_trace] shifts in time units

        # Inference attributes (set by _run_mcmc())
        self.mcmc: MCMC | None = None
        self.samples: dict[str, Any] | None = None
        self._posterior: InferenceData | None = None
```

Then add this helper method inside the class, just after `_validate`:

```python
    def _resolve_trace_sigma_noise(
        self, supplied: NDArray[np.float64] | None
    ) -> NDArray[np.float64]:
        """Return per-trace sigma_noise, auto-computing from signal rows when missing."""
        from chromhandler.trace_statistics import compute_trace_statistics

        if supplied is not None:
            arr = np.asarray(supplied, dtype=float)
            if arr.shape != (self.n_traces,):
                raise ValueError(
                    f"trace_sigma_noise must have length n_traces={self.n_traces}, got shape {arr.shape}."
                )
            if not np.all(np.isfinite(arr)) or not np.all(arr > 0.0):
                raise ValueError("trace_sigma_noise must be finite and positive for every trace.")
            return arr

        out = np.empty(self.n_traces, dtype=float)
        for t in range(self.n_traces):
            try:
                out[t] = compute_trace_statistics(
                    np.asarray(self.time[t], dtype=float),
                    np.asarray(self.signal[t], dtype=float),
                ).sigma_noise
            except ValueError as exc:
                raise ValueError(f"trace row {t}: {exc}") from exc
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/fitting/test_noise_plumbing.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check chromhandler/fitting/fitter.py tests/unit/fitting/test_noise_plumbing.py`
Run: `uv run pyright chromhandler/fitting/fitter.py tests/unit/fitting/test_noise_plumbing.py`
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add chromhandler/fitting/fitter.py tests/unit/fitting/test_noise_plumbing.py
git commit -m "feat(fitter): add trace_sigma_noise attribute with auto-compute default"
```

---

## Task 2: Strict `from_handler` — require `trace_stats` on every surviving chromatogram

Promote `Fitter.from_handler` from "defensively compute stats" to "strictly require stats, or raise with a listed remediation." Populate `trace_sigma_noise` from each chromatogram's `trace_stats`.

**Files:**
- Modify: `chromhandler/fitting/fitter.py:308-370` (`Fitter.from_handler`)
- Modify: `tests/unit/fitting/test_noise_plumbing.py` (extend)

- [ ] **Step 1: Write failing tests (extend `test_noise_plumbing.py`)**

Append these tests to `tests/unit/fitting/test_noise_plumbing.py`:

```python
# ---------------------------------------------------------------------------
# from_handler tests
# ---------------------------------------------------------------------------


def _handler_with_noisy_traces(
    n_samples: int = 2, n_points: int = 4000, sigma: float = 1.0, seed: int = 0
):
    from chromhandler.handler import Handler
    from chromhandler.model import Chromatogram, Sample

    rng = np.random.default_rng(seed)
    handler = Handler()
    for i in range(n_samples):
        time = np.linspace(0.0, 10.0, n_points)
        signal = 100.0 + rng.normal(0.0, sigma, size=n_points)
        chrom = Chromatogram(
            id=f"c{i}", sample_id=f"s{i}", time=time.tolist(), signal=signal.tolist()
        )
        handler.samples.append(Sample(id=f"s{i}", chromatograms=[chrom]))
    return handler


@pytest.mark.unit
def test_from_handler_populates_trace_sigma_noise_from_trace_stats() -> None:
    """Fitter.from_handler copies sigma_noise from chrom.trace_stats in trace order."""
    handler = _handler_with_noisy_traces(n_samples=2, sigma=1.2)
    fitter = Fitter.from_handler(handler)

    assert fitter.trace_sigma_noise.shape == (2,)
    for t, sample in enumerate(handler.samples):
        chrom = sample.chromatograms[0]
        assert chrom.trace_stats is not None
        assert fitter.trace_sigma_noise[t] == pytest.approx(chrom.trace_stats.sigma_noise)


@pytest.mark.unit
def test_from_handler_raises_when_any_trace_stats_missing() -> None:
    """Fitter.from_handler raises ValueError listing chromatograms without trace_stats."""
    from chromhandler.handler import Handler
    from chromhandler.model import Chromatogram, Sample

    # c1 is all-NaN → compute_trace_statistics silently leaves trace_stats=None.
    healthy = Chromatogram(
        id="c0",
        sample_id="s0",
        time=np.linspace(0.0, 10.0, 1000).tolist(),
        signal=(100.0 + np.random.default_rng(0).normal(0.0, 1.0, size=1000)).tolist(),
    )
    bad = Chromatogram(
        id="c1",
        sample_id="s0",
        time=[0.0, 0.1, 0.2, 0.3],
        signal=[float("nan")] * 4,
    )
    handler = Handler()
    handler.samples.append(Sample(id="s0", chromatograms=[healthy, bad]))

    with pytest.raises(ValueError, match=r"missing trace_stats.*c1"):
        Fitter.from_handler(handler)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/fitting/test_noise_plumbing.py::test_from_handler_populates_trace_sigma_noise_from_trace_stats tests/unit/fitting/test_noise_plumbing.py::test_from_handler_raises_when_any_trace_stats_missing -v`
Expected: FAIL — `trace_sigma_noise` is auto-computed from signal (not from `trace_stats`), and the missing-stats case does not currently raise.

- [ ] **Step 3: Modify `Fitter.from_handler`**

Replace the body of `from_handler` in `chromhandler/fitting/fitter.py` starting at the line `handler.compute_trace_statistics(overwrite=False)` down through the final `return cls(...)`:

```python
        # Ensure every chromatogram has full-trace stats before we read
        # signal arrays. No-op if already populated.
        handler.compute_trace_statistics(overwrite=False)

        samples = [
            s
            for s in handler.samples
            if sample_ids is None or s.id in sample_ids
        ]
        if not samples:
            raise ValueError("No matching samples found in handler.")

        # Strict check: every included chromatogram must carry trace_stats.
        # Degenerate (all-NaN / <3-point) chromatograms are skipped silently
        # by compute_trace_statistics and surface here.
        missing = [
            c.id
            for s in samples
            for c in s.chromatograms
            if c.trace_stats is None
        ]
        if missing:
            raise ValueError(
                "Fitter.from_handler: chromatograms missing trace_stats after "
                f"compute_trace_statistics: {missing}. These traces are likely "
                "all-NaN or too short — drop them via handler.cut_chromatograms "
                "or filter upstream."
            )

        time_lists: list[list[float]] = [c.time for s in samples for c in s.chromatograms]
        signal_lists: list[list[float]] = [c.signal for s in samples for c in s.chromatograms]
        trace_sample_ids: list[str] = [s.id for s in samples for _c in s.chromatograms]
        trace_chrom_ids: list[str] = [c.id for s in samples for c in s.chromatograms]
        trace_sigma_noise = np.array(
            [c.trace_stats.sigma_noise for s in samples for c in s.chromatograms],
            dtype=float,
        )

        time_arr, signal_arr = pad_traces(time_lists, signal_lists)

        inherited_peaks = list(handler.peak_annotations.values())

        return cls(
            time_arr,
            signal_arr,
            peaks=inherited_peaks or None,
            baselines=None,
            trace_sample_ids=trace_sample_ids,
            trace_chromatogram_ids=trace_chrom_ids,
            trace_sigma_noise=trace_sigma_noise,
        )
```

Note: `c.trace_stats.sigma_noise` is safe at this point — the `missing` check above guarantees `c.trace_stats is not None` for every included chromatogram. Pyright may flag it; add `# type: ignore[union-attr]` if needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/fitting/test_noise_plumbing.py -v`
Expected: PASS (8 tests total — 6 from Task 1 + 2 new).

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check chromhandler/fitting/fitter.py tests/unit/fitting/test_noise_plumbing.py`
Run: `uv run pyright chromhandler/fitting/fitter.py tests/unit/fitting/test_noise_plumbing.py`
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add chromhandler/fitting/fitter.py tests/unit/fitting/test_noise_plumbing.py
git commit -m "feat(fitter): strict from_handler requires trace_stats on every trace"
```

---

## Task 3: `Fitter.noise_prior()` becomes passthrough

Strip the MAD-in-baseline-regions logic and the `1e-3 * signal_range` numerical floor from `noise_prior()`. It now returns the per-trace `trace_sigma_noise` directly.

**Files:**
- Modify: `chromhandler/fitting/fitter.py:224-263` (`noise_prior`)
- Modify: `tests/unit/fitting/test_noise_plumbing.py` (extend)

- [ ] **Step 1: Write failing test (extend `test_noise_plumbing.py`)**

Append to `tests/unit/fitting/test_noise_plumbing.py`:

```python
@pytest.mark.unit
def test_noise_prior_is_trace_sigma_noise_passthrough() -> None:
    """Fitter.noise_prior() returns self.trace_sigma_noise unchanged."""
    time, signal = _noisy_matrix(n_trace=3, sigma=1.5)
    explicit = np.array([0.5, 1.0, 2.0])
    fitter = Fitter(time, signal, trace_sigma_noise=explicit)

    np.testing.assert_array_equal(fitter.noise_prior(), explicit)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/fitting/test_noise_plumbing.py::test_noise_prior_is_trace_sigma_noise_passthrough -v`
Expected: FAIL — current `noise_prior()` derives noise from MAD, not from `trace_sigma_noise`.

- [ ] **Step 3: Replace `Fitter.noise_prior` body**

Replace the full body of `Fitter.noise_prior` in `chromhandler/fitting/fitter.py` (lines ~224-263):

```python
    def noise_prior(self) -> NDArray[np.float64]:
        """Per-trace observation noise prior — equals ``self.trace_sigma_noise``.

        The DER_SNR estimate computed on the full untruncated signal
        (see :class:`chromhandler.trace_statistics.TraceStatistics`) is used
        directly. No additional floors or window-local estimators.

        Returns
        -------
        np.ndarray
            Shape ``[n_trace]``, strictly positive.
        """
        return self.trace_sigma_noise
```

Now check whether `baseline_to_mask` is still imported/used anywhere else in `fitter.py`; delete the import if this was its only user.

- [ ] **Step 4: Check and prune `baseline_to_mask` import if now unused**

Run: `uv run ruff check chromhandler/fitting/fitter.py`
If ruff reports `F401: 'baseline_to_mask' imported but unused`, remove the import. Otherwise leave it.

- [ ] **Step 5: Run the noise-plumbing tests**

Run: `uv run pytest tests/unit/fitting/test_noise_plumbing.py -v`
Expected: PASS (9 tests total).

- [ ] **Step 6: Run the full fitting test suite (sanity check — nothing else should break yet)**

Run: `uv run pytest tests/unit/fitting/ -v`
Expected: anything that hit the old `noise_prior()` MAD path still passes because `trace_sigma_noise` recovers the same value within statistical agreement. Failures here signal something subtler — stop and investigate before committing.

- [ ] **Step 7: Lint and type-check**

Run: `uv run ruff check chromhandler/fitting/fitter.py tests/unit/fitting/test_noise_plumbing.py`
Run: `uv run pyright chromhandler/fitting/fitter.py tests/unit/fitting/test_noise_plumbing.py`
Expected: both clean.

- [ ] **Step 8: Commit**

```bash
git add chromhandler/fitting/fitter.py tests/unit/fitting/test_noise_plumbing.py
git commit -m "refactor(fitter): noise_prior() is a trace_sigma_noise passthrough"
```

---

## Task 4: `estimate_baseline` takes per-trace `sigma_noise` — per-trace floors

Replace `_MIN_INTERCEPT_SCALE = 1.0` and `_MIN_SLOPE_SCALE = 1e-3` with per-trace, physics-grounded floors: intercept floor = `sigma_noise`, slope floor = `sigma_noise / time_span`.

**Files:**
- Modify: `chromhandler/fitting/baseline.py:14-18, 31-95, 243-251`
- Modify: `chromhandler/fitting/fitter.py:198-210` (`baseline_priors` — call site update)
- Create: `tests/unit/fitting/test_baseline.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/fitting/test_baseline.py`:

```python
"""Unit tests for estimate_baseline per-trace sigma_noise floors."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chromhandler.annotations import PeakAnnotation
from chromhandler.fitting.baseline import estimate_baseline


def _single_peak(rt_min: float = 4.0, rt_max: float = 6.0) -> PeakAnnotation:
    return PeakAnnotation(molecule_id="m", rt_min=rt_min, rt_max=rt_max)


@pytest.mark.unit
def test_estimate_baseline_requires_sigma_noise_kwarg() -> None:
    """estimate_baseline's sigma_noise kwarg is required (no default)."""
    t = np.linspace(0.0, 10.0, 200)
    time = jnp.asarray(np.tile(t, (2, 1)))
    signal = jnp.asarray(np.full((2, 200), 100.0))
    with pytest.raises(TypeError, match="sigma_noise"):
        estimate_baseline(time, signal, peaks=[_single_peak()])  # type: ignore[call-arg]


@pytest.mark.unit
def test_estimate_baseline_floors_use_per_trace_sigma_noise() -> None:
    """With a degenerate single-anchor fit, the floor equals sigma_noise per trace."""
    # Build a signal where every peak-window edge is *identical* across the window,
    # so the OLS fit has effectively zero standard error and must hit the floor.
    t = np.linspace(0.0, 10.0, 200)
    time = jnp.asarray(np.tile(t, (2, 1)))
    signal = jnp.asarray(np.full((2, 200), 100.0))
    sigma_noise = jnp.asarray([0.5, 1.5])
    time_span = float(t[-1] - t[0])  # 10.0

    priors = estimate_baseline(
        time, signal, peaks=[_single_peak()], sigma_noise=sigma_noise
    )

    # Intercept floor == sigma_noise (per trace).
    np.testing.assert_allclose(np.asarray(priors.intercept_scale), [0.5, 1.5], atol=1e-6)
    # Slope floor == sigma_noise / time_span.
    np.testing.assert_allclose(
        np.asarray(priors.slope_scale), [0.5 / time_span, 1.5 / time_span], atol=1e-6
    )


@pytest.mark.unit
def test_estimate_baseline_floors_tolerate_zero_time_span() -> None:
    """If a trace's time span is zero, slope floor falls back to sigma_noise."""
    # A single repeated time value makes time_span == 0.
    t = np.full(200, 2.0)
    time = jnp.asarray(np.tile(t, (1, 1)))
    signal = jnp.asarray(np.full((1, 200), 100.0))
    sigma_noise = jnp.asarray([0.7])

    priors = estimate_baseline(
        time, signal, peaks=[_single_peak(rt_min=1.5, rt_max=2.5)], sigma_noise=sigma_noise
    )
    np.testing.assert_allclose(np.asarray(priors.slope_scale), [0.7], atol=1e-6)


@pytest.mark.unit
def test_estimate_baseline_scales_above_floor_when_ols_is_informative() -> None:
    """When anchors give an informative OLS fit, scales can exceed the floor."""
    rng = np.random.default_rng(0)
    n = 2000
    t = np.linspace(0.0, 10.0, n)
    # Noisy baseline 100 + 2*t with σ=0.5 noise; peak window with content.
    signal_np = np.stack(
        [
            100.0 + 2.0 * t + rng.normal(0.0, 0.5, size=n),
            100.0 + 2.0 * t + rng.normal(0.0, 0.5, size=n),
        ]
    )
    time = jnp.asarray(np.tile(t, (2, 1)))
    signal = jnp.asarray(signal_np)
    # Pretend sigma_noise is tiny so floors cannot dominate.
    sigma_noise = jnp.asarray([1e-6, 1e-6])

    priors = estimate_baseline(
        time, signal, peaks=[_single_peak()], sigma_noise=sigma_noise
    )

    # With 2000 anchors of noise=0.5, OLS SE for intercept is well above 1e-6.
    assert float(priors.intercept_scale[0]) > 1e-3
    assert float(priors.slope_scale[0]) > 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/fitting/test_baseline.py -v`
Expected: FAIL — `sigma_noise` kwarg not accepted.

- [ ] **Step 3: Modify `chromhandler/fitting/baseline.py`**

Delete the two module-level constants and update the signature and helpers. Full file changes:

Remove lines 17-18:

```python
_MIN_INTERCEPT_SCALE = 1.0
_MIN_SLOPE_SCALE = 1e-3
```

Update `estimate_baseline` signature to add required `sigma_noise` kwarg, and replace the two `_scale_from_se` calls:

```python
def estimate_baseline(
    time: jax.Array,
    signal: jax.Array,
    *,
    peaks: list[PeakAnnotation],
    sigma_noise: jax.Array,
    baselines: list[BaselineAnnotation] | None = None,
    edge_fraction: float = _DEFAULT_EDGE_FRACTION,
    percentile: float = _DEFAULT_PERCENTILE,
) -> BaselinePriors:
    """Estimate per-trace linear baseline priors.

    Anchor points are collected from:

    - **Explicit baseline sections** — all finite points within each
      :class:`~.data.BaselineAnnotation` region.
    - **Peak window edges** — bottom ``percentile``% of the left+right
      ``edge_fraction`` of each :class:`~.data.PeakAnnotation` window.
      Falls back to bottom ``percentile``% of the full window when the
      window is too narrow for edge extraction.

    A per-trace OLS line is then fitted through the anchor points.
    Prior scales are derived from the OLS standard errors, floored at a
    per-trace physical scale: intercept floor = ``sigma_noise``, slope
    floor = ``sigma_noise / time_span`` (falling back to ``sigma_noise``
    when ``time_span <= 0``).

    Args:
        time:          ``[n_trace, n_time]`` retention-time axis.
        signal:        ``[n_trace, n_time]`` signal matrix.
        peaks:         Peak window annotations.
        sigma_noise:   ``[n_trace]`` per-trace noise estimate (DER_SNR).
        baselines:     Optional explicit baseline region annotations.
        edge_fraction: Fraction of each window to use for edge anchors.
        percentile:    Signal percentile threshold for anchor selection.

    Returns:
        :class:`BaselinePriors` with per-trace intercept/slope and scales.
    """
    if time.ndim != 2 or signal.ndim != 2:
        raise ValueError("time and signal must be 2-D [n_trace, n_time].")
    if time.shape != signal.shape:
        raise ValueError("time and signal shape mismatch.")
    sigma_noise = jnp.asarray(sigma_noise)
    if sigma_noise.shape != (time.shape[0],):
        raise ValueError(
            f"sigma_noise must have shape [n_trace]={time.shape[0]}, got {sigma_noise.shape}."
        )
    if not (0.0 < float(percentile) <= 100.0):
        raise ValueError("percentile must satisfy 0 < percentile <= 100.")
    if not (0.0 < float(edge_fraction) <= 0.5):
        raise ValueError("edge_fraction must satisfy 0 < edge_fraction <= 0.5.")

    baseline_regions = [] if baselines is None else baselines

    anchor_mask = _select_anchors(
        time,
        signal,
        peaks=peaks,
        baselines=baseline_regions,
        edge_fraction=float(edge_fraction),
        percentile=float(percentile),
    )
    intercept, slope, se_intercept, se_slope = _fit_line(time, signal, anchor_mask)

    # Per-trace time span for slope floor. Fall back to sigma_noise when span <= 0.
    time_span = time[:, -1] - time[:, 0]
    slope_floor = jnp.where(time_span > 0.0, sigma_noise / jnp.where(time_span > 0.0, time_span, 1.0), sigma_noise)

    intercept_scale = _scale_from_se(se_intercept, floor=sigma_noise)
    slope_scale = _scale_from_se(se_slope, floor=slope_floor)

    return BaselinePriors(
        intercept=intercept,
        slope=slope,
        intercept_scale=intercept_scale,
        slope_scale=slope_scale,
    )
```

Replace `_scale_from_se` at the bottom of the file:

```python
def _scale_from_se(se: jax.Array, *, floor: jax.Array) -> jax.Array:
    """Use OLS standard errors as prior scales, with a per-trace floor.

    Args:
        se:    ``[n_trace]`` standard errors (may contain NaN for degenerate fits).
        floor: ``[n_trace]`` per-trace minimum scale, strictly positive.
    """
    return jnp.where(
        jnp.isfinite(se) & (se > 0.0),
        jnp.maximum(se, floor),
        floor,
    )
```

- [ ] **Step 4: Update `Fitter.baseline_priors()` to pass `sigma_noise`**

In `chromhandler/fitting/fitter.py`, update the cached call to `estimate_baseline`:

```python
    def baseline_priors(self) -> BaselinePriors:
        """Per-trace OLS linear baseline priors.

        Cached after first call.
        """
        if "_bp_direct" not in self.__dict__:
            self._bp_direct: BaselinePriors = estimate_baseline(
                jnp.asarray(self.time),
                jnp.asarray(self.signal),
                peaks=self.peaks,
                baselines=self.baselines,
                sigma_noise=jnp.asarray(self.trace_sigma_noise),
            )
        return self._bp_direct
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/fitting/test_baseline.py tests/unit/fitting/test_noise_plumbing.py -v`
Expected: PASS.

- [ ] **Step 6: Lint and type-check**

Run: `uv run ruff check chromhandler/fitting/baseline.py chromhandler/fitting/fitter.py tests/unit/fitting/test_baseline.py`
Run: `uv run pyright chromhandler/fitting/baseline.py chromhandler/fitting/fitter.py tests/unit/fitting/test_baseline.py`
Expected: both clean.

- [ ] **Step 7: Commit**

```bash
git add chromhandler/fitting/baseline.py chromhandler/fitting/fitter.py tests/unit/fitting/test_baseline.py
git commit -m "feat(baseline): per-trace sigma_noise floors replace absolute constants"
```

---

## Task 5: `_estimate_snr` + `build_peak_priors` take `sigma_noise`

Replace first-difference MAD in `_estimate_snr` with `apex_height / sigma_noise`. Thread `sigma_noise` through `build_peak_priors`. Update every call site (internal + tests).

**Files:**
- Modify: `chromhandler/fitting/priors.py:583-596` (`_estimate_snr`)
- Modify: `chromhandler/fitting/priors.py:604-713` (`build_peak_priors` signature + call-site)
- Modify: `chromhandler/fitting/fitter.py:178-196` (`_compute_position_priors` call site)
- Modify: `tests/unit/fitting/test_priors.py` (all `build_peak_priors` + `_estimate_snr` call sites)
- Modify: `tests/integration/test_prior_pipeline.py` (all `build_peak_priors` call sites)

- [ ] **Step 1: Write failing tests (extend `test_priors.py`)**

Add two new tests to `tests/unit/fitting/test_priors.py` (place near the existing `_estimate_snr` tests, approximately after line 210):

```python
@pytest.mark.unit
def test_estimate_snr_uses_supplied_sigma_noise() -> None:
    """_estimate_snr returns apex_height / sigma_noise, elementwise."""
    from chromhandler.fitting.priors import _estimate_snr

    apex_height = np.array([100.0, 200.0, 50.0])
    sigma_noise = np.array([2.0, 4.0, 1.0])
    snr = _estimate_snr(apex_height, sigma_noise)
    np.testing.assert_allclose(snr, [50.0, 50.0, 50.0])


@pytest.mark.unit
def test_build_peak_priors_requires_sigma_noise_kwarg() -> None:
    """build_peak_priors' sigma_noise kwarg is required (no default)."""
    # Minimal fixture — reuse the module's existing conftest-provided peaks/x/signal/baseline.
    from chromhandler.fitting.priors import build_peak_priors

    x = np.linspace(0.0, 10.0, 200)
    signal = np.full((1, 200), 100.0) + np.exp(-((x - 5.0) ** 2))[None, :]
    baseline = np.full((1, 200), 100.0)
    peaks = [PeakAnnotation(molecule_id="m", rt_min=4.0, rt_max=6.0)]
    with pytest.raises(TypeError, match="sigma_noise"):
        build_peak_priors(peaks, x, signal, baseline)  # type: ignore[call-arg]
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/unit/fitting/test_priors.py::test_estimate_snr_uses_supplied_sigma_noise tests/unit/fitting/test_priors.py::test_build_peak_priors_requires_sigma_noise_kwarg -v`
Expected: FAIL — signatures not updated.

- [ ] **Step 3: Rewrite `_estimate_snr`**

Replace `_estimate_snr` in `chromhandler/fitting/priors.py` (lines ~583-596):

```python
def _estimate_snr(
    apex_height: jax.Array | NDArray[np.float64],
    sigma_noise: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-trace signal-to-noise ratio = ``apex_height / sigma_noise``.

    Both inputs are per-trace arrays of shape ``[n_trace]``. ``sigma_noise``
    comes from :class:`chromhandler.trace_statistics.TraceStatistics` via
    :attr:`Fitter.trace_sigma_noise` and is already strictly positive.
    """
    return np.maximum(
        np.asarray(apex_height, dtype=float) / np.asarray(sigma_noise, dtype=float),
        0.0,
    )
```

- [ ] **Step 4: Update `build_peak_priors` signature and internal call**

In `chromhandler/fitting/priors.py`:

Update the signature (line ~604):

```python
def build_peak_priors(
    peaks: list[PeakAnnotation],
    x: NDArray[np.float64],
    signal: NDArray[np.float64],
    baseline: NDArray[np.float64],
    *,
    sigma_noise: NDArray[np.float64],
) -> tuple[list[GeometricPeakPriors], PeakApexTraces]:
```

Update the docstring `Parameters` section — add a new entry for `sigma_noise`:

```
    sigma_noise:
        Per-trace noise estimate, shape ``[n_trace]``.  Used for the
        signal-to-noise diagnostic on ``GeometricPeakPriors``.
```

Update the validation block (just after the `baseline.shape` check at ~line 653) to also validate `sigma_noise`:

```python
    sigma_noise = np.asarray(sigma_noise, dtype=float)
    if sigma_noise.shape != (signal.shape[0],):
        raise ValueError(
            f"sigma_noise must have shape [n_trace]={signal.shape[0]}, got {sigma_noise.shape}."
        )
```

Update the internal `_estimate_snr` call (line ~713):

```python
        # --- Signal-to-noise ratio ---
        snr_per_trace = _estimate_snr(geometry.apex_height, sigma_noise)
```

- [ ] **Step 5: Update `Fitter._compute_position_priors` call site**

In `chromhandler/fitting/fitter.py`, update the `build_peak_priors` call inside `_compute_position_priors` (~line 185):

```python
        priors, apex_traces = build_peak_priors(
            self.peaks, x, self.signal, baseline, sigma_noise=self.trace_sigma_noise
        )
```

- [ ] **Step 6: Update all existing call sites in tests**

Update every `build_peak_priors(peaks, x, signal, baseline)` call site in both test files to pass `sigma_noise`.

**`tests/unit/fitting/test_priors.py`:**
Add a default-sigma fixture at the top of the file (after imports, before the first test that needs it):

```python
def _default_sigma(n_trace: int) -> np.ndarray:
    """Small constant per-trace sigma_noise for unit tests."""
    return np.full(n_trace, 1.0, dtype=float)
```

For every `build_peak_priors(peaks, x, signal, baseline)` call in the file, append `sigma_noise=_default_sigma(signal.shape[0])`. Example:

Before:
```python
priors, _ = build_peak_priors(peaks, x, signal, baseline)
```
After:
```python
priors, _ = build_peak_priors(peaks, x, signal, baseline, sigma_noise=_default_sigma(signal.shape[0]))
```

For every `_estimate_snr(y, geo.apex_height)` call in the file (two sites around lines 184 and 205), replace with `_estimate_snr(geo.apex_height, _default_sigma(y.shape[0]))`.

**`tests/integration/test_prior_pipeline.py`:**
Add the same helper after imports:

```python
def _default_sigma(n_trace: int) -> np.ndarray:
    return np.full(n_trace, 1.0, dtype=float)
```

Update every `build_peak_priors(peaks, x, signal, baseline)` call to pass `sigma_noise=_default_sigma(signal.shape[0])` (or use the true noise where the fixture already defines it — e.g. tests that vary the noise level should use the matching array).

For the specific test at `test_prior_pipeline.py:213-214` (which compares `signal_low` vs `signal_high`), pass `sigma_noise=_default_sigma(signal_low.shape[0])` in both calls.

- [ ] **Step 7: Run the full priors test suite**

Run: `uv run pytest tests/unit/fitting/test_priors.py tests/integration/test_prior_pipeline.py -v`
Expected: PASS. If any existing test was relying on the old first-diff MAD noise estimate (e.g. the SNR assertions in `test_priors.py:181-210`), adjust the expected values to match `apex_height / 1.0` (the default sigma).

- [ ] **Step 8: Lint and type-check**

Run: `uv run ruff check chromhandler/fitting/priors.py chromhandler/fitting/fitter.py tests/unit/fitting/test_priors.py tests/integration/test_prior_pipeline.py`
Run: `uv run pyright chromhandler/fitting/priors.py chromhandler/fitting/fitter.py tests/unit/fitting/test_priors.py tests/integration/test_prior_pipeline.py`
Expected: both clean.

- [ ] **Step 9: Commit**

```bash
git add chromhandler/fitting/priors.py chromhandler/fitting/fitter.py tests/unit/fitting/test_priors.py tests/integration/test_prior_pipeline.py
git commit -m "feat(priors): _estimate_snr + build_peak_priors take per-trace sigma_noise"
```

---

## Task 6: Lock `trace_sigma_noise` contract in existing Fitter tests

Add one-line assertions to the two existing direct-construction test files so `trace_sigma_noise` is exercised in their scenarios too. Catches regressions in the auto-compute path.

**Files:**
- Modify: `tests/unit/fitting/test_fitter_inputs.py`
- Modify: `tests/unit/fitting/test_fitter_diagnostics.py`

- [ ] **Step 1: Extend `test_fitter_inputs.py`**

Find each function that constructs `Fitter(time, signal)` directly (at least the one at line ~47). Immediately after the construction line, add:

```python
assert fitter.trace_sigma_noise.shape == (fitter.n_traces,)
assert np.all(fitter.trace_sigma_noise > 0.0)
```

- [ ] **Step 2: Extend `test_fitter_diagnostics.py`**

Same thing — find each `Fitter(time, signal)` direct construction (at least the ones at lines ~46 and ~212). Add immediately after each:

```python
assert fitter.trace_sigma_noise.shape == (fitter.n_traces,)
assert np.all(fitter.trace_sigma_noise > 0.0)
```

If `numpy as np` is not already imported at the top of either file, add it.

- [ ] **Step 3: Run the modified tests**

Run: `uv run pytest tests/unit/fitting/test_fitter_inputs.py tests/unit/fitting/test_fitter_diagnostics.py -v`
Expected: PASS (all existing tests + the new assertions pass via the auto-compute path).

- [ ] **Step 4: Lint and type-check**

Run: `uv run ruff check tests/unit/fitting/test_fitter_inputs.py tests/unit/fitting/test_fitter_diagnostics.py`
Run: `uv run pyright tests/unit/fitting/test_fitter_inputs.py tests/unit/fitting/test_fitter_diagnostics.py`
Expected: both clean.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/fitting/test_fitter_inputs.py tests/unit/fitting/test_fitter_diagnostics.py
git commit -m "test(fitter): assert trace_sigma_noise contract in existing tests"
```

---

## Task 7: Full-suite verification

Run the full test suite and repo-wide lint/type-check to catch any call site the earlier tasks missed.

- [ ] **Step 1: Run the full unit + integration suite**

Run: `uv run pytest tests/ -v`
Expected: all existing + new tests pass.

If anything fails: identify the call site, update it to pass `sigma_noise`, commit as a separate "test: fix missed call site for <file>" commit.

- [ ] **Step 2: Repo-wide lint**

Run: `uv run ruff check .`
Expected: only pre-existing errors, nothing new from this feature.

- [ ] **Step 3: Type-check the touched modules**

Run: `uv run pyright chromhandler/fitting/fitter.py chromhandler/fitting/baseline.py chromhandler/fitting/priors.py`
Expected: clean.

- [ ] **Step 4: Done — no additional commit needed unless fix-ups were required**

If Step 1 surfaced missed call sites, the fix-up commits land here. Otherwise the feature is complete.

---

## Self-review notes

- **Spec coverage:** A (`priors.py::_estimate_snr`) → Task 5. B (`fitter.py::noise_prior` floor) → Task 3. C (`fitter.py::noise_prior` MAD block) → Task 3. D (`baseline.py` floors) → Task 4. Plumbing → Tasks 1-2. Error handling → Task 2 (strict) + Task 1 (shape/finiteness validation). All testing strategy sections → Tasks 1-6.
- **Type consistency:** `trace_sigma_noise` is `NDArray[np.float64]` throughout. `sigma_noise` param name is used consistently in `estimate_baseline`, `build_peak_priors`, `_estimate_snr`. Shape invariant `[n_trace]` holds at every interface.
- **No `subset` method on Fitter** — the design spec mentioned slicing consistency, but `Fitter` actually uses `select_trace_indices` to return indices (not a copy). No new slicing helper is required; the existing index-returning API remains unchanged and callers that index `trace_chromatogram_ids` can index `trace_sigma_noise` identically.
- **`_FLOAT_MIN` stays in `priors.py`** — used for many unrelated epsilon guards (the rewrite of `_estimate_snr` removes only the `noise_est = max(noise_est, _FLOAT_MIN)` usage).
