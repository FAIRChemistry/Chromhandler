"""Minimal chromatographic fitter using window-geometry-based Bayesian priors.

Replaces the FWHM-based prior pipeline of ``nu_bayes.py`` with the geometry-only
approach from ``priors.py``.  This file contains only what is needed to:

1. Accept time/signal data + peak/baseline annotations via a native subset API.
2. Estimate a linear baseline via ``baseline.py``.
3. Compute window-geometry priors via ``priors.py``.
4. Run MCMC inference via ``better_model.py`` using NUTS sampler.
5. Print a human-readable prior summary and posterior statistics via ArviZ.

Subset API overview
-------------------
Case 1 — single group (all traces share the same peak windows)::

    fitter = BetterFitter.from_handler(handler)
    fitter.add_baseline_annotation(BaselineAnnotation(rt_min=0.5, rt_max=1.0))
    fitter.add_peak_annotation(PeakAnnotation(molecule_id="s0", rt_min=2.8, rt_max=3.2, mode="single"))
    fitter.fit()
    fitter.posteriors  # {"__default__": InferenceData}

Case 2 — multiple groups with different peak windows or trace selections::

    fitter = BetterFitter.from_handler(handler)
    s1 = fitter.add_subset("col_A", sample_ids=["run1", "run2"])
    s1.add_peak_annotation(PeakAnnotation(molecule_id="NAD", rt_min=2.8, rt_max=3.2, mode="single"))

    s2 = fitter.add_subset("col_B", sample_ids=["run3", "run4"])
    s2.add_peak_annotation(PeakAnnotation(molecule_id="NAD", rt_min=2.9, rt_max=3.3, mode="single"))

    fitter.fit()                        # all subsets
    fitter.fit(subsets=["col_A"])       # selective
    fitter.posteriors                   # {"col_A": InferenceData, "col_B": InferenceData}
"""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from chromhandler.model import Peak

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.optim as numpyro_optim
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import (
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
    AutoNormal,
)
from rich import print

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation

from . import better_model
from .baseline import BaselinePriors, estimate_baseline
from .data import (
    PEAK_MODE_TO_CODE,
    baseline_to_mask,
    peak_is_artefact_mode,
    peak_is_free_mode,
    stack_and_pad_signal,
)
from .priors import (
    FwhmShapeDiagnostics,
    GeometricPeakPriors,
    build_geometric_priors,
    geometric_priors_to_arrays,
    refine_apex_priors_with_trace_shift,
    summarise_priors,
)
from .priors import (
    compute_fwhm_shape_diagnostics as build_fwhm_shape_diagnostics,
)
from .subsets import AreaRecord, Subset

# _DEFAULT_SUBSET_NAME is the implicit subset created by add_peak_annotation()
# when no explicit subsets have been registered.
_DEFAULT_SUBSET_NAME = "__default__"

GuideType = Literal["diagonal", "full_rank", "low_rank"]


def _build_guide(guide_type: GuideType, model: object, low_rank_rank: int) -> object:
    """Construct a NumPyro autoguide for the given guide type."""
    if guide_type == "diagonal":
        return AutoNormal(model)
    elif guide_type == "full_rank":
        return AutoMultivariateNormal(model)
    elif guide_type == "low_rank":
        return AutoLowRankMultivariateNormal(model, rank=low_rank_rank)
    raise ValueError(f"guide_type must be 'diagonal', 'full_rank', or 'low_rank'; got {guide_type!r}")


@dataclasses.dataclass(frozen=True)
class PosteriorCurves:
    """Precomputed posterior HDI curves for a fitted subset.

    All arrays are plain numpy.  Shapes:

    - ``x``: ``[n_x]`` — evaluation axis (minutes)
    - ``total_*``, ``baseline_*``: ``[n_trace, n_x]``
    - ``comp_l_*``, ``comp_r_*``: ``[n_trace, n_peak, n_x]``
    - ``trace_indices``: ``[n_trace]`` — global indices into the parent fitter's
      full trace array (result of :meth:`~BetterFitter.select_trace_indices`)
    - ``chromatogram_ids``: per-trace labels (or ``None`` if unavailable)
    """

    x: np.ndarray
    total_median: np.ndarray
    total_lower: np.ndarray
    total_upper: np.ndarray
    baseline_median: np.ndarray
    baseline_lower: np.ndarray
    baseline_upper: np.ndarray
    comp_l_median: np.ndarray
    comp_l_lower: np.ndarray
    comp_l_upper: np.ndarray
    comp_r_median: np.ndarray
    comp_r_lower: np.ndarray
    comp_r_upper: np.ndarray
    trace_indices: np.ndarray
    chromatogram_ids: list[str] | None


class BetterFitter:
    """Chromatographic fitter with native multi-subset support.

    Parameters
    ----------
    time:
        Retention-time matrix, shape ``[n_trace, n_time]``.
        Rows may have a slowly drifting time axis (e.g. from different runs);
        a common 1-D axis is derived as the row-wise median.
    signal:
        Signal matrix, shape ``[n_trace, n_time]``.
    peaks:
        Optional annotated peak windows applied to **all** traces (shorthand
        that auto-creates a ``"__default__"`` subset).  Mutually exclusive with
        :meth:`add_subset`.  Prefer using :meth:`add_peak_annotation` instead.
    baselines:
        Global baseline regions used by all subsets that do not define their
        own baselines.  An empty list is acceptable; the baseline estimation
        also uses the edges of each peak window.
    """

    def __init__(
        self,
        time: np.ndarray,
        signal: np.ndarray,
        *,
        peaks: list[PeakAnnotation] | None = None,
        baselines: list[BaselineAnnotation] | None = None,
        trace_sample_ids: list[str] | None = None,
        trace_chromatogram_ids: list[str] | None = None,
    ) -> None:
        self.time = np.asarray(time, dtype=float)
        self.signal = np.asarray(signal, dtype=float)
        # Global fallback peaks/baselines (used when _subsets is empty, i.e. on views)
        self.peaks: list[PeakAnnotation] = []
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
        self.trace_sample_ids: np.ndarray | None = (
            np.asarray(trace_sample_ids, dtype=object) if trace_sample_ids is not None else None
        )
        self.trace_chromatogram_ids: np.ndarray | None = (
            np.asarray(trace_chromatogram_ids, dtype=object) if trace_chromatogram_ids is not None else None
        )

        # trace_subsets[i] holds the name of the Subset that trace i belongs to.
        # Set when add_subset() is called; None means no subsets registered yet.
        self.trace_subsets: np.ndarray | None = None  # [n_trace] str, lazy-init

        # --- Subset storage (new unified design) ---
        self._subsets: dict[str, Subset] = {}
        # Per-subset posteriors (populated after fit())
        self._posteriors: dict[str, object] = {}  # subset_name → InferenceData
        self._samples_dict: dict[str, dict] = {}  # subset_name → {param: array}
        self._elbo_losses: dict[str, np.ndarray] = {}  # subset_name → [n_steps] (SVI only)
        self._subset_trace_ids: dict[str, np.ndarray] = {}  # subset_name → chromatogram_ids
        # Per-subset baseline-prior cache
        self._baseline_priors_cache: dict[str, BaselinePriors] = {}
        # Per-subset shape-features cache
        self._shape_features_cache: dict[str, FwhmShapeDiagnostics] = {}
        # Per-subset masked time axis (set by fit() from view.x_masked)
        self._x_masked: dict[str, np.ndarray] = {}

        # Alignment attributes (set by .align())
        self.shift_samples: np.ndarray | None = None  # [n_trace] shifts in samples
        self.shift_time: np.ndarray | None = None  # [n_trace] shifts in time units
        self.shift_result: object | None = None  # ShiftAlignmentResult

        # Inference attributes (set by _run_mcmc() / _run_svi() on views only)
        self.mcmc: MCMC | None = None
        self.samples: dict | None = None
        self.svi_result: object | None = None
        self.elbo_losses: np.ndarray | None = None
        # Note: self.posterior is NOT initialized here; only views get it via _run_mcmc()/_run_svi()

        # If peaks were provided at construction time, auto-register them as __default__
        if peaks:
            for ann in peaks:
                self.add_peak_annotation(ann)

    @staticmethod
    def _stabilize_area_prior_matrix(area_pt: np.ndarray) -> np.ndarray:
        """Ensure area prior centers are strictly positive for LogNormal priors."""
        col_median = np.where(
            np.nanmedian(area_pt, axis=0) > 0,
            np.nanmedian(area_pt, axis=0),
            1e-3,
        )
        return np.where(area_pt > 0, area_pt, col_median[None, :] * 0.01)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        if self.time.ndim != 2 or self.signal.ndim != 2:
            raise ValueError("time and signal must be 2-D [n_trace, n_time].")
        if self.time.shape != self.signal.shape:
            raise ValueError(
                f"time and signal must have the same shape; got {self.time.shape} vs {self.signal.shape}."
            )

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def n_traces(self) -> int:
        return int(self.time.shape[0])

    @property
    def n_timepoints(self) -> int:
        return int(self.time.shape[1])

    def common_time(self) -> np.ndarray:
        """Row-wise median time axis → 1-D ``[n_time]``.

        When all traces share an identical time grid this is exact.  For small
        per-trace drift it is a robust representative axis.
        """
        return np.nanmedian(self.time, axis=0)

    # ------------------------------------------------------------------
    # Subset resolution helpers
    # ------------------------------------------------------------------

    def _resolve_subset(self, subset: str | None) -> Subset:
        """Resolve *subset* name to a :class:`Subset` object.

        If *subset* is ``None`` and exactly one subset is registered, returns
        it automatically.  Raises for any ambiguity.
        """
        if subset is None:
            if len(self._subsets) == 1:
                return next(iter(self._subsets.values()))
            if not self._subsets:
                raise RuntimeError("No subsets registered. Call add_peak_annotation() or add_subset() first.")
            raise ValueError(f"Multiple subsets registered: {list(self._subsets)}. Specify subset='<name>'.")
        if subset not in self._subsets:
            raise KeyError(f"No subset named '{subset}'. Registered subsets: {list(self._subsets)}.")
        return self._subsets[subset]

    def _compute_subset_mask(self, subset: Subset) -> np.ndarray:
        """Return a boolean mask ``[n_trace]`` selecting traces for *subset*."""
        if subset._match_all:
            return np.ones(self.n_traces, dtype=bool)
        if self.trace_sample_ids is None or self.trace_chromatogram_ids is None:
            raise RuntimeError(
                "_compute_subset_mask() requires trace ID arrays. "
                "Build the fitter with BetterFitter.from_handler()."
            )
        mask = np.zeros(self.n_traces, dtype=bool)
        for sid in subset.sample_ids:
            mask |= self.trace_sample_ids == sid
        for cid in subset.chromatogram_ids:
            mask |= self.trace_chromatogram_ids == cid
        return mask

    # ------------------------------------------------------------------
    # Prior computation (subset-aware)
    # ------------------------------------------------------------------

    def _compute_position_priors(
        self,
        subset: str | None = None,
    ) -> tuple[list[GeometricPeakPriors], float]:
        """Compute apex priors plus the shared trace-shift scale."""
        if not self._subsets:
            peaks = self.peaks
            x = self.common_time()
            signal = self.signal
            baseline = self.baseline_signal()
        else:
            s = self._resolve_subset(subset)
            mask = self._compute_subset_mask(s)
            peaks = s.peaks
            x = np.nanmedian(self.time[mask], axis=0)
            signal = self.signal[mask]
            baseline = self.baseline_signal(subset=s.name)

        priors = build_geometric_priors(peaks, x, signal, baseline)
        diagnostics = build_fwhm_shape_diagnostics(peaks, x, signal, baseline)
        x_finite = np.asarray(x, dtype=float)
        x_finite = x_finite[np.isfinite(x_finite)]
        if x_finite.size >= 2:
            apex_scale_floor = max(
                float(np.nanmedian(np.abs(np.diff(np.sort(x_finite))))),
                1e-6,
            )
        else:
            apex_scale_floor = 1e-6
        return refine_apex_priors_with_trace_shift(
            priors,
            diagnostics,
            apex_scale_floor=apex_scale_floor,
            trace_shift_scale_floor=1e-6,
        )

    def baseline_priors(self, subset: str | None = None) -> BaselinePriors:
        """Per-trace OLS linear baseline priors for *subset*.

        Cached per subset after first call.  For views (no subsets registered)
        uses the fitter's own data directly.
        """
        if not self._subsets:
            # View mode: compute from self directly (legacy path)
            if "_bp_direct" not in self.__dict__:
                self._bp_direct: BaselinePriors = estimate_baseline(
                    self.time,
                    self.signal,
                    peaks=self.peaks,
                    baselines=self.baselines,
                )
            return self._bp_direct

        s = self._resolve_subset(subset)
        if s.name not in self._baseline_priors_cache:
            mask = self._compute_subset_mask(s)
            effective_baselines = s.baselines if s.baselines else self.baselines
            self._baseline_priors_cache[s.name] = estimate_baseline(
                self.time[mask],
                self.signal[mask],
                peaks=s.peaks,
                baselines=effective_baselines,
            )
        return self._baseline_priors_cache[s.name]

    def baseline_signal(self, subset: str | None = None) -> np.ndarray:
        """Reconstructed linear baseline matrix for *subset*.

        Returns shape ``[n_trace_subset, n_time]`` for explicit subsets,
        or ``[n_trace, n_time]`` for the view / default case.
        """
        if not self._subsets:
            bp = self.baseline_priors()
            intercept = np.asarray(bp.intercept, dtype=float)[:, None]
            slope = np.asarray(bp.slope, dtype=float)[:, None]
            return intercept + slope * self.time

        s = self._resolve_subset(subset)
        mask = self._compute_subset_mask(s)
        bp = self.baseline_priors(subset=s.name)
        intercept = np.asarray(bp.intercept, dtype=float)[:, None]
        slope = np.asarray(bp.slope, dtype=float)[:, None]
        return intercept + slope * self.time[mask]

    def compute_priors(self, subset: str | None = None) -> list[GeometricPeakPriors]:
        """Compute window-geometry priors for *subset* (or the single registered subset)."""
        priors, _ = self._compute_position_priors(subset=subset)
        return priors

    def compute_fwhm_shape_diagnostics(self, subset: str | None = None) -> FwhmShapeDiagnostics:
        """Per-trace FWHM-derived main-peak shape diagnostics for *subset*."""
        if not self._subsets:
            x = self.common_time()
            baseline = self.baseline_signal()
            return build_fwhm_shape_diagnostics(self.peaks, x, self.signal, baseline)

        s = self._resolve_subset(subset)
        mask = self._compute_subset_mask(s)
        baseline = self.baseline_signal(subset=s.name)
        x = np.nanmedian(self.time[mask], axis=0)
        return build_fwhm_shape_diagnostics(s.peaks, x, self.signal[mask], baseline)

    def get_shape_features(self, subset: str | None = None) -> FwhmShapeDiagnostics:
        """Per-trace FWHM-derived shape diagnostics with NaN for invalid entries.

        Continuous fields (``sigma_trace``, ``alpha_trace``,
        ``fwhm_apex_trace``, ``fwhm_trace``, ``apex_height_trace``) are set
        to ``float('nan')`` wherever ``fwhm_valid_trace`` is ``False``.

        The result is cached per subset after first access; call
        :meth:`compute_fwhm_shape_diagnostics` directly for an uncached copy.
        """
        cache_key = self._resolve_subset(subset).name if self._subsets else "_direct_"
        if cache_key not in self._shape_features_cache:
            diag = self.compute_fwhm_shape_diagnostics(subset=(subset if self._subsets else None))
            invalid = ~diag.fwhm_valid_trace
            self._shape_features_cache[cache_key] = dataclasses.replace(
                diag,
                sigma_trace=np.where(invalid, np.nan, diag.sigma_trace),
                alpha_trace=np.where(invalid, np.nan, diag.alpha_trace),
                fwhm_apex_trace=np.where(invalid, np.nan, diag.fwhm_apex_trace),
                fwhm_trace=np.where(invalid, np.nan, diag.fwhm_trace),
                apex_height_trace=np.where(invalid, np.nan, diag.apex_height_trace),
            )
        return self._shape_features_cache[cache_key]

    # Backward-compat alias (views only; will raise on parent with subsets)
    @functools.cached_property
    def shape_features(self) -> FwhmShapeDiagnostics:
        """Deprecated: use :meth:`get_shape_features` instead."""
        return self.get_shape_features()

    def noise_prior(self, subset: str | None = None) -> np.ndarray:
        """Estimate per-trace observation noise from baseline-corrected signal.

        Uses median absolute deviation in baseline regions, or falls back to
        signal std if no baseline regions defined.

        Returns
        -------
        np.ndarray
            Shape ``[n_trace_subset]``, noise level for each trace (positive).
        """
        if not self._subsets:
            # View mode: use self directly
            time_s = self.time
            signal_s = self.signal
            effective_baselines = self.baselines
        else:
            s = self._resolve_subset(subset)
            mask = self._compute_subset_mask(s)
            time_s = self.time[mask]
            signal_s = self.signal[mask]
            effective_baselines = s.baselines if s.baselines else self.baselines

        bp = (
            self.baseline_priors()
            if not self._subsets
            else self.baseline_priors(subset=self._resolve_subset(subset).name)
        )
        intercept = np.asarray(bp.intercept, dtype=float)[:, None]
        slope = np.asarray(bp.slope, dtype=float)[:, None]
        baseline = intercept + slope * time_s
        signal_corrected = signal_s - baseline

        n_traces_s = signal_s.shape[0]
        if effective_baselines:
            x_jax = jnp.asarray(time_s, dtype=float)
            baseline_mask = baseline_to_mask(effective_baselines, x_jax)
            baseline_mask_np = np.asarray(baseline_mask, dtype=bool)
            sigma_y = np.array(
                [
                    float(np.median(np.abs(signal_corrected[t][baseline_mask_np[t]]))) * 1.4826
                    for t in range(n_traces_s)
                ]
            )
        else:
            sigma_y = np.std(signal_corrected, axis=1)

        return np.maximum(sigma_y, 1.0)

    def create_observation_mask(self) -> np.ndarray:
        """Create boolean mask for timepoints to include in likelihood.

        Covers baseline regions and peak windows from this fitter's own
        ``peaks``/``baselines`` (view mode) or aggregated across all
        registered subsets (parent mode).

        Returns
        -------
        np.ndarray
            Shape ``[n_time]``, dtype bool.
        """
        x = self.common_time()
        mask = np.zeros(x.shape[0], dtype=bool)

        # Collect all baselines: global + subset-specific
        all_baselines = list(self.baselines)
        for s in self._subsets.values():
            all_baselines.extend(s.baselines)
        for bl in all_baselines:
            lo, hi = float(bl.rt_min), float(bl.rt_max)
            mask |= (x >= lo) & (x <= hi)

        # Collect all peak windows: global self.peaks + subset peaks
        all_peaks = list(self.peaks)
        for s in self._subsets.values():
            all_peaks.extend(s.peaks)
        for pk in all_peaks:
            lo, hi = float(pk.rt_min), float(pk.rt_max)
            mask |= (x >= lo) & (x <= hi)

        return mask

    def slice_to_observed_windows(self) -> tuple[np.ndarray, np.ndarray]:
        """Slice time and signal to include only baseline regions and peak windows.

        Returns rectangular arrays using the aligned per-trace time axis.

        Returns
        -------
        tuple of np.ndarray
            ``(time_masked, signal_masked)`` where each has shape
            ``[n_trace, n_masked_time]``.
        """
        mask = self.create_observation_mask()
        return self.time[:, mask], self.signal[:, mask]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_handler(
        cls,
        handler: object,
        sample_ids: list[str] | None = None,
    ) -> BetterFitter:
        """Construct a :class:`BetterFitter` from a :class:`~chromhandler.handler.Handler`.

        Chromatograms are gathered from all samples (or *sample_ids* subset),
        then NaN-padded to a common length so the resulting arrays are
        rectangular.

        The returned fitter has **no peak or baseline annotations** registered.
        Use :meth:`add_baseline_annotation` and :meth:`add_peak_annotation` (or
        :meth:`add_subset`) to attach annotations before calling :meth:`fit`.

        Args:
            handler: A :class:`~chromhandler.handler.Handler` instance.
            sample_ids: Optional list of sample IDs to include.  When ``None``
                all samples are used.

        Returns:
            A fully initialised :class:`BetterFitter` with no annotations.

        Example::

            fitter = BetterFitter.from_handler(handler)
            fitter.add_baseline_annotation(BaselineAnnotation(rt_min=0.5, rt_max=1.0))
            fitter.add_peak_annotation(
                PeakAnnotation(molecule_id="s0", rt_min=2.8, rt_max=3.2, mode="single")
            )
            fitter.align()
            fitter.fit()
        """
        samples = [
            s
            for s in handler.samples  # type: ignore[attr-defined]
            if sample_ids is None or s.id in sample_ids
        ]
        if not samples:
            raise ValueError("No matching samples found in handler.")

        time_lists: list[list[float]] = [c.time for s in samples for c in s.chromatograms]
        signal_lists: list[list[float]] = [c.signal for s in samples for c in s.chromatograms]
        trace_sample_ids: list[str] = [s.id for s in samples for c in s.chromatograms]
        trace_chrom_ids: list[str] = [c.id for s in samples for c in s.chromatograms]

        time_arr, signal_arr = stack_and_pad_signal(time_lists, signal_lists)

        return cls(
            np.asarray(time_arr, dtype=float),
            np.asarray(signal_arr, dtype=float),
            peaks=None,
            baselines=None,
            trace_sample_ids=trace_sample_ids,
            trace_chromatogram_ids=trace_chrom_ids,
        )

    # ------------------------------------------------------------------
    # Annotation management
    # ------------------------------------------------------------------

    def add_peak_annotation(
        self,
        ann: PeakAnnotation,
        subset_id: str | None = None,
    ) -> None:
        """Register a peak-window annotation.

        Args:
            ann: The :class:`~chromhandler.annotations.PeakAnnotation` to add.
            subset_id: Name of the target subset.  When ``None`` and no explicit
                subsets have been registered, a ``"__default__"`` subset covering
                all traces is auto-created.  When ``None`` and explicit subsets
                exist, a :exc:`ValueError` is raised asking the caller to specify
                *subset_id*.  When the named subset does not exist, a
                :exc:`KeyError` is raised.

        Examples::

            # Case 1 — single group (auto-creates __default__ subset):
            fitter.add_peak_annotation(
                PeakAnnotation(molecule_id="s0", rt_min=2.8, rt_max=3.2, mode="single")
            )

            # Case 2 — add to an existing named subset:
            fitter.add_peak_annotation(ann, subset_id="col_A")
        """
        if subset_id is None:
            # Check for forbidden mixing of default + explicit subsets
            if self._subsets and _DEFAULT_SUBSET_NAME not in self._subsets:
                raise ValueError(
                    "Explicit subsets are already registered. "
                    "Specify subset_id='<name>' or use subset.add_peak_annotation()."
                )
            if _DEFAULT_SUBSET_NAME not in self._subsets:
                # Auto-create the default subset (matches all traces)
                self._subsets[_DEFAULT_SUBSET_NAME] = Subset(
                    name=_DEFAULT_SUBSET_NAME,
                    _match_all=True,
                )
            self._subsets[_DEFAULT_SUBSET_NAME].add_peak_annotation(ann)
        else:
            if subset_id not in self._subsets:
                raise KeyError(f"No subset named '{subset_id}'. Call add_subset('{subset_id}', ...) first.")
            self._subsets[subset_id].add_peak_annotation(ann)

    def add_baseline_annotation(
        self,
        ann: BaselineAnnotation,
        subset_id: str | None = None,
    ) -> None:
        """Register a baseline-region annotation.

        Args:
            ann: The :class:`~chromhandler.annotations.BaselineAnnotation` to add.
            subset_id: When ``None``, the annotation is added to the fitter's
                global ``baselines`` list (fallback for all subsets that define
                no subset-specific baselines).  When a name is given the
                annotation is added to that subset's baseline list, overriding
                the global fallback for that subset only.

        Examples::

            # Global baseline (applies to all subsets):
            fitter.add_baseline_annotation(BaselineAnnotation(rt_min=0.5, rt_max=1.0))

            # Subset-specific baseline:
            fitter.add_baseline_annotation(BaselineAnnotation(rt_min=0.5, rt_max=1.0), subset_id="col_A")
        """
        if subset_id is None:
            self.baselines.append(ann)
            # Invalidate all baseline-prior caches since global baselines changed
            self._baseline_priors_cache.clear()
            if "_bp_direct" in self.__dict__:
                del self._bp_direct
        else:
            if subset_id not in self._subsets:
                raise KeyError(f"No subset named '{subset_id}'. Call add_subset('{subset_id}', ...) first.")
            self._subsets[subset_id].add_baseline_annotation(ann)
            # Invalidate cache for this subset
            self._baseline_priors_cache.pop(subset_id, None)

    # ------------------------------------------------------------------
    # Subset management
    # ------------------------------------------------------------------

    def add_subset(
        self,
        name: str,
        *,
        sample_ids: list[str] | None = None,
        chromatogram_ids: list[str] | None = None,
    ) -> Subset:
        """Register a named fitting subset and return its builder object.

        After calling :meth:`add_subset`, attach peak/baseline annotations to
        the returned :class:`~chromhandler.fitting.subsets.Subset` object (or
        via :meth:`add_peak_annotation` with ``subset_id=name``).

        Args:
            name: Unique subset label.
            sample_ids: Include all chromatograms whose *sample_id* appears here.
            chromatogram_ids: Include specific chromatograms by ID.

        Returns:
            The newly created :class:`~chromhandler.fitting.subsets.Subset` builder.

        Raises:
            RuntimeError: If trace ID arrays are not available (build with
                :meth:`from_handler`).
            ValueError: If the ``"__default__"`` auto-subset already exists
                (mixing case 1 and case 2), or if *name* is already registered,
                or if neither *sample_ids* nor *chromatogram_ids* is provided.

        Example::

            s = fitter.add_subset("col_A", sample_ids=["run1", "run2"])
            s.add_peak_annotation(PeakAnnotation(molecule_id="NAD", rt_min=2.8, rt_max=3.2, mode="single"))
        """
        if self.trace_sample_ids is None or self.trace_chromatogram_ids is None:
            raise RuntimeError(
                "add_subset() requires trace_sample_ids and trace_chromatogram_ids. "
                "Build the fitter with BetterFitter.from_handler()."
            )
        if _DEFAULT_SUBSET_NAME in self._subsets:
            raise ValueError(
                f"A '{_DEFAULT_SUBSET_NAME}' subset already exists (created by "
                "add_peak_annotation() without subset_id). "
                "Cannot mix the default subset with explicit subsets."
            )
        if name in self._subsets:
            raise ValueError(f"A subset named '{name}' is already registered.")
        if not sample_ids and not chromatogram_ids:
            raise ValueError(
                f"add_subset('{name}'): at least one of sample_ids or chromatogram_ids must be provided."
            )

        subset = Subset(
            name=name,
            sample_ids=list(sample_ids) if sample_ids else [],
            chromatogram_ids=list(chromatogram_ids) if chromatogram_ids else [],
        )

        # Validate that at least one trace matches
        mask = self._compute_subset_mask(subset)
        if not np.any(mask):
            raise ValueError(f"Subset '{name}' matched no traces. Check sample_ids and chromatogram_ids.")

        # Lazy-init trace_subsets array
        if self.trace_subsets is None:
            self.trace_subsets = np.full(self.n_traces, "", dtype=object)
        self.trace_subsets[mask] = name
        self._subsets[name] = subset
        return subset

    def get_subset(self, name: str) -> Subset:
        """Return the :class:`~chromhandler.fitting.subsets.Subset` builder for *name*.

        Args:
            name: Subset name as registered via :meth:`add_subset` or
                ``"__default__"`` for the auto-created default subset.

        Raises:
            KeyError: If no subset with *name* is registered.
        """
        if name not in self._subsets:
            raise KeyError(f"No subset named '{name}'. Registered subsets: {list(self._subsets)}.")
        return self._subsets[name]

    def _make_subset_view(self, name: str) -> BetterFitter:
        """Build a transient BetterFitter restricted to *name*'s traces and peaks.

        The returned view has no subsets registered (``_subsets`` is empty) so
        all methods fall through to the direct / view-mode path.  The view is
        intended for use by :meth:`_run_mcmc` only and should not be stored.
        """
        subset = self._subsets[name]
        mask = self._compute_subset_mask(subset)
        effective_baselines = subset.baselines if subset.baselines else self.baselines

        view = BetterFitter(
            self.time[mask],
            self.signal[mask],
            peaks=None,  # do NOT auto-create subset — set self.peaks directly below
            baselines=effective_baselines,
            trace_sample_ids=(
                list(self.trace_sample_ids[mask])  # type: ignore[index]
                if self.trace_sample_ids is not None
                else None
            ),
            trace_chromatogram_ids=(
                list(self.trace_chromatogram_ids[mask])  # type: ignore[index]
                if self.trace_chromatogram_ids is not None
                else None
            ),
        )
        # Directly assign peaks — bypass add_peak_annotation to avoid subset creation
        view.peaks = list(subset.peaks)
        view.trace_subsets = np.full(int(mask.sum()), name, dtype=object)

        # Propagate alignment shifts
        if self.shift_samples is not None:
            view.shift_samples = self.shift_samples[mask]
        if self.shift_time is not None:
            view.shift_time = self.shift_time[mask]

        return view

    # ------------------------------------------------------------------
    # Chromatogram alignment
    # ------------------------------------------------------------------

    def align(
        self,
        *,
        lr: float = 1e-2,
        n_steps: int = 500,
        center_weight: float = 1e3,
        max_shift_samples: float | None = None,
        enforce_zero_mean: bool = True,
        n_starts: int = 16,
        sigma_perturb: float = 3.0,
        seed: int = 0,
        verbose: bool = True,
    ) -> None:
        """Align traces in-place by optimising per-trace retention-time shifts.

        Builds an alignment mask from all annotated peak windows and baseline
        regions (aggregated across all registered subsets), then runs
        multi-start Adam optimisation on the MSE alignment loss.  After
        alignment ``self.time`` is updated in-place; all cached quantities
        (baseline priors, shape features) are invalidated automatically.

        Parameters
        ----------
        lr : float
            Adam learning rate (default 1e-2).
        n_steps : int
            Adam iterations per start (default 500).
        center_weight : float
            Penalty on mean shift — keeps traces zero-centred (default 1e3).
        max_shift_samples : float or None
            Hard bound on shift magnitude in samples.  None = unconstrained.
        enforce_zero_mean : bool
            Re-centre shifts after every step (default True).
        n_starts : int
            Number of independent Adam restarts (default 16).
        sigma_perturb : float
            Std-dev (samples) of perturbation noise for starts 1+ (default 3.0).
        seed : int
            PRNG seed for perturbation noise (default 0).
        verbose : bool
            Print per-trace shift diagnostics after alignment (default True).
        """
        from .shift import align_chromatograms

        obs_mask_1d = self.create_observation_mask()  # [n_time]
        alignment_mask = np.tile(obs_mask_1d, (self.n_traces, 1))  # [n_trace, n_time]

        result = align_chromatograms(
            self.signal,
            mask=alignment_mask,
            lr=lr,
            n_steps=n_steps,
            center_weight=center_weight,
            max_shift_samples=max_shift_samples,
            enforce_zero_mean=enforce_zero_mean,
            n_starts=n_starts,
            sigma_perturb=sigma_perturb,
            seed=seed,
        )

        self.shift_result = result
        self.shift_samples = np.asarray(result.shifts_samples, dtype=float)

        dt_per_trace = np.nanmedian(np.abs(np.diff(self.time, axis=1)), axis=1)
        self.shift_time = self.shift_samples * dt_per_trace

        self.time = self.time + self.shift_time[:, None]

        # Invalidate all cached baseline priors and shape features
        self._baseline_priors_cache.clear()
        self._shape_features_cache.clear()
        if "_bp_direct" in self.__dict__:
            del self._bp_direct
        # Invalidate cached_property shape_features if it was computed
        if "shape_features" in self.__dict__:
            del self.__dict__["shape_features"]

        if verbose:
            self._print_alignment_result(result)

    def _print_alignment_result(self, result: object) -> None:
        """Print per-trace shift summary to stdout."""
        print(f"  loss: {result.loss_initial:.3e} → {result.loss_final:.3e}")  # type: ignore[attr-defined]
        print(f"  {'Trace':>5}  {'Shift (samp)':>12}  {'Shift (min)':>12}")
        print(f"  {'-' * 5}  {'-' * 12}  {'-' * 12}")
        for t in range(self.n_traces):
            print(
                f"  {t:>5}  "
                f"{float(self.shift_samples[t]):>+12.4f}  "  # type: ignore[index]
                f"{float(self.shift_time[t]):>+12.6f}"  # type: ignore[index]
            )

    def peak_structure(self) -> dict[str, np.ndarray]:
        """Extract mode-specific peak structure arrays from ``self.peaks``.

        Returns local indices splitting free-doublet peaks by separation
        variance:

        - ``free_fixed_local_index``: positions within the *n_free* axis where
          ``vary_separation=False`` (default).  These peaks share a single
          common separation across all traces.
        - ``free_vary_local_index``: positions within the *n_free* axis where
          ``vary_separation=True``.  These peaks get per-trace separation with
          a sampled per-peak trace-scale hyperparameter.
        """
        n_peak = len(self.peaks)
        peak_mode_code = np.zeros(n_peak, dtype=np.int32)
        artefact_side = np.zeros(n_peak, dtype=np.int32)
        artefact_indices: list[int] = []
        free_indices: list[int] = []
        nonfree_indices: list[int] = []
        free_fixed_local: list[int] = []
        free_vary_local: list[int] = []

        for i, peak in enumerate(self.peaks):
            peak_mode_code[i] = PEAK_MODE_TO_CODE[peak.mode]
            if not peak_is_free_mode(peak.mode):
                nonfree_indices.append(i)
            if peak_is_artefact_mode(peak.mode):
                artefact_indices.append(i)
                artefact_side[i] = -1 if peak.artefact_side == "left" else 1
            elif peak_is_free_mode(peak.mode):
                local_pos = len(free_indices)
                free_indices.append(i)
                if peak.vary_separation:
                    free_vary_local.append(local_pos)
                else:
                    free_fixed_local.append(local_pos)

        return {
            "peak_mode_code": peak_mode_code,
            "artefact_side": artefact_side,
            "artefact_peak_index": np.array(artefact_indices, dtype=np.int32),
            "free_peak_index": np.array(free_indices, dtype=np.int32),
            "nonfree_peak_index": np.array(nonfree_indices, dtype=np.int32),
            "free_fixed_local_index": np.array(free_fixed_local, dtype=np.int32),
            "free_vary_local_index": np.array(free_vary_local, dtype=np.int32),
        }

    def compute_model_inputs(self) -> dict[str, np.ndarray]:
        """Assemble all model inputs from data, priors, and baseline.

        Intended to be called on views (no subsets) by :meth:`_run_mcmc`.

        Returns
        -------
        dict[str, np.ndarray]
            Keys: all parameters expected by ``better_model.model()``.
        """
        priors, trace_shift_scale = self._compute_position_priors()
        prior_arrays = geometric_priors_to_arrays(priors)
        prior_arrays["trace_shift_scale"] = np.asarray(trace_shift_scale, dtype=np.float32)

        prior_arrays["dominant_area_loc_per_trace"] = self._stabilize_area_prior_matrix(
            prior_arrays["dominant_area_loc_per_trace"].T
        )
        prior_arrays["area_total_loc_per_trace"] = self._stabilize_area_prior_matrix(
            prior_arrays["area_total_loc_per_trace"].T
        )

        peak_structure = self.peak_structure()

        baseline_bp = self.baseline_priors()
        baseline_arrays = {
            "baseline_intercept_loc": np.asarray(baseline_bp.intercept, dtype=float),
            "baseline_intercept_scale": np.asarray(baseline_bp.intercept_scale, dtype=float),
            "baseline_slope_loc": np.asarray(baseline_bp.slope, dtype=float),
            "baseline_slope_scale": np.asarray(baseline_bp.slope_scale, dtype=float),
        }

        noise_arrays = {
            "sigma_y_prior_loc": self.noise_prior(),
        }

        return {
            **prior_arrays,
            **peak_structure,
            **baseline_arrays,
            **noise_arrays,
        }

    def print_priors(self, subset: str | None = None) -> None:
        """Compute and print all prior summaries for *subset* to stdout."""
        print("[Baseline Priors]")
        self._print_baseline_priors(subset=subset)
        print()
        print("[Noise Prior]")
        self._print_noise_prior(subset=subset)
        print()
        print("[Peak Geometry Priors]")
        priors = self.compute_priors(subset=subset)
        print(summarise_priors(priors))

    def _print_baseline_priors(self, subset: str | None = None) -> None:
        bp = self.baseline_priors(subset=subset)
        intercept = np.asarray(bp.intercept, dtype=float)
        slope = np.asarray(bp.slope, dtype=float)
        intercept_scale = np.asarray(bp.intercept_scale, dtype=float)
        slope_scale = np.asarray(bp.slope_scale, dtype=float)

        print(f"{'Trace':>5}  {'Intercept':>12}  {'Int Scale':>10}  {'Slope':>12}  {'Slope Scale':>12}")
        print("-" * 60)
        for t in range(len(intercept)):
            print(
                f"{t:>5}  {intercept[t]:>12.4e}  {intercept_scale[t]:>10.3e}  "
                f"{slope[t]:>12.5e}  {slope_scale[t]:>12.5e}"
            )

    def _print_noise_prior(self, subset: str | None = None) -> None:
        sigma_y = self.noise_prior(subset=subset)
        print(f"{'Trace':>5}  {'Noise s_y':>12}")
        print("-" * 20)
        for t in range(len(sigma_y)):
            print(f"{t:>5}  {sigma_y[t]:>12.3f}")

    def plot_sigma_alpha_prior_diagnostics(
        self,
        *,
        subset: str | None = None,
        figsize: tuple[float, float] | None = None,
        cmap: str = "viridis",
        colorize_by: Literal[None, "sample_id", "subset"] = None,
        prior_colors: list[str] | None = None,
        prior_linecolor: str = "white",
        label_fontsize: float = 9,
        title_fontsize: float = 10,
        tick_fontsize: float = 8,
        spine_linewidth: float = 0.8,
        marker_size: float = 40,
        marker_linewidth: float = 0.9,
        transparent: bool = True,
        show_prior_density: bool = True,
    ) -> tuple[object, np.ndarray]:
        """Plot per-trace FWHM-derived sigma-vs-alpha scatter for each peak.

        Args:
            subset: Subset name to plot.  Resolved automatically when only one
                subset is registered.
            figsize: Figure size (width, height).
            cmap: Colormap for prior density background.
            colorize_by: How to color scatter points.
            prior_colors: Colors for prior density background.
            prior_linecolor: Color for prior density lines.
            label_fontsize: Font size for x/y axis labels.
            title_fontsize: Font size for subplot titles.
            tick_fontsize: Font size for tick labels.
            spine_linewidth: Line width for axis spines and tick marks.
            marker_size: Base scatter marker size (pt²).  When
                *apex_height_trace* is available the height-derived sizes are
                scaled proportionally so both paths stay consistent.
            marker_linewidth: Line width of scatter marker edges.
            transparent: When ``True`` (default) figure and axes backgrounds
                are transparent.
            show_prior_density: When ``True`` (default) show 2-D KDE contours
                for prior density. Set to ``False`` for a cleaner scatter plot.
        """
        from .better_visualize import plot_sigma_alpha_scatter

        diagnostics = self.compute_fwhm_shape_diagnostics(subset=subset)
        priors = self.compute_priors(subset=subset)

        # Get effective peaks for the resolved subset
        if self._subsets:
            s = self._resolve_subset(subset)
            effective_peaks = s.peaks
            mask = self._compute_subset_mask(s)
            sample_ids_arr = self.trace_sample_ids[mask] if self.trace_sample_ids is not None else None
            subset_arr = self.trace_subsets[mask] if self.trace_subsets is not None else None
        else:
            effective_peaks = self.peaks
            sample_ids_arr = self.trace_sample_ids
            subset_arr = self.trace_subsets

        sample_ids: list[str] | None = None
        subset_ids: list[str] | None = None
        if colorize_by == "sample_id" and sample_ids_arr is not None:
            sample_ids = list(sample_ids_arr)
        elif colorize_by == "subset" and subset_arr is not None:
            subset_ids = [str(s) for s in subset_arr]

        fig, axes = plot_sigma_alpha_scatter(
            effective_peaks,
            diagnostics.sigma_trace,
            diagnostics.alpha_trace,
            diagnostics.fwhm_valid_trace,
            apex_height_trace=diagnostics.apex_height_trace,
            sigma_loc=np.asarray([p.sigma_loc for p in priors], dtype=float),
            sigma_scale=np.asarray([p.sigma_scale for p in priors], dtype=float),
            alpha_loc=np.asarray([p.alpha_loc for p in priors], dtype=float),
            alpha_scale=np.asarray([p.alpha_scale for p in priors], dtype=float),
            figsize=figsize,
            cmap=cmap,
            prior_colors=prior_colors,
            prior_linecolor=prior_linecolor,
            colorize_by=colorize_by,
            sample_ids=sample_ids,
            subset_ids=subset_ids,
            label_fontsize=label_fontsize,
            title_fontsize=title_fontsize,
            tick_fontsize=tick_fontsize,
            spine_linewidth=spine_linewidth,
            marker_size=marker_size,
            marker_linewidth=marker_linewidth,
            transparent=transparent,
            show_prior_density=show_prior_density,
        )
        return fig, axes

    def plot_trace_rows(
        self,
        *,
        subset: str | None = None,
        figsize: tuple[float, float] | None = None,
        t_min: float | None = None,
        t_max: float | None = None,
        trace_color: str = "black",
        trace_linewidth: float = 1.0,
        peak_alpha: float = 0.14,
        show_peak_legend: bool = True,
    ) -> tuple[object, np.ndarray]:
        """Plot all chromatograms as stacked full-trace rows.

        Args:
            subset: Subset to plot.  Defaults to the single registered subset.
        """
        from .better_visualize import plot_trace_rows

        if self._subsets:
            s = self._resolve_subset(subset)
            mask = self._compute_subset_mask(s)
            time_plot = self.time[mask]
            signal_plot = self.signal[mask]
            peaks_plot = s.peaks
        else:
            time_plot = self.time
            signal_plot = self.signal
            peaks_plot = self.peaks

        fig, axes = plot_trace_rows(
            time_plot,
            signal_plot,
            peaks_plot,
            figsize=figsize,
            t_min=t_min,
            t_max=t_max,
            trace_color=trace_color,
            trace_linewidth=trace_linewidth,
            peak_alpha=peak_alpha,
            show_peak_legend=show_peak_legend,
        )
        return fig, axes

    def plot_prior_traces(
        self,
        *,
        subset: str | None = None,
        figsize: tuple[float, float] | None = None,
        cmap: str = "viridis",
        show_baseline: bool = True,
        show_apex_prior: bool = True,
        show_gaussian_prior_peak: bool = True,
        show_peak_bounds: bool = True,
    ) -> tuple[object, np.ndarray]:
        """Plot raw traces with baseline, apex prior, and Gaussian peak prior.

        Args:
            subset: Subset to plot.  Defaults to the single registered subset.
        """
        from .better_visualize import plot_prior_traces

        if self._subsets:
            s = self._resolve_subset(subset)
            mask = self._compute_subset_mask(s)
            time_plot = self.time[mask]
            signal_plot = self.signal[mask]
            peaks_plot = s.peaks
        else:
            time_plot = self.time
            signal_plot = self.signal
            peaks_plot = self.peaks

        bp = self.baseline_priors(subset=subset if self._subsets else None)
        peak_priors = self.compute_priors(subset=subset if self._subsets else None)
        diagnostics = self.compute_fwhm_shape_diagnostics(subset=subset if self._subsets else None)

        fig, axes = plot_prior_traces(
            time_plot,
            signal_plot,
            peaks_plot,
            np.asarray(bp.intercept, dtype=float),
            np.asarray(bp.slope, dtype=float),
            np.asarray(bp.intercept_scale, dtype=float),
            np.asarray(bp.slope_scale, dtype=float),
            np.asarray([p.apex_loc for p in peak_priors], dtype=float),
            np.asarray([p.apex_scale for p in peak_priors], dtype=float),
            approx_apex_trace=diagnostics.approx_apex_trace,
            approx_height_trace=diagnostics.approx_height_trace,
            approx_sigma_trace=diagnostics.approx_sigma_trace,
            approx_valid_trace=diagnostics.approx_valid_trace,
            approx_fallback_trace=diagnostics.approx_fallback_trace,
            show_baseline=show_baseline,
            show_apex_prior=show_apex_prior,
            show_gaussian_prior_peak=show_gaussian_prior_peak,
            show_peak_bounds=show_peak_bounds,
            figsize=figsize,
            cmap=cmap,
        )
        return fig, axes

    # ------------------------------------------------------------------
    # Trace selection and posterior evaluation helpers
    # ------------------------------------------------------------------

    def select_trace_indices(
        self,
        *,
        sample_ids: list[str] | None = None,
        chromatogram_ids: list[str] | None = None,
        subset_id: str | None = None,
    ) -> np.ndarray:
        """Return integer indices of traces matching all supplied filters.

        Filters are applied as an intersection (AND).  When no filter is given
        every trace index is returned.

        Args:
            sample_ids: Keep only traces whose ``trace_sample_ids`` value is
                in this list.  Requires the fitter to have been built with
                :meth:`from_handler`.
            chromatogram_ids: Keep only traces whose
                ``trace_chromatogram_ids`` value is in this list.
            subset_id: Keep only traces belonging to the named subset.

        Returns:
            ``np.ndarray`` of shape ``[n_selected]``, dtype ``int``.
        """
        indices = np.arange(self.n_traces)

        if subset_id is not None:
            s = self._resolve_subset(subset_id)
            mask = self._compute_subset_mask(s)
            indices = np.intersect1d(indices, np.where(mask)[0])

        if sample_ids is not None:
            if self.trace_sample_ids is None:
                raise RuntimeError(
                    "sample_ids filter requires trace_sample_ids. "
                    "Build the fitter with BetterFitter.from_handler()."
                )
            keep = np.where(np.isin(self.trace_sample_ids, sample_ids))[0]
            indices = np.intersect1d(indices, keep)

        if chromatogram_ids is not None:
            if self.trace_chromatogram_ids is None:
                raise RuntimeError(
                    "chromatogram_ids filter requires trace_chromatogram_ids. "
                    "Build the fitter with BetterFitter.from_handler()."
                )
            keep = np.where(np.isin(self.trace_chromatogram_ids, chromatogram_ids))[0]
            indices = np.intersect1d(indices, keep)

        return indices

    def window_mask(self, rt_min: float, rt_max: float) -> np.ndarray:
        """Boolean mask on :meth:`common_time` for ``[rt_min, rt_max]``.

        Args:
            rt_min: Window left edge (minutes).
            rt_max: Window right edge (minutes).

        Returns:
            1-D ``bool`` array of shape ``[n_time]``.
        """
        t = self.common_time()
        return (t >= float(rt_min)) & (t <= float(rt_max))

    def posterior_curves(
        self,
        x: np.ndarray,
        *,
        subset: str | None = None,
        hdi_prob: float = 0.95,
        trace_indices: np.ndarray | None = None,
        n_samples_max: int = 2000,
    ) -> PosteriorCurves:
        """Evaluate posterior HDI curves on an arbitrary time grid.

        Re-evaluates the skew-normal components at every point in *x* using
        the stored posterior samples; no interpolation.

        Args:
            x: Evaluation axis, shape ``[n_x]`` (minutes).
            subset: Subset name.  Resolved automatically when only one subset
                is fitted.
            hdi_prob: Credible-interval probability (default 0.95).
            trace_indices: Global trace indices (into the full fitter array) to
                include.  ``None`` = all traces in the subset.
            n_samples_max: Maximum number of posterior draws to use (capped for
                memory).  Default 2000.

        Returns:
            :class:`PosteriorCurves` with pre-computed median / lower / upper
            for total signal, baseline, and per-component curves.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
            KeyError: If the requested subset has no fitted posterior.
        """
        from scipy.special import ndtr as _ndtr

        if not self._posteriors:
            raise RuntimeError("posterior_curves() requires a fitted posterior. Call fit() first.")

        # Resolve subset name
        if subset is not None:
            subset_name = subset
        elif len(self._posteriors) == 1:
            subset_name = next(iter(self._posteriors))
        else:
            raise ValueError(f"Multiple fitted subsets: {list(self._posteriors)}. Specify subset=<name>.")
        if subset_name not in self._posteriors:
            raise KeyError(
                f"No fitted posterior for subset '{subset_name}'. Fitted subsets: {list(self._posteriors)}."
            )

        post = self._posteriors[subset_name]
        subset_obj = self._subsets[subset_name]
        subset_mask = self._compute_subset_mask(subset_obj)
        all_subset_idx = np.where(subset_mask)[0]  # global indices
        n_subset = len(all_subset_idx)

        # Map requested trace_indices to local (within-subset posterior) positions
        if trace_indices is None:
            local_idx = np.arange(n_subset)
            global_idx = all_subset_idx
        else:
            local_list, global_list = [], []
            for gi in np.asarray(trace_indices, dtype=int):
                pos = np.where(all_subset_idx == gi)[0]
                if len(pos) > 0:
                    local_list.append(int(pos[0]))
                    global_list.append(int(gi))
            local_idx = np.array(local_list, dtype=int)
            global_idx = np.array(global_list, dtype=int)

        n_selected = len(local_idx)
        if n_selected == 0:
            raise ValueError(
                "posterior_curves(): no traces matched the given trace_indices "
                f"within subset '{subset_name}'."
            )

        # Extract 4-D posterior arrays [n_chain, n_draw, n_trace, n_peak]
        pvar = post.posterior
        xi_l_raw = np.asarray(pvar["xi_l"].values)
        xi_r_raw = np.asarray(pvar["xi_r"].values)
        sigma_l_raw = np.asarray(pvar["sigma_l"].values)
        sigma_r_raw = np.asarray(pvar["sigma_r"].values)
        alpha_l_raw = np.asarray(pvar["alpha_l"].values)
        alpha_r_raw = np.asarray(pvar["alpha_r"].values)
        area_l_raw = np.asarray(pvar["area_l"].values)
        area_r_raw = np.asarray(pvar["area_r"].values)
        bl_int_raw = np.asarray(pvar["baseline_intercept"].values)  # [n_chain,n_draw,n_trace]
        bl_slp_raw = np.asarray(pvar["baseline_slope"].values)

        n_chain, n_draw = xi_l_raw.shape[:2]
        n_total = n_chain * n_draw

        def _flatten(arr: np.ndarray) -> np.ndarray:
            return arr.reshape(n_total, *arr.shape[2:])

        def _select(arr: np.ndarray) -> np.ndarray:
            """Flatten chains then select traces."""
            return _flatten(arr)[:, local_idx]

        xi_l = _select(xi_l_raw)  # [n_total, n_sel, n_peak]
        xi_r = _select(xi_r_raw)
        sigma_l = _select(sigma_l_raw)
        sigma_r = _select(sigma_r_raw)
        alpha_l = _select(alpha_l_raw)
        alpha_r = _select(alpha_r_raw)
        area_l = _select(area_l_raw)
        area_r = _select(area_r_raw)
        bl_int = _flatten(bl_int_raw)[:, local_idx]  # [n_total, n_sel]
        bl_slp = _flatten(bl_slp_raw)[:, local_idx]

        # Subsample draws
        if n_total > n_samples_max:
            rng = np.random.default_rng(0)
            idx = rng.choice(n_total, size=n_samples_max, replace=False)
            xi_l, xi_r = xi_l[idx], xi_r[idx]
            sigma_l, sigma_r = sigma_l[idx], sigma_r[idx]
            alpha_l, alpha_r = alpha_l[idx], alpha_r[idx]
            area_l, area_r = area_l[idx], area_r[idx]
            bl_int, bl_slp = bl_int[idx], bl_slp[idx]
            n_samp = n_samples_max
        else:
            n_samp = n_total

        x_eval = np.asarray(x, dtype=float)  # [n_x]
        n_x = len(x_eval)
        n_peak = xi_l.shape[2]

        # Merge sample x trace dims for vectorized PDF evaluation
        n_flat = n_samp * n_selected

        def _merge(arr: np.ndarray) -> np.ndarray:
            return arr.reshape(n_flat, n_peak)

        xi_l_f = _merge(xi_l)
        xi_r_f = _merge(xi_r)
        sigma_l_f = _merge(sigma_l)
        sigma_r_f = _merge(sigma_r)
        alpha_l_f = _merge(alpha_l)
        alpha_r_f = _merge(alpha_r)
        area_l_f = _merge(area_l)
        area_r_f = _merge(area_r)

        # Broadcast x: [n_flat, n_x]
        x_flat = np.broadcast_to(x_eval[None, :], (n_flat, n_x)).copy()

        def _skew_normal_pdf(
            x_2d: np.ndarray,  # [n, n_x]
            xi: np.ndarray,  # [n, n_comp]
            sigma: np.ndarray,  # [n, n_comp]
            alpha: np.ndarray,  # [n, n_comp]
        ) -> np.ndarray:  # [n, n_comp, n_x]
            sig = np.maximum(sigma[:, :, None], 1e-6)
            z = (x_2d[:, None, :] - xi[:, :, None]) / sig
            log_pdf = (
                np.log(2.0)
                - np.log(sig)
                - 0.5 * z**2
                - 0.5 * np.log(2.0 * np.pi)
                + np.log(np.clip(_ndtr(alpha[:, :, None] * z), 1e-300, None))
            )
            return np.exp(log_pdf)

        pdf_l = _skew_normal_pdf(x_flat, xi_l_f, sigma_l_f, alpha_l_f)  # [n_flat, n_peak, n_x]
        pdf_r = _skew_normal_pdf(x_flat, xi_r_f, sigma_r_f, alpha_r_f)

        comp_l_f = area_l_f[:, :, None] * pdf_l  # [n_flat, n_peak, n_x]
        comp_r_f = area_r_f[:, :, None] * pdf_r

        # Reshape back to [n_samp, n_sel, n_peak, n_x]
        comp_l = comp_l_f.reshape(n_samp, n_selected, n_peak, n_x)
        comp_r = comp_r_f.reshape(n_samp, n_selected, n_peak, n_x)

        # Baseline: [n_samp, n_sel, n_x]
        baseline_samps = bl_int[:, :, None] + bl_slp[:, :, None] * x_eval[None, None, :]

        # Total: [n_samp, n_sel, n_x]
        total_samps = comp_l.sum(axis=2) + comp_r.sum(axis=2) + baseline_samps

        # HDI percentiles
        lo_pct = 100.0 * (1.0 - hdi_prob) / 2.0
        hi_pct = 100.0 - lo_pct

        def _pct(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return (
                np.percentile(arr, 50, axis=0),
                np.percentile(arr, lo_pct, axis=0),
                np.percentile(arr, hi_pct, axis=0),
            )

        t_med, t_lo, t_hi = _pct(total_samps)  # each [n_sel, n_x]
        bl_med, bl_lo, bl_hi = _pct(baseline_samps)
        cl_med, cl_lo, cl_hi = _pct(comp_l)  # each [n_sel, n_peak, n_x]
        cr_med, cr_lo, cr_hi = _pct(comp_r)

        chrom_ids: list[str] | None = (
            list(self.trace_chromatogram_ids[global_idx]) if self.trace_chromatogram_ids is not None else None
        )

        return PosteriorCurves(
            x=x_eval,
            total_median=t_med,
            total_lower=t_lo,
            total_upper=t_hi,
            baseline_median=bl_med,
            baseline_lower=bl_lo,
            baseline_upper=bl_hi,
            comp_l_median=cl_med,
            comp_l_lower=cl_lo,
            comp_l_upper=cl_hi,
            comp_r_median=cr_med,
            comp_r_lower=cr_lo,
            comp_r_upper=cr_hi,
            trace_indices=global_idx,
            chromatogram_ids=chrom_ids,
        )

    def plot_fit(
        self,
        *,
        subset: str | None = None,
        sample_ids: list[str] | None = None,
        chromatogram_ids: list[str] | None = None,
        hdi_prob: float = 0.95,
        n_samples_max: int = 2000,
        figsize: tuple[float, float] | None = None,
        colors: list[str] | None = None,
    ) -> tuple[object, np.ndarray]:
        """Plot raw data and posterior fit for *subset*.

        Raw scatter is shown for every **display trace** (determined by
        *sample_ids* / *chromatogram_ids*, or all traces in the subset when
        neither is specified).  Posterior curves are overlaid only on traces
        that belong to the fitted subset.

        Args:
            subset: Subset to display/plot posterior for.  Resolved
                automatically when only one subset is registered.
            sample_ids: Restrict display to traces with these sample IDs.
            chromatogram_ids: Restrict display to traces with these
                chromatogram IDs.
            hdi_prob: Credible-interval probability passed to
                :meth:`posterior_curves` (default 0.95).
            n_samples_max: Max posterior draws used for HDI evaluation.
            figsize: Figure size; auto-scaled when ``None``.
            colors: List of hex color codes (e.g., ['#FF5733', '#33FF57'])
                for the total fitted signal per peak.  Length must match the
                number of peaks in the subset.  When ``None`` (default), uses
                blue ('C0') for all peaks.

        Returns:
            ``(fig, axes)`` — the matplotlib Figure and ``[n_display, n_col]``
            axes array.

        Raises:
            RuntimeError: If no subsets are registered.
        """
        from .better_visualize import plot_fit as _bv_plot_fit

        if not self._subsets:
            raise RuntimeError(
                "plot_fit() requires at least one subset with peak annotations. "
                "Call add_peak_annotation() or add_subset() first."
            )

        # Resolve subset name
        if subset is not None:
            subset_name = subset
        elif len(self._subsets) == 1:
            subset_name = next(iter(self._subsets))
        else:
            raise ValueError(f"Multiple subsets: {list(self._subsets)}. Specify subset=<name>.")

        subset_obj = self._subsets[subset_name]
        subset_mask = self._compute_subset_mask(subset_obj)
        all_subset_idx = np.where(subset_mask)[0]  # global indices

        # Determine display traces
        if sample_ids is not None or chromatogram_ids is not None:
            display_idx = self.select_trace_indices(
                sample_ids=sample_ids,
                chromatogram_ids=chromatogram_ids,
            )
        else:
            display_idx = all_subset_idx

        # Traces that are BOTH displayed and fitted
        fitted_display_idx = np.intersect1d(display_idx, all_subset_idx)

        # Build evaluation range from peak windows + baseline annotations
        peaks = subset_obj.peaks
        eff_baselines = subset_obj.baselines if subset_obj.baselines else self.baselines
        rt_bounds: list[float] = []
        for p in peaks:
            rt_bounds += [float(p.rt_min), float(p.rt_max)]
        for bl in eff_baselines:
            rt_bounds += [float(bl.rt_min), float(bl.rt_max)]
        if rt_bounds:
            rt_lo, rt_hi = min(rt_bounds), max(rt_bounds)
        else:
            common_t = self.common_time()
            rt_lo, rt_hi = float(common_t[0]), float(common_t[-1])

        x_eval = np.linspace(rt_lo, rt_hi, 300)

        # Compute posterior curves if the subset has been fitted
        curves: PosteriorCurves | None = None
        if subset_name in self._posteriors and len(fitted_display_idx) > 0:
            curves = self.posterior_curves(
                x_eval,
                subset=subset_name,
                hdi_prob=hdi_prob,
                trace_indices=fitted_display_idx,
                n_samples_max=n_samples_max,
            )

        # Rows in display_idx that have posterior curves
        if curves is not None:
            fitted_rows = np.where(np.isin(display_idx, fitted_display_idx))[0]
        else:
            fitted_rows = np.array([], dtype=int)

        time_display = self.time[display_idx]
        signal_display = self.signal[display_idx]
        chrom_ids_display: list[str] | None = (
            list(self.trace_chromatogram_ids[display_idx])
            if self.trace_chromatogram_ids is not None
            else None
        )

        return _bv_plot_fit(
            time_display,
            signal_display,
            peaks,
            curves,
            fitted_rows=fitted_rows,
            baselines=eff_baselines if eff_baselines else None,
            chromatogram_ids=chrom_ids_display,
            hdi_prob=hdi_prob,
            figsize=figsize,
            colors=colors,
        )

    # ------------------------------------------------------------------
    # Posteriors property
    # ------------------------------------------------------------------

    @property
    def posteriors(self) -> dict[str, object]:
        """Fitted posteriors keyed by subset name.

        Returns an empty dict before :meth:`fit` is called, and a dict mapping
        subset names to ArviZ ``InferenceData`` objects after fitting.

        Use ``fitter.posteriors["__default__"]`` for single-group fits, or
        ``fitter.posteriors["subset_name"]`` for multi-subset fits.
        """
        return dict(self._posteriors)

    @property
    def elbo_history(self) -> dict[str, np.ndarray]:
        """ELBO loss history keyed by subset name.

        Populated only after :meth:`fit` is called with ``backend="svi"``.
        Returns an empty dict for MCMC runs.
        """
        return dict(self._elbo_losses)

    def plot_elbo(
        self,
        subset: str | None = None,
        *,
        log_scale: bool = False,
        window: int | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> object:
        """Plot SVI ELBO convergence history for *subset*.

        Args:
            subset: Subset name.  Auto-resolved when only one subset exists.
            log_scale: If ``True``, use a symlog y-axis.
            window: If not ``None``, overlay a rolling mean with this window
                size over the raw ELBO curve.
            figsize: Figure size ``(width, height)``.

        Returns:
            ``matplotlib.figure.Figure`` — caller is responsible for showing
            or saving.

        Raises:
            RuntimeError: If :meth:`fit` with ``backend="svi"`` has not been
                called, or if ELBO history is unavailable for *subset*.
        """
        import matplotlib.pyplot as plt

        subset_name = self._resolve_subset(subset).name
        if subset_name not in self._elbo_losses:
            raise RuntimeError(f"No ELBO history for subset '{subset_name}'. Call fit(backend='svi') first.")
        losses = self._elbo_losses[subset_name]
        steps = np.arange(len(losses))

        fig, ax = plt.subplots(figsize=figsize or (8, 3))
        ax.plot(steps, losses, alpha=0.4, linewidth=0.8, color="steelblue", label="ELBO")
        if window is not None:
            rolled = np.convolve(losses, np.ones(window) / window, mode="valid")
            ax.plot(
                steps[window - 1 :],
                rolled,
                linewidth=1.5,
                color="steelblue",
                label=f"Rolling mean (window={window})",
            )
            ax.legend(fontsize=8)
        ax.set_xlabel("Step")
        ax.set_ylabel("ELBO loss")
        if log_scale:
            ax.set_yscale("symlog")
        ax.set_title(f"SVI convergence — subset '{subset_name}'")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Inference (MCMC or SVI)
    # ------------------------------------------------------------------

    def fit(
        self,
        # --- MCMC parameters (backend="mcmc") ---
        num_samples: int = 1000,
        num_warmup: int = 500,
        num_chains: int = 1,
        # --- SVI parameters (backend="svi") ---
        num_steps: int = 10_000,
        lr: float = 1e-3,
        guide_type: GuideType = "diagonal",
        low_rank_rank: int = 10,
        n_posterior_samples: int = 2000,
        save_elbo: str | None = None,
        # --- shared ---
        backend: Literal["mcmc", "svi"] = "mcmc",
        seed: int = 0,
        progress_bar: bool = True,
        save_summary: str | None = None,
        subsets: list[str] | None = None,
    ) -> None:
        """Run Bayesian inference on all registered subsets.

        Each registered subset is fitted as an independent inference run.
        Results are stored in :attr:`posteriors` keyed by subset name.

        Parameters
        ----------
        backend : {"mcmc", "svi"}
            Inference backend (default ``"mcmc"``).

            - ``"mcmc"`` — NUTS sampler via NumPyro (accurate, slow).
            - ``"svi"`` — Stochastic Variational Inference via Adam
              (approximate, fast).

        num_samples : int
            MCMC: number of samples to draw per chain (default 1000).
        num_warmup : int
            MCMC: number of warmup (burn-in) iterations (default 500).
        num_chains : int
            MCMC: number of independent chains (default 1).
        num_steps : int
            SVI: number of Adam optimisation steps (default 10 000).
        lr : float
            SVI: Adam learning rate (default 1e-3).
        guide_type : {"diagonal", "full_rank", "low_rank"}
            SVI: variational family.

            - ``"diagonal"`` — AutoNormal mean-field (fastest, default).
            - ``"low_rank"`` — AutoLowRankMultivariateNormal (captures
              some posterior correlations; use ``low_rank_rank`` to set
              the rank).
            - ``"full_rank"`` — AutoMultivariateNormal (full covariance;
              expensive for large models).

        low_rank_rank : int
            SVI: rank for ``guide_type="low_rank"`` (default 10).
        n_posterior_samples : int
            SVI: number of samples to draw from the trained guide for
            downstream posterior evaluation (default 2000).
        save_elbo : str or None
            SVI: if provided, save the ELBO loss array to this path.
        seed : int
            Random seed for reproducibility (default 0).  Incremented by
            one for each successive subset.
        progress_bar : bool
            Whether to show a progress bar during inference (default True).
        save_summary : str or None
            If provided, save the ArviZ posterior summary to this file path
            (or path prefix when fitting multiple subsets — the subset name
            is appended before the file extension).
        subsets : list[str] or None
            When not ``None``, fit only the named subsets.  All registered
            subsets are fitted when ``None`` (default).
        """
        if not self._subsets:
            raise RuntimeError(
                "fit() requires at least one peak annotation. "
                "Call add_peak_annotation() or add_subset() + subset.add_peak_annotation() first."
            )

        # Determine which subsets to fit
        if subsets is not None:
            unknown = [n for n in subsets if n not in self._subsets]
            if unknown:
                raise ValueError(f"Unknown subset(s): {unknown}. Registered subsets: {list(self._subsets)}.")
            active = {n: self._subsets[n] for n in subsets}
        else:
            active = dict(self._subsets)

        # Validate: each active subset must have peaks
        for name, subset in active.items():
            if not subset.peaks:
                raise RuntimeError(
                    f"Subset '{name}' has no peak annotations. "
                    "Call subset.add_peak_annotation() before fit()."
                )

        for i, (name, _subset) in enumerate(active.items()):
            if len(active) > 1:
                print(f"\n{'=' * 80}")
                print(f"Fitting subset '{name}' ({i + 1}/{len(active)})")
                print(f"{'=' * 80}")

            view = self._make_subset_view(name)

            subset_summary: str | None = None
            if save_summary is not None:
                import os

                base, ext = os.path.splitext(save_summary)
                subset_summary = f"{base}_{name}{ext}" if len(active) > 1 else save_summary

            if backend == "mcmc":
                view._run_mcmc(
                    num_samples=num_samples,
                    num_warmup=num_warmup,
                    num_chains=num_chains,
                    seed=seed + i,
                    progress_bar=progress_bar,
                    save_summary=subset_summary,
                )
            elif backend == "svi":
                import os as _os

                subset_elbo: str | None = None
                if save_elbo is not None:
                    _base, _ext = _os.path.splitext(save_elbo)
                    subset_elbo = f"{_base}_{name}{_ext}" if len(active) > 1 else save_elbo
                view._run_svi(
                    num_steps=num_steps,
                    lr=lr,
                    guide_type=guide_type,
                    low_rank_rank=low_rank_rank,
                    n_posterior_samples=n_posterior_samples,
                    seed=seed + i,
                    progress_bar=progress_bar,
                    save_summary=subset_summary,
                    save_elbo=subset_elbo,
                )
            else:
                raise ValueError(f"backend must be 'mcmc' or 'svi'; got {backend!r}")

            # Transfer results from the transient view to parent storage
            self._posteriors[name] = view.posterior  # type: ignore[attr-defined]
            self._samples_dict[name] = view.samples  # type: ignore[arg-type]
            self._subset_trace_ids[name] = (
                view.trace_chromatogram_ids
                if view.trace_chromatogram_ids is not None
                else np.array([], dtype=object)
            )
            if view.x_masked is not None:
                self._x_masked[name] = np.asarray(view.x_masked)
            if view.elbo_losses is not None:
                self._elbo_losses[name] = view.elbo_losses
            # view is intentionally discarded here

    def _run_mcmc(
        self,
        num_samples: int = 1000,
        num_warmup: int = 500,
        num_chains: int = 1,
        seed: int = 0,
        progress_bar: bool = True,
        save_summary: str | None = None,
    ) -> None:
        """Execute a single MCMC run on this fitter's traces and peaks.

        Intended to be called on views produced by :meth:`_make_subset_view`.
        Sets ``self.mcmc``, ``self.samples``, and ``self.posterior`` on *self*.
        """
        model_inputs = self.compute_model_inputs()

        x_masked, y_masked = self.slice_to_observed_windows()

        self.x_masked = x_masked
        self.y_masked = y_masked

        model_inputs["x"] = jnp.asarray(x_masked, dtype=jnp.float32)
        model_inputs["y"] = jnp.asarray(y_masked, dtype=jnp.float32)

        for key in model_inputs:
            if isinstance(model_inputs[key], np.ndarray):
                value = model_inputs[key]
                if np.issubdtype(value.dtype, np.integer):
                    model_inputs[key] = jnp.asarray(value, dtype=jnp.int32)
                elif np.issubdtype(value.dtype, np.bool_):
                    model_inputs[key] = jnp.asarray(value, dtype=bool)
                else:
                    model_inputs[key] = jnp.asarray(value, dtype=jnp.float32)

        model_param_names = {
            "x",
            "y",
            "peak_mode_code",
            "artefact_side",
            "artefact_peak_index",
            "free_peak_index",
            "nonfree_peak_index",
            "free_fixed_local_index",
            "free_vary_local_index",
            "apex_loc",
            "apex_scale",
            "trace_shift_scale",
            "sigma_loc",
            "sigma_scale",
            "alpha_loc",
            "alpha_scale",
            "dominant_area_loc_per_trace",
            "area_total_loc_per_trace",
            "artefact_area_loc_shared",
            "window_lo",
            "window_hi",
            "baseline_intercept_loc",
            "baseline_intercept_scale",
            "baseline_slope_loc",
            "baseline_slope_scale",
            "sigma_y_prior_loc",
        }
        model_inputs_filtered = {k: v for k, v in model_inputs.items() if k in model_param_names}

        self.mcmc = MCMC(
            NUTS(better_model.model),
            num_warmup=int(num_warmup),
            num_samples=int(num_samples),
            num_chains=int(num_chains),
            progress_bar=bool(progress_bar),
            chain_method="parallel" if num_chains > 1 else "sequential",
        )

        self.mcmc.run(jax.random.PRNGKey(int(seed)), **model_inputs_filtered)

        self.samples = self.mcmc.get_samples()

        import arviz as az

        self.posterior = az.from_numpyro(self.mcmc)

        available_vars = list(self.posterior.posterior.data_vars)
        summary_vars = [v for v in better_model.SUMMARY_PARAMETER_NAMES if v in available_vars]
        summary_df = az.summary(self.posterior, var_names=summary_vars)
        print("\n" + "=" * 80)
        print("ArviZ Posterior Summary")
        print("=" * 80)
        print(summary_df.to_string())

        if save_summary is not None:
            with open(save_summary, "w", encoding="utf-8") as f:
                f.write(summary_df.to_string())
            print(f"\n✓ Summary saved to: {save_summary}")

    @staticmethod
    def _svi_samples_to_idata(
        guide: object,
        svi_params: dict,
        model: object,
        model_inputs: dict,
        n_posterior_samples: int,
        seed: int,
    ) -> tuple[object, dict]:
        """Draw samples from a trained VI guide and convert to ArviZ InferenceData.

        Returns ``(idata, samples_dict)`` where:

        - ``idata.posterior`` has shape ``[n_chain=1, n_draw=n_posterior_samples, ...]``
          for all variables (sampled + deterministic), matching what
          :meth:`posterior_curves` expects.
        - ``samples_dict`` has shape ``[n_draw, n_trace, n_peak]`` for ``area_l``,
          ``area_r``, ``apex_l``, ``apex_r``, matching what
          :meth:`_peaks_from_samples` and :meth:`_records_from_samples` expect.
        """
        import arviz as az

        key1, key2 = jax.random.split(jax.random.PRNGKey(seed + 1))

        # ── Step 1: sample all LATENT variables from the variational posterior ──
        #
        # Predictive(model, guide=guide, ...) internally calls
        # condition(model, guide_samples), which marks every guided site as
        # "observed" in the forward-pass trace.  NumPyro's Predictive then
        # EXCLUDES those conditioned sample sites from its output dict — only
        # numpyro.deterministic sites survive.  That is why baseline_intercept,
        # sigma_y, etc. went missing while baseline_curve (a deterministic
        # derived from them) appeared.
        #
        # Fix: run the guide alone to get all latent samples explicitly.
        guide_predictive = Predictive(guide, params=svi_params, num_samples=n_posterior_samples)
        latent_samples: dict = guide_predictive(key1, **model_inputs)
        # latent_samples: {baseline_intercept: [N, n_trace], sigma_y: [N, n_trace],
        #                  apex: [N, n_trace, n_peak], …}

        # ── Step 2: run the model forward to collect DETERMINISTIC sites ──
        #
        # Predictive with posterior_samples conditions the model on the guide's
        # draws and returns all numpyro.deterministic sites (xi_l, area_l, …).
        # Conditioned sample sites are still absent here — that's fine, because
        # we already have them from step 1.
        det_predictive = Predictive(
            model,
            posterior_samples=latent_samples,
            exclude_deterministic=False,
        )
        det_raw: dict = det_predictive(key2, **model_inputs)

        # ── Step 3: merge — latents (guide) U deterministics (model forward) ──
        raw: dict = {**det_raw, **latent_samples}
        raw.pop("y", None)  # drop the posterior-predictive obs draw

        # ArviZ expects shape [n_chain, n_draw, ...]; add the chain dim via [None].
        posterior_dict = {k: np.asarray(v)[None] for k, v in raw.items()}
        idata = az.from_dict(posterior=posterior_dict)

        # samples_dict is consumed by _peaks_from_samples / _records_from_samples
        # which expect [n_draw, n_trace, n_peak] — no chain dim.
        samples_dict = {k: np.asarray(v) for k, v in raw.items()}

        return idata, samples_dict

    def _run_svi(
        self,
        num_steps: int = 10_000,
        lr: float = 1e-3,
        guide_type: GuideType = "diagonal",
        low_rank_rank: int = 10,
        n_posterior_samples: int = 2000,
        seed: int = 0,
        progress_bar: bool = True,
        save_summary: str | None = None,
        save_elbo: str | None = None,
    ) -> None:
        """Execute a single SVI run on this fitter's traces and peaks.

        Intended to be called on views produced by :meth:`_make_subset_view`.
        Sets ``self.svi_result``, ``self.elbo_losses``, ``self.samples``, and
        ``self.posterior`` on *self*.
        """
        import arviz as az

        model_inputs = self.compute_model_inputs()

        x_masked, y_masked = self.slice_to_observed_windows()
        self.x_masked = x_masked
        self.y_masked = y_masked

        model_inputs["x"] = jnp.asarray(x_masked, dtype=jnp.float32)
        model_inputs["y"] = jnp.asarray(y_masked, dtype=jnp.float32)

        for key in model_inputs:
            if isinstance(model_inputs[key], np.ndarray):
                value = model_inputs[key]
                if np.issubdtype(value.dtype, np.integer):
                    model_inputs[key] = jnp.asarray(value, dtype=jnp.int32)
                elif np.issubdtype(value.dtype, np.bool_):
                    model_inputs[key] = jnp.asarray(value, dtype=bool)
                else:
                    model_inputs[key] = jnp.asarray(value, dtype=jnp.float32)

        model_param_names = {
            "x",
            "y",
            "peak_mode_code",
            "artefact_side",
            "artefact_peak_index",
            "free_peak_index",
            "nonfree_peak_index",
            "free_fixed_local_index",
            "free_vary_local_index",
            "apex_loc",
            "apex_scale",
            "trace_shift_scale",
            "sigma_loc",
            "sigma_scale",
            "alpha_loc",
            "alpha_scale",
            "dominant_area_loc_per_trace",
            "area_total_loc_per_trace",
            "artefact_area_loc_shared",
            "window_lo",
            "window_hi",
            "baseline_intercept_loc",
            "baseline_intercept_scale",
            "baseline_slope_loc",
            "baseline_slope_scale",
            "sigma_y_prior_loc",
        }
        model_inputs_filtered = {k: v for k, v in model_inputs.items() if k in model_param_names}

        guide = _build_guide(guide_type, better_model.model, low_rank_rank)
        optimizer = numpyro_optim.Adam(step_size=lr)
        svi = SVI(better_model.model, guide, optimizer, loss=Trace_ELBO())

        svi_result = svi.run(
            jax.random.PRNGKey(int(seed)),
            num_steps=int(num_steps),
            progress_bar=bool(progress_bar),
            **model_inputs_filtered,
        )
        self.svi_result = svi_result
        self.elbo_losses = np.asarray(svi_result.losses)

        self.posterior, self.samples = BetterFitter._svi_samples_to_idata(
            guide=guide,
            svi_params=svi_result.params,
            model=better_model.model,
            model_inputs=model_inputs_filtered,
            n_posterior_samples=int(n_posterior_samples),
            seed=int(seed),
        )

        available_vars = list(self.posterior.posterior.data_vars)
        summary_vars = [v for v in better_model.SUMMARY_PARAMETER_NAMES if v in available_vars]
        summary_df = az.summary(self.posterior, var_names=summary_vars)
        print("\n" + "=" * 80)
        print("ArviZ Posterior Summary (SVI)")
        print("=" * 80)
        print(summary_df.to_string())

        if save_summary is not None:
            with open(save_summary, "w", encoding="utf-8") as f:
                f.write(summary_df.to_string())
            print(f"\n✓ Summary saved to: {save_summary}")

        if save_elbo is not None:
            np.savetxt(save_elbo, self.elbo_losses, header="elbo_loss")
            print(f"✓ ELBO losses saved to: {save_elbo}")

    # ------------------------------------------------------------------
    # Posterior area extraction (view-level helpers)
    # ------------------------------------------------------------------

    @staticmethod
    def _molecule_area_slice(
        peak: PeakAnnotation,
        area_l: np.ndarray,
        area_r: np.ndarray,
    ) -> np.ndarray:
        """Return the area array that represents the molecule's concentration.

        Centralises mode-dependent area-selection so every downstream method
        uses identical logic:

        * ``single`` — left component only (right is zero by construction).
        * ``free_doublet`` — sum of both components.
        * ``artefact_doublet`` — dominant-side component only, **unless**
          ``include_artefact_in_area=True``, in which case both are summed.
        """
        if peak.mode == "single":
            return area_l
        if peak.mode == "free_doublet":
            return area_l + area_r
        # artefact_doublet
        if peak.include_artefact_in_area:
            return area_l + area_r
        if peak.artefact_side == "left":
            return area_r  # artefact on left → dominant on right
        return area_l  # artefact on right → dominant on left

    @property
    def posterior_area_matrix(self) -> np.ndarray:
        """Median posterior area matrix ``[n_trace, n_peak, 2]``.

        Available on views (after :meth:`_run_mcmc`) and for single-subset
        parent fitters (after :meth:`fit`).

        Axis ``-1`` holds ``[left_component, right_component]``.
        """
        samples = self._get_view_samples()
        area_l = np.asarray(samples["area_l"])
        area_r = np.asarray(samples["area_r"])
        return np.stack(
            [np.median(area_l, axis=0), np.median(area_r, axis=0)],
            axis=-1,
        )

    def molecule_areas(
        self,
        *,
        quantiles: tuple[float, float, float] = (0.05, 0.5, 0.95),
    ) -> np.ndarray:
        """Posterior median molecule-relevant area ``[n_trace, n_peak]``.

        Available on views and single-subset parent fitters.
        """
        samples = self._get_view_samples()
        peaks = self._get_view_peaks()
        area_l = np.asarray(samples["area_l"])
        area_r = np.asarray(samples["area_r"])
        mol_area = np.empty_like(area_l)
        for p_idx, peak in enumerate(peaks):
            mol_area[..., p_idx] = self._molecule_area_slice(peak, area_l[..., p_idx], area_r[..., p_idx])
        _, q_med, _ = quantiles
        return np.quantile(mol_area, q_med, axis=0)

    def _get_view_samples(self) -> dict:
        """Return samples dict — works on views and single-subset parent fitters."""
        if self.samples is not None:
            return self.samples
        if len(self._samples_dict) == 1:
            return next(iter(self._samples_dict.values()))
        raise RuntimeError(
            "posterior_area_matrix / molecule_areas() require a fitted posterior. "
            "Call fit() first, or use to_peaks() / area_records() for multi-subset fitters."
        )

    def _get_view_peaks(self) -> list[PeakAnnotation]:
        """Return peaks — works on views and single-subset parent fitters."""
        if self.peaks:
            return self.peaks
        if len(self._subsets) == 1:
            return next(iter(self._subsets.values())).peaks
        raise RuntimeError("Cannot determine peaks for multi-subset fitter.")

    # ------------------------------------------------------------------
    # Static extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _peaks_from_samples(
        peaks: list[PeakAnnotation],
        samples: dict,
        trace_chromatogram_ids: np.ndarray,
        quantiles: tuple[float, float, float],
        n_samples: int | None,
    ) -> list:
        """Convert posterior *samples* into Peak objects for the given *peaks*.

        Returns a list of :class:`~chromhandler.model.Peak` objects with
        ``Estimate`` area and location.
        """
        from chromhandler.model import Estimate, Peak  # local import — avoids circular

        area_l = np.asarray(samples["area_l"])  # [n_sample, n_trace, n_peak]
        area_r = np.asarray(samples["area_r"])
        apex_l = np.asarray(samples["apex_l"])
        apex_r = np.asarray(samples["apex_r"])

        mol_area = np.empty_like(area_l)
        mol_apex = np.empty_like(apex_l)

        for p_idx, peak in enumerate(peaks):
            al = area_l[..., p_idx]
            ar = area_r[..., p_idx]
            mol_area[..., p_idx] = BetterFitter._molecule_area_slice(peak, al, ar)
            if peak.mode == "single":
                mol_apex[..., p_idx] = apex_l[..., p_idx]
            elif peak.mode == "free_doublet":
                total = al + ar
                mol_apex[..., p_idx] = (apex_l[..., p_idx] * al + apex_r[..., p_idx] * ar) / np.where(
                    total > 0, total, 1.0
                )
            else:  # artefact_doublet
                if peak.include_artefact_in_area:
                    # Both components contribute — use area-weighted centroid
                    total = al + ar
                    mol_apex[..., p_idx] = (apex_l[..., p_idx] * al + apex_r[..., p_idx] * ar) / np.where(
                        total > 0, total, 1.0
                    )
                elif peak.artefact_side == "left":
                    mol_apex[..., p_idx] = apex_r[..., p_idx]
                else:
                    mol_apex[..., p_idx] = apex_l[..., p_idx]

        q_low, _, q_high = quantiles
        peaks_out: list = []

        for t, chrom_id in enumerate(trace_chromatogram_ids):
            for p_idx, ann in enumerate(peaks):
                a_samp = mol_area[:, t, p_idx]
                x_samp = mol_apex[:, t, p_idx]

                if n_samples is not None:
                    n_draw = min(n_samples, len(a_samp))
                    idx = np.random.choice(len(a_samp), size=n_draw, replace=False)
                    a_stored = a_samp[idx].tolist()
                    x_stored = x_samp[idx].tolist()
                else:
                    a_stored = []
                    x_stored = []

                area_est = Estimate(
                    mean=float(np.mean(a_samp)),
                    median=float(np.median(a_samp)),
                    std=float(np.std(a_samp)),
                    q05=float(np.quantile(a_samp, q_low)),
                    q95=float(np.quantile(a_samp, q_high)),
                    samples=a_stored,
                )
                loc_est = Estimate(
                    mean=float(np.mean(x_samp)),
                    median=float(np.median(x_samp)),
                    std=float(np.std(x_samp)),
                    q05=float(np.quantile(x_samp, q_low)),
                    q95=float(np.quantile(x_samp, q_high)),
                    samples=x_stored,
                )
                peaks_out.append(
                    Peak(
                        chromatogram_id=str(chrom_id),
                        location=loc_est,
                        area=area_est,
                        molecule_id=ann.molecule_id,
                    )
                )

        return peaks_out

    @staticmethod
    def _records_from_samples(
        peaks: list[PeakAnnotation],
        samples: dict,
        trace_chromatogram_ids: np.ndarray,
        subset_name: str,
        quantiles: tuple[float, float, float],
    ) -> list[AreaRecord]:
        """Flatten posterior *samples* into :class:`~chromhandler.fitting.subsets.AreaRecord` list."""
        area_l = np.asarray(samples["area_l"])
        area_r = np.asarray(samples["area_r"])

        mol_area = np.empty_like(area_l)
        for p_idx, peak in enumerate(peaks):
            mol_area[..., p_idx] = BetterFitter._molecule_area_slice(
                peak, area_l[..., p_idx], area_r[..., p_idx]
            )

        q_data = np.moveaxis(np.quantile(mol_area, quantiles, axis=0), 0, -1)  # [n_trace, n_peak, 3]

        records: list[AreaRecord] = []
        for t, chrom_id in enumerate(trace_chromatogram_ids):
            for p, peak in enumerate(peaks):
                records.append(
                    AreaRecord(
                        chromatogram_id=str(chrom_id),
                        molecule_id=peak.molecule_id,
                        subset_name=subset_name,
                        area_q05=float(q_data[t, p, 0]),
                        area_median=float(q_data[t, p, 1]),
                        area_q95=float(q_data[t, p, 2]),
                    )
                )
        return records

    # ------------------------------------------------------------------
    # Public posterior extraction
    # ------------------------------------------------------------------

    def to_peaks(
        self,
        *,
        quantiles: tuple[float, float, float] = (0.05, 0.5, 0.95),
        n_samples: int | None = None,
    ) -> list[Peak]:
        """Convert posterior samples into Peak objects with Estimate area/location.

        One Peak is produced per (trace, annotation-peak) pair, aggregated
        across all fitted subsets.

        Args:
            quantiles: Three quantile levels ``(q_low, q_median, q_high)``.
            n_samples: If not ``None``, embed this many posterior samples in
                ``Estimate.samples`` for downstream visualisation.

        Returns:
            List of :class:`~chromhandler.model.Peak` objects sorted by
            ``chromatogram_id`` then ``molecule_id``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called or if subsets were
                registered but not yet fitted.
        """
        if not self._samples_dict:
            if self._subsets:
                raise RuntimeError("to_peaks() requires fitted subset posteriors. Call fit() first.")
            raise RuntimeError("to_peaks() requires a fitted posterior. Call fit() first.")

        all_peaks: list = []
        seen_keys: set[tuple[str, str | None]] = set()

        for name, subset in self._subsets.items():
            if name not in self._samples_dict:
                # This subset was registered but not yet fitted (selective fit)
                continue
            child_peaks = self._peaks_from_samples(
                subset.peaks,
                self._samples_dict[name],
                self._subset_trace_ids[name],
                quantiles,
                n_samples,
            )
            for peak in child_peaks:
                key = (str(peak.chromatogram_id), peak.molecule_id)
                if key in seen_keys:
                    raise ValueError(
                        "to_peaks() found duplicate fitted peaks across subsets for "
                        f"chromatogram_id='{key[0]}' and molecule_id='{key[1]}'."
                    )
                seen_keys.add(key)
                all_peaks.append(peak)

        return sorted(
            all_peaks,
            key=lambda peak: (str(peak.chromatogram_id), str(peak.molecule_id)),
        )

    def area_records(
        self,
        *,
        quantiles: tuple[float, float, float] = (0.05, 0.5, 0.95),
    ) -> list[AreaRecord]:
        """Flatten posterior areas into :class:`~chromhandler.fitting.subsets.AreaRecord` list.

        Records from all fitted subsets are aggregated and sorted.

        Args:
            quantiles: Three quantile levels ``(q_low, q_median, q_high)``.

        Returns:
            List of :class:`~chromhandler.fitting.subsets.AreaRecord` sorted by
            ``chromatogram_id`` then ``molecule_id``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if not self._samples_dict:
            raise RuntimeError("area_records() requires a fitted posterior. Call fit() first.")

        records: list[AreaRecord] = []
        for name, subset in self._subsets.items():
            if name not in self._samples_dict:
                continue
            records.extend(
                self._records_from_samples(
                    subset.peaks,
                    self._samples_dict[name],
                    self._subset_trace_ids[name],
                    name,
                    quantiles,
                )
            )
        return sorted(records, key=lambda r: (r.chromatogram_id, r.molecule_id))


# ---------------------------------------------------------------------------
# Entry point — mirrors the data setup in nu_bayes.py __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from .better_visualize import (
        plot_posterior_predictive,
        plot_trace,
    )

    print("=" * 80)
    print("BetterFitter — Window-Geometry-Based Bayesian Priors")
    print("=" * 80)
    print()

    def save_figure(fig: object, filename: str, *, dpi: int = 150) -> None:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Saved: {filename}")

    def save_sigma_alpha_plot(fitter_obj: BetterFitter, label: str) -> None:
        fig, _ = fitter_obj.plot_sigma_alpha_prior_diagnostics()
        save_figure(fig, f"better_fitter_sigma_alpha_{label}.png")

    def fit_and_plot_dataset(
        fitter_obj: BetterFitter,
        label: str,
        *,
        num_samples: int = 1000,
        num_warmup: int = 1000,
        num_chains: int = 8,
        seed: int = 42,
    ) -> None:
        print()
        print("=" * 80)
        print(f"Running MCMC Inference: {label} ({fitter_obj.n_traces} traces)")
        print("=" * 80)
        fitter_obj.fit(
            num_samples=num_samples,
            num_warmup=num_warmup,
            num_chains=num_chains,
            seed=seed,
        )

        if not fitter_obj.posteriors:
            print(f"Skipping posterior plots for {label}: posteriors unavailable.")
            return

        # Use the first/only posterior for trace plots
        posterior = next(iter(fitter_obj.posteriors.values()))
        fig = plot_trace(posterior, var_names=list(better_model.TRACE_PARAMETER_NAMES))
        save_figure(fig, f"better_fitter_trace_{label}.png", dpi=100)

        # Resolve subset for posterior predictive plot
        subset_name = next(iter(fitter_obj._subsets))
        subset = fitter_obj._subsets[subset_name]
        mask = fitter_obj._compute_subset_mask(subset)
        fig, _ = plot_posterior_predictive(
            fitter_obj.time[mask],
            fitter_obj.signal[mask],
            subset.peaks,
            posterior,
            x_posterior=None,
            y_posterior=None,
            chromatogram_ids=(
                list(fitter_obj._subset_trace_ids[subset_name])
                if subset_name in fitter_obj._subset_trace_ids
                else None
            ),
        )
        save_figure(fig, f"better_fitter_posterior_{label}.png", dpi=100)

    arr = np.load("/Users/max/code/sahh-kinetics-hplc/chromatograms.npy").reshape(-1, 3000)[:, :1000]
    time = np.load("/Users/max/code/sahh-kinetics-hplc/times.npy").reshape(-1, 3000)[:, :1000]

    g0 = [
        *range(58),
        63,
        64,
        70,
        71,
        77,
        78,
        84,
        85,
        91,
        92,
        98,
        99,
        105,
        106,
        112,
        113,
        119,
        120,
        126,
        127,
        133,
        134,
        140,
        141,
        147,
        148,
    ]

    arr_g0 = arr[g0]
    time_g0 = time[g0]

    baselines = [
        BaselineAnnotation(rt_min=2.5, rt_max=2.55),
        BaselineAnnotation(rt_min=3.5, rt_max=3.52),
    ]
    peaks = [
        PeakAnnotation(molecule_id="ino", rt_min=2.55, rt_max=2.9, mode="single"),
        PeakAnnotation(
            molecule_id="peak2",
            rt_min=2.85,
            rt_max=3.15,
            mode="artefact_doublet",
            artefact_side="right",
        ),
        PeakAnnotation(
            molecule_id="peak3",
            rt_min=3.15,
            rt_max=3.5,
            mode="artefact_doublet",
            artefact_side="left",
        ),
    ]

    fitter = BetterFitter(time_g0, arr_g0, peaks=peaks, baselines=baselines)

    fig, axes = fitter.plot_trace_rows(t_min=2.3, t_max=3.7)
    save_figure(fig, "traces.png", dpi=100)
    fitter.print_priors()

    save_sigma_alpha_plot(fitter, "g0")
    fit_and_plot_dataset(fitter, "g0", seed=42)
