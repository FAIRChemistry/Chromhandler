"""Minimal chromatographic fitter using window-geometry-based Bayesian priors.

Replaces the FWHM-based prior pipeline of ``nu_bayes.py`` with the geometry-only
approach from ``priors.py``.  This file contains only what is needed to:

1. Accept pre-selected time/signal data + peak/baseline annotations.
2. Estimate a linear baseline via ``baseline.py``.
3. Compute window-geometry priors via ``priors.py``.
4. Run MCMC inference via ``better_model.py`` using NUTS sampler.
5. Print a human-readable prior summary and posterior statistics via ArviZ.

Usage::

    fitter = Fitter.from_handler(handler)
    fitter.add_baseline_annotation(BaselineAnnotation(rt_min=0.5, rt_max=1.0))
    fitter.add_peak_annotation(PeakAnnotation(molecule_id="s0", rt_min=2.8, rt_max=3.2, mode="single"))
    fitter.fit()
    fitter.posterior  # InferenceData
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arviz import InferenceData
    from matplotlib.figure import Figure as MplFigure
    from numpy.typing import NDArray

    from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
    from chromhandler.model import Peak

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS
from rich import print

from . import better_model
from .baseline import BaselinePriors, estimate_baseline
from .data import (
    PEAK_MODE_TO_CODE,
    ModelHyperparams,
    baseline_to_mask,
    pad_traces,
    peak_is_artefact_mode,
    peak_is_free_mode,
)
from .priors import (
    GeometricPeakPriors,
    build_peak_priors,
    geometric_priors_to_arrays,
    refine_apex_priors_with_trace_shift,
)


@dataclasses.dataclass(frozen=True)
class AreaRecord:
    """Posterior area summary for one (chromatogram, molecule) pair.

    Attributes:
        chromatogram_id: Identifier of the source chromatogram.
        molecule_id: Identifier of the molecule.
        subset_name: Label of the fitting subset that produced this record.
        area_median: Posterior median of the molecule-relevant peak area.
        area_q05: 5th-percentile credible interval bound.
        area_q95: 95th-percentile credible interval bound.
    """

    chromatogram_id: str
    molecule_id: str
    subset_name: str
    area_median: float
    area_q05: float
    area_q95: float


@dataclasses.dataclass(frozen=True)
class PosteriorCurves:
    """Precomputed posterior HDI curves for a fitted subset.

    All arrays are plain numpy.  Shapes:

    - ``x``: ``[n_x]`` — evaluation axis (minutes)
    - ``total_*``, ``baseline_*``: ``[n_trace, n_x]``
    - ``comp_l_*``, ``comp_r_*``: ``[n_trace, n_peak, n_x]``
    - ``trace_indices``: ``[n_trace]`` — global indices into the parent fitter's
      full trace array (result of :meth:`~Fitter.select_trace_indices`)
    - ``chromatogram_ids``: per-trace labels (or ``None`` if unavailable)
    """

    x: NDArray[np.float64]  # [n_x]
    total_median: NDArray[np.float64]  # [n_trace, n_x]
    total_lower: NDArray[np.float64]  # [n_trace, n_x]
    total_upper: NDArray[np.float64]  # [n_trace, n_x]
    baseline_median: NDArray[np.float64]  # [n_trace, n_x]
    baseline_lower: NDArray[np.float64]  # [n_trace, n_x]
    baseline_upper: NDArray[np.float64]  # [n_trace, n_x]
    comp_l_median: NDArray[np.float64]  # [n_trace, n_peak, n_x]
    comp_l_lower: NDArray[np.float64]  # [n_trace, n_peak, n_x]
    comp_l_upper: NDArray[np.float64]  # [n_trace, n_peak, n_x]
    comp_r_median: NDArray[np.float64]  # [n_trace, n_peak, n_x]
    comp_r_lower: NDArray[np.float64]  # [n_trace, n_peak, n_x]
    comp_r_upper: NDArray[np.float64]  # [n_trace, n_peak, n_x]
    trace_indices: NDArray[np.intp]  # [n_trace]
    chromatogram_ids: list[str] | None


class Fitter:
    """Chromatographic fitter for a pre-selected set of traces.

    Parameters
    ----------
    time:
        Retention-time matrix, shape ``[n_trace, n_time]``.
        Rows may have a slowly drifting time axis (e.g. from different runs);
        a common 1-D axis is derived as the row-wise median.
    signal:
        Signal matrix, shape ``[n_trace, n_time]``.
    peaks:
        Optional peak-window annotations.  Prefer :meth:`add_peak_annotation`.
    baselines:
        Optional baseline region annotations.  An empty list is acceptable;
        the baseline estimation also uses the edges of each peak window.
    """

    def __init__(
        self,
        time: NDArray[np.float64],
        signal: NDArray[np.float64],
        *,
        peaks: list[PeakAnnotation] | None = None,
        baselines: list[BaselineAnnotation] | None = None,
        trace_sample_ids: list[str] | None = None,
        trace_chromatogram_ids: list[str] | None = None,
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

        self.hyperparams: ModelHyperparams = hyperparams if hyperparams is not None else ModelHyperparams()

        self.shift_samples: NDArray[np.float64] | None = None  # [n_trace] shifts in samples
        self.shift_time: NDArray[np.float64] | None = None  # [n_trace] shifts in time units

        # Inference attributes (set by _run_mcmc())
        self.mcmc: MCMC | None = None
        self.samples: dict[str, Any] | None = None
        self._posterior: InferenceData | None = None

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

    def common_time(self) -> NDArray[np.float64]:
        """Row-wise median time axis → 1-D ``[n_time]``.

        When all traces share an identical time grid this is exact.  For small
        per-trace drift it is a robust representative axis.
        """
        return np.nanmedian(self.time, axis=0)

    # ------------------------------------------------------------------
    # Prior computation (subset-aware)
    # ------------------------------------------------------------------

    def _compute_position_priors(
        self,
    ) -> tuple[list[GeometricPeakPriors], float]:
        """Compute apex priors plus the shared trace-shift scale."""
        x = self.common_time()
        baseline = self.baseline_signal()  # cached; compute_model_inputs hits the same cache

        priors, apex_traces = build_peak_priors(self.peaks, x, self.signal, baseline)

        x_finite = x[np.isfinite(x)]
        apex_scale_floor = (
            max(float(np.nanmedian(np.abs(np.diff(np.sort(x_finite))))), 1e-6)
            if x_finite.size >= 2
            else 1e-6
        )
        return refine_apex_priors_with_trace_shift(
            priors,
            apex_traces,
            apex_scale_floor=apex_scale_floor,
            trace_shift_scale_floor=1e-6,
        )

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
            )
        return self._bp_direct

    def baseline_signal(self) -> NDArray[np.float64]:
        """Reconstructed linear baseline matrix, shape ``[n_trace, n_time]``."""
        bp = self.baseline_priors()
        intercept = np.asarray(bp.intercept, dtype=float)[:, None]
        slope = np.asarray(bp.slope, dtype=float)[:, None]
        return intercept + slope * self.time

    def compute_priors(self) -> list[GeometricPeakPriors]:
        """Compute window-geometry priors."""
        priors, _ = self._compute_position_priors()
        return priors

    def noise_prior(self) -> NDArray[np.float64]:
        """Estimate per-trace observation noise from baseline-corrected signal.

        Uses median absolute deviation in baseline regions, or falls back to
        signal std if no baseline regions defined.

        Returns
        -------
        np.ndarray
            Shape ``[n_trace]``, noise level for each trace (positive).
        """
        bp = self.baseline_priors()
        intercept = np.asarray(bp.intercept, dtype=float)[:, None]
        slope = np.asarray(bp.slope, dtype=float)[:, None]
        baseline = intercept + slope * self.time
        signal_corrected: NDArray[np.float64] = np.asarray(self.signal - baseline, dtype=float)

        if self.baselines:
            x_jax = jnp.asarray(self.time, dtype=float)
            baseline_mask = baseline_to_mask(self.baselines, x_jax)
            baseline_mask_np = np.asarray(baseline_mask, dtype=bool)
            sigma_y = np.array(
                [
                    float(
                        np.median(
                            np.abs(
                                np.asarray(signal_corrected[t])[np.asarray(baseline_mask_np[t], dtype=bool)]
                            )
                        )
                    )
                    * 1.4826
                    for t in range(self.n_traces)
                ]
            )
        else:
            sigma_y = np.std(signal_corrected, axis=1)

        signal_range = np.ptp(self.signal, axis=1)
        noise_floor = 1e-3 * np.maximum(signal_range, 1e-6)
        return np.maximum(sigma_y, noise_floor)

    def create_observation_mask(self) -> NDArray[np.bool_]:
        """Create boolean mask for timepoints to include in likelihood.

        Covers all registered baseline regions and peak windows.

        Returns
        -------
        np.ndarray
            Shape ``[n_time]``, dtype bool.
        """
        x = self.common_time()
        mask = np.zeros(int(x.shape[0]), dtype=bool)

        for bl in self.baselines:
            lo, hi = float(bl.rt_min), float(bl.rt_max)
            mask |= (x >= lo) & (x <= hi)

        for pk in self.peaks:
            lo, hi = float(pk.rt_min), float(pk.rt_max)
            mask |= (x >= lo) & (x <= hi)

        return mask

    def slice_to_observed_windows(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
    ) -> Fitter:
        """Construct a :class:`Fitter` from a :class:`~chromhandler.handler.Handler`.

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
            A fully initialised :class:`Fitter` with no annotations.

        Example::

            fitter = Fitter.from_handler(handler)
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
        trace_sample_ids: list[str] = [s.id for s in samples for _c in s.chromatograms]
        trace_chrom_ids: list[str] = [c.id for s in samples for c in s.chromatograms]

        time_arr, signal_arr = pad_traces(time_lists, signal_lists)

        return cls(
            time_arr,
            signal_arr,
            peaks=None,
            baselines=None,
            trace_sample_ids=trace_sample_ids,
            trace_chromatogram_ids=trace_chrom_ids,
        )

    # ------------------------------------------------------------------
    # Annotation management
    # ------------------------------------------------------------------

    def add_peak_annotation(self, ann: PeakAnnotation) -> None:
        """Register a peak-window annotation.

        Args:
            ann: The :class:`~chromhandler.annotations.PeakAnnotation` to add.

        Example::

            fitter.add_peak_annotation(
                PeakAnnotation(molecule_id="s0", rt_min=2.8, rt_max=3.2, mode="single")
            )
        """
        self.peaks.append(ann)
        # Invalidate cached baseline priors (peak windows affect mask)
        if "_bp_direct" in self.__dict__:
            del self._bp_direct

    def add_baseline_annotation(self, ann: BaselineAnnotation) -> None:
        """Register a baseline-region annotation.

        Args:
            ann: The :class:`~chromhandler.annotations.BaselineAnnotation` to add.

        Example::

            fitter.add_baseline_annotation(BaselineAnnotation(rt_min=0.5, rt_max=1.0))
        """
        self.baselines.append(ann)
        if "_bp_direct" in self.__dict__:
            del self._bp_direct

    def peak_structure(self) -> dict[str, NDArray[Any]]:
        """Extract mode-specific peak structure arrays from ``self.peaks``.

        Returns local indices splitting free-doublet peaks by separation
        variance:
        """
        n_peak = len(self.peaks)
        peak_mode_code = np.zeros(n_peak, dtype=np.int32)
        artefact_side = np.zeros(n_peak, dtype=np.int32)
        artefact_indices: list[int] = []
        free_indices: list[int] = []

        for i, peak in enumerate(self.peaks):
            peak_mode_code[i] = PEAK_MODE_TO_CODE[peak.mode]
            if peak_is_artefact_mode(peak.mode):
                artefact_indices.append(i)
                artefact_side[i] = -1 if peak.artefact_side == "left" else 1
            elif peak_is_free_mode(peak.mode):
                free_indices.append(i)

        nonfree_mask = peak_mode_code != PEAK_MODE_TO_CODE["free_doublet"]
        nonfree_idx = np.where(nonfree_mask)[0].astype(np.int32)
        nonfree_position = (np.cumsum(nonfree_mask.astype(np.int32)) - 1).astype(np.int32)

        return {
            "peak_mode_code": peak_mode_code,
            "artefact_side": artefact_side,
            "artefact_peak_index": np.array(artefact_indices, dtype=np.int32),
            "free_peak_index": np.array(free_indices, dtype=np.int32),
            "nonfree_idx": nonfree_idx,
            "nonfree_position": nonfree_position,
        }

    def compute_model_inputs(self) -> dict[str, NDArray[Any]]:
        """Assemble all model inputs from data, priors, and baseline.

        Intended to be called on views (no subsets) by :meth:`_run_mcmc`.

        Returns
        -------
        dict[str, np.ndarray]
            Keys: all parameters expected by ``better_model.model()``.
        """
        priors, trace_shift_scale = self._compute_position_priors()
        prior_arrays = geometric_priors_to_arrays(priors)
        prior_arrays["trace_shift_scale"] = np.asarray(trace_shift_scale, dtype=np.float64)

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

    # ------------------------------------------------------------------
    # Trace selection and posterior evaluation helpers
    # ------------------------------------------------------------------

    def select_trace_indices(
        self,
        *,
        sample_ids: list[str] | None = None,
        chromatogram_ids: list[str] | None = None,
    ) -> NDArray[np.intp]:
        """Return integer indices of traces matching all supplied filters.

        Filters are applied as an intersection (AND).  When no filter is given
        every trace index is returned.

        Args:
            sample_ids: Keep only traces whose ``trace_sample_ids`` value is
                in this list.  Requires the fitter to have been built with
                :meth:`from_handler`.
            chromatogram_ids: Keep only traces whose
                ``trace_chromatogram_ids`` value is in this list.

        Returns:
            ``np.ndarray`` of shape ``[n_selected]``, dtype ``int``.
        """
        indices = np.arange(self.n_traces)

        if sample_ids is not None:
            if self.trace_sample_ids is None:
                raise RuntimeError(
                    "sample_ids filter requires trace_sample_ids. "
                    "Build the fitter with Fitter.from_handler()."
                )
            keep = np.where(np.isin(self.trace_sample_ids, sample_ids))[0]
            indices = np.intersect1d(indices, keep)

        if chromatogram_ids is not None:
            if self.trace_chromatogram_ids is None:
                raise RuntimeError(
                    "chromatogram_ids filter requires trace_chromatogram_ids. "
                    "Build the fitter with Fitter.from_handler()."
                )
            keep = np.where(np.isin(self.trace_chromatogram_ids, chromatogram_ids))[0]
            indices = np.intersect1d(indices, keep)

        return indices

    def window_mask(self, rt_min: float, rt_max: float) -> NDArray[np.bool_]:
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
        x: NDArray[np.float64],
        *,
        hdi_prob: float = 0.95,
        trace_indices: NDArray[np.intp] | None = None,
        n_samples_max: int = 2000,
    ) -> PosteriorCurves:
        """Evaluate posterior HDI curves on an arbitrary time grid.

        Re-evaluates the skew-normal components at every point in *x* using
        the stored posterior samples; no interpolation.

        Args:
            x: Evaluation axis, shape ``[n_x]`` (minutes).
            hdi_prob: Credible-interval probability (default 0.95).
            trace_indices: Indices of traces to include.  ``None`` = all traces.
            n_samples_max: Maximum number of posterior draws to use (capped for
                memory).  Default 2000.

        Returns:
            :class:`PosteriorCurves` with pre-computed median / lower / upper
            for total signal, baseline, and per-component curves.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        from scipy.special import ndtr as _ndtr

        if self._posterior is None:
            raise RuntimeError("posterior_curves() requires a fitted posterior. Call fit() first.")

        n_traces = self.n_traces

        # Select requested traces
        if trace_indices is None:
            local_idx = np.arange(n_traces)
            global_idx = np.arange(n_traces)
        else:
            global_idx = np.asarray(trace_indices, dtype=int)
            local_idx = global_idx

        n_selected = len(local_idx)
        if n_selected == 0:
            raise ValueError("posterior_curves(): no traces matched the given trace_indices.")

        # Extract posterior arrays from self.samples (shape [n_total, n_trace, n_peak])
        # where n_total = n_chains x n_draws.  Derived quantities (xi_l, sigma_l, ...)
        # were computed post-sampling by compute_derived_quantities() and merged into
        # self.samples, so they are available here without going through ArviZ.
        smp = self._get_view_samples()

        def _sel(key: str) -> NDArray[np.float64]:
            """Fetch from samples and select the requested trace subset."""
            return np.asarray(smp[key], dtype=float)[:, local_idx]

        xi_l: NDArray[np.float64] = _sel("xi_l")       # [n_total, n_sel, n_peak]
        xi_r: NDArray[np.float64] = _sel("xi_r")
        sigma_l: NDArray[np.float64] = _sel("sigma_l")
        sigma_r: NDArray[np.float64] = _sel("sigma_r")
        alpha_l: NDArray[np.float64] = _sel("alpha_l")
        alpha_r: NDArray[np.float64] = _sel("alpha_r")
        area_l: NDArray[np.float64] = _sel("area_l")
        area_r: NDArray[np.float64] = _sel("area_r")
        bl_int: NDArray[np.float64] = np.asarray(smp["baseline_intercept"], dtype=float)[:, local_idx]
        bl_slp: NDArray[np.float64] = np.asarray(smp["baseline_slope"], dtype=float)[:, local_idx]

        n_total = xi_l.shape[0]

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

        def _merge(arr: NDArray[np.float64]) -> NDArray[np.float64]:
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
            x_2d: NDArray[np.float64],  # [n, n_x]
            xi: NDArray[np.float64],  # [n, n_comp]
            sigma: NDArray[np.float64],  # [n, n_comp]
            alpha: NDArray[np.float64],  # [n, n_comp]
        ) -> NDArray[np.float64]:  # [n, n_comp, n_x]
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

        def _pct(
            arr: NDArray[np.float64],
        ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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

    # ------------------------------------------------------------------
    # Posteriors property
    # ------------------------------------------------------------------

    @property
    def posterior(self) -> InferenceData | None:
        """Fitted posterior.

        Returns ``None`` before :meth:`fit` is called, and an ArviZ
        ``InferenceData`` object after fitting.
        """
        return self._posterior

    # ------------------------------------------------------------------
    # Diagnostic outputs
    # ------------------------------------------------------------------

    def save_summary(self, path: str | Path) -> None:
        """Save ArviZ posterior summary to a text file.

        Parameters
        ----------
        path : str or Path
            Destination file path. Created or overwritten.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if self._posterior is None:
            raise RuntimeError(
                "save_summary() requires a fitted posterior. Call fit() first."
            )

        import arviz as az

        available_vars = list(self._posterior.posterior.data_vars)  # type: ignore[union-attr]
        summary_vars = [
            v for v in better_model.SUMMARY_PARAMETER_NAMES if v in available_vars
        ]
        summary_df = az.summary(self._posterior, var_names=summary_vars or None)
        Path(path).write_text(summary_df.to_string(), encoding="utf-8")

    def plot_traces(
        self,
        path: str | Path | None = None,
        var_names: list[str] | None = None,
    ) -> MplFigure:
        """MCMC trace plot for chain-mixing / Rhat convergence diagnostics.

        Parameters
        ----------
        path : str, Path, or None
            If given, saves the figure to this path and closes it.
        var_names : list[str] or None
            Parameters to plot. Defaults to ``TRACE_PARAMETER_NAMES`` from
            ``better_model``, filtered to variables present in the posterior.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if self._posterior is None:
            raise RuntimeError(
                "plot_traces() requires a fitted posterior. Call fit() first."
            )

        import matplotlib.pyplot as _plt

        from . import better_visualize

        fig = better_visualize.plot_trace(self._posterior, var_names=var_names)
        if path is not None:
            fig.savefig(str(path), dpi=150, bbox_inches="tight")
            _plt.close(fig)
        return fig

    def plot_fit(
        self,
        path: str | Path | None = None,
        *,
        trace_indices: NDArray[np.intp] | None = None,
    ) -> tuple[MplFigure, np.ndarray[Any, Any]]:
        """Plot raw data with posterior fit curves (or scatter-only before fit).

        Rows = traces, columns = one per peak window + optional combined column.

        Parameters
        ----------
        path : str, Path, or None
            If given, saves the figure to this path and closes it.
        trace_indices : array of int or None
            Which traces to build posterior curves for. ``None`` = all traces.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : np.ndarray  shape [n_traces, n_col]
        """
        import matplotlib.pyplot as _plt

        from . import better_visualize

        curves = None
        fitted_rows: np.ndarray[Any, Any] | None = None

        if self._posterior is not None:
            curves = self.posterior_curves(
                self.common_time(),
                trace_indices=trace_indices,
            )
            fitted_rows = (
                trace_indices
                if trace_indices is not None
                else np.arange(self.n_traces, dtype=np.intp)
            )

        chromatogram_ids: list[str] | None = (
            list(self.trace_chromatogram_ids.astype(str))
            if self.trace_chromatogram_ids is not None
            else None
        )

        fig, axes = better_visualize.plot_fit(
            self.time,
            self.signal,
            self.peaks,
            curves,
            fitted_rows=fitted_rows,
            baselines=self.baselines,
            chromatogram_ids=chromatogram_ids,
        )
        if path is not None:
            fig.savefig(str(path), dpi=150, bbox_inches="tight")
            _plt.close(fig)
        return fig, axes

    # ------------------------------------------------------------------
    # Inference (MCMC)
    # ------------------------------------------------------------------

    def fit(
        self,
        num_samples: int = 1000,
        num_warmup: int = 500,
        num_chains: int = 1,
        seed: int = 0,
        progress_bar: bool = True,
        save_summary: str | None = None,
    ) -> None:
        """Run MCMC inference.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw per chain (default 1000).
        num_warmup : int
            Number of warmup (burn-in) iterations (default 500).
        num_chains : int
            Number of independent chains (default 1).
        seed : int
            Random seed for reproducibility (default 0).
        progress_bar : bool
            Whether to show a progress bar during inference (default True).
        save_summary : str or None
            If provided, save the ArviZ posterior summary to this file path.
        """
        if not self.peaks:
            raise RuntimeError(
                "fit() requires at least one peak annotation. Call add_peak_annotation() first."
            )

        self._run_mcmc(
            num_samples=num_samples,
            num_warmup=num_warmup,
            num_chains=num_chains,
            seed=seed,
            progress_bar=progress_bar,
            save_summary=save_summary,
        )

    def _prepare_model_inputs(self) -> dict[str, Any]:
        """Compute priors + structure + baseline + noise → JAX-ready arrays.

        Converts all numpy arrays to JAX with appropriate dtypes and applies
        safety clamps to scale parameters.  Uses ``inspect.signature`` to
        filter to only the parameters accepted by ``better_model.model()``.
        """
        import inspect

        model_inputs: dict[str, Any] = dict(self.compute_model_inputs())

        x_masked, y_masked = self.slice_to_observed_windows()
        self.x_masked = x_masked
        self.y_masked = y_masked

        model_inputs["x"] = jnp.asarray(x_masked, dtype=jnp.float32)
        model_inputs["y"] = jnp.asarray(y_masked, dtype=jnp.float32)

        # Safety clamps for scale/loc parameters that must be strictly positive.
        _float_clamps: dict[str, float] = {
            "trace_shift_scale": 1e-6,
            "w_left_loc": 1e-6,
            "w_left_scale": 1e-6,
            "w_right_loc": 1e-6,
            "w_right_scale": 1e-6,
            "area_gaussian_pt": 1e-8,
            "area_trapz_pt": 1e-8,
            "area_art_shared": 1e-8,
            "baseline_intercept_scale": 1e-6,
            "baseline_slope_scale": 1e-6,
            "sigma_y_prior_loc": 1e-6,
        }
        for key in model_inputs:
            if isinstance(model_inputs[key], np.ndarray):
                value = model_inputs[key]
                if np.issubdtype(value.dtype, np.integer):
                    model_inputs[key] = jnp.asarray(value, dtype=jnp.int32)
                elif np.issubdtype(value.dtype, np.bool_):
                    model_inputs[key] = jnp.asarray(value, dtype=bool)
                else:
                    arr = jnp.asarray(value, dtype=jnp.float32)
                    if key in _float_clamps:
                        arr = jnp.maximum(arr, _float_clamps[key])
                    model_inputs[key] = arr

        # Filter to only the parameters accepted by model().
        sig = inspect.signature(better_model.model)
        model_param_names = set(sig.parameters.keys()) - {"hyperparams"}
        return {k: v for k, v in model_inputs.items() if k in model_param_names}

    def _run_nuts(
        self,
        model_inputs: dict[str, Any],
        *,
        num_samples: int = 1000,
        num_warmup: int = 500,
        num_chains: int = 1,
        seed: int = 0,
        progress_bar: bool = True,
    ) -> MCMC:
        """Execute NUTS sampler. Returns the MCMC object."""
        import functools

        model_fn = functools.partial(better_model.model, hyperparams=self.hyperparams)

        mcmc = MCMC(
            NUTS(model_fn),
            num_warmup=int(num_warmup),
            num_samples=int(num_samples),
            num_chains=int(num_chains),
            progress_bar=bool(progress_bar),
            chain_method="parallel" if num_chains > 1 else "sequential",
        )
        mcmc.run(jax.random.PRNGKey(int(seed)), **model_inputs)
        return mcmc

    def _process_posterior(
        self,
        mcmc: MCMC,
        model_inputs: dict[str, Any],
        num_samples: int,
        *,
        save_summary: str | None = None,
    ) -> None:
        """Reconstruct derived quantities, build ArviZ InferenceData, print summary."""
        import arviz as az
        import xarray as xr

        self.mcmc = mcmc
        raw_samples: dict[str, Any] = mcmc.get_samples()

        derived = better_model.compute_derived_quantities(
            raw_samples, model_inputs, self.hyperparams  # type: ignore[arg-type]
        )
        self.samples = {**raw_samples, **derived}

        self._posterior = az.from_numpyro(mcmc)

        # Merge derived quantities into ArviZ InferenceData.
        n_chains = int(mcmc.num_chains)
        n_draws = int(num_samples)
        new_xr_vars: dict[str, xr.DataArray] = {}
        for key, arr in derived.items():
            shaped = np.asarray(arr).reshape(n_chains, n_draws, *np.asarray(arr).shape[1:])
            dims = ["chain", "draw"] + [f"{key}_dim_{i}" for i in range(shaped.ndim - 2)]
            new_xr_vars[key] = xr.DataArray(shaped, dims=dims)
        self._posterior.posterior = self._posterior.posterior.assign(new_xr_vars)  # type: ignore[union-attr]

        available_vars = list(self.posterior.posterior.data_vars)  # type: ignore[union-attr]
        summary_vars = [v for v in better_model.SUMMARY_PARAMETER_NAMES if v in available_vars]
        summary_df = az.summary(self.posterior, var_names=summary_vars)
        print("\n" + "=" * 80)
        print("ArviZ Posterior Summary")
        print("=" * 80)
        print(summary_df.to_string())

        if save_summary is not None:
            self.save_summary(save_summary)
            print(f"\n✓ Summary saved to: {save_summary}")

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

        Delegates to :meth:`_prepare_model_inputs`, :meth:`_run_nuts`, and
        :meth:`_process_posterior`.
        """
        model_inputs = self._prepare_model_inputs()
        mcmc = self._run_nuts(
            model_inputs,
            num_samples=num_samples,
            num_warmup=num_warmup,
            num_chains=num_chains,
            seed=seed,
            progress_bar=progress_bar,
        )
        self._process_posterior(mcmc, model_inputs, num_samples, save_summary=save_summary)

    # ------------------------------------------------------------------
    # Posterior area extraction (view-level helpers)
    # ------------------------------------------------------------------

    @staticmethod
    def _molecule_area_slice(
        peak: PeakAnnotation,
        area_l: NDArray[np.float64],
        area_r: NDArray[np.float64],
    ) -> NDArray[np.float64]:
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
    def posterior_area_matrix(self) -> NDArray[np.float64]:
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
    ) -> NDArray[np.float64]:
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

    def _get_view_samples(self) -> dict[str, Any]:
        if self.samples is None:
            raise RuntimeError(
                "posterior_area_matrix / molecule_areas() require a fitted posterior. Call fit() first."
            )
        return self.samples

    def _get_view_peaks(self) -> list[PeakAnnotation]:
        return self.peaks

    # ------------------------------------------------------------------
    # Static extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _peaks_from_samples(
        peaks: list[PeakAnnotation],
        samples: dict[str, Any],
        trace_chromatogram_ids: NDArray[Any],
        quantiles: tuple[float, float, float],
        n_samples: int | None,
    ) -> list[Any]:
        """Convert posterior *samples* into Peak objects for the given *peaks*.

        Returns a list of :class:`~chromhandler.model.Peak` objects with
        ``Estimate`` area and location.
        """
        from chromhandler.model import Estimate, Peak  # local import — avoids circular

        # [n_sample, n_trace, n_peak]
        area_l: NDArray[np.float64] = np.asarray(samples["area_l"], dtype=float)
        area_r: NDArray[np.float64] = np.asarray(samples["area_r"], dtype=float)
        apex_l: NDArray[np.float64] = np.asarray(samples["apex_l"], dtype=float)
        apex_r: NDArray[np.float64] = np.asarray(samples["apex_r"], dtype=float)

        mol_area = np.empty_like(area_l)
        mol_apex = np.empty_like(apex_l)

        for p_idx, peak in enumerate(peaks):
            al = area_l[..., p_idx]
            ar = area_r[..., p_idx]
            mol_area[..., p_idx] = Fitter._molecule_area_slice(peak, al, ar)
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
        peaks_out: list[Any] = []

        for t, chrom_id in enumerate(trace_chromatogram_ids):
            for p_idx, ann in enumerate(peaks):
                a_samp = mol_area[:, t, p_idx]
                x_samp = mol_apex[:, t, p_idx]

                a_stored: list[float]
                x_stored: list[float]
                if n_samples is not None:
                    n_draw = min(n_samples, len(a_samp))
                    idx = np.random.choice(len(a_samp), size=n_draw, replace=False)
                    a_stored = [float(v) for v in a_samp[idx]]
                    x_stored = [float(v) for v in x_samp[idx]]
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
        samples: dict[str, Any],
        trace_chromatogram_ids: NDArray[Any],
        subset_name: str,
        quantiles: tuple[float, float, float],
    ) -> list[AreaRecord]:
        """Flatten posterior *samples* into :class:`~chromhandler.fitting.subsets.AreaRecord` list."""
        area_l: NDArray[np.float64] = np.asarray(samples["area_l"], dtype=float)
        area_r: NDArray[np.float64] = np.asarray(samples["area_r"], dtype=float)

        mol_area = np.empty_like(area_l)
        for p_idx, peak in enumerate(peaks):
            mol_area[..., p_idx] = Fitter._molecule_area_slice(
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

        One Peak is produced per (trace, annotation-peak) pair.

        Args:
            quantiles: Three quantile levels ``(q_low, q_median, q_high)``.
            n_samples: If not ``None``, embed this many posterior samples in
                ``Estimate.samples`` for downstream visualisation.

        Returns:
            List of :class:`~chromhandler.model.Peak` objects sorted by
            ``chromatogram_id`` then ``molecule_id``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self.samples is None:
            raise RuntimeError("to_peaks() requires a fitted posterior. Call fit() first.")

        chrom_ids = (
            self.trace_chromatogram_ids
            if self.trace_chromatogram_ids is not None
            else np.arange(self.n_traces, dtype=object)
        )

        all_peaks = self._peaks_from_samples(
            self.peaks,
            self.samples,
            chrom_ids,
            quantiles,
            n_samples,
        )

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

        Args:
            quantiles: Three quantile levels ``(q_low, q_median, q_high)``.

        Returns:
            List of :class:`~chromhandler.fitting.subsets.AreaRecord` sorted by
            ``chromatogram_id`` then ``molecule_id``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self.samples is None:
            raise RuntimeError("area_records() requires a fitted posterior. Call fit() first.")

        chrom_ids = (
            self.trace_chromatogram_ids
            if self.trace_chromatogram_ids is not None
            else np.arange(self.n_traces, dtype=object)
        )

        records = self._records_from_samples(
            self.peaks,
            self.samples,
            chrom_ids,
            "",
            quantiles,
        )
        return sorted(records, key=lambda r: (r.chromatogram_id, r.molecule_id))
