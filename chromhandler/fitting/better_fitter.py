"""Minimal chromatographic fitter using window-geometry-based Bayesian priors.

Replaces the FWHM-based prior pipeline of ``nu_bayes.py`` with the geometry-only
approach from ``priors.py``.  This file contains only what is needed to:

1. Accept time/signal data + peak/baseline annotations.
2. Estimate a linear baseline via ``baseline.py``.
3. Compute window-geometry priors via ``priors.py``.
4. Run MCMC inference via ``better_model.py`` using NUTS sampler.
5. Print a human-readable prior summary and posterior statistics via ArviZ.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS
from rich import print

from . import better_model
from .baseline import BaselinePriors, estimate_baseline
from .data import (
    PEAK_MODE_TO_CODE,
    BaselineAnnotation,
    PeakAnnotation,
    baseline_to_mask,
    peak_is_artefact_mode,
    peak_is_free_mode,
)
from .priors import (
    FwhmShapeDiagnostics,
    GeometricPeakPriors,
    build_geometric_priors,
    geometric_priors_to_arrays,
    summarise_priors,
)
from .priors import (
    compute_fwhm_shape_diagnostics as build_fwhm_shape_diagnostics,
)


class BetterFitter:
    """Minimal fitter: baseline estimation + window-geometry priors.

    Parameters
    ----------
    time:
        Retention-time matrix, shape ``[n_trace, n_time]``.
        Rows may have a slowly drifting time axis (e.g. from different runs);
        a common 1-D axis is derived as the row-wise median.
    signal:
        Signal matrix, shape ``[n_trace, n_time]``.
    peaks:
        Annotated peak windows. ``mode`` controls single vs artefact-doublet
        or free-doublet behaviour.
    baselines:
        Explicit baseline regions used to anchor the linear baseline fit.
        The baseline estimation also uses the edges of each peak window, so
        an empty list is acceptable.
    """

    def __init__(
        self,
        time: np.ndarray,
        signal: np.ndarray,
        *,
        peaks: list[PeakAnnotation],
        baselines: list[BaselineAnnotation],
    ) -> None:
        self.time = np.asarray(time, dtype=float)
        self.signal = np.asarray(signal, dtype=float)
        self.peaks = list(peaks)
        self.baselines = list(baselines)
        self._validate()

        # Alignment attributes (set by .align())
        self.shift_samples: np.ndarray | None = None  # [n_trace] shifts in samples
        self.shift_time: np.ndarray | None = None  # [n_trace] shifts in time units
        self.shift_result: object | None = None  # ShiftAlignmentResult

        # Inference-related attributes (initialized on fit())
        self.mcmc: MCMC | None = None
        self.samples: dict | None = None
        self.posterior: object | None = None  # arviz.InferenceData

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
                f"time and signal must have the same shape; "
                f"got {self.time.shape} vs {self.signal.shape}."
            )
        if not self.peaks:
            raise ValueError("At least one PeakAnnotation is required.")

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

    def baseline_priors(self) -> BaselinePriors:
        """Per-trace OLS linear baseline priors (cached after first call)."""
        if not hasattr(self, "_baseline_priors"):
            self._baseline_priors = estimate_baseline(
                self.time,
                self.signal,
                peaks=self.peaks,
                baselines=self.baselines,
            )
        return self._baseline_priors

    def baseline_signal(self) -> np.ndarray:
        """Reconstructed linear baseline matrix, shape ``[n_trace, n_time]``."""
        bp = self.baseline_priors()
        intercept = np.asarray(bp.intercept, dtype=float)[:, None]  # [n_trace, 1]
        slope = np.asarray(bp.slope, dtype=float)[:, None]  # [n_trace, 1]
        return intercept + slope * self.time  # [n_trace, n_time]

    # ------------------------------------------------------------------
    # Prior computation
    # ------------------------------------------------------------------

    def compute_priors(self) -> list[GeometricPeakPriors]:
        """Compute window-geometry priors for all annotated peaks."""
        x = self.common_time()
        baseline = self.baseline_signal()
        return build_geometric_priors(self.peaks, x, self.signal, baseline)

    def compute_fwhm_shape_diagnostics(self) -> FwhmShapeDiagnostics:
        """Compute per-trace FWHM-derived main-peak shape diagnostics."""
        x = self.common_time()
        baseline = self.baseline_signal()
        return build_fwhm_shape_diagnostics(self.peaks, x, self.signal, baseline)

    def noise_prior(self) -> np.ndarray:
        """Estimate per-trace observation noise from baseline-corrected signal.

        Uses median absolute deviation in baseline regions, or falls back to
        signal std if no baseline regions defined.

        Returns
        -------
        np.ndarray
            Shape ``[n_trace]``, noise level for each trace (positive).
        """
        baseline = self.baseline_signal()
        signal_corrected = self.signal - baseline

        if self.baselines:
            # Estimate noise from signal in baseline regions only

            x_jax = jnp.asarray(self.time, dtype=float)
            baseline_mask = baseline_to_mask(self.baselines, x_jax)
            baseline_mask_np = np.asarray(baseline_mask, dtype=bool)

            # Per-trace noise from baseline regions
            sigma_y = np.array(
                [
                    float(np.median(np.abs(signal_corrected[t][baseline_mask_np[t]])))
                    * 1.4826  # MAD → std conversion
                    for t in range(self.n_traces)
                ]
            )
        else:
            # Fall back to std of signal in peak windows
            sigma_y = np.std(signal_corrected, axis=1)

        # Guard: ensure positive
        return np.maximum(sigma_y, 1.0)

    def create_observation_mask(self) -> np.ndarray:
        """Create boolean mask for timepoints to include in likelihood.

        The likelihood should only evaluate over:
        - All baseline regions (for noise estimation)
        - All peak windows (for peak fitting)

        This prevents sigma_y from being inflated by "dead zones" between
        baselines and peaks. The model works with a common time axis.

        Returns
        -------
        np.ndarray
            Shape [n_time], dtype bool. True where observations should be included.
        """
        x = self.common_time()  # [n_time]
        mask = np.zeros(x.shape[0], dtype=bool)

        # Include all baseline regions
        for baseline_annot in self.baselines:
            lo, hi = float(baseline_annot.low), float(baseline_annot.high)
            baseline_mask = (x >= lo) & (x <= hi)
            mask |= baseline_mask

        # Include all peak windows
        for peak_annot in self.peaks:
            lo, hi = float(peak_annot.low), float(peak_annot.high)
            peak_mask = (x >= lo) & (x <= hi)
            mask |= peak_mask

        return mask

    def slice_to_observed_windows(self) -> tuple[np.ndarray, np.ndarray]:
        """Slice time and signal to include only baseline regions and peak windows.

        Returns rectangular arrays using the aligned per-trace time axis. The
        column mask is shared, but each trace keeps its own shifted x-values.

        Returns
        -------
        tuple of np.ndarray
            (time_masked, signal_masked) where:
            - time_masked: [n_trace, n_masked_time]
            - signal_masked: [n_trace, n_masked_time]
        """
        mask = self.create_observation_mask()
        x_masked = self.time[:, mask]  # [n_trace, n_masked_time]
        signal_masked = self.signal[:, mask]  # [n_trace, n_masked_time]

        return x_masked, signal_masked

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
        regions, then runs multi-start Adam optimisation on the MSE alignment
        loss.  After alignment ``self.time`` is updated in-place; all cached
        quantities (baseline priors) are invalidated automatically.

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
            Number of independent Adam restarts (default 16).  Start 0 uses
            the coarse-correlation estimate; starts 1+ are perturbed copies.
            Use 1 for a faster single-start run.
        sigma_perturb : float
            Std-dev (samples) of perturbation noise for starts 1+ (default 3.0).
        seed : int
            PRNG seed for perturbation noise (default 0).
        verbose : bool
            Print per-trace shift diagnostics after alignment (default True).
        """
        from .shift import align_chromatograms

        # Build 2-D alignment mask [n_trace, n_time] from peak windows + baselines
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

        # Convert sample shifts → time units via per-trace median dt  [n_trace]
        dt_per_trace = np.nanmedian(np.abs(np.diff(self.time, axis=1)), axis=1)
        self.shift_time = self.shift_samples * dt_per_trace

        # Apply in-place — all downstream methods automatically see aligned time
        self.time = self.time + self.shift_time[:, None]

        # Invalidate cached baseline priors
        if hasattr(self, "_baseline_priors"):
            del self._baseline_priors

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
        """Extract mode-specific peak structure arrays from annotations."""
        n_peak = len(self.peaks)
        peak_mode_code = np.zeros(n_peak, dtype=np.int32)
        artefact_side = np.zeros(n_peak, dtype=np.int32)
        artefact_indices: list[int] = []
        free_indices: list[int] = []
        nonfree_indices: list[int] = []

        for i, peak in enumerate(self.peaks):
            peak_mode_code[i] = PEAK_MODE_TO_CODE[peak.mode]
            if not peak_is_free_mode(peak.mode):
                nonfree_indices.append(i)
            if peak_is_artefact_mode(peak.mode):
                artefact_indices.append(i)
                artefact_side[i] = -1 if peak.shoulder == "left" else 1
            elif peak_is_free_mode(peak.mode):
                free_indices.append(i)

        return {
            "peak_mode_code": peak_mode_code,
            "artefact_side": artefact_side,
            "artefact_peak_index": np.array(artefact_indices, dtype=np.int32),
            "free_peak_index": np.array(free_indices, dtype=np.int32),
            "nonfree_peak_index": np.array(nonfree_indices, dtype=np.int32),
        }

    def compute_model_inputs(self) -> dict[str, np.ndarray]:
        """Assemble all model inputs from data, priors, and baseline.

        Combines priors, peak structure, baseline estimates, and noise priors
        into a single dict suitable for ``better_model.model()``.

        Returns
        -------
        dict[str, np.ndarray]
            Keys: all parameters expected by ``model()``, values as numpy arrays.
        """
        # Priors
        priors = self.compute_priors()
        prior_arrays = geometric_priors_to_arrays(priors)

        # Transpose per-trace area priors: [n_peak, n_trace] → [n_trace, n_peak]
        prior_arrays["dominant_area_loc_per_trace"] = self._stabilize_area_prior_matrix(
            prior_arrays["dominant_area_loc_per_trace"].T
        )
        prior_arrays["area_total_loc_per_trace"] = self._stabilize_area_prior_matrix(
            prior_arrays["area_total_loc_per_trace"].T
        )

        # Peak structure
        peak_structure = self.peak_structure()

        # Baseline
        baseline_bp = self.baseline_priors()
        baseline_arrays = {
            "baseline_intercept_loc": np.asarray(baseline_bp.intercept, dtype=float),
            "baseline_intercept_scale": np.asarray(
                baseline_bp.intercept_scale, dtype=float
            ),
            "baseline_slope_loc": np.asarray(baseline_bp.slope, dtype=float),
            "baseline_slope_scale": np.asarray(baseline_bp.slope_scale, dtype=float),
        }

        # Noise
        noise_arrays = {
            "sigma_y_prior_loc": self.noise_prior(),
        }

        # Assemble
        return {
            **prior_arrays,
            **peak_structure,
            **baseline_arrays,
            **noise_arrays,
        }

    def print_priors(self) -> None:
        """Compute and print all prior summaries to stdout."""
        print("[Baseline Priors]")
        self._print_baseline_priors()
        print()
        print("[Noise Prior]")
        self._print_noise_prior()
        print()
        print("[Peak Geometry Priors]")
        priors = self.compute_priors()
        print(summarise_priors(priors))

    def _print_baseline_priors(self) -> None:
        """Print per-trace baseline priors."""
        bp = self.baseline_priors()
        intercept = np.asarray(bp.intercept, dtype=float)
        slope = np.asarray(bp.slope, dtype=float)
        intercept_scale = np.asarray(bp.intercept_scale, dtype=float)
        slope_scale = np.asarray(bp.slope_scale, dtype=float)

        print(
            f"{'Trace':>5}  {'Intercept':>12}  {'Int Scale':>10}  {'Slope':>12}  {'Slope Scale':>12}"
        )
        print("-" * 60)
        for t in range(self.n_traces):
            print(
                f"{t:>5}  {intercept[t]:>12.4e}  {intercept_scale[t]:>10.3e}  "
                f"{slope[t]:>12.5e}  {slope_scale[t]:>12.5e}"
            )

    def _print_noise_prior(self) -> None:
        """Print per-trace noise prior."""
        sigma_y = self.noise_prior()
        print(f"{'Trace':>5}  {'Noise σ_y':>12}")
        print("-" * 20)
        for t in range(self.n_traces):
            print(f"{t:>5}  {sigma_y[t]:>12.3f}")

    def plot_sigma_alpha_prior_diagnostics(
        self,
        *,
        figsize: tuple[float, float] | None = None,
        cmap: str = "viridis",
    ) -> tuple[object, np.ndarray]:
        """Plot per-trace FWHM-derived sigma-vs-alpha scatter for each peak."""
        from .better_visualize import plot_sigma_alpha_scatter

        diagnostics = self.compute_fwhm_shape_diagnostics()
        priors = self.compute_priors()

        fig, axes = plot_sigma_alpha_scatter(
            self.peaks,
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
        )
        return fig, axes

    def plot_prior_traces(
        self,
        *,
        figsize: tuple[float, float] | None = None,
        cmap: str = "viridis",
        show_baseline: bool = True,
        show_apex_anchor_prior: bool = True,
        show_gaussian_prior_peak: bool = True,
        show_peak_bounds: bool = True,
    ) -> tuple[object, np.ndarray]:
        """Plot raw traces with baseline, apex-anchor prior, and Gaussian peak prior."""
        from .better_visualize import plot_prior_traces

        bp = self.baseline_priors()
        peak_priors = self.compute_priors()
        diagnostics = self.compute_fwhm_shape_diagnostics()

        fig, axes = plot_prior_traces(
            self.time,
            self.signal,
            self.peaks,
            np.asarray(bp.intercept, dtype=float),
            np.asarray(bp.slope, dtype=float),
            np.asarray(bp.intercept_scale, dtype=float),
            np.asarray(bp.slope_scale, dtype=float),
            np.asarray([p.mu_loc for p in peak_priors], dtype=float),
            np.asarray([p.mu_scale for p in peak_priors], dtype=float),
            approx_center_trace=diagnostics.approx_center_trace,
            approx_height_trace=diagnostics.approx_height_trace,
            approx_sigma_trace=diagnostics.approx_sigma_trace,
            approx_valid_trace=diagnostics.approx_valid_trace,
            approx_fallback_trace=diagnostics.approx_fallback_trace,
            show_baseline=show_baseline,
            show_apex_anchor_prior=show_apex_anchor_prior,
            show_gaussian_prior_peak=show_gaussian_prior_peak,
            show_peak_bounds=show_peak_bounds,
            figsize=figsize,
            cmap=cmap,
        )
        return fig, axes

    # ------------------------------------------------------------------
    # MCMC Inference
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
        """Run MCMC inference on the Bayesian peak model using NUTS sampler.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw per chain (default 1000).
        num_warmup : int
            Number of warmup (burn-in) iterations (default 500).
        num_chains : int
            Number of independent MCMC chains (default 1).
        seed : int
            Random seed for reproducibility (default 0).
        progress_bar : bool
            Whether to show progress bar during sampling (default True).
        save_summary : str or None
            If provided, save ArviZ summary to this file path. If None, only print to stdout.
        """
        # Assemble model inputs (data + priors)
        model_inputs = self.compute_model_inputs()

        # Use windowed likelihood: restrict to baseline regions + peak windows only
        # This prevents sigma_y inflation from unrelated noisy baseline data
        x_masked, y_masked = self.slice_to_observed_windows()

        # Store masked time axis for use in posterior plots
        self.x_masked = x_masked
        self.y_masked = y_masked

        # Use the aligned per-trace masked time tensor directly in the model
        x_for_model = x_masked

        # Add masked data as JAX arrays
        model_inputs["x"] = jnp.asarray(x_for_model, dtype=jnp.float32)
        model_inputs["y"] = jnp.asarray(y_masked, dtype=jnp.float32)

        # Convert all priors to JAX arrays for consistency
        for key in model_inputs:
            if isinstance(model_inputs[key], np.ndarray):
                value = model_inputs[key]
                if np.issubdtype(value.dtype, np.integer):
                    model_inputs[key] = jnp.asarray(value, dtype=jnp.int32)
                elif np.issubdtype(value.dtype, np.bool_):
                    model_inputs[key] = jnp.asarray(value, dtype=bool)
                else:
                    model_inputs[key] = jnp.asarray(value, dtype=jnp.float32)

        # Filter to only keys that the model expects
        model_param_names = {
            "x",
            "y",
            "peak_mode_code",
            "artefact_side",
            "artefact_peak_index",
            "free_peak_index",
            "nonfree_peak_index",
            "apex_anchor_loc",
            "apex_anchor_scale",
            "sigma_loc",
            "sigma_scale",
            "alpha_loc",
            "alpha_scale",
            "dominant_area_loc_per_trace",
            "area_total_loc_per_trace",
            "artefact_area_loc_shared",
            "baseline_intercept_loc",
            "baseline_intercept_scale",
            "baseline_slope_loc",
            "baseline_slope_scale",
            "sigma_y_prior_loc",
        }
        model_inputs_filtered = {
            k: v for k, v in model_inputs.items() if k in model_param_names
        }

        # Create MCMC sampler with NUTS kernel.
        self.mcmc = MCMC(
            NUTS(better_model.model),
            num_warmup=int(num_warmup),
            num_samples=int(num_samples),
            num_chains=int(num_chains),
            progress_bar=bool(progress_bar),
            chain_method="parallel" if num_chains > 1 else "sequential",
        )

        # Run inference
        print("\n" + "=" * 80)
        print("Running MCMC Inference (NUTS Sampler)")
        print("=" * 80)
        self.mcmc.run(jax.random.PRNGKey(int(seed)), **model_inputs_filtered)

        # Extract samples
        self.samples = self.mcmc.get_samples()

        # Convert to ArviZ format
        import arviz as az

        self.posterior = az.from_numpyro(self.mcmc)

        # Print ArviZ summary (filter to only vars that exist in posterior)
        available_vars = list(self.posterior.posterior.data_vars)
        summary_vars = [
            v for v in better_model.SUMMARY_PARAMETER_NAMES if v in available_vars
        ]
        summary_df = az.summary(self.posterior, var_names=summary_vars)
        print("\n" + "=" * 80)
        print("ArviZ Posterior Summary")
        print("=" * 80)
        print(summary_df.to_string())

        # Optionally save summary to file
        if save_summary is not None:
            with open(save_summary, "w", encoding="utf-8") as f:
                f.write(summary_df.to_string())
            print(f"\n✓ Summary saved to: {save_summary}")


# ---------------------------------------------------------------------------
# Entry point — mirrors the data setup in nu_bayes.py __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    from .better_visualize import (
        plot_posterior_predictive,
        plot_trace,
    )

    print("=" * 80)
    print("BetterFitter — Window-Geometry-Based Bayesian Priors")
    print("=" * 80)
    print()

    arr = jnp.load("/Users/max/code/sahh-kinetics-hplc/chromatograms.npy").reshape(
        -1, 3000
    )[-4:, :1000]
    time = jnp.load("/Users/max/code/sahh-kinetics-hplc/times.npy").reshape(-1, 3000)[
        -4:, :1000
    ]

    baselines = [
        BaselineAnnotation(low=2.5, high=2.52),
        BaselineAnnotation(low=3.5, high=3.6),
    ]
    peaks = [
        PeakAnnotation(name="ino", low=2.52, high=2.9, mode="free_doublet"),
        # PeakAnnotation(
        #     name="peak2",
        #     low=2.85,
        #     high=3.15,
        #     mode="artefact_doublet",
        #     shoulder="right",
        # ),
        PeakAnnotation(
            name="peak3",
            low=3.15,
            high=3.5,
            mode="free_doublet",
            shoulder=None,
        ),
    ]

    fitter = BetterFitter(time, arr, peaks=peaks, baselines=baselines)

    # print("=" * 80)
    # print("Aligning Chromatograms")
    # print("=" * 80)
    # fitter.align(n_starts=16, verbose=True)

    fitter.print_priors()

    fig, axes = fitter.plot_sigma_alpha_prior_diagnostics()
    plt.savefig(
        "better_fitter_sigma_alpha_prior_diagnostics.png", dpi=150, bbox_inches="tight"
    )
    print("✓ Saved: better_fitter_sigma_alpha_prior_diagnostics.png")

    # Visualize priors
    print()
    print("=" * 80)
    print("Generating Prior Visualization")
    print("=" * 80)
    fig, axes = fitter.plot_prior_traces(
        show_baseline=True,
        show_apex_anchor_prior=True,
        show_gaussian_prior_peak=True,
        show_peak_bounds=True,
    )
    plt.savefig("better_fitter_priors.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: better_fitter_priors.png")

    print()
    print("=" * 80)
    print("Running MCMC Inference (small sample for testing)")
    print("=" * 80)
    fitter.fit(num_samples=1000, num_warmup=1000, seed=42, num_chains=8)

    # Plot trace for convergence diagnostics
    print()
    print("=" * 80)
    print("Generating MCMC Trace Plots")
    print("=" * 80)
    if fitter.posterior is not None:
        trace_var_names = [
            name for name in better_model.SUMMARY_PARAMETER_NAMES if name != "sigma_y"
        ]
        fig = plot_trace(
            fitter.posterior,
            var_names=trace_var_names,
        )
        plt.savefig("better_fitter_trace.png", dpi=100, bbox_inches="tight")
        print("✓ Saved: better_fitter_trace.png")

    # Plot posterior predictive (fitted signal + components)
    print()
    print("=" * 80)
    print("Generating Posterior Predictive Plots")
    print("=" * 80)
    if fitter.posterior is not None:
        # If windowed likelihood was used, also pass the masked time/signal
        x_posterior = getattr(fitter, "x_masked", None)
        y_posterior = getattr(fitter, "y_masked", None)
        fig, axes = plot_posterior_predictive(
            fitter.time,
            fitter.signal,
            fitter.peaks,
            fitter.posterior,
            x_posterior=x_posterior,
            y_posterior=y_posterior,
        )
        plt.savefig("better_fitter_posterior.png", dpi=100, bbox_inches="tight")
        print("✓ Saved: better_fitter_posterior.png")
