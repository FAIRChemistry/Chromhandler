from functools import partial
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_median
from rich.console import Console
from rich.table import Table
from rich.text import Text

from chromhandler.fitting.moments import (
    compute_peak_moment_metrics_batch,
    metrics_list_to_arrays,
    summarize_metrics,
)
from chromhandler.fitting.peak_models import (
    log_skew_normal_pdf,
    model,
    skew_mixture_area,
)
from chromhandler.fitting.shift import (
    ShiftAlignmentResult,
    align_groupwise_sample_shifts,
)

numpyro.set_host_device_count(8)

num_devices = jax.local_device_count()
console = Console()

print(f"Number of devices: {num_devices}")


def relabel_by_sort(idata: az.InferenceData, key: str = "mu") -> az.InferenceData:
    """
    Sort each draw by ascending μ and permute all component-indexed variables.
    Handles component-indexed arrays with either:
    - the same shape rank as `mu`, or
    - one fewer rank than `mu` (e.g. global/shared [chain, draw, K] vars when
      `mu` is [chain, draw, S, K]).

    Variables that are not shape-compatible are left unchanged.
    """
    post = idata.posterior
    mu = jnp.asarray(post[key].values)
    order = jnp.argsort(mu, axis=-1)
    K = mu.shape[-1]

    comp_vars = [
        v for v in post.data_vars if post[v].shape[-1] == K and post[v].ndim >= 3
    ]

    reordered = {}
    for v in comp_vars:
        arr = jnp.asarray(post[v].values)

        if arr.ndim == order.ndim:
            # Example: arr and order are both [chain, draw, S, K]
            if arr.shape[:-1] != order.shape[:-1]:
                continue
            idx = order
        elif arr.ndim == order.ndim - 1:
            # Example: arr is [chain, draw, K] while order is [chain, draw, S, K]
            if arr.shape[:-1] != order.shape[:-2]:
                continue
            idx = order[..., 0, :]
            if order.shape[-2] > 1:
                is_consistent = bool(jnp.all(order == idx[..., None, :]))
                if not is_consistent:
                    # No unique order across the dropped axis; skip relabeling.
                    continue
        else:
            continue

        new = jnp.take_along_axis(arr, idx, axis=-1)
        reordered[v] = (post[v].dims, jnp.asarray(new))

    post_rl = post.assign(**reordered)

    other_groups = {g: getattr(idata, g) for g in idata.groups() if g != "posterior"}
    return az.InferenceData(posterior=post_rl, **other_groups)


# =====================================================================
# Prediction helper
# =====================================================================


def _predict_single_sample(
    x: jnp.ndarray,
    A_s: jnp.ndarray,
    mu_s: jnp.ndarray,
    sigma_s: jnp.ndarray,
    alpha_s: jnp.ndarray,
) -> jnp.ndarray:
    """Helper to predict mean from single posterior sample."""
    return skew_mixture_area(x, A_s, mu_s, sigma_s, alpha_s)


def predict_mean(
    x: jnp.ndarray,
    samples: dict[str, Any],
) -> jnp.ndarray:
    """Reconstruct mean signal μ_y from posterior samples.

    Args:
        x: Time/retention points, shape [..., N]
        samples: Posterior samples dict

    Returns:
        Mean predictions, shape [num_samples, ..., N]
    """
    x = jnp.asarray(x, dtype=jnp.float32)
    A = jnp.asarray(samples["A"], dtype=jnp.float32)
    mu = jnp.asarray(samples["mu"], dtype=jnp.float32)
    sigma = jnp.asarray(samples["sigma"], dtype=jnp.float32)
    alpha = jnp.asarray(samples["alpha"], dtype=jnp.float32)

    predict_fn = partial(_predict_single_sample, x)
    mu_y = jax.vmap(predict_fn)(A, mu, sigma, alpha)

    # Add linear baseline if present in posterior samples.
    if "b0" in samples and "b1" in samples:
        b0 = jnp.asarray(samples["b0"], dtype=jnp.float32)
        b1 = jnp.asarray(samples["b1"], dtype=jnp.float32)
        if x.ndim == 1:
            b0_flat = b0.reshape(b0.shape[0], -1)[:, 0]
            b1_flat = b1.reshape(b1.shape[0], -1)[:, 0]
            if mu_y.ndim == 3 and mu_y.shape[1] == 1:
                mu_y = jnp.squeeze(mu_y, axis=1)
            mu_y = mu_y + b0_flat[:, None] + b1_flat[:, None] * x[None, :]
        else:
            # Support both global baseline params [draw] and legacy [draw, S...].
            x_exp = x[None, ...]
            if b0.ndim == 1:
                expand_shape = (b0.shape[0],) + (1,) * x.ndim
                b0_exp = b0.reshape(expand_shape)
                b1_exp = b1.reshape(expand_shape)
            else:
                b0_exp = b0.reshape((b0.shape[0],) + tuple(x.shape[:-1]) + (1,))
                b1_exp = b1.reshape((b1.shape[0],) + tuple(x.shape[:-1]) + (1,))
            mu_y = mu_y + b0_exp + b1_exp * x_exp
    elif x.ndim == 1 and mu_y.ndim == 3 and mu_y.shape[1] == 1:
        # Backward-compatible single-spectrum shape: [samples, 1, N] -> [samples, N]
        mu_y = jnp.squeeze(mu_y, axis=1)

    return mu_y


# =====================================================================
# ChromFitter class
# =====================================================================


class ChromFitter:
    """Bayesian skew-normal-mixture fitter for chromatographic peaks.

    Attributes:
        x: Retention time/data points
        y: Observed signal
        background: Estimated baseline/background, same shape as `y`
        mcmc: NumPyro MCMC object (after fit)
        idata: ArviZ InferenceData object (after fit)
        samples: Posterior samples dictionary (after fit)
    """

    def __init__(
        self,
        time: jnp.ndarray,
        signal: jnp.ndarray,
        peak_windows: Optional[list[PeakWindow]] = None,
        sample_names: Optional[list[str]] = None,
        chromatogram_names: Optional[list[list[str]]] = None,
        time_unit: str = "min",
        signal_unit: str = "a.u.",
    ):
        """Initialize ChromFitter with data and constraints.
        S is the number of samples, C is the number of chromatograms, N is the number of data points

        Args:
            time: Retention time/data points, shape [S, C, N]
            signal: Observed signal, shape [S, C, N]
            peak_windows: List of PeakWindow objects
            sample_names: List of sample names [S]
            chromatogram_names: List of chromatogram names [S, C]
            time_unit: Unit of time
            signal_unit: Unit of signal
        """
        self.time = jnp.asarray(time)
        self.signal = jnp.asarray(signal)
        self.x = self.time
        self.y = self.signal
        self.baseline_mask = jnp.zeros(self.signal.shape, dtype=bool)
        self.peak_mask = jnp.zeros(self.signal.shape, dtype=bool)
        # Baseline container always mirrors current signal dimensionality.
        self.background = jnp.zeros_like(self.signal, dtype=jnp.float32)
        self.peak_windows = [] if peak_windows is None else list(peak_windows)

        if self.signal.ndim != 3:
            raise ValueError(
                f"Expected signal with shape [S, C, N], got {self.signal.shape}"
            )
        n_samples = int(self.signal.shape[0])
        n_chrom = int(self.signal.shape[1])

        if sample_names is None or len(sample_names) == 0:
            self.sample_names = [f"sample_{s}" for s in range(n_samples)]
        else:
            self.sample_names = list(sample_names)

        if chromatogram_names is None or len(chromatogram_names) == 0:
            self.chromatogram_names = np.asarray(
                [[f"chrom_{c}" for c in range(n_chrom)] for _ in range(n_samples)],
                dtype=object,
            )
        else:
            self.chromatogram_names = np.asarray(chromatogram_names, dtype=object)

        self.time_unit = time_unit
        self.signal_unit = signal_unit

        # Peak-model state defaults.
        self.K = 0
        self.peak_names: list[str] = []
        self.peak_definitions: list[PeakDefinition] = []
        self.component_to_logical_index: list[int] = []
        self.component_is_shoulder: list[bool] = []
        self.component_shoulder_side: list[Literal["left", "right"] | None] = []
        self.component_include_in_total_area: list[bool] = []
        self.mu_lo = jnp.zeros((0,), dtype=jnp.float32)
        self.mu_hi = jnp.zeros((0,), dtype=jnp.float32)
        self.mu_init = jnp.zeros((n_samples * n_chrom, 0), dtype=jnp.float32)
        self.sigma_init = jnp.zeros((n_samples * n_chrom, 0), dtype=jnp.float32)
        self.A_init = jnp.zeros((n_samples * n_chrom, 0), dtype=jnp.float32)

        # Default prior/constraint hyperparameters.
        dx = jnp.median(jnp.abs(jnp.diff(self.time, axis=-1)))
        dx = jnp.maximum(dx, jnp.array(1e-6, dtype=jnp.float32))
        self.sigma_min = float(4.0 * dx)
        self.sigma_max = float(12.0 * dx)
        self.alpha_prior_sd = 1.0
        self.baseline_slope_quantile = 0.10
        self.baseline_curvature_quantile = 0.25
        self.baseline_anchor_mask = self._compute_baseline_anchor_mask(
            slope_quantile=self.baseline_slope_quantile,
            curvature_quantile=self.baseline_curvature_quantile,
        )

        @property
        def sampling_frequency(self) -> float:
            "The sampling frequency of the data"
            return float(jnp.min(jnp.diff(self.x, axis=-1)))

        self.shift_result: Optional[ShiftAlignmentResult] = None
        self.shift_chrom_deltas_samples: Optional[jnp.ndarray] = None
        self.shift_sample_deltas_samples: Optional[jnp.ndarray] = None
        self.shift_deltas_samples: Optional[jnp.ndarray] = None
        self.shift_deltas_time: Optional[jnp.ndarray] = None

        # Placeholders for results
        self.mcmc: Optional[MCMC] = None
        self.idata: Optional[az.InferenceData] = None
        self.samples: Optional[dict[str, Any]] = None
        self.moment_metrics: dict[str, dict[str, Any]] = {}
        self.figure_dir = Path("figs")
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self._figure_counts: dict[str, int] = {}

        # Display initialization summary with rich styling
        self._print_initialization_summary()

    def add_baseline_region(self, low: float, high: float) -> None:
        """Add a baseline region to the model.



        Args:
            low: The lower bound of the baseline region.
            high: The upper bound of the baseline region.
        """
        # update baseline mask by getting indices of time points within the region
        indices = jnp.where((self.time >= low) & (self.time <= high))
        self.baseline_mask = self.baseline_mask.at[indices].set(True)

    def _as_2d(self, arr: jnp.ndarray) -> jnp.ndarray:
        """Flatten leading axes and keep the last axis as data-point axis."""
        arr_jnp = jnp.asarray(arr, dtype=jnp.float32)
        if arr_jnp.ndim == 1:
            return arr_jnp[None, :]
        if arr_jnp.ndim == 2:
            return arr_jnp
        if arr_jnp.ndim >= 3:
            return arr_jnp.reshape(-1, arr_jnp.shape[-1])
        raise ValueError(f"Unsupported array shape for _as_2d: {arr_jnp.shape}")

    def _flat_trace_labels(self) -> list[str]:
        """Build flattened trace labels that match ``self._as_2d`` ordering."""
        n_samples, n_chrom, _ = self.signal.shape
        sample_names = list(self.sample_names)
        chromatogram_names = np.asarray(self.chromatogram_names, dtype=object)
        labels: list[str] = []

        for sample_index in range(n_samples):
            if sample_index < len(sample_names):
                sample_label = str(sample_names[sample_index])
            else:
                sample_label = f"sample_{sample_index}"

            for chromatogram_index in range(n_chrom):
                chrom_label = f"chrom_{chromatogram_index}"
                if chromatogram_names.ndim >= 2:
                    if (
                        sample_index < chromatogram_names.shape[0]
                        and chromatogram_index < chromatogram_names.shape[1]
                    ):
                        chrom_label = str(
                            chromatogram_names[sample_index, chromatogram_index]
                        )
                elif chromatogram_names.ndim == 1:
                    flat_index = sample_index * n_chrom + chromatogram_index
                    if flat_index < chromatogram_names.shape[0]:
                        chrom_label = str(chromatogram_names[flat_index])

                labels.append(f"{sample_label} | {chrom_label}")

        return labels

    def compute_peak_moment_metrics(
        self,
        peak_names: Optional[Sequence[str]] = None,
        start_quantile: float = 0.005,
        end_quantile: float = 0.995,
        tail_window_sigma: float = 2.0,
        use_background: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Compute moment-based diagnostics within logical peak windows.

        The user-defined logical peak bounds from `add_peak(...)` are used as
        broad initialization windows. Within each window, this method estimates
        tighter quantile bounds and shape metrics for every flattened trace.
        """
        if len(self.peak_definitions) == 0:
            raise RuntimeError("No peak definitions available. Add peaks first.")

        if peak_names is None:
            selected_definitions = list(self.peak_definitions)
        else:
            by_name = {
                definition.name: definition for definition in self.peak_definitions
            }
            missing = [name for name in peak_names if name not in by_name]
            if missing:
                raise ValueError(
                    f"Unknown peak names: {missing}. Available: {list(by_name.keys())}"
                )
            selected_definitions = [by_name[name] for name in peak_names]

        x2d = np.asarray(self._as_2d(self.x), dtype=float)
        y2d = np.asarray(self._as_2d(self.y), dtype=float)
        if use_background:
            background2d = np.asarray(self._as_2d(self.background), dtype=float)
            if background2d.shape != y2d.shape:
                raise ValueError(
                    "background must have the same flattened shape as signal."
                )
            y2d = y2d - background2d

        trace_labels = np.asarray(self._flat_trace_labels(), dtype=object)
        if trace_labels.shape[0] != x2d.shape[0]:
            trace_labels = np.asarray(
                [f"trace_{index}" for index in range(x2d.shape[0])], dtype=object
            )

        output: dict[str, dict[str, Any]] = {}
        for definition in selected_definitions:
            metrics_list = compute_peak_moment_metrics_batch(
                x_matrix=x2d,
                y_matrix=y2d,
                window_low=float(definition.low),
                window_high=float(definition.high),
                start_quantile=float(start_quantile),
                end_quantile=float(end_quantile),
                tail_window_sigma=float(tail_window_sigma),
            )
            metric_arrays = metrics_list_to_arrays(metrics_list)
            metric_summary = summarize_metrics(metric_arrays)
            metric_arrays["trace_index"] = np.arange(x2d.shape[0], dtype=int)
            metric_arrays["trace_label"] = trace_labels

            output[definition.name] = {
                "definition": definition,
                "metrics": metric_arrays,
                "summary": metric_summary,
            }

        self.moment_metrics = output
        return output

    def plot_data(
        self,
        linestyle: Literal["scatter", "line"] = "scatter",
        size: float = 4,
        alpha: float = 0.2,
        ymax: Optional[float] = None,
        ymin: Optional[float] = None,
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Plot chromatogram data with global scatter + peak-window line overlays.

        Args:
            linestyle: Kept for API compatibility (ignored by rendering logic).
            size: Scatter marker size for the global data cloud.
            alpha: Scatter alpha for the global data cloud.
            ymax: If specified, sets the upper y-axis limit.
            ymin: If specified, sets the lower y-axis limit.
            save_path: If specified, save the figure to this path.
            dpi: Resolution for saved figure.

        Returns:
            fig: The figure object.
            axes: The array of axes (one per sample).

        Note: One can specify both `ymax` and `ymin`.
        """

        _ = linestyle  # Backward-compatible arg; rendering is now fixed-style.
        n_samples, n_chrom, n_time = self.signal.shape

        cmap = plt.get_cmap("viridis")
        chrom_colors = np.asarray(
            [cmap(i / max(n_chrom - 1, 1)) for i in range(n_chrom)],
            dtype=float,
        )  # (n_chrom, 4)

        fig, axes = plt.subplots(
            n_samples, 1, figsize=(8, n_samples * 4), squeeze=False
        )

        # Resolve peak windows per sample from mu bounds if available; fallback to peak_windows.
        sample_windows: list[list[tuple[float, float]]] = [[] for _ in range(n_samples)]
        if (
            hasattr(self, "mu_lo")
            and hasattr(self, "mu_hi")
            and hasattr(self, "K")
            and int(getattr(self, "K", 0)) > 0
        ):
            mu_lo_arr = np.asarray(self.mu_lo, dtype=float)
            mu_hi_arr = np.asarray(self.mu_hi, dtype=float)
            if mu_lo_arr.ndim == 1 and mu_hi_arr.ndim == 1:
                shared = [
                    (float(lo), float(hi))
                    for lo, hi in zip(mu_lo_arr, mu_hi_arr)
                    if np.isfinite(lo) and np.isfinite(hi) and hi > lo
                ]
                for s in range(n_samples):
                    sample_windows[s] = shared
            elif mu_lo_arr.ndim == 2 and mu_hi_arr.ndim == 2:
                if mu_lo_arr.shape[0] == n_samples and mu_hi_arr.shape[0] == n_samples:
                    for s in range(n_samples):
                        sample_windows[s] = [
                            (float(lo), float(hi))
                            for lo, hi in zip(mu_lo_arr[s], mu_hi_arr[s])
                            if np.isfinite(lo) and np.isfinite(hi) and hi > lo
                        ]
                elif (
                    mu_lo_arr.shape[0] == n_samples * n_chrom
                    and mu_hi_arr.shape[0] == n_samples * n_chrom
                ):
                    for s in range(n_samples):
                        row = s * n_chrom
                        sample_windows[s] = [
                            (float(lo), float(hi))
                            for lo, hi in zip(mu_lo_arr[row], mu_hi_arr[row])
                            if np.isfinite(lo) and np.isfinite(hi) and hi > lo
                        ]
        elif hasattr(self, "peak_windows") and self.peak_windows:
            shared = [
                (float(w.x1), float(w.x2))
                for w in self.peak_windows
                if np.isfinite(w.x1) and np.isfinite(w.x2) and float(w.x2) > float(w.x1)
            ]
            for s in range(n_samples):
                sample_windows[s] = shared

        for s in range(n_samples):
            ax = axes[s, 0]
            ax.set_title(str(self.sample_names[s]))

            x = np.asarray(self.time[s])
            y = np.asarray(self.signal[s])

            # Accept either (n_chrom, n_time) or (n_time, n_chrom) and normalize to (n_time, n_chrom)
            if x.shape == (n_chrom, n_time):
                x = x.T
            elif x.shape != (n_time, n_chrom):
                raise ValueError(
                    f"time[{s}] has shape {x.shape}, expected {(n_chrom, n_time)} or {(n_time, n_chrom)}"
                )

            if y.shape == (n_chrom, n_time):
                y = y.T
            elif y.shape != (n_time, n_chrom):
                raise ValueError(
                    f"signal[{s}] has shape {y.shape}, expected {(n_chrom, n_time)} or {(n_time, n_chrom)}"
                )

            for k in range(n_chrom):
                label = str(self.chromatogram_names[s, k])
                xk = np.asarray(x[:, k], dtype=float)
                yk = np.asarray(y[:, k], dtype=float)

                # Global raw data view.
                ax.scatter(
                    xk,
                    yk,
                    color=chrom_colors[k],
                    s=size,
                    linewidths=0,
                    alpha=alpha,
                    label=label,
                )

                # Peak-window overlay with full opacity.
                peak_mask = np.zeros((n_time,), dtype=bool)
                for lo, hi in sample_windows[s]:
                    peak_mask |= (xk >= lo) & (xk <= hi)
                if np.any(peak_mask):
                    y_line = np.where(peak_mask, yk, np.nan)
                    ax.plot(
                        xk,
                        y_line,
                        color=chrom_colors[k],
                        linewidth=1.2,
                        alpha=1.0,
                    )

            ax.legend(loc="best", fontsize="small")
            ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))
            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
            ax.tick_params(axis="x", which="minor", bottom=True, length=3)
            ax.tick_params(axis="y", which="minor", left=True, length=3)
            ax.grid(True, which="both", alpha=0.2)

            ax.set_xlabel(f"Time [{self.time_unit}]")
            ax.set_ylabel(f"Signal [{self.signal_unit}]")
            # Allow specifying ymin and ymax, with input validation
            current_ymin, current_ymax = ax.get_ylim()

            final_ymin = current_ymin if ymin is None else float(ymin)
            final_ymax = current_ymax if ymax is None else float(ymax)

            if final_ymax <= final_ymin:
                raise ValueError(
                    f"ymax ({final_ymax}) must be greater than ymin ({final_ymin})."
                )
            ax.set_ylim(final_ymin, final_ymax)

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return fig, axes

    def _refresh_after_signal_update(self, clear_posterior: bool = True) -> None:
        """Recompute data-derived caches after `self.y` is modified."""
        # Shift/baseline preprocessing should work even before any peaks are defined.
        if not hasattr(self, "K"):
            self.K = 0
        if not hasattr(self, "mu_lo"):
            self.mu_lo = jnp.zeros((0,), dtype=jnp.float32)
        if not hasattr(self, "mu_hi"):
            self.mu_hi = jnp.zeros((0,), dtype=jnp.float32)
        if not hasattr(self, "peak_names"):
            self.peak_names = []
        if not hasattr(self, "peak_definitions"):
            self.peak_definitions = []
        if not hasattr(self, "component_to_logical_index"):
            self.component_to_logical_index = []
        if not hasattr(self, "component_is_shoulder"):
            self.component_is_shoulder = []
        if not hasattr(self, "component_shoulder_side"):
            self.component_shoulder_side = []
        if not hasattr(self, "component_include_in_total_area"):
            self.component_include_in_total_area = []
        if not hasattr(self, "baseline_slope_quantile"):
            self.baseline_slope_quantile = 0.10
        if not hasattr(self, "baseline_curvature_quantile"):
            self.baseline_curvature_quantile = 0.25

        self.mu_init, self.sigma_init, self.A_init = (
            self._compute_all_peak_initializers()
        )
        self.baseline_anchor_mask = self._compute_baseline_anchor_mask(
            slope_quantile=self.baseline_slope_quantile,
            curvature_quantile=self.baseline_curvature_quantile,
        )
        if clear_posterior:
            self.mcmc = None
            self.idata = None
            self.samples = None

    def apply_retention_shift_correction(
        self,
        lr: float = 1e-2,
        n_steps: int = 500,
        center_weight: float = 1e3,
        max_shift_samples: Optional[float] = None,
        enforce_zero_mean: bool = True,
        return_history: bool = False,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Apply two-stage retention-shift correction on ``self.time``.

        Stage 1: align chromatograms within each sample.
        Stage 2: align all samples jointly (shared shift per sample).
        """
        signal_arr = jnp.asarray(self.signal, dtype=jnp.float32)
        time_arr = jnp.asarray(self.time, dtype=jnp.float32)
        if signal_arr.ndim != 3 or time_arr.ndim != 3:
            raise ValueError(
                f"Expected signal/time shape [S, C, N], got "
                f"signal={signal_arr.shape}, time={time_arr.shape}"
            )
        if signal_arr.shape != time_arr.shape:
            raise ValueError(
                f"signal/time shape mismatch: {signal_arr.shape} vs {time_arr.shape}"
            )

        result = align_groupwise_sample_shifts(
            signal=signal_arr,
            lr=lr,
            n_steps=n_steps,
            center_weight=center_weight,
            max_shift_samples=max_shift_samples,
            enforce_zero_mean=enforce_zero_mean,
            return_history=return_history,
        )

        self.shift_result = result
        self.shift_chrom_deltas_samples = jnp.asarray(
            result.chromatogram_shifts_samples, dtype=jnp.float32
        )
        self.shift_sample_deltas_samples = jnp.asarray(
            result.sample_shifts_samples, dtype=jnp.float32
        )
        self.shift_deltas_samples = jnp.asarray(
            result.total_shifts_samples, dtype=jnp.float32
        )

        # Convert shift from sample-index units to time units per [S,C] trace.
        s_count, _, n_points = signal_arr.shape
        if n_points > 1:
            time_np = np.asarray(time_arr, dtype=np.float64)
            c_count = int(time_np.shape[1])
            dx_sc_np = np.ones((s_count, c_count), dtype=np.float32)
            for s in range(s_count):
                for c in range(c_count):
                    trace_diffs = np.diff(time_np[s, c]).reshape(-1)
                    trace_diffs = np.abs(trace_diffs[np.isfinite(trace_diffs)])
                    if trace_diffs.size > 0:
                        dx_sc_np[s, c] = float(np.median(trace_diffs))
            dx_sc = jnp.asarray(dx_sc_np, dtype=jnp.float32)
        else:
            dx_sc = jnp.ones((s_count, signal_arr.shape[1]), dtype=jnp.float32)
        self.shift_deltas_time = self.shift_deltas_samples * dx_sc

        # Shift x-axis (time) per sample; keep measured signal unchanged.
        self.time = jnp.asarray(
            time_arr + self.shift_deltas_time[:, :, None], dtype=jnp.float32
        )
        self.x = self.time
        self.y = self.signal
        self._refresh_after_signal_update(clear_posterior=True)

        loss_initial = float(result.loss_initial)
        loss_final = float(result.loss_final)
        loss_drop = loss_initial - loss_final
        loss_drop_pct = 100.0 * loss_drop / max(abs(loss_initial), 1e-8)

        if verbose:
            print("\n[shift] Groupwise retention shift correction")
            print(f"  samples (S): {s_count}")
            print(f"  chromatograms per sample (C): {signal_arr.shape[1]}")
            print(f"  points per chromatogram (N): {signal_arr.shape[2]}")
            print(
                f"  optimizer: Adam-like (pure JAX), lr={lr:.4g}, n_steps={n_steps}, "
                f"center_weight={center_weight:.4g}"
            )
            if max_shift_samples is not None:
                print(f"  max |shift| constraint: {max_shift_samples:.4f} samples")
            print(f"  loss initial: {loss_initial:.6e}")
            print(f"  loss final:   {loss_final:.6e}")
            print(f"  loss delta:   {loss_drop:.6e} ({loss_drop_pct:.2f}%)")
            print(
                f"  stage 1 (within-sample) loss: "
                f"{float(result.within_loss_initial):.6e} -> {float(result.within_loss_final):.6e}"
            )
            print(
                f"  stage 2 (across-sample) loss: "
                f"{float(result.across_loss_initial):.6e} -> {float(result.across_loss_final):.6e}"
            )
            print("  per-sample joint shifts:")
            for s in range(int(self.shift_deltas_samples.shape[0])):
                ds_joint = float(self.shift_sample_deltas_samples[s])
                dt_joint = float(jnp.mean(self.shift_deltas_time[s]))
                within_min = float(jnp.min(self.shift_chrom_deltas_samples[s]))
                within_max = float(jnp.max(self.shift_chrom_deltas_samples[s]))
                print(
                    f"    sample {s:>2d}: joint_shift_samples={ds_joint:+.5f}, "
                    f"joint_shift_time≈{dt_joint:+.6f}, "
                    f"within_chrom_shifts=[{within_min:+.5f}, {within_max:+.5f}]"
                )

        return {
            "chrom_deltas_samples": self.shift_chrom_deltas_samples,
            "sample_deltas_samples": self.shift_sample_deltas_samples,
            "deltas_samples": self.shift_deltas_samples,
            "deltas_time": self.shift_deltas_time,
            "loss_initial": loss_initial,
            "loss_final": loss_final,
            "loss_delta": loss_drop,
            "loss_delta_percent": loss_drop_pct,
            "within_loss_initial": float(result.within_loss_initial),
            "within_loss_final": float(result.within_loss_final),
            "across_loss_initial": float(result.across_loss_initial),
            "across_loss_final": float(result.across_loss_final),
            "template": result.template,
            "within_loss_history": result.within_loss_history,
            "across_loss_history": result.across_loss_history,
        }

    # ---------------------------------------------------------------------------
    # Orchestration  (handles the [S, C, N] matrix, NaN interpolation, verbosity)
    # ---------------------------------------------------------------------------

    def clip_signal(
        self,
        threshold: float,
        clip_value: float,
        mode: Literal["below", "above"] = "below",
        verbose: bool = True,
    ) -> jnp.ndarray:
        """Clip signal values either below or above a threshold.

        Parameters
        ----------
        threshold : float
            Threshold to compare against.
        clip_value : float
            Replacement value applied to clipped points.
        mode : {"below", "above"}
            - "below": replace points where signal < threshold
            - "above": replace points where signal > threshold
        verbose : bool
            Print clipping summary.

        Returns
        -------
        jnp.ndarray
            Clipped signal array (also stored back to ``self.signal`` and ``self.y``).
        """
        signal_src = self.signal if hasattr(self, "signal") else self.y
        signal_arr = jnp.asarray(signal_src, dtype=jnp.float32)
        threshold_f = jnp.asarray(threshold, dtype=jnp.float32)
        clip_f = jnp.asarray(clip_value, dtype=jnp.float32)

        if mode == "below":
            mask = signal_arr < threshold_f
        elif mode == "above":
            mask = signal_arr > threshold_f
        else:
            raise ValueError(f"Invalid mode `{mode}`. Use 'below' or 'above'.")

        clipped = jnp.where(mask, clip_f, signal_arr)
        n_clipped = int(jnp.sum(mask))
        n_total = int(signal_arr.size)

        self.signal = clipped
        self.y = clipped

        # Refresh model caches only when legacy model state is initialized.
        if all(
            hasattr(self, n)
            for n in (
                "mu_lo",
                "mu_hi",
                "K",
                "baseline_slope_quantile",
                "baseline_curvature_quantile",
            )
        ):
            self._refresh_after_signal_update(clear_posterior=True)

        if verbose:
            pct = 100.0 * n_clipped / max(n_total, 1)
            print("\n[clip] Applied signal clipping")
            print(f"  mode: {mode}")
            print(f"  threshold: {float(threshold_f):.6g}")
            print(f"  clip_value: {float(clip_f):.6g}")
            print(f"  clipped points: {n_clipped}/{n_total} ({pct:.2f}%)")

        return clipped

    def slice_time_ranges(
        self,
        ranges: Sequence[tuple[float, float]],
        verbose: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Keep only points whose time lies within provided x-ranges.

        Parameters
        ----------
        ranges : sequence of (xmin, xmax)
            Time windows to keep. The kept points are concatenated in their
            original order along the last axis.
        verbose : bool
            Print a short summary.

        Returns
        -------
        (time, signal) : tuple[jnp.ndarray, jnp.ndarray]
            Updated arrays after slicing.
        """
        if len(ranges) == 0:
            raise ValueError("ranges must contain at least one (xmin, xmax) tuple.")

        time_arr = np.asarray(self.time, dtype=np.float64)
        signal_arr = np.asarray(self.signal, dtype=np.float64)
        baseline_arr = np.asarray(self.background, dtype=np.float64)

        if time_arr.shape != signal_arr.shape:
            raise ValueError(
                f"time and signal shapes must match, got {time_arr.shape} vs {signal_arr.shape}"
            )
        if baseline_arr.shape != signal_arr.shape:
            raise ValueError(
                f"background shape {baseline_arr.shape} must match signal shape {signal_arr.shape}"
            )
        if time_arr.ndim < 1:
            raise ValueError(
                "time/signal must have at least one axis with time on the last axis."
            )

        # Build union mask for requested windows.
        keep_mask = np.zeros_like(time_arr, dtype=bool)
        validated_ranges: list[tuple[float, float]] = []
        for x_min, x_max in ranges:
            if x_max <= x_min:
                raise ValueError(f"Invalid range ({x_min}, {x_max}); need xmax > xmin.")
            x_min_f = float(x_min)
            x_max_f = float(x_max)
            validated_ranges.append((x_min_f, x_max_f))
            keep_mask |= (time_arr >= x_min_f) & (time_arr <= x_max_f)

        # Every trace must keep the same number of points to preserve tensor shape.
        n_points_per_trace = keep_mask.sum(axis=-1)
        unique_counts = np.unique(n_points_per_trace)
        if unique_counts.size != 1:
            raise ValueError(
                "Requested ranges keep different point counts across traces. "
                "Ensure aligned time axes across traces before slicing."
            )

        kept_points = int(unique_counts[0])
        if kept_points == 0:
            raise ValueError("No points selected. Check the provided ranges.")

        leading_shape = time_arr.shape[:-1]
        n_traces = int(np.prod(leading_shape)) if leading_shape else 1
        n_total = int(time_arr.shape[-1])

        time_2d = time_arr.reshape(n_traces, n_total)
        signal_2d = signal_arr.reshape(n_traces, n_total)
        baseline_2d = baseline_arr.reshape(n_traces, n_total)
        mask_2d = keep_mask.reshape(n_traces, n_total)

        sliced_time_2d = np.empty((n_traces, kept_points), dtype=np.float64)
        sliced_signal_2d = np.empty((n_traces, kept_points), dtype=np.float64)
        sliced_baseline_2d = np.empty((n_traces, kept_points), dtype=np.float64)

        for trace_idx in range(n_traces):
            keep_idx = np.flatnonzero(mask_2d[trace_idx])
            sliced_time_2d[trace_idx] = time_2d[trace_idx, keep_idx]
            sliced_signal_2d[trace_idx] = signal_2d[trace_idx, keep_idx]
            sliced_baseline_2d[trace_idx] = baseline_2d[trace_idx, keep_idx]

        new_shape = leading_shape + (kept_points,)
        self.time = jnp.asarray(sliced_time_2d.reshape(new_shape), dtype=jnp.float32)
        self.signal = jnp.asarray(
            sliced_signal_2d.reshape(new_shape), dtype=jnp.float32
        )
        self.background = jnp.asarray(
            sliced_baseline_2d.reshape(new_shape), dtype=jnp.float32
        )

        # Keep backward-compat aliases in sync.
        self.x = self.time
        self.y = self.signal

        # Refresh model caches only when legacy model state is initialized.
        if all(
            hasattr(self, n)
            for n in (
                "mu_lo",
                "mu_hi",
                "K",
                "baseline_slope_quantile",
                "baseline_curvature_quantile",
            )
        ):
            self._refresh_after_signal_update(clear_posterior=True)

        if verbose:
            print("\n[slice] Applied time-range slicing")
            print(f"  ranges: {validated_ranges}")
            print(f"  old shape: {time_arr.shape}")
            print(f"  new shape: {tuple(self.time.shape)}")
            print(f"  kept points per trace: {kept_points}/{n_total}")

        return self.time, self.signal

    def _window_mask(self, x_s: jnp.ndarray, low: float, high: float) -> jnp.ndarray:
        return (x_s >= low) & (x_s <= high)

    def _window_baseline(self, x_w: jnp.ndarray, y_w: jnp.ndarray) -> jnp.ndarray:
        if x_w.size < 2:
            return jnp.zeros_like(y_w)
        x0 = x_w[0]
        x1 = x_w[-1]
        y0 = y_w[0]
        y1 = y_w[-1]
        dx = jnp.maximum(x1 - x0, jnp.array(1e-8, dtype=jnp.float32))
        slope = (y1 - y0) / dx
        return y0 + slope * (x_w - x0)

    def _window_weights(self, x_w: jnp.ndarray, y_w: jnp.ndarray) -> jnp.ndarray:
        baseline = self._window_baseline(x_w, y_w)
        return jnp.clip(y_w - baseline, a_min=0.0, a_max=None)

    def _moment_mu(
        self, x_w: jnp.ndarray, w_w: jnp.ndarray, low: float, high: float
    ) -> jnp.ndarray:
        w_sum = jnp.sum(w_w)
        fallback = jnp.array(0.5 * (low + high), dtype=jnp.float32)
        return jnp.where(w_sum > 1e-8, jnp.sum(x_w * w_w) / w_sum, fallback)

    def _moment_sigma(
        self,
        x_w: jnp.ndarray,
        w_w: jnp.ndarray,
        mu0: jnp.ndarray,
        low: float,
        high: float,
    ) -> jnp.ndarray:
        w_sum = jnp.sum(w_w)
        fallback = jnp.maximum(
            jnp.array((high - low) / 6.0, dtype=jnp.float32),
            jnp.array(1e-4, dtype=jnp.float32),
        )
        var = jnp.sum(w_w * (x_w - mu0) ** 2) / jnp.maximum(
            w_sum, jnp.array(1e-8, dtype=jnp.float32)
        )
        return jnp.where(
            w_sum > 1e-8,
            jnp.sqrt(jnp.maximum(var, jnp.array(1e-10, dtype=jnp.float32))),
            fallback,
        )

    def _trapezoid_area(self, x_w: jnp.ndarray, w_w: jnp.ndarray) -> jnp.ndarray:
        if x_w.size < 2:
            return jnp.maximum(jnp.sum(w_w), jnp.array(1e-8, dtype=jnp.float32))
        return jnp.maximum(jnp.trapezoid(w_w, x_w), jnp.array(1e-8, dtype=jnp.float32))

    def _window_bounds_mats(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        x2d = self._as_2d(self.x)
        S = x2d.shape[0]
        if self.mu_lo.ndim == 1 and self.mu_hi.ndim == 1:
            lo = jnp.broadcast_to(self.mu_lo[None, :], (S, self.K))
            hi = jnp.broadcast_to(self.mu_hi[None, :], (S, self.K))
        elif self.mu_lo.ndim == 2 and self.mu_hi.ndim == 2:
            lo = self.mu_lo
            hi = self.mu_hi
        else:
            raise ValueError(
                "mu_lo and mu_hi must both be 1D [K] or both be 2D [S, K]."
            )
        return lo, hi

    def _outside_peak_window_mask(self) -> jnp.ndarray:
        x2d = self._as_2d(self.x)
        y2d = self._as_2d(self.y)
        S, N = x2d.shape

        if self.K == 0:
            inside_window = jnp.zeros((S, N), dtype=bool)
        else:
            lo_mat, hi_mat = self._window_bounds_mats()
            inside_window = jnp.any(
                (x2d[:, None, :] >= lo_mat[:, :, None])
                & (x2d[:, None, :] <= hi_mat[:, :, None]),
                axis=1,
            )
        outside_window = ~inside_window
        finite_xy = jnp.isfinite(x2d) & jnp.isfinite(y2d)
        return outside_window & finite_xy

    def _compute_baseline_anchor_mask(
        self, slope_quantile: float, curvature_quantile: float
    ) -> jnp.ndarray:
        if not (0.0 < slope_quantile <= 1.0):
            raise ValueError("slope_quantile must be in (0, 1].")
        if not (0.0 < curvature_quantile <= 1.0):
            raise ValueError("curvature_quantile must be in (0, 1].")

        x2d = self._as_2d(self.x)
        y2d = self._as_2d(self.y)
        S, N = x2d.shape
        if N < 2:
            raise ValueError("Need at least 2 points per spectrum to compute slopes.")

        outside_window = self._outside_peak_window_mask()

        dx = jnp.diff(x2d, axis=1)
        dy = jnp.diff(y2d, axis=1)
        dx_safe = jnp.where(jnp.abs(dx) > 1e-12, dx, jnp.nan)
        slope_abs = jnp.abs(dy / dx_safe)  # [S, N-1]

        if N >= 3:
            d2y = y2d[:, 2:] - 2.0 * y2d[:, 1:-1] + y2d[:, :-2]  # [S, N-2]
            dx_left = x2d[:, 1:-1] - x2d[:, :-2]
            dx_right = x2d[:, 2:] - x2d[:, 1:-1]
            dx_mid = 0.5 * (dx_left + dx_right)
            dx2_safe = jnp.where(jnp.abs(dx_mid) > 1e-12, dx_mid**2, jnp.nan)
            curv_mid = jnp.abs(d2y / dx2_safe)  # [S, N-2]

            curv_pt = jnp.full((S, N), jnp.nan, dtype=jnp.float32)
            curv_pt = curv_pt.at[:, 1:-1].set(curv_mid)
            curv_interval = jnp.maximum(curv_pt[:, :-1], curv_pt[:, 1:])  # [S, N-1]
        else:
            curv_interval = jnp.zeros((S, N - 1), dtype=jnp.float32)

        interval_ok = (
            outside_window[:, :-1]
            & outside_window[:, 1:]
            & jnp.isfinite(slope_abs)
            & jnp.isfinite(curv_interval)
        )

        selected_points = jnp.zeros((S, N), dtype=bool)
        for s in range(S):
            cand_slope = slope_abs[s][interval_ok[s]]
            cand_curv = curv_interval[s][interval_ok[s]]
            if int(cand_slope.size) == 0:
                continue

            slope_thresh = jnp.quantile(cand_slope, slope_quantile)
            curv_thresh = jnp.quantile(cand_curv, curvature_quantile)
            low_intervals = (
                interval_ok[s]
                & (slope_abs[s] <= slope_thresh)
                & (curv_interval[s] <= curv_thresh)
            )

            selected_row = jnp.zeros((N,), dtype=bool)
            selected_row = selected_row.at[:-1].set(selected_row[:-1] | low_intervals)
            selected_row = selected_row.at[1:].set(selected_row[1:] | low_intervals)
            selected_points = selected_points.at[s].set(selected_row)

        return selected_points

    def _combined_baseline_likelihood_mask(
        self,
        x2d: jnp.ndarray,
        y2d: jnp.ndarray,
        peak_mask2d: jnp.ndarray,
    ) -> jnp.ndarray:
        """Build baseline-likelihood mask from manual and anchor baseline masks.

        The returned mask:
        1) combines `self.baseline_mask` and `self.baseline_anchor_mask`,
        2) keeps only finite x/y points,
        3) excludes peak-mask points,
        4) falls back per trace to all non-peak finite points if no baseline points
           remain for that trace.
        """
        finite_mask = jnp.isfinite(x2d) & jnp.isfinite(y2d)

        manual_mask = self._as_2d(self.baseline_mask).astype(bool)
        if manual_mask.shape != x2d.shape:
            manual_mask = jnp.zeros_like(finite_mask)

        anchor_mask = self._as_2d(self.baseline_anchor_mask).astype(bool)
        if anchor_mask.shape != x2d.shape:
            anchor_mask = jnp.zeros_like(finite_mask)

        combined_mask = (manual_mask | anchor_mask) & finite_mask & (~peak_mask2d)
        fallback_mask = finite_mask & (~peak_mask2d)

        has_points = jnp.sum(combined_mask, axis=-1) > 0
        return jnp.where(has_points[:, None], combined_mask, fallback_mask)

    def _baseline_linear_regression_initializers(
        self,
        x2d: jnp.ndarray,
        y2d: jnp.ndarray,
        baseline_mask2d: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        """Estimate baseline linear-regression initializers for model priors.

        Args:
            x2d: Flattened retention-time matrix with shape ``[S, N]``.
            y2d: Flattened signal matrix with shape ``[S, N]``.
            baseline_mask2d: Boolean baseline mask with shape ``[S, N]``.

        Returns:
            Dictionary with per-trace initializers (`b0_init`, `b1_init`) and
            pooled hyper initializers/scales (`b0_hyper_*`, `b1_hyper_*`).
        """
        finite_xy = jnp.isfinite(x2d) & jnp.isfinite(y2d)
        active_mask = baseline_mask2d & finite_xy
        weights = active_mask.astype(jnp.float32)

        n_points = jnp.sum(weights, axis=-1)
        safe_n_points = jnp.maximum(n_points, jnp.array(1.0, dtype=jnp.float32))

        sum_x = jnp.sum(weights * x2d, axis=-1)
        sum_y = jnp.sum(weights * y2d, axis=-1)
        sum_xx = jnp.sum(weights * x2d * x2d, axis=-1)
        sum_xy = jnp.sum(weights * x2d * y2d, axis=-1)

        denominator = safe_n_points * sum_xx - sum_x * sum_x
        valid_fit = (n_points >= 2.0) & (jnp.abs(denominator) > 1e-12)

        slope_estimate = jnp.where(
            valid_fit,
            (safe_n_points * sum_xy - sum_x * sum_y) / denominator,
            0.0,
        )
        intercept_estimate = jnp.where(
            n_points > 0.0,
            (sum_y - slope_estimate * sum_x) / safe_n_points,
            0.0,
        )

        per_trace_median = jnp.nanmedian(
            jnp.where(jnp.isfinite(y2d), y2d, jnp.nan), axis=-1
        )
        per_trace_median = jnp.where(
            jnp.isfinite(per_trace_median), per_trace_median, 0.0
        )
        intercept_estimate = jnp.where(
            n_points > 0.0, intercept_estimate, per_trace_median
        )

        weights_all = weights
        n_all = jnp.sum(weights_all)
        safe_n_all = jnp.maximum(n_all, jnp.array(1.0, dtype=jnp.float32))
        sum_x_all = jnp.sum(weights_all * x2d)
        sum_y_all = jnp.sum(weights_all * y2d)
        sum_xx_all = jnp.sum(weights_all * x2d * x2d)
        sum_xy_all = jnp.sum(weights_all * x2d * y2d)

        denominator_all = safe_n_all * sum_xx_all - sum_x_all * sum_x_all
        valid_fit_all = (n_all >= 2.0) & (jnp.abs(denominator_all) > 1e-12)

        slope_hyper_init = jnp.where(
            valid_fit_all,
            (safe_n_all * sum_xy_all - sum_x_all * sum_y_all) / denominator_all,
            jnp.median(slope_estimate),
        )
        intercept_hyper_init = jnp.where(
            valid_fit_all,
            (sum_y_all - slope_hyper_init * sum_x_all) / safe_n_all,
            jnp.median(intercept_estimate),
        )

        intercept_median = jnp.median(intercept_estimate)
        intercept_mad = jnp.median(jnp.abs(intercept_estimate - intercept_median))
        intercept_robust_sd = 1.4826 * intercept_mad

        slope_median = jnp.median(slope_estimate)
        slope_mad = jnp.median(jnp.abs(slope_estimate - slope_median))
        slope_robust_sd = 1.4826 * slope_mad

        y_scale = jnp.maximum(
            jnp.nanmax(jnp.abs(jnp.where(jnp.isfinite(y2d), y2d, jnp.nan))),
            jnp.array(1.0, dtype=jnp.float32),
        )
        x_min = jnp.nanmin(jnp.where(jnp.isfinite(x2d), x2d, jnp.nan))
        x_max = jnp.nanmax(jnp.where(jnp.isfinite(x2d), x2d, jnp.nan))
        x_span = jnp.maximum(x_max - x_min, jnp.array(1e-6, dtype=jnp.float32))

        intercept_hyper_sd_init = jnp.maximum(
            intercept_robust_sd, jnp.array(0.01, dtype=jnp.float32) * y_scale
        )
        slope_hyper_sd_init = jnp.maximum(
            slope_robust_sd,
            jnp.array(0.01, dtype=jnp.float32) * y_scale / x_span,
        )

        return {
            "baseline_mask": baseline_mask2d.astype(bool),
            "b0_init": intercept_estimate.astype(jnp.float32),
            "b1_init": slope_estimate.astype(jnp.float32),
            "b0_hyper_init": jnp.asarray(intercept_hyper_init, dtype=jnp.float32),
            "b1_hyper_init": jnp.asarray(slope_hyper_init, dtype=jnp.float32),
            "b0_hyper_sd_init": jnp.asarray(
                jnp.maximum(intercept_hyper_sd_init, 1e-4), dtype=jnp.float32
            ),
            "b1_hyper_sd_init": jnp.asarray(
                jnp.maximum(slope_hyper_sd_init, 1e-6), dtype=jnp.float32
            ),
        }

    def _estimate_peak_initializers(
        self, low_vec: jnp.ndarray, high_vec: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x2d = self._as_2d(self.x)
        y2d = self._as_2d(self.y)
        S = x2d.shape[0]

        mu_vals = []
        sigma_vals = []
        area_vals = []
        for s in range(S):
            x_s = x2d[s]
            y_s = jnp.nan_to_num(y2d[s], nan=0.0, posinf=0.0, neginf=0.0)
            low = float(low_vec[s])
            high = float(high_vec[s])
            mask = self._window_mask(x_s, low, high)
            if bool(jnp.any(mask)):
                x_w = x_s[mask]
                y_w = y_s[mask]
            else:
                x_w = x_s
                y_w = jnp.zeros_like(y_s)

            w_w = self._window_weights(x_w, y_w)
            mu0 = self._moment_mu(x_w, w_w, low, high)
            sigma0 = self._moment_sigma(x_w, w_w, mu0, low, high)
            A0 = self._trapezoid_area(x_w, w_w)

            mu_vals.append(mu0)
            sigma_vals.append(sigma0)
            area_vals.append(A0)

        return (
            jnp.asarray(mu_vals, dtype=jnp.float32),
            jnp.asarray(sigma_vals, dtype=jnp.float32),
            jnp.asarray(area_vals, dtype=jnp.float32),
        )

    def _compute_all_peak_initializers(
        self,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x2d = self._as_2d(self.x)
        S = x2d.shape[0]
        if self.K == 0:
            z = jnp.zeros((S, 0), dtype=jnp.float32)
            return z, z, z

        lo_mat, hi_mat = self._window_bounds_mats()
        mu_cols = []
        sigma_cols = []
        area_cols = []
        for k in range(self.K):
            mu0_k, sigma0_k, A0_k = self._estimate_peak_initializers(
                lo_mat[:, k], hi_mat[:, k]
            )
            mu_cols.append(mu0_k[:, None])
            sigma_cols.append(sigma0_k[:, None])
            area_cols.append(A0_k[:, None])

        mu_init = jnp.concatenate(mu_cols, axis=1)
        sigma_init = jnp.concatenate(sigma_cols, axis=1)
        A_init = jnp.concatenate(area_cols, axis=1)
        return mu_init, sigma_init, A_init

    def _append_peak_bounds(
        self, low_vals: list[float], high_vals: list[float]
    ) -> None:
        x2d = self._as_2d(self.x)
        S = x2d.shape[0]
        low_arr = jnp.asarray(low_vals, dtype=jnp.float32)
        high_arr = jnp.asarray(high_vals, dtype=jnp.float32)
        if self.mu_lo.ndim == 1:
            self.mu_lo = jnp.concatenate([self.mu_lo, low_arr], axis=0)
            self.mu_hi = jnp.concatenate([self.mu_hi, high_arr], axis=0)
        elif self.mu_lo.ndim == 2:
            self.mu_lo = jnp.concatenate(
                [self.mu_lo, jnp.broadcast_to(low_arr[None, :], (S, low_arr.shape[0]))],
                axis=1,
            )
            self.mu_hi = jnp.concatenate(
                [
                    self.mu_hi,
                    jnp.broadcast_to(high_arr[None, :], (S, high_arr.shape[0])),
                ],
                axis=1,
            )
        else:
            raise ValueError("mu_lo/mu_hi must be 1D or 2D.")

    def add_peak(
        self,
        name: str,
        low: float,
        high: float,
        shoulder: Literal["left", "right"] | None = None,
        exclude_shoulder: bool = False,
    ) -> dict[str, Any]:
        """Register a logical peak and compile corresponding model components.

        Args:
            name: Logical peak name.
            low: Lower retention-time bound.
            high: Upper retention-time bound.
            shoulder: Shoulder side. ``None`` creates a single component; otherwise
                creates a double peak with one main and one shoulder component.
            exclude_shoulder: If ``True`` and ``shoulder`` is not ``None``, shoulder
                area can be excluded from logical total area reporting.

        Returns:
            Dictionary with logical/component indices and initializer slices.
        """
        if high <= low:
            raise ValueError(
                f"Peak `{name}` has invalid bounds: low={low}, high={high}"
            )
        if shoulder not in (None, "left", "right"):
            raise ValueError(
                f"Invalid shoulder `{shoulder}`. Use None, 'left', or 'right'."
            )
        if any(defn.name == name for defn in self.peak_definitions):
            raise ValueError(f"Peak name `{name}` already exists.")

        if shoulder is None and exclude_shoulder:
            # Flag is only meaningful when a shoulder component exists.
            exclude_shoulder = False

        logical_index = int(len(self.peak_definitions))
        logical_definition = PeakDefinition(
            name=name,
            low=float(low),
            high=float(high),
            shoulder=shoulder,
            exclude_shoulder=bool(exclude_shoulder),
        )
        self.peak_definitions.append(logical_definition)
        self.peak_windows.append(PeakWindow(x1=float(low), x2=float(high), centers=[]))

        if shoulder is None:
            component_specs: list[dict[str, Any]] = [
                {
                    "component_name": name,
                    "role": "main",
                    "shoulder_side": None,
                    "include_in_total_area": True,
                }
            ]
        else:
            component_specs = [
                {
                    "component_name": name,
                    "role": "main",
                    "shoulder_side": shoulder,
                    "include_in_total_area": True,
                },
                {
                    "component_name": f"{name}_shoulder",
                    "role": "shoulder",
                    "shoulder_side": shoulder,
                    "include_in_total_area": not bool(exclude_shoulder),
                },
            ]

        component_names = [str(spec["component_name"]) for spec in component_specs]
        for component_name in component_names:
            if component_name in self.peak_names:
                raise ValueError(
                    f"Component name `{component_name}` already exists. "
                    "Choose a different logical peak name."
                )

        num_new_components = len(component_specs)
        low_vals = [float(low)] * num_new_components
        high_vals = [float(high)] * num_new_components
        self._append_peak_bounds(low_vals, high_vals)

        component_start_index = int(self.K)
        self.peak_names.extend(component_names)
        for spec in component_specs:
            is_shoulder = str(spec["role"]) == "shoulder"
            shoulder_side = spec["shoulder_side"]
            include_in_total = bool(spec["include_in_total_area"])
            self.component_to_logical_index.append(logical_index)
            self.component_is_shoulder.append(is_shoulder)
            self.component_shoulder_side.append(shoulder_side)
            self.component_include_in_total_area.append(include_in_total)

        self.K = int(self.mu_lo.shape[-1])
        self.mu_init, self.sigma_init, self.A_init = (
            self._compute_all_peak_initializers()
        )
        self.baseline_anchor_mask = self._compute_baseline_anchor_mask(
            slope_quantile=self.baseline_slope_quantile,
            curvature_quantile=self.baseline_curvature_quantile,
        )

        # Update global peak mask over the full [S, C, N] data cube.
        peak_region = (self.time >= float(low)) & (self.time <= float(high))
        self.peak_mask = jnp.asarray(self.peak_mask | peak_region, dtype=bool)

        component_end_index = component_start_index + num_new_components
        mu0_slice = self.mu_init[:, component_start_index:component_end_index]
        sigma0_slice = self.sigma_init[:, component_start_index:component_end_index]
        area0_slice = self.A_init[:, component_start_index:component_end_index]

        if num_new_components == 1:
            mu0_out: jnp.ndarray = mu0_slice[:, 0]
            sigma0_out: jnp.ndarray = sigma0_slice[:, 0]
            area0_out: jnp.ndarray = area0_slice[:, 0]
        else:
            mu0_out = mu0_slice
            sigma0_out = sigma0_slice
            area0_out = area0_slice

        return {
            "name": name,
            "logical_index": logical_index,
            "component_indices": list(
                range(component_start_index, component_end_index)
            ),
            "component_names": component_names,
            "shoulder": shoulder,
            "exclude_shoulder": bool(exclude_shoulder),
            "low": low,
            "high": high,
            "mu0": mu0_out,
            "sigma0": sigma0_out,
            "A0": area0_out,
        }

    def _build_model_inputs(self, y: Optional[jnp.ndarray]) -> dict[str, Any]:
        """Compile and validate inputs for the mixed peak `model`.

        This bridges user-facing logical peak definitions to component-level
        metadata used by the NumPyro model.
        """
        if self.K <= 0:
            raise RuntimeError("No model components defined. Add peaks before fitting.")
        if not self.peak_definitions:
            raise RuntimeError(
                "No logical peak definitions found. Use `add_peak(...)` to configure the model."
            )

        if len(self.component_to_logical_index) != self.K:
            raise RuntimeError("component_to_logical_index is inconsistent with K.")
        if len(self.component_is_shoulder) != self.K:
            raise RuntimeError("component_is_shoulder is inconsistent with K.")
        if len(self.component_shoulder_side) != self.K:
            raise RuntimeError("component_shoulder_side is inconsistent with K.")
        if len(self.component_include_in_total_area) != self.K:
            raise RuntimeError(
                "component_include_in_total_area is inconsistent with K."
            )

        x2d = self._as_2d(self.x)
        y2d = None if y is None else self._as_2d(y)

        peak_mask2d = self._as_2d(self.peak_mask).astype(bool)
        peak_mask_arg: Optional[jnp.ndarray]
        if peak_mask2d.shape != x2d.shape:
            peak_mask_arg = None
        else:
            peak_mask_arg = peak_mask2d if bool(jnp.any(peak_mask2d)) else None

        logical_mu_lo: list[float] = []
        logical_mu_hi: list[float] = []
        logical_main_component_index: list[int] = []
        logical_shoulder_component_index: list[int] = []
        logical_shoulder_side: list[int] = []

        for logical_index, definition in enumerate(self.peak_definitions):
            component_indices = [
                component_index
                for component_index, mapped_logical_index in enumerate(
                    self.component_to_logical_index
                )
                if int(mapped_logical_index) == logical_index
            ]

            if definition.shoulder is None:
                if len(component_indices) != 1:
                    raise RuntimeError(
                        f"Logical peak `{definition.name}` expected 1 component, got {len(component_indices)}."
                    )
                main_component_index = component_indices[0]
                shoulder_component_index = -1
                shoulder_side_code = 0
            else:
                if len(component_indices) != 2:
                    raise RuntimeError(
                        f"Logical peak `{definition.name}` expected 2 components, got {len(component_indices)}."
                    )
                shoulder_candidates = [
                    idx
                    for idx in component_indices
                    if bool(self.component_is_shoulder[idx])
                ]
                main_candidates = [
                    idx
                    for idx in component_indices
                    if not bool(self.component_is_shoulder[idx])
                ]
                if len(main_candidates) != 1 or len(shoulder_candidates) != 1:
                    raise RuntimeError(
                        f"Logical peak `{definition.name}` has invalid main/shoulder mapping."
                    )
                main_component_index = main_candidates[0]
                shoulder_component_index = shoulder_candidates[0]

                side = self.component_shoulder_side[shoulder_component_index]
                if side is None:
                    side = definition.shoulder
                if side not in ("left", "right"):
                    raise RuntimeError(
                        f"Logical peak `{definition.name}` has invalid shoulder side metadata: {side}."
                    )
                shoulder_side_code = -1 if side == "left" else 1

            logical_mu_lo.append(float(definition.low))
            logical_mu_hi.append(float(definition.high))
            logical_main_component_index.append(int(main_component_index))
            logical_shoulder_component_index.append(int(shoulder_component_index))
            logical_shoulder_side.append(int(shoulder_side_code))

        return {
            "x": x2d,
            "y": y2d,
            "mu_lo": self.mu_lo,
            "mu_hi": self.mu_hi,
            "mu_init": self.mu_init,
            "sigma_init": self.sigma_init,
            "A_init": self.A_init,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "peak_mask": peak_mask_arg,
            "alpha_prior_sd": float(self.alpha_prior_sd),
            "logical_mu_lo": jnp.asarray(logical_mu_lo, dtype=jnp.float32),
            "logical_mu_hi": jnp.asarray(logical_mu_hi, dtype=jnp.float32),
            "logical_main_component_index": jnp.asarray(
                logical_main_component_index, dtype=jnp.int32
            ),
            "logical_shoulder_component_index": jnp.asarray(
                logical_shoulder_component_index, dtype=jnp.int32
            ),
            "logical_shoulder_side": jnp.asarray(
                logical_shoulder_side, dtype=jnp.int32
            ),
            "component_to_logical_index": jnp.asarray(
                self.component_to_logical_index, dtype=jnp.int32
            ),
            "component_include_in_total_area": jnp.asarray(
                self.component_include_in_total_area, dtype=bool
            ),
        }

    def set_baseline_anchor_quantiles(
        self, slope_quantile: float = 0.10, curvature_quantile: float = 0.25
    ) -> None:
        """Update baseline-anchor quantiles and recompute the model baseline mask."""
        self.baseline_slope_quantile = float(slope_quantile)
        self.baseline_curvature_quantile = float(curvature_quantile)
        self.baseline_anchor_mask = self._compute_baseline_anchor_mask(
            slope_quantile=self.baseline_slope_quantile,
            curvature_quantile=self.baseline_curvature_quantile,
        )

    def _next_figure_path(self, stem: str) -> Path:
        count = self._figure_counts.get(stem, 0) + 1
        self._figure_counts[stem] = count
        return self.figure_dir / f"{stem}_{count:03d}.png"

    def _save_and_close_current_figure(
        self,
        stem: str,
        dpi: int = 150,
        filename: Optional[str] = None,
    ) -> Path:
        if filename is None:
            out_path = self._next_figure_path(stem)
        else:
            out_path = self.figure_dir / Path(filename).name

        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(plt.gcf())
        console.print(f"[dim]Saved figure:[/dim] {out_path}")
        return out_path

    def _print_initialization_summary(self) -> None:
        """Print a rich-styled summary of initialization parameters."""
        # Create main panel
        title = Text("🧬 ChromFitter Initialization Summary", style="bold cyan")

        # Data summary table
        data_table = Table(
            title="📊 Data Summary", show_header=True, header_style="bold magenta"
        )
        data_table.add_column("Parameter", style="cyan", no_wrap=True)
        data_table.add_column("Value", style="green")
        data_table.add_column("Description", style="dim")

        data_table.add_row(
            "Data points (N)", f"{self.time.shape[-1]:,}", "Points per spectrum"
        )
        data_table.add_row(
            "Batch shape",
            f"{self.time.shape if self.time.shape else '(single)'}",
            "Leading dimensions (evolution axes)",
        )
        data_table.add_row(
            "Components (K)", f"{len(self.peak_windows)}", "Number of peak windows"
        )
        data_table.add_row(
            "Time range",
            f"{float(self.time.min()):.3f} - {float(self.time.max()):.3f}",
            "Retention time span",
        )
        data_table.add_row(
            "Signal range",
            f"{float(jnp.nanmin(self.signal)):.3f} - {float(jnp.nanmax(self.signal)):.3f}",
            "Observed signal span",
        )
        data_table.add_row(
            "Sampling Δx",
            f"{float(jnp.median(jnp.diff(self.time))):.6f}",
            "Median time step",
        )

    def fit(
        self,
        num_warmup: int = 100,
        num_samples: int = 100,
        num_chains: int = num_devices,
        seed: int = 42,
        progress_bar: bool = True,
    ) -> "ChromFitter":
        """Run MCMC sampling with parallel chains.

        Args:
            num_warmup: Number of warmup iterations
            num_samples: Number of samples per chain
            num_chains: Number of parallel chains
            seed: Random seed
            progress_bar: Whether to show progress bar
            dense_mass: Whether to use dense mass matrix

        Returns:
            self for method chaining
        """
        model_inputs = self._build_model_inputs(y=self.y)

        self.mcmc = MCMC(
            NUTS(model, init_strategy=init_to_median),
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=progress_bar,
            chain_method="parallel" if num_chains > 1 else "sequential",
        )

        rng_key = jax.random.PRNGKey(seed)
        self.mcmc.run(rng_key, **model_inputs)

        self.samples = self.mcmc.get_samples()
        idata_raw = az.from_numpyro(self.mcmc)  # unchanged draws

        # --- new: relabel so every chain uses the same component order
        self.idata = relabel_by_sort(idata_raw, key="mu")

        self.samples = {
            v: self.idata.posterior[v].values.reshape(
                -1, *self.idata.posterior[v].shape[2:]
            )
            for v in self.idata.posterior.data_vars
        }

        return self

    def predict(self, x: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Predict mean signal from posterior samples.

        Args:
            x: Prediction points (uses self.x if None)

        Returns:
            Posterior mean predictions, shape [num_samples, N]
        """
        if self.samples is None:
            raise RuntimeError("Must call fit() before predict()")

        x_pred = self.x if x is None else jnp.asarray(x, dtype=jnp.float32)
        return predict_mean(x_pred, self.samples)

    def summary(self, var_names: Optional[list[str]] = None, round_to: int = 3) -> Any:
        """Generate summary statistics for posterior.

        Args:
            var_names: Variables to include (None for all, excluding ``mu_y``)
            round_to: Decimal places to round to

        Returns:
            Styled ArviZ summary DataFrame
        """
        if self.idata is None:
            raise RuntimeError("Must call fit() before summary()")

        posterior_vars = list(self.idata.posterior.data_vars)
        if var_names is None:
            summary_vars = [v for v in posterior_vars if v != "mu_y"]
        else:
            summary_vars = [v for v in var_names if v in posterior_vars and v != "mu_y"]

        if not summary_vars:
            raise ValueError(
                "No valid variables available for summary after exclusions."
            )

        df = az.summary(
            self.idata,
            var_names=summary_vars,
            round_to=round_to,
        )

        def color_rhat(val: float) -> str:
            color = "red" if val > 1.05 else "green"
            return f"color: {color}"

        styled = (
            df.style.applymap(color_rhat, subset=["r_hat"]).format(
                precision=round_to
            )  # ✅ apply same rounding
        )
        return styled

    def plot_trace(
        self, var_names: Optional[list[str]] = None, compact: bool = True
    ) -> None:
        """Plot posterior traces using vanilla ArviZ behavior."""
        if self.idata is None:
            raise RuntimeError("Must call fit() before plot_trace()")

        posterior = self.idata.posterior
        if var_names is None:
            var_names_use = list(posterior.data_vars)
        else:
            var_names_use = [v for v in var_names if v in posterior.data_vars]
            missing = [v for v in var_names if v not in posterior.data_vars]
            if missing:
                console.print(
                    f"[yellow]Skipping missing trace vars:[/yellow] {', '.join(missing)}"
                )

        if not var_names_use:
            raise ValueError("No valid posterior variables available for trace plot.")

        # Guard against fully non-finite variables that break KDE internals.
        finite_vars: list[str] = []
        dropped_nonfinite: list[str] = []
        for v in var_names_use:
            arr = jnp.asarray(posterior[v].values)
            if bool(jnp.any(jnp.isfinite(arr))):
                finite_vars.append(v)
            else:
                dropped_nonfinite.append(v)

        if dropped_nonfinite:
            console.print(
                "[yellow]Dropping non-finite trace vars:[/yellow] "
                + ", ".join(dropped_nonfinite)
            )

        if not finite_vars:
            raise ValueError("All requested trace variables are non-finite.")

        # Guard against near-degenerate coordinates inside a variable (e.g. one
        # component fixed/constant), which can crash ArviZ KDE internals.
        stable_vars: list[str] = []
        dropped_degenerate: list[str] = []
        for v in finite_vars:
            arr = jnp.asarray(posterior[v].values)
            if arr.ndim < 3:
                stable_vars.append(v)
                continue

            flat = arr.reshape(arr.shape[0] * arr.shape[1], -1)
            # Keep variable only if every coordinate has finite spread.
            col_finite = jnp.all(jnp.isfinite(flat), axis=0)
            col_std = jnp.std(jnp.where(jnp.isfinite(flat), flat, 0.0), axis=0)
            col_ok = col_finite & (col_std > 1e-12)
            if bool(jnp.all(col_ok)):
                stable_vars.append(v)
            else:
                dropped_degenerate.append(v)

        if dropped_degenerate:
            console.print(
                "[yellow]Dropping degenerate trace vars (constant/non-finite coords):[/yellow] "
                + ", ".join(dropped_degenerate)
            )

        if not stable_vars:
            raise ValueError(
                "All requested trace variables are degenerate for KDE plotting."
            )

        try:
            az.plot_trace(self.idata, var_names=stable_vars, compact=compact)  # type: ignore
        except Exception as exc:
            # ArviZ KDE can fail on near-degenerate variables; fallback to rank view.
            console.print(
                "[yellow]Trace KDE failed; retrying with rank_vlines[/yellow] "
                f"({type(exc).__name__}: {exc})"
            )
            plt.close("all")
            try:
                az.plot_trace(
                    self.idata,
                    var_names=stable_vars,
                    compact=compact,
                    kind="rank_vlines",
                )  # type: ignore
            except Exception as exc2:
                console.print(
                    "[yellow]rank_vlines failed; falling back to az.plot_rank[/yellow] "
                    f"({type(exc2).__name__}: {exc2})"
                )
                plt.close("all")
                az.plot_rank(self.idata, var_names=stable_vars)  # type: ignore
        plt.tight_layout()
        self._save_and_close_current_figure("trace")

    def plot_rank(self, var_names: Optional[list[str]] = None) -> None:
        """Plot rank statistics for convergence diagnostics."""
        if self.idata is None:
            raise RuntimeError("Must call fit() before plot_rank()")
        az.plot_rank(self.idata, var_names=var_names)  # type: ignore
        plt.tight_layout()
        self._save_and_close_current_figure("rank")

    def plot_autocorr(self, var_names: Optional[list[str]] = None) -> None:
        """Plot autocorrelation for selected variables."""
        if self.idata is None:
            raise RuntimeError("Must call fit() before plot_autocorr()")
        az.plot_autocorr(self.idata, var_names=var_names)  # type: ignore
        plt.tight_layout()
        self._save_and_close_current_figure("autocorr")

    def plot_pair(
        self,
        var_names: Optional[list[str]] = None,
        kind: str = "kde",
        save_path: Optional[str] = None,  # ⇦ new
        dpi: int = 100,
    ) -> None:
        """
        Corner plot of the requested variables.

        Args
        ----
        var_names   : list of variable names to plot (None = all)
        kind        : "scatter", "kde", "hexbin", …
        save_path   : filesystem path — supports .png, .pdf, .svg, etc.
                    • .png/.tiff use `dpi`
                    • .pdf/.svg are vector-based (best for unlimited zoom)
        dpi         : resolution for raster formats
        """
        if self.idata is None:
            raise RuntimeError("Must call fit() before plot_pair()")

        # lift ArviZ subplot cap
        with az.rc_context(rc={"plot.max_subplots": None}):
            az.plot_pair(
                self.idata,
                var_names=var_names,
                kind=kind,
                marginals=True,
                divergences=True,
            )

            plt.tight_layout()

            if save_path is not None:
                self._save_and_close_current_figure("pair", dpi=dpi, filename=save_path)
            else:
                self._save_and_close_current_figure("pair", dpi=dpi)

    def plot_fit(
        self,
        figsize: tuple[int, int] = (10, 4),
        fill_alpha: float = 0.30,
        linewidth: float = 1.6,
        data_color: str = "C0",
        dense_points: int = 300,
    ) -> None:
        """Plot observed data with component and logical-peak posterior curves.

        Rendering order per spectrum:
        1) raw data points
        2) individual component curves in gray with 95% HDI bands around the curve
        3) logical-peak total curves in tab10 colors with 95% HDI bands
        """
        if self.samples is None:
            raise RuntimeError("Must call fit() before plot_fit()")

        x2d = self._as_2d(self.x)
        y2d = self._as_2d(self.y)
        n_spectra = int(x2d.shape[0])

        A = jnp.asarray(self.samples["A"], dtype=jnp.float32)
        mu = jnp.asarray(self.samples["mu"], dtype=jnp.float32)
        sigma = jnp.asarray(self.samples["sigma"], dtype=jnp.float32)
        alpha_draws = jnp.asarray(self.samples["alpha"], dtype=jnp.float32)
        b0 = (
            jnp.asarray(self.samples["b0"], dtype=jnp.float32)
            if "b0" in self.samples
            else None
        )
        b1 = (
            jnp.asarray(self.samples["b1"], dtype=jnp.float32)
            if "b1" in self.samples
            else None
        )

        if A.ndim == 2:
            A = A[:, None, :]
        if mu.ndim == 2:
            mu = mu[:, None, :]
        if sigma.ndim == 1:
            sigma = sigma[:, None, None]
        elif sigma.ndim == 2:
            sigma = sigma[:, None, :]
        if alpha_draws.ndim == 1:
            alpha_draws = alpha_draws[:, None, None]
        elif alpha_draws.ndim == 2:
            alpha_draws = alpha_draws[:, None, :]

        if A.shape[1] == 1 and n_spectra > 1:
            A = jnp.broadcast_to(A, (A.shape[0], n_spectra, A.shape[-1]))
        if mu.shape[1] == 1 and n_spectra > 1:
            mu = jnp.broadcast_to(mu, (mu.shape[0], n_spectra, mu.shape[-1]))
        if sigma.shape[1] == 1 and n_spectra > 1:
            sigma = jnp.broadcast_to(
                sigma, (sigma.shape[0], n_spectra, sigma.shape[-1])
            )
        if alpha_draws.shape[1] == 1 and n_spectra > 1:
            alpha_draws = jnp.broadcast_to(
                alpha_draws, (alpha_draws.shape[0], n_spectra, alpha_draws.shape[-1])
            )

        if A.shape[1] != n_spectra or mu.shape[1] != n_spectra:
            raise ValueError(
                f"Posterior shapes incompatible with spectra count S={n_spectra}: "
                f"A={A.shape}, mu={mu.shape}"
            )
        if alpha_draws.shape[1] != n_spectra:
            raise ValueError(
                f"Posterior alpha shape incompatible with spectra count S={n_spectra}: "
                f"alpha={alpha_draws.shape}"
            )
        if sigma.shape[1] != n_spectra:
            raise ValueError(
                f"Posterior sigma shape incompatible with spectra count S={n_spectra}: "
                f"sigma={sigma.shape}"
            )

        cmap = plt.get_cmap("tab10")
        _, axes = plt.subplots(
            n_spectra,
            1,
            figsize=(figsize[0], figsize[1] * n_spectra),
            squeeze=False,
        )
        axes_flat = axes.reshape(-1)
        q_low = 0.025
        q_high = 0.975

        if len(self.component_to_logical_index) == self.K:
            component_to_logical = jnp.asarray(
                self.component_to_logical_index, dtype=jnp.int32
            )
            n_logical = int(jnp.max(component_to_logical)) + 1 if self.K > 0 else 0
        else:
            component_to_logical = jnp.arange(self.K, dtype=jnp.int32)
            n_logical = self.K

        logical_names = []
        if len(self.peak_definitions) == n_logical:
            logical_names = [d.name for d in self.peak_definitions]
        else:
            logical_names = [f"logical_{idx + 1}" for idx in range(n_logical)]

        for s_idx in range(n_spectra):
            ax = axes_flat[s_idx]
            x_s = x2d[s_idx]
            y_s = y2d[s_idx]
            finite_xy = jnp.isfinite(x_s) & jnp.isfinite(y_s)
            if not bool(jnp.any(finite_xy)):
                continue

            x_fin = x_s[finite_xy]
            y_fin = y_s[finite_xy]
            x_min = float(jnp.min(x_fin))
            x_max = float(jnp.max(x_fin))
            ax.plot(x_fin, y_fin, ".", ms=3, alpha=0.6, color=data_color)

            n_dense = max(int(dense_points), 200)
            x_dense = jnp.linspace(x_min, x_max, n_dense, dtype=jnp.float32)

            mu_s = mu[:, s_idx, :]  # [draw, K]
            sigma_s = jnp.maximum(
                sigma[:, s_idx, :], jnp.array(1e-6, dtype=jnp.float32)
            )
            alpha_s = alpha_draws[:, s_idx, :]  # [draw, K]
            A_s = A[:, s_idx, :]  # [draw, K]

            log_pdf = log_skew_normal_pdf(x_dense, mu_s, sigma_s, alpha_s)
            component_draws = A_s[:, :, None] * jnp.exp(log_pdf)  # [draw, K, N_dense]

            component_label_added = False
            for k in range(self.K):
                comp_k = component_draws[:, k, :]
                comp_med = jnp.median(comp_k, axis=0)
                comp_low = jnp.quantile(comp_k, q_low, axis=0)
                comp_high = jnp.quantile(comp_k, q_high, axis=0)

                ax.fill_between(
                    x_dense,
                    comp_low,
                    comp_high,
                    color="0.65",
                    alpha=0.22 * fill_alpha / max(fill_alpha, 1e-6),
                    label="Components (95% HDI)" if not component_label_added else None,
                    zorder=1,
                )
                ax.plot(
                    x_dense,
                    comp_med,
                    color="0.35",
                    lw=max(0.9, 0.75 * linewidth),
                    alpha=0.9,
                    zorder=2,
                )
                component_label_added = True

            for logical_index in range(n_logical):
                component_indices = jnp.where(component_to_logical == logical_index)[0]
                if int(component_indices.size) == 0:
                    continue
                logical_draws = jnp.sum(
                    component_draws[:, component_indices, :], axis=1
                )
                logical_median = jnp.median(logical_draws, axis=0)
                logical_low = jnp.quantile(logical_draws, q_low, axis=0)
                logical_high = jnp.quantile(logical_draws, q_high, axis=0)
                logical_color = cmap(logical_index % 10)

                ax.fill_between(
                    x_dense,
                    logical_low,
                    logical_high,
                    color=logical_color,
                    alpha=fill_alpha,
                    zorder=3,
                )
                ax.plot(
                    x_dense,
                    logical_median,
                    color=logical_color,
                    lw=linewidth,
                    label=logical_names[logical_index],
                    zorder=4,
                )

            ax.set_ylabel("Signal")
            ax.set_title(f"Spectrum {s_idx}")
            handles, labels = ax.get_legend_handles_labels()
            uniq = dict(zip(labels, handles))
            if uniq:
                ax.legend(
                    list(uniq.values()),
                    list(uniq.keys()),
                    loc="best",
                    frameon=False,
                    fontsize=8,
                )
            ax.grid(True, alpha=0.20)

        axes_flat[-1].set_xlabel("Retention time")
        plt.tight_layout()
        self._save_and_close_current_figure("fit")

    def plot_component_windows_posterior(
        self,
        nsigma: float = 4.0,
        figsize: Optional[tuple[int, int]] = None,
        scatter_size: float = 7.0,
        fill_alpha: float = 0.30,
        include_baseline: bool = False,
        dense_points: int = 300,
        hdi_prob: float = 0.95,
        component_median_color: str = "0.45",
    ) -> None:
        """Plot per-(spectrum, component) posterior curves in a grid."""
        if self.samples is None:
            raise RuntimeError(
                "Must call fit() before plot_component_windows_posterior()"
            )
        if self.K <= 0:
            raise RuntimeError("No components defined (K=0).")
        if not (0.0 < hdi_prob < 1.0):
            raise ValueError("hdi_prob must be in (0, 1).")

        x2d = self._as_2d(self.x)
        y2d = self._as_2d(self.y)
        S = int(x2d.shape[0])
        K = int(self.K)

        A = jnp.asarray(self.samples["A"], dtype=jnp.float32)
        mu = jnp.asarray(self.samples["mu"], dtype=jnp.float32)
        sigma = jnp.asarray(self.samples["sigma"], dtype=jnp.float32)
        alpha = jnp.asarray(self.samples["alpha"], dtype=jnp.float32)
        b0 = (
            jnp.asarray(self.samples["b0"], dtype=jnp.float32)
            if "b0" in self.samples
            else None
        )
        b1 = (
            jnp.asarray(self.samples["b1"], dtype=jnp.float32)
            if "b1" in self.samples
            else None
        )

        if A.ndim == 2:
            A = A[:, None, :]
        if mu.ndim == 2:
            mu = mu[:, None, :]
        if sigma.ndim == 1:
            sigma = sigma[:, None, None]
        elif sigma.ndim == 2:
            sigma = sigma[:, None, :]
        if alpha.ndim == 1:
            alpha = alpha[:, None, None]
        elif alpha.ndim == 2:
            alpha = alpha[:, None, :]

        if A.shape[1] == 1 and S > 1:
            A = jnp.broadcast_to(A, (A.shape[0], S, A.shape[-1]))
        if mu.shape[1] == 1 and S > 1:
            mu = jnp.broadcast_to(mu, (mu.shape[0], S, mu.shape[-1]))
        if sigma.shape[1] == 1 and S > 1:
            sigma = jnp.broadcast_to(sigma, (sigma.shape[0], S, sigma.shape[-1]))
        if alpha.shape[1] == 1 and S > 1:
            alpha = jnp.broadcast_to(alpha, (alpha.shape[0], S, alpha.shape[-1]))

        if A.shape[1] != S or mu.shape[1] != S:
            raise ValueError(
                f"Posterior shapes incompatible with spectra count S={S}: "
                f"A={A.shape}, mu={mu.shape}"
            )
        if A.shape[-1] != self.K or mu.shape[-1] != self.K:
            raise ValueError(
                f"Posterior component shape mismatch K={self.K}: "
                f"A={A.shape}, mu={mu.shape}"
            )
        if sigma.shape[-1] != self.K or alpha.shape[-1] != self.K:
            raise ValueError(
                f"Posterior component shape mismatch K={self.K}: "
                f"sigma={sigma.shape}, alpha={alpha.shape}"
            )
        if sigma.shape[1] != S:
            raise ValueError(
                f"Posterior sigma shape incompatible with spectra count S={S}: "
                f"sigma={sigma.shape}"
            )
        if alpha.shape[1] != S:
            raise ValueError(
                f"Posterior alpha shape incompatible with spectra count S={S}: "
                f"alpha={alpha.shape}"
            )

        if figsize is None:
            figsize = (max(3.8 * K, 6.0), max(2.4 * S, 3.0))

        fig, axes = plt.subplots(
            S,
            K,
            figsize=figsize,
            sharex="col",
            squeeze=False,
        )
        cmap = plt.get_cmap("tab10")
        q_low = 0.5 * (1.0 - hdi_prob)
        q_high = 1.0 - q_low

        for k in range(K):
            color = cmap(k % 10)
            peak_name = (
                self.peak_names[k] if k < len(self.peak_names) else f"peak_{k + 1}"
            )
            for s in range(S):
                ax = axes[s, k]
                x_s = x2d[s]
                y_s = y2d[s]

                finite_xy = jnp.isfinite(x_s) & jnp.isfinite(y_s)
                if not bool(jnp.any(finite_xy)):
                    continue

                x_fin = x_s[finite_xy]
                x_min = float(jnp.min(x_fin))
                x_max = float(jnp.max(x_fin))

                mu_sk_draw = mu[:, s, k]
                sigma_k_draw = jnp.maximum(
                    sigma[:, s, k], jnp.array(1e-6, dtype=jnp.float32)
                )
                mu_mean = float(jnp.mean(mu_sk_draw))
                sigma_mean = float(jnp.mean(sigma_k_draw))
                x0 = max(mu_mean - nsigma * sigma_mean, x_min)
                x1 = min(mu_mean + nsigma * sigma_mean, x_max)
                if x1 <= x0:
                    x0, x1 = x_min, x_max

                mask = finite_xy & (x_s >= x0) & (x_s <= x1)
                x_w = x_s[mask]
                y_w = y_s[mask]

                n_win = int(jnp.sum(mask))
                n_dense = max(int(dense_points), max(40, 5 * n_win))
                x_dense = jnp.linspace(x0, x1, n_dense, dtype=jnp.float32)

                n_draws = int(A.shape[0])
                baseline_draws = jnp.zeros((n_draws, n_dense), dtype=jnp.float32)

                if include_baseline and b0 is not None and b1 is not None:
                    if b0.ndim == 1:
                        b0_s = b0[:, None]
                        b1_s = b1[:, None]
                    else:
                        b0_flat = b0.reshape(b0.shape[0], -1)
                        b1_flat = b1.reshape(b1.shape[0], -1)
                        s_idx = min(s, b0_flat.shape[1] - 1)
                        b0_s = b0_flat[:, s_idx][:, None]
                        b1_s = b1_flat[:, s_idx][:, None]
                    baseline_draws = b0_s + b1_s * x_dense[None, :]

                alpha_k_draw = alpha[:, s, k]
                A_sk_draw = A[:, s, k]
                mu_draw = mu_sk_draw[:, None]
                sigma_draw = sigma_k_draw[:, None]
                alpha_draw = alpha_k_draw[:, None]
                log_pdf = log_skew_normal_pdf(x_dense, mu_draw, sigma_draw, alpha_draw)
                comp_draws = A_sk_draw[:, None] * jnp.exp(log_pdf[:, 0, :])
                comp_med = jnp.median(comp_draws, axis=0)

                y_draws = baseline_draws + comp_draws
                y_med = jnp.median(y_draws, axis=0)
                y_low = jnp.quantile(y_draws, q_low, axis=0)
                y_high = jnp.quantile(y_draws, q_high, axis=0)

                area_mean = float(jnp.mean(A_sk_draw))
                area_sd = float(jnp.std(A_sk_draw))
                area_pct = 100.0 * area_sd / max(abs(area_mean), 1e-8)
                mu_sd = float(jnp.std(mu_sk_draw))
                mu_pct = 100.0 * mu_sd / max(abs(mu_mean), 1e-8)
                sigma_sd = float(jnp.std(sigma_k_draw))
                sigma_pct = 100.0 * sigma_sd / max(abs(sigma_mean), 1e-8)
                alpha_mean = float(jnp.mean(alpha_k_draw))
                alpha_sd = float(jnp.std(alpha_k_draw))
                alpha_pct = 100.0 * alpha_sd / max(abs(alpha_mean), 1e-8)
                area_label = (
                    f"Area: {area_mean:.2f} ± {area_pct:.1f}% (1σ),\n"
                    f"mu: {mu_mean:.4f} ± {mu_pct:.1f}% (1σ),\n"
                    f"sigma: {sigma_mean:.4f} ± {sigma_pct:.1f}% (1σ),\n"
                    f"alpha: {alpha_mean:.3f} ± {alpha_pct:.1f}% (1σ),\n"
                    f"{int(round(100 * hdi_prob))}% HDI"
                )

                ax.fill_between(
                    x_dense,
                    y_low,
                    y_high,
                    color=color,
                    alpha=fill_alpha,
                    label=area_label,
                    zorder=1,
                )
                ax.plot(
                    x_dense,
                    comp_med,
                    color=component_median_color,
                    linewidth=1.1,
                    alpha=0.95,
                    zorder=2,
                )
                ax.scatter(
                    x_w,
                    y_w,
                    s=scatter_size,
                    color="0.55",
                    alpha=0.70,
                    linewidths=0.0,
                    zorder=2,
                )
                ax.plot(
                    x_dense,
                    y_med,
                    color=color,
                    linewidth=1.9,
                    zorder=4,
                )

                if s == 0:
                    ax.set_title(str(peak_name))
                if k == 0:
                    ax.set_ylabel(f"Spectrum {s}")
                if s == S - 1:
                    ax.set_xlabel("Retention time")
                ax.grid(True, alpha=0.20)
                ax.legend(loc="best", frameon=False, fontsize=7)

        plt.tight_layout()
        self._save_and_close_current_figure("component_windows_posterior")

    def plot_corr(
        self,
        var_names: Optional[list[str]] = None,
        figsize: tuple[int, int] = (12, 10),
        cmap: str = "coolwarm",
        vmin: float = -1.0,
        vmax: float = 1.0,
    ) -> None:
        """
        Show a parameter–parameter correlation matrix.

        Parameters
        ----------
        var_names : list[str] | None
            Variables to include.  None = all component-indexed variables
            (A, mu, sigma, alpha, …).
        figsize   : (w, h)
            Figure size in inches.
        cmap      : str
            Matplotlib colormap.
        vmin/vmax : float
            Color scale limits.
        """
        if self.idata is None:
            raise RuntimeError("Must call fit() before plot_corr()")

        # -------- pick variables --------
        posterior = self.idata.posterior
        if var_names is None:
            var_names = [
                v
                for v in posterior.data_vars
                if posterior[v].ndim >= 3 and posterior[v].shape[-1] == self.K
            ]

        cols, labels = [], []
        for v in var_names:
            arr = posterior[v].values  # (chain, draw, K) or scalar per comp
            flat = arr.reshape(-1, *arr.shape[2:])  # collapse chains & draws
            if flat.ndim == 1:  # scalar per draw
                cols.append(flat)
                labels.append(v)
            else:  # one column per component
                for k in range(flat.shape[-1]):
                    cols.append(flat[..., k])
                    labels.append(f"{v}[{k}]")

        X = jnp.column_stack(cols)  # (samples, features)
        C = jnp.corrcoef(X, rowvar=False)

        # -------- plot --------
        plt.figure(figsize=figsize)
        im = plt.imshow(C, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(im, fraction=0.046)
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)
        plt.title("Posterior correlation matrix")
        plt.tight_layout()
        self._save_and_close_current_figure("corr")

    def get_peak_summaries(self) -> dict[str, dict[str, float]]:
        """Extract summary statistics for each peak.

        Returns:
            Dictionary with peak parameters (areas, centers, widths)
        """
        if self.samples is None:
            raise RuntimeError("Must call fit() before get_peak_summaries()")

        summaries = {}
        for k in range(self.K):
            area_k = self.samples["A"][..., k].reshape(-1)
            mu_k = self.samples["mu"][..., k].reshape(-1)
            sigma_k = self.samples["sigma"][..., k].reshape(-1)
            alpha_k = self.samples["alpha"][..., k].reshape(-1)
            summaries[f"peak_{k + 1}"] = {
                "area_mean": float(area_k.mean()),
                "area_std": float(area_k.std()),
                "mu_mean": float(mu_k.mean()),
                "mu_std": float(mu_k.std()),
                "sigma_mean": float(sigma_k.mean()),
                "sigma_std": float(sigma_k.std()),
                "alpha_mean": float(alpha_k.mean()),
                "alpha_std": float(alpha_k.std()),
            }

        return summaries

    def plot_prior_predictive_chromatograms(
        self,
        num_draws: int = 50,
        seed: int = 0,
        figsize: Optional[tuple[float, float]] = None,
        draw_color: str = "tab:blue",
        draw_alpha: float = 0.2,
        draw_linewidth: float = 1.0,
        data_color: str = "black",
        data_linewidth: float = 1.2,
    ) -> None:
        """Plot prior predictive draws against observed chromatograms.

        Creates a single-column subplot layout with one chromatogram per axis:
        1) plot prior predictive draws of ``mu_y``
        2) overlay observed data as line plot
        """
        from numpyro.infer import Predictive

        if self.K <= 0:
            raise RuntimeError("No components defined (K=0).")
        if num_draws <= 0:
            raise ValueError("num_draws must be > 0.")

        x2d = self._as_2d(self.x)
        y2d = self._as_2d(self.y)
        n_chromatograms = int(x2d.shape[0])

        model_inputs = self._build_model_inputs(y=None)
        predictive = Predictive(
            model,
            num_samples=int(num_draws),
            return_sites=("mu_y",),
        )
        prior_samples = predictive(jax.random.PRNGKey(seed), **model_inputs)

        prior_mu_y = jnp.asarray(prior_samples["mu_y"], dtype=jnp.float32)
        if prior_mu_y.ndim == 2:
            prior_mu_y = prior_mu_y[:, None, :]
        if prior_mu_y.shape[1] == 1 and n_chromatograms > 1:
            prior_mu_y = jnp.broadcast_to(
                prior_mu_y, (prior_mu_y.shape[0], n_chromatograms, prior_mu_y.shape[-1])
            )
        if prior_mu_y.shape[1] != n_chromatograms:
            raise RuntimeError(
                "Prior predictive shape mismatch: "
                f"mu_y has {prior_mu_y.shape[1]} chromatograms, expected {n_chromatograms}."
            )

        if figsize is None:
            figsize = (11.0, max(2.4 * n_chromatograms, 4.0))

        fig, axes = plt.subplots(
            n_chromatograms,
            1,
            figsize=figsize,
            sharex=True,
            squeeze=False,
        )
        axes_1d = axes[:, 0]

        draw_label_used = False
        data_label_used = False
        for chromatogram_index, ax in enumerate(axes_1d):
            x_vals = np.asarray(x2d[chromatogram_index], dtype=float)
            y_vals = np.asarray(y2d[chromatogram_index], dtype=float)

            for draw_index in range(prior_mu_y.shape[0]):
                label = "Prior predictive draws" if not draw_label_used else None
                ax.plot(
                    x_vals,
                    np.asarray(prior_mu_y[draw_index, chromatogram_index], dtype=float),
                    color=draw_color,
                    alpha=draw_alpha,
                    linewidth=draw_linewidth,
                    zorder=1,
                    label=label,
                )
                draw_label_used = True

            label = "Observed data" if not data_label_used else None
            ax.plot(
                x_vals,
                y_vals,
                color=data_color,
                linewidth=data_linewidth,
                zorder=2,
                label=label,
            )
            data_label_used = True

            ax.set_ylabel(f"Chrom {chromatogram_index}")
            ax.grid(True, alpha=0.20)

        axes_1d[-1].set_xlabel("Retention time")
        axes_1d[0].legend(loc="best", frameon=False)
        plt.tight_layout()
        self._save_and_close_current_figure("prior_predictive_chromatograms")

    def plot_prior_hairlines(
        self,
        num_draws: int = 30,
        seed: int = 0,
        figsize: Optional[tuple[int, int]] = None,
        alpha: float = 0.30,
        lw: float = 1.8,
        data_color: str = "C0",
        nsigma: float = 3.0,
        dense_points: int = 500,
        hdi_prob: float = 0.95,
        include_baseline: bool = False,
        scatter_size: float = 5.0,
    ) -> None:
        """Plot prior predictive component windows in component-grid layout.

        Layout matches `plot_component_windows_posterior`:
        - rows = spectra
        - cols = components
        - tab10 color per component column
        - per-cell prior median + HDI band over a dense x-grid
        - observed data in the same local window as gray scatter
        """
        from numpyro.infer import Predictive

        if self.K <= 0:
            raise RuntimeError("No components defined (K=0).")
        if not (0.0 < hdi_prob < 1.0):
            raise ValueError("hdi_prob must be in (0, 1).")

        # ===================================================================
        # 1) PRIOR PREDICTIVE SAMPLING
        # ===================================================================
        # Sample from prior by passing y=None (no conditioning on data)
        rng_key = jax.random.PRNGKey(seed)
        model_inputs = self._build_model_inputs(y=None)
        predictive = Predictive(
            model,
            num_samples=num_draws,
        )
        prior_samples = predictive(rng_key, **model_inputs)

        # ===================================================================
        # 2) ARVIZ INTEGRATION
        # ===================================================================
        # Convert to ArviZ format with proper structure
        # Predictive returns shape (num_samples, ...), ArviZ expects (chain, draw, ...)
        # We treat all samples as coming from a single chain
        prior_dict = {}
        for k, v in prior_samples.items():
            # Add chain dimension: (num_samples, ...) → (1, num_samples, ...)
            prior_dict[k] = jnp.expand_dims(v, axis=0)

        # Create/update InferenceData with prior_predictive group
        if self.idata is None:
            self.idata = az.from_dict(prior_predictive=prior_dict)
        else:
            # Add prior_predictive group to existing InferenceData
            prior_group = az.from_dict(prior_predictive=prior_dict).prior_predictive
            self.idata.add_groups(prior_predictive=prior_group)

        # ===================================================================
        # 3) PLOTTING
        # ===================================================================
        x2d = self._as_2d(self.x)
        y2d = self._as_2d(self.y)
        S = int(x2d.shape[0])
        K = int(self.K)

        A = jnp.asarray(prior_samples["A"], dtype=jnp.float32)
        mu = jnp.asarray(prior_samples["mu"], dtype=jnp.float32)
        sigma = jnp.asarray(prior_samples["sigma"], dtype=jnp.float32)
        alpha_draws = jnp.asarray(prior_samples["alpha"], dtype=jnp.float32)
        b0 = (
            jnp.asarray(prior_samples["b0"], dtype=jnp.float32)
            if "b0" in prior_samples
            else None
        )
        b1 = (
            jnp.asarray(prior_samples["b1"], dtype=jnp.float32)
            if "b1" in prior_samples
            else None
        )

        if A.ndim == 2:
            A = A[:, None, :]
        if mu.ndim == 2:
            mu = mu[:, None, :]
        if sigma.ndim == 1:
            sigma = sigma[:, None, None]
        elif sigma.ndim == 2:
            sigma = sigma[:, None, :]
        if alpha_draws.ndim == 1:
            alpha_draws = alpha_draws[:, None, None]
        elif alpha_draws.ndim == 2:
            alpha_draws = alpha_draws[:, None, :]

        if A.shape[1] == 1 and S > 1:
            A = jnp.broadcast_to(A, (A.shape[0], S, A.shape[-1]))
        if mu.shape[1] == 1 and S > 1:
            mu = jnp.broadcast_to(mu, (mu.shape[0], S, mu.shape[-1]))
        if sigma.shape[1] == 1 and S > 1:
            sigma = jnp.broadcast_to(sigma, (sigma.shape[0], S, sigma.shape[-1]))
        if alpha_draws.shape[1] == 1 and S > 1:
            alpha_draws = jnp.broadcast_to(
                alpha_draws, (alpha_draws.shape[0], S, alpha_draws.shape[-1])
            )

        if figsize is None:
            figsize = (max(4.2 * K, 8.0), max(2.4 * S, 4.0))

        q_low = 0.5 * (1.0 - hdi_prob)
        q_high = 1.0 - q_low
        cmap = plt.get_cmap("tab10")
        fig, axes = plt.subplots(
            S,
            K,
            figsize=figsize,
            sharex="col",
            squeeze=False,
        )

        for k in range(K):
            color = cmap(k % 10)
            peak_name = (
                self.peak_names[k] if k < len(self.peak_names) else f"peak_{k + 1}"
            )
            for s in range(S):
                ax = axes[s, k]
                x_s = x2d[s]
                y_s = y2d[s]
                finite_xy = jnp.isfinite(x_s) & jnp.isfinite(y_s)
                if not bool(jnp.any(finite_xy)):
                    continue

                x_fin = x_s[finite_xy]
                x_min = float(jnp.min(x_fin))
                x_max = float(jnp.max(x_fin))

                mu_sk_draw = mu[:, s, k]
                sigma_k_draw = jnp.maximum(
                    sigma[:, s, k], jnp.array(1e-6, dtype=jnp.float32)
                )
                mu_mean = float(jnp.mean(mu_sk_draw))
                sigma_mean = float(jnp.mean(sigma_k_draw))
                x0 = max(mu_mean - nsigma * sigma_mean, x_min)
                x1 = min(mu_mean + nsigma * sigma_mean, x_max)
                if x1 <= x0:
                    x0, x1 = x_min, x_max

                mask = finite_xy & (x_s >= x0) & (x_s <= x1)
                x_w = x_s[mask]
                y_w = y_s[mask]

                n_win = int(jnp.sum(mask))
                n_dense = max(int(dense_points), max(40, 5 * n_win))
                x_dense = jnp.linspace(x0, x1, n_dense, dtype=jnp.float32)

                A_sk = A[:, s, k]
                alpha_k = alpha_draws[:, s, k]
                mu_draw = mu_sk_draw[:, None]
                sigma_draw = sigma_k_draw[:, None]
                alpha_draw = alpha_k[:, None]
                log_pdf = log_skew_normal_pdf(x_dense, mu_draw, sigma_draw, alpha_draw)
                comp_draws = A_sk[:, None] * jnp.exp(log_pdf[:, 0, :])

                baseline_draws = jnp.zeros_like(comp_draws)
                if include_baseline and b0 is not None and b1 is not None:
                    if b0.ndim == 1:
                        b0_s = b0[:, None]
                        b1_s = b1[:, None]
                    else:
                        b0_flat = b0.reshape(b0.shape[0], -1)
                        b1_flat = b1.reshape(b1.shape[0], -1)
                        s_idx = min(s, b0_flat.shape[1] - 1)
                        b0_s = b0_flat[:, s_idx][:, None]
                        b1_s = b1_flat[:, s_idx][:, None]
                    baseline_draws = b0_s + b1_s * x_dense[None, :]

                y_draws = baseline_draws + comp_draws
                y_med = jnp.median(y_draws, axis=0)
                y_low = jnp.quantile(y_draws, q_low, axis=0)
                y_high = jnp.quantile(y_draws, q_high, axis=0)

                ax.fill_between(
                    x_dense,
                    y_low,
                    y_high,
                    color=color,
                    alpha=alpha,
                    zorder=1,
                )
                ax.scatter(
                    x_w,
                    y_w,
                    s=scatter_size,
                    color=data_color,
                    alpha=0.55,
                    linewidths=0.0,
                    zorder=2,
                )
                ax.plot(
                    x_dense,
                    y_med,
                    color=color,
                    lw=lw,
                    zorder=3,
                )

                if s == 0:
                    ax.set_title(str(peak_name))
                if k == 0:
                    ax.set_ylabel(f"Spectrum {s}")
                if s == S - 1:
                    ax.set_xlabel("Retention time")
                ax.grid(True, alpha=0.20)

        plt.tight_layout()
        self._save_and_close_current_figure("prior_hairlines")

        console.print(
            "\n[green]✓[/green] Prior predictive samples saved to "
            "[cyan]self.idata.prior_predictive[/cyan]"
        )
        console.print(
            "  Use [yellow]az.plot_ppc(fitter.idata, group='prior_predictive')[/yellow] "
            "for more diagnostics\n"
        )

    def plot_prior_draws(
        self,
        num_draws: int = 30,
        seed: int = 0,
        figsize: tuple[int, int] = (10, 4),
        alpha: float = 0.08,
        lw: float = 1.0,
    ) -> None:
        self.plot_prior_hairlines(
            num_draws=num_draws,
            seed=seed,
            figsize=figsize,
            alpha=alpha,
            lw=lw,
        )

    @staticmethod
    def simulate(
        A: list[float],
        mu: list[float],
        sigma: list[float],
        alpha: list[float],
        x_min: float,
        x_max: float,
        sampling_rate: float = 1200.0,
        noise_level: float = 0.02,
        seed: int = 0,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Simulate a chromatographic spectrum with multiple skew-normal peaks.

        Args:
            A: Peak areas, length K
            mu: Peak centers (retention times), length K
            sigma: Skew-normal scale parameters, length K
            alpha: Skewness parameters, length K
            x_min: Minimum retention time
            x_max: Maximum retention time
            sampling_rate: Samples per time unit (default: 1200 Hz = 20 Hz * 60)
            noise_level: Relative noise level (fraction of max signal)
            seed: Random seed for noise generation

        Returns:
            Tuple of (x, y_clean, y_noisy) arrays

        Example:
            >>> x, y_clean, y_noisy = ChromFitter.simulate(
            ...     A=[100, 150, 80],
            ...     mu=[6.5, 6.7, 6.9],
            ...     sigma=[0.05, 0.07, 0.03],
            ...     alpha=[2.0, 0.0, -3.0],
            ...     x_min=5.0,
            ...     x_max=8.0,
            ...     noise_level=0.02
            ... )
        """
        # Convert to arrays
        A = jnp.asarray(A, dtype=jnp.float32)
        mu = jnp.asarray(mu, dtype=jnp.float32)
        sigma = jnp.asarray(sigma, dtype=jnp.float32)
        alpha = jnp.asarray(alpha, dtype=jnp.float32)

        # Validate inputs
        K = len(A)
        if not (len(mu) == len(sigma) == len(alpha) == K):
            raise ValueError("All peak parameter lists must have the same length")

        if x_min >= x_max:
            raise ValueError("x_min must be less than x_max")

        # Generate time points
        N = int((x_max - x_min) * sampling_rate) + 1
        x = jnp.linspace(x_min, x_max, N, dtype=jnp.float32)

        # Generate clean signal
        y_clean = skew_mixture_area(x, A, mu, sigma, alpha)

        # Add noise
        rng = jax.random.PRNGKey(seed)
        rel_noise = 0.05  # 5 % of the local signal height

        abs_noise = 10.0  # μV (or whatever units) absolute noise SD

        rng = jax.random.PRNGKey(seed)

        # per-point standard deviation:  σ_i = sqrt((ρ·y_i)² + σ₀²)
        sigma_pts = jnp.sqrt((rel_noise * y_clean) ** 2 + abs_noise**2)

        noise = jax.random.normal(rng, shape=y_clean.shape) * sigma_pts
        y_noisy = y_clean + noise

        return x, y_clean, y_noisy.astype(jnp.float32)

    def plot_chromatogram_summed(self) -> None:
        plt.plot(self.x[0, :], self.y.sum(axis=0))

        # individual spectra as fine lines
        for i in range(self.y.shape[0]):
            plt.plot(
                self.x[i, :],
                self.y[i, :],
                linewidth=0.5,
                color="tab:blue",
                linestyle=":",
            )

        # show fine grid lines with dense x ticks
        plt.grid(
            True, linestyle="--", linewidth=0.5, which="both", axis="both", alpha=0.5
        )

        # Show dense x ticks and increase number of minor ticks
        ax = plt.gca()
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.tick_params(axis="x", which="minor", bottom=True, length=3)
        ax.tick_params(axis="y", which="minor", left=True, length=3)

        # add legend
        plt.xlabel("Retention Time [min]")
        plt.ylabel("Signal")

        # save figure
        plt.savefig("chromatogram_summed.png")

        plt.tight_layout()
        plt.show()

    def plot_data_all_spectra(
        self,
        figsize: tuple[int, int] = (12, 5),
        linewidth: float = 0.8,
        alpha: float = 0.95,
        cmap_name: str = "coolwarm",
        show_baseline: bool = False,
        baseline_linewidth: float = 0.8,
        baseline_alpha: float = 0.9,
        baseline_linestyle: str = "--",
    ) -> None:
        """Plot all spectra in one axis using a coolwarm color mapping."""
        x2d = self._as_2d(self.x)
        y2d = self._as_2d(self.y)
        bg2d = self._as_2d(self.background)
        s_count = int(y2d.shape[0])

        plt.figure(figsize=figsize)
        cmap = plt.get_cmap(cmap_name)

        if s_count == 1:
            colors = [cmap(0.5)]
        else:
            colors = [cmap(i / (s_count - 1)) for i in range(s_count)]

        for s in range(s_count):
            plt.plot(
                x2d[s],
                y2d[s],
                color=colors[s],
                linewidth=linewidth,
                alpha=alpha,
            )
            if show_baseline:
                plt.plot(
                    x2d[s],
                    bg2d[s],
                    color=colors[s],
                    linewidth=baseline_linewidth,
                    alpha=baseline_alpha,
                    linestyle=baseline_linestyle,
                    label="Estimated baseline" if s == 0 else None,
                )

        ax = plt.gca()
        ax.minorticks_on()
        ax.grid(True, which="major", alpha=0.1)
        ax.grid(True, which="minor", alpha=0.1)
        plt.xlabel("Retention time")
        plt.ylabel("Signal")
        if show_baseline:
            plt.title("All spectra with baseline")
            plt.legend(loc="best", frameon=False)
        else:
            plt.title("All spectra")
        plt.tight_layout()
        self._save_and_close_current_figure("all_spectra")

    def plot_low_slope_outside_windows(
        self,
        quantile: float = 0.10,
        curvature_quantile: float = 0.25,
        use_model_mask: bool = False,
        figsize: tuple[int, int] = (12, 6),
        line_color: str = "0.55",
        line_alpha: float = 0.55,
        line_width: float = 0.9,
        point_color: str = "tab:blue",
        point_size: float = 10.0,
    ) -> None:
        """Plot all spectra and highlight low-slope/low-curvature points outside windows.

        Steps:
        1) Compute absolute local slope |dy/dx| for adjacent point pairs.
        2) Keep only intervals where both points are outside all peak windows.
        3) Compute curvature proxy |d2y/dx2| and map it to intervals.
        4) Select intervals in the lowest `quantile` of |dy/dx| AND lowest
           `curvature_quantile` of |d2y/dx2| per spectrum.
        5) Highlight both endpoints of those intervals in blue.

        If `use_model_mask=True`, plot the currently configured baseline-anchor mask
        used by the model (`self.baseline_anchor_mask`) instead of recomputing from
        the provided quantiles.
        """
        x2d = self._as_2d(self.x)
        y2d = self._as_2d(self.y)
        S = int(x2d.shape[0])
        outside_window = self._outside_peak_window_mask()

        if use_model_mask:
            selected_points = self.baseline_anchor_mask
            slope_q = self.baseline_slope_quantile
            curv_q = self.baseline_curvature_quantile
        else:
            selected_points = self._compute_baseline_anchor_mask(
                slope_quantile=quantile,
                curvature_quantile=curvature_quantile,
            )
            slope_q = quantile
            curv_q = curvature_quantile

        plt.figure(figsize=figsize)

        line_label_added = False
        point_label_added = False
        for s in range(S):
            x_s = x2d[s]
            y_s = y2d[s]
            m_s = selected_points[s]

            plt.plot(
                x_s,
                y_s,
                color=line_color,
                alpha=line_alpha,
                linewidth=line_width,
                label="Spectra" if not line_label_added else None,
            )
            line_label_added = True

            if bool(jnp.any(m_s)):
                plt.scatter(
                    x_s[m_s],
                    y_s[m_s],
                    s=point_size,
                    color=point_color,
                    alpha=0.95,
                    linewidths=0.0,
                    label=(
                        "Outside windows + lowest "
                        f"{int(100 * slope_q)}% |dy/dx| and "
                        f"{int(100 * curv_q)}% |d2y/dx2|"
                        if not point_label_added
                        else None
                    ),
                )
                point_label_added = True

        n_sel = int(jnp.sum(selected_points))
        n_out = int(jnp.sum(outside_window))
        plt.title(
            "Low-slope/low-curvature points outside peak windows "
            f"({n_sel}/{n_out} points selected)"
        )
        plt.xlabel("Retention time")
        plt.ylabel("Signal")
        if line_label_added or point_label_added:
            plt.legend(loc="best", frameon=False)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        self._save_and_close_current_figure("outside_low_slope_curv")

    def plot_peak_moment_diagnostics(
        self,
        peak_names: Optional[Sequence[str]] = None,
        start_quantile: float = 0.005,
        end_quantile: float = 0.995,
        tail_window_sigma: float = 2.0,
        use_background: bool = False,
        overlay_alpha: float = 0.20,
        overlay_linewidth: float = 0.8,
        dpi: int = 150,
    ) -> dict[str, dict[str, Any]]:
        """Visualize moment-based metrics within user-defined logical windows.

        For each selected logical peak, this method creates a 2x2 diagnostic
        figure showing:
        1) overlay of raw traces in the broad user window with robust bound bands,
        2) per-trace positions (start/end/apex/centroid),
        3) asymmetry indicators (z-shift, skewness, log tail ratio),
        4) scale indicators (sigma, left/right sigma, relative area).
        """
        diagnostics = self.compute_peak_moment_metrics(
            peak_names=peak_names,
            start_quantile=start_quantile,
            end_quantile=end_quantile,
            tail_window_sigma=tail_window_sigma,
            use_background=use_background,
        )

        x2d = np.asarray(self._as_2d(self.x), dtype=float)
        y2d = np.asarray(self._as_2d(self.y), dtype=float)
        if use_background:
            y2d = y2d - np.asarray(self._as_2d(self.background), dtype=float)

        trace_index = np.arange(x2d.shape[0], dtype=int)

        for peak_name, payload in diagnostics.items():
            definition = payload["definition"]
            metric = payload["metrics"]

            start_times = np.asarray(metric["start_time"], dtype=float)
            end_times = np.asarray(metric["end_time"], dtype=float)
            centroids = np.asarray(metric["centroid"], dtype=float)
            apex_times = np.asarray(metric["apex_time"], dtype=float)
            sigmas = np.asarray(metric["sigma"], dtype=float)
            left_sigmas = np.asarray(metric["left_sigma"], dtype=float)
            right_sigmas = np.asarray(metric["right_sigma"], dtype=float)
            areas = np.asarray(metric["area"], dtype=float)
            centroid_apex_z = np.asarray(metric["centroid_apex_z"], dtype=float)
            skewness = np.asarray(metric["skewness"], dtype=float)
            log_tail_ratio = np.asarray(metric["log_tail_ratio"], dtype=float)

            area_median = np.nanmedian(areas)
            if not np.isfinite(area_median) or area_median <= 1e-12:
                area_relative = np.full_like(areas, np.nan)
            else:
                area_relative = areas / area_median

            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            ax_overlay = axes[0, 0]
            ax_positions = axes[0, 1]
            ax_shape = axes[1, 0]
            ax_scale = axes[1, 1]

            for trace_id in range(x2d.shape[0]):
                x_trace = x2d[trace_id]
                y_trace = y2d[trace_id]
                trace_mask = (
                    np.isfinite(x_trace)
                    & np.isfinite(y_trace)
                    & (x_trace >= float(definition.low))
                    & (x_trace <= float(definition.high))
                )
                if int(np.sum(trace_mask)) < 2:
                    continue
                ax_overlay.plot(
                    x_trace[trace_mask],
                    y_trace[trace_mask],
                    color="0.5",
                    alpha=overlay_alpha,
                    linewidth=overlay_linewidth,
                )

            def _add_iqr_band(
                axis,
                values: np.ndarray,
                color: str,
                label: str,
            ) -> None:
                finite_values = values[np.isfinite(values)]
                if finite_values.size == 0:
                    return
                q25, q75 = np.percentile(finite_values, [25, 75])
                median = np.median(finite_values)
                axis.axvspan(q25, q75, color=color, alpha=0.12, label=f"{label} IQR")
                axis.axvline(
                    median,
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                    label=f"{label} median",
                )

            _add_iqr_band(ax_overlay, start_times, "tab:green", "start")
            _add_iqr_band(ax_overlay, end_times, "tab:red", "end")
            _add_iqr_band(ax_overlay, centroids, "tab:blue", "centroid")
            _add_iqr_band(ax_overlay, apex_times, "tab:orange", "apex")

            ax_overlay.set_title("Window overlay with robust moment bands")
            ax_overlay.set_xlabel("Retention time")
            ax_overlay.set_ylabel("Signal")
            ax_overlay.grid(True, alpha=0.25)
            ax_overlay.legend(loc="best", frameon=False, fontsize=8)

            ax_positions.plot(
                trace_index,
                start_times,
                color="tab:green",
                marker="o",
                markersize=2.5,
                linewidth=1.0,
                label="start",
            )
            ax_positions.plot(
                trace_index,
                end_times,
                color="tab:red",
                marker="o",
                markersize=2.5,
                linewidth=1.0,
                label="end",
            )
            ax_positions.plot(
                trace_index,
                centroids,
                color="tab:blue",
                marker="o",
                markersize=2.5,
                linewidth=1.0,
                label="centroid",
            )
            ax_positions.plot(
                trace_index,
                apex_times,
                color="tab:orange",
                marker="o",
                markersize=2.5,
                linewidth=1.0,
                label="apex",
            )
            ax_positions.axhline(
                float(definition.low),
                color="0.35",
                linestyle=":",
                linewidth=1.0,
                label="window low",
            )
            ax_positions.axhline(
                float(definition.high),
                color="0.20",
                linestyle=":",
                linewidth=1.0,
                label="window high",
            )
            ax_positions.set_title("Per-trace location metrics")
            ax_positions.set_xlabel("Flattened trace index")
            ax_positions.set_ylabel("Retention time")
            ax_positions.grid(True, alpha=0.25)
            ax_positions.legend(loc="best", frameon=False, fontsize=8)

            ax_shape.plot(
                trace_index,
                centroid_apex_z,
                color="tab:purple",
                marker="o",
                markersize=2.5,
                linewidth=1.0,
                label="(centroid-apex)/sigma",
            )
            ax_shape.plot(
                trace_index,
                skewness,
                color="tab:brown",
                marker="o",
                markersize=2.5,
                linewidth=1.0,
                label="skewness",
            )
            ax_shape.plot(
                trace_index,
                log_tail_ratio,
                color="tab:pink",
                marker="o",
                markersize=2.5,
                linewidth=1.0,
                label="log tail ratio",
            )
            ax_shape.axhline(0.0, color="0.2", linestyle="--", linewidth=1.0)
            ax_shape.set_title("Asymmetry metrics")
            ax_shape.set_xlabel("Flattened trace index")
            ax_shape.grid(True, alpha=0.25)
            ax_shape.legend(loc="best", frameon=False, fontsize=8)
            ax_shape.set_ylim(-2.0, 2.0)

            ax_scale.plot(
                trace_index,
                sigmas,
                color="tab:blue",
                marker="o",
                markersize=2.5,
                linewidth=1.0,
                label="sigma",
            )
            ax_scale.plot(
                trace_index,
                left_sigmas,
                color="tab:cyan",
                marker="o",
                markersize=2.5,
                linewidth=1.0,
                label="left sigma",
            )
            ax_scale.plot(
                trace_index,
                right_sigmas,
                color="tab:olive",
                marker="o",
                markersize=2.5,
                linewidth=1.0,
                label="right sigma",
            )
            ax_scale.plot(
                trace_index,
                area_relative,
                color="tab:gray",
                marker="o",
                markersize=2.5,
                linewidth=1.0,
                label="area / median(area)",
            )
            ax_scale.axhline(1.0, color="0.2", linestyle="--", linewidth=1.0)
            ax_scale.set_title("Scale metrics")
            ax_scale.set_xlabel("Flattened trace index")
            ax_scale.grid(True, alpha=0.25)
            ax_scale.legend(loc="best", frameon=False, fontsize=8)
            ax_scale.set_ylim(-2.0, 2.0)

            fig.suptitle(
                f"Moment diagnostics: {peak_name} [{definition.low:.4f}, {definition.high:.4f}]",
                fontsize=13,
            )
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

            safe_name = "".join(
                char if (char.isalnum() or char in ("_", "-")) else "_"
                for char in str(peak_name)
            ).strip("_")
            if not safe_name:
                safe_name = "peak"
            self._save_and_close_current_figure(
                stem=f"moment_metrics_{safe_name}",
                dpi=dpi,
            )

            console.print(
                "[cyan]Moment diagnostics[/cyan] "
                f"{peak_name}: "
                f"start≈{np.nanmedian(start_times):.4f}, "
                f"end≈{np.nanmedian(end_times):.4f}, "
                f"z≈{np.nanmedian(centroid_apex_z):.3f}, "
                f"skew≈{np.nanmedian(skewness):.3f}, "
                f"log-tail≈{np.nanmedian(log_tail_ratio):.3f}"
            )

        return diagnostics


if __name__ == "__main__":
    from rich import print

    print("\n[main] Loading data...")
    arr = jnp.load("/Users/max/code/sahh-kinetics-hplc/chromatograms.npy")
    time = jnp.load("/Users/max/code/sahh-kinetics-hplc/times.npy")
    sample_names = jnp.load("/Users/max/code/sahh-kinetics-hplc/folder_names.npy")
    chromatogram_names = jnp.load("/Users/max/code/sahh-kinetics-hplc/sample_names.npy")
    n_keep_chromatograms = 22 * 7
    if arr.ndim < 3 or time.ndim < 3:
        raise ValueError(
            f"Expected arr/time with shape [S, C, N], got arr={arr.shape}, time={time.shape}"
        )
    if arr.shape[1] != time.shape[1]:
        raise ValueError(
            f"Chromatogram axis mismatch: arr.shape[1]={arr.shape[1]} vs time.shape[1]={time.shape[1]}"
        )

    n_samples = int(arr.shape[0])
    n_chrom_per_sample = int(arr.shape[1])
    total_chromatograms = n_samples * n_chrom_per_sample
    traces_to_keep = max(int(n_keep_chromatograms), 1)

    sample_names_arr = np.asarray(sample_names, dtype=object)
    chromatogram_names_arr = np.asarray(chromatogram_names, dtype=object)

    if traces_to_keep >= total_chromatograms:
        # Keep the full [S, C, N] cube.
        sample_names = sample_names_arr
        chromatogram_names = chromatogram_names_arr
        print("[main] Keeping all chromatograms:")
        print(
            f"  requested traces: {traces_to_keep}, total available: {total_chromatograms}, "
            f"kept traces: {total_chromatograms}"
        )
        print(f"  new shape: {arr.shape}")
    else:
        # Keep the last N chromatograms across all samples (global flattened order).
        keep_start = total_chromatograms - traces_to_keep
        arr_flat = arr.reshape(total_chromatograms, arr.shape[-1])
        time_flat = time.reshape(total_chromatograms, time.shape[-1])
        arr = arr_flat[keep_start:][None, :, :]
        time = time_flat[keep_start:][None, :, :]

        sample_index_flat = np.repeat(np.arange(n_samples), n_chrom_per_sample)
        chromatogram_index_flat = np.tile(np.arange(n_chrom_per_sample), n_samples)
        selected_labels: list[str] = []
        for flat_index in range(keep_start, total_chromatograms):
            sample_index = int(sample_index_flat[flat_index])
            chromatogram_index = int(chromatogram_index_flat[flat_index])
            if sample_index < sample_names_arr.shape[0]:
                sample_label = str(sample_names_arr[sample_index])
            else:
                sample_label = f"sample_{sample_index}"

            chrom_label = f"chrom_{chromatogram_index}"
            if chromatogram_names_arr.ndim >= 2:
                if (
                    sample_index < chromatogram_names_arr.shape[0]
                    and chromatogram_index < chromatogram_names_arr.shape[1]
                ):
                    chrom_label = str(
                        chromatogram_names_arr[sample_index, chromatogram_index]
                    )
            elif chromatogram_names_arr.ndim == 1:
                flat_name_index = sample_index * n_chrom_per_sample + chromatogram_index
                if flat_name_index < chromatogram_names_arr.shape[0]:
                    chrom_label = str(chromatogram_names_arr[flat_name_index])
            selected_labels.append(f"{sample_label} | {chrom_label}")

        sample_names = np.asarray(["selected_chromatograms"], dtype=object)
        chromatogram_names = np.asarray([selected_labels], dtype=object)

        print("[main] Keeping last chromatograms across all samples:")
        print(
            f"  requested traces: {traces_to_keep}, total available: {total_chromatograms}, "
            f"kept traces: {arr.shape[1]}"
        )
        print(f"  new shape: {arr.shape}")

    F = ChromFitter(time, arr, [], sample_names, chromatogram_names)

    F.add_baseline_region(low=0.5, high=1.3)
    F.add_baseline_region(low=1.9, high=2.1)
    F.add_baseline_region(low=4, high=4.5)

    print(F.add_peak(name="sah", low=2.60, high=2.9, shoulder="right"))
    print(F.add_peak(name="ado", low=3.1, high=3.45, shoulder="right"))

    # F.add_peak(name="ado", low=2.60, high=2.85)
    F.slice_time_ranges([(2.6, 3.6)])

    F.apply_retention_shift_correction()
    F.plot_data(
        save_path="figs/data.png",
        dpi=300,
    )
    print("[main] Plotting prior predictive chromatogram draws...")
    F.plot_prior_predictive_chromatograms(
        num_draws=50,
        seed=0,
        draw_color="tab:blue",
        draw_alpha=0.2,
    )
    print("[main] Plotting peak moment diagnostics...")
    F.plot_peak_moment_diagnostics(
        start_quantile=0.005,
        end_quantile=0.995,
        tail_window_sigma=2.0,
        use_background=False,
        overlay_alpha=0.18,
        overlay_linewidth=0.8,
        dpi=200,
    )
    raise SystemExit(0)

    F.fit(num_warmup=3000, num_samples=2000, num_chains=8)
    if F.idata is not None:
        summary_vars = [v for v in F.idata.posterior.data_vars if v != "mu_y"]
        summary_df = az.summary(F.idata, var_names=summary_vars, round_to=3)
        summary_text = summary_df.to_string()

        summary_path = F.figure_dir / "arviz_summary.txt"
        summary_path.write_text(summary_text + "\n", encoding="utf-8")
        print(f"[main] Saved ArviZ summary: {summary_path}")
    else:
        print("[main] No inference data available; skipped ArviZ summary export.")

    F.plot_fit()

    F.plot_trace()

    F.slice_time_ranges([(0.5, 5)])
    F.plot_data(
        save_path="figs/data_shifted.png",
        dpi=300,
    )

    # subtract background and plot data
    F.plot_data(
        save_path="figs/data_subtracted.png",
        dpi=300,
        ymax=1500,
        ymin=-200,
    )

    # Start with no peaks, then add user-defined peaks via add_peak().
    empty = jnp.array([], dtype=jnp.float32)
    fitter = ChromFitter(time, arr, empty, empty)

    def print_prep_state(step: str) -> None:
        print(f"\n[prep] {step}")
        print(f"  K={fitter.K}, peak_names={fitter.peak_names}")
        print(f"  mu_lo: {jnp.asarray(fitter.mu_lo)}")
        print(f"  mu_hi: {jnp.asarray(fitter.mu_hi)}")
        print(f"  mu_init shape={fitter.mu_init.shape}")
        print(f"  sigma_init shape={fitter.sigma_init.shape}")
        print(f"  A_init shape={fitter.A_init.shape}")
        print(
            "  background: "
            f"shape={fitter.background.shape}, "
            f"min={float(jnp.nanmin(fitter.background)):.3f}, "
            f"max={float(jnp.nanmax(fitter.background)):.3f}"
        )
        print(
            "  baseline mask: "
            f"shape={fitter.baseline_anchor_mask.shape}, "
            f"selected={int(jnp.sum(fitter.baseline_anchor_mask))}, "
            f"slope_q={fitter.baseline_slope_quantile:.2f}, "
            f"curv_q={fitter.baseline_curvature_quantile:.2f}"
        )
        if fitter.K > 0:
            print(f"  mu_init:\n{jnp.asarray(fitter.mu_init)}")
            print(f"  sigma_init:\n{jnp.asarray(fitter.sigma_init)}")
            print(f"  A_init:\n{jnp.asarray(fitter.A_init)}")

    print_prep_state("initial (no peaks)")

    peak_specs = [
        {"name": "ado", "low": 2.60, "high": 2.85},
        {"name": "ino", "low": 2.87, "high": 3.14},
        {"name": "spi", "low": 3.18, "high": 3.43},
    ]
    for spec in peak_specs:
        info = fitter.add_peak(**spec)
        name = spec["name"]
        low = float(spec["low"])
        high = float(spec["high"])
        print(f"\n[prep] added peak `{name}` [{low:.2f}, {high:.2f}]")
        print(f"  index={info['index']}")
        print(f"  mu0={jnp.asarray(info['mu0'])}")
        print(f"  sigma0={jnp.asarray(info['sigma0'])}")
        print(f"  A0={jnp.asarray(info['A0'])}")
        print_prep_state(f"after add_peak({name})")

    # ------------------------------------------------------------------
    # Plotting presets (tune here)
    # ------------------------------------------------------------------
    n_spectra = int(fitter._as_2d(fitter.y).shape[0])
    n_components = max(1, fitter.K)
    plot_all_kwargs: dict[str, Any] = {
        "figsize": (13, 5),
        "linewidth": 0.9,
        "alpha": 0.90,
        "cmap_name": "coolwarm",
    }
    plot_baseline_kwargs: dict[str, Any] = {
        "show_baseline": True,
        "baseline_linewidth": 1.0,
        "baseline_alpha": 0.90,
        "baseline_linestyle": "--",
    }
    prior_plot_kwargs: dict[str, Any] = {
        "num_draws": 200,
        "seed": 0,
        "figsize": (max(4.2 * n_components, 8.0), max(2.4 * n_spectra, 4.0)),
        "alpha": 0.30,
        "lw": 1.8,
        "nsigma": 3.0,
        "dense_points": 700,
        "hdi_prob": 0.95,
        "include_baseline": False,
        "scatter_size": 5.0,
        "data_color": "0.45",
    }
    trace_var_names = [
        "A",
        "mu",
        "sigma",
        "alpha",
        "sigma_y",
    ]
    pair_var_names = [
        "A",
        "mu",
        "sigma",
        "alpha",
        "sigma_y",
    ]
    component_plot_kwargs: dict[str, Any] = {
        "nsigma": 3.0,
        "scatter_size": 5.0,
        "fill_alpha": 0.30,
        "include_baseline": False,
        "dense_points": 700,
        "hdi_prob": 0.95,
        "component_median_color": "0.45",
        "figsize": (max(4.2 * n_components, 8.0), max(2.4 * n_spectra, 4.0)),
    }
    fit_plot_kwargs: dict[str, Any] = {
        "figsize": (12, 2),
        "fill_alpha": 0.26,
        "linewidth": 1.8,
        "data_color": "0.45",
        "dense_points": 700,
    }

    print("\n[main] Plotting all spectra (before shift correction)...")
    fitter.plot_data_all_spectra(**plot_all_kwargs)

    print("\n[main] Applying groupwise retention shift correction...")
    fitter.apply_retention_shift_correction(
        lr=1e-2,
        n_steps=500,
        center_weight=1e3,
        max_shift_samples=None,
        enforce_zero_mean=True,
        return_history=False,
        verbose=True,
    )
    print_prep_state("after retention shift correction")

    print("\n[main] Plotting all spectra (after shift correction)...")
    fitter.plot_data_all_spectra(**plot_all_kwargs)
    print("\n[main] Estimating baseline (collab_pls, arpls)...")
    fitter.estimate_background(
        mode="collab_pls",
        collab_method="arpls",
        average_dataset=True,
        arpls_kwargs={"lam": 1e6, "diff_order": 2, "max_iter": 50, "tol": 1e-3},
        verbose=True,
    )
    print("\n[main] Plotting all spectra with estimated baseline...")
    fitter.plot_data_all_spectra(**plot_all_kwargs, **plot_baseline_kwargs)
    print("\n[main] Subtracting baseline from spectra...")
    fitter.subtract_baseline(estimate_first=False, verbose=True)
    print_prep_state("after baseline subtraction")
    print("\n[main] Plotting baseline-corrected spectra...")
    fitter.plot_data_all_spectra(**plot_all_kwargs, show_baseline=False)
    print("[main] Plotting prior predictive hairlines...")
    fitter.plot_prior_hairlines(**prior_plot_kwargs)

    # assert False

    print("[main] Running MCMC fit...")
    fitter.fit(num_warmup=1000, num_samples=1000)

    print("[main] Diagnostics (area posterior only)...")
    if fitter.idata is not None:
        print(az.summary(fitter.idata, var_names=["A"], round_to=3))

    A_post = jnp.asarray(fitter.samples["A"], dtype=jnp.float32)
    if A_post.ndim == 2:
        A_post = A_post[:, None, :]
    n_spectra = int(A_post.shape[1])

    print("[main] Area posterior 1σ (std) per component:")
    for k, name in enumerate(fitter.peak_names):
        vals_all = A_post[:, :, k].reshape(-1)
        mean_all = float(jnp.mean(vals_all))
        std_all = float(jnp.std(vals_all))
        print(f"  {name}: mean={mean_all:.3f}, 1σ={std_all:.3f}")

    print("[main] Area posterior 1σ (std) per component and spectrum:")
    for k, name in enumerate(fitter.peak_names):
        for s in range(n_spectra):
            vals_sk = A_post[:, s, k]
            mean_sk = float(jnp.mean(vals_sk))
            std_sk = float(jnp.std(vals_sk))
            print(f"  {name} | spectrum {s}: mean={mean_sk:.3f}, 1σ={std_sk:.3f}")

    print("\n[main] Plotting trace...")
    fitter.plot_trace(var_names=trace_var_names)
    print("\n[main] Plotting component windows posterior...")
    fitter.plot_component_windows_posterior(**component_plot_kwargs)
    print("\n[main] Plotting fit...")
    fitter.plot_fit(**fit_plot_kwargs)
    print("\n[main] Plotting pair...")
    fitter.plot_pair(var_names=pair_var_names)
