from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class BaselineEstimate:
    slope: float
    intercept: float
    r2: float


def linear_baseline_estimate(
    time: jnp.ndarray,
    signal: jnp.ndarray,
) -> list[BaselineEstimate]:
    """Estimate linear baseline. Ignores NaN (use NaN for points to exclude)."""
    xm = jnp.nanmean(time, axis=-1, keepdims=True)
    ym = jnp.nanmean(signal, axis=-1, keepdims=True)

    xc = time - xm
    yc = signal - ym

    sxx = jnp.nansum(xc * xc, axis=-1)
    sxy = jnp.nansum(xc * yc, axis=-1)

    slope = jnp.where(sxx > 1e-12, sxy / sxx, 0.0)
    intercept = jnp.nan_to_num(ym[..., 0], nan=0.0) - slope * jnp.nan_to_num(
        xm[..., 0], nan=0.0
    )

    yhat = slope[..., None] * time + intercept[..., None]

    ss_res = jnp.nansum((signal - yhat) ** 2, axis=-1)
    ss_tot = jnp.nansum((signal - ym) ** 2, axis=-1)
    r2 = jnp.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, 0.0)

    assert (
        slope.shape == intercept.shape == r2.shape
    ), "slope, intercept, and r2 must have the same shape"

    baseline_estimates = []
    for i in range(slope.shape[0]):
        baseline_estimates.append(
            BaselineEstimate(slope=slope[i], intercept=intercept[i], r2=r2[i])
        )

    return baseline_estimates

    # def estimate_background(
    #     self,
    #     mode: Literal["collab_pls", "arpls", "asls"] = "collab_pls",
    #     collab_method: Literal["arpls", "asls"] = "arpls",
    #     average_dataset: bool = True,
    #     arpls_kwargs: Optional[dict[str, Any]] = None,
    #     asls_kwargs: Optional[dict[str, Any]] = None,
    #     lam: float = 1e6,
    #     diff_order: int = 2,
    #     max_iter: int = 50,
    #     tol: float = 1e-3,
    #     verbose: bool = True,
    # ) -> jnp.ndarray:
    #     """Estimate baseline for ``self.signal`` ([S, C, N]).

    #     Baseline backends:
    #     - ``mode="collab_pls"`` (default): uses ``pybaselines.Baseline.collab_pls``
    #       collaboratively across samples for each chromatogram index.
    #     - ``mode="arpls"``: per-trace local arPLS using the in-house implementation.
    #     - ``mode="asls"``: per-trace local AsLS using ``pybaselines.Baseline.asls``.

    #     Hyperparameters for arPLS / AsLS should be passed via ``arpls_kwargs`` /
    #     ``asls_kwargs`` dictionaries. Legacy ``lam/diff_order/max_iter/tol`` are
    #     used as defaults and are overridden by explicit kwargs.
    #     """
    #     signal_np = np.asarray(self.signal, dtype=np.float64)  # (S, C, N)
    #     if signal_np.ndim != 3:
    #         raise ValueError(
    #             f"Expected self.signal with shape [S, C, N], got {signal_np.shape}"
    #         )
    #     S, C, N = signal_np.shape

    #     arpls_options: dict[str, Any] = {
    #         "lam": lam,
    #         "diff_order": diff_order,
    #         "max_iter": max_iter,
    #         "tol": tol,
    #     }
    #     if arpls_kwargs is not None:
    #         arpls_options.update(arpls_kwargs)

    #     asls_options: dict[str, Any] = {
    #         "lam": lam,
    #         "p": 0.01,
    #         "diff_order": diff_order,
    #         "max_iter": max_iter,
    #         "tol": tol,
    #     }
    #     if asls_kwargs is not None:
    #         asls_options.update(asls_kwargs)

    #     baseline_np = np.zeros_like(signal_np, dtype=np.float64)
    #     valid_trace = np.zeros((S, C), dtype=bool)
    #     prepared_signal = np.zeros_like(signal_np, dtype=np.float64)

    #     idx = np.arange(N, dtype=np.int64)
    #     min_pts = max(4, int(diff_order) + 2)

    #     # Prepare finite traces for baseline fitting (interpolate NaN/Inf where possible).
    #     for s in range(S):
    #         for c in range(C):
    #             trace = signal_np[s, c]
    #             finite_mask = np.isfinite(trace)
    #             n_finite = int(finite_mask.sum())

    #             if n_finite < min_pts:
    #                 prepared_signal[s, c] = 0.0
    #                 continue

    #             if n_finite < N:
    #                 finite_idx = idx[finite_mask]
    #                 trace_interp = trace.copy()
    #                 trace_interp[~finite_mask] = np.interp(
    #                     idx[~finite_mask],
    #                     finite_idx,
    #                     trace[finite_mask],
    #                     left=float(trace[finite_mask][0]),
    #                     right=float(trace[finite_mask][-1]),
    #                 )
    #             else:
    #                 trace_interp = trace

    #             prepared_signal[s, c] = trace_interp
    #             valid_trace[s, c] = True

    #     collab_params_by_chrom: dict[int, dict[str, Any]] = {}
    #     if mode == "arpls":
    #         for s in range(S):
    #             for c in range(C):
    #                 if not valid_trace[s, c]:
    #                     continue
    #                 baseline_np[s, c] = self._arpls_baseline_1d(
    #                     prepared_signal[s, c],
    #                     **arpls_options,
    #                 )
    #     elif mode == "asls":
    #         try:
    #             from pybaselines import Baseline
    #         except Exception as exc:
    #             raise ImportError(
    #                 "pybaselines is required for mode='asls'. "
    #                 "Install it with `pip install pybaselines`."
    #             ) from exc

    #         baseline_solver = Baseline()
    #         for s in range(S):
    #             for c in range(C):
    #                 if not valid_trace[s, c]:
    #                     continue
    #                 baseline_trace, _ = baseline_solver.asls(
    #                     prepared_signal[s, c],
    #                     **asls_options,
    #                 )
    #                 baseline_np[s, c] = np.asarray(baseline_trace, dtype=np.float64)
    #     elif mode == "collab_pls":
    #         if collab_method not in {"arpls", "asls"}:
    #             raise ValueError(
    #                 f"Unsupported collab_method '{collab_method}'. Use 'arpls' or 'asls'."
    #             )
    #         try:
    #             from pybaselines import Baseline
    #         except Exception as exc:
    #             raise ImportError(
    #                 "pybaselines is required for mode='collab_pls'. "
    #                 "Install it with `pip install pybaselines`."
    #             ) from exc

    #         method_kwargs = arpls_options if collab_method == "arpls" else asls_options

    #         for c in range(C):
    #             valid_rows = valid_trace[:, c]
    #             if not np.any(valid_rows):
    #                 continue
    #             data_matrix = prepared_signal[valid_rows, c, :]  # [M, N]
    #             baseline_solver = Baseline()
    #             baselines_c, params_c = baseline_solver.collab_pls(
    #                 data_matrix,
    #                 average_dataset=average_dataset,
    #                 method=collab_method,
    #                 method_kwargs=method_kwargs,
    #             )
    #             baseline_np[valid_rows, c, :] = np.asarray(baselines_c, dtype=np.float64)
    #             collab_params_by_chrom[c] = params_c
    #     else:
    #         raise ValueError(
    #             f"Invalid mode '{mode}'. Use 'collab_pls', 'arpls', or 'asls'."
    #         )

    #     self.background = jnp.asarray(baseline_np, dtype=jnp.float32)
    #     self.background_params = {
    #         "mode": mode,
    #         "collab_method": collab_method if mode == "collab_pls" else None,
    #         "average_dataset": bool(average_dataset) if mode == "collab_pls" else None,
    #         "collab_params_by_chrom": collab_params_by_chrom if mode == "collab_pls" else {},
    #         "arpls_kwargs": arpls_options,
    #         "asls_kwargs": asls_options,
    #     }

    #     if verbose:
    #         n_traces = S * C
    #         mean_abs = float(jnp.mean(jnp.abs(self.background)))
    #         print("\n[baseline] Baseline estimated")
    #         print(f"  data shape : {signal_np.shape}  (S={S}, C={C}, N={N})")
    #         print(f"  traces     : {n_traces}")
    #         print(f"  mode       : {mode}")
    #         if mode == "collab_pls":
    #             print(f"  collab_method   : {collab_method}")
    #             print(f"  average_dataset : {average_dataset}")
    #             print(f"  method_kwargs   : {method_kwargs}")
    #             if collab_params_by_chrom:
    #                 print("  collab_pls results by chromatogram:")
    #                 for c_idx, params_c in collab_params_by_chrom.items():
    #                     weights = np.asarray(params_c.get("average_weights", np.array([])))
    #                     if weights.size > 0:
    #                         print(
    #                             f"    chrom {c_idx:>2d}: avg_weights "
    #                             f"(min={float(np.min(weights)):.4g}, "
    #                             f"max={float(np.max(weights)):.4g}, "
    #                             f"mean={float(np.mean(weights)):.4g})"
    #                         )
    #                     method_params = params_c.get("method_params", {})
    #                     tol_histories = method_params.get("tol_history", [])
    #                     last_tols = [
    #                         float(hist[-1]) for hist in tol_histories if hasattr(hist, "__len__") and len(hist) > 0
    #                     ]
    #                     if last_tols:
    #                         print(
    #                             f"      tol_history last (mean/min/max): "
    #                             f"{float(np.mean(last_tols)):.3e} / "
    #                             f"{float(np.min(last_tols)):.3e} / "
    #                             f"{float(np.max(last_tols)):.3e}"
    #                         )
    #         elif mode == "arpls":
    #             print(f"  arpls_kwargs: {arpls_options}")
    #         elif mode == "asls":
    #             print(f"  asls_kwargs : {asls_options}")
    #         print(f"  mean |background| : {mean_abs:.6f}")

    #     return self.background

    # def subtract_baseline(
    #     self,
    #     estimate_first: bool = True,
    #     mode: Literal["collab_pls", "arpls", "asls"] = "collab_pls",
    #     collab_method: Literal["arpls", "asls"] = "arpls",
    #     average_dataset: bool = True,
    #     arpls_kwargs: Optional[dict[str, Any]] = None,
    #     asls_kwargs: Optional[dict[str, Any]] = None,
    #     lam: float = 1e6,
    #     diff_order: int = 2,
    #     max_iter: int = 50,
    #     tol: float = 1e-3,
    #     verbose: bool = True,
    # ) -> None:
    #     """Subtract baseline from ``self.signal`` in-place.

    #     Parameters
    #     ----------
    #     estimate_first : bool
    #         If True, call :func:`estimate_background` first.
    #         If False, ``self.background`` must already exist and match
    #         ``self.signal`` in shape.
    #     """
    #     if estimate_first:
    #         self.estimate_background(
    #             mode=mode,
    #             collab_method=collab_method,
    #             average_dataset=average_dataset,
    #             arpls_kwargs=arpls_kwargs,
    #             asls_kwargs=asls_kwargs,
    #             lam=lam,
    #             diff_order=diff_order,
    #             max_iter=max_iter,
    #             tol=tol,
    #             verbose=verbose,
    #         )
    #     elif not hasattr(self, "background") or self.background.shape != self.signal.shape:
    #         raise ValueError(
    #             "self.background is missing or shape-mismatched. "
    #             "Run estimate_background() first or pass estimate_first=True."
    #         )

    #     self.signal = jnp.asarray(self.signal - self.background, dtype=jnp.float32)

    #     if verbose:
    #         mean_abs = float(jnp.mean(jnp.abs(self.signal)))
    #         print("\n[baseline] Subtracted background from self.signal")
    #         print(f"  post-subtraction mean |signal| : {mean_abs:.6f}")

    # def plot_background_validation(
    #     self,
    #     point_size: float = 6.0,
    #     point_alpha: float = 0.80,
    #     baseline_linewidth: float = 1.2,
    #     baseline_alpha: float = 0.95,
    #     baseline_linestyle: str = ":",
    #     cmap_name: str = "viridis",
    #     fig_width: float = 10.0,
    #     row_height: float = 2.1,
    #     group_by_sample: bool = False,
    #     xmin: Optional[float] = None,
    #     xmax: Optional[float] = None,
    #     save_path: Optional[str] = None,
    #     dpi: int = 150,
    # ) -> tuple[plt.Figure, np.ndarray]:
    #     """Plot raw signal + estimated baseline for baseline-validation.

    #     Modes:
    #     - `group_by_sample=False`: one chromatogram per axis (S*C rows)
    #     - `group_by_sample=True`: one axis per sample, all chromatograms overlaid
    #       with a per-sample color scale across chromatograms.
    #     """
    #     if xmin is not None and xmax is not None and xmax <= xmin:
    #         raise ValueError(f"Invalid x-window: xmin={xmin}, xmax={xmax} (need xmax > xmin)")

    #     signal_np = np.asarray(self.signal)
    #     time_np = np.asarray(self.time)
    #     baseline_np = np.asarray(self.background)

    #     if signal_np.ndim != 3 or time_np.ndim != 3:
    #         raise ValueError(
    #             "Expected `signal` and `time` with shape [S, C, N] (or [S, N, C] per sample)."
    #         )
    #     if baseline_np.shape != signal_np.shape:
    #         raise ValueError(
    #             f"background shape {baseline_np.shape} does not match signal shape {signal_np.shape}"
    #         )

    #     n_samples = int(signal_np.shape[0])
    #     n_chrom = int(signal_np.shape[1])
    #     n_time = int(signal_np.shape[2])

    #     sample_names_arr = np.asarray(self.sample_names, dtype=object)
    #     chrom_names_arr = np.asarray(self.chromatogram_names, dtype=object)

    #     normalized_payload: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
    #     for sample_idx in range(n_samples):
    #         time_s = np.asarray(time_np[sample_idx])
    #         signal_s = np.asarray(signal_np[sample_idx])
    #         baseline_s = np.asarray(baseline_np[sample_idx])

    #         # Normalize each sample to [C, N].
    #         if time_s.shape == (n_time, n_chrom):
    #             time_s = time_s.T
    #         elif time_s.shape != (n_chrom, n_time):
    #             raise ValueError(
    #                 f"time[{sample_idx}] has shape {time_s.shape}, expected "
    #                 f"{(n_chrom, n_time)} or {(n_time, n_chrom)}"
    #             )

    #         if signal_s.shape == (n_time, n_chrom):
    #             signal_s = signal_s.T
    #         elif signal_s.shape != (n_chrom, n_time):
    #             raise ValueError(
    #                 f"signal[{sample_idx}] has shape {signal_s.shape}, expected "
    #                 f"{(n_chrom, n_time)} or {(n_time, n_chrom)}"
    #             )

    #         if baseline_s.shape == (n_time, n_chrom):
    #             baseline_s = baseline_s.T
    #         elif baseline_s.shape != (n_chrom, n_time):
    #             raise ValueError(
    #                 f"background[{sample_idx}] has shape {baseline_s.shape}, expected "
    #                 f"{(n_chrom, n_time)} or {(n_time, n_chrom)}"
    #             )

    #         sample_name = (
    #             str(sample_names_arr[sample_idx])
    #             if sample_names_arr.size > sample_idx
    #             else f"sample_{sample_idx}"
    #         )
    #         normalized_payload.append((time_s, signal_s, baseline_s, sample_name))

    #     n_axes = n_samples if group_by_sample else n_samples * n_chrom
    #     fig, axes = plt.subplots(
    #         n_axes,
    #         1,
    #         figsize=(fig_width, max(2.0, row_height * n_axes)),
    #         squeeze=False,
    #     )
    #     axes_flat = axes.reshape(-1)
    #     cmap = plt.get_cmap(cmap_name)

    #     axis_cursor = 0
    #     for sample_idx, (time_sc, signal_sc, baseline_sc, sample_name) in enumerate(normalized_payload):
    #         if group_by_sample:
    #             ax = axes_flat[axis_cursor]
    #             axis_cursor += 1
    #             colors = (
    #                 [cmap(0.5)]
    #                 if n_chrom == 1
    #                 else [cmap(i / max(n_chrom - 1, 1)) for i in range(n_chrom)]
    #             )
    #             baseline_values_for_limits: list[np.ndarray] = []

    #             for chrom_idx in range(n_chrom):
    #                 x_trace = np.asarray(time_sc[chrom_idx], dtype=float)
    #                 y_trace = np.asarray(signal_sc[chrom_idx], dtype=float)
    #                 baseline_trace = np.asarray(baseline_sc[chrom_idx], dtype=float)
    #                 color = colors[chrom_idx]

    #                 if (
    #                     chrom_names_arr.ndim == 2
    #                     and chrom_names_arr.shape[0] > sample_idx
    #                     and chrom_names_arr.shape[1] > chrom_idx
    #                 ):
    #                     chrom_name = str(chrom_names_arr[sample_idx, chrom_idx])
    #                 else:
    #                     chrom_name = f"chrom_{chrom_idx}"

    #                 window_mask = np.ones_like(x_trace, dtype=bool)
    #                 if xmin is not None:
    #                     window_mask &= x_trace >= xmin
    #                 if xmax is not None:
    #                     window_mask &= x_trace <= xmax

    #                 finite_data = np.isfinite(x_trace) & np.isfinite(y_trace) & window_mask
    #                 finite_baseline = (
    #                     np.isfinite(x_trace) & np.isfinite(baseline_trace) & window_mask
    #                 )

    #                 ax.plot(
    #                     x_trace[finite_baseline],
    #                     baseline_trace[finite_baseline],
    #                     linestyle=baseline_linestyle,
    #                     linewidth=baseline_linewidth,
    #                     alpha=baseline_alpha,
    #                     color=color,
    #                 )

    #                 ax.plot(
    #                     x_trace[finite_data],
    #                     y_trace[finite_data],
    #                     alpha=point_alpha,
    #                     linewidth=1,
    #                     color=color,
    #                     label=chrom_name,
    #                 )
    #                 if np.any(finite_baseline):
    #                     baseline_values_for_limits.append(baseline_trace[finite_baseline])

    #             if baseline_values_for_limits:
    #                 baseline_vals = np.concatenate(baseline_values_for_limits, axis=0)
    #                 baseline_min = float(np.min(baseline_vals))
    #                 baseline_max = float(np.max(baseline_vals))
    #                 ymax = 2.0 * baseline_max
    #                 ymin_padding = 0.05 * max(abs(baseline_min), abs(baseline_max), 1e-12)
    #                 ymin = baseline_min - ymin_padding
    #                 if ymax <= ymin:
    #                     ymax = baseline_max + max(1e-6, ymin_padding)
    #                 ax.set_ylim(ymin, ymax)

    #             if xmin is not None or xmax is not None:
    #                 x_lo = xmin
    #                 x_hi = xmax
    #                 if x_lo is not None and x_hi is not None and x_hi > x_lo:
    #                     ax.set_xlim(x_lo, x_hi)

    #             ax.set_title(f"{sample_name}", fontsize=10)
    #             ax.legend(loc="best", fontsize=7, frameon=False, ncol=2)
    #             ax.set_ylabel(f"Signal [{self.signal_unit}]")
    #             ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))
    #             ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    #             ax.tick_params(axis="x", which="minor", bottom=True, length=3)
    #             ax.tick_params(axis="y", which="minor", left=True, length=3)
    #             ax.grid(True, which="both", alpha=0.2)
    #         else:
    #             for chrom_idx in range(n_chrom):
    #                 ax = axes_flat[axis_cursor]
    #                 axis_cursor += 1

    #                 x_trace = np.asarray(time_sc[chrom_idx], dtype=float)
    #                 y_trace = np.asarray(signal_sc[chrom_idx], dtype=float)
    #                 baseline_trace = np.asarray(baseline_sc[chrom_idx], dtype=float)
    #                 color = cmap(
    #                     0.5 if n_chrom == 1 else chrom_idx / max(n_chrom - 1, 1)
    #                 )

    #                 if (
    #                     chrom_names_arr.ndim == 2
    #                     and chrom_names_arr.shape[0] > sample_idx
    #                     and chrom_names_arr.shape[1] > chrom_idx
    #                 ):
    #                     chrom_name = str(chrom_names_arr[sample_idx, chrom_idx])
    #                 else:
    #                     chrom_name = f"chrom_{chrom_idx}"

    #                 window_mask = np.ones_like(x_trace, dtype=bool)
    #                 if xmin is not None:
    #                     window_mask &= x_trace >= xmin
    #                 if xmax is not None:
    #                     window_mask &= x_trace <= xmax

    #                 finite_data = np.isfinite(x_trace) & np.isfinite(y_trace) & window_mask
    #                 finite_baseline = (
    #                     np.isfinite(x_trace) & np.isfinite(baseline_trace) & window_mask
    #                 )

    #                 ax.scatter(
    #                     x_trace[finite_data],
    #                     y_trace[finite_data],
    #                     s=point_size,
    #                     alpha=point_alpha,
    #                     linewidths=0.0,
    #                     color=color,
    #                     label="Raw signal",
    #                 )
    #                 ax.plot(
    #                     x_trace[finite_baseline],
    #                     baseline_trace[finite_baseline],
    #                     linestyle=baseline_linestyle,
    #                     linewidth=baseline_linewidth,
    #                     alpha=baseline_alpha,
    #                     color=color,
    #                     label="Estimated baseline",
    #                 )

    #                 if np.any(finite_baseline):
    #                     baseline_vals = baseline_trace[finite_baseline]
    #                     baseline_min = float(np.min(baseline_vals))
    #                     baseline_max = float(np.max(baseline_vals))
    #                     ymax = 2.0 * baseline_max
    #                     ymin_padding = 0.05 * max(abs(baseline_min), abs(baseline_max), 1e-12)
    #                     ymin = baseline_min - ymin_padding
    #                     if ymax <= ymin:
    #                         ymax = baseline_max + max(1e-6, ymin_padding)
    #                     ax.set_ylim(ymin, ymax)

    #                 if xmin is not None or xmax is not None:
    #                     x_lo = xmin
    #                     x_hi = xmax
    #                     if x_lo is not None and x_hi is not None and x_hi > x_lo:
    #                         ax.set_xlim(x_lo, x_hi)

    #                 ax.set_title(f"{sample_name} | {chrom_name}", fontsize=10)
    #                 ax.set_ylabel(f"Signal [{self.signal_unit}]")
    #                 ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))
    #                 ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    #                 ax.tick_params(axis="x", which="minor", bottom=True, length=3)
    #                 ax.tick_params(axis="y", which="minor", left=True, length=3)
    #                 ax.grid(True, which="both", alpha=0.2)
    #                 if axis_cursor == 1:
    #                     ax.legend(loc="best", fontsize="small", frameon=False)

    #     axes_flat[-1].set_xlabel(f"Time [{self.time_unit}]")
    #     plt.tight_layout()

    #     if save_path is not None:
    #         fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    #     return fig, axes
