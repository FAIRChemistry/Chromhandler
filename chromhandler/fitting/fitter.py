"""User-facing fitter entry point.

Exposes ``fit()`` which orchestrates build_priors -> run_mcmc -> FitResult,
and the ``FitResult`` class with debug-plot methods.

Single-mode peaks only at present; doublet hooks documented inline.
TODO(doublet): extend plot methods to overlay right components when ready.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import arviz
import numpy as np

from chromhandler.fitting.model import ModelConfig
from chromhandler.fitting.posterior import (
    compute_posterior_predictive as _compute_pp,
)
from chromhandler.fitting.posterior import (
    compute_prior_predictive as _compute_prior_pp,
)
from chromhandler.fitting.posterior import (
    diagnostics as _diagnostics_fn,
)

if TYPE_CHECKING:
    from pathlib import Path

    import matplotlib.figure
    import pandas as pd

    from chromhandler.fitting.prepared_dataset import PreparedDataset
    from chromhandler.fitting.priors import PriorConfig, SkewNormalPriors


@dataclass(frozen=False)  # mutable: lazy groups added to idata over time
class FitResult:
    """Bundle of MCMC output, original inputs, and debug-plot methods.

    Attributes:
        idata: ArviZ InferenceData. `posterior` and `observed_data` are
            present from the moment `fit()` returns. `posterior_predictive`
            is added on the first `plot_fit()` call (lazy); `prior` and
            `prior_predictive` on the first `plot_prior_predictive()` call.
        dataset: The PreparedDataset that was fit.
        priors: The SkewNormalPriors that were used.
        model_config: The ModelConfig that was used.
    """

    idata: arviz.InferenceData
    dataset: PreparedDataset
    priors: list[SkewNormalPriors]
    model_config: ModelConfig

    def save(self, path: Path | str) -> None:
        """Write the full InferenceData to netCDF.

        Whatever groups are currently in `idata` get saved — call
        `plot_fit()` / `plot_prior_predictive()` first if you want the
        predictive samples persisted.
        """
        self.idata.to_netcdf(str(path))

    def summary(self, var_names: list[str] | None = None) -> pd.DataFrame:
        """ArviZ summary (mean / sd / hdi / r_hat / ess) as a DataFrame."""
        return arviz.summary(self.idata, var_names=var_names)  # type: ignore[return-value]

    def diagnostics(self) -> dict[str, Any]:
        """Quick convergence summary dict (see posterior.diagnostics)."""
        return _diagnostics_fn(self.idata)

    def plot_traces(self, var_names: list[str] | None = None) -> matplotlib.figure.Figure:
        """ArviZ trace plot for the listed variables (or all if None).

        Uses ``combined=True`` so all chains are merged before KDE estimation,
        which avoids OverflowError when individual chains are very short.
        """
        axes = arviz.plot_trace(self.idata, var_names=var_names, combined=True)  # type: ignore[call-overload]
        # arviz returns a 2D ndarray of Axes; grab the parent Figure
        if hasattr(axes, "flat"):
            return axes.flat[0].figure  # type: ignore[return-value]
        return axes[0].figure if hasattr(axes, "__iter__") else axes.figure  # type: ignore[return-value]

    def plot_prior_overlay(self) -> matplotlib.figure.Figure:
        """For each non-control trace, plot data + prior loc curve at the
        per-trace amplitude. Single-mode peaks only.

        TODO(doublet): when doublet ships, add a right-component dashed
        curve in panels for doublet peaks.
        """
        import matplotlib.pyplot as plt

        from chromhandler.fitting.skew_normal import density_cp

        dataset = self.dataset
        priors_list = self.priors
        n_peak = len(priors_list)
        non_control_idx = np.where(~dataset.is_control)[0]

        fig, axes = plt.subplots(
            n_peak, len(non_control_idx),
            figsize=(3.5 * len(non_control_idx), 2.8 * n_peak),
            squeeze=False,
        )

        for peak_idx, p in enumerate(priors_list):
            sigma_loc = float(np.exp(p.log_sigma_left_loc))
            t_dense = np.linspace(p.mu_left_low, p.mu_left_high, 500)
            _mu = np.asarray(p.mu_left_loc)
            _sig = np.asarray(sigma_loc)
            _g1 = np.asarray(p.gamma1_left_loc)
            sn_unit = np.asarray(density_cp(t_dense, _mu, _sig, _g1))  # type: ignore[arg-type]
            for col, tr in enumerate(non_control_idx):
                ax = axes[peak_idx, col]
                t = dataset.time[tr]
                s = dataset.signal[tr]
                bs = s - (dataset.baseline_intercept[tr] + dataset.baseline_slope[tr] * t)
                mask = ((t >= p.mu_left_low) & (t <= p.mu_left_high) & np.isfinite(bs))
                ax.plot(t[mask], bs[mask], color="C0", lw=1.0, label="data")
                A = float(np.exp(p.log_A_left_loc_per_trace[tr]))
                ax.plot(t_dense, A * sn_unit, "k--", lw=1.2, label="prior loc")
                ax.set_title(f"trace {dataset.trace_ids[tr]} (peak {peak_idx})", fontsize=8)
                ax.axhline(0, color="k", lw=0.3, alpha=0.3)
                if col == 0 and peak_idx == 0:
                    ax.legend(fontsize=7)
        fig.tight_layout()
        return fig

    def plot_fit(self) -> matplotlib.figure.Figure:
        """Posterior predictive 95% HDI band + median + observed data per trace.

        Lazily computes posterior predictive on first call; caches in `idata`.

        TODO(doublet): for doublet peaks, overlay separate dashed lines for
        left and right components.
        """
        if not hasattr(self.idata, "posterior_predictive"):
            _compute_pp(self.idata, self.dataset, self.priors, self.model_config)

        return self._plot_band(
            samples_group="posterior_predictive",
            label="posterior",
            band_color="tab:blue",
        )

    def plot_prior_predictive(self) -> matplotlib.figure.Figure:
        """Prior predictive 95% HDI band + median + observed data per trace.

        Lazily computes prior + prior_predictive on first call; caches both.
        """
        if not hasattr(self.idata, "prior_predictive"):
            _compute_prior_pp(self.idata, self.dataset, self.priors, self.model_config)

        return self._plot_band(
            samples_group="prior_predictive",
            label="prior",
            band_color="tab:purple",
        )

    def _plot_band(
        self,
        samples_group: str,
        label: str,
        band_color: str,
    ) -> matplotlib.figure.Figure:
        """Shared implementation for plot_fit + plot_prior_predictive.

        Layout: one axis per (trace, annotated peak window). Each axis
        is restricted to ``[ann.rt_min, ann.rt_max]`` so narrow peaks
        are actually legible and the band is comparable to data inside
        the window the model is fitting.
        """
        import matplotlib.pyplot as plt

        group = getattr(self.idata, samples_group)
        obs = np.asarray(group["obs"])  # [chain, draw, trace, time_idx]
        flat = obs.reshape(-1, obs.shape[-2], obs.shape[-1])  # [draws, trace, time]
        n_trace = self.dataset.n_trace
        peak_anns = self.dataset.peak_annotations
        n_peak = len(peak_anns)
        if n_peak == 0:
            # Degenerate dataset: no peak windows. Fall back to a single
            # full-trace strip so the method still produces a figure.
            peak_anns_for_plot: list[Any] = [None]
            n_cols = 1
        else:
            peak_anns_for_plot = list(peak_anns)
            n_cols = n_peak

        fig, axes = plt.subplots(
            n_trace, n_cols,
            figsize=(3.8 * n_cols, 2.2 * n_trace),
            squeeze=False, sharex=False,
        )

        for tr in range(n_trace):
            t = self.dataset.time[tr]
            s = self.dataset.signal[tr]
            for col, ann in enumerate(peak_anns_for_plot):
                ax = axes[tr, col]
                if ann is None:
                    mask = np.isfinite(s)
                else:
                    mask = (
                        (t >= ann.rt_min) & (t <= ann.rt_max)
                        & np.isfinite(s)
                    )

                samples_tr = flat[:, tr, :]  # [draws, time]
                try:
                    hdi_da = arviz.hdi(samples_tr[None, :, :], hdi_prob=0.95)
                    if hasattr(hdi_da, "data_vars"):
                        hdi_arr = np.asarray(  # type: ignore[reportUnknownArgumentType]
                            next(iter(hdi_da.data_vars.values()))  # type: ignore[attr-defined,reportUnknownArgumentType]
                        )
                    else:
                        hdi_arr = np.asarray(hdi_da)
                    if hdi_arr.shape[0] == 2 and hdi_arr.shape[-1] != 2:
                        hdi_arr = hdi_arr.T
                except Exception:
                    hdi_arr = np.quantile(
                        samples_tr, [0.025, 0.975], axis=0,
                    ).T  # [time, 2]
                median = np.median(samples_tr, axis=0)

                ax.fill_between(
                    t[mask], hdi_arr[mask, 0], hdi_arr[mask, 1],
                    color=band_color, alpha=0.35,
                    label=f"{label} 95% HDI",
                )
                ax.plot(
                    t[mask], median[mask],
                    color=band_color, lw=1.4, label=f"{label} median",
                )
                ax.plot(t[mask], s[mask], color="k", lw=0.8, label="data")
                if ann is not None:
                    ax.axvline(ann.rt_min, color="0.6", lw=0.5, ls="--")
                    ax.axvline(ann.rt_max, color="0.6", lw=0.5, ls="--")
                    title = (
                        f"{self.dataset.trace_ids[tr]} — "
                        f"{ann.molecule_id} ({ann.rt_min:.2f}-{ann.rt_max:.2f})"
                    )
                else:
                    title = self.dataset.trace_ids[tr]
                ax.set_title(title, fontsize=8)
                if tr == 0 and col == 0:
                    ax.legend(fontsize=7)
        fig.tight_layout()
        return fig


def fit(
    dataset: PreparedDataset,
    *,
    prior_config: PriorConfig | None = None,
    model_config: ModelConfig | None = None,
) -> FitResult:
    """Build priors, run MCMC, return a FitResult.

    Args:
        dataset: PreparedDataset to fit.
        prior_config: Optional PriorConfig override. Defaults to PriorConfig().
        model_config: Optional ModelConfig override. Defaults to ModelConfig().

    Returns:
        FitResult with .plot_* methods and .idata for raw access.
    """
    from chromhandler.fitting.model import run_mcmc
    from chromhandler.fitting.priors import PriorConfig, build_priors

    pc = prior_config if prior_config is not None else PriorConfig()
    mc = model_config if model_config is not None else ModelConfig()
    priors = build_priors(dataset, config=pc)
    idata = run_mcmc(dataset, priors, mc)
    return FitResult(idata=idata, dataset=dataset, priors=priors, model_config=mc)
