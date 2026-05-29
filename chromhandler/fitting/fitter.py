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
        """Write the full InferenceData (DataTree) to netCDF.

        Whatever groups are currently in `idata` get saved — call
        `plot_fit()` / `plot_prior_predictive()` first if you want the
        predictive samples persisted.
        """
        self.idata.to_netcdf(str(path), engine="h5netcdf")

    def _default_user_facing_var_names(self) -> list[str]:
        """All posterior variables except internal ``*_raw`` sample sites.

        The model exposes physical quantities as ``numpyro.deterministic``
        sites and samples them via ``Normal(0, 1)`` reparameterisation
        sites suffixed ``_raw``. Plotting and summary defaults skip the
        raw sites so users see only the natural-space parameters.
        """
        data_vars = self.idata.posterior.data_vars  # type: ignore[attr-defined]
        return [str(n) for n in data_vars if not str(n).endswith("_raw")]  # type: ignore[reportUnknownArgumentType]

    def summary(self, var_names: list[str] | None = None) -> pd.DataFrame:
        """ArviZ summary (mean / sd / hdi / r_hat / ess) as a DataFrame.

        Defaults to all user-facing deterministic sites (excludes ``*_raw``).
        """
        if var_names is None:
            var_names = self._default_user_facing_var_names()
        return arviz.summary(self.idata, var_names=var_names)  # type: ignore[return-value]

    def diagnostics(self) -> dict[str, Any]:
        """Quick convergence summary dict (see posterior.diagnostics)."""
        return _diagnostics_fn(self.idata)

    def plot_baseline_prior(
        self,
        *,
        overlay: str = "single",
        ax_size: tuple[float, float] = (10.0, 2.8),
        save: Path | str | None = None,
    ) -> matplotlib.figure.Figure:
        """Plot the baseline prior (median + uncertainty band) per group.

        Thin wrapper over
        :func:`chromhandler.fitting.plotting.plot_baseline_prior`; see that
        function for parameter details.
        """
        from chromhandler.fitting.plotting import (
            plot_baseline_prior as _plot_baseline_prior,
        )

        return _plot_baseline_prior(
            self.dataset, overlay=overlay, ax_size=ax_size, save=save,
        )

    def plot_traces(
        self,
        var_names: list[str] | None = None,
        *,
        compact: bool = True,
        combined: bool = False,
        figsize: tuple[float, float] | None = None,
    ) -> matplotlib.figure.Figure:
        """ArviZ trace plot for the listed variables (or all user-facing if None).

        Each chain is drawn as a separate line so divergent or stuck chains
        are visible.

        Args:
            var_names: Variables to plot. ``None`` plots all user-facing
                deterministic sites (excludes internal ``*_raw`` reparam
                draws).
            compact: Accepted for backwards compatibility; ignored under
                ArviZ 1.x (the new ``plot_trace`` lays out variables
                automatically).
            combined: Accepted for backwards compatibility; ignored under
                ArviZ 1.x.
            figsize: Accepted for backwards compatibility; ignored under
                ArviZ 1.x (figure sizing is handled by the backend).
        """
        if var_names is None:
            var_names = self._default_user_facing_var_names()
        # ArviZ 1.x plot_trace returns a PlotCollection (not a 2-D axes
        # array). The old compact/combined/figsize kwargs no longer exist.
        pc = arviz.plot_trace(  # type: ignore[call-overload]
            self.idata,
            var_names=var_names,
        )
        # Retrieve the matplotlib Figure from the first axes object stored
        # in the PlotCollection's viz DataTree.
        plot_ds = pc.viz["plot"].ds  # type: ignore[index]
        first_var = next(iter(plot_ds.data_vars))  # type: ignore[attr-defined]
        first_ax = plot_ds[first_var].values.flat[0]  # type: ignore[index]
        fig: matplotlib.figure.Figure = first_ax.figure  # type: ignore[assignment]
        # Prevent axis titles from clipping the plot above (common when
        # many rows are packed into the default auto-sized figure).
        fig.tight_layout()
        return fig

    def plot_prior_overlay(self) -> matplotlib.figure.Figure:
        """Per-(peak, trace) panel: baseline-subtracted data + prior median
        skew-normal scaled to each trace's prior area centre.

        Supported traces (gate passed) show ``area_loc * SN(mu, width, skew)``
        as a black dashed curve; unsupported traces (gate failed) show no
        prior curve since their area prior is centred on zero.
        """
        import matplotlib.pyplot as plt

        from chromhandler.fitting.skew_normal import density_cp

        dataset = self.dataset
        priors_list = self.priors
        n_peak = len(priors_list)
        n_trace = dataset.n_trace

        fig, axes = plt.subplots(
            n_peak, n_trace,
            figsize=(3.5 * n_trace, 2.8 * n_peak),
            squeeze=False,
        )

        for peak_idx, p in enumerate(priors_list):
            ann = dataset.peak_annotations[peak_idx]
            t_dense = np.linspace(ann.rt_min, ann.rt_max, 500)
            _mu = np.asarray(p.mu_loc)
            _width = np.asarray(p.width_loc)
            _skew = np.asarray(p.skew_loc)
            sn_unit = np.asarray(density_cp(t_dense, _mu, _width, _skew))  # type: ignore[arg-type]
            for tr in range(n_trace):
                ax = axes[peak_idx, tr]
                t = dataset.time[tr]
                s = dataset.signal[tr]
                bs = s - (dataset.baseline_intercept[tr] + dataset.baseline_slope[tr] * t)
                mask = ((t >= ann.rt_min) & (t <= ann.rt_max) & np.isfinite(bs))
                ax.plot(t[mask], bs[mask], color="C0", lw=1.0, label="data")
                if p.has_support_per_trace[tr]:
                    area_prior = float(p.area_loc_per_trace[tr])
                    ax.plot(t_dense, area_prior * sn_unit, "k--", lw=1.2, label="prior loc")
                else:
                    ax.text(
                        0.02, 0.95, "no support", transform=ax.transAxes,
                        fontsize=7, va="top", color="0.4",
                    )
                ax.set_title(f"trace {dataset.trace_ids[tr]} (peak {peak_idx})", fontsize=8)
                ax.axhline(0, color="k", lw=0.3, alpha=0.3)
                if tr == 0 and peak_idx == 0:
                    ax.legend(fontsize=7)
        fig.tight_layout()
        return fig

    def plot_fit(self) -> matplotlib.figure.Figure:
        """Posterior predictive 95% HDI band + median + observed data per trace.

        Lazily computes posterior predictive on first call; caches in `idata`.

        TODO(doublet): for doublet peaks, overlay separate dashed lines for
        left and right components.
        """
        if "posterior_predictive" not in self.idata.children:
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
        if "prior_predictive" not in self.idata.children:
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
