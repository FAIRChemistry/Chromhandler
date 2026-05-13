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

from chromhandler.fitting.posterior import diagnostics as _diagnostics_fn

if TYPE_CHECKING:
    from pathlib import Path

    import matplotlib.figure
    import pandas as pd

    from chromhandler.fitting.model import ModelConfig
    from chromhandler.fitting.prepared_dataset import PreparedDataset
    from chromhandler.fitting.priors import SkewNormalPriors


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
