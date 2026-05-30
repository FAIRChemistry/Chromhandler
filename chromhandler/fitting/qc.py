"""Quality-control views over a fit's InferenceData.

Pure functions: each takes an arviz DataTree (`idata`) and returns a
matplotlib Figure or a metric dict. No FitResult/dataset dependency, so
they are unit-testable on a synthetic idata. Plots are chosen by parameter
ROLE so the view scales to the model's many parameters:
  - shape (mu/width/skew): few, interpretable      -> rank plots
  - area[trace, peak]:      many, the estimands      -> forest
  - warp + noise (per trace): nuisance               -> forest
  - everything:             R-hat / ESS overview     -> scatter + scalar gate
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import arviz
import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure


def _figure(pc: Any) -> matplotlib.figure.Figure:
    """Extract the matplotlib Figure from an arviz 1.x PlotCollection."""
    return pc.viz["figure"].item()


def _present(idata: arviz.InferenceData, names: tuple[str, ...]) -> list[str]:
    """Subset of ``names`` actually present in the posterior."""
    data_vars = idata.posterior.data_vars  # type: ignore[attr-defined]
    return [n for n in names if n in data_vars]


def plot_warp(idata: arviz.InferenceData) -> matplotlib.figure.Figure:
    """Forest of per-trace warp + noise.

    Answers: are the time shifts/stretches small and zero-centred, and is any
    single trace an outlier? A large ``time_shift[trace]`` flags a problematic
    injection; an inflated ``noise[trace]`` flags model misfit on that trace.
    """
    names = _present(idata, ("time_shift", "time_stretch", "noise"))
    return _figure(arviz.plot_forest(idata, var_names=names))


def plot_area_forest(idata: arviz.InferenceData) -> matplotlib.figure.Figure:
    """Forest of area[trace, peak] — the estimands (-> concentration).

    One row per (trace, peak) with point + interval, so the whole kinetic
    series and its uncertainties are visible at a glance and any area with a
    too-wide or shifted interval stands out.
    """
    return _figure(arviz.plot_forest(idata, var_names=["area"]))


def plot_convergence(
    idata: arviz.InferenceData,
    var_names: tuple[str, ...] = ("mu", "width", "skew", "emg_mu", "emg_sigma", "emg_tau"),
) -> matplotlib.figure.Figure:
    """Rank plots for the shared per-peak shape parameters.

    Rank plots (not trace caterpillars) are arviz's recommended convergence
    diagnostic — they reveal between-chain bias that overlapping traces hide.
    Defaults to the small, interpretable shape set so the figure never blows up.
    """
    names = _present(idata, var_names)
    return _figure(arviz.plot_rank(idata, var_names=names))


# ---------------------------------------------------------------------------
# Role grouping
# ---------------------------------------------------------------------------

# Map a posterior variable name to its QC role group.
_GROUP = {
    "mu": "shape", "width": "shape", "skew": "shape",
    "emg_mu": "shape", "emg_sigma": "shape", "emg_tau": "shape",
    "area": "area",
    "time_shift": "warp", "time_stretch": "warp",
    "noise": "noise",
    "baseline_intercept": "baseline", "baseline_slope": "baseline",
}


def _var_of(label: str) -> str:
    """'area[0, 0]' -> 'area'."""
    return label.split("[", 1)[0]


def _bfmi(idata: Any) -> np.ndarray:
    """Per-chain BFMI from sample_stats.energy (empty array if absent)."""
    if not (hasattr(idata, "sample_stats") and "energy" in idata.sample_stats):
        return np.array([])
    e = np.asarray(idata.sample_stats["energy"])  # type: ignore[reportUnknownArgumentType]  # [chain, draw]
    num = np.mean(np.diff(e, axis=1) ** 2, axis=1)
    den = np.var(e, axis=1)
    return num / np.where(den > 0, den, np.nan)


def _user_var_names(idata: Any) -> list[str]:
    return [
        str(n) for n in idata.posterior.data_vars
        if not (str(n).endswith("_raw") or str(n).endswith("_warped"))
    ]


def qc_summary(idata: Any) -> dict[str, Any]:
    """Scalar QC gate + per-role worst R-hat / min ESS.

    Returns: fit_healthy(bool), n_divergent(int), bfmi_min(float),
    rhat_max(float), ess_min(float), groups({group: {rhat_max, ess_min}}).
    Healthy iff r_hat <= 1.01 and ess_bulk > 400 everywhere, no divergences,
    and bfmi_min >= 0.3.
    """
    names = _user_var_names(idata)
    summ = arviz.summary(idata, var_names=names)
    rhat = summ["r_hat"].astype(float)
    ess = summ["ess_bulk"].astype(float)

    groups: dict[str, dict[str, float]] = {}
    for label in summ.index:
        grp = _GROUP.get(_var_of(str(label)), "other")  # type: ignore[reportUnknownArgumentType]
        d = groups.setdefault(grp, {"rhat_max": -np.inf, "ess_min": np.inf})
        d["rhat_max"] = max(d["rhat_max"], float(rhat[label]))  # type: ignore[reportUnknownArgumentType]
        d["ess_min"] = min(d["ess_min"], float(ess[label]))  # type: ignore[reportUnknownArgumentType]

    n_div = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        n_div = int(np.asarray(idata.sample_stats["diverging"]).sum())  # type: ignore[reportUnknownArgumentType]
    bfmi = _bfmi(idata)
    bfmi_min = float(np.nanmin(bfmi)) if bfmi.size else float("nan")

    rhat_max = float(rhat.max())  # type: ignore[reportUnknownArgumentType]
    ess_min = float(ess.min())  # type: ignore[reportUnknownArgumentType]
    rhat_ok = bool(np.isnan(rhat_max) or rhat_max <= 1.01)
    bfmi_ok = bool(np.isnan(bfmi_min) or bfmi_min >= 0.3)
    fit_healthy = bool(rhat_ok and ess_min > 400 and n_div == 0 and bfmi_ok)

    return {
        "fit_healthy": fit_healthy,
        "n_divergent": n_div,
        "bfmi_min": bfmi_min,
        "rhat_max": rhat_max,
        "ess_min": ess_min,
        "groups": groups,
    }


def plot_qc_overview(idata: Any) -> matplotlib.figure.Figure:
    """R-hat-vs-ESS scatter over all params, coloured by role group.

    Catches any single problematic parameter among hundreds without one
    subplot per scalar. Reference lines at r_hat = 1.01 and ESS = 400.
    """
    import matplotlib.pyplot as plt

    summ = arviz.summary(idata, var_names=_user_var_names(idata))
    groups = [_GROUP.get(_var_of(str(i)), "other") for i in summ.index]  # type: ignore[reportUnknownArgumentType]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for grp in sorted(set(groups)):
        m = np.asarray([g == grp for g in groups])
        ax.scatter(
            summ["ess_bulk"][m].astype(float),  # type: ignore[reportUnknownArgumentType]
            summ["r_hat"][m].astype(float),  # type: ignore[reportUnknownArgumentType]
            s=18, alpha=0.7, label=grp,
        )
    ax.axhline(1.01, color="r", lw=0.8, ls="--")
    ax.axvline(400, color="r", lw=0.8, ls="--")
    ax.set_xlabel("ESS (bulk)")
    ax.set_ylabel("R-hat")
    ax.set_title("Convergence overview (target: r_hat < 1.01, ESS > 400)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
