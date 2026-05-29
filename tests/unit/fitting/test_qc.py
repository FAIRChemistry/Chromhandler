"""Unit tests for the QC layer (synthetic idata, no MCMC)."""
import arviz as az
import matplotlib.figure
import numpy as np


def synthetic_idata(n_trace=7, n_peak=3, n_chain=4, n_draw=200):
    """An idata mirroring the fitter's posterior group structure."""
    rng = np.random.default_rng(0)
    sh = (n_chain, n_draw)
    post = {
        "mu": rng.normal(3.0, 0.01, (*sh, n_peak)),
        "width": rng.normal(0.03, 0.001, (*sh, n_peak)),
        "skew": rng.normal(0.0, 0.1, (*sh, n_peak)),
        "area": rng.normal(1000.0, 50.0, (*sh, n_trace, n_peak)),
        "time_shift": rng.normal(0.0, 0.01, (*sh, n_trace)),
        "time_stretch": rng.normal(1.0, 0.01, (*sh, n_trace)),
        "noise": rng.normal(5.0, 0.5, (*sh, n_trace)),
        "baseline_intercept": rng.normal(1.0, 0.1, (*sh, n_trace)),
        "baseline_slope": rng.normal(0.0, 0.01, (*sh, n_trace)),
        "mu_warped": rng.normal(3.0, 0.01, (*sh, n_trace, n_peak)),
        "width_warped": rng.normal(0.03, 0.001, (*sh, n_trace, n_peak)),
    }
    stats = {"energy": rng.normal(0.0, 1.0, sh), "diverging": np.zeros(sh, dtype=bool)}
    return az.from_dict({"posterior": post, "sample_stats": stats})


def test_figure_helper_returns_matplotlib_figure():
    from chromhandler.fitting.qc import _figure

    pc = az.plot_forest(synthetic_idata(), var_names=["mu"])
    assert isinstance(_figure(pc), matplotlib.figure.Figure)


def test_plot_warp_returns_figure():
    from chromhandler.fitting.qc import plot_warp

    fig = plot_warp(synthetic_idata())
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_areas_returns_figure():
    from chromhandler.fitting.qc import plot_area_forest

    fig = plot_area_forest(synthetic_idata())
    assert isinstance(fig, matplotlib.figure.Figure)
