import numpy as np

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.prepared_dataset import prepare_dataset
from chromhandler.fitting.priors import EMGPriors, PriorConfig, build_priors


def _toy_emg_dataset():
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 10.0, 0.02)
    peak = 1000.0 * np.exp(-0.5 * ((t - 5.0) / 0.05) ** 2)
    tail = np.where(t > 5.0, 400.0 * np.exp(-(t - 5.0) / 0.15), 0.0)
    s = peak + tail + 1.0 + rng.normal(0, 0.5, t.shape)
    pk = [PeakAnnotation(molecule_id="x", rt_min=4.7, rt_max=5.8,
                         mode="single", peak_model="emg")]
    bs = [BaselineAnnotation(rt_min=0.0, rt_max=1.0),
          BaselineAnnotation(rt_min=9.0, rt_max=10.0)]
    return prepare_dataset([t], [s], pk, bs)


def test_build_priors_emits_emg_priors():
    ds = _toy_emg_dataset()
    p = build_priors(ds, PriorConfig(signal_threshold=50.0))[0]
    assert isinstance(p, EMGPriors)
    assert p.emg_sigma_loc > 0 and p.emg_tau_loc > 0
    assert np.all(p.area_loc_per_trace > 0)
    assert p.area_log_scale == 1.0
    assert p.emg_mu_loc < 5.05  # mu_G sits left of the apex (~5.0)


def test_build_priors_mixed_types():
    from chromhandler.fitting.priors import SkewNormalPriors
    rng = np.random.default_rng(1)
    t = np.arange(0.0, 10.0, 0.02)
    s = (1000.0 * np.exp(-0.5 * ((t - 3.0) / 0.05) ** 2)
         + 1000.0 * np.exp(-0.5 * ((t - 6.0) / 0.05) ** 2) + 1.0 + rng.normal(0, 0.5, t.shape))
    pk = [PeakAnnotation(molecule_id="a", rt_min=2.7, rt_max=3.3, mode="single"),
          PeakAnnotation(molecule_id="b", rt_min=5.7, rt_max=6.3, mode="single", peak_model="emg")]
    bs = [BaselineAnnotation(rt_min=0.0, rt_max=1.0), BaselineAnnotation(rt_min=9.0, rt_max=10.0)]
    priors = build_priors(prepare_dataset([t], [s], pk, bs), PriorConfig(signal_threshold=50.0))
    assert isinstance(priors[0], SkewNormalPriors)
    assert isinstance(priors[1], EMGPriors)


def _mixed_dataset():
    rng = np.random.default_rng(3)
    t = np.arange(0.0, 10.0, 0.02)
    s = (1000.0 * np.exp(-0.5 * ((t - 3.0) / 0.05) ** 2)
         + 1000.0 * np.exp(-0.5 * ((t - 6.0) / 0.05) ** 2) + 1.0 + rng.normal(0, 0.5, t.shape))
    pk = [PeakAnnotation(molecule_id="a", rt_min=2.7, rt_max=3.3, mode="single"),
          PeakAnnotation(molecule_id="b", rt_min=5.7, rt_max=6.3, mode="single", peak_model="emg")]
    bs = [BaselineAnnotation(rt_min=0.0, rt_max=1.0), BaselineAnnotation(rt_min=9.0, rt_max=10.0)]
    return prepare_dataset([t], [s], pk, bs)


def test_summarise_priors_handles_mixed():
    from chromhandler.fitting.priors import summarise_priors
    pc = PriorConfig(signal_threshold=50.0)
    txt = summarise_priors(build_priors(_mixed_dataset(), pc), pc)
    assert "emg_mu" in txt and "emg_tau" in txt   # EMG peak rows
    assert "skew" in txt                          # SN peak rows


def test_plot_prior_overlay_handles_mixed():
    import arviz as az
    import matplotlib.figure

    from chromhandler.fitting.fitter import FitResult
    ds = _mixed_dataset()
    priors = build_priors(ds, PriorConfig(signal_threshold=50.0))
    r = FitResult(idata=az.from_dict({"posterior": {"mu": np.zeros((1, 2))}}),
                  dataset=ds, priors=priors, model_config=None)  # type: ignore[arg-type]
    assert isinstance(r.plot_prior_overlay(), matplotlib.figure.Figure)
