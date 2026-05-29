import numpyro
numpyro.set_host_device_count(4)
import numpy as np

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting import ModelConfig, fit
from chromhandler.fitting.prepared_dataset import prepare_dataset
from chromhandler.fitting.priors import PriorConfig


def _emg_trace(mu_g=5.0, sigma=0.05, tau=0.12, area=3000.0, seed=0):
    from scipy.stats import exponnorm
    rng = np.random.default_rng(seed)
    t = np.arange(2.0, 8.0, 0.02)
    s = area * exponnorm.pdf(t, tau / sigma, loc=mu_g, scale=sigma) + 2.0 + rng.normal(0, 1.0, t.shape)
    return t, s


def test_emg_recovers_known_tau_no_divergence():
    t, s0 = _emg_trace(tau=0.12, seed=0)
    _, s1 = _emg_trace(tau=0.12, seed=1)
    pk = [PeakAnnotation(molecule_id="x", rt_min=4.6, rt_max=6.0, mode="single", peak_model="emg")]
    bs = [BaselineAnnotation(rt_min=2.0, rt_max=2.5), BaselineAnnotation(rt_min=7.5, rt_max=8.0)]
    ds = prepare_dataset([t, t], [s0, s1], pk, bs)
    r = fit(ds, prior_config=PriorConfig(signal_threshold=50.0),
            model_config=ModelConfig(num_warmup=400, num_samples=400, num_chains=2, seed=0))
    tau_post = float(np.asarray(r.idata.posterior["emg_tau"]).mean())
    assert 0.08 < tau_post < 0.18                  # recovers true tau ~0.12
    assert r.diagnostics()["n_divergent"] == 0     # non-centred geometry OK


def test_all_skew_normal_unchanged():
    rng = np.random.default_rng(2)
    t = np.arange(2.0, 8.0, 0.02)
    s = 3000.0 * np.exp(-0.5 * ((t - 5.0) / 0.05) ** 2) + 2.0 + rng.normal(0, 1.0, t.shape)
    pk = [PeakAnnotation(molecule_id="x", rt_min=4.6, rt_max=5.6, mode="single")]
    bs = [BaselineAnnotation(rt_min=2.0, rt_max=2.5), BaselineAnnotation(rt_min=7.5, rt_max=8.0)]
    ds = prepare_dataset([t, t], [s, s], pk, bs)
    r = fit(ds, prior_config=PriorConfig(signal_threshold=50.0),
            model_config=ModelConfig(num_warmup=200, num_samples=200, num_chains=2, seed=0))
    assert "mu" in r.idata.posterior and "emg_tau" not in r.idata.posterior


def _load_atp():
    from pathlib import Path

    p = Path("tests/fixtures/atp_tailing/ATP_sig.csv")
    data = np.genfromtxt(p, delimiter=",", names=True)
    t = data["RTminutes__NOT_USED_BY_IMPORT"]
    s = data["260"]
    m = (t >= 4.6) & (t <= 6.4)
    return t[m], s[m]


def test_atp_emg_area_matches_trapezoid_and_beats_sn_tail():
    t, s = _load_atp()
    # Pre-peak baseline only: the ATP EMG tail is still elevated at 6.4 min
    # (~1.16 M counts), so the post-peak region is not flat enough for OLS.
    # Two pre-peak windows [4.60, 4.68] and [4.76, 4.87] are both flat at
    # ~24 k counts (std < 400) and are well separated from the peak window.
    bs = [
        BaselineAnnotation(rt_min=4.60, rt_max=4.68),
        BaselineAnnotation(rt_min=4.76, rt_max=4.87),
    ]
    cfg = ModelConfig(num_warmup=400, num_samples=400, num_chains=2, seed=0)
    out = {}
    for model_name in ("skew_normal", "emg"):
        pk = [
            PeakAnnotation(
                molecule_id="ATP",
                rt_min=4.9,
                rt_max=5.9,
                mode="single",
                peak_model=model_name,
            )
        ]
        ds = prepare_dataset([t], [s], pk, bs)
        out[model_name] = (
            fit(
                ds,
                prior_config=PriorConfig(signal_threshold=1e6),
                model_config=cfg,
            ),
            ds,
        )
    r, ds = out["emg"]
    area_emg = float(np.asarray(r.idata.posterior["area"]).mean())
    b_int = float(np.asarray(r.idata.posterior["baseline_intercept"]).mean())
    b_slp = float(np.asarray(r.idata.posterior["baseline_slope"]).mean())
    tt = np.asarray(ds.time)[0]
    ss = np.asarray(ds.signal)[0]
    vmask = np.asarray(ds.valid_mask)[0]
    win = vmask & (tt >= 4.9) & (tt <= 5.9)
    trap = float(np.trapezoid((ss - (b_int + b_slp * tt))[win], tt[win]))
    assert abs(area_emg - trap) / trap < 0.05
    noise_emg = float(np.asarray(out["emg"][0].idata.posterior["noise"]).mean())
    noise_sn = float(np.asarray(out["skew_normal"][0].idata.posterior["noise"]).mean())
    assert noise_emg < noise_sn
