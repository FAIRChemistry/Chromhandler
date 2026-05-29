"""EMG vs skew-normal comparison on the ATP tailing-peak fixture.

Loads the ATP chromatogram (apex ~5.146 min, strong right tail), fits both
``peak_model="emg"`` and ``peak_model="skew_normal"``, prints areas + noise
+ diagnostics for each, and saves an overlay plot to
``dev/emg_vs_skewnormal.png``.

Run:
    uv run python dev/emg_vs_skewnormal.py
"""

import math
from pathlib import Path

import numpyro

numpyro.set_host_device_count(8)

# All chromhandler / JAX imports must follow set_host_device_count.
import numpy as np  # noqa: E402

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation  # noqa: E402
from chromhandler.fitting import ModelConfig, fit  # noqa: E402
from chromhandler.fitting.prepared_dataset import PreparedDataset, prepare_dataset  # noqa: E402
from chromhandler.fitting.priors import PriorConfig  # noqa: E402

_FIXTURE = Path("tests/fixtures/atp_tailing/ATP_sig.csv")

# Analysis window (minutes)
_T_MIN, _T_MAX = 4.6, 6.4

# Peak window (wider right edge to capture EMG tail)
_PEAK_MIN, _PEAK_MAX = 4.9, 5.9

# Baseline: two pre-peak flat regions (std < 400 counts each)
_BASELINES = [
    BaselineAnnotation(rt_min=4.60, rt_max=4.68),
    BaselineAnnotation(rt_min=4.76, rt_max=4.87),
]

_MCMC_CFG = ModelConfig(num_warmup=400, num_samples=400, num_chains=2, seed=0)
_PRIOR_CFG = PriorConfig(signal_threshold=1e6)


def load_atp() -> tuple[np.ndarray, np.ndarray]:
    """Load ATP trace, return (time, signal) clipped to [_T_MIN, _T_MAX]."""
    raw = np.genfromtxt(_FIXTURE, delimiter=",", names=True)
    t = raw["RTminutes__NOT_USED_BY_IMPORT"]
    s = raw["260"]
    m = (t >= _T_MIN) & (t <= _T_MAX)
    return t[m], s[m]


def run_fits(
    t: np.ndarray, s: np.ndarray
) -> dict[str, tuple[object, PreparedDataset]]:
    """Fit both models; return {model_name: (FitResult, PreparedDataset)}."""
    results = {}
    for model_name in ("skew_normal", "emg"):
        pk = [
            PeakAnnotation(
                molecule_id="ATP",
                rt_min=_PEAK_MIN,
                rt_max=_PEAK_MAX,
                mode="single",
                peak_model=model_name,
            )
        ]
        ds = prepare_dataset([t], [s], pk, _BASELINES)
        r = fit(ds, prior_config=_PRIOR_CFG, model_config=_MCMC_CFG)
        results[model_name] = (r, ds)
    return results


def print_comparison(
    results: dict[str, tuple[object, PreparedDataset]],
) -> None:
    """Print area, noise, and diagnostics side-by-side."""
    print("\n" + "=" * 60)
    print("ATP tailing peak: EMG vs skew-normal comparison")
    print("=" * 60)
    for model_name, (r, _) in results.items():
        area = float(np.asarray(r.idata.posterior["area"]).mean())
        noise = float(np.asarray(r.idata.posterior["noise"]).mean())
        try:
            diag = r.diagnostics()
        except Exception as exc:
            diag = {"error": str(exc)}
        label = model_name.replace("_", "-").upper()
        print(f"\n  [{label}]")
        print(f"    area (posterior mean): {area:.4e}")
        print(f"    noise (posterior mean): {noise:.4e}")
        print("    diagnostics:")
        for k, v in diag.items():
            if isinstance(v, float):
                print(f"      {k:<22} {v:.4g}")
            else:
                print(f"      {k:<22} {v}")
    print()


def _peak_curve_emg(
    t: np.ndarray, post: object, area: float
) -> np.ndarray:
    """Posterior-median EMG peak curve on array t."""
    from scipy.stats import exponnorm

    mu_e = float(np.asarray(post["emg_mu"]).mean())
    sigma_e = float(np.asarray(post["emg_sigma"]).mean())
    tau_e = float(np.asarray(post["emg_tau"]).mean())
    # scipy exponnorm: K = tau/sigma, loc=mu, scale=sigma
    K = tau_e / sigma_e
    return area * exponnorm.pdf(t, K, loc=mu_e, scale=sigma_e)


def _peak_curve_sn(
    t: np.ndarray, post: object, area: float
) -> np.ndarray:
    """Posterior-median skew-normal peak curve on array t using CP->DP.

    The model stores centred parameters (mu, width=sigma_cp, skew=gamma1).
    Convert to direct parameters (xi, omega, alpha) via the closed-form
    cp_to_dp bijection, then evaluate with scipy.stats.skewnorm.
    """
    from scipy.stats import skewnorm

    mu = float(np.asarray(post["mu"]).mean())
    width = float(np.asarray(post["width"]).mean())  # sigma_cp
    gamma1 = float(np.asarray(post["skew"]).mean())  # skewness coefficient

    # cp_to_dp (same arithmetic as chromhandler/fitting/skew_normal.py)
    # Clip gamma1 to valid range to avoid domain issues.
    gamma1_max = ((4.0 - math.pi) / 2.0) * (math.sqrt(2.0 / math.pi) ** 3) / (
        1.0 - 2.0 / math.pi
    ) ** 1.5
    gamma1 = float(np.clip(gamma1, -0.999 * gamma1_max, 0.999 * gamma1_max))
    c = (2.0 * gamma1 / (4.0 - math.pi)) ** (1.0 / 3.0)
    delta = c * math.sqrt(math.pi / 2.0) / math.sqrt(1.0 + c**2 * math.pi / 2.0)
    alpha = delta / math.sqrt(1.0 - delta**2)
    # omega from sigma_cp
    b = math.sqrt(2.0 / math.pi)
    omega = width / math.sqrt(1.0 - b**2 * delta**2)
    # xi from mu_cp
    xi = mu - omega * b * delta
    return area * skewnorm.pdf(t, alpha, loc=xi, scale=omega)


def build_overlay(
    t: np.ndarray,
    s: np.ndarray,
    results: dict[str, tuple[object, PreparedDataset]],
    out_path: Path,
) -> None:
    """Posterior-median fit overlay + residuals saved to out_path."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_fit, ax_res = axes

    # Raw data
    ax_fit.plot(t, s, color="0.6", lw=0.8, label="data")

    colors = {"skew_normal": "#2e86ab", "emg": "#e84855"}
    labels = {"skew_normal": "skew-normal", "emg": "EMG"}

    t_fine = np.linspace(_PEAK_MIN, _PEAK_MAX, 2000)

    for model_name, (r, _ds) in results.items():
        post = r.idata.posterior
        b_int = float(np.asarray(post["baseline_intercept"]).mean())
        b_slp = float(np.asarray(post["baseline_slope"]).mean())
        area = float(np.asarray(post["area"]).mean())
        baseline_fine = b_int + b_slp * t_fine
        baseline = b_int + b_slp * t

        if model_name == "emg":
            peak_fine = _peak_curve_emg(t_fine, post, area)
            peak_at_t = _peak_curve_emg(t, post, area)
        else:
            peak_fine = _peak_curve_sn(t_fine, post, area)
            peak_at_t = _peak_curve_sn(t, post, area)

        model_signal_fine = baseline_fine + peak_fine
        color = colors[model_name]
        ax_fit.plot(t_fine, model_signal_fine, color=color, lw=1.8,
                    label=f"{labels[model_name]} (area={area:.3e})")

        # Residuals normalised by posterior-mean noise
        predicted = baseline + peak_at_t
        noise_val = float(np.asarray(post["noise"]).mean())
        ax_res.plot(t, (s - predicted) / noise_val, color=color, lw=0.8, alpha=0.7,
                    label=f"{labels[model_name]} residual/noise")

    # Annotate baseline windows (light shading)
    for ba in _BASELINES:
        ax_fit.axvspan(ba.rt_min, ba.rt_max, alpha=0.08, color="green")
    ax_fit.axvspan(_PEAK_MIN, _PEAK_MAX, alpha=0.05, color="orange")

    ax_fit.set_ylabel("Signal (AU)")
    ax_fit.legend(fontsize=9)
    ax_fit.set_title("ATP tailing peak: EMG vs skew-normal (posterior median)")

    ax_res.axhline(0, color="k", lw=0.6)
    ax_res.axhline(+1, color="k", lw=0.4, ls="--")
    ax_res.axhline(-1, color="k", lw=0.4, ls="--")
    ax_res.set_ylabel("Residual / noise")
    ax_res.set_xlabel("Retention time (min)")
    ax_res.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved overlay to {out_path}")


def main() -> None:
    t, s = load_atp()
    print(f"Loaded ATP fixture: {len(t)} points in [{_T_MIN}, {_T_MAX}] min")

    results = run_fits(t, s)
    print_comparison(results)
    build_overlay(t, s, results, Path("dev/emg_vs_skewnormal.png"))


if __name__ == "__main__":
    main()
