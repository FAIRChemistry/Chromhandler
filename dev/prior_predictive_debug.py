"""Standalone debug script: canonical prior-predictive on CV10 only.

The actual bug
==============

``chromhandler.fitting.posterior.compute_prior_predictive`` passes the
model directly to ``numpyro.infer.Predictive``. The model contains:

    numpyro.sample(
        "obs",
        dist.Normal(predicted, noise[:, None]),
        obs=jnp.asarray(dataset.signal),     # <-- conditioning
    )

``Predictive`` will *not* sample a site that already has ``obs=`` set —
it returns the conditioned value verbatim. So every "prior predictive"
draw was a byte-for-byte copy of the observed signal (verified:
``np.allclose(obs_buggy[draw], dataset.signal)`` is True for every draw).

That is why ``plot_prior_predictive`` looked like a single line through
the data: every draw IS the data. The non-trivial spread we saw printed
for ``mu_anchor_left`` / ``log_sigma_left`` / ``log_A_left`` / etc. was
real, but it was being discarded — those samples never feed back into
``obs`` because ``obs`` was overridden.

Fix
===

For a *prior* predictive, you have to sample from the likelihood at
``obs`` without conditioning. The cleanest numpyro idiom is to call
``Predictive`` against a copy of the model with ``obs=`` removed. This
script defines :func:`model_unconditioned` for exactly that, builds the
same ``priors_list`` ``fit()`` would build, and plots the 95% HDI band
per (trace, annotated peak window).

Run:
    uv run python3 dev/prior_predictive_debug.py
Output:
    dev/prior_predictive_debug.png
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import arviz
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.model import ModelConfig, _compute_baseline_se
from chromhandler.fitting.priors import PriorConfig, build_priors
from chromhandler.fitting.skew_normal import GAMMA1_MAX, density_cp
from chromhandler.handler import Handler

if TYPE_CHECKING:
    from chromhandler.fitting.prepared_dataset import PreparedDataset
    from chromhandler.fitting.priors import SkewNormalPriors

# ---- Paths ----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
ASM_DIR = REPO_ROOT / "tests" / "fixtures" / "asm_kinetic_series"
CONDITIONS_CSV = ASM_DIR / "conditions.csv"
OUTPUT_PNG = Path(__file__).with_suffix(".png")

# ---- Config ---------------------------------------------------------------
N_PRIOR_SAMPLES = 400
HDI_PROB = 0.95
SEED = 0


def load_cv10_dataset() -> PreparedDataset:
    """Build a PreparedDataset from the CV10 timecourse only."""
    handler = Handler.read_asm(path=ASM_DIR, mode="timecourse")
    for mol_id in ("SIH", "Hyp", "Ino"):
        handler.create_molecule(id=mol_id, pubchem_cid=1)
    handler.load_initial_conditions(CONDITIONS_CSV, conc_unit="umol / l")

    h_sub = Handler()
    h_sub.molecules = deepcopy(handler.molecules)
    h_sub.samples = [deepcopy(s) for s in handler.samples if s.id == "CV10"]

    peak_anns = [
        PeakAnnotation(molecule_id="SIH", rt_min=2.80, rt_max=3.15, mode="single"),
    ]
    base_anns = [
        BaselineAnnotation(rt_min=2.50, rt_max=2.52),
        BaselineAnnotation(rt_min=3.55, rt_max=3.58),
    ]
    return h_sub.prepare_dataset(peak_anns, base_anns)


def model_unconditioned(
    dataset: PreparedDataset,
    priors_list: list[SkewNormalPriors],
    config: ModelConfig,
) -> None:
    """Identical to ``fitting.model.model`` except that ``obs`` is NOT
    conditioned on ``dataset.signal``.

    This is the model that ``numpyro.infer.Predictive`` needs to call to
    produce a real prior predictive: the ``obs`` site samples from the
    Normal likelihood at every draw, instead of being short-circuited by
    the ``obs=`` argument.
    """
    n_trace = dataset.n_trace
    n_peak = len(priors_list)
    dt_global = float(dataset.dt_global)

    mu_anchor_left = numpyro.sample(
        "mu_anchor_left",
        dist.TruncatedNormal(
            loc=jnp.asarray([p.mu_left_loc for p in priors_list]),
            scale=jnp.asarray([p.mu_left_scale for p in priors_list]),
            low=jnp.asarray([p.mu_left_low for p in priors_list]),
            high=jnp.asarray([p.mu_left_high for p in priors_list]),
        ),
    )
    log_sigma_left = numpyro.sample(
        "log_sigma_left",
        dist.TruncatedNormal(
            loc=jnp.asarray([p.log_sigma_left_loc for p in priors_list]),
            scale=jnp.asarray([p.log_sigma_left_scale for p in priors_list]),
            low=jnp.asarray([p.log_sigma_left_low for p in priors_list]),
            high=jnp.asarray([p.log_sigma_left_high for p in priors_list]),
        ),
    )
    gamma1_bound = 0.99 * float(GAMMA1_MAX)
    gamma1_left = numpyro.sample(
        "gamma1_left",
        dist.TruncatedNormal(
            loc=jnp.asarray([p.gamma1_left_loc for p in priors_list]),
            scale=jnp.asarray([p.gamma1_left_scale for p in priors_list]),
            low=-gamma1_bound, high=gamma1_bound,
        ),
    )
    log_A_left = numpyro.sample(
        "log_A_left",
        dist.Normal(
            loc=jnp.asarray(
                np.stack([p.log_A_left_loc_per_trace for p in priors_list], axis=1)
            ),
            scale=jnp.asarray([p.log_A_left_scale for p in priors_list])[None, :],
        ),
    )
    drift_scale = config.trace_shift_scale_dt_multiplier * dt_global
    trace_shift = numpyro.sample(
        "trace_shift",
        dist.Normal(loc=jnp.zeros(n_trace), scale=drift_scale),
    )
    intercept_se, slope_se = _compute_baseline_se(dataset)
    intercept_se_eff = np.maximum(intercept_se, config.baseline_intercept_se_floor)
    slope_se_eff = np.maximum(slope_se, config.baseline_slope_se_floor)
    baseline_intercept = numpyro.sample(
        "baseline_intercept",
        dist.Normal(
            loc=jnp.asarray(dataset.baseline_intercept),
            scale=jnp.asarray(intercept_se_eff),
        ),
    )
    baseline_slope = numpyro.sample(
        "baseline_slope",
        dist.Normal(
            loc=jnp.asarray(dataset.baseline_slope),
            scale=jnp.asarray(slope_se_eff),
        ),
    )

    sigma_left = jnp.exp(log_sigma_left)
    A_left = jnp.exp(log_A_left)
    mu = mu_anchor_left[None, :] + trace_shift[:, None]
    time_arr = jnp.asarray(dataset.time)
    baseline = baseline_intercept[:, None] + baseline_slope[:, None] * time_arr
    left_contrib = jnp.zeros_like(time_arr)
    for peak in range(n_peak):
        dens = density_cp(
            time_arr,
            mu[:, peak : peak + 1],  # type: ignore[arg-type]
            sigma_left[peak],
            gamma1_left[peak],  # type: ignore[arg-type]
        )
        left_contrib = left_contrib + A_left[:, peak : peak + 1] * dens
    predicted = baseline + left_contrib
    noise = jnp.asarray(dataset.noise_per_trace)
    # NO obs= — this is the canonical prior predictive.
    numpyro.sample("obs", dist.Normal(predicted, noise[:, None]))


def print_actual_priors(priors_list: list[SkewNormalPriors]) -> None:
    print("\n=== Actual priors used by model() ===")
    for i, p in enumerate(priors_list):
        print(f"\nPeak {i} (n_components={p.n_components}):")
        print(
            f"  mu_left:       loc={p.mu_left_loc:.4f}  "
            f"scale={p.mu_left_scale:.4f}  "
            f"low={p.mu_left_low:.4f}  high={p.mu_left_high:.4f}"
        )
        print(
            f"  log_sigma_left:loc={p.log_sigma_left_loc:.4f}  "
            f"scale={p.log_sigma_left_scale:.4f}  "
            f"low={p.log_sigma_left_low:.4f}  high={p.log_sigma_left_high:.4f}"
        )
        print(
            f"  gamma1_left:   loc={p.gamma1_left_loc:.4f}  "
            f"scale={p.gamma1_left_scale:.4f}"
        )
        per_trace = np.asarray(p.log_A_left_loc_per_trace)
        print(
            f"  log_A_left:    per_trace_locs="
            f"[{per_trace.min():.3f}..{per_trace.max():.3f}]  "
            f"scale={p.log_A_left_scale:.4f}"
        )


def sample_prior_predictive(
    dataset: PreparedDataset,
    priors_list: list[SkewNormalPriors],
    config: ModelConfig,
    n_samples: int,
    seed: int,
) -> dict[str, np.ndarray[Any, np.dtype[Any]]]:
    """Canonical numpyro prior predictive via the unconditioned model."""
    predictive = numpyro.infer.Predictive(model_unconditioned, num_samples=n_samples)
    samples = predictive(jax.random.PRNGKey(seed), dataset, priors_list, config)
    return {name: np.asarray(arr) for name, arr in samples.items()}


def plot_prior_predictive_per_window(
    dataset: PreparedDataset,
    peak_anns: list[PeakAnnotation],
    prior_obs: np.ndarray[Any, np.dtype[Any]],
    output_path: Path,
) -> None:
    """One axis per (trace, peak annotation): data + 95% HDI band + median."""
    n_trace = dataset.n_trace
    n_peak = len(peak_anns)
    fig, axes = plt.subplots(
        n_trace, n_peak,
        figsize=(3.8 * n_peak, 2.2 * n_trace),
        squeeze=False, sharex=False,
    )

    for tr in range(n_trace):
        t = dataset.time[tr]
        s = dataset.signal[tr]
        for col, ann in enumerate(peak_anns):
            ax = axes[tr, col]
            mask = (t >= ann.rt_min) & (t <= ann.rt_max) & np.isfinite(s)
            samples_tr_t = prior_obs[:, tr, :]

            # arviz.hdi: pass [chain=1, draw, time] to compute per-time HDI.
            try:
                hdi_arr = np.asarray(  # type: ignore[reportUnknownArgumentType]
                    arviz.hdi(samples_tr_t[None, :, :], hdi_prob=HDI_PROB)
                )
                # Normalise to [time, 2]
                if hdi_arr.shape[0] == 2 and hdi_arr.shape[-1] != 2:
                    hdi_arr = hdi_arr.T
            except Exception:
                hdi_arr = np.quantile(samples_tr_t, [0.025, 0.975], axis=0).T

            median = np.median(samples_tr_t, axis=0)
            ax.fill_between(
                t[mask], hdi_arr[mask, 0], hdi_arr[mask, 1],
                color="tab:purple", alpha=0.30,
                label=f"prior {int(HDI_PROB*100)}% HDI",
            )
            ax.plot(
                t[mask], median[mask],
                color="tab:purple", lw=1.2, label="prior median",
            )
            ax.plot(t[mask], s[mask], color="k", lw=0.9, label="data")
            ax.axvline(ann.rt_min, color="0.6", lw=0.5, ls="--")
            ax.axvline(ann.rt_max, color="0.6", lw=0.5, ls="--")
            ax.set_title(
                f"{dataset.trace_ids[tr]} — peak {col} ({ann.molecule_id})",
                fontsize=8,
            )
            if tr == 0 and col == 0:
                ax.legend(fontsize=7)
    fig.suptitle(
        "Canonical prior predictive (model obs= unconditioned) — CV10",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nSaved figure: {output_path}")


def main() -> None:
    dataset = load_cv10_dataset()
    print(f"Loaded CV10: n_trace={dataset.n_trace}, "
          f"trace_ids={dataset.trace_ids}")

    priors_list = build_priors(dataset, config=PriorConfig())
    print_actual_priors(priors_list)

    config = ModelConfig()
    prior_samples = sample_prior_predictive(
        dataset, priors_list, config,
        n_samples=N_PRIOR_SAMPLES, seed=SEED,
    )
    obs = prior_samples["obs"]
    print(
        f"\nPrior predictive obs shape: {obs.shape}  "
        f"(draws, n_trace, n_time)"
    )

    # Spot-check spread at apex per trace — proves the band is real.
    t0 = np.asarray(dataset.time[0])
    apex_idx = int(np.argmin(np.abs(t0 - priors_list[0].mu_left_loc)))
    print(f"\nApex t={t0[apex_idx]:.4f} — prior 95% interval vs data:")
    for tr in range(dataset.n_trace):
        col = obs[:, tr, apex_idx]
        q_low, q_high = np.quantile(col, [0.025, 0.975])
        print(
            f"  {dataset.trace_ids[tr]:<22s} "
            f"[{q_low:>8.0f}, {q_high:>8.0f}]   "
            f"data={dataset.signal[tr, apex_idx]:>8.0f}"
        )

    # --- Cross-check against the library ---
    # After the library fix, numpyro.infer.Predictive(model) (the actual
    # library `model`, NOT `model_unconditioned`) should produce the same
    # spread we just demonstrated. If this assertion fails, the library
    # fix did not land.
    from chromhandler.fitting.model import model as lib_model

    lib_pred = numpyro.infer.Predictive(lib_model, num_samples=N_PRIOR_SAMPLES)
    lib_samples = lib_pred(jax.random.PRNGKey(SEED), dataset, priors_list, config)
    lib_obs = np.asarray(lib_samples["obs"])
    lib_apex_var = lib_obs[:, :, apex_idx].var(axis=0)
    assert np.all(lib_apex_var > 0.0), (
        "Library model still conditions obs= — apex variance is zero. "
        "The library fix from this plan did not land."
    )
    print(
        f"\nLibrary cross-check OK: library model produces non-zero apex "
        f"variance per trace (min={lib_apex_var.min():.2f})."
    )

    plot_prior_predictive_per_window(
        dataset, dataset.peak_annotations, obs, OUTPUT_PNG,
    )


if __name__ == "__main__":
    main()
