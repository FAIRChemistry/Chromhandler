"""End-to-end fit on ASM kinetic-series fixture (CV10, SIH peak).

Builds priors, sanity-plots them, then runs MCMC and reports diagnostics.
"""

from pathlib import Path

import numpyro

numpyro.set_host_device_count(8)

# Imports follow set_host_device_count, which must run before JAX is imported.
from chromhandler.annotations import BaselineAnnotation, PeakAnnotation  # noqa: E402
from chromhandler.fitting import ModelConfig, fit  # noqa: E402
from chromhandler.fitting.priors import (  # noqa: E402
    PriorConfig,
    build_priors,
    summarise_priors,
)
from chromhandler.handler import Handler  # noqa: E402


def main() -> None:
    data = Path("tests/fixtures/asm_kinetic_series")
    conditions = "tests/fixtures/asm_kinetic_series/conditions.csv"

    handler = Handler.read_asm(path=data, mode="timecourse")
    handler.load_initial_conditions(conditions, conc_unit="umol / l")
    # All samples — CV (kinetic series) + CW (calibration / control wells)
    handler.create_molecule(id="SIH", pubchem_cid=135398693, name="S-inosyl-L-homocysteine")
    handler.create_molecule(id="other", pubchem_cid=0, name="other")
    handler.create_molecule(id="third", pubchem_cid=0, name="third")

    peak_anns = [
        PeakAnnotation(molecule_id="other", rt_min=2.55, rt_max=2.80, mode="single", peak_model="emg"),
        PeakAnnotation(molecule_id="SIH", rt_min=2.80, rt_max=3.15, mode="single"),
        PeakAnnotation(molecule_id="third", rt_min=3.15, rt_max=3.45, mode="single"),
    ]
    base_anns = [
        BaselineAnnotation(rt_min=2.50, rt_max=2.52),
        BaselineAnnotation(rt_min=3.55, rt_max=3.58),
    ]

    # Coarse retention-time pre-alignment (opt-in, explicit two-step): removes
    # gross per-trace offsets in place before the fit so the model's tight
    # time_shift prior only absorbs the fine residual. Aligns on the peak-
    # bearing span; mutates each chromatogram's time axis.
    align = handler.align_chromatograms(lower_rt=2.55, upper_rt=3.45)
    print("\n=== retention-time alignment ===")
    print(f"  delta_rt (min): {[round(float(x), 4) for x in align.delta_rt]}")
    print(f"  loss: {align.loss_initial:.4g} -> {align.loss_final:.4g}")

    dataset = handler.prepare_dataset(peak_anns, base_anns)

    pc = PriorConfig(min_height_frac=0.05)
    priors = build_priors(dataset, config=pc)
    print(summarise_priors(priors, pc))

    result = fit(
        dataset,
        prior_config=pc,
        model_config=ModelConfig(num_warmup=500, num_samples=500, num_chains=4, seed=0),
    )

    print("\n=== diagnostics ===")
    for k, v in result.diagnostics().items():
        print(f"  {k:<22} {v}")
    print("\n=== user-facing parameters ===")
    print(result.summary())

    result.plot_prior_overlay().savefig("prior_overlay.png", bbox_inches="tight", dpi=120)
    result.plot_fit().savefig("fit.png", bbox_inches="tight", dpi=120)
    result.plot_traces(var_names=["area", "mu", "width", "skew"]).savefig(
        "traces.png", bbox_inches="tight", dpi=100
    )
    print("\nwrote prior_overlay.png, fit.png, traces.png")

    print("\n=== QC report ===")
    qc = result.qc_report()
    print(
        f"  fit_healthy={qc['fit_healthy']} n_divergent={qc['n_divergent']} "
        f"bfmi_min={qc['bfmi_min']:.2f} rhat_max={qc['rhat_max']:.3f} ess_min={qc['ess_min']:.0f}"
    )
    for grp, d in sorted(qc["groups"].items()):
        print(f"    {grp:<9} rhat_max={d['rhat_max']:.3f}  ess_min={d['ess_min']:.0f}")

    result.plot_qc_overview().savefig("qc_overview.png", bbox_inches="tight", dpi=120)
    result.plot_convergence().savefig("qc_convergence.png", bbox_inches="tight", dpi=120)
    result.plot_areas().savefig("qc_areas.png", bbox_inches="tight", dpi=120)
    result.plot_warp().savefig("qc_warp.png", bbox_inches="tight", dpi=120)
    print("wrote qc_overview.png, qc_convergence.png, qc_areas.png, qc_warp.png")


if __name__ == "__main__":
    main()
