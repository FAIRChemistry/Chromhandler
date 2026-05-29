"""End-to-end fit on ASM kinetic-series fixture (CV10, SIH peak).

Builds priors, sanity-plots them, then runs MCMC and reports diagnostics.
"""

from pathlib import Path

import numpyro

numpyro.set_host_device_count(8)

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting import ModelConfig, fit
from chromhandler.fitting.priors import PriorConfig, build_priors, summarise_priors
from chromhandler.handler import Handler


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
        PeakAnnotation(molecule_id="other", rt_min=2.55, rt_max=2.80, mode="single"),
        PeakAnnotation(molecule_id="SIH", rt_min=2.80, rt_max=3.15, mode="single"),
        PeakAnnotation(molecule_id="third", rt_min=3.15, rt_max=3.45, mode="single"),
    ]
    base_anns = [
        BaselineAnnotation(rt_min=2.50, rt_max=2.52),
        BaselineAnnotation(rt_min=3.55, rt_max=3.58),
    ]
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
    result.plot_traces(var_names=["area", "mu", "width", "skew"]).savefig("traces.png", bbox_inches="tight", dpi=100)
    print("\nwrote prior_overlay.png, fit.png, traces.png")

    import json
    from pathlib import Path as _P

    golden_path = _P("tests/fixtures/asm_kinetic_series/golden_baseline_model.json")
    golden = json.loads(golden_path.read_text())
    new_summ = result.summary(var_names=["area", "mu", "width", "skew"])
    new = json.loads(new_summ[["mean", "sd", "ess_bulk"]].to_json(orient="index"))
    g = golden["summary"]

    print("\n=== A/B vs golden (current) model ===")
    print(f"{'param':<16}{'old_mean':>12}{'new_mean':>12}{'Δ/σ':>8}{'old_ess':>9}{'new_ess':>9}")
    failures: list[str] = []
    for key in g:
        if key not in new:
            failures.append(f"missing param {key} in new model")
            continue
        om, nm = float(g[key]["mean"]), float(new[key]["mean"])
        osd = max(float(g[key]["sd"]), float(new[key]["sd"]), 1e-12)
        z = abs(nm - om) / osd
        print(f"{key:<16}{om:>12.4g}{nm:>12.4g}{z:>8.2f}"
              f"{g[key]['ess_bulk']:>9.0f}{new[key]['ess_bulk']:>9.0f}")
        if z > 0.5:
            failures.append(f"{key}: |Δ|/σ = {z:.2f} > 0.5  (old={om:.4g} new={nm:.4g})")

    new_diag = result.diagnostics()
    old_diag = golden["diagnostics"]
    print(f"\ndivergences: old={old_diag['n_divergent']} new={new_diag['n_divergent']}")
    print(f"ess_min:     old={old_diag['ess_min_bulk']:.0f} new={new_diag['ess_min_bulk']:.0f}")
    print(f"wall:        old={golden['wall_seconds']:.1f}s")
    if new_diag["n_divergent"] > old_diag["n_divergent"]:
        failures.append(f"divergences regressed: {old_diag['n_divergent']} -> {new_diag['n_divergent']}")

    if failures:
        print("\n[A/B FAILURES]")
        for f in failures:
            print(f"   - {f}")
    else:
        print("\n[A/B PASS] marginalised model agrees with golden within tolerance.")


if __name__ == "__main__":
    main()
