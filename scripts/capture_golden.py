# scripts/capture_golden.py
"""One-time capture of CURRENT-model posteriors as the A/B golden reference.
Run on unmodified HEAD before the marginalisation change."""
import json
import time as _time
from pathlib import Path

import numpyro

numpyro.set_host_device_count(8)

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting import ModelConfig, fit
from chromhandler.fitting.priors import PriorConfig
from chromhandler.handler import Handler


def build_dataset():
    data = Path("tests/fixtures/asm_kinetic_series")
    handler = Handler.read_asm(path=data, mode="timecourse")
    handler.load_initial_conditions(
        "tests/fixtures/asm_kinetic_series/conditions.csv", conc_unit="umol / l"
    )
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
    return handler.prepare_dataset(peak_anns, base_anns)


def main():
    dataset = build_dataset()
    pc = PriorConfig(min_height_frac=0.05)
    t0 = _time.perf_counter()
    result = fit(dataset, prior_config=pc,
                 model_config=ModelConfig(num_warmup=500, num_samples=500, num_chains=4, seed=0))
    wall = _time.perf_counter() - t0

    summ = result.summary(var_names=["area", "mu", "width", "skew"])
    diag = result.diagnostics()
    payload = {
        "wall_seconds": wall,
        "diagnostics": {k: (float(v) if isinstance(v, (int, float)) else v)
                        for k, v in diag.items()},
        "summary": json.loads(summ[["mean", "sd", "ess_bulk"]].to_json(orient="index")),
    }
    out = Path("tests/fixtures/asm_kinetic_series/golden_baseline_model.json")
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out} (wall={wall:.1f}s, ess_min={diag['ess_min_bulk']:.0f}, "
          f"div={diag['n_divergent']})")


if __name__ == "__main__":
    main()
