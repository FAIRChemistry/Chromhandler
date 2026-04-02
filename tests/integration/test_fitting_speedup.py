"""Integration test: fit the SAHH kinetics dataset and assert convergence.

Mirrors the reference script that exercises Fitter end-to-end with
three annotated peaks (single, artefact_doublet x 2) on real ASM data.

Requires external data at /Users/max/code/sahh-kinetics-hplc/data/asm.
The test is skipped when that path does not exist so it can run locally
without affecting CI.

Marks:
    fitting  - uses JAX/NumPyro MCMC
    slow     - typically ~40 s on 8 CPU chains
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from chromhandler.fitting import Fitter

DATA_DIR = Path("/Users/max/code/sahh-kinetics-hplc/data")

pytestmark = [pytest.mark.fitting, pytest.mark.slow]


@pytest.fixture(scope="module")
def fitted_fitter() -> tuple[Fitter, float]:
    """Run the full fit once; share across all tests in this module."""
    import chromhandler as ch
    from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
    from chromhandler.fitting import Fitter

    if not (DATA_DIR / "asm").exists():
        pytest.skip(f"External data not found at {DATA_DIR / 'asm'}")

    handler = ch.Handler.read(path=DATA_DIR / "asm")
    handler.cut_chromatograms((2.5, 3.6))
    handler.samples = handler.samples[:1]
    handler.load_initial_conditions(
        DATA_DIR / "conditions.csv",
        conc_unit="umol / l",
    )

    handler.create_molecule(id="Ino", pubchem_cid=135398641, name="Inosine")
    handler.create_molecule(id="SIH", pubchem_cid=135398693, name="S-inosyl-L-homocysteine")
    handler.create_molecule(id="Hyp", pubchem_cid=790, name="Hypoxanthine")

    fitter = Fitter.from_handler(handler)

    fitter.add_peak_annotation(PeakAnnotation(molecule_id="Inosine", rt_min=2.6, rt_max=2.85, mode="single"))
    fitter.add_peak_annotation(
        PeakAnnotation(
            molecule_id="S-inosyl-L-homocysteine",
            rt_min=2.85,
            rt_max=3.15,
            mode="artefact_doublet",
            artefact_side="right",
        )
    )
    fitter.add_peak_annotation(
        PeakAnnotation(
            molecule_id="Hypoxanthine",
            rt_min=3.15,
            rt_max=3.48,
            mode="artefact_doublet",
            artefact_side="left",
            include_artefact_in_area=True,
        )
    )
    fitter.add_baseline_annotation(BaselineAnnotation(rt_min=2.58, rt_max=2.6))
    fitter.add_baseline_annotation(BaselineAnnotation(rt_min=3.5, rt_max=3.52))

    t0 = time.perf_counter()
    fitter.fit(
        num_warmup=1000,
        num_samples=500,
        num_chains=8,
        seed=42,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n[timing] fit() elapsed: {elapsed:.1f}s")

    return fitter, elapsed


def test_fit_completes(fitted_fitter: tuple[Fitter, float]) -> None:
    """fit() must return a non-None posterior."""
    fitter, _ = fitted_fitter
    assert fitter.posterior is not None


def test_area_rhat_below_threshold(fitted_fitter: tuple[Fitter, float]) -> None:
    """90th-percentile Rhat of area_l and area_r must be ≤ 1.05.

    Uses the 90th percentile rather than nanmax because kinetics data has
    near-zero-concentration traces where area is unidentified and individual
    Rhat values can be inflated — that is expected, not a convergence bug.

    area_l/area_r live in fitter.samples (reconstructed post-sampling).
    """
    import arviz as az

    fitter, _ = fitted_fitter
    assert fitter.samples is not None
    n_chains, n_draws = 8, 500
    samples = fitter.samples

    def _rhat_p90(key: str) -> float:
        arr = np.asarray(samples[key])  # [n_chains*n_draws, ...]
        chain_arr = arr.reshape(n_chains, n_draws, *arr.shape[1:])
        values = np.asarray(az.rhat({key: chain_arr})[key].values)  # type: ignore[arg-type]
        finite = values[np.isfinite(values)]
        return float(np.percentile(finite, 90)) if finite.size else 1.0

    for key in ("area_l", "area_r"):
        p90 = _rhat_p90(key)
        assert p90 <= 1.05, f"90th-pct Rhat {p90:.4f} > 1.05 for {key}"


def test_samples_contain_derived_keys(fitted_fitter: tuple[Fitter, float]) -> None:
    """After fit(), self.samples must contain reconstructed derived quantities."""
    fitter, _ = fitted_fitter
    assert fitter.samples is not None
    required = {
        "area_l",
        "area_r",
        "apex_l",
        "apex_r",
        "xi_l",
        "xi_r",
        "sigma_l",
        "sigma_r",
        "alpha_l",
        "alpha_r",
        "baseline_slope",
    }
    missing = required - set(fitter.samples.keys())
    assert not missing, f"Missing derived keys in samples: {missing}"


def test_area_l_shape(fitted_fitter: tuple[Fitter, float]) -> None:
    """area_l must have shape [n_total_samples, n_trace, n_peak]."""
    fitter, _ = fitted_fitter
    assert fitter.samples is not None
    area_l = np.asarray(fitter.samples["area_l"])
    # n_chains=8, n_samples=500 → n_total=4000; n_trace depends on data
    assert area_l.ndim == 3, f"expected 3D, got {area_l.ndim}D"
    assert area_l.shape[0] == 8 * 500  # 4000 draws


def test_elapsed_time_printed(
    fitted_fitter: tuple[Fitter, float], capsys: pytest.CaptureFixture[str]
) -> None:
    """Elapsed time is printed (informational, not asserted)."""
    _, elapsed = fitted_fitter
    # Just verify timing was captured; no hard limit to avoid flakiness
    assert elapsed > 0
    print(f"fit() took {elapsed:.1f}s")
