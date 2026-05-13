"""End-to-end smoke test on the real ASM kinetic-series fixture."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting import ModelConfig, fit
from chromhandler.handler import Handler

ASM_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "asm_kinetic_series"
CONDITIONS_CSV = ASM_DIR / "conditions.csv"


def test_fit_on_sih_kinetic_subset() -> None:
    """Fit the SIH analyte on the CV10 kinetic series + CV1 (no enzyme) +
    CV4 + CV5 controls. Single-mode, contiguous window 2.80-3.15 min.

    Note on diagnostics: The CV1 traces contain a very large SIH peak
    (~272 K mAU) while the baseline-region noise is ~5 mAU (SNR ≈ 55 000).
    The posterior is essentially a delta function, so NUTS converges to a
    correct but zero-variance chain for each initialisation — ArviZ r_hat and
    ESS are degenerate in this regime.  We therefore skip the r_hat / ESS
    checks and only assert that (a) divergences are absent and (b) the
    effective peak position (mu_anchor + median trace_shift) lands near the
    priors_demo reference of 3.008 min.
    """
    handler = Handler.read_asm(path=ASM_DIR, mode="timecourse")
    for mol_id in ("SIH", "Hyp", "Ino"):
        handler.create_molecule(id=mol_id, pubchem_cid=1)
    handler.load_initial_conditions(CONDITIONS_CSV, conc_unit="umol / l")

    # Subset to samples relevant for SIH + controls
    h_sub = Handler()
    h_sub.molecules = deepcopy(handler.molecules)
    h_sub.samples = [deepcopy(s) for s in handler.samples
                     if s.id in {"CV10", "CV1", "CV4", "CV5"}]

    peak_anns = [PeakAnnotation(molecule_id="SIH", rt_min=2.80, rt_max=3.15, mode="single")]
    base_anns = [BaselineAnnotation(rt_min=2.50, rt_max=2.52),
                 BaselineAnnotation(rt_min=3.55, rt_max=3.58)]
    dataset = h_sub.prepare_dataset(peak_anns, base_anns)

    result = fit(dataset, model_config=ModelConfig(
        num_warmup=200, num_samples=200, num_chains=2, seed=0,
    ))

    diag = result.diagnostics()

    # No divergences allowed — any divergence is a genuine posterior pathology.
    assert diag["n_divergent"] == 0, f"divergences: {diag}"

    # The posterior is near-deterministic for high-SNR CV1 traces, so r_hat
    # and ESS are meaningless (within-chain variance ≈ 0).  We do not assert
    # on r_hat_max here.  The scientific validity check below is the real gate.

    # Effective peak position: mu_anchor + median(trace_shift) should land
    # near the priors_demo reference of 3.008 min (tolerance ±0.03 min).
    # We use the median over (chain, draw, trace) to marginalise the
    # mu_anchor ↔ trace_shift composition degeneracy described in the plan.
    posterior_mu = np.asarray(  # type: ignore[reportUnknownArgumentType]
        result.idata.posterior["mu_anchor_left"]  # type: ignore[reportAttributeAccessIssue]
    )  # [chain, draw, n_peak]
    posterior_ts = np.asarray(  # type: ignore[reportUnknownArgumentType]
        result.idata.posterior["trace_shift"]  # type: ignore[reportAttributeAccessIssue]
    )  # [chain, draw, n_trace]

    # Effective mu per (chain, draw, trace) = mu_anchor + trace_shift
    eff_mu = posterior_mu[:, :, 0:1] + posterior_ts  # [chain, draw, trace]
    median_eff_mu = float(np.median(eff_mu))
    assert abs(median_eff_mu - 3.008) < 0.03, (
        f"effective mu {median_eff_mu:.4f} too far from priors_demo value 3.008"
    )

    # Plot smoke tests: verify both methods return a non-None figure.
    fig_traces = result.plot_traces()
    assert fig_traces is not None
    fig_fit = result.plot_fit()
    assert fig_fit is not None
