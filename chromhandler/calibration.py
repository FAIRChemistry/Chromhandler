"""Calibration types and fitting logic for chromhandler."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .model import Sample
    from .molecule import Molecule


# ---------------------------------------------------------------------------
# Calibration method enum
# ---------------------------------------------------------------------------


class CalibrationMethod(str, Enum):
    """Calibration method used to build a :class:`LinearCalibration`."""

    EXTERNAL = "external"
    INTERNAL = "internal"


# ---------------------------------------------------------------------------
# LinearCalibration model
# ---------------------------------------------------------------------------


class LinearCalibration(BaseModel):
    """Linear peak-area → concentration calibration model.

    Built by :func:`calibrate_molecules` from calibration-standard samples
    (``reaction_time == 0``) and stored on :attr:`Molecule.calibration` after
    fitting.

    The regression direction is::

        area = slope * conc (+ intercept)

    so the inverse (area → concentration) is::

        conc = (area - intercept) / slope

    When all calibration peaks carry posterior draws
    (``Peak.area.samples`` non-empty), per-draw regressions are stored in
    :attr:`slope_samples` / :attr:`intercept_samples` and
    :meth:`area_to_conc` returns a full :class:`~chromhandler.model.Estimate`
    with uncertainty.

    Attributes:
        method: Calibration method (always ``"external"`` at present).
        fit_intercept: Whether the regression included an intercept term.
        slope: OLS slope in *area / conc* units.
        intercept: OLS intercept in *area* units (0.0 when
            ``fit_intercept=False``).
        r_squared: Coefficient of determination R².
        residual_std: Standard error of the estimate.
        n_standards: Number of calibration points used.
        max_area: Highest measured area in the calibration set.
        min_area: Lowest measured area in the calibration set.
        max_conc: Highest known concentration in the calibration set.
        min_conc: Lowest known concentration in the calibration set.
        slope_samples: Per-draw OLS slopes (non-empty if posterior regression
            was performed).
        intercept_samples: Per-draw OLS intercepts (parallel to
            ``slope_samples``).
        conc_unit: String representation of the concentration unit.
        chromatogram_ids: Chromatogram IDs whose peaks were used as standards.
        concentrations: Known concentrations used for calibration (one per
            standard), in the same order as :attr:`areas_mean`.
        areas_mean: Mean areas used for calibration (one per standard).
    """

    model_config: ConfigDict = ConfigDict(validate_assignment=True)  # type: ignore

    method: CalibrationMethod = CalibrationMethod.EXTERNAL
    fit_intercept: bool = False

    # --- Point-estimate regression parameters ---
    slope: float = Field(description="OLS slope (area / conc).")
    intercept: float = Field(default=0.0, description="OLS intercept (area units).")
    r_squared: float = Field(description="Coefficient of determination R².")
    residual_std: float = Field(description="Standard error of the estimate.")
    n_standards: int = Field(description="Number of calibration points used.")

    # --- Calibration range (for extrapolation guard) ---
    max_area: float = Field(description="Highest measured area in calibration set.")
    min_area: float = Field(description="Lowest measured area in calibration set.")
    max_conc: float = Field(description="Highest known concentration in calibration set.")
    min_conc: float = Field(description="Lowest known concentration in calibration set.")

    # --- Posterior regression distributions ---
    slope_samples: list[float] = Field(
        default_factory=lambda: [],
        description="Per-draw OLS slopes (non-empty when posterior regression was performed).",
    )
    intercept_samples: list[float] = Field(
        default_factory=lambda: [],
        description="Per-draw OLS intercepts, parallel to slope_samples.",
    )

    # --- Provenance ---
    conc_unit: str = Field(default="", description="String representation of concentration unit.")
    chromatogram_ids: list[str] = Field(
        default_factory=list,
        description="Chromatogram IDs whose peaks were used as standards.",
    )

    # --- Raw data (useful for plotting the calibration curve) ---
    concentrations: list[float] = Field(
        default_factory=lambda: [],
        description="Known concentrations used for calibration.",
    )
    areas_mean: list[float] = Field(
        default_factory=lambda: [],
        description="Mean areas per standard, in the same order as concentrations.",
    )

    def area_to_conc(
        self,
        area: float,
        extrapolate: bool = False,
        n_samples: int | None = None,
    ) -> Any:
        """Convert a peak area to a concentration :class:`~chromhandler.model.Estimate`.

        When posterior regression distributions are available
        (``slope_samples`` non-empty) and *n_samples* is given, the returned
        :class:`~chromhandler.model.Estimate` carries full uncertainty from the
        calibration posteriors.

        Args:
            area: The peak area to convert.
            extrapolate: If ``False`` (default), raises :exc:`ValueError` when
                *area* is outside the calibration range
                [``min_area``, ``max_area``].
            n_samples: If not ``None`` and posterior distributions exist, draw
                this many random samples for the posterior-predictive
                :class:`~chromhandler.model.Estimate`.

        Returns:
            :class:`~chromhandler.model.Estimate` with at minimum ``mean`` set.
            When posterior samples are available, ``std``, ``q05``, ``q95``,
            and optionally ``samples`` are also populated.

        Raises:
            ValueError: If ``extrapolate=False`` and *area* is outside the
                calibration range.
        """
        from chromhandler.model import Estimate  # local — avoids circular import

        if not extrapolate and (area < self.min_area or area > self.max_area):
            raise ValueError(
                f"area={area:.4g} is outside the calibration range "
                f"[{self.min_area:.4g}, {self.max_area:.4g}]. "
                "Pass extrapolate=True to allow extrapolation."
            )

        if self.slope_samples:
            slopes = np.asarray(self.slope_samples)
            intercepts = np.asarray(self.intercept_samples)
            if n_samples is not None:
                n_draw = min(n_samples, len(slopes))
                idx = np.random.choice(len(slopes), size=n_draw, replace=False)
                slopes = slopes[idx]
                intercepts = intercepts[idx]
            concs = (area - intercepts) / slopes
            return Estimate(
                mean=float(np.mean(concs)),
                median=float(np.median(concs)),
                std=float(np.std(concs)),
                q05=float(np.quantile(concs, 0.05)),
                q95=float(np.quantile(concs, 0.95)),
                samples=concs.tolist() if n_samples is not None else [],
            )

        # Point estimate only
        conc = (area - self.intercept) / self.slope
        return Estimate(mean=conc)


# ---------------------------------------------------------------------------
# OLS helper
# ---------------------------------------------------------------------------


def _fit_linear(
    x: npt.NDArray[np.floating[Any]],
    y: npt.NDArray[np.floating[Any]],
    *,
    fit_intercept: bool,
) -> tuple[float, float]:
    """OLS linear regression ``y = slope*x (+ intercept)``.

    Args:
        x: Predictor array (concentrations).
        y: Response array (areas).
        fit_intercept: Whether to include an intercept term.

    Returns:
        ``(slope, intercept)`` tuple.  *intercept* is ``0.0`` when
        *fit_intercept* is ``False``.
    """
    if fit_intercept:
        A = np.column_stack([x, np.ones_like(x)])
        result, *_ = np.linalg.lstsq(A, y, rcond=None)
        return float(result[0]), float(result[1])
    slope = float(np.dot(x, y) / np.dot(x, x))
    return slope, 0.0


# ---------------------------------------------------------------------------
# Molecule target resolution helper
# ---------------------------------------------------------------------------


def _resolve_molecule_targets(
    molecules: dict[str, Molecule],
    molecule_ids: list[str] | None,
) -> list[Molecule]:
    """Return the target molecules for calibration.

    When *molecule_ids* is ``None`` every registered molecule is returned.
    Otherwise all IDs are validated upfront and a :exc:`ValueError` is raised
    for any that are not found — no silent skipping.
    """
    if molecule_ids is None:
        return list(molecules.values())
    missing = set(molecule_ids) - molecules.keys()
    if missing:
        raise ValueError(
            f"calibrate_molecules: unknown molecule IDs: {missing}. "
            "Register the molecule first with create_molecule() or register_molecule()."
        )
    return [molecules[mid] for mid in molecule_ids]


# ---------------------------------------------------------------------------
# Public calibration functions
# ---------------------------------------------------------------------------


def external_calibration(
    molecule: Molecule,
    samples: list[Sample],
    *,
    fit_intercept: bool,
) -> LinearCalibration | None:
    """Build a :class:`LinearCalibration` from t=0 calibration samples.

    Iterates all *samples* whose chromatograms include ``reaction_time == 0``,
    collects ``(init_conc, peak_area)`` pairs for *molecule*, and fits an OLS
    linear model.  Returns ``None`` when fewer than 1 data point is found.

    Args:
        molecule: The molecule to calibrate.
        samples: All samples held by the handler.
        fit_intercept: Whether to include an intercept term in the regression.

    Returns:
        A fitted :class:`LinearCalibration`, or ``None`` if insufficient data.
    """
    known_concs: list[float] = []
    areas_mean: list[float] = []
    area_samples_per_point: list[list[float]] = []
    chrom_ids: list[str] = []
    conc_unit_ref: str | None = None

    for sample in samples:
        # Only calibration standards (at least one chromatogram at t=0)
        if not any(c.reaction_time is not None and c.reaction_time == 0.0 for c in sample.chromatograms):
            continue

        # Known concentration for this molecule
        ic = next(
            (ic for ic in sample.initial_conditions if ic.molecule_id == molecule.id),
            None,
        )
        if ic is None:
            continue

        # Store conc_unit from the first valid ic
        if conc_unit_ref is None:
            conc_unit_ref = ic.conc_unit if isinstance(ic.conc_unit, str) else ic.conc_unit.id

        # Chromatogram at t=0 that contains a peak for this molecule
        chrom = next(
            (
                c
                for c in sample.chromatograms
                if c.reaction_time is not None
                and c.reaction_time == 0.0
                and any(p.molecule_id == molecule.id for p in c.peaks)
            ),
            None,
        )
        if chrom is None:
            continue

        peak = next(p for p in chrom.peaks if p.molecule_id == molecule.id)

        # Extract area: posterior samples > median > mean
        if peak.area.samples:
            area_pt = float(np.mean(peak.area.samples))
            area_samples_per_point.append(list(peak.area.samples))
        elif peak.area.median is not None:
            area_pt = float(peak.area.median)
            area_samples_per_point.append([])
        else:
            area_pt = float(peak.area.mean)
            area_samples_per_point.append([])

        known_concs.append(float(ic.init_conc))
        areas_mean.append(area_pt)
        chrom_ids.append(str(chrom.id))

    if len(known_concs) < 1:
        return None

    conc_arr = np.asarray(known_concs, dtype=float)
    area_arr = np.asarray(areas_mean, dtype=float)

    # --- Point-estimate regression ---
    slope, intercept = _fit_linear(conc_arr, area_arr, fit_intercept=fit_intercept)
    area_pred = slope * conc_arr + intercept
    ss_res = float(np.sum((area_arr - area_pred) ** 2))
    ss_tot = float(np.sum((area_arr - np.mean(area_arr)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    n_params = 2 if fit_intercept else 1
    residual_std = float(np.sqrt(ss_res / max(len(conc_arr) - n_params, 1)))

    # --- Per-draw regression (only when ALL points have posterior samples) ---
    slope_samples: list[float] = []
    intercept_samples: list[float] = []
    if all(len(s) > 0 for s in area_samples_per_point):
        n_draws = min(len(s) for s in area_samples_per_point)
        for i in range(n_draws):
            a_i = np.asarray([s[i] for s in area_samples_per_point])
            s_i, ic_i = _fit_linear(conc_arr, a_i, fit_intercept=fit_intercept)
            slope_samples.append(float(s_i))
            intercept_samples.append(float(ic_i))

    return LinearCalibration(
        method=CalibrationMethod.EXTERNAL,
        fit_intercept=fit_intercept,
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        residual_std=residual_std,
        n_standards=len(known_concs),
        max_area=float(np.max(area_arr)),
        min_area=float(np.min(area_arr)),
        max_conc=float(np.max(conc_arr)),
        min_conc=float(np.min(conc_arr)),
        slope_samples=slope_samples,
        intercept_samples=intercept_samples,
        conc_unit=conc_unit_ref or "",
        chromatogram_ids=chrom_ids,
        concentrations=known_concs,
        areas_mean=areas_mean,
    )


def calibrate_molecules(
    molecules: dict[str, Molecule],
    samples: list[Sample],
    molecule_ids: list[str] | None = None,
    *,
    fit_intercept: bool = False,
    method: str = "external",
    verbose: bool = True,
) -> None:
    """Fit linear calibration curves for the given molecules.

    Uses samples with ``reaction_time == 0`` as calibration standards.
    Each such sample must have an :class:`~chromhandler.model.InitialCondition`
    with ``init_conc`` set for the target molecule.  Peak areas are read from
    :attr:`~chromhandler.model.Chromatogram.peaks`, preferring
    ``Peak.area.samples`` (posterior draws) → ``Peak.area.median`` →
    ``Peak.area.mean``.

    The fitted :class:`LinearCalibration` is stored on ``Molecule.calibration``
    as a side effect.

    Args:
        molecules: Registry of molecules keyed by ID.
        samples: All samples to search for calibration standards.
        molecule_ids: Which molecules to calibrate.  Default: every molecule
            in the registry.
        fit_intercept: Include an intercept term in the regression
            (default ``False`` — forces the curve through the origin).
        method: Calibration method.  Only ``"external"`` is implemented;
            ``"internal"`` raises :exc:`NotImplementedError`.
        verbose: Print a rich calibration-summary table to stdout
            (default ``True``).  Pass ``False`` to suppress all output.

    Raises:
        NotImplementedError: When *method* is ``"internal"``.
        ValueError: When *method* is unrecognised or *molecule_ids* contains
            unknown IDs.
    """
    from . import pretty  # local — only needed when called

    if method == "internal":
        raise NotImplementedError("Internal standard calibration is not yet implemented.")
    if method != "external":
        raise ValueError(f"Unknown calibration method: {method!r}. Use 'external'.")

    targets = _resolve_molecule_targets(molecules, molecule_ids)

    summary: list[tuple[Molecule, LinearCalibration | None]] = []
    for molecule in targets:
        cal = external_calibration(molecule, samples, fit_intercept=fit_intercept)
        if cal is not None:
            molecule.calibration = cal
        summary.append((molecule, cal))

    if verbose:
        pretty.display_calibration_summary(summary)


__all__ = [
    "CalibrationMethod",
    "LinearCalibration",
    "calibrate_molecules",
    "external_calibration",
]
