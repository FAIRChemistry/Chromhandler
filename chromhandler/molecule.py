from __future__ import annotations

import json
from enum import Enum
from typing import TYPE_CHECKING, Optional

import numpy as np
from calipytion.model import Calibration
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from chromhandler.model import Estimate


class CalibrationMethod(str, Enum):
    """Calibration method used to build a :class:`LinearCalibration`."""

    EXTERNAL = "external"
    INTERNAL = "internal"


class LinearCalibration(BaseModel):
    """Linear peak-area → concentration calibration model.

    Built by :meth:`~chromhandler.handler.Handler.calibrate_molecules` from
    calibration-standard samples (``reaction_time == 0``) and stored on
    :attr:`Molecule.calibration` after fitting.

    The regression direction is::

        area = slope × conc (+ intercept)

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
        default_factory=list,
        description="Per-draw OLS slopes (non-empty when posterior regression was performed).",
    )
    intercept_samples: list[float] = Field(
        default_factory=list,
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
        default_factory=list,
        description="Known concentrations used for calibration.",
    )
    areas_mean: list[float] = Field(
        default_factory=list,
        description="Mean areas per standard, in the same order as concentrations.",
    )

    def area_to_conc(
        self,
        area: float,
        extrapolate: bool = False,
        n_samples: int | None = None,
    ) -> "Estimate":
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


class Molecule(BaseModel):
    model_config: ConfigDict = ConfigDict(  # type: ignore
        validate_assignment=True,
        use_enum_values=True,
    )

    id: str = Field(
        description="ID of the molecule",
    )
    pubchem_cid: int = Field(
        description="PubChem CID of the molecule",
    )
    name: str = Field(
        description="Name of the molecule",
    )
    standard: Optional[Calibration] = Field(
        description="Standard associated with the molecule",
        default=None,
    )
    constant: bool = Field(
        description="Boolean indicating whether the molecule concentration is constant throughout the experiment",
        default=False,
    )
    internal_standard: bool = Field(
        description="Boolean indicating whether the molecule is an internal standard",
        default=False,
    )
    calibration: Optional[LinearCalibration] = Field(
        default=None,
        description="Linear calibration model fitted from t=0 calibration standards.",
    )

    @classmethod
    def read_json(cls, path: str) -> Molecule:
        """Creates a Molecule instance from a JSON file.

        Args:
            path (str): The path to the JSON file.

        Returns:
            Molecule: The created Molecule instance.
        """

        with open(path, "r") as f:
            data = json.load(f)

        return cls(**data)

    def save_json(self, path: str) -> None:
        """Saves the Molecule instance to a JSON file.

        Args:
            path (str): The path to the JSON file.

        Returns:
            None
        """

        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=4))
