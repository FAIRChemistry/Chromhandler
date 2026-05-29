"""Handler-level peak and baseline window models.

Distinct from :mod:`chromhandler.fitting.data` which holds MCMC fitting
helpers. This module defines the retention-time windows users place on a
:class:`~chromhandler.handler.Handler` as well as the fitter-facing
``PeakAnnotation`` type used by the current Bayesian fitting pipeline.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, computed_field, model_validator

# Vocabulary aligned with the fitting module internals.
PeakMode = Literal["single", "artefact_doublet", "free_doublet"]
ArtefactSide = Literal["left", "right"]
PeakModel = Literal["skew_normal", "emg"]


class PeakAnnotation(BaseModel):
    """A retention-time window annotation attached to a molecule.

    Used by both :class:`~chromhandler.handler.Handler` (for molecule peak
    assignment over reader-detected peaks) and
    :class:`~chromhandler.fitting.fitter.Fitter` (for Bayesian peak fitting).

    Attributes:
        molecule_id: ID of the molecule this window belongs to.
        rt_min: Lower retention-time bound (minutes, inclusive).
        rt_max: Upper retention-time bound (minutes, inclusive).
        mode: Peak shape assumption — ``"single"`` (default, one component),
            ``"free_doublet"`` (two independently positioned components),
            ``"artefact_doublet"`` (main peak + fixed-side artefact shoulder).
            Ignored by handler-level peak assignment.
        artefact_side: Which side carries the shoulder; required when
            *mode* is ``"artefact_doublet"``, must be ``None`` otherwise.
        vary_separation: When ``True`` and *mode* is ``"free_doublet"``,
            the separation is allowed to vary across traces.
        include_artefact_in_area: When ``True`` and *mode* is
            ``"artefact_doublet"``, the artefact shoulder area is summed
            with the dominant component's area.
        wavelength: When set, :meth:`~chromhandler.handler.Handler.assign_molecules`
            only considers chromatograms at this wavelength (nm). Ignored by
            the fitter.

    Example::

        # Handler-only usage (peak assignment over reader-detected peaks):
        ann = PeakAnnotation(molecule_id="Ino", rt_min=2.8, rt_max=3.2)

        # Fitter usage with an artefact shoulder:
        ann = PeakAnnotation(
            molecule_id="s0",
            rt_min=2.8,
            rt_max=3.2,
            mode="artefact_doublet",
            artefact_side="right",
        )
    """

    model_config = ConfigDict(frozen=True)

    molecule_id: str
    rt_min: float
    rt_max: float
    mode: PeakMode = "single"
    peak_model: PeakModel = "skew_normal"
    artefact_side: ArtefactSide | None = None
    vary_separation: bool = False
    include_artefact_in_area: bool = False
    wavelength: float | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_components(self) -> int:
        """Number of skew-normal components implied by ``mode``.

        Returns:
            1 for ``"single"``, 2 for ``"artefact_doublet"`` or ``"free_doublet"``.
        """
        return 1 if self.mode == "single" else 2

    @model_validator(mode="after")
    def _validate(self) -> PeakAnnotation:
        if self.mode == "artefact_doublet" and self.artefact_side is None:
            raise ValueError(
                "artefact_side must be 'left' or 'right' when mode='artefact_doublet'."
            )
        if self.mode != "artefact_doublet" and self.artefact_side is not None:
            raise ValueError(f"artefact_side must be None when mode='{self.mode}'.")
        if self.vary_separation and self.mode != "free_doublet":
            raise ValueError(
                f"vary_separation=True is only valid for mode='free_doublet', "
                f"got mode='{self.mode}'."
            )
        if self.include_artefact_in_area and self.mode != "artefact_doublet":
            raise ValueError(
                f"include_artefact_in_area=True is only valid for "
                f"mode='artefact_doublet', got mode='{self.mode}'."
            )
        if self.rt_max <= self.rt_min:
            raise ValueError(
                f"rt_max ({self.rt_max}) must be greater than rt_min ({self.rt_min})."
            )
        return self


class BaselineAnnotation(BaseModel):
    """A retention-time window marking a baseline region.

    Attributes:
        rt_min: Lower retention-time bound (minutes, inclusive).
        rt_max: Upper retention-time bound (minutes, inclusive).

    Example::

        bl = BaselineAnnotation(rt_min=0.5, rt_max=1.0)
    """

    model_config = ConfigDict(frozen=True)

    rt_min: float
    rt_max: float

    @model_validator(mode="after")
    def _validate(self) -> BaselineAnnotation:
        if self.rt_max <= self.rt_min:
            raise ValueError(
                f"rt_max ({self.rt_max}) must be greater than rt_min ({self.rt_min})."
            )
        return self


def check_baseline_peak_disjoint(
    peaks: list[PeakAnnotation],
    baselines: list[BaselineAnnotation],
) -> None:
    """Raise ``ValueError`` if any baseline window overlaps any peak window.

    Peak-peak and baseline-baseline overlaps are explicitly allowed and not
    checked. Touching boundaries (``a.rt_max == b.rt_min``) are not overlap.

    Args:
        peaks: Peak annotations to validate.
        baselines: Baseline annotations to validate.

    Raises:
        ValueError: If a baseline window overlaps any peak window.
    """
    for b in baselines:
        for p in peaks:
            if b.rt_min < p.rt_max and p.rt_min < b.rt_max:
                raise ValueError(
                    f"Baseline window [{b.rt_min}, {b.rt_max}] overlaps peak "
                    f"window [{p.rt_min}, {p.rt_max}] for molecule "
                    f"{p.molecule_id!r}. Baseline regions must be peak-free."
                )
