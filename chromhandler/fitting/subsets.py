"""Subset specification and area record types for multi-subset fitting.

A :class:`Subset` defines which traces belong to a fitting group and which
peak/baseline annotations to use for that group.  After fitting, each subset
produces a list of :class:`AreaRecord` objects that map chromatogram IDs to
per-molecule peak areas.

:class:`SubsetSpec` is kept for backward compatibility but is deprecated; use
:class:`Subset` instead.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import field

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation

__all__ = ["Subset", "SubsetSpec", "AreaRecord"]


# ---------------------------------------------------------------------------
# Subset — mutable builder (new canonical API)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Subset:
    """Mutable fitting-subset builder.

    Defines which traces belong to this group (via *sample_ids* or
    *chromatogram_ids*) and accumulates peak/baseline annotations via
    :meth:`add_peak_annotation` and :meth:`add_baseline_annotation`.

    Use :meth:`~chromhandler.fitting.better_fitter.BetterFitter.add_subset`
    to create instances attached to a fitter.

    Attributes:
        name: Unique label for this subset.
        sample_ids: Include all chromatograms whose *sample_id* appears here.
        chromatogram_ids: Include individual chromatograms by *chromatogram_id*.
        peaks: Peak-window annotations accumulated via :meth:`add_peak_annotation`.
        baselines: Baseline annotations accumulated via
            :meth:`add_baseline_annotation`.  When empty the fitter's global
            baselines are used as a fallback.

    Example::

        fitter = BetterFitter.from_handler(handler)
        s = fitter.add_subset("column_A", sample_ids=["run1", "run2"])
        s.add_peak_annotation(
            PeakAnnotation(molecule_id="NAD", rt_min=2.8, rt_max=3.2, mode="single")
        )
        s.add_baseline_annotation(BaselineAnnotation(rt_min=0.5, rt_max=1.0))
    """

    name: str
    sample_ids: list[str] = field(default_factory=list)
    chromatogram_ids: list[str] = field(default_factory=list)
    peaks: list[PeakAnnotation] = field(default_factory=list)
    baselines: list[BaselineAnnotation] = field(default_factory=list)
    # Internal flag: True for the auto-created "__default__" subset which
    # covers all traces in the fitter without explicit ID filtering.
    _match_all: bool = field(default=False, repr=False)

    def add_peak_annotation(self, ann: PeakAnnotation) -> None:
        """Append *ann* to this subset's peak windows.

        Args:
            ann: The :class:`~chromhandler.annotations.PeakAnnotation` to add.
        """
        self.peaks.append(ann)

    def add_baseline_annotation(self, ann: BaselineAnnotation) -> None:
        """Append *ann* to this subset's baseline regions.

        When at least one baseline annotation is registered on the subset it
        takes precedence over the fitter's global baseline list.

        Args:
            ann: The :class:`~chromhandler.annotations.BaselineAnnotation` to add.
        """
        self.baselines.append(ann)


# ---------------------------------------------------------------------------
# SubsetSpec — deprecated alias kept for backward compatibility
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SubsetSpec:
    """Deprecated: use :class:`Subset` instead.

    This class is kept for backward compatibility only and will be removed in
    a future release.  Pass a :class:`Subset` (obtained from
    :meth:`~chromhandler.fitting.better_fitter.BetterFitter.add_subset`) to
    add subset configurations to a fitter.
    """

    name: str
    peaks: list[PeakAnnotation]
    baselines: list[BaselineAnnotation] = field(default_factory=list)
    sample_ids: list[str] = field(default_factory=list)
    chromatogram_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        warnings.warn(
            "SubsetSpec is deprecated. Use BetterFitter.add_subset() which returns "
            "a Subset builder object instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self.peaks:
            raise ValueError(f"SubsetSpec '{self.name}': peaks must not be empty.")
        if not self.sample_ids and not self.chromatogram_ids:
            raise ValueError(
                f"SubsetSpec '{self.name}': at least one of sample_ids or "
                "chromatogram_ids must be non-empty."
            )


# ---------------------------------------------------------------------------
# AreaRecord — posterior area summary (unchanged)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AreaRecord:
    """Posterior area summary for one (chromatogram, molecule) pair.

    Attributes:
        chromatogram_id: Identifier of the source chromatogram — links back to
            :attr:`~chromhandler.model.Chromatogram.id` on the handler.
        molecule_id: Identifier of the molecule, taken from
            :attr:`~chromhandler.annotations.PeakAnnotation.molecule_id`.
        subset_name: Name of the :class:`Subset` that produced this record.
        area_median: Posterior median of the molecule-relevant peak area.
        area_q05: 5th-percentile credible interval bound.
        area_q95: 95th-percentile credible interval bound.

    Example::

        rec = AreaRecord(
            chromatogram_id="cw10_0min",
            molecule_id="s0",
            subset_name="group_A",
            area_median=12_450.3,
            area_q05=11_800.0,
            area_q95=13_100.5,
        )
    """

    chromatogram_id: str
    molecule_id: str
    subset_name: str
    area_median: float
    area_q05: float
    area_q95: float
