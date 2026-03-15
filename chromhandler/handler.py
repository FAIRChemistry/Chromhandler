from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from calipytion.tools.utility import pubchem_request_molecule_name
from loguru import logger
from pydantic import BaseModel, Field
from rich.console import Console, Group

from . import pretty, visualize
from .annotations import PeakAnnotation
from .model import Chromatogram, InitialCondition, Peak, Sample
from .molecule import CalibrationMethod, LinearCalibration, Molecule
from .protein import Protein
from .readers.abstractreader import AbstractReader
from .utility import _resolve_chromatogram

# Matches numeric value + unit at the end (or anywhere) of a filename stem,
# e.g. "CV10_120min", "sample_30sec", "run_2h".
_TIME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(min|sec|h)\b", re.IGNORECASE)


def _fit_linear(
    x: np.ndarray,
    y: np.ndarray,
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
# EnzymeML export helpers (module-level pure functions)
# ---------------------------------------------------------------------------


def _enzml_filter_samples(
    samples: list[Sample],
    sample_ids: list[str] | None,
) -> list[Sample]:
    """Return the subset of *samples* matching *sample_ids* (or all if ``None``)."""
    if sample_ids is None:
        return list(samples)
    id_set = set(sample_ids)
    missing = id_set - {s.id for s in samples}
    if missing:
        raise ValueError(f"to_enzymeml: sample_ids not found in handler: {missing}")
    return [s for s in samples if s.id in id_set]


def _enzml_assert_calibrations(molecules: list[Molecule]) -> None:
    """Raise ``ValueError`` if any non-constant molecule is missing a calibration."""
    missing = [m.id for m in molecules if not m.constant and m.calibration is None]
    if missing:
        raise ValueError(
            f"to_enzymeml: to_concentration=True but the following molecules "
            f"have no calibration — run handler.calibrate_molecules() first: {missing}"
        )


def _enzml_add_species(
    doc: Any, molecules: list[Molecule], proteins: list[Protein]
) -> None:
    """Populate *doc* with SmallMolecule and Protein entries."""
    from pyenzyme import Protein as EnzymeMLProtein
    from pyenzyme import SmallMolecule

    pubchem_base = "https://pubchem.ncbi.nlm.nih.gov/compound/"
    for mol in molecules:
        kwargs: dict[str, Any] = {
            "id": mol.id,
            "name": mol.name,
            "constant": mol.constant,
        }
        if mol.pubchem_cid and mol.pubchem_cid > 0:
            kwargs["ld_id"] = f"{pubchem_base}{mol.pubchem_cid}"
        doc.small_molecules.append(SmallMolecule(**kwargs))

    for prot in proteins:
        doc.proteins.append(
            EnzymeMLProtein(
                id=prot.id,
                name=prot.name,
                constant=prot.constant,
                sequence=prot.sequence or "",
            )
        )


def _enzml_build_measurement(
    *,
    sample: Sample,
    molecules: list[Molecule],
    proteins: list[Protein],
    temperature: float,
    temperature_unit: str,
    ph: float,
    to_concentration: bool,
    extrapolate: bool,
    draw_idx: int | None,
) -> Any:
    """Build one pyenzyme ``Measurement`` for a single *sample* / *draw_idx* pair."""
    from pyenzyme import DataTypes, MeasurementData
    from pyenzyme import Measurement as EnzymeMLMeasurement

    meas_id = sample.id if draw_idx is None else f"{sample.id}_draw{draw_idx:04d}"

    # Chromatograms sorted by reaction_time (skip those with no time assigned)
    chroms = sorted(
        [c for c in sample.chromatograms if c.reaction_time is not None],
        key=lambda c: c.reaction_time,  # type: ignore[arg-type]
    )

    species_data = []

    for mol in molecules:
        md = _enzml_build_measurement_data(
            chroms=chroms,
            sample=sample,
            species=mol,
            to_concentration=to_concentration,
            extrapolate=extrapolate,
            draw_idx=draw_idx,
        )
        if md is not None:
            species_data.append(md)

    # Proteins: constant concentration stored in initial/prepared, no timecourse
    for prot in proteins:
        init_conc = prot.init_conc
        unit_str = str(prot.conc_unit) if prot.conc_unit else ""
        species_data.append(
            MeasurementData(
                species_id=prot.id,
                initial=init_conc,
                prepared=init_conc,
                data_unit=unit_str,
                data_type=DataTypes.CONCENTRATION,
                time_unit="min",
                data=[],
                time=[],
            )
        )

    return EnzymeMLMeasurement(
        id=meas_id,
        name=meas_id,
        temperature=temperature,
        temperature_unit=temperature_unit,
        ph=ph,
        species_data=species_data,
    )


def _enzml_build_measurement_data(
    *,
    chroms: list[Chromatogram],
    sample: Sample,
    species: Molecule,
    to_concentration: bool,
    extrapolate: bool,
    draw_idx: int | None,
) -> Any:
    """Build one ``MeasurementData`` for a single *(sample, molecule)* pair."""
    import math

    from pyenzyme import DataTypes, MeasurementData

    times: list[float] = []
    values: list[float] = []
    time_unit = "min"

    for chrom in chroms:
        if chrom.reaction_time_unit:
            time_unit = str(chrom.reaction_time_unit)

        peak = next((p for p in chrom.peaks if p.molecule_id == species.id), None)
        area = _enzml_resolve_area(peak, draw_idx)

        # Convert area → concentration when requested and calibration is available
        if to_concentration and area is not None and species.calibration is not None:
            try:
                value: float | None = species.calibration.area_to_conc(
                    area, extrapolate=extrapolate
                ).mean
            except ValueError:
                value = None  # outside calibration range and extrapolate=False
        else:
            value = area

        times.append(float(chrom.reaction_time))  # type: ignore[arg-type]
        values.append(value if value is not None else math.nan)

    # prepared = user-set initial concentration from InitialCondition
    ic = next(
        (ic for ic in sample.initial_conditions if ic.molecule_id == species.id),
        None,
    )
    prepared: float | None = (
        float(ic.init_conc) if ic is not None and ic.init_conc is not None else None
    )

    # initial = actual measured value at t=0 (fallback: prepared)
    t0_idx = next((i for i, t in enumerate(times) if t == 0.0), None)
    if t0_idx is not None and not math.isnan(values[t0_idx]):
        initial: float | None = values[t0_idx]
    else:
        initial = prepared

    data_type = DataTypes.CONCENTRATION if to_concentration else DataTypes.PEAK_AREA
    if to_concentration and species.calibration is not None:
        print(f"we are here: {species.calibration.conc_unit}")
        unit_str = species.calibration.conc_unit
    else:
        unit_str = ic.conc_unit if ic is not None and ic.conc_unit is not None else "AU"

    print(f"type of unit_str: {type(unit_str)}")
    return MeasurementData(
        species_id=species.id,
        initial=initial,
        prepared=prepared,
        data_unit=unit_str,
        data_type=data_type,
        time_unit=time_unit,
        data=values,
        time=times,
    )


def _enzml_resolve_area(peak: Any, draw_idx: int | None) -> float | None:
    """Extract a scalar area value from a ``Peak.area`` ``Estimate``.

    With *draw_idx* given and posterior samples available, returns the
    sample at position ``draw_idx % len(samples)``.  Otherwise returns
    the median (preferred) or mean point estimate.
    """
    if peak is None:
        return None
    if draw_idx is not None and peak.area.samples:
        return float(peak.area.samples[draw_idx % len(peak.area.samples)])
    if peak.area.median is not None:
        return float(peak.area.median)
    return float(peak.area.mean)


def _parse_reaction_time(stem: str) -> float:
    """Extract a reaction time (in minutes) from a filename stem.

    Supported units: ``min`` → as-is, ``sec`` → ÷60, ``h`` → ×60.
    The *last* match in the stem is used so that prefixes like ``CV10_`` are
    ignored when the time is at the end.

    Args:
        stem: Filename without extension, e.g. ``"CV10_120min"``.

    Returns:
        Reaction time in minutes.

    Raises:
        ValueError: If no time pattern is found in *stem*.
    """
    matches = _TIME_RE.findall(stem)
    if not matches:
        raise ValueError(
            f"Cannot extract reaction time from filename stem '{stem}'. "
            "Expected a pattern like '30min', '120sec', or '2h'."
        )
    value_str, unit = matches[-1]
    value = float(value_str)
    match unit.lower():
        case "min":
            return value
        case "sec":
            return value / 60.0
        case "h":
            return value * 60.0
        case _:
            raise ValueError(f"Unrecognised time unit '{unit}' in stem '{stem}'.")


class Handler(BaseModel):
    """Entry point for chromatographic data loading and analysis.

    Holds a collection of :class:`~chromhandler.model.Sample` objects, each
    containing one or more :class:`~chromhandler.model.Chromatogram` instances.
    Molecules and proteins are registered separately for peak annotation and
    downstream quantification.

    Example::

        handler = Handler.read_asm(
            path="/data/sahh-kinetics/asm",
            id="sahh-exp-01",
            name="SAHH kinetics HPLC",
        )
        handler.define_molecule(
            id="s0",
            pubchem_cid=439176,
            retention_time=4.2,
            auto_assign=True,
        )
    """

    id: str = Field(description="Unique identifier of the Handler.")
    name: str = Field(default="", description="Human-readable name.")

    molecules: list[Molecule] = Field(
        default_factory=list,
        description="Molecules registered for peak annotation / quantification.",
    )
    proteins: list[Protein] = Field(
        default_factory=list,
        description="Proteins present in the reaction.",
    )
    samples: list[Sample] = Field(
        default_factory=list,
        description="Samples, each holding one or more chromatograms.",
    )
    internal_standard: Molecule | None = Field(
        default=None,
        description="Internal standard molecule used for concentration normalisation.",
    )
    # ------------------------------------------------------------------
    # Core read method
    # ------------------------------------------------------------------

    def read_chromatogram(
        self,
        sample_id: str,
        chromatogram_id: str,
        file_path: Path | str,
        reaction_time: float,
        reader: AbstractReader,
    ) -> Chromatogram:
        """Parse one chromatogram file and attach it to the appropriate sample.

        If a :class:`~chromhandler.model.Sample` with *sample_id* already
        exists it is reused; otherwise a new one is created and appended.

        Args:
            sample_id: Identifier of the parent sample (e.g. ``"CV10"``).
            chromatogram_id: Identifier for this chromatogram (e.g. ``"CV10_0min"``).
            file_path: Path to the instrument data file.
            reaction_time: Time since reaction start, **in minutes**.
            reader: Any object implementing :class:`~chromhandler.readers.abstractreader.AbstractReader`.

        Returns:
            The newly created :class:`~chromhandler.model.Chromatogram`.
        """
        data = reader.read_file(Path(file_path), chromatogram_id=chromatogram_id)

        chromatogram = Chromatogram(
            id=chromatogram_id,
            sample_id=sample_id,
            signal=data.signal,
            time=data.time,
            peaks=data.peaks,
            wavelength=data.wavelength,
            reaction_time=reaction_time,
        )

        sample = self._get_or_create_sample(sample_id)
        sample.chromatograms.append(chromatogram)

        return chromatogram

    # ------------------------------------------------------------------
    # Convenience classmethods
    # ------------------------------------------------------------------

    @classmethod
    def read_asm(
        cls,
        path: Path | str,
        id: str,
        name: str = "",
    ) -> Handler:
        """Read a directory (or directory-of-directories) of ASM JSON files.

        **Dir-of-dirs** layout (one sample per sub-directory)::

            asm/
            ├── CV10/
            │   ├── CV10_0min.json
            │   └── CV10_30min.json
            └── CV11/
                └── CV11_0min.json

        **Flat** layout (single sample, files directly in *path*)::

            asm/
            ├── CV10_0min.json
            └── CV10_30min.json

        In the flat case *path.name* is used as the sample ID.

        Reaction times are extracted from each filename stem via
        :func:`_parse_reaction_time` (e.g. ``"CV10_120min"`` → ``120.0``).

        Args:
            path: Root directory containing ASM JSON files or sub-directories.
            id: Handler identifier.
            name: Optional human-readable name.

        Returns:
            A fully populated :class:`Handler`.
        """
        from .readers.asm import ASMReader

        root = Path(path)
        if not root.is_dir():
            raise NotADirectoryError(f"'{root}' is not a directory.")

        handler = cls(id=id, name=name)
        reader = ASMReader()

        subdirs = sorted(
            p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
        json_files = sorted(
            p
            for p in root.iterdir()
            if p.is_file() and p.suffix == ".json" and not p.name.startswith(".")
        )

        if subdirs:
            # Dir-of-dirs: each sub-directory is one sample.
            for sample_dir in subdirs:
                sample_id = sample_dir.name
                for file in sorted(
                    p
                    for p in sample_dir.iterdir()
                    if p.is_file()
                    and p.suffix == ".json"
                    and not p.name.startswith(".")
                ):
                    reaction_time = _parse_reaction_time(file.stem)
                    handler.read_chromatogram(
                        sample_id, file.stem, file, reaction_time, reader
                    )
        elif json_files:
            # Flat layout: all files belong to one sample named after the directory.
            sample_id = root.name
            for file in json_files:
                reaction_time = _parse_reaction_time(file.stem)
                handler.read_chromatogram(
                    sample_id, file.stem, file, reaction_time, reader
                )
        else:
            raise FileNotFoundError(f"No ASM JSON files found under '{root}'.")

        for sample in handler.samples:
            sample.chromatograms.sort(
                key=lambda c: (c.reaction_time is None, c.reaction_time or 0)
            )

        return handler

    # ------------------------------------------------------------------
    # Peak annotation management
    # ------------------------------------------------------------------

    def add_initial_condition(
        self,
        sample_id: str,
        molecule_id: str,
        init_conc: float,
        conc_unit: str,
    ) -> InitialCondition:
        """Register an initial condition for a molecule."""
        sample = self._get_or_create_sample(sample_id)
        sample.initial_conditions.append(
            InitialCondition(
                molecule_id=molecule_id,
                init_conc=init_conc,
                conc_unit=conc_unit,
            )
        )

    def load_initial_conditions(
        self,
        path: Path | str | pd.DataFrame,
        *,
        conc_unit: str,
    ) -> None:
        """Load initial conditions from a tabular layout.

        Columns = molecule_ids, rows = sample_ids. Values = floats (init_conc).
        sample_id: column, index level, or implicit index.

        Example::

            # sample_id as column
            df = pd.DataFrame({"sample_id": ["s1"], "A": [1.0], "B": [2.0]})

            # sample_id as index
            df = pd.DataFrame({"A": [1.0], "B": [2.0]}, index=pd.Index(["s1"], name="sample_id"))
        """
        if isinstance(path, Path) or isinstance(path, str):
            df = pd.read_csv(path)
        elif isinstance(path, pd.DataFrame):
            df = path.copy()
        else:
            raise ValueError(f"Invalid path type: {type(path)}")

        # Resolve sample_id: column, index level, or implicit index
        if "sample_id" in df.columns:
            sample_ids = df["sample_id"].astype(str)
            df_mol = df.drop(columns=["sample_id"])
        elif "sample_id" in (df.index.names or []):
            df = df.reset_index()
            sample_ids = df["sample_id"].astype(str)
            df_mol = df.drop(columns=["sample_id"])
        else:
            # Implicit index: index values are sample_ids
            sample_ids = pd.Series(df.index).astype(str)
            df_mol = df

        if df_mol.empty:
            raise ValueError("No molecule columns found.")

        for col in df_mol.columns:
            df_mol[col] = pd.to_numeric(df_mol[col], errors="coerce")

        for i, sample_id in enumerate(sample_ids):
            for mol_id in df_mol.columns:
                val = df_mol.iloc[i, df_mol.columns.get_loc(mol_id)]
                if pd.notna(val):
                    self.add_initial_condition(
                        sample_id, str(mol_id), float(val), conc_unit
                    )

    # ------------------------------------------------------------------
    # Posterior area collection
    # ------------------------------------------------------------------

    def collect_areas(
        self,
        fitter: object,
    ) -> dict[str, list[tuple[float, float]]]:
        """Map posterior molecule areas from a fitted fitter to reaction times.

        Iterates the :class:`~chromhandler.fitting.subsets.AreaRecord` objects
        produced by ``fitter.area_records()`` and joins each record to its
        chromatogram's ``reaction_time`` via the chromatogram ID.  Records
        whose chromatogram ID is not found in this handler are silently skipped.

        Args:
            fitter: A fitted :class:`~chromhandler.fitting.better_fitter.BetterFitter`
                instance (with or without subsets).

        Returns:
            Mapping of ``molecule_id`` → ``list[(reaction_time, area_median)]``
            sorted by ``reaction_time`` within each molecule.

        Example::

            timecourse = handler.collect_areas(fitter)
            for molecule_id, points in timecourse.items():
                times, areas = zip(*points)
        """
        chrom_to_rt: dict[str, float | None] = {
            c.id: c.reaction_time for s in self.samples for c in s.chromatograms
        }
        result: dict[str, list[tuple[float, float]]] = {}
        for rec in fitter.area_records():  # type: ignore[attr-defined]
            rt = chrom_to_rt.get(rec.chromatogram_id)
            if rt is None:
                continue
            result.setdefault(rec.molecule_id, []).append((rt, rec.area_median))
        # Sort each molecule's timecourse by reaction_time
        for mol_id in result:
            result[mol_id].sort(key=lambda x: x[0])
        return result

    def write_fitted_peaks(
        self,
        fitter: object,
        *,
        quantiles: tuple[float, float, float] = (0.05, 0.5, 0.95),
        n_samples: int | None = None,
    ) -> list[Peak]:
        """Write Bayesian posterior peak estimates into matching Chromatograms.

        Calls :meth:`~chromhandler.fitting.better_fitter.BetterFitter.to_peaks`
        and upserts each returned :class:`~chromhandler.model.Peak` into the
        :class:`~chromhandler.model.Chromatogram` whose ``id`` matches
        ``Peak.chromatogram_id``.  An existing peak whose ``molecule_id``
        matches is replaced in-place; otherwise the new peak is appended.

        After this call the handler's chromatograms carry full posterior
        statistics (mean, std, q05, q95, and optionally raw samples) in their
        ``Peak.area`` and ``Peak.location`` :class:`~chromhandler.model.Estimate`
        fields.

        Args:
            fitter: A fitted :class:`~chromhandler.fitting.better_fitter.BetterFitter`
                instance (:meth:`~chromhandler.fitting.better_fitter.BetterFitter.fit`
                must have been called).  Subset-mode fitters are supported; in
                that case peaks are aggregated across fitted child subsets.
            quantiles: ``(q_low, q_median, q_high)`` percentile levels forwarded
                to ``to_peaks()``.
            n_samples: Number of randomly-drawn posterior samples to embed in
                each :class:`~chromhandler.model.Estimate`.  ``None`` (default)
                stores no samples.

        Returns:
            The list of :class:`~chromhandler.model.Peak` objects that were
            written (one per chromatogram × molecule pair).

        Example::

            fitter.fit(num_samples=1000, num_warmup=500)
            handler.write_fitted_peaks(fitter)

            chrom = handler.samples[0].chromatograms[0]
            peak  = chrom.peaks[0]
            print(peak.area.mean, peak.area.std, peak.area.q05, peak.area.q95)
        """
        peaks = fitter.to_peaks(  # type: ignore[attr-defined]
            quantiles=quantiles, n_samples=n_samples
        )

        # Build a flat chromatogram-id → Chromatogram index
        chrom_index: dict[str, Chromatogram] = {
            c.id: c for s in self.samples for c in s.chromatograms if c.id is not None
        }

        for peak in peaks:
            chrom = chrom_index.get(peak.chromatogram_id)
            if chrom is None:
                logger.warning(
                    f"write_fitted_peaks: chromatogram {peak.chromatogram_id} not found — skipping.",
                )
                continue
            # Replace existing peak for this molecule, or append
            for i, existing in enumerate(chrom.peaks):
                if existing.molecule_id == peak.molecule_id:
                    chrom.peaks[i] = peak
                    break
            else:
                chrom.peaks.append(peak)

        return peaks

    def calibrate_molecules(
        self,
        molecule_ids: list[str] | None = None,
        *,
        fit_intercept: bool = False,
        method: str = "external",
    ) -> dict[str, LinearCalibration]:
        """Fit linear calibration curves for registered molecules.

        Uses samples with ``reaction_time == 0`` as calibration standards.
        Each such sample must have an :class:`~chromhandler.model.InitialCondition`
        with ``init_conc`` set for the target molecule.  Peak areas are read from
        :attr:`~chromhandler.model.Chromatogram.peaks`, preferring
        ``Peak.area.samples`` (posterior draws) → ``Peak.area.median`` →
        ``Peak.area.mean``.

        The fitted :class:`~chromhandler.molecule.LinearCalibration` is stored on
        ``Molecule.calibration`` and returned for immediate inspection.

        Args:
            molecule_ids: Which molecules to calibrate.  Default: every molecule
                registered on this handler.
            fit_intercept: Include an intercept term in the regression
                (default ``False`` — forces the curve through the origin).
            method: Calibration method.  Only ``"external"`` is implemented;
                ``"internal"`` raises :exc:`NotImplementedError`.

        Returns:
            Mapping of ``molecule_id`` → fitted
            :class:`~chromhandler.molecule.LinearCalibration`.

        Raises:
            NotImplementedError: When *method* is ``"internal"``.
            ValueError: When *method* is unrecognised.

        Example::

            handler.write_fitted_peaks(fitter)
            cals = handler.calibrate_molecules(molecule_ids=["Ino", "Hyp"])

            mol = handler.get_molecule("Ino")
            est = mol.calibration.area_to_conc(12_500.0)
            print(est.mean, est.std)
        """
        if method == "internal":
            raise NotImplementedError(
                "Internal standard calibration is not yet implemented."
            )
        if method != "external":
            raise ValueError(f"Unknown calibration method: {method!r}. Use 'external'.")

        targets = (
            [self.get_molecule(mid) for mid in molecule_ids]
            if molecule_ids is not None
            else list(self.molecules)
        )

        results: dict[str, LinearCalibration] = {}
        for molecule in targets:
            cal = self._external_calibration(molecule, fit_intercept=fit_intercept)
            if cal is not None:
                molecule.calibration = cal
                self._update_molecule(molecule)
                results[molecule.id] = cal

        return results

    def _external_calibration(
        self,
        molecule: Molecule,
        *,
        fit_intercept: bool,
    ) -> LinearCalibration | None:
        """Build a :class:`~chromhandler.molecule.LinearCalibration` from t=0 samples.

        Iterates all samples whose chromatograms include ``reaction_time == 0``,
        collects ``(init_conc, peak_area)`` pairs, and fits an OLS linear model.
        Returns ``None`` (with a warning) when fewer than 2 data points are found.
        """
        known_concs: list[float] = []
        areas_mean: list[float] = []
        area_samples_per_point: list[list[float]] = []
        chrom_ids: list[str] = []

        for sample in self.samples:
            # Only calibration standards (at least one chromatogram at t=0)
            if not any(
                c.reaction_time is not None and c.reaction_time == 0.0
                for c in sample.chromatograms
            ):
                continue

            # Known concentration for this molecule
            ic = next(
                (
                    ic
                    for ic in sample.initial_conditions
                    if ic.molecule_id == molecule.id
                ),
                None,
            )
            if ic is None:
                continue

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
            logger.warning(
                f"calibrate_molecules: molecule {molecule.id} has only {len(known_concs)} calibration point(s) "
                "(need ≥ 1) — skipping.",
            )
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
            conc_unit=ic.conc_unit.id,
            chromatogram_ids=chrom_ids,
            concentrations=known_concs,
            areas_mean=areas_mean,
        )

    # ------------------------------------------------------------------
    # EnzymeML export
    # ------------------------------------------------------------------

    def to_enzymeml(
        self,
        *,
        sample_ids: list[str] | None = None,
        temperature: float,
        temperature_unit: str,
        ph: float,
        to_concentration: bool = False,
        n_samples: int | None = None,
        extrapolate: bool = False,
        name: str | None = None,
    ) -> Any:
        """Export handler data as an :class:`EnzymeMLDocument`.

        Each :class:`~chromhandler.model.Sample` in *sample_ids* (or all
        samples if ``None``) is converted to one or more pyenzyme
        ``Measurement`` objects, where each molecule's timecourse is
        assembled from its :class:`~chromhandler.model.Chromatogram` peaks
        sorted by ``reaction_time``.

        Args:
            sample_ids: Samples to include.  ``None`` includes all.
            temperature: Reaction temperature value.
            temperature_unit: Temperature unit string (e.g. ``"celsius"``).
            ph: Reaction pH.
            to_concentration: Convert peak areas → concentrations via each
                molecule's :attr:`~chromhandler.molecule.Molecule.calibration`.
                Requires all non-constant molecules to have a fitted
                :class:`~chromhandler.molecule.LinearCalibration`; call
                :meth:`calibrate_molecules` first.
            n_samples: If ``None`` (default), produce **one** pyenzyme
                ``Measurement`` per sample using the point estimate
                (median → mean) of each peak area.  If an integer *N*,
                produce *N* measurements per sample — one per posterior
                draw — yielding a full posterior-predictive ensemble of
                concentration trajectories.
            extrapolate: Allow extrapolation beyond the calibration range
                (passed to :meth:`~chromhandler.molecule.LinearCalibration.area_to_conc`).
                Default ``False``.
            name: Document name.  Defaults to ``self.name`` or ``self.id``.

        Returns:
            A :class:`pyenzyme.EnzymeMLDocument` ready for
            ``doc.to_json()`` / ``doc.to_yaml()``.

        Raises:
            ValueError: If *to_concentration* is ``True`` and any
                non-constant molecule is missing a calibration, or if any
                *sample_id* is not found.
            ImportError: If ``pyenzyme`` is not installed.
        """
        from pyenzyme import EnzymeMLDocument

        targets = _enzml_filter_samples(self.samples, sample_ids)
        active_mols = [m for m in self.molecules if not m.internal_standard]

        if to_concentration:
            _enzml_assert_calibrations(active_mols)

        doc = EnzymeMLDocument(name=name or self.name or self.id)
        _enzml_add_species(doc, active_mols, self.proteins)

        draw_indices: list[int | None] = (
            list(range(n_samples)) if n_samples is not None else [None]
        )
        for sample in targets:
            for draw_idx in draw_indices:
                meas = _enzml_build_measurement(
                    sample=sample,
                    molecules=active_mols,
                    proteins=self.proteins,
                    temperature=temperature,
                    temperature_unit=temperature_unit,
                    ph=ph,
                    to_concentration=to_concentration,
                    extrapolate=extrapolate,
                    draw_idx=draw_idx,
                )
                doc.measurements.append(meas)

        return doc

    # ------------------------------------------------------------------
    # Molecule / protein management
    # ------------------------------------------------------------------

    def define_molecule(
        self,
        id: str,
        pubchem_cid: int,
        name: Optional[str] = None,
        is_internal_standard: bool = False,
    ) -> Molecule:
        """Define and register a molecule.

        Args:
            id: Internal identifier (e.g. ``"s0"``).
            pubchem_cid: PubChem compound ID.
            retention_time: Expected retention time in minutes, or ``None``.
            retention_tolerance: Tolerance in minutes for peak matching.
            init_conc: Initial concentration (optional).
            conc_unit: Concentration unit string (optional).
            name: Display name; fetched from PubChem if omitted.
            is_internal_standard: Mark as internal standard.

        Returns:
            The created/updated :class:`~chromhandler.molecule.Molecule`.
        """
        if name is None:
            name = pubchem_request_molecule_name(pubchem_cid)

        molecule = Molecule(
            id=id,
            pubchem_cid=pubchem_cid,
            name=name,
            internal_standard=is_internal_standard,
        )

        self._update_molecule(molecule)

        return molecule

    def add_molecule(
        self,
        molecule: Molecule,
        init_conc: Optional[float] = None,
        conc_unit: Optional[str] = None,
        retention_tolerance: Optional[float] = None,
        min_signal: float = 0.0,
        auto_assign: bool = False,
    ) -> None:
        """Add (or update) a molecule.

        Args:
            molecule: The molecule to add.
            init_conc: Override initial concentration.
            conc_unit: Override concentration unit.
            retention_tolerance: Override retention time tolerance.
            min_signal: Minimum peak area threshold.
            auto_assign: Immediately run peak assignment after adding.
        """
        new_mol = copy.deepcopy(molecule)

        if init_conc is not None:
            new_mol.init_conc = init_conc
        if conc_unit is not None:
            new_mol.conc_unit = conc_unit
        if retention_tolerance is not None:
            new_mol.retention_tolerance = retention_tolerance

        new_mol.min_signal = min_signal

        self._update_molecule(new_mol)

        if auto_assign and new_mol.has_retention_time:
            self._register_peaks(
                new_mol, new_mol.retention_tolerance, new_mol.wavelength
            )

    def get_molecule(self, molecule_id: str) -> Molecule:
        """Return the molecule with `molecule_id`.

        Raises:
            ValueError: If no molecule with that ID exists.
        """
        for molecule in self.molecules:
            if molecule.id == molecule_id:
                return molecule
        raise ValueError(f"Molecule with ID '{molecule_id}' not found.")

    def define_protein(
        self,
        id: str,
        name: str,
        sequence: str | None = None,
        organism: str | None = None,
        organism_tax_id: str | None = None,
        constant: bool = True,
    ) -> None:
        """Define and register a protein."""
        protein = Protein(
            id=id,
            name=name,
            sequence=sequence,
            organism=organism,
            organism_tax_id=organism_tax_id,
            constant=constant,
        )
        self._update_protein(protein)

    def add_protein(
        self,
        protein: Protein,
    ) -> None:
        """Add (or update) a protein."""
        nu_prot = copy.deepcopy(protein)

        self._update_protein(nu_prot)

    # ------------------------------------------------------------------
    # Peak assignment
    # ------------------------------------------------------------------

    def get_peaks(
        self, molecule_id: str, *, wavelength: float | None = None
    ) -> list[Peak]:
        """Collect all peaks assigned to *molecule_id* across all samples.

        Args:
            molecule_id: ID of the molecule.
            wavelength: Wavelength of the signal in nm.

        Returns:
            List of peaks assigned to the molecule.

        Raises:
            ValueError: If no matching peaks are found.
        """
        peaks: list[Peak] = []
        for sample in self.samples:
            if len(sample.chromatograms) > 1 and wavelength is None:
                raise ValueError(
                    f"Multiple chromatograms found for sample '{sample.id}', but no wavelength is specified."
                )

            for chrom in sample.chromatograms:
                if wavelength is not None and chrom.wavelength != wavelength:
                    continue

                peaks.extend(
                    peak for peak in chrom.peaks if peak.molecule_id == molecule_id
                )

        return peaks

    def _register_peaks(
        self,
        molecule: Molecule,
        ret_tolerance: float,
        wavelength: float | None,
        silent: bool = False,
    ) -> dict[str, Any]:
        """Assign peaks to *molecule* across all chromatograms.

        For each sample the chromatogram is resolved (by wavelength if given),
        then all peaks within ±*ret_tolerance* minutes of
        ``molecule.retention_time`` are candidates. The single closest peak
        is assigned; if multiple candidates exist, a warning is issued.

        Returns:
            Assignment summary dict with keys ``molecule``,
            ``assigned_peak_count``, ``measurements_with_multiple_peaks``,
            ``measurements_with_no_peaks``, ``retention_tolerance``.
        """
        if not molecule.has_retention_time:
            raise ValueError(f"Molecule '{molecule.id}' has no retention time.")

        assigned_peak_count = 0
        samples_with_multiple_peaks: list[dict[str, Any]] = []
        samples_with_no_peaks: list[str] = []

        for sample in self.samples:
            chrom = _resolve_chromatogram(sample.chromatograms, wavelength)

            candidate_peaks = [
                peak
                for peak in chrom.peaks
                if (
                    molecule.retention_time is not None
                    and abs(peak.location.mean - molecule.retention_time)
                    <= ret_tolerance
                    and peak.area.mean >= molecule.min_signal
                )
            ]

            if not candidate_peaks:
                samples_with_no_peaks.append(sample.id)

            elif len(candidate_peaks) == 1:
                candidate_peaks[0].molecule_id = molecule.id
                assigned_peak_count += 1
                logger.debug(
                    f"'{molecule.id}' assigned to peak at "
                    f"{candidate_peaks[0].location.mean} in sample '{sample.id}'."
                )

            else:
                closest = min(
                    candidate_peaks,
                    key=lambda p: abs(p.location.mean - molecule.retention_time)  # type: ignore[operator]
                    if molecule.retention_time
                    else float("inf"),
                )
                closest.molecule_id = molecule.id
                assigned_peak_count += 1
                samples_with_multiple_peaks.append(
                    {
                        "sample_id": sample.id,
                        "num_peaks": len(candidate_peaks),
                        "assigned_rt": closest.location.mean,
                        "all_rts": [p.location.mean for p in candidate_peaks],
                    }
                )
                logger.debug(
                    f"'{molecule.id}' assigned to closest peak at "
                    f"{closest.location.mean} in sample '{sample.id}'."
                )

        result: dict[str, Any] = {
            "molecule": molecule,
            "assigned_peak_count": assigned_peak_count,
            "measurements_with_multiple_peaks": samples_with_multiple_peaks,
            "measurements_with_no_peaks": samples_with_no_peaks,
            "retention_tolerance": ret_tolerance,
        }

        if not silent:
            self._print_peak_assignment_summary(
                molecule,
                assigned_peak_count,
                samples_with_multiple_peaks,
                samples_with_no_peaks,
                ret_tolerance,
            )

        return result

    def assign_all_peaks(self, silent_individual: bool = True) -> None:
        """Assign peaks for all molecules that have a retention time.

        Args:
            silent_individual: Suppress per-molecule output; show consolidated
                report only.
        """
        if not self.molecules:
            print("No molecules defined for peak assignment.")
            return

        assignment_results = []
        for molecule in self.molecules:
            if molecule.has_retention_time:
                result = self._register_peaks(
                    molecule,
                    molecule.retention_tolerance,
                    molecule.wavelength,
                    silent=silent_individual,
                )
                assignment_results.append(result)

        if assignment_results:
            self._display_consolidated_assignment_report(assignment_results)

    # ------------------------------------------------------------------
    # Sample / chromatogram utilities
    # ------------------------------------------------------------------

    def set_dilution_factor(self, dilution_factor: float) -> None:
        """Set a uniform dilution factor on all samples.

        Args:
            dilution_factor: The dilution factor to apply.

        Raises:
            ValueError: If *dilution_factor* is not a number.
        """
        if not isinstance(dilution_factor, float | int):
            raise ValueError("Dilution factor must be a float or integer.")

        for sample in self.samples:
            sample.dilution_factor = dilution_factor

    def get_chromatograms_by_wavelength(self, wavelength: float) -> list[Chromatogram]:
        """Return all chromatograms recorded at *wavelength* nm.

        Args:
            wavelength: Target wavelength in nm.

        Returns:
            Flat list of matching :class:`~chromhandler.model.Chromatogram` objects.
        """
        return [
            chrom
            for sample in self.samples
            for chrom in sample.chromatograms
            if chrom.wavelength == wavelength
        ]

    def cut_chromatograms(
        self,
        ranges: (slice | tuple[float, float] | list[slice] | list[tuple[float, float]]),
    ) -> None:
        """Restrict chromatograms to given time ranges (inclusive).

        Removes signal/time points and peaks whose x-values (time) fall outside
        the specified ranges. Boundaries are inclusive.

        Args:
            ranges: One or more time ranges. Each range is either a
                ``slice(start, stop)`` or ``(min, max)`` tuple, interpreted as
                [start, stop] inclusive. Data is kept where time falls in ANY
                of the ranges.

        Example::
            handler.cut_chromatograms([(0, 1), (2.5, 4)])
        """
        norm = self._normalize_cut_ranges(ranges)
        for sample in self.samples:
            for chrom in sample.chromatograms:
                self._cut_chromatogram(chrom, norm)

    def _normalize_cut_ranges(
        self,
        ranges: (slice | tuple[float, float] | list[slice] | list[tuple[float, float]]),
    ) -> list[tuple[float, float]]:
        """Convert slice/tuple input to list of (lo, hi) inclusive ranges."""
        if not isinstance(ranges, list):
            ranges = [ranges]
        out: list[tuple[float, float]] = []
        for r in ranges:
            if isinstance(r, slice):
                lo = r.start if r.start is not None else float("-inf")
                hi = r.stop if r.stop is not None else float("inf")
            else:
                lo, hi = r[0], r[1]
            out.append((lo, hi))
        return out

    def _cut_chromatogram(
        self, chrom: Chromatogram, ranges: list[tuple[float, float]]
    ) -> None:
        """Restrict chromatogram signal/time and peaks to ranges (in-place)."""

        def in_ranges(t: float) -> bool:
            return any(lo <= t <= hi for lo, hi in ranges)

        # Filter signal/time
        if chrom.time and chrom.signal:
            keep = [in_ranges(t) for t in chrom.time]
            chrom.signal = [s for s, k in zip(chrom.signal, keep) if k]
            chrom.time = [t for t, k in zip(chrom.time, keep) if k]

        # Filter peaks
        chrom.peaks = [p for p in chrom.peaks if in_ranges(p.location.mean)]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self, path: str | Path) -> None:
        """Serialise to a JSON file, creating parent directories as needed."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.model_dump_json(indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> Handler:
        """Deserialise from a JSON file produced by :meth:`to_json`.

        Raises:
            FileNotFoundError: If *path* does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        data = json.loads(Path(path).read_text())
        return cls(**data)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def plot(
        self,
        samples: list[str] | None = None,
        figsize: tuple[float, float] | None = None,
        show_balance: bool = False,
        colors: dict[str, str] | None = None,
    ) -> tuple[object, np.ndarray]:
        """Plot peak areas over reaction time for one or more samples.

        Creates one subplot per selected sample in a single column. Each point
        is a peak area from one chromatogram, plotted against that
        chromatogram's ``reaction_time``.

        Args:
            samples: Optional list of sample IDs to plot. If ``None``, all
                samples in :attr:`samples` are included.
            figsize: Optional matplotlib figure size. When ``None``, the figure
                height grows with the number of plotted samples.
            show_balance: If ``True``, draw a faint dashed line showing the sum
                of assigned peak areas at each reaction time.
            colors: Optional dict mapping molecule ID to a hex color string
                (e.g. ``{"SAHH": "#FF5733"}``).  Molecule IDs not present in
                the dict fall back to the default ``tab10`` colormap.

        Returns:
            ``(fig, axes)`` where *axes* is a 1-D numpy array of matplotlib axes.
        """
        import matplotlib.pyplot as plt

        if samples is None:
            selected_samples = list(self.samples)
        else:
            requested_ids = list(samples)
            requested_set = set(requested_ids)
            selected_samples = [s for s in self.samples if s.id in requested_set]
            found_ids = {s.id for s in selected_samples}
            missing = [
                sample_id for sample_id in requested_ids if sample_id not in found_ids
            ]
            if missing:
                raise ValueError(f"Unknown sample IDs: {missing}")

        if not selected_samples:
            raise ValueError("No samples available to plot.")

        n_samples = len(selected_samples)
        if figsize is None:
            figsize = (8.0, max(3.0 * n_samples, 3.5))

        fig, axes = plt.subplots(n_samples, 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        molecule_ids: list[str] = []
        for sample in selected_samples:
            for chrom in sample.chromatograms:
                for peak in chrom.peaks:
                    if peak.molecule_id is None:
                        continue
                    mol_id = peak.molecule_id
                    if mol_id not in molecule_ids:
                        molecule_ids.append(mol_id)

        cmap = plt.get_cmap("tab10")
        molecule_colors: dict[str, object] = {}
        for idx, mol_id in enumerate(molecule_ids):
            if colors is not None and mol_id in colors:
                molecule_colors[mol_id] = colors[mol_id]
            else:
                molecule_colors[mol_id] = cmap(idx % 10)

        for ax, sample in zip(axes, selected_samples):
            time_unit = "min"
            points_by_molecule: dict[str, tuple[list[float], list[float]]] = {}
            balance_by_time: dict[float, float] = {}

            chromatograms = sorted(
                sample.chromatograms,
                key=lambda c: (
                    c.reaction_time is None,
                    float(c.reaction_time)
                    if c.reaction_time is not None
                    else float("inf"),
                ),
            )

            for chrom in chromatograms:
                if chrom.reaction_time is None:
                    continue
                if chrom.reaction_time_unit is not None:
                    time_unit = getattr(
                        chrom.reaction_time_unit,
                        "id",
                        str(chrom.reaction_time_unit),
                    )
                for peak in chrom.peaks:
                    if peak.molecule_id is None:
                        continue
                    area_value = (
                        float(peak.area.median)
                        if peak.area.median is not None
                        else float(peak.area.mean)
                    )
                    mol_id = peak.molecule_id
                    x_vals, y_vals = points_by_molecule.setdefault(mol_id, ([], []))
                    reaction_time = float(chrom.reaction_time)
                    x_vals.append(reaction_time)
                    y_vals.append(area_value)
                    balance_by_time[reaction_time] = (
                        balance_by_time.get(reaction_time, 0.0) + area_value
                    )

            for mol_id, (x_vals, y_vals) in points_by_molecule.items():
                ax.scatter(
                    x_vals,
                    y_vals,
                    s=32,
                    alpha=0.9,
                    color=molecule_colors.get(mol_id, "C0"),
                    label=mol_id,
                )

            if show_balance and balance_by_time:
                balance_points = sorted(balance_by_time.items())
                ax.plot(
                    [time for time, _ in balance_points],
                    [total for _, total in balance_points],
                    linestyle="--",
                    linewidth=1.5,
                    color="0.25",
                    alpha=0.4,
                    label="summed area",
                )

            if not points_by_molecule:
                ax.text(
                    0.5,
                    0.5,
                    "No peak areas available",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="0.4",
                )

            ax.set_title(str(sample.id), loc="left")
            ax.set_xlabel(f"time [{time_unit}]")
            ax.set_ylabel("signal [AU]")
            ax.grid(True, alpha=0.3)
            ax.patch.set_facecolor("none")

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                by_label = dict(zip(labels, handles))
                ax.legend(
                    by_label.values(),
                    by_label.keys(),
                    fontsize=8,
                    loc="upper left",
                    bbox_to_anchor=(1.01, 1.0),
                    borderaxespad=0.0,
                    frameon=True,
                )

        fig.patch.set_facecolor("none")
        fig.tight_layout()
        return fig, axes

    def visualize(
        self,
        n_cols: int = 2,
        figsize: tuple[float, float] | None = None,
        width_per_ax: float = 5.0,
        height_per_ax: float = 4.0,
        show_peaks: bool = True,
        rt_min: float | None = None,
        rt_max: float | None = None,
        save_path: str | None = None,
        assigned_only: bool = False,
        overlay: bool = False,
        show_peak_annotations: bool = True,
        peak_annotations: list[PeakAnnotation] | None = None,
        show_legend: bool = True,
    ) -> None:
        """Plot all chromatograms in a matplotlib grid.

        Args:
            peak_annotations: Optional list of
                :class:`~chromhandler.annotations.PeakAnnotation` objects to
                overlay as shaded windows.  Typically obtained from a
                :class:`~chromhandler.fitting.better_fitter.BetterFitter` via
                ``fitter.get_subset("__default__").peaks``.
        """
        visualize.visualize(
            self,
            n_cols=n_cols,
            figsize=figsize,
            width_per_ax=width_per_ax,
            height_per_ax=height_per_ax,
            show_peaks=show_peaks,
            rt_min=rt_min,
            rt_max=rt_max,
            save_path=save_path,
            assigned_only=assigned_only,
            overlay=overlay,
            show_peak_annotations=show_peak_annotations,
            peak_annotations=peak_annotations,
            show_legend=show_legend,
        )

    def rich_display(self, console: Console | None = None, debug: bool = False) -> None:
        """Display a rich-formatted overview of this Handler."""
        pretty.display_rich_handler(self, console, debug)

    def __rich__(self) -> Group:
        return pretty.create_rich_handler_group(self)

    def __call__(self) -> None:
        self.rich_display()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_or_create_sample(self, sample_id: str) -> Sample:
        for sample in self.samples:
            if sample.id == sample_id:
                return sample
        new_sample = Sample(id=sample_id)
        self.samples.append(new_sample)
        return new_sample

    def _update_molecule(self, molecule: Molecule) -> None:
        for idx, mol in enumerate(self.molecules):
            if mol.id == molecule.id:
                self.molecules[idx] = molecule
                return
        self.molecules.append(molecule)

    def _update_protein(self, protein: Protein) -> None:
        for idx, prot in enumerate(self.proteins):
            if prot.id == protein.id:
                self.proteins[idx] = protein
                return
        self.proteins.append(protein)

    def _display_consolidated_assignment_report(
        self, assignment_results: list[dict[str, Any]]
    ) -> None:
        pretty.display_consolidated_assignment_report(self, assignment_results)

    def _print_peak_assignment_summary(
        self,
        molecule: Molecule,
        assigned_peak_count: int,
        samples_with_multiple_peaks: list[dict[str, Any]],
        samples_with_no_peaks: list[str],
        ret_tolerance: float,
    ) -> None:
        pretty.print_peak_assignment_summary(
            self,
            molecule,
            assigned_peak_count,
            samples_with_multiple_peaks,
            samples_with_no_peaks,
            ret_tolerance,
        )
