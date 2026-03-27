from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import pandas as pd
from dotted_dict import DottedDict
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import pretty, visualize
from .annotations import PeakAnnotation, PeakWindow
from .enzymeml import handler_to_enzymeml_document
from .model import Chromatogram, InitialCondition, Peak, Sample
from .molecule import Molecule
from .protein import Protein
from .readers.utils import parse_reaction_time

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import numpy.typing as npt
    from matplotlib.figure import Figure
    from rich.console import Console, Group

    from .calibration import LinearCalibration
    from .fitting.better_fitter import BetterFitter
    from .readers.abstractreader import AbstractReader



class Handler(BaseModel):
    """Entry point for chromatographic data loading and analysis.

    Holds a collection of :class:`~chromhandler.model.Sample` objects, each
    containing one or more :class:`~chromhandler.model.Chromatogram` instances.
    Molecules and proteins are registered separately for peak annotation and
    downstream quantification, keyed by species id in
    ``DottedDict`` from ``dotted_dict`` (attribute access works when
    the id is a valid Python identifier). :attr:`peak_windows` uses the same type,
    keyed by molecule id.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    molecules: dict[str, Molecule] = Field(
        default_factory=DottedDict,
        description="Molecules keyed by :attr:`~chromhandler.molecule.Molecule.id`.",
    )
    proteins: dict[str, Protein] = Field(
        default_factory=DottedDict,
        description="Proteins keyed by :attr:`~chromhandler.protein.Protein.id`.",
    )
    samples: list[Sample] = Field(
        default_factory=list,
        description="Samples, each holding one or more chromatograms.",
    )
    peak_windows: dict[str, PeakWindow] = Field(
        default_factory=DottedDict,
        description="Retention-time peak windows keyed by molecule id.",
    )

    @model_validator(mode="after")
    def _validate_registries(self) -> Handler:
        """Validate registry key/ID alignment and ensure DottedDict containers."""
        for key, mol in self.molecules.items():
            if key != mol.id:
                raise ValueError(
                    f"molecules key {key!r} does not match Molecule.id {mol.id!r}. "
                    "Use create_molecule() or register_molecule() to add molecules."
                )
        for key, prot in self.proteins.items():
            if key != prot.id:
                raise ValueError(
                    f"proteins key {key!r} does not match Protein.id {prot.id!r}. "
                    "Use create_protein() or register_protein() to add proteins."
                )
        # Pydantic rebuilds dicts as plain dict during validation — restore DottedDict.
        if not isinstance(self.molecules, DottedDict):
            self.molecules = DottedDict(self.molecules)
        if not isinstance(self.proteins, DottedDict):
            self.proteins = DottedDict(self.proteins)
        if not isinstance(self.peak_windows, DottedDict):
            self.peak_windows = DottedDict(self.peak_windows)
        return self

    # ------------------------------------------------------------------
    # Core read method
    # ------------------------------------------------------------------

    def read_chromatogram(
        self,
        sample_id: str,
        chromatogram_id: str,
        file_path: Path | str,
        reaction_time: float | None,
        reader: AbstractReader,
    ) -> Chromatogram:
        """Parse one chromatogram file and attach it to the appropriate sample.

        If a :class:`~chromhandler.model.Sample` with *sample_id* already
        exists it is reused; otherwise a new one is created and appended.

        Args:
            sample_id: Identifier of the parent sample (e.g. ``"CV10"``).
            chromatogram_id: Identifier for this chromatogram (e.g. ``"CV10_0min"``).
            file_path: Path to the instrument data file.
            reaction_time: Time since reaction start in minutes, or ``None``.
            reader: Any object implementing :class:`~chromhandler.readers.abstractreader.AbstractReader`.

        Returns:
            The newly created :class:`~chromhandler.model.Chromatogram`.
        """
        chromatogram = reader.read_file(
            Path(file_path),
            chromatogram_id=chromatogram_id,
            sample_id=sample_id,
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
        *,
        mode: Literal["timecourse", "endpoint"] = "timecourse",
    ) -> Handler:
        """Read a directory (or directory-of-directories) of ASM JSON files.

        **Timecourse mode** (default) — reaction time extracted from each
        filename stem (e.g. ``"CV10_120min"`` → ``120.0 min``):

        *Dir-of-dirs* (one sample per sub-directory)::

            asm/
            ├── CV10/
            │   ├── CV10_0min.json
            │   └── CV10_30min.json
            └── CV11/
                └── CV11_0min.json

        *Flat* (all files are chromatograms of a single sample named after the
        directory)::

            asm/
            ├── CV10_0min.json
            └── CV10_30min.json

        **Endpoint mode** — each file in *path* becomes its own sample
        (``sample_id = file.stem``); no reaction time is extracted::

            asm/
            ├── condition_A.json   → sample "condition_A"
            └── condition_B.json   → sample "condition_B"

        Args:
            path: Root directory containing ASM JSON files or sub-directories.
            mode: ``"timecourse"`` (default) or ``"endpoint"``.

        Returns:
            A fully populated :class:`Handler`.
        """
        from .readers.asm import ASMReader

        root = Path(path)
        if not root.is_dir():
            raise NotADirectoryError(f"'{root}' is not a directory.")

        handler = cls()
        reader = ASMReader()

        if mode == "endpoint":
            json_files = sorted(
                p
                for p in root.iterdir()
                if p.is_file() and p.suffix == ".json" and not p.name.startswith(".")
            )
            if not json_files:
                raise FileNotFoundError(f"No ASM JSON files found under '{root}'.")
            for file in json_files:
                handler.read_chromatogram(file.stem, file.stem, file, None, reader)
            return handler

        # timecourse mode
        subdirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
        json_files = sorted(
            p for p in root.iterdir() if p.is_file() and p.suffix == ".json" and not p.name.startswith(".")
        )

        if subdirs:
            # Dir-of-dirs: each sub-directory is one sample.
            for sample_dir in subdirs:
                sample_id = sample_dir.name
                for file in sorted(
                    p
                    for p in sample_dir.iterdir()
                    if p.is_file() and p.suffix == ".json" and not p.name.startswith(".")
                ):
                    reaction_time = parse_reaction_time(file.stem)
                    handler.read_chromatogram(sample_id, file.stem, file, reaction_time, reader)
        elif json_files:
            # Flat layout: all files belong to one sample named after the directory.
            sample_id = root.name
            for file in json_files:
                reaction_time = parse_reaction_time(file.stem)
                handler.read_chromatogram(sample_id, file.stem, file, reaction_time, reader)
        else:
            raise FileNotFoundError(f"No ASM JSON files found under '{root}'.")

        for sample in handler.samples:
            sample.chromatograms.sort(key=lambda c: (c.reaction_time is None, c.reaction_time or 0))

        return handler

    @classmethod
    def read_knauer(
        cls,
        path: Path | str,
        *,
        mode: Literal["timecourse", "endpoint"] = "timecourse",
    ) -> Handler:
        """Read a directory of ClarityChrom (Knauer HPLC) TXT files.

        **Timecourse mode** (default) — reaction time extracted from each
        filename stem (e.g. ``"knauer_30_min"`` → ``30.0 min``):

        *Dir-of-dirs* (one sample per sub-directory)::

            data/
            ├── CV10/
            │   ├── knauer_0_min.txt
            │   └── knauer_30_min.txt
            └── CV11/
                └── knauer_0_min.txt

        *Flat* (all files are chromatograms of a single sample)::

            data/
            ├── knauer_0_min.txt
            └── knauer_30_min.txt

        **Endpoint mode** — each file becomes its own sample::

            data/
            ├── condition_A.txt   → sample "condition_A"
            └── condition_B.txt   → sample "condition_B"

        Args:
            path: Root directory containing TXT files or sub-directories.
            mode: ``"timecourse"`` (default) or ``"endpoint"``.

        Returns:
            A fully populated :class:`Handler`.
        """
        from .readers.knauer_txt import KnauerTXTReader

        root = Path(path)
        if not root.is_dir():
            raise NotADirectoryError(f"'{root}' is not a directory.")

        handler = cls()
        reader = KnauerTXTReader()

        if mode == "endpoint":
            txt_files = sorted(
                p for p in root.iterdir() if p.is_file() and p.suffix == ".txt" and not p.name.startswith(".")
            )
            if not txt_files:
                raise FileNotFoundError(f"No TXT files found under '{root}'.")
            for file in txt_files:
                handler.read_chromatogram(file.stem, file.stem, file, None, reader)
            return handler

        # timecourse mode
        subdirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
        txt_files = sorted(
            p for p in root.iterdir() if p.is_file() and p.suffix == ".txt" and not p.name.startswith(".")
        )

        if subdirs:
            for sample_dir in subdirs:
                sample_id = sample_dir.name
                for file in sorted(
                    p
                    for p in sample_dir.iterdir()
                    if p.is_file() and p.suffix == ".txt" and not p.name.startswith(".")
                ):
                    rt = parse_reaction_time(file.stem)
                    handler.read_chromatogram(sample_id, file.stem, file, rt, reader)
        elif txt_files:
            sample_id = root.name
            for file in txt_files:
                rt = parse_reaction_time(file.stem)
                handler.read_chromatogram(sample_id, file.stem, file, rt, reader)
        else:
            raise FileNotFoundError(f"No TXT files found under '{root}'.")

        for sample in handler.samples:
            sample.chromatograms.sort(key=lambda c: (c.reaction_time is None, c.reaction_time or 0))

        return handler

    @classmethod
    def read_shimadzu(
        cls,
        path: Path | str,
        *,
        mode: Literal["timecourse", "endpoint"] = "timecourse",
    ) -> Handler:
        """Read a directory of Shimadzu LabSolutions TXT export files.

        **Timecourse mode** (default) — reaction time extracted from each
        filename stem (e.g. ``"P0-0.0_min"`` → ``0.0 min``):

        *Dir-of-dirs* (one sample per sub-directory)::

            data/
            ├── sample_A/
            │   ├── P0-0.0_min.txt
            │   └── P1-30.0_min.txt
            └── sample_B/
                └── P0-0.0_min.txt

        *Flat* (all files are chromatograms of a single sample)::

            data/
            ├── P0-0.0_min.txt
            └── P1-30.0_min.txt

        **Endpoint mode** — each file becomes its own sample::

            data/
            ├── condition_A.txt   → sample "condition_A"
            └── condition_B.txt   → sample "condition_B"

        Args:
            path: Root directory containing TXT files or sub-directories.
            mode: ``"timecourse"`` (default) or ``"endpoint"``.

        Returns:
            A fully populated :class:`Handler`.
        """
        from .readers.shimadzu import ShimadzuReader

        root = Path(path)
        if not root.is_dir():
            raise NotADirectoryError(f"'{root}' is not a directory.")

        handler = cls()
        reader = ShimadzuReader()

        if mode == "endpoint":
            txt_files = sorted(
                p for p in root.iterdir() if p.is_file() and p.suffix == ".txt" and not p.name.startswith(".")
            )
            if not txt_files:
                raise FileNotFoundError(f"No TXT files found under '{root}'.")
            for file in txt_files:
                handler.read_chromatogram(file.stem, file.stem, file, None, reader)
            return handler

        # timecourse mode
        subdirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
        txt_files = sorted(
            p for p in root.iterdir() if p.is_file() and p.suffix == ".txt" and not p.name.startswith(".")
        )

        if subdirs:
            for sample_dir in subdirs:
                sample_id = sample_dir.name
                for file in sorted(
                    p
                    for p in sample_dir.iterdir()
                    if p.is_file() and p.suffix == ".txt" and not p.name.startswith(".")
                ):
                    rt = parse_reaction_time(file.stem)
                    handler.read_chromatogram(sample_id, file.stem, file, rt, reader)
        elif txt_files:
            sample_id = root.name
            for file in txt_files:
                rt = parse_reaction_time(file.stem)
                handler.read_chromatogram(sample_id, file.stem, file, rt, reader)
        else:
            raise FileNotFoundError(f"No TXT files found under '{root}'.")

        for sample in handler.samples:
            sample.chromatograms.sort(key=lambda c: (c.reaction_time is None, c.reaction_time or 0))

        return handler

    @classmethod
    def read_agilent(
        cls,
        path: Path | str,
        *,
        mode: Literal["timecourse", "endpoint"] = "timecourse",
        wavelength: float | None = None,
        channel: str | None = None,
    ) -> Handler:
        """Read a directory of Agilent ``.D`` chromatogram directories.

        Each ``.D`` sub-directory represents one injection.  The ``rainbow``
        library is used to parse the proprietary binary files.

        **Timecourse mode** (default) — reaction time extracted from each
        ``.D`` directory stem (e.g. ``"run_30min.D"`` → ``30.0 min``):

        *Dir-of-dirs* (one sample per sub-directory)::

            data/
            ├── sample_A/
            │   ├── 0min.D
            │   └── 30min.D
            └── sample_B/
                └── 0min.D

        *Flat* (all ``.D`` dirs are chromatograms of a single sample)::

            data/
            ├── 0min.D
            └── 30min.D

        **Endpoint mode** — each ``.D`` directory becomes its own sample::

            data/
            ├── condition_A.D   → sample "condition_A"
            └── condition_B.D   → sample "condition_B"

        **Detector-file selection** — Agilent ``.D`` directories may contain
        multiple detector files (e.g. ``FID1A.CH``, ``TCD2B.CH``):

        * ``channel`` selects by file name (case-insensitive).
        * ``wavelength`` selects by DAD wavelength in nm (only for UV/DAD data).
        * If both are ``None`` and the directory holds exactly one file, that
          file is used automatically; otherwise a :exc:`ValueError` is raised.

        Args:
            path: Root directory containing ``.D`` sub-directories.
            mode: ``"timecourse"`` (default) or ``"endpoint"``.
            wavelength: DAD wavelength in nm to select, or ``None``.
            channel: Detector file name to select (e.g. ``"FID1A.CH"``), or
                ``None``.

        Returns:
            A fully populated :class:`Handler`.
        """
        from .readers.agilent import AgilentReader

        root = Path(path)
        if not root.is_dir():
            raise NotADirectoryError(f"'{root}' is not a directory.")

        handler = cls()
        reader = AgilentReader(wavelength=wavelength, channel=channel)

        def _is_d_dir(p: Path) -> bool:
            return p.is_dir() and p.name.endswith(".D") and not p.name.startswith(".")

        if mode == "endpoint":
            d_dirs = sorted(p for p in root.iterdir() if _is_d_dir(p))
            if not d_dirs:
                raise FileNotFoundError(f"No '.D' directories found under '{root}'.")
            for d_dir in d_dirs:
                handler.read_chromatogram(d_dir.stem, d_dir.stem, d_dir, None, reader)
            return handler

        # timecourse mode
        subdirs = sorted(
            p
            for p in root.iterdir()
            if p.is_dir() and not p.name.endswith(".D") and not p.name.startswith(".")
        )
        d_dirs_flat = sorted(p for p in root.iterdir() if _is_d_dir(p))

        if subdirs:
            for sample_dir in subdirs:
                sample_id = sample_dir.name
                for d_dir in sorted(p for p in sample_dir.iterdir() if _is_d_dir(p)):
                    rt = parse_reaction_time(d_dir.stem)
                    handler.read_chromatogram(sample_id, d_dir.stem, d_dir, rt, reader)
        elif d_dirs_flat:
            sample_id = root.name
            for d_dir in d_dirs_flat:
                rt = parse_reaction_time(d_dir.stem)
                handler.read_chromatogram(sample_id, d_dir.stem, d_dir, rt, reader)
        else:
            raise FileNotFoundError(f"No '.D' directories found under '{root}'.")

        for sample in handler.samples:
            sample.chromatograms.sort(key=lambda c: (c.reaction_time is None, c.reaction_time or 0))

        return handler

    @classmethod
    def read(
        cls,
        path: Path | str,
        *,
        mode: Literal["timecourse", "endpoint"] = "timecourse",
        channel: str | None = None,
        wavelength: float | None = None,
    ) -> Handler:
        """Auto-detect instrument format and read chromatography data.

        Tries each registered reader in order (Agilent → ASM → Knauer →
        Shimadzu) and delegates to the matching ``read_*`` classmethod.

        Agilent-specific kwargs (``channel``, ``wavelength``) are forwarded
        only when the Agilent format is detected; they are silently ignored
        for all other formats.

        Args:
            path: Root directory containing chromatography data.
            mode: ``"timecourse"`` (default) or ``"endpoint"``.
            channel: Agilent detector-file name (e.g. ``"FID1A.CH"``).
            wavelength: Agilent DAD wavelength in nm.

        Returns:
            A fully populated :class:`Handler`.

        Raises:
            NotADirectoryError: If *path* is not a directory.
            ValueError: If no registered reader recognises the contents of
                *path*.
        """
        from .readers import READERS, AgilentReader

        root = Path(path)
        if not root.is_dir():
            raise NotADirectoryError(f"'{root}' is not a directory.")

        for reader_cls in READERS:
            if reader_cls.can_read(root):
                if reader_cls is AgilentReader:
                    return cls.read_agilent(
                        root, mode=mode, channel=channel, wavelength=wavelength
                    )
                dispatch = {
                    "ASMReader": cls.read_asm,
                    "KnauerTXTReader": cls.read_knauer,
                    "ShimadzuReader": cls.read_shimadzu,
                }
                return dispatch[reader_cls.__name__](root, mode=mode)

        # Build a helpful error listing what was actually found.
        try:
            found = sorted({p.suffix or p.name for p in root.iterdir()})
        except OSError:
            found = []
        found_str = ", ".join(found) if found else "nothing"
        raise ValueError(
            f"No reader recognised the contents of '{root}' (found: {found_str}). "
            "Use a specific read_* method: read_agilent, read_asm, read_knauer, read_shimadzu."
        )

    # ------------------------------------------------------------------
    # Peak annotation management
    # ------------------------------------------------------------------

    def add_initial_condition(
        self,
        sample_id: str,
        molecule_id: str,
        init_conc: float,
        conc_unit: str,
    ) -> None:
        """Register an initial condition for a molecule."""
        sample = self._get_sample(sample_id)
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

        Only rows for samples that already exist in the handler are processed;
        extra sample_ids in the file are ignored. Raises if an existing sample
        has no non-null initial conditions in its row.

        Example::

            # sample_id as column
            df = pd.DataFrame({"sample_id": ["s1"], "A": [1.0], "B": [2.0]})

            # sample_id as index
            df = pd.DataFrame({"A": [1.0], "B": [2.0]}, index=pd.Index(["s1"], name="sample_id"))
        """
        if isinstance(path, (Path, str)):
            df = pd.read_csv(path)
        else:
            df = path.copy()

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

        existing_ids = {s.id for s in self.samples}
        for i, sample_id in enumerate(sample_ids):
            if sample_id not in existing_ids:
                continue
            added_any = False
            for mol_id in df_mol.columns:
                val = df_mol.iloc[i, df_mol.columns.get_loc(mol_id)]
                if not pd.isna(val):  # type: ignore[arg-type]
                    self.add_initial_condition(sample_id, str(mol_id), float(val), conc_unit)  # type: ignore[arg-type]
                    added_any = True
            if not added_any:
                raise ValueError(f"Sample '{sample_id}' has no initial conditions in the file.")

    # ------------------------------------------------------------------
    # Posterior area collection
    # ------------------------------------------------------------------

    def collect_areas(
        self,
        fitter: BetterFitter,
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
        for rec in fitter.area_records():
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
        fitter: BetterFitter,
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
            written (one per chromatogram x molecule pair).

        Example::

            fitter.fit(num_samples=1000, num_warmup=500)
            handler.write_fitted_peaks(fitter)

            chrom = handler.samples[0].chromatograms[0]
            peak  = chrom.peaks[0]
            print(peak.area.mean, peak.area.std, peak.area.q05, peak.area.q95)
        """
        peaks = fitter.to_peaks(quantiles=quantiles, n_samples=n_samples)

        # Build a flat chromatogram-id → Chromatogram index
        chrom_index: dict[str, Chromatogram] = {c.id: c for s in self.samples for c in s.chromatograms}

        for peak in peaks:
            chrom = chrom_index.get(peak.chromatogram_id)
            if chrom is None:
                print(
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
        verbose: bool = True,
    ) -> None:
        """Fit linear calibration curves for registered molecules.

        Uses samples with ``reaction_time == 0`` as calibration standards.
        Each such sample must have an :class:`~chromhandler.model.InitialCondition`
        with ``init_conc`` set for the target molecule.  Peak areas are read from
        :attr:`~chromhandler.model.Chromatogram.peaks`, preferring
        ``Peak.area.samples`` (posterior draws) → ``Peak.area.median`` →
        ``Peak.area.mean``.

        The fitted :class:`~chromhandler.calibration.LinearCalibration` is stored
        on ``Molecule.calibration`` as a side effect.  Retrieve it afterwards via
        ``handler.molecules["Ino"].calibration``.

        Args:
            molecule_ids: Which molecules to calibrate.  Default: every molecule
                registered on this handler.
            fit_intercept: Include an intercept term in the regression
                (default ``False`` — forces the curve through the origin).
            method: Calibration method.  Only ``"external"`` is implemented;
                ``"internal"`` raises :exc:`NotImplementedError`.
            verbose: Print a rich calibration-summary table to stdout
                (default ``True``).  Pass ``False`` to suppress all output.

        Raises:
            NotImplementedError: When *method* is ``"internal"``.
            ValueError: When *method* is unrecognised.

        Example::

            handler.write_fitted_peaks(fitter)
            handler.calibrate_molecules(molecule_ids=["Ino", "Hyp"])

            est = handler.molecules["Ino"].calibration.area_to_conc(12_500.0)
            print(est.mean, est.std)
        """
        from .calibration import calibrate_molecules as _cal

        _cal(
            self.molecules,
            self.samples,
            molecule_ids,
            fit_intercept=fit_intercept,
            method=method,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # EnzymeML export
    # ------------------------------------------------------------------

    def to_enzymeml(
        self,
        name: str,
        *,
        sample_ids: list[str] | None = None,
        temperature: float,
        temperature_unit: str,
        ph: float,
        to_concentration: bool = False,
        n_samples: int | None = None,
        extrapolate: bool = False,
    ) -> Any:
        """Export handler data as an :class:`EnzymeMLDocument`.

        Each :class:`~chromhandler.model.Sample` in *sample_ids* (or all
        samples if ``None``) is converted to one or more pyenzyme
        ``Measurement`` objects, where each molecule's timecourse is
        assembled from its :class:`~chromhandler.model.Chromatogram` peaks
        sorted by ``reaction_time``.

        Args:
            name: Document name.
            sample_ids: Samples to include.  ``None`` includes all.
            temperature: Reaction temperature value.
            temperature_unit: Temperature unit string (e.g. ``"Celsius"`` for pyenzyme/astropy).
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


        Returns:
            A :class:`pyenzyme.EnzymeMLDocument` ready for
            ``doc.to_json()`` / ``doc.to_yaml()``.

        Raises:
            ValueError: If *to_concentration* is ``True`` and any
                non-constant molecule is missing a calibration, or if any
                *sample_id* is not found.
            ImportError: If ``pyenzyme`` is not installed.
        """
        active_mols = [m for m in self.molecules.values() if not m.internal_standard]
        proteins = list(self.proteins.values())
        return handler_to_enzymeml_document(
            samples=self.samples,
            molecules=active_mols,
            proteins=proteins,
            name=name,
            sample_ids=sample_ids,
            temperature=temperature,
            temperature_unit=temperature_unit,
            ph=ph,
            to_concentration=to_concentration,
            n_samples=n_samples,
            extrapolate=extrapolate,
        )

    # ------------------------------------------------------------------
    # Molecule / protein management
    # ------------------------------------------------------------------

    def create_molecule(
        self,
        id: str,
        pubchem_cid: int | str,
        name: str | None = None,
        *,
        constant: bool = False,
        internal_standard: bool = False,
        calibration: LinearCalibration | None = None,
    ) -> Molecule:
        """Build a :class:`~chromhandler.molecule.Molecule`, register it, and return it.

        Same *id* replaces an existing molecule. If *name* is omitted, the title is
        fetched from PubChem using the CID.

        Args:
            id: Internal identifier (e.g. ``"Ino"``).
            pubchem_cid: PubChem compound ID (``int`` or digits-only ``str``).
            name: Display name; PubChem title when ``None``.
            constant: Concentration treated as constant over the experiment.
            internal_standard: Mark as internal standard.
            calibration: Optional pre-fitted linear calibration.

        Returns:
            The registered molecule instance.
        """
        cid_int = int(pubchem_cid)
        if name is None:
            # TODO: Implement PubChem API integration to fetch molecule name
            name = f"CID_{cid_int}"

        molecule = Molecule(
            id=id,
            pubchem_cid=cid_int,
            name=name,
            constant=constant,
            internal_standard=internal_standard,
            calibration=calibration,
        )

        self.molecules[molecule.id] = molecule

        return molecule

    def register_molecule(
        self,
        molecule: Molecule,
    ) -> None:
        """Register a pre-built molecule (deep-copied).

        Use after :meth:`~chromhandler.molecule.Molecule.read_json` or when reusing a
        ``Molecule`` across handlers. Matching :attr:`~chromhandler.molecule.Molecule.id`
        replaces the previous entry.
        """
        registered = copy.deepcopy(molecule)
        self.molecules[registered.id] = registered

    def create_protein(
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
        self.proteins[protein.id] = protein

    def register_protein(
        self,
        protein: Protein,
    ) -> None:
        """Add (or update) a protein."""
        nu_prot = copy.deepcopy(protein)

        self.proteins[nu_prot.id] = nu_prot

    def add_peak_window(
        self,
        molecule_id: str,
        rt_min: float,
        rt_max: float,
        *,
        wavelength: float | None = None,
    ) -> PeakWindow:
        """Add or replace the peak window for *molecule_id* on this handler."""
        if self.molecules.get(molecule_id) is None:
            raise ValueError(
                f"Molecule {molecule_id} not found. Define the molecule first "
                "with create_molecule() or register_molecule()."
            )
        window = PeakWindow(
            molecule_id=molecule_id,
            rt_min=rt_min,
            rt_max=rt_max,
            wavelength=wavelength,
        )
        self.peak_windows[molecule_id] = window
        return window

    # ------------------------------------------------------------------
    # Peak assignment
    # ------------------------------------------------------------------

    def get_peaks(self, molecule_id: str, *, wavelength: float | None = None) -> list[Peak]:
        """Collect all peaks assigned to *molecule_id* across all samples.

        Args:
            molecule_id: ID of the molecule.
            wavelength: If set, only chromatograms at this wavelength (nm) are
                searched; otherwise every chromatogram in each sample is included.

        Returns:
            Peaks with matching ``molecule_id`` (possibly empty).
        """
        peaks: list[Peak] = []
        for sample in self.samples:
            for chrom in sample.chromatograms:
                if wavelength is not None and chrom.wavelength != wavelength:
                    continue

                peaks.extend(peak for peak in chrom.peaks if peak.molecule_id == molecule_id)

        return peaks

    def unassign_peaks(
        self,
        *,
        chromatogram_ids: Sequence[str] | None = None,
        sample_ids: Sequence[str] | None = None,
        molecule_ids: Sequence[str] | None = None,
        reaction_times: Sequence[float] | None = None,
        time_tolerance: float = 1e-6,
    ) -> list[dict[str, Any]]:
        """Clear ``peak.molecule_id`` for peaks matching the provided filters.

        This is the manual counterpart to :meth:`assign_molecules`. It is
        useful after reviewing :meth:`plot` and deciding that one chromatogram
        or one molecule/time-point should be excluded from downstream
        quantification.

        Examples::

            # Remove every assigned peak from a bad chromatogram
            handler.unassign_peaks(chromatogram_ids=["CW10_120min"])

            # Remove only the Hyp assignment at 120 min in sample CW10
            handler.unassign_peaks(
                sample_ids=["CW10"],
                molecule_ids=["Hyp"],
                reaction_times=[120.0],
            )

        Args:
            chromatogram_ids: Restrict to specific chromatogram IDs.
            sample_ids: Restrict to specific sample IDs.
            molecule_ids: Restrict to specific assigned molecule IDs.
            reaction_times: Restrict to chromatograms whose
                ``reaction_time`` matches any provided value within
                ``time_tolerance``.
            time_tolerance: Absolute tolerance for matching
                ``reaction_times``.

        Returns:
            List of records describing the assignments that were removed.
        """
        if (
            chromatogram_ids is None
            and sample_ids is None
            and molecule_ids is None
            and reaction_times is None
        ):
            raise ValueError(
                "unassign_peaks requires at least one filter "
                "(chromatogram_ids, sample_ids, molecule_ids, or reaction_times)."
            )
        if time_tolerance < 0:
            raise ValueError("time_tolerance must be non-negative.")

        chromatogram_id_set: set[str] | None = None
        if chromatogram_ids is not None:
            requested_chrom_ids = list(dict.fromkeys(chromatogram_ids))
            available_chrom_ids = {chrom.id for sample in self.samples for chrom in sample.chromatograms}
            missing = [chrom_id for chrom_id in requested_chrom_ids if chrom_id not in available_chrom_ids]
            if missing:
                raise ValueError(f"Unknown chromatogram IDs: {missing}")
            chromatogram_id_set = set(requested_chrom_ids)

        sample_id_set: set[str] | None = None
        if sample_ids is not None:
            requested_sample_ids = list(dict.fromkeys(sample_ids))
            available_sample_ids = {sample.id for sample in self.samples}
            missing = [
                sample_id for sample_id in requested_sample_ids if sample_id not in available_sample_ids
            ]
            if missing:
                raise ValueError(f"Unknown sample IDs: {missing}")
            sample_id_set = set(requested_sample_ids)

        molecule_id_set: set[str] | None = None
        if molecule_ids is not None:
            requested_molecule_ids = list(dict.fromkeys(molecule_ids))
            available_molecule_ids = set(self.molecules.keys())
            available_molecule_ids.update(
                peak.molecule_id
                for sample in self.samples
                for chrom in sample.chromatograms
                for peak in chrom.peaks
                if peak.molecule_id is not None
            )
            missing = [
                molecule_id
                for molecule_id in requested_molecule_ids
                if molecule_id not in available_molecule_ids
            ]
            if missing:
                raise ValueError(f"Unknown molecule IDs: {missing}")
            molecule_id_set = set(requested_molecule_ids)

        reaction_time_values: list[float] | None = None
        if reaction_times is not None:
            reaction_time_values = [float(reaction_time) for reaction_time in dict.fromkeys(reaction_times)]

        removed: list[dict[str, Any]] = []
        for sample in self.samples:
            if sample_id_set is not None and sample.id not in sample_id_set:
                continue

            for chrom in sample.chromatograms:
                if chromatogram_id_set is not None and chrom.id not in chromatogram_id_set:
                    continue

                if reaction_time_values is not None:
                    if chrom.reaction_time is None:
                        continue
                    chrom_reaction_time = float(chrom.reaction_time)
                    if not any(
                        abs(chrom_reaction_time - reaction_time) <= time_tolerance
                        for reaction_time in reaction_time_values
                    ):
                        continue

                for peak_idx, peak in enumerate(chrom.peaks):
                    if peak.molecule_id is None:
                        continue
                    if molecule_id_set is not None and peak.molecule_id not in molecule_id_set:
                        continue

                    removed.append(
                        {
                            "sample_id": sample.id,
                            "chromatogram_id": chrom.id,
                            "reaction_time": chrom.reaction_time,
                            "peak_index": peak_idx,
                            "peak_rt": float(peak.location.mean),
                            "previous_molecule_id": peak.molecule_id,
                        }
                    )
                    peak.molecule_id = None

        return removed

    def assign_molecules(
        self,
        *,
        min_amplitude: float | None = None,
        on_multiple: Literal["raise", "skip"] = "raise",
        silent: bool = False,
    ) -> None:
        """Assign molecules to existing peaks using the handler's peak windows.

        For each configured window, **every** chromatogram in each sample is
        considered (typical time-course: one sample, many traces at different
        reaction times). If :attr:`~chromhandler.annotations.PeakWindow.wavelength`
        is set, only chromatograms with that wavelength are used.

        A molecule is assigned when exactly one peak per chromatogram falls
        inside its window after optional amplitude filtering. If multiple peaks
        match within the same chromatogram, the behavior depends on
        ``on_multiple``: ``"raise"`` aborts immediately and ``"skip"``
        leaves that chromatogram unassigned for the current molecule.
        """
        if min_amplitude is not None and min_amplitude < 0:
            raise ValueError("min_amplitude must be non-negative.")
        if on_multiple not in {"raise", "skip"}:
            raise ValueError("on_multiple must be either 'raise' or 'skip'.")
        self._validate_peak_windows()

        if not self.peak_windows:
            return

        targeted_ids = set(self.peak_windows)
        peak_targets: dict[tuple[str, int], str] = {}
        pending_assignments: list[tuple[str, int, str]] = []
        results: list[dict[str, Any]] = []
        chrom_index: dict[str, Chromatogram] = {
            chrom.id: chrom for sample in self.samples for chrom in sample.chromatograms
        }

        for molecule_id, window in self.peak_windows.items():
            molecule = self.molecules[molecule_id]
            assigned_peak_count = 0
            chromatograms_with_no_peaks: list[str] = []
            chromatograms_with_multiple_peaks: list[str] = []
            chromatograms_considered: list[str] = []

            for sample in self.samples:
                for chrom in self._chromatograms_for_peak_window(sample, window):
                    chromatograms_considered.append(chrom.id)

                    candidate_indices = self._peak_indices_in_window(
                        chrom,
                        window,
                        min_amplitude=min_amplitude,
                    )

                    if not candidate_indices:
                        chromatograms_with_no_peaks.append(chrom.id)
                        continue

                    if len(candidate_indices) > 1:
                        candidate_rts = [float(chrom.peaks[idx].location.mean) for idx in candidate_indices]
                        if on_multiple == "raise":
                            raise ValueError(
                                f"assign_molecules: molecule '{molecule_id}' matched multiple peaks "
                                f"in chromatogram '{chrom.id}' for window "
                                f"[{window.rt_min:.3f}, {window.rt_max:.3f}] "
                                f"(candidate RTs: {candidate_rts}). "
                                "Increase min_amplitude or split the handler first."
                            )
                        chromatograms_with_multiple_peaks.append(chrom.id)
                        continue

                    peak_idx = candidate_indices[0]
                    peak = chrom.peaks[peak_idx]

                    if (
                        peak.molecule_id is not None
                        and peak.molecule_id not in targeted_ids
                        and peak.molecule_id != molecule_id
                    ):
                        raise ValueError(
                            f"assign_molecules: peak at {peak.location.mean:.3f} min in "
                            f"chromatogram '{chrom.id}' is already assigned to molecule "
                            f"'{peak.molecule_id}', cannot also assign '{molecule_id}'."
                        )

                    peak_key = (chrom.id, peak_idx)
                    existing_target = peak_targets.get(peak_key)
                    if existing_target is not None and existing_target != molecule_id:
                        raise ValueError(
                            f"assign_molecules: chromatogram '{chrom.id}' peak at "
                            f"{peak.location.mean:.3f} min matched both '{existing_target}' "
                            f"and '{molecule_id}'. Split the handler or change peak windows."
                        )

                    peak_targets[peak_key] = molecule_id
                    pending_assignments.append((chrom.id, peak_idx, molecule_id))
                    assigned_peak_count += 1

            results.append(
                {
                    "molecule": molecule,
                    "window": window,
                    "assigned_peak_count": assigned_peak_count,
                    "chromatograms_with_no_peaks": chromatograms_with_no_peaks,
                    "chromatograms_with_multiple_peaks": chromatograms_with_multiple_peaks,
                    "chromatograms_considered": chromatograms_considered,
                    "min_amplitude": min_amplitude,
                    "on_multiple": on_multiple,
                }
            )

        for sample in self.samples:
            for chrom in sample.chromatograms:
                for peak in chrom.peaks:
                    if peak.molecule_id in targeted_ids:
                        peak.molecule_id = None

        for chrom_id, peak_idx, molecule_id in pending_assignments:
            chrom_index[chrom_id].peaks[peak_idx].molecule_id = molecule_id

        if not silent:
            pretty.display_molecule_assignment_report(self, results)

    # ------------------------------------------------------------------
    # Sample / chromatogram utilities
    # ------------------------------------------------------------------

    def set_dilution_factor(self, dilution_factor: float | int) -> None:
        """Set a uniform dilution factor on all samples.

        Args:
            dilution_factor: The dilution factor to apply.
        """

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

    def subset(self, chromatogram_ids: Sequence[str]) -> Handler:
        """Return a deep-copied handler containing only *chromatogram_ids*."""
        requested_ids = list(dict.fromkeys(chromatogram_ids))
        available_ids = {chrom.id for sample in self.samples for chrom in sample.chromatograms}
        missing = [chrom_id for chrom_id in requested_ids if chrom_id not in available_ids]
        if missing:
            raise ValueError(f"subset: chromatogram_ids not found in handler: {missing}")

        selected = set(requested_ids)
        child = copy.deepcopy(self)
        child.samples = []
        for sample in self.samples:
            chromatograms = [copy.deepcopy(chrom) for chrom in sample.chromatograms if chrom.id in selected]
            if not chromatograms:
                continue
            sample_copy = copy.deepcopy(sample)
            sample_copy.chromatograms = chromatograms
            child.samples.append(sample_copy)
        return child

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
        out: list[tuple[float, float]] = []
        range_list: list[slice | tuple[float, float]] = (
            [ranges] if not isinstance(ranges, list) else ranges  # type: ignore[arg-type]
        )
        for r in range_list:
            if isinstance(r, slice):
                lo = r.start if r.start is not None else float("-inf")
                hi = r.stop if r.stop is not None else float("inf")
            else:
                lo, hi = r[0], r[1]  # type: ignore[index]
            out.append((lo, hi))
        return out

    def _cut_chromatogram(self, chrom: Chromatogram, ranges: list[tuple[float, float]]) -> None:
        """Restrict chromatogram signal/time and peaks to ranges (in-place)."""

        def in_ranges(t: float) -> bool:
            return any(lo <= t <= hi for lo, hi in ranges)

        # Filter signal/time
        if chrom.time and chrom.signal:
            keep = [in_ranges(t) for t in chrom.time]
            chrom.signal = [s for s, k in zip(chrom.signal, keep, strict=False) if k]
            chrom.time = [t for t, k in zip(chrom.time, keep, strict=False) if k]

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
        show_chromatogram_ids: bool = False,
    ) -> tuple[Figure, npt.NDArray[np.object_]]:
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
            show_chromatogram_ids: If ``True``, annotate each plotted point
                with its chromatogram ID. This is handy when you want to
                identify a point first and then remove it with
                :meth:`unassign_peaks`.

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
            missing = [sample_id for sample_id in requested_ids if sample_id not in found_ids]
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

        for ax, sample in zip(axes, selected_samples, strict=False):  # type: ignore[arg-type]
            time_unit = "min"
            points_by_molecule: dict[str, list[tuple[float, float, str]]] = {}
            balance_by_time: dict[float, float] = {}

            chromatograms = sorted(
                sample.chromatograms,
                key=lambda c: (
                    c.reaction_time is None,
                    float(c.reaction_time) if c.reaction_time is not None else float("inf"),
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
                        float(peak.area.median) if peak.area.median is not None else float(peak.area.mean)
                    )
                    mol_id = peak.molecule_id
                    reaction_time = float(chrom.reaction_time)
                    points_by_molecule.setdefault(mol_id, []).append((reaction_time, area_value, chrom.id))
                    balance_by_time[reaction_time] = balance_by_time.get(reaction_time, 0.0) + area_value

            for mol_id, point_records in points_by_molecule.items():
                x_vals = [point[0] for point in point_records]
                y_vals = [point[1] for point in point_records]
                point_color = molecule_colors.get(mol_id, "C0")
                ax.scatter(
                    x_vals,
                    y_vals,
                    s=32,
                    alpha=0.9,
                    color=point_color,
                    label=mol_id,
                )
                if show_chromatogram_ids:
                    for x_val, y_val, chrom_id in point_records:
                        ax.annotate(
                            str(chrom_id),
                            (x_val, y_val),
                            textcoords="offset points",
                            xytext=(4.0, 4.0),
                            fontsize=7,
                            alpha=0.8,
                            color=point_color,
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
                by_label = dict(zip(labels, handles, strict=False))
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
        assigned_only: bool = False,
        overlay: bool = False,
        share_y: bool = False,
        show_peak_annotations: bool = True,
        peak_annotations: list[PeakAnnotation | PeakWindow] | None = None,
        chromatogram_ids: list[str] | None = None,
        show_legend: bool = True,
    ) -> tuple[Figure, Any]:
        """Plot all chromatograms; returns ``(fig, ax)`` or ``(fig, axes)``.

        The figure is not saved automatically. Use ``fig.savefig(...)`` or
        ``plt.show()`` as needed.

        Args:
            peak_annotations: Optional list of
                window-like objects to overlay as shaded regions. When omitted,
                the handler's stored :attr:`peak_windows` are shown.
            chromatogram_ids: Chromatogram IDs to display. An empty list means
                all chromatograms. Unknown IDs raise ``ValueError``.
        """
        if peak_annotations is None:
            peak_annotations = list(self.peak_windows.values())
        if chromatogram_ids is None:
            chromatogram_ids = []
        return visualize.visualize(
            self,
            n_cols=n_cols,
            figsize=figsize,
            width_per_ax=width_per_ax,
            height_per_ax=height_per_ax,
            show_peaks=show_peaks,
            rt_min=rt_min,
            rt_max=rt_max,
            assigned_only=assigned_only,
            overlay=overlay,
            share_y=share_y,
            show_peak_annotations=show_peak_annotations,
            peak_annotations=cast("list[PeakAnnotation] | None", peak_annotations),
            chromatogram_ids=chromatogram_ids,
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

    def _get_sample(self, sample_id: str) -> Sample:
        """Return sample by id. Raises ValueError if not found."""
        for sample in self.samples:
            if sample.id == sample_id:
                return sample
        available = [s.id for s in self.samples]
        raise ValueError(f"Sample '{sample_id}' not found. Available samples: {available}")

    def _get_or_create_sample(self, sample_id: str) -> Sample:
        for sample in self.samples:
            if sample.id == sample_id:
                return sample
        new_sample = Sample(id=sample_id)
        self.samples.append(new_sample)
        return new_sample

    def _validate_peak_windows(self) -> None:
        molecule_ids = set(self.molecules.keys())
        missing = sorted(set(self.peak_windows) - molecule_ids)
        if missing:
            raise ValueError(
                f"Peak windows reference unknown molecule ids: {missing}. "
                "Define the molecules on the handler before assigning peak windows."
            )

    @staticmethod
    def _chromatograms_for_peak_window(sample: Sample, window: PeakWindow) -> list[Chromatogram]:
        """Chromatograms in *sample* to search for peaks for this *window*."""
        if window.wavelength is not None:
            matching = [c for c in sample.chromatograms if c.wavelength == window.wavelength]
            if not matching:
                raise ValueError(
                    f"assign_molecules: no chromatogram with wavelength "
                    f"{window.wavelength} nm in sample '{sample.id}' for window "
                    f"'{window.molecule_id}'."
                )
            return matching
        return list(sample.chromatograms)

    @staticmethod
    def _peak_indices_in_window(
        chrom: Chromatogram,
        window: PeakWindow,
        *,
        min_amplitude: float | None,
    ) -> list[int]:
        matches: list[int] = []
        for idx, peak in enumerate(chrom.peaks):
            rt = float(peak.location.mean)
            if not (window.rt_min <= rt <= window.rt_max):
                continue
            if min_amplitude is not None and peak.amplitude is not None and peak.amplitude < min_amplitude:
                continue
            matches.append(idx)
        return matches
