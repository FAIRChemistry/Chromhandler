from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from chromhandler.model import Chromatogram

if TYPE_CHECKING:
    from pathlib import Path


class AgilentReader:
    """Reader for Agilent chromatography ``.D`` directories via the ``rainbow`` library.

    Implements the :class:`AbstractReader` protocol: parses a single ``.D``
    directory and returns a fully constructed :class:`~chromhandler.model.Chromatogram`.

    Selection of which detector file to read is controlled by two mutually
    exclusive constructor arguments:

    * ``channel`` — name of the detector file (e.g. ``"FID1A.CH"``); matched
      case-insensitively.
    * ``wavelength`` — DAD wavelength in nm; only available when the detector
      file stores wavelength labels (i.e. UV/DAD data).

    If both are ``None`` and the ``.D`` directory contains exactly one detector
    file, that file is used automatically.  If multiple files are present and
    no selector is given, a :exc:`ValueError` is raised.

    Example::

        reader = AgilentReader(channel="FID1A.CH")
        chrom = reader.read_file(
            Path("0min.D"),
            chromatogram_id="run_0min",
            sample_id="sample_A",
            reaction_time=0.0,
        )
    """

    def __init__(
        self,
        *,
        wavelength: float | None = None,
        channel: str | None = None,
    ) -> None:
        self.wavelength = wavelength
        self.channel = channel

    @classmethod
    def can_read(cls, path: Path) -> bool:
        """Return True if *path* (or a direct sub-directory) contains a ``.D`` directory."""
        try:
            if any(p.is_dir() and p.name.endswith(".D") for p in path.iterdir()):
                return True
            for sub in (
                p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")
            ):
                try:
                    if any(p.is_dir() and p.name.endswith(".D") for p in sub.iterdir()):
                        return True
                except OSError:  # noqa: PERF203
                    continue
        except OSError:
            pass
        return False

    def read_file(
        self,
        path: Path,
        *,
        chromatogram_id: str,
        sample_id: str,
        reaction_time: float | None = None,
    ) -> Chromatogram:
        """Parse a single Agilent ``.D`` directory.

        Args:
            path: Path to the ``.D`` directory.
            chromatogram_id: Identifier for this chromatogram.
            sample_id: Identifier of the parent sample.
            reaction_time: Time since reaction start in minutes, or ``None``.

        Returns:
            A :class:`~chromhandler.model.Chromatogram` with signal and time
            arrays (both in minutes).  ``peaks`` is always ``[]`` because
            ``rainbow`` does not expose peak data.

        Raises:
            ValueError: If the channel or wavelength cannot be resolved.
        """
        import rainbow as rb

        datadir = rb.read(str(path))
        datafile = self._select_datafile(datadir, path)
        time, signal, resolved_wavelength = self._extract_signal(datafile, path)

        return Chromatogram(
            id=chromatogram_id,
            sample_id=sample_id,
            signal=signal,
            time=time,
            peaks=[],
            reaction_time=reaction_time,
            wavelength=resolved_wavelength,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_datafile(self, datadir: object, path: Path) -> object:  # type: ignore[return]
        """Pick the detector file according to *channel* / *wavelength* / auto."""
        datafiles: list[object] = list(datadir.datafiles)  # type: ignore[attr-defined]

        if not datafiles:
            raise ValueError(f"No detector files found in '{path}'.")

        if self.channel is not None:
            return self._select_by_channel(datafiles, path)

        if self.wavelength is not None:
            return self._select_by_wavelength(datafiles, path)

        # Auto: only valid when a single datafile exists.
        if len(datafiles) == 1:
            return datafiles[0]

        names = [str(df.name) for df in datafiles]  # type: ignore[attr-defined]
        raise ValueError(
            f"'{path}' contains {len(datafiles)} detector files: {names}. "
            "Specify 'channel' (e.g. channel='FID1A.CH') to select one."
        )

    def _select_by_channel(self, datafiles: list[object], path: Path) -> object:
        """Return the datafile whose name matches *self.channel* case-insensitively."""
        assert self.channel is not None
        target = self.channel.upper()
        for df in datafiles:
            if str(df.name).upper() == target:  # type: ignore[attr-defined]
                return df
        available = [str(df.name) for df in datafiles]  # type: ignore[attr-defined]
        raise ValueError(
            f"Channel '{self.channel}' not found in '{path}'. "
            f"Available channels: {available}."
        )

    def _select_by_wavelength(self, datafiles: list[object], path: Path) -> object:
        """Return the datafile whose ylabels contain *self.wavelength* (±1 nm)."""
        assert self.wavelength is not None

        # Collect wavelength-capable datafiles (ylabels with at least one
        # non-empty label that can be parsed as a float).
        wl_datafiles: list[tuple[object, list[float]]] = []
        for df in datafiles:
            labels = cast("np.ndarray[tuple[int], np.dtype[np.str_]]", df.ylabels)  # type: ignore[attr-defined]
            parsed = _parse_wavelength_labels(labels)
            if parsed:
                wl_datafiles.append((df, parsed))

        if not wl_datafiles:
            names = [str(df.name) for df in datafiles]  # type: ignore[attr-defined]
            raise ValueError(
                f"None of the detector files in '{path}' contain wavelength data "
                f"(files: {names}). "
                "Use 'channel' to select a detector file instead."
            )

        # Search for the requested wavelength across all wavelength-capable files.
        for df, wavelengths in wl_datafiles:
            diffs = [abs(w - self.wavelength) for w in wavelengths]
            if min(diffs) <= 1.0:
                return df

        all_wavelengths = sorted({w for _, wls in wl_datafiles for w in wls})
        raise ValueError(
            f"Wavelength {self.wavelength} nm not found in '{path}' "
            f"(tolerance ±1 nm). "
            f"Available wavelengths: {all_wavelengths} nm."
        )

    def _extract_signal(
        self, datafile: object, path: Path
    ) -> tuple[list[float], list[float], float | None]:
        """Return (time_min, signal, resolved_wavelength).

        ``rainbow`` returns ``xlabels`` in minutes for Agilent data.
        ``data`` has shape ``(n_timepoints, n_wavelengths)``.
        """
        xlabels = cast("np.ndarray[tuple[int], np.dtype[np.float64]]", datafile.xlabels)  # type: ignore[attr-defined]
        data = cast("np.ndarray[tuple[int, int], np.dtype[np.float64]]", datafile.data)  # type: ignore[attr-defined]
        ylabels = cast("np.ndarray[tuple[int], np.dtype[np.str_]]", datafile.ylabels)  # type: ignore[attr-defined]

        resolved_wavelength: float | None = None
        if self.wavelength is not None:
            # Find the column index closest to the requested wavelength.
            parsed = _parse_wavelength_labels(ylabels)
            if parsed:
                diffs = [abs(w - self.wavelength) for w in parsed]
                col_idx = int(np.argmin(diffs))
                resolved_wavelength = parsed[col_idx]
                raw_signal = data[:, col_idx]
            else:
                raw_signal = data[:, 0]
        else:
            raw_signal = data[:, 0]

        # Some Agilent files have xlabels/data row counts that differ by one.
        # Truncate both to the shorter length to guarantee equal-length arrays.
        n = min(len(xlabels), len(raw_signal))
        time: list[float] = xlabels[:n].tolist()
        signal: list[float] = raw_signal[:n].tolist()

        return time, signal, resolved_wavelength


def _parse_wavelength_labels(
    labels: np.ndarray[tuple[int], np.dtype[np.str_]],
) -> list[float]:
    """Convert a numpy array of label strings to a list of floats.

    Returns an empty list for labels that are empty, blank, or non-numeric.
    """
    result: list[float] = []
    for label in labels:
        stripped = str(label).strip()
        if stripped:
            try:
                result.append(float(stripped))
            except ValueError:
                pass
    return result
