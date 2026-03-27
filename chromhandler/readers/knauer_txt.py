from __future__ import annotations

from pathlib import Path  # noqa: TC003 - used at runtime for path.read_text()

from chromhandler.model import Chromatogram

_KNAUER_HEADER_WORDS = ("Analyst", "SampleID", "Sample", "Sample", "Range")


class KnauerTXTReader:
    """Reader for ClarityChrom (Knauer HPLC) ASCII export files.

    Implements the :class:`AbstractReader` protocol: parses a single
    ClarityChrom ``.txt`` export and returns a
    :class:`~chromhandler.model.Chromatogram`.

    File format:
    - Header key/value lines (``Key : Value``)
    - Blank line
    - Column header: ``[Min.]\\t[mAU]``
    - Tab-separated time/signal rows with **comma decimal separator**

    Time is already in minutes per the ``[Min.]`` column header.
    The format contains no peak table, so ``peaks`` is always empty.

    Example::

        reader = KnauerTXTReader()
        chrom = reader.read_file(
            Path("knauer_0_min.txt"),
            chromatogram_id="knauer_0_min",
            sample_id="experiment_1",
            reaction_time=0.0,
        )
    """

    @classmethod
    def can_read(cls, path: Path) -> bool:
        """Return True if *path* contains a ClarityChrom (Knauer) TXT file."""
        try:
            txt_files = [p for p in path.iterdir() if p.is_file() and p.suffix == ".txt"]
            if not txt_files:
                return False
            lines = txt_files[0].read_text(encoding="utf-8", errors="ignore").splitlines()
            if len(lines) < 5:
                return False
            return all(
                lines[i].split()[0] == word
                for i, word in enumerate(_KNAUER_HEADER_WORDS)
                if lines[i].split()
            )
        except OSError:
            return False

    def read_file(
        self,
        path: Path,
        *,
        chromatogram_id: str,
        sample_id: str,
        reaction_time: float | None = None,
    ) -> Chromatogram:
        """Parse a single ClarityChrom TXT export file.

        Args:
            path: Path to the ``.txt`` file.
            chromatogram_id: Identifier for this chromatogram.
            sample_id: Identifier of the parent sample.
            reaction_time: Time since reaction start in minutes, or ``None``.

        Returns:
            A :class:`~chromhandler.model.Chromatogram` with signal and time
            arrays (both in minutes).  ``peaks`` is always ``[]``.

        Raises:
            ValueError: If the ``[Min.]`` data header line is not found.
        """
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        header_end = self._find_data_header(lines, path)
        time, signal = self._parse_data(lines[header_end + 1 :])

        return Chromatogram(
            id=chromatogram_id,
            sample_id=sample_id,
            signal=signal,
            time=time,
            reaction_time=reaction_time,
        )

    def _find_data_header(self, lines: list[str], path: Path) -> int:
        """Return the index of the ``[Min.]`` column header line.

        Raises:
            ValueError: If the header is not found.
        """
        for i, line in enumerate(lines):
            if line.startswith("[Min.]"):
                return i
        raise ValueError(f"Data header '[Min.]' not found in '{path}'.")

    def _parse_data(self, data_lines: list[str]) -> tuple[list[float], list[float]]:
        """Parse tab-separated time/signal rows.

        Replaces comma decimal separators with periods before conversion.
        Skips blank lines silently.
        """
        time: list[float] = []
        signal: list[float] = []
        for line in data_lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                time.append(float(parts[0].replace(",", ".")))
                signal.append(float(parts[1].replace(",", ".")))
        return time, signal
