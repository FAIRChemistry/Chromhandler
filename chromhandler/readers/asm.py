from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 - used at runtime for path.read_text()
from typing import Any, cast

from loguru import logger

from chromhandler.model import Chromatogram, Estimate, Peak


def _point_estimate(value: float) -> Estimate:
    """Wrap a scalar as a point estimate (mean only)."""
    return Estimate(mean=value)


class ASMReader:
    """Reader for Allotrope Simple Model (ASM) JSON files.

    Implements the :class:`AbstractReader` protocol: parses a single ASM file
    and returns a fully constructed :class:`~chromhandler.model.Chromatogram`.
    Supports both LC and GC aggregate document formats.

    Example::

        reader = ASMReader()
        chrom = reader.read_file(
            Path("CV10_0min.json"),
            chromatogram_id="CV10_0min",
            sample_id="CV10",
            reaction_time=0.0,
        )
    """

    @classmethod
    def can_read(cls, path: Path) -> bool:
        """Return True if *path* contains at least one ``.json`` file."""
        try:
            return any(p.is_file() and p.suffix == ".json" for p in path.iterdir())
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
        """Parse a single ASM JSON file.

        Args:
            path: Path to the ASM JSON file.
            chromatogram_id: Identifier for this chromatogram.
            sample_id: Identifier of the parent sample.
            reaction_time: Time since reaction start in minutes, or ``None``.

        Returns:
            A :class:`~chromhandler.model.Chromatogram` with signal, time
            (both in minutes), peaks, and wavelength if available.

        Raises:
            ValueError: If the document type is not recognised.
        """
        content: dict[str, Any] = json.loads(path.read_text())

        if "liquid chromatography aggregate document" in content:
            return self._map_lc(content, path, chromatogram_id, sample_id, reaction_time)
        if "gas chromatography aggregate document" in content:
            return self._map_gc(content, path, chromatogram_id, sample_id, reaction_time)

        raise ValueError(
            f"Unrecognised ASM document type in '{path}'. "
            "Expected 'liquid chromatography aggregate document' or "
            "'gas chromatography aggregate document'."
        )

    # ------------------------------------------------------------------
    # LC mapping
    # ------------------------------------------------------------------

    def _map_lc(
        self,
        content: dict[str, Any],
        path: Path,
        chromatogram_id: str,
        sample_id: str,
        reaction_time: float | None,
    ) -> Chromatogram:
        doc = content["liquid chromatography aggregate document"][
            "liquid chromatography document"
        ]

        if len(doc) > 1:
            logger.warning(
                f"More than one chromatogram found in '{path}'. Using the first one."
            )

        try:
            meas_document: Any = doc[0]["measurement document"]
        except KeyError:
            meas_document = doc[0]["measurement aggregate document"][
                "measurement document"
            ]

        if isinstance(meas_document, list):
            meas_document = meas_document[0]

        meas_document = cast("dict[str, Any]", meas_document)
        signal, time = self._extract_signal_time(meas_document, path)
        peaks = self._extract_lc_peaks(doc[0], meas_document, path, chromatogram_id)

        return Chromatogram(
            id=chromatogram_id,
            sample_id=sample_id,
            signal=signal,
            time=time,
            peaks=peaks,
            reaction_time=reaction_time,
        )

    # ------------------------------------------------------------------
    # GC mapping
    # ------------------------------------------------------------------

    def _map_gc(
        self,
        content: dict[str, Any],
        path: Path,
        chromatogram_id: str,
        sample_id: str,
        reaction_time: float | None,
    ) -> Chromatogram:
        doc = content["gas chromatography aggregate document"][
            "gas chromatography document"
        ]

        if len(doc) > 1:
            logger.warning(
                f"More than one chromatogram found in '{path}'. Using the first one."
            )

        meas_document: Any = doc[0]["measurement aggregate document"]["measurement document"]
        if isinstance(meas_document, list):
            meas_document = meas_document[0]

        meas_document = cast("dict[str, Any]", meas_document)
        signal, time = self._extract_signal_time(meas_document, path)
        peaks = self._extract_gc_peaks(meas_document, path, chromatogram_id)

        return Chromatogram(
            id=chromatogram_id,
            sample_id=sample_id,
            signal=signal,
            time=time,
            peaks=peaks,
            reaction_time=reaction_time,
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _extract_lc_peaks(
        self,
        doc_element: dict[str, Any],
        meas_document: dict[str, Any],
        path: Path,
        chromatogram_id: str,
    ) -> list[Peak]:
        """Extract peaks from LC document. Returns [] if peak list is absent."""
        raw: list[dict[str, Any]] = []
        if "peak list" in meas_document:
            raw = meas_document["peak list"].get("peak", [])
        elif "analyte aggregate document" in doc_element:
            analyte_docs = doc_element["analyte aggregate document"].get(
                "analyteDocument", []
            )
            if analyte_docs:
                raw = analyte_docs[0].get("peak list", {}).get("peak", [])
            if len(analyte_docs) > 1:
                logger.warning(
                    f"More than one analyte document in '{path}'. Using the first."
                )
        return self._map_peaks_safe(raw, path, chromatogram_id)

    def _extract_gc_peaks(
        self,
        meas_document: dict[str, Any],
        path: Path,
        chromatogram_id: str,
    ) -> list[Peak]:
        """Extract peaks from GC document. Returns [] if peak list is absent."""
        raw: list[dict[str, Any]] = []
        try:
            raw = meas_document["processed data document"]["peak list"]["peak"]
        except KeyError:
            pass
        return self._map_peaks_safe(raw, path, chromatogram_id)

    def _map_peaks_safe(
        self,
        raw_peaks: list[dict[str, Any]],
        path: Path,
        chromatogram_id: str,
    ) -> list[Peak]:
        """Map raw peak dicts to Peak, skipping malformed entries."""
        peaks: list[Peak] = []
        for p in raw_peaks:
            try:
                peaks.append(self._map_peak(p, chromatogram_id))
            except (KeyError, TypeError):  # noqa: PERF203 - expected to skip malformed peaks
                logger.debug(f"Skipping malformed peak in '{path}'")
                continue
        return peaks

    def _extract_signal_time(
        self, meas_document: dict[str, Any], path: Path
    ) -> tuple[list[float], list[float]]:
        """Extract and normalise signal and time arrays to minutes."""
        cube = meas_document["chromatogram data cube"]
        signal: list[float] = cube["data"]["measures"][0]
        time: list[float] = cube["data"]["dimensions"][0]

        time_unit: str = cube["cube-structure"]["dimensions"][0]["unit"]
        if time_unit == "s":
            time = [t / 60.0 for t in time]
        elif time_unit != "min":
            raise ValueError(f"Unrecognised time unit '{time_unit}' in '{path}'.")

        return signal, time

    def _map_peak(self, peak_dict: dict[str, Any], chromatogram_id: str) -> Peak:
        """Convert a raw ASM peak dict to a :class:`Peak` model instance."""
        area_entry = peak_dict["peak area"]
        area_value: float = area_entry["value"]
        area_unit: str = area_entry.get("unit", "")
        if area_unit == "mAU.s":
            area_value = area_value * 60.0

        def _to_min(entry: dict[str, Any]) -> float:
            value: float = entry["value"]
            if entry.get("unit") == "s":
                value /= 60.0
            return value

        width_est: Estimate | None = None
        if "peak width at half height" in peak_dict:
            width_est = _point_estimate(_to_min(peak_dict["peak width at half height"]))

        skew_est: Estimate | None = None
        try:
            skew_val = peak_dict["chromatographic peak asymmetry factor"]["value"]
            skew_est = _point_estimate(skew_val)
        except (KeyError, TypeError):
            pass

        return Peak(
            chromatogram_id=chromatogram_id,
            location=_point_estimate(_to_min(peak_dict["retention time"])),
            area=_point_estimate(area_value),
            skew=skew_est,
            width=width_est,
            amplitude=peak_dict["peak height"]["value"],
            percent_area=peak_dict["relative peak area"]["value"],
            peak_start=_to_min(peak_dict["peak start"]),
            peak_end=_to_min(peak_dict["peak end"]),
        )
