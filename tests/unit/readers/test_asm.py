from __future__ import annotations

from typing import Any

import pytest

from chromhandler.readers.asm import ASMReader
from chromhandler.readers.utils import parse_reaction_time

READER = ASMReader()
CHROM_ID = "test_chrom"

# ---------------------------------------------------------------------------
# Helpers: minimal dicts that mimic ASM JSON structure
# ---------------------------------------------------------------------------

def _make_cube(time: list[float], signal: list[float], time_unit: str) -> dict[str, Any]:
    return {
        "chromatogram data cube": {
            "cube-structure": {"dimensions": [{"unit": time_unit}]},
            "data": {
                "dimensions": [time],
                "measures": [signal],
            },
        }
    }


def _make_peak(
    rt_value: float,
    rt_unit: str = "s",
    area_value: float = 100.0,
    area_unit: str = "mAU.s",
    height: float = 50.0,
    start_value: float = 90.0,
    start_unit: str = "s",
    end_value: float = 110.0,
    end_unit: str = "s",
    percent_area: float = 25.0,
    asymmetry: float | None = None,
    width_value: float | None = None,
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "retention time": {"value": rt_value, "unit": rt_unit},
        "peak area": {"value": area_value, "unit": area_unit},
        "peak height": {"value": height, "unit": "mAU"},
        "peak start": {"value": start_value, "unit": start_unit},
        "peak end": {"value": end_value, "unit": end_unit},
        "relative peak area": {"value": percent_area, "unit": "%"},
        "chromatographic peak asymmetry factor": (
            {"value": asymmetry} if asymmetry is not None else None
        ),
    }
    if width_value is not None:
        d["peak width at half height"] = {"value": width_value, "unit": "s"}
    return d


# ---------------------------------------------------------------------------
# _extract_signal_time
# ---------------------------------------------------------------------------

def test_extract_signal_time_seconds() -> None:
    cube = _make_cube([0.0, 60.0, 120.0], [1.0, 2.0, 3.0], "s")
    signal, time = READER._extract_signal_time(cube, path=None)  # type: ignore[reportPrivateUsage,arg-type]
    assert time == pytest.approx([0.0, 1.0, 2.0])
    assert signal == [1.0, 2.0, 3.0]


def test_extract_signal_time_minutes() -> None:
    cube = _make_cube([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], "min")
    _, time = READER._extract_signal_time(cube, path=None)  # type: ignore[reportPrivateUsage,arg-type]
    assert time == [0.0, 1.0, 2.0]


def test_extract_signal_time_unknown_unit() -> None:
    cube = _make_cube([0.0], [1.0], "ms")
    with pytest.raises(ValueError, match="Unrecognised time unit"):
        READER._extract_signal_time(cube, path=None)  # type: ignore[reportPrivateUsage,arg-type]


# ---------------------------------------------------------------------------
# _map_peak
# ---------------------------------------------------------------------------

def test_map_peak_basic() -> None:
    """Values from CV6_20min.json first peak."""
    p = READER._map_peak(  # type: ignore[reportPrivateUsage]
        _make_peak(
            rt_value=162.815,
            area_value=3820321.2235922245,
            area_unit="mAU.s",
            height=78459.4296875,
            start_value=156.815,
            end_value=170.815,
            percent_area=13.519827166781642,
        ),
        CHROM_ID,
    )
    assert p.location.mean == pytest.approx(162.815 / 60, rel=1e-6)
    assert p.area.mean == pytest.approx(3820321.2235922245 * 60, rel=1e-6)
    assert p.amplitude == pytest.approx(78459.4296875, rel=1e-6)
    assert p.peak_start == pytest.approx(156.815 / 60, rel=1e-6)
    assert p.peak_end == pytest.approx(170.815 / 60, rel=1e-6)
    assert p.percent_area == pytest.approx(13.519827166781642, rel=1e-6)
    assert p.chromatogram_id == CHROM_ID


def test_map_peak_null_asymmetry() -> None:
    p = READER._map_peak(  # type: ignore[reportPrivateUsage]
        _make_peak(rt_value=100.0, asymmetry=None), CHROM_ID)
    assert p.skew is None


def test_map_peak_with_asymmetry() -> None:
    p = READER._map_peak(  # type: ignore[reportPrivateUsage]
        _make_peak(rt_value=100.0, asymmetry=1.23), CHROM_ID)
    assert p.skew is not None
    assert p.skew.mean == pytest.approx(1.23)


def test_map_peak_maus_area_unit() -> None:
    p = READER._map_peak(  # type: ignore[reportPrivateUsage]
        _make_peak(rt_value=100.0, area_value=1000.0, area_unit="mAU.s"), CHROM_ID)
    assert p.area.mean == pytest.approx(1000.0 * 60)


def test_map_peak_non_maus_area_unit() -> None:
    p = READER._map_peak(  # type: ignore[reportPrivateUsage]
        _make_peak(rt_value=100.0, area_value=1000.0, area_unit="mAU.min"), CHROM_ID)
    assert p.area.mean == pytest.approx(1000.0)


def test_map_peak_seconds_rt() -> None:
    p = READER._map_peak(  # type: ignore[reportPrivateUsage]
        _make_peak(rt_value=120.0, rt_unit="s"), CHROM_ID)
    assert p.location.mean == pytest.approx(2.0)


def test_map_peak_width_extracted() -> None:
    p = READER._map_peak(  # type: ignore[reportPrivateUsage]
        _make_peak(rt_value=100.0, width_value=6.0), CHROM_ID)
    assert p.width is not None
    assert p.width.mean == pytest.approx(6.0 / 60)


# ---------------------------------------------------------------------------
# _map_peaks_safe
# ---------------------------------------------------------------------------

def test_map_peaks_safe_skips_malformed() -> None:
    good = _make_peak(rt_value=100.0)
    bad: dict[str, Any] = {}  # missing all required keys
    peaks = READER._map_peaks_safe([good, bad, good], path=None, chromatogram_id=CHROM_ID)  # type: ignore[reportPrivateUsage,arg-type]
    assert len(peaks) == 2


def test_map_peaks_safe_empty() -> None:
    assert READER._map_peaks_safe([], path=None, chromatogram_id=CHROM_ID) == []  # type: ignore[reportPrivateUsage,arg-type]


# ---------------------------------------------------------------------------
# parse_reaction_time
# ---------------------------------------------------------------------------

def test_parse_reaction_time_min() -> None:
    assert parse_reaction_time("CV6_20min") == pytest.approx(20.0)


def test_parse_reaction_time_sec() -> None:
    assert parse_reaction_time("run_30sec") == pytest.approx(0.5)


def test_parse_reaction_time_h() -> None:
    assert parse_reaction_time("exp_2h") == pytest.approx(120.0)


def test_parse_reaction_time_float() -> None:
    assert parse_reaction_time("sample_1.5min") == pytest.approx(1.5)


def test_parse_reaction_time_uses_last_match() -> None:
    # prefix "CV10_" should not interfere
    assert parse_reaction_time("CV10_120min") == pytest.approx(120.0)


def test_parse_reaction_time_no_match() -> None:
    with pytest.raises(ValueError, match="Cannot extract reaction time"):
        parse_reaction_time("sample_A")
