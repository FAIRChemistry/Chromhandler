"""Tests for the Agilent OpenLab summary-report (RDL) reader.

Four report files across two template revisions:

- ``docs/usage/data/agilent_rdl/`` — the two shipped examples, rev 2
  ("Cross Sequence Summary Report", two-space left margin). These parsed
  correctly before this reader was rewritten and must keep parsing identically.
- ``tests/test_readers/data/agilent_rdl/ATP_1_mM.txt`` — rev 2, but with a
  retention time the report wrapped across two physical lines.
- ``tests/test_readers/data/agilent_rdl/GATP_1.00_mM.txt`` — rev 1
  ("Sequence Summary Report", no left margin, narrower columns).
"""

from pathlib import Path

import pytest

from chromhandler.model import Measurement
from chromhandler.readers.agilent_rdl import AgilentRDLReader

REPO_ROOT = Path(__file__).parent.parent.parent
SHIPPED = REPO_ROOT / "docs" / "usage" / "data" / "agilent_rdl"
FIXTURES = Path(__file__).parent / "data" / "agilent_rdl"

REV1 = FIXTURES / "GATP_1.00_mM.txt"
REV2_WRAPPED = FIXTURES / "ATP_1_mM.txt"


def read_one(path: Path) -> Measurement:
    """Read a single report and return its Measurement."""
    reader = AgilentRDLReader(
        dirpath=str(path.parent),
        file_paths=[str(path)],
        values=[0.0],
        unit="min",
        ph=7.4,
        temperature=25.0,
        temperature_unit="Celsius",
        silent=True,
        mode="timecourse",
    )
    return reader.read()[0]


# ---------------------------------------------------------------------------
# rev 1 — the template that used to crash
# ---------------------------------------------------------------------------


def test_rev1_report_yields_all_eight_peaks() -> None:
    peaks = read_one(REV1).chromatograms[0].peaks
    assert len(peaks) == 8


def test_rev1_first_and_last_peak() -> None:
    peaks = read_one(REV1).chromatograms[0].peaks
    assert peaks[0].retention_time == pytest.approx(2.744)
    assert peaks[0].area == pytest.approx(37.2216)
    assert peaks[-1].retention_time == pytest.approx(13.198)
    assert peaks[-1].area == pytest.approx(1726.4999)


def test_rev1_peak_fields_are_all_populated() -> None:
    peak = read_one(REV1).chromatograms[0].peaks[0]
    assert peak.type == "BB"
    assert peak.width == pytest.approx(0.4746)
    assert peak.amplitude == pytest.approx(4.6769)
    assert peak.percent_area == pytest.approx(0.4221)


# ---------------------------------------------------------------------------
# rev 2 with a wrapped retention time — the silently dropped peak
# ---------------------------------------------------------------------------


def test_wrapped_retention_time_peak_is_not_dropped() -> None:
    peaks = read_one(REV2_WRAPPED).chromatograms[0].peaks
    assert len(peaks) == 4
    assert peaks[-1].retention_time == pytest.approx(12.015)
    assert peaks[-1].area == pytest.approx(5556.9249)


def test_wrapped_report_first_peak() -> None:
    peaks = read_one(REV2_WRAPPED).chromatograms[0].peaks
    assert peaks[0].retention_time == pytest.approx(3.417)
    assert peaks[0].area == pytest.approx(28.3015)


# ---------------------------------------------------------------------------
# Shared behaviour across both templates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("report", [REV1, REV2_WRAPPED])
def test_wavelength_is_read_from_the_signal_box(report: Path) -> None:
    assert read_one(report).chromatograms[0].wavelength == 254


@pytest.mark.parametrize("report", [REV1, REV2_WRAPPED])
def test_percent_areas_sum_to_one_hundred(report: Path) -> None:
    peaks = read_one(report).chromatograms[0].peaks
    assert sum(p.percent_area or 0.0 for p in peaks) == pytest.approx(100.0, abs=0.5)


def test_both_templates_read_in_one_pass() -> None:
    reader = AgilentRDLReader(
        dirpath=str(FIXTURES),
        file_paths=[str(REV2_WRAPPED), str(REV1)],
        values=[0.0, 1.0],
        unit="min",
        ph=7.4,
        temperature=25.0,
        temperature_unit="Celsius",
        silent=True,
        mode="timecourse",
    )
    measurements = reader.read()
    assert [len(m.chromatograms[0].peaks) for m in measurements] == [4, 8]
    assert [m.data.value for m in measurements] == [0.0, 1.0]


# ---------------------------------------------------------------------------
# Regression: the shipped example data must parse exactly as before
# ---------------------------------------------------------------------------

_SHIPPED_PEAKS = [
    (0.698, "BV", 0.4062, 53.0992, 7.2642, 0.3671),
    (1.169, "VV", 0.8468, 6094.3336, 783.0775, 42.1289),
    (2.756, "VB", 0.4315, 14.2114, 1.9341, 0.0982),
    (3.331, "BV", 0.7823, 7620.7030, 925.5433, 52.6804),
    (3.974, "VB", 1.5648, 381.8593, 55.0016, 2.6397),
    (5.770, "BB", 1.0452, 301.7014, 22.1066, 2.0856),
]


@pytest.mark.parametrize("name", ["M2_MJ_100_min.txt", "M3_102_min.txt"])
def test_shipped_example_data_is_unchanged(name: str) -> None:
    peaks = read_one(SHIPPED / name).chromatograms[0].peaks
    assert len(peaks) == len(_SHIPPED_PEAKS)
    for peak, (rt, typ, width, area, amplitude, percent) in zip(peaks, _SHIPPED_PEAKS):
        assert peak.retention_time == pytest.approx(rt)
        assert peak.type == typ
        assert peak.width == pytest.approx(width)
        assert peak.area == pytest.approx(area)
        assert peak.amplitude == pytest.approx(amplitude)
        assert peak.percent_area == pytest.approx(percent)
