from __future__ import annotations

import pytest

from chromhandler.readers.shimadzu import (
    ShimadzuReader,
    _find_chromatogram_section,  # type: ignore[reportPrivateUsage]
    _find_peak_table_section,  # type: ignore[reportPrivateUsage]
    _parse_chromatogram_body,  # type: ignore[reportPrivateUsage]
    _parse_peak_table,  # type: ignore[reportPrivateUsage]
    _parse_sections,  # type: ignore[reportPrivateUsage]
)

READER = ShimadzuReader()

# ---------------------------------------------------------------------------
# Minimal fixture data
# ---------------------------------------------------------------------------

_CHROM_BODY = """\
# of Points	10
Interval(msec)	640
Wavelength(nm)	215
Bandwidth(nm)	4
R.Time (min)	Intensity
0,00000	0
0,01067	100
0,02133	-287
"""

_PEAK_BODY = """\
# of Peaks	2
Peak#\tR.Time\tI.Time\tF.Time\tArea\tHeight\tA/H\tConc.\tMark\tID#\tName\tk'\
\tPlate #\tPlate Ht.\tTailing\tResolution\tSep.Factor\tArea Ratio\tHeight Ratio\
\tConc. %\tNorm Conc.
1\t10,618\t10,517\t10,912\t25899\t3994\t6,484\t2,191\t   \t\t\t0,000\t54166\
\t2,769\t1,507\t0,000\t0,000\t0\t0\t0,000\t0,000
2\t11,160\t10,933\t11,509\t1156011\t125524\t9,209\t97,809\t   \t\t\t0,051\
\t30810\t4,868\t1,201\t2,484\t0,000\t0\t0\t0,000\t0,000
"""

_SECTIONS = {
    "Header": "Application Name\tLabSolutions\n",
    "Peak Table(PDA-Ch1)": _PEAK_BODY,
    "LC Status Trace(Pump A Pressure)": "some\tdata\n",
    "PDA Multi Chromatogram(Ch1)": _CHROM_BODY,
}

_FILE_CONTENT = """\
[Header]
Application Name\tLabSolutions

[Peak Table(PDA-Ch1)]
# of Peaks\t2
Peak#\tR.Time\tI.Time\tF.Time\tArea\tHeight\tA/H\tConc.\tMark\tID#\tName\tk'\
\tPlate #\tPlate Ht.\tTailing\tResolution\tSep.Factor\tArea Ratio\tHeight Ratio\
\tConc. %\tNorm Conc.
1\t10,618\t10,517\t10,912\t25899\t3994\t6,484\t2,191\t   \t\t\t0,000\t54166\
\t2,769\t1,507\t0,000\t0,000\t0\t0\t0,000\t0,000

[PDA Multi Chromatogram(Ch1)]
# of Points\t10
Wavelength(nm)\t215
R.Time (min)\tIntensity
0,00000\t0
0,01067\t100
"""


# ---------------------------------------------------------------------------
# _parse_sections
# ---------------------------------------------------------------------------


def test_parse_sections_returns_dict() -> None:
    from pathlib import Path

    sections = _parse_sections(_FILE_CONTENT, Path("dummy.txt"))
    assert "Header" in sections
    assert "PDA Multi Chromatogram(Ch1)" in sections
    assert "Peak Table(PDA-Ch1)" in sections


def test_parse_sections_bad_file_raises() -> None:
    from pathlib import Path

    with pytest.raises(ValueError, match=r"dummy\.txt"):
        _parse_sections("no section header here\n[Start]\ndata\n", Path("dummy.txt"))


# ---------------------------------------------------------------------------
# _find_chromatogram_section
# ---------------------------------------------------------------------------


def test_find_chromatogram_section_returns_correct_body() -> None:
    from pathlib import Path

    body = _find_chromatogram_section(_SECTIONS, Path("dummy.txt"))
    assert "Wavelength(nm)" in body
    assert "R.Time (min)" in body


def test_find_chromatogram_section_skips_lc_status() -> None:
    from pathlib import Path

    sections = {"LC Status Trace(Pump A Pressure)": "some\tdata\n"}
    with pytest.raises(ValueError, match=r"dummy\.txt"):
        _find_chromatogram_section(sections, Path("dummy.txt"))


# ---------------------------------------------------------------------------
# _find_peak_table_section
# ---------------------------------------------------------------------------


def test_find_peak_table_section_found() -> None:
    body = _find_peak_table_section(_SECTIONS)
    assert body is not None
    assert "Peak#" in body


def test_find_peak_table_section_missing_returns_none() -> None:
    assert _find_peak_table_section({"Header": "data\n"}) is None


# ---------------------------------------------------------------------------
# _parse_chromatogram_body
# ---------------------------------------------------------------------------


def test_parse_chromatogram_body_values() -> None:
    from pathlib import Path

    time, signal, wavelength = _parse_chromatogram_body(_CHROM_BODY, Path("dummy.txt"))
    assert time == pytest.approx([0.0, 0.01067, 0.02133])
    assert signal == pytest.approx([0.0, 100.0, -287.0])
    assert wavelength == pytest.approx(215.0)


def test_parse_chromatogram_body_no_wavelength() -> None:
    from pathlib import Path

    body = "R.Time (min)\tIntensity\n0,00000\t0\n"
    time, _, wavelength = _parse_chromatogram_body(body, Path("dummy.txt"))
    assert wavelength is None
    assert time == pytest.approx([0.0])


def test_parse_chromatogram_body_missing_header_raises() -> None:
    from pathlib import Path

    with pytest.raises(ValueError, match=r"dummy\.txt"):
        _parse_chromatogram_body("Wavelength(nm)\t215\nno data header\n", Path("dummy.txt"))


def test_parse_chromatogram_body_skips_blank_lines() -> None:
    from pathlib import Path

    body = "R.Time (min)\tIntensity\n0,00000\t1\n\n0,01067\t2\n"
    time, signal, _ = _parse_chromatogram_body(body, Path("dummy.txt"))
    assert len(time) == 2
    assert len(signal) == 2


# ---------------------------------------------------------------------------
# _parse_peak_table
# ---------------------------------------------------------------------------


def test_parse_peak_table_count() -> None:
    peaks = _parse_peak_table(_PEAK_BODY, "chrom_1")
    assert len(peaks) == 2


def test_parse_peak_table_values() -> None:
    peaks = _parse_peak_table(_PEAK_BODY, "chrom_1")
    assert peaks[0].location.mean == pytest.approx(10.618)
    assert peaks[0].area.mean == pytest.approx(25899.0)
    assert peaks[0].peak_start == pytest.approx(10.517)
    assert peaks[0].peak_end == pytest.approx(10.912)
    assert peaks[0].amplitude == pytest.approx(3994.0)


def test_parse_peak_table_chromatogram_id() -> None:
    peaks = _parse_peak_table(_PEAK_BODY, "my_chrom")
    assert all(p.chromatogram_id == "my_chrom" for p in peaks)


def test_parse_peak_table_no_header_returns_empty() -> None:
    peaks = _parse_peak_table("# of Peaks\t0\n", "chrom_1")
    assert peaks == []


def test_parse_peak_table_empty_returns_empty() -> None:
    peaks = _parse_peak_table("", "chrom_1")
    assert peaks == []
