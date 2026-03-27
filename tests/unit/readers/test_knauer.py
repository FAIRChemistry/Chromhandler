from __future__ import annotations

import pytest

from chromhandler.readers.knauer_txt import KnauerTXTReader
from chromhandler.readers.utils import parse_reaction_time

READER = KnauerTXTReader()

_SAMPLE_LINES = [
    "Analyst : Nucleoside",
    "Rate : 1 per sec.",
    "",
    "[Min.]\t[mAU]",
    "0\t0,001",
    "0,5\t1,25",
    "1\t-0,5",
]


# ---------------------------------------------------------------------------
# _find_data_header
# ---------------------------------------------------------------------------

def test_find_data_header() -> None:
    from pathlib import Path
    assert READER._find_data_header(_SAMPLE_LINES, Path("dummy.txt")) == 3  # type: ignore[reportPrivateUsage]


def test_find_data_header_missing() -> None:
    from pathlib import Path
    with pytest.raises(ValueError, match=r"dummy\.txt"):
        READER._find_data_header(["foo", "bar"], Path("dummy.txt"))  # type: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# _parse_data
# ---------------------------------------------------------------------------

def test_parse_data_comma_decimal() -> None:
    _, signal = READER._parse_data(  # type: ignore[reportPrivateUsage]
        ["0,5\t1,25"])
    assert signal == pytest.approx([1.25])


def test_parse_data_integer_values() -> None:
    _, signal = READER._parse_data(  # type: ignore[reportPrivateUsage]
        ["0\t0"])
    assert signal == pytest.approx([0.0])


def test_parse_data_negative_signal() -> None:
    _, signal = READER._parse_data(  # type: ignore[reportPrivateUsage]
        ["1\t-0,5"])
    assert signal == pytest.approx([-0.5])


def test_parse_data_skips_blank_lines() -> None:
    _, signal = READER._parse_data(  # type: ignore[reportPrivateUsage]
        ["0\t1", "", "1\t2"])
    assert len(signal) == 2


def test_parse_data_multiple_rows() -> None:
    data_lines = _SAMPLE_LINES[4:]  # ["0\t0,001", "0,5\t1,25", "1\t-0,5"]
    _, signal = READER._parse_data(  # type: ignore[reportPrivateUsage]
        data_lines)
    assert signal == pytest.approx([0.001, 1.25, -0.5])


def test_parse_data_empty() -> None:
    _, signal = READER._parse_data(  # type: ignore[reportPrivateUsage]
        [])
    assert signal == []


# ---------------------------------------------------------------------------
# parse_reaction_time — underscore separator (new regex behaviour)
# ---------------------------------------------------------------------------

def test_parse_reaction_time_underscore_zero() -> None:
    assert parse_reaction_time("knauer_0_min") == pytest.approx(0.0)


def test_parse_reaction_time_underscore_sep() -> None:
    assert parse_reaction_time("knauer_15_min") == pytest.approx(15.0)


def test_parse_reaction_time_underscore_large() -> None:
    assert parse_reaction_time("knauer_60_min") == pytest.approx(60.0)


def test_parse_reaction_time_existing_compact_still_works() -> None:
    """Existing format (no separator) must not regress."""
    assert parse_reaction_time("CV6_20min") == pytest.approx(20.0)
