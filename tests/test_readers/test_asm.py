import pytest

from chromhandler.readers.asm import ASMReader


@pytest.fixture
def asm_lc_1() -> ASMReader:
    reader = ASMReader(
        dirpath="docs/usage/data/asm",
        values=[0.0, 0.33, 3.4, 10],
        unit="min",
        ph=7.0,
        temperature=25.0,
        temperature_unit="Celsius",
        silent=False,
        mode="timecourse",
    )
    return reader


@pytest.fixture
def asm_lc_2() -> ASMReader:
    reader = ASMReader(
        dirpath="docs/usage/data/asm_2",
        values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        unit="min",
        ph=7.0,
        temperature=25.0,
        temperature_unit="Celsius",
        silent=False,
        mode="timecourse",
    )
    return reader


@pytest.fixture
def asm_gc_1() -> ASMReader:
    reader = ASMReader(
        dirpath="docs/usage/data/asm_3",
        values=[0.0, 0.33, 3.4, 10, 11],
        unit="min",
        ph=7.0,
        temperature=25.0,
        temperature_unit="Celsius",
        silent=False,
        mode="timecourse",
    )
    return reader


def test_read_asm_lc(asm_lc_1: ASMReader) -> None:
    measurements = asm_lc_1.read()
    assert len(measurements) == 4
    assert measurements[0].data.value == 0.0
    assert measurements[-1].data.value == 10.0
    assert measurements[0].chromatograms[0].peaks[0].peak_start == pytest.approx(
        81.962 / 60, rel=1e-3
    )
    assert measurements[0].chromatograms[0].peaks[0].peak_end == pytest.approx(
        129.762 / 60, rel=1e-3
    )
    assert measurements[0].chromatograms[0].peaks[0].amplitude == pytest.approx(
        2510394.25, rel=1e-1
    )
    assert measurements[0].chromatograms[0].peaks[0].retention_time == pytest.approx(
        86.562 / 60, rel=1e-3
    )
    assert measurements[0].chromatograms[0].peaks[0].percent_area == pytest.approx(
        0.1907055105844727, rel=1e-6
    )
    assert measurements[0].chromatograms[0].peaks[0].area == pytest.approx(
        1.4715719691118047e8 * 60, rel=1e-1
    )


def test_read_asm_lc_2(asm_lc_2: ASMReader) -> None:
    measurements = asm_lc_2.read()
    assert len(measurements) == 10
    assert measurements[0].data.value == 0.0
    assert measurements[-1].data.value == 9.0
    assert measurements[0].chromatograms[0].peaks[0].peak_start == pytest.approx(
        70.777 / 60, rel=1e-3
    )
    assert measurements[0].chromatograms[0].peaks[0].peak_end == pytest.approx(
        74.242 / 60, rel=1e-3
    )
    assert measurements[0].chromatograms[0].peaks[0].amplitude == pytest.approx(
        1893986.625, rel=1e-3
    )
    assert measurements[0].chromatograms[0].peaks[0].retention_time == pytest.approx(
        72.51 / 60, rel=1e-3
    )
    assert measurements[0].chromatograms[0].peaks[0].percent_area == pytest.approx(
        19.69565464074257, rel=1e-6
    )
    assert measurements[0].chromatograms[0].peaks[0].area == pytest.approx(
        1.702556410265625e7, rel=1e-1
    )


def test_read_asm_gc(asm_gc_1: ASMReader) -> None:
    measurements = asm_gc_1.read()
    assert len(measurements) == 5
    assert measurements[0].data.value == 0.0
    assert measurements[-1].data.value == 11.0
    assert measurements[0].chromatograms[0].peaks[0].peak_start == pytest.approx(
        1.76 / 60, rel=1e-3
    )
    assert measurements[0].chromatograms[0].peaks[0].peak_end == pytest.approx(
        2.56 / 60, rel=1e-3
    )
    assert measurements[0].chromatograms[0].peaks[0].amplitude == pytest.approx(
        80845.0, rel=1e-3
    )
    assert measurements[0].chromatograms[0].peaks[0].retention_time == pytest.approx(
        2.16 / 60, rel=1e-3
    )
    assert measurements[0].chromatograms[0].peaks[0].percent_area == pytest.approx(
        0.00567822271062627, rel=1e-6
    )
    assert measurements[0].chromatograms[0].peaks[0].area == pytest.approx(
        316139.0365283202, rel=1e-3
    )


def test_missing_peak_list_returns_empty_peaks(asm_lc_1: ASMReader) -> None:
    content = {
        "liquid chromatography aggregate document": {
            "liquid chromatography document": [
                {
                    "sample document": {
                        "sample identifier": "",
                        "written name": "broken",
                    },
                    "measurement document": {
                        "chromatogram data cube": {
                            "cube-structure": {
                                "dimensions": [
                                    {"unit": "s"},
                                ]
                            },
                            "data": {
                                "measures": [[1.0, 2.0]],
                                "dimensions": [[1.0, 2.0]],
                            },
                        }
                    },
                }
            ]
        }
    }

    measurement = asm_lc_1._map_lc_measurement(content, 0.0, "broken.json")
    assert measurement.id == "broken"
    assert len(measurement.chromatograms) == 1
    assert measurement.chromatograms[0].peaks == []
