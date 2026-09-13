"""`values=[...]` must follow the files in numeric file-name order, not text order.

Sorted as text, ``s_10.38min`` comes before ``s_2.88min``; the values are zipped onto the
files positionally, so the time course was silently scrambled.
"""

import shutil
from pathlib import Path

from chromhandler.handler import Handler

SHIMADZU_DIR = Path("docs/usage/data/shimadzu")
VALUES = [2.88, 10.38]  # numeric order, which is NOT text order


def _value_in_name(measurement_id: str) -> float:
    return float(measurement_id.split("_")[1].removesuffix("min"))


def test_shimadzu_values_follow_numeric_file_order(tmp_path: Path) -> None:
    for value, src in zip(VALUES, ["Output-sample 7.txt", "Output-sample 8.txt"]):
        shutil.copy(SHIMADZU_DIR / src, tmp_path / f"s_{value}min.txt")

    handler = Handler.read_shimadzu(
        path=tmp_path,
        values=VALUES,
        unit="min",
        ph=7.4,
        temperature=25.0,
        mode="timecourse",
        silent=True,
    )

    assert [m.data.value for m in handler.measurements] == VALUES
    for meas in handler.measurements:
        assert meas.data.value == _value_in_name(meas.id)


def test_csv_values_follow_numeric_file_order(tmp_path: Path) -> None:
    for value in VALUES:
        (tmp_path / f"s_{value}min.csv").write_text(f"rt,area\n1.0,{value * 100}\n")

    handler = Handler.read_csv(
        path=tmp_path,
        retention_time_col_name="rt",
        peak_area_col_name="area",
        header=0,
        temperature_unit="Celsius",
        values=VALUES,
        unit="min",
        ph=7.4,
        temperature=25.0,
        mode="timecourse",
        silent=True,
    )

    assert [m.data.value for m in handler.measurements] == VALUES
    for meas in handler.measurements:
        assert meas.chromatograms[0].peaks[0].area == meas.data.value * 100
