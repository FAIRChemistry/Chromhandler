"""`Handler.visualize()` against a real fixture.

The reason this file exists: `visualize()` called `plt.cm.get_cmap`, which
matplotlib removed in 3.9, so every call raised AttributeError on any modern
install. Nothing caught it, because nothing called `visualize()` outside the
docs notebook — and the notebook was not passing either. This test is the guard.
"""

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from chromhandler import Handler

ASM_DIR = Path(__file__).parent.parent / "docs" / "usage" / "data" / "asm"


@pytest.fixture
def handler() -> Handler:
    return Handler.read_asm(
        path=str(ASM_DIR), ph=7.4, temperature=25.0, mode="timecourse", silent=True
    )


def test_visualize_runs(handler: Handler) -> None:
    handler.visualize()
    plt.close("all")


def test_visualize_overlay_runs(handler: Handler) -> None:
    handler.visualize(overlay=True)
    plt.close("all")
