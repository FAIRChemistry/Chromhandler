"""Every test under tests/test_readers/ belongs to the `readers` marker group."""

from pathlib import Path

import pytest

_READERS_DIR = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if _READERS_DIR in item.path.parents:
            item.add_marker(pytest.mark.readers)
