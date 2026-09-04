"""Every test under tests/test_readers/ belongs to the `readers` marker group."""

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        item.add_marker(pytest.mark.readers)
