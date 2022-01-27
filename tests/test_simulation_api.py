"""Test the API functionality."""

import pytest


@pytest.mark.cat("all", "subassembly", "cable", "service_equipment")
def test_pass():
    pass


@pytest.mark.cat("all")
def test_things():
    pass
