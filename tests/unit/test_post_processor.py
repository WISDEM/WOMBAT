"""Test the post-processing functionality."""

import pytest

from wombat.core import Frequency
from wombat.core.post_processor import _check_frequency


def test_check_frequency():
    """Test `_check_frequency`."""
    assert _check_frequency("project").value == "project"
    assert _check_frequency("annual").value == "annual"
    assert _check_frequency("monthly").value == "monthly"
    assert _check_frequency("month year").value == "month_year"

    frequency = _check_frequency("project")
    assert isinstance(frequency, Frequency)
    assert frequency is Frequency.PROJECT

    with pytest.raises(ValueError):
        _check_frequency("annual", which="project")

    with pytest.raises(ValueError):
        _check_frequency("monthly", which="project")

    with pytest.raises(ValueError):
        _check_frequency("month-year", which="project")
