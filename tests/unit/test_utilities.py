"""Tests for wombat/utilities/utilities.py, except setup_logger()."""

import datetime

import numpy as np
import pytest
import numpy.testing as nptest

from wombat.utilities.time import convert_dt_to_hours, hours_until_future_hour
from wombat.utilities.logging import format_events_log_message
from wombat.utilities.utilities import IEC_power_curve, _mean


def test_convert_dt_to_hours():
    """Tests `convert_dt_to_hours`."""
    # Test for a clean number of hours
    correct_hours = 10.0
    diff = convert_dt_to_hours(datetime.timedelta(hours=10))
    assert diff == correct_hours

    # Test for a days, hours, and minutes
    correct_hours = 36.5
    diff = convert_dt_to_hours(datetime.timedelta(days=1, hours=12, minutes=30))
    assert diff == correct_hours

    # Test for seconds
    correct_hours = 0.01
    diff = convert_dt_to_hours(datetime.timedelta(seconds=36))
    assert diff == correct_hours


def test_hours_until_future_hour():
    """Tests `hours_until_future_hour`."""
    # Test for a time later in the day, with an exact number of hours
    correct_hours = 10.0
    date = datetime.datetime(2021, 12, 20, 8, 0, 0)
    diff = hours_until_future_hour(date, 18)
    assert diff == correct_hours

    # Test for a time later in the day, with a decimal number of hours
    correct_hours = 10.26
    date = datetime.datetime(2021, 12, 20, 7, 44, 24)
    diff = hours_until_future_hour(date, 18)
    assert diff == correct_hours

    # Test for a time in the next day
    correct_hours = 23.26
    date = datetime.datetime(2021, 12, 20, 7, 44, 24)
    diff = hours_until_future_hour(date, 7)
    assert pytest.approx(diff) == correct_hours  # floating point error at ~16th digit

    # Test for next day, but using hours as 24 + 7 (same result as above)
    correct_hours = 23.26
    date = datetime.datetime(2021, 12, 20, 7, 44, 24)
    diff = hours_until_future_hour(date, 31)
    assert pytest.approx(diff) == correct_hours  # floating point error at ~16th digit

    # Test for multiple days
    correct_hours = 47.26
    date = datetime.datetime(2021, 12, 20, 7, 44, 24)
    diff = hours_until_future_hour(date, 55)
    assert pytest.approx(diff) == correct_hours  # floating point error at ~16th digit


def test_mean():
    """Tests `_mean`."""
    correct_mean = 2.0
    args = [1, 2, 3, 1, 2, 3]

    # Test for individual inputs
    assert _mean(1, 2, 3, 1, 2, 3) == correct_mean

    # Test for unpacking
    assert _mean(*args) == correct_mean


def test_format_events_log_message():
    """Tests `format_events_log_message`."""
    # Test that any floating point value gets updated for to be printed as a float,
    # which is a standard 6 decimal digit value printing.
    message = format_events_log_message(
        simulation_time=datetime.datetime(2021, 12, 20, 9, 45, 0),
        env_time=9.75,
        system_id="unit_tests",
        system_name="unit testing",
        part_id="utilities_tests",
        part_name="utilities testing",
        system_ol=1.0,
        part_ol=1,  # integer will get printed to float
        agent="pytest",
        action="test for errors",
        reason="software quality",
        additional="testing the software for correctness",
        duration=1,  # integer will get printed to float
        request_id="M00004",
        materials_cost=0,  # integer will get printed to float
        hourly_labor_cost=0.0,
        salary_labor_cost=140.00000009,  # 7 decimals wil, so will get clipped 6
        equipment_cost=0,  # integer will get printed to float
    )
    correct_message = (
        "2021-12-20 09:45:00|9.750000|unit_tests|unit testing"
        "|utilities_tests|utilities testing|1.000000|1.000000|pytest"
        "|test for errors|software quality"
        "|testing the software for correctness|1.000000|M00004|na"
        "|0.000000|0.000000|140.000000|0.000000|140.000000|140.000000"
    )
    assert message == correct_message


def test_IEC_power_curve():
    """Tests `IEC_power_curve`."""
    # tests are built from the same as OpenOA
    # power curve source: https://github.com/NREL/turbine-models/blob/master/Offshore/2020ATB_NREL_Reference_15MW_240.csv
    nrel_15mw_wind = np.arange(4, 26)
    nrel_15mw_power = np.array(
        [
            720,
            1239,
            2271,
            3817,
            5876,
            8450,
            11536,
            15000,
            15000,
            15000,
            15000,
            15000,
            15000,
            15000,
            15000,
            15000,
            15000,
            15000,
            15000,
            15000,
            15000,
            1500,
        ]
    )

    cut_in = 4
    cut_out = 25
    curve = IEC_power_curve(
        nrel_15mw_wind,
        nrel_15mw_power,
        windspeed_start=cut_in,
        windspeed_end=cut_out,
        bin_width=1,
    )

    # Create the test data
    test_windspeeds = np.arange(0, 31)
    test_power = curve(test_windspeeds)

    # Test all windspeeds outside of cut-in and cut-out windspeeds produce no power
    should_be_zeros = test_power[
        (test_windspeeds < cut_in) | (test_windspeeds > cut_out)
    ]
    nptest.assert_array_equal(should_be_zeros, np.zeros(should_be_zeros.shape))

    # Test all the valid windspeeds are equal
    valid_power = test_power[(test_windspeeds >= cut_in) & (test_windspeeds <= cut_out)]
    nptest.assert_array_equal(nrel_15mw_power, valid_power)
