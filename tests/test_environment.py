"""Tests wombat/core/environment.py."""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import numpy.testing as npt
from polars.testing import assert_series_equal

from wombat.core import RepairManager, WombatEnvironment
from wombat.windfarm import Windfarm

from tests.conftest import TEST_DATA


def test_data_dir():
    """Tests the data_dir creation."""
    relative_library = Path(__file__).resolve().parent / "library"
    assert TEST_DATA == relative_library


def test_setup():
    """Tests the setup of the `WombatEnvironment` and logging infrastructure."""
    # Test an invalid directory
    with pytest.raises(FileNotFoundError):
        WombatEnvironment(
            data_dir="TEST_DATA",
            weather_file="test_weather_quick_load.csv",
            workday_start=8,
            workday_end=18,
            random_seed=2022,
        )

    # Test invalid starting hour - too low
    with pytest.raises(ValueError):
        WombatEnvironment(
            data_dir=TEST_DATA,
            weather_file="test_weather_quick_load.csv",
            workday_start=-1,
            workday_end=18,
            random_seed=2022,
        )

    # Test invalid starting hour - too high
    with pytest.raises(ValueError):
        WombatEnvironment(
            data_dir=TEST_DATA,
            weather_file="test_weather_quick_load.csv",
            workday_start=24,
            workday_end=18,
            random_seed=2022,
        )

    # Test invalid ending hour - too low
    with pytest.raises(ValueError):
        WombatEnvironment(
            data_dir=TEST_DATA,
            weather_file="test_weather_quick_load.csv",
            workday_start=8,
            workday_end=-1,
            random_seed=2022,
        )

    # Test invalid ending hour - before start
    with pytest.raises(ValueError):
        WombatEnvironment(
            data_dir=TEST_DATA,
            weather_file="test_weather_quick_load.csv",
            workday_start=8,
            workday_end=3,
            random_seed=2022,
        )

    # Test invalid ending hour - at the same time as start
    with pytest.raises(ValueError):
        WombatEnvironment(
            data_dir=TEST_DATA,
            weather_file="test_weather_quick_load.csv",
            workday_start=8,
            workday_end=8,
            random_seed=2022,
        )

    # Test invalid start_year - later than file limits (2002 through 2003)
    with pytest.raises(ValueError):
        WombatEnvironment(
            data_dir=TEST_DATA,
            weather_file="test_weather.csv",
            workday_start=8,
            workday_end=16,
            simulation_name="testing_setup",
            start_year=2004,
            end_year=2003,
            random_seed=2022,
        )

    # Test invalid end_year - start year is after end year
    with pytest.raises(ValueError):
        WombatEnvironment(
            data_dir=TEST_DATA,
            weather_file="test_weather.csv",
            workday_start=8,
            workday_end=16,
            simulation_name="testing_setup",
            start_year=2003,
            end_year=2002,
            random_seed=2022,
        )

    # Test the data are read correctly with default starting and ending year
    # and the logging infrastructure
    env = WombatEnvironment(
        data_dir=TEST_DATA,
        weather_file="test_weather.csv",
        workday_start=8,
        workday_end=16,
        simulation_name="testing_setup",
        random_seed=2022,
    )
    assert env.workday_start == 8
    assert env.workday_end == 16
    assert env.shift_length == 8
    assert isinstance(env.weather, pl.DataFrame)
    assert env.start_datetime == datetime.datetime(2002, 1, 1, 0, 0)
    assert env.end_datetime == datetime.datetime(2003, 12, 31, 23, 0)
    assert env.max_run_time == 8760 * 2  # 2 year data file

    assert Path(env.events_log_fname).is_file()
    assert Path(env.operations_log_fname).is_file()

    # Delete the logged files, and check they not longer exist
    env.cleanup_log_files()
    assert not Path(env.events_log_fname).is_file()
    assert not Path(env.operations_log_fname).is_file()

    # Test custom start year
    env = WombatEnvironment(
        data_dir=TEST_DATA,
        weather_file="test_weather.csv",
        workday_start=8,
        workday_end=16,
        simulation_name="testing_setup",
        start_year=2003,
        random_seed=2022,
    )
    assert env.start_datetime == datetime.datetime(2003, 1, 1, 0, 0)
    assert env.end_datetime == datetime.datetime(2003, 12, 31, 23, 0)
    env.cleanup_log_files()  # delete the logged data

    # Test custom end year
    env = WombatEnvironment(
        data_dir=TEST_DATA,
        weather_file="test_weather.csv",
        workday_start=8,
        workday_end=16,
        simulation_name="testing_setup",
        end_year=2002,
        random_seed=2022,
    )
    assert env.start_datetime == datetime.datetime(2002, 1, 1, 0, 0)
    assert env.end_datetime == datetime.datetime(2002, 12, 31, 23, 0)
    env.cleanup_log_files()  # delete the logged data

    # Test custom start and end year
    env = WombatEnvironment(
        data_dir=TEST_DATA,
        weather_file="test_weather.csv",
        workday_start=8,
        workday_end=16,
        simulation_name="testing_setup",
        start_year=2003,
        end_year=2003,
        random_seed=2022,
    )
    assert env.start_datetime == datetime.datetime(2003, 1, 1, 0, 0)
    assert env.end_datetime == datetime.datetime(2003, 12, 31, 23, 0)
    env.cleanup_log_files()  # delete the logged data


def test_timing():
    """Test basic movements and time coordination for `WombatEnvironment`.

    Methods that are completely tested:
     - ``date_ix``
     - ``simulation_time``
     - ``weather_now``

    Methods that are partially tested for no inputs:
     - ``hours_to_next_shift``
     - ``is_workshift``

    """
    # Setup a basic environment
    env = WombatEnvironment(
        data_dir=TEST_DATA,
        weather_file="test_weather_quick_load.csv",
        workday_start=8,
        workday_end=16,
        simulation_name="testing_setup",
        random_seed=2022,
    )
    manager = RepairManager(env)
    Windfarm(env, "layout.csv", manager)

    # Run for 96 hours from the start to this point in the data
    # 1/5/02 0:00,8.238614098,0.37854216
    correct_index = 96
    correct_date = datetime.date(2002, 1, 5)
    correct_hour = 0
    correct_datetime = datetime.datetime(2002, 1, 5, 0, 0, 0)
    correct_wind = 8.238614098
    correct_wave = 0.37854216

    assert env.date_ix(correct_date) == correct_index
    assert env.date_ix(correct_datetime) == correct_index

    env.run(until=correct_index)
    assert env.now == correct_index
    assert env.simulation_time == correct_datetime
    assert env.hours_to_next_shift() == 8
    current_conditions = env.weather_now.to_numpy().flatten()[2:]
    assert all(current_conditions == (correct_wind, correct_wave, correct_hour))
    assert not env.is_workshift()

    # Test for a non-even timing of 128.7 hours
    # 1/6/02 8:00,8.321216829,0.795805105
    until = 128.7
    correct_index = 120
    correct_date = datetime.date(2002, 1, 6)
    correct_datetime = datetime.datetime(2002, 1, 6, 8, 42)
    correct_hours_to_next_shift = 23.3
    correct_hour = 8
    correct_wind = 8.321216829
    correct_wave = 0.795805105
    assert env.date_ix(correct_date) == correct_index
    assert env.date_ix(correct_datetime) == correct_index

    env.run(until=until)
    assert env.now == until
    assert env.simulation_time == correct_datetime
    assert env.hours_to_next_shift() == correct_hours_to_next_shift
    current_conditions = env.weather_now.to_numpy().flatten()[2:]
    assert all(current_conditions == (correct_wind, correct_wave, correct_hour))
    assert env.is_workshift()

    env.cleanup_log_files()  # delete the logged data


def test_is_workshift():
    """Tests the ``is_workshift`` method for a variety of inputs."""
    # Setup a basic environment
    env = WombatEnvironment(
        data_dir=TEST_DATA,
        weather_file="test_weather_quick_load.csv",
        workday_start=8,
        workday_end=16,
        simulation_name="testing_setup",
        random_seed=2022,
    )
    manager = RepairManager(env)
    Windfarm(env, "layout.csv", manager)
    # Test for the 24 hours of the day with a basic 8am to 4pm workday
    assert not env.is_workshift()
    for i in range(23):
        env.run(i + 1)
        if 7 <= i < 15:
            # THe hours from 8am until, but not at, 4pm are working hours
            assert env.is_workshift()
        else:
            assert not env.is_workshift()

        # Test the boundary conditions

        # Start and current are the same time, but end is midnight
        assert env.is_workshift(workday_start=i + 1, workday_end=24)

        # Start is just before current, but the end is midnight
        assert env.is_workshift(workday_start=i, workday_end=24)

        # Start is just after, but the end is midnight
        assert not env.is_workshift(workday_start=i + 2, workday_end=24)

        # Start is before current, but end is the current hour
        assert not env.is_workshift(workday_start=i, workday_end=i + 1)

    env.cleanup_log_files()  # delete the logged data


def test_hour_in_shift():
    """Tests the ``hour_in_shift`` method for a variety of inputs."""
    # Setup a basic environment
    env = WombatEnvironment(
        data_dir=TEST_DATA,
        weather_file="test_weather_quick_load.csv",
        workday_start=8,
        workday_end=16,
        simulation_name="testing_setup",
        random_seed=2022,
    )
    manager = RepairManager(env)
    Windfarm(env, "layout.csv", manager)

    # Test the boundary conditions at midnight

    i = 0
    # Start and current are the same time, but end is midnight
    assert env.hour_in_shift(hour=i, workday_start=i, workday_end=24)

    # Start is just before current (-1), so reverts to env workday settings
    assert not env.hour_in_shift(hour=i, workday_start=-1, workday_end=24)

    # Start is just after, but the end is midnight
    assert not env.hour_in_shift(hour=i, workday_start=i + 1, workday_end=24)

    # Start is before current, but end is the current hour
    assert not env.hour_in_shift(hour=i, workday_start=i, workday_end=i)

    # Test for the 24 hours of the day, post midnight
    for i in range(1, 24):
        # Start and current are the same time, but end is midnight
        assert env.hour_in_shift(hour=i, workday_start=i, workday_end=24)

        # Start is just before current, but the end is midnight
        assert env.hour_in_shift(hour=i, workday_start=i - 1, workday_end=24)

        # Start is just after, but the end is midnight
        assert not env.hour_in_shift(hour=i, workday_start=i + 1, workday_end=24)

        # Start is before current, but end is the current hour
        assert not env.hour_in_shift(hour=i, workday_start=i, workday_end=i)

    env.cleanup_log_files()  # delete the logged data


def test_hours_to_next_shift():
    """Tests the ``hours_to_next_shift`` method for a variety of inputs."""
    # Setup a basic environment
    env = WombatEnvironment(
        data_dir=TEST_DATA,
        weather_file="test_weather_quick_load.csv",
        workday_start=8,
        workday_end=16,
        simulation_name="testing_setup",
        random_seed=2022,
    )
    manager = RepairManager(env)
    Windfarm(env, "layout.csv", manager)

    # Test that at hour 0, there are 8 hours until 8am
    assert env.hours_to_next_shift() == 8

    # Test that at hour 0, with start time 0, there are 24 hours until the next period
    assert env.hours_to_next_shift(workday_start=0)

    env.run(until=7)

    # Test that there is still 1 hour left
    assert env.hours_to_next_shift() == 1

    # Test if the starting shift is at the same time, the we get 24 hours
    assert env.hours_to_next_shift(workday_start=7) == 24

    # Test that it's 24 hours left
    env.run(until=8)
    assert env.hours_to_next_shift() == 24

    # Test for an uneven time until the next shift
    env.run(until=9.3)
    assert env.hours_to_next_shift() == pytest.approx(22.7, rel=1e-4)  # weird floats

    env.cleanup_log_files()  # delete the logged data


def test_weather_forecast():
    """Tests the ``weather_forecast`` method for a variety of inputs."""
    # Setup a basic environment
    env = WombatEnvironment(
        data_dir=TEST_DATA,
        weather_file="test_weather_quick_load.csv",
        workday_start=8,
        workday_end=16,
        simulation_name="testing_setup",
        random_seed=2022,
    )
    manager = RepairManager(env)
    Windfarm(env, "layout.csv", manager)

    # Test for the next 5 hours, but at the top of the hour
    # Starts with hour 0, and ends with the 5th hour following
    correct_index = pl.from_pandas(
        pd.date_range(
            "1/1/2002 00:00:00", "1/1/2002 04:00:00", freq="1h", name="datetime"
        )
    )
    correct_hour = np.array([0, 1, 2, 3, 4], dtype=float)
    correct_wind = np.array(
        [11.75561096, 10.41321252, 8.959270788, 9.10014808, 9.945059601]
    )
    correct_wave = np.array(
        [1.281772405, 1.586584315, 1.725690828, 1.680982063, 1.552232138]
    )

    ix, hour, wind, wave = env.weather_forecast(hours=5)
    assert ix.shape[0] == wind.shape[0] == wave.shape[0] == 5
    assert_series_equal(ix, correct_index)
    npt.assert_equal(hour, correct_hour)
    npt.assert_equal(wind, correct_wind)
    npt.assert_equal(wave, correct_wave)

    # Test for 5 hours at an uneven start time increment
    # Starts at hour 1, and ends with the 5th hour following
    env.run(until=0.1)
    correct_index = pl.from_pandas(
        pd.date_range(
            "1/1/2002 00:00:00", "1/1/2002 05:00:00", freq="1h", name="datetime"
        )
    )
    correct_hour = np.arange(6, dtype=float)
    correct_wind = np.array(
        [11.75561096, 10.41321252, 8.959270788, 9.10014808, 9.945059601, 11.50807971]
    )
    correct_wave = np.array(
        [1.281772405, 1.586584315, 1.725690828, 1.680982063, 1.552232138, 1.335083363]
    )

    ix, hour, wind, wave = env.weather_forecast(hours=5)
    assert ix.shape[0] == hour.shape[0] == wind.shape[0] == wave.shape[0] == 6
    assert_series_equal(ix, correct_index)
    npt.assert_equal(hour, correct_hour)
    npt.assert_equal(wind, correct_wind)
    npt.assert_equal(wave, correct_wave)

    # Test for 5.5 hours at an uneven start time increment
    # Starts at hour 1, and ends at the 7th hour following
    env.run(until=1.2)
    correct_index = pl.from_pandas(
        pd.date_range(
            "1/1/2002 01:00:00", "1/1/2002 07:00:00", freq="1h", name="datetime"
        )
    )
    correct_hour = np.arange(1, 8, dtype=float)
    correct_wind = np.array(
        [
            10.41321252,
            8.959270788,
            9.10014808,
            9.945059601,
            11.50807971,
            12.44757098,
            13.33747206,
        ]
    )
    correct_wave = np.array(
        [
            1.586584315,
            1.725690828,
            1.680982063,
            1.552232138,
            1.335083363,
            1.294392727,
            1.187261471,
        ]
    )

    ix, hour, wind, wave = env.weather_forecast(hours=5.5)
    assert ix.shape[0] == hour.shape[0] == wind.shape[0] == wave.shape[0] == 7
    assert_series_equal(ix, correct_index)
    npt.assert_equal(hour, correct_hour)
    npt.assert_equal(wind, correct_wind)
    npt.assert_equal(wave, correct_wave)

    env.cleanup_log_files()  # delete the logged data
