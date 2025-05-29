"""Tests wombat/core/environment.py."""

from __future__ import annotations

import datetime

from wombat.core import RepairManager, WombatEnvironment
from wombat.windfarm import Windfarm

from tests.conftest import TEST_DATA


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
        maintenance_start="01/06/2002",
    )
    manager = RepairManager(env)
    wf = Windfarm(env, "layout.csv", manager)

    # Ensure maintenance start has passed through to the generator maintenance task
    for tid in wf.turbine_id:
        turb = wf.system(tid)
        gen, *_ = (el for el in turb.subassemblies if el.data.name == "generator")
        assert gen.data.maintenance[0].start_date == env.maintenance_start

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
