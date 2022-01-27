"""Test the ServiceEquipment class."""

from pprint import pprint

import numpy as np
import pytest
import numpy.testing as npt
from sqlalchemy import false

from wombat.windfarm import Windfarm
from wombat.core.library import load_yaml
from wombat.core.environment import WombatEnvironment
from wombat.core.data_classes import Maintenance
from wombat.core.repair_management import StrategyMap, EquipmentMap, RepairManager
from wombat.core.service_equipment import ServiceEquipment, consecutive_groups

from tests.conftest import TEST_DATA, env_setup, env_setup_full_profile


@pytest.mark.cat("all", "subassembly", "cable", "service_equipment")
def test_pass():
    pass


@pytest.mark.cat("all", "service_equipment")
def test_consecutive_groups():
    """Tests the ``consecutive_groups`` function in a similar way that this would be used."""
    all_clear = np.array(
        [True, True, False, True, True, True, False, True, True, True, True, False]
    )
    windows = consecutive_groups(np.where(all_clear)[0])
    correct_windows = [np.array([0, 1]), np.array([3, 4, 5]), np.array([7, 8, 9, 10])]
    print(windows)
    assert len(windows) == len(correct_windows)
    for window, correct_window in zip(windows, correct_windows):
        npt.assert_equal(window, correct_window)

    # Flip the weather window and test that we only have 1 hour openings
    all_clear = ~all_clear
    windows = consecutive_groups(np.where(all_clear)[0])
    correct_windows = [np.array([2]), np.array([6]), np.array([11])]
    print(windows)
    assert len(windows) == len(correct_windows)
    for window, correct_window in zip(windows, correct_windows):
        npt.assert_equal(window, correct_window)


@pytest.mark.cat("all", "service_equipment")
def test_service_equipment_init(env_setup):
    env = env_setup
    manager = RepairManager(env)
    windfarm = Windfarm(env, "layout.csv", manager)

    # TEST 1
    # Check that a basic CTV initialization looks good

    ctv = ServiceEquipment(env, windfarm, manager, "ctv_quick_load.yaml")
    ctv_dict = load_yaml(env.data_dir / "repair" / "transport", "ctv_quick_load.yaml")

    # Check the basic attribute assignments
    assert ctv.env == env
    assert ctv.windfarm == windfarm
    assert ctv.manager == manager
    assert not ctv.onsite
    assert not ctv.enroute
    assert not ctv.at_system
    assert not ctv.at_port
    assert ctv.current_system is None

    # Check that the starting and ending year is the same as the input file
    assert ctv.settings.start_year == ctv_dict["start_year"]
    assert ctv.settings.end_year == ctv_dict["end_year"]

    # Check that the default inputs are set to the environment settings
    assert ctv.settings.workday_start == env.workday_start
    assert ctv.settings.workday_end == env.workday_end

    # TEST 2
    # Check that a CTV with invalid start years gets corrected
    ctv = ServiceEquipment(env, windfarm, manager, "ctv_invalid_years.yaml")

    # Check the basic attribute assignments
    assert ctv.env == env
    assert ctv.windfarm == windfarm
    assert ctv.manager == manager
    assert not ctv.onsite
    assert not ctv.enroute
    assert not ctv.at_system
    assert not ctv.at_port
    assert ctv.current_system is None

    # Check that the starting and ending year match the environment limits
    assert ctv.settings.start_year == env.start_year
    assert ctv.settings.end_year == env.end_year

    # Check that the default inputs are set to the environment settings
    assert ctv.settings.workday_start == env.workday_start
    assert ctv.settings.workday_end == env.workday_end

    # TEST 3
    # Check the initializtion for an unscheduled vessel for a downtime-basis
    hlv = ServiceEquipment(env, windfarm, manager, "hlv_downtime.yaml")
    hlv_dict = load_yaml(env.data_dir / "repair" / "transport", "hlv_downtime.yaml")

    # Check the basic attribute assignments
    assert hlv.env == env
    assert hlv.windfarm == windfarm
    assert hlv.manager == manager
    assert not hlv.onsite
    assert not hlv.enroute
    assert not hlv.at_system
    assert not hlv.at_port
    assert hlv.current_system is None

    # Check that the hours are the custom set hours
    assert hlv.settings.workday_start == hlv_dict["workday_start"]
    assert hlv.settings.workday_end == hlv_dict["workday_end"]

    # Check that the vessel is registered correctly in the repair manager
    assert manager.downtime_based_equipment.is_running
    assert not manager.request_based_equipment.is_running  # No vessels to run or check
    assert manager.downtime_based_equipment.LCN == [
        EquipmentMap(hlv_dict["strategy_threshold"], hlv)
    ]
    for capability in ("CTV", "SCN", "CAB", "RMT", "DRN", "DSV"):
        assert getattr(manager.downtime_based_equipment, capability) == []

    # TEST 4
    # Check the initializtion for an unscheduled vessel for a request-basis
    hlv = ServiceEquipment(env, windfarm, manager, "hlv_requests.yaml")
    hlv_dict = load_yaml(env.data_dir / "repair" / "transport", "hlv_requests.yaml")

    # Check the basic attribute assignments
    assert hlv.env == env
    assert hlv.windfarm == windfarm
    assert hlv.manager == manager
    assert not hlv.onsite
    assert not hlv.enroute
    assert not hlv.at_system
    assert not hlv.at_port
    assert hlv.current_system is None

    # Check that the hours are the custom set hours
    assert hlv.settings.workday_start == hlv_dict["workday_start"]
    assert hlv.settings.workday_end == hlv_dict["workday_end"]

    # Check that the vessel is registered correctly in the repair manager
    assert (
        manager.downtime_based_equipment.is_running
    )  # Same manager, just another HLV added with a different strategy
    assert manager.request_based_equipment.is_running
    assert manager.request_based_equipment.LCN == [
        EquipmentMap(hlv_dict["strategy_threshold"], hlv)
    ]
    for capability in ("CTV", "SCN", "CAB", "RMT", "DRN", "DSV"):
        assert getattr(manager.request_based_equipment, capability) == []


@pytest.mark.cat("all", "service_equipment")
def test_calculate_salary_cost(env_setup):
    """Tests the ``calculate_salary_cost`` method."""
    env = env_setup
    manager = RepairManager(env)
    windfarm = Windfarm(env, "layout.csv", manager)
    ctv = ServiceEquipment(env, windfarm, manager, "ctv_wages.yaml")
    ctv_dict = load_yaml(env.data_dir / "repair" / "transport", "ctv_wages.yaml")
    ctv_crew = ctv_dict["crew"]

    # Test a small, even number
    n_hours = 4
    cost = ctv.calculate_salary_cost(n_hours)
    assert cost == n_hours / 24 * ctv_crew["day_rate"] * ctv_crew["n_day_rate"]

    # Test a multi-day number
    n_hours = 74.333
    cost = ctv.calculate_salary_cost(n_hours)
    assert cost == n_hours / 24 * ctv_crew["day_rate"] * ctv_crew["n_day_rate"]


@pytest.mark.cat("all", "service_equipment")
def test_calculate_hourly_cost(env_setup):
    """Tests the ``calculate_hourly_cost`` method."""
    env = env_setup
    manager = RepairManager(env)
    windfarm = Windfarm(env, "layout.csv", manager)
    ctv = ServiceEquipment(env, windfarm, manager, "ctv_wages.yaml")
    ctv_dict = load_yaml(env.data_dir / "repair" / "transport", "ctv_wages.yaml")
    ctv_crew = ctv_dict["crew"]

    n_hours = 4
    cost = ctv.calculate_hourly_cost(n_hours)
    assert cost == n_hours * ctv_crew["hourly_rate"] * ctv_crew["n_hourly_rate"]

    n_hours = 74.333
    cost = ctv.calculate_hourly_cost(n_hours)
    assert cost == n_hours * ctv_crew["hourly_rate"] * ctv_crew["n_hourly_rate"]


@pytest.mark.cat("all", "service_equipment")
def test_calculate_equipment_cost(env_setup):
    """Tests the ``calculate_equipment_cost`` method."""
    env = env_setup
    manager = RepairManager(env)
    windfarm = Windfarm(env, "layout.csv", manager)
    ctv = ServiceEquipment(env, windfarm, manager, "ctv.yaml")
    ctv_dict = load_yaml(env.data_dir / "repair" / "transport", "ctv.yaml")

    n_hours = 4
    cost = ctv.calculate_equipment_cost(n_hours)
    assert cost == n_hours / 24 * ctv_dict["equipment_rate"]

    n_hours = 74.333
    cost = ctv.calculate_equipment_cost(n_hours)
    assert cost == n_hours / 24 * ctv_dict["equipment_rate"]


@pytest.mark.cat("all", "service_equipment")
def test_onsite_scheduled_equipment_logic(env_setup_full_profile):
    """ "Test the simulation logic of a scheduled CTV."""
    env = env_setup_full_profile
    manager = RepairManager(env)
    windfarm = Windfarm(env, "layout_simulation.csv", manager)

    ctv = ServiceEquipment(env, windfarm, manager, "ctv.yaml")
    fsv = ServiceEquipment(env, windfarm, manager, "fsv_scheduled.yaml")
    hlv = ServiceEquipment(env, windfarm, manager, "hlv_scheduled.yaml")

    # Start the simulation to ensure everything is in place as required
    env.run(1)
    assert (
        ctv.at_port
        is ctv.at_system
        is ctv.transferring_crew
        is ctv.enroute
        is (not ctv.onsite)
        is False
    )
    assert (
        fsv.at_port
        is fsv.at_system
        is fsv.transferring_crew
        is fsv.enroute
        is fsv.onsite
        is False
    )
    assert (
        hlv.at_port
        is hlv.at_system
        is hlv.transferring_crew
        is hlv.enroute
        is hlv.onsite
        is False
    )

    # Fast-forward to the start of the environment workshift to ensure all are
    # non-operational (CTV has no tasks, nor mobilization to check)
    env.run(7)
    assert (
        ctv.at_port
        is ctv.at_system
        is ctv.transferring_crew
        is ctv.enroute
        is (not ctv.onsite)
        is False
    )
    assert (
        fsv.at_port
        is fsv.at_system
        is fsv.transferring_crew
        is fsv.enroute
        is fsv.onsite
        is False
    )
    assert (
        hlv.at_port
        is hlv.at_system
        is hlv.transferring_crew
        is hlv.enroute
        is hlv.onsite
        is False
    )

    # Check that at Day 15 there are 6 turbine generator maintenance tasks requested
    timeout = 15 * 24 + 1
    env.run(timeout)
    assert len(manager.items) == 6
    assert all(isinstance(el.details, Maintenance) for el in manager.items)
    assert all(el.details.description == "ctv call" for el in manager.items)

    # The first maintenance task is at day 15 for the CTV, fast forward to this point in
    # time to check for operations to be running and inspect every step of the task
    timeout += 6 + 1 / 60  # Shift start is 7AM, but it's already 1AM
    env.run(timeout + 0.01)
    assert ctv.onsite is ctv.at_system is ctv.transferring_crew is True
    assert not ctv.enroute
    assert ctv.current_system == "S00T1"
    assert (
        fsv.at_port
        is fsv.at_system
        is fsv.onsite
        is fsv.enroute
        is fsv.transferring_crew
        is False
    )
    assert (
        hlv.at_port
        is hlv.at_system
        is hlv.onsite
        is hlv.enroute
        is hlv.transferring_crew
        is False
    )

    # Move to just before the 15 minute transfer is complete to ensure it hasn't ended early
    timeout += 13 / 60
    env.run(timeout)
    assert ctv.transferring_crew is ctv.onsite is ctv.at_system is True
    assert ctv.current_system == "S00T1"
    assert not ctv.enroute

    # Move to the end of the 15 minute to ensure the transfer is complete
    timeout += 2 / 60
    env.run(timeout)
    assert ctv.onsite is ctv.at_system is True
    assert ctv.current_system == "S00T1"
    assert ctv.enroute is ctv.transferring_crew is False

    # The weather window is clear on 1/16/2002, so there will not be weather delays, so
    # the CTV should be transferring the crew back onto the vessel
    timeout += 1
    env.run(timeout)
    assert ctv.onsite is ctv.at_system is ctv.transferring_crew is True
    assert not ctv.enroute
    assert ctv.current_system == "S00T1"
    assert (
        fsv.at_port
        is fsv.at_system
        is fsv.onsite
        is fsv.enroute
        is fsv.transferring_crew
        is False
    )
    assert (
        hlv.at_port
        is hlv.at_system
        is hlv.onsite
        is hlv.enroute
        is hlv.transferring_crew
        is False
    )

    # Check the transfer is still progressing
    timeout += 13 / 60
    env.run(timeout)
    assert ctv.transferring_crew is ctv.onsite is ctv.at_system is True
    assert ctv.current_system == "S00T1"
    assert not ctv.enroute
    assert len(manager.items) == 5

    ## TURBINE 2 GENERATOR CTV repair

    # Move to the end of the 15 minute to ensure the transfer is complete, and there is
    # no travel time between site objects, so the crew should be transferring again
    timeout += 2 / 60
    env.run(timeout)
    assert ctv.onsite is ctv.at_system is ctv.transferring_crew is True
    assert ctv.current_system == "S00T2"
    assert ctv.enroute is False

    # Move to just before the 15 minute transfer is complete to ensure it hasn't ended early
    timeout += 13 / 60
    env.run(timeout)
    assert ctv.transferring_crew is ctv.onsite is ctv.at_system is True
    assert ctv.current_system == "S00T2"
    assert not ctv.enroute

    # Move to the end of the 15 minute to ensure the transfer is complete
    timeout += 2 / 60
    env.run(timeout)
    assert ctv.onsite is ctv.at_system is True
    assert ctv.current_system == "S00T2"
    assert ctv.enroute is ctv.transferring_crew is False

    # The weather window is clear on 1/16/2002, so there will not be weather delays, so
    # the CTV should be transferring the crew back onto the vessel
    timeout += 1
    env.run(timeout)
    assert ctv.onsite is ctv.at_system is ctv.transferring_crew is True
    assert not ctv.enroute
    assert ctv.current_system == "S00T2"
    assert (
        fsv.at_port
        is fsv.at_system
        is fsv.onsite
        is fsv.enroute
        is fsv.transferring_crew
        is False
    )
    assert (
        hlv.at_port
        is hlv.at_system
        is hlv.onsite
        is hlv.enroute
        is hlv.transferring_crew
        is False
    )

    # Check the transfer is still progressing
    timeout += 13 / 60
    env.run(timeout)
    assert ctv.transferring_crew is ctv.onsite is ctv.at_system is True
    assert ctv.current_system == "S00T2"
    assert not ctv.enroute
    assert len(manager.items) == 4

    ## TURBINE 3 GENERATOR CTV repair

    # Move to the end of the 15 minute to ensure the transfer is complete, and there is
    # no travel time between site objects, so the crew should be transferring again
    timeout += 2 / 60
    env.run(timeout)
    assert ctv.onsite is ctv.at_system is ctv.transferring_crew is True
    assert ctv.current_system == "S00T3"
    assert ctv.enroute is False

    # Move to just before the 15 minute transfer is complete to ensure it hasn't ended early
    timeout += 13 / 60
    env.run(timeout)
    assert ctv.transferring_crew is ctv.onsite is ctv.at_system is True
    assert ctv.current_system == "S00T3"
    assert not ctv.enroute

    # Move to the end of the 15 minute to ensure the transfer is complete
    timeout += 2 / 60
    env.run(timeout)
    assert ctv.onsite is ctv.at_system is True
    assert ctv.current_system == "S00T3"
    assert ctv.enroute is ctv.transferring_crew is False

    # The weather window is clear on 1/16/2002, so there will not be weather delays, so
    # the CTV should be transferring the crew back onto the vessel
    timeout += 1
    env.run(timeout)
    assert ctv.onsite is ctv.at_system is ctv.transferring_crew is True
    assert not ctv.enroute
    assert ctv.current_system == "S00T3"
    assert (
        fsv.at_port
        is fsv.at_system
        is fsv.onsite
        is fsv.enroute
        is fsv.transferring_crew
        is False
    )
    assert (
        hlv.at_port
        is hlv.at_system
        is hlv.onsite
        is hlv.enroute
        is hlv.transferring_crew
        is False
    )

    # Check the transfer is still progressing
    timeout += 13 / 60
    env.run(timeout)
    assert ctv.transferring_crew is ctv.onsite is ctv.at_system is True
    assert ctv.current_system == "S00T3"
    assert not ctv.enroute
    assert len(manager.items) == 3

    ## TURBINE 4 GENERATOR CTV repair

    # Move to the end of the 15 minute to ensure the transfer is complete, and there is
    # no travel time between site objects, so the crew should be transferring again
    timeout += 2 / 60
    env.run(timeout)
    assert ctv.onsite is ctv.at_system is ctv.transferring_crew is True
    assert ctv.current_system == "S01T4"
    assert ctv.enroute is False

    # Move to just before the 15 minute transfer is complete to ensure it hasn't ended early
    timeout += 13 / 60
    env.run(timeout)
    assert ctv.transferring_crew is ctv.onsite is ctv.at_system is True
    assert ctv.current_system == "S01T4"
    assert not ctv.enroute

    # Move to the end of the 15 minute to ensure the transfer is complete
    timeout += 2 / 60
    env.run(timeout)
    assert ctv.onsite is ctv.at_system is True
    assert ctv.current_system == "S01T4"
    assert ctv.enroute is ctv.transferring_crew is False

    # The weather window is clear on 1/16/2002, so there will not be weather delays, so
    # the CTV should be transferring the crew back onto the vessel
    timeout += 1
    env.run(timeout)
    assert ctv.onsite is ctv.at_system is ctv.transferring_crew is True
    assert not ctv.enroute
    assert ctv.current_system == "S01T4"
    assert (
        fsv.at_port
        is fsv.at_system
        is fsv.onsite
        is fsv.enroute
        is fsv.transferring_crew
        is False
    )
    assert (
        hlv.at_port
        is hlv.at_system
        is hlv.onsite
        is hlv.enroute
        is hlv.transferring_crew
        is False
    )

    # Check the transfer is still progressing
    timeout += 13 / 60
    env.run(timeout)
    assert ctv.transferring_crew is ctv.onsite is ctv.at_system is True
    assert ctv.current_system == "S01T4"
    assert not ctv.enroute
    assert len(manager.items) == 2

    ## TURBINE 5 GENERATOR CTV repair

    # Move to the end of the 15 minute to ensure the transfer is complete, and there is
    # no travel time between site objects, so the crew should be transferring again
    timeout += 2 / 60
    env.run(timeout)
    assert ctv.onsite is ctv.at_system is ctv.transferring_crew is True
    assert ctv.current_system == "S01T5"
    assert ctv.enroute is False

    # Move to just before the 15 minute transfer is complete to ensure it hasn't ended early
    timeout += 13 / 60
    env.run(timeout)
    assert ctv.transferring_crew is ctv.onsite is ctv.at_system is True
    assert ctv.current_system == "S01T5"
    assert not ctv.enroute

    # Move to the end of the 15 minute to ensure the transfer is complete
    timeout += 2 / 60
    env.run(timeout)
    assert ctv.onsite is ctv.at_system is True
    assert ctv.current_system == "S01T5"
    assert ctv.enroute is ctv.transferring_crew is False

    # The weather window is clear on 1/16/2002, so there will not be weather delays, so
    # the CTV should be transferring the crew back onto the vessel
    timeout += 1
    env.run(timeout)
    assert ctv.onsite is ctv.at_system is ctv.transferring_crew is True
    assert not ctv.enroute
    assert ctv.current_system == "S01T5"
    assert (
        fsv.at_port
        is fsv.at_system
        is fsv.onsite
        is fsv.enroute
        is fsv.transferring_crew
        is False
    )
    assert (
        hlv.at_port
        is hlv.at_system
        is hlv.onsite
        is hlv.enroute
        is hlv.transferring_crew
        is False
    )

    # Check the transfer is still progressing
    timeout += 13 / 60
    env.run(timeout)
    assert ctv.transferring_crew is ctv.onsite is ctv.at_system is True
    assert ctv.current_system == "S01T5"
    assert not ctv.enroute
    assert len(manager.items) == 1

    ## TURBINE 6 GENERATOR CTV repair

    # Move to the end of the 15 minute to ensure the transfer is complete, and there is
    # no travel time between site objects, so the crew should be transferring again
    timeout += 2 / 60
    env.run(timeout)
    assert ctv.onsite is ctv.at_system is ctv.transferring_crew is True
    assert ctv.current_system == "S01T6"
    assert ctv.enroute is False

    # Move to just before the 15 minute transfer is complete to ensure it hasn't ended early
    timeout += 13 / 60
    env.run(timeout)
    assert ctv.transferring_crew is ctv.onsite is ctv.at_system is True
    assert ctv.current_system == "S01T6"
    assert not ctv.enroute

    # Move to the end of the 15 minute to ensure the transfer is complete
    timeout += 2 / 60
    env.run(timeout)
    assert ctv.onsite is ctv.at_system is True
    assert ctv.current_system == "S01T6"
    assert ctv.enroute is ctv.transferring_crew is False

    # The weather window is clear on 1/16/2002, so there will not be weather delays, so
    # the CTV should be transferring the crew back onto the vessel
    timeout += 1
    env.run(timeout)
    assert ctv.onsite is ctv.at_system is ctv.transferring_crew is True
    assert not ctv.enroute
    assert ctv.current_system == "S01T6"
    assert (
        fsv.at_port
        is fsv.at_system
        is fsv.onsite
        is fsv.enroute
        is fsv.transferring_crew
        is False
    )
    assert (
        hlv.at_port
        is hlv.at_system
        is hlv.onsite
        is hlv.enroute
        is hlv.transferring_crew
        is False
    )

    # Check the transfer is still progressing
    timeout += 13 / 60
    env.run(timeout)
    assert ctv.transferring_crew is ctv.onsite is ctv.at_system is True
    assert ctv.current_system == "S01T6"
    assert not ctv.enroute
    assert len(manager.items) == 0

    # The CTV process will repeat every 15 days (as this is a maintenance task), but no
    # more events will be checked as this process is considered "vetted". Now we will
    # check in on the HLV, then FSV scheduling.
    timeout += 2 / 60
    env.run(timeout)
    assert ctv.onsite
    assert ctv.current_system is None
    assert ctv.transferring_crew is ctv.enroute is ctv.at_system is False

    current = env.simulation_time
    assert current.date().day == 16
    assert current.date().month == 1
    assert current.date().year == 2002
    assert current.hour == 16
    assert current.minute == 1


@pytest.mark.cat("all", "service_equipment")
def test_scheduled_equipment_logic(env_setup_full_profile):
    """ "Test the simulation logic of a scheduled HLV."""
    env = env_setup_full_profile
    manager = RepairManager(env)
    windfarm = Windfarm(env, "layout_simulation.csv", manager)

    # ctv = ServiceEquipment(env, windfarm, manager, "ctv.yaml")
    fsv = ServiceEquipment(env, windfarm, manager, "fsv_scheduled.yaml")
    hlv = ServiceEquipment(env, windfarm, manager, "hlv_scheduled.yaml")

    # Start the simulation to ensure everything is in place as required
    env.run(1)
    # assert ctv.at_port is ctv.at_system is ctv.transferring_crew is ctv.enroute is (not ctv.onsite) is False
    assert (
        fsv.at_port
        is fsv.at_system
        is fsv.transferring_crew
        is fsv.enroute
        is fsv.onsite
        is False
    )
    assert (
        hlv.at_port
        is hlv.at_system
        is hlv.transferring_crew
        is hlv.enroute
        is hlv.onsite
        is False
    )

    # Moving on to the HLV, scheduled in May with 60 days mobilization, and
    # default working hours
    timeout = env.date_ix(hlv.settings.operating_dates[0]) + 8
    mobilization_timeout = timeout - 24 * 60

    env.run(mobilization_timeout - 1)
    assert (
        hlv.at_port
        is hlv.at_system
        is hlv.transferring_crew
        is hlv.enroute
        is hlv.onsite
        is False
    )
    assert (
        fsv.at_port
        is fsv.at_system
        is fsv.transferring_crew
        is fsv.enroute
        is fsv.onsite
        is False
    )

    env.run(mobilization_timeout + 1)
    assert hlv.enroute
    assert hlv.at_port is hlv.at_system is hlv.transferring_crew is hlv.onsite is False
    assert (
        fsv.at_port
        is fsv.at_system
        is fsv.transferring_crew
        is fsv.enroute
        is fsv.onsite
        is False
    )

    env.run(timeout - 1)  # starting hours
    assert hlv.enroute
    assert hlv.at_port is hlv.at_system is hlv.transferring_crew is hlv.onsite is False
    assert (
        fsv.at_port
        is fsv.at_system
        is fsv.transferring_crew
        is fsv.enroute
        is fsv.onsite
        is False
    )

    # Check for the correct number of items before the HLV is operating
    N_days = np.floor(env.now / 24)
    correct_N_ctv_items = (
        np.floor(N_days / 15) * 6
    )  # calls every 15 days for 6 turbines
    correct_N_hlv_items = (
        np.floor(N_days / 50) * 6
    )  # calls every 50 days for 6 turbines
    correct_N_fsv_items = (
        np.floor(N_days / 25) * 6
    )  # calls every 25 days for 6 turbines
    correct_N_items = correct_N_ctv_items + correct_N_hlv_items + correct_N_fsv_items
    n_ctv_items = len(
        [el for el in manager.items if el.details.description == "ctv call"]
    )
    n_hlv_items = len(
        [el for el in manager.items if el.details.description == "hlv call"]
    )
    n_fsv_items = len(
        [el for el in manager.items if el.details.description == "fsv call"]
    )
    assert len(
        manager.items == correct_N_items
    )  # all smaller requests should have been
    assert n_ctv_items == correct_N_ctv_items
    assert n_hlv_items == correct_N_hlv_items
    assert n_fsv_items == correct_N_fsv_items

    # HLV repairs are 3 hours long
    # Repair 1 has operating hours 8-16 with inclimate weather the whole first day, so
    # when the HLV attempts to transfer the crew, it will go back to port.
    # TODO: Have a better logic to pre-empt wasted travel like this
    env.run(timeout + 1)
    assert hlv.at_port
    assert hlv.transferring_crew is hlv.enroute is hlv.onsite is hlv.at_system is False
    assert (
        len([el for el in manager.items if el.details.description == "hlv call"]) == 11
    )

    # HLV should be in the same spot at the end of the shift
    timeout += 8
    env.run(timeout - 0.1)
    assert hlv.at_port
    assert hlv.transferring_crew is hlv.enroute is hlv.at_system is hlv.onsite is False
    assert (
        len([el for el in manager.items if el.details.description == "hlv call"]) == 11
    )

    env.run(timeout + 0.1)
    assert hlv.at_port
    assert hlv.at_system is hlv.onsite is hlv.transferring_crew is hlv.enroute is False

    # The next shift has a clear weather window for the first two hours so the repair
    # should be partially complete at that point, but require an additional 15 minutes for
    # both crew transfer and completion after a windy hour
    timeout += 16
    env.run(timeout - 0.1)
    assert hlv.at_port
    assert hlv.at_system is hlv.onsite is hlv.transferring_crew is hlv.enroute is False
    assert (
        len([el for el in manager.items if el.details.description == "hlv call"]) == 11
    )

    env.run(timeout + 0.1)
    assert hlv.at_system is hlv.onsite is hlv.transferring_crew is True
    assert hlv.at_port is hlv.enroute is False
    assert hlv.current_system == "S00T1"
    assert (
        len([el for el in manager.items if el.details.description == "hlv call"]) == 11
    )

    timeout += 14 / 60
    env.run(timeout)
    assert hlv.at_system is hlv.onsite is hlv.transferring_crew is True
    assert hlv.at_port is hlv.enroute is False
    assert hlv.current_system == "S00T1"
    assert (
        len([el for el in manager.items if el.details.description == "hlv call"]) == 11
    )

    timeout += 2 / 60
    env.run(timeout)
    assert hlv.at_system is hlv.onsite is True
    assert hlv.at_port is hlv.enroute is hlv.transferring_crew is False
    assert hlv.current_system == "S00T1"
    assert (
        len([el for el in manager.items if el.details.description == "hlv call"]) == 11
    )

    # Move to just past the end of the repair when the crew transfer is ongoing
    timeout += 4  # 3 hour repair + 1 hour delay
    env.run(timeout)
    assert hlv.at_system is hlv.onsite is hlv.transferring_crew is True
    assert hlv.at_port is hlv.enroute is False
    assert hlv.current_system == "S00T1"
    assert (
        len([el for el in manager.items if el.details.description == "hlv call"]) == 11
    )

    timeout += 13 / 60
    assert hlv.at_system is hlv.onsite is hlv.transferring_crew is True
    assert hlv.at_port is hlv.enroute is False
    assert hlv.current_system == "S00T1"
    assert (
        len([el for el in manager.items if el.details.description == "hlv call"]) == 11
    )

    # Now move to the 2nd repair for the day on S00T2 (no travel time between turbines)

    timeout += 2 / 60
    env.run(timeout)
    assert hlv.at_system is hlv.onsite is hlv.transferring_crew is True
    assert hlv.at_port is hlv.enroute is False
    assert hlv.current_system == "S00T2"
    assert (
        len([el for el in manager.items if el.details.description == "hlv call"]) == 10
    )

    timeout += 13 / 60
    env.run(timeout)
    assert hlv.at_system is hlv.onsite is hlv.transferring_crew is True
    assert hlv.at_port is hlv.enroute is False
    assert hlv.current_system == "S00T2"
    assert (
        len([el for el in manager.items if el.details.description == "hlv call"]) == 10
    )

    timeout += 2 / 60
    env.run(timeout)
    assert hlv.at_system is hlv.onsite is True
    assert hlv.at_port is hlv.enroute is hlv.transferring_crew is False
    assert hlv.current_system == "S00T2"
    assert (
        len([el for el in manager.items if el.details.description == "hlv call"]) == 10
    )

    # Move to just past the end of the repair (no delays) to ensure crew is transferring
    timeout += 3
    env.run(timeout)
    assert hlv.at_system is hlv.onsite is hlv.transferring_crew is True
    assert hlv.at_port is hlv.enroute is False
    assert hlv.current_system == "S00T2"
    assert (
        len([el for el in manager.items if el.details.description == "hlv call"]) == 10
    )

    # Move to just after the crew transfer, at which time the HLV will return to port
    timeout += 16 / 60
    env.run(timeout)
    assert hlv.at_port
    assert hlv.enroute is hlv.transferring_crew is hlv.at_system is hlv.onsite is False
    assert hlv.current_system is None
    assert (
        len([el for el in manager.items if el.details.description == "hlv call"]) == 10
    )

    # Now, check the logic for repair # 3 on S00T3

    # Move to just before the start of the shift to ensure we're still at port

    # Ensure crew is transferring

    print(env.simulation_time)
    # for k, v in hlv.__dict__.items():
    #     if k == "settings":
    #         continue
    #     print(k, v)
    assert False


# hours_required = 1
# hours_available = 1
# hours_to_process = 1

# if hours_to_process > 0:
#     hours_to_process = min(hours_available, hours_to_process)
#     if hours_required < hours_to_process:
#         hours_to_process = hours_required
