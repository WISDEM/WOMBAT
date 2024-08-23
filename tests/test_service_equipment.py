"""Test the ServiceEquipment class."""

from __future__ import annotations

import numpy as np
import pytest
import numpy.testing as npt

from wombat.windfarm import Windfarm
from wombat.core.library import load_yaml
from wombat.core.data_classes import (
    VALID_EQUIPMENT,
    Maintenance,
    EquipmentMap,
    RepairRequest,
)
from wombat.core.repair_management import RepairManager
from wombat.core.service_equipment import ServiceEquipment, consecutive_groups


def get_items_by_description(
    manager: RepairManager, description: str
) -> list[RepairRequest]:
    """Get all items in the repair manager by description."""
    return [el for el in manager.items if el.details.description == description]


def test_consecutive_groups():
    """Test the consecutive_groups function in a similar way that this would be used."""
    all_clear = np.array(
        [True, True, False, True, True, True, False, True, True, True, True, False]
    )
    windows = consecutive_groups(np.where(all_clear)[0])
    correct_windows = [np.array([0, 1]), np.array([3, 4, 5]), np.array([7, 8, 9, 10])]
    assert len(windows) == len(correct_windows)
    for window, correct_window in zip(windows, correct_windows):
        npt.assert_equal(window, correct_window)

    # Flip the weather window and test that we only have 1 hour openings
    all_clear = ~all_clear
    windows = consecutive_groups(np.where(all_clear)[0])
    correct_windows = [np.array([2]), np.array([6]), np.array([11])]
    assert len(windows) == len(correct_windows)
    for window, correct_window in zip(windows, correct_windows):
        npt.assert_equal(window, correct_window)


def test_service_equipment_init(env_setup):
    """Test the ServiceEquipment initialization."""
    env = env_setup
    manager = RepairManager(env)
    windfarm = Windfarm(env, "layout.csv", manager)

    # TEST 1
    # Check that a basic CTV initialization looks good

    ctv = ServiceEquipment(env, windfarm, manager, "ctv_quick_load.yaml")
    ctv.finish_setup_with_environment_variables()
    ctv_dict = load_yaml(env.data_dir / "vessels", "ctv_quick_load.yaml")

    # Check the basic attribute assignments
    assert ctv.env == env
    assert ctv.windfarm == windfarm
    assert ctv.manager == manager
    assert not ctv.onsite
    assert not ctv.enroute
    assert not ctv.at_system
    assert not ctv.at_port
    assert not ctv.at_site
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
    ctv.finish_setup_with_environment_variables()

    # Check the basic attribute assignments
    assert ctv.env == env
    assert ctv.windfarm == windfarm
    assert ctv.manager == manager
    assert not ctv.onsite
    assert not ctv.enroute
    assert not ctv.at_system
    assert not ctv.at_site
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
    hlv.finish_setup_with_environment_variables()
    hlv_dict = load_yaml(env.data_dir / "vessels", "hlv_downtime.yaml")

    # Check the basic attribute assignments
    assert hlv.env == env
    assert hlv.windfarm == windfarm
    assert hlv.manager == manager
    assert not hlv.onsite
    assert not hlv.enroute
    assert not hlv.at_system
    assert not hlv.at_site
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
    for capability in VALID_EQUIPMENT:
        if capability == "LCN":
            assert len(getattr(manager.downtime_based_equipment, capability)) == 1
        else:
            assert getattr(manager.downtime_based_equipment, capability) == []

    # TEST 4
    # Check the initializtion for an unscheduled vessel for a request-basis
    hlv = ServiceEquipment(env, windfarm, manager, "hlv_requests.yaml")
    hlv.finish_setup_with_environment_variables()
    hlv_dict = load_yaml(env.data_dir / "vessels", "hlv_requests.yaml")

    # Check the basic attribute assignments
    assert hlv.env == env
    assert hlv.windfarm == windfarm
    assert hlv.manager == manager
    assert not hlv.onsite
    assert not hlv.enroute
    assert not hlv.at_system
    assert not hlv.at_site
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
    for capability in (
        "CTV",
        "SCN",
        "MCN",
        "CAB",
        "RMT",
        "DRN",
        "DSV",
        "AHV",
        "TOW",
        "VSG",
    ):
        assert getattr(manager.request_based_equipment, capability) == []


def test_port_distance_setup(env_setup):
    """Test that the environment port_distance is passed through correctly."""
    env = env_setup
    manager = RepairManager(env)
    windfarm = Windfarm(env, "layout.csv", manager)

    # Check that a None for WombatEnvironment.port_distance does nothing
    ctv = ServiceEquipment(env, windfarm, manager, "ctv_quick_load.yaml")
    ctv.finish_setup_with_environment_variables()
    assert env.port_distance is None
    assert ctv.settings.port_distance == 0

    # Check that <=0 does not override no inputs to servicing equipment settings
    env.port_distance = -0.1
    ctv = ServiceEquipment(env, windfarm, manager, "ctv_quick_load.yaml")
    ctv.finish_setup_with_environment_variables()
    assert env.port_distance == -0.1
    assert ctv.settings.port_distance == 0

    # Check that a real port_distance overrides the default servicing equipment settings
    env.port_distance = 10
    ctv = ServiceEquipment(env, windfarm, manager, "ctv_quick_load.yaml")
    ctv.finish_setup_with_environment_variables()
    assert env.port_distance == ctv.settings.port_distance == 10

    # Check that an already set value to port_distance will not be overridden
    ctv = ServiceEquipment(
        env, windfarm, manager, "ctv_quick_load_with_port_distance.yaml"
    )
    ctv.finish_setup_with_environment_variables()
    assert ctv.settings.port_distance == 1


def test_calculate_salary_cost(env_setup):
    """Test the ``calculate_salary_cost`` method."""
    env = env_setup
    manager = RepairManager(env)
    windfarm = Windfarm(env, "layout.csv", manager)
    ctv = ServiceEquipment(env, windfarm, manager, "ctv_wages.yaml")
    ctv.finish_setup_with_environment_variables()
    ctv_dict = load_yaml(env.data_dir / "vessels", "ctv_wages.yaml")
    ctv_crew = ctv_dict["crew"]

    # Test a small, even number
    n_hours = 4
    cost = ctv.calculate_salary_cost(n_hours)
    assert cost == n_hours / 24 * ctv_crew["day_rate"] * ctv_crew["n_day_rate"]

    # Test a multi-day number
    n_hours = 74.333
    cost = ctv.calculate_salary_cost(n_hours)
    assert cost == n_hours / 24 * ctv_crew["day_rate"] * ctv_crew["n_day_rate"]


def test_calculate_hourly_cost(env_setup):
    """Test the ``calculate_hourly_cost`` method."""
    env = env_setup
    manager = RepairManager(env)
    windfarm = Windfarm(env, "layout.csv", manager)
    ctv = ServiceEquipment(env, windfarm, manager, "ctv_wages.yaml")
    ctv.finish_setup_with_environment_variables()
    ctv_dict = load_yaml(env.data_dir / "vessels", "ctv_wages.yaml")
    ctv_crew = ctv_dict["crew"]

    n_hours = 4
    cost = ctv.calculate_hourly_cost(n_hours)
    assert cost == n_hours * ctv_crew["hourly_rate"] * ctv_crew["n_hourly_rate"]

    n_hours = 74.333
    cost = ctv.calculate_hourly_cost(n_hours)
    assert cost == n_hours * ctv_crew["hourly_rate"] * ctv_crew["n_hourly_rate"]


def test_calculate_equipment_cost(env_setup):
    """Test the ``calculate_equipment_cost`` method."""
    env = env_setup
    manager = RepairManager(env)
    windfarm = Windfarm(env, "layout.csv", manager)
    ctv = ServiceEquipment(env, windfarm, manager, "ctv.yaml")
    ctv.finish_setup_with_environment_variables()
    ctv_dict = load_yaml(env.data_dir / "vessels", "ctv.yaml")

    n_hours = 4
    cost = ctv.calculate_equipment_cost(n_hours)
    assert cost == n_hours / 24 * ctv_dict["equipment_rate"]

    n_hours = 74.333
    cost = ctv.calculate_equipment_cost(n_hours)
    assert cost == n_hours / 24 * ctv_dict["equipment_rate"]


def test_onsite_scheduled_equipment_logic(env_setup_full_profile):
    """Test the simulation logic of a scheduled CTV."""
    env = env_setup_full_profile
    manager = RepairManager(env)
    windfarm = Windfarm(env, "layout_simulation.csv", manager)

    ctv = ServiceEquipment(env, windfarm, manager, "ctv.yaml")
    ctv.finish_setup_with_environment_variables()
    fsv = ServiceEquipment(env, windfarm, manager, "fsv_scheduled.yaml")
    fsv.finish_setup_with_environment_variables()
    hlv = ServiceEquipment(env, windfarm, manager, "hlv_scheduled.yaml")
    hlv.finish_setup_with_environment_variables()

    # Start the simulation to ensure everything is in place as required
    env.run(1)
    assert (
        ctv.at_port
        is ctv.at_site
        is ctv.at_system
        is ctv.transferring_crew
        is ctv.enroute
        is ctv.onsite
        is False
    )
    assert (
        fsv.at_port
        is fsv.at_site
        is fsv.at_system
        is fsv.transferring_crew
        is fsv.enroute
        is fsv.onsite
        is False
    )
    assert (
        hlv.at_port
        is hlv.at_site
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
        is ctv.at_site
        is ctv.at_system
        is ctv.transferring_crew
        is ctv.enroute
        is ctv.onsite
        is False
    )
    assert (
        fsv.at_port
        is fsv.at_site
        is fsv.at_system
        is fsv.transferring_crew
        is fsv.enroute
        is fsv.onsite
        is False
    )
    assert (
        hlv.at_port
        is hlv.at_site
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
    assert ctv.onsite is ctv.at_site is ctv.at_system is ctv.transferring_crew is True
    assert not ctv.enroute
    assert ctv.current_system == "S00T1"
    assert (
        fsv.at_port
        is fsv.at_site
        is fsv.at_system
        is fsv.onsite
        is fsv.enroute
        is fsv.transferring_crew
        is False
    )
    assert (
        hlv.at_port
        is hlv.at_site
        is hlv.at_system
        is hlv.onsite
        is hlv.enroute
        is hlv.transferring_crew
        is False
    )

    # Move to just before the 15 minute transfer is complete to ensure it hasn't ended
    timeout += 13 / 60
    env.run(timeout)
    assert ctv.transferring_crew is ctv.at_site is ctv.at_system is True
    assert ctv.current_system == "S00T1"
    assert not ctv.enroute

    # Move to the end of the 15 minute to ensure the transfer is complete
    timeout += 2 / 60
    env.run(timeout)
    assert ctv.at_site is ctv.at_system is True
    assert ctv.current_system == "S00T1"
    assert ctv.enroute is ctv.at_port is ctv.transferring_crew is False

    # The weather window is clear on 1/16/2002, so there will not be weather delays, so
    # the CTV should be transferring the crew back onto the vessel
    timeout += 1
    env.run(timeout)
    assert ctv.at_site is ctv.at_system is ctv.transferring_crew is True
    assert ctv.at_port is ctv.enroute is False
    assert ctv.current_system == "S00T1"
    assert (
        fsv.at_port
        is fsv.at_site
        is fsv.at_system
        is fsv.onsite
        is fsv.enroute
        is fsv.transferring_crew
        is False
    )
    assert (
        hlv.at_port
        is hlv.at_site
        is hlv.at_system
        is hlv.onsite
        is hlv.enroute
        is hlv.transferring_crew
        is False
    )

    # Check the transfer is still progressing
    timeout += 13 / 60
    env.run(timeout)
    assert ctv.transferring_crew is ctv.at_site is ctv.at_system is True
    assert ctv.current_system == "S00T1"
    assert ctv.enroute is ctv.at_port is False
    assert len(manager.items) == 5

    ## TURBINE 2 GENERATOR CTV repair

    # Move to the end of the 15 minute to ensure the transfer is complete, and there is
    # no travel time between site objects, so the crew should be transferring again
    timeout += 2 / 60
    env.run(timeout)
    assert ctv.at_site is ctv.at_system is ctv.transferring_crew is True
    assert ctv.current_system == "S00T2"
    assert ctv.at_port is ctv.enroute is False

    # Move to just before the 15 minute transfer is complete to ensure it hasn't ended
    timeout += 13 / 60
    env.run(timeout)
    assert ctv.transferring_crew is ctv.at_site is ctv.at_system is True
    assert ctv.current_system == "S00T2"
    assert ctv.at_port is ctv.enroute is False

    # Move to the end of the 15 minute to ensure the transfer is complete
    timeout += 2 / 60
    env.run(timeout)
    assert ctv.at_site is ctv.at_system is True
    assert ctv.current_system == "S00T2"
    assert ctv.enroute is ctv.at_port is ctv.transferring_crew is False

    # The weather window is clear on 1/16/2002, so there will not be weather delays, so
    # the CTV should be transferring the crew back onto the vessel
    timeout += 1
    env.run(timeout)
    assert ctv.at_site is ctv.at_system is ctv.transferring_crew is True
    assert ctv.enroute is ctv.at_port is False
    assert ctv.current_system == "S00T2"
    assert (
        fsv.at_port
        is fsv.at_site
        is fsv.at_system
        is fsv.onsite
        is fsv.enroute
        is fsv.transferring_crew
        is False
    )
    assert (
        hlv.at_port
        is hlv.at_site
        is hlv.at_system
        is hlv.onsite
        is hlv.enroute
        is hlv.transferring_crew
        is False
    )

    # Check the transfer is still progressing
    timeout += 13 / 60
    env.run(timeout)
    assert ctv.transferring_crew is ctv.at_site is ctv.at_system is True
    assert ctv.current_system == "S00T2"
    assert ctv.enroute is ctv.at_port is False
    assert len(manager.items) == 4

    ## TURBINE 3 GENERATOR CTV repair

    # Move to the end of the 15 minute to ensure the transfer is complete, and there is
    # no travel time between site objects, so the crew should be transferring again
    timeout += 2 / 60
    env.run(timeout)
    assert ctv.at_site is ctv.at_system is ctv.transferring_crew is True
    assert ctv.current_system == "S00T3"
    assert ctv.enroute is ctv.at_port is False

    # Move to just before the 15 minute transfer is complete to ensure it hasn't ended
    timeout += 13 / 60
    env.run(timeout)
    assert ctv.transferring_crew is ctv.at_site is ctv.at_system is True
    assert ctv.current_system == "S00T3"
    assert ctv.enroute is ctv.at_port is False

    # Move to the end of the 15 minute to ensure the transfer is complete
    timeout += 2 / 60
    env.run(timeout)
    assert ctv.at_site is ctv.at_system is True
    assert ctv.current_system == "S00T3"
    assert ctv.enroute is ctv.at_port is ctv.transferring_crew is False

    # The weather window is clear on 1/16/2002, so there will not be weather delays, so
    # the CTV should be transferring the crew back onto the vessel
    timeout += 1
    env.run(timeout)
    assert ctv.at_site is ctv.at_system is ctv.transferring_crew is True
    assert ctv.enroute is ctv.at_port is False
    assert ctv.current_system == "S00T3"
    assert (
        fsv.at_port
        is fsv.at_site
        is fsv.at_system
        is fsv.onsite
        is fsv.enroute
        is fsv.transferring_crew
        is False
    )
    assert (
        hlv.at_port
        is hlv.at_site
        is hlv.at_system
        is hlv.onsite
        is hlv.enroute
        is hlv.transferring_crew
        is False
    )

    # Check the transfer is still progressing
    timeout += 13 / 60
    env.run(timeout)
    assert ctv.transferring_crew is ctv.at_site is ctv.at_system is True
    assert ctv.current_system == "S00T3"
    assert ctv.enroute is ctv.at_port is False
    assert len(manager.items) == 3

    ## TURBINE 4 GENERATOR CTV repair

    # Move to the end of the 15 minute to ensure the transfer is complete, and there is
    # no travel time between site objects, so the crew should be transferring again
    timeout += 2 / 60
    env.run(timeout)
    assert ctv.at_site is ctv.at_system is ctv.transferring_crew is True
    assert ctv.current_system == "S01T4"
    assert ctv.enroute is ctv.at_port is False

    # Move to just before the 15 minute transfer is complete to ensure it hasn't ended
    timeout += 13 / 60
    env.run(timeout)
    assert ctv.transferring_crew is ctv.onsite is ctv.at_site is ctv.at_system is True
    assert ctv.current_system == "S01T4"
    assert ctv.enroute is ctv.at_port is False

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
    assert ctv.onsite is ctv.at_site is ctv.at_system is ctv.transferring_crew is True
    assert ctv.enroute is ctv.at_port is False
    assert ctv.current_system == "S01T4"
    assert (
        fsv.at_port
        is fsv.at_site
        is fsv.at_system
        is fsv.onsite
        is fsv.enroute
        is fsv.transferring_crew
        is False
    )
    assert (
        hlv.at_port
        is hlv.at_site
        is hlv.at_system
        is hlv.onsite
        is hlv.enroute
        is hlv.transferring_crew
        is False
    )

    # Check the transfer is still progressing
    timeout += 13 / 60
    env.run(timeout)
    assert ctv.transferring_crew is ctv.onsite is ctv.at_site is ctv.at_system is True
    assert ctv.current_system == "S01T4"
    assert ctv.enroute is ctv.at_port is False
    assert len(manager.items) == 2

    ## TURBINE 5 GENERATOR CTV repair

    # Move to the end of the 15 minute to ensure the transfer is complete, and there is
    # no travel time between site objects, so the crew should be transferring again
    timeout += 2 / 60
    env.run(timeout)
    assert ctv.onsite is ctv.at_site is ctv.at_system is ctv.transferring_crew is True
    assert ctv.current_system == "S01T5"
    assert ctv.enroute is ctv.at_port is False

    # Move to just before the 15 minute transfer is complete to ensure it hasn't ended
    timeout += 13 / 60
    env.run(timeout)
    assert ctv.transferring_crew is ctv.onsite is ctv.at_site is ctv.at_system is True
    assert ctv.current_system == "S01T5"
    assert ctv.enroute is ctv.at_port is False

    # Move to the end of the 15 minute to ensure the transfer is complete
    timeout += 2 / 60
    env.run(timeout)
    assert ctv.onsite is ctv.at_site is ctv.at_system is True
    assert ctv.current_system == "S01T5"
    assert ctv.enroute is ctv.at_port is ctv.transferring_crew is False

    # The weather window is clear on 1/16/2002, so there will not be weather delays, so
    # the CTV should be transferring the crew back onto the vessel
    timeout += 1
    env.run(timeout)
    assert ctv.onsite is ctv.at_site is ctv.at_system is ctv.transferring_crew is True
    assert ctv.enroute is ctv.at_port is False
    assert ctv.current_system == "S01T5"
    assert (
        fsv.at_port
        is fsv.at_site
        is fsv.at_system
        is fsv.onsite
        is fsv.enroute
        is fsv.transferring_crew
        is False
    )
    assert (
        hlv.at_port
        is hlv.at_site
        is hlv.at_system
        is hlv.onsite
        is hlv.enroute
        is hlv.transferring_crew
        is False
    )

    # Check the transfer is still progressing
    timeout += 13 / 60
    env.run(timeout)
    assert ctv.transferring_crew is ctv.onsite is ctv.at_site is ctv.at_system is True
    assert ctv.current_system == "S01T5"
    assert ctv.enroute is ctv.at_port is False
    assert len(manager.items) == 1

    ## TURBINE 6 GENERATOR CTV repair

    # Move to the end of the 15 minute to ensure the transfer is complete, and there is
    # no travel time between site objects, so the crew should be transferring again
    timeout += 2 / 60
    env.run(timeout)
    assert ctv.onsite is ctv.at_site is ctv.at_system is ctv.transferring_crew is True
    assert ctv.current_system == "S01T6"
    assert ctv.enroute is ctv.at_port is False

    # Move to just before the 15 minute transfer is complete to ensure it hasn't ended
    timeout += 13 / 60
    env.run(timeout)
    assert ctv.transferring_crew is ctv.onsite is ctv.at_site is ctv.at_system is True
    assert ctv.current_system == "S01T6"
    assert ctv.enroute is ctv.at_port is False

    # Move to the end of the 15 minute to ensure the transfer is complete
    timeout += 2 / 60
    env.run(timeout)
    assert ctv.at_site is ctv.at_system is True
    assert ctv.current_system == "S01T6"
    assert ctv.enroute is ctv.transferring_crew is False

    # The weather window is clear on 1/16/2002, so there will not be weather delays, so
    # the CTV should be transferring the crew back onto the vessel
    timeout += 1
    env.run(timeout)
    assert ctv.onsite is ctv.at_site is ctv.at_system is ctv.transferring_crew is True
    assert ctv.enroute is ctv.at_port is False
    assert ctv.current_system == "S01T6"
    assert (
        fsv.at_port
        is fsv.at_site
        is fsv.at_system
        is fsv.onsite
        is fsv.enroute
        is fsv.transferring_crew
        is False
    )
    assert (
        hlv.at_port
        is hlv.at_site
        is hlv.at_system
        is hlv.onsite
        is hlv.enroute
        is hlv.transferring_crew
        is False
    )

    # Check the transfer is still progressing
    timeout += 13 / 60
    env.run(timeout)
    assert ctv.transferring_crew is ctv.onsite is ctv.at_site is ctv.at_system is True
    assert ctv.current_system == "S01T6"
    assert ctv.enroute is ctv.at_port is False
    assert len(manager.items) == 0

    # The CTV process will repeat every 15 days (as this is a maintenance task), but no
    # more events will be checked as this process is considered "vetted".
    timeout += 2 / 60
    env.run(timeout)
    assert ctv.at_port and ctv.onsite
    assert ctv.current_system is None
    assert (
        ctv.at_site
        is ctv.transferring_crew
        is ctv.at_site
        is ctv.enroute
        is ctv.at_system
        is False
    )

    current = env.simulation_time
    assert current.date().day == 16
    assert current.date().month == 1
    assert current.date().year == 2002
    assert current.hour == 16
    assert current.minute == 1


@pytest.mark.skip(reason="The timing of the failures needs to be updated")
def test_scheduled_equipment_logic(env_setup_full_profile):
    """Test the simulation logic of a scheduled HLV."""
    env = env_setup_full_profile
    manager = RepairManager(env)
    windfarm = Windfarm(env, "layout_simulation.csv", manager)

    # ctv = ServiceEquipment(env, windfarm, manager, "ctv.yaml")
    fsv = ServiceEquipment(env, windfarm, manager, "fsv_scheduled.yaml")
    fsv.finish_setup_with_environment_variables()
    hlv = ServiceEquipment(env, windfarm, manager, "hlv_scheduled.yaml")
    hlv.finish_setup_with_environment_variables()

    # Start the simulation to ensure everything is in place as required
    env.run(1)
    # assert (
    #     ctv.at_port is ctv.at_system
    #     is ctv.transferring_crew
    #     is ctv.enroute
    #     is (not ctv.onsite)
    #     is False
    # )
    assert (
        fsv.at_port
        is fsv.at_site
        is fsv.at_system
        is fsv.transferring_crew
        is fsv.enroute
        is fsv.onsite
        is False
    )
    assert (
        hlv.at_port
        is hlv.at_site
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
        is hlv.at_site
        is hlv.at_system
        is hlv.transferring_crew
        is hlv.enroute
        is hlv.onsite
        is False
    )
    assert (
        fsv.at_port
        is fsv.at_site
        is fsv.at_system
        is fsv.transferring_crew
        is fsv.enroute
        is fsv.onsite
        is False
    )

    env.run(mobilization_timeout + 1)
    assert hlv.enroute
    assert (
        hlv.at_port
        is hlv.at_site
        is hlv.at_system
        is hlv.transferring_crew
        is hlv.onsite
        is False
    )
    assert (
        fsv.at_port
        is fsv.at_site
        is fsv.at_system
        is fsv.transferring_crew
        is fsv.enroute
        is fsv.onsite
        is False
    )

    env.run(timeout - 1)  # starting hours
    assert hlv.enroute
    assert (
        hlv.at_port
        is hlv.at_site
        is hlv.at_system
        is hlv.transferring_crew
        is hlv.onsite
        is False
    )
    assert (
        fsv.at_port
        is fsv.at_site
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
    n_ctv_items = len(get_items_by_description(manager, "ctv call"))
    n_hlv_items = len(get_items_by_description(manager, "hlv call"))
    n_fsv_items = len(get_items_by_description(manager, "fsv call"))
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
    assert hlv.at_port is hlv.onsite is True
    assert hlv.transferring_crew is hlv.enroute is hlv.at_site is hlv.at_system is False
    assert len(get_items_by_description(manager, "hlv call")) == 11

    # HLV should be in the same spot at the end of the shift
    timeout += 8
    env.run(timeout - 0.1)
    assert hlv.at_port is hlv.at_site is True
    assert hlv.transferring_crew is hlv.enroute is hlv.at_system is hlv.onsite is False
    assert len(get_items_by_description(manager, "hlv call")) == 11

    env.run(timeout + 0.1)
    assert hlv.at_port is hlv.onsite is True
    assert hlv.at_system is hlv.at_site is hlv.transferring_crew is hlv.enroute is False

    # The next shift has a clear weather window for the first two hours so the repair
    # should be partially complete at that point, but require an additional 15 minutes
    # for both crew transfer and completion after a windy hour
    timeout += 16
    env.run(timeout - 0.1)
    assert hlv.at_port is hlv.onsite is True
    assert hlv.at_system is hlv.at_site is hlv.transferring_crew is hlv.enroute is False
    assert len(get_items_by_description(manager, "hlv call")) == 11

    env.run(timeout + 0.1)
    assert hlv.at_system is hlv.onsite is hlv.at_site is hlv.transferring_crew is True
    assert hlv.at_port is hlv.enroute is False
    assert hlv.current_system == "S00T1"
    assert len(get_items_by_description(manager, "hlv call")) == 11

    timeout += 14 / 60
    env.run(timeout)
    assert hlv.at_system is hlv.onsite is hlv.at_site is hlv.transferring_crew is True
    assert hlv.at_port is hlv.enroute is False
    assert hlv.current_system == "S00T1"
    assert len(get_items_by_description(manager, "hlv call")) == 11

    timeout += 2 / 60
    env.run(timeout)
    assert hlv.at_system is hlv.onsite is hlv.at_site is True
    assert hlv.at_port is hlv.enroute is hlv.transferring_crew is False
    assert hlv.current_system == "S00T1"
    assert len(get_items_by_description(manager, "hlv call")) == 11

    # Move to just past the end of the repair when the crew transfer is ongoing
    timeout += 5  # 3 hour repair + 1 hour delay
    env.run(timeout)
    assert hlv.at_system is hlv.onsite is hlv.at_site is hlv.transferring_crew is True
    assert hlv.at_port is hlv.enroute is False
    assert hlv.current_system == "S00T1"
    assert len(get_items_by_description(manager, "hlv call")) == 11

    timeout += 13 / 60
    assert hlv.at_system is hlv.onsite is hlv.at_site is hlv.transferring_crew is True
    assert hlv.at_port is hlv.enroute is False
    assert hlv.current_system == "S00T1"
    assert len(get_items_by_description(manager, "hlv call")) == 11

    # Now move to the 2nd repair for the day on S00T2 (no travel time between turbines)

    timeout += 2 / 60
    env.run(timeout)
    assert hlv.at_system is hlv.onsite is hlv.at_site is hlv.transferring_crew is True
    assert hlv.at_port is hlv.enroute is False
    assert hlv.current_system == "S00T2"
    assert len(get_items_by_description(manager, "hlv call")) == 10

    timeout += 13 / 60
    env.run(timeout)
    assert hlv.at_system is hlv.onsite is hlv.at_site is hlv.transferring_crew is True
    assert hlv.at_port is hlv.enroute is False
    assert hlv.current_system == "S00T2"
    assert len(get_items_by_description(manager, "hlv call")) == 10

    timeout += 2 / 60
    env.run(timeout)
    assert hlv.at_system is hlv.onsite is hlv.at_site is True
    assert hlv.at_port is hlv.enroute is hlv.transferring_crew is False
    assert hlv.current_system == "S00T2"
    assert len(get_items_by_description(manager, "hlv call")) == 10

    # Move to just past the end of the repair (no delays) to ensure crew is transferring
    timeout += 2
    env.run(timeout)
    assert hlv.at_system is hlv.onsite is hlv.at_site is hlv.transferring_crew is True
    assert hlv.at_port is hlv.enroute is False
    assert hlv.current_system == "S00T2"
    assert len(get_items_by_description(manager, "hlv call")) == 10

    # Move to just after the crew transfer, at which time the HLV will return to port
    timeout += 16 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.onsite is True
    assert hlv.enroute is hlv.transferring_crew is hlv.at_system is hlv.at_site is False
    assert hlv.current_system is None
    assert len(get_items_by_description(manager, "hlv call")) == 10

    # Now, check the logic for repair # 3 on S00T3

    # Move to just before the start of the shift to ensure it's starting correctly
    timeout += 16 - 3 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.onsite is True
    assert hlv.enroute is hlv.transferring_crew is hlv.at_system is hlv.at_site is False
    assert hlv.current_system is None
    assert len(get_items_by_description(manager, "hlv call")) == 10

    # Ensure crew is transferring
    timeout += 2 / 60
    env.run(timeout)
    assert hlv.at_system is hlv.onsite is hlv.at_site is hlv.transferring_crew is True
    assert hlv.at_port is hlv.enroute is False
    assert hlv.current_system == "S00T2"
    assert len(get_items_by_description(manager, "hlv call")) == 10

    # Ensure crew is still transferring
    timeout += 13 / 60
    env.run(timeout)
    assert hlv.at_system is hlv.onsite is hlv.at_site is hlv.transferring_crew is True
    assert hlv.at_port is hlv.enroute is False
    assert hlv.current_system == "S00T2"
    assert len(get_items_by_description(manager, "hlv call")) == 10

    # TODO: This is where the timing breaks due to now mistimed weather and shift delays
    # with new logic

    # Ensure repair is processed with a 2 hour delay
    print(env.now)
    timeout += 3 + 2 + 2 / 60
    env.run(timeout)
    assert hlv.at_system is hlv.onsite is hlv.at_site is hlv.transferring_crew is True
    assert hlv.at_port is hlv.enroute is False
    assert hlv.current_system == "S00T2"
    assert len(get_items_by_description(manager, "hlv call")) == 10

    # Ensure crew is transferring to the vessel
    timeout += 2 / 60
    env.run(timeout)
    assert hlv.at_system is hlv.onsite is hlv.at_site is hlv.transferring_crew is True
    assert hlv.at_port is hlv.enroute is False
    assert hlv.current_system == "S00T2"
    assert len(get_items_by_description(manager, "hlv call")) == 10

    # Ensure crew is transferring to the next turbine
    timeout += 14 / 60
    env.run(timeout)
    assert hlv.at_system is hlv.onsite is hlv.at_site is hlv.transferring_crew is True
    assert hlv.at_port is hlv.enroute is False
    assert hlv.current_system == "S01T4"
    assert len(get_items_by_description(manager, "hlv call")) == 8

    # Ensure crew stopped transferring
    timeout += 14 / 60
    env.run(timeout)
    assert hlv.at_system is hlv.onsite is hlv.at_site is True
    assert hlv.at_port is hlv.enroute is hlv.transferring_crew is False
    assert hlv.current_system == "S01T4"
    assert len(get_items_by_description(manager, "hlv call")) == 8

    # Ensure repair is hit with a delay at the end of the shift
    timeout += 2
    env.run(timeout)
    assert hlv.at_system is hlv.onsite is hlv.at_site is hlv.transferring_crew is True
    assert hlv.at_port is hlv.enroute is False
    assert hlv.current_system == "S01T4"
    assert len(get_items_by_description(manager, "hlv call")) == 8

    # Check the crew is transferred
    timeout += 1 / 60
    env.run(timeout)
    assert hlv.at_system is hlv.onsite is hlv.at_site is hlv.transferring_crew is True
    assert hlv.at_port is hlv.enroute is False
    assert hlv.current_system == "S01T4"
    assert len(get_items_by_description(manager, "hlv call")) == 8

    timeout += 13 / 60
    env.run(timeout)
    assert hlv.at_system is hlv.onsite is hlv.at_site is hlv.transferring_crew is True
    assert hlv.at_port is hlv.enroute is False
    assert hlv.current_system == "S01T4"
    assert len(get_items_by_description(manager, "hlv call")) == 8

    timeout += 1 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.onsite
    assert hlv.at_system is hlv.at_site is hlv.transferring_crew is hlv.enroute is False
    assert hlv.current_system is None
    assert len(get_items_by_description(manager, "hlv call")) == 8

    # Wait until the start of the next shift and check for the crew transferring
    timeout += 16
    env.run(timeout)
    assert hlv.at_port is hlv.onsite
    assert hlv.at_system is hlv.at_site is hlv.transferring_crew is hlv.enroute is False
    assert hlv.current_system is None
    assert len(get_items_by_description(manager, "hlv call")) == 8

    # Check the crew is unable to transfer and there is a shift delay until 5/5 at 14:00
    # TODO: The hlv will travel to site as an edge case in the crew_transfer delay
    timeout += 24
    env.run(timeout)
    assert hlv.onsite is hlv.at_site is hlv.at_system is True
    assert hlv.transferring_crew is hlv.enroute is hlv.at_port is False
    assert hlv.current_system == "S01T4"
    assert len(get_items_by_description(manager, "hlv call")) == 8

    # Check we haven't gone anywhere
    timeout += 6 - 1 / 60
    env.run(timeout)
    assert hlv.at_system is hlv.at_site is hlv.onsite is True
    assert hlv.at_port is hlv.transferring_crew is hlv.enroute is False
    assert hlv.current_system == "S01T4"
    assert len(get_items_by_description(manager, "hlv call")) == 8

    # Will be transferring crew now
    timeout += 1 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.enroute is False
    assert hlv.at_system is hlv.at_site is hlv.onsite is hlv.transferring_crew is True
    assert hlv.current_system == "S01T4"
    assert len(get_items_by_description(manager, "hlv call")) == 8

    # Check that the maintenance is now going for the remained 1 hour after the transfer
    timeout += 1 + 13 / 60
    env.run(timeout)
    assert hlv.at_system is hlv.at_site is hlv.onsite is True
    assert hlv.at_port is hlv.enroute is hlv.transferring_crew is False
    assert hlv.current_system == "S01T4"
    assert len(get_items_by_description(manager, "hlv call")) == 8

    # Will be transferring crew back to the HLV
    timeout += 2 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.enroute is False
    assert hlv.at_system is hlv.at_site is hlv.onsite is hlv.transferring_crew is True
    assert hlv.current_system == "S01T4"
    assert len(get_items_by_description(manager, "hlv call")) == 8

    # Repair 5 on S01T5 will be delayed overnight due to insufficient time to transfer
    # the crew back and forth with a safety buffer

    # Will be waiting overnight at port
    timeout += 15 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.onsite is True
    assert hlv.enroute is hlv.at_site is hlv.at_system is hlv.transferring_crew is False
    assert hlv.current_system is None
    assert len(get_items_by_description(manager, "hlv call")) == 7

    # Delayed until 8am
    timeout += 16 + 29 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.onsite is True
    assert hlv.enroute is hlv.at_system is hlv.at_site is hlv.transferring_crew is False
    assert hlv.current_system is None
    assert len(get_items_by_description(manager, "hlv call")) == 7

    # Transferring crew to the turbine
    timeout += 15 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.enroute is False
    assert hlv.at_system is hlv.onsite is hlv.at_site is hlv.transferring_crew is True
    assert hlv.current_system == "S01T5"
    assert len(get_items_by_description(manager, "hlv call")) == 7

    # After the 3 hour repair and prior crew transfer, crew is being transferred back
    timeout += 3 + 14 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.enroute is False
    assert hlv.at_system is hlv.onsite is hlv.at_site is hlv.transferring_crew is True
    assert hlv.current_system == "S01T5"
    assert len(get_items_by_description(manager, "hlv call")) == 7

    # Repair 6, S01T6 with no weather or shift delays

    # Crew is transferred to the next turbine
    timeout += 15 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.enroute is False
    assert hlv.at_system is hlv.onsite is hlv.at_site is hlv.transferring_crew is True
    assert hlv.current_system == "S01T6"
    assert len(get_items_by_description(manager, "hlv call")) == 6

    # After a crew transfer, and 3 hour repair, the crew is being transferred back
    timeout += 3 + 15 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.enroute is False
    assert hlv.at_system is hlv.onsite is hlv.at_site is hlv.transferring_crew is True
    assert hlv.current_system == "S01T6"
    assert len(get_items_by_description(manager, "hlv call")) == 6

    # Repair 7 will start on the next day due to time constraints for ensuring a safe
    # window for crew transfer, then will complete without delay on the next day

    # Crew is transferred, and the HLV is back at port waiting to start on 5/6 8AM
    timeout += 2 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.onsite
    assert hlv.at_system is hlv.at_site is hlv.transferring_crew is hlv.enroute is False
    assert hlv.current_system is None
    assert len(get_items_by_description(manager, "hlv call")) == 5

    # Timeout to 5/6 at 8:01AM for crew transferring to the turbine
    timeout += 1 + 16  # 1 hour left in shift + 16 hours overnight
    env.run(timeout)
    assert hlv.at_port is hlv.enroute is False
    assert hlv.at_system is hlv.at_site is hlv.onsite is hlv.transferring_crew is True
    assert hlv.current_system == "S00T1"
    assert len(get_items_by_description(manager, "hlv call")) == 5

    # Crew is transferring back to the vessel after a completed repair
    timeout += 3 + 15 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.enroute is False
    assert hlv.at_system is hlv.at_site is hlv.onsite is hlv.transferring_crew is True
    assert hlv.current_system == "S00T1"
    assert len(get_items_by_description(manager, "hlv call")) == 5

    # Repair 8, S00T2 has no weather or shift delays

    # Crew is transferring back to the vessel after a completed repair
    timeout += 15 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.enroute is False
    assert hlv.at_system is hlv.at_site is hlv.onsite is hlv.transferring_crew is True
    assert hlv.current_system == "S00T2"
    assert len(get_items_by_description(manager, "hlv call")) == 4

    # Crew is transferring back to the vessel after a completed repair
    timeout += 3 + 15 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.enroute is False
    assert hlv.at_system is hlv.at_site is hlv.onsite is hlv.transferring_crew is True
    assert hlv.current_system == "S00T2"
    assert len(get_items_by_description(manager, "hlv call")) == 4

    # Repair 9, S00T3 will start the next day due to the safety buffer and be processed
    # without delays

    # Crew is back at port
    timeout += 15 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.onsite is True
    assert hlv.at_system is hlv.at_site is hlv.transferring_crew is hlv.enroute is False
    assert hlv.current_system is None
    assert len(get_items_by_description(manager, "hlv call")) == 3

    # Crew is still at port just before the start of the shift
    timeout += 1 + 16 - 2 / 60  # 1 hour remainder + overnight: 5/8 at 7:59AM
    env.run(timeout)
    assert hlv.at_port is hlv.onsite is True
    assert hlv.at_system is hlv.at_site is hlv.transferring_crew is hlv.enroute is False
    assert hlv.current_system is None
    assert len(get_items_by_description(manager, "hlv call")) == 3

    # Crew is transferring to the turbine
    timeout += 15 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.enroute is False
    assert hlv.at_system is hlv.at_site is hlv.onsite is hlv.transferring_crew is True
    assert hlv.current_system == "S00T3"
    assert len(get_items_by_description(manager, "hlv call")) == 3

    # Crew is transferring back to the HLV
    timeout += 3 + 2 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.enroute is False
    assert hlv.at_system is hlv.at_site is hlv.onsite is hlv.transferring_crew is True
    assert hlv.current_system == "S00T3"
    assert len(get_items_by_description(manager, "hlv call")) == 3

    # Repair 10, S01T4 will process without delays

    # Crew is transferring to the turbine
    timeout += 15 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.enroute is False
    assert hlv.at_system is hlv.at_site is hlv.onsite is hlv.transferring_crew is True
    assert hlv.current_system == "S01T4"
    assert len(get_items_by_description(manager, "hlv call")) == 2

    # Crew is still at the turbine
    timeout += 3
    env.run(timeout)
    assert hlv.at_system is hlv.at_site is hlv.onsite is True
    assert hlv.at_port is hlv.enroute is hlv.transferring_crew is False
    assert hlv.current_system == "S01T4"
    assert len(get_items_by_description(manager, "hlv call")) == 2

    # Crew is transferring to the HLV
    timeout += 15 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.enroute is False
    assert hlv.at_system is hlv.at_site is hlv.onsite is hlv.transferring_crew is True
    assert hlv.current_system == "S01T4"
    assert len(get_items_by_description(manager, "hlv call")) == 2

    # Repair 11, S01T5 is delayed to start until the next day because of the safety,
    # but is unable to transfer the crew safely until 5/10 at 13:00 so is effectively
    # stuck at port

    # HLV goes back to port due to the safety buffer
    timeout += 15 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.onsite is True
    assert hlv.at_system is hlv.at_site is hlv.transferring_crew is hlv.enroute is False
    assert hlv.current_system is None
    assert len(get_items_by_description(manager, "hlv call")) == 1

    # HLV is at port after the delay still due to inclimate weather
    timeout += 1 + 16  # 1 hour remainder + 16 hours overnight
    env.run(timeout)
    assert hlv.at_port is hlv.onsite is True
    assert hlv.at_system is hlv.at_site is hlv.transferring_crew is hlv.enroute is False
    assert hlv.current_system is None
    assert len(get_items_by_description(manager, "hlv call")) == 1

    # HLV travels to site, but is unable to transfer crew just yet
    timeout += 24
    env.run(timeout)
    assert hlv.at_system is hlv.at_site is hlv.at_site is hlv.onsite is True
    assert hlv.at_port is hlv.transferring_crew is hlv.enroute is False
    assert hlv.current_system == "S01T5"
    assert len(get_items_by_description(manager, "hlv call")) == 1

    # HLV is transferring crew to the turbine after a 5 hour delay
    timeout += 5
    env.run(timeout)
    assert hlv.at_port is hlv.enroute is False
    assert hlv.at_system is hlv.at_site is hlv.onsite is hlv.transferring_crew is True
    assert hlv.current_system == "S01T5"
    assert len(get_items_by_description(manager, "hlv call")) == 1

    # HLV is transferring crew back to the HLV after processing 2.75 hours of the repair
    timeout += 2.75
    env.run(timeout)
    assert hlv.at_port is hlv.enroute is False
    assert hlv.at_system is hlv.at_site is hlv.onsite is hlv.transferring_crew is True
    assert hlv.current_system == "S01T5"
    assert len(get_items_by_description(manager, "hlv call")) == 1

    # The HLV is back at port
    timeout += 15 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.onsite is True
    assert hlv.at_system is hlv.at_site is hlv.transferring_crew is hlv.enroute is False
    assert hlv.current_system is None
    assert len(get_items_by_description(manager, "hlv call")) == 1

    # Check that the Field Support Vessel has been deployed at midnight to arrive for
    # the start of its first workshift
    timeout += 8
    env.run(timeout)
    assert fsv.enroute
    assert (
        fsv.at_system
        is fsv.at_site
        is fsv.onsite
        is fsv.transferring_crew
        is fsv.at_port
        is False
    )
    assert len(get_items_by_description(manager, "fsv call")) == 30

    # The HLV is transferring the crew to the turbine
    timeout += 8
    env.run(timeout)
    assert hlv.at_port is hlv.enroute is False
    assert hlv.at_system is hlv.at_site is hlv.onsite is hlv.transferring_crew is True
    assert hlv.current_system == "S01T5"
    assert len(get_items_by_description(manager, "hlv call")) == 1

    # The HLV is transferring the crew back to the HLV
    # 15 minutes transfer + 45 minutes to finish the repair
    timeout += (13 + 45) / 60
    env.run(timeout)
    assert hlv.at_port is hlv.enroute is False
    assert hlv.at_system is hlv.at_site is hlv.onsite is hlv.transferring_crew is True
    assert hlv.current_system == "S01T5"
    assert len(get_items_by_description(manager, "hlv call")) == 1

    # Repair 12 S01T6 runs without delays

    # The HLV is transferring the crew to the turbine
    timeout += 15 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.enroute is False
    assert hlv.at_system is hlv.at_site is hlv.onsite is hlv.transferring_crew is True
    assert hlv.current_system == "S01T6"
    assert len(get_items_by_description(manager, "hlv call")) == 0

    # The crew is transferring back to the HLV
    timeout += 3 + 15 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.enroute is False
    assert hlv.at_system is hlv.at_site is hlv.onsite is hlv.transferring_crew is True
    assert hlv.current_system == "S01T6"
    assert len(get_items_by_description(manager, "hlv call")) == 0

    # Repair 13 S01T4 will finish without delalys

    # The HLV is back at port because there are no more requests.
    timeout += 2 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.onsite is True
    assert hlv.at_system is hlv.at_site is hlv.transferring_crew is hlv.enroute is False
    assert hlv.current_system is None
    assert len(get_items_by_description(manager, "hlv call")) == 0

    # The HLV will have no more repairs for the remainder of its visit, so check the
    # start of each remaining shift

    # Move to 8:01 AM the following day
    timeout += 19 + 30 / 60
    env.run(timeout)
    assert hlv.at_port is hlv.onsite is True
    assert hlv.at_system is hlv.onsite is hlv.transferring_crew is hlv.enroute is False
    assert hlv.current_system is None
    assert len(get_items_by_description(manager, "hlv call")) == 0

    # The HLV will remain at port
    timeout += 24
    env.run(timeout)
    assert hlv.at_port is hlv.onsite is True
    assert hlv.at_system is hlv.at_site is hlv.transferring_crew is hlv.enroute is False
    assert hlv.current_system is None
    assert len(get_items_by_description(manager, "hlv call")) == 0

    # The HLV will remain at port
    timeout += 24
    env.run(timeout)
    assert hlv.at_port is hlv.onsite is True
    assert hlv.at_system is hlv.at_site is hlv.transferring_crew is hlv.enroute is False
    assert hlv.current_system is None
    assert len(get_items_by_description(manager, "hlv call")) == 0

    # The HLV will remain at port
    timeout += 24
    env.run(timeout)
    assert hlv.at_port is hlv.onsite is True
    assert hlv.at_system is hlv.at_site is hlv.transferring_crew is hlv.enroute is False
    assert hlv.current_system is None
    assert len(get_items_by_description(manager, "hlv call")) == 0

    # The HLV will no longer be available and is off to another charter
    # It is enroute because it's technically be chartered, but not
    timeout += 24
    env.run(timeout)
    assert (
        hlv.at_system
        is hlv.onsite
        is hlv.transferring_crew
        is hlv.at_site
        is hlv.at_port
        is hlv.enroute
        is False
    )
    assert hlv.current_system is None
    assert len(get_items_by_description(manager, "hlv call")) == 0

    # Field Support Vessel Checks
    # This is going to only check in on the first day and last day of processes
    # due to the extensive timing checks performed above for the HLV, which cover
    # numerous edge cases for repair timing, and so we mostly care about the continuous
    # shift timing checks. In addition because the FSV only has a waveheight
    # restriction, the only weather delays should occur on 6/28 when there are
    # waves >= 1.5m.

    # The FSV be traveling still
    assert fsv.enroute
    assert (
        fsv.at_port
        is fsv.at_site
        is fsv.at_system
        is fsv.onsite
        is fsv.transferring_crew
        is False
    )
    assert fsv.current_system is None

    # This is not 30 due to delays in the timing caused by repairs when the maintenance
    # timeout clock is not ticking
    assert len(get_items_by_description(manager, "fsv call")) == 30

    # The FSV should not have arrived on 5/31 at 23:59
    timeout += 16 * 24 - 8 - 2 / 60
    env.run(timeout)
    assert fsv.enroute
    assert (
        fsv.at_port
        is fsv.at_site
        is fsv.at_system
        is fsv.onsite
        is fsv.transferring_crew
        is False
    )
    assert fsv.current_system is None
    # More of the requests should have been submitted at this stage
    assert len(get_items_by_description(manager, "fsv call")) == 33

    # The FSV should now be on site for its first request and tranferring the crew
    timeout += 2 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.onsite is fsv.transferring_crew is fsv.at_site is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S00T1"
    assert len(get_items_by_description(manager, "fsv call")) == 33

    # The FSV should be starting the repair after its 30 minute transfer
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is True
    assert fsv.at_port is fsv.enroute is fsv.transferring_crew is False
    assert fsv.current_system == "S00T1"
    assert len(get_items_by_description(manager, "fsv call")) == 33

    # Tranferring the crew back to the FSV
    timeout += 2
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S00T1"
    # Normally, this number will be decremented, but another request was submitted
    assert len(get_items_by_description(manager, "fsv call")) == 33

    # Transferring crew to the next turbine
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S00T2"
    assert len(get_items_by_description(manager, "fsv call")) == 32

    # Maintenance
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is True
    assert fsv.at_port is fsv.enroute is fsv.transferring_crew is False
    assert fsv.current_system == "S00T2"
    assert len(get_items_by_description(manager, "fsv call")) == 32

    # Tranferring the crew back to the FSV
    timeout += 2
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S00T2"
    assert len(get_items_by_description(manager, "fsv call")) == 32

    # Transferring crew to the next turbine
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S00T3"
    assert len(get_items_by_description(manager, "fsv call")) == 31

    # Maintenance
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is True
    assert fsv.at_port is fsv.enroute is fsv.transferring_crew is False
    assert fsv.current_system == "S00T3"
    assert len(get_items_by_description(manager, "fsv call")) == 31

    # Tranferring the crew back to the FSV
    timeout += 2
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S00T3"
    assert len(get_items_by_description(manager, "fsv call")) == 31

    # Transferring crew to the next turbine
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S01T4"
    assert len(get_items_by_description(manager, "fsv call")) == 30

    # Maintenance
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is True
    assert fsv.at_port is fsv.enroute is fsv.transferring_crew is False
    assert fsv.current_system == "S01T4"
    assert len(get_items_by_description(manager, "fsv call")) == 30

    # Tranferring the crew back to the FSV
    timeout += 2
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S01T4"
    assert len(get_items_by_description(manager, "fsv call")) == 30

    # Transferring crew to the next turbine
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S01T5"
    assert len(get_items_by_description(manager, "fsv call")) == 29

    # Maintenance
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is True
    assert fsv.at_port is fsv.enroute is fsv.transferring_crew is False
    assert fsv.current_system == "S01T5"
    assert len(get_items_by_description(manager, "fsv call")) == 29

    # Tranferring the crew back to the FSV
    timeout += 2
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S01T5"
    assert len(get_items_by_description(manager, "fsv call")) == 29

    # Transferring crew to the next turbine
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S01T6"
    assert len(get_items_by_description(manager, "fsv call")) == 28

    # Maintenance
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is True
    assert fsv.at_port is fsv.enroute is fsv.transferring_crew is False
    assert fsv.current_system == "S01T6"
    assert len(get_items_by_description(manager, "fsv call")) == 28

    # Tranferring the crew back to the FSV
    timeout += 2
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S01T6"
    assert len(get_items_by_description(manager, "fsv call")) == 28

    # Transferring crew to the next turbine
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S00T1"
    assert len(get_items_by_description(manager, "fsv call")) == 27

    # Maintenance
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is True
    assert fsv.at_port is fsv.enroute is fsv.transferring_crew is False
    assert fsv.current_system == "S00T1"
    assert len(get_items_by_description(manager, "fsv call")) == 27

    # Tranferring the crew back to the FSV
    timeout += 2
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S00T1"
    assert len(get_items_by_description(manager, "fsv call")) == 27

    # Transferring crew to the next turbine
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S00T2"
    assert len(get_items_by_description(manager, "fsv call")) == 26

    # Maintenance
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is True
    assert fsv.at_port is fsv.enroute is fsv.transferring_crew is False
    assert fsv.current_system == "S00T2"
    assert len(get_items_by_description(manager, "fsv call")) == 26

    # Tranferring the crew back to the FSV
    timeout += 2
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S00T2"
    assert len(get_items_by_description(manager, "fsv call")) == 26

    # Transferring crew to the next turbine
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S00T3"
    # The final 3 requests will have been submitted, now that they've caught up with
    # their delays
    assert len(get_items_by_description(manager, "fsv call")) == 27

    # Maintenance
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is True
    assert fsv.at_port is fsv.enroute is fsv.transferring_crew is False
    assert fsv.current_system == "S00T3"
    assert len(get_items_by_description(manager, "fsv call")) == 27

    # Tranferring the crew back to the FSV
    timeout += 2
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S00T3"
    assert len(get_items_by_description(manager, "fsv call")) == 27

    # Transferring crew to the next turbine
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S01T4"
    assert len(get_items_by_description(manager, "fsv call")) == 26

    # Maintenance
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is True
    assert fsv.at_port is fsv.enroute is fsv.transferring_crew is False
    assert fsv.current_system == "S01T4"
    assert len(get_items_by_description(manager, "fsv call")) == 26

    # Tranferring the crew back to the FSV
    timeout += 2
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S01T4"
    assert len(get_items_by_description(manager, "fsv call")) == 26

    # Transferring crew to the next turbine
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S01T5"
    assert len(get_items_by_description(manager, "fsv call")) == 25

    # Maintenance
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is True
    assert fsv.at_port is fsv.enroute is fsv.transferring_crew is False
    assert fsv.current_system == "S01T5"
    assert len(get_items_by_description(manager, "fsv call")) == 25

    # Tranferring the crew back to the FSV
    timeout += 2
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S01T5"
    assert len(get_items_by_description(manager, "fsv call")) == 25

    # Transferring crew to the next turbine
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S01T6"
    assert len(get_items_by_description(manager, "fsv call")) == 24

    # Maintenance
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is True
    assert fsv.at_port is fsv.enroute is fsv.transferring_crew is False
    assert fsv.current_system == "S01T6"
    assert len(get_items_by_description(manager, "fsv call")) == 24

    # Tranferring the crew back to the FSV
    timeout += 2
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S01T6"
    assert len(get_items_by_description(manager, "fsv call")) == 24

    # Transferring crew to the next turbine
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S00T1"
    assert len(get_items_by_description(manager, "fsv call")) == 23

    # Maintenance
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is True
    assert fsv.at_port is fsv.enroute is fsv.transferring_crew is False
    assert fsv.current_system == "S00T1"
    assert len(get_items_by_description(manager, "fsv call")) == 23

    # Tranferring the crew back to the FSV
    timeout += 2
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S00T1"
    assert len(get_items_by_description(manager, "fsv call")) == 23

    # Transferring crew to the next turbine
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S00T2"
    assert len(get_items_by_description(manager, "fsv call")) == 22

    # Maintenance
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is True
    assert fsv.at_port is fsv.enroute is fsv.transferring_crew is False
    assert fsv.current_system == "S00T2"
    assert len(get_items_by_description(manager, "fsv call")) == 22

    # Tranferring the crew back to the FSV
    timeout += 2
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S00T2"
    assert len(get_items_by_description(manager, "fsv call")) == 22

    # Transferring crew to the next turbine
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S00T3"
    assert len(get_items_by_description(manager, "fsv call")) == 21

    # Maintenance
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is True
    assert fsv.at_port is fsv.enroute is fsv.transferring_crew is False
    assert fsv.current_system == "S00T3"
    assert len(get_items_by_description(manager, "fsv call")) == 21

    # Tranferring the crew back to the FSV
    timeout += 2
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S00T3"
    assert len(get_items_by_description(manager, "fsv call")) == 21

    # Transferring crew to the next turbine
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S01T4"
    assert len(get_items_by_description(manager, "fsv call")) == 20

    # Maintenance
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is True
    assert fsv.at_port is fsv.enroute is fsv.transferring_crew is False
    assert fsv.current_system == "S01T4"
    assert len(get_items_by_description(manager, "fsv call")) == 20

    # Tranferring the crew back to the FSV
    timeout += 2
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S01T4"
    assert len(get_items_by_description(manager, "fsv call")) == 20

    # Transferring crew to the next turbine
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S01T5"
    assert len(get_items_by_description(manager, "fsv call")) == 19

    # Maintenance
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is True
    assert fsv.at_port is fsv.enroute is fsv.transferring_crew is False
    assert fsv.current_system == "S01T5"
    assert len(get_items_by_description(manager, "fsv call")) == 19

    # Tranferring the crew back to the FSV
    timeout += 2
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S01T5"
    assert len(get_items_by_description(manager, "fsv call")) == 19

    # Transferring crew to the next turbine
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is fsv.transferring_crew is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S01T6"
    assert len(get_items_by_description(manager, "fsv call")) == 18

    # Maintenance
    timeout += 30 / 60
    env.run(timeout)
    assert fsv.at_system is fsv.at_site is fsv.onsite is True
    assert fsv.at_port is fsv.enroute is fsv.transferring_crew is False
    assert fsv.current_system == "S01T6"
    assert len(get_items_by_description(manager, "fsv call")) == 18

    # Now that the timing continued overnight, let's ensure the last day looks good too
    timeout += 25 * 24 - (3 + 30 / 60)
    env.run(timeout)
    assert fsv.at_port is fsv.onsite is True
    assert fsv.at_system is fsv.at_site is fsv.enroute is fsv.transferring_crew is False
    assert fsv.current_system is None
    assert len(get_items_by_description(manager, "fsv call")) == 0

    # Now check that the FSV has left
    timeout += 24
    env.run(timeout)
    assert (
        fsv.at_system
        is fsv.at_site
        is fsv.at_port
        is fsv.enroute
        is fsv.transferring_crew
        is fsv.onsite
        is False
    )
    assert fsv.current_system is None
    assert len(get_items_by_description(manager, "fsv call")) == 0


def test_unscheduled_service_equipment_call(env_setup_full_profile):
    """Tests the calling of downtime-based and requests-based service equipment. This
    test will only consider that the equipment is mobilized when required, arrives at
    site, and leaves as the previous tests will cover repair logics.
    """
    env = env_setup_full_profile
    manager = RepairManager(env)
    windfarm = Windfarm(env, "layout_simulation.csv", manager)

    # ctv = ServiceEquipment(env, windfarm, manager, "ctv.yaml")
    fsv = ServiceEquipment(env, windfarm, manager, "fsv_requests.yaml")
    fsv.finish_setup_with_environment_variables()
    hlv = ServiceEquipment(env, windfarm, manager, "hlv_downtime.yaml")
    hlv.finish_setup_with_environment_variables()

    # Start the simulation to ensure everything is in place as required
    env.run(1)

    # for system in windfarm.system_list:
    #     system = windfarm.system(system)
    #     for pname, p in system.subassemblies[0].processes.items():
    #         if not isinstance(pname, tuple):
    #             continue
    #         if pname[1] != "catastrophic failure":
    #             continue
    #         print(system.id, pname, f"{p._target._delay:,.10f}")

    assert (
        fsv.at_port
        is fsv.at_site
        is fsv.at_system
        is fsv.transferring_crew
        is fsv.enroute
        is fsv.onsite
        is False
    )
    assert (
        hlv.at_port
        is hlv.at_site
        is hlv.at_system
        is hlv.transferring_crew
        is hlv.enroute
        is hlv.onsite
        is False
    )

    # The FSV has a 10 request threshold, with a request being submitted simultaneously
    # by each turbine every 25 days, so at day 50, the FSV will be mobilized
    timeout = 50 * 24 + 1 / 60
    env.run(timeout)
    assert fsv.enroute
    assert (
        fsv.transferring_crew
        is fsv.at_system
        is fsv.at_site
        is fsv.onsite
        is fsv.at_port
        is False
    )
    assert fsv.current_system is None

    # After 21 days the FSV will arrive at site, but have to wait until the start of the
    # work shift before repairs can happen
    timeout += 21 * 24
    env.run(timeout)
    assert fsv.at_port is fsv.onsite is True
    assert fsv.transferring_crew is fsv.at_system is fsv.at_site is fsv.enroute is False
    assert fsv.current_system is None

    # Check that the FSV is working at the start of the work shift
    timeout += 9
    env.run(timeout)
    assert fsv.onsite is fsv.at_site is fsv.transferring_crew is fsv.at_system is True
    assert fsv.at_port is fsv.enroute is False
    assert fsv.current_system == "S00T1"

    # The FSV should no longer be on site after 28 days, plus the time to finish the
    # repair that it's started
    timeout += 28 * 24 + 2 + 1  # 28 days + 2 hour repair + 1 hour in crew transfer
    env.run(timeout)
    assert (
        fsv.transferring_crew
        is fsv.at_system
        is fsv.onsite
        is fsv.at_site
        is fsv.at_port
        is fsv.enroute
        is False
    )
    assert fsv.current_system is None

    # The first HLV call is at 14012.2435309168 hours when S00T2's generator has a
    # catastrophic failure putting the windfarm at 83.3% operations, which is less than
    # the 90% threshold. However, because the timing will be delayed during repairs,
    # realized timeout will be at 14185.753253
    timeout = 14185.753253
    env.run(timeout + 1)
    # For checking in on the actual timing when it eventually changes
    # df = env.load_events_log_dataframe()
    # print(
    #     df.loc[
    #         df.agent == "Heavy Lift Vessel",
    #         [
    #             "env_datetime", "env_time", "agent", "action", "reason", "additional",
    #             "duration", "system_id", "part_id", "request_id"
    #         ]
    #     ].tail(20).to_string()
    # )
    assert hlv.enroute
    assert (
        hlv.transferring_crew
        is hlv.at_system
        is hlv.at_site
        is hlv.onsite
        is hlv.at_port
        is False
    )
    assert hlv.current_system is None

    # Test that the HLV was successfully mobilized
    timeout += 60 * 24
    env.run(timeout + 1 / 60)
    assert hlv.transferring_crew is hlv.at_site is hlv.at_system is hlv.onsite is True
    assert hlv.enroute is hlv.at_port is False
    assert hlv.current_system == "S00T2"

    # Ensure it's still here at the end, accounting for completing its last request
    timeout += 30 * 24 + 10 + 15 / 60
    env.run(timeout)
    assert hlv.onsite is hlv.at_site is hlv.at_system is True
    assert hlv.transferring_crew is hlv.at_port is hlv.enroute is False
    assert hlv.current_system == "S00T2"

    # Check that it's gone
    timeout += 24 + 15 / 60
    env.run(timeout + 24)
    assert hlv.onsite is False
