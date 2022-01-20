"""Test the ServiceEquipment class."""

import numpy as np
import numpy.testing as npt

from wombat.windfarm import Windfarm
from wombat.core.library import load_yaml
from wombat.core.repair_management import StrategyMap, EquipmentMap, RepairManager
from wombat.core.service_equipment import ServiceEquipment, consecutive_groups

from tests.conftest import env_setup


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
