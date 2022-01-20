"""Tests the Windfarm class."""

from re import M
from copy import deepcopy

import numpy as np
import pytest

from wombat.core import repair_management, service_equipment
from wombat.core.library import load_yaml
from wombat.core.data_classes import (
    Failure,
    Maintenance,
    RepairRequest,
    ServiceEquipmentData,
)
from wombat.windfarm.windfarm import Windfarm
from wombat.core.repair_management import StrategyMap, EquipmentMap, RepairManager
from wombat.core.service_equipment import ServiceEquipment
from wombat.windfarm.system.system import System
from wombat.windfarm.system.subassembly import Subassembly

from tests.conftest import VESTAS_V90, GENERATOR_SUBASSEMBLY, env_setup


def test_equipment_map(env_setup):
    """Tests the ``EquipmentMap`` class."""
    env = env_setup

    # Test for a requests-based equipment
    hlv = ServiceEquipmentData(
        load_yaml(env.data_dir / "repair" / "transport", "hlv_requests.yaml")
    ).determine_type()
    mapping = EquipmentMap(equipment=hlv, strategy_threshold=hlv.strategy_threshold)
    assert mapping.equipment == hlv
    assert mapping.strategy_threshold == hlv.strategy_threshold

    # Test for a downtime-based equipment
    hlv = ServiceEquipmentData(
        load_yaml(env.data_dir / "repair" / "transport", "hlv_downtime.yaml")
    ).determine_type()
    mapping = EquipmentMap(equipment=hlv, strategy_threshold=hlv.strategy_threshold)
    assert mapping.equipment == hlv
    assert mapping.strategy_threshold == hlv.strategy_threshold


def test_strategy_map(env_setup):
    """Tests the ``StrategyMap`` class."""
    env = env_setup

    mapping = StrategyMap()
    assert mapping.CTV == []
    assert mapping.SCN == []
    assert mapping.LCN == []
    assert mapping.CAB == []
    assert mapping.RMT == []
    assert mapping.DRN == []
    assert mapping.DSV == []
    assert not mapping.is_running

    # Test for bad input
    hlv = load_yaml(env.data_dir / "repair" / "transport", "hlv_requests.yaml")
    hlv["capability"] = ["SCN", "LCN"]
    hlv = ServiceEquipmentData(hlv).determine_type()
    hlv_map = EquipmentMap(hlv.strategy_threshold, hlv)

    # List of inputs will fail
    with pytest.raises(ValueError):
        mapping.update(hlv.capability, hlv.strategy_threshold, hlv)

    # Single bad input will fail
    with pytest.raises(ValueError):
        mapping.update("CRV", hlv.strategy_threshold, hlv)

    # Check that the update did not complete
    assert not mapping.is_running

    # Create a few equipment and test for inclusion

    # HLV with multiple capabilities
    hlv = load_yaml(env.data_dir / "repair" / "transport", "hlv_requests.yaml")
    hlv["capability"] = ["SCN", "LCN"]
    hlv = ServiceEquipmentData(hlv).determine_type()
    hlv_map = EquipmentMap(hlv.strategy_threshold, hlv)
    for capability in hlv.capability:
        mapping.update(capability, hlv.strategy_threshold, hlv)
    assert mapping.SCN == mapping.LCN == [hlv_map]
    assert mapping.is_running

    # Cabling vessel with one capability
    cab = load_yaml(env.data_dir / "repair" / "transport", "cabling.yaml")
    cab = ServiceEquipmentData(cab).determine_type()
    for capability in cab.capability:
        mapping.update(capability, cab.strategy_threshold, cab)
    cab_map = EquipmentMap(cab.strategy_threshold, cab)
    assert mapping.CAB == [cab_map]
    assert mapping.is_running

    # Diving support vessel
    dsv = load_yaml(env.data_dir / "repair" / "transport", "dsv.yaml")
    dsv = ServiceEquipmentData(dsv).determine_type()
    for capability in dsv.capability:
        mapping.update(capability, dsv.strategy_threshold, dsv)
    dsv_map = EquipmentMap(dsv.strategy_threshold, dsv)
    assert mapping.DSV == [dsv_map]
    assert mapping.is_running

    # Remote reset
    rmt = load_yaml(env.data_dir / "repair" / "transport", "rmt.yaml")
    rmt = ServiceEquipmentData(rmt).determine_type()
    for capability in rmt.capability:
        mapping.update(capability, rmt.strategy_threshold, rmt)
    rmt_map = EquipmentMap(rmt.strategy_threshold, rmt)
    assert mapping.RMT == [rmt_map]
    assert mapping.is_running

    # Drone repair
    drn = load_yaml(env.data_dir / "repair" / "transport", "drn.yaml")
    drn = ServiceEquipmentData(drn).determine_type()
    for capability in drn.capability:
        mapping.update(capability, drn.strategy_threshold, drn)
    drn_map = EquipmentMap(drn.strategy_threshold, drn)
    assert mapping.DRN == [drn_map]
    assert mapping.is_running

    # Crew tranfer
    ctv = load_yaml(env.data_dir / "repair" / "transport", "ctv.yaml")
    ctv = ServiceEquipmentData(ctv).determine_type()
    for capability in ctv.capability:
        mapping.update(capability, ctv.strategy_threshold, ctv)
    ctv_map = EquipmentMap(ctv.strategy_threshold, ctv)
    assert mapping.CTV == [ctv_map]
    assert mapping.is_running

    # Update an existing category with another piece of equipment
    fsv = load_yaml(env.data_dir / "repair" / "transport", "fsv_downtime.yaml")
    fsv = ServiceEquipmentData(fsv).determine_type()
    for capability in fsv.capability:
        mapping.update(capability, fsv.strategy_threshold, fsv)
    fsv_map = EquipmentMap(fsv.strategy_threshold, fsv)
    assert mapping.SCN == [hlv_map, fsv_map]
    assert mapping.is_running


def test_repair_manager_init(env_setup):
    env = env_setup

    # Test the capacity input
    manager = RepairManager(env, capacity=1)
    assert manager.env == env
    assert manager._current_id == 0
    assert manager.capacity == 1
    assert manager.items == []
    assert manager.downtime_based_equipment == StrategyMap()
    assert manager.request_based_equipment == StrategyMap()

    # Test the defaults
    manager = RepairManager(env)
    assert manager.env == env
    assert manager._current_id == 0
    assert manager.capacity == np.inf
    assert manager.items == []
    assert manager.downtime_based_equipment == StrategyMap()
    assert manager.request_based_equipment == StrategyMap()

    # Ensure the repair manager has an initialized windfarm registered
    windfarm = Windfarm(env, "layout.csv", manager)
    assert manager.windfarm == windfarm

    # Ensure the repair manager has an initialized equipment registered
    # NOTE: This will only check for the first addition, otherwise the StrategyMap tests
    # above, are sufficient

    # Add a downtime-based piece of equipment
    mapping = StrategyMap()
    fsv = load_yaml(env.data_dir / "repair" / "transport", "fsv_downtime.yaml")
    fsv_data = ServiceEquipmentData(fsv).determine_type()
    for capability in fsv_data.capability:
        mapping.update(capability, fsv_data.strategy_threshold, fsv_data)
    fsv = ServiceEquipment(env, windfarm, manager, "fsv_downtime.yaml")
    fsv_map = EquipmentMap(fsv.settings.strategy_threshold, fsv)
    assert manager.request_based_equipment == StrategyMap()
    assert manager.downtime_based_equipment.SCN == [fsv_map]
    assert manager.downtime_based_equipment.LCN == []
    assert manager.downtime_based_equipment.DRN == []
    assert manager.downtime_based_equipment.CTV == []
    assert manager.downtime_based_equipment.CAB == []
    assert manager.downtime_based_equipment.RMT == []
    assert manager.downtime_based_equipment.DSV == []

    # Add a requests-based piece of equipment
    fsv = ServiceEquipment(env, windfarm, manager, "fsv_requests.yaml")
    fsv_map = EquipmentMap(fsv.settings.strategy_threshold, fsv)
    assert manager.request_based_equipment.SCN == [fsv_map]
    assert manager.request_based_equipment.LCN == []
    assert manager.request_based_equipment.DRN == []
    assert manager.request_based_equipment.CTV == []
    assert manager.request_based_equipment.CAB == []
    assert manager.request_based_equipment.RMT == []
    assert manager.request_based_equipment.DSV == []

    # Check for registering of equipment with multiple capabilities
    hlv = ServiceEquipment(env, windfarm, manager, "hlv_requests_multi_capability.yaml")
    hlv_map = EquipmentMap(hlv.settings.strategy_threshold, hlv)
    assert manager.request_based_equipment.SCN == [fsv_map, hlv_map]
    assert manager.request_based_equipment.LCN == [hlv_map]
    assert manager.request_based_equipment.DRN == []
    assert manager.request_based_equipment.CTV == []
    assert manager.request_based_equipment.CAB == []
    assert manager.request_based_equipment.RMT == []
    assert manager.request_based_equipment.DSV == []

    # Ensure that a scheduled equipment type will raise an error, though this should
    # never be called within the code
    ctv = ServiceEquipment(env, windfarm, manager, "ctv_quick_load.yaml")
    with pytest.raises(ValueError):
        manager._register_equipment(ctv)


def test_submit_request_and_get_request(env_setup):
    """Tests the ``RepairManager.submit_request`` method."""

    # Set up all the required infrastructure
    env = env_setup
    manager = RepairManager(env)
    turbine = System(env, manager, "WTG001", "Vestas V90 001", VESTAS_V90, "turbine")
    generator = Subassembly(turbine, env, "GEN001", GENERATOR_SUBASSEMBLY)

    # Test that a submitted failure is registered correctly
    manual_reset = generator.data.failures[1]
    repair_request_failure = RepairRequest(
        turbine.id,
        turbine.name,
        generator.id,
        generator.name,
        manual_reset.level,
        manual_reset,
    )
    repair_request_failure = turbine.repair_manager.submit_request(
        repair_request_failure
    )
    assert manager._current_id == 1
    assert repair_request_failure.request_id == "RPR00000000"
    assert len(manager.items) == 1
    assert not manager.downtime_based_equipment.is_running
    assert not manager.request_based_equipment.is_running

    # Test that a submitted maintenance task is registered correctly
    annual_service = generator.data.maintenance[0]
    repair_request_maintenance = RepairRequest(
        turbine.id,
        turbine.name,
        generator.id,
        generator.name,
        annual_service.level,
        annual_service,
    )
    repair_request_maintenance = turbine.repair_manager.submit_request(
        repair_request_maintenance
    )
    assert manager._current_id == 2
    assert repair_request_maintenance.request_id == "MNT00000001"
    assert len(manager.items) == 2
    assert not manager.downtime_based_equipment.is_running
    assert not manager.request_based_equipment.is_running

    # Get the failure request back to ensure it's still the same object
    # NOTE: this will also purge it from the system
    retrieved_request = manager.get_request_by_system(["CTV"], system_id=turbine.id)
    assert retrieved_request.value == repair_request_failure

    # Get the request back to ensure it's the same object
    retrieved_request = manager.get_request_by_system(["CTV"], system_id=turbine.id)
    assert retrieved_request.value == repair_request_maintenance

    assert manager.items == []


def test_request_map(env_setup):
    """Tests the ``RepairManager.request_map`` property."""

    # Set up all the required infrastructure
    env = env_setup
    manager = RepairManager(env)
    turbine = System(env, manager, "WTG001", "Vestas V90 001", VESTAS_V90, "turbine")
    generator = Subassembly(turbine, env, "GEN001", GENERATOR_SUBASSEMBLY)

    # Test nothing is created yet
    assert manager.request_map == {}

    # Test that all the failures and maintenance tasks are registered correctly
    correct_request_map = {"CTV": 4, "SCN": 1, "LCN": 1}
    # Submit a request for all of the maintenance and failure objects in the generator
    for repair in generator.data.maintenance + [*generator.data.failures.values()]:
        repair_request = RepairRequest(
            turbine.id, turbine.name, generator.id, generator.name, repair.level, repair
        )
        repair_request = turbine.repair_manager.submit_request(repair_request)

    assert manager.request_map == correct_request_map


def test_get_requests(env_setup):
    """Tests the ``RepairManager.get_request_by_system`` and
    ``RepairManager.get_next_highest_severity_request`` methods.
    """

    # Set up all the required infrastructure
    env = env_setup
    manager = RepairManager(env)
    capability_list = ["SCN", "LCN", "CTV", "CAB", "DSV", "RMT", "DRN"]
    turbine1 = System(env, manager, "WTG001", "Vestas V90 001", VESTAS_V90, "turbine")
    turbine2 = System(env, manager, "WTG002", "Vestas V90 002", VESTAS_V90, "turbine")
    turbine3 = System(env, manager, "WTG003", "Vestas V90 003", VESTAS_V90, "turbine")
    generator1 = Subassembly(turbine1, env, "GEN001", GENERATOR_SUBASSEMBLY)
    generator2 = Subassembly(turbine2, env, "GEN002", GENERATOR_SUBASSEMBLY)
    generator3 = Subassembly(turbine3, env, "GEN003", GENERATOR_SUBASSEMBLY)

    # Ensure no requests are returned
    for _id in ("WTG001", "WTG002", "WTG003"):
        request = manager.get_request_by_system(capability_list, _id)
        assert request is None

    for severity in range(6):
        request = manager.get_next_highest_severity_request(capability_list, severity)
        assert request is None

    # Submit a handful of requests with variables indicating what the request is and
    # which turbine it is originating from
    medium_repair1 = generator1.data.failures[3]
    medium_repair1 = RepairRequest(
        turbine1.id,
        turbine1.name,
        generator1.id,
        generator1.name,
        medium_repair1.level,
        medium_repair1,
    )
    medium_repair1 = turbine1.repair_manager.submit_request(medium_repair1)

    annual_reset2 = generator2.data.maintenance[0]
    annual_reset2 = RepairRequest(
        turbine2.id,
        turbine2.name,
        generator2.id,
        generator2.name,
        annual_reset2.level,
        annual_reset2,
    )
    annual_reset2 = turbine2.repair_manager.submit_request(annual_reset2)

    minor_repair2 = generator2.data.failures[2]
    minor_repair2 = RepairRequest(
        turbine2.id,
        turbine2.name,
        generator2.id,
        generator2.name,
        minor_repair2.level,
        minor_repair2,
    )
    minor_repair2 = turbine2.repair_manager.submit_request(minor_repair2)

    major_replacement3 = generator3.data.failures[5]
    major_replacement3 = RepairRequest(
        turbine3.id,
        turbine3.name,
        generator3.id,
        generator3.name,
        major_replacement3.level,
        major_replacement3,
    )
    major_replacement3 = turbine3.repair_manager.submit_request(major_replacement3)

    medium_repair3 = generator3.data.failures[3]
    medium_repair3 = RepairRequest(
        turbine3.id,
        turbine3.name,
        generator3.id,
        generator1.name,
        medium_repair3.level,
        medium_repair3,
    )
    medium_repair3 = turbine3.repair_manager.submit_request(medium_repair3)

    major_repair2 = generator2.data.failures[4]
    major_repair2 = RepairRequest(
        turbine2.id,
        turbine2.name,
        generator2.id,
        generator2.name,
        major_repair2.level,
        major_repair2,
    )
    major_repair2 = turbine2.repair_manager.submit_request(major_repair2)

    # Basic checks that the submissions are all there
    assert len(manager.items) == 6

    # Test that the major replacement is retrieved first when requesting by severity
    # level 5 and that other invalid requests return None
    request = manager.get_next_highest_severity_request(capability_list, 6)
    assert request is None

    request = manager.get_next_highest_severity_request(capability_list, 1)
    assert request is None

    request = manager.get_next_highest_severity_request(capability_list, 5)
    assert request.value == major_replacement3

    # Ensure the next highest severity level request is (major repair 2) is picked
    request = manager.get_next_highest_severity_request(capability_list)
    assert request.value == major_repair2

    # Test that the Turbine 2 requests are retrieved in order and with respect to
    # capability limitations and anything outside of the submitted capabilties is None
    request = manager.get_request_by_system(
        ["LCN", "CAB", "DSV", "RMT", "DRN"], turbine2.id
    )
    assert request is None

    request = manager.get_request_by_system(["CTV"], turbine2.id)
    assert request.value == annual_reset2

    request = manager.get_request_by_system(capability_list, turbine2.id)
    assert request.value == minor_repair2

    # With 2 requests left, ensure they're produced in the correct order and that
    # specifying incorrect severity levesls returns None
    request = manager.get_next_highest_severity_request(capability_list)
    assert request.value == medium_repair1

    request = manager.get_next_highest_severity_request(capability_list, 4)
    assert request is None

    request = manager.get_next_highest_severity_request(capability_list, 2)
    assert request is None

    request = manager.get_next_highest_severity_request(capability_list, 3)
    assert request.value == medium_repair3
