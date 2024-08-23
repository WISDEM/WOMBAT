"""Tests wombat/core/repair_management.py."""

import numpy as np
import pytest

from wombat.core.library import load_yaml
from wombat.core.data_classes import (
    VALID_EQUIPMENT,
    StrategyMap,
    EquipmentMap,
    RepairRequest,
    ServiceEquipmentData,
)
from wombat.windfarm.windfarm import Windfarm
from wombat.core.repair_management import RepairManager
from wombat.core.service_equipment import ServiceEquipment
from wombat.windfarm.system.system import System
from wombat.windfarm.system.subassembly import Subassembly

from tests.conftest import VESTAS_V90, GENERATOR_SUBASSEMBLY


def test_repair_manager_init(env_setup):
    """Test the initialization steps."""
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
    fsv = load_yaml(env.data_dir / "vessels", "fsv_downtime.yaml")
    fsv_data = ServiceEquipmentData(fsv).determine_type()
    for capability in fsv_data.capability:
        mapping.update(capability, fsv_data.strategy_threshold, fsv_data)
    fsv = ServiceEquipment(env, windfarm, manager, "fsv_downtime.yaml")
    fsv_map = EquipmentMap(fsv.settings.strategy_threshold, fsv)
    assert manager.request_based_equipment == StrategyMap()
    assert manager.downtime_based_equipment.SCN == [fsv_map]
    assert manager.downtime_based_equipment.MCN == []
    assert manager.downtime_based_equipment.LCN == []
    assert manager.downtime_based_equipment.DRN == []
    assert manager.downtime_based_equipment.CTV == []
    assert manager.downtime_based_equipment.CAB == []
    assert manager.downtime_based_equipment.RMT == []
    assert manager.downtime_based_equipment.DSV == []
    assert manager.downtime_based_equipment.TOW == []
    assert manager.downtime_based_equipment.AHV == []
    assert manager.downtime_based_equipment.VSG == []

    # Add a requests-based piece of equipment
    fsv = ServiceEquipment(env, windfarm, manager, "fsv_requests.yaml")
    fsv_map = EquipmentMap(fsv.settings.strategy_threshold, fsv)
    assert manager.request_based_equipment.SCN == [fsv_map]
    assert manager.downtime_based_equipment.MCN == []
    assert manager.downtime_based_equipment.LCN == []
    assert manager.downtime_based_equipment.DRN == []
    assert manager.downtime_based_equipment.CTV == []
    assert manager.downtime_based_equipment.CAB == []
    assert manager.downtime_based_equipment.RMT == []
    assert manager.downtime_based_equipment.DSV == []
    assert manager.downtime_based_equipment.TOW == []
    assert manager.downtime_based_equipment.AHV == []
    assert manager.downtime_based_equipment.VSG == []

    # Check for registering of equipment with multiple capabilities
    hlv = ServiceEquipment(env, windfarm, manager, "hlv_requests_multi_capability.yaml")
    hlv_map = EquipmentMap(hlv.settings.strategy_threshold, hlv)
    assert manager.request_based_equipment.SCN == [fsv_map, hlv_map]
    assert manager.request_based_equipment.LCN == [hlv_map]
    assert manager.downtime_based_equipment.MCN == []
    assert manager.downtime_based_equipment.DRN == []
    assert manager.downtime_based_equipment.CTV == []
    assert manager.downtime_based_equipment.CAB == []
    assert manager.downtime_based_equipment.RMT == []
    assert manager.downtime_based_equipment.DSV == []
    assert manager.downtime_based_equipment.TOW == []
    assert manager.downtime_based_equipment.AHV == []
    assert manager.downtime_based_equipment.VSG == []

    # Ensure that a scheduled equipment type will raise an error, though this should
    # never be called within the code
    ctv = ServiceEquipment(env, windfarm, manager, "ctv_quick_load.yaml")
    with pytest.raises(ValueError):
        manager._register_equipment(ctv)


def test_register_request_and_submit_request_and_get_request(env_setup):
    """Tests the ``RepairManager.submit_request`` method."""
    # Set up all the required infrastructure
    env = env_setup
    manager = RepairManager(env)
    turbine = System(env, manager, "WTG001", "Vestas V90 001", VESTAS_V90, "turbine")
    generator = Subassembly(turbine, env, "GEN001", GENERATOR_SUBASSEMBLY)

    # Test that a submitted failure is registered correctly
    manual_reset = generator.data.failures[0]
    repair_request_failure = RepairRequest(
        turbine.id,
        turbine.name,
        generator.id,
        generator.name,
        manual_reset.level,
        manual_reset,
    )
    repair_request_failure = turbine.repair_manager.register_request(
        repair_request_failure
    )
    assert manager._current_id == 1
    assert repair_request_failure.request_id == "RPR00000000"

    turbine.repair_manager.submit_request(repair_request_failure)
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
    repair_request_maintenance = turbine.repair_manager.register_request(
        repair_request_maintenance
    )
    assert manager._current_id == 2
    assert repair_request_maintenance.request_id == "MNT00000001"

    turbine.repair_manager.submit_request(repair_request_maintenance)
    assert len(manager.items) == 2
    assert not manager.downtime_based_equipment.is_running
    assert not manager.request_based_equipment.is_running

    # Get the failure request back to ensure it's still the same object
    # NOTE: this will also purge it from the system
    retrieved_request = manager.get_request_by_system(["CTV"], system_id=turbine.id)
    assert retrieved_request.value == repair_request_failure

    retrieved_request = manager.get_request_by_system(["CTV"], system_id=turbine.id)
    assert retrieved_request.value is repair_request_maintenance

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
    for repair in generator.data.maintenance + generator.data.failures:
        repair_request = RepairRequest(
            turbine.id, turbine.name, generator.id, generator.name, repair.level, repair
        )
        repair_request = turbine.repair_manager.register_request(repair_request)
        turbine.repair_manager.submit_request(repair_request)

    assert manager.request_map == correct_request_map


def test_get_requests(env_setup):
    """Tests the ``RepairManager.get_request_by_system`` and
    ``RepairManager.get_request_by_severity`` methods.
    """
    # Set up all the required infrastructure
    env = env_setup
    manager = RepairManager(env)
    windfarm = Windfarm(env, "layout_single_subassembly.csv", manager)
    capability_list = VALID_EQUIPMENT
    turbine1 = windfarm.system("WTG001")
    turbine2 = windfarm.system("WTG002")
    turbine3 = windfarm.system("WTG003")
    generator1 = [el for el in turbine1.subassemblies if el.name == "generator"][0]
    generator2 = [el for el in turbine2.subassemblies if el.name == "generator"][0]
    generator3 = [el for el in turbine3.subassemblies if el.name == "generator"][0]

    # Ensure no requests are returned
    for _id in ("WTG001", "WTG002", "WTG003"):
        request = manager.get_request_by_system(capability_list, _id)
        assert request is None

    for severity in range(6):
        request = manager.get_request_by_severity(capability_list, severity)
        assert request is None

    # Submit a handful of requests with variables indicating what the request is and
    # which turbine it is originating from
    medium_repair1 = generator1.data.failures[2]
    medium_repair1 = RepairRequest(
        turbine1.id,
        turbine1.name,
        generator1.id,
        generator1.name,
        medium_repair1.level,
        medium_repair1,
    )
    medium_repair1 = turbine1.repair_manager.register_request(medium_repair1)
    turbine1.repair_manager.submit_request(medium_repair1)

    annual_reset2 = generator2.data.maintenance[0]
    annual_reset2 = RepairRequest(
        turbine2.id,
        turbine2.name,
        generator2.id,
        generator2.name,
        annual_reset2.level,
        annual_reset2,
    )
    annual_reset2 = turbine2.repair_manager.register_request(annual_reset2)
    turbine2.repair_manager.submit_request(annual_reset2)

    minor_repair2 = generator2.data.failures[1]
    minor_repair2 = RepairRequest(
        turbine2.id,
        turbine2.name,
        generator2.id,
        generator2.name,
        minor_repair2.level,
        minor_repair2,
    )
    minor_repair2 = turbine2.repair_manager.register_request(minor_repair2)
    turbine2.repair_manager.submit_request(minor_repair2)

    major_replacement3 = generator3.data.failures[4]
    major_replacement3 = RepairRequest(
        turbine3.id,
        turbine3.name,
        generator3.id,
        generator3.name,
        major_replacement3.level,
        major_replacement3,
    )
    major_replacement3 = turbine3.repair_manager.register_request(major_replacement3)
    turbine3.repair_manager.submit_request(major_replacement3)

    medium_repair3 = generator3.data.failures[2]
    medium_repair3 = RepairRequest(
        turbine3.id,
        turbine3.name,
        generator3.id,
        generator1.name,
        medium_repair3.level,
        medium_repair3,
    )
    medium_repair3 = turbine3.repair_manager.register_request(medium_repair3)
    turbine3.repair_manager.submit_request(medium_repair3)

    major_repair2 = generator2.data.failures[3]
    major_repair2 = RepairRequest(
        turbine2.id,
        turbine2.name,
        generator2.id,
        generator2.name,
        major_repair2.level,
        major_repair2,
    )
    major_repair2 = turbine2.repair_manager.register_request(major_repair2)
    turbine2.repair_manager.submit_request(major_repair2)

    # Basic checks that the submissions are all there
    assert len(manager.items) == 6

    # Test that the first submitted request is returned with no system filtering
    request = manager.get_request_by_system(capability_list)
    assert request.value == medium_repair1

    # Test that the major replacement is retrieved first when requesting by severity
    # level 5 and that other invalid requests return None
    request = manager.get_request_by_severity(capability_list, 6)
    assert request is None

    request = manager.get_request_by_severity(capability_list, 1)
    assert request is None

    request = manager.get_request_by_severity(capability_list, 5)
    assert request.value == major_replacement3

    # Ensure the next highest severity level request is (major repair 2) is picked
    request = manager.get_request_by_severity(capability_list)
    assert request.value == major_repair2

    # Test that the Turbine 2 requests are retrieved in order and with respect to
    # capability limitations and anything outside of the submitted capabilties is None
    manager.invalidate_system(turbine2)
    manager.interrupt_system(turbine2)
    request = manager.get_request_by_system(
        ["LCN", "CAB", "DSV", "RMT", "DRN"], turbine2.id
    )
    assert request is None
    manager.enable_requests_for_system(turbine2)

    request = manager.get_request_by_system(["CTV"], turbine2.id)
    manager.invalidate_system(turbine2)
    manager.interrupt_system(turbine2)
    assert request.value == annual_reset2
    manager.enable_requests_for_system(turbine2)

    request = manager.get_request_by_system(capability_list, turbine2.id)
    manager.invalidate_system(turbine2)
    manager.interrupt_system(turbine2)
    assert request.value == minor_repair2
    manager.enable_requests_for_system(turbine2)

    # With 1 requests left, ensure they're produced in the correct order and that
    # specifying incorrect severity levesls returns None
    request = manager.get_request_by_severity(capability_list, 4)
    assert request is None

    request = manager.get_request_by_severity(capability_list, 2)
    assert request is None

    # Activate turbine 3 and check that its medium repair is the next in the list
    manager.invalidate_system(turbine3)
    manager.interrupt_system(turbine3)
    manager.enable_requests_for_system(turbine3)
    request = manager.get_request_by_severity(capability_list, 3)
    assert request.value == medium_repair3


def test_purge_subassembly_requests(env_setup):
    """Tests the ``RepairManager.purge_subassembly_requests`` method."""
    env = env_setup
    manager = RepairManager(env)
    windfarm = Windfarm(env, "layout_with_gearboxes.csv", manager)
    turbine1 = windfarm.system("S00T1")
    turbine2 = windfarm.system("S00T2")
    turbine3 = windfarm.system("S00T3")
    generator1 = turbine1.generator
    gearbox1 = turbine1.gearbox
    generator2 = turbine2.generator
    gearbox2 = turbine2.gearbox
    generator3 = turbine3.generator
    gearbox3 = turbine3.gearbox

    # Check there are no submitted items
    assert manager.items == []

    # Check that None is returned when there are not items
    response = manager.purge_subassembly_requests(turbine1.id, generator1.id)
    assert response is None

    # Submit a handful of requests with variables indicating what the request is and
    # which turbine it is originating from
    gen_medium_repair1 = generator1.data.failures[2]
    gen_medium_repair1 = RepairRequest(
        turbine1.id,
        turbine1.name,
        generator1.id,
        generator1.name,
        gen_medium_repair1.level,
        gen_medium_repair1,
    )
    gen_medium_repair1 = turbine1.repair_manager.register_request(gen_medium_repair1)
    turbine1.repair_manager.submit_request(gen_medium_repair1)

    gen_major_repair1 = generator1.data.failures[3]
    gen_major_repair1 = RepairRequest(
        turbine1.id,
        turbine1.name,
        generator1.id,
        generator1.name,
        gen_major_repair1.level,
        gen_major_repair1,
    )
    gen_major_repair1 = turbine1.repair_manager.register_request(gen_major_repair1)
    turbine1.repair_manager.submit_request(gen_major_repair1)

    gbx_medium_repair1 = gearbox1.data.failures[2]
    gbx_medium_repair1 = RepairRequest(
        turbine1.id,
        turbine1.name,
        gearbox1.id,
        gearbox1.name,
        gbx_medium_repair1.level,
        gbx_medium_repair1,
    )
    gbx_medium_repair1 = turbine1.repair_manager.register_request(gbx_medium_repair1)
    turbine1.repair_manager.submit_request(gbx_medium_repair1)

    gen_medium_repair2 = generator2.data.failures[2]
    gen_medium_repair2 = RepairRequest(
        turbine2.id,
        turbine2.name,
        generator2.id,
        generator2.name,
        gen_medium_repair2.level,
        gen_medium_repair2,
    )
    gen_medium_repair2 = turbine2.repair_manager.register_request(gen_medium_repair2)
    turbine2.repair_manager.submit_request(gen_medium_repair2)

    gen_major_repair2 = generator2.data.failures[3]
    gen_major_repair2 = RepairRequest(
        turbine2.id,
        turbine2.name,
        generator2.id,
        generator2.name,
        gen_major_repair2.level,
        gen_major_repair2,
    )
    gen_major_repair2 = turbine2.repair_manager.register_request(gen_major_repair2)
    turbine2.repair_manager.submit_request(gen_major_repair2)

    gbx_medium_repair2 = gearbox2.data.failures[2]
    gbx_medium_repair2 = RepairRequest(
        turbine2.id,
        turbine2.name,
        gearbox2.id,
        gearbox2.name,
        gbx_medium_repair2.level,
        gbx_medium_repair2,
    )
    gbx_medium_repair2 = turbine2.repair_manager.register_request(gbx_medium_repair2)
    turbine2.repair_manager.submit_request(gbx_medium_repair2)

    gen_medium_repair3 = generator3.data.failures[2]
    gen_medium_repair3 = RepairRequest(
        turbine3.id,
        turbine3.name,
        generator3.id,
        generator3.name,
        gen_medium_repair3.level,
        gen_medium_repair3,
    )
    gen_medium_repair3 = turbine3.repair_manager.register_request(gen_medium_repair3)
    turbine3.repair_manager.submit_request(gen_medium_repair3)

    gen_major_repair3 = generator3.data.failures[3]
    gen_major_repair3 = RepairRequest(
        turbine3.id,
        turbine3.name,
        generator3.id,
        generator3.name,
        gen_major_repair3.level,
        gen_major_repair3,
    )
    gen_major_repair3 = turbine3.repair_manager.register_request(gen_major_repair3)
    turbine3.repair_manager.submit_request(gen_major_repair3)

    gbx_medium_repair3 = gearbox3.data.failures[2]
    gbx_medium_repair3 = RepairRequest(
        turbine3.id,
        turbine3.name,
        gearbox3.id,
        gearbox3.name,
        gbx_medium_repair3.level,
        gbx_medium_repair3,
    )
    gbx_medium_repair3 = turbine3.repair_manager.register_request(gbx_medium_repair3)
    turbine3.repair_manager.submit_request(gbx_medium_repair3)

    # Check for the correct number of submissions
    assert len(manager.items) == 9

    # Check that mismatched identifiers don't delete results
    for subassembly in turbine1.subassemblies:
        if subassembly.id in ("generator", "gearbox"):
            continue
        response = manager.purge_subassembly_requests(
            system_id=turbine1.id, subassembly_id=subassembly.id
        )
        assert response is None
        assert len(manager.items) == 9

    for subassembly in turbine2.subassemblies:
        if subassembly.id in ("generator", "gearbox"):
            continue
        response = manager.purge_subassembly_requests(
            system_id=turbine2.id, subassembly_id=subassembly.id
        )
        assert response is None
        assert len(manager.items) == 9

    for subassembly in turbine3.subassemblies:
        if subassembly.id in ("generator", "gearbox"):
            continue
        response = manager.purge_subassembly_requests(
            system_id=turbine3.id, subassembly_id=subassembly.id
        )
        assert response is None
        assert len(manager.items) == 9

    # Check that only the 1 gearbox submission for turbine 1 is all that is removed
    response = manager.purge_subassembly_requests(
        system_id=turbine1.id, subassembly_id=gearbox1.id
    )
    assert len(response) == 1
    assert response[0] == gbx_medium_repair1
    assert len(manager.items) == 8

    # Check that only the turbine 1 medium repair submission is removed
    response = manager.purge_subassembly_requests(
        system_id=turbine1.id,
        subassembly_id=generator1.id,
        exclude=[gen_major_repair1.request_id],
    )
    response = list(response)
    assert len(response) == 1
    assert response[0] == gen_medium_repair1
    assert len(manager.items) == 7

    # Check that only the turbine 1 major repair submission is removed
    response = manager.purge_subassembly_requests(
        system_id=turbine1.id, subassembly_id=generator1.id
    )
    response = list(response)
    assert len(response) == 1
    assert response[0] == gen_major_repair1
    assert len(manager.items) == 6

    # Check that the turbine 2 generator submissions are removed
    response = manager.purge_subassembly_requests(
        system_id=turbine2.id, subassembly_id=generator2.id
    )
    response = list(response)
    assert len(response) == 2
    assert response[0] == gen_medium_repair2
    assert response[1] == gen_major_repair2
    assert len(manager.items) == 4

    # Check that only the 1 gearbox submission for turbine 2 is all that is removed
    response = manager.purge_subassembly_requests(
        system_id=turbine2.id, subassembly_id=gearbox2.id
    )
    assert len(response) == 1
    assert response[0] == gbx_medium_repair2
    assert len(manager.items) == 3

    # Check that the turbine 3 generator submissions are removed
    response = manager.purge_subassembly_requests(
        system_id=turbine3.id, subassembly_id=generator3.id
    )
    response = list(response)
    assert len(response) == 2
    assert response[0] == gen_medium_repair3
    assert response[1] == gen_major_repair3
    assert len(manager.items) == 1

    # Check that only the 1 gearbox submission for turbine 3 is all that is removed
    response = manager.purge_subassembly_requests(
        system_id=turbine3.id, subassembly_id=gearbox3.id
    )
    assert len(response) == 1
    assert response[0] == gbx_medium_repair3
    assert len(manager.items) == 0
