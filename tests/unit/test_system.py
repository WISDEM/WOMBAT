"""Provides unit tests for the `System` class."""

import numpy as np
import pytest
import numpy.testing as npt

from wombat.core import RepairManager
from wombat.windfarm.system import System

from tests.conftest import (
    SUBSTATION,
    VESTAS_V90,
    ELECTROLYZER_POLY,
    VESTAS_POWER_CURVE,
    ELECTROLYZER_LINEAR,
    VESTAS_V90_1_SUBASSEMBLY,
    VESTAS_V90_NO_SUBASSEMBLY,
)


def test_turbine_initialization_complete_setup(env_setup):
    """Tests a complete turbine setup."""
    ENV = env_setup
    MANAGER = RepairManager(ENV)
    turbine_id = "WTG001"
    turbine_name = "Vestas V90 - 001"

    system = System(
        env=ENV,
        repair_manager=MANAGER,
        t_id=turbine_id,
        name=turbine_name,
        subassemblies=VESTAS_V90,
        system="turbine",
    )
    assert system.env == ENV
    assert system.repair_manager == MANAGER
    assert system.id == turbine_id
    assert system.name == turbine_name
    assert system.capacity == VESTAS_V90["capacity_kw"]
    assert system.value == VESTAS_V90["capacity_kw"] * VESTAS_V90["capex_kw"]
    subassemblies = [
        "electrical_system",
        "electronic_control",
        "sensors",
        "hydraulic_system",
        "yaw_system",
        "rotor_blades",
        "mechanical_brake",
        "rotor_hub",
        "gearbox",
        "generator",
        "supporting_structure",
        "drive_train",
    ]
    for subassembly in subassemblies:
        assert hasattr(system, subassembly)
    assert [el.name for el in system.subassemblies] == subassemblies

    windspeed = np.arange(0, 30, 0.2)
    correct_power = VESTAS_POWER_CURVE(windspeed)
    npt.assert_equal(system.power_curve(windspeed), correct_power)
    npt.assert_equal(system.power(windspeed), correct_power)

    # Test that we're starting at 100% operations
    assert system.operating_level == 1.0


def test_turbine_initialization_minimal_setup(env_setup):
    """Test an incomplete, or partial turbine defintion for correctness."""
    ENV = env_setup
    MANAGER = RepairManager(ENV)
    turbine_id = "WTG001"
    turbine_name = "Vestas V90 - 001"

    system = System(
        env=ENV,
        repair_manager=MANAGER,
        t_id=turbine_id,
        name=turbine_name,
        subassemblies=VESTAS_V90_1_SUBASSEMBLY,
        system="turbine",
    )
    assert system.env == ENV
    assert system.repair_manager == MANAGER
    assert system.id == turbine_id
    assert system.name == turbine_name
    assert system.capacity == VESTAS_V90["capacity_kw"]
    assert system.value == VESTAS_V90["capacity_kw"] * VESTAS_V90["capex_kw"]

    assert hasattr(system, "generator")
    assert [el.name for el in system.subassemblies] == ["generator"]

    # Check that there is no power curve defined, aside from all zero output
    windspeed = np.arange(0, 30, 0.2)
    correct_power = np.zeros(windspeed.shape)
    npt.assert_equal(system.power_curve(windspeed), correct_power)
    npt.assert_equal(system.power(windspeed), correct_power)

    # Test that we're starting at 100% operations
    assert system.operating_level == 1.0


def test_substation_initialization(env_setup):
    """Tests a complete substation setup."""
    ENV = env_setup
    MANAGER = RepairManager(ENV)
    substation_id = "OSS001"
    substation_name = "SUBSTATION - 001"

    system = System(
        env=ENV,
        repair_manager=MANAGER,
        t_id=substation_id,
        name=substation_name,
        subassemblies=SUBSTATION,
        system="substation",
    )
    assert system.env == ENV
    assert system.repair_manager == MANAGER
    assert system.id == substation_id
    assert system.name == substation_name
    assert system.capacity == SUBSTATION["capacity_kw"]
    assert system.value == SUBSTATION["capacity_kw"] * SUBSTATION["capex_kw"]

    # Check that only the turbine subassemblies are created
    assert hasattr(system, "transformer")
    assert [el.name for el in system.subassemblies] == ["transformer"]

    # Test that we're starting at 100% operations
    assert system.operating_level == 1.0


def test_polynomial_electrolyzer_setup(env_setup):
    """Tests a complete electrolyzer with polynomial efficiency curve setup."""
    ENV = env_setup
    MANAGER = RepairManager(ENV)
    electrolyzer_id = "ELC1"
    electrolyzer_name = "ELECTROLYZER - 001"

    system = System(
        env=ENV,
        repair_manager=MANAGER,
        t_id=electrolyzer_id,
        name=electrolyzer_name,
        subassemblies=ELECTROLYZER_POLY,
        system="electrolyzer",
    )
    assert system.env == ENV
    assert system.repair_manager == MANAGER
    assert system.id == electrolyzer_id
    assert system.name == electrolyzer_name
    assert system.n_stacks == ELECTROLYZER_POLY["n_stacks"]
    assert (
        system.capacity
        == ELECTROLYZER_POLY["n_stacks"] * ELECTROLYZER_POLY["stack_capacity_kw"]
    )
    assert (
        system.value
        == ELECTROLYZER_POLY["n_stacks"]
        * ELECTROLYZER_POLY["stack_capacity_kw"]
        * ELECTROLYZER_POLY["capex_kw"]
    )
    assert system.rated_production == 19.791303299083083

    # Check that only the turbine subassemblies are created
    assert hasattr(system, "power_system")
    assert [el.name for el in system.subassemblies] == ["Power System"]

    # Test that we're starting at 100% operations
    assert system.operating_level == 1.0


def test_linear_electrolyzer_setup(env_setup):
    """Tests a complete electrolyzer with polynomial efficiency curve setup."""
    ENV = env_setup
    MANAGER = RepairManager(ENV)
    electrolyzer_id = "ELC1"
    electrolyzer_name = "ELECTROLYZER - 001"

    system = System(
        env=ENV,
        repair_manager=MANAGER,
        t_id=electrolyzer_id,
        name=electrolyzer_name,
        subassemblies=ELECTROLYZER_LINEAR,
        system="electrolyzer",
    )
    assert system.env == ENV
    assert system.repair_manager == MANAGER
    assert system.id == electrolyzer_id
    assert system.name == electrolyzer_name
    assert system.n_stacks == ELECTROLYZER_LINEAR["n_stacks"]
    assert (
        system.capacity
        == ELECTROLYZER_LINEAR["n_stacks"] * ELECTROLYZER_LINEAR["stack_capacity_kw"]
    )
    assert (
        system.value
        == ELECTROLYZER_LINEAR["n_stacks"]
        * ELECTROLYZER_LINEAR["stack_capacity_kw"]
        * ELECTROLYZER_LINEAR["capex_kw"]
    )
    assert system.rated_production == 25.354969574036513

    # Check that only the turbine subassemblies are created
    assert hasattr(system, "power_system")
    assert [el.name for el in system.subassemblies] == ["Power System"]

    # Test that we're starting at 100% operations
    assert system.operating_level == 1.0


def test_initialization_error_cases(env_setup):
    """Tests the cases where a System definition should fail."""
    ENV = env_setup
    MANAGER = RepairManager(ENV)

    # No subassemblies provided
    turbine_id = "WTG001"
    turbine_name = "Vestas V90 - 001"
    with pytest.raises(ValueError):
        System(
            env=ENV,
            repair_manager=MANAGER,
            t_id=turbine_id,
            name=turbine_name,
            subassemblies=VESTAS_V90_NO_SUBASSEMBLY,
            system="turbine",
        )

    # Incorrect system definitions
    incorrect_system = "invalid"
    with pytest.raises(ValueError):
        System(
            env=ENV,
            repair_manager=MANAGER,
            t_id=turbine_id,
            name=turbine_name,
            subassemblies=VESTAS_V90,
            system=incorrect_system,
        )

    incorrect_system = "turbbine"
    with pytest.raises(ValueError):
        System(
            env=ENV,
            repair_manager=MANAGER,
            t_id=turbine_id,
            name=turbine_name,
            subassemblies=VESTAS_V90,
            system=incorrect_system,
        )

    incorrect_system = "ssubstation"
    with pytest.raises(ValueError):
        System(
            env=ENV,
            repair_manager=MANAGER,
            t_id=turbine_id,
            name=turbine_name,
            subassemblies=VESTAS_V90,
            system=incorrect_system,
        )


def test_operating_level(env_setup):
    """Tests the ``operating_level`` method."""
    ENV = env_setup
    MANAGER = RepairManager(ENV)
    turbine_id = "WTG001"
    turbine_name = "Vestas V90 - 001"

    system = System(
        env=ENV,
        repair_manager=MANAGER,
        t_id=turbine_id,
        name=turbine_name,
        subassemblies=VESTAS_V90,
        system="turbine",
    )

    # Test the starting operating level
    assert system.operating_level == 1

    # Test updating a single subassembly
    system.generator.operating_level = 0.9
    assert system.operating_level == 0.9

    # Test updating another subassembly
    system.electronic_control.operating_level = 0.9
    assert system.operating_level == 0.9 * 0.9

    # Test a cable failure
    system.cable_failure = ENV.event()
    assert system.operating_level == 0.0
    system.cable_failure.succeed()

    # Test a failed subassembly shuts down the turbine
    system.generator.operating_level = 0.0
    assert system.operating_level == 0.0
