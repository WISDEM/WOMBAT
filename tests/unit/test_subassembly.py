"""Tests the Subassembly class."""

from wombat.windfarm.system import System, Subassembly
from wombat.core.repair_management import RepairManager

from tests.conftest import RNG, SUBSTATION, VESTAS_V90, GENERATOR_SUBASSEMBLY


def test_subassembly_initialization(env_setup):
    """Test the initialization of a subassembly."""
    # Define the basic items
    ENV = env_setup
    MANAGER = RepairManager(ENV)
    TURBINE = System(ENV, MANAGER, "WTG001", "Vestas V90 001", VESTAS_V90, "turbine")
    OSS = System(ENV, MANAGER, "OSS1", "OSS1", SUBSTATION, "substation")
    TRANSFORMER = SUBSTATION["transformer"]

    # Test the initialization for a turbine and generator
    subassembly_id = "GEN001"
    data_dict = GENERATOR_SUBASSEMBLY  # same data as the one in the library
    generator = Subassembly(TURBINE, ENV, subassembly_id, data_dict)
    correct_N_maintenance = len(GENERATOR_SUBASSEMBLY["maintenance"])
    correct_N_failures = len(GENERATOR_SUBASSEMBLY["failures"])
    correct_descriptions = [
        (el["level"], el["description"]) for el in GENERATOR_SUBASSEMBLY["failures"]
    ]
    correct_descriptions.extend(
        [el["description"] for el in GENERATOR_SUBASSEMBLY["maintenance"]]
    )

    assert generator.env == ENV
    assert generator.system == TURBINE
    assert generator.id == subassembly_id
    assert generator.name == data_dict["name"]
    # assert subassembly.data == TURBINE.generator.data  # TODO: equality testing
    assert generator.broken.triggered
    assert generator.operating_level == 1.0
    assert len(generator.processes) == correct_N_maintenance + correct_N_failures
    # Maintenance keys are descriptions
    # Failure keys are tuples of severity and description
    N_failures = len([el for el in generator.processes if isinstance(el, tuple)])
    assert N_failures == correct_N_failures
    assert [*generator.processes] == correct_descriptions

    # Run the same tests for a substation and transformer
    subassembly_id = "TRNS1"
    data_dict = TRANSFORMER
    data_dict.update({"rng": RNG})
    correct_N_maintenance = len(TRANSFORMER["maintenance"])
    correct_N_failures = len(TRANSFORMER["failures"])
    transformer = Subassembly(OSS, ENV, subassembly_id, data_dict)

    assert transformer.env == ENV
    assert transformer.system == OSS
    assert transformer.id == subassembly_id
    assert transformer.name == data_dict["name"]
    # assert subassembly.data == OSS.transformer.data  # TODO: equality testing
    assert transformer.broken.triggered
    assert transformer.operating_level == 1.0
    assert len(transformer.processes) == correct_N_maintenance + correct_N_failures

    # Maintenance keys are descriptions
    # Failure keys are tuples of severity and description
    N_failures = len([el for el in transformer.processes if isinstance(el, tuple)])
    assert N_failures == correct_N_failures
