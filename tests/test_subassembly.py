"""Tests the Subassembly class."""

import numpy as np
import pytest

from wombat.windfarm import Windfarm
from wombat.windfarm.system import System, Subassembly
from wombat.core.data_classes import Failure, Maintenance
from wombat.core.repair_management import RepairManager

from tests.conftest import (
    RNG,
    SUBSTATION,
    VESTAS_V90,
    GENERATOR_SUBASSEMBLY,
    VESTAS_V90_TEST_TIMEOUTS,
)


np.random.seed(2022)  # for test_interruptions()


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


@pytest.mark.skip(reason="The timing of the failures needs to be updated")
def test_interruptions_and_request_submission(env_setup):
    """Test the initialization of a subassembly.

    NOTE: This test will not run correctly if not running the whole pytest suite because
    it relies on the when/where the the random variable is set in the process and the
    order in which the files run. This is considered to be acceptable behavior because
    it is not testing for the actual timing, but the behavior when certain events happen
    and that they are logged.
    """
    # Define the basic items
    ENV = env_setup
    MANAGER = RepairManager(ENV)
    TURBINE = System(
        ENV, MANAGER, "WTG001", "Vestas V90 001", VESTAS_V90_TEST_TIMEOUTS, "turbine"
    )

    # Timeout turbine will have (randomness controlled by random seed at the top):
    #  1) a generator maintenance event every 5 days (120 hours/time steps)
    #  2) a generator failure event with a Weibull of scale 0.5, shape 1, with
    #      frequencies at:
    #      2.47, 13.47, 303.92 hours (last one not reached)
    #  3) a gearbox failure event with a Weibull of scale 5, shape 1, with
    #      frequencies at:
    #      1054.20 and 9132.90 hours (last one not reached)
    #  4) a generator catastrophic failure event at 3027.74 hours that stops everything

    # The resets and maintenance tasks will be removed from the record for the
    # generator upon failure
    correct_N_generator_failures = 2  # 2 resets reached prior to catastrophic failure
    correct_N_generator_maintenance = 4  # 4 prior to failure
    correct_N_gearbox_failures = 1  # none reached
    correct_N_gearbox_maintenance = 0  # none reached

    # PROCESS FOR DETERMINING ACTUAL EVENTS IN CASE OF CHANGE OF RANDOM BEHAVIOR
    """
    # TODO
    t = 2.47 + 13.47 + 303.92 + 175.41 + 599.03 + 597.26
    print(t)
    ENV.run(1 + t + 2)
    print("generator")
    for key, process in TURBINE.generator.processes.items():
        print(key, process.__dict__)
    print("gearbox")
    for key, process in TURBINE.gearbox.processes.items():
        print(key, process.__dict__)
    print(ENV.now)
    assert False
    """

    catastrophic_timeout = 527.1  # see notes above for catastrophic failure timing

    # Test that all the failures that will get purged actually happen
    ENV.run(catastrophic_timeout - 1)

    requests = MANAGER.items
    generator_requests = [el for el in requests if el.subassembly_id == "generator"]
    gearbox_requests = [el for el in requests if el.subassembly_id == "gearbox"]
    generator_maintenance = [
        el for el in generator_requests if isinstance(el.details, Maintenance)
    ]
    generator_failures = [
        el for el in generator_requests if isinstance(el.details, Failure)
    ]
    gearbox_maintenance = [
        el for el in gearbox_requests if isinstance(el.details, Maintenance)
    ]
    gearbox_failures = [
        el for el in gearbox_requests if isinstance(el.details, Failure)
    ]

    # See the notes above for timing
    assert len(generator_failures) == correct_N_generator_failures
    assert len(generator_maintenance) == correct_N_generator_maintenance
    assert len(gearbox_failures) == correct_N_gearbox_failures
    assert len(gearbox_maintenance) == correct_N_gearbox_maintenance

    # Test that the correct failures/tasks get purged and all the processes are
    # timed out appropriately
    ENV.run(catastrophic_timeout + 1)

    # Check that the generator is set to broken
    assert TURBINE.generator.broken is True
    assert TURBINE.gearbox.broken is False
    assert TURBINE.generator.operating_level == 0
    assert TURBINE.gearbox.operating_level == 1
    assert TURBINE.operating_level == 0

    # Check that the 24 hour default timeout has been setup, which indicates that all
    # processes have been interrupted. This is the 1 day timeout to see if the turbine
    # is operating again in case of repairs
    for process in TURBINE.generator.processes.values():
        assert process._target._delay == 24
    for process in TURBINE.gearbox.processes.values():
        assert process._target._delay == 24

    requests = MANAGER.items
    generator_requests = [el for el in requests if el.subassembly_id == "generator"]
    generator_maintenance = [
        el for el in generator_requests if isinstance(el.details, Maintenance)
    ]
    generator_failures = [
        el for el in generator_requests if isinstance(el.details, Failure)
    ]
    gearbox_requests = [el for el in requests if el.subassembly_id == "gearbox"]
    gearbox_maintenance = [
        el for el in gearbox_requests if isinstance(el.details, Maintenance)
    ]
    gearbox_failures = [
        el for el in gearbox_requests if isinstance(el.details, Failure)
    ]

    # Check the total number of request submitted from each subassembly is correct
    assert len(generator_requests) == 1  # all others will have been purged
    assert (
        len(gearbox_requests)
        == correct_N_gearbox_failures + correct_N_gearbox_maintenance
    )

    # Check the number of maintenance and failure requests for each subassembly
    assert len(gearbox_failures) == correct_N_gearbox_failures
    assert len(gearbox_maintenance) == correct_N_gearbox_maintenance
    assert len(generator_failures) == 1  # all but catastrophic failure removed
    assert len(generator_maintenance) == 0  # all removed

    # Run for another 24 hours to ensure nothing has changed
    ENV.run(catastrophic_timeout + 24)

    assert TURBINE.generator.broken is True
    assert TURBINE.gearbox.broken is False
    assert TURBINE.generator.operating_level == 0
    assert TURBINE.gearbox.operating_level == 1
    assert TURBINE.operating_level == 0

    for process in TURBINE.generator.processes.values():
        assert process._target._delay == 24
    for process in TURBINE.gearbox.processes.values():
        assert process._target._delay == 24

    # Run until the end to ensure nothig has changed
    ENV.run(ENV.max_run_time)

    assert TURBINE.generator.broken is True
    assert TURBINE.gearbox.broken is False
    assert TURBINE.generator.operating_level == 0
    assert TURBINE.gearbox.operating_level == 1
    assert TURBINE.operating_level == 0

    for process in TURBINE.generator.processes.values():
        assert process._target._delay == 24
    for process in TURBINE.gearbox.processes.values():
        assert process._target._delay == 24


def test_timeouts_for_zeroed_out(env_setup):
    """Tests that the timeouts for any zeroed out subassembly maintenance or failure
    does not occur.
    """
    # Define the basic items
    ENV = env_setup
    MANAGER = RepairManager(ENV)
    TURBINE = System(ENV, MANAGER, "WTG001", "Vestas V90 001", VESTAS_V90, "turbine")
    Windfarm(ENV, "layout.csv", MANAGER)

    # Run the simulation 1 timestep to get it started, then inspec the process timeouts
    ENV.run(1)

    # Only the generator should have timeouts not equal to max run time
    for subassembly in TURBINE.subassemblies:
        if subassembly.id == "generator":
            continue
        for process in subassembly.processes.values():
            assert process._target._delay in (ENV.max_run_time, ENV.max_run_time + 23)

    # Catastrophic failure at 284508.82985483576
    # TODO: Don't have a way to test that all update to max_run_time - failure time
