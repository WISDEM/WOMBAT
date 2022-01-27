"""Test the Cable system and subassembly classes."""

import pytest

from wombat.core import RepairManager
from wombat.windfarm import Windfarm
from wombat.windfarm.system import Cable

from tests.conftest import ARRAY_33KV_240MM, ARRAY_33KV_630MM, env_setup


@pytest.fixture()
def category(pytestconfig):
    return pytestconfig.getoption("run_category")


@pytest.mark.cat("all", "subassembly", "cable", "service_equipment")
def test_pass():
    pass


@pytest.mark.cat("all", "cable")
def test_cable_init(env_setup):
    """Tests the initialization of a `Cable` object. Much of this exists in the windfarm
    tests, but is repeated here for the sake of redundnacy and clarity that tests exist.
    """

    env = env_setup
    manager = RepairManager(env)
    windfarm = Windfarm(env, "layout.csv", manager)

    correct_edge_keys = [
        ("OSS1", "S00T1"),
        ("S00T1", "S00T2"),
        ("S00T2", "S00T3"),  # string 1
        ("OSS1", "S01T4"),
        ("S01T4", "S01T5"),
        ("S01T5", "S01T6"),  # string 2
    ]
    correct_cable_ids = ["::".join(("cable", *el)) for el in correct_edge_keys]
    correct_cable_list = [
        Cable(
            windfarm,
            env,
            "OSS1",
            "S00T1",
            ARRAY_33KV_630MM,
        ),
        Cable(
            windfarm,
            env,
            "S00T1",
            "S00T2",
            ARRAY_33KV_240MM,
        ),
        Cable(windfarm, env, "S00T2", "S00T3", ARRAY_33KV_240MM),
        Cable(
            windfarm,
            env,
            "OSS1",
            "S01T4",
            ARRAY_33KV_630MM,
        ),
        Cable(
            windfarm,
            env,
            "S01T4",
            "S01T5",
            ARRAY_33KV_240MM,
        ),
        Cable(windfarm, env, "S01T5", "S01T6", ARRAY_33KV_240MM),
    ]

    for key, _id, correct_cable in zip(
        correct_edge_keys, correct_cable_ids, correct_cable_list
    ):
        cable = windfarm.graph.edges[key]["cable"]
        assert cable.env == correct_cable.env
        assert cable.id == correct_cable.id
        assert cable.name == correct_cable.name
        assert cable.turbine == correct_cable.turbine
        assert cable.data.name == cable.data.name
        assert cable.operating_level == correct_cable.operating_level == 1
        assert cable.broken == correct_cable.broken is False
        assert cable.downstream_failure == correct_cable.downstream_failure is False

        # Rough check of the subassemblies and maintenance/failure creation
        assert cable.processes.keys() == correct_cable.processes.keys()


@pytest.mark.cat("all", "cable")
def test_cable_failures(category, env_setup):
    """Test that failing cable disable upstream turbines"""
    env = env_setup
    manager = RepairManager(env)
    windfarm = Windfarm(env, "layout_cable_test.csv", manager)

    # env.run(1)
    # from pprint import pprint
    # for name, cable in windfarm.graph.edges.items():
    #     if name != ("OSS1", "S00T1"):
    #         continue
    #     cable = cable["cable"]
    #     print(name)
    #     for pname, p in cable.processes.items():
    #         print(pname)
    #         pprint(p.__dict__)

    # A failure will occur at 13.85 for the cable connecting the OSS to the first turbine,
    # so check that all turbines are down, and this should effectively shut down the farm.
    catastrophic_timeout = 13.85
    env.run(catastrophic_timeout + 1)

    assert windfarm.current_availability == 0.5  # half the farm is off

    # Check the cable is broken
    cable = windfarm.cable(("OSS1", "S00T1"))
    assert cable.broken
    assert cable.operating_level == 0

    # Assert that all items are timed out for a day until the next check or
    # until the end of the simulation
    for p in cable.processes.values():
        print(p._target._delay)
    assert all(
        p._target._delay in (24, 1402.148783264902) for p in cable.processes.values()
    )

    # Check the failure was submitted and no other items exist for this cable
    assert sum(req.system_id == "cable::OSS1::S00T1" for req in manager.items) == 1

    # Check the substation is operating (no downstream nodes are shut off)
    oss = windfarm.system("OSS1")
    assert not oss.cable_failure
    assert oss.operating_level == 1

    # Check the appropriate turbines are broken
    for turbine in ("S00T1", "S00T2", "S00T3"):
        turbine = windfarm.system(turbine)
        assert turbine.cable_failure
        assert turbine.operating_level == 0

        # Check that operating level for subassemblies is unaffected, and 0 turbine
        # operations is controlled by cable failure flag
        assert all((s.operating_level == 1 for s in turbine.subassemblies))

    # Check the appropriate turbines are operating
    for turbine in ("S01T4", "S01T5", "S01T6"):
        turbine = windfarm.system(turbine)
        assert not turbine.cable_failure
        assert turbine.operating_level == 1
        assert all((s.operating_level == 1 for s in turbine.subassemblies))
