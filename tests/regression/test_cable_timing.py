"""Test the Cable system and subassembly classes."""

import pytest

from wombat.core import RepairManager
from wombat.windfarm import Windfarm


def test_cable_failures(env_setup):
    """Test that failing cable disable upstream turbines."""
    env = env_setup
    manager = RepairManager(env)
    windfarm = Windfarm(env, "layout_cable_test.csv", manager)

    # NOTE: If timing changes, this is the process to inpsect timeouts
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

    # A failure will occur at 10.79 for the cable connecting the OSS to the first
    # turbine, so check that all turbines are down, and this should effectively shut
    # down the farm.
    catastrophic_timeout = 10.793391659337129
    env.run(catastrophic_timeout + 1)
    assert windfarm.current_availability == 0.5  # 50% of the farm is off

    # Check the cable is broken
    cable = windfarm.cable(("OSS1", "S00T1"))
    assert cable.broken
    assert cable.operating_level == 0

    # Assert that all items that are timed are non existent or set to the end + buffer
    # from pprint import pprint
    # for p in cable.processes.values():
    #     pprint(p._target.__dict__)
    non_modeled_timeout = getattr(
        list(cable.processes.values())[0]._target, "_delay", None
    )
    assert non_modeled_timeout is None

    post_sim_timeout = getattr(
        list(cable.processes.values())[1]._target, "_delay", None
    )
    assert post_sim_timeout == pytest.approx(
        env.max_run_time - catastrophic_timeout + 23
    )

    # Check the failure was submitted and no other items exist for this cable
    assert sum(req.system_id == "cable::OSS1::S00T1" for req in manager.items) == 1

    # Check the substation is operating (no downstream nodes are shut off)
    oss = windfarm.system("OSS1")
    assert oss.cable_failure.triggered
    assert oss.operating_level == 1

    # Check the appropriate turbines are broken
    for turbine in ("S00T1", "S00T2", "S00T3"):
        turbine = windfarm.system(turbine)
        assert not turbine.cable_failure.triggered
        assert turbine.operating_level == 0

        # Check that operating level for subassemblies is unaffected, and 0 turbine
        # operations is controlled by cable failure flag
        assert all(s.operating_level == 1 for s in turbine.subassemblies)

    # Check the appropriate turbines are operating
    for turbine in ("S01T4", "S01T5", "S01T6"):
        turbine = windfarm.system(turbine)
        assert turbine.cable_failure.triggered
        assert turbine.operating_level == 1
        assert all(s.operating_level == 1 for s in turbine.subassemblies)
