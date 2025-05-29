"""Test the Cable system and subassembly classes."""

from wombat.core import RepairManager
from wombat.windfarm import Windfarm
from wombat.windfarm.system import Cable

from tests.conftest import ARRAY_33KV_240MM, ARRAY_33KV_630MM


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
        Cable(windfarm, env, "array", "OSS1", "S00T1", ARRAY_33KV_630MM),
        Cable(windfarm, env, "array", "S00T1", "S00T2", ARRAY_33KV_240MM),
        Cable(windfarm, env, "array", "S00T2", "S00T3", ARRAY_33KV_240MM),
        Cable(windfarm, env, "array", "OSS1", "S01T4", ARRAY_33KV_630MM),
        Cable(windfarm, env, "array", "S01T4", "S01T5", ARRAY_33KV_240MM),
        Cable(windfarm, env, "array", "S01T5", "S01T6", ARRAY_33KV_240MM),
    ]

    for key, _id, correct_cable in zip(
        correct_edge_keys, correct_cable_ids, correct_cable_list
    ):
        cable = windfarm.graph.edges[key]["cable"]
        assert cable.env == correct_cable.env
        assert cable.id == correct_cable.id
        assert cable.name == correct_cable.name
        assert cable.system == correct_cable.system
        assert cable.data.name == cable.data.name
        assert cable.operating_level == correct_cable.operating_level == 1
        assert cable.broken.triggered
        assert cable.servicing.triggered
        assert cable.downstream_failure.triggered

        # Rough check of the subassemblies and maintenance/failure creation
        assert cable.processes.keys() == correct_cable.processes.keys()
