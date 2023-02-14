"""Tests the Windfarm class."""

from pathlib import Path

import numpy as np
import pytest
import numpy.testing as npt
from networkx.classes.digraph import DiGraph

from wombat.core import RepairManager
from wombat.windfarm import Windfarm
from wombat.windfarm.system import Cable, System

from tests.conftest import (
    EXPORT,
    SUBSTATION,
    VESTAS_V90,
    ARRAY_33KV_240MM,
    ARRAY_33KV_630MM,
)


def test_windfarm_init(env_setup):
    """Tests the setup of ``Windfarm``."""
    env = env_setup
    manager = RepairManager(env)
    windfarm = Windfarm(env, "layout.csv", manager)

    correct_substation_id = ["OSS1"]
    correct_substation = System(env, manager, "OSS1", "OSS1", SUBSTATION, "substation")
    correct_turbine_ids = ["S00T1", "S00T2", "S00T3", "S01T4", "S01T5", "S01T6"]
    correct_turbine_names = ["WTG001", "WTG002", "WTG003", "WTG004", "WTG005", "WTG006"]
    correct_turbine_list = [
        System(env, manager, _id, _name, VESTAS_V90, "turbine")
        for _id, _name in zip(correct_turbine_ids, correct_turbine_names)
    ]
    correct_edge_keys = [
        ("OSS1", "OSS1"),
        ("OSS1", "S00T1"),
        ("S00T1", "S00T2"),
        ("S00T2", "S00T3"),  # string 1
        ("OSS1", "S01T4"),
        ("S01T4", "S01T5"),
        ("S01T5", "S01T6"),  # string 2
    ]
    correct_cable_ids = ["::".join(("cable", *el)) for el in correct_edge_keys]
    correct_cable_list = [
        Cable(windfarm, env, "export", "OSS1", "OSS1", EXPORT),
        Cable(windfarm, env, "array", "OSS1", "S00T1", ARRAY_33KV_630MM),
        Cable(windfarm, env, "array", "S00T1", "S00T2", ARRAY_33KV_240MM),
        Cable(windfarm, env, "array", "S00T2", "S00T3", ARRAY_33KV_240MM),
        Cable(windfarm, env, "array", "OSS1", "S01T4", ARRAY_33KV_630MM),
        Cable(windfarm, env, "array", "S01T4", "S01T5", ARRAY_33KV_240MM),
        Cable(windfarm, env, "array", "S01T5", "S01T6", ARRAY_33KV_240MM),
    ]
    correct_N_edges = 7
    correct_N_nodes = 7
    correct_capacity = VESTAS_V90["capacity_kw"] * len(correct_turbine_ids)
    correct_substation_turbine_map = {
        "OSS1": {
            "turbines": np.array(
                ["S00T1", "S01T4", "S00T2", "S00T3", "S01T5", "S01T6"]
            ),
            "weights": np.full(
                correct_N_nodes - 1, VESTAS_V90["capacity_kw"] / correct_capacity
            ),
        }
    }

    # Basic init items
    assert windfarm.repair_manager == manager
    assert manager.windfarm == windfarm
    assert Path(windfarm.env.operations_log_fname).is_file()

    # Windfarm layout checking: 6 turbines, 1 substation, and 6 cables
    assert windfarm.env == env
    assert isinstance(windfarm.graph, DiGraph)
    assert len(windfarm.graph.nodes) == correct_N_nodes
    assert len(windfarm.graph.edges) == correct_N_edges
    assert sorted(windfarm.graph.nodes) == sorted(
        correct_substation_id + correct_turbine_ids
    )
    assert sorted(windfarm.graph.edges) == sorted(correct_edge_keys)
    assert windfarm.substation_id == correct_substation_id
    assert sorted(windfarm.turbine_id) == sorted(correct_turbine_ids)

    assert windfarm.current_availability == 1
    assert windfarm.capacity == correct_capacity

    assert (
        windfarm.substation_turbine_map.keys() == correct_substation_turbine_map.keys()
    )
    oss1_map = windfarm.substation_turbine_map["OSS1"]
    correct_oss1_map = correct_substation_turbine_map["OSS1"]

    # For the substation map with turbines and their weights, the ordering of the
    # turbines doesn't matter when comparing because the weights are in the same order
    # as the turbines and the capacities in this model are all the same
    npt.assert_equal(sorted(oss1_map["turbines"]), sorted(correct_oss1_map["turbines"]))
    npt.assert_equal(oss1_map["weights"], correct_oss1_map["weights"])

    # TODO: CHECK THE DISTANCE MATRIX
    # TODO: WHERE IS THE LAYOUT DISTANCE FIELD BEING USED IN WINDFARM????
    correct_distances = np.array([0.0, np.inf])  # no distances
    distances = np.unique(windfarm.distance_matrix)
    npt.assert_equal(distances, correct_distances)

    # Check the actual windfarm objects for correctness. This is not a complete check
    # because that is the purpose of test_system.py and test_subassembly.py
    for _id, correct_turbine in zip(correct_turbine_ids, correct_turbine_list):
        turbine = windfarm.graph.nodes[_id]["system"]
        assert turbine.env == correct_turbine.env
        assert turbine.repair_manager == correct_turbine.repair_manager
        assert turbine.id == correct_turbine.id
        assert turbine.name == correct_turbine.name
        assert turbine.capacity == correct_turbine.capacity
        assert turbine.value == correct_turbine.value
        assert turbine.operating_level == correct_turbine.operating_level == 1
        windspeeds = np.arange(0, 30, 0.1)
        npt.assert_equal(turbine.power(windspeeds), correct_turbine.power(windspeeds))

        # Rough check of the subassemblies and maintenance/failure creation
        assert (
            turbine.electrical_system.processes.keys()
            == correct_turbine.electrical_system.processes.keys()
        )
        assert (
            turbine.electronic_control.processes.keys()
            == correct_turbine.electronic_control.processes.keys()
        )
        assert (
            turbine.sensors.processes.keys() == correct_turbine.sensors.processes.keys()
        )
        assert (
            turbine.hydraulic_system.processes.keys()
            == correct_turbine.hydraulic_system.processes.keys()
        )
        assert (
            turbine.yaw_system.processes.keys()
            == correct_turbine.yaw_system.processes.keys()
        )
        assert (
            turbine.rotor_blades.processes.keys()
            == correct_turbine.rotor_blades.processes.keys()
        )
        assert (
            turbine.mechanical_brake.processes.keys()
            == correct_turbine.mechanical_brake.processes.keys()
        )
        assert (
            turbine.rotor_hub.processes.keys()
            == correct_turbine.rotor_hub.processes.keys()
        )
        assert (
            turbine.gearbox.processes.keys() == correct_turbine.gearbox.processes.keys()
        )
        assert (
            turbine.generator.processes.keys()
            == correct_turbine.generator.processes.keys()
        )
        assert (
            turbine.supporting_structure.processes.keys()
            == correct_turbine.supporting_structure.processes.keys()
        )
        assert (
            turbine.drive_train.processes.keys()
            == correct_turbine.drive_train.processes.keys()
        )

    # Check the substation
    substation = windfarm.graph.nodes[correct_substation_id[0]]["system"]
    assert substation.env == correct_substation.env
    assert substation.repair_manager == correct_substation.repair_manager
    assert substation.id == correct_substation.id
    assert substation.name == correct_substation.name
    assert substation.capacity == correct_substation.capacity
    assert substation.value == correct_substation.value
    assert substation.operating_level == correct_substation.operating_level == 1

    # Rough check of the subassemblies and maintenance/failure creation
    assert (
        substation.transformer.processes.keys()
        == correct_substation.transformer.processes.keys()
    )

    # Check the cable initialization
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
        assert cable.broken.triggered == correct_cable.broken.triggered is True
        assert (
            cable.downstream_failure.triggered
            == correct_cable.downstream_failure.triggered
            is True
        )

        # Rough check of the subassemblies and maintenance/failure creation
        assert cable.processes.keys() == correct_cable.processes.keys()


def test_windfarm_failed_init(env_setup):
    """Tests the failing cases for setup of ``Windfarm`` where no data is provided."""
    env = env_setup
    manager = RepairManager(env)

    # Test for bad substation file
    with pytest.raises(ValueError):
        Windfarm(env, "layout_substation_invalid.csv", manager)

    # Test for bad subassembly file
    with pytest.raises(ValueError):
        Windfarm(env, "layout_subassembly_invalid.csv", manager)

    # Test for bad substation file
    with pytest.raises(ValueError):
        Windfarm(env, "layout_array_invalid.csv", manager)
