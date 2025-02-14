"""Test the API functionality."""

import sys

import pytest

from wombat import Simulation
from wombat.core import Metrics

from tests.conftest import TEST_DATA


def test_simulation_file_setup():
    """Test that a primarily file-defined simulation initializes correctly."""
    sim = Simulation.from_config(TEST_DATA, "base.yml")

    assert len(sim.service_equipment) == 5
    assert len(sim.windfarm.turbine_id) == 6
    assert len(sim.windfarm.substation_id) == 1
    assert len(sim.windfarm.graph.edges) == 7

    sim.run()

    assert isinstance(sim.metrics, Metrics)

    sim.env.cleanup_log_files()


def test_simulation_consolidated_setup():
    """Test that a primarily dictionary-defined simulation initializes correctly."""
    sim = Simulation.from_config(TEST_DATA, "base_consolidated.yml")

    assert len(sim.service_equipment) == 5
    assert len(sim.windfarm.turbine_id) == 6
    assert len(sim.windfarm.substation_id) == 1
    assert len(sim.windfarm.graph.edges) == 7

    sim.run()

    assert isinstance(sim.metrics, Metrics)

    sim.env.cleanup_log_files()


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="Why does Windows load it differently?"
)
def test_metrics_save_load():
    """Tests that the saved metrics data can be reused."""
    sim = Simulation.from_config(TEST_DATA, "base_consolidated.yml")
    sim.run()

    metrics = sim.metrics
    sim.save_metrics_inputs()

    metrics_reloaded = Metrics.from_simulation_outputs(
        fpath=TEST_DATA / "results", fname=sim.env.metrics_input_fname
    )

    assert metrics == metrics_reloaded

    sim.env.cleanup_log_files()
