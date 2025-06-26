"""Test the API functionality."""

import sys
import datetime

import pandas as pd
import pytest

from wombat import Simulation, load_yaml
from wombat.core import Metrics

from tests.conftest import TEST_DATA


def test_simulation_file_setup():
    """Test that a primarily file-defined simulation initializes correctly."""
    sim = Simulation.from_config(TEST_DATA, "base.yml")

    assert sim.env.maintenance_start == datetime.datetime(2002, 6, 1)
    assert len(sim.service_equipment) == 5
    assert len(sim.windfarm.turbine_id) == 6
    assert len(sim.windfarm.substation_id) == 1
    assert len(sim.windfarm.graph.edges) == 7

    sim.run()

    assert isinstance(sim.metrics, Metrics)

    sim.env.cleanup_log_files()


def test_consolidated_config_simulation():
    """Test that the consolidated configuration setup works for all possible setups."""
    sim = Simulation(TEST_DATA, "poly_electrolyzer.yml")
    assert isinstance(sim, Simulation)
    assert len(sim.windfarm.turbine_id) == 1
    assert len(sim.windfarm.substation_id) == 1
    assert len(sim.windfarm.electrolyzer_id) == 1

    sim = Simulation(TEST_DATA, "linear_electrolyzer.yml")
    assert isinstance(sim, Simulation)
    assert len(sim.windfarm.turbine_id) == 1
    assert len(sim.windfarm.substation_id) == 1
    assert len(sim.windfarm.electrolyzer_id) == 1


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


def test_simulation_df_layout():
    """Test that the simulation successfully loads with a DataFrame layout."""
    library_path = TEST_DATA
    config = load_yaml(library_path / "project/config", "linear_electrolyzer.yml")
    layout = pd.read_csv(library_path / "project/plant/layout_electrolyzer_linear.csv")
    config["layout"] = layout

    sim = Simulation(library_path, config)
    sim.run(delete_logs=True, save_metrics_inputs=False)


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="Why does Windows load it differently?"
)
def test_metrics_save_load():
    """Tests that the saved metrics data can be reused."""
    sim = Simulation.from_config(TEST_DATA, "base_consolidated.yml")
    sim.run(save_metrics_inputs=True)

    metrics = sim.metrics

    metrics_reloaded = Metrics.from_simulation_outputs(
        fpath=TEST_DATA / "results", fname=sim.env.metrics_input_fname
    )

    assert metrics == metrics_reloaded

    sim.env.cleanup_log_files()
