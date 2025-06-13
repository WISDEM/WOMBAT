"""Unit tests configuration file."""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wombat.core import Metrics, Simulation, WombatEnvironment
from wombat.utilities import IEC_power_curve
from wombat.core.library import load_yaml


ROOT = Path(__file__).parent
TEST_DATA = ROOT / "library"


def pytest_addoption(parser):  # noqa: D103
    parser.addoption(
        "--unit", action="store_true", default=False, help="run tests in test/unit/."
    )
    parser.addoption(
        "--regression",
        action="store_true",
        default=False,
        help="run tests in test/regression/.",
    )


def pytest_configure(config):  # noqa: D103
    # Check for the options
    unit = config.getoption("--unit")
    regression = config.getoption("--regression")

    # Provide the appropriate directories
    unit_tests = [el for el in (ROOT / "unit").iterdir() if el.suffix == ".py"]
    regression_tests = [
        el for el in (ROOT / "regression").iterdir() if el.suffix == ".py"
    ]

    # If both, run them all; if neither skip any modifications; otherwise run just the
    # appropriate subset
    if regression and unit:
        config.args = unit_tests + regression_tests
    elif regression:
        config.args = regression_tests
    elif unit:
        config.args = unit_tests


def pytest_assertrepr_compare(op, left, right):
    """Custom failure messages."""
    if isinstance(left, Metrics) and isinstance(right, Metrics) and op == "==":
        return left._repr_compare(right)


@pytest.fixture
def env_setup():
    """Ensure the proper setup and teardown of an environment so no logging
    files can accumulate.
    """
    env = WombatEnvironment(
        data_dir=TEST_DATA,
        weather_file="test_weather_quick_load.csv",
        workday_start=8,
        workday_end=16,
        simulation_name="testing_setup",
        random_seed=2022,
    )
    yield env
    env.cleanup_log_files()


@pytest.fixture
def env_setup_full_profile():
    """Create a full weather profile environment with proper teardown."""
    env = WombatEnvironment(
        data_dir=TEST_DATA,
        weather_file="test_weather.csv",
        workday_start=8,
        workday_end=16,
        simulation_name="testing_setup",
        random_seed=2022,
    )
    yield env
    env.cleanup_log_files()


@pytest.fixture(scope="module")
def setup_ttp():
    """Create a full weather profile environment with proper teardown."""
    sim = Simulation("COREWIND", "morro_bay_tow_to_port.yaml", random_seed=2022)
    sim.run(8760 * 5)
    yield sim
    sim.env.cleanup_log_files()


SUBSTATION = load_yaml(TEST_DATA / "substations", "offshore_substation.yaml")
VESTAS_V90 = load_yaml(TEST_DATA / "turbines", "vestas_v90.yaml")
VESTAS_V90_1_SUBASSEMBLY = load_yaml(
    TEST_DATA / "turbines", "vestas_v90_single_subassembly.yaml"
)
VESTAS_V90_NO_SUBASSEMBLY = load_yaml(
    TEST_DATA / "turbines", "vestas_v90_no_subassemblies.yaml"
)
VESTAS_V90_TEST_TIMEOUTS = load_yaml(
    TEST_DATA / "turbines", "vestas_v90_test_timeouts.yaml"
)
ARRAY_33KV_240MM = load_yaml(TEST_DATA / "cables", "array_33kv_240mm.yaml")
ARRAY_33KV_630MM = load_yaml(TEST_DATA / "cables", "array_33kv_630mm.yaml")
EXPORT = load_yaml(TEST_DATA / "cables", "export.yaml")
ELECTROLYZER_POLY = load_yaml(TEST_DATA / "project/config", "poly_electrolyzer.yml")[
    "electrolyzers"
]["test_electrolyzer_poly"]
ELECTROLYZER_LINEAR = load_yaml(
    TEST_DATA / "project/config", "linear_electrolyzer.yml"
)["electrolyzers"]["test_electrolyzer_linear"]


power_curve = TEST_DATA / "turbines" / "vestas_v90_power_curve.csv"
power_curve = pd.read_csv(f"{power_curve}")
power_curve = power_curve.loc[power_curve.power_kw != 0].reset_index(drop=True)
VESTAS_POWER_CURVE = IEC_power_curve(
    power_curve.windspeed_ms,
    power_curve.power_kw,
    windspeed_start=4,
    windspeed_end=25,
    bin_width=0.5,
)
del power_curve

RNG = np.random.default_rng(seed=34)


GENERATOR_SUBASSEMBLY = {
    "name": "generator",
    "system_value": 39000000,
    "maintenance_start": datetime.datetime(2000, 6, 1),
    "maintenance": [
        {
            "description": "annual service",
            "time": 60,
            "materials": 18500,
            "service_equipment": "CTV",
            "frequency": 365,
            "system_value": 39000000,
        }
    ],
    "failures": [
        {
            "scale": 0.1333,
            "shape": 1,
            "time": 3,
            "materials": 0,
            "service_equipment": "CTV",
            "operation_reduction": 0.0,
            "level": 1,
            "description": "manual reset",
            "system_value": 39000000,
        },
        {
            "scale": 0.3333,
            "shape": 1,
            "time": 7.5,
            "materials": 1000,
            "service_equipment": "CTV",
            "operation_reduction": 0.0,
            "level": 2,
            "description": "minor repair",
            "system_value": 39000000,
        },
        {
            "scale": 3.6363,
            "shape": 1,
            "time": 22,
            "materials": 18500,
            "service_equipment": "CTV",
            "operation_reduction": 0.0,
            "level": 3,
            "description": "medium repair",
            "system_value": 39000000,
        },
        {
            "scale": 25,
            "shape": 1,
            "time": 26,
            "materials": 73500,
            "service_equipment": "SCN",
            "operation_reduction": 0.0,
            "level": 4,
            "description": "major repair",
            "system_value": 39000000,
        },
        {
            "scale": 12.5,
            "shape": 1,
            "time": 52,
            "materials": 334500,
            "service_equipment": "LCN",
            "operation_reduction": 1.0,
            "level": 5,
            "description": "major replacement",
            "system_value": 39000000,
        },
    ],
    "rng": RNG,
}

UNSCHEDULED_VESSEL_REQUESTS = {
    "name": "Heavy Lift Vessel",
    "equipment_rate": 150000,
    "charter_days": 30,
    "strategy": "requests",
    "strategy_threshold": 10,
    "start_year": 2002,
    "end_year": 2014,
    "onsite": False,
    "capability": "LCN",
    "mobilization_cost": 500000,
    "mobilization_days": 60,
    "speed": 12.66,
    "max_windspeed_transport": 10,
    "max_windspeed_repair": 10,
    "max_waveheight_transport": 2,
    "max_waveheight_repair": 2,
    "workday_start": 0,
    "workday_end": 24,
    "crew_transfer_time": 0.25,
    "method": "severity",
    "n_crews": 1,
    "crew": {
        "day_rate": 0,
        "n_day_rate": 0,
        "hourly_rate": 0,
        "n_hourly_rate": 0,
    },
}


UNSCHEDULED_VESSEL_DOWNTIME = {
    "name": "Heavy Lift Vessel",
    "equipment_rate": 150000,
    "charter_days": 30,
    "strategy": "downtime",
    "strategy_threshold": 0.9,
    "start_year": 2002,
    "end_year": 2014,
    "onsite": False,
    "capability": "LCN",
    "mobilization_cost": 500000,
    "mobilization_days": 60,
    "speed": 12.66,
    "max_windspeed_transport": 10,
    "max_windspeed_repair": 10,
    "max_waveheight_transport": 2,
    "max_waveheight_repair": 2,
    "workday_start": 0,
    "workday_end": 24,
    "crew_transfer_time": 0.25,
    "n_crews": 1,
    "method": "turbine",
    "crew": {
        "day_rate": 0,
        "n_day_rate": 0,
        "hourly_rate": 0,
        "n_hourly_rate": 0,
    },
}


SCHEDULED_VESSEL = {
    "name": "Crew Transfer Vessel",
    "equipment_rate": 1750,
    "start_month": 1,
    "start_day": 1,
    "end_month": 12,
    "end_day": 31,
    "start_year": 2002,
    "end_year": 2014,
    "onsite": True,
    "capability": "CTV",
    "max_severity": 10,
    "mobilization_cost": 0,
    "mobilization_days": 0,
    "speed": 37.04,
    "max_windspeed_transport": 99,
    "max_windspeed_repair": 99,
    "max_waveheight_transport": 1.5,
    "max_waveheight_repair": 1.5,
    "workday_start": 8,
    "workday_end": 20,
    "strategy": "scheduled",
    "method": "turbine",
    "crew_transfer_time": 0.25,
    "n_crews": 1,
    "crew": {
        "day_rate": 0,
        "n_day_rate": 0,
        "hourly_rate": 0,
        "n_hourly_rate": 0,
    },
}
