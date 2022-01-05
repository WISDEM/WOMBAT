"""Unit tests configuration file."""
import pandas as pd
from pathlib import Path

from wombat.core import RepairManager, WombatEnvironment
from wombat.core.library import load_yaml
from wombat.utilities.utilities import IEC_power_curve


TEST_DATA = Path(__file__).resolve().parent / "library"


ENV = WombatEnvironment(
    data_dir=TEST_DATA,
    weather_file="test_weather_quick_load.csv",
    workday_start=8,
    workday_end=16,
    simulation_name="testing_setup",
)
MANAGER = RepairManager(ENV)
SUBSTATION = load_yaml(TEST_DATA / "windfarm", "offshore_substation.yaml")
VESTAS_V90 = load_yaml(TEST_DATA / "windfarm", "vestas_v90.yaml")
VESTAS_V90_1_SUBASSEMBLY = load_yaml(
    TEST_DATA / "windfarm", "vestas_v90_single_subassembly.yaml"
)
VESTAS_V90_NO_SUBASSEMBLY = load_yaml(
    TEST_DATA / "windfarm", "vestas_v90_no_subassemblies.yaml"
)


power_curve = TEST_DATA / "windfarm" / "vestas_v90_power_curve.csv"
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


GENERATOR_SUBASSEMBLY = dict(
    name="generator",
    system_value=39000000,
    maintenance=[
        dict(
            description="annual service",
            time=60,
            materials=18500,
            service_equipment="CTV",
            frequency=365,
            system_value=39000000,
        )
    ],
    failures={
        1: dict(
            scale=0.1333,
            shape=1,
            time=3,
            materials=0,
            service_equipment="CTV",
            operation_reduction=0.0,
            level=1,
            description="manual reset",
            system_value=39000000,
        ),
        2: dict(
            scale=0.3333,
            shape=1,
            time=7.5,
            materials=1000,
            service_equipment="CTV",
            operation_reduction=0.0,
            level=2,
            description="minor repair",
            system_value=39000000,
        ),
        3: dict(
            scale=3.6363,
            shape=1,
            time=22,
            materials=18500,
            service_equipment="CTV",
            operation_reduction=0.0,
            level=3,
            description="medium repair",
            system_value=39000000,
        ),
        4: dict(
            scale=25,
            shape=1,
            time=26,
            materials=73500,
            service_equipment="SCN",
            operation_reduction=0.0,
            level=4,
            description="major repair",
            system_value=39000000,
        ),
        5: dict(
            scale=12.5,
            shape=1,
            time=52,
            materials=334500,
            service_equipment="LCN",
            operation_reduction=1.0,
            level=5,
            description="major replacement",
            system_value=39000000,
        ),
    },
)

UNSCHEDULED_VESSEL_REQUESTS = dict(
    name="Heavy Lift Vessel",
    equipment_rate=150000,
    charter_days=30,
    strategy="requests",
    strategy_threshold=10,
    start_year=2002,
    end_year=2014,
    onsite=False,
    capability="LCN",
    mobilization_cost=500000,
    mobilization_days=60,
    speed=12.66,
    max_windspeed_transport=10,
    max_windspeed_repair=10,
    max_waveheight_transport=2,
    max_waveheight_repair=2,
    workday_start=0,
    workday_end=24,
    crew_transfer_time=0.25,
    method="severity",
    n_crews=1,
    crew=dict(
        day_rate=0,
        n_day_rate=0,
        hourly_rate=0,
        n_hourly_rate=0,
    ),
)


UNSCHEDULED_VESSEL_DOWNTIME = dict(
    name="Heavy Lift Vessel",
    equipment_rate=150000,
    charter_days=30,
    strategy="downtime",
    strategy_threshold=0.9,
    start_year=2002,
    end_year=2014,
    onsite=False,
    capability="LCN",
    mobilization_cost=500000,
    mobilization_days=60,
    speed=12.66,
    max_windspeed_transport=10,
    max_windspeed_repair=10,
    max_waveheight_transport=2,
    max_waveheight_repair=2,
    workday_start=0,
    workday_end=24,
    crew_transfer_time=0.25,
    n_crews=1,
    method="turbine",
    crew=dict(
        day_rate=0,
        n_day_rate=0,
        hourly_rate=0,
        n_hourly_rate=0,
    ),
)


SCHEDULED_VESSEL = dict(
    name="Crew Transfer Vessel",
    equipment_rate=1750,
    start_month=1,
    start_day=1,
    end_month=12,
    end_day=31,
    start_year=2002,
    end_year=2014,
    onsite=True,
    capability="CTV",
    max_severity=10,
    mobilization_cost=0,
    mobilization_days=0,
    speed=37.04,
    max_windspeed_transport=99,
    max_windspeed_repair=99,
    max_waveheight_transport=1.5,
    max_waveheight_repair=1.5,
    workday_start=8,
    workday_end=20,
    strategy="scheduled",
    method="turbine",
    crew_transfer_time=0.25,
    n_crews=1,
    crew=dict(
        day_rate=0,
        n_day_rate=0,
        hourly_rate=0,
        n_hourly_rate=0,
    ),
)
