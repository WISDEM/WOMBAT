"""Tests for wombat/core/library.py."""


from wombat.core.library import load_yaml, library_map


def test_load_yaml():
    """Tests `load_yaml`."""
    # Test loading the offshore substation from the Dinwoodie library
    library = library_map["DINWOODIE"]
    folder = "substations"
    correct_substation = {
        "capacity_kw": 0,
        "capex_kw": 0,
        "transformer": {
            "name": "substation",
            "maintenance": [
                {
                    "description": "n/a",
                    "time": 0,
                    "materials": 0,
                    "service_equipment": "CTV",
                    "frequency": 0,
                }
            ],
            "failures": {
                1: {
                    "scale": 0,
                    "shape": 0,
                    "time": 0,
                    "materials": 0,
                    "service_equipment": ["CTV"],
                    "operation_reduction": 0,
                    "level": 1,
                    "description": "n/a",
                }
            },
        },
    }

    substation = load_yaml(library / folder, "offshore_substation.yaml")
    assert substation == correct_substation

    # Test loading CTV1 from the IEA26 project
    library = library_map["IEA26"]
    folder = "vessels"
    correct_ctv = {
        "name": "Crew Transfer Vessel 1",
        "equipment_rate": 3500,
        "start_month": 1,
        "start_day": 1,
        "end_month": 12,
        "end_day": 31,
        "start_year": 1996,
        "end_year": 2015,
        "onsite": True,
        "capability": "CTV",
        "workday_start": 7,
        "workday_end": 18,
        "max_severity": 10,
        "mobilization_cost": 0,
        "mobilization_days": 0,
        "speed": 40.74,
        "max_windspeed_transport": 16,
        "max_windspeed_repair": 16,
        "max_waveheight_transport": 2,
        "max_waveheight_repair": 2,
        "strategy": "scheduled",
        "crew_transfer_time": 0.25,
        "n_crews": 1,
        "crew": {
            "day_rate": 0,
            "n_day_rate": 0,
            "hourly_rate": 0,
            "n_hourly_rate": 0,
        },
    }

    ctv = load_yaml(library / folder, "ctv1.yaml")
    assert ctv == correct_ctv

    # Test loading CTV1 from the IEA_26 project
    library = library_map["IEA_26"]
    folder = "vessels"
    correct_ctv = {
        "name": "Crew Transfer Vessel 1",
        "equipment_rate": 3500,
        "start_month": 1,
        "start_day": 1,
        "end_month": 12,
        "end_day": 31,
        "start_year": 1996,
        "end_year": 2015,
        "onsite": True,
        "capability": "CTV",
        "workday_start": 7,
        "workday_end": 18,
        "max_severity": 10,
        "mobilization_cost": 0,
        "mobilization_days": 0,
        "speed": 40.74,
        "max_windspeed_transport": 16,
        "max_windspeed_repair": 16,
        "max_waveheight_transport": 2,
        "max_waveheight_repair": 2,
        "strategy": "scheduled",
        "crew_transfer_time": 0.25,
        "n_crews": 1,
        "crew": {
            "day_rate": 0,
            "n_day_rate": 0,
            "hourly_rate": 0,
            "n_hourly_rate": 0,
        },
    }

    ctv = load_yaml(library / folder, "ctv1.yaml")
    assert ctv == correct_ctv
