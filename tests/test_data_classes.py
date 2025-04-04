"""Tests for wombat/core/data_classes.py."""

from __future__ import annotations

import datetime
from copy import deepcopy

import attr
import numpy as np
import pytest
import numpy.testing as npt
from dateutil.relativedelta import relativedelta

from wombat.core.library import load_yaml
from wombat.utilities.time import convert_dt_to_hours
from wombat.core.data_classes import (
    Failure,
    FixedCosts,
    Maintenance,
    ServiceCrew,
    StrategyMap,
    EquipmentMap,
    FromDictMixin,
    RepairRequest,
    SubassemblyData,
    ServiceEquipmentData,
    ScheduledServiceEquipmentData,
    UnscheduledServiceEquipmentData,
    valid_hour,
    convert_to_list,
    annual_date_range,
    clean_string_input,
    convert_to_list_lower,
    validate_0_1_inclusive,
    convert_ratio_to_absolute,
)

from tests.conftest import (
    RNG,
    SCHEDULED_VESSEL,
    GENERATOR_SUBASSEMBLY,
    UNSCHEDULED_VESSEL_DOWNTIME,
    UNSCHEDULED_VESSEL_REQUESTS,
)


def test_convert_to_list():
    """Tests ``convert_to_list``."""
    # Tests for int conversion to list[int]
    correct_conversion = [1]
    assert convert_to_list(1) == correct_conversion

    # Tests for int conversion to list[float]
    correct_conversion = [234.56]
    assert convert_to_list(234.56) == correct_conversion

    # Tests for int conversion to list[str]
    correct_conversion = ["word"]
    assert convert_to_list("word") == correct_conversion

    # Tests for int conversion to list[str] with upper case string conversion
    correct_conversion = ["WORD"]
    assert convert_to_list("word", manipulation=str.upper) == correct_conversion


def test_clean_string_input():
    """Tests ``clean_string_input``."""
    correct = "this is a statement."
    assert clean_string_input(" THIS is a STATEMENT.    ") == correct


def test_annual_date_range_good_endpoints():
    """Tests ``annual_date_range``."""
    correct_date_range = [
        datetime.date(2019, 12, 12),
        datetime.date(2019, 12, 13),
        datetime.date(2019, 12, 14),
        datetime.date(2020, 12, 12),
        datetime.date(2020, 12, 13),
        datetime.date(2020, 12, 14),
        datetime.date(2021, 12, 12),
        datetime.date(2021, 12, 13),
        datetime.date(2021, 12, 14),
    ]
    correct_date_range = np.array(correct_date_range)
    date_range = annual_date_range(12, 14, 12, 12, 2019, 2021)
    npt.assert_equal(date_range, correct_date_range)

    # Test that bad year endpoints fail
    with pytest.raises(ValueError):
        annual_date_range(12, 14, 12, 12, 2022, 2021)

    # Test that bad month endpoints fail
    with pytest.raises(ValueError):
        annual_date_range(12, 14, 12, 1, 2019, 2021)

    # Test that bad month and day endpoints fail
    with pytest.raises(ValueError):
        annual_date_range(14, 12, 12, 12, 2019, 2021)


def test_convert_ratio_to_absolute():
    """Tests ``convert_ratio_to_absolute``."""
    # Test for <= 1
    ratio = 0.9
    total = 100
    correct_amount = 90.0
    assert convert_ratio_to_absolute(ratio, total) == correct_amount

    # Test for > 1, but small
    ratio = 1.1
    total = 100
    correct_amount = 1.1
    assert convert_ratio_to_absolute(ratio, total) == correct_amount

    # Test for large number
    ratio = 50
    total = 100
    correct_amount = 50.0
    assert convert_ratio_to_absolute(ratio, total) == correct_amount


def test_valid_hour():
    """Tests ``valid_hour``. The function being checked is an attrs validator, and is
    used in conjunction with a integer conversion, so all passed values will be forced
    to intgers, and will therefore not be checking that here.
    """

    @attr.s(auto_attribs=True)
    class HourClass:
        """Dummy class for testing ``valid_hour``."""

        hour: int = attr.ib(converter=int, validator=valid_hour)

    # Test for out of bounds below hour range.
    with pytest.raises(ValueError):
        HourClass(hour=-2)

    # Test for out of bounds above hour range.
    with pytest.raises(ValueError):
        HourClass(hour=25)

    # Test for valid null.
    hour = HourClass(hour=-1)
    assert hour.hour == -1

    # Test for valid start of bounds hour.
    hour = HourClass(hour=0)
    assert hour.hour == 0

    # Test for valid end of bounds hour.
    hour = HourClass(hour=24)
    assert hour.hour == 24


def test_validate_0_1_inclusive():
    """Tests the ``validate_0_1_inclusive`` validator."""

    @attr.s(auto_attribs=True)
    class ReductionClass:
        """Dummy class for testing ``validate_0_1_inclusive``."""

        speed_reduction: int = attr.ib(
            converter=float, validator=validate_0_1_inclusive
        )

    # Test the fringes
    with pytest.raises(ValueError):
        ReductionClass(-0.000001)
    with pytest.raises(ValueError):
        ReductionClass(-1)
    with pytest.raises(ValueError):
        ReductionClass(1.000001)
    with pytest.raises(ValueError):
        ReductionClass(2)

    # Test that valid positive decimals work
    r = ReductionClass(0.2)
    assert r.speed_reduction == 0.2

    r = ReductionClass(0.999)
    assert r.speed_reduction == 0.999


def test_FromDictMixin():
    """Test the ``FromDictMixin`` mix in class."""

    @attr.s(auto_attribs=True)
    class DictClass(FromDictMixin):
        """A dummy class for testing the ``FromDictMixin`` methods."""

        x: int = attr.ib(converter=int)
        y: float = attr.ib(default=1, converter=float)
        z: list[str] = attr.ib(default=["empty"], converter=convert_to_list_lower)

    # Test for default values
    inputs = {"x": 2}
    cls = DictClass.from_dict(inputs)
    assert cls == DictClass(x=2)
    assert cls.x == 2
    assert cls.y == 1.0
    assert cls.z == ["empty"]

    # Test custom inputs and that kwarg overloading is allowed without storing
    # the extra inputs
    inputs = {"x": 3, "y": 3.2, "z": ["one", "two"], "arr": [3, 4, 5.5]}
    cls = DictClass.from_dict(inputs)
    assert inputs["x"] == cls.x
    assert inputs["y"] == cls.y
    assert inputs["z"] == cls.z
    assert getattr(cls, "arr", None) is None

    # Test that an error is raised if the required inputs are not passed
    with pytest.raises(AttributeError):
        DictClass.from_dict({})


def test_Maintenance_conversion():
    """Tests the `Maintenance` class for all inputs being provided and converted."""
    start = datetime.datetime(2000, 1, 1)
    end = datetime.datetime(2001, 1, 1)
    max_run_time = (end - start).days * 24
    inputs_all = {
        "description": "test",
        "time": 14,
        "materials": 100,
        "frequency": 200,
        "frequency_basis": "days",
        "start_date": "02-01-2000",
        "service_equipment": "ctv",
        "operation_reduction": 0.5,
        "system_value": 100000,
    }
    cls = Maintenance.from_dict(inputs_all)
    cls._update_event_timing(start, end, max_run_time)
    assert cls.description == "test"
    assert cls.time == 14.0
    assert cls.materials == 100.0
    assert cls.frequency == relativedelta(days=200)
    assert cls.frequency_basis == "days"
    assert cls.start_date == datetime.datetime(2000, 2, 1)
    assert cls.hours_to_next_event(start) == (31 * 24, False)
    assert cls.service_equipment == ["CTV"]
    assert cls.operation_reduction == 0.5
    assert cls.system_value == 100000.0


def test_Maintenance_frequency():
    """Tests the `Maintenance` class ``frequency`` and ``frequency_basis`` handling."""
    start = datetime.datetime(2000, 1, 1)
    end = datetime.datetime(2001, 1, 1)
    max_run_time = (end - start).days * 24

    # Test maximum time limit replaces a 0-valued input
    inputs_all = {
        "description": "test",
        "time": 14,
        "materials": 100,
        "frequency": 0,
        "frequency_basis": "days",
        "service_equipment": "ctv",
        "operation_reduction": 0.5,
        "system_value": 100000,
    }
    cls = Maintenance.from_dict(inputs_all)
    cls._update_event_timing(start, end, max_run_time)
    assert cls.frequency == relativedelta(days=(end - start).days)
    assert cls.frequency_basis == "date-hours"

    # Test the conversion for standard, non-date inputs
    start = datetime.datetime(2000, 2, 1)
    end = datetime.datetime(2001, 1, 1)
    max_run_time = (end - start).days * 24

    # Prior starting date gets adjusted to first post-simulation start
    inputs_all["frequency"] = 3
    inputs_all["frequency_basis"] = "months"
    inputs_all["start_date"] = "01/01/2000"
    expected_first_dt = datetime.datetime(2000, 4, 1)
    cls = Maintenance.from_dict(inputs_all)
    cls._update_event_timing(start, end, max_run_time)
    assert cls.frequency == relativedelta(months=3)
    assert cls.frequency_basis == "months"
    assert cls.start_date == expected_first_dt
    assert cls.event_dates == []
    current_dt = start
    expected_hours = convert_dt_to_hours(expected_first_dt - current_dt)
    assert (expected_hours, False) == cls.hours_to_next_event(current_dt)

    inputs_all["frequency"] = 4
    inputs_all["frequency_basis"] = "years"
    inputs_all["start_date"] = None
    cls = Maintenance.from_dict(inputs_all)
    cls._update_event_timing(start, end, max_run_time)
    assert cls.frequency == relativedelta(years=4)
    current_dt = datetime.datetime(2000, 2, 1)
    expected_dt = datetime.datetime(2004, 2, 1)
    expected_hours = convert_dt_to_hours(expected_dt - current_dt)
    assert (expected_hours, False) == cls.hours_to_next_event(current_dt)
    current_dt = datetime.datetime(2002, 2, 1)
    expected_dt = datetime.datetime(2006, 2, 1)
    expected_hours = convert_dt_to_hours(expected_dt - current_dt)
    assert (expected_hours, False) == cls.hours_to_next_event(current_dt)

    # Staggered start date for timing based
    start = datetime.datetime(2000, 1, 1)
    end = datetime.datetime(2010, 12, 31)
    max_run_time = (end - start).days * 24
    inputs_all["frequency"] = 1
    inputs_all["frequency_basis"] = "years"
    inputs_all["start_date"] = "01/01/2008"
    cls = Maintenance.from_dict(inputs_all)
    cls._update_event_timing(start, end, max_run_time)
    expected_dt = datetime.datetime(2008, 1, 1)
    assert cls.start_date == expected_dt
    current_dt = start
    expected_hours = convert_dt_to_hours(expected_dt - current_dt)
    assert (expected_hours, False) == cls.hours_to_next_event(current_dt)

    # Test the conversion for date-based inputs
    # Months and basic start date
    start = datetime.datetime(2000, 2, 1)
    end = datetime.datetime(2001, 1, 1)
    max_run_time = (end - start).days * 24
    inputs_all["frequency"] = 3
    inputs_all["frequency_basis"] = "date-months"
    inputs_all["start_date"] = "01/01/2000"
    cls = Maintenance.from_dict(inputs_all)
    cls._update_event_timing(start, end, max_run_time)
    assert cls.frequency == relativedelta(months=3)
    assert cls.event_dates == [
        datetime.datetime(2000, 4, 1),
        datetime.datetime(2000, 7, 1),
        datetime.datetime(2000, 10, 1),
        datetime.datetime(2001, 1, 1),
    ]
    current_dt = datetime.datetime(2000, 12, 1)
    expected_dt = datetime.datetime(2001, 1, 1)
    expected_hours = convert_dt_to_hours(expected_dt - current_dt)
    assert (expected_hours, True) == cls.hours_to_next_event(current_dt)

    # Years and default start date
    start = datetime.datetime(2000, 2, 1)
    end = datetime.datetime(2006, 1, 1)
    max_run_time = (end - start).days * 24
    inputs_all["frequency"] = 4
    inputs_all["frequency_basis"] = "date-years"
    inputs_all["start_date"] = None
    cls = Maintenance.from_dict(inputs_all)
    cls._update_event_timing(start, end, max_run_time)
    assert cls.frequency == relativedelta(years=4)
    assert cls.event_dates == [
        datetime.datetime(2004, 2, 1),
        datetime.datetime(2008, 2, 1),
    ]
    current_dt = start
    expected_dt = datetime.datetime(2004, 2, 1)
    expected_hours = convert_dt_to_hours(expected_dt - current_dt)
    assert (expected_hours, True) == cls.hours_to_next_event(current_dt)

    # Years and staggered start date
    inputs_all["start_date"] = "06-30-2002"
    cls = Maintenance.from_dict(inputs_all)
    cls._update_event_timing(start, end, max_run_time)
    assert cls.frequency == relativedelta(years=4)
    assert cls.event_dates == [
        datetime.datetime(2002, 6, 30),
        datetime.datetime(2006, 6, 30),
    ]

    # Days and prior start date
    start = datetime.datetime(2000, 1, 1)
    end = datetime.datetime(2001, 1, 1)
    max_run_time = (end - start).days * 24
    inputs_all["frequency"] = 180
    inputs_all["frequency_basis"] = "date-days"
    inputs_all["start_date"] = "12/31/1999"
    cls = Maintenance.from_dict(inputs_all)
    cls._update_event_timing(start, end, max_run_time)
    assert cls.event_dates == [
        datetime.datetime(2000, 6, 28),
        datetime.datetime(2000, 12, 25),
        datetime.datetime(2001, 6, 23),
    ]


def test_Maintenance_defaults():
    """Tests the `Maintenance` class for default values being populated."""
    start = datetime.datetime(2000, 1, 1)
    end = datetime.datetime(2001, 1, 1)
    max_run_time = (end - start).days * 24
    inputs_defaults = {
        "time": 14,
        "materials": 100,
        "frequency": 200,
        "service_equipment": "ctv",
        "system_value": 100000,
    }
    cls = Maintenance.from_dict(inputs_defaults)
    cls._update_event_timing(start, end, max_run_time)
    class_data = attr.fields_dict(Maintenance)
    assert cls.description == class_data["description"].default
    assert cls.time == 14.0
    assert cls.materials == 100.0
    assert cls.frequency == relativedelta(days=200)
    assert cls.frequency_basis == "days"
    assert cls.start_date == start
    assert cls.event_dates == []
    assert cls.hours_to_next_event(start) == (200.0 * 24, False)
    assert cls.service_equipment == ["CTV"]
    assert cls.operation_reduction == class_data["operation_reduction"].default
    assert cls.system_value == 100000.0


def test_Maintenance_no_events():
    """Tests the `Maintenance` class for when there is no modeled maintenance."""
    start = datetime.datetime(2000, 1, 1)
    end = datetime.datetime(2001, 1, 1)
    max_run_time = (end - start).days * 24
    inputs = {
        "time": 0,
        "materials": 0,
        "frequency": 0,
        "service_equipment": "ctv",
        "system_value": 0,
    }
    cls = Maintenance.from_dict(inputs)
    cls._update_event_timing(start, end, max_run_time)
    class_data = attr.fields_dict(Maintenance)
    assert cls.description == class_data["description"].default
    assert cls.time == 0
    assert cls.materials == 0
    assert cls.frequency == relativedelta(hours=max_run_time)
    assert cls.frequency_basis == "date-hours"
    assert cls.start_date == start
    assert cls.event_dates == [end + relativedelta(days=1)]
    assert cls.hours_to_next_event(start) == (max_run_time + 24, True)
    assert cls.service_equipment == ["CTV"]
    assert cls.operation_reduction == class_data["operation_reduction"].default
    assert cls.system_value == 0


def test_Maintenance_proportional_materials():
    """Tests the `Maintenance` classfor proportional materials cost, relative to system
    value.
    """
    start = datetime.datetime(2000, 1, 1)
    end = datetime.datetime(2001, 1, 1)
    max_run_time = (end - start).days * 24
    inputs_system_value = {
        "description": "test",
        "time": 14,
        "materials": 0.25,
        "frequency": 200,
        "service_equipment": ["ctv", "dsv"],
        "operation_reduction": 0.5,
        "system_value": 100000,
    }
    cls = Maintenance.from_dict(inputs_system_value)
    cls._update_event_timing(start, end, max_run_time)
    assert cls.description == "test"
    assert cls.time == 14.0
    assert cls.materials == 25000.0
    assert cls.frequency == relativedelta(days=200)
    assert cls.hours_to_next_event(start) == (200.0 * 24, False)
    assert cls.service_equipment == ["CTV", "DSV"]
    assert cls.operation_reduction == 0.5
    assert cls.system_value == 100000.0


def test_Maintenance_assign_id():
    """Tests the `Maintenance` class."""
    start = datetime.datetime(2000, 1, 1)
    end = datetime.datetime(2001, 1, 1)
    max_run_time = (end - start).days * 24
    inputs_system_value = {
        "description": "test",
        "time": 14,
        "materials": 0.25,
        "frequency": 200,
        "service_equipment": ["ctv", "dsv"],
        "operation_reduction": 0.5,
        "system_value": 100000,
    }
    cls = Maintenance.from_dict(inputs_system_value)
    cls._update_event_timing(start, end, max_run_time)
    correct_id = "M00001"
    cls.assign_id(request_id=correct_id)
    assert cls.request_id == correct_id

    # Test class is frozen by trying to set any attribute
    with pytest.raises(attr.exceptions.FrozenInstanceError):
        cls.time = 100


def test_Failure():
    """Tests the `Failure` class."""
    # Test that all inputs work
    inputs_all = {
        "scale": 0.2,
        "shape": 1.0,
        "time": 45,
        "materials": 200,
        "operation_reduction": 0.2,
        "level": 0,
        "service_equipment": "lcn",
        "system_value": 100000,
        "description": "test",
        "rng": RNG,
    }
    np.random.seed(0)
    cls = Failure.from_dict(inputs_all)
    assert cls.scale == 0.2
    assert cls.shape == 1.0
    assert cls.time == 45.0
    assert cls.materials == 200.0
    assert cls.operation_reduction == 0.2
    assert cls.level == 0
    assert cls.service_equipment == ["LCN"]
    assert cls.system_value == 100000
    assert cls.description == "test"
    # TODO: UPDATE THIS BEFORE PR
    # assert cls.hours_to_next_event(0, 0) == 1394.372138301769, False

    # Test that the default values work
    inputs_all = {
        "scale": 0.2,
        "shape": 1.0,
        "time": 45,
        "materials": 200,
        "operation_reduction": 0.2,
        "level": 0,
        "service_equipment": "lcn",
        "system_value": 100000,
        "rng": RNG,
    }
    cls = Failure.from_dict(inputs_all)
    class_data = attr.fields_dict(Failure)
    assert cls.scale == 0.2
    assert cls.shape == 1.0
    assert cls.time == 45.0
    assert cls.materials == 200.0
    assert cls.operation_reduction == 0.2
    assert cls.level == 0
    assert cls.service_equipment == ["LCN"]
    assert cls.system_value == 100000
    assert cls.description == class_data["description"].default

    # Test that the proportional materials cost works
    inputs_all = {
        "scale": 0.2,
        "shape": 1.0,
        "time": 45,
        "materials": 0.01,
        "operation_reduction": 0.2,
        "level": 0,
        "service_equipment": "lcn",
        "system_value": 100000,
        "rng": RNG,
    }
    cls = Failure.from_dict(inputs_all)
    class_data = attr.fields_dict(Failure)
    assert cls.scale == 0.2
    assert cls.shape == 1.0
    assert cls.time == 45.0
    assert cls.materials == 1000.0
    assert cls.operation_reduction == 0.2
    assert cls.level == 0
    assert cls.service_equipment == ["LCN"]
    assert cls.system_value == 100000
    assert cls.description == class_data["description"].default

    # Test for assign_id
    correct_id = "R00001"
    cls.assign_id(request_id=correct_id)
    assert cls.request_id == correct_id

    # Test for a missing value
    inputs_missing = {
        "shape": 1.0,
        "time": 45,
        "materials": 200,
        "operation_reduction": 0.2,
        "level": 0,
        "service_equipment": "lcn",
        "system_value": 100000,
    }
    with pytest.raises(AttributeError):
        Failure.from_dict(inputs_missing)

    # Tests that with no weibull parameters, no timeouts will happen
    inputs_all = {
        "scale": 0.0,
        "shape": 0.0,
        "time": 45,
        "materials": 200,
        "operation_reduction": 0.2,
        "level": 0,
        "service_equipment": "lcn",
        "system_value": 100000,
        "description": "test",
        "rng": RNG,
    }
    cls = Failure.from_dict(inputs_all)
    assert cls.hours_to_next_event() is None

    # TODO: Write a test for the weibull function


def test_SubassemblyData():
    """Tests the `SubassemblyData` class."""
    N_maintenance = len(GENERATOR_SUBASSEMBLY["maintenance"])
    failure_levels = [f["level"] for f in GENERATOR_SUBASSEMBLY["failures"]]
    N_failure = len(failure_levels)

    subassembly = SubassemblyData.from_dict(GENERATOR_SUBASSEMBLY)
    maintenance_list = [
        Maintenance.from_dict(m) for m in GENERATOR_SUBASSEMBLY["maintenance"]
    ]
    failure_list = [
        Failure.from_dict({**f, "rng": RNG}) for f in GENERATOR_SUBASSEMBLY["failures"]
    ]

    assert subassembly.name == GENERATOR_SUBASSEMBLY["name"]
    assert subassembly.system_value == GENERATOR_SUBASSEMBLY["system_value"]
    assert len(subassembly.maintenance) == N_maintenance

    # Set the request ID for all maintenance tasks to avoid an AttributeError
    request = "M0000001"
    for m in maintenance_list:
        m.assign_id(request_id=request)
    for m in subassembly.maintenance:
        m.assign_id(request_id=request)
    assert subassembly.maintenance == maintenance_list
    assert len(subassembly.failures) == N_failure
    assert [f.level for f in subassembly.failures] == failure_levels

    # Set the request ID for all failure tasks to avoid an AttributeError
    request = "R000000"
    for i, failure in enumerate(failure_list):
        rid = f"{request}{i}"
        failure.assign_id(request_id=rid)
        subassembly.failures[i].assign_id(request_id=rid)
        assert failure_list[i] == subassembly.failures[i]


def test_RepairRequest():
    """Tests the `RepairRequest` class."""
    failure = GENERATOR_SUBASSEMBLY["failures"][1]
    failure.update({"rng": RNG})
    maintenance = GENERATOR_SUBASSEMBLY["maintenance"][0]

    # Test for a failure repair request
    request = {
        "system_id": "WTG-01",
        "system_name": "turbine 1",
        "subassembly_id": "generator",
        "subassembly_name": "generator",
        "severity_level": failure["level"],
        "details": Failure.from_dict(failure),
        "cable": 1,
    }
    cls = RepairRequest.from_dict(request)
    assert cls.system_id == request["system_id"]
    assert cls.system_name == request["system_name"]
    assert cls.subassembly_id == request["subassembly_id"]
    assert cls.subassembly_name == request["subassembly_name"]
    assert cls.severity_level == 2
    assert cls.cable
    assert cls.upstream_turbines == []

    # Set the id
    mid = "M00001"
    cls.assign_id(mid)
    request["details"].assign_id(mid)
    assert cls.details == request["details"]

    # Test for a maintenance repair request and default cable value
    request = {
        "system_id": "WTG-01",
        "system_name": "turbine 1",
        "subassembly_id": "generator",
        "subassembly_name": "generator",
        "severity_level": 0,
        "details": Maintenance.from_dict(maintenance),
    }
    cls = RepairRequest.from_dict(request)
    assert cls.system_id == request["system_id"]
    assert cls.system_name == request["system_name"]
    assert cls.subassembly_id == request["subassembly_id"]
    assert cls.subassembly_name == request["subassembly_name"]
    assert cls.severity_level == 0
    assert not cls.cable
    assert cls.upstream_turbines == []

    # Set the id
    mid = "R00001"
    cls.assign_id(mid)
    request["details"].assign_id(mid)
    assert cls.details == request["details"]

    # Test that details needs to be a Maintenance or Failure class
    request = {
        "system_id": "WTG-01",
        "system_name": "turbine 1",
        "subassembly_id": "generator",
        "subassembly_name": "generator",
        "severity_level": 0,
        "details": "",
    }
    with pytest.raises(TypeError):
        cls = RepairRequest.from_dict(request)


def test_ServiceCrew():
    """Tests the `ServiceCrew` class."""
    inputs = {"n_day_rate": 4, "day_rate": 100, "n_hourly_rate": 10, "hourly_rate": 6.1}
    crew = ServiceCrew.from_dict(inputs)
    assert crew.n_day_rate == 4
    assert crew.n_hourly_rate == 10
    assert crew.day_rate == 100.0
    assert crew.hourly_rate == 6.1


def test_ServiceEquipmentData_determine_type():
    """Tests the creation of the servicing equipment data classes."""
    # Test creation with requests strategy as kwarg
    vessel = ServiceEquipmentData(
        UNSCHEDULED_VESSEL_REQUESTS, strategy="requests"
    ).determine_type()
    assert isinstance(vessel, UnscheduledServiceEquipmentData)

    # Test creation with requests strategy without kwarg
    vessel = ServiceEquipmentData(UNSCHEDULED_VESSEL_REQUESTS).determine_type()
    assert isinstance(vessel, UnscheduledServiceEquipmentData)

    # Test creation with downtime strategy as kwarg
    vessel = ServiceEquipmentData(
        UNSCHEDULED_VESSEL_DOWNTIME, strategy="downtime"
    ).determine_type()
    assert isinstance(vessel, UnscheduledServiceEquipmentData)

    # Test creation with downtime strategy without kwarg
    vessel = ServiceEquipmentData(UNSCHEDULED_VESSEL_DOWNTIME).determine_type()
    assert isinstance(vessel, UnscheduledServiceEquipmentData)

    # Test creation with scheduled strategy as kwarg
    vessel = ServiceEquipmentData(
        SCHEDULED_VESSEL, strategy="scheduled"
    ).determine_type()
    assert isinstance(vessel, ScheduledServiceEquipmentData)

    # Test creation with scheduled strategy without kwarg
    vessel = ServiceEquipmentData(SCHEDULED_VESSEL).determine_type()
    assert isinstance(vessel, ScheduledServiceEquipmentData)

    # Test for error with mismatched definitions. This is a ValueError because the
    # default values are meant to cause a failure
    with pytest.raises(ValueError):
        ServiceEquipmentData(SCHEDULED_VESSEL, strategy="requests").determine_type()

    with pytest.raises(ValueError):
        ServiceEquipmentData(SCHEDULED_VESSEL, strategy="downtime").determine_type()

    with pytest.raises(ValueError):
        ServiceEquipmentData(
            UNSCHEDULED_VESSEL_REQUESTS, strategy="scheduled"
        ).determine_type()

    # Test that an invalid strategy raises a ValueError
    with pytest.raises(ValueError):
        ServiceEquipmentData(
            UNSCHEDULED_VESSEL_REQUESTS, strategy="invalid"
        ).determine_type()


def test_BaseServiceEquipmentData():
    """Tests the basic setups common to un/scheduled servicing equipment."""
    # Test that no start/end dates produce empty date sets
    vessel = ScheduledServiceEquipmentData.from_dict(SCHEDULED_VESSEL)

    vessel.set_non_operational_dates(start_year=2000, end_year=2020)
    vessel.set_reduced_speed_parameters(start_year=2000, end_year=2020)
    assert vessel.non_operational_dates.size == 0
    assert vessel.non_operational_dates_set == set()
    assert vessel.reduced_speed_dates.size == 0
    assert vessel.reduced_speed_dates_set == set()

    # Test for a single date provided that errors are raised
    data = deepcopy(SCHEDULED_VESSEL)
    data["reduced_speed_start"] = "11/14"
    with pytest.raises(ValueError):
        ScheduledServiceEquipmentData.from_dict(data)

    data["reduced_speed_start"] = None
    data["non_operational_end"] = "11/14"
    with pytest.raises(ValueError):
        ScheduledServiceEquipmentData.from_dict(data)

    # Test that same dates create an error
    data["reduced_speed_start"] = "11/14"
    data["reduced_speed_end"] = "11-14"
    with pytest.raises(ValueError):
        ScheduledServiceEquipmentData.from_dict(data)

    data["reduced_speed_start"] = None
    data["reduced_speed_end"] = None
    data["non_operational_start"] = "11-14"
    data["non_operational_end"] = "11/14"
    with pytest.raises(ValueError):
        ScheduledServiceEquipmentData.from_dict(data)

    # Test valid same-year dates
    # NOTE: 3 days x 3 years should create 9 valid dates in the set
    data = deepcopy(SCHEDULED_VESSEL)
    data["reduced_speed_start"] = "11/14"
    data["reduced_speed_end"] = "11/16"

    vessel = ScheduledServiceEquipmentData.from_dict(data)
    vessel.set_reduced_speed_parameters(start_year=2000, end_year=2002)
    assert len(vessel.reduced_speed_dates) == 9
    assert min(vessel.reduced_speed_dates) == datetime.date(2000, 11, 14)
    assert max(vessel.reduced_speed_dates) == datetime.date(2002, 11, 16)

    # Test valid overlapping year dates
    # NOTE: 4 days x 3 years, but each date chunk is split between two years
    data = deepcopy(SCHEDULED_VESSEL)
    data["non_operational_start"] = "12-30"
    data["non_operational_end"] = "1/2"
    vessel = ScheduledServiceEquipmentData.from_dict(data)
    vessel.set_non_operational_dates(start_year=2000, end_year=2002)
    assert len(vessel.non_operational_dates) == 12
    assert min(vessel.non_operational_dates) == datetime.date(2000, 1, 1)
    assert max(vessel.non_operational_dates) == datetime.date(2002, 12, 31)


def test_ScheduledServiceEquipmentData():
    """Tests the initialized attributes of ScheduledServicingEquipmentData."""
    # Test that provided values are correctly mapped
    vessel = ScheduledServiceEquipmentData.from_dict(SCHEDULED_VESSEL)
    assert vessel.name == SCHEDULED_VESSEL["name"]
    assert vessel.equipment_rate == SCHEDULED_VESSEL["equipment_rate"]
    assert vessel.n_crews == SCHEDULED_VESSEL["n_crews"]
    assert vessel.crew.day_rate == SCHEDULED_VESSEL["crew"]["day_rate"]
    assert vessel.crew.n_day_rate == SCHEDULED_VESSEL["crew"]["n_day_rate"]
    assert vessel.crew.hourly_rate == SCHEDULED_VESSEL["crew"]["hourly_rate"]
    assert vessel.crew.n_hourly_rate == SCHEDULED_VESSEL["crew"]["n_hourly_rate"]
    assert vessel.start_month == SCHEDULED_VESSEL["start_month"]
    assert vessel.start_day == SCHEDULED_VESSEL["start_day"]
    assert vessel.start_year == SCHEDULED_VESSEL["start_year"]
    assert vessel.end_month == SCHEDULED_VESSEL["end_month"]
    assert vessel.end_day == SCHEDULED_VESSEL["end_day"]
    assert vessel.end_year == SCHEDULED_VESSEL["end_year"]
    assert vessel.capability == [SCHEDULED_VESSEL["capability"].upper()]
    assert vessel.mobilization_cost == SCHEDULED_VESSEL["mobilization_cost"]
    assert vessel.mobilization_days == SCHEDULED_VESSEL["mobilization_days"]
    assert vessel.speed == SCHEDULED_VESSEL["speed"]
    assert vessel.max_windspeed_transport == SCHEDULED_VESSEL["max_windspeed_transport"]
    assert vessel.max_windspeed_repair == SCHEDULED_VESSEL["max_windspeed_repair"]
    assert (
        vessel.max_waveheight_transport == SCHEDULED_VESSEL["max_waveheight_transport"]
    )
    assert vessel.max_waveheight_repair == SCHEDULED_VESSEL["max_waveheight_repair"]
    assert vessel.workday_start == SCHEDULED_VESSEL["workday_start"]
    assert vessel.workday_end == SCHEDULED_VESSEL["workday_end"]
    assert vessel.crew_transfer_time == SCHEDULED_VESSEL["crew_transfer_time"]
    assert vessel.onsite == SCHEDULED_VESSEL["onsite"]
    assert vessel.method == SCHEDULED_VESSEL["method"].lower()
    assert vessel.operating_dates[0] == datetime.date(
        SCHEDULED_VESSEL["start_year"],
        SCHEDULED_VESSEL["start_month"],
        SCHEDULED_VESSEL["start_day"],
    )
    assert vessel.operating_dates[-1] == datetime.date(
        SCHEDULED_VESSEL["end_year"],
        SCHEDULED_VESSEL["end_month"],
        SCHEDULED_VESSEL["end_day"],
    )
    assert vessel.strategy == SCHEDULED_VESSEL["strategy"]

    # Test default values
    vessel_dict = deepcopy(SCHEDULED_VESSEL)
    has_default_value = (
        "method",
        "workday_start",
        "workday_end",
        "max_waveheight_transport",
        "max_waveheight_repair",
        "crew_transfer_time",
    )
    for parameter in has_default_value:
        vessel_dict.pop(parameter)
    vessel = ScheduledServiceEquipmentData.from_dict(vessel_dict)
    assert vessel.workday_start == vessel.workday_end == -1
    assert vessel.max_waveheight_repair == vessel.max_waveheight_transport == 1000.0
    assert vessel.method == "severity"
    assert vessel.strategy == "scheduled"
    assert vessel.crew_transfer_time == 0.0

    # Test that setting the environment's shift will work
    start_shift = 8
    end_shift = 20
    vessel._set_environment_shift(start_shift, end_shift)
    assert vessel.workday_start == start_shift
    assert vessel.workday_end == end_shift

    # Test the date range works for return visits
    vessel_dict = deepcopy(SCHEDULED_VESSEL)
    vessel_dict["onsite"] = False
    vessel_dict["start_day"] = 10
    vessel_dict["end_day"] = 11
    vessel_dict["start_month"] = 2
    vessel_dict["end_month"] = 2
    vessel_dict["start_year"] = 2019
    vessel_dict["end_year"] = 2021
    correct_date_range = np.array(
        [
            datetime.date(2019, 2, 10),
            datetime.date(2019, 2, 11),
            datetime.date(2020, 2, 10),
            datetime.date(2020, 2, 11),
            datetime.date(2021, 2, 10),
            datetime.date(2021, 2, 11),
        ]
    )
    vessel = ScheduledServiceEquipmentData.from_dict(vessel_dict)
    npt.assert_equal(vessel.operating_dates, correct_date_range)


def test_UnscheduledServiceEquipmentData():
    """Tests the initialized attributes of UnscheduledServicingEquipmentData."""
    # Test that provided values are correctly mapped for the requests-based equipment
    vessel = UnscheduledServiceEquipmentData.from_dict(UNSCHEDULED_VESSEL_REQUESTS)
    assert vessel.name == UNSCHEDULED_VESSEL_REQUESTS["name"]
    assert vessel.equipment_rate == UNSCHEDULED_VESSEL_REQUESTS["equipment_rate"]
    assert vessel.charter_days == UNSCHEDULED_VESSEL_REQUESTS["charter_days"]
    assert vessel.n_crews == UNSCHEDULED_VESSEL_REQUESTS["n_crews"]
    assert vessel.crew.day_rate == UNSCHEDULED_VESSEL_REQUESTS["crew"]["day_rate"]
    assert vessel.crew.n_day_rate == UNSCHEDULED_VESSEL_REQUESTS["crew"]["n_day_rate"]
    assert vessel.crew.hourly_rate == UNSCHEDULED_VESSEL_REQUESTS["crew"]["hourly_rate"]
    assert (
        vessel.crew.n_hourly_rate
        == UNSCHEDULED_VESSEL_REQUESTS["crew"]["n_hourly_rate"]
    )
    assert vessel.capability == [UNSCHEDULED_VESSEL_REQUESTS["capability"].upper()]
    assert vessel.mobilization_cost == UNSCHEDULED_VESSEL_REQUESTS["mobilization_cost"]
    assert vessel.mobilization_days == UNSCHEDULED_VESSEL_REQUESTS["mobilization_days"]
    assert vessel.speed == UNSCHEDULED_VESSEL_REQUESTS["speed"]
    assert vessel.strategy == UNSCHEDULED_VESSEL_REQUESTS["strategy"]
    assert (
        vessel.strategy_threshold == UNSCHEDULED_VESSEL_REQUESTS["strategy_threshold"]
    )
    assert (
        vessel.max_windspeed_transport
        == UNSCHEDULED_VESSEL_REQUESTS["max_windspeed_transport"]
    )
    assert (
        vessel.max_windspeed_repair
        == UNSCHEDULED_VESSEL_REQUESTS["max_windspeed_repair"]
    )
    assert (
        vessel.max_waveheight_transport
        == UNSCHEDULED_VESSEL_REQUESTS["max_waveheight_transport"]
    )
    assert (
        vessel.max_waveheight_repair
        == UNSCHEDULED_VESSEL_REQUESTS["max_waveheight_repair"]
    )
    assert vessel.workday_start == UNSCHEDULED_VESSEL_REQUESTS["workday_start"]
    assert vessel.workday_end == UNSCHEDULED_VESSEL_REQUESTS["workday_end"]
    assert (
        vessel.crew_transfer_time == UNSCHEDULED_VESSEL_REQUESTS["crew_transfer_time"]
    )
    assert vessel.onsite == UNSCHEDULED_VESSEL_REQUESTS["onsite"]
    assert vessel.method == UNSCHEDULED_VESSEL_REQUESTS["method"].lower()

    # Test default values
    vessel_dict = deepcopy(UNSCHEDULED_VESSEL_REQUESTS)
    has_default_value = (
        "method",
        "workday_start",
        "workday_end",
        "max_waveheight_transport",
        "max_waveheight_repair",
        "crew_transfer_time",
    )
    for parameter in has_default_value:
        vessel_dict.pop(parameter)
    vessel = UnscheduledServiceEquipmentData.from_dict(vessel_dict)
    assert vessel.workday_start == vessel.workday_end == -1
    assert vessel.max_waveheight_repair == vessel.max_waveheight_transport == 1000.0
    assert vessel.method == "severity"
    assert vessel.crew_transfer_time == 0.0

    # Test that setting the environment's shift will work
    start_shift = 8
    end_shift = 20
    vessel._set_environment_shift(start_shift, end_shift)
    assert vessel.workday_start == start_shift
    assert vessel.workday_end == end_shift

    # Test the strategy and threshold bounds with edge of in bounds for requests
    vessel_dict = deepcopy(UNSCHEDULED_VESSEL_REQUESTS)
    vessel_dict["strategy_threshold"] = 1
    vessel = UnscheduledServiceEquipmentData.from_dict(vessel_dict)
    assert vessel.strategy_threshold == 1

    # Test for edge of out of bounds for requests
    vessel_dict = deepcopy(UNSCHEDULED_VESSEL_REQUESTS)
    vessel_dict["strategy_threshold"] = 0
    with pytest.raises(ValueError):
        UnscheduledServiceEquipmentData.from_dict(vessel_dict)

    # Test the strategy and threshold bounds with edge of in bounds for downtime
    vessel_dict = deepcopy(UNSCHEDULED_VESSEL_DOWNTIME)
    vessel_dict["strategy_threshold"] = 0.0000000000000001
    vessel = UnscheduledServiceEquipmentData.from_dict(vessel_dict)
    assert vessel.strategy_threshold == 0.0000000000000001

    vessel_dict = deepcopy(UNSCHEDULED_VESSEL_DOWNTIME)
    vessel_dict["strategy_threshold"] = 0.99999999999999
    vessel = UnscheduledServiceEquipmentData.from_dict(vessel_dict)
    assert vessel.strategy_threshold == 0.99999999999999

    # Test the strategy and threshold bounds with edge for out of bounds for downtime
    vessel_dict = deepcopy(UNSCHEDULED_VESSEL_DOWNTIME)
    vessel_dict["strategy_threshold"] = 0
    with pytest.raises(ValueError):
        UnscheduledServiceEquipmentData.from_dict(vessel_dict)

    vessel_dict = deepcopy(UNSCHEDULED_VESSEL_DOWNTIME)
    vessel_dict["strategy_threshold"] = 1
    with pytest.raises(ValueError):
        UnscheduledServiceEquipmentData.from_dict(vessel_dict)

    # Test that setting the environment's shift will work
    start_shift = 8
    end_shift = 20
    vessel._set_environment_shift(start_shift, end_shift)
    assert vessel.workday_start == start_shift
    assert vessel.workday_end == end_shift


def test_FixedCosts_operations_provided():
    """Tests high resolution inptus go to 0 when `operations` is provided."""
    high_res = FixedCosts(
        operations=100,  # should be 100
        operations_management_administration=100,
        project_management_administration=100,
        marine_management=111,
        weather_forecasting=22,
        condition_monitoring=47,
        operating_facilities=25,
        environmental_health_safety_monitoring=45,
        insurance=100,
        brokers_fee=44,
        operations_all_risk=100,
        business_interruption=76,
        third_party_liability=82,
        storm_coverage=98,
        annual_leases_fees=100,
        submerge_land_lease_costs=80,
        transmission_charges_rights=90,
        onshore_electrical_maintenance=60,
        labor=100,
    )

    assert high_res.operations == 100
    assert high_res.operations_management_administration == 0
    assert high_res.project_management_administration == 0
    assert high_res.marine_management == 0
    assert high_res.weather_forecasting == 0
    assert high_res.condition_monitoring == 0
    assert high_res.operating_facilities == 0
    assert high_res.environmental_health_safety_monitoring == 0
    assert high_res.insurance == 0
    assert high_res.brokers_fee == 0
    assert high_res.operations_all_risk == 0
    assert high_res.business_interruption == 0
    assert high_res.third_party_liability == 0
    assert high_res.storm_coverage == 0
    assert high_res.annual_leases_fees == 0
    assert high_res.submerge_land_lease_costs == 0
    assert high_res.transmission_charges_rights == 0
    assert high_res.onshore_electrical_maintenance == 0
    assert high_res.labor == 0


def test_FixedCosts_category_values_provided():
    """Test that High resolution inputs get overwritten in the case of providing
    category values.
    """
    high_res = FixedCosts(
        operations=0,  # should be 530
        operations_management_administration=100,
        project_management_administration=100,
        marine_management=111,
        weather_forecasting=22,
        condition_monitoring=47,
        operating_facilities=25,  # independent
        environmental_health_safety_monitoring=45,  # independent
        insurance=100,
        brokers_fee=44,
        operations_all_risk=100,
        business_interruption=76,
        third_party_liability=82,
        storm_coverage=98,
        annual_leases_fees=100,
        submerge_land_lease_costs=80,
        transmission_charges_rights=90,
        onshore_electrical_maintenance=60,  # independent
        labor=100,  # independent
    )
    assert high_res.operations == 530
    assert high_res.operations_management_administration == 100
    assert high_res.project_management_administration == 0
    assert high_res.marine_management == 0
    assert high_res.weather_forecasting == 0
    assert high_res.condition_monitoring == 0
    assert high_res.operating_facilities == 25
    assert high_res.environmental_health_safety_monitoring == 45
    assert high_res.insurance == 100
    assert high_res.brokers_fee == 0
    assert high_res.operations_all_risk == 0
    assert high_res.business_interruption == 0
    assert high_res.third_party_liability == 0
    assert high_res.storm_coverage == 0
    assert high_res.annual_leases_fees == 100
    assert high_res.submerge_land_lease_costs == 0
    assert high_res.transmission_charges_rights == 0
    assert high_res.onshore_electrical_maintenance == 60
    assert high_res.labor == 100


def test_equipment_map(env_setup):
    """Tests the ``EquipmentMap`` class."""
    env = env_setup

    # Test for a requests-based equipment
    hlv = ServiceEquipmentData(
        load_yaml(env.data_dir / "vessels", "hlv_requests.yaml")
    ).determine_type()
    mapping = EquipmentMap(equipment=hlv, strategy_threshold=hlv.strategy_threshold)
    assert mapping.equipment == hlv
    assert mapping.strategy_threshold == hlv.strategy_threshold

    # Test for a downtime-based equipment
    hlv = ServiceEquipmentData(
        load_yaml(env.data_dir / "vessels", "hlv_downtime.yaml")
    ).determine_type()
    mapping = EquipmentMap(equipment=hlv, strategy_threshold=hlv.strategy_threshold)
    assert mapping.equipment == hlv
    assert mapping.strategy_threshold == hlv.strategy_threshold


def test_strategy_map(env_setup):
    """Tests the ``StrategyMap`` class."""
    env = env_setup

    mapping = StrategyMap()
    assert mapping.CTV == []
    assert mapping.SCN == []
    assert mapping.MCN == []
    assert mapping.LCN == []
    assert mapping.CAB == []
    assert mapping.RMT == []
    assert mapping.DRN == []
    assert mapping.DSV == []
    assert mapping.TOW == []
    assert mapping.AHV == []
    assert mapping.VSG == []
    assert not mapping.is_running

    # Test for bad input
    hlv = load_yaml(env.data_dir / "vessels", "hlv_requests.yaml")
    hlv["capability"] = ["SCN", "LCN"]
    hlv = ServiceEquipmentData(hlv).determine_type()
    hlv_map = EquipmentMap(hlv.strategy_threshold, hlv)

    # List of inputs will fail
    with pytest.raises(ValueError):
        mapping.update(hlv.capability, hlv.strategy_threshold, hlv)

    # Single bad input will fail
    with pytest.raises(ValueError):
        mapping.update("CRV", hlv.strategy_threshold, hlv)

    # Check that the update did not complete
    assert not mapping.is_running

    # Create a few equipment and test for inclusion

    # HLV with multiple capabilities
    hlv = load_yaml(env.data_dir / "vessels", "hlv_requests.yaml")
    hlv["capability"] = ["SCN", "LCN"]
    hlv = ServiceEquipmentData(hlv).determine_type()
    hlv_map = EquipmentMap(hlv.strategy_threshold, hlv)
    for capability in hlv.capability:
        mapping.update(capability, hlv.strategy_threshold, hlv)
    assert mapping.SCN == mapping.LCN == [hlv_map]
    assert mapping.is_running

    # Cabling vessel with one capability
    cab = load_yaml(env.data_dir / "vessels", "cabling.yaml")
    cab = ServiceEquipmentData(cab).determine_type()
    for capability in cab.capability:
        mapping.update(capability, cab.strategy_threshold, cab)
    cab_map = EquipmentMap(cab.strategy_threshold, cab)
    assert mapping.CAB == [cab_map]
    assert mapping.is_running

    # Diving support vessel
    dsv = load_yaml(env.data_dir / "vessels", "dsv.yaml")
    dsv = ServiceEquipmentData(dsv).determine_type()
    for capability in dsv.capability:
        mapping.update(capability, dsv.strategy_threshold, dsv)
    dsv_map = EquipmentMap(dsv.strategy_threshold, dsv)
    assert mapping.DSV == [dsv_map]
    assert mapping.is_running

    # Remote reset
    rmt = load_yaml(env.data_dir / "vessels", "rmt.yaml")
    rmt = ServiceEquipmentData(rmt).determine_type()
    for capability in rmt.capability:
        mapping.update(capability, rmt.strategy_threshold, rmt)
    rmt_map = EquipmentMap(rmt.strategy_threshold, rmt)
    assert mapping.RMT == [rmt_map]
    assert mapping.is_running

    # Drone repair
    drn = load_yaml(env.data_dir / "vessels", "drn.yaml")
    drn = ServiceEquipmentData(drn).determine_type()
    for capability in drn.capability:
        mapping.update(capability, drn.strategy_threshold, drn)
    drn_map = EquipmentMap(drn.strategy_threshold, drn)
    assert mapping.DRN == [drn_map]
    assert mapping.is_running

    # Crew tranfer
    ctv = load_yaml(env.data_dir / "vessels", "ctv_requests.yaml")
    ctv = ServiceEquipmentData(ctv).determine_type()
    for capability in ctv.capability:
        mapping.update(capability, ctv.strategy_threshold, ctv)
    ctv_map = EquipmentMap(ctv.strategy_threshold, ctv)
    assert mapping.CTV == [ctv_map]
    assert mapping.is_running

    # Update an existing category with another piece of equipment
    fsv = load_yaml(env.data_dir / "vessels", "fsv_downtime.yaml")
    fsv = ServiceEquipmentData(fsv).determine_type()
    for capability in fsv.capability:
        mapping.update(capability, fsv.strategy_threshold, fsv)
    fsv_map = EquipmentMap(fsv.strategy_threshold, fsv)
    assert mapping.SCN == [hlv_map, fsv_map]
    assert mapping.is_running

    # MCM
    fsv_dict = load_yaml(env.data_dir / "vessels", "fsv_downtime.yaml")
    fsv_dict["capability"] = ["MCN"]
    mcn = ServiceEquipmentData(fsv_dict).determine_type()
    for capability in mcn.capability:
        mapping.update(capability, mcn.strategy_threshold, mcn)
    mcn_map = EquipmentMap(mcn.strategy_threshold, mcn)
    assert mapping.MCN == [mcn_map]
    assert mapping.is_running

    # TOW/AHV
    fsv_dict["capability"] = ["AHV", "TOW"]
    tow = ServiceEquipmentData(fsv_dict).determine_type()
    for capability in tow.capability:
        mapping.update(capability, tow.strategy_threshold, tow)
    tow_map = EquipmentMap(tow.strategy_threshold, tow)
    assert mapping.TOW == [tow_map]
    assert mapping.AHV == [tow_map]
    assert mapping.is_running

    # VSG
    fsv_dict["capability"] = ["VSG"]
    vsg = ServiceEquipmentData(fsv_dict).determine_type()
    for capability in vsg.capability:
        mapping.update(capability, vsg.strategy_threshold, vsg)
    vsg_map = EquipmentMap(vsg.strategy_threshold, vsg)
    assert mapping.VSG == [vsg_map]
    assert mapping.is_running


def test_FixedCosts_high_resolution_provided():
    """Test that high resolution inputs work and categories sum correctly."""
    high_res = FixedCosts(
        operations=0,  # should be 1080
        operations_management_administration=0,  # should be 280
        project_management_administration=100,
        marine_management=111,
        weather_forecasting=22,
        condition_monitoring=47,
        operating_facilities=25,  # independent
        environmental_health_safety_monitoring=45,  # independent
        insurance=0,  # should be 400
        brokers_fee=44,
        operations_all_risk=100,
        business_interruption=76,
        third_party_liability=82,
        storm_coverage=98,
        annual_leases_fees=0,  # should be 170
        submerge_land_lease_costs=80,
        transmission_charges_rights=90,
        onshore_electrical_maintenance=60,  # independent
        labor=100,  # independent
    )
    assert high_res.operations == 1080
    assert high_res.operations_management_administration == 280
    assert high_res.insurance == 400
    assert high_res.annual_leases_fees == 170
