"""Turbine and turbine component shared utilities."""


import attr  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import typing  # type: ignore
import datetime  # type: ignore
from math import fsum  # type: ignore
from typing import Any, Dict, List, Union, Callable, Optional, Sequence  # type: ignore
from scipy.stats import weibull_min  # type: ignore


HOURS_IN_YEAR = 8760
HOURS_IN_DAY = 24


def convert_to_list(
    value: Union[Sequence, Union[str, int, float]],
    manipulation: Optional[Union[Callable, None]] = None,
):
    """Converts an unknown element that could be a list or single, non-sequence element
    to a list of elements.

    Parameters
    ----------
    value : Union[Sequence, Union[str, int, float]]
        The unknown element to be converted to a list of element(s).
    manipulation: Callable
        A function to be performed upon the individual elements

    Returns
    -------
    List[Any]
        The new list of elements.
    """

    if isinstance(value, (str, int, float)):
        value = [value]
    if manipulation is not None:
        return [manipulation(el) for el in value]
    return list(value)


def str_lower(value: str) -> str:
    """Converts a string to all lower case.

    Parameters
    ----------
    value : str
        The string to turn to lower case.

    Returns
    -------
    str
        The lower case string.
    """
    return value.lower()


def str_upper(value: str) -> str:
    """Converts a string to all upper case.

    Parameters
    ----------
    value : str
        The string to turn to upper case.

    Returns
    -------
    str
        The upper case string.
    """
    return value.upper()


def clean_string_input(value: str) -> str:
    """Converts a string to lower case and and removes leading and trailing white spaces.

    Parameters
    ----------
    value: str
        The user input string.

    Returns
    -------
    str
        value.lower().strip()
    """
    return value.lower().strip()


def annual_date_range(
    start_day: int,
    end_day: int,
    start_month: int,
    end_month: int,
    start_year: int,
    end_year: int,
) -> np.ndarray:
    """Creates a series of date ranges across multiple years.

    Parameters
    ----------
    start_day : int
        Starting day of the month.
    end_day : int
        Ending day of the month.
    start_month : int
        Starting month.
    end_month : int
        Ending month.
    start_year : int
        First year of the date range.
    end_year : int
        Last year of the date range.

    Yields
    -------
    np.ndarray
        A ``numpy.ndarray`` of ``datetime.date`` objects.
    """
    date_ranges = []
    if end_month >= start_month:
        for year in range(start_year, end_year + 1):
            start = datetime.datetime(year, start_month, start_day)
            end = datetime.datetime(year, end_month, end_day)
            date_ranges.extend(pd.date_range(start, end).date)
    else:
        for year in range(start_year, end_year + 1):
            start1 = datetime.datetime(year, start_month, start_day)
            end1 = datetime.datetime(year, 12, 31)
            date_ranges.extend(pd.date_range(start1, end1).date)
            if year != end_year:
                start2 = datetime.datetime(year + 1, 1, 1)
                end2 = datetime.datetime(year + 1, end_month, end_day)
                date_ranges.extend(pd.date_range(start2, end2).date)

    date_range = np.array(date_ranges)
    return date_range


def convert_ratio_to_absolute(
    ratio: Union[int, float], total: Union[int, float]
) -> Union[int, float]:
    """Converts a proportional cost to an absolute cost.

    Parameters
    ----------
    ratio : Union[int, float]
        The proportional materials cost for a ``Maintenance`` or ``Failure``.
    total: Union[int, float]
        The turbine's replacement cost.

    Returns
    -------
    Union[int, float]
        The absolute cost of the materials.
    """

    if ratio <= 1:
        return ratio * total
    return ratio


def valid_hour(
    instance: Any, attribute: attr.Attribute, value: int  # pylint: disable=W0613
) -> None:
    """Validator that ensures that the input is a valid time or null value.

    Parameters
    ----------
    instance : Any
        A class object.
    attribute : attr.Attribute
        The attribute being validated.
    value : int
        A whole number, hour of the day or -1 (null).

    Raises
    ------
    ValueError
        Raised if ``value`` is not between -1 and 24, inclusive.
    """
    if not -1 <= value <= 24:
        raise ValueError(f"Input {attribute.name} must be between 0 and 24, inclusive.")


def check_capability(
    instance: Any,  # pylint: disable=W0613
    attribute: attr.Attribute,
    value: Union[str, List[str]],
) -> None:
    """Validator that ensures capability has a valid input.

    Parameters
    ----------
    instance : Any
        A class object.
    attribute : attr.Attribute
        The attribute being validated.
    value : Union[str, List[str]]
        The servicing equipment's capability. Should be one of the following:
         - CTV: crew transfer vehicle/vessel
         - SCN: small crane
         - LCN: large crane
         - CAB: cabling equipment/vessel
         - RMT: remote reset
         - DRN: drone
         - DSV: diving support vessel

    Raises
    ------
    ValueError
        Raised if the input is not of the valid inputs.
    """
    valid = set(("CTV", "SCN", "LCN", "CAB", "RMT", "DRN", "DSV"))
    values = set(convert_to_list(value, str.upper))
    invalid = values - valid
    if invalid:
        raise ValueError(f"Input {attribute.name} must be any combination of {valid}.")


def check_method(
    instance: Any, attribute: attr.Attribute, value: str  # pylint: disable=W0613
) -> None:
    """Validator that ensures that method is a valid input.

    Parameters
    ----------
    instance : Any
        A class object.
    attribute : attr.Attribute
        The attribute being validated.
    value : str
        The priority method for retreiving the next repair. Should be one of:
         - turbine: get all repairs on a turbine/system first, regardless of severity.
         - severity: move from highest to lowest priority, regardless of location.

    Raises
    ------
    ValueError
        Raised ``value`` is not one of valid inputs.
    """
    valid = ("turbine", "severity")
    if value not in valid:
        raise ValueError(f"Input {attribute.name} must be one of {valid}.")


@attr.s(auto_attribs=True)
class FromDictMixin:
    """A Mixin class to allow for kwargs overloading when a data class doesn't
    have a specific parameter definied. This allows passing of larger dictionaries
    to a data class without throwing an error.
    """

    @classmethod
    @typing.no_type_check
    def from_dict(cls, data: dict):
        """Maps a data dictionary to an ``attr``-defined class.

        Args:
            data : dict
                The data dictionary to be mapped.
        Returns:
            cls
                The ``attr``-defined class.
        """
        return cls(  # type: ignore
            **{a.name: data[a.name] for a in cls.__attrs_attrs__ if a.name in data}
        )


@attr.s(frozen=True, auto_attribs=True)
class Maintenance(FromDictMixin):
    """Data class to store maintenance data used in subassembly and cable modeling.

    Parameters
    ----------
    time : float
        Amount of time required to perform maintenance, in hours.
    materials : float
        Cost of materials required to perform maintenance, in USD.
    frequency : float
        Optimal number of days between performing maintenance, in days.
    service_equipment: Union[List[str], str]
        Any combination of the ``Equipment.capability`` options.
         - RMT: remote (no actual equipment BUT no special implementation)
         - DRN: drone
         - CTV: crew transfer vessel/vehicle
         - SCN: small crane (i.e., field support vessel)
         - LCN: large crane (i.e., heavy lift vessel)
         - CAB: cabling vessel/vehicle
         - DSV: diving support vessel
    system_value : Union[int, float]
        Turbine replacement value. Used if the materials cost is a proportional cost.
    description : str
        A short text description to be used for logging.
    """

    time: float
    materials: float
    frequency: float
    service_equipment: List[str] = attr.ib(
        converter=convert_to_list, validator=check_capability
    )
    system_value: Union[int, float]
    description: str = "routine maintenance"
    operation_reduction: float = 0.0
    request_id: str = attr.ib(init=False)
    level: int = attr.ib(default=0, init=False)

    def __attrs_post_init__(self):
        """Convert frequency to hours (simulation time scale) and the equipment
        requirement to a list.
        """
        object.__setattr__(self, "frequency", self.frequency * HOURS_IN_DAY)
        object.__setattr__(
            self,
            "service_equipment",
            convert_to_list(self.service_equipment, str.upper),
        )
        object.__setattr__(
            self,
            "materials",
            convert_ratio_to_absolute(self.materials, self.system_value),
        )

    def assign_id(self, request_id: str) -> None:
        """Assigns a unique identifier to the request.

        Parameters
        ----------
        request_id : str
            The ``wombat.core.RepairManager`` generated identifier.
        """
        object.__setattr__(self, "request_id", request_id)


@attr.s(frozen=True, auto_attribs=True)
class Failure(FromDictMixin):
    """Data class to store failure data used in subassembly and cable modeling.

    Parameters
    ----------
    scale : float
        Weibull scale parameter for a failure classification.
    shape : float
        Weibull shape parameter for a failure classification.
    time : float
        Amount of time required to complete the repair, in hours.
    materials : float
        Cost of the materials required to complete the repair, in $USD.
    operation_reduction : float
        Performance reduction caused by the failure, between (0, 1].
    level : int, optional
        Level of severity, will be generated in the ``ComponentData.create_severities``
        method.
    service_equipment: Union[List[str], str]
        Any combination of the ``Equipment.capability`` options.
         - RMT: remote (no actual equipment BUT no special implementation)
         - DRN: drone
         - CTV: crew transfer vessel/vehicle
         - SCN: small crane (i.e., field support vessel)
         - LCN: large crane (i.e., heavy lift vessel)
         - CAB: cabling vessel/vehicle
         - DSV: diving support vessel
    system_value : Union[int, float]
        Turbine replacement value. Used if the materials cost is a proportional cost.
    description : str
        A short text description to be used for logging.
    """

    scale: float
    shape: float
    time: float
    materials: float
    operation_reduction: float
    level: int
    service_equipment: Union[List[str], str] = attr.ib(
        converter=convert_to_list, validator=check_capability
    )
    system_value: Union[int, float]
    description: str = "failure"
    weibull: weibull_min = attr.ib(init=False)
    request_id: str = attr.ib(init=False)

    def __attrs_post_init__(self):
        """Creates the actual Weibull distribution and converts equipment requirements
        to a list.
        """
        object.__setattr__(self, "weibull", weibull_min(self.shape, scale=self.scale))
        object.__setattr__(
            self,
            "service_equipment",
            convert_to_list(self.service_equipment, str.upper),
        )
        object.__setattr__(
            self,
            "materials",
            convert_ratio_to_absolute(self.materials, self.system_value),
        )

    def hours_to_next_failure(self) -> Union[None, float]:
        """Samples the next time to failure in a Weibull distribution. If the ``scale``
        and ``shape`` parameters are set to 0, then the model will return ``None`` to cause
        the subassembly to timeout to the end of the simulation.

        Returns
        -------
        Union[None, float]
            Returns ``None`` for a non-modelled failure, or the time until the next
            simulated failure.
        """
        if self.scale == self.shape == 0:
            return None

        return self.weibull.rvs(size=1)[0] * HOURS_IN_YEAR

    def assign_id(self, request_id: str) -> None:
        """Assigns a unique identifier to the request.

        Parameters
        ----------
        request_id : str
            The ``wombat.core.RepairManager`` generated identifier.
        """
        object.__setattr__(self, "request_id", request_id)


@attr.s(frozen=True, auto_attribs=True)
class SubassemblyData(FromDictMixin):
    """Data storage and validation class for the subassemblies.

    Parameters
    ----------
    name : str
        Name of the component/subassembly.
    maintenance : List[Dict[str, Union[float, str]]]
        List of the maintenance classification dictionaries. This will be converted
        to a list of ``Maintenance`` objects in the post initialization hook.
    failures : Dict[int, Dict[str, Union[float, str]]]
        Dictionary of failure classifications in a numerical (ordinal) categorization
        order. This will be converted to a dictionary of ``Failure`` objects in the
        post initialization hook.
    system_value : Union[int, float]
        Turbine's cost of replacement. Used in case percentages of turbine cost are used
        in place of an absolute cost.
    """

    name: str
    maintenance: List[Union[Maintenance, Dict[str, Union[float, str]]]]
    failures: Dict[int, Union[Failure, Dict[str, Union[float, str]]]]
    system_value: Union[int, float]

    def __attrs_post_init__(self):
        """Converts the maintenance and failure data to ``Maintenance`` and ``Failure`` objects, respectively."""
        for kwargs in self.maintenance:
            assert isinstance(kwargs, dict)
            kwargs.update({"system_value": self.system_value})
        object.__setattr__(
            self, "maintenance", [Maintenance.from_dict(kw) for kw in self.maintenance]
        )

        for kwargs in self.failures.values():  # type: ignore
            assert isinstance(kwargs, dict)
            kwargs.update({"system_value": self.system_value})
        object.__setattr__(
            self,
            "failures",
            {level: Failure.from_dict(kw) for level, kw in self.failures.items()},
        )


@attr.s(frozen=True, auto_attribs=True)
class RepairRequest(FromDictMixin):
    """Repair/Maintenance request data class.

    Parameters
    ----------
    system_id : str
        ``System.id``.
    system_name : str
        ``System.name``.
    subassembly_id : str
        ``Subassembly.id``.
    subassembly_name : str
        ``Subassembly.name``.
    severity_level : int
        ``Maintenance.level`` or ``Failure.level``.
    details : Union[Failure, Maintenance]
        The actual data class.
    cable : bool
        Indicator that the request is for a cable, by default False.
    upstream_turbines : List[str]
        The cable's upstream turbines, by default []. No need to use this if ``cable`` == False.
    """

    system_id: str
    system_name: str
    subassembly_id: str
    subassembly_name: str
    severity_level: int
    details: Union[Failure, Maintenance]
    cable: bool = attr.ib(default=False)
    upstream_turbines: List[str] = attr.Factory(list)
    request_id: str = attr.ib(default="")

    def assign_id(self, request_id: str) -> None:
        """Assigns a unique identifier to the request.

        Parameters
        ----------
        request_id : str
            The ``wombat.core.RepairManager`` generated identifier.
        """
        object.__setattr__(self, "request_id", request_id)
        self.details.assign_id(request_id)


@attr.s(frozen=True, auto_attribs=True)
class ServiceCrew(FromDictMixin):
    """An internal data class for the indivdual crew units that are on the servicing equipment.

    Parameters
    ----------
    n_day_rate: int
        Number of salaried workers.
    day_rate: float
        Day rate for salaried workers, in USD.
    n_hourly_rate: int
        Number of hourly/subcontractor workers.
    hourly_rate: float
        Hourly labor rate for subcontractors, in USD.
    """

    n_day_rate: int = attr.ib(converter=int)
    day_rate: float = attr.ib(converter=float)
    n_hourly_rate: int = attr.ib(converter=int)
    hourly_rate: float = attr.ib(converter=float)


@attr.s(frozen=True, auto_attribs=True)
class ScheduledServiceEquipmentData(FromDictMixin):
    """The data class specification for servicing equipment that will use a pre-scheduled
    basis for returning to site.

    Parameters
    ----------
    name: str
        Name of the piece of servicing equipment.
    equipment_rate: float
        Day rate for the equipment/vessel, in USD.
    n_crews : int
        Number of crew units for the equipment.
        .. note: the input to this does not matter yet, as multi-crew functionality
        is not yet implemented.
    crew : ServiceCrew
        The crew details, see ``ServiceCrew`` for more information.
    start_month : int
        The day to start operations for the rig and crew.
    start_day : int
        The month to start operations for the rig and crew.
    start_year : int
        The year to start operations for the rig and crew.
    end_month : int
        The month to end operations for the rig and crew.
    end_day : int
        The day to end operations for the rig and crew.
    end_year : int
        The year to end operations for the rig and crew.
        ... note:: if the rig comes annually, then the enter the year for the last year
        that the rig and crew will be available.
    capability : str
        The type of capabilities the equipment contains. Must be one of:
         - RMT: remote (no actual equipment BUT no special implementation)
         - DRN: drone
         - CTV: crew transfer vessel/vehicle
         - SCN: small crane (i.e., field support vessel)
         - LCN: large crane (i.e., heavy lift vessel)
         - CAB: cabling vessel/vehicle
         - DSV: diving support vessel
    mobilization_cost : float
        Cost to mobilize the rig and crew.
    mobilization_days : int
        Number of days it takes to mobilize the equipment.
    speed : float
        Maximum transit speed, km/hr.
    max_windspeed_transport : float
        Maximum windspeed for safe transport, m/s.
    max_windspeed_repair : float
        Maximum windspeed for safe operations, m/s.
    max_waveheight_transport : float
        Maximum waveheight for safe transport, m, default 1000 (land-based).
    max_waveheight_repair : float
        Maximum waveheight for safe operations, m, default 1000 (land-based).
    workday_start : int
        The starting hour of a workshift, in 24 hour time.
    workday_end : int
        The ending hour of a workshift, in 24 hour time.
    crew_transfer_time : float
        The number of hours it takes to transfer the crew from the equipment to the
        system, e.g. how long does it take to transfer the crew from the CTV to the turbine,
        default 0.
    onsite : bool
        Indicator for if the rig and crew are based onsite.
        ... note:: if the rig and crew are onsite be sure that the start and end dates
        represent the first and last day/month of the year, respectively, and the start
        and end years represent the fist and last year in the weather file.
    method : str
        Determines if the ship will do all maximum severity repairs first or do all
        the repairs at one turbine before going to the next, by default severity.
        Should by one of "severity" or "turbine".
    """

    name: str = attr.ib(converter=str)
    equipment_rate: float = attr.ib(converter=float)
    n_crews: int = attr.ib(converter=int)
    crew: ServiceCrew = attr.ib(converter=ServiceCrew.from_dict)  # type: ignore
    start_month: int = attr.ib(converter=int)
    start_day: int = attr.ib(converter=int)
    start_year: int = attr.ib(converter=int)
    end_month: int = attr.ib(converter=int)
    end_day: int = attr.ib(converter=int)
    end_year: int = attr.ib(converter=int)
    capability: Union[List[str], str] = attr.ib(
        converter=convert_to_list, validator=check_capability
    )
    mobilization_cost: float = attr.ib(converter=float)
    mobilization_days: int = attr.ib(converter=int)
    speed: float = attr.ib(converter=float)
    max_windspeed_transport: float = attr.ib(converter=float)
    max_windspeed_repair: float = attr.ib(converter=float)
    max_waveheight_transport: float = attr.ib(default=1000.0, converter=float)
    max_waveheight_repair: float = attr.ib(default=1000.0, converter=float)
    workday_start: int = attr.ib(default=-1, converter=int, validator=valid_hour)
    workday_end: int = attr.ib(default=-1, converter=int, validator=valid_hour)
    crew_transfer_time: float = attr.ib(converter=float, default=0.0)
    onsite: bool = attr.ib(default=False, converter=bool)
    method: str = attr.ib(  # type: ignore
        default="severity",
        converter=[str, str_lower],  # type: ignore
        validator=check_method,
    )
    operating_dates: np.ndarray = attr.ib(init=False)
    strategy: str = attr.ib(default="scheduled")

    def create_date_range(self) -> np.ndarray:
        """Creates an ``np.ndarray`` of valid operational dates for the service equipment."""
        start_date = datetime.datetime(
            self.start_year, self.start_month, self.start_day
        )
        if self.end_year - self.start_year <= 1 or self.onsite:
            end_date = datetime.datetime(self.end_year, self.end_month, self.end_day)
            date_range = pd.date_range(start_date, end_date).date
        else:
            date_range = annual_date_range(
                self.start_day,
                self.end_day,
                self.start_month,
                self.end_month,
                self.start_year,
                self.end_year,
            )
        return date_range

    def _set_environment_shift(self, start: int, end: int) -> None:
        """Used to set the ``workday_start`` and ``workday_end`` to the environment's values.

        Parameters
        ----------
        start : int
            Starting hour of a workshift.
        end : int
            Ending hour of a workshift.
        """
        object.__setattr__(self, "workday_start", start)
        object.__setattr__(self, "workday_end", end)

    def __attrs_post_init__(self) -> None:
        object.__setattr__(
            self, "capability", convert_to_list(self.capability, str.upper)
        )
        object.__setattr__(self, "operating_dates", self.create_date_range())


@attr.s(frozen=True, auto_attribs=True)
class UnscheduledServiceEquipmentData(FromDictMixin):
    """The data class specification for servicing equipment that will use either a
    basis of windfarm downtime or total number of requests serviceable by the equipment.

    Parameters
    ----------
    name: str
        Name of the piece of servicing equipment.
    equipment_rate: float
        Day rate for the equipment/vessel, in USD.
    n_crews : int
        Number of crew units for the equipment.
        .. note: the input to this does not matter yet, as multi-crew functionality
        is not yet implemented.
    crew : ServiceCrew
        The crew details, see ``ServiceCrew`` for more information.
    charter_days : int
        The number of days the servicing equipment can be chartered for.
    capability : str
        The type of capabilities the equipment contains. Must be one of:
         - RMT: remote (no actual equipment BUT no special implementation)
         - DRN: drone
         - CTV: crew transfer vessel/vehicle
         - SCN: small crane (i.e., field support vessel)
         - LCN: large crane (i.e., heavy lift vessel)
         - CAB: cabling vessel/vehicle
         - DSV: diving support vessel
    mobilization_cost : float
        Cost to mobilize the rig and crew.
    mobilization_days : int
        Number of days it takes to mobilize the equipment.
    speed : float
        Maximum transit speed, km/hr.
    max_windspeed_transport : float
        Maximum windspeed for safe transport, m/s.
    max_windspeed_repair : float
        Maximum windspeed for safe operations, m/s.
    max_waveheight_transport : float
        Maximum waveheight for safe transport, m, default 1000 (land-based).
    max_waveheight_repair : float
        Maximum waveheight for safe operations, m, default 1000 (land-based).
    strategy : str
        For any unscheduled maintenance servicing equipment, this determines the
        strategy for dispatching. Should be on of "downtime" or "requests".
    strategy_threshold : str
        For downtime-based scenarios, this is based on the operating level, and should
        be in the range (0, 1). For reqest-based scenarios, this is the maximum number
        of requests that are allowed to build up for any given type of unscheduled
        servicing equipment, should be an integer >= 1.
    workday_start : int
        The starting hour of a workshift, in 24 hour time.
    workday_end : int
        The ending hour of a workshift, in 24 hour time.
    crew_transfer_time : float
        The number of hours it takes to transfer the crew from the equipment to the
        system, e.g. how long does it take to transfer the crew from the CTV to the turbine,
        default 0.
    onsite : bool
        Indicator for if the rig and crew are based onsite.
        ... note:: if the rig and crew are onsite be sure that the start and end dates
        represent the first and last day/month of the year, respectively, and the start
        and end years represent the fist and last year in the weather file.
    method : str
        Determines if the ship will do all maximum severity repairs first or do all
        the repairs at one turbine before going to the next, by default severity.
        Should by one of "severity" or "turbine".
    """

    name: str = attr.ib(converter=str)
    equipment_rate: float = attr.ib(converter=float)
    n_crews: int = attr.ib(converter=int)
    crew: ServiceCrew = attr.ib(converter=ServiceCrew.from_dict)  # type: ignore
    charter_days: int = attr.ib(converter=int)
    capability: Union[List[str], str] = attr.ib(
        converter=convert_to_list, validator=check_capability
    )
    mobilization_cost: float = attr.ib(converter=float)
    mobilization_days: int = attr.ib(converter=int)
    speed: float = attr.ib(converter=float)
    max_windspeed_transport: float = attr.ib(converter=float)
    max_windspeed_repair: float = attr.ib(converter=float)
    strategy: str = attr.ib(converter=clean_string_input)
    strategy_threshold: Union[int, float] = attr.ib()
    max_waveheight_transport: float = attr.ib(default=1000)
    max_waveheight_repair: float = attr.ib(default=1000)
    workday_start: int = attr.ib(default=-1, converter=int, validator=valid_hour)
    workday_end: int = attr.ib(default=-1, converter=int, validator=valid_hour)
    crew_transfer_time: float = attr.ib(converter=float, default=0.0)
    onsite: bool = attr.ib(default=False)
    method: str = attr.ib(  # type: ignore
        default="severity",
        converter=[str, str_lower],  # type: ignore
        validator=check_method,
    )

    @strategy.validator  # type: ignore
    def _validate_strategy(  # pylint: disable=R0201
        self, attribute: attr.Attribute, value: str  # pylint: disable=W0613
    ) -> None:
        """Determines if the provided strategy is valid"""
        valid = ("requests", "downtime")
        if value not in valid:
            raise ValueError(
                "``strategy`` for unscheduled servicing equipment must be one of",
                "'requests' or 'downtime'!",
            )

    @strategy_threshold.validator  # type: ignore
    def _validate_threshold(
        self,
        attribute: attr.Attribute,  # pylint: disable=W0613
        value: Union[int, float],
    ) -> None:
        """Ensures a valid threshold is provided for a given ``strategy``."""
        if not isinstance(value, (int, float)):
            raise TypeError("``strategy_threshold`` must be an ``int`` or ``float``!")

        if self.strategy == "downtime":
            if value <= 0 or value >= 1:
                raise ValueError(
                    "Downtime-based strategies must have a ``strategy_threshold``",
                    "between 0 and 1, non-inclusive!",
                )
        if self.strategy == "requests":
            if value <= 0:
                raise ValueError(
                    "Requests-based strategies must have a ``strategy_threshold``",
                    "greater than 0!",
                )

    def _set_environment_shift(self, start: int, end: int) -> None:
        """Used to set the ``workday_start`` and ``workday_end`` to the environment's values.

        Parameters
        ----------
        start : int
            Starting hour of a workshift.
        end : int
            Ending hour of a workshift.
        """
        object.__setattr__(self, "workday_start", start)
        object.__setattr__(self, "workday_end", end)

    def __attrs_post_init__(self) -> None:
        object.__setattr__(
            self, "capability", convert_to_list(self.capability, str.upper)
        )


@attr.s(frozen=True, auto_attribs=True)
class ServiceEquipmentData(FromDictMixin):
    """Helps to determine the type ServiceEquipment that should be used, based on the
    repair strategy for its operation. See
    :py:class:`~data_classes.ScheduledServiceEquipmentData` or
    :py:class:`~data_classes.UnscheduledServiceEquipmentData` for more details on each
    classifcation.

    Parameters
    ----------
    data_dict : dict
        The dictionary that will be used to create the appropriate ServiceEquipmentData.
        This should contain a field called 'strategy' with either "scheduled" or
        "unscheduled" as a value if strategy is not provided as a keyword argument.
    strategy : str, optional
        Should be one of "scheduled", "requests", "downtime". If nothing is provided,
        the equipment configuration will be checked.

    Raises
    ------
    ValueError
        If ``strategy`` is not one of "scheduled" or "unscheduled" an error will be raised.

    Examples
    --------
    The below workflow is how a new data :py:class:`~data_classes.ScheduledServiceEquipmentData`
    object could be created via a generic/routinized creation method, and is how the
    :py:class:`~service_equipment.ServiceEquipment`'s ``__init__`` method creates the
    settings data.

    >>> from wombat.core.data_classes import  ServiceEquipmentData
    >>>
    >>> data_dict = {
    >>>     "name": "Crew Transfer Vessel 1",
    >>>     "equipment_rate": 1750,
    >>>     "start_month": 1,
    >>>     "start_day": 1,
    >>>     "end_month": 12,
    >>>     "end_day": 31,
    >>>     "start_year": 2002,
    >>>     "end_year": 2014,
    >>>     "onsite": True,
    >>>     "capability": "CTV",
    >>>     "max_severity": 10,
    >>>     "mobilization_cost": 0,
    >>>     "mobilization_days": 0,
    >>>     "speed": 37.04,
    >>>     "max_windspeed_transport": 99,
    >>>     "max_windspeed_repair": 99,
    >>>     "max_waveheight_transport": 1.5,
    >>>     "max_waveheight_repair": 1.5,
    >>>     "strategy": scheduled,
    >>>     "crew_transfer_time": 0.25,
    >>>     "n_crews": 1,
    >>>     "crew": {
    >>>         "day_rate": 0,
    >>>         "n_day_rate": 0,
    >>>         "hourly_rate": 0,
    >>>         "n_hourly_rate": 0,
    >>>     },
    >>> }
    >>> equipment = ServiceEquipmentData(data_dict).determine_type()
    """

    data_dict: dict
    strategy: str = attr.ib(kw_only=True, default=None)

    def __attrs_post_init__(self):
        if self.strategy is None:
            object.__setattr__(
                self, "strategy", clean_string_input(self.data_dict["strategy"])
            )
        if self.strategy not in ("scheduled", "requests", "downtime"):
            raise ValueError(
                "ServiceEquipment strategy should be one of 'scheduled' or 'unscheduled'."
            )

    def determine_type(
        self,
    ) -> Union[ScheduledServiceEquipmentData, UnscheduledServiceEquipmentData]:
        """Generates the appropriate ServiceEquipmentData variation.

        Returns
        -------
        Union[ScheduledServiceEquipmentData, UnscheduledServiceEquipmentData]
            The appropriate ``xxServiceEquipmentData`` schema depending on the strategy the
            ``ServiceEquipment`` will use.
        """
        if self.strategy == "scheduled":
            return ScheduledServiceEquipmentData.from_dict(self.data_dict)
        elif self.strategy in ("requests", "downtime"):
            return UnscheduledServiceEquipmentData.from_dict(self.data_dict)
        else:
            raise ValueError("Invalid strategy provided!")


@attr.s(frozen=True, auto_attribs=True)
class FixedCosts(FromDictMixin):
    """The fixed costs for operating a windfarm. All values are assumed to be in $/kW/yr.

    Parameters
    ----------
    operations : float
        Non-maintenance costs of operating the project.

        If a value is provided for this attribute, then it will zero out all other
        values, otherwise it will be set to the sum of the remaining values.
    operations_management_administration: float
        Activities necessary to forecast, dispatch, sell, and manage the production of
        power from the plant. Includes both the onsite and offsite personnel, software,
        and equipment to coordinate high voltage equipment, switching, port activities,
        and marine activities.

        This should only be used when not breaking down the cost into the following
        categories: ``project_management_administration``,
        ``operation_management_administration``, ``marine_management``, and/or
        ``weather_forecasting``

    project_management_administration : float
        Financial reporting, public relations, procurement, parts and stock management,
        H&SE management, training, subcontracts, and general administration.
    marine_management : float
        Coordination of port equipment, vessels, and personnel to carry out inspections
        and maintenance of generation and transmission equipment.
    weather_forecasting : float
        Daily forecast of metocean conditions used to plan maintenance visits and
        estimate project power production.
    condition_monitoring : float
        Monitoring of SCADA data from wind turbine components to optimize performance
        and identify component faults.
    operating_facilities : float
        Co-located offices, parts store, quayside facilities, helipad, refueling
        facilities, hanger (if necesssary), etc.
    environmental_health_safety_monitoring : float
        Coordination and monitoring to ensure comp0liance with HSE requirements during
        operations.
    insurance : float
        Insurance policies during operational period including All Risk Property,
        Buisness Interuption, Third Party Liability, and Brokers Fee, and Storm Coverage.

        This should only be used when not breaking down the cost into the following
        categories: ``brokers_fee``, ``operations_all_risk``, ``business_interruption``,
        ``third_party_liability``, and/or ``storm_coverage``
    brokers_fee : float
        Fees for arranging the insurance package.
    operations_all_risk : float
        All Risk Property (physical damage). Sudden and unforseen physical loss or
        physical damage to teh plant/assets during the operational phase of a project.
    business_interruption : float
        Sudden and unforseen loss or physical damage to the plant/assets during the
        operational phase of a project causing an interruption.
    third_party_liability : float
        Liability imposed by law, and/or Express Contractual Liability, for bodily
        injury or property damage.
    storm_coverage : float
        Coverage from huricane and tropical storm events (tyipcally for Atlantic Coast
        projects).
    annual_leases_fees : float
        Ongoing payments, including but not limited to: payments to regulatory body for
        permission to operate at project site (terms defined within lease); payments to
        Transmission Systems Operators or Transmission Asseet Owners for rights to
        transport generated power.

        This should only be used when not breaking down the cost into the following
        categories: ``submerge_land_lease_costs`` and/or ``transmission_charges_rights``

    submerge_land_lease_costs : float
        Payments to submerged land owners for rights to build project during operations.
    transmission_charges_rights : float
        Any payments to Transmissions Systems Operators or Transmission Asset Owners for
        rights to transport generated power.
    onshore_electrical_maintenance : float
        Inspections of cables, transformer, switch gears, power compensation equipment,
        etc. and infrequent repairs

        This should only be used if not modeling these as processes within the model.
        Currently, onshore modeling is not included.
    labor : float
        The costs associated with labor, if not being modeled through the simulated
        processes.
    """

    # Total Cost
    operations: float = attr.ib(default=0)

    # Operations, Management, and General Administration
    operations_management_administration: float = attr.ib(default=0)
    project_management_administration: float = attr.ib(default=0)
    marine_management: float = attr.ib(default=0)
    weather_forecasting: float = attr.ib(default=0)
    condition_monitoring: float = attr.ib(default=0)

    operating_facilities: float = attr.ib(default=0)
    environmental_health_safety_monitoring: float = attr.ib(default=0)

    # Insurance
    insurance: float = attr.ib(default=0)
    brokers_fee: float = attr.ib(default=0)
    operations_all_risk: float = attr.ib(default=0)
    business_interruption: float = attr.ib(default=0)
    third_party_liability: float = attr.ib(default=0)
    storm_coverage: float = attr.ib(default=0)

    # Annual Leases and Fees
    annual_leases_fees: float = attr.ib(default=0)
    submerge_land_lease_costs: float = attr.ib(default=0)
    transmission_charges_rights: float = attr.ib(default=0)

    onshore_electrical_maintenance: float = attr.ib(default=0)

    labor: float = attr.ib(default=0)

    resolution: dict = attr.ib(init=False)
    hierarchy: dict = attr.ib(init=False)

    def cost_category_validator(self, name: str, sub_names: List[str]):
        """Either updates the higher-level cost to be the sum of the category's
        lower-level costs or uses a supplied higher-level cost and zeros out the
        lower-level costs.

        Parameters
        ----------
        name : str
            The higher-level cost category's name.
        sub_names : List[str]
            The lower-level cost names associated with the higher-level category.
        """
        if getattr(self, name) <= 0:
            object.__setattr__(self, name, fsum(getattr(self, el) for el in sub_names))
        else:
            for cost in sub_names:
                object.__setattr__(self, cost, 0)

    def __attrs_post_init__(self):
        grouped_names = {
            "operations_management_administration": [
                "project_management_administration",
                "marine_management",
                "weather_forecasting",
                "condition_monitoring",
            ],
            "insurance": [
                "brokers_fee",
                "operations_all_risk",
                "business_interruption",
                "third_party_liability",
                "storm_coverage",
            ],
            "annual_leases_fees": [
                "submerge_land_lease_costs",
                "transmission_charges_rights",
            ],
        }
        individual_names = [
            "operating_facilities",
            "environmental_health_safety_monitoring",
            "onshore_electrical_maintenance",
            "labor",
        ]

        # Check each category for aggregation
        for _name, _sub_names in grouped_names.items():
            self.cost_category_validator(_name, _sub_names)

        # Check for single value aggregation
        self.cost_category_validator("operations", [*grouped_names] + individual_names)

        # Create the value resolution mapping
        object.__setattr__(
            self,
            "resolution",
            {
                "low": ["operations"],
                "medium": [
                    "operations_management_administration",
                    "insurance",
                    "annual_leases_fees",
                    "operating_facilities",
                    "environmental_health_safety_monitoring",
                    "onshore_electrical_maintenance",
                    "labor",
                ],
                "high": [
                    "project_management_administration",
                    "marine_management",
                    "weather_forecasting",
                    "condition_monitoring",
                    "brokers_fee",
                    "operations_all_risk",
                    "business_interruption",
                    "third_party_liability",
                    "storm_coverage",
                    "submerge_land_lease_costs",
                    "transmission_charges_rights",
                    "operating_facilities",
                    "environmental_health_safety_monitoring",
                    "onshore_electrical_maintenance",
                    "labor",
                ],
            },
        )

        object.__setattr__(
            self,
            "hierarchy",
            {
                "operations": {
                    "operations_management_administration": [
                        "project_management_administration",
                        "marine_management",
                        "weather_forecasting",
                        "condition_monitoring",
                    ],
                    "insurance": [
                        "brokers_fee",
                        "operations_all_risk",
                        "business_interruption",
                        "third_party_liability",
                        "storm_coverage",
                    ],
                    "annual_leases_fees": [
                        "submerge_land_lease_costs",
                        "transmission_charges_rights",
                    ],
                    "operating_facilities": [],
                    "environmental_health_safety_monitoring": [],
                    "onshore_electrical_maintenance": [],
                    "labor": [],
                }
            },
        )
