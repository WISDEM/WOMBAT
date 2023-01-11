"""Turbine and turbine component shared utilities."""


from __future__ import annotations

import datetime
from math import fsum
from typing import TYPE_CHECKING, Any, Callable, Sequence
from pathlib import Path
from functools import partial, update_wrapper

import attr
import attrs
import numpy as np
import pandas as pd
from attrs import Factory, Attribute, field, define
from scipy.stats import weibull_min

from wombat.utilities import HOURS_IN_DAY, HOURS_IN_YEAR
from wombat.utilities.time import parse_date


if TYPE_CHECKING:
    from wombat.core import ServiceEquipment


# Define the valid servicing equipment types
VALID_EQUIPMENT = (
    "CTV",  # Crew tranfer vessel or onsite vehicle
    "SCN",  # Small crane
    "LCN",  # Large crane
    "CAB",  # Cabling equipment
    "RMT",  # Remote reset or anything performed remotely
    "DRN",  # Drone
    "DSV",  # Diving support vessel
    "TOW",  # Tugboat or support vessel for moving a turbine to a repair facility
    "AHV",  # Anchor handling vessel, tpyically a tugboat, but doesn't trigger tow-to-port
)

# Define the valid unscheduled and valid strategies
UNSCHEDULED_STRATEGIES = ("requests", "downtime")
VALID_STRATEGIES = tuple(["scheduled"] + list(UNSCHEDULED_STRATEGIES))


def convert_to_list(
    value: Sequence | str | int | float,
    manipulation: Callable | None = None,
) -> list:
    """Converts an unknown element that could be a list or single, non-sequence element
    to a list of elements.

    Parameters
    ----------
    value : Sequence | str | int | float
        The unknown element to be converted to a list of element(s).
    manipulation: Callable | None
        A function to be performed upon the individual elements, by default None.

    Returns
    -------
    list
        The new list of elements.
    """

    if isinstance(value, (str, int, float)):
        value = [value]
    if manipulation is not None:
        return [manipulation(el) for el in value]
    return list(value)


convert_to_list_upper = partial(convert_to_list, manipulation=str.upper)
update_wrapper(convert_to_list_upper, convert_to_list)

convert_to_list_lower = partial(convert_to_list, manipulation=str.lower)
update_wrapper(convert_to_list_lower, convert_to_list)


def clean_string_input(value: str | None) -> str | None:
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
    if value is None:
        return value
    return value.lower().strip()


def annual_date_range(
    start_day: int,
    end_day: int,
    start_month: int,
    end_month: int,
    start_year: int,
    end_year: int,
) -> np.ndarray:
    """Creates a series of date ranges across multiple years between the starting date
    and ending date for each year from starting year to ending year. Will only work if
    the end year is the same as the starting year or later, and the ending month, day
    combinatoion is after the starting month, day combination.

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

    Raises
    ------
    ValueError: Raised if the ending month, day combination occur after the starting
        month, day combination.
    ValueError: Raised if the ending year is prior to the starting year.

    Yields
    -------
    np.ndarray
        A ``numpy.ndarray`` of ``datetime.date`` objects.
    """
    # Check the year bounds
    if end_year < start_year:
        raise ValueError(
            f"The end_year ({start_year}) is later than the start_year ({end_year})."
        )

    # Check the month, date combination bounds
    start = datetime.datetime(2022, start_month, start_day)
    end = datetime.datetime(2022, end_month, end_day)
    if end < start:
        raise ValueError(
            f"The starting month and day {start} is after the ending month and day {end}."
        )

    # Create a list of arrays of date ranges for each year
    start = datetime.datetime(1, start_month, start_day)
    end = datetime.datetime(1, end_month, end_day)
    date_ranges = [
        pd.date_range(start.replace(year=year), end.replace(year=year)).date
        for year in range(start_year, end_year + 1)
    ]

    # Return a 1-D array
    return np.hstack(date_ranges)


def annualized_date_range(
    start_date: datetime.datetime,
    start_year: int,
    end_date: datetime.datetime,
    end_year: int,
) -> np.ndarray:
    """Creates an annualized list of dates based on the simulation years.

    Parameters
    ----------
    start_date : str
        The string start month and day in MM/DD or MM-DD format.
    start_year : int
        _description_
    end_date : str
        The string end month and day in MM/DD or MM-DD format.
    end_year : int
        _description_

    Returns
    -------
    set[datetime.datetime]
        _description_
    """
    additional = 1 if end_date < start_date else 0
    date_list = [
        pd.date_range(
            start_date.replace(year=year),
            end_date.replace(year=year + additional),
            freq="D",
        ).date
        for year in range(start_year - additional, end_year + 1)
    ]
    dates = np.hstack(date_list)
    start = datetime.date(start_year, 1, 1)
    end = datetime.date(end_year, 12, 31)
    dates = dates[(dates >= start) & (dates <= end)]
    return dates


def convert_ratio_to_absolute(ratio: int | float, total: int | float) -> int | float:
    """Converts a proportional cost to an absolute cost.

    Parameters
    ----------
    ratio : int | float
        The proportional materials cost for a ``Maintenance`` or ``Failure``.
    total: int | float
        The turbine's replacement cost.

    Returns
    -------
    int | float
        The absolute cost of the materials.
    """

    if ratio <= 1:
        return ratio * total
    return ratio


def check_start_stop_dates(
    instance: Any, attribute: attr.Attribute, value: datetime.datetime | None
) -> None:
    """Ensures the date range start and stop points are not the same."""
    if attribute.name == "non_operational_end":
        start_date = instance.non_operational_start
        start_name = "non_operational_start"
    else:
        start_date = instance.reduced_speed_start
        start_name = "reduced_speed_start"

    if value is None:
        if start_date is None:
            return
        raise ValueError(
            f"A starting date was provided, but no ending date was provided for `{attribute.name}`."
        )

    if start_date is None:
        raise ValueError(
            f"An ending date was provided, but no starting date was provided for `{start_name}`."
        )

    if start_date == value:
        raise ValueError(
            f"Starting date (`{start_name}`={start_date} and ending date"
            f" (`{attribute.name}`={value}) cannot be the same date."
        )


def valid_hour(
    instance: Any, attribute: Attribute, value: int  # pylint: disable=W0613
) -> None:
    """Validator that ensures that the input is a valid time or null value (-1).

    Parameters
    ----------
    instance : Any
        A class object.
    attribute : Attribute
        The attribute being validated.
    value : int
        A whole number, hour of the day or -1 (null), where 24 indicates end of the day.

    Raises
    ------
    ValueError
        Raised if ``value`` is not between -1 and 24, inclusive.
    """
    if value == -1:
        pass
    elif value < 0 or value > 24:
        raise ValueError(f"Input {attribute.name} must be between 0 and 24, inclusive.")


def valid_reduction(
    instance, attribute: Attribute, value: int | float  # pylint: disable=W0613
) -> None:
    """Checks to see if the reduction factor is between 0 and 1, inclusive.

    Parameters
    ----------
    value : int | float
        The input value for `speed_reduction_factor`.
    """
    if value < 0 or value > 1:
        raise ValueError(
            f"Input for {attribute.name}'s `speed_reduction_factor` must be between 0 and 1, inclusive."
        )


def greater_than_zero(
    instance,  # pylint: disable=W0613
    attribute: Attribute,  # pylint: disable=W0613
    value: int | float,
) -> None:
    """Checks if an input is greater than 0.

    Parameters
    ----------
    instance : Any
        The class containing the attribute to be checked.
    attribute : Attribute
        The attribute's properties.
    value : int | float
        The user-input value for the ``attribute``.

    Raises
    ------
    ValueError
        Raised if ``value`` is less than or equal to zero.
    """
    if value <= 0:
        raise ValueError("Input must be greater than 0.")


def greater_than_equal_to_zero(
    instance,  # pylint: disable=W0613
    attribute: Attribute,  # pylint: disable=W0613
    value: int | float,
) -> None:
    """Checks if an input is at least 0.

    Parameters
    ----------
    instance : Any
        The class containing the attribute to be checked.
    attribute : Attribute
        The attribute's properties.
    value : int | float
        The user-input value for the ``attribute``.

    Raises
    ------
    ValueError
        Raised if ``value`` is less than or equal to zero.
    """
    if value < 0:
        raise ValueError("Input must be at least 0.")


@define
class FromDictMixin:
    """A Mixin class to allow for kwargs overloading when a data class doesn't
    have a specific parameter definied. This allows passing of larger dictionaries
    to a data class without throwing an error.

    Raises
    ------
    AttributeError
        Raised if the required class inputs are not provided.
    """

    @classmethod
    def from_dict(cls, data: dict):
        """Maps a data dictionary to an `attrs`-defined class.

        TODO: Add an error to ensure that either none or all the parameters are passed in

        Parameters
        ----------
            data : dict
                The data dictionary to be mapped.
        Returns
        -------
            cls : Any
                The `attrs`-defined class.
        """
        # Get all parameters from the input dictionary that map to the class initialization
        kwargs = {  # type: ignore
            a.name: data[a.name]  # type: ignore
            for a in cls.__attrs_attrs__  # type: ignore
            if a.name in data and a.init  # type: ignore
        }

        # Map the inputs must be provided: 1) must be initialized, 2) no default value defined
        required_inputs = [  # type: ignore
            a.name  # type: ignore
            for a in cls.__attrs_attrs__  # type: ignore
            if a.init and isinstance(a.default, attr._make._Nothing)  # type: ignore
        ]
        undefined = sorted(set(required_inputs) - set(kwargs))
        if undefined:
            raise AttributeError(
                f"The class defintion for {cls.__name__} is missing the following inputs: {undefined}"
            )
        return cls(**kwargs)  # type: ignore


@define(frozen=True, auto_attribs=True)
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
         - TOW: tugboat or towing equipment
         - AHV: anchor handling vessel (tugboat that doesn't trigger tow-to-port)
    system_value : Union[int, float]
        Turbine replacement value. Used if the materials cost is a proportional cost.
    description : str
        A short text description to be used for logging.
    operation_reduction : float
        Performance reduction caused by the failure, between (0, 1]. Defaults to 0.
    level : int, optional
        Severity level of the maintenance. Defaults to 0.
    """

    time: float = field(converter=float)
    materials: float = field(converter=float)
    frequency: float = field(converter=float)
    service_equipment: list[str] = field(
        converter=convert_to_list_upper,  # type: ignore
        validator=attrs.validators.deep_iterable(
            member_validator=attrs.validators.in_(VALID_EQUIPMENT),
            iterable_validator=attrs.validators.instance_of(list),
        ),
    )
    system_value: int | float = field(converter=float)
    description: str = field(default="routine maintenance", converter=str)
    operation_reduction: float = field(default=0.0, converter=float)
    level: int = field(default=0, converter=int)
    request_id: str = field(init=False)
    replacement: bool = field(default=False, init=False)

    def __attrs_post_init__(self):
        """Convert frequency to hours (simulation time scale) and the equipment
        requirement to a list.
        """
        object.__setattr__(self, "frequency", self.frequency * HOURS_IN_DAY)
        # object.__setattr__(
        #     self,
        #     "service_equipment",
        #     convert_to_list(self.service_equipment, str.upper),
        # )
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


@define(frozen=True, auto_attribs=True)
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
         - TOW: tugboat or towing equipment
         - AHV: anchor handling vessel (tugboat that doesn't trigger tow-to-port)
    system_value : Union[int, float]
        Turbine replacement value. Used if the materials cost is a proportional cost.
    replacement : bool
        True if triggering the failure requires a subassembly replacement, False, if
        only a repair is necessary. Defaults to False
    description : str
        A short text description to be used for logging.
    """

    scale: float = field(converter=float)
    shape: float = field(converter=float)
    time: float = field(converter=float)
    materials: float = field(converter=float)
    operation_reduction: float = field(converter=float)
    level: int = field(converter=int)
    service_equipment: list[str] | str = field(
        converter=convert_to_list_upper,  # type: ignore
        validator=attrs.validators.deep_iterable(
            member_validator=attrs.validators.in_(VALID_EQUIPMENT),
            iterable_validator=attrs.validators.instance_of(list),
        ),
    )
    system_value: int | float = field(converter=float)
    replacement: bool = field(default=False, converter=bool)
    description: str = field(default="failure", converter=str)
    weibull: weibull_min = field(init=False, eq=False)
    request_id: str = field(init=False)

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

    def hours_to_next_failure(self) -> float | None:
        """Samples the next time to failure in a Weibull distribution. If the ``scale``
        and ``shape`` parameters are set to 0, then the model will return ``None`` to cause
        the subassembly to timeout to the end of the simulation.

        Returns
        -------
        float | None
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


@define(frozen=True, auto_attribs=True)
class SubassemblyData(FromDictMixin):
    """Data storage and validation class for the subassemblies.

    Parameters
    ----------
    name : str
        Name of the component/subassembly.
    maintenance : list[dict[str, float | str]]
        List of the maintenance classification dictionaries. This will be converted
        to a list of ``Maintenance`` objects in the post initialization hook.
    failures : fict[int, dict[str, float | str]]
        Dictionary of failure classifications in a numerical (ordinal) categorization
        order. This will be converted to a dictionary of ``Failure`` objects in the
        post initialization hook.
    system_value : int | float
        Turbine's cost of replacement. Used in case percentages of turbine cost are used
        in place of an absolute cost.
    """

    name: str = field(converter=str)
    maintenance: list[Maintenance | dict[str, float | str]]
    failures: dict[int, Failure | dict[str, float | str]]
    system_value: int | float = field(converter=float)

    def __attrs_post_init__(self):
        """Converts the maintenance and failure data to ``Maintenance`` and ``Failure`` objects, respectively."""
        for kwargs in self.maintenance:
            assert isinstance(kwargs, dict)
            kwargs.update({"system_value": self.system_value})
        object.__setattr__(
            self,
            "maintenance",
            [
                Maintenance.from_dict(kw) if isinstance(kw, dict) else kw
                for kw in self.maintenance
            ],
        )

        for kwargs in self.failures.values():  # type: ignore
            assert isinstance(kwargs, dict)
            kwargs.update({"system_value": self.system_value})
        object.__setattr__(
            self,
            "failures",
            {
                level: Failure.from_dict(kw) if isinstance(kw, dict) else kw
                for level, kw in self.failures.items()
            },
        )


@define(frozen=True, auto_attribs=True)
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
    details : Failure | Maintenance
        The actual data class.
    cable : bool
        Indicator that the request is for a cable, by default False.
    upstream_turbines : list[str]
        The cable's upstream turbines, by default []. No need to use this if ``cable`` == False.
    upstream_cables : list[str]
        The cable's upstream cables, by default []. No need to use this if ``cable`` == False.
    """

    system_id: str = field(converter=str)
    system_name: str = field(converter=str)
    subassembly_id: str = field(converter=str)
    subassembly_name: str = field(converter=str)
    severity_level: int = field(converter=int)
    details: Failure | Maintenance = field(
        validator=attrs.validators.instance_of((Failure, Maintenance))
    )
    cable: bool = field(default=False, converter=bool, kw_only=True)
    upstream_turbines: list[str] = field(default=Factory(list), kw_only=True)
    upstream_cables: list[str] = field(default=Factory(list), kw_only=True)
    request_id: str = field(init=False)

    def assign_id(self, request_id: str) -> None:
        """Assigns a unique identifier to the request.

        Parameters
        ----------
        request_id : str
            The ``wombat.core.RepairManager`` generated identifier.
        """
        object.__setattr__(self, "request_id", request_id)
        self.details.assign_id(request_id)


@define(frozen=True, auto_attribs=True)
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

    n_day_rate: int = field(converter=int)
    day_rate: float = field(converter=float)
    n_hourly_rate: int = field(converter=int)
    hourly_rate: float = field(converter=float)


class DateLimitsMixin:
    """Base servicing equpment dataclass. Only meant to reduce repeated code across the
    ``ScheduledServiceEquipmentData`` and ``UnscheduledServiceEquipmentData`` classes.
    """

    # MyPy type hints
    port_distance: float = field(converter=float)
    non_operational_start: datetime.datetime = field(converter=parse_date)
    non_operational_end: datetime.datetime = field(converter=parse_date)
    reduced_speed_start: datetime.datetime = field(converter=parse_date)
    reduced_speed_end: datetime.datetime = field(converter=parse_date)

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

    def _set_port_distance(self, distance: int | float | None) -> None:
        """Used to set ``port_distance`` from the environment's or port's variables.

        Parameters
        ----------
        distance : int | float
            The distance to port that must be traveled for servicing equipment.
        """
        if distance is None:
            return
        if distance <= 0:
            return
        if self.port_distance <= 0:  # type: ignore
            object.__setattr__(self, "port_distance", float(distance))

    def _compare_dates(
        self,
        new_start: datetime.datetime | None,
        new_end: datetime.datetime | None,
        which: str,
    ) -> tuple[datetime.datetime | None, datetime.datetime | None]:
        """Compares the orignal and newly input starting ane ending date for either the
        non-operational date range or the reduced speed date range.

        Parameters
        ----------
        new_start : datetime.datetime | None
            The overriding start date if it is an earlier date than the original.
        new_end : datetime.datetime | None
            The overriding end date if it is a later date than the original.
        which : str
            One of "reduced_speed" or "non_operational"

        Returns
        -------
        tuple[datetime.datetime | None, datetime.datetime | None]
            The more conservative date pair between the new values and original values.

        Raises
        ------
        ValueError
            Raised if an invalid value to ``which`` is provided.
        """
        if which in ("reduced_speed", "non_operational"):
            original_start = getattr(self, f"{which}_start")
            original_end = getattr(self, f"{which}_end")
        else:
            raise ValueError(
                "`which` must be one of 'reduced_speed' or 'non_operational'."
            )

        if original_start is not None:
            if new_start is not None:
                new_start = min(new_start, original_start)
            else:
                new_start = original_start
        else:
            object.__setattr__(self, f"{which}_start", new_start)

        if original_end is not None:
            if new_end is not None:
                new_end = max(new_end, original_end)
            else:
                new_end = original_end
        else:
            object.__setattr__(self, f"{which}_end", new_end)

        return new_start, new_end

    def set_non_operational_dates(
        self,
        start_date: datetime.datetime | None = None,
        start_year: int | None = None,
        end_date: datetime.datetime | None = None,
        end_year: int | None = None,
    ) -> None:
        """Creates an annualized list of dates where servicing equipment should be
        unavailable. Takes a a ``start_year`` and ``end_year`` to determine how many
        years should be covered.

        Parameters
        ----------
        start_date : datetime.datetime | None
            The starting date for the annual date range of non-operational dates, by
            default None.
        start_year : int | None
            The first year that has a non-operational date range, by default None.
        end_date : datetime.datetime | None
            The ending date for the annual range of non-operational dates, by default
            None.
        end_year : int | None
            The last year that should have a non-operational date range, by default None.

        Raises
        ------
        ValueError
            Raised if the starting and ending dates are the same date.
        """
        start_date, end_date = self._compare_dates(
            start_date, end_date, "non_operational"
        )
        object.__setattr__(self, "non_operational_start", start_date)
        object.__setattr__(self, "non_operational_end", end_date)

        # Check that the base dates are valid
        if self.non_operational_start is None or self.non_operational_end is None:
            object.__setattr__(
                self, "non_operational_dates", np.empty(0, dtype="object")
            )
            object.__setattr__(self, "non_operational_dates_set", set())
            return

        # Check that the input year range is valid
        if not isinstance(start_year, int):
            raise ValueError(
                f"Input to `start_year`: {start_year}, must be an integer."
            )
        if not isinstance(end_year, int):
            raise ValueError(f"Input to `end_year`: {end_year}, must be an integer.")
        if end_year < start_year:
            raise ValueError(
                f"`start_year`: {start_year}, must less than or equal to the `end_year`: {end_year}"
            )

        # Create the date range
        dates = annualized_date_range(
            self.non_operational_start, start_year, self.non_operational_end, end_year
        )
        object.__setattr__(self, "non_operational_dates", dates)
        object.__setattr__(self, "non_operational_dates_set", set(dates))

        # Update the operating dates field to ensure there is no overlap
        if hasattr(self, "operating_dates"):
            operating = np.setdiff1d(self.operating_dates, self.non_operational_dates)  # type: ignore
            object.__setattr__(self, "operating_dates", operating)
            object.__setattr__(self, "operating_dates_set", set(operating))

    def set_reduced_speed_parameters(
        self,
        start_date: datetime.datetime | None = None,
        start_year: int | None = None,
        end_date: datetime.datetime | None = None,
        end_year: int | None = None,
        speed: float = 0.0,
    ) -> None:
        """Creates an annualized list of dates where servicing equipment should be
        operating with reduced speeds. The passed ``start_date``, ``end_date``, and
        ``speed`` will override provided values if they are more conservative than the
        current settings, or a value for ``speed`` will only override the existing value
        if the passed value is slower. Takes a ``start_year`` and ``end_year`` to
        determine how many years should be covered by this setting.

        Parameters
        ----------
        start_date : datetime.datetime | None
            The starting date for the annual date range of reduced speeds, by default
            None.
        start_year : int | None
            The first year that has a reduced speed date range, by default None.
        end_date : datetime.datetime | None
            The ending date for the annual range of reduced speeds, by default None.
        end_year : int | None
            The last year that should have a reduced speed date range, by default None.
        speed : float
            The maximum operating speed under a speed-restricted scenario.

        Raises
        ------
        ValueError
            Raised if the starting and ending dates are the same date.
        """
        # Check that the base dates are valid and override if no values were provided
        # or if the new value is more conservative than the originally provided value
        start_date, end_date = self._compare_dates(
            start_date, end_date, "reduced_speed"
        )
        object.__setattr__(self, "reduced_speed_start", start_date)
        object.__setattr__(self, "reduced_speed_end", end_date)

        if start_date is None or end_date is None:
            object.__setattr__(self, "reduced_speed_dates", np.empty(0, dtype="object"))
            object.__setattr__(self, "reduced_speed_dates_set", set())
            return

        # Check that the input year range is valid
        if not isinstance(start_year, int):
            raise ValueError(
                f"Input to `start_year`: {start_year}, must be an integer."
            )
        if not isinstance(end_year, int):
            raise ValueError(f"Input to `end_year`: {end_year}, must be an integer.")
        if end_year < start_year:
            raise ValueError(
                f"`start_year`: {start_year}, must less than or equal to the `end_year`: {end_year}"
            )

        # Create the date range
        dates = annualized_date_range(
            self.reduced_speed_start, start_year, self.reduced_speed_end, end_year
        )
        object.__setattr__(self, "reduced_speed_dates", dates)
        object.__setattr__(self, "reduced_speed_dates_set", set(dates))

        # Update the reduced speed if none was originally provided
        if speed != 0 and (self.reduced_speed == 0 or speed < self.reduced_speed):  # type: ignore
            object.__setattr__(self, "reduced_speed", speed)


@define(frozen=True, auto_attribs=True)
class ScheduledServiceEquipmentData(FromDictMixin, DateLimitsMixin):
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

        .. note:: the input to this does not matter yet, as multi-crew functionality
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

        .. note:: if the rig comes annually, then the enter the year for the last year
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

        Please note that "TOW" is unavailable for scheduled servicing equipment
    mobilization_cost : float
        Cost to mobilize the rig and crew.
    mobilization_days : int
        Number of days it takes to mobilize the equipment.
    speed : float
        Maximum transit speed, km/hr.
    speed_reduction_factor : flaot
        Reduction factor for traveling in inclement weather, default 0. When 0, travel
        is stopped when either `max_windspeed_transport` or `max_waveheight_transport`
        is reached, and when 1, `speed` is used.
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

        .. note:: if the rig and crew are onsite be sure that the start and end dates
            represent the first and last day/month of the year, respectively, and the
            start and end years represent the fist and last year in the weather file.

    method : str
        Determines if the ship will do all maximum severity repairs first or do all
        the repairs at one turbine before going to the next, by default severity.
        Should by one of "severity" or "turbine".
    port_distance : int | float
        The distance, in km, the equipment must travel to go between port and site, by
        default 0.
    non_operational_start : str | datetime.datetime | None
        The starting month and day, e.g., MM/DD, M/D, MM-DD, etc. for an annualized
        period of prohibited operations. When defined at the environment level, an
        undefined or later starting date will be overridden, by default None.
    non_operational_end : str | datetime.datetime | None
        The ending month and day, e.g., MM/DD, M/D, MM-DD, etc. for an annualized
        period of prohibited operations. When defined at the environment level, an
        undefined or earlier ending date will be overridden, by default None.
    reduced_speed_start : str | datetime.datetime | None
        The starting month and day, e.g., MM/DD, M/D, MM-DD, etc. for an annualized
        period of reduced speed operations. When defined at the environment level, an
        undefined or later starting date will be overridden, by default None.
    reduced_speed_end : str | datetime.datetime | None
        The ending month and day, e.g., MM/DD, M/D, MM-DD, etc. for an annualized
        period of reduced speed operations. When defined at the environment level, an
        undefined or earlier ending date will be overridden, by default None.
    reduced_speed : float
        The maximum operating speed during the annualized reduced speed operations.
        When defined at the environment level, an undefined or faster value will be
        overridden, by default 0.0.
    """

    name: str = field(converter=str)
    equipment_rate: float = field(converter=float)
    n_crews: int = field(converter=int)
    crew: ServiceCrew = field(converter=ServiceCrew.from_dict)  # type: ignore
    capability: list[str] = field(
        converter=convert_to_list_upper,  # type: ignore
        validator=attrs.validators.deep_iterable(
            member_validator=attrs.validators.in_(VALID_EQUIPMENT),
            iterable_validator=attrs.validators.instance_of(list),
        ),
    )
    speed: float = field(converter=float, validator=greater_than_zero)
    max_windspeed_transport: float = field(converter=float)
    max_windspeed_repair: float = field(converter=float)

    mobilization_cost: float = field(default=0, converter=float)
    mobilization_days: int = field(default=0, converter=int)
    max_waveheight_transport: float = field(default=1000.0, converter=float)
    max_waveheight_repair: float = field(default=1000.0, converter=float)
    workday_start: int = field(default=-1, converter=int, validator=valid_hour)
    workday_end: int = field(default=-1, converter=int, validator=valid_hour)
    crew_transfer_time: float = field(converter=float, default=0.0)
    speed_reduction_factor: float = field(
        default=0.0, converter=float, validator=valid_reduction
    )
    port_distance: float = field(default=0.0, converter=float)
    onsite: bool = field(default=False, converter=bool)
    method: str = field(  # type: ignore
        default="severity",
        converter=[str, str.lower],  # type: ignore
        validator=attrs.validators.in_(["turbine", "severity"]),
    )
    start_month: int = field(
        default=-1, converter=int, validator=attrs.validators.ge(0)  # type: ignore
    )
    start_day: int = field(default=-1, converter=int, validator=attrs.validators.ge(0))  # type: ignore
    start_year: int = field(default=-1, converter=int, validator=attrs.validators.ge(0))  # type: ignore
    end_month: int = field(default=-1, converter=int, validator=attrs.validators.ge(0))  # type: ignore
    end_day: int = field(default=-1, converter=int, validator=attrs.validators.ge(0))  # type: ignore
    end_year: int = field(default=-1, converter=int, validator=attrs.validators.ge(0))  # type: ignore
    strategy: str = field(
        default="scheduled", validator=attrs.validators.in_(["scheduled"])
    )
    operating_dates: np.ndarray = field(factory=list, init=False)
    operating_dates_set: set = field(factory=set, init=False)

    non_operational_start: datetime.datetime = field(default=None, converter=parse_date)
    non_operational_end: datetime.datetime = field(
        default=None, converter=parse_date, validator=check_start_stop_dates
    )
    reduced_speed_start: datetime.datetime = field(default=None, converter=parse_date)
    reduced_speed_end: datetime.datetime = field(
        default=None, converter=parse_date, validator=check_start_stop_dates
    )
    reduced_speed: float = field(default=0, converter=float)
    non_operational_dates: pd.DatetimeIndex = field(factory=list, init=False)
    non_operational_dates_set: pd.DatetimeIndex = field(factory=set, init=False)
    reduced_speed_dates: pd.DatetimeIndex = field(factory=set, init=False)

    def create_date_range(self) -> np.ndarray:
        """Creates an ``np.ndarray`` of valid operational dates for the service equipment."""
        start_date = datetime.datetime(
            self.start_year, self.start_month, self.start_day
        )

        # If `onsite` is True, then create the range in one go because there should be
        # no discrepancies in the range.
        # Othewise, create a specified date range between two dates in the year range.
        if self.onsite:
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

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, "operating_dates", self.create_date_range())
        object.__setattr__(self, "operating_dates_set", set(self.operating_dates))


@define(frozen=True, auto_attribs=True)
class UnscheduledServiceEquipmentData(FromDictMixin, DateLimitsMixin):
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
        - TOW: tugboat or towing equipment
        - AHV: anchor handling vessel (tugboat that doesn't trigger tow-to-port)
    speed : float
        Maximum transit speed, km/hr.
    tow_speed : float
        The maximum transit speed when towing, km/hr.

        .. note:: This is only required for when the servicing equipment is tugboat
            enabled for a tow-to-port scenario (capability = "TOW")

    speed_reduction_factor : flaot
        Reduction factor for traveling in inclement weather, default 0. When 0, travel
        is stopped when either `max_windspeed_transport` or `max_waveheight_transport`
        is reached, and when 1, `speed` is used.
    max_windspeed_transport : float
        Maximum windspeed for safe transport, m/s.
    max_windspeed_repair : float
        Maximum windspeed for safe operations, m/s.
    max_waveheight_transport : float
        Maximum waveheight for safe transport, m, default 1000 (land-based).
    max_waveheight_repair : float
        Maximum waveheight for safe operations, m, default 1000 (land-based).
    mobilization_cost : float
        Cost to mobilize the rig and crew, default 0.
    mobilization_days : int
        Number of days it takes to mobilize the equipment, default 0.
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
        system, e.g. how long does it take to transfer the crew from the CTV to the
        turbine, default 0.
    onsite : bool
        Indicator for if the rig and crew are based onsite.

        .. note:: if the rig and crew are onsite be sure that the start and end dates
            represent the first and last day/month of the year, respectively, and the
            start and end years represent the fist and last year in the weather file.

    method : str
        Determines if the ship will do all maximum severity repairs first or do all
        the repairs at one turbine before going to the next, by default severity.
        Should by one of "severity" or "turbine".
    unmoor_hours : int | float
        The number of hours required to unmoor a floating offshore wind turbine in order
        to tow it to port, by default 0.

        .. note:: Required for the tugboat/towing capability, otherwise unused.

    reconnection_hours : int | float
        The number of hours required to reconnect a floating offshore wind turbine after
        being towed back to site, by default 0.

        .. note:: Required for the tugboat/towing capability, otherwise unused.

    port_distance : int | float
        The distance, in km, the equipment must travel to go between port and site, by
        default 0.
    non_operational_start : str | datetime.datetime | None
        The starting month and day, e.g., MM/DD, M/D, MM-DD, etc. for an annualized
        period of prohibited operations. When defined at the environment level or the
        port level, if a tugboat, an undefined or later starting date will be overridden,
        by default None.
    non_operational_end : str | datetime.datetime | None
        The ending month and day, e.g., MM/DD, M/D, MM-DD, etc. for an annualized
        period of prohibited operations. When defined at the environment level or the
        port level, if a tugboat, an undefined or earlier ending date will be overridden,
        by default None.
    reduced_speed_start : str | datetime.datetime | None
        The starting month and day, e.g., MM/DD, M/D, MM-DD, etc. for an annualized
        period of reduced speed operations. When defined at the environment level or the
        port level, if a tugboat, an undefined or later starting date will be overridden,
        by default None.
    reduced_speed_end : str | datetime.datetime | None
        The ending month and day, e.g., MM/DD, M/D, MM-DD, etc. for an annualized
        period of reduced speed operations. When defined at the environment level or the
        port level, if a tugboat, an undefined or earlier ending date will be overridden,
        by default None.
    reduced_speed : float
        The maximum operating speed during the annualized reduced speed operations.
        When defined at the environment level, an undefined or faster value will be
        overridden, by default 0.0.
    """

    name: str = field(converter=str)
    equipment_rate: float = field(converter=float)
    n_crews: int = field(converter=int)
    crew: ServiceCrew = field(converter=ServiceCrew.from_dict)  # type: ignore
    capability: list[str] = field(
        converter=convert_to_list_upper,  # type: ignore
        validator=attrs.validators.deep_iterable(
            member_validator=attrs.validators.in_(VALID_EQUIPMENT),
            iterable_validator=attrs.validators.instance_of(list),
        ),
    )
    speed: float = field(converter=float, validator=greater_than_zero)
    max_windspeed_transport: float = field(converter=float)
    max_windspeed_repair: float = field(converter=float)
    mobilization_cost: float = field(default=0, converter=float)
    mobilization_days: int = field(default=0, converter=int)
    max_waveheight_transport: float = field(default=1000.0, converter=float)
    max_waveheight_repair: float = field(default=1000.0, converter=float)
    workday_start: int = field(default=-1, converter=int, validator=valid_hour)
    workday_end: int = field(default=-1, converter=int, validator=valid_hour)
    crew_transfer_time: float = field(converter=float, default=0.0)
    speed_reduction_factor: float = field(
        default=0.0, converter=float, validator=valid_reduction
    )
    port_distance: float = field(default=0.0, converter=float)
    onsite: bool = field(default=False, converter=bool)
    method: str = field(  # type: ignore
        default="severity",
        converter=[str, str.lower],  # type: ignore
        validator=attrs.validators.in_(["turbine", "severity"]),
    )
    strategy: str | None = field(
        default="unscheduled",
        converter=clean_string_input,
        validator=attrs.validators.in_(UNSCHEDULED_STRATEGIES),
    )
    strategy_threshold: int | float = field(default=-1, converter=float)
    charter_days: int = field(
        default=-1, converter=int, validator=attrs.validators.gt(0)  # type: ignore
    )
    tow_speed: float = field(default=1, converter=float, validator=greater_than_zero)
    unmoor_hours: int | float = field(default=0, converter=float)
    reconnection_hours: int | float = field(default=0, converter=float)
    non_operational_start: datetime.datetime = field(default=None, converter=parse_date)
    non_operational_end: datetime.datetime = field(default=None, converter=parse_date)
    reduced_speed_start: datetime.datetime = field(default=None, converter=parse_date)
    reduced_speed_end: datetime.datetime = field(default=None, converter=parse_date)
    reduced_speed: float = field(default=0, converter=float)
    non_operational_dates: pd.DatetimeIndex = field(factory=list, init=False)
    non_operational_dates_set: pd.DatetimeIndex = field(factory=set, init=False)
    reduced_speed_dates: pd.DatetimeIndex = field(factory=set, init=False)

    @strategy_threshold.validator  # type: ignore
    def _validate_threshold(
        self,
        attribute: Attribute,  # pylint: disable=W0613
        value: int | float,
    ) -> None:
        """Ensures a valid threshold is provided for a given ``strategy``."""
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


@define(frozen=True, auto_attribs=True)
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
    >>> type(equipment)
    """

    data_dict: dict
    strategy: str | None = field(
        kw_only=True, default=None, converter=clean_string_input
    )

    def __attrs_post_init__(self):
        if self.strategy is None:
            object.__setattr__(
                self, "strategy", clean_string_input(self.data_dict["strategy"])
            )
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"ServiceEquipment strategy should be one of {VALID_STRATEGIES}; input: {self.strategy}."
            )

    def determine_type(
        self,
    ) -> ScheduledServiceEquipmentData | UnscheduledServiceEquipmentData:
        """Generates the appropriate ServiceEquipmentData variation.

        Returns
        -------
        Union[ScheduledServiceEquipmentData, UnscheduledServiceEquipmentData]
            The appropriate ``xxServiceEquipmentData`` schema depending on the strategy the
            ``ServiceEquipment`` will use.
        """
        if self.strategy == "scheduled":
            return ScheduledServiceEquipmentData.from_dict(self.data_dict)
        elif self.strategy in UNSCHEDULED_STRATEGIES:
            return UnscheduledServiceEquipmentData.from_dict(self.data_dict)
        else:
            # This should not be able to be reached
            raise ValueError("Invalid strategy provided!")


@define(auto_attribs=True)
class EquipmentMap:
    """Internal mapping for servicing equipment strategy information."""

    strategy_threshold: int | float
    equipment: "ServiceEquipment"  # noqa: disable=F821


@define(auto_attribs=True)
class StrategyMap:
    """Internal mapping for equipment capabilities and their data."""

    CTV: list[EquipmentMap] = field(factory=list)
    SCN: list[EquipmentMap] = field(factory=list)
    LCN: list[EquipmentMap] = field(factory=list)
    CAB: list[EquipmentMap] = field(factory=list)
    RMT: list[EquipmentMap] = field(factory=list)
    DRN: list[EquipmentMap] = field(factory=list)
    DSV: list[EquipmentMap] = field(factory=list)
    TOW: list[EquipmentMap] = field(factory=list)
    AHV: list[EquipmentMap] = field(factory=list)
    is_running: bool = field(default=False, init=False)

    def update(
        self,
        capability: str,
        threshold: int | float,
        equipment: "ServiceEquipment",  # noqa: disable=F821
    ) -> None:
        """A method to update the strategy mapping between capability types and the
        available ``ServiceEquipment`` objects.

        Parameters
        ----------
        capability : str
            The ``equipment``'s capability.
        threshold : int | float
            The threshold for ``equipment``'s strategy.
        equipment : ServiceEquipment
            The actual ``ServiceEquipment`` object to be logged.

        Raises
        ------
        ValueError
            Raised if there is an invalid capability, though this shouldn't be able to
            be reached.
        """
        # Using a mypy ignore because of an unpatched bug using data classes
        if capability == "CTV":
            self.CTV.append(EquipmentMap(threshold, equipment))  # type: ignore
        elif capability == "SCN":
            self.SCN.append(EquipmentMap(threshold, equipment))  # type: ignore
        elif capability == "LCN":
            self.LCN.append(EquipmentMap(threshold, equipment))  # type: ignore
        elif capability == "CAB":
            self.CAB.append(EquipmentMap(threshold, equipment))  # type: ignore
        elif capability == "RMT":
            self.RMT.append(EquipmentMap(threshold, equipment))  # type: ignore
        elif capability == "DRN":
            self.DRN.append(EquipmentMap(threshold, equipment))  # type: ignore
        elif capability == "DSV":
            self.DSV.append(EquipmentMap(threshold, equipment))  # type: ignore
        elif capability == "TOW":
            self.TOW.append(EquipmentMap(threshold, equipment))  # type: ignore
        elif capability == "AHV":
            self.AHV.append(EquipmentMap(threshold, equipment))  # type: ignore
        else:
            # This should not even be able to be reached
            raise ValueError(
                f"Invalid servicing equipment '{capability}' has been provided!"
            )
        self.is_running = True


@define(frozen=True, auto_attribs=True)
class PortConfig(FromDictMixin, DateLimitsMixin):
    """Port configurations for offshore wind power plant scenarios.

    Parameters
    ----------
    name : str
        The name of the port, if multiple are used, then be sure this is unique.
    tugboats : list[str]
        file, or list of files to create the port's tugboats.

        .. note:: Each tugboat is considered to be a tugboat + supporting vessels as
        the primary purpose to tow turbines between a repair port and site.
    n_crews : int
        The number of service crews available to be working on repairs simultaneously;
        each crew is able to service exactly one repair.
    crew : ServiceCrew
        The crew details, see ``ServiceCrew`` for more information.
    max_operations : int
        Total number of turbines the port can handle simultaneously.
    workday_start : int
        The starting hour of a workshift, in 24 hour time.
    workday_end : int
        The ending hour of a workshift, in 24 hour time.
    site_distance : int | float
        Distance, in km, a tugboat has to travel to get between site and port.
    annual_fee : int | float
        The annualized fee for access to the repair port that will be distributed
        monthly in the simulation and accounted for on the first of the month from the
        start of the simulation to the end of the simulation.

        .. note:: Don't include this cost in both this category and either the
        ``FixedCosts.operations_management_administration`` bucket or
        ``FixedCosts.marine_management`` category.

    non_operational_start : str | datetime.datetime | None
        The starting month and day, e.g., MM/DD, M/D, MM-DD, etc. for an annualized
        period of prohibited operations. When defined at the port level, an undefined or
        later starting date will be overridden by the environment, and any associated
        tubboats will have this value overridden using the same logic, by default None.
    non_operational_end : str | datetime.datetime | None
        The ending month and day, e.g., MM/DD, M/D, MM-DD, etc. for an annualized
        period of prohibited operations. When defined at the port level, an undefined
        or earlier ending date will be overridden by the environment, and any associated
        tubboats will have this value overridden using the same logic, by default None.
    reduced_speed_start : str | datetime.datetime | None
        The starting month and day, e.g., MM/DD, M/D, MM-DD, etc. for an annualized
        period of reduced speed operations. When defined at the port level, an undefined
        or later starting date will be overridden by the environment, and any associated
        tubboats will have this value overridden using the same logic, by default None.
    reduced_speed_end : str | datetime.datetime | None
        The ending month and day, e.g., MM/DD, M/D, MM-DD, etc. for an annualized
        period of reduced speed operations. When defined at the port level, an undefined
        or earlier ending date will be overridden by the environment, and any associated
        tubboats will have this value overridden using the same logic, by default None.
    reduced_speed : float
        The maximum operating speed during the annualized reduced speed operations.
        When defined at the port level, an undefined or faster value will be overridden
        by the environment, and any associated tubboats will have this value overridden
        using the same logic, by default 0.0.
    """

    name: str = field(converter=str)
    tugboats: list[str | Path] = field(converter=convert_to_list)
    crew: ServiceCrew = field(converter=ServiceCrew.from_dict)  # type: ignore
    n_crews: int = field(default=1, converter=int)
    max_operations: int = field(default=1, converter=int)
    workday_start: int = field(default=-1, converter=int, validator=valid_hour)
    workday_end: int = field(default=-1, converter=int, validator=valid_hour)
    site_distance: float = field(default=0.0, converter=float)
    annual_fee: float = field(
        default=0, converter=float, validator=greater_than_equal_to_zero
    )
    non_operational_start: datetime.datetime = field(default=None, converter=parse_date)
    non_operational_end: datetime.datetime = field(default=None, converter=parse_date)
    reduced_speed_start: datetime.datetime = field(default=None, converter=parse_date)
    reduced_speed_end: datetime.datetime = field(default=None, converter=parse_date)
    reduced_speed: float = field(default=0, converter=float)
    non_operational_dates: pd.DatetimeIndex = field(factory=set, init=False)
    reduced_speed_dates: pd.DatetimeIndex = field(factory=set, init=False)
    non_operational_dates_set: pd.DatetimeIndex = field(factory=set, init=False)
    reduced_speed_dates_set: pd.DatetimeIndex = field(factory=set, init=False)


@define(frozen=True, auto_attribs=True)
class FixedCosts(FromDictMixin):
    """The fixed costs for operating a windfarm. All values are assumed to be in $/kW/yr.

    Parameters
    ----------
    operations : float
        Non-maintenance costs of operating the project. If a value is provided for this
        attribute, then it will zero out all other values, otherwise it will be set to
        the sum of the remaining values.
    operations_management_administration: float
        Activities necessary to forecast, dispatch, sell, and manage the production of
        power from the plant. Includes both the onsite and offsite personnel, software,
        and equipment to coordinate high voltage equipment, switching, port activities,
        and marine activities.

        .. note:: This should only be used when not breaking down the cost into the following
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
        Coordination and monitoring to ensure compliance with HSE requirements during
        operations.
    insurance : float
        Insurance policies during operational period including All Risk Property,
        Buisness Interuption, Third Party Liability, and Brokers Fee, and Storm Coverage.

        .. note:: This should only be used when not breaking down the cost into the
            following categories: ``brokers_fee``, ``operations_all_risk``,
            ``business_interruption``, ``third_party_liability``, and/or ``storm_coverage``

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

        .. note:: This should only be used when not breaking down the cost into the
            following categories: ``submerge_land_lease_costs`` and/or
            ``transmission_charges_rights``

    submerge_land_lease_costs : float
        Payments to submerged land owners for rights to build project during operations.
    transmission_charges_rights : float
        Any payments to Transmissions Systems Operators or Transmission Asset Owners for
        rights to transport generated power.
    onshore_electrical_maintenance : float
        Inspections of cables, transformer, switch gears, power compensation equipment,
        etc. and infrequent repairs

        .. warning:: This should only be used if not modeling these as processes within
            the model. Currently, onshore modeling is not included.

    labor : float
        The costs associated with labor, if not being modeled through the simulated
        processes.
    """

    # Total Cost
    operations: float = field(default=0)

    # Operations, Management, and General Administration
    operations_management_administration: float = field(default=0)
    project_management_administration: float = field(default=0)
    marine_management: float = field(default=0)
    weather_forecasting: float = field(default=0)
    condition_monitoring: float = field(default=0)

    operating_facilities: float = field(default=0)
    environmental_health_safety_monitoring: float = field(default=0)

    # Insurance
    insurance: float = field(default=0)
    brokers_fee: float = field(default=0)
    operations_all_risk: float = field(default=0)
    business_interruption: float = field(default=0)
    third_party_liability: float = field(default=0)
    storm_coverage: float = field(default=0)

    # Annual Leases and Fees
    annual_leases_fees: float = field(default=0)
    submerge_land_lease_costs: float = field(default=0)
    transmission_charges_rights: float = field(default=0)

    onshore_electrical_maintenance: float = field(default=0)

    labor: float = field(default=0)

    resolution: dict = field(init=False)
    hierarchy: dict = field(init=False)

    def cost_category_validator(self, name: str, sub_names: list[str]):
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


@define(auto_attribs=True)
class SubString:
    """
    A list of the upstream connections for a turbine and its downstream connector.

    Parameters
    ----------
    downstream : str
        The downstream turbine/substation connection id.
    upstream : list[str]
        A list of the upstream turbine connections.
    """

    downstream: str
    upstream: list[str]


@define(auto_attribs=True)
class String:
    """
    All of the connection information for a complete string in a wind farm.

    Parameters
    ----------
    start : str
        The substation's ID (``System.id``)
    upstream_map : dict[str, SubString]
        The dictionary of each turbine ID in the string and it's upstream ``SubString``.
    """

    start: str
    upstream_map: dict[str, SubString]


@define(auto_attribs=True)
class SubstationMap:
    """
    A mapping of every ``String`` connected to a substation, excluding export
    connections to other substations.

    Parameters
    ----------
    string_starts : list[str]
        A list of every first turbine's ``System.id`` in a string connected to the substation.
    string_map : dict[str, String]
        A dictionary mapping each string starting turbine to its ``String`` data.
    downstream : str
        The ``System.id`` of where the export cable leads. This should be the same
        ``System.id`` as the substation for an interconnection point, or another
        connecting substation.
    """

    string_starts: list[str]
    string_map: dict[str, String]
    downstream: str


@define(auto_attribs=True)
class WindFarmMap:
    """
    A list of the upstream connections for a turbine and its downstream connector.

    Parameters
    ----------
    substation_map : list[str]
        A dictionary mapping of each substation and its ``SubstationMap``.
    export_cables : list[tuple[str, str]]
        A list of the export cable connections.
    """

    substation_map: dict[str, SubstationMap]
    export_cables: list[tuple[str, str]]

    def get_upstream_connections(
        self, substation: str, string_start: str, node: str, return_cables: bool = True
    ) -> list[str] | tuple[list[str], list[str]]:
        """Retrieves the upstream turbines (and optionally cables) within the wind farm graph.

        Parameters
        ----------
        substation : str
            The substation's ``System.id``.
        string_start : str
            The ``System.id`` of the first turbine in the string.
        node : str
            The ``System.id`` of the ending node for a cable connection.
        return_cables : bool
            Indicates if the ``Cable.id`` should be generated for each of the turbines,
            by default True.

        Returns
        -------
        list[str] | tuple[list[str], list[str]]
            A list of ``System.id`` for all of the upstream turbines of ``node`` if
            ``cables=False``, otherwise the upstream turbine and the ``Cable.id`` lists
            are returned.
        """
        strings = self.substation_map[substation].string_map
        upstream = strings[string_start].upstream_map
        turbines = upstream[node].upstream
        if return_cables:
            cables = [
                f"cable::{node if i == 0 else turbines[i - 1]}::{t}"
                for i, t in enumerate(turbines)
            ]
            return turbines, cables
        return turbines

    def get_upstream_connections_from_substation(
        self, substation: str, return_cables: bool = True, by_string: bool = True
    ) -> list[str] | tuple[list[str], list[str]] | list[list[str]] | tuple[
        list[list[str]], list[list[str]]
    ]:
        """Retrieves the upstream turbines (and optionally, cables) connected to a
        py:attr:`substation` in the wind farm graph.

        Parameters
        ----------
        substation : str
            The py:attr:`System.id` for the substation.
        return_cables : bool, optional
            Indicates if the ``Cable.id`` should be generated for each of the turbines,
            by default True
        by_string : bool, optional
            Indicates if the list of turbines (and cables) should be a nested list for
            each string (py:obj:`True`), or as 1-D list (py:obj:`False`), by default True.

        Returns
        -------
        list[str] | tuple[list[str], list[str]]
            A list of ``System.id`` for all of the upstream turbines of ``node`` if
            ``return_cables=False``, otherwise the upstream turbine and the ``Cable.id`` lists
            are returned. These are bifurcated in lists of lists for each string if ``by_string=True``
        """
        turbines, cables = [], []
        substation_map = self.substation_map[substation]
        start_nodes = substation_map.string_starts
        for start_node in start_nodes:

            # Add the starting node of the string and substation-turbine array cable
            _turbines = [start_node]
            _cables = [f"cable::{substation}::{start_node}"]

            # Add the main components of the string
            _t, _c = self.get_upstream_connections(substation, start_node, start_node)
            _turbines.extend(_t)
            _cables.extend(_c)
            turbines.append(_turbines)
            cables.append(_cables)

        if not by_string:
            turbines = [el for string in turbines for el in string]  # type: ignore
            cables = [el for string in cables for el in string]  # type: ignore

        if return_cables:
            return turbines, cables
        return turbines
