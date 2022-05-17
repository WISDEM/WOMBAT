"""A common location for base class mixins that provide functionality shared across
different types of classes.
"""

from __future__ import annotations

from typing import Generator

import numpy as np
from attrs import field, define
from simpy.events import Process, Timeout

from wombat.utilities.utilities import check_working_hours

from .data_classes import Failure, Maintenance


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
        kwargs = {
            a.name: data[a.name]
            for a in cls.__attrs_attrs__  # type: ignore
            if a.name in data and a.init
        }

        # Map the inputs must be provided: 1) must be initialized, 2) no default value defined
        required_inputs = [
            a.name
            for a in cls.__attrs_attrs__  # type: ignore
            if a.init and isinstance(a.default, attr._make._Nothing)  # type: ignore
        ]
        undefined = sorted(set(required_inputs) - set(kwargs))
        if undefined:
            raise AttributeError(
                f"The class defintion for {cls.__name__} is missing the following inputs: {undefined}"
            )
        return cls(**kwargs)  # type: ignore


class RepairsMixin:
    """A Mixin class that provides some common repair and timing methods that are used
    across different repair-providing classes, such as: servicing equipment and ports.
    """

    def _check_working_hours(self) -> None:
        """Checks the working hours of the port and overrides a default (-1) to
        the ``env`` settings, otherwise hours remain the same.
        """
        self.settings._set_environment_shift(
            *check_working_hours(
                self.env.workday_start,
                self.env.workday_end,
                self.settings.workday_start,
                self.settings.workday_end,
            )
        )

    def _is_workshift(self, datetime_ix: DatetimeIndex) -> np.ndarray:
        """Determines which timestamps are in the servicing equipment's working hours.

        Parameters
        ----------
        datetime_ix : DatetimeIndex
            A pandas DatetimeIndex that needs to be assessed.

        Returns
        -------
        np.ndarray
            A boolean array for which values in working hours (True), and which values
            are outside working hours (False).
        """
        start = self.settings.workday_start
        end = self.settings.workday_end
        hour = datetime_ix.hour
        return (start <= hour) & (hour <= end)

    def wait_until_next_shift(self, **kwargs) -> Generator[Timeout, None, None]:
        """Delays the process until the start of the next shift.

        Yields
        -------
        Generator[Timeout, None, None]
            Delay until the start of the next shift.
        """
        kwargs["additional"] = kwargs.get(
            "additional", "work shift has ended; waiting for next shift to start"
        )
        kwargs["action"] = kwargs.get("action", "delay")
        delay = self.env.hours_to_next_shift(workday_start=self.settings.workday_start)
        salary_cost = self.calculate_salary_cost(delay)
        hourly_cost = self.calculate_hourly_cost(0)
        equpipment_cost = self.calculate_equipment_cost(delay)
        self.env.log_action(
            duration=delay,
            salary_labor_cost=salary_cost,
            hourly_labor_cost=hourly_cost,
            equipment_cost=equpipment_cost,
            **kwargs,
        )
        yield self.env.timeout(delay)

    def process_repair(
        self, hours: int | float, request_details: Maintenance | Failure, **kwargs
    ) -> None | Generator[Timeout | Process, None, None]:
        """The logging and timeout process for performing a repair or doing maintenance.

        Parameters
        ----------
        hours : int | float
            The lenght, in hours, of the repair or maintenance task.
        request_details : Maintenance | Failure
            The deatils of the request, this is only being checked for the type.

        Yields
        -------
        None | Generator[Timeout | Process, None, None]
            A ``Timeout`` is yielded of length ``hours``.
        """
        action = "repair" if isinstance(request_details, Failure) else "maintenance"
        salary_cost = self.calculate_salary_cost(hours)
        hourly_cost = self.calculate_hourly_cost(hours)
        equipment_cost = self.calculate_equipment_cost(hours)
        self.env.log_action(
            action=action,
            duration=hours,
            salary_labor_cost=salary_cost,
            hourly_labor_cost=hourly_cost,
            equipment_cost=equipment_cost,
            additional=action,
            **kwargs,
        )
        yield self.env.timeout(hours)
