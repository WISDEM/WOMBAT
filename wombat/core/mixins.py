"""A common location for base class mixins that provide functionality shared across
different types of classes.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING
from functools import partial
from collections.abc import Generator

import numpy as np
import pandas as pd
from simpy.events import Process, Timeout

from wombat.utilities import HOURS_IN_DAY, check_working_hours
from wombat.utilities.time import calculate_cost
from wombat.core.environment import WombatEnvironment

from .data_classes import (
    Failure,
    PortConfig,
    Maintenance,
    ScheduledServiceEquipmentData,
    UnscheduledServiceEquipmentData,
)


class RepairsMixin:
    """A Mixin class that provides some common repair and timing methods that are used
    across different repair-providing classes, such as: servicing equipment and ports.
    """

    env: WombatEnvironment
    settings: (
        PortConfig | ScheduledServiceEquipmentData | UnscheduledServiceEquipmentData
    )  # noqa: E501

    def initialize_cost_calculators(self, which: str) -> None:
        """Creates the cost calculators for each of the subclasses that will need to
        calculate hourly costs.

        Parameters
        ----------
        which : str
            One of "port" or "equipment" to to indicate how to access equipment costs
        """
        self.calculate_salary_cost = partial(
            calculate_cost,
            rate=self.settings.crew.day_rate,
            n_rate=self.settings.crew.n_day_rate,
            daily_rate=True,
        )
        self.calculate_hourly_cost = partial(
            calculate_cost,
            rate=self.settings.crew.hourly_rate,
            n_rate=self.settings.crew.n_hourly_rate,
        )
        self.calculate_equipment_cost = partial(
            calculate_cost,
            rate=0 if which == "port" else self.settings.equipment_rate,  # type: ignore
            daily_rate=True,
        )

    def _check_working_hours(self, which: str) -> None:
        """Checks the working hours of the port and overrides a default (-1) to
        the ``env`` settings, otherwise hours remain the same.

        Parameters
        ----------
        which : str
            One of "env" or "port" to determine from which overarching environment
            variable should be used to override unset settings.
        """
        if which == "env":
            start = self.env.workday_start
            end = self.env.workday_end
        elif which == "port":
            if TYPE_CHECKING:
                assert hasattr(self, "port")
            start = self.port.settings.workday_start
            end = self.port.settings.workday_end
        else:
            raise ValueError(
                "Can only set the workday settings from a 'port' or 'env'."
            )

        self.settings._set_environment_shift(
            *check_working_hours(
                start, end, self.settings.workday_start, self.settings.workday_end
            )
        )

    def _is_workshift(self, hour_ix: np.ndarray | float | int) -> np.ndarray | bool:
        """Determines which timestamps are in the servicing equipment's working hours.

        Parameters
        ----------
        hour_ix : np.ndarray | float | int
            The hour of day component of the datetime stamp.

        Returns
        -------
        np.ndarray | bool
            A boolean array for which values in working hours (True), and which values
            are outside working hours (False).
        """
        if not self.settings.non_stop_shift:
            is_workshift = self.settings.workday_start <= hour_ix
            is_workshift &= hour_ix <= self.settings.workday_end
            return is_workshift

        if isinstance(hour_ix, np.ndarray):
            return np.ones(hour_ix.shape, dtype=bool)
        return True

    def wait_until_next_shift(self, **kwargs) -> Generator[Timeout]:
        """Delays the process until the start of the next shift.

        Yields
        ------
        Generator[Timeout, None, None]
            Delay until the start of the next shift.
        """
        kwargs["additional"] = kwargs.get(
            "additional", "work shift has ended; waiting for next shift to start"
        )
        kwargs["action"] = kwargs.get("action", "delay")

        # Assume at port, unless otherwise indicated
        if "location" not in kwargs:
            kwargs["location"] = "port"

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
    ) -> Generator[Timeout | Process]:
        """The logging and timeout process for performing a repair or doing maintenance.

        Parameters
        ----------
        hours : int | float
            The lenght, in hours, of the repair or maintenance task.
        request_details : Maintenance | Failure
            The deatils of the request, this is only being checked for the type.
        kwargs : dict
            Additional parameters to be passed to ``WombatEnvironment.log_action``.

        Yields
        ------
        Generator[Timeout | Process, None, None]
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
            location="port" if isinstance(self.settings, PortConfig) else "system",
            **kwargs,
        )
        yield self.env.timeout(hours)

    def hours_to_next_operational_date(
        self,
        start_search_date: datetime.datetime | datetime.date | pd.Timestamp,
        exclusion_days: int = 0,
    ) -> float:
        """Calculates the number of hours until the first date that is not a part of
        the ``non_operational_dates`` given a starting datetime and for search criteria.
        Optionally, ``exclusion_days`` can be used to account for a mobilization period
        so that mobilization can occur during the non operational period.

        Parameters
        ----------
        start_search_date : datetime.datetime | datetime.date | pd.Timestamp
            The first date to be considered in the search.
        exclusion_days : int, optional
            The number of days to subtract from the next available datetime that
            represents a non operational action that can occur during the non
            operational period, such as mobilization, by default 0.

        Returns
        -------
        float
            The total number of hours until the next operational date.
        """
        current = self.env.simulation_time

        if isinstance(start_search_date, datetime.date):
            start_search_date = datetime.datetime.combine(
                start_search_date, datetime.datetime.max.time()
            )

        # Get the forecast and filter out dates prior to the start of the search
        all_dates, *_ = self.env.weather_forecast(8760)
        dates = all_dates.filter(all_dates > start_search_date)
        if dates.shape[0] == 0:
            dates = all_dates.slice(-1)

        # Find the next operational date, and if no available dates, return the time
        # until the end of the forecast period
        date_diff = set(dates.dt.date()).difference(
            self.settings.non_operational_dates_set
        )
        if not date_diff:
            diff = ((max(dates) - current).days + 1) * HOURS_IN_DAY
            return diff
        next_available = (
            (dates.filter(dates.dt.date() == min(date_diff))).slice(0, 1).item()
        )

        # Compute the difference between the current time and the future date
        diff = next_available - current
        hours_to_next = diff.days * HOURS_IN_DAY + diff.seconds / 60 / 60
        hours_to_next -= exclusion_days * HOURS_IN_DAY
        hours_to_next = max(0, hours_to_next)
        return hours_to_next
