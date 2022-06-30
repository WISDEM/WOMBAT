"""A common location for base class mixins that provide functionality shared across
different types of classes.
"""

from __future__ import annotations

from typing import Generator
from functools import partial

import numpy as np
import pandas as pd
from simpy.events import Process, Timeout

from wombat.utilities import check_working_hours
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
    settings: PortConfig | ScheduledServiceEquipmentData | UnscheduledServiceEquipmentData

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

    def _check_working_hours(self) -> None:
        """Checks the working hours of the port and overrides a default (-1) to
        the ``env`` settings, otherwise hours remain the same.
        """
        self.settings._set_environment_shift(
            *check_working_hours(
                self.env.workday_start,  # type: ignore
                self.env.workday_end,
                self.settings.workday_start,
                self.settings.workday_end,
            )
        )

    def _is_workshift(self, datetime_ix: pd.DatetimeIndex) -> np.ndarray:
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
            location="port" if isinstance(self.settings, PortConfig) else "site",
            **kwargs,
        )
        yield self.env.timeout(hours)
