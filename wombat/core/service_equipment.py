"""The servicing equipment module provides a small number of utility functions specific
to the operations of servicing equipment and the `ServiceEquipment` class that provides
the repair and transportation logic for scheduled, unscheduled, and unscheduled towing
servicing equipment.
"""
# TODO: NEED A SPECIFIC STARTUP METHOD
from __future__ import annotations

import logging
from copy import deepcopy
from math import ceil
from typing import TYPE_CHECKING, Any, Optional, Generator
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
from simpy.events import Event, Process, Timeout
from pandas.core.indexes.datetimes import DatetimeIndex

from wombat.core import (
    Maintenance,
    RepairManager,
    RepairRequest,
    WombatEnvironment,
    ServiceEquipmentData,
)
from wombat.windfarm import Windfarm
from wombat.utilities import HOURS_IN_DAY, cache, hours_until_future_hour
from wombat.core.mixins import RepairsMixin
from wombat.core.library import load_yaml
from wombat.windfarm.system import System
from wombat.core.data_classes import (
    UNSCHEDULED_STRATEGIES,
    ScheduledServiceEquipmentData,
    UnscheduledServiceEquipmentData,
)
from wombat.windfarm.system.cable import Cable
from wombat.windfarm.system.subassembly import Subassembly


if TYPE_CHECKING:
    from wombat.core import Port


def consecutive_groups(data: np.ndarray, step_size: int = 1) -> list[np.ndarray]:
    """Generates the subgroups of an array where the difference between two sequential
    elements is equal to the ``step_size``. The intent is to find the length of delays
    in a weather forecast

    Parameters
    ----------
    data : np.ndarray
        An array of integers.
    step_size : int, optional
        The step size to be considered a consecutive number, by default 1.

    Returns
    -------
    list[np.ndarray]
        A list of arrays of the consecutive elements of ``data``.
    """
    # Consecutive groups in delay
    groups = np.split(data, np.where(np.diff(data) != step_size)[0] + 1)
    return groups


def calculate_delay_from_forecast(
    forecast: np.ndarray, hours_required: int | float
) -> tuple[bool, int]:
    """Calculates the delay from the binary weather forecast for if that hour is all
    clear for operations.

    Parameters
    ----------
    forecast : np.ndarray
        Truth array to indicate if that hour satisfies the weather limit requirements.
    hours_required : np.ndarray
        The minimum clear weather window required, in hours.

    Returns
    -------
    tuple[bool, int]
        Indicator if a window is found (``True``) or not (``False``), and the number
        of hours the event needs to be delayed in order to start.
    """
    safe_operating_windows = consecutive_groups(np.where(forecast)[0])
    window_lengths = np.array([window.size for window in safe_operating_windows])
    clear_windows = np.where(window_lengths >= hours_required)[0]
    if clear_windows.size == 0:
        return False, forecast.size
    return True, safe_operating_windows[clear_windows[0]][0]


def validate_end_points(start: str, end: str, no_intrasite: bool = False) -> None:
    """Checks the starting and ending locations for traveling and towing.

    Parameters
    ----------
    start : str
        The starting location; should be on of: "site", "system", or "port".
    end : str
        The ending location; should be on of: "site", "system", or "port".
    no_intrasite : bool
        A flag to disable intrasite travel, so that ``start`` and ``end`` cannot
        both be "system", by default False.

    Raises
    ------
    ValueError
        Raised if the starting location is invalid.
    ValueError
        Raised if the ending location is invalid
    ValueError
        Raised if "system" is provided to both ``start`` and ``end``, but
        ``no_intrasite`` is set to ``True``.
    """
    if start not in ("port", "site", "system"):
        raise ValueError(
            "``start`` location must be one of 'port', 'site', or 'system'!"
        )
    if end not in ("port", "site", "system"):
        raise ValueError("``end`` location must be one of 'port', 'site', or 'system'!")
    if no_intrasite and (start == end == "system"):
        raise ValueError("No travel within the site is allowed for this process")


def reset_system_operations(system: System) -> None:
    """Completely resets the failure and maintenance events for a given system
    and its subassemblies, and puts each ``Subassembly.operating_level`` back to 100%.

    .. note:: This is only intended to be used in conjunction with a tow-to-port
        repair where a turbine will be completely serviced.

    Parameters
    ----------
    system : System
        The turbine to be reset.
    """
    for subassembly in system.subassemblies:
        subassembly.operating_level = 1.0
        subassembly.recreate_processes()
    system.servicing.succeed()


class ServiceEquipment(RepairsMixin):
    """Provides a servicing equipment object that can handle various maintenance and
    repair tasks.

    Parameters
    ----------
    env : WombatEnvironment
        The simulation environment.
    windfarm : Windfarm
        The ``Windfarm`` object.
    repair_manager : RepairManager
        The ``RepairManager`` object.
    equipment_data_file : str
        The equipment settings file name with extension.

    Attributes
    ----------
    env : WombatEnvironment
        The simulation environment instance.
    windfarm : Windfarm
        The simulation windfarm instance.
    manager : RepairManager
        The simulation repair manager instance.
    settings : PortConfig
        The port's configuration settings, as provided by the user.
    onsite : bool
        Indicates if the servicing equipment is at the site (``True``), or not
        (``False``).
    enroute : bool
        Indicates if the servicing equipment is on its way to the site (``True``),
        or not (``False``).
    at_port : bool
        Indicates if the servicing equipment is at the port, or similar location for
        land-based, (``True``), or not (``False``).
    at_system : bool
        Indications if the servicing equipment is at a cable, substation, or turbine
        while on the site (``True``), or not (``False``).
    current_system : str | None
        Either the ``System.id`` if ``at_system``, or ``None`` if not.
    transferring_crew : bool
        Indications if the servicing equipment is at a cable, substation, or turbine
        and transferring the crew to or from that system (``True``), or not
        (``False``).
    """

    def __init__(
        self,
        env: WombatEnvironment,
        windfarm: Windfarm,
        repair_manager: RepairManager,
        equipment_data_file: str | Path | dict,
    ):
        """Initializes the ``ServiceEquipment`` class.

        Parameters
        ----------
        env : WombatEnvironment
            The simulation environment.
        windfarm : Windfarm
            The ``Windfarm`` object.
        repair_manager : RepairManager
            The ``RepairManager`` object.
        equipment_data_file : str
            The equipment settings file name with extension.
        """
        self.env = env
        self.windfarm = windfarm
        self.manager = repair_manager
        self.onsite = False
        self.dispatched = False
        self.enroute = False
        self.at_port = False
        self.at_system = False
        self.transferring_crew = False
        self.current_system = None  # type: str | None

        if isinstance(equipment_data_file, (str, Path)):
            try:
                data = load_yaml(env.data_dir / "vessels", equipment_data_file)
            except FileNotFoundError:
                data = load_yaml(env.data_dir / "repair/transport", equipment_data_file)
                logging.warning(
                    "DeprecationWarning: In v0.7, all servicing equipment configurations must be located in: '<library>/vessels/"
                )
        else:
            data = equipment_data_file

        try:
            if data["start_year"] < self.env.start_year:
                data["start_year"] = self.env.start_year
            if data["end_year"] > self.env.end_year:
                data["end_year"] = self.env.end_year
        except KeyError:
            # Ignores for unscheduled maintenace equipment that would not have this input
            pass

        # NOTE: mypy is not caught up with attrs yet :(
        self.settings: ScheduledServiceEquipmentData | UnscheduledServiceEquipmentData = ServiceEquipmentData(data).determine_type()  # type: ignore

        # Register servicing equipment with the repair manager if it is using an
        # unscheduled maintenance scenario, so it can be dispatched as needed
        if self.settings.strategy in UNSCHEDULED_STRATEGIES:
            self.manager._register_equipment(self)

        # Only run the equipment if it is on a scheduled basis, otherwise wait
        # for it to be dispatched
        if self.settings.strategy == "scheduled":
            self.env.process(self.run_scheduled_in_situ())

        # Create partial functions for the labor and equipment costs for clarity
        self.initialize_cost_calculators(which="equipment")

    def finish_setup_with_environment_variables(self) -> None:
        """A post-initialization step that will override unset parameters with those
        from the the environemt that may have already been set.
        """
        self._check_working_hours(which="env")
        self.settings._set_port_distance(self.env.port_distance)
        self.settings.set_non_operational_dates(
            self.env.non_operational_start,
            self.env.start_year,
            self.env.non_operational_end,
            self.env.end_year,
        )
        self.settings.set_reduced_speed_parameters(
            self.env.reduced_speed_start,
            self.env.start_year,
            self.env.reduced_speed_end,
            self.env.end_year,
            self.env.reduced_speed,
        )

    def _register_port(self, port: "Port") -> None:
        """Method for a tugboat at attach the port for two-way communications. This also
        sets the vessel to be at the port, and updates the port_distance.

        Parameters
        ----------
        port : Port
            The port where the tugboat is based.
        """
        self.port = port
        self.at_port = True
        self.settings._set_port_distance(port.settings.site_distance)  # type: ignore

        self._check_working_hours(which="port")

        # Set the non-operational start/end dates if needed
        self.settings.set_non_operational_dates(
            port.settings.non_operational_start,
            self.env.start_year,
            port.settings.non_operational_end,
            self.env.end_year,
        )

        # Set the reduced speed start/end dates if needed
        self.settings.set_reduced_speed_parameters(
            port.settings.reduced_speed_start,
            self.env.start_year,
            port.settings.reduced_speed_end,
            self.env.end_year,
            port.settings.reduced_speed,
        )

    def _set_location(self, end: str, set_current: Optional[str] = None) -> None:
        """Keeps track of the servicing equipment by setting the location at either:
        site, port, or a specific system

        Parameters
        ----------
        start : str
            The starting location; one of "site", or "port"
        end : str
            The ending location; one of "site", or "port"
        set_current : str
            The ``System.id`` for the new current location, if one is to be set.
        """
        if end == "port":
            self.at_port = True
            self.onsite = False
        else:
            self.at_port = False
            self.onsite = True
        self.at_system = True if set_current is not None else False
        self.current_system = set_current

    def _weather_forecast(
        self, hours: int | float, which: str
    ) -> tuple[DatetimeIndex, np.ndarray, np.ndarray]:
        """Retrieves the weather forecast from the simulation environment, and
        translates it to a boolean for satisfactory (True) and unsatisfactory (False)
        weather conditions.

        Parameters
        ----------
        hours : int | float
            The number of hours of weather data that should be retrieved.
        which : str
            One of "repair" or "transport" to indicate which weather limits to be using.

        Returns
        -------
        tuple[DatetimeIndex, np.ndarray, np.ndarray]
            The pandas DatetimeIndex, the hour of day array, and the boolean array of
            where the weather conditions are within safe operating limits for the
            servicing equipment (True) or not (False).
        """
        if which == "repair":
            max_wind = self.settings.max_windspeed_repair
            max_wave = self.settings.max_waveheight_repair
        elif which == "transport":
            max_wind = self.settings.max_windspeed_transport
            max_wave = self.settings.max_waveheight_transport
        else:
            raise ValueError("`which` must be one of `repair` or `transport`.")

        dt, hour, wind, wave = self.env.weather_forecast(hours)
        all_clear = (wind <= max_wind) & (wave <= max_wave)
        return dt, hour, all_clear

    def get_speed(self, tow: bool = False) -> float:
        """Determines the appropriate speed that the servicing equipment should be
        traveling at for towing or traveling, and if the timeframe is during a reduced
        speed scenario.

        Parameters
        ----------
        tow : bool, optional
            True indicates the servicing equipment should be using the towing speed,
            and if False, then the traveling speed should be used, by default False.

        Returns
        -------
        float
            The maximum speed the servicing equipment should be traveling/towing at.
        """
        speed = self.settings.tow_speed if tow else self.settings.speed  # type: ignore
        if self.env.simulation_time.date() in self.settings.reduced_speed_dates_set:  # type: ignore
            if speed > self.settings.reduced_speed:
                speed = self.settings.reduced_speed
        return speed

    def get_next_request(self):
        """Gets the next request by the rig's method for processing repairs.

        Returns
        -------
        simpy.resources.store.FilterStoreGet
            The next ``RepairRequest`` to be processed.
        """
        if self.settings.method == "turbine":
            return self.manager.get_request_by_system(self.settings.capability)
        elif self.settings.method == "severity":
            return self.manager.get_next_highest_severity_request(
                self.settings.capability
            )

    def enable_string_operations(self, cable: Cable) -> None:
        """Traverses the upstream cable and turbine connections and resets the
        ``System.cable_failure`` and ``Cable.downstream_failure`` until it hits
        another cable failure, then the loop exits.

        Parameters
        ----------
        subassembly : Cable
            The `Cable` or `System`
        """
        farm = self.manager.windfarm
        if cable.connection_type == "array":
            # If there is another failure downstream of the repaired cable, do nothing
            if cable.downstream_failure:
                return

            # For each upstream turbine and cable, reset their operations
            for t_id, c_id in zip(cable.upstream_nodes, cable.upstream_cables):
                cable = farm.cable(c_id)
                if cable.operating_level == 0:
                    break
                cable.downstream_failure.succeed()
                farm.system(t_id).cable_failure.succeed()

        if cable.connection_type == "export":
            # Reset the substation's cable failure
            farm.system(cable.end_node).cable_failure.succeed()

            # For each string connected to the substation reset all the turbines and cables
            # until another cable failure is encoountered, then move to the next string
            for turbines, cables in zip(cable.upstream_nodes, cable.upstream_cables):
                for t_id, c_id in zip(turbines, cables):
                    cable = farm.cable(c_id)
                    if cable.operating_level == 0:
                        break
                    cable.downstream_failure.succeed()
                    farm.system(t_id).cable_failure.succeed()

    def register_repair_with_subassembly(
        self,
        subassembly: Subassembly | Cable,
        repair: RepairRequest,
        starting_operating_level: float,
    ) -> None:
        """Goes into the repaired subassembly, component, or cable and returns its
        ``operating_level`` back to good as new for the specific repair. For fatal cable
        failures, all upstream turbines are turned back on unless there is another fatal
        cable failure preventing any more from operating.

        Parameters
        ----------
        subassembly : Subassembly | Cable
            The subassembly or cable that was repaired.
        repair : RepairRequest
            The request for repair that was submitted.
        starting_operating_level : float
            The operating level before a repair was started.
        """
        operation_reduction = repair.details.operation_reduction

        if repair.cable and repair.details.operation_reduction == 1:
            assert isinstance(subassembly, Cable)  # mypy helper
            self.enable_string_operations(subassembly)

        # Put the subassembly/component back to good as new condition
        if repair.details.replacement:
            subassembly.operating_level = 1
            _ = self.manager.purge_subassembly_requests(
                repair.system_id, repair.subassembly_id
            )
        elif operation_reduction == 1:
            subassembly.operating_level = starting_operating_level
            subassembly.broken.succeed()
        elif operation_reduction == 0:
            subassembly.operating_level = starting_operating_level
        else:
            subassembly.operating_level /= 1 - operation_reduction

        # Register that the servicing is now over
        system = subassembly if isinstance(subassembly, Cable) else subassembly.system
        system.servicing.succeed()
        self.manager.enable_requests_for_system(repair.system_id)

    def wait_until_next_operational_period(
        self, *, less_mobilization_hours: int = 0
    ) -> Generator[Timeout, None, None]:
        """Delays the crew and equipment until the start of the next operational
        period.

        TODO: Need a custom error if weather doesn't align with the equipment dates.

        Parameters
        ----------
        less_mobilization_hours : int
            The number of hours required for mobilization that will be subtracted from
            the waiting period to account for mobilization, by default 0.

        Yields
        -------
        Generator[Timeout, None, None]
            A Timeout event for the number of hours between when the function is called
            and when the next operational period starts.
        """
        current = self.env.simulation_time.date()
        ix_match = np.where(current < self.settings.operating_dates)[0]  # type: ignore
        if ix_match.size > 0:
            still_in_operation = True
            next_operating_date = self.settings.operating_dates[ix_match[0]]  # type: ignore
            hours_to_next_shift = (
                self.env.date_ix(next_operating_date)
                - self.env.now
                + self.settings.workday_start
                - less_mobilization_hours
            )
        else:
            still_in_operation = False
            hours_to_next_shift = self.env.max_run_time - self.env.now

        if still_in_operation:
            # TODO: This message needs to account for unscheduled equipment as well, but
            # the post processor filtering needs to be updated as well
            additional = "will return next year"
        else:
            additional = "no more return visits will be made"

        # Ensures that the statuses are correct
        self.at_port = False
        self.enroute = False
        self.onsite = False

        self.env.log_action(
            agent=self.settings.name,
            action="delay",
            reason="work is complete",
            additional=additional,
            duration=hours_to_next_shift,
            location="enroute",
        )

        yield self.env.timeout(hours_to_next_shift)

    def mobilize_scheduled(self) -> Generator[Timeout, None, None]:
        """Mobilizes the ServiceEquipment object by waiting for the next operational
        period, less the mobilization time, then logging the mobiliztion cost.

        NOTE: weather delays are not accounted for in this process.

        Yields
        -------
        Generator[Timeout, None, None]
            A Timeout event for the number of hours between when the function is called
            and when the next operational period starts.
        """
        mobilization_hours = self.settings.mobilization_days * HOURS_IN_DAY
        yield self.env.process(  # type: ignore
            self.wait_until_next_operational_period(
                less_mobilization_hours=mobilization_hours
            )
        )
        self.env.log_action(
            agent=self.settings.name,
            action="mobilization",
            reason=f"{self.settings.name} is being mobilized",
            additional="mobilization",
            location="enroute",
        )

        self.enroute = True
        yield self.env.timeout(mobilization_hours)
        self.onsite = True
        self.enroute = False

        self.env.log_action(
            agent=self.settings.name,
            action="mobilization",
            reason=f"{self.settings.name} has arrived on site",
            additional="mobilization",
            duration=mobilization_hours,
            equipment_cost=self.settings.mobilization_cost,
            location="site",
        )

    def mobilize(self) -> Generator[Timeout, None, None]:
        """Mobilizes the ServiceEquipment object.

        NOTE: weather delays are not accounted for at this stage.

        Yields
        -------
        Generator[Timeout, None, None]
            A Timeout event for the number of hours the ServiceEquipment requires for
            mobilizing to the windfarm site.
        """
        self.env.log_action(
            agent=self.settings.name,
            action="mobilization",
            reason=f"{self.settings.name} is being mobilized",
            additional="mobilization",
            location="enroute",
        )

        self.enroute = True
        mobilization_hours = self.settings.mobilization_days * HOURS_IN_DAY
        yield self.env.timeout(mobilization_hours)
        self.onsite = True
        self.enroute = False

        self.env.log_action(
            agent=self.settings.name,
            action="mobilization",
            reason=f"{self.settings.name} has arrived on site",
            additional="mobilization",
            duration=mobilization_hours,
            equipment_cost=self.settings.mobilization_cost,
            location="site",
        )

    def find_uninterrupted_weather_window(
        self, hours_required: int | float
    ) -> tuple[int | float, bool]:
        """Finds the delay required before starting on a process that won't be able to
        be interrupted by a weather delay.

        TODO: WEATHER FORECAST NEEDS TO BE DONE FOR ``math.floor(self.now)``, not the ceiling
        or there will be a whole lot of rounding up errors on process times.

        Parameters
        ----------
        hours_required : int | float
            The number of uninterrupted of hours that a process requires for completion.

        Returns
        -------
        tuple[int | float, bool]
            The number of hours in weather delays before a process can be completed, and
            an indicator for if the process has to be delayed until the next shift for
            a safe transfer.
        """
        # If the hours required for the window is 0, then return 0 and indicate there is
        # no shift delay to be processed
        if hours_required == 0:
            return 0, False

        current = self.env.simulation_time

        # If the time required for a transfer is longer than the time left in the shift,
        # then return the hours left in the shift and indicate a shift delay
        max_hours = hours_until_future_hour(
            self.env.simulation_time, self.settings.workday_end
        )
        if hours_required > max_hours:
            return max_hours, True

        _, hour, all_clear = self._weather_forecast(max_hours, which="repair")
        all_clear &= self._is_workshift(hour)
        safe_operating_windows = consecutive_groups(np.where(all_clear)[0])
        window_lengths = np.array([window.size for window in safe_operating_windows])

        # If all the safe operating windows are shorter than the time required, then
        # return the time until the end of the shift and indicate a shift delay
        if all([window < hours_required for window in window_lengths]):
            return max_hours, True

        # Find the length of the delay
        delay = safe_operating_windows[
            np.where(window_lengths >= hours_required)[0][0]
        ][0]

        # If there is no delay, simply return 0 and the shift delay indicator
        if delay == 0:
            return delay, False

        # If the delay is non-zero, ensure we get the correct hour difference from now
        # until the top of the first available hour
        delay = hours_until_future_hour(current, current.hour + delay)
        return delay, False

    def find_interrupted_weather_window(
        self, hours_required: int | float
    ) -> tuple[DatetimeIndex, np.ndarray, bool]:
        """Assesses at which points in the repair window, the wind (and wave)
        constraints for safe operation are met.

        The initial search looks for a weather window of length ``hours_required``, and
        adds 24 hours to the window for the proceeding 9 days. If no satisfactory window
        is found, then the calling process must make another call to this function, but
        likely there is something wrong with the constraints or weather conditions if
        no window is found within 10 days.

        Parameters
        ----------
        hours_required : int | float
            The number of hours required to complete the repair.

        Returns
        -------
        tuple[DatetimeIndex, np.ndarray, bool]
            The pandas DatetimeIndex, and a corresponding boolean array for what points
            in the time window are safe to complete a maintenance task or repair.
        """
        window_found = False
        hours_required = ceil(hours_required)

        for i in range(10):  # no more than 10 attempts to find a window
            dt_ix, hour_ix, weather = self._weather_forecast(
                hours_required + (i * 24), which="repair"
            )
            working_hours = self._is_workshift(hour_ix)
            window = weather & working_hours
            if window.sum() >= hours_required:
                window_found = True
                break

        return dt_ix, weather, window_found

    def weather_delay(self, hours: int | float, **kwargs) -> Generator[Event, Any, Any]:
        """Processes a weather delay of length ``hours`` hours. If ``hours`` = 0, then
        a Timeout is still processed, but not logging is done (this is to increase
        consistency and do better typing validation across the codebase).

        Parameters
        ----------
        hours : int | float
            The lenght, in hours, of the weather delay.

        Yields
        -------
        Generator[Event, Any, Any]
            If the delay is more than 0 hours, then a ``Timeout`` is yielded of length ``hours``.
        """
        if hours > 0:
            salary_cost = self.calculate_salary_cost(hours)
            hourly_cost = 0  # contractors not paid for delays
            equipment_cost = self.calculate_equipment_cost(hours)
            self.env.log_action(
                duration=hours,
                action="delay",
                additional="weather delay",
                salary_labor_cost=salary_cost,
                hourly_labor_cost=hourly_cost,
                equipment_cost=equipment_cost,
                **kwargs,
            )
            yield self.env.timeout(hours)

        yield self.env.timeout(hours)

    @cache
    def _calculate_intra_site_time(
        self, start: str | None, end: str | None
    ) -> tuple[float, float]:
        """Calculates the time it takes to travel between port and site or between
        systems on site.

        Parameters
        ----------
        start : str | None
            The starting onsite location. If ``None``, then 0 is returned.
        end : str | None
            The ending onsite location. If ``None``, then 0 is returned.

        Returns
        -------
        tuple[float, float]
            The travel time and distance between two locations.
        """
        distance = 0.0  # setting for invalid cases to have no traveling
        valid_sys = self.windfarm.distance_matrix.columns
        intra_site = start in valid_sys and end in valid_sys
        if intra_site:
            distance = self.windfarm.distance_matrix.loc[start, end]

        # if no speed is set, then there is no traveling time
        speed = self.get_speed()
        travel_time = distance / speed if speed > 0 else 0.0

        # Infinity is the result of "traveling" between a system twice in a row
        if travel_time == float("inf"):
            travel_time = 0
        return travel_time, distance

    def _calculate_uninterrupted_travel_time(
        self, distance: float, tow: bool = False
    ) -> tuple[float, float]:
        """Calculates the delay to the start of traveling and the amount of time it
        will take to travel between two locations.

        Parameters
        ----------
        distance : float
            The distance to be traveled.
        tow : bool
            Indicates if this travel is for towing (True), or not (False), by default
            False.

        Returns
        -------
        tuple[float, float]
            The length of the delay and the length of travel time, in hours.` -1 is
            returned if there no weather windows, and the process will have to be attempted
            again.
        """
        # Return 0 delay and travel time in the distance is zero
        if distance == 0:
            return 0, 0

        speed = self.get_speed(tow=tow)
        hours_required = distance / speed

        n = 1
        max_extra_days = 4
        while n <= max_extra_days:  # max tries before failing
            hours = hours_required + (24 * n)
            *_, all_clear = self._weather_forecast(hours, which="repair")
            found, delay = calculate_delay_from_forecast(all_clear, hours_required)
            if found:
                return delay, hours_required
            n += 1

        # Return -1 for delay if no weather window was found
        return -1, hours_required

    def _calculate_interrupted_travel_time(
        self, distance: float, tow: bool = False
    ) -> float:
        """Calculates the travel time with speed reductions for inclement weather, but
        without shift interruptions.

        Parameters
        ----------
        distance : flaot
            The total distance to be traveled, in km.
        tow : bool
            Indicates if this travel is for towing (True), or not (False), by default
            False.

        Returns
        -------
        float
            _description_
        """
        speed = self.get_speed(tow=tow)
        reduction_factor = 1 - self.settings.speed_reduction_factor
        reduction_factor = 0.01 if reduction_factor == 0 else reduction_factor

        # get the weather forecast with this time for the max travel time
        max_hours = 1 + distance / speed * (1 / reduction_factor)
        dt, _, all_clear = self._weather_forecast(max_hours, which="transport")

        # calculate the distance able to be traveled in each 1-hour window
        distance_traveled = speed * all_clear.astype(float)
        distance_traveled[distance_traveled == 0] = speed * reduction_factor

        # Reduce the first time step by the time lapsed since the start of the hour
        # before proceeding
        distance_traveled[0] *= 1 - self.env.now % 1

        # Cumulative sum at the end of each full hour
        distance_traveled_sum = distance_traveled.cumsum()

        # Get the index for the end of the hour where the distance requred to be
        # traveled is reached.
        try:
            ix_hours = np.where(distance_traveled_sum >= distance)[0][0]
        except IndexError as e:
            # If an error occurs because an index maxes out the weather window, check
            # that it's not due to having reached the end of the simulation. If so,
            # return the max amount of time, but if that's not the case re-raise the error.
            if self.env.weather.index.values[-1] in dt:
                ix_hours = distance_traveled.size - 1
            else:
                raise e

        # Shave off the extra timing to get the exact travel time
        total_hours = ix_hours + 1  # add 1 for 0-indexing
        traveled = distance_traveled_sum[ix_hours]
        if traveled > distance:
            difference = traveled - distance
            speed_at_hour = distance_traveled[ix_hours]
            reduction = difference / speed_at_hour
            total_hours -= reduction

        return total_hours

    def travel(
        self,
        start: str,
        end: str,
        set_current: str | None = None,
        hours: float | None = None,
        distance: float | None = None,
        **kwargs,
    ) -> Generator[Timeout | Process, None, None]:
        """The process for traveling between port and site, or two systems onsite.

        NOTE: This does not currently take the weather conditions into account.

        Parameters
        ----------
        start : str
            The starting location, one of "site", "port", or "system".
        end : str
            The starting location, one of "site", "port", or "system".
        set_current : str, optional
            Where to set ``current_system`` to be if traveling to site or a different
            system onsite, by default None.
        hours : float, optional
            The number hours required for traveling between ``start`` and ``end``.
            If provided, no internal travel time will be calculated.
        distance : float, optional
            The distance, in km, to be traveled. Only used if hours is provided

        Yields
        -------
        Generator[Timeout | Process, None, None]
            The timeout event for traveling.
        """
        validate_end_points(start, end)
        if hours is None:
            hours = 0.0
            distance = 0.0
            additional = f"traveling from {start} to {end}"

            if start == end == "system":
                additional = f"traveling from {self.current_system} to {set_current}"
                hours, distance = self._calculate_intra_site_time(
                    self.current_system, set_current
                )
            elif set((start, end)) == set(("site", "port")):
                additional = f"traveling from {start} to {end}"
                distance = self.settings.port_distance
                hours = self._calculate_interrupted_travel_time(distance)
        else:
            if distance is None:
                raise ValueError("`distance` must be provided if `hours` is provided.")

        # MyPy helper
        assert isinstance(hours, float)
        assert isinstance(distance, float)

        # If the the equipment will arive after the shift is over, then it must travel
        # back to port (if needed), and wait for the next shift
        future_time = self.env.simulation_time + timedelta(hours=hours)
        is_shift = self._is_workshift(future_time.hour)
        if not is_shift and end != "port" and not self.at_port:
            kw: dict[str, str] = {
                "additional": "insufficient time to complete travel before end of the shift"
            }
            kw.update(kwargs)
            yield self.env.process(self.travel(start=start, end="port", **kw))  # type: ignore
            yield self.env.process(self.wait_until_next_shift(**kwargs))
            yield self.env.process(
                self.travel(start=start, end=end, set_current=set_current, **kwargs)
            )
            return
        elif not is_shift and self.at_port:
            kw = {
                "additional": "insufficient time to complete travel before end of the shift"
            }
            kw.update(kwargs)
            yield self.env.process(self.wait_until_next_shift(**kw))
            yield self.env.process(
                self.travel(start=start, end=end, set_current=set_current, **kwargs)
            )
            return

        salary_cost = self.calculate_salary_cost(hours)
        hourly_cost = 0  # contractors not paid for traveling
        equipment_cost = self.calculate_equipment_cost(hours)
        self.env.log_action(
            action="traveling",
            additional=additional,
            duration=hours,
            distance_km=distance,
            salary_labor_cost=salary_cost,
            hourly_labor_cost=hourly_cost,
            equipment_cost=equipment_cost,
            location="enroute",
            **kwargs,
        )
        yield self.env.timeout(hours)

        self._set_location(end, set_current)
        self.env.log_action(
            action="complete travel",
            additional=f"arrived at {set_current if set_current is not None else end}",
            location=end,
            **kwargs,
        )

    def tow(
        self,
        start: str,
        end: str,
        set_current: str | None = None,
        **kwargs,
    ) -> Generator[Timeout | Process, None, None]:
        """The process for towing a turbine to/from port.

        Parameters
        ----------
        start : str
            The starting location; one of "site" or "port".
        end : str
            The ending location; one of "site" or "port".
        set_current : str | None, optional
            The ``System.id`` if the turbine is being returned to site, by default None

        Yields
        ------
        Generator[Timeout | Process, None, None]
            The series of SimPy events that will be processed for the actions to occur.
        """
        validate_end_points(start, end, no_intrasite=True)
        # Get the distance that needs to be traveled, then calculate the delay and time
        # traveling, and log each of them
        distance = self.settings.port_distance
        delay, hours = self._calculate_uninterrupted_travel_time(distance, tow=True)
        if delay == -1:
            if start == "site":
                kw = deepcopy(kwargs)
                kw.update({"reason": "Insufficient weather window, return to port"})
                yield self.env.process(self.travel("site", "port", **kw))
                yield self.env.timeout(HOURS_IN_DAY * 4)
                yield self.env.process(
                    self.tow(start, end, set_current=set_current, **kwargs)
                )
            elif start == "port":
                kw = deepcopy(kwargs)
                kw.update(
                    {"reason": "Insufficient weather window, will try again later"}
                )
                yield self.env.timeout(HOURS_IN_DAY * 4)
                yield self.env.process(
                    self.tow(start, end, set_current=set_current, **kwargs)
                )

        else:
            self.env.process(self.weather_delay(delay, location=end, **kwargs))

        salary_labor_cost = self.calculate_salary_cost(hours)
        hourly_labor_cost = self.calculate_hourly_cost(hours)
        equipment_cost = self.calculate_equipment_cost(hours)
        self.env.log_action(
            duration=hours,
            distance_km=distance,
            action="towing",
            salary_labor_cost=salary_labor_cost,
            hourly_labor_cost=hourly_labor_cost,
            equipment_cost=equipment_cost,
            location="enroute",
            **kwargs,
        )

        yield self.env.timeout(hours)
        self._set_location(end, set_current)
        self.env.log_action(
            action="complete towing",
            salary_labor_cost=salary_labor_cost,
            hourly_labor_cost=hourly_labor_cost,
            equipment_cost=equipment_cost,
            additional="complete",
            location=end,
            **kwargs,
        )

    def crew_transfer(
        self,
        system: System,
        subassembly: Subassembly,
        request: RepairRequest,
        to_system: bool,
    ) -> None | Generator[Timeout | Process, None, None]:
        """The process of transfering the crew from the equipment to the ``System``
        for servicing using an uninterrupted weather window to ensure safe transfer.

        Parameters
        ----------
        system : System
            The System where the crew needs to be transferred to.
        subassembly : Subassembly
            The Subassembly that is being worked on.
        request : RepairRequest
            The repair to be processed.
        to_system : bool
            True if the crew is being transferred to the system, or False if the crew
            is being transferred off the system.

        Returns
        -------
        None
            None is returned when this is run recursively to not repeat the crew
            transfer process.

        Yields
        ------
        Generator[Timeout | Process, None, None]
            Yields a timeout event for the crew transfer once an uninterrupted weather
            window can be found.
        """
        hours_to_process = self.settings.crew_transfer_time
        salary_cost = self.calculate_salary_cost(hours_to_process)
        hourly_cost = self.calculate_hourly_cost(hours_to_process)
        equipment_cost = self.calculate_equipment_cost(hours_to_process)

        shared_logging = dict(
            system_id=system.id,
            system_name=system.name,
            part_id=subassembly.id,
            part_name=subassembly.name,
            system_ol=system.operating_level,
            part_ol=subassembly.operating_level,
            agent=self.settings.name,
            reason=request.details.description,
            request_id=request.request_id,
        )

        delay, shift_delay = self.find_uninterrupted_weather_window(hours_to_process)
        # If there is a shift delay, then travel to port, wait, and travel back, and finally try again.
        if shift_delay:
            travel_time = self.env.now
            yield self.env.process(
                self.travel(start="site", end="port", **shared_logging)
            )
            travel_time -= self.env.now  # will be negative value, but is flipped
            delay -= abs(travel_time)  # decrement the delay by the travel time
            yield self.env.process(
                self.weather_delay(delay, location="port", **shared_logging)
            )
            yield self.env.process(
                self.wait_until_next_shift(
                    **shared_logging,
                    **{"additional": "weather unsuitable to transfer crew"},
                )
            )
            yield self.env.process(
                self.travel(
                    start="port", end="site", set_current=system.id, **shared_logging
                )
            )
            yield self.env.process(
                self.crew_transfer(system, subassembly, request, to_system=to_system)  # type: ignore
            )
            return

        yield self.env.process(
            self.weather_delay(
                delay, location="site" if to_system else "system", **shared_logging
            )
        )

        if to_system:
            additional = f"transferring crew from {self.settings.name} to {system.id}"
        else:
            additional = f"transferring crew from {system.id} to {self.settings.name}"
        self.env.log_action(
            action="transferring crew",
            additional=additional,
            duration=hours_to_process,
            salary_labor_cost=salary_cost,
            hourly_labor_cost=hourly_cost,
            equipment_cost=equipment_cost,
            location="system",
            **shared_logging,
        )
        self.transferring_crew = True
        yield self.env.timeout(hours_to_process)
        self.transferring_crew = False
        if to_system:
            self.current_system = system.id
        else:
            self.current_system = None
            self.at_system = False
        self.env.log_action(
            action="complete transfer",
            additional="complete",
            location="system" if to_system else "site",
            **shared_logging,
        )

    def mooring_connection(
        self,
        system: System,
        request: RepairRequest,
        which: str,
    ) -> None | Generator[Timeout | Process, None, None]:
        """The process of either umooring a floating turbine to prepare for towing it to
        port, or reconnecting it after its repairs have been completed.

        Parameters
        ----------
        system : System
            The System that needs unmooring/reconnecting.
        request : RepairRequest
            The repair to be processed.
        which : bool
            "unmoor" for unmooring the turbine, "reconnect" for reconnecting the turbine.

        Returns
        -------
        None
            None is returned when this is run recursively to not repeat the process.

        Yields
        ------
        Generator[Timeout | Process, None, None]
            Yields a timeout event for the unmooring/reconnection once an uninterrupted
            weather window can be found.
        """
        assert isinstance(self.settings, UnscheduledServiceEquipmentData)  # mypy check
        which = which.lower().strip()
        if which == "unmoor":
            hours_to_process = self.settings.unmoor_hours
        elif which == "reconnect":
            hours_to_process = self.settings.reconnection_hours
        else:
            raise ValueError(
                f"Only `unmoor` and `reconnect` are allowable inputs, not {which}"
            )

        salary_cost = self.calculate_salary_cost(hours_to_process)
        hourly_cost = self.calculate_hourly_cost(hours_to_process)
        equipment_cost = self.calculate_equipment_cost(hours_to_process)

        shared_logging = dict(
            system_id=system.id,
            system_name=system.name,
            system_ol=system.operating_level,
            agent=self.settings.name,
            reason=request.details.description,
            request_id=request.request_id,
        )

        which_text = "unmooring" if which == "unmoor" else "mooring reconnection"

        delay, shift_delay = self.find_uninterrupted_weather_window(hours_to_process)
        # If there is a shift delay, then wait try again.
        if shift_delay:
            yield self.env.process(
                self.weather_delay(delay, location="site", **shared_logging)
            )
            yield self.env.process(
                self.wait_until_next_shift(
                    **shared_logging,
                    **{
                        "location": "site",
                        "additional": f"weather unsuitable for {which_text}",
                    },
                )
            )
            yield self.env.process(
                self.mooring_connection(system, request, which=which)  # type: ignore
            )
            return

        # If no shift delay, then process any weather delays before dis/connection
        yield self.env.process(
            self.weather_delay(delay, location="site", **shared_logging)
        )

        if which == "unmoor":
            additional = f"Unmooring {system.id} to tow to port"
        else:
            additional = f"Reconnecting the mooring to {system.id}"
        self.env.log_action(
            action=which_text,
            additional=additional,
            duration=hours_to_process,
            salary_labor_cost=salary_cost,
            hourly_labor_cost=hourly_cost,
            equipment_cost=equipment_cost,
            location="site",
            **shared_logging,  # type: ignore
        )

        if which == "unmoor":
            yield self.env.timeout(self.settings.unmoor_hours)
        else:
            yield self.env.timeout(self.settings.reconnection_hours)

        self.env.log_action(
            action=f"complete {which_text}", additional="complete", **shared_logging  # type: ignore
        )

    def in_situ_repair(
        self,
        request: RepairRequest,
        time_processed: int | float = 0,
        prior_operation_level: float = -1.0,
    ) -> Generator[Timeout | Process, None, None]:
        """Processes the repair including any weather and shift delays.

        Parameters
        ----------
        request : RepairRequest
            The ``Maintenance`` or ``Failure`` receiving attention.
        time_processed : int | float, optional
            Time that has already been processed, by default 0.
        prior_operation_level : float, optional
            The operating level of the ``System`` just before the repair has begun, by
            default -1.0.

        Yields
        -------
        Generator[Timeout | Process, None, None]
            Timeouts for the repair process.
        """

        """
        NOTE: THE PROCESS
        1. Travel to site/turbine, if not there already
        2. Transfer Crew
        3. Number of hours required, or hours until the end of the shift
        4. Do repairs for the above
        5. When shift/repair is complete, transfer crew
        6. Travel to next turbine, back to port

        7. Repeat until the repair is complete
        8. Exit this function
        """
        shift_delay = False

        if request.cable:
            system = self.windfarm.cable(request.system_id)
            cable = request.subassembly_id.split("::")[1:]
            subassembly = self.windfarm.graph.edges[cable]["cable"]
        else:
            system = self.windfarm.system(request.system_id)  # type: ignore
            subassembly = getattr(system, request.subassembly_id)

        starting_operational_level = max(
            prior_operation_level, subassembly.operating_level
        )
        shared_logging = dict(
            system_id=system.id,
            system_name=system.name,
            part_id=subassembly.id,
            part_name=subassembly.name,
            system_ol=system.operating_level,
            part_ol=subassembly.operating_level,
            agent=self.settings.name,
            reason=request.details.description,
            request_id=request.request_id,
        )

        # Ensure there is enough time to transfer the crew back and forth with a buffer
        # of twice the travel time or travel back to port and try again the next shift
        start_shift = self.settings.workday_start
        end_shift = self.settings.workday_end
        current = self.env.simulation_time
        hours_required = request.details.time - time_processed
        hours_available = hours_until_future_hour(current, end_shift)

        if hours_available <= self.settings.crew_transfer_time * 4:
            yield self.env.process(
                self.travel(
                    start="site",
                    end="port",
                    **shared_logging,
                )
            )
            yield self.env.process(self.wait_until_next_shift(**shared_logging))

            yield self.env.process(
                self.in_situ_repair(
                    request, prior_operation_level=starting_operational_level
                )
            )
            return

        # Travel to site or the next system on site
        if not self.at_system and self.at_port:
            yield self.env.process(
                self.travel(
                    start="port", end="site", set_current=system.id, **shared_logging
                )
            )
            # First turn off the turbine, then proceed with the servicing so the
            # turbine is not registered as operating when the turbine is being worked on
            if system.servicing.triggered:
                self.manager.halt_requests_for_system(system)
            yield self.env.process(
                self.crew_transfer(system, subassembly, request, to_system=True)  # type: ignore
            )
        elif self.at_system is not None and not self.at_port:
            yield self.env.process(
                self.travel(
                    start="system",
                    end="system",
                    set_current=system.id,
                    **shared_logging,
                )
            )
            # First turn off the turbine, then proceed with the servicing so the
            # turbine is not registered as operating when the turbine is being worked on
            if system.servicing.triggered:
                self.manager.halt_requests_for_system(system)
            yield self.env.process(
                self.crew_transfer(system, subassembly, request, to_system=True)  # type: ignore
            )
        else:
            raise RuntimeError(f"{self.settings.name} is lost!")

        current = self.env.simulation_time

        # If the hours required is longer than the shift time, then reset the available
        # number of hours and the appropriate weather forecast, accounting for crew
        # transfer, otherwise get an adequate weather window, allowing for interruptions
        if not (start_shift == 0 and end_shift == 24):
            hours_available = hours_until_future_hour(current, end_shift)
            hours_available -= self.settings.crew_transfer_time
            *_, weather_forecast = self._weather_forecast(
                hours_available, which="repair"
            )
        else:
            _, weather_forecast, _ = self.find_interrupted_weather_window(
                hours_required
            )

        # Check that all times are within the windfarm's working hours and cut off any
        # time points that fall outside of the work shifts
        if hours_required > hours_available:
            shift_delay = True

        hours_processed = 0
        weather_delay_groups = consecutive_groups(np.where(~weather_forecast)[0])
        while weather_forecast.size > 0 and hours_available > 0:
            delays = np.where(~weather_forecast)[0]
            if delays.size > 0:
                hours_to_process = delays[0]
                delay = weather_delay_groups.pop(0).size
            else:
                hours_to_process = weather_forecast.size
                delay = 0
            weather_forecast = weather_forecast[hours_to_process + delay :]

            # If the delay is at the start, hours_to_process is 0, and a delay gets
            # processed, otherwise the crew works for the minimum of
            # ``hours_to_process`` or maximum time that can be worked until the shift's
            # end, and maxed out by the hours required for the actual repair.
            if hours_to_process > 0:
                if hours_available <= hours_to_process:
                    hours_to_process = hours_available
                else:
                    current = self.env.simulation_time
                    hours_to_process = hours_until_future_hour(
                        current, current.hour + int(hours_to_process)
                    )
                if hours_required < hours_to_process:
                    hours_to_process = hours_required
                # Ensure this gets the correct float hours to the start of the target
                # hour, unless the hours to process is between (0, 1]
                yield self.env.process(
                    self.process_repair(  # type: ignore
                        hours_to_process, request.details, **shared_logging
                    )
                )
                hours_processed += hours_to_process
                hours_available -= hours_to_process
                hours_required -= hours_to_process

            # If a delay is the first part of the process or a delay occurs after the
            # some work is performed, then that delay is processed here.
            if delay > 0 and hours_required > 0:
                current = self.env.simulation_time
                hours_to_process = hours_until_future_hour(
                    current, current.hour + delay
                )
                yield self.env.process(
                    self.weather_delay(
                        hours_to_process, location="system", **shared_logging
                    )
                )
                hours_available -= hours_to_process

        # Reached the end of the shift or the end of the repair, and need to unload crew
        # from the system
        yield self.env.process(
            self.crew_transfer(system, subassembly, request, to_system=False)  # type: ignore
        )
        if shift_delay:
            yield self.env.process(
                self.travel(start="site", end="port", **shared_logging)
            )
            yield self.env.process(self.wait_until_next_shift(**shared_logging))

            yield self.env.process(
                self.in_situ_repair(
                    request,
                    time_processed=hours_processed + time_processed,
                    prior_operation_level=starting_operational_level,
                )
            )
            return

        # Register the repair
        self.register_repair_with_subassembly(
            subassembly, request, starting_operational_level
        )
        action = "maintenance" if isinstance(request.details, Maintenance) else "repair"
        self.env.log_action(
            system_id=system.id,
            system_name=system.name,
            part_id=subassembly.id,
            part_name=subassembly.name,
            system_ol=system.operating_level,
            part_ol=subassembly.operating_level,
            agent=self.settings.name,
            action=f"{action} complete",
            reason=request.details.description,
            materials_cost=request.details.materials,
            additional="complete",
            request_id=request.request_id,
            location="site",
        )

        # If this is the end of the shift, ensure that we're traveling back to port
        if not self.env.is_workshift(start_shift, end_shift):
            yield self.env.process(
                self.travel(start="site", end="port", **shared_logging)
            )

    def run_scheduled_in_situ(self) -> Generator[Process, None, None]:
        """Runs the simulation of in situ repairs for scheduled servicing equipment
        that have the `onsite` designation or don't require mobilization.

        Yields
        -------
        Generator[Process, None, None]
            The simulation.
        """
        assert isinstance(self.settings, ScheduledServiceEquipmentData)  # mypy controls

        # If the starting operation date is the same as the simulations, set to be onsite
        if self.settings.operating_dates[0] == self.env.simulation_time.date():
            self.onsite = True

        while True:
            # Wait for a valid operational period to start
            if self.env.simulation_time.date() not in self.settings.operating_dates_set:  # type: ignore
                yield self.env.process(self.mobilize_scheduled())

            # Wait for next shift to start
            is_workshift = self.env.is_workshift(
                workday_start=self.settings.workday_start,
                workday_end=self.settings.workday_end,
            )
            if not is_workshift:
                yield self.env.process(
                    self.wait_until_next_shift(
                        **dict(
                            agent=self.settings.name,
                            reason="not in working hours",
                            additional="not in working hours",
                        )
                    )
                )

            request = self.get_next_request()
            if request is None:
                if not self.at_port:
                    yield self.env.process(
                        self.travel(
                            start="site",
                            end="port",
                            reason="no requests",
                            agent=self.settings.name,
                        )
                    )
                yield self.env.process(
                    self.wait_until_next_shift(
                        **dict(
                            agent=self.settings.name,
                            reason="no requests",
                            additional="no work requests, waiting until the next shift",
                        )
                    )
                )
            else:
                yield self.env.process(self.in_situ_repair(request.value))

    def run_unscheduled_in_situ(self) -> Generator[Process, None, None]:
        """Runs an in situ repair simulation for unscheduled servicing equipment, or
        those that have to be mobilized before performing repairs and maintenance.

        Yields
        -------
        Generator[Process, None, None]
            The simulation
        """
        self.dispatched = True
        assert isinstance(self.settings, UnscheduledServiceEquipmentData)  # mypy helper
        mobilization_days = self.settings.mobilization_days
        charter_end_env_time = self.settings.charter_days * HOURS_IN_DAY
        charter_end_env_time += mobilization_days * HOURS_IN_DAY
        charter_end_env_time += self.env.now  # type: ignore

        current = self.env.simulation_time
        charter_start = current + pd.Timedelta(mobilization_days, "D")
        charter_end = (
            current
            + pd.Timedelta(mobilization_days, "D")
            + pd.Timedelta(self.settings.charter_days, "D")
        )
        charter_period = set(pd.date_range(charter_start, charter_end).date)

        # If the charter period contains any non-operational date, then, try again
        # at the next available date after the end of the attempted charter period
        intersection = charter_period.intersection(
            self.settings.non_operational_dates_set
        )
        if intersection:
            hours_to_next = self.hours_to_next_operational_date(
                start_search_date=max(intersection),
                exclusion_days=mobilization_days,
            )
            self.env.log_action(
                agent=self.settings.name,
                action="delay",
                reason="non-operational period",
                additional="waiting for next operational period",
                duration=hours_to_next,
            )
            yield self.env.timeout(hours_to_next)  # type: ignore
            yield self.env.process(self.run_unscheduled_in_situ())
            return

        while True:
            if self.env.now >= charter_end_env_time:
                self.onsite = False
                self.at_port = False
                self.at_system = False
                self.dispatched = False
                self.current_system = None
                self.env.log_action(
                    agent=self.settings.name,
                    action="leaving site",
                    reason="charter period has ended",
                )
                break

            if not self.onsite:
                yield self.env.process(self.mobilize())

            # Wait for next shift to start
            is_workshift = self.env.is_workshift(
                workday_start=self.settings.workday_start,
                workday_end=self.settings.workday_end,
            )
            if not is_workshift:
                yield self.env.process(
                    self.wait_until_next_shift(
                        **dict(
                            agent=self.settings.name,
                            reason="not in working hours",
                            additional="not in working hours",
                        )
                    )
                )
            request = self.get_next_request()
            if request is None:
                yield self.env.process(
                    self.wait_until_next_shift(
                        **dict(
                            agent=self.settings.name,
                            reason="no requests",
                            additional="no work requests submitted by start of shift",
                        )
                    )
                )
            else:
                yield self.env.process(self.in_situ_repair(request.value))

    def run_tow_to_port(self, request: RepairRequest) -> Generator[Process, None, None]:
        """Runs the tow to port logic, so a turbine can be repaired at port.

        Parameters
        ----------
        request : RepairRequest
            The request the triggered the tow-to-port strategy.

        Yields
        ------
        Generator[Process, None, None]
            The series of events that simulate the complete towing logic.

        Raises
        ------
        ValueError
            Raised if the equipment is not currently at port
        """
        system = self.windfarm.system(request.system_id)

        shared_logging = dict(
            system_id=request.system_id,
            system_name=request.system_name,
            request_id=request.request_id,
            agent=self.settings.name,
            reason=request.details.description,
        )

        # Travel to the turbine
        yield self.env.process(
            self.travel("port", "site", set_current=system.id, **shared_logging)  # type: ignore
        )

        # Unmoor the turbine and tow it back to port
        yield self.env.process(self.mooring_connection(system, request, which="unmoor"))  # type: ignore
        yield self.env.process(self.tow("site", "port", **shared_logging))

    def run_tow_to_site(self, request: RepairRequest) -> Generator[Process, None, None]:
        """Runs the tow to site logic for after a turbine has had its repairs completed
        at port.

        Parameters
        ----------
        request : RepairRequest
            The request the triggered the tow-to-port strategy.

        Yields
        ------
        Generator[Process, None, None]
            The series of events that simulate the complete towing logic.

        Raises
        ------
        ValueError
            Raised if the equipment is not currently at port
        """
        system = self.windfarm.system(request.system_id)
        shared_logging = dict(
            agent=self.settings.name,
            request_id=request.request_id,
            system_id=request.system_id,
            system_name=request.system_name,
        )

        # Tow the turbine back to site and reconnect it to the mooring and electrical systems
        yield self.env.process(
            self.tow(
                "port",
                "site",
                set_current=system.id,
                reason="towing turbine back to site",
                **shared_logging,
            )
        )
        yield self.env.process(
            self.mooring_connection(system, request, which="reconnect")  # type: ignore
        )

        # Reset the turbine back to operating and return to port
        reset_system_operations(system)
        self.manager.enable_requests_for_system(system.id)
        yield self.env.process(
            self.travel(
                "site",
                "port",
                reason="traveling back to port after returning turbine",
                **shared_logging,  # type: ignore
            )
        )

    def run_unscheduled(self, request: RepairRequest):
        """Runs the appropriate repair logic for unscheduled servicing equipment, or
        those that can be immediately dispatched.

        Parameters
        ----------
        request : RepairRequest
            The request that trigged the repair logic.
        """
        if "TOW" in request.details.service_equipment:
            yield self.env.process(self.port.run_tow_to_port(request))
        elif "AHV" in request.details.service_equipment:
            yield self.env.process(self.port.run_unscheduled_in_situ(request))
        else:
            yield self.env.process(self.run_unscheduled_in_situ())
