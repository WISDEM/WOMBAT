"""Creates the ServiceEquipment class for simulating repairs."""
# TODO: NEED A SPECIFIC STARTUP METHODs


import os  # type: ignore
import numpy as np  # type: ignore
from math import ceil
from typing import List, Tuple, Union, Generator  # type: ignore
from datetime import timedelta
from simpy.events import Process, Timeout  # type: ignore
from pandas.core.indexes.datetimes import DatetimeIndex

from wombat.core import (
    Failure,
    Maintenance,
    RepairManager,
    RepairRequest,
    WombatEnvironment,
    ServiceEquipmentData,
)
from wombat.windfarm import Windfarm
from wombat.utilities import hours_until_future_hour
from wombat.core.library import load_yaml
from wombat.windfarm.system import System
from wombat.windfarm.system.cable import Cable
from wombat.windfarm.system.subassembly import Subassembly


try:
    from functools import cache  # type: ignore
except ImportError:
    from functools import lru_cache

    cache = lru_cache(None)


HOURS_IN_DAY = 24


def consecutive_groups(data: np.ndarray, step_size: int = 1) -> List[np.ndarray]:
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
    List[np.ndarray]
        A list of arrays of the consecutive elements of ``data``.
    """
    # Consecutive groups in delay
    groups = np.split(data, np.where(np.diff(data) != step_size)[0] + 1)
    return groups


class ServiceEquipment:
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
    """

    def __init__(
        self,
        env: WombatEnvironment,
        windfarm: Windfarm,
        repair_manager: RepairManager,
        equipment_data_file: str,
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
        self.enroute = False
        self.at_system = False
        self.at_port = False
        self.current_system = None  # type: Union[str, None]

        data = load_yaml(
            os.path.join(env.data_dir, "repair", "transport"), equipment_data_file
        )
        try:
            if data["start_year"] < self.env.start_year:
                data["start_year"] = self.env.start_year
            if data["end_year"] > self.env.end_year:
                data["end_year"] = self.env.end_year
        except KeyError:
            # Ignores for unscheduled maintenace equipment that would not have this input
            pass

        self.settings = ServiceEquipmentData(data).determine_type()
        self._check_working_hours()

        # Register servicing equipment with the repair manager if it is using an
        # unscheduled maintenance scenario, so it can be dispatched as needed
        if self.settings.strategy != "scheduled":
            self.manager._register_equipment(self)

        # Only run the equipment if it is on a scheduled basis, otherwise wait
        # for it to be dispatched
        if self.settings.strategy == "scheduled":
            self.env.process(self.run_scheduled())

    def _check_working_hours(self) -> None:
        """Checks the working hours of the equipment and overrides a default (-1) to
        the ``env`` settings, otherwise hours remain the same.
        """
        start_is_invalid = self.settings.workday_start == -1
        end_is_invalid = self.settings.workday_end == -1

        start = (
            self.env.workday_start if start_is_invalid else self.settings.workday_start
        )
        end = self.env.workday_end if end_is_invalid else self.settings.workday_end

        self.settings._set_environment_shift(start, end)

    def calculate_salary_cost(self, duration: Union[int, float]) -> float:
        """The total salaried labor cost implications for a given action.

        Parameters
        ----------
        duration : Union[int, float]
            Number of hours to charge for the salaried labor.

        Returns
        -------
        float
            The total cost of salaried workers for a given action.
        """
        return (
            (duration / HOURS_IN_DAY)
            * self.settings.crew.day_rate
            * self.settings.crew.n_day_rate
        )

    def calculate_hourly_cost(self, duration: Union[int, float]) -> float:
        """The total hourly labor cost implications for a given action.

        Parameters
        ----------
        duration : Union[int, float]
            Number of hours to charge for the hourly labor.

        Returns
        -------
        float
            The total cost of hourly workers for a given action.
        """
        return (
            duration * self.settings.crew.hourly_rate * self.settings.crew.n_hourly_rate
        )

    def calculate_equipment_cost(self, duration: Union[int, float]) -> float:
        """The total equipment cost implications for a given action.

        Parameters
        ----------
        duration : Union[int, float]
            Number of hours to charge for the equipment.

        Returns
        -------
        float
            The total cost of equipment for a given action.
        """
        return (duration / HOURS_IN_DAY) * self.settings.equipment_rate

    def register_repair_with_subassembly(
        self,
        subassembly: Union[Subassembly, Cable],
        repair: RepairRequest,
        starting_operating_level: float,
    ) -> None:
        """Goes into the repaired subassembly, component, or cable and returns its
        ``operating_level`` back to good as new for the specific repair. For fatal cable
        failures, all upstream turbines are turned back on unless there is another fatal
        cable failure preventing any more from operating.

        Parameters
        ----------
        subassembly : Union[Subassembly, Cable]
            The subassembly or cable that was repaired.
        repair : RepairRequest
            The request for repair that was submitted.
        starting_operating_level : float
            The operating level before a repair was started.
        """
        # If this is a cable repair and returning it back to operating, then reset all
        # upstream turbines to be operating as long as there isn't another fatal cable
        # failure preventing it from operating
        start: str = ""
        if repair.cable and repair.details.operation_reduction == 1:
            for i, turbine in enumerate(repair.upstream_turbines):
                if i > 0:
                    edge = self.windfarm.graph.edges[start, turbine]
                    if edge["cable"].operating_level == 0:
                        break
                self.windfarm.graph.nodes[turbine]["system"].cable_failure = False
                start = turbine

        # Put the subassembly/component back to good as new condition
        if repair.details.operation_reduction == 1:
            subassembly.operating_level = 1
            _ = self.manager.purge_subassembly_requests(
                repair.system_id, repair.subassembly_id
            )
        elif repair.details.operation_reduction == 0:
            subassembly.operating_level = starting_operating_level
        else:
            subassembly.operating_level /= 1 - repair.details.operation_reduction

    def wait_until_next_shift(self, **kwargs) -> Generator[Timeout, None, None]:
        """Delays the crew until the start of the next shift.

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
                + self.env.workday_start
                - less_mobilization_hours
            )
        else:
            still_in_operation = False
            hours_to_next_shift = self.env.max_run_time - self.env.now

        if still_in_operation:
            additional = "will return next year"
        else:
            additional = "no more return visits will be made"

        self.env.log_action(
            agent=self.settings.name,
            action="delay",
            reason="work is complete",
            additional=additional,
            duration=hours_to_next_shift,
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
        yield self.env.process(
            self.wait_until_next_operational_period(
                less_mobilization_hours=mobilization_hours
            )
        )
        self.env.log_action(
            agent=self.settings.name,
            action="mobilization",
            reason=f"{self.settings.name} is being mobilized",
            additional="mobilization",
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
        )

    def _weather_forecast(
        self, hours: Union[int, float]
    ) -> Tuple[DatetimeIndex, np.ndarray]:
        """Retrieves the weather forecast from the simulation environment, and
        translates it to a boolean for satisfactory (True) and unsatisfactory (False)
        weather conditions.

        Parameters
        ----------
        hours : Union[int, float]
            The whole number of hours that should be pulled

        Returns
        -------
        Tuple[DatetimeIndex, np.ndarray]
            The pandas DatetimeIndex and the boolean array of where the weather
            conditions are within safe operating limits for the servicing equipment
            (True) or not (False).
        """
        dt, wind, wave = self.env.weather_forecast(hours)
        all_clear = wind <= self.settings.max_windspeed_repair
        all_clear &= wave <= self.settings.max_waveheight_repair
        return dt, all_clear

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
        return (start <= datetime_ix.hour) & (datetime_ix.hour <= end)

    def find_uninterrupted_weather_window(
        self, hours_required: Union[int, float]
    ) -> Tuple[Union[int, float], bool]:
        """Finds the delay required before starting on a process that won't be able to
        be interrupted by a weather delay.

        TODO: WEATHER FORECAST NEEDS TO BE DONE FOR ``math.floor(self.now)``, not the ceiling
        or there will be a whole lot of rounding up errors on process times.

        Parameters
        ----------
        hours_required : Union[int, float]
            The number of uninterrupted of hours that a process requires for completion.

        Returns
        -------
        Tuple[Union[int, float], bool]
            The number of hours in weather delays before a process can be completed, and
            an indicator for if the process has to be delayed until the next shift for
            a safe transfer.
        """
        if not isinstance(hours_required, int):
            hours_required = ceil(hours_required)

        current = self.env.simulation_time

        # If the time required for a transfer is longer than the time left in the shift,
        # then return the hours left in the shift and indicate a shift delay
        max_hours = hours_until_future_hour(
            self.env.simulation_time, self.settings.workday_end
        )
        if hours_required > max_hours:
            return max_hours, True

        dt_ix, all_clear = self._weather_forecast(max_hours)
        all_clear &= self._is_workshift(dt_ix)
        safe_operating_windows = consecutive_groups(np.where(all_clear)[0])
        window_lengths = np.array([window.size for window in safe_operating_windows])

        # If all the safe operating windows are shorter than the time required, then
        # return the time until the end of the shift and indicate a shift delay
        if all([window < hours_required for window in window_lengths]):
            return max_hours, True

        # Return the number of hours until the first adequate operating window
        delay = safe_operating_windows[
            np.where(window_lengths >= hours_required)[0][0]
        ][0]
        delay = hours_until_future_hour(current, current.hour + delay)
        return delay, False

    def find_interrupted_weather_window(
        self, hours_required: Union[int, float]
    ) -> Tuple[DatetimeIndex, np.ndarray, bool]:
        """Assesses at which points in the repair window, the wind (and wave)
        constraints for safe operation are met.

        The initial search looks for a weather window of length ``hours_required``, and
        adds 24 hours to the window for the proceeding 9 days. If no satisfactory window
        is found, then the calling process must make another call to this function, but
        likely there is something wrong with the constraints or weather conditions if
        no window is found within 10 days.

        Parameters
        ----------
        hours_required : Union[int, float]
            The number of hours required to complete the repair.

        Returns
        -------
        Tuple[DatetimeIndex, np.ndarray, bool]
            The pandas DatetimeIndex, and a corresponding boolean array for what points
            in the time window are safe to complete a maintenance task or repair.
        """
        window_found = False
        hours_required = np.ceil(hours_required).astype(int)

        for i in range(10):  # no more than 10 attempts to find a window
            dt_ix, weather = self._weather_forecast(hours_required + (i * 24))
            working_hours = self._is_workshift(dt_ix)
            window = weather & working_hours
            if window.sum() >= hours_required:
                window_found = True
                break

        return dt_ix, weather, window_found

    def get_next_request(self):
        """Gets the next request by the rig's method for processing repairs.

        Returns
        -------
        simpy.resources.store.FilterStoreGet
            The next ``RepairRequest`` to be processed.
        """
        if self.settings.method == "turbine":
            return self.manager.get_request_by_turbine(self.settings.capability)
        elif self.settings.method == "severity":
            return self.manager.get_next_highest_severity_request(
                self.settings.capability
            )

    def weather_delay(
        self, hours: Union[int, float], **kwargs
    ) -> Union[None, Generator[Union[Timeout, Process], None, None]]:
        """Processes a weather delay of length ``hours`` hours.

        Parameters
        ----------
        hours : Union[int, float]
            The lenght, in hours, of the weather delay.

        Returns
        -------
        None
            If the delay is 0 hours, then nothing happens.

        Yields
        -------
        Union[None, Generator[Union[Timeout, Process], None, None]]
            If the delay is more than 0 hours, then a ``Timeout`` is yielded of length ``hours``.
        """
        if hours == 0:
            return

        salary_cost = self.calculate_salary_cost(hours)
        hourly_cost = self.calculate_hourly_cost(0)
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

    def repair(
        self,
        hours: Union[int, float],
        request_details: Union[Maintenance, Failure],
        **kwargs,
    ) -> Union[None, Generator[Union[Timeout, Process], None, None]]:
        """[summary]

        Parameters
        ----------
        hours : Union[int, float]
            The lenght, in hours, of the repair or maintenance task.
        request_details : Union[Maintenance, Failure]
            The deatils of the request, this is only being checked for the type.

        Yields
        -------
        Union[None, Generator[Union[Timeout, Process], None, None]]
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

    @cache
    def _calculate_intra_site_time(
        self, start: Union[str, None], end: Union[str, None]
    ) -> float:
        """Calculates the time it takes to travel between port and site or between
        systems on site.

        Parameters
        ----------
        start : Union[str, None]
            The starting onsite location. If ``None``, then 0 is returned.
        end : Union[str, None]
            The ending onsite location. If ``None``, then 0 is returned.

        Returns
        -------
        float
            The travel time between two locations.
        """
        distance = 0  # setting for invalid cases to have no traveling
        valid_sys = self.windfarm.distance_matrix.columns
        intra_site = start in valid_sys and end in valid_sys
        if intra_site:
            distance = self.windfarm.distance_matrix.loc[start, end]
        travel_time = distance / self.settings.speed

        # Infinity is the result of "traveling" between a system twice in a row
        if travel_time == float("inf"):
            travel_time = 0
        return travel_time

    def travel(
        self, start: str, end: str, set_current: Union[str, None] = None, **kwargs
    ) -> Generator[Union[Timeout, Process], None, None]:
        """The process for traveling between port and site, or two systems onsite.

        Parameters
        ----------
        start : str
            The starting location, one of "site", "port", or "system".
        end : str
            The starting location, one of "site", "port", or "system".
        set_current : str, optional
            Where to set ``current_system`` to be if traveling to site or a different
            system onsite, by default None.

        Yields
        -------
        Generator[Union[Timeout, Process], None, None]
            The timeout event for traveling.

        Raises
        ------
        ValueError
            Raised if ``start`` is not one of the required inputs.
        ValueError
            Raised if ``end`` is not one of the required inputs.
        """
        if start not in ("port", "site", "system"):
            raise ValueError(
                "``start`` location must be one of 'port', 'site', or 'system'!"
            )
        if end not in ("port", "site", "system"):
            raise ValueError(
                "``end`` location must be one of 'port', 'site', or 'system'!"
            )

        # TODO: Need a port-to-site distance to be implemented
        hours = 0  # takes care of invalid cases
        if start == end == "system":
            additional = f"traveling from {self.current_system} to {set_current}"
            hours = self._calculate_intra_site_time(self.current_system, set_current)
        else:
            additional = f"traveling from {start} to {end}"
            hours = 0

        # If the the equipment will arive after the shift is over, then it must travel
        # back to port (if needed), and wait for the next shift
        if not self._is_workshift(self.env.simulation_time + timedelta(hours=hours)):
            if not self.at_port:
                kw = {
                    "additional": "insufficient time to complete travel before end of the shift"
                }
                kw.update(kwargs)
                yield self.env.process(self.travel(start="site", end="port", **kw))
            yield self.env.process(self.wait_until_next_shift(**kwargs))
            yield self.env.process(
                self.travel(start=start, end=end, set_current=set_current, **kwargs)
            )

        salary_cost = self.calculate_salary_cost(hours)
        hourly_cost = self.calculate_hourly_cost(0)
        equipment_cost = self.calculate_equipment_cost(hours)
        self.env.log_action(
            action="traveling",
            additional=additional,
            duration=hours,
            salary_labor_cost=salary_cost,
            hourly_labor_cost=hourly_cost,
            equipment_cost=equipment_cost,
            **kwargs,
        )
        yield self.env.timeout(hours)
        self.at_port = True if end == "port" else False
        self.at_system = True if set_current is not None else False
        self.current_system = set_current
        self.env.log_action(
            action="complete travel",
            additional=f"arrived at {set_current if set_current is not None else end}",
            **kwargs,
        )

    def crew_transfer(
        self,
        system: System,
        subassembly: Subassembly,
        request: RepairRequest,
        to_system: bool,
    ) -> Union[None, Generator[Union[Timeout, Process], None, None]]:
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
        Generator[Union[Timeout, Process], None, None]
            Yields a timeout event for the crew transfer once an interrupted weather
            window can be found.
        """
        hours_to_process = self.settings.crew_transfer_time
        salary_cost = self.calculate_salary_cost(hours_to_process)
        hourly_cost = self.calculate_hourly_cost(0)
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
            yield self.env.process(self.weather_delay(delay, **shared_logging))
            yield self.env.process(
                self.travel(start="site", end="port", **shared_logging)
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
                self.crew_transfer(system, subassembly, request, to_system=to_system)
            )
            return

        self.weather_delay(delay, **shared_logging)

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
            **shared_logging,
        )
        yield self.env.timeout(hours_to_process)
        self.env.log_action(
            action="complete transfer", additional="complete", **shared_logging
        )

    def process_repair(
        self,
        request: RepairRequest,
        time_processed: Union[int, float] = 0,
        prior_operation_level: float = -1.0,
    ) -> Generator[Union[Timeout, Process], None, None]:
        """Processes the repair including any weather and shift delays.

        Parameters
        ----------
        request : RepairRequest
            The ``Maintenance`` or ``Failure`` receiving attention.
        time_processed : Union[int, float], optional
            Time that has already been processed, by default 0.
        prior_operation_level : float, optional
            The operating level of the ``System`` just before the repair has begun, by
            default -1.0.

        Yields
        -------
        Generator[Union[Timeout, Process], None, None]
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

        system = self.windfarm.graph.nodes[request.system_id]["system"]
        if request.cable:
            cable = request.subassembly_id.split("::")[1:]
            subassembly = self.windfarm.graph.edges[cable]["cable"]
        else:
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

        # Travel to site or the next system on site
        if not self.at_system and self.at_port:
            yield self.env.process(
                self.travel(
                    start="port", end="site", set_current=system.id, **shared_logging
                )
            )
            yield self.env.process(
                self.crew_transfer(system, subassembly, request, to_system=True)
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
            yield self.env.process(
                self.crew_transfer(system, subassembly, request, to_system=True)
            )
        else:
            raise RuntimeError(f"{self.settings.name} is lost!")

        hours_required = hours_available = request.details.time - time_processed
        start_shift = self.settings.workday_start
        end_shift = self.settings.workday_end
        current = self.env.simulation_time

        # If the hours required is longer than the shift time, then reset the available
        # number of hours and the appropriate weather forecast, otherwise get an
        # adequate weather window, allowing for interruptions
        if not (start_shift == 0 and end_shift == 24):
            hours_available = hours_until_future_hour(current, end_shift)
            _, weather_forecast = self._weather_forecast(hours_available)
        else:
            _, weather_forecast, _ = self.find_interrupted_weather_window(
                hours_available
            )

        # Check that all times are within the windfarm's working hours and cut off any
        # time points that fall outside of the work shifts
        if hours_required > hours_available:
            shift_delay = True

        # First turn off the turbine, then proceed with the repair or maintenance so the
        # turbine is not registered as operating when the turbine is being worked on
        subassembly.operating_level = 0
        subassembly.interrupt_all_subassembly_processes()

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
            # end.
            if hours_to_process > 0:
                hours_to_process = int(min(hours_available, hours_to_process))
                current = self.env.simulation_time
                hours_to_process = hours_until_future_hour(
                    current, current.hour + hours_to_process
                )

                yield self.env.process(
                    self.repair(hours_to_process, request.details, **shared_logging)
                )
                hours_processed += hours_to_process
                hours_available -= hours_to_process

            # If a delay is the first part of the process or a delay occurs after the
            # some work is performed, then that delay is processed here.
            if delay > 0:
                current = self.env.simulation_time
                hours_to_process = hours_until_future_hour(
                    current, current.hour + delay
                )
                yield self.env.process(
                    self.weather_delay(hours_to_process, **shared_logging)
                )
                hours_available -= hours_to_process

        # Reached the end of the shift or the end of the repair, and need to unload crew
        # from the system
        yield self.env.process(
            self.crew_transfer(system, subassembly, request, to_system=False)
        )

        if shift_delay:
            yield self.env.process(
                self.travel(start="site", end="port", **shared_logging)
            )
            yield self.env.process(self.wait_until_next_shift(**shared_logging))

            yield self.env.process(
                self.process_repair(
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
        )

    def run_scheduled(self) -> Generator[Process, None, None]:
        """Runs the simulation.

        Yields
        -------
        Generator[Process, None, None]
            The simulation.
        """
        while True:
            # Wait for a valid operational period to start
            if self.env.simulation_time.date() not in self.settings.operating_dates:  # type: ignore

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
                yield self.env.process(self.process_repair(request.value))

    def run_unscheduled(self) -> Generator[Process, None, None]:
        """Runs an unscheduled maintenance simulation

        Yields
        -------
        Generator[Process, None, None]
            The simulation
        """

        charter_end_env_time = self.settings.charter_days * HOURS_IN_DAY  # type: ignore
        charter_end_env_time += self.settings.mobilization_days * HOURS_IN_DAY
        charter_end_env_time += self.env.now
        while True:
            if self.env.now >= charter_end_env_time:
                self.onsite = False
                self.at_port = False
                self.at_system = False
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
                yield self.env.process(self.process_repair(request.value))
