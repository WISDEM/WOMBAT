"""Creates the RepairCrew class."""
# TODO: NEED A SPECIFIC STARTUP METHODs


import os  # type: ignore
from typing import Generator, List, Union  # type: ignore

import numpy as np  # type: ignore
from simpy.events import Process, Timeout  # type: ignore

from wombat.core import (
    Failure,
    Maintenance,
    RepairManager,
    RepairRequest,
    ServiceEquipmentData,
    WombatEnvironment,
)
from wombat.core.library import load_yaml
from wombat.utilities import hours_until_future_hour
from wombat.windfarm import Windfarm
from wombat.windfarm.system.cable import Cable
from wombat.windfarm.system.subassembly import Subassembly


HOURS_IN_DAY = 24


def consecutive_groups(data: np.ndarray, step_size: int = 1) -> List[np.ndarray]:
    """Generates the subgroups of an array where the difference between two sequential
    elements is equal to the `step_size`. The intent is to find the length of delays
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
        A list of arrays of the consecutive elements of `data`.
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
        The `Windfarm` object.
    repair_manager : RepairManager
        The `RepairManager` object.
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
        """Initializes the `ServiceEquipment` class.

        Parameters
        ----------
        env : WombatEnvironment
            The simulation environment.
        windfarm : Windfarm
            The `Windfarm` object.
        repair_manager : RepairManager
            The `RepairManager` object.
        equipment_data_file : str
            The equipment settings file name with extension.
        """
        self.env = env
        self.windfarm = windfarm
        self.manager = repair_manager

        data = load_yaml(
            os.path.join(env.data_dir, "repair", "transport"), equipment_data_file
        )
        if data["start_year"] < self.env.start_year:
            data["start_year"] = self.env.start_year
        if data["end_year"] > self.env.end_year:
            data["end_year"] = self.env.end_year
        self.settings = ServiceEquipmentData.from_dict(data)

        # Register servicing equipment with the repair manager if it is using an
        # unscheduled maintenance scenario, so it can be dispatched as needed
        if self.settings.strategy == "unscheduled":
            self.manager._register_equipment(self)

        # Only run the equipment if it is on a scheduled basis, otherwise wait
        # for it to be dispatched
        if self.settings.strategy == "scheduled":
            self.env.process(self.run_scheduled())

    def _check_working_hours(self) -> None:
        """Checks the working hours of the equipment and overrides a default (-1) to
        the `env` settings, otherwise hours remain the same.
        """
        start_is_valid = self.settings.workday_start == -1
        end_is_valid = self.settings.workday_end == -1

        start = (
            self.env.workday_start if start_is_valid else self.settings.workday_start
        )
        end = self.env.workday_end if end_is_valid else self.settings.workday_end

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
            * self.settings.day_rate
            * self.settings.n_day_rate
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
        return duration * self.settings.hourly_rate * self.settings.n_hourly_rate

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
        `operating_level` back to good as new for the specific repair. For fatal cable
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
        else:
            subassembly.operating_level += starting_operating_level
            subassembly.operating_level /= 1 - repair.details.operation_reduction

    def wait_until_next_shift(self, **kwargs) -> Generator[Timeout, None, None]:
        """Delays the crew until the start of the next shift.

        Yields
        -------
        Generator[Timeout, None, None]
            Delay until the start of the next shift.
        """
        delay = self.env.hours_to_next_shift(workday_start=self.settings.workday_start)
        salary_cost = self.calculate_salary_cost(delay)
        hourly_cost = self.calculate_hourly_cost(0)
        equpipment_cost = self.calculate_equipment_cost(delay)
        self.env.log_action(
            system_id=kwargs.get("system_id", ""),
            system_name=kwargs.get("system_name", ""),
            part_id=kwargs.get("part_id", ""),
            part_name=kwargs.get("part_name", ""),
            system_ol=kwargs.get("system_ol", ""),
            part_ol=kwargs.get("part_ol", ""),
            agent=kwargs.get("agent", ""),
            action=kwargs.get("action", "delay"),
            reason=kwargs.get("reason", ""),
            additional=kwargs.get(
                "additional", "work shift has ended; waiting for next shift to start"
            ),
            duration=delay,
            request_id=kwargs.get("request_id", "na"),
            salary_labor_cost=salary_cost,
            hourly_labor_cost=hourly_cost,
            equipment_cost=equpipment_cost,
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
        ix_match = np.where(current < self.settings.operating_dates)[0]
        if ix_match.size > 0:
            still_in_operation = True
            next_operating_date = self.settings.operating_dates[ix_match[0]]
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
            system_id="",
            system_name="",
            part_id="",
            part_name="",
            system_ol="",
            part_ol="",
            agent=self.settings.name,
            action="delay",
            reason="work is complete",
            additional=additional,
            duration=hours_to_next_shift,
            salary_labor_cost=0,
            hourly_labor_cost=0,
            equipment_cost=0,
        )

        yield self.env.timeout(hours_to_next_shift)

    def mobilize_scheduled(self) -> Generator[Timeout, None, None]:
        """Mobilizes the ServiceEquipment object by waiting for the next operational
        period, less the mobilization time, then logging the mobiliztion cost.

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
        yield self.env.timeout(mobilization_hours)
        self.onsite = True
        self.env.log_action(
            system_id="",
            system_name="",
            part_id="",
            part_name="",
            system_ol="",
            part_ol="",
            agent=self.settings.name,
            action="mobilization",
            reason=f"{self.settings.name} is scheduled",
            additional="mobilization",
            duration=mobilization_hours,
            salary_labor_cost=0,
            hourly_labor_cost=0,
            equipment_cost=self.settings.mobilization_cost,
        )

    def mobilize(self) -> Generator[Timeout, None, None]:
        """Mobilizes the ServiceEquipment object.

        Yields
        -------
        Generator[Timeout, None, None]
            A Timeout event for the number of hours the ServiceEquipment requires for
            mobilizing to the windfarm site.
        """
        mobilization_hours = self.settings.mobilization_days * HOURS_IN_DAY
        yield self.env.timeout(mobilization_hours)
        self.onsite = True
        self.env.log_action(
            system_id="",
            system_name="",
            part_id="",
            part_name="",
            system_ol="",
            part_ol="",
            agent=self.settings.name,
            action="mobilization",
            reason=f"{self.settings.name} is scheduled",
            additional="mobilization",
            duration=mobilization_hours,
            salary_labor_cost=0,
            hourly_labor_cost=0,
            equipment_cost=self.settings.mobilization_cost,
        )

    def assess_weather(
        self, hours_required: Union[int, float], n_tries: int = 0
    ) -> np.ndarray:
        """Assesses at which points in the repair window, the wind (and wave)
        constraints for safe operation are met. If there is a delay in the time window,
        then the a new window with the added delay is processed to ensure that an array
        with enough safe operating times is available within the returned array.

        Parameters
        ----------
        hours_required : Union[int, float]
            The number of hours required to complete the repair.
        n_tries : int
            The number of previous attempts to find a suitable weather window.

        Returns
        -------
        np.ndarray
            A boolean array for what points in the time window are safe to complete a
            maintenance task or repair.
        """
        hours_required = np.ceil(hours_required).astype(int)
        wind, wave = self.env.weather_forecast(hours_required).T  # type: ignore
        satisfactory_wind = wind <= self.settings.max_windspeed_repair
        satisfactory_wave = wave <= self.settings.max_waveheight_repair
        all_clear = satisfactory_wind & satisfactory_wave

        total_clear_hours = np.count_nonzero(all_clear)
        delay = all_clear.size - total_clear_hours
        enough_windows = hours_required <= total_clear_hours
        shift_exceeded = all_clear.size > self.env.shift_length
        if enough_windows or shift_exceeded:
            return all_clear

        # TODO: Need a better solution to for the recursion error
        if n_tries < 10:
            return self.assess_weather(hours_required + delay, n_tries=n_tries + 1)
        else:
            return all_clear

    def get_next_request(self):
        """Gets the next request by the rig's method for processing repairs.

        Returns
        -------
        simpy.resources.store.FilterStoreGet
            The next `RepairRequest` to be processed.
        """
        if self.settings.method == "turbine":
            return self.manager.get_request_by_turbine(self.settings.capability)
        elif self.settings.method == "severity":
            return self.manager.get_next_highest_severity_request(
                self.settings.capability
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
            The `Maintenance` or `Failure` receiving attention.
        time_processed : Union[int, float], optional
            Time that has already been processed, by default 0.
        prior_operation_level : float, optional
            The operating level of the `System` just before the repair has begun, by
            default -1.0.

        Yields
        -------
        Generator[Union[Timeout, Process], None, None]
            Timeouts for the repair process.
        """
        shift_delay = False
        hours_required = request.details.time - time_processed
        system = self.windfarm.graph.nodes[request.system_id]["system"]
        if request.cable:
            cable = request.subassembly_id.split("::")[1:]
            subassembly = self.windfarm.graph.edges[cable]["cable"]
        else:
            subassembly = getattr(system, request.subassembly_id)

        staring_operational_level = max(
            prior_operation_level, subassembly.operating_level
        )

        # Obtain the weather forecast for the duration of the required time
        weather_forecast = self.assess_weather(hours_required + 1)

        # Check that all times are within the windfarm's working hours and cut off any
        # time points that fall outside of the work shifts
        current = self.env.simulation_time
        time_in_work_shift = np.arange(
            current.hour, current.hour + weather_forecast.size
        )
        working_hours = np.array(
            [
                self.env.hour_in_shift(
                    hour,
                    workday_start=self.settings.workday_start,
                    workday_end=self.settings.workday_end,
                )
                for hour in time_in_work_shift
            ]
        )
        if np.any(~working_hours):
            shift_delay = True
            hours_until_end = np.where(~working_hours)[0][0]
            weather_forecast = weather_forecast[:hours_until_end]

        # First turn off the turbine, then proceed with the repair or maintenance so the
        # turbine is not registered as operating when the turbine is being worked on
        subassembly.operating_level = 0
        subassembly.interrupt_all_subassembly_processes()

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
        hours_processed = 0
        delay_groups = consecutive_groups(np.where(~weather_forecast)[0])
        while weather_forecast.size > 0 and hours_required > 0:
            delays = np.where(~weather_forecast)[0]
            if delays.size > 0:
                hours_to_process = delays[0]
                delay = delay_groups.pop(0).size
            else:
                hours_to_process = weather_forecast.size
                delay = 0
            weather_forecast = weather_forecast[hours_to_process + delay :]

            if hours_to_process > 0:
                hours_to_process = int(min(hours_required, hours_to_process))
                current = self.env.simulation_time
                hours_to_process = hours_until_future_hour(
                    current, current.hour + hours_to_process
                )

                action = (
                    "repair" if isinstance(request.details, Failure) else "maintenance"
                )
                salary_cost = self.calculate_salary_cost(hours_to_process)
                hourly_cost = self.calculate_hourly_cost(hours_to_process)
                equipment_cost = self.calculate_equipment_cost(hours_to_process)
                self.env.log_action(
                    action=action,
                    duration=hours_to_process,
                    salary_labor_cost=salary_cost,
                    hourly_labor_cost=hourly_cost,
                    equipment_cost=equipment_cost,
                    additional=action,
                    **shared_logging,
                )
                yield self.env.timeout(hours_to_process)

                hours_processed += hours_to_process
                hours_required -= hours_to_process

            if delay > 0:
                current = self.env.simulation_time
                hours_to_process = hours_until_future_hour(
                    current, current.hour + delay
                )

                salary_cost = self.calculate_salary_cost(hours_to_process)
                hourly_cost = self.calculate_hourly_cost(0)
                equipment_cost = self.calculate_equipment_cost(hours_to_process)
                self.env.log_action(
                    action="delay",
                    additional="weather delay",
                    duration=hours_to_process,
                    salary_labor_cost=salary_cost,
                    hourly_labor_cost=hourly_cost,
                    equipment_cost=equipment_cost,
                    **shared_logging,
                )
                yield self.env.timeout(hours_to_process)

        if shift_delay:
            yield self.env.process(self.wait_until_next_shift(**shared_logging))

            yield self.env.process(
                self.process_repair(
                    request,
                    time_processed=hours_processed + time_processed,
                    prior_operation_level=staring_operational_level,
                )
            )
            return

        # Register the repair
        self.register_repair_with_subassembly(
            subassembly, request, staring_operational_level
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
            duration=0,
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
            if self.env.simulation_time.date() not in self.settings.operating_dates:

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

        charter_end_env_time = self.settings.charter_days * HOURS_IN_DAY
        charter_end_env_time += self.settings.mobilization_days * HOURS_IN_DAY
        charter_end_env_time += self.env.now
        while True:
            if self.env.now >= charter_end_env_time:
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
