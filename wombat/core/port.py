"""Creates the Port class that provies the tow-to-port repair capabilities."""

from __future__ import annotations

from typing import Optional, Generator
from pathlib import Path

import numpy as np
from simpy.events import Process, Timeout  # type: ignore
from simpy.resources.store import FilterStore, FilterStoreGet

from wombat.core.library import load_yaml
from wombat.core.environment import WombatEnvironment
from wombat.core.data_classes import PortConfig, RepairRequest
from wombat.utilities.utilities import check_working_hours, hours_until_future_hour
from wombat.core.repair_management import RepairManager
from wombat.core.service_equipment import ServiceEquipment


try:
    from functools import cache  # type: ignore
except ImportError:
    from functools import lru_cache

    cache = lru_cache(None)


HOURS_IN_DAY = 24


class Port(FilterStore):
    def __init__(
        self,
        env: WombatEnvironment,
        config: str | Path,
        repair_manager: RepairManager,
        capacity: float = np.inf,
    ) -> None:
        super().__init__(env, capacity)

        self.env = env
        self.manager = repair_manager

        settings = load_yaml(env.data_dir / "repair", config)
        self.settings = PortConfig.from_dict(settings)

        self._check_working_hours()

        # Instantiate the crews and tugboats
        self.availible_crews = self.settings.n_crews
        self.availible_tugboats = [
            ServiceEquipment(self.env, self.env.windfarm, repair_manager, tug)
            for tug in self.settings.tugboats
        ]

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

    def get_system_requests_from_manager(self, system_id: str) -> None:
        """Gets all of a given system's repair requests from the simulation's repair
        manager, removes them from that queue, and puts them in the port's queue.

        Parameters
        ----------
        system_id : str
            The ``System.id`` attribute from the system that will be repaired at port.
        """
        requests = self.manager.get_all_requests_for_system(
            self.settings.name, system_id
        )
        if requests is None:
            return requests

        self.items.extend(requests)
        for request in requests:
            self.env.log_action(
                system_id=request.system_id,
                system_name=request.system_name,
                part_id=request.subassembly_id,
                part_name=request.subassembly_name,
                system_ol=float("nan"),
                part_ol=float("nan"),
                agent=self.settings.name,
                action="requests moved to port",
                reason="at-port repair can now proceed",
                request_id=request.request_id,
            )

    def get_request_by_system(self, system_id: str) -> Optional[FilterStoreGet]:
        """Gets all repair requests for a certain turbine with given a sequence of
        ``equipment_capability``.

        Parameters
        ----------
        system_id : Optional[str], optional
            ID of the turbine or OSS; should correspond to ``System.id``.

        Returns
        -------
        Optional[FilterStoreGet]
            The first repair request for the focal system or None, if self.items is
            empty, or there is no matching system.
        """
        if not self.items:
            return None

        # Filter the requests by equipment capability and return the first valid request
        # or None, if no requests were found
        requests = self.get(lambda x: x.system_id == system_id)
        if requests == []:
            return None
        else:
            yield requests

    def tow_to_port(self, tugboat: ServiceEquipment, system_id: str) -> None:
        """Process for a tugboat to initiate the tow-to-port repair process."""

        """
        TODO
        1) Calculate travel time to site
        2) Travel to site
        3) Unmoor the turbine
        4) Calculate travel time to port
        5) Travel to port
        6) Line it up for repairs
        """

    def return_system(self, tugboat: ServiceEquipment, system_id: str) -> None:
        """Process returns the system to site, and turns it back on."""

        """
        TODO
        1) Tow turbine back to port
            a) get travel time with speed reductions
            b) travel
        2) Reconnect turbine
        3) Reset to fully operating
        4) Reset all failure/maintenance processes?

        """

        # Tow turbine back to site
        hours = tugboat._calculate_travel_time(self.settings.site_distance, towing=True)
        yield self.env.process(
            tugboat.travel(
                "port",
                "site",
                system_id,
                hours=hours,
                action=f"transporting {system_id} back to site",
                agent=tugboat.settings.name,
                reason="at-port repairs complete",
                additional="anything else about slow downs, tbd?",
            )
        )

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

    def repair_single(self, repair: RepairRequest) -> None:
        """
        TODO

        """
        # Check for enough time in shift
        # - if not enough, wait a shift
        # Perform operations for enough shifts/hours to complete the repair w/o weather
        # Complete the repair
        hours_processed = 0
        current = self.env.simulation_time
        start_shift = self.settings.workday_start
        end_shift = self.settings.workday_end
        hours_to_process = hours_required = repair.details.time

        while hours_processed < hours_required:
            # Check if the workday is limited by shifts and adjust to stay within shift
            if not (start_shift == 0 and end_shift == 24):
                hours_to_process = hours_until_future_hour(current, end_shift)

            # Delay until the next shift if we're at the end
            if hours_to_process == 0:
                yield self.env.process(self.w)
