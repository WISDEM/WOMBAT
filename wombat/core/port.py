"""Creates the `Port` class that provies the tow-to-port repair capabilities for
offshore floating wind farms. The `Port` will control a series of tugboats enabled
through the "TOW" capability that get automatically dispatched once a tow-to-port repair
is submitted and a tugboat is available (`ServiceEquipment.at_port`). The `Port` also
controls any mooring repairs through the "AHV" capability, which operates similarly to
the tow-to-port except that it will not be released until the repair is completed, and
operates on a strict shift scheduling basis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
from collections.abc import Generator

import numpy as np
import simpy
import pandas as pd
import polars as pl
from simpy.events import Process, Timeout
from simpy.resources.store import FilterStore, FilterStoreGet

from wombat.windfarm import Windfarm
from wombat.core.mixins import RepairsMixin
from wombat.core.library import load_yaml
from wombat.utilities.time import hours_until_future_hour
from wombat.core.environment import WombatEnvironment
from wombat.core.data_classes import PortConfig, Maintenance, RepairRequest
from wombat.core.repair_management import RepairManager
from wombat.core.service_equipment import ServiceEquipment


class Port(RepairsMixin, FilterStore):
    """The offshore wind base port that operates tugboats and performs tow-to-port
    repairs.

    .. note:: The operating costs for the port are incorporated into the ``FixedCosts``
        functionality in the high-levl cost bucket:
        ``operations_management_administration`` or the more granula cost bucket:
        ``marine_management``

    Parameters
    ----------
    env : WombatEnvironment
        The simulation environment instance.
    windfarm : Windfarm
        The simulation windfarm instance.
    repair_manager : RepairManager
        The simulation repair manager instance.
    config : dict | str | Path
        A path to a YAML object or dictionary encoding the port's configuration
        settings. This will be loaded into a ``PortConfig`` object during
        initialization.

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
    requests_serviced : set[str]
        The set of requests that have already been serviced to ensure there are no
        duplications of labor when splitting out the repair requests to be processed.
    turbine_manager : simpy.Resource
        A SimPy ``Resource`` object that limits the number of turbines that can be towed
        to port, so as not to overload the quayside waters, which is controlled by
        ``settings.max_operations``.
    crew_manager : simpy.Resource
        A SimPy ``Resource`` object that limts the number of repairs that can be
        occurring at any given time, which is controlled by ``settings.n_crews``.
    service_equipment_manager : simpy.FilterStore
        A SimPy ``FilterStore`` object that acts as a coordination system for the
        registered tugboats to tow turbines between port and site. In order to tow
        in either direction they must be filtered by ``ServiceEquipment.at_port``. This
        is generated from the tugboat definitions in ``settings.tugboats``.
    active_repairs : dict[str, dict[str, simpy.events.Event]]
        A nested dictionary of turbines, and its associated request IDs with a SimPy
        ``Event``. The use of events allows them to automatically succeed at the end of
        repairs, and once all repairs are processed on a turbine, the tow-to-site
        process can commence.
    """

    def __init__(
        self,
        env: WombatEnvironment,
        windfarm: Windfarm,
        repair_manager: RepairManager,
        config: dict | str | Path,
    ) -> None:
        super().__init__(env, np.inf)

        self.env = env
        self.windfarm = windfarm
        self.manager = repair_manager
        self.system_request_map: dict[str, list[RepairRequest]] = {}
        self.requests_serviced: set[str] = set()
        self.invalid_systems: list[str] = []

        self.manager._register_port(self)

        if not isinstance(config, dict):
            config = load_yaml(env.data_dir / "project/port", config)
        if TYPE_CHECKING:
            assert isinstance(config, dict)
        self.settings = PortConfig.from_dict(config)

        self._check_working_hours(which="env")
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

        # Instantiate the crews, tugboats, and turbine availability
        if TYPE_CHECKING:
            assert isinstance(self.settings, PortConfig)
        self.turbine_manager = simpy.Resource(env, self.settings.max_operations)
        self.crew_manager = simpy.Resource(env, self.settings.n_crews)

        service_equipment = []
        for t in self.settings.tugboats:
            tugboat = ServiceEquipment(self.env, self.windfarm, repair_manager, t)
            tugboat._register_port(self)
            service_equipment.append(tugboat)

        self.service_equipment_manager = simpy.FilterStore(env, len(service_equipment))
        self.service_equipment_manager.items = service_equipment
        self.active_repairs: dict[str, dict[str, simpy.events.Event]] = {}
        self.subassembly_resets: dict[str, list[str]] = {}

        # Create partial functions for the labor and equipment costs for clarity
        self.initialize_cost_calculators(which="port")

        # Run the annualized fee logger
        if self.settings.annual_fee > 0:
            self.env.process(self._log_annual_fee())

    def _log_annual_fee(self):
        """Logs the annual port lease fee on a monthly-basis."""
        if TYPE_CHECKING:
            assert isinstance(self.settings, PortConfig)
        monthly_fee = self.settings.annual_fee / 12.0
        ix_month_starts = self.env.weather.filter(
            (pl.col("datetime").dt.day() == 1)
            & (pl.col("datetime").dt.hour() == 0)
            & (pl.col("index") > 0)
        ).select(pl.col("index"))

        # At time 0 log the first monthly fee
        self.env.log_action(
            agent=self.settings.name,
            action="monthly lease fee",
            reason="port lease",
            equipment_cost=monthly_fee,
        )
        for i, (ix_month,) in enumerate(ix_month_starts.rows()):
            # Get the time to the start of the next month
            time_to_next = (
                ix_month
                if i == 0
                else ix_month - ix_month_starts.slice(i - 1, 1).item()
            )

            # Log the fee at the start of each month at midnight
            yield self.env.timeout(time_to_next)
            self.env.log_action(
                agent=self.settings.name,
                action="monthly lease fee",
                reason="port lease",
                equipment_cost=monthly_fee,
            )

    def repair_single(self, request: RepairRequest) -> Generator[Timeout | Process]:
        """Simulation logic to process a single repair request.

        Parameters
        ----------
        request : RepairRequest
            The submitted repair or maintenance request.
        """
        # Request a service crew
        crew_request = self.crew_manager.request()
        yield crew_request

        # Once a crew is available, process the acutal repair
        end_shift = self.settings.workday_end

        # Set the default hours to process and remaining hours for the repair
        hours_to_process = hours_remaining = request.details.time

        # Create the shared logging among the processes
        system = self.windfarm.system(request.system_id)
        subassembly = getattr(system, request.subassembly_id)
        shared_logging = {
            "agent": self.settings.name,
            "reason": request.details.description,
            "system_id": system.id,
            "system_name": system.name,
            "part_id": subassembly.id,
            "part_name": subassembly.name,
            "request_id": request.request_id,
        }

        # Continue repairing and waiting a shift until the remaining hours to complete
        # the repair is zero
        while hours_remaining > 0:
            current = self.env.simulation_time

            # Check if the workday is limited by shifts and adjust to stay within shift
            if not self.settings.non_stop_shift:
                hours_to_process = hours_until_future_hour(current, end_shift)

            # Delay until the next shift if we're at the end
            if hours_to_process == 0:
                additional = "end of shift; will resume work in the next shift"
                yield self.env.process(
                    self.wait_until_next_shift(additional=additional, **shared_logging)
                )
                continue

            # Process the repair for the minimum of remaining hours to completion and
            # hours available in the shift
            hours_to_process = min(hours_to_process, hours_remaining)
            yield self.env.process(
                self.process_repair(hours_to_process, request.details, **shared_logging)
            )

            # Decrement the remaining hours and reset the default hours to process back
            # to the remaining repair time
            hours_remaining -= hours_to_process
            hours_to_process = hours_remaining

        # Log the completion of the repair
        action = "maintenance" if isinstance(request.details, Maintenance) else "repair"
        self.env.log_action(
            system_id=system.id,
            part_id=subassembly.id,
            part_name=subassembly.name,
            agent=self.settings.name,
            action=f"{action} complete",
            reason=request.details.description,
            materials_cost=request.details.materials,
            additional="complete",
            request_id=request.request_id,
        )

        # Make the crew available again
        self.crew_manager.release(crew_request)
        self.active_repairs[request.system_id][request.request_id].succeed()
        yield self.env.process(self.manager.register_repair(request))

    def transfer_requests_from_manager(
        self, system_id: str
    ) -> None | list[RepairRequest] | Generator:
        """Gets all of a given system's repair requests from the simulation's repair
        manager, removes them from that queue, and puts them in the port's queue.

        Parameters
        ----------
        system_id : str
            The ``System.id`` attribute from the system that will be repaired at port.

        Returns
        -------
        None | list[RepairRequest]
            The list of repair requests that need to be completed at port.
        """
        requests = self.manager.get_all_requests_for_system(
            self.settings.name, system_id
        )
        if requests is None:
            return requests
        requests = [r.value for r in requests]  # type: ignore

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
            _ = yield self.manager.in_process_requests.put(request)
            self.active_repairs[system_id][request.request_id] = self.env.event()
        request_ids = {el.request_id for el in requests}
        self.manager.request_status_map["pending"].difference_update(request_ids)
        self.manager.request_status_map["processing"].update(request_ids)
        return None

    def run_repairs(self, system_id: str) -> Generator | None:
        """Method that transfers the requests from the repair manager and initiates the
        repair sequence.

        Parameters
        ----------
        system_id : str
            The ``System.id`` that is has been towed to port.
        """
        self.active_repairs[system_id] = {}
        self.subassembly_resets[system_id] = []
        yield self.env.process(self.transfer_requests_from_manager(system_id))

        # Get all the requests and run them.
        # NOTE: this will all fail if there are somehow no requests, which also means
        # something else is completely wrong with the simulation
        request_list = self.get_all_requests_for_system(system_id)
        for request in request_list:  # type: ignore
            if TYPE_CHECKING:
                assert isinstance(request, FilterStoreGet)
            request = request.value
            self.requests_serviced.update([request.request_id])
            self.subassembly_resets[system_id].append(request.subassembly_id)
            yield self.env.process(self.repair_single(request))
        return None

    def get_all_requests_for_system(
        self, system_id: str
    ) -> None | Generator[FilterStoreGet]:
        """Gets all repair requests for a specific ``system_id``.

        Parameters
        ----------
        system_id : Optional[str], optional
            ID of the turbine or OSS; should correspond to ``System.id``.
            the first repair requested.

        Returns
        -------
        Optional[Generator[FilterStoreGet]]
            All repair requests for a given system. If no matching requests are found,
            or there aren't any items in the queue yet, then None is returned.
        """
        if not self.items:
            return None

        # Filter the requests by system
        requests = self.items
        if system_id is not None:
            requests = [el for el in self.items if el.system_id == system_id]
        if requests == []:
            return None

        # Loop the requests and pop them from the queue
        for request in requests:
            _ = yield self.get(lambda x: x is request)  # pylint: disable=W0640

        return requests

    def run_tow_to_port(self, request: RepairRequest) -> Generator[Process]:
        """The method to initiate a tow-to-port repair sequence.

        The process follows the following following routine:

        1. Request a tugboat from the tugboat resource manager and wait
        2. Runs ``ServiceEquipment.tow_to_port``, which encapsulates the traveling to
            site, unmooring, and return tow with a turbine
        3. Transfers the the turbine's repair log to the port, and gets all available
           crews to work on repairs immediately
        4. Requests a tugboat to return the turbine to site
        5. Runs ``ServiceEquipment.tow_to_site()``, which encapsulates the tow back to
           site, reconnection, resetting the operating status, and returning back to
           port

        Parameters
        ----------
        request : RepairRequest
            The request that initiated the process. This is primarily used for logging
            purposes.

        Yields
        ------
        Generator[Process, None, None]
            The series of events constituting the tow-to-port repairs
        """
        # If the request has already been addressed, return
        if request.request_id in self.requests_serviced:
            return

        system_id = request.system_id
        request_id = request.request_id

        # Double check in case a delay causes multiple vessels to be interacting with
        # the same turbine

        # Add the requested system to the list of systems undergoing or registered to be
        # undergoing repairs, so this method can't be run again on the same system
        self.invalid_systems.append(system_id)

        # If the system is already undergoing repairs from other servicing equipment,
        # then wait until it's done being serviced
        servicing = self.windfarm.system(system_id).servicing

        # Wait for a spot to open up in the port queue
        turbine_request = self.turbine_manager.request()

        yield turbine_request & servicing
        yield self.env.timeout(self.env.get_random_seconds())
        yield self.windfarm.system(system_id).servicing

        # Request a tugboat to retrieve the turbine
        tugboat = yield self.service_equipment_manager.get(
            lambda x: x.at_port
            and (not x.dispatched)
            and "TOW" in x.settings.capability
        )

        # Check that there is enough time to complete towing, connection, and repairs
        # before starting the process, otherwise, wait until the next operational period
        # TODO: use a more sophisticated guess on timing, other than 20 days
        current = self.env.simulation_time.date()
        check_range = set(
            pd.date_range(current, current + pd.Timedelta(days=20), freq="D").date
        )
        intersection = check_range.intersection(self.settings.non_operational_dates_set)
        if intersection:
            hours_to_next = self.hours_to_next_operational_date(
                start_search_date=max(intersection)
            )
            self.env.log_action(
                agent=self.settings.name,
                action="delay",
                reason="non-operational period",
                additional="waiting for next operational period",
                duration=hours_to_next,
            )
            yield self.env.timeout(hours_to_next)

        self.requests_serviced.update([request_id])

        if TYPE_CHECKING:
            assert isinstance(tugboat, ServiceEquipment)
        yield self.env.process(tugboat.run_tow_to_port(request))

        # Make the tugboat available again
        yield self.service_equipment_manager.put(tugboat)

        # Transfer the repairs to the port queue, which will initiate the repair process
        yield self.env.process(self.run_repairs(system_id))

        # Wait for the repairs to complete
        yield simpy.AllOf(self.env, self.active_repairs[system_id].values())

        # Request a tugboat to tow the turbine back to site, and open the turbine queue
        tugboat = yield self.service_equipment_manager.get(
            lambda x: x.at_port
            and (not x.dispatched)
            and "TOW" in x.settings.capability
        )
        self.turbine_manager.release(turbine_request)
        self.subassembly_resets[system_id] = list(
            set(self.subassembly_resets[system_id])
        )
        yield self.env.process(
            tugboat.run_tow_to_site(request, self.subassembly_resets[system_id])
        )
        self.invalid_systems.pop(self.invalid_systems.index(system_id))

        # Make the tugboat available again
        yield self.service_equipment_manager.put(tugboat)

    def run_unscheduled_in_situ(
        self, request: RepairRequest, initial: bool = False
    ) -> Generator[Process]:
        """Runs the in-situ repair processes for port-based servicing equipment such as
        tugboats that will always return back to port, but are not necessarily a feature
        of the windfarm itself, such as a crew transfer vessel.

        Parameters
        ----------
        request : RepairRequest
            The request that triggered the non tow-to-port, but port-based servicing
            equipment repair.

        Yields
        ------
        Generator[Process, None, None]
            The travel and repair processes.
        """
        # If the request has already been addressed, return
        if request.request_id in self.requests_serviced:
            return

        system = self.windfarm.system(request.system_id)

        if initial:
            _ = self.manager.get(lambda x: x is request)
            self.manager.in_process_requests.put(request)
            self.manager.request_status_map["pending"].difference_update(
                [request.request_id]
            )
            self.manager.request_status_map["processing"].update([request.request_id])

        # If the system is already undergoing repairs from other servicing equipment,
        # then wait until it's done being serviced, then double check
        yield system.servicing
        seconds_to_wait, *_ = (
            self.env.random_generator.integers(low=0, high=30, size=1) / 3600.0
        )
        yield self.env.timeout(seconds_to_wait)
        yield system.servicing

        # Halt the turbine before going further to avoid issue with requests being
        # being submitted between now and when the tugboat gets to the turbine
        self.requests_serviced.update([request.request_id])

        # Request a vessel that isn't solely a towing vessel
        vessel = yield self.service_equipment_manager.get(
            lambda x: x.at_port
            and (not x.dispatched)
            and x.settings.capability != ["TOW"]
        )
        if TYPE_CHECKING:
            assert isinstance(vessel, ServiceEquipment)
        request = yield self.manager.get(lambda x: x is request)
        yield self.env.process(vessel.in_situ_repair(request, initial=True))

        # If the tugboat finished mid-shift, the in-situ repair logic will keep it
        # there, so ensure it returns back to port once it's complete
        if not vessel.at_port:
            yield self.env.process(
                vessel.travel(
                    start="site",
                    end="port",
                    agent=vessel.settings.name,
                    reason=f"{request.details.description} complete",
                    system_id=request.system_id,
                    system_name=request.system_name,
                    part_id=request.subassembly_id,
                    part_name=request.subassembly_name,
                    request_id=request.request_id,
                )
            )

        # Make the tugboat available again
        yield self.service_equipment_manager.put(vessel)
