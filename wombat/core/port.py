"""
Creates the `Port` class that provies the tow-to-port repair capabilities for offshore
floating wind farms. The `Port` will control a series of tugboats enabled through the
"TOW" capability that get automatically dispatched once a tow-to-port repair is
submitted and a tugboat is available (`ServiceEquipment.at_port`). The `Port` also
controls any mooring repairs through the "AHV" capability, which operates similarly to
the tow-to-port except that it will not be released until the repair is completed, and
operates on a strict shift scheduling basis.
"""

from __future__ import annotations

import logging
from typing import Generator
from pathlib import Path

import numpy as np
import simpy
import pandas as pd
from simpy.events import Process, Timeout
from simpy.resources.store import FilterStore

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
        functionality in the high-levl cost bucket: ``operations_management_administration``
        or the more granula cost bucket: ``marine_management``

    Parameters
    ----------
    env : WombatEnvironment
        The simulation environment instance.
    windfarm : Windfarm
        The simulation windfarm instance.
    repair_manager : RepairManager
        The simulation repair manager instance.
    config : dict | str | Path
        A path to a YAML object or dictionary encoding the port's configuration settings.
        This will be loaded into a ``PortConfig`` object during initialization.

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
    tugboat_manager : simpy.FilterStore
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
        self.requests_serviced: set[str] = set()

        self.manager._register_port(self)

        if not isinstance(config, dict):
            try:
                config = load_yaml(env.data_dir / "project/port", config)
            except FileNotFoundError:
                config = load_yaml(env.data_dir / "repair", config)  # type: ignore
                logging.warning(
                    "DeprecationWarning: In v0.7, all port configurations must be located in: '<library>/project/port/"
                )
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
        assert isinstance(self.settings, PortConfig)
        self.turbine_manager = simpy.Resource(env, self.settings.max_operations)
        self.crew_manager = simpy.Resource(env, self.settings.n_crews)
        self.tugboat_manager = simpy.FilterStore(env, len(self.settings.tugboats))

        tugboats = []
        for t in self.settings.tugboats:
            tugboat = ServiceEquipment(self.env, self.windfarm, repair_manager, t)
            tugboat._register_port(self)
            tugboats.append(tugboat)

        self.tugboat_manager.items = tugboats
        self.active_repairs: dict[str, dict[str, simpy.events.Event]] = {}

        # Create partial functions for the labor and equipment costs for clarity
        self.initialize_cost_calculators(which="port")

        # Run the annualized fee logger
        if self.settings.annual_fee > 0:
            self.env.process(self._log_annual_fee())

    def _log_annual_fee(self):
        """Logs the annual port lease fee on a monthly-basis"""
        assert isinstance(self.settings, PortConfig)
        monthly_fee = self.settings.annual_fee / 12.0
        ix_month_starts = self.env.weather.index.day == 1  # 1st of the month
        ix_month_starts &= self.env.weather.index.hour == 0  # at midnight
        ix_month_starts = np.where(ix_month_starts)[0]
        ix_month_starts = ix_month_starts[ix_month_starts > 0]

        # At time 0 log the first monthly fee
        self.env.log_action(
            agent=self.settings.name,
            action="monthly lease fee",
            reason="port lease",
            equipment_cost=monthly_fee,
        )

        for i, ix_month in enumerate(ix_month_starts):
            # Get the time to the start of the next month
            time_to_next = ix_month if i == 0 else ix_month - ix_month_starts[i - 1]

            # Log the fee at the start of each month at midnight
            yield self.env.timeout(time_to_next)
            self.env.log_action(
                agent=self.settings.name,
                action="monthly lease fee",
                reason="port lease",
                equipment_cost=monthly_fee,
            )

    def repair_single(
        self, request: RepairRequest
    ) -> Generator[Timeout | Process, None, None]:
        """Simulation logic to process a single repair request.

        Parameters
        ----------
        request : RepairRequest
            The submitted repair or maintenance request.
        """
        # Retrieve the actual request
        request = yield self.get(lambda r: r == request)  # type: ignore

        # Request a service crew
        crew_request = self.crew_manager.request()
        yield crew_request  # type: ignore

        # Once a crew is available, process the acutal repair
        # Get the shift parameters
        start_shift = self.settings.workday_start
        end_shift = self.settings.workday_end
        continuous_operations = start_shift == 0 and end_shift == 24

        # Set the default hours to process and remaining hours for the repair
        hours_to_process = hours_remaining = request.details.time

        # Create the shared logging among the processes
        system = self.windfarm.system(request.system_id)
        subassembly = getattr(system, request.subassembly_id)
        shared_logging = dict(
            agent=self.settings.name,
            reason=request.details.description,
            system_id=system.id,
            system_name=system.name,
            part_id=subassembly.id,
            part_name=subassembly.name,
            request_id=request.request_id,
        )

        # Continue repairing and waiting a shift until the remaining hours to complete
        # the repair is zero
        while hours_remaining > 0:
            current = self.env.simulation_time

            # Check if the workday is limited by shifts and adjust to stay within shift
            if not continuous_operations:
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
                self.process_repair(hours_to_process, request.details, **shared_logging)  # type: ignore
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

    def transfer_requests_from_manager(
        self, system_id: str
    ) -> None | list[RepairRequest]:
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
            self.active_repairs[system_id][request.request_id] = self.env.event()
        return requests

    def run_repairs(self, system_id: str) -> None:
        """Method that transfers the requests from the repair manager and initiates the
        repair sequence.

        Parameters
        ----------
        system_id : str
            The ``System.id`` that is has been towed to port.
        """
        self.active_repairs[system_id] = {}
        requests = self.transfer_requests_from_manager(system_id)
        if requests is not None:
            self.requests_serviced.update([r.request_id for r in requests])
            [self.env.process(self.repair_single(request)) for request in requests]  # type: ignore

    def run_tow_to_port(self, request: RepairRequest) -> Generator[Process, None, None]:
        """The method to initiate a tow-to-port repair sequence.

        The process follows the following following routine:

        1. Request a tugboat from the tugboat resource manager and wait
        2. Runs ``ServiceEquipment.tow_to_port``, which encapsulates the traveling to
            site, unmooring, and return tow with a turbine
        3. Transfers the the turbine's repair log to the port, and gets all available
           crews to work on repairs immediately
        4. Requests a tugboat to return the turbine to site
        5. Runs ``ServiceEquipment.tow_to_site()``, which encapsulates the tow back to
           site, reconnection, resetting the operating status, and returning back to port

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

        # If the system is already undergoing repairs from other servicing equipment,
        # then wait until it's done being serviced
        if request.system_id in self.manager.invalid_systems:
            yield self.windfarm.system(request.system_id).servicing  # type: ignore

        # Halt the turbine before going further to avoid issue with requests being
        # being submitted between now and when the tugboat gets to the turbine
        self.manager.halt_requests_for_system(self.windfarm.system(request.system_id))

        # Check that there is enough time to complete towing, connection, and repairs
        # before starting the process, otherwise, wait until the next operational period
        # TODO: use a more sophisticated guess on timing, other than 20 days
        current = self.env.simulation_time.date()
        check_range = set(
            pd.date_range(current, current + pd.Timedelta(days=20), freq="D").date
        )
        intersection = check_range.intersection(self.settings.non_operational_dates_set)  # type: ignore
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
            yield self.env.timeout(hours_to_next)  # type: ignore

        self.requests_serviced.update([request.request_id])

        # Wait for a spot to open up in the port queue
        turbine_request = self.turbine_manager.request()
        yield turbine_request  # type: ignore

        # Request a tugboat to retrieve the turbine
        tugboat = yield self.tugboat_manager.get(lambda x: x.at_port)  # type: ignore
        yield self.env.process(tugboat.run_tow_to_port(request))  # type: ignore

        # Make the tugboat available again
        self.tugboat_manager.put(tugboat)

        # Transfer the repairs to the port queue, which will initiate the repair process
        self.run_repairs(request.system_id)

        # Wait for the repairs to complete
        yield simpy.AllOf(self.env, self.active_repairs[request.system_id].values())  # type: ignore

        # Request a tugboat to tow the turbine back to site, and open up the turbine queue
        tugboat = yield self.tugboat_manager.get(lambda x: x.at_port)  # type: ignore
        self.turbine_manager.release(turbine_request)
        yield self.env.process(tugboat.run_tow_to_site(request))  # type: ignore

        # Make the tugboat available again
        self.tugboat_manager.put(tugboat)

    def run_unscheduled_in_situ(
        self, request: RepairRequest
    ) -> Generator[Process, None, None]:
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

        self.requests_serviced.update([request.request_id])

        # Request a tugboat to retrieve the tugboat
        tugboat = yield self.tugboat_manager.get(lambda x: x.at_port)  # type: ignore
        assert isinstance(tugboat, ServiceEquipment)  # mypy: helper
        request = yield self.manager.get(lambda x: x == request)
        yield self.env.process(tugboat.in_situ_repair(request))

        # If the tugboat finished mid-shift, the in-situ repair logic will keep it there,
        # so ensure it returns back to port once it's complete
        if not tugboat.at_port:
            yield self.env.process(
                tugboat.travel(
                    start="site",
                    end="port",
                    agent=tugboat.settings.name,
                    reason=f"{request.details.description} complete",
                    system_id=request.system_id,
                    system_name=request.system_name,
                    part_id=request.subassembly_id,
                    part_name=request.subassembly_name,
                    request_id=request.request_id,
                )
            )

        # Make the tugboat available again
        self.tugboat_manager.put(tugboat)
