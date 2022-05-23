"""Creates the Port class that provies the tow-to-port repair capabilities."""

from __future__ import annotations

from typing import Optional, Generator
from pathlib import Path

import numpy as np
import simpy
from simpy.events import Process
from simpy.resources.store import FilterStore, FilterStoreGet

from wombat.core.mixins import RepairsMixin
from wombat.core.library import load_yaml
from wombat.utilities.time import hours_until_future_hour
from wombat.core.environment import WombatEnvironment
from wombat.core.data_classes import PortConfig, Maintenance, RepairRequest
from wombat.core.repair_management import RepairManager
from wombat.core.service_equipment import ServiceEquipment


class Port(FilterStore, RepairsMixin):
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

        # Instantiate the crews, tugboats, and turbine availability
        assert isinstance(self.settings, PortConfig)
        self.turbine_manager = simpy.Resource(env, self.settings.max_operations)
        self.crew_manager = simpy.Resource(env, self.settings.n_crews)
        self.tugboat_manager = simpy.Resource(env, len(self.settings.tugboats))

        # Instantiate the crews and tugboats
        # TODO: put this outside of port or in a register method bc mananger isn't used
        # anywhere else
        self.availible_tugboats = [
            ServiceEquipment(self.env, self.env.windfarm, repair_manager, tug)  # type: ignore
            for tug in self.settings.tugboats
        ]
        for tugboat in self.availible_tugboats:
            tugboat._register_port(self)

        # Create partial functions for the labor and equipment costs for clarity
        self.initialize_cost_calculators(which="port")

    def transfer_requests_from_manager(self, system_id: str) -> None:
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
        _ = [self.env.process(self.repair_single(request)) for request in requests]  # type: ignore

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

    def repair_single(self, request: RepairRequest) -> None:
        """Simulation logic to process a single repair request.

        Parameters
        ----------
        request : RepairRequest
            The submitted repair or maintenance request.
        """
        # Retrieve the actual request
        request = yield self.get(lambda r: r == request)

        # Request a service crew
        crew_request = self.crew_manager.request()
        yield crew_request

        # Once a crew is available, process the acutal repair
        # Get the shift parameters
        start_shift = self.settings.workday_start
        end_shift = self.settings.workday_end
        continuous_operations = start_shift == 0 and end_shift == 24

        # Set the default hours to process and remaining hours for the repair
        hours_to_process = hours_remaining = request.details.time

        # Create the shared logging among the processes
        system = self.env.windfarm.system(request.system_id)
        subassembly = getattr(system, request.subassembly_id)
        shared_logging = dict(
            agent=self.settings.name,
            reason=request.details.description,
            system_id=system.id,
            system_name=system.name,
            subassembly_id=subassembly.id,
            subassembly_name=subassembly.name,
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

    def run_tow_to_port(self, request: RepairRequest) -> Generator[Process, None, None]:
        """The method to initiate a tow-to-port repair sequence.

        Process:
        - Request a tugboat from the tugboat resource manager and wait
        - Runs ``ServiceEquipment.tow_to_port()``, which encapsulates the traveling to
          site, unmooring, and return tow with a turbine
        - Transfers the the turbine's repair log to the port, and gets all available
          crews to work on repairs immediately
        - Requests a tugboat to return the turbine to site
        - Runs ``ServiceEquipment.tow_to_site()``, which encapsulates the tow back to
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
        # Wait for a spot to open up in the port queue
        turbine_request = self.turbine_manager.request()
        yield turbine_request

        # Request a tugboat to retrieve the tugboat
        tugboat_request = self.tugboat_manager.request()
        yield tugboat_request
        tugboat = self.availible_tugboats.pop(0)
        yield self.env.process(tugboat.run_tow_to_port(request))

        # Make the tugboat available again
        self.availible_tugboats.append(tugboat)
        self.tugboat_manager.release(tugboat_request)

        # Transfer the repairs to the port queue, which will initiate the repair process
        self.transfer_requests_from_manager(request.system_id)

        # Request a tugboat to tow the turbine back to site, and open up the turbine queue
        tugboat_request = self.tugboat_manager.request()
        yield tugboat_request
        tugboat = self.availible_tugboats.pop(0)
        self.turbine_manager.release(turbine_request)
        yield self.env.process(tugboat.run_tow_to_site(request))

        # Make the tugboat available again
        self.availible_tugboats.append(tugboat)
        self.tugboat_manager.release(tugboat_request)
