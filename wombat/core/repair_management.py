"""Creates the necessary repair classes."""


from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence
from itertools import chain
from collections import Counter

import numpy as np
from simpy.resources.store import FilterStore, FilterStoreGet

from wombat.core import (
    Failure,
    Maintenance,
    StrategyMap,
    RepairRequest,
    WombatEnvironment,
)


if TYPE_CHECKING:
    from wombat.core import Port, ServiceEquipment
    from wombat.windfarm import Windfarm
    from wombat.windfarm.system import Cable, System


class RepairManager(FilterStore):
    """Provides a class to manage repair and maintenance tasks.

    Parameters
    ----------
    FilterStore : simpy.resources.store.FilterStore
        The ``simpy`` class on which RepairManager is based to manage the repair and
        maintenance tasks.
    env : wombat.core.WombatEnvironment
        The simulation environment.
    capacity : float
        The maximum number of tasks that can be submitted to the manager, by default ``np.inf``.

    Attributes
    ----------
    env : wombat.core.WombatEnvironment
        The simulation environment.
    windfarm: wombat.windfarm.Windfarm
        The simulated windfarm. This is only used for getting the operational capacity.
    _current_id : int
        The logged and auto-incrememented integer base for the ID generated for each
        submitted repair request.
    downtime_based_equipment: StrategyMap
        The mapping between downtime-based servicing equipment and their capabilities.
    request_based_equipment: StrategyMap
        The mapping between request-based servicing equipment and their capabilities.
    """

    def __init__(self, env: WombatEnvironment, capacity: float = np.inf) -> None:
        super().__init__(env, capacity)

        self.env = env
        self._current_id = 0
        self.invalid_systems: list[str] = []

        self.downtime_based_equipment = StrategyMap()
        self.request_based_equipment = StrategyMap()

    def _update_equipment_map(self, service_equipment: ServiceEquipment) -> None:
        """Updates ``equipment_map`` with a provided servicing equipment object."""
        capability = service_equipment.settings.capability
        strategy = service_equipment.settings.strategy

        if strategy == "downtime":
            mapping = self.downtime_based_equipment
        elif strategy == "requests":
            mapping = self.request_based_equipment
        else:
            # Shouldn't be possible to get here!
            raise ValueError("Invalid servicing equipment!")

        strategy_threshold = service_equipment.settings.strategy_threshold  # type: ignore
        if isinstance(capability, list):
            for c in capability:
                mapping.update(c, strategy_threshold, service_equipment)

    def _register_windfarm(self, windfarm: Windfarm) -> None:
        """Adds the simulation windfarm to the class attributes"""
        self.windfarm = windfarm

    def _register_equipment(self, service_equipment: ServiceEquipment) -> None:
        """Adds the servicing equipment to the class attributes and adds it to the
        capabilities mapping.
        """
        self._update_equipment_map(service_equipment)

    def _register_port(self, port: "Port") -> None:
        """Registers the port with the repair manager, so that they can communicate as needed.

        Parameters
        ----------
        port : Port
            The port where repairs will occur.
        """
        self.port = port

    def _create_request_id(self, request: RepairRequest) -> str:
        """Creates a unique ``request_id`` to be logged in the ``request``.

        Parameters
        ----------
        request : RepairRequest
            The request object.

        Returns
        -------
        str
            An 11-digit identifier starting with "MNT" for maintenance tasks or "RPR"
            for repairs.

        Raises
        ------
        ValueError
            If the ``request.details`` property is not a ``Failure`` or ``Maintenance`` object,
            then a ValueError will be raised.
        """
        if isinstance(request.details, Failure):
            prefix = "RPR"
        elif isinstance(request.details, Maintenance):
            prefix = "MNT"
        else:
            # Note this is a safety, and shouldn't be able to be reached
            raise ValueError("A valid ``RepairRequest`` must be submitted")

        request_id = f"{prefix}{str(self._current_id).zfill(8)}"
        self._current_id += 1
        return request_id

    def _run_equipment_downtime(self, request: RepairRequest) -> None:
        """Run any equipment that has a pending request where the current windfarm
        operating capacity is less than or equal to the servicing equipment's threshold.
        """
        operating_capacity = self.windfarm.current_availability_wo_servicing
        for capability in self.request_map:
            equipment_mapping = getattr(self.downtime_based_equipment, capability)
            for equipment in equipment_mapping:
                if operating_capacity > equipment.strategy_threshold:
                    continue
                if capability in ("TOW", "AHV"):
                    # Don't dispatch a second piece of equipment for tow-to-port
                    if (
                        request.system_id in self.invalid_systems
                        and capability == "TOW"
                    ):
                        continue
                    try:
                        self.env.process(equipment.equipment.run_unscheduled(request))
                    except ValueError:
                        # ValueError is raised when a duplicate request is called for any of
                        # the port-based servicing equipment
                        pass
                if equipment.equipment.dispatched:
                    continue
                self.env.process(equipment.equipment.run_unscheduled(request))

    def _run_equipment_requests(self, request: RepairRequest) -> None:
        """Run the first piece of equipment (if none are onsite) for each equipment
        capability category where the number of requests is greater than or equal to the
        equipment's threshold.
        """
        for capability, n_requests in self.request_map.items():
            if capability not in request.details.service_equipment:
                continue
            equipment_mapping = getattr(self.request_based_equipment, capability)
            for i, equipment in enumerate(equipment_mapping):
                if n_requests < equipment.strategy_threshold:
                    continue
                # Run only the first piece of equipment in the mapping list, but ensure
                # that it moves to the back of the line after being used
                if capability in ("TOW", "AHV"):
                    # Don't dispatch a second piece of equipment for tow-to-port
                    if (
                        request.system_id in self.invalid_systems
                        and capability == "TOW"
                    ):
                        continue
                    try:
                        self.env.process(equipment.equipment.run_unscheduled(request))
                    except ValueError:
                        # ValueError is raised when a duplicate request is called for any of
                        # the port-based servicing equipment
                        pass
                    break

                if equipment.equipment.dispatched:
                    equipment_mapping.append(equipment_mapping.pop(i))
                    break

                self.env.process(equipment.equipment.run_unscheduled(request))
                equipment_mapping.append(equipment_mapping.pop(i))
                break

    def register_request(self, request: RepairRequest) -> RepairRequest:
        """The method to submit requests to the repair mananger and adds a unique
        identifier for logging.

        Parameters
        ----------
        request : RepairRequest
            The request that needs to be tracked and logged.

        Returns
        -------
        RepairRequest
            The same request as passed into the method, but with a unique identifier
            used for logging.
        """
        request_id = self._create_request_id(request)
        request.assign_id(request_id)
        self.put(request)
        return request

    def submit_request(self, request: RepairRequest) -> None:
        """The method to submit requests to the repair mananger and adds a unique
        identifier for logging.

        Parameters
        ----------
        request : RepairRequest
            The request that needs to be tracked and logged.

        Returns
        -------
        RepairRequest
            The same request as passed into the method, but with a unique identifier
            used for logging.
        """
        if self.downtime_based_equipment.is_running:
            self._run_equipment_downtime(request)
        if self.request_based_equipment.is_running:
            self._run_equipment_requests(request)

    def get_request_by_system(
        self, equipment_capability: Sequence[str], system_id: Optional[str] = None
    ) -> Optional[FilterStoreGet]:
        """Gets all repair requests for a certain turbine with given a sequence of
        ``equipment_capability`` as long as it isn't registered as unable to be serviced.

        Parameters
        ----------
        equipment_capability : list[str]
            The capability of the servicing equipment requesting repairs to process.
        system_id : Optional[str], optional
            ID of the turbine or OSS; should correspond to ``System.id``, by default
            None. If None, then it will simply sort the list by ``System.id`` and return
            the first repair requested.

        Returns
        -------
        Optional[FilterStoreGet]
            The first repair request for the focal system or None, if self.items is
            empty, or there is no matching system.
        """
        if not self.items:
            return None

        if system_id in self.invalid_systems:
            return None

        equipment_capability = set(equipment_capability)  # type: ignore

        # Filter the requests by system
        requests = self.items
        if system_id is not None:
            requests = [el for el in self.items if el.system_id == system_id]
        if requests == []:
            return None

        # Filter the requests by equipment capability and return the first valid request
        for request in requests:
            if equipment_capability.intersection(request.details.service_equipment):  # type: ignore
                # If this is the first request for the system, make sure no other servicing
                # equipment can access it
                if request.system_id not in self.invalid_systems:
                    self.invalid_systems.append(request.system_id)
                return self.get(lambda x: x == requests[0])

        # There were no matching equipment requirements to match the equipment
        # attempting to retrieve its next request
        return None

    def get_next_highest_severity_request(
        self,
        equipment_capability: list[str] | set[str],
        severity_level: Optional[int] = None,
    ) -> Optional[FilterStoreGet]:
        """Gets the next repair request by ``severity_level``.

        Parameters
        ----------
        equipment_capability : list[str]
            The capability of the equipment requesting possible repairs to make.
        severity_level : int
            Severity level of focus, default None.

        Returns
        -------
        Optional[FilterStoreGet]
            Repair request meeting the required criteria.
        """
        if not self.items:
            return None

        equipment_capability = set(equipment_capability)

        requests = self.items
        if severity_level is not None:
            # Capture only the desired severity level, if specified
            requests = [
                el for el in self.items if (el.severity_level == severity_level)
            ]
            if requests == []:
                return None

        # Re-order requests by severity (high to low) and the order they were submitted
        # Need to ensure that the requests are in submission order in case any get put
        # back
        requests = sorted(requests, key=lambda x: x.severity_level, reverse=True)
        for request in requests:
            if request.system_id not in self.invalid_systems:
                if equipment_capability.intersection(request.details.service_equipment):
                    self.invalid_systems.append(request.system_id)
                    return self.get(lambda x: x == request)

        # Ensure None is returned if nothing is found in the loop just as a FilterGet
        # would if allowed to oeprate without the above looping to identify multiple
        # criteria and acting on the request before it's processed
        return None

    def halt_requests_for_system(self, system: System | Cable) -> None:
        """Disables the ability for servicing equipment to service a specific system,
        sets the turbine status to be in servicing, and interrupts all the processes
        to turn off operations.

        Parameters
        ----------
        system_id : str
            The system to disable repairs.
        """
        if system.id not in self.invalid_systems:
            self.invalid_systems.append(system.id)
        system.servicing = self.env.event()
        system.interrupt_all_subassembly_processes()

    def enable_requests_for_system(self, system_id: str) -> None:
        """Reenables service equipment operations on the provided system.

        Parameters
        ----------
        system_id : str
            The ``System.id`` of the turbine that can be operated on again
        """
        _ = self.invalid_systems.pop(self.invalid_systems.index(system_id))

    def get_all_requests_for_system(
        self, agent: str, system_id: str
    ) -> Optional[list[RepairRequest]]:
        """Gets all repair requests for a specific ``system_id``.

        Parameters
        ----------
        agent : str
            The name of the entity requesting all of a system's repair requests.
        system_id : Optional[str], optional
            ID of the turbine or OSS; should correspond to ``System.id``.
            the first repair requested.

        Returns
        -------
        Optional[list[RepairRequest]]
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
            self.env.log_action(
                system_id=request.system_id,
                system_name=request.system_name,
                part_id=request.subassembly_id,
                part_name=request.subassembly_name,
                system_ol=float("nan"),
                part_ol=float("nan"),
                agent=agent,
                action="requests being moved",
                reason="",
                request_id=request.request_id,
            )
            _ = self.get(lambda x: x == request)  # pylint: disable=W0640

        return requests

    def purge_subassembly_requests(
        self, system_id: str, subassembly_id: str, exclude: list[str] = []
    ) -> Optional[list[RepairRequest]]:
        """Yields all the requests for a system/subassembly combination. This is intended
        to be used to remove erroneous requests after a subassembly has been replaced.

        Parameters
        ----------
        system_id : str
            Either the ``System.id`` or ``Cable.id``.
        subassembly_id : str
            Either the ``Subassembly.id`` or the ``Cable.id`` repeated for cables.
        exclude : list[str]
            A list of ``request_id`` to exclude from the purge. This is a specific use
            case for the combined cable system/subassembly, but can be to exclude
            certain requests from the purge.

        Yields
        -------
        Optional[list[RepairRequest]]
            All requests made to the repair manager for the provided system/subassembly
            combination. Returns None if self.items is empty or the loop terminates
            without finding what it is looking for.
        """
        if not self.items:
            return None

        requests = [
            request
            for request in self.items
            if (
                request.system_id == system_id
                and request.subassembly_id == subassembly_id
                and request.request_id not in exclude
            )
        ]
        if requests == []:
            return None

        for request in requests:
            self.env.log_action(
                system_id=request.system_id,
                system_name=request.system_name,
                part_id=request.subassembly_id,
                part_name=request.subassembly_name,
                system_ol=float("nan"),
                part_ol=float("nan"),
                agent="RepairManager",
                action="request canceled",
                reason="replacement required",
                request_id=request.request_id,
            )
            _ = self.get(lambda x: x == request)  # pylint: disable=W0640
        return requests

    @property
    def request_map(self) -> dict[str, int]:
        """Creates an updated mapping between the servicing equipment capabilities and
        the number of requests that fall into each capability category (nonzero values only).
        """
        requests = dict(
            Counter(
                chain.from_iterable((r.details.service_equipment for r in self.items))
            )
        )
        return requests
