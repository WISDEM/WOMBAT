"""Creates the necessary repair classes."""

from __future__ import annotations

from typing import TYPE_CHECKING
from itertools import chain
from collections import Counter
from collections.abc import Generator

import numpy as np
from simpy.resources.store import FilterStore, FilterStoreGet

from wombat.core import (
    Failure,
    Maintenance,
    StrategyMap,
    RepairRequest,
    WombatEnvironment,
    UnscheduledServiceEquipmentData,
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
        The maximum number of tasks that can be submitted to the manager, by default
        ``np.inf``.

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
        self.systems_in_tow: list[str] = []
        self.systems_waiting_for_tow: list[str] = []

        self.downtime_based_equipment = StrategyMap()
        self.request_based_equipment = StrategyMap()
        self.completed_requests = FilterStore(self.env)
        self.in_process_requests = FilterStore(self.env)
        self.request_status_map: dict[str, set] = {
            "pending": set(),
            "processing": set(),
            "completed": set(),
        }

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

        if TYPE_CHECKING:
            assert isinstance(
                service_equipment.settings, UnscheduledServiceEquipmentData
            )
        strategy_threshold = service_equipment.settings.strategy_threshold
        if isinstance(capability, list):
            for c in capability:
                mapping.update(c, strategy_threshold, service_equipment)

    def _register_windfarm(self, windfarm: Windfarm) -> None:
        """Adds the simulation windfarm to the class attributes."""
        self.windfarm = windfarm

    def _register_equipment(self, service_equipment: ServiceEquipment) -> None:
        """Adds the servicing equipment to the class attributes and adds it to the
        capabilities mapping.
        """
        self._update_equipment_map(service_equipment)

    def _register_port(self, port: Port) -> None:
        """Registers the port with the repair manager, so that they can communicate as
        needed.

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
            If the ``request.details`` property is not a ``Failure`` or ``Maintenance``
            object,
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

    def _is_request_processing(self, request: RepairRequest) -> bool:
        """Checks if a repair is being performed, or has already been completed.

        Parameters
        ----------
        request : RepairRequest
            The request that is about to be submitted to servicing equipment, but needs
            to be double-checked against ongoing processes.

        Returns
        -------
        bool
            True if the request is ongoing or completed, False, if it's ok to processed
            with the operation.
        """
        if self.items == []:
            return False

        rid = request.request_id
        processing = (lambda: rid in self.request_status_map["processing"])()
        completed = (lambda: rid in self.request_status_map["completed"])()
        return processing or completed

    def _run_equipment_downtime(self, request: RepairRequest) -> None | Generator:
        """Run any equipment that has a pending request where the current windfarm
        operating capacity is less than or equal to the servicing equipment's threshold.

        TODO: This methodology needs to better resolve dispatching every equipment
        relating to a request vs just the one(s) that are required. Basically, don't
        dispatch every available HLV, but just one plus one of every other capability
        category that has pending requests
        """
        # Add an initial check to help avoid simultaneous dispatching
        seconds_to_wait, *_ = (
            self.env.random_generator.integers(low=0, high=10, size=1) / 3600.0
        )
        yield self.env.timeout(seconds_to_wait)

        # Port-based servicing equipment should be handled by the port and does not
        # have an operating reduction threshold to meet at this time
        if "TOW" in request.details.service_equipment:
            if request.system_id not in self.port.invalid_systems:
                system = self.windfarm.system(request.system_id)
                yield system.servicing_queue & system.servicing
                yield self.env.timeout(self.env.get_random_seconds(high=1))
                if request.system_id in self.systems_in_tow:
                    return None
                self.invalidate_system(system, tow=True)
                yield self.env.process(self.port.run_tow_to_port(request))
            return None

        # Wait for the actual system or cable to be available
        if request.cable:
            yield self.windfarm.cable(request.system_id).servicing
        else:
            yield self.windfarm.system(request.system_id).servicing

        operating_capacity = self.windfarm.current_availability_wo_servicing
        for capability in self.request_map:
            equipment_mapping = self.downtime_based_equipment.get_mapping(capability)
            for i, equipment in enumerate(equipment_mapping):
                if operating_capacity > equipment.strategy_threshold:
                    continue

                # Avoid simultaneous dispatches by waiting a random number of seconds
                seconds_to_wait, *_ = (
                    self.env.random_generator.integers(low=0, high=30, size=1) / 3600.0
                )
                yield self.env.timeout(seconds_to_wait)

                equipment_obj = equipment.equipment
                if equipment_obj.dispatched:
                    continue

                # Equipment-based logic does not manage system availability, so
                # ensure it's available prior to dispatching, and double check in
                # case delays causing a timing collision
                if equipment_obj.port_based:
                    if request.system_id in self.port.invalid_systems:
                        break

                    yield self.windfarm.system(request.system_id).servicing
                    self.env.process(
                        self.port.run_unscheduled_in_situ(request, initial=True)
                    )
                else:
                    self.env.process(equipment_obj.run_unscheduled_in_situ())

                    # Move the dispatched capability to the end of list to ensure proper
                    # cycling of available servicing equipment
                    self.downtime_based_equipment.move_equipment_to_end(capability, i)
        return None

    def _run_equipment_requests(self, request: RepairRequest) -> None | Generator:
        """Run the first piece of equipment (if none are onsite) for each equipment
        capability category where the number of requests is greater than or equal to the
        equipment's threshold.
        """
        # Add an initial check to help avoid simultaneous dispatching
        yield self.env.timeout(self.env.get_random_seconds(high=30))

        # Port-based servicing equipment should be handled by the port and does not have
        # a requests-based threshold to meet at this time
        if "TOW" in request.details.service_equipment:
            if request.system_id not in self.port.invalid_systems:
                self.systems_waiting_for_tow.append(request.system_id)
                system = self.windfarm.system(request.system_id)
                yield system.servicing_queue & system.servicing
                yield self.env.timeout(self.env.get_random_seconds(high=1))
                if request.system_id in self.systems_in_tow:
                    return None
                self.invalidate_system(system, tow=True)
                yield self.env.process(self.port.run_tow_to_port(request))
            return None

        # Wait for the actual system or cable to be available
        if request.cable:
            yield self.windfarm.cable(request.system_id).servicing
        else:
            yield self.windfarm.system(request.system_id).servicing

        dispatched = None
        for capability, n_requests in self.request_map.items():
            # For a requests basis, the capability and submitted request must match
            if capability not in request.details.service_equipment:
                continue
            equipment_mapping = self.request_based_equipment.get_mapping(capability)
            for i, equipment in enumerate(equipment_mapping):
                if n_requests < equipment.strategy_threshold:
                    continue

                # Avoid simultaneous dispatches by waiting a random number of seconds
                yield self.env.timeout(self.env.get_random_seconds(high=30))

                # Run only the first piece of equipment in the mapping list, but ensure
                # that it moves to the back of the line after being used
                equipment_obj = equipment.equipment
                if equipment_obj.dispatched:
                    break

                # Either run the repair logic from the port for port-based servicing
                # equipment, so that it can self-mangge or dispatch the servicing
                # equipment directly, when port is an implicitly modeled aspect
                if equipment_obj.port_based:
                    # Equipment-based logic does not manage system availability, so
                    # ensure it's available prior to dispatching
                    if request.system_id in self.port.invalid_systems:
                        break

                    yield self.env.process(
                        self.port.run_unscheduled_in_situ(request, initial=True)
                    )
                else:
                    yield self.env.process(equipment_obj.run_unscheduled_in_situ())

                    # Move the dispatched capability to the end of list to ensure proper
                    # cycling of available servicing equipment
                    self.request_based_equipment.move_equipment_to_end(capability, i)
                dispatched = capability
                break

        yield self.env.timeout(1 / 60)  # wait one minute for repair to register

        # Double check the the number of reqeusts is still below the threshold following
        # the dispatching of a piece of servicing equipment. This mostly pertains to
        # highly frequent request with long repair times and low thresholds.
        if (
            dispatched is None
            or equipment_obj.port_based
            or dispatched not in self.request_map
        ):
            return None
        n_requests = self.request_map[dispatched]
        threshold_check = [
            n_requests >= eq.strategy_threshold
            for eq in self.request_based_equipment.get_mapping(dispatched)
        ]
        if any(threshold_check):
            new_request_check = [
                x
                for x in self.items
                if dispatched in x.details.service_equipment
                and not self._is_request_processing(x)
                and x is not request
            ]
            if new_request_check:
                new_request = self.get(lambda x: x == new_request_check[0]).value
                yield self.env.process(self._run_equipment_requests(new_request))
        return None

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
        self.request_status_map["pending"].update([request.request_id])
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
            self.env.process(self._run_equipment_downtime(request))
        if self.request_based_equipment.is_running:
            self.env.process(self._run_equipment_requests(request))

    def get_request_by_system(
        self, equipment_capability: list[str], system_id: str | None = None
    ) -> FilterStoreGet | None:
        """Gets all repair requests for a certain turbine with given a sequence of
        ``equipment_capability`` as long as it isn't registered as unable to be
        serviced.

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
        if TYPE_CHECKING:
            assert isinstance(equipment_capability, set)
        for request in requests:
            if self._is_request_processing(request):
                continue
            if request.system_id in self.systems_waiting_for_tow:
                continue
            if equipment_capability.intersection(request.details.service_equipment):
                # If this is the first request for the system, make sure no other
                # servicing equipment can access i
                self.request_status_map["pending"].difference_update(
                    [requests[0].request_id]
                )
                self.request_status_map["processing"].update([requests[0].request_id])
                return self.get(lambda x: x is requests[0])

        # There were no matching equipment requirements to match the equipment
        # attempting to retrieve its next request
        return None

    def get_request_by_severity(
        self,
        equipment_capability: list[str] | set[str],
        severity_level: int | None = None,
    ) -> FilterStoreGet | None:
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
            if self._is_request_processing(request):
                continue
            if request.system_id in self.systems_waiting_for_tow:
                continue
            if request.cable:
                if not self.windfarm.cable(request.system_id).servicing.triggered:
                    continue
            else:
                if not self.windfarm.system(request.system_id).servicing.triggered:
                    continue
            if request.system_id not in self.invalid_systems:
                if equipment_capability.intersection(request.details.service_equipment):
                    self.request_status_map["pending"].difference_update(
                        [request.request_id]
                    )
                    self.request_status_map["processing"].update([request.request_id])
                    return self.get(lambda x: x is request)

        # Ensure None is returned if nothing is found in the loop just as a FilterGet
        # would if allowed to oeprate without the above looping to identify multiple
        # criteria and acting on the request before it's processed
        return None

    def invalidate_system(
        self, system: System | Cable | str, tow: bool = False
    ) -> None:
        """Disables the ability for servicing equipment to service a specific system,
        sets the turbine status to be in servicing, and interrupts all the processes
        to turn off operations.

        Parameters
        ----------
        system : System | Cable | str
            The system, cable, or ``str`` id of one to disable repairs.
        tow : bool, optional
            Set to True if this is for a tow-to-port request.
        """
        if isinstance(system, str):
            if "::" in system:
                system = self.windfarm.cable(system)
            else:
                system = self.windfarm.system(system)

        if system.id not in self.invalid_systems and system.servicing.triggered:
            system.servicing_queue = self.env.event()
            self.invalid_systems.append(system.id)
        else:
            raise RuntimeError(
                f"{self.env.simulation_time} {system.id} already being serviced"
            )
        if tow:
            self.systems_in_tow.append(system.id)
            _ = self.systems_waiting_for_tow.pop(
                self.systems_waiting_for_tow.index(system.id)
            )

    def interrupt_system(
        self, system: System | Cable, replacement: str | None = None
    ) -> None:
        """Sets the turbine status to be in servicing, and interrupts all the processes
        to turn off operations.

        Parameters
        ----------
        system_id : str
            The system to disable repairs.
        replacement: str | None, optional
            If a subassebly `id` is provided, this indicates the interruption is caused
            by its replacement event. Defaults to None.
        """
        if system.servicing.triggered and system.id in self.invalid_systems:
            system.servicing = self.env.event()
            system.interrupt_all_subassembly_processes(replacement=replacement)
        else:
            raise RuntimeError(
                f"{self.env.simulation_time} {system.id} already being serviced"
            )

    def register_repair(self, repair: RepairRequest) -> Generator:
        """Registers the repair as complete with the repair managiner.

        Parameters
        ----------
        repair : RepairRequest
            The repair that has been completed.
        port : bool, optional
            If True, indicates that a port handled the repair, otherwise that a managed
            servicing equipment handled the repair, by default False.

        Yields
        ------
        Generator
            The ``completed_requests.put()`` that registers completion.
        """
        self.request_status_map["processing"].difference_update([repair.request_id])
        self.request_status_map["completed"].update([repair.request_id])
        yield self.completed_requests.put(repair)
        yield self.in_process_requests.get(lambda x: x is repair)

    def enable_requests_for_system(
        self, system: System | Cable, tow: bool = False
    ) -> None:
        """Reenables service equipment operations on the provided system.

        Parameters
        ----------
        system_id : System | Cable
            The ``System`` or ``Cable`` that can be operated on again.
        tow : bool, optional
            Set to True if this is for a tow-to-port request.
        """
        if system.servicing.triggered:
            raise RuntimeError(
                f"{self.env.simulation_time} Repairs were already completed"
                f" at {system.id}"
            )
        _ = self.invalid_systems.pop(self.invalid_systems.index(system.id))
        if tow:
            _ = self.systems_in_tow.pop(self.systems_in_tow.index(system.id))
        system.servicing.succeed()
        system.servicing_queue.succeed()
        system.interrupt_all_subassembly_processes()

    def get_all_requests_for_system(
        self, agent: str, system_id: str
    ) -> list[RepairRequest] | None | Generator:
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
            self.request_status_map["pending"].difference_update([request.request_id])
            self.request_status_map["processing"].update([request.request_id])
            _ = yield self.get(lambda x: x is request)  # pylint: disable=W0640

        return requests

    def purge_subassembly_requests(
        self, system_id: str, subassembly_id: str, exclude: list[str] = []
    ) -> list[RepairRequest] | None:
        """Yields all the requests for a system/subassembly combination. This is
        intended to be used to remove erroneous requests after a subassembly has been
        replaced.

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
        ------
        Optional[list[RepairRequest]]
            All requests made to the repair manager for the provided system/subassembly
            combination. Returns None if self.items is empty or the loop terminates
            without finding what it is looking for.
        """
        if not self.items:
            return None

        # First check the system matches because we'll need these separated later to
        # ensure we don't incorrectly remove towing requests from other subassemblies
        system_requests = [
            request
            for request in self.items
            if request.system_id == system_id and request.request_id not in exclude
        ]
        if system_requests == []:
            return None

        subassembly_requests = [
            request
            for request in system_requests
            if request.subassembly_id == subassembly_id
        ]
        if subassembly_requests == []:
            return None

        for request in subassembly_requests:
            which = "repair" if isinstance(request.details, Failure) else "maintenance"
            self.env.log_action(
                system_id=request.system_id,
                system_name=request.system_name,
                part_id=request.subassembly_id,
                part_name=request.subassembly_name,
                system_ol=float("nan"),
                part_ol=float("nan"),
                agent="RepairManager",
                action=f"{which} canceled",
                reason="replacement required",
                request_id=request.request_id,
            )
            self.request_status_map["pending"].difference_update([request.request_id])
            self.request_status_map["processing"].update([request.request_id])
            _ = self.get(lambda x: x is request)  # pylint: disable=W0640
        sid = request.system_id

        # Ensure that if it was reset, and a tow was waiting, that it gets cleared,
        # unless a separate subassembly required the tow
        if sid in self.systems_waiting_for_tow:
            other_subassembly_match = [
                r for r in system_requests if "TOW" in r.details.service_equipment
            ]
            if sid not in self.systems_in_tow and other_subassembly_match == []:
                _ = self.systems_waiting_for_tow.pop(
                    self.systems_waiting_for_tow.index(sid)
                )

        return subassembly_requests

    @property
    def request_map(self) -> dict[str, int]:
        """Creates an updated mapping between the servicing equipment capabilities and
        the number of requests that fall into each capability category (nonzero values
        only).
        """
        requests = dict(
            Counter(
                chain.from_iterable(r.details.service_equipment for r in self.items)
            )
        )
        return requests
