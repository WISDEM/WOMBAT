"""Creates the necessary repair classes."""


from __future__ import annotations

import attr
import numpy as np
from typing import (  # type: ignore
    TYPE_CHECKING,
    Dict,
    List,
    Union,
    Iterator,
    Optional,
    Sequence,
)
from itertools import chain
from collections import Counter
from simpy.resources.store import FilterStore, FilterStoreGet  # type: ignore

from wombat.core import Failure, Maintenance, RepairRequest, WombatEnvironment


if TYPE_CHECKING:
    from wombat.windfarm import Windfarm
    from wombat.core.service_equipment import ServiceEquipment


@attr.s(auto_attribs=True)
class EquipmentMap:
    """Internal mapping for servicing equipment strategy information."""

    strategy_threshold: Union[int, float] = attr.ib()
    equipment: ServiceEquipment = attr.ib()


@attr.s(auto_attribs=True)
class StrategyMap:
    """Internal mapping for equipment capabilities and their data."""

    CTV: List[EquipmentMap] = attr.ib(factory=list)
    SCN: List[EquipmentMap] = attr.ib(factory=list)
    LCN: List[EquipmentMap] = attr.ib(factory=list)
    CAB: List[EquipmentMap] = attr.ib(factory=list)
    RMT: List[EquipmentMap] = attr.ib(factory=list)
    DRN: List[EquipmentMap] = attr.ib(factory=list)
    DSV: List[EquipmentMap] = attr.ib(factory=list)

    def update(
        self, capability: str, threshold: Union[int, float], equipment: ServiceEquipment
    ) -> None:
        """A method to update the strategy mapping between capability types and the
        available ``ServiceEquipment`` objects.

        Parameters
        ----------
        capability : str
            The ``equipment``'s capability.
        threshold : Union[int, float]
            The threshold for ``equipment``'s strategy.
        equipment : ServiceEquipment
            The actual ``ServiceEquipment`` object to be logged.

        Raises
        ------
        ValueError
            Raised if there is an invalid capability, though this shouldn't be able to
            be reached.
        """
        if capability == "CTV":
            self.CTV.append(EquipmentMap(threshold, equipment))
        elif capability == "SCN":
            self.SCN.append(EquipmentMap(threshold, equipment))
        elif capability == "LCN":
            self.LCN.append(EquipmentMap(threshold, equipment))
        elif capability == "CAB":
            self.CAB.append(EquipmentMap(threshold, equipment))
        elif capability == "RMT":
            self.RMT.append(EquipmentMap(threshold, equipment))
        elif capability == "DRN":
            self.DRN.append(EquipmentMap(threshold, equipment))
        elif capability == "DSV":
            self.DSV.append(EquipmentMap(threshold, equipment))
        else:
            # This should not even be able to be reached
            raise ValueError("Invalid servicing equipment has been provided!")


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

        self.downtime_based_equipment = StrategyMap()
        self.request_based_equipment = StrategyMap()

    def _update_equipment_map(self, service_equipment: ServiceEquipment) -> None:
        """Updates ``equipment_map`` with a provided servicing equipment object."""
        capability = service_equipment.settings.capability
        strategy = service_equipment.settings.strategy
        strategy_threshold = service_equipment.settings.strategy_threshold  # type: ignore

        if strategy == "downtime":
            mapping = self.downtime_based_equipment
        elif strategy == "requests":
            mapping = self.request_based_equipment
        else:
            # Shouldn't be possible to get here!
            raise ValueError("Invalid servicing equipment!")

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
            raise ValueError("A valid ``RepairRequest`` must be submitted")

        request_id = f"{prefix}{str(self._current_id).zfill(8)}"
        self._current_id += 1
        return request_id

    def _run_equipment_downtime(self) -> None:
        """Run any equipment that has a pending request where the current windfarm
        operating capacity is less than or equal to the servicing equipment's threshold.
        """
        operating_capacity = self.windfarm.current_availability
        for capability in self.request_map:
            equipment_mapping = getattr(self.downtime_based_equipment, capability)
            for equipment in equipment_mapping:
                if operating_capacity > equipment.strategy_threshold:
                    continue
                if equipment.equipment.onsite or equipment.equipment.enroute:
                    continue
                self.env.process(equipment.equipment.run_unscheduled())
                break

    def _run_equipment_requests(self) -> None:
        """Run the first piece of equipment (if none are onsite) for each equipment
        capability category where the number of requests is greater than or equal to the
        equipment's threshold.
        """
        for capability, n_requests in self.request_map.items():
            equipment_mapping = getattr(self.request_based_equipment, capability)
            for i, equipment in enumerate(equipment_mapping):
                if n_requests > equipment.strategy_threshold:
                    continue
                # Run only the first piece of equipment in the mapping list, but ensure
                # that it moves to the back of the line after being used
                if equipment.equipment.onsite or equipment.equipment.enroute:
                    equipment_mapping.append(equipment_mapping.pop(i))
                    break
                self.env.process(equipment.equipment.run_unscheduled())
                equipment_mapping.append(equipment_mapping.pop(i))

    def submit_request(self, request: RepairRequest) -> RepairRequest:
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

        self._run_equipment_downtime()
        self._run_equipment_requests()

        return request

    def get_request_by_turbine(
        self, equipment_capability: Sequence[str], turbine_id: Optional[str] = None
    ) -> Optional[FilterStoreGet]:
        """Gets all repair requests for a certain turbine with maximum severity of ``severity_max``.

        Note: For now, this will simply order the items by ``Turbine.id`` and
        return the first turbine.

        Parameters
        ----------
        equipment_capability : List[str]
            The capability of the equipment requesting possible repairs to make.
        turbine_id : Optional[str], optional
            ID of the turbine; should correspond to ``Turbine.id``, by default None.
            If None, then it will simply sort the list by ``Turbine.id`` and
            return the first repair requested.

        Returns
        -------
        Optional[FilterStoreGet]
            All of the repair requests for the focal turbine or None, if self.items
            is empty.
        """
        if not self.items:
            return None

        equipment_capability = set(equipment_capability)  # type: ignore
        for request in sorted(self.items, key=lambda x: x.turbine_id):
            capability_overlap = equipment_capability.intersection(request.equipment)  # type: ignore
            valid_turbine = turbine_id is None or request.turbine_id == turbine_id
            if valid_turbine and capability_overlap:
                return self.get(lambda x: x == request)

        # In case the loop above iterates all the way to the end, nothing was
        # found so return None. This is probably an error for which an error
        # should be raised, but this keeps the type hints correct for now.

        return None

    def get_next_highest_severity_request(
        self, equipment_capability: Sequence[str], severity_level: Optional[int] = None
    ) -> Optional[FilterStoreGet]:
        """Gets the next repair request by ``severity_level``.

        Parameters
        ----------
        equipment_capability : List[str]
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

        equipment_capability = set(equipment_capability)  # type: ignore
        requests = sorted(self.items, key=lambda x: x.severity_level, reverse=True)
        for request in requests:
            capability_overlap = equipment_capability & set(  # type: ignore
                request.details.service_equipment
            )
            requested_severity: bool = severity_level in (request.details.level, None)
            if requested_severity and capability_overlap:
                return self.get(lambda x: x == request)

        # This return statement ensures there is always a known return type,
        # but consider raising an error here if the loop is always assumed to
        # break when something is found.

        return None

    def purge_subassembly_requests(
        self, system_id: str, subassembly_id: str
    ) -> Optional[Iterator]:
        """Yields all the requests for a system/subassembly combination. This is intended
        to be used to remove erroneous requests after a subassembly has been replaced.

        Parameters
        ----------
        system_id : str
            ``System.id``.
        subassembly_id : str
            ``Subassembly.id``.

        Yields
        -------
        Optional[Iterator]
            All requests made to the repair manager for the provided system/subassembly
            combination. Returns None if self.items is empty or the loop terminates
            without finding what it is looking for.
        """
        if not self.items:
            return None

        for request in self.items:
            system_match = request.system_id == system_id
            subassembly_match = request.subassembly_id == subassembly_id
            if system_match and subassembly_match:
                yield self.get(lambda x: x == request)  # pylint: disable=W0640

                self.env.log_action(
                    system_id=request.system_id,
                    system_name=request.system_name,
                    part_id=request.system_name,
                    part_name=request.part_name,
                    system_ol="n/a",
                    part_ol="n/a",
                    agent="RepairManager",
                    action="request canceled",
                    reason="replacement required",
                    request_id=request.request_id,
                )

        # This is here to ensure there is always an explicit return value,
        # but you may want to consider raising an error here.

        return None

    @property
    def request_map(self) -> Dict[str, int]:
        """Creates an updated mapping between the servicing equipment capabilities and
        the number of requests that fall into each capability category (nonzero values only).
        """
        # mapping = dict(CTV=0, SCN=0, LCN=0, CAB=0, RMT=0, DRN=0, DSV=0)
        all_requests = [request.details.service_equipment for request in self.items]
        requests = dict(Counter(chain.from_iterable(all_requests)))
        # mapping.update(requests)
        return requests
