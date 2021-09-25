"""Creates the necessary repair classes."""


from __future__ import annotations

from typing import (  # type: ignore
    TYPE_CHECKING,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
)

import attr
import numpy as np
from simpy.resources.store import FilterStore, FilterStoreGet  # type: ignore

from wombat.core import Failure, Maintenance, RepairRequest, WombatEnvironment


if TYPE_CHECKING:
    from wombat.core.service_equipment import ServiceEquipment
    from wombat.windfarm import Windfarm


@attr.s(auto_attribs=True)
class RepairManager(FilterStore):
    """Provides a class to manage repair and maintenance tasks.

    Parameters
    ----------
    FilterStore : simpy.resources.store.FilterStore
        The `simpy` class on which RepairManager is based to manage the repair and
        maintenance tasks.
    env : wombat.core.WombatEnvironment
        The simulation environment.
    capacity : float
        The maximum number of tasks that can be submitted to the manager, by default `np.inf`.
    strategy : str
        For any unscheduled maintenance servicing equipment, this determines the
        strategy for dispatching. Should be on of "downtime" or "requests". By default,
        None, to indicate that all servicing equipment is pre-scheduled.
    strategy_threshold : str
        For downtime-based scenarios, this is based on the operating level, and should
        be in the range (0, 1). For reqest-based scenarios, this is the maximum number
        of requests that are allowed to build up for any given type of unscheduled
        servicing equipment, should be an integer >= 1.
    """

    env: WombatEnvironment
    capacity: float = attr.ib(default=np.inf, converter=float)
    strategy: str = attr.ib(default=None)
    strategy_threshold: Union[int, float] = attr.ib(default=0)

    # Registering attributes without initialization
    equipment: List[ServiceEquipment] = attr.ib(factory=list, init=False)
    windfarm: Windfarm = attr.ib(init=False)
    equipment_map: Dict[str, List[ServiceEquipment]] = attr.ib(
        default=dict(CTV=[], SCN=[], LCN=[], CAB=[], RMT=[], DRN=[], DSV=[]), init=False
    )
    _current_id: int = attr.ib(default=0, init=False)

    def __init__(self, env: WombatEnvironment, capacity: float = np.inf) -> None:
        super().__init__(env, capacity)

    @strategy.validator
    def check_strategy(self, attribute: str, value: str) -> None:
        """Ensures the `strategy` provided is valid."""
        if value not in ("downtime", "requests"):
            raise ValueError(
                "`strategy` for unscheduled maintence handling must be one of",
                "'downtime' or 'requests'",
            )

    @strategy_threshold.validator
    def validate_threshold(self, attribute: str, value: Union[int, float]) -> None:
        """Ensures a valid threshold is provided for a given `strategy`."""
        if not isinstance(value, (int, float)):
            raise TypeError("`strategy_threshold` must be an `int` or `float`!")

        if self.strategy == "downtime":
            if value <= 0 or value >= 1:
                raise ValueError(
                    "Downtime-based strategies must have a `strategy_threshold`",
                    "between 0 and 1, non-inclusive!",
                )
        if self.strategy == "requests":
            if value <= 0:
                raise ValueError(
                    "Requests-based strategies must have a `strategy_threshold`",
                    "greater than 0!",
                )

    def _update_equipment_map(self, service_equipment: ServiceEquipment) -> None:
        """Updates `equipment_map` with a provided servicing equipment object."""
        capability = service_equipment.settings.capability
        if isinstance(capability, list):
            for c in capability:
                self.equipment_map[c].append(service_equipment)
        else:
            self.equipment_map[capability].append(service_equipment)

    def _register_windfarm(self, windfarm: Windfarm) -> None:
        """Adds the simulation windfarm to the class attributes"""
        self.windfarm = windfarm

    def _register_equipment(self, service_equipment: ServiceEquipment) -> None:
        """Adds the servicing equipment to the class attributes and adds it to the
        capabilities mapping.
        """
        self._update_equipment_map(service_equipment)

    def _create_request_id(self, request: RepairRequest) -> str:
        """Creates a unique `request_id` to be logged in the `request`.

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
            If the `request.details` property is not a `Failure` or `Maintenance` object,
            then a ValueError will be raised.
        """
        if isinstance(request.details, Failure):
            prefix = "RPR"
        elif isinstance(request.details, Maintenance):
            prefix = "MNT"
        else:
            raise ValueError("A valid `RepairRequest` must be submitted")

        request_id = f"{prefix}{str(self._current_id).zfill(8)}"
        self._current_id += 1
        return request_id

    def _run_equipment_downtime(self, request_mapping: Dict[str, int]) -> None:
        """Run any equipment that has a pending request for its capability category.

        Parameters
        ----------
        request_mapping: Dict[str, int]
            A copy of the `request_map` object.
        """
        for capability, n_requests in request_mapping.items():
            if n_requests > 0:
                # Run only the first piece of equipment in the mapping list
                equipment = self.equipment_map[capability].pop(0)
                self.env.process(equipment.run_unscheduled())
                self.equipment_map[capability].append(equipment)

    def _run_equipment_requests(self, request_mapping: Dict[str, int]) -> None:
        for capability, n_requests in request_mapping.items():
            if n_requests >= self.strategy_threshold:
                # Run only the first piece of equipment in the mapping list, but ensure
                # that it moves to the back of the line after being used
                equipment = self.equipment_map[capability].pop(0)
                self.env.process(equipment.run_unscheduled())
                self.equipment_map[capability].append(equipment)

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
        self.update_equipment_request_map(request)

        request_mapping = self.request_map
        if self.strategy == "downtime":
            if self.windfarm.current_availability <= self.strategy_threshold:
                self.run_equipment_downtime(request_mapping)
        elif self.strategy == "requests":
            if any(val >= self.strategy_threshold for val in request_mapping.values()):
                self.run_equipment_requests(request_mapping)

        return request

    def get_request_by_turbine(
        self, equipment_capability: Sequence[str], turbine_id: Optional[str] = None
    ) -> Optional[FilterStoreGet]:
        """Gets all repair requests for a certain turbine with maximum severity of `severity_max`.

        Note: For now, this will simply order the items by `Turbine.id` and
        return the first turbine.

        Parameters
        ----------
        equipment_capability : List[str]
            The capability of the equipment requesting possible repairs to make.
        turbine_id : Optional[str], optional
            ID of the turbine; should correspond to `Turbine.id`, by default None.
            If None, then it will simply sort the list by `Turbine.id` and
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
        """Gets the next repair request by `severity_level`.

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
            `System.id`.
        subassembly_id : str
            `Subassembly.id`.

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
                yield self.get(lambda x: x == request)

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
        """Creates an update mapping between the servicing equipment and the number
        of requests that each category of capability can service.
        """
        mapping = dict(CTV=0, SCN=0, LCN=0, CAB=0, RMT=0, DRN=0, DSV=0)
        for request in self.items:
            capability = request.details.service_equipment
            if isinstance(capability, list):
                for c in capability:
                    mapping[c] += 1
            else:
                mapping[capability] += 1
        return mapping
