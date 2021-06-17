"""Creates the necessary repair classes."""


from typing import Iterator, Optional, Sequence  # type: ignore

import numpy as np  # type: ignore
from simpy.resources.store import FilterStore, FilterStoreGet  # type: ignore

from wombat.core import Failure, Maintenance, RepairRequest, WombatEnvironment


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
    """

    def __init__(self, env: WombatEnvironment, capacity: float = np.inf) -> None:
        """Initializes the class.

        Parameters
        ----------
        env : WombatEnvironment
            The simualation environment.
        capacity : float, optional
            The maximum number of tasks that the manager should be storing, by default
            `np.inf`.
        """
        super().__init__(env, capacity)
        self.env = env
        self._current_id = 0

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
