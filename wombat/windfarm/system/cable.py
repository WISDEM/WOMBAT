""""Defines the Cable class and cable simulations."""


from typing import Generator  # type: ignore
from itertools import chain

import numpy as np  # type: ignore
import simpy  # type: ignore
import networkx as nx

from wombat.core import (
    Failure,
    Maintenance,
    RepairRequest,
    SubassemblyData,
    WombatEnvironment,
)


# TODO: Need a better method for checking if a repair has been made to bring the
# subassembly back online
HOURS = 8760
TIMEOUT = 24  # Wait time of 1 day for replacement to occur


class Cable:
    """The cable system/asset class.

    Parameters
    ----------
    windfarm : ``wombat.windfarm.Windfarm``
        The ``Windfarm`` object.
    env : WombatEnvironment
        The simulation environment.
    cable_id : str
        The unique identifier for the cable.
    start_node : str
        The starting point (``system.id``) (turbine or substation) of the cable segment.
    cable_data : dict
        The dictionary defining the cable segment.
    """

    def __init__(
        self,
        windfarm,
        env: WombatEnvironment,
        start_node: str,
        end_node: str,
        cable_data: dict,
    ) -> None:
        """Initializes the ``Cable`` class.

        Parameters
        ----------
        windfarm : ``wombat.windfarm.Windfarm``
            The ``Windfarm`` object.
        env : WombatEnvironment
            The simulation environment.
        cable_id : str
            The unique identifier for the cable.
        start_node : str
            The starting point (``system.id``) (turbine or substation) of the cable segment.
        end_node : str
            The ending point (``system.id``) (turbine or substation) of the cable segment.
        cable_data : dict
            The dictionary defining the cable segment.
        """

        self.env = env
        self.windfarm = windfarm
        self.start_node = start_node
        self.end_node = end_node
        self.id = f"cable::{start_node}::{end_node}"
        # TODO: need to be able to handle substations, which are not being modeled currently
        self.system = windfarm.graph.nodes(data=True)[start_node][
            "system"
        ]  # MAKE THIS START

        # Map the upstream substations and turbines, and cables
        upstream = nx.dfs_successors(self.windfarm.graph, end_node)
        self.upstream_nodes = [self.end_node]
        if len(upstream) > 0:
            self.upstream_nodes = list(chain(self.upstream_nodes, *upstream.values()))
        self.upstream_cables = list(nx.edge_dfs(self.windfarm.graph, end_node))

        cable_data = {**cable_data, "system_value": self.system.value}
        self.data = SubassemblyData.from_dict(cable_data)
        self.name = self.data.name

        self.downstream_failure = False
        self.operating_level = 1.0
        self.servicing = False
        self.broken = False

        # TODO: need to get the time scale of a distribution like this
        self.processes = dict(self._create_processes())

    def _create_processes(self):
        """Creates the processes for each of the failure and maintenance types.

        Yields
        -------
        Tuple[Union[str, int], simpy.events.Process]
            Creates a dictionary to keep track of the running processes within the
            subassembly.
        """
        for level, failure in self.data.failures.items():
            yield level, self.env.process(self.run_single_failure(failure))

        for i, maintenance in enumerate(self.data.maintenance):
            yield f"m{i}", self.env.process(self.run_single_maintenance(maintenance))

    def interrupt_processes(self) -> None:
        """Interrupts all of the running processes within the subassembly except for the
        process associated with failure that triggers the catastrophic failure.

        Parameters
        ----------
        subassembly : Subassembly
            The subassembly that should have all processes interrupted.
        """
        for _, process in self.processes.items():
            try:
                process.interrupt()
            except RuntimeError:
                # This error occurs for the process halting all other processes.
                pass

    def interrupt_all_subassembly_processes(self) -> None:
        """Thin wrapper for ``interrupt_processes`` to keep usage the same as systems."""
        self.interrupt_processes()

    def stop_all_upstream_processes(self, failure: Failure) -> None:
        """Stops all upstream turbines from producing power by setting their
        ``System.cable_failure`` to ``True``.

        Parameters
        ----------
        failure : Failre
            The ``Failure`` that is causing a string shutdown.
        """
        # Shut down all upstream objects and set the flag for an downstream cable failure
        for node in self.upstream_nodes:
            system = self.windfarm.system(node)
            system.interrupt_all_subassembly_processes()
            system.cable_failure = True
            self.env.log_action(
                system_id=node,
                system_name=system.name,
                system_ol=system.operating_level,
                part_ol=np.nan,
                agent=self.name,
                action="repair request",
                reason=failure.description,
                additional="cable failure shutting off all upstream cables and turbines that are still operating",
                request_id=failure.request_id,
            )

        for edge in self.upstream_cables:
            cable = self.windfarm.cable(edge)
            cable.interrupt_processes()
            cable.downstream_failure = True
            self.env.log_action(
                part_id=cable.id,
                part_name=cable.name,
                system_ol=np.nan,
                part_ol=cable.operating_level,
                agent=self.name,
                action="repair request",
                reason=failure.description,
                additional="cable failure shutting off all upstream cables and turbines that are still operating",
                request_id=failure.request_id,
            )

    def run_single_maintenance(self, maintenance: Maintenance) -> Generator:
        """Runs a process to trigger one type of maintenance request throughout the simulation.

        Parameters
        ----------
        maintenance : Maintenance
            A maintenance category.

        Yields
        -------
        simpy.events.Timeout
            Time between maintenance requests.
        """
        while True:
            hours_to_next = maintenance.frequency
            if hours_to_next == 0:
                remainder = self.env.max_run_time - self.env.now
                try:
                    yield self.env.timeout(remainder)
                except simpy.Interrupt:
                    remainder -= self.env.now

            while hours_to_next > 0:
                try:
                    # If the replacement has not been completed, then wait another minute
                    if self.broken or self.downstream_failure or self.servicing:
                        yield self.env.timeout(TIMEOUT)
                        continue

                    start = self.env.now
                    yield self.env.timeout(hours_to_next)
                    hours_to_next = 0

                    # Automatically submit a repair request
                    # NOTE: mypy is not caught up with attrs yet :(
                    repair_request = RepairRequest(  # type: ignore
                        self.system.id,
                        self.system.name,
                        self.id,
                        self.name,
                        0,
                        maintenance,
                        cable=True,
                        upstream_turbines=self.upstream_nodes,
                    )
                    repair_request = self.system.repair_manager.register_request(
                        repair_request
                    )
                    self.env.log_action(
                        system_id=self.system.id,
                        system_name=self.system.name,
                        part_id=self.id,
                        part_name=self.name,
                        system_ol=self.system.operating_level,
                        part_ol=self.operating_level,
                        agent=self.name,
                        action="maintenance request",
                        reason=maintenance.description,
                        additional="request",
                        request_id=repair_request.request_id,
                    )
                    self.system.repair_manager.submit_request(repair_request)

                except simpy.Interrupt:
                    if self.broken:
                        # The subassembly had so restart the maintenance cycle
                        hours_to_next = 0
                        continue
                    else:
                        # A different subassembly failed so we need to subtract the
                        # amount of passed time
                        hours_to_next -= self.env.now - start

    def run_single_failure(self, failure: Failure) -> Generator:
        """Runs a process to trigger one type of failure repair request throughout the simulation.

        Parameters
        ----------
        failure : Failure
            A failure classification.

        Yields
        -------
        simpy.events.Timeout
            Time between failure events that need to request a repair.
        """
        while True:
            hours_to_next = failure.hours_to_next_failure()
            if hours_to_next is None:
                remainder = self.env.max_run_time - self.env.now
                try:
                    yield self.env.timeout(remainder)
                except simpy.Interrupt:
                    remainder -= self.env.now

            while hours_to_next > 0:  # type: ignore
                try:
                    if self.operating_level == 0 or self.downstream_failure:
                        yield self.env.timeout(TIMEOUT)
                        continue

                    start = self.env.now
                    yield self.env.timeout(hours_to_next)
                    hours_to_next = 0
                    self.operating_level *= 1 - failure.operation_reduction

                    # Automatically submit a repair request
                    # NOTE: mypy is not caught up with attrs yet :(
                    repair_request = RepairRequest(  # type: ignore
                        self.id,
                        self.name,
                        self.id,
                        self.name,
                        failure.level,
                        failure,
                        cable=True,
                        upstream_turbines=self.upstream_nodes,
                    )
                    repair_request = self.system.repair_manager.register_request(
                        repair_request
                    )

                    if failure.operation_reduction == 1:
                        self.broken = True

                        # Remove previously submitted requests as a replacement is required
                        _ = self.system.repair_manager.purge_subassembly_requests(
                            self.id, self.id, exclude=[repair_request.request_id]
                        )
                        self.interrupt_processes()
                        self.stop_all_upstream_processes(failure)

                    self.env.log_action(
                        system_id=self.id,
                        system_name=self.name,
                        part_id=self.id,
                        part_name=self.name,
                        system_ol=self.system.operating_level,
                        part_ol=self.operating_level,
                        agent=self.name,
                        action="repair request",
                        reason=failure.description,
                        additional=f"severity level {failure.level}",
                        request_id=repair_request.request_id,
                    )
                    self.system.repair_manager.submit_request(repair_request)

                except simpy.Interrupt:
                    if self.broken:
                        # The subassembly had to be replaced so the timing to next failure
                        # will reset
                        hours_to_next = 0
                        continue
                    else:
                        # A different subassembly failed so we need to subtract the
                        # amount of passed time
                        hours_to_next -= self.env.now - start
