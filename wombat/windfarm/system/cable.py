""""Defines the Cable class and cable simulations."""
from __future__ import annotations

from typing import Generator

import numpy as np
import simpy

from wombat.core import (
    Failure,
    Maintenance,
    RepairRequest,
    SubassemblyData,
    WombatEnvironment,
)


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
    connection_type : str
        The type of cable. Must be one of "array" or "export".
    start_node : str
        The starting point (``system.id``) (turbine or substation) of the cable segment.
    cable_data : dict
        The dictionary defining the cable segment.
    """

    def __init__(
        self,
        windfarm,
        env: WombatEnvironment,
        connection_type: str,
        start_node: str,
        end_node: str,
        cable_data: dict,
        name: str | None = None,
    ) -> None:
        """Initializes the ``Cable`` class.

        Parameters
        ----------
        windfarm : ``wombat.windfarm.Windfarm``
            The ``Windfarm`` object.
        env : WombatEnvironment
            The simulation environment.
        connection_type : str
            One of "export" or "array".
        cable_id : str
            The unique identifier for the cable.
        start_node : str
            The starting point (``system.id``) (turbine or substation) of the cable segment.
        end_node : str
            The ending point (``system.id``) (turbine or substation) of the cable segment.
        cable_data : dict
            The dictionary defining the cable segment.
        name : str | None
            The name of the cable to use during logging.
        """

        self.env = env
        self.windfarm = windfarm
        self.connection_type = connection_type
        self.start_node = start_node
        self.end_node = end_node
        self.id = f"cable::{start_node}::{end_node}"
        self.system = windfarm.system(start_node)

        if self.connection_type not in ("array", "export"):
            raise ValueError(
                f"Input to `connection_type` for {self.id} must be one of 'array' or 'export'."
            )

        cable_data = {**cable_data, "system_value": self.system.value}
        self.data = SubassemblyData.from_dict(cable_data)
        self.name = self.data.name if name is None else name

        self.operating_level = 1.0
        self.servicing = self.env.event()
        self.downstream_failure = self.env.event()
        self.broken = self.env.event()

        # Ensure events start as processed and inactive
        self.servicing.succeed()
        self.downstream_failure.succeed()
        self.broken.succeed()

        # TODO: need to get the time scale of a distribution like this
        self.processes = dict(self._create_processes())

    def set_string_details(self, start_node: str, substation: str):
        """Sets the starting turbine for the string to be used for traversing the
        correct upstream connections when resetting after a failure.

        Parameters
        ----------
        start_node : str
            The ``System.id`` for the starting turbine on a string.
        substation : str
            The ``System.id`` for the string's connecting substation.
        """
        self.string_start = start_node
        self.substation = substation

    def finish_setup(self) -> None:
        """Creates the ``upstream_nodes`` and ``upstream_cables`` attributes for use by
        the cable when triggering usptream failures and resetting them after the repair
        is complete.
        """

        wf_map = self.windfarm.wind_farm_map
        if self.connection_type == "array":
            turbines = []
            if self.end_node == self.string_start:
                turbines.append(self.end_node)
                _turbines, cables = wf_map.get_upstream_connections(
                    self.substation, self.string_start, self.string_start
                )
            else:
                _turbines, cables = wf_map.get_upstream_connections(
                    self.substation, self.string_start, self.end_node
                )
            turbines.extend(_turbines)

        if self.connection_type == "export":
            turbines, cables = wf_map.get_upstream_connections_from_substation(
                self.substation
            )

        self.upstream_nodes = turbines
        self.upstream_cables = cables

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

    def stop_all_upstream_processes(self, failure: Failure | Maintenance) -> None:
        """Stops all upstream turbines and cables from producing power by creating a
        ``env.event()`` for each ``System.cable_failure`` and
        ``Cable.downstream_failure``, respectively. In the case of an export cable, each
        string is traversed to stop the substation and upstream turbines and cables.

        Parameters
        ----------
        failure : Failre
            The ``Failure`` that is causing a string shutdown.
        """
        shared_logging = dict(
            agent=self.id,
            action="repair_request",
            reason=failure.description,
            additional="downstream cable failure",
            request_id=failure.request_id,
        )
        upstream_nodes = self.upstream_nodes
        upstream_cables = self.upstream_cables
        if self.connection_type == "export":
            # Flatten the list of lists for shutting down the upstream connections
            upstream_nodes = [el for string in upstream_nodes for el in string]
            upstream_cables = [el for string in upstream_cables for el in string]

            # Turn off the subation
            substation = self.windfarm.system(self.end_node)
            substation.interrupt_all_subassembly_processes()
            substation.cable_failure = self.env.event()
            self.env.log_action(
                system_id=self.end_node,
                system_name=substation.name,
                system_ol=substation.operating_level,
                part_ol=np.nan,
                **shared_logging,  # type: ignore
            )

        for t_id, c_id in zip(upstream_nodes, upstream_cables):
            turbine = self.windfarm.system(t_id)
            turbine.cable_failure = self.env.event()
            turbine.interrupt_all_subassembly_processes()
            self.env.log_action(
                system_id=t_id,
                system_name=turbine.name,
                system_ol=turbine.operating_level,
                part_ol=np.nan,
                **shared_logging,  # type: ignore
            )
            cable = self.windfarm.cable(c_id)
            cable.interrupt_processes()
            cable.downstream_failure = self.env.event()
            self.env.log_action(
                system_id=c_id,
                system_name=cable.name,
                part_id=c_id,
                part_name=cable.name,
                system_ol=np.nan,
                part_ol=cable.operating_level,
                **shared_logging,  # type: ignore
            )

    def trigger_request(self, action: Maintenance | Failure):
        """Triggers the actual repair or maintenance logic for a failure or maintenance
        event, respectively.

        Parameters
        ----------
        action : Maintenance | Failure
            The maintenance or failure event that triggers a ``RepairRequest``.
        """
        which = "maintenance" if isinstance(action, Maintenance) else "repair"
        self.operating_level *= 1 - action.operation_reduction

        # Automatically submit a repair request
        # NOTE: mypy is not caught up with attrs yet :(
        repair_request = RepairRequest(  # type: ignore
            self.id,
            self.name,
            self.id,
            self.name,
            action.level,
            action,
            cable=True,
            upstream_turbines=self.upstream_nodes,
            upstream_cables=self.upstream_cables,
        )
        repair_request = self.system.repair_manager.register_request(repair_request)
        self.env.log_action(
            system_id=self.id,
            system_name=self.name,
            part_id=self.id,
            part_name=self.name,
            system_ol=self.operating_level,
            part_ol=self.operating_level,
            agent=self.name,
            action=f"{which} request",
            reason=action.description,
            additional=f"severity level {action.level}",
            request_id=repair_request.request_id,
        )

        if action.operation_reduction == 1:
            self.broken = self.env.event()
            self.interrupt_processes()
            self.stop_all_upstream_processes(action)

        # Remove previously submitted requests as a replacement is required
        if action.replacement:
            _ = self.system.repair_manager.purge_subassembly_requests(
                self.id, self.id, exclude=[repair_request.request_id]
            )

        self.system.repair_manager.submit_request(repair_request)

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
                    yield self.servicing & self.downstream_failure & self.broken

                    start = self.env.now
                    yield self.env.timeout(hours_to_next)
                    hours_to_next = 0
                    self.trigger_request(maintenance)
                except simpy.Interrupt:
                    if not self.broken.triggered:
                        # The subassembly had to restart the maintenance cycle
                        hours_to_next = 0
                    else:
                        # A different interruption occurred, so subtract the elapsed time
                        hours_to_next -= self.env.now - start  # pylint: disable=E0601

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

            assert isinstance(hours_to_next, (int, float))  # mypy helper
            while hours_to_next > 0:  # type: ignore
                try:
                    yield self.servicing & self.downstream_failure & self.broken

                    start = self.env.now
                    yield self.env.timeout(hours_to_next)
                    hours_to_next = 0
                    self.trigger_request(failure)
                except simpy.Interrupt:
                    if not self.broken.triggered:
                        # Restart after fixing
                        hours_to_next = 0
                    else:
                        # A different interruption occurred, so subtract the elapsed time
                        hours_to_next -= self.env.now - start  # pylint: disable=E0601
