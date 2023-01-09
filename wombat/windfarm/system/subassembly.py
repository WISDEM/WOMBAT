"""Provides the Subassembly class"""

from __future__ import annotations

from typing import Generator

import simpy

from wombat.core import (
    Failure,
    Maintenance,
    RepairRequest,
    SubassemblyData,
    WombatEnvironment,
)


class Subassembly:
    """A major system composes the turbine or substation objects."""

    def __init__(
        self,
        system,
        env: WombatEnvironment,
        s_id: str,
        subassembly_data: dict,
    ) -> None:
        """Creates a subassembly object that models various maintenance and failure types.

        Parameters
        ----------
        system : wombat.windfarm.System
            The system containing subassembly object(s).
        env : WombatEnvironment
            The simulation environment.
        s_id : str
            A unique identifier for the subassembly within the system.
        subassembly_data : dict
            A dictionary to be passed to ``SubassemblyData`` for creation and validation.
        """

        self.env = env
        self.system = system
        self.id = s_id

        subassembly_data = {**subassembly_data, "system_value": self.system.value}
        self.data = SubassemblyData.from_dict(subassembly_data)
        self.name = self.data.name

        self.operating_level = 1.0
        self.broken = self.env.event()
        self.broken.succeed()  # start the event as inactive

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

    def recreate_processes(self) -> None:
        """If a turbine is being entirely reset after a tow-to-port repair, then all
        processes are assumed to be reset to 0, and not pick back up where they left off.
        """
        self.processes = dict(self._create_processes())

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
        """Thin wrapper for ``system.interrupt_all_subassembly_processes``."""
        self.system.interrupt_all_subassembly_processes()

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
        if action.operation_reduction == 1:
            self.broken = self.env.event()
            self.interrupt_all_subassembly_processes()

        # Remove previously submitted requests if a replacement is required
        if action.replacement:
            _ = self.system.repair_manager.purge_subassembly_requests(
                self.system.id, self.id
            )

        # Automatically submit a repair request
        # NOTE: mypy is not caught up with attrs yet :(
        repair_request = RepairRequest(  # type: ignore
            system_id=self.system.id,
            system_name=self.system.name,
            subassembly_id=self.id,
            subassembly_name=self.name,
            severity_level=action.level,
            details=action,
        )
        repair_request = self.system.repair_manager.register_request(repair_request)
        self.env.log_action(
            system_id=self.system.id,
            system_name=self.system.name,
            part_id=self.id,
            part_name=self.name,
            system_ol=self.system.operating_level,
            part_ol=self.operating_level,
            agent=self.name,
            action=f"{which} request",
            reason=action.description,
            additional=f"severity level {action.level}",
            request_id=repair_request.request_id,
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
        simpy.events. HOURS_IN_DAY
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
                    # Wait until these events are triggered and back to operational
                    yield self.system.servicing & self.system.cable_failure & self.broken

                    start = self.env.now
                    yield self.env.timeout(hours_to_next)
                    hours_to_next = 0
                    self.trigger_request(maintenance)

                except simpy.Interrupt:
                    if not self.broken.triggered:
                        # The subassembly had to restart the maintenance cycle
                        hours_to_next = 0
                    else:
                        # A different subassembly failed, so subtract the elapsed time
                        hours_to_next -= self.env.now - start  # pylint: disable=E0601

    def run_single_failure(self, failure: Failure) -> Generator:
        """Runs a process to trigger one type of failure repair request throughout the simulation.

        Parameters
        ----------
        failure : Failure
            A failure classification.

        Yields
        -------
        simpy.events. HOURS_IN_DAY
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
                continue
            while hours_to_next > 0:  # type: ignore
                try:
                    yield self.system.servicing & self.system.cable_failure & self.broken
                    start = self.env.now
                    yield self.env.timeout(hours_to_next)
                    hours_to_next = 0
                    self.trigger_request(failure)

                except simpy.Interrupt:
                    if not self.broken.triggered:
                        # The subassembly had to be replaced so reset the timing
                        hours_to_next = 0
                    else:
                        # A different subassembly failed, so subtract the elapsed time
                        hours_to_next -= self.env.now - start  # pylint: disable=E0601
