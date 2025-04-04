"""Provides the Subassembly class."""

from __future__ import annotations

from collections.abc import Generator

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
        """Creates a subassembly object that models various maintenance and failure
        types.

        Parameters
        ----------
        system : wombat.windfarm.System
            The system containing subassembly object(s).
        env : WombatEnvironment
            The simulation environment.
        s_id : str
            A unique identifier for the subassembly within the system.
        subassembly_data : dict
            A dictionary to be passed to ``SubassemblyData`` for creation and
            validation.
        """
        self.env = env
        self.system = system
        self.id = s_id

        subassembly_data = {
            **subassembly_data,
            "system_value": self.system.value,
            "rng": self.env.random_generator,
        }
        self.data = SubassemblyData.from_dict(subassembly_data)
        self.name = self.data.name

        self.operating_level = 1.0
        self.broken = self.env.event()
        self.broken.succeed()  # start the event as inactive

        self.processes = dict(self._create_processes(first=True))

    def _create_processes(self, *, first: bool = False):
        """Creates the processes for each of the failure and maintenance types.

        Parameters
        ----------
        first : bool, optional
            Indicate if this is the initial creation (True), or a recreation following
            a replacement event (False). When True, the maintenance data are updated
            from their original inputs to a simulation-ready format.

        Yields
        ------
        Tuple[Union[str, int], simpy.events.Process]
            Creates a dictionary to keep track of the running processes within the
            subassembly.
        """
        if first:
            for maintenance in self.data.maintenance:
                maintenance._update_event_timing(
                    self.env.start_datetime,
                    self.env.end_datetime,
                    self.env.max_run_time,
                )

        for failure in self.data.failures:
            level = failure.level
            desc = failure.description
            yield (level, desc), self.env.process(self.run_single_failure(failure))

        for maintenance in self.data.maintenance:
            desc = maintenance.description
            yield desc, self.env.process(self.run_single_maintenance(maintenance))

    def recreate_processes(self) -> None:
        """If a turbine is being reset after a tow-to-port repair or replacement, then
        all processes are assumed to be reset to 0, and not pick back up where they left
        off.
        """
        self.processes = dict(self._create_processes())

    def interrupt_processes(
        self, origin: Subassembly | None = None, replacement: str | None = None
    ) -> None:
        """Interrupts all of the running processes within the subassembly except for the
        process associated with failure that triggers the catastrophic failure.

        Parameters
        ----------
        origin : Subassembly
            The subassembly that triggered the request, if the method call is coming
            from a subassembly shutdown event. If provided, and it is the same as the
            current subassembly, then a try/except flow is used to ensure the process
            that initiated the shutdown is not interrupting itself.
        replacement: bool, optional
            If a subassebly `id` is provided, this indicates the interruption is caused
            by its replacement event. Defaults to None.
        """
        cause = "failure"
        if self.id == replacement:
            cause = "replacement"
        if origin is not None and id(origin) == id(self):
            for _, process in self.processes.items():
                try:
                    process.interrupt(cause=cause)
                except RuntimeError:  # Process initiating process can't be interrupted
                    pass
            return

        for _, process in self.processes.items():
            process.interrupt(cause=cause)

    def interrupt_all_subassembly_processes(self) -> None:
        """Thin wrapper for ``system.interrupt_all_subassembly_processes``."""
        self.system.interrupt_all_subassembly_processes(origin=self)

    def trigger_request(self, action: Maintenance | Failure):
        """Triggers the actual repair or maintenance logic for a failure or maintenance
        event, respectively.

        Parameters
        ----------
        action : Maintenance | Failure
            The maintenance or failure event that triggers a ``RepairRequest``.
        """
        which = "maintenance" if isinstance(action, Maintenance) else "repair"
        current_ol = self.operating_level
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
            prior_operating_level=current_ol,
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
        """Runs a process to trigger one type of maintenance request throughout the
        simulation.

        Parameters
        ----------
        maintenance : Maintenance
            A maintenance category.

        Yields
        ------
        simpy.events. HOURS_IN_DAY
            Time between maintenance requests.
        """
        while True:
            hours_to_next, accrue_downtime = maintenance.hours_to_next_event(
                self.env.simulation_time
            )
            while hours_to_next > 0:
                start = -1  # Ensure an interruption before processing is caught
                try:
                    # Downtime doesn't accrue for date-based maintenance
                    if not accrue_downtime:
                        yield (
                            self.system.servicing
                            & self.system.cable_failure
                            & self.broken
                        )
                    start = self.env.now
                    yield self.env.timeout(hours_to_next)
                    hours_to_next = 0
                    self.trigger_request(maintenance)

                except simpy.Interrupt as i:
                    if i.cause == "replacement":
                        return
                    hours_to_next -= 0 if start == -1 else self.env.now - start

    def run_single_failure(self, failure: Failure) -> Generator:
        """Runs a process to trigger one type of failure repair request throughout the
        simulation.

        Parameters
        ----------
        failure : Failure
            A failure classification.

        Yields
        ------
        simpy.events. HOURS_IN_DAY
            Time between failure events that need to request a repair.
        """
        while True:
            hours_to_next = failure.hours_to_next_event()
            if hours_to_next is None:
                remainder = self.env.max_run_time - self.env.now
                try:
                    yield self.env.timeout(remainder)
                except simpy.Interrupt:
                    remainder -= self.env.now
                continue
            else:
                while hours_to_next > 0:  # type: ignore
                    start = -1  # Ensure an interruption before processing is caught
                    try:
                        yield (
                            self.system.servicing
                            & self.system.cable_failure
                            & self.broken
                        )
                        start = self.env.now
                        yield self.env.timeout(hours_to_next)
                        hours_to_next = 0
                        self.trigger_request(failure)

                    except simpy.Interrupt as i:
                        if i.cause == "replacement":
                            return
                        hours_to_next -= 0 if start == -1 else self.env.now - start
