"""Provides the Subassembly class"""


from typing import Generator  # type: ignore

import simpy  # type: ignore

from wombat.core import (
    Failure,
    Maintenance,
    RepairRequest,
    SubassemblyData,
    WombatEnvironment,
)
from wombat.utilities import HOURS_IN_DAY


# TODO: Need a better method for checking if a repair has been made to bring the
# subassembly back online


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

        self.broken = False
        self.operating_level = 1.0

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
                    # If the replacement has not been completed, then wait another minute
                    if self.system.operating_level == 0 or self.system.servicing:
                        yield self.env.timeout(HOURS_IN_DAY)
                        continue

                    start = self.env.now
                    yield self.env.timeout(hours_to_next)
                    hours_to_next = 0

                    # Automatically submit a repair request
                    # NOTE: mypy is not caught up with attrs yet :(
                    repair_request = RepairRequest(  # type: ignore
                        system_id=self.system.id,
                        system_name=self.system.name,
                        subassembly_id=self.id,
                        subassembly_name=self.name,
                        severity_level=0,
                        details=maintenance,
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

            else:
                while hours_to_next > 0:  # type: ignore
                    try:
                        if self.system.operating_level == 0 or self.system.servicing:
                            yield self.env.timeout(HOURS_IN_DAY)
                            continue

                        start = self.env.now
                        yield self.env.timeout(hours_to_next)
                        hours_to_next = 0
                        self.operating_level *= 1 - failure.operation_reduction
                        if failure.operation_reduction == 1:
                            self.broken = True
                            self.interrupt_all_subassembly_processes()

                            # Remove previously submitted requests as a replacement is required
                            _ = self.system.repair_manager.purge_subassembly_requests(
                                self.system.id, self.id
                            )

                        # Automatically submit a repair request
                        # NOTE: mypy is not caught up with attrs yet :(
                        repair_request = RepairRequest(  # type: ignore
                            self.system.id,
                            self.system.name,
                            self.id,
                            self.name,
                            failure.level,
                            failure,
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
