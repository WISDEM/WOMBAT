"""Provides the Subassembly class"""


import simpy  # type: ignore
from typing import Generator  # type: ignore

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


class Subassembly:
    """A major system composes the turbine or substation objects."""

    def __init__(
        self,
        turbine,
        env: WombatEnvironment,
        s_id: str,
        subassembly_data: dict,
    ) -> None:
        """Creates a subassembly object that models various maintenance and failure types.

        Parameters
        ----------
        turbine : wombat.windfarm.System
            The turbine containing subassembly object(s).
        env : WombatEnvironment
            The simulation environment.
        s_id : str
            A unique identifier for the subassembly within the turbine.
        subassembly_data : dict
            A dictionary to be passed to ``SubassemblyData`` for creation and validation.
        """

        self.env = env
        self.turbine = turbine
        self.id = s_id

        subassembly_data = {**subassembly_data, "system_value": self.turbine.value}
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

    @staticmethod
    def interrupt_subassembly_processes(subassembly) -> None:
        """Interrupts all of the running processes within the subassembly except for the
        process associated with failure that triggers the catastrophic failure.

        Parameters
        ----------
        subassembly : Subassembly
            The subassembly that should have all processes interrupted.
        """
        for _, process in subassembly.processes.items():
            try:
                process.interrupt()
            except RuntimeError:
                # This error occurs for the process halting all other processes.
                pass

    def interrupt_all_subassembly_processes(self) -> None:
        """Interrupts the running processes in all of the subassemblies within ``turbine``.

        Parameters
        ----------
        failure : Failure
            The failure that triggers the turbine shutting down.
        """
        for subassembly in self.turbine.subassemblies:
            self.interrupt_subassembly_processes(subassembly)

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
                    self.env.log_action(
                        system_id=self.turbine.id,
                        system_name=self.turbine.name,
                        part_id=self.id,
                        part_name=self.name,
                        system_ol=self.turbine.operating_level,
                        part_ol=self.operating_level,
                        agent=self.name,
                        action="none",
                        reason=f"{self.name} is not modeled",
                        additional="no maintenance will be modeled for select subassembly",
                        duration=remainder,
                    )
                    yield self.env.timeout(remainder)
                except simpy.Interrupt:
                    remainder -= self.env.now

            while hours_to_next > 0:
                try:
                    # If the replacement has not been completed, then wait another minute
                    if self.turbine.operating_level == 0:
                        yield self.env.timeout(TIMEOUT)
                        continue

                    start = self.env.now
                    yield self.env.timeout(hours_to_next)
                    hours_to_next = 0

                    # Automatically submit a repair request
                    repair_request = RepairRequest(
                        system_id=self.turbine.id,
                        system_name=self.turbine.name,
                        subassembly_id=self.id,
                        subassembly_name=self.name,
                        severity_level=0,
                        details=maintenance,
                    )
                    repair_request = self.turbine.repair_manager.submit_request(
                        repair_request
                    )
                    self.env.log_action(
                        system_id=self.turbine.id,
                        system_name=self.turbine.name,
                        part_id=self.id,
                        part_name=self.name,
                        system_ol=self.turbine.operating_level,
                        part_ol=self.operating_level,
                        agent=self.name,
                        action="maintenance request",
                        reason=maintenance.description,
                        additional="request",
                        request_id=repair_request.request_id,
                    )

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
                    self.env.log_action(
                        system_id=self.turbine.id,
                        system_name=self.turbine.name,
                        part_id=self.id,
                        part_name=self.name,
                        system_ol=self.turbine.operating_level,
                        part_ol=self.operating_level,
                        agent=self.name,
                        action="none",
                        reason=f"{self.name} is not modeled",
                        additional="no failures will be modeled for select subassembly",
                        duration=remainder,
                    )
                    yield self.env.timeout(remainder)
                except simpy.Interrupt:
                    remainder -= self.env.now

            else:
                while hours_to_next > 0:  # type: ignore
                    try:
                        if self.turbine.operating_level == 0:
                            yield self.env.timeout(TIMEOUT)
                            continue

                        start = self.env.now
                        yield self.env.timeout(hours_to_next)
                        hours_to_next = 0
                        self.operating_level *= 1 - failure.operation_reduction
                        if failure.operation_reduction == 1:
                            self.broken = True
                            self.interrupt_all_subassembly_processes()

                            # Remove previously submitted requests as a replacement is required
                            _ = self.turbine.repair_manager.purge_subassembly_requests(
                                self.turbine.id, self.id
                            )

                        # Automatically submit a repair request
                        repair_request = RepairRequest(
                            self.turbine.id,
                            self.turbine.name,
                            self.id,
                            self.name,
                            failure.level,
                            failure,
                        )
                        repair_request = self.turbine.repair_manager.submit_request(
                            repair_request
                        )

                        self.env.log_action(
                            system_id=self.turbine.id,
                            system_name=self.turbine.name,
                            part_id=self.id,
                            part_name=self.name,
                            system_ol=self.turbine.operating_level,
                            part_ol=self.operating_level,
                            agent=self.name,
                            action="repair request",
                            reason=failure.description,
                            additional=f"severity level {failure.level}",
                            request_id=repair_request.request_id,
                        )

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
