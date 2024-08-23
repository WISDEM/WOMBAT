""" "Defines the Mooring class and mooring simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Generator

import numpy as np
import simpy

from wombat.core import (
    Failure,
    Maintenance,
    RepairRequest,
    SubassemblyData,
    WombatEnvironment,
)


class Mooring:
    """The mooring system/asset class for a wind farm simulation.

    Attributes
    ----------
    windfarm : wombat.windfarm.Windfarm
        The Windfarm object.
    env : WombatEnvironment
        The simulation environment.
    anchor_id : str
        The unique identifier for the anchor.
    connected_systems : List[str]
        Identifiers for systems (turbines, substations) connected to the anchor.
    mooring_data : dict
        The dictionary defining the mooring segment.
    """

    def __init__(
        self,
        windfarm,
        env: WombatEnvironment,
        anchor_id: str,
        mooringline_ids: list[str],
        connected_systems: list[str],
        anchor_data: dict,
        mooringlines_data: dict,
        #        mooring_data: dict,
        name: str | None = None,
    ) -> None:
        """Initializes the Mooring class."""
        self.env = env
        self.windfarm = windfarm
        self.anchor_id = anchor_id
        self.mooringline_ids = mooringline_ids
        self.connected_systems = [
            sys_id.strip() for sys_id in connected_systems
        ]  # connected_systems
        self.id = f"{anchor_id}"  # _{','.join(mooringline_ids)}" if mooringline_ids else anchor_id
        self.name = name if name is not None else self.id

        # Handle multiple connected systems
        self.systems = (
            [self.windfarm.system(sys_id.strip()) for sys_id in connected_systems]
            if connected_systems
            else []
        )

        # Set mooring_id based on context
        if mooringline_ids:
            # It's a mooring line
            self.mooring_id = f"M-{','.join(mooringline_ids)}"
        else:
            # It's an anchor
            self.mooring_id = f"A-{anchor_id}"

        # Assuming the first system's value is representative
        if self.systems:
            system_value = self.systems[0].value
        else:
            system_value = 0  # Default or error value

        # Merge anchor and mooring line data into mooring_data
        self.mooring_data = {
            "name": anchor_data.get("name", "Default Anchor Name"),
            "maintenance": anchor_data.get("maintenance", []),
            "failures": anchor_data.get("failures", {}),
        }

        # Integrate mooring line data
        for mooringline_id, data in mooringlines_data.items():
            self.mooring_data["maintenance"].extend(data.get("maintenance", []))
            self.mooring_data["failures"].update(data.get("failures", {}))

        # Now, add system_value and random number generator to the dictionary for SubassemblyData creation
        self.data = SubassemblyData.from_dict(
            {
                **self.mooring_data,
                "system_value": system_value,  # Could be adjusted based on context
                "rng": self.env.random_generator,
            }
        )

        self.name = self.data.name

        # print(f"Preparing to create SubassemblyData with mooring_data: {self.mooring_data}")

        self.operating_level = 1.0
        self.servicing = self.env.event()
        self.servicing_queue = self.env.event()

        self.mooring_failure = self.env.event()

        # self.anchor_failure = self.env.event()
        # self.mooringline_failure = self.env.event()
        self.broken = self.env.event()

        # Ensure events start as processed and inactive
        self.servicing.succeed()
        self.servicing_queue.succeed()

        self.mooring_failure.succeed()

        self.broken.succeed()

        # need to get the time scale of a distribution like this
        self.processes = dict(self._create_processes())

        self.maintenance = Maintenance

    def _init_subassembly_data(self):
        combined_maintenance = self.mooring_data["anchor"]["maintenance"][:]
        combined_failures = self.mooring_data["anchor"]["failures"].copy()
        names = [self.mooring_data["anchor"]["name"]] + [
            data["name"] for data in self.mooring_data["mooringlines"].values()
        ]

        for mooringline_data in self.mooring_data["mooringlines"].values():
            combined_maintenance.extend(mooringline_data["maintenance"])
            combined_failures.update(
                mooringline_data["failures"]
            )  # If 'failures' is a dict

        self.data = SubassemblyData.from_dict(
            {
                "name": ", ".join(names),
                "maintenance": combined_maintenance,
                "failures": combined_failures,
                "system_value": self.systems[0].value if self.systems else 0,
                "rng": self.env.random_generator,
            }
        )

    def finish_setup(self) -> None:
        """Finalizes the setup of mooring connections after initial attributes are defined."""
        mooring_details = self.windfarm.mooring_map.get(self.id)
        if not mooring_details:
            raise ValueError(
                f"Details for mooring ID {self.id} are missing in the mooring map."
            )

        # Extract and store the connected systems as System objects
        self.connected_systems_details = [
            self.windfarm.system(sys_id)
            for sys_id in mooring_details.get("connected_systems", [])
        ]

        # Handle different connection types if applicable
        if hasattr(self, "connection_type"):
            self.handle_connection_types(mooring_details)

        # Store additional mooring line details if necessary
        self.mooring_lines = mooring_details.get("mooring_lines", [])

    def _create_processes(self):
        """Creates the processes for each of the failure and maintenance types.

        Yields
        ------
        Tuple[Union[str, int], simpy.events.Process]
            Creates a dictionary to keep track of the running processes within the
            subassembly.
        """
        for failure in self.data.failures:
            desc = failure.description
            yield desc, self.env.process(self.run_single_failure(failure))

        for i, maintenance in enumerate(self.data.maintenance):
            desc = maintenance.description
            yield desc, self.env.process(self.run_single_maintenance(maintenance))

    def recreate_processes(self) -> None:
        """If an anchor or mooringline is being reset after a replacement, then all processes are
        assumed to be reset to 0, and not pick back up where they left off.
        """
        self.processes = dict(self._create_processes())

    def interrupt_processes(self, replacement: str | None = None) -> None:
        """Interrupts all of the running processes within the subassembly except for the
        process associated with failure that triggers the catastrophic failure.

        Parameters
        ----------
        subassembly : Subassembly
            The subassembly that should have all processes interrupted.
        replacement: bool, optional
            If a subassebly `id` is provided, this indicates the interruption is caused
            by its replacement event. Defaults to None.
        """
        cause = "failure"
        if self.id == replacement:
            cause = "replacement"

        for _, process in self.processes.items():
            try:
                process.interrupt(cause=cause)
            except RuntimeError:
                # This error occurs for the process halting all other processes.
                pass

    def interrupt_all_subassembly_processes(
        self, replacement: str | None = None
    ) -> None:
        """Thin wrapper for ``interrupt_processes`` for consistent usage with system.

        Parameters
        ----------
        replacement: bool, optional
            If a subassebly `id` is provided, this indicates the interruption is caused
            by its replacement event. Defaults to None.
        """
        self.interrupt_processes(replacement=replacement)

    def log_event(self, action, detail):
        """Log detailed actions for debugging."""
        self.env.log_action(
            agent=self.id,
            action=action,
            detail=detail,
            timestamp=self.env.now,
            connected_systems=self.connected_systems,
            mooring_id=self.id,
        )

    def stop_all_mooring_processes(self, failure: Failure | Maintenance) -> None:
        """Stops all turbines from producing power by creating an event for each System.mooring_failure
        and handles both direct and indirect impacts based on shared mooring connections.
        """
        logging_context = {
            "agent": self.mooring_id,
            # "action": "stop processes",
            # "reason": "Connectivity failure",
            "request_id": failure.request_id,  # "some_unique_identifier",
        }

        # Log the start of the process
        self.env.log_action(
            system_id=self.mooring_id,
            system_name="Mooring System",
            action="Process Stop Initiated",
            reason="Connectivity failure triggered",
            additional="Starting shutdown of all connected systems",
            **logging_context,
        )

        # print(f"Process Stop Initiated for mooring_id: {self.mooring_id}")

        # Function to safely interrupt processes within subassemblies
        def safe_interrupt(system):
            for subassembly in system.subassemblies:
                for process in subassembly.processes.values():
                    if (
                        not process.triggered
                    ):  # Check if the process has not been triggered
                        try:
                            process.interrupt(cause="mooring failure")
                        except RuntimeError as e:
                            self.env.log_action(
                                system_id=system.id,
                                system_name=system.name,
                                action="Interrupt Failed",
                                reason=str(e),
                                **logging_context,
                            )

        #                 print(f"Interrupt Failed for system_id: {system.id}, reason: {str(e)}")

        # Directly affected systems
        directly_affected_systems = set()
        for system_id in self.connected_systems:
            if self.windfarm.is_anchor(system_id):
                # Skip any shutdown logic for anchors since they don't influence power production
                continue
            else:
                system = self.windfarm.system(system_id)
                if (
                    system.system_type == "turbine"
                    or system.system_type == "substation"
                ):
                    system.mooring_failure = self.env.event()
                    safe_interrupt(system)
                    # system.interrupt_all_subassembly_processes()
                    directly_affected_systems.add(system_id)
                    self.env.log_action(
                        system_id=system_id,
                        system_name=system.name,
                        action="Direct Impact - Failure Triggered",
                        reason="Directly connected to failed mooring",
                        system_ol=0,
                        part_ol=np.nan,
                        **logging_context,
                    )

        #          print(f"Direct Impact - Failure Triggered for system_id: {system_id}")

        # print(f"Directly affected systems: {directly_affected_systems}")

        # Indirectly affected systems due to shared mooring connections
        indirectly_affected_systems = set()
        all_checked_systems = set(
            directly_affected_systems
        )  # Keep track of all checked systems
        anchors_to_check = set()

        # First, gather all anchors connected to directly affected systems
        for system_id in directly_affected_systems:
            anchors_to_check.update(
                self.windfarm.mooring_map.get_anchor_connections(system_id)
            )

        # print(f"Initial anchors to check: {anchors_to_check}")

        # Now gather all systems connected to these anchors
        for anchor_id in anchors_to_check:
            connected_systems = self.windfarm.mooring_map.get_connected_turbines(
                anchor_id
            ) + self.windfarm.mooring_map.get_connected_substations(anchor_id)
            #   print(f"Connected systems for anchor_id {anchor_id}: {connected_systems}")
            for connected_system_id in connected_systems:
                if connected_system_id not in directly_affected_systems:
                    indirectly_affected_systems.add(connected_system_id)
                    all_checked_systems.add(connected_system_id)

        # Print final indirectly affected systems
        # print(f"Indirectly affected systems: {indirectly_affected_systems}")

        for system_id in indirectly_affected_systems:
            if self.windfarm.is_anchor(system_id):
                # Skip any shutdown logic for anchors since they don't influence power production
                continue
            else:
                system = self.windfarm.system(system_id)
                if (
                    system.system_type == "turbine"
                    or system.system_type == "substation"
                ):
                    system.mooring_failure = self.env.event()
                    safe_interrupt(system)
                    # system.interrupt_all_subassembly_processes()
                    self.env.log_action(
                        system_id=system_id,
                        system_name=system.name,
                        action="Indirect Impact - Failure Triggered",
                        reason="Affected by shared mooring connectivity",
                        system_ol=0,
                        part_ol=np.nan,
                        **logging_context,
                    )

            #        print(f"Indirect Impact - Failure Triggered for system_id: {system_id}")

        # Log the completion of the process
        self.env.log_action(
            system_id="N/A",
            system_name="Global Mooring System",
            action="Process Stop Completed",
            reason="All relevant systems have been processed",
            additional="Shutdown complete",
            **logging_context,
        )

        # print("Process Stop Completed for Global Mooring System")

    def trigger_request(self, action: Maintenance | Failure):
        """Triggers the actual repair or maintenance logic for a failure or maintenance
        event, respectively.

        Parameters
        ----------
        action : Maintenance | Failure
            The maintenance or failure event that triggers a `RepairRequest`.
        """
        which = "maintenance" if isinstance(action, Maintenance) else "repair"
        self.operating_level *= 1 - action.operation_reduction

        # Automatically submit a repair request
        repair_request = RepairRequest(
            system_id=self.id,
            system_name=self.name,
            subassembly_id=self.id,
            subassembly_name=self.name,
            severity_level=action.level,
            details=action,
            mooring=True,
            connected_systems=self.connected_systems,
            # part_ol=self.operating_level,
        )
        repair_request = self.windfarm.repair_manager.register_request(repair_request)
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

        if action.operation_reduction == 0:
            self.broken = self.env.event()
            self.interrupt_all_subassembly_processes()
            self.stop_all_mooring_processes(action)

        # Remove previously submitted requests as a replacement is required
        if action.replacement:
            _ = self.windfarm.repair_manager.purge_subassembly_requests(
                self.id, self.id, exclude=[repair_request.request_id]
            )
        self.windfarm.repair_manager.submit_request(repair_request)

    def run_single_maintenance(self, maintenance: Maintenance) -> Generator:
        """Runs a process to trigger one type of maintenance request throughout the
        simulation.

        Parameters
        ----------
        maintenance : Maintenance
            A maintenance category.

        Yields
        ------
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
            else:
                while hours_to_next > 0:
                    start = -1  # Ensure an interruption before processing is caught
                    try:
                        # If the replacement has not been completed, then wait
                        yield self.servicing & self.mooring_failure & self.broken

                        start = self.env.now
                        yield self.env.timeout(hours_to_next)
                        hours_to_next = 0
                        self.trigger_request(maintenance)
                    except simpy.Interrupt:
                        if not self.broken.triggered:
                            # The subassembly had to restart the maintenance cycle
                            hours_to_next = 0
                        else:
                            # A different process failed, so subtract the elapsed time
                            # only if it had started to be processed
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

            else:
                if TYPE_CHECKING:
                    assert isinstance(hours_to_next, (int, float))  # mypy helper
                while hours_to_next > 0:  # type: ignore
                    start = -1  # Ensure an interruption before processing is caught
                    try:
                        yield self.servicing & self.mooring_failure & self.broken

                        start = self.env.now
                        yield self.env.timeout(hours_to_next)
                        hours_to_next = 0
                        self.trigger_request(failure)
                    except simpy.Interrupt as i:
                        if i.cause == "replacement":
                            return
                        hours_to_next -= 0 if start == -1 else self.env.now - start
                    # except simpy.Interrupt:
                    #     if not self.broken.triggered:
                    #         # Restart after fixing
                    #         hours_to_next = 0
                    #     else:
                    #         # A different process failed, so subtract the elapsed time
                    #         # only if it had started to be processed
                    #         hours_to_next -= 0 if start == -1 else self.env.now - start
