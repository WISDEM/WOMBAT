## [UNRELEASES]
- Testing now included!
- `pathlib.Path` is used in place of `os` throughout for easier to read file manuevering.
- `attrs.define` and `attrs.field` have been adopted in place of `attr.s` and `attr.ib`, respectively.
- typing style is updated for future python type annotations as, e.g., `Union[List[str], str]` ->
  `list[str] | str`.
- Unused imports have been further cleaned up.
- Better caching of commonly computed values for faster simulations

- `wombat.windfarm.system.System` no longer requires every subassembly model to be
  defined for increased flexibility and ease of definition of turbine simulations

- `wombat.utilities.hours_until_future_hour()` correctly moves to the expected hour in the future,
  regardless of the number of days that are added.
- The IEA Task 26 validation cabling vessel was renamed from "cabling.yaml" to "cabling_scheduled.yaml"
  for consistent naming conventions.
- Improved error messages throughout.

- `wombat.core.simulation_api.Simulation` no longer accepts `name` as an input, and reads it
  directly from the configuration.
- `wombat.core.simulation_api.Simulation.run` now has a `create_metrics` flag that defaults to `True`
  to indicate if the metrics suite should be created, so that `False` can be used if numerous
  analyses will be run and post-processed after the fact.

- `wombat.core.environment.WombatEnvironment.workday_start` and
  `wombat.core.environment.WombatEnvironment.workday_end` are better validated for edge cases.
- `wombat.core.environment.WombatEnvironment._weather_setup` validates the start and end points
  against the provided weather profile and against each other to ensure they are valid boundaries.
- `wombat.core.environment.WombatEnvironment.cleanup_log_files` will only try to delete files that actually
  exist to avoid unexpected simulation failures at the last step.
- `wombat.core.environment.WombatEnvironment.date_ix` accepts `datetime.datetime` and `datetime.date` inputs to
  avoid unnecesary errors or manipulations in a simulation.
- `wombat.core.environment.WombatEnvironment.log_action` now only accepts numeric inputs for
  `system_ol` and `part_ol`.
- `wombat.core.environment.WombatEnvironment.weather_forecast` now rounds the starting time down to
  include the current hour, which fixes a bug in some edge cases where a crew transfer at the end
  of the shift gets stuck and has to wait until the next shift despite ample time to transfer.
- `wombat.core.environment.WombatEnvironment.is_workshift` correctly accounts for around the clock
  shifts.

- `wombat.core.repair_management.RepairManager.submit_request` will only attempt to deploy servicing
  equipment for strategies that have been initialized to reduce unnecessarily running code.
- `wombat.core.repair_management.RepairManager.get_request_by_turbine` was renamed to
  `wombat.core.repair_management.RepairManager.get_request_by_system` to explicitly include substations.
  to remove repeated calculations and checks.
- `wombat.core.repair_management.RepairManager.get_request_by_severity` request checking was optimized
  to remove repeated calculations and checks.
- `wombat.core.repair_management.RepairManager.purge_subassembly_requests` no longer prematurely exits
  the loop to purge unnecessary requests from the logs in case of a replacement being required.
  Additionally, an `exclude` parameter has been added for niche use cases where all but a couple
  of requests will need to be removed.

- `wombat.core.service_equipment.ServiceEquipment` has more attributes for tracking its location
  throughout a simulation.
- `wombat.core.service_equipment.ServiceEquipment.find_uninterrupted_weather_window` now rounds
  float inputs up to the nearest whole number to ensure the proper length window is retreived.
- `wombat.core.service_equipment.ServiceEquipment.find_interrupted_weather_window` uses `math.ceil`
  in place of `np.ceil().astype(int)` for strictly python data types in computation.
- `wombat.core.service_equipment.ServiceEquipment.travel` consistently resets the location attributes
  when moving between site and port to fix bugs in the end location.
- `wombat.core.service_equipment.ServiceEquipment.crew_transfer` now indicates when the crew is
  actually being transferred, and updates the equipment's location post-transfer. Additionally, the
  travel time in delays is now accounted for, so the actually elapsed time is accounted for in
  processing weather delays. A bug in processing weather delays is fixed and uses the correct `SimPy`
  timeout call so that the simulation does not get stuck due to improper calling.
- `wombat.core.service_equipment.ServiceEquipment.process_repair` adds a safety window to account
  for the crew having enough time to safely transfer back to the equipment. This logic is likely
  not entirely fixed, but is sufficiently fixed for the time being.
- `wombat.core.service_equipment.ServiceEquipment.process_repair`:
  - fixes a bug in the hours to process calculation by reconfiguring the control flow for both
    simplicity and accuracy.
  - Additionally, the equipment will travel back to port if the workshift is over, so the equipment
    does not skip steps the repair logic.
  - Weather delays will also not be processed if the actual repair is finished, and will be moved to
    the crew transfer step.
  - The hours required vs hours available is reset for equipment that have around the clock shifts,
    so the timing is not arbitrarily capped, which also reduces the number of steps in the simulation.
  - The workshift indicator check now accounts for the equipment's specific workshift settings, and
    not the environment's to ensure the proper operating parameters are used.

- `wombat.core.service_equipment.ServiceEquipment.run_scheduled` now sets the `onsite` variable
  for equipment that are always on site for more consistent handling of the onsite attribute.
- `wombat.core.service_equipment.ServiceEquipment.wait_until_next_operational_period` now resets
  the equipment's location attributes so its location can't incorrectly persist from the last step.

- `wombat.windfarm.Windfarm` operations logging message generation routines were optimized.
- `wombat.windfarm.Windfarm.system` was created in place of `wombat.windfarm.Windfarm.node_system`
  as a convenience method for grabbing the `System` object from the windfarm graph layout.
- `wombat.windfarm.Windfarm.cable` was created to replicate the `system` method for cable objects.
- `wombat.windfarm.Windfarm`, `wombat.windfarm.system.cable.Cable`, and
  `wombat.windfarm.system.system.System` all have an operating level property that is the current
  operating level and one that does not account for if a system is being serviced, and therefore
  registered as non-operational. This fixes a bug in the dispatching of unscheduled service equipment
  when a repair request is submitted while a repair is occuring that drops the operating level below
  the strategy threshold. In the dispatching check we now are able to consider the operating level
  before a system was shut down for repairs and avoid improper dispatching.

- `wombat.windfarm.system.cable.Cable` now takes `end_node` in place of `upstream_nodes` to simplify
  the initialization process and be more explicit about the starting and ending points on the cable.
- `wombat.windfarm.system.cable.Cable.interrupt_subassembly_processes` is now `interrupt_processes`
  and only interrupts the cable's own maintenance and failure simulation processes.
- `wombat.windfarm.system.cable.Cable.interrupt_all_subassembly_processes` has been removed
  in place of calling each system's own interruption method, which
- `wombat.windfarm.system.cable.Cable` logging now properly records itself as the target of repais
  and maintenance tasks instead of its starting node.

- `wombat.windfarm.system.subassembly.Subassembly.interrupt_subassembly_processes` is now
  `interrupt_processes` and only interrupts the cable's own maintenance and failure simulation processes.
- `wombat.windfarm.system.system.System.interrupt_all_subassembly_processes` was added so that
  subassemblies are no longer directly interacting with each other, and the interruptions are caused
  by the system's themselves.
- `wombat.windfarm.system.system.System.log_action` no returns the `Subassembly.id` instead of the
  `__repr__` for the object to create a legible log.


## 0.3.5 (2021-December-9)

- Downtime-based and requests-based unscheduled maintenance models have been added to
  servicing equipment.
- `Windfarm.current_availability` is able to account for multiple substations and is now
  an accurate calculation of the operating level across the windfarm.
- `Simulation.from_inputs()` is replaced by `Simulation.from_config()` to be more
  straightfoward for users.
- `ServiceCrew` has been added for expansion to multiple crews.
- Servicing equipment now have separate `travel` and `crew_transfer`, `weather_delay`,
  and `repair` functions to more realistically simulate the separate processes involved.
- The windfarm now has a distance matrix to calculate the distance between any two
  systems oniste, for instance, substation to cable, cable to turbine, etc.
- Bug fix in the task completion rate metric.
- `WombatEnvironment.log_action()` enforces the use of keyword arguments with only 3
  required inputs.
- `WombatEnvironment.weather_forecast()` outputs now include the datetime index.
- If no PySAM settings are provided, the `financial_model` will be set to `None` with a
  `NotImplementedError` being raised for any attempted usage of the PySAM-powered
  functionality.
- Fix bugs in `Maintenance` and `Failure` initializations.
- Fix bugs in operating level accounting from recent updates.
- Update the unicode typing in all docstrings.
- Update the documentation to account for all the recent changes, including adding a new
  demonstration notebook for the 3 new strategies.

## 0.3.1 (2021-March-2)

- Updated the the `simulation/` folder to be a single-level `core/` directory.
- Updates the order of the keyword arguments for WombatEnvironment.log_action(), and makes all arguments keyword-only arguments.
