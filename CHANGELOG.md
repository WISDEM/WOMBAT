## Unreleased (TBD)
- Replace Flake8 and Pylint in the pre-commit workflow with ruff, and fix/ignore the resulting errors as appropriate
- Bug fixes:
  - Maintenance and failure simulation process interruptions were occuring prior to starting the process timing, and causing simulation failures.
  - Duplicated parameters were being processed in `WombatEnvironment.log_action` stemming from improper handling of varying parameters in some of the more complex control flow logic in *in situ* repairs.
  - Another edge case of negative delays during crew transfers where there is insufficient time remaining in the shift after account for weather, so the method was called recursively, but not exiting the original loop.
  - `Port` managment of *in situ* and tow-to-port capable tugboats wasn't properly accounting for tugboas of varying capabilities, and assuming all tugboats could do both. The vessel management and repair processing were out of sync causing duplicated turbine servicing/towing.
  - `ServiceEquipment.weather_delay()` no longer silently processes a second weather delay.

## v0.6.2 (3 February 2023)
- Warnings from Pandas `.groupby()` calls have been silenced by shifting the column filtering to before the groupby method call.
- Fixed the towing to port logging message, and subsequent `Metrics.number_of_tows()` search criteria to correctly calculate the number of tows in each direction.
- Fixes a bug in the crew transfer logic where the travel time back to port is longer than the weather delay itself, and a negative time delay is incorrectly attempted to be processed.

## v0.6.1 (31 January 2023)
- A hot fix for the code comparison data not being found when installing from PyPI, which was caused by missing `__init__.py` files in the reconfigured library structure.
## v0.6.0 (10 January 2023)

In v0.6, due to a series of bug fixes, logic improvements, and feature additions (all described below), users can expect mild to significant shifts in their results. These shifts, while startling, move WOMBAT towards more accurate results now that servicing equipment can't be dispatched multiple times in a row, statuses can't be reset without failure. Additionally, in our validation cases this has led to an average speedup of 71%, or 3.5x faster run times.
### New and Updated Features
- Environmental and logistics considerations via prohibited operations periods and speed reduction periods.
  - All periods can be set for each servicing equipment, or across the board when set at the environment level
  - For ports, the same logic applies where the environment can set the port variables and the port can set its associated tugboats' variables
  - When a setting is defined in multiple locations, the more conservative restriction is applied
  - Variables `non_operational_start`, `non_operational_end` create the annualized period where operations are prohibited, resulting in the creation of the array `non_operational_dates` and set `non_operational_dates_set`.
  - Variables `reduced_speed_start`, `reduced_speed_end` create the annualized period where the maximum speed, `reduced_speed`, for all operations is imposed, resulting in the creation of the array `reduced_speed_dates` and set `reduced_speed_dates_set`.
- Export Cables
  - Models the export cabling system as a single cable between the substation and the interconnection point or as a connection between multiple subsations. for multiple connected substations, the model assumes they are independent systems.
  - Adds support for "type" in the wind farm layout CSV file, which should be filled with either "substation" or "turbine". This column supports multi-substation farms so that accurate plots and connections can be made. For instance, multiple connected substatation, can now be accurately rendered and modeled in the farm.
  - Adds support for "upstream_cable_name" in the wind farm layout CSV file, to provide an individualized name to a cable in place of using the name field in the cable settings file for all similar cables.
- New library structure that mirrors ORBIT (see below diagram)! In v0.7, the orignal library structure will be officially deprecated in favor of the below, and during the v0.6 lifecycle a warning will be raised to instruct users where to place and structure folders going forward.
  ```
  <library>
    ├── project
      ├── config       <- Project-level configuration files
      ├── port         <- Port configuration files
      ├── plant        <- Wind farm layout files
    ├── cables         <- Export and Array cable configuration files
    ├── substructures  <- Substructure configuration files
    ├── turbines       <- Turbine configuration and power curve files
    ├── vessels        <- Land-based and offshore servicing equipment configuration files
    ├── weather        <- Weather profiles
    ├── results        <- The analysis log files and any saved output data
  ```
  - Adds `create_library_structure` to `wombat.core.library` so that users can create the appropriate folder structure for a new project without having to manually create each and every folder.
- ``Maintenance.operation_reduction`` has been enabled to better resemble the effect of unaddressed maintenance.
- ``Failure`` now has a boolean flag for ``replacement`` to indicate if a replacement is required, which allows for operational shutdowns without necessitating a full replacement of the subassembly. Additionally, this flag enables a replacement event for non-shutdown failures.

### General Improvements
- Bump Python versioning requirements to 3.8+
- Add PyArrow dependency for fasting save/load processes for CSV reading and writing
- Convert boolean operational statuses for `System`, `Subassembly`, `Cable`, and `ServiceEquipment` to SimPy events for more efficient processing and accurate delays for restarting
- Fix numerous bugs in the repair logic introduced by the use of boolean checks and status switches, which also improve simulation performance. These issues were primarily caused by erroneously resetting the status, but with the new event setting and `.succeed()` logic to clear an operation, the previously incorrect resetting is much harder to do.
- Continue to improve the performance of low-level simulation operations to realize further improvements in memory usage and simulation performance
- Logging is now based on directly writing to CSV in place of the `logging`-based infrastructure
  - All underlying infrastructure withing the simulation have also been updated to accommodate the different file types, which allows for more direct interation at the end of the simulation and enables PyArrow csv read/write
  - This enables:
    - deprecation warnings to be passed directly to the terminal/notebook without interfering with the file handling
    - reasonable speedups to simulation times by not having additional overhead from the logging and buffering
- `Metrics.process_times()` has been converted to efficient pandas code for fast computation.
- The Metrics example usage documentation page has been rewritten and reformatted to provide more helpful information to users
- `Metrics` methods now accurately account for the effect of substation oeprating reductions on upstream turbines so that each substation's subgraph multiplies the turbine operating capacity by the subastation operating capacity.
- Fixes a bug in the `RepairManager.get_next_highest_severity_request()` where requests aren't processed in first in, first out order with a serverity level priority.
- Remove duplicated logic in the the ``Subassembly`` and ``Cable`` maintenance and failure modeling ensuring that repetitive logic is identical between scenarios.

## 0.5.1 (22 July 2022)
- Updates to use the most recent pandas API/recommendations, which fixes numerous warnings in the `Metrics` class
- Fixes inconsistency in returning floats vs `DataFrame`s in the  `Metrics` class
- Updates the examples to work with the returned `DataFrame` values, and adds warnings about the change in usage
- Updates the documenation configuration to be compatible with the latest sphinx book theme API usage
- Adds a potential fix to an occasional issue where the logging files can't be deleted using `WombatEnvironment.cleanup_log_files()` because the file is still considered to be in use

## 0.5.0 (30 June 2022)
- Adds capabilities: "TOW" and "AHV" for tugboat/towing equipment and anchor-handling vessels
- Adds a tow-to-port strategy that is activated for repairs with the "TOW" capability in the servicing
- Adds a `Port` class to handle the tow-to-port class and tugboat-based service requests
- Allows for any name to define the subassemblies of a turbine or substation to enable users to use the naming conventions they are most familiar with or most meaningful for their own work
- Minor bug fixes in the `Metrics` class to improve stability, improve code reuse, and documentation\
- Adds nearly all documentation udpates from PR #39 as a result of an internal code review, but makes the changes in the source files that generate the example notebooks, so is not a direct merge
- Adds an annual fee to `PortConfig.annual_fee` that gets applied monthly, though is not included in any metrics as of yet.
- Adds `UnscheduledServiceEquipmentData.tow_speed` to differentiate between towing speeds and traveling speeds required between port and site, and implements the towing speed application appropriately
- Adds a `location` flag to the events logging infrastructure and implements its usage across the simulation architecture
- Creates the metric `Metrics.number_of_tows` to track the number of tows and provides breakdowns as needed
- Creates the metric `Metrics.vessel_crew_hours_at_sea` to track the number of vessel or crew hours at sea
- Creates the metric `Metrics.port_fees` to calculate any port fees associated with a project
- Creates the metric `Metrics.opex` to calculate a project's operational expenditures
- Creates the metric `Metrics.NPV` to calculate a project's net present value
- Modifies `Metrics.project_fixed_costs` to have more time resolutions to align with the OpEx calculation options
- Fixes some results formatting inconsistencies in the `Metrics` class

## 0.4.1 (2022-March-8)
- Adds code diagrams to demonstrate how the various components connect
- Updates the documentation to be better in line with the current state of the software
- Fixes a bug that allowed the x_metrics_inputs.yaml file to persist after the cleanup method is called.
- Updates the provided library content structure to account for future updates

## 0.4.0 (2022-February-4)
- Testing now included!
- `pathlib.Path` is used in place of `os` throughout for easier to read file manuevering.
- `attrs.define` and `attrs.field` have been adopted in place of `attr.s` and `attr.ib`, respectively.
- typing style is updated for future python type annotations as, e.g., `Union[List[str], str]` ->
  `list[str] | str`.
- Unused imports have been further cleaned up.
- Better caching of commonly computed values for faster simulations
- Simulation metrics are now able to be saved for later reloading and use.
- Minor documentation updates, mostly in the examples section for changed API usage.
- Validation cases have been re-run for the most up-to-date version.

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
- `wombat.core.simulation_api.Simulation.save_metrics_inputs` added to allow for saving
  the elements required to recreate a `Metrics` object at a later point in time.

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

- `wombat.core.data_classes.annual_date_range` now uses `numpy.hstack` to construct 1-D
  multiple year date ranges to ensure there is no nesting of data.

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
  - The control flow for if there isn't enough time to perform teh repair is updated to accurately
    account for when and where the equipment needs to travel
  - Cables are no longer retrieved through system and causing an error when processing their repairs.
- `wombat.core.service_equipment.ServiceEquipment._calculate_intra_site_time` now accounts for a
  speed of 0 km/hr so that function will not error out when there is a travel distance, but no speed.
- `wombat.core.service_equipment.ServiceEquipment.run_scheduled` now sets the `onsite` variable
  for equipment that are always on site for more consistent handling of the onsite attribute.
- `wombat.core.service_equipment.ServiceEquipment.wait_until_next_operational_period` now resets
  the equipment's location attributes so its location can't incorrectly persist from the last step.
- `wombat.core.service_equipment.ServiceEquipment.register_repair_with_subassembly` correctly
  retrieves the cable information for upstream cables to be reset.

- `wombat.core.post_processor.Metrics.service_equipment_utilization` has a new methodolgy that uses the
  acual number of days in operation instead of a backwards computation that consistently and
  accurately accounts for the days where the servicing equipment is in operation. Additionally, the
  filtering is updated to match the filter for total days, which also improves accuracy of results.
- `wombat.core.post_processor.Metrics.from_simulation_outputs` classmethod was added to easily load
  simulation data for after-the-fact analyses.

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
- `wombat.windfarm.system.cable.Cable.interrupt_all_subassembly_processes` wraps `interrupt_processes`
  to ensure similar functionality between system and subassembly methods for use in the simulations.
- `wombat.windfarm.system.cable.Cable` logging now properly records itself as the target of repais
  and maintenance tasks instead of its starting node.
- `wombat.windfarm.system.cable.Cable.upstream_nodes` has been updated to be in the correct order
  the nodes sit on the string

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
