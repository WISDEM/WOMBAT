# CHANGELOG

## v0.10.1 [4 April 2025]

Patch release addressing improper setup logic for the new date-based
maintenance following subassembly replacements.

## v0.10 [26 March 2025]

### Features

#### Date-based maintenance and improved timing

The frequency of maintenance events is now significantly more customizable by enabling
custom starting dates and more resolved frequency inputs. There is no change required
for existing configurations as the default value will `frequency` number of days until
the next event.

- `frequency` is an integer input (0 for unmodeled)
- `frequency_basis` indicates the time basis for `frequency`, which is modeled in two
  forms:
  - timing based: amount of operational time that should pass before an event occurs,
    which does not count downtime (repairs, upstream failure, or system offline) in the
    time until the next event.
    - "days": number of days between events (default)
    - "months": number of months between events
    - "years": number of years between events
  - date based: uses a set schedule for when events should occur, regardless of downtime
    and will use `start_date` as the first occurrence of the event.
    - "date-days": number of days between events
    - "date-months": number of months between events
    - "date-years": number of years between events
- `start_date`: The first occurrence of the maintenance event, which defaults to the
  simulation start date + the expected interval. If the input is prior to the simulation
  start date, then it is treated as a staggered timing, so, for instance, a biannual
  event starting the year prior to the simulation ends up being staggered with the first
  occurrence in the first year of the simulation, not the second. Similarly, this
  allows for maintenance activities to start well into the simulation period allowing
  for costs on OEM-warrantied maintenance activities to be unmodeled.

A few examples of more complex scenarios assuming a 1/1/2000 simulation starting date:

- Semiannual event to occur starting in the 3rd month of the simulation:

  ```yaml
  frequency: 6
  frequency_basis: months
  start_date: "9/1/1999"
  ```

- Summer-based annual event with the first occurrence in the 3rd year of the simulation:

  ```yaml
  frequency: 1
  frequency_basis: years
  start_date: "6/1/2003"
  ```

- Annual, June maintenance activity:

  ```yaml
  frequency: 1
  frequency_basis: "date-years"
  start_date: "6/1/2000"
  ```

- Biannual, June maintenance activity that should start in the first year, and every
  other year after that:

  ```yaml
  frequency: 1
  frequency_basis: "date-years"
  start_date: "6/1/2000"
  ```

#### Repeat vessel configuration simplification

Multiple instances of a servicing equipment can be created with a list of the name
and number of them in the following forms. This creates copies where the equipment's
name is suffixed with a 1-indexed indication of which number it is, such as
"Crew Transfer Vessel -1" through "Crew Transfer Vessel - 4" for the below example.

```yaml
...
servicing_equipment:
  - - ctv.yaml
    - 4
  - [hlv.yaml, 2]
  - dsv.yaml
...
```

#### Single(ish) file configuration

WOMBAT configurations now allow for the embedding of servicing equipment, turbine,
substation, cable, and port data within the main configuration. Now, the only files
required outside the primary configuration YAML are the weather profile and layout.
To utilize this update, data can be included in the following form:

```yaml
servicing_equipment:
  - [7, ctv]
...  # Other configuration details
vessels:
  ctv:
    ...  # Contents of library/corewind/vessels/ctv.yaml
turbines:
  corewind_15MW:
    ...  # Contents of library/corewind/turbines/corewind_15MW.yaml
substations:
  corewind_substation:
    ...  # Contents of library/corewind/substations/corewind_substation.yaml
cables:
  corewind_array:
    ...  # Contents of library/corewind/cables/corewind_array.yaml
  corewind_export:
    ...  # Contents of library/corewind/cables/corewind_export.yaml
```

### Updates

- Update the `Failure` definition to use lists of failure configurations, not
  dictionaries. Users can use the following function to update their cable, turbine,
  substation, and consolidated configurations:
  `wombat/core/library.py::convert_failure_data` Documentation is available at
  https://wisdem.github.io/WOMBAT/API/utilities.html#importing-and-converting-from-old-versions.
- Updates the minimum Python version to 3.10.
- The wind farm operation level calculation was moved to `wombat/utilities/utilities.py`
  so it can be reused when `Metrics` loads the operational data.
- Adds a CI check for code linting (pre-commit) and for the documentation building.
- Basic tests added for the Simulation API
- Fixes the `FutureWarnings` from Pandas about changing offset strings.
- Updates the COREWIND severity levels to those originally listed in the publication now
  that severity levels can be used multiple times within a single subassembly.
- Updates the GH Pages uploading workflow steps.

## 0.9.7 (12 February 2025)

- Fixes a new bug where YAML is now sensitive to the implicit closing of files by using
  a context manager to open a YAML file and return the contents.
- Removes PyPI secret usage now that trusted publishing fails with redundant permissions.

## 0.9.6 (11 December 2024)

- Fixes a discrepancy where the wind farm vs turbine availability losses do not match. A slight difference in total availability will be noticeable as a result.

## 0.9.5 (6 August 2024)

- Fixes a bug that causes delayed mobilizations. The underlying cause was the lack of
  resetting the index column of `WombatEnvironment.weather` after filtering out rows
  that exist prior to the starting year of the simulation.
- The Polars minimum version was bumped to avoid a deprecation error with the previous
  index column generation method.

## 0.9.4 (1 July 2024)

- Adds support for Python 3.11 and 3.12.
- Adds the following capability and servicing equipment codes:
  - MCN for medium cranes, which should enable greater options for land-based wind.
  - VSG for vessel support groups, or any representation of multiple vessels used for a
    single operation.
- Updates Polars API usage to account for a series of deprecation and future warnings.
- Changes the metrics demonstration to use the COREWIND Morro Bay in situ example, and
  adds the availability plotting to the demonstration example.
- `RepairRequest.prior_operating_level` has been added to allow 100% reduction factor failures to correctly and consistently restore the operating level of a subassembly following a repair.
- Replaces the `valid_reduction` attrs validator with `validate_0_1_inclusive` to reuse the logic in multiple places without duplicating checking methods.
- Adds a `replacement` flag for interruption methods, so that a failure or replacement comment can be added as a cause for `simpy.process.interrupt`. This update allows the failure and maintenance processes to check if an interruption should cause the process to exit completely. Additionally, the forced exit ensures that processes can't persist after a replacement event when a process is recreated, which was happening in isolated cases.
- Fixes a bug in `RepairManager.purge_subassemble_requests()` where the pending tows are cleared regardless of whether or not the focal subassembly is the cause of the tow, leading to a simulation failure.
- Fixes a bug in `utilities/utilities.py:create_variable_from_string()` to operate in a way that is expected. The original method was removing all numerics, but only leading punctuation and numerics should be replaced, with any punctuation being replaced with an underscore.
- Adds additional inline comments for clarification on internal methods.
- Update README.md to be inline with current conda and Python standards.
- Fully adopts `functools.cache` now that older Python versions where
  `functools.lru_cache` was the available caching method.
- Fixes a Pandas `FutureWarning` by removing a now unnecessary piece of code in
  `wombat/core/post_processor.py:equipment_costs`

## v0.9.3 (15 February 2024)

- Reinstate the original time-based availability methodology, which is based on all
  turbines, not the wind farm total.
- Replace the `black` formatter with `ruff`.
- Adopt `pyupgrade` to ensure modern Python language usage.
- Add the NumPy 2.0 integration tool to Ruff.

## v0.9.2 (13 November 2023)

### General

- Adds the PowerPoint file used in the NAWEA/WindTech 2023 Workshop.
- Updates all coming soon links and typos.

### Bug Fixes

- Fixes an edge-case introduced in v0.9.1 where traveling during a crew transfer process is able to occur after the end of the simulation period.
- The documentation has been completely converted to a Jupyter Book paradigm due to a build issue found in the GitHub Actions.

## v0.9.1 (27 October 2023)

- Removes the pre-public release analysis data and results, located at`library/original-outdated`, from the repository, but can be found in all releases prior to v0.9. This is being done to manage the package size and ensure it can be published to PyPI.

## v0.9 (27 October 2023)

### Bug Fixes

- Repairs that had a weather delay extended repairs past the end of the shift now properly extend the repair into future shifts instead of completing early. This has a small negative impact on availability because that means some repairs can take longer than they had originally.
- Traveling to a system for a repair where the timing extends beyond the end of the shift, but into the next shift, is now registered as a shift delay just like travel weather delays that extend beyond the end of the current shift but before the start of the next shift. This has a small positive impact on availability because a turbine or cable may not start being repaired until weather is more consistently clear, rather than starting it and extending it for many shifts.
- `Windfarm.cable()` now correctly identifies 2 and 3 length cable naming conventions to differentiate which version of the cable id is being retrieved.
- An edge case where events occurred just after the end of a simulation has been resolved by checking the datetime stamp of that event and not adding any action log to the simulation that is after the `WombatEnvironment.end_datetime`.
- A bug in how the total wind farm operating level was calculated is updated to account for substation downtime, rather than using a sum of all turbine values.
- `Metrics.time_based_availability` and `Metrics.production_based_availability` have been updated to use to take advantage of the above fix. Similarly, the time-based availability skews higher now, as is expected when taking into account all availability greater than 0, and the energy-based availability drops moderately as a result of accounting for the substation downtime.

### General Updates

- `Metrics.equipment_labor_cost_breakdowns` now has a `by_equipment` boolean flag, so that the labor and equipment costs can be broken down by category and equipment. Additionally, `total_hours` has been added to the results, resulting in fewer computed metrics across the same set of breakdowns.
- "request canceled" and "complete" are now updated in the logging to directly state if it's a "repair" or "maintenance" task that was completed or canceled to ensure consistency across the logging messages. As a result, `Metrics.task_completion_rate()` can now correctly differentiate between the completed tasks effectively.
- The use of unique naming for the servicing equipment is now enforced to ensure that there is no overlap and potential confusion in the model.
- New, experimental plotting functionality has been added via `wombat.utilities.plot`.
  - `plot_farm_layout` plots the graph layout of the farm. Note that this will not work if realistic lat/lon pairs have not been provided in the layout CSV.
  - `plot_farm_availability` plots a line chart of the monthly overall windfarm availability. Additional toggles allow for the plotting of individual turbines in the background and/or a 95% confidence interval band around the farm's availability
  - `plot_detailed_availability` plots a heatmap of the hourly turbine and farm operational levels to help in debugging simulations where availability may be running suspiciously low (i.e., a turbine might have shut down or a cable failure didn't get repaired within a reasonable time frame).
- `Simulation.service_equipment` is now a dictionary to provide clear access to the servicing equipment details.
- `Simulation.from_config()` requires both a library input and a file name input for the configuration after deprecating the `library` field from the configuration requirements due to it being cumbersome
- `Metrics.opex()` provides the opex output by category if `by_category` is set to be `True`

### Methodology Updates

- Subassemblies and cables are now able to resample their next times to failure for all maintenance and failure activities, so that replacement events reset the timing for failures across the board.
- Systems are no longer interrupted once the servicing equipment addressing a repair or maintenance task is dispatched, and instead are interrupted just before the crew is transferred to begin repair, maintenance, or unmooring. This has a small net positive increase in the availability for short travel distances, but can vary if the travel time is more than a few hours (i.e., large distance to port, slow travel due to weather, & etc.)
- When a tow to port repair is in the queue, no other repairs or maintenance activities will occur for that turbine until it's brought into port for repairs.

### Deprecations

- The original library structure has been fully deprecated, please see the Reference/How To example for more details on converting data that has not yet been updated.
- All PySAM functionality has been deprecated as it's no longer relevant to this work.
- The `Simulation.config` no longer requires the `library` field to reduce conflicts with sharing data between users.

## v0.8.1 (28 August 2023)

- Fixes a bug where servicing equipment waiting for the next operational period at the end of a simulation get stuck in an infinite loop because the timeout is set for just prior to the end of the simulation, and not just after the end of the simulation's maximum run time.

## v0.8.0 (16 August 2023)

### Bug Fixes

- Most of the `# type: ignore` comments have been removed, or the past errors have been resolved
- Failure and maintenance logic in the `Cable` and `Subassembly` models have been wrapped in `if`/`else` blocks to ensure previously unreachable code still can't be reached under limited conditions
- Fixes a bug in where the mobilization duration is logged. Previously, the mobilization duration was logged upon completion, but now is logged while during the actual mobilizing, with costs still being logged upon completion.
- Fixes an uncommon error where repairs are made twice causing the simulation to fail. This was caused by a control mismatch in the unscheduled repair process, and was fixed by better tracking and controlling when a repair moves from "pending" (submitted), to "processing" (handed off to the servicing equipment for repair or to the port for a tow-to-port repair cycle), to "completed" (the repair is registered as complete). Additionally, the servicing equipment no longer relies on receiving the first repair from the repair manager on dispatch, and follows the same `get_next_request()` logic that will occur following the initial dispatch and repair.
- Fixes an error where a repair has `replacement=True` and `operation_reduction` < 1 both resets the `operating_level` back to 1 and readjusts for the `operation_reduction`, which can cause `operating_level` > 100%
- `ServiceEquipment.crew_transfer()` uses a while loop in place of an `if` with recursion strategy to handle unsafe transfer conditions to continuously wait until the next shift's available window.

### Features

- Adds a `non_stop_shift` attribute to `ServiceEquipmentData`, `UnscheduledServiceEquipmentData`, `ScheduledServiceEquipmentData`, and `PortConfig` that is set in the post-initialization hook or through `DateLimitsMixin._set_environment_shift()` to ensure it is updated appropriately. Additionally, all checks for a 24 hour shift now check for the `non_stop_shift` attribute.
- `Metrics.emissions()` has been added to the list of available metrics to calculate the emissions from idling at port or sea, tranisiting, and maneuvering. Co-authored by and inspired by analysis work from @hemezz.
- `Simulation` now accepts a `random_seed` or `random_generator` variable to seed the random number generators for Weibull failure timeouts and wait timing between event completions. Setting the `random_seed` to the same value from one simulation to the next will net the same results between different simulations, whereas the `random_generator` can be used to use the same generator for a batch of simulations.

### General

- All `assert` statements are now only called when type checking is performed
- Replaces all `.get(lamda x: x == request)` with a 10x faster `.get(lambda x: x is request)` to more efficiently filter out the desired event to be removed from the repair manager and port repair management.
- `WombatEnvironment.weather` is now a Polars DataFrame to improve efficiency and indexing bottlenecks introduced in Pandas 2.0.
- All subassembly cable files are read in once, and stored in a dictionary to provide a modest speed up for the simulation initialization.

## v0.7.1 (4 May 2023)

- Features
  - `Metrics.process_times()` now includes the time_to_start, representing the time between when a request is submitted, and when the repairs officially start.
  - Expand the acceptable date formats for the weather profiles to allow for year-first.

## v0.7.0 (3 May 2023)

- Replace Flake8 and Pylint in the pre-commit workflow with ruff, and fix/ignore the resulting errors as appropriate
- Features:
  - Weather data now has the ability to contain more than just the required "windspeed" and "waveheight" columns. This will allow for easier expansion of the weather model in the future, and increase compatibility with other NREL techno economic modeling frameworks.
- Bug fixes:
  - Maintenance and failure simulation process interruptions were occuring prior to starting the process timing, and causing simulation failures.
  - Duplicated parameters were being processed in `WombatEnvironment.log_action` stemming from improper handling of varying parameters in some of the more complex control flow logic in *in situ* repairs.
  - Another edge case of negative delays during crew transfers where there is insufficient time remaining in the shift after account for weather, so the method was called recursively, but not exiting the original loop.
  - `Port` management of *in situ* and tow-to-port capable tugboats wasn't properly accounting for tugboats of varying capabilities, and assuming all tugboats could do both. The vessel management and repair processing were out of sync causing duplicated turbine servicing/towing.
  - `RepairManager` has consolidated the dispatching of servicing equipment to be the following categories instead of the previously complex logic:
    - If a request is a tow-to-port category, have the port initiate the process
    - If the dispatching thresholds are met, then have the port initiate a repair for port-based servicing equipment, otherwise the repair manager will dispatch the appropriate servicing equipment.
  - `ServiceEquipment.weather_delay()` no longer silently processes a second weather delay.
  - Intra-site travel is now correctly logging the distance and duration of traveling between systems at the site by moving the location setting logic out of `crew_transfer` method and being entirely maintained within, or next to, the `travel` method of `ServiceEquipment`.
  - `RepairManager` now properly waits to dispatch servicing equipment until they are no longer under service from another servicing equipment unit.
  - Servicing equipment are no longer simultaneously dispatched from both the `Port` and `RepairManager` causing simulations to potentially error out when the `System.servicing` or `Cable.servicing` statuses are overridden by the second-in-line servicing equipment. This was resolved by the port waiting for the turbine, a tugboat, and a spot at port to become available, then immediately halting the turbine, and starting the tow-in logic.
  - Missing `ServiceEquipment.dispatched` status updates have been amended, so no matter the operation, a piece of servicing equipment should be set to `dispatched = True` until the crew is back and repair is completed.
  - To avoid multiple dispatches to a single turbine, cable, or substation, or multiple dispatches of a single vessel/onsite equipment, a random timeout between 0 and 30 seconds (in simulation time) is processed, then a status double-checking occurs.
  - `Metrics.process_times()` now includes the "N" column to indicate the number of processes that fall into each category.

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
  - Models the export cabling system as a single cable between the substation and the interconnection point or as a connection between multiple substations. For multiple connected substations, the model assumes they are independent systems.
  - Adds support for "type" in the wind farm layout CSV file, which should be filled with either "substation" or "turbine". This column supports multi-substation farms so that accurate plots and connections can be made. For instance, multiple connected substations, can now be accurately rendered and modeled in the farm.
  - Adds support for "upstream_cable_name" in the wind farm layout CSV file, to provide an individualized name to a cable in place of using the name field in the cable settings file for all similar cables.
- New library structure that mirrors ORBIT (see below diagram)! In v0.7, the original library structure will be officially deprecated in favor of the below, and during the v0.6 lifecycle a warning will be raised to instruct users where to place and structure folders going forward.
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
  - All underlying infrastructure withing the simulation have also been updated to accommodate the different file types, which allows for more direct interaction at the end of the simulation and enables PyArrow CSV read/write
  - This enables:
    - deprecation warnings to be passed directly to the terminal/notebook without interfering with the file handling
    - reasonable speedups to simulation times by not having additional overhead from the logging and buffering
- `Metrics.process_times()` has been converted to efficient pandas code for fast computation.
- The Metrics example usage documentation page has been rewritten and reformatted to provide more helpful information to users
- `Metrics` methods now accurately account for the effect of substation operating reductions on upstream turbines so that each substation's subgraph multiplies the turbine operating capacity by the substation operating capacity.
- Fixes a bug in the `RepairManager.get_next_highest_severity_request()` where requests aren't processed in first in, first out order with a severity level priority.
- Remove duplicated logic in the ``Subassembly`` and ``Cable`` maintenance and failure modeling ensuring that repetitive logic is identical between scenarios.

## 0.5.1 (22 July 2022)

- Updates to use the most recent pandas API/recommendations, which fixes numerous warnings in the `Metrics` class
- Fixes inconsistency in returning floats vs `DataFrame`s in the `Metrics` class
- Updates the examples to work with the returned `DataFrame` values, and adds warnings about the change in usage
- Updates the documentation configuration to be compatible with the latest sphinx book theme API usage
- Adds a potential fix to an occasional issue where the logging files can't be deleted using `WombatEnvironment.cleanup_log_files()` because the file is still considered to be in use

## 0.5.0 (30 June 2022)

- Adds capabilities: "TOW" and "AHV" for tugboat/towing equipment and anchor-handling vessels
- Adds a tow-to-port strategy that is activated for repairs with the "TOW" capability in the servicing
- Adds a `Port` class to handle the tow-to-port class and tugboat-based service requests
- Allows for any name to define the subassemblies of a turbine or substation to enable users to use the naming conventions they are most familiar with or most meaningful for their own work
- Minor bug fixes in the `Metrics` class to improve stability, improve code reuse, and documentation\
- Adds nearly all documentation updates from PR #39 as a result of an internal code review, but makes the changes in the source files that generate the example notebooks, so is not a direct merge
- Adds an annual fee to `PortConfig.annual_fee` that gets applied monthly, though is not included in any metrics yet.
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
- `pathlib.Path` is used in place of `os` throughout for easier to read file maneuvering.
- `attrs.define` and `attrs.field` have been adopted in place of `attr.s` and `attr.ib`, respectively.
- Typing style is updated for future python type annotations as, e.g., `Union[List[str], str]` ->
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
  float inputs up to the nearest whole number to ensure the proper length window is retrieved.
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
  - Additionally, the equipment will travel back to port if the work shift is over, so the equipment
    does not skip steps the repair logic.
  - Weather delays will also not be processed if the actual repair is finished, and will be moved to
    the crew transfer step.
  - The hours required vs hours available is reset for equipment that have around the clock shifts,
    so the timing is not arbitrarily capped, which also reduces the number of steps in the simulation.
  - The work shift indicator check now accounts for the equipment's specific work shift settings, and
    not the environment's to ensure the proper operating parameters are used.
  - The control flow for if there isn't enough time to perform the repair is updated to accurately
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
  actual number of days in operation instead of a backwards computation that consistently and
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
  when a repair request is submitted while a repair is occurring that drops the operating level below
  the strategy threshold. In the dispatching check we now are able to consider the operating level
  before a system was shut down for repairs and avoid improper dispatching.

- `wombat.windfarm.system.cable.Cable` now takes `end_node` in place of `upstream_nodes` to simplify
  the initialization process and be more explicit about the starting and ending points on the cable.
- `wombat.windfarm.system.cable.Cable.interrupt_subassembly_processes` is now `interrupt_processes`
  and only interrupts the cable's own maintenance and failure simulation processes.
- `wombat.windfarm.system.cable.Cable.interrupt_all_subassembly_processes` wraps `interrupt_processes`
  to ensure similar functionality between system and subassembly methods for use in the simulations.
- `wombat.windfarm.system.cable.Cable` logging now properly records itself as the target of repairs
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
  straightforward for users.
- `ServiceCrew` has been added for expansion to multiple crews.
- Servicing equipment now have separate `travel` and `crew_transfer`, `weather_delay`,
  and `repair` functions to more realistically simulate the separate processes involved.
- The windfarm now has a distance matrix to calculate the distance between any two
  systems onsite, for instance, substation to cable, cable to turbine, etc.
- Bug fix in the task completion rate metric.
- `WombatEnvironment.log_action()` enforces the use of keyword arguments with only 3
  required inputs.
- `WombatEnvironment.weather_forecast()` outputs now include the datetime index.
- If no PySAM settings are provided, the `financial_model` will be set to `None` with a
  `NotImplementedError` being raised for any attempted usage of the PySAM powered
  functionality.
- Fix bugs in `Maintenance` and `Failure` initialization.
- Fix bugs in operating level accounting from recent updates.
- Update the Unicode typing in all docstrings.
- Update the documentation to account for all the recent changes, including adding a new
  demonstration notebook for the 3 new strategies.

## 0.3.1 (2021-March-2)

- Updated the `simulation/` folder to be a single-level `core/` directory.
- Updates the order of the keyword arguments for `WombatEnvironment.log_action()`, and makes all arguments keyword-only arguments.
