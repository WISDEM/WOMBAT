# [Unreleased]

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

# 0.3.1 (2021-March-2)

 - Updated the the `simulation/` folder to be a single-level `core/` directory.
 - Updates the order of the keyword arguments for WombatEnvironment.log_action(), and makes all arguments keyword-only arguments.
