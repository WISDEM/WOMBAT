(simulation-api)=
# Simulation API

```{image} ../images/simulation_api.svg
:alt:
:align: center
:width: 2400px
```

(simulation-api:config)=
## Configuration

```{eval-rst}
.. autoclass:: wombat.core.simulation_api.Configuration
   :members:
   :undoc-members:
   :exclude-members: name, library, layout, service_equipment, weather, workday_start,
    workday_end, inflation_rate, fixed_costs, project_capacity, start_year, end_year,
    port, port_distance, non_operational_start, non_operational_end, reduced_speed_start,
    reduced_speed_end, reduced_speed, random_seed, random_generator, cables, vessels,
    turbines, substations
```

(simulation-api:interface)=
## Simulation Interface

```{eval-rst}
.. autoclass:: wombat.core.simulation_api.Simulation
   :members:
   :undoc-members:
   :exclude-members: setup_simulation, config, env, initialize_metrics, library_path,
    metrics, repair_manager, service_equipment, windfarm, port, random_seed,
    random_generator
```

(simulation-api:metrics)=
## Metrics Computation

For example usage of the Metrics class and its associated methods, please see the
[examples documentation page](../examples/metrics_demonstration)

```{eval-rst}
.. autoclass:: wombat.core.post_processor.Metrics
   :members: from_simulation_outputs, power_production, time_based_availability,
    production_based_availability, capacity_factor, task_completion_rate,
    equipment_costs, service_equipment_utilization, vessel_crew_hours_at_sea,
    number_of_tows, labor_costs, equipment_labor_cost_breakdowns, emissions,
    process_times, component_costs, port_fees, project_fixed_costs, opex, npv
   :member-order: bysource
   :undoc-members:
   :exclude-members: _yield_comparisons, _repr_compare
```
