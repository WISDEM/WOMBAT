# Simulation API

```{image} ../images/simulation_api.svg
:alt:
:align: center
:width: 2400px
```

## Configuration
```{eval-rst}
.. autoclass:: wombat.core.simulation_api.Configuration
   :members:
   :undoc-members:
   :exclude-members: name, library, layout, service_equipment, weather, workday_start,
    workday_end, inflation_rate, fixed_costs, project_capacity, start_year, end_year,
    port, SAM_settings
```

## Simulation Interface
```{eval-rst}
.. autoclass:: wombat.core.simulation_api.Simulation
   :members:
   :undoc-members:
   :exclude-members: setup_simulation, config, env, initialize_metrics, library_path,
    metrics, repair_manager, service_equipment, windfarm, port
```


## Metrics Computation
For example usage of the Metrics class and its associated methods, please see the [examples documentation page](../examples/metrics_demonstration)
```{eval-rst}
.. autoclass:: wombat.core.post_processor.Metrics
   :members: from_simulation_outputs, power_production, time_based_availability,
    production_based_availability, capacity_factor, task_completion_rate,
    equipment_costs, service_equipment_utilization, vessel_crew_hours_at_sea,
    number_of_tows, labor_costs, equipment_labor_cost_breakdowns, component_costs,
    opex, npv, project_fixed_costs, process_times, pysam_npv, pysam_lcoe_real,
    pysam_lcoe_nominal, pysam_irr, pysam_all_outputs
   :member-order: bysource
   :undoc-members:
   :exclude-members:
```
