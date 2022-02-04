# Simulation API

## Configuration
```{eval-rst}
.. autoclass:: wombat.core.simulation_api.Configuration
   :members:
   :undoc-members:
   :exclude-members: name, library, layout, service_equipment, weather, workday_start,
    workday_end, inflation_rate, fixed_costs, project_capacity, start_year, end_year
```

## Simulation Interface
```{eval-rst}
.. autoclass:: wombat.core.simulation_api.Simulation
   :members:
   :undoc-members:
   :exclude-members: setup_simulation, config, env, initialize_metrics, library_path,
    metrics, repair_manager, service_equipment, windfarm
```


## Metrics Computation
```{eval-rst}
.. autoclass:: wombat.core.post_processor.Metrics
   :members:
   :undoc-members:
   :exclude-members:
```
