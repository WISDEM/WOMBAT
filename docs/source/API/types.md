# Data Classes

The WOMBAT architecture relies heavily on a base set of data classes to process most of
the model's inputs. This enables a rigid, yet robust data model to properly define a
simulation.

```{image} ../images/data_classes.svg
:alt:
:align: center
:width: 2400px
```

## What is this FromDictMixin I keep seeing in the code diagrams?

The `FromDictMixin` class provides a standard method for providing dictionary definitions
to `attrs` dataclasses without worrying about overloading the definition. Plus, you get
the convenience of writing `cls.from_dict(data_dict)` instead of cls(**data_dict), and
hoping for the best.

```{eval-rst}
.. autoclass:: wombat.core.data_classes.FromDictMixin
   :members:
   :undoc-members:
   :exclude-members:
```

## Scheduled and Unscheduled Maintenance

### Maintenance Tasks
```{eval-rst}
.. autoclass:: wombat.core.data_classes.Maintenance
   :members:
   :undoc-members:
   :exclude-members: time, materials, frequency, equipment, system_value, description,
    level, operation_reduction, rng, service_equipment, replacement
```

### Failures
```{eval-rst}
.. autoclass:: wombat.core.data_classes.Failure
   :members:
   :undoc-members:
   :exclude-members: scale, shape, time, materials, operation_reduction, weibull, name,
    maintenance, failures, level, equipment, system_value, description, rng,
    service_equipment, replacement
```

### Repair Requests
```{eval-rst}
.. autoclass:: wombat.core.data_classes.RepairRequest
   :members:
   :undoc-members:
   :exclude-members: system_id, system_name, subassembly_id, subassembly_name,
    severity_level, details, cable, upstream_turbines, upstream_cables,
```

## Servicing Equipment and Crews

### Service Equipment
```{eval-rst}
.. autoclass:: wombat.core.data_classes.ServiceEquipmentData
   :members:
   :undoc-members:
   :exclude-members: data_dict, strategy
```

### ServiceCrew
```{eval-rst}
.. autoclass:: wombat.core.data_classes.ServiceCrew
   :members:
   :undoc-members:
   :exclude-members: n_day_rate, day_rate, n_hourly_rate, hourly_rate
```

### Scheduled Service Equipment
```{eval-rst}
.. autoclass:: wombat.core.data_classes.ScheduledServiceEquipmentData
   :members:
   :undoc-members:
   :exclude-members: name, equipment_rate, day_rate, n_day_rate, hourly_rate,
    n_hourly_rate, start_month, start_day, start_year, end_month, end_day, end_year,
    capability, mobilization_cost, mobilization_days, speed, max_windspeed_repair,
    max_windspeed_transport, max_waveheight_transport, max_waveheight_repair, onsite,
    method, max_severity, operating_dates, create_date_range, workday_start,
    workday_end, crew, crew_transfer_time, n_crews, strategy, port_distance,
    reduced_speed_start, reduced_speed_end, reduced_speed, speed_reduction_factor,
    non_operational_end, non_operational_start,
```

### Unscheduled Service Equipment
```{eval-rst}
.. autoclass:: wombat.core.data_classes.UnscheduledServiceEquipmentData
   :members:
   :undoc-members:
   :exclude-members: name, equipment_rate, day_rate, n_day_rate, hourly_rate,
    n_hourly_rate, start_month, start_day, start_year, end_month, end_day, end_year,
    capability, mobilization_cost, mobilization_days, speed, max_windspeed_repair,
    max_windspeed_transport, max_waveheight_transport, max_waveheight_repair, onsite,
    method, max_severity, operating_dates, create_date_range, workday_start,
    workday_end, crew, crew_transfer_time, n_crews, strategy, strategy_threshold,
    speed_reduction_factor, non_operational_end, non_operational_start, unmoor_hours,
    reconnection_hours, port_distance, reduced_speed_start, reduced_speed_end,
    reduced_speed, tow_speed, charter_days
```

### Port Configuration
```{eval-rst}
.. autoclass:: wombat.core.data_classes.PortConfig
   :members:
   :undoc-members:
   :exclude-members: name, tugboats, crew, n_crews, max_operations, workday_start,
    workday_end, site_distance, annual_fee, non_operational_start, non_operational_end,
    reduced_speed_start, reduced_speed_end, reduced_speed, non_operational_dates_set,
    reduced_speed_dates_set, non_stop_shift
```

## Wind Farm Support

### Subassembly Model
```{eval-rst}
.. autoclass:: wombat.core.data_classes.SubassemblyData
   :members:
   :undoc-members:
   :exclude-members: name, maintenance, failures, system_id, system_name, system_value,
    subassembly_id, subassembly_name, severity_level, details, cable, upstream_turbines,
    rng
```

### Wind Farm Map
```{eval-rst}
.. autoclass:: wombat.core.data_classes.WindFarmMap
   :members:
   :undoc-members:
   :exclude-members: substation_map, export_cables
```

### Substation Map
```{eval-rst}
.. autoclass:: wombat.core.data_classes.SubstationMap
   :members:
   :undoc-members:
   :exclude-members: string_starts, string_map, downstream
```

### String
```{eval-rst}
.. autoclass:: wombat.core.data_classes.String
   :members:
   :undoc-members:
   :exclude-members: start, upstream_map
```

### Sub String
```{eval-rst}
.. autoclass:: wombat.core.data_classes.SubString
   :members:
   :undoc-members:
   :exclude-members: downstream, upstream
```

## Miscellaneous

### Fixed Cost Model
```{eval-rst}
.. autoclass:: wombat.core.data_classes.FixedCosts
   :members:
   :undoc-members:
   :exclude-members: operations, operations_management_administration,
    project_management_administration, marine_management, weather_forecasting,
    condition_monitoring, operating_facilities, environmental_health_safety_monitoring,
    insurance, brokers_fee, operations_all_risk, business_interruption,
    third_party_liability, storm_coverage, annual_leases_fees, submerge_land_lease_costs,
    transmission_charges_rights, onshore_electrical_maintenance, labor, resolution,
    hierarchy, cost_category_validator
```
