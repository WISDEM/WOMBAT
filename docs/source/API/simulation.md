# Simulation Tools and Methodology


## Simulation Classes
There are a variety of components that enable a simulation to be run, from the
environment, to the windfarm and its composition, and the servicing equipment. The
below will show how each of the APIs are powered to enable the full flexibility of modeling.

```{image} ../images/simulation_tools.svg
:alt:
:align: center
:width: 2400px
```

### Environment
```{eval-rst}
.. autoclass:: wombat.core.environment.WombatEnvironment
   :members:
   :undoc-members:
```

### Repair Management
```{eval-rst}
.. autoclass:: wombat.core.repair_management.RepairManager
   :members:
   :undoc-members:
   :exclude-members: _current_id
```

### Servicing Equipment
```{eval-rst}
.. automodule:: wombat.core.service_equipment
   :members:
   :inherited-members:
   :exclude-members: ServiceEquipment

.. autoclass:: wombat.core.service_equipment.ServiceEquipment
   :members:
   :inherited-members:
   :undoc-members:
   :exclude-members: env, windfarm, manager, settings
```

### Port
```{eval-rst}
.. automodule:: wombat.core.port
   :exclude-members: Port

.. autoclass:: wombat.core.port.Port
   :members: transfer_requests_from_manager, repair_single, run_repairs,
      wait_until_next_shift, run_tow_to_port, run_unscheduled_in_situ
   :inherited-members:
   :undoc-members:
   :exclude-members: env, windfarm, manager, settings, requests_serviced, turbine_manager,
      crew_manager, tugboat_manager, active_repairs, GetQueue, PutQueue, capacity, get,
      get_queue, items, put, put_queue
```


## Windfarm Classes

### Windfarm
```{eval-rst}
.. automodule:: wombat.windfarm.windfarm
   :members:
   :undoc-members:
```

### System: Wind Turbine or Substation
```{eval-rst}
.. automodule:: wombat.windfarm.system.system
   :members:
   :undoc-members:
```

### Subassembly
```{eval-rst}
.. automodule:: wombat.windfarm.system.subassembly
   :members:
   :undoc-members:
```

### Cable
```{eval-rst}
.. automodule:: wombat.windfarm.system.cable
   :members:
   :undoc-members:
```
