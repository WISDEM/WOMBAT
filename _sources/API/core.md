(core)=
# Simulation Core Classes

There are a variety of components that enable a simulation to be run, from the
environment to the management of repairs to the servicing equipment. The below will show
how each of the APIs are powered to enable the full flexibility of modeling.

```{image} ../images/simulation_tools.svg
:alt:
:align: center
:width: 2400px
```

(core:environment)=
## Environment

```{eval-rst}
.. autoclass:: wombat.core.environment.WombatEnvironment
   :members:
   :undoc-members:
```

(core:repair-manager)=
## Repair Management

```{eval-rst}
.. autoclass:: wombat.core.repair_management.RepairManager
   :members:
   :undoc-members:
   :exclude-members: _current_id
```

(core:service-equipment)=
## Servicing Equipment

```{eval-rst}
.. automodule:: wombat.core.service_equipment
   :members:
   :inherited-members:
   :exclude-members: ServiceEquipment

.. autoclass:: wombat.core.service_equipment.ServiceEquipment
   :members:
   :inherited-members:
   :undoc-members:
   :exclude-members: env, windfarm, manager, settings, port
```

(core:port)=
## Port

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
